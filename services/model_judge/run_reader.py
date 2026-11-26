# services/model_judge/run_reader.py
"""
Run Reader for Model Judge.

Builds one candidate per registered run by reading the run's own folder, the
same files the graph manager plots from. Nothing is copied into an index:

    <project>/artifacts/run_registry.json, test_registry.json
        lifecycle status, env, model_id, the run folder (``path``)

    <run>/model/<name>_checkpoint_manifest.json, <run>/config/<name>_params.yaml
        the tested epoch: the best-validation epoch when validation
        checkpointing is on and evaluation.test.load_model is best_validation,
        otherwise the final epoch

    <run>/metrics/attack_metrics/scalars/<name>_metrics_<process>_history.jsonl
        process train / validation: the row of the tested epoch. A row's "dependencies" rank like its metrics.

    <run>/metrics/key_recovery/validation_best.json
        the key-recovery outcome of process validation

    the latest test of each kind (utils/latest_test.py, owner 2026-10-27: never an earlier one)
        process test: the latest All test (the training's recorded test, or a later Retest All / Test All), its
        test history row and key-recovery report; process test_budget: the latest Budget test's (its data entry's
        attack_trace_budget per attack). candidate["tests"] names the test each was read from.

    <run>/cross_validation/aggregate/summary.yaml, mean_rank_curve.json,
    cross_validation_diagnostics.json
        process cross_validation: the fold-mean headline, the crossings of the
        fold-mean curve, the fold mean of every scalar and the diagnostic
        statistics

Candidate shape::

    {
      "model_name", "model_hash", "env", "model_id",
      "train_status", "test_status", "run_dir", "run_path",
      "processes": {process: {metric: value}},
      "cv_verdict": <diagnostic verdict code or None>,
    }

A process is present when the run recorded something for it; a registered run
whose folder is missing, or that recorded nothing yet, is not a candidate.

Author: Shraddha Mishra
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from services.side_channel_core.utils.latest_test import latest_test
from services.side_channel_core.utils.seal import check as check_seal, strip as strip_seal

logger = logging.getLogger("ModelJudge")

# Layout of a run folder (paths.* of the project's params.yaml).
MANIFEST = "model/{name}_checkpoint_manifest.json"
SNAPSHOT = "config/{name}_params.yaml"
HISTORY = "metrics/attack_metrics/scalars/{name}_metrics_{process}_history.jsonl"
KEY_RECOVERY = {
    "validation": "metrics/key_recovery/validation_best.json",
    "test": "metrics/key_recovery/test.json",
}
CV_AGGREGATE = "cross_validation/aggregate"

# Identity of a history row, not a metric of the model.
_ROW_HEADER = {"epoch", "timestamp", "model_name", "model_hash", "process", "env", "dependencies"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        check_seal(path)  # utils/seal.py: a changed record is refused, not judged
        data = strip_seal(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc.msg}") from exc
    return data if isinstance(data, dict) else {}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    check_seal(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _run_dir(project_root: Path, path: Any) -> Path | None:
    """The run folder of a registry ``path`` (relative to projects/ or to the project)."""
    if not path:
        return None
    for base in (project_root.parent, project_root):
        candidate = base / str(path)
        if candidate.is_dir():
            return candidate
    return None


def _tested(run_dir: Path, name: str) -> tuple[int | None, bool]:
    """(tested epoch, whether the validation set chose it); see tested_epoch."""
    manifest = _read_json(run_dir / MANIFEST.format(name=name))
    final = (manifest.get("final") or {}).get("epoch")
    if final is None:
        return None, False
    selection = manifest.get("selection") or {}
    if manifest.get("validation_checkpointing") is True and selection.get("epoch") is not None:
        evaluation = _read_yaml(run_dir / SNAPSHOT.format(name=name)).get("evaluation") or {}
        policy = str((evaluation.get("test") or {}).get("load_model") or "").strip().lower()
        if policy == "best_validation":
            return int(selection["epoch"]), True
    return int(final), False


def tested_epoch(run_dir: Path, name: str) -> int | None:
    """
    Epoch whose weights the test loads, and that the train/validation metrics
    must describe: the best-validation epoch only when validation
    checkpointing is on and evaluation.test.load_model is best_validation.
    None while the run has no checkpoint manifest (training not finished).
    """
    return _tested(run_dir, name)[0]


def _history_metrics(run_dir: Path, name: str, process: str, epoch: int | None, *, path: Path | None = None) -> dict[str, Any]:
    """Scalar columns of the process's row at ``epoch`` (its last row when None); ``path``: another history file."""
    path = path or run_dir / HISTORY.format(name=name, process=process)
    if not path.is_file():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    try:
        check_seal(path)
    except json.JSONDecodeError:
        pass  # reported below with its line number
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = strip_seal(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSON: {exc.msg}") from exc
        if isinstance(row, dict) and isinstance(row.get("epoch"), int):
            rows[row["epoch"]] = row            # the latest record of an epoch
    if not rows:
        return {}
    row = rows.get(epoch) if epoch is not None else rows[max(rows)]
    if row is None:
        logger.warning("%s has no row for the tested epoch %s", path, epoch)
        return {}
    dependencies = row.get("dependencies") if isinstance(row.get("dependencies"), dict) else {}
    # A metric of the same name as a dependency wins; array pointers are not rankable.
    merged = {**dependencies, **{k: v for k, v in row.items() if k not in _ROW_HEADER}}
    return {key: value for key, value in merged.items() if _scalar(value)}


def _key_recovery_metrics(run_dir: Path, process: str, *, path: Path | None = None) -> dict[str, Any]:
    """Outcome of the process's key-recovery report, as rankable fields; ``path``: another report."""
    relative = KEY_RECOVERY.get(process)
    report = _read_json(path) if path else (_read_json(run_dir / relative) if relative else {})
    if not report:
        return {}
    key = report.get("key") or {}
    enumeration = report.get("enumeration") or {}
    out: dict[str, Any] = {}
    for section in ("key", "recovery"):
        for name, value in (report.get(section) or {}).items():
            if _scalar(value):
                out[name] = value
    margin = (report.get("diagnostics") or {}).get("top1_top2_score_margin")
    if margin is not None:
        out["top1_top2_score_margin"] = margin
    # Without enumeration the model rank is the whole story; with it the
    # headline is the rank the enumeration reports.
    model_rank = key.get("final_true_key_rank")
    out["key_rank_reported"] = enumeration.get("key_rank_reported") or model_rank
    out["key_recovery"] = enumeration.get("key_recovery") or ("DONE" if key.get("recovered") is True else "FAILED")
    out["key_recovered_through"] = enumeration.get("key_rank_source") or "MODEL"
    if enumeration.get("enumeration_position") is not None:
        out["key_enumeration_position"] = enumeration["enumeration_position"]
    out["attack_byte"] = (report.get("target") or {}).get("attack_byte")
    return out


def _cv_metrics(run_dir: Path) -> tuple[dict[str, Any], Any]:
    """(rankable cross-validation metrics, diagnostic verdict code) of a run."""
    aggregate = run_dir / CV_AGGREGATE
    summary = _read_yaml(aggregate / "summary.yaml")
    if not summary:
        return {}, None

    metrics = {
        key: value
        for key, value in (summary.get("headline") or {}).items()
        if key not in {"model_name", "model_hash", "aggregate_path"}
    }
    # "k/n" -> fraction, so it can be ranked (higher is better).
    for key, value in list(metrics.items()):
        if key.startswith("folds_reaching_ge_") and isinstance(value, str) and "/" in value:
            reached, total = value.split("/", 1)
            try:
                metrics[f"{key}_fraction"] = int(reached) / int(total)
            except (ValueError, ZeroDivisionError):
                pass

    # Crossings of the fold-mean rank curve; the stable crossing is the headline.
    for k, item in (_read_json(aggregate / "mean_rank_curve.json").get("traces_to_ge") or {}).items():
        if isinstance(item, dict):
            metrics[f"traces_to_ge_{k}_mean_curve"] = item.get(
                "mean_curve_stable_trace", item.get("mean_curve_first_trace")
            )
            metrics[f"first_traces_to_ge_{k}_mean_curve"] = item.get("mean_curve_first_trace")
            if "median_curve_stable_trace" in item:            # runs aggregated since 2026-10-22
                metrics[f"traces_to_ge_{k}_median_curve"] = item["median_curve_stable_trace"]
                metrics[f"first_traces_to_ge_{k}_median_curve"] = item.get("median_curve_first_trace")
            if item.get("n_folds"):
                metrics[f"folds_reaching_ge_{k}_fraction"] = item.get("folds_reached", 0) / item["n_folds"]

    # Fold mean of every scalar: <metric>_train, <metric>_validation, and the
    # plain <metric> for the held-out value (a fit time: the training one).
    means: dict[str, dict[str, Any]] = {"train": {}, "validation": {}}
    for name, statistics in (summary.get("scalar_metrics") or {}).items():
        process, _, metric = str(name).partition(".")
        if process in means and metric and isinstance(statistics, dict) and "mean" in statistics:
            means[process][metric] = statistics["mean"]
    for process, values in means.items():
        for metric, value in values.items():
            metrics.setdefault(f"{metric}_{process}", value)
    for metric in {**means["train"], **means["validation"]}:
        first, second = ("train", "validation") if "fit_time" in metric else ("validation", "train")
        metrics.setdefault(metric, means[first][metric] if metric in means[first] else means[second][metric])

    report = _read_json(aggregate / "cross_validation_diagnostics.json")
    rank = (report.get("analyses") or {}).get("rank_learning_and_stability") or {}
    for key in ("folds_better_than_chance", "normalized_rank_advantage"):
        if key in rank:
            metrics[key] = rank[key]
    if "sample_std" in (rank.get("endpoint_rank") or {}):
        metrics["endpoint_rank_std"] = rank["endpoint_rank"]["sample_std"]
    return metrics, (report.get("verdict") or {}).get("code")


def load_candidates(project_root: str | Path) -> list[dict[str, Any]]:
    """Every registered run of one project that recorded something, from its run folder."""
    project_root = Path(project_root)
    artifacts = project_root / "artifacts"
    if not artifacts.is_dir():
        raise FileNotFoundError(f"No artifacts directory, nothing was trained: {artifacts}")

    run_registry = _read_json(artifacts / "run_registry.json")
    test_registry = _read_json(artifacts / "test_registry.json")

    candidates: list[dict[str, Any]] = []
    for model_hash, run_info in run_registry.items():
        if not isinstance(run_info, dict) or not run_info.get("model_name"):
            continue
        name = str(run_info["model_name"])
        test_info = test_registry.get(model_hash) if isinstance(test_registry.get(model_hash), dict) else {}
        run_dir = _run_dir(project_root, run_info.get("path"))
        if run_dir is None:
            logger.info("Run %s (%s) has no run folder %r; not a candidate", name, model_hash, run_info.get("path"))
            continue

        epoch, chosen_on_validation = _tested(run_dir, name)
        processes: dict[str, dict[str, Any]] = {}
        # the test processes: the latest test of each kind, never an earlier one (utils/latest_test.py)
        latest = {"test": latest_test(run_dir, test_info, "all"), "test_budget": latest_test(run_dir, test_info, "budget")}
        for process in ("train", "validation", "test", "test_budget"):
            if process in latest:
                source = latest[process]
                if source is None:
                    continue
                history = source["metrics_dir"] / "scalars" / f"{name}_metrics_test_history.jsonl"
                metrics = _history_metrics(run_dir, name, "test", None, path=history)
                for key, value in _key_recovery_metrics(run_dir, "test", path=source["key_recovery_dir"] / "test.json").items():
                    metrics.setdefault(key, value)
            else:
                metrics = _history_metrics(run_dir, name, process, epoch)
                for key, value in _key_recovery_metrics(run_dir, process).items():
                    metrics.setdefault(key, value)
            if not metrics:
                continue
            if process not in latest and epoch is not None:
                metrics["selected_epoch"] = epoch
            if process == "validation":
                # The best of the epochs on the set that picked it: optimistic, not an
                # estimate of the model's performance (the test is).
                metrics["selection_biased"] = chosen_on_validation
            processes[process] = metrics
        cv_metrics, verdict = _cv_metrics(run_dir)
        if cv_metrics:
            processes["cross_validation"] = cv_metrics
        if not processes:
            continue

        candidates.append({
            "model_name": name,
            "model_hash": str(model_hash),
            "env": run_info.get("env"),
            "model_id": run_info.get("model_id") or test_info.get("model_id"),
            "train_status": str(run_info.get("status") or "").upper() or None,
            "test_status": str(test_info.get("status") or "").upper() or None,
            "run_dir": run_info.get("path"),
            "run_path": str(run_dir),
            "processes": processes,
            "cv_verdict": verdict,
            "tests": {p: (t["id"] if t else None) for p, t in latest.items()},   # recorded, or <hash>_et_<index>
        })
    return candidates


__all__ = ["load_candidates", "tested_epoch"]
