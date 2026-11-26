# services/model_judge/model_judge.py

"""
Model Judge for SCA results.

Ranks every trained model of one project/env and selects the best one.

Inputs (see run_reader.py): artifacts/run_registry.json and
artifacts/test_registry.json, and, for every registered run, its own run
folder: the metric history rows of the tested epoch, the key-recovery reports
and the cross-validation aggregate. No separate index is kept.

Ranking (projects/<PROJECT>/config/model_judge.yaml):
1. primary metric, then tie-breakers, each as {process, metric};
2. process is train | validation | test | cross_validation;
3. direction is inferred from the metric name (see metric_direction);
4. a candidate missing a ranking value sorts after every candidate that has it
   (keys stay aligned position by position).

Model selection should use validation or cross-validation evidence only. A
ranking item on process "test" is allowed but logged as a warning and
recorded as ``selection_uses_test_data: true`` because it makes the reported
test result of the selected model optimistically biased. A ranking item on a
quality metric of process "train" (accuracy, loss; rank metrics are never
computed on training traces) is allowed but logged as a warning and listed
under ``selection_uses_train_quality_metrics``: it is measured on the traces
the model was fitted to. Costs such as fit_time_s are legitimate train metrics.

Outputs are written to projects/<PROJECT>/<output_dir> (report_writer.py).

Author: Shraddha Mishra
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import yaml

from services.model_judge.run_reader import load_candidates
from services.model_judge.report_writer import write_outputs
from services.side_channel_core.utils.graph_config_sync import update_default_job_models
from services.side_channel_core.utils.logger import console, pretty_print
from services.side_channel_core.utils.timestamps import now_iso


logger = logging.getLogger("ModelJudge")

SCHEMA_VERSION = "model_selection.v2"
# test: the latest All test; test_budget: the latest Budget test (run_reader.py, utils/latest_test.py)
PROCESSES = {"train", "validation", "test", "test_budget", "cross_validation"}

# Higher or lower is better: one rule with the training monitor (early stopping,
# validation checkpointing), read through the names run_reader builds.
from services.side_channel_core.utils.monitor_metric import metric_direction  # noqa: E402


# Train metrics that are a cost, not model quality measured on the fitted traces.
_TRAIN_COST = re.compile(r"(mean_)?fit_time_(s|h|per_epoch)|trainable_parameters")


# Numeric fields that are counters, settings or provenance, not model quality.
# Key values (true_key_byte, top_key_hypothesis) and true_key_tie_count identify the
# attack's outcome, recovery_rank is a setting, and top1_top2_score_margin has no
# direction: a wide margin is good only when the top hypothesis is the true key.
_BOOKKEEPING = re.compile(
    r".*_(correct|total|successes|trials)|attack_byte|rank_base|epoch|n_folds|evaluated_traces"
    r"|trace_population_.*|traces_(available|used)|trace_order_repetitions|full_population_used"
    r"|key_enumeration_.*|key_recovery_.*|target_key_byte_recovered|selected_validation_epoch|selected_epoch"
    r"|selection_score|selection_metric|mean_curve_traces|rank_contract_version|running_train_accuracy"
    r"|true_key_.*|top_key_hypothesis|recovery_rank|top1_top2_score_margin"
)


def available_metrics(candidates: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Meaningful numeric metrics per process across all candidate models."""
    found: dict[str, set[str]] = {}
    for candidate in candidates:
        for process, metrics in candidate["processes"].items():
            names = found.setdefault(process, set())
            names.update(
                name for name in metrics
                if get_metric(candidate, process, name) is not None
                and not _BOOKKEEPING.fullmatch(name.lower())
            )
    return {process: sorted(names) for process, names in sorted(found.items())}


def recorded_fields(candidates: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Every field name recorded per process, including empty (null) ones."""
    found: dict[str, set[str]] = {}
    for candidate in candidates:
        for process, metrics in candidate["processes"].items():
            found.setdefault(process, set()).update(metrics)
    return found


def print_metric_choices(
    available: dict[str, list[str]],
    candidates: list[dict[str, Any]],
) -> None:
    """Table of the process/metric pairs the recorded runs offer."""
    from rich.table import Table
    from rich.text import Text

    table = Table(
        title="What you can choose, from your current data",
        title_style="bold bright_cyan",
        caption="green: higher is better | blue: lower is better | gray: empty for every model (threshold never reached)",
        caption_style="bright_black",
        header_style="bold bright_yellow",
        show_lines=True,
    )
    table.add_column("process", style="bold bright_magenta", no_wrap=True)
    table.add_column("metrics")
    for process, names in available.items():
        cell = Text()
        for name in names:
            style = "bright_green" if metric_direction(name) == "max" else "bright_blue"
            cell.append(name, style=style).append(", ")
        # Null for every model, e.g. traces_to_ge_k when the threshold was never reached.
        seen = [c["processes"][process] for c in candidates if process in c["processes"]]
        empty = sorted(
            name for name in set().union(*seen) - set(names)
            if all(m.get(name) is None for m in seen)
            and not _BOOKKEEPING.fullmatch(name.lower())
            and not name.endswith("_message")
        )
        for name in empty:
            cell.append(name, style="bright_black").append(", ")
        cell.rstrip()
        if cell.plain.endswith(","):
            cell.right_crop(1)
        table.add_row(process, cell)
    console.print(table)


def config_hints(
    judge_cfg: dict[str, Any],
    available: dict[str, list[str]],
    recorded: dict[str, set[str]] | None = None,
) -> list[str]:
    """One hint per configured process/metric that no candidate model provides."""
    recorded = recorded or {}
    hints: list[str] = []
    configured = [("primary", judge_cfg["ranking"]["primary"])]
    configured += [("tie_breaker", item) for item in judge_cfg["ranking"].get("tie_breakers") or []]
    configured += [("report_metric", item) for item in judge_cfg.get("report_metrics") or []]
    for role, item in configured:
        process, metric = item.get("process"), item.get("metric")
        if process not in available:
            hints.append(
                f"{role} {process}.{metric}: process {process!r} has no data; "
                f"available processes: {list(available)}"
            )
        elif metric in recorded.get(process, set()) and metric not in available[process]:
            hints.append(
                f"{role} {process}.{metric}: recorded but empty for every model "
                f"(e.g. the threshold was never reached); models without it rank last"
            )
        elif metric not in available[process]:
            hints.append(
                f"{role} {process}.{metric}: metric {metric!r} not found; "
                f"available {process} metrics: {available[process]}"
            )
    for process in (judge_cfg.get("filters") or {}).get("require_processes") or []:
        if process not in available:
            hints.append(
                f"filters.require_processes {process!r}: no data; "
                f"available processes: {list(available)}"
            )
    return hints


def load_model_judge_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"model_judge.yaml not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    from services.side_channel_core.utils import config_schema       # a key the judge does not read is refused
    config_schema.check_keys(cfg, config_schema.JUDGE, source=path.name)
    if "model_judge" not in cfg:
        raise KeyError(f"Missing top-level key 'model_judge' in {path}")
    judge_cfg = cfg["model_judge"]

    ranking = judge_cfg.get("ranking") or {}
    items = [ranking.get("primary"), *(ranking.get("tie_breakers") or [])]
    for item in items:
        if not isinstance(item, dict) or not item.get("process") or not item.get("metric"):
            raise ValueError(f"Every ranking item needs 'process' and 'metric': {item!r}")
        if item["process"] not in PROCESSES:
            raise ValueError(f"Unknown ranking process {item['process']!r}; use one of {sorted(PROCESSES)}")
    return judge_cfg


def ranking_items(judge_cfg: dict[str, Any]) -> list[dict[str, str]]:
    ranking = judge_cfg["ranking"]
    return [ranking["primary"], *(ranking.get("tie_breakers") or [])]


def get_metric(candidate: dict[str, Any], process: str, metric: str) -> float | None:
    value = (candidate.get("processes") or {}).get(process, {}).get(metric)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def sort_key(candidate: dict[str, Any], items: list[dict[str, str]]) -> tuple:
    """One (missing, value) pair per ranking item, so positions never shift."""
    key: list[tuple[int, float]] = []
    for item in items:
        value = get_metric(candidate, item["process"], item["metric"])
        if value is None:
            key.append((1, 0.0))
        else:
            key.append((0, value if metric_direction(item["metric"]) == "min" else -value))
    return (*key, candidate["model_name"])


def rejection_reason(candidate: dict[str, Any], judge_cfg: dict[str, Any], env: str | None) -> str | None:
    filters = judge_cfg.get("filters") or {}
    if env is not None and candidate.get("env") != env:
        return f"env {candidate.get('env')!r} != {env!r}"
    if filters.get("require_done", True):
        if candidate.get("train_status") != "DONE":
            return f"train status {candidate.get('train_status')}"
        # SKIPPED: the run's config leaves the test traces unattacked
        # (perform_attack.test.attack_times: 0), so there is no test to wait for.
        if candidate.get("test_status") not in ("DONE", "SKIPPED"):
            return f"test status {candidate.get('test_status')}"
    missing = [p for p in filters.get("require_processes") or [] if p not in candidate["processes"]]
    if missing:
        return f"missing processes {missing}"
    primary = judge_cfg["ranking"]["primary"]
    if get_metric(candidate, primary["process"], primary["metric"]) is None:
        available = available_metrics([candidate]).get(primary["process"], [])
        return (
            f"no numeric {primary['process']}.{primary['metric']}; "
            f"available {primary['process']} metrics: {available}"
        )
    return None


def _item_summary(candidate: dict[str, Any], item: dict[str, str]) -> dict[str, Any]:
    return {
        "process": item["process"],
        "metric": item["metric"],
        "value": get_metric(candidate, item["process"], item["metric"]),
        "direction": metric_direction(item["metric"]),
    }


def summarize_candidate(
    candidate: dict[str, Any],
    judge_cfg: dict[str, Any],
    rank: int,
    project_root: Path,
) -> dict[str, Any]:
    items = ranking_items(judge_cfg)
    report = judge_cfg.get("report_metrics") or []
    run_path = Path(candidate["run_path"]) if candidate.get("run_path") else None
    name = candidate["model_name"]
    return {
        "rank": rank,
        "model_name": name,
        "model_hash": candidate["model_hash"],
        "env": candidate.get("env"),
        "model_id": candidate.get("model_id"),
        "primary": _item_summary(candidate, items[0]),
        "tie_breakers": [_item_summary(candidate, item) for item in items[1:]],
        "selection_metrics": {
            f"{item['process']}.{item['metric']}": get_metric(candidate, item["process"], item["metric"])
            for item in items
        },
        # Shown for context only; never used for ranking.
        "reported_metrics": {
            f"{item['process']}.{item['metric']}": get_metric(candidate, item["process"], item["metric"])
            for item in report
        },
        "cv_verdict": candidate.get("cv_verdict"),
        # the test each test process was read from: recorded (the training's) or <hash>_et_<index>, the latest of its kind
        "tests": candidate.get("tests") or {},
        # True when the tested epoch is the best-validation one: its validation values are
        # the maximum over epochs on the set that chose it, an optimistic number.
        "validation_selection_biased": bool(
            (candidate["processes"].get("validation") or {}).get("selection_biased")
        ),
        "artifacts": {
            "run_dir": str(run_path) if run_path else None,
            "final_model": str(run_path / "model" / f"{name}_trained.pt") if run_path else None,
            "checkpoint_manifest": (
                str(run_path / "model" / f"{name}_checkpoint_manifest.json") if run_path else None
            ),
            "cv_aggregate": (
                str(run_path / "cross_validation" / "aggregate")
                if run_path and "cross_validation" in candidate["processes"]
                else None
            ),
        },
    }


def list_choices(
    config_path: str | Path,
    project: str,
    env: str | None = None,
    projects_root: str | Path = "projects",
) -> dict[str, Any]:
    """What a judge can rank on here: {"available": {process: [metric, ...]}, "direction": {metric: max|min},
    "current": {"process", "metric"} (model_judge.yaml's primary), "candidates": n}. Read only."""
    judge_cfg = load_model_judge_config(config_path)
    project_root = Path(projects_root) / project
    wanted = str(env or "").strip().lower()
    candidates = [c for c in load_candidates(project_root)
                  if not wanted or str(c.get("env") or "").strip().lower() == wanted]
    available = available_metrics(candidates)
    primary = (judge_cfg.get("ranking") or {}).get("primary") or {}
    return {
        "available": available,
        "direction": {name: metric_direction(name) for names in available.values() for name in names},
        "current": {"process": str(primary.get("process") or ""), "metric": str(primary.get("metric") or "")},
        "candidates": len(candidates),
    }


def run_model_judge(
    config_path: str | Path,
    project: str,
    env: str | None = None,
    projects_root: str | Path = "projects",
) -> dict[str, Any]:
    judge_cfg = load_model_judge_config(config_path)
    project_root = Path(projects_root) / project
    items = ranking_items(judge_cfg)

    uses_test = [f"{i['process']}.{i['metric']}" for i in items if i["process"] in ("test", "test_budget")]
    if uses_test:
        logger.warning(
            "Model selection ranks on test data (%s); the selected model's test "
            "result is then optimistically biased. Prefer validation or "
            "cross_validation metrics for selection.",
            ", ".join(uses_test),
        )
    uses_train = [
        f"{i['process']}.{i['metric']}" for i in items
        if i["process"] == "train" and not _TRAIN_COST.fullmatch(str(i["metric"]).lower())
    ]
    if uses_train:
        logger.warning(
            "Model selection ranks on training-set quality metrics (%s): they are "
            "measured on the traces the model was fitted to and favour overfitted "
            "models. Prefer validation or cross_validation metrics for selection; "
            "use process train only for costs such as fit_time_s.",
            ", ".join(uses_train),
        )

    candidates = load_candidates(project_root)
    available = available_metrics(candidates)
    recorded = recorded_fields(candidates)
    hints = config_hints(judge_cfg, available, recorded)
    for hint in hints:
        pretty_print(f"[ModelJudge] Config hint: {hint}", color="yellow")
    if hints:
        print_metric_choices(available, candidates)
    eligible, rejected = [], []
    for candidate in candidates:
        reason = rejection_reason(candidate, judge_cfg, env)
        if reason is None:
            eligible.append(candidate)
        else:
            rejected.append({
                "model_name": candidate["model_name"],
                "model_hash": candidate["model_hash"],
                "reason": reason,
            })
    if not eligible:
        raise RuntimeError(
            f"No eligible model in {project_root} (env={env}); "
            f"{len(candidates)} candidate run(s), rejected: {rejected}"
            + "".join(f"\n  hint: {hint}" for hint in hints)
        )

    ranked = sorted(eligible, key=lambda c: sort_key(c, items))
    n_top = int((judge_cfg.get("ranking") or {}).get("N_top_candidates", len(ranked)))
    leaderboard = [
        summarize_candidate(c, judge_cfg, rank=i + 1, project_root=project_root)
        for i, c in enumerate(ranked[:n_top])
    ]
    selected = leaderboard[0]

    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "project": project,
        "env": env,
        "artifacts_dir": str(project_root / "artifacts"),
        "ranking": items,
        "selection_uses_test_data": bool(uses_test),
        "selection_uses_train_quality_metrics": uses_train,
        "n_candidates": len(candidates),
        "n_eligible": len(eligible),
        "selected": {**selected, "decision": judge_cfg.get("decision", "SELECTED")},
        "leaderboard": leaderboard,
        "rejected": rejected,
        "config_hints": hints,
        "available_metrics": available,
    }
    output_dir = project_root / judge_cfg.get("output_dir", "model_selection")
    result["output_dir"] = str(output_dir)
    write_outputs(result, output_dir, judge_cfg.get("outputs") or {})

    # Ranked models become job 0 'top_models' of graph_config.yaml.
    graph_cfg = project_root / "config" / "graph_config.yaml"
    if graph_cfg.is_file():
        try:
            result["graph_top_models"] = update_default_job_models(
                graph_cfg, "top_models", [item["model_name"] for item in leaderboard], replace=True,
            )
        except Exception as exc:
            logger.warning("graph_config.yaml job 0 top_models not updated: %s", exc)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Select the best trained model of a project.")
    parser.add_argument("--project", "-p", required=True)
    parser.add_argument("--env", "-e", default=None)
    parser.add_argument("--config", default=None, help="default: projects/<project>/config/model_judge.yaml")
    parser.add_argument("--list-json", action="store_true",
                        help="print the process/metric pairs the recorded runs offer, as JSON, and stop "
                             "(the panel's Judge asks with them); nothing is ranked or written")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR if args.list_json else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    config = args.config or f"projects/{args.project}/config/model_judge.yaml"
    if args.list_json:
        print(json.dumps(list_choices(config, project=args.project, env=args.env)))
        return
    result = run_model_judge(config, project=args.project, env=args.env)
    s = result["selected"]
    print("=" * 80)
    print("MODEL SELECTION")
    print("=" * 80)
    print(f"{'Selected model':<26}: {s['model_name']} ({s['model_hash']})")
    print(f"{'Primary':<26}: {s['primary']['process']}.{s['primary']['metric']} = {s['primary']['value']}")
    for tb in s["tie_breakers"]:
        print(f"{'Tie-breaker':<26}: {tb['process']}.{tb['metric']} = {tb['value']}")
    for key, value in s["reported_metrics"].items():
        print(f"{'Reported (not ranked)':<26}: {key} = {value}")
    if s.get("cv_verdict"):
        print(f"{'CV verdict':<26}: {s['cv_verdict']}")
    print(f"{'Final model':<26}: {s['artifacts']['final_model']}")
    print(f"{'Candidates':<26}: {result['n_eligible']} eligible / {result['n_candidates']} trained")
    print(f"{'Uses test data to select':<26}: {result['selection_uses_test_data']}")
    if result["selection_uses_train_quality_metrics"]:
        print(f"{'Uses train quality':<26}: {', '.join(result['selection_uses_train_quality_metrics'])}")
    print(f"{'Output':<26}: {result['output_dir']}")
    if result.get("graph_top_models"):
        print(f"{'Graph job 0 top_models':<26}: {', '.join(result['graph_top_models'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
