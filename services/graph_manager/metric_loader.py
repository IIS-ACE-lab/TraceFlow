# services/graph_manager/metric_loader.py
"""
Metric loader for TraceFlow GraphManager.

Purpose:
- Load scalar metric histories from metrics/scalars.
- Load array / curve metric artifacts from metrics/arrays.
- Issue terminal warning when missing.
- Never create placeholder images.
- Keep graph_config.yaml job-driven while keeping file discovery centralized here.

Author: Shraddha Mishra
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from services.side_channel_core.utils.seal import SEAL_KEY, check as check_seal, load_json, read_jsonl

from .metric_router import MetricRoute, route_metric
from .missing_plotter import MissingMetric, report_missing_metric


PROCESS_ALIASES = {
    "val": "validation",
    "valid": "validation",
    "validate": "validation",
}

ARRAY_EXTENSIONS = (".npy", ".npz", ".json", ".jsonl", ".csv", ".txt")

EPOCH_FILE_RE = re.compile(r"^ep(\d+)$", re.IGNORECASE)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []

    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)

    return out


def _epoch_sort_key(path: Path) -> tuple[int, str]:
    match = EPOCH_FILE_RE.match(path.stem)
    if match:
        return int(match.group(1)), path.name
    return -1, path.name

def _epoch_name_to_number(epoch_name: str) -> int:
    match = EPOCH_FILE_RE.match(str(epoch_name).strip())
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid epoch name: {epoch_name!r}. Expected format like ep0001.")


def _normalize_epoch_name(epoch: str | int) -> str:
    if isinstance(epoch, int):
        if epoch < 0:
            raise ValueError(f"Epoch must be non-negative, got {epoch}")
        return f"ep{epoch:04d}"

    text = str(epoch).strip()

    if not text:
        raise ValueError("Epoch name cannot be empty")

    if text.isdigit():
        return f"ep{int(text):04d}"

    if EPOCH_FILE_RE.match(text):
        return text.lower()

    raise ValueError(f"Invalid epoch name: {epoch!r}. Expected ep0001 or integer-like value.")


def _epoch_file_map(epoch_files: list[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}

    for path in epoch_files:
        out[_normalize_epoch_name(path.stem)] = path

    return out


def _as_curve(data: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)``: x is the trace count of each point.

    ``[N, 2]`` arrays are ``[trace_count, value]`` tables (rank curves are
    sampled every ``perform_attack.attack_step`` traces), so x is their first
    column. Arrays without an x column (1-D, or ``[R, N]`` repetitions averaged
    over R) are taken as one point per trace, x = 1..N.
    """
    if isinstance(data, pd.DataFrame):
        numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
        if not numeric_cols:
            raise ValueError("DataFrame contains no numeric columns")
        arr = data[numeric_cols[-1]].to_numpy(dtype=float)
    else:
        arr = np.asarray(data, dtype=float)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 0].astype(float), arr[:, 1].astype(float)
    if arr.ndim == 2:
        arr = np.nanmean(arr.astype(float), axis=0)
    if arr.ndim != 1:
        raise ValueError(f"Cannot summarize array with ndim={arr.ndim}")
    return np.arange(1, arr.size + 1, dtype=float), arr.astype(float)


def _summarize_curve(
    data: Any,
    *,
    summary_cfg: Mapping[str, Any],
) -> float:
    x, arr = _as_curve(data)

    if arr.size == 0:
        raise ValueError("Cannot summarize empty curve")

    summary_type = str(summary_cfg.get("type", "final_value")).strip().lower()

    if summary_type == "final_value":
        return float(arr[-1])

    if summary_type == "min_value":
        return float(np.nanmin(arr))

    if summary_type == "value_at_trace":
        trace_index = int(summary_cfg.get("trace_index", -1))
        if trace_index < 1:
            raise ValueError("summary.trace_index must be >= 1 for value_at_trace")
        # trace_index is a trace count; it must be a point of the curve,
        # never clamped or rounded to a neighbouring point.
        hits = np.flatnonzero(x == trace_index)
        if hits.size != 1:
            raise ValueError(
                f"summary.trace_index={trace_index} is not a point of the curve "
                f"(points {int(x[0])}, ..., {int(x[-1])}); use a multiple of "
                "perform_attack.attack_step within the attacked population"
            )
        return float(arr[int(hits[0])])

    if summary_type == "traces_to_threshold":
        # Stable crossing, as traces_to_ge_k: the first trace count from which
        # the value stays <= threshold to the end of the curve.
        threshold = float(summary_cfg.get("threshold", 1.0))
        held = np.logical_and.accumulate((arr <= threshold)[::-1])[::-1]
        hits = np.flatnonzero(held)
        if hits.size == 0:
            return float("nan")
        return float(x[int(hits[0])])

    raise ValueError(f"Unsupported summary type: {summary_type!r}")

def array_metric_roots(
    metrics_dir: Path,
    process: str,
    metric: str,
) -> list[Path]:
    process = normalize_process_name(process)

    roots = [
        metrics_dir / "arrays" / process / metric,
        metrics_dir / "arrays" / process / metric.lower(),
        metrics_dir / "arrays" / metric,
        metrics_dir / "arrays" / metric.lower(),
    ]

    return _dedupe_paths(roots)


def epoch_array_candidates(
    metrics_dir: Path,
    process: str,
    metric: str,
) -> list[Path]:
    supported_suffixes = {ext.lower() for ext in ARRAY_EXTENSIONS}
    found: list[Path] = []

    for root in array_metric_roots(metrics_dir, process, metric):
        if not root.exists() or not root.is_dir():
            continue

        for path in root.iterdir():
            if not path.is_file():
                continue

            if path.suffix.lower() not in supported_suffixes:
                continue

            if EPOCH_FILE_RE.match(path.stem):
                found.append(path)

    return sorted(_dedupe_paths(found), key=_epoch_sort_key)
@dataclass
class LoadedMetric:
    metric: str
    canonical_metric: str
    model: str
    process: str
    route: MetricRoute
    data: Any
    source_path: Path | None
    missing: MissingMetric | None = None

    @property
    def ok(self) -> bool:
        return self.missing is None and self.data is not None


def normalize_process_name(process: str) -> str:
    p = str(process).strip().lower()
    return PROCESS_ALIASES.get(p, p)


def _with_dependency_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    History rows keep what a requested metric depends on (e.g. fit_time_s of
    mean_fit_time_s) under "dependencies"; expose those values as columns, a
    metric of the same name first.
    """
    if "dependencies" not in df.columns:
        return df
    nested = pd.DataFrame(
        [value if isinstance(value, dict) else {} for value in df["dependencies"]],
        index=df.index,
    )
    df = df.drop(columns="dependencies")
    for column in nested.columns:
        if column not in df.columns:
            df[column] = nested[column]
    return df


def _read_json_or_jsonl(path: Path) -> pd.DataFrame:
    check_seal(path)  # utils/seal.py: a changed record is refused, not plotted
    if path.suffix.lower() == ".jsonl":
        return _with_dependency_columns(pd.read_json(path, lines=True).drop(columns=[SEAL_KEY], errors="ignore"))

    obj = load_json(path)
    try:
        return pd.read_json(io.StringIO(json.dumps(obj)))
    except Exception:

        if isinstance(obj, list):
            return pd.DataFrame(obj)

        if isinstance(obj, dict):
            if "history" in obj and isinstance(obj["history"], list):
                return pd.DataFrame(obj["history"])
            return pd.DataFrame([obj])

        raise ValueError(f"Unsupported JSON metric format: {path}")


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".json", ".jsonl"}:
        return _read_json_or_jsonl(path)

    if suffix == ".txt":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported table metric file: {path}")


def _load_array(path: Path) -> Any:
    suffix = path.suffix.lower()

    check_seal(path)  # utils/seal.py: a changed record is refused, not plotted

    if suffix == ".npy":
        return np.load(path, allow_pickle=True)

    if suffix == ".npz":
        return dict(np.load(path, allow_pickle=True))

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".jsonl":
        return pd.read_json(path, lines=True).drop(columns=[SEAL_KEY], errors="ignore")

    if suffix == ".json":
        obj = load_json(path)

        try:
            return np.asarray(obj)
        except Exception:
            return obj

    if suffix == ".txt":
        return np.loadtxt(path)

    raise ValueError(f"Unsupported array metric file: {path}")


def scalar_candidates(metrics_dir: Path, model: str, process: str) -> list[Path]:
    process = normalize_process_name(process)

    return [
        metrics_dir / "scalars" / f"{model}_metrics_{process}_history.jsonl",
        metrics_dir / "scalars" / f"{model}_metrics_{process}_history.json",
        metrics_dir / "scalars" / f"{process}_history.jsonl",
        metrics_dir / "scalars" / f"{process}_history.json",
        metrics_dir / "scalars" / f"{model}_metrics_{process}_history.csv",
        metrics_dir / "scalars" / f"{process}_history.csv",
    ]


def array_candidates(
    metrics_dir: Path,
    model: str,
    process: str,
    metric: str,
) -> list[Path]:
    candidates: list[Path] = []

    for root in array_metric_roots(metrics_dir, process, metric):
        for ext in ARRAY_EXTENSIONS:
            candidates.append(root / f"{model}_{metric}{ext}")
            candidates.append(root / f"{metric}{ext}")
            candidates.append(root.with_suffix(ext))

    return _dedupe_paths(candidates)

def load_scalar_history(
    *,
    metrics_dir: Path,
    model: str,
    process: str,
    metric: str,
) -> tuple[pd.DataFrame | None, Path | None, list[Path]]:
    checked = scalar_candidates(metrics_dir, model, process)

    for path in checked:
        if not path.exists():
            continue

        df = _load_table(path)

        if metric not in df.columns:
            continue

        if "epoch" not in df.columns:
            df = df.copy()
            df.insert(0, "epoch", range(1, len(df) + 1))
        else:
            # Re-running a stage appends rows (e.g. every test run writes
            # epoch 1 again); the latest record of an epoch is the current one.
            df = df.drop_duplicates("epoch", keep="last").sort_values("epoch")

        keep = ["epoch", metric]
        if "process" in df.columns:
            keep.append("process")
        if "model" in df.columns:
            keep.append("model")
        if "model_name" in df.columns:
            keep.append("model_name")

        return df[keep].copy(), path, checked

    return None, None, checked


def load_array_metric(
    *,
    metrics_dir: Path,
    model: str,
    process: str,
    metric: str,
    array_epoch: Mapping[str, Any] | None = None,
) -> tuple[Any | None, Path | None, list[Path]]:
    checked = array_candidates(metrics_dir, model, process, metric)

    # 1. Existing behavior: direct single-file artifact.
    for path in checked:
        if not path.exists() or not path.is_file():
            continue

        return _load_array(path), path, checked

    # 2. Epoch-sliced artifact directory.
    epoch_files = epoch_array_candidates(metrics_dir, process, metric)

    if not epoch_files:
        return None, None, checked

    checked = checked + epoch_files

    policy = dict(array_epoch or {})
    mode = str(policy.get("mode", "latest")).strip().lower()

    epoch_map = _epoch_file_map(epoch_files)

    if mode == "latest":
        latest = epoch_files[-1]
        return _load_array(latest), latest, checked

    if mode == "selected":
        requested = _normalize_epoch_name(policy.get("epoch", ""))
        selected = epoch_map.get(requested)

        if selected is None:
            available = ", ".join(sorted(epoch_map))
            raise FileNotFoundError(
                f"Requested array_epoch.epoch={requested!r} not found for metric={metric}. "
                f"Available epochs: {available}"
            )

        return _load_array(selected), selected, checked

    if mode == "overlay":
        requested_epochs = policy.get("epochs", []) or []
        if isinstance(requested_epochs, str):
            requested_epochs = [requested_epochs]

        if not requested_epochs:
            raise ValueError("array_epoch.mode='overlay' requires non-empty array_epoch.epochs")

        series: dict[str, Any] = {}

        for epoch in requested_epochs:
            epoch_name = _normalize_epoch_name(epoch)
            path = epoch_map.get(epoch_name)

            if path is None:
                available = ", ".join(sorted(epoch_map))
                raise FileNotFoundError(
                    f"Requested overlay epoch={epoch_name!r} not found for metric={metric}. "
                    f"Available epochs: {available}"
                )

            series[epoch_name] = _load_array(path)

        # source_path cannot represent multiple files, so use parent directory.
        return series, epoch_files[0].parent, checked

    if mode == "summary":
        summary_cfg = policy.get("summary", {}) or {}
        if not isinstance(summary_cfg, Mapping):
            raise ValueError("array_epoch.summary must be mapping/dict-like")

        rows: list[dict[str, Any]] = []

        for path in epoch_files:
            epoch_name = _normalize_epoch_name(path.stem)
            epoch_number = _epoch_name_to_number(epoch_name)
            data = _load_array(path)
            value = _summarize_curve(data, summary_cfg=summary_cfg)

            rows.append(
                {
                    "epoch": epoch_number,
                    metric: value,
                    "epoch_name": epoch_name,
                    "source_path": str(path),
                }
            )

        df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)

        # source_path cannot represent multiple files, so use parent directory.
        return df, epoch_files[0].parent, checked

    raise ValueError(
        f"Unsupported array_epoch.mode={mode!r}. "
        "Allowed: latest, selected, overlay, summary."
    )

def load_metric(
    *,
    metrics_dir: Path,
    model: str,
    process: str,
    metric: str,
    mode: str = "auto",
    array_epoch: Mapping[str, Any] | None = None,
) -> LoadedMetric:
    """
    Load one metric for one model/process pair.

    Missing behavior:
    - prints terminal warning
    - returns LoadedMetric(ok=False)
    """

    process = normalize_process_name(process)
    route = route_metric(metric, mode=mode)
    canonical = route.canonical_metric

    try:
        if route.storage == "scalar_history":
            data, source_path, checked = load_scalar_history(
                metrics_dir=Path(metrics_dir),
                model=model,
                process=process,
                metric=canonical,
            )

        elif route.storage in {"array_curve", "array_table"}:
            data, source_path, checked = load_array_metric(
                metrics_dir=Path(metrics_dir),
                model=model,
                process=process,
                metric=canonical,
                array_epoch=array_epoch,
            )

        else:
            data, source_path, checked = None, None, []

        if data is None:
            missing = report_missing_metric(
                metric=canonical,
                model=model,
                process=process,
                reason=f"no metric data found for route storage={route.storage}",
                expected_paths=[str(p) for p in checked],
            )

            return LoadedMetric(
                metric=metric,
                canonical_metric=canonical,
                model=model,
                process=process,
                route=route,
                data=None,
                source_path=None,
                missing=missing,
            )

        return LoadedMetric(
            metric=metric,
            canonical_metric=canonical,
            model=model,
            process=process,
            route=route,
            data=data,
            source_path=source_path,
            missing=None,
        )

    except Exception as exc:
        missing = report_missing_metric(
            metric=canonical,
            model=model,
            process=process,
            reason=f"failed to load metric: {exc}",
            expected_paths=[],
        )

        return LoadedMetric(
            metric=metric,
            canonical_metric=canonical,
            model=model,
            process=process,
            route=route,
            data=None,
            source_path=None,
            missing=missing,
        )


# ----------------------------------------------------------------------
# Sources: metrics/{attack_metrics, cross_validation, key_recovery}
# ----------------------------------------------------------------------

SOURCES = ("attack_metrics", "cross_validation", "key_recovery")

# Per-fold arrays that are pooled across folds instead of averaged.
_POOLED_ACROSS_FOLDS = {"per_attack_curves", "final_ranks"}


class CrossValidationIncomplete(ValueError):
    """A run's cross-validation is not one finished set of folds a figure can average."""


def fold_metric_dirs(metrics_root: Path, *, strict: bool = True) -> list[Path]:
    """metrics/cross_validation/fold_NN directories of the run's finished cross-validation, in fold order.

    The run's cross_validation/manifest.yaml decides which folds count (TODO §19): its status must be
    completed, every one of its contract.n_folds folds completed and assessed at one epoch, each with
    its fold_NN directory. A fold directory the manifest does not list (left by an earlier run of the
    hash) is never read. strict=False (the panel's menus) lists the manifest's completed folds without
    refusing an unfinished run. A run without a manifest (written before it existed) keeps every
    fold_NN directory."""
    root = Path(metrics_root) / "cross_validation"
    if not root.is_dir():
        return []
    found = {p.name: p for p in root.glob("fold_*") if p.is_dir()}
    manifest_path = Path(metrics_root).parent / "cross_validation" / "manifest.yaml"
    if not manifest_path.is_file():
        return [found[name] for name in sorted(found)]
    import yaml
    from services.side_channel_core.utils.seal import strip as strip_seal

    manifest = strip_seal(yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {})
    folds = [f for f in manifest.get("folds") or [] if isinstance(f, dict)]
    done = [f for f in folds if f.get("status") == "completed"]
    problems = []
    if manifest.get("status") != "completed":
        problems.append(f"its status is {manifest.get('status')!r}, not 'completed'")
    n_folds = int((manifest.get("contract") or {}).get("n_folds") or len(folds))
    if len(done) != n_folds:
        unfinished = [f"{f.get('tag')} ({f.get('status')})" for f in folds if f.get("status") != "completed"]
        problems.append(f"{len(done)} of {n_folds} folds completed" + (f": {', '.join(unfinished)}" if unfinished else ""))
    epochs = {(f.get("result") or {}).get("assessed_epoch") for f in done}
    if len(epochs) > 1:
        problems.append(f"its folds were assessed at different epochs: {sorted(epochs, key=str)}")
    absent = [str(f.get("tag")) for f in done if str(f.get("tag")) not in found]
    if absent:
        problems.append(f"no metrics directory for completed fold(s) {', '.join(absent)}")
    if strict and problems:
        raise CrossValidationIncomplete(
            f"{manifest_path.parent.parent.name}: the cross-validation is not finished ({'; '.join(problems)}); "
            "a figure averages only the folds of a completed cross-validation"
        )
    return [found[str(f.get("tag"))] for f in sorted(done, key=lambda f: int(f.get("fold_number") or 0))
            if str(f.get("tag")) in found]


# Per-epoch values a fold keeps in its fold_result.yaml trajectory only (they
# are not metrics, so no history row has them): metric -> its process.
TRAJECTORY_METRICS = {"training_objective": "train", "validation_objective": "validation"}


# One value per fold, kept in fold_result.yaml whatever metrics_to_log says:
# metric -> its process. training_time_s is the time of the optimisation passes
# of the whole fold (the stopwatch of fit_time_s, read at the fold's last epoch).
FOLD_RESULT_SCALARS = {"training_time_s": "train"}


def _trajectory_part(fold: Path, process: str, metric: str) -> pd.DataFrame | None:
    """
    [epoch, metric] of one fold from <run>/cross_validation/folds/<fold>/fold_result.yaml.

    training_objective is the loss the optimiser minimised (trainer.loss.type),
    mean per trace over the epoch; validation_objective the same loss on the
    held-out traces. None when this fold has no such trajectory.
    """
    if process not in (TRAJECTORY_METRICS.get(metric), FOLD_RESULT_SCALARS.get(metric)):
        return None
    path = fold.parent.parent.parent / "cross_validation" / "folds" / fold.name / "fold_result.yaml"
    if not path.is_file():
        return None
    import yaml

    check_seal(path)
    record = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if metric in FOLD_RESULT_SCALARS:
        value, epoch = record.get(metric), record.get("epochs_completed")
        if value is None or epoch is None:
            return None
        return pd.DataFrame({"epoch": [int(epoch)], metric: [float(value)]})
    trajectory = record.get("trajectory") or {}
    epochs, values = trajectory.get("epochs"), trajectory.get(metric)
    if not epochs or not values or len(epochs) != len(values):
        return None
    return pd.DataFrame({"epoch": epochs, metric: pd.to_numeric(pd.Series(values), errors="coerce")})


def _strict_mean(values: pd.Series) -> float:
    # A fold without a value (e.g. GE threshold never reached) makes the
    # fold mean undefined; averaging only the folds that have one would bias it.
    return float(values.mean()) if values.notna().all() else float("nan")


def combine_folds(metric: str, parts: Mapping[str, Any], *, with_std: bool = True) -> Any:
    """
    Fold-level data -> one cross-validation series.

    DataFrame (per-epoch scalars): mean over folds per epoch, plus
        <metric>_std (sample std) and n_folds.
    [N, 2] curve: mean rank curve over folds (Eq. 7) as [N, 3] = x, mean, std.
    with_std=False (style.std_band.enabled: false) leaves the std out: no
        <metric>_std column, and the curve is [N, 2] = x, mean.
    per_attack_curves / final_ranks: pooled across folds.
    dict (array_epoch overlay): combined per epoch.
    """
    values = list(parts.values())
    first = values[0]

    if isinstance(first, pd.DataFrame):
        df = pd.concat(values, ignore_index=True)
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        grouped = df.groupby("epoch")[metric]
        out = grouped.agg(_strict_mean).rename(metric).to_frame()
        if with_std:
            out[f"{metric}_std"] = grouped.std(ddof=1)
        out["n_folds"] = grouped.size()
        return out.reset_index()

    if isinstance(first, Mapping):
        return {
            key: combine_folds(metric, {fold: part[key] for fold, part in parts.items()}, with_std=with_std)
            for key in first
        }

    arrays = [np.asarray(v, dtype=float) for v in values]

    if metric in _POOLED_ACROSS_FOLDS:
        if metric == "final_ranks":
            return np.concatenate([a.reshape(-1) for a in arrays])
        return np.concatenate([a.reshape((-1,) + a.shape[-2:]) for a in arrays], axis=0)

    curves = [np.squeeze(a) for a in arrays]
    ref = curves[0]
    if ref.ndim != 2 or ref.shape[1] < 2:
        raise ValueError(f"fold data for {metric} is not an [N, 2] curve: shape={ref.shape}")
    for curve in curves[1:]:
        if curve.shape != ref.shape or not np.array_equal(curve[:, 0], ref[:, 0]):
            raise ValueError(f"fold curves of {metric} use different trace grids; cannot average")
    ys = np.stack([c[:, 1] for c in curves])
    if not with_std:
        return np.column_stack([ref[:, 0], ys.mean(axis=0)])
    std = ys.std(axis=0, ddof=1) if len(ys) > 1 else np.zeros(ys.shape[1])
    return np.column_stack([ref[:, 0], ys.mean(axis=0), std])


def _key_recovery_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one key-recovery record into numeric columns."""
    epoch = record.get("epoch")
    row: dict[str, Any] = {"epoch": 1 if epoch is None else epoch}
    for section in ("key", "recovery"):
        for key, value in (record.get(section) or {}).items():
            if isinstance(value, bool):
                row[key] = int(value)
            elif value is None or isinstance(value, (int, float)):
                row[key] = value
    margin = (record.get("diagnostics") or {}).get("top1_top2_score_margin")
    if margin is not None:
        row["top1_top2_score_margin"] = margin
    return row


def _best_is_key(record: Mapping[str, Any]) -> int | None:
    """1 when the record's best key hypothesis is the true key, 0 when not, None when the record names either not
    (the colour of a run's dot in a key-confidence strip, plotters.plot_strip)."""
    key = record.get("key") or {}
    if key.get("top_key_hypothesis") is None or key.get("true_key_byte") is None:
        return None
    return int(key["top_key_hypothesis"] == key["true_key_byte"])


def _key_recovery_files(kr_dir: Path, process: str) -> tuple[Path | None, Path | None]:
    """(history.jsonl, single-record json) for one process."""
    history = kr_dir / f"{process}_history.jsonl"
    single = next(
        (p for p in (kr_dir / f"{process}_best.json", kr_dir / f"{process}.json") if p.is_file()),
        None,
    )
    return (history if history.is_file() else None), single


def _key_candidate_scores(record: Mapping[str, Any]) -> dict[str, Any]:
    candidates = (record.get("diagnostics") or {}).get("top_key_candidates") or []
    if not candidates:
        raise ValueError("record has no diagnostics.top_key_candidates")
    ordered = sorted(candidates, key=lambda c: int(c["key_guess"]))
    return {
        "key_guess": np.array([int(c["key_guess"]) for c in ordered]),
        "score": np.array([float(c["score"]) for c in ordered]),
        "true_key": (record.get("key") or {}).get("true_key_byte"),
        "true_key_rank": (record.get("key") or {}).get("final_true_key_rank"),
    }


def _load_key_recovery_dir(kr_dir: Path, process: str, metric: str) -> tuple[Any, list[Path]]:
    history, single = _key_recovery_files(kr_dir, process)
    checked = [kr_dir / f"{process}_history.jsonl", kr_dir / f"{process}_best.json", kr_dir / f"{process}.json"]

    if metric == "key_candidate_scores":
        if single is None:
            return None, checked
        return _key_candidate_scores(load_json(single)), checked

    if history is not None:
        records = read_jsonl(history)
    elif single is not None:
        records = [load_json(single)]
    else:
        return None, checked

    rows = [_key_recovery_row(record) for record in records]
    # the score margin carries best_is_key beside it (plotters.plot_strip); every other metric is [epoch, metric]
    keep = ["epoch", metric]
    if metric == "top1_top2_score_margin" and any(_best_is_key(record) is not None for record in records):
        for row, record in zip(rows, records):
            row["best_is_key"] = _best_is_key(record)
        keep.append("best_is_key")
    df = pd.DataFrame(rows)
    if metric not in df.columns:
        return None, checked
    df = df.drop_duplicates("epoch", keep="last").sort_values("epoch")
    return df[keep].reset_index(drop=True), checked


def _missing(metric: str, route: MetricRoute, model: str, process: str, reason: str, checked: list[Path]) -> LoadedMetric:
    return LoadedMetric(
        metric=metric,
        canonical_metric=route.canonical_metric,
        model=model,
        process=process,
        route=route,
        data=None,
        source_path=None,
        missing=report_missing_metric(
            metric=route.canonical_metric,
            model=model,
            process=process,
            reason=reason,
            expected_paths=[str(p) for p in checked],
        ),
    )


def load_source_metric(
    *,
    source: str,
    metrics_dir: Path,
    model: str,
    process: str,
    metric: str,
    mode: str = "auto",
    array_epoch: Mapping[str, Any] | None = None,
    key_recovery_dir: Path | None = None,
    with_std: bool = True,
) -> LoadedMetric:
    """
    Load one metric of one model/process from a metrics source.

    metrics_dir is the model's metrics/attack_metrics directory (paths.metrics);
    the other sources live next to it under metrics/. with_std is passed to
    combine_folds: the std over folds is computed only when a band is drawn.
    """
    process = normalize_process_name(process)
    metrics_dir = Path(metrics_dir)
    metrics_root = metrics_dir.parent
    route = route_metric(metric, mode=mode)
    canonical = route.canonical_metric

    if source == "attack_metrics":
        return load_metric(
            metrics_dir=metrics_dir, model=model, process=process,
            metric=metric, mode=mode, array_epoch=array_epoch,
        )

    folds = fold_metric_dirs(metrics_root)

    try:
        if source == "cross_validation":
            if not folds:
                return _missing(metric, route, model, process,
                                "no fold directories (cross_validation disabled for this model?)",
                                [metrics_root / "cross_validation" / "fold_NN"])
            inner = "mean_attack_rank_curve" if route.plotter == "fold_curves" else metric
            parts: dict[str, Any] = {}
            for fold in folds:
                # The objectives are kept in the fold's trajectory, not in its
                # metric history: read them there first (no missing-history notice).
                part = _trajectory_part(fold, process, canonical)
                if part is not None:
                    parts[fold.name] = part
                    continue
                loaded = load_metric(
                    metrics_dir=fold, model=model, process=process,
                    metric=inner, mode=mode, array_epoch=array_epoch,
                )
                if not loaded.ok:
                    return LoadedMetric(metric, canonical, model, process, route, None, None, loaded.missing)
                parts[fold.name] = loaded.data
            if route.plotter == "fold_curves":
                if not all(isinstance(v, np.ndarray) for v in parts.values()):
                    return _missing(metric, route, model, process,
                                    "fold_rank_curves needs array_epoch.mode latest or selected", [])
                data = parts
            else:
                data = combine_folds(canonical, parts, with_std=with_std)
            return LoadedMetric(metric, canonical, model, process, route, data,
                                metrics_root / "cross_validation")

        if source == "key_recovery":
            kr_dir = Path(key_recovery_dir) if key_recovery_dir else metrics_root / "key_recovery"
            data, checked = _load_key_recovery_dir(kr_dir, process, canonical)
            if data is None and process == "validation" and folds:
                # CV: validation key recovery is per fold (fold_NN/key_recovery).
                parts = {}
                for fold in folds:
                    part, fold_checked = _load_key_recovery_dir(fold / "key_recovery", process, canonical)
                    checked += fold_checked
                    if part is None:
                        parts = {}
                        break
                    parts[fold.name] = part
                if parts:
                    data = parts if canonical == "key_candidate_scores" else combine_folds(canonical, parts, with_std=with_std)
            if data is None:
                return _missing(metric, route, model, process,
                                "no key-recovery record with this field", checked)
            return LoadedMetric(metric, canonical, model, process, route, data, kr_dir)

    except Exception as exc:
        return _missing(metric, route, model, process, f"failed to load metric: {exc}", [])

    raise ValueError(f"Unknown metrics source={source!r}. Allowed: {list(SOURCES)}")
