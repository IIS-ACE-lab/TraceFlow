# services/side_channel_core/utils/metrics_logger.py
"""
Metrics logger for Side-Channel Analysis (SCA) — Option A (JSONL + array artifacts)

Writes ONE JSONL file per (model, process) under metrics/scalars/.
Saves array-like metrics as .npy under metrics/arrays/<process>/<metric>/epXXXX.npy
and stores pointers in the JSONL record.

A row holds its header (epoch, timestamp, model_name, model_hash, process,
env), the metrics metrics_to_log names, and, under "dependencies", the values
a requested metric is derived from or read with (sca.metrics.split_requested).
Nothing else is stored. Readers take a dependency as a column through
flat_history_row().

Author: Shraddha Mishra
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from utils.logger import get_logger
from utils.path_utils import resolve_path
from utils.timestamps import now_iso
from utils.seal import SEAL_KEY, check as check_seal, seal_jsonl_rows, seal_npy_file

logger = get_logger("Metrics")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _now_iso() -> str:
    return now_iso()


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (int, float, str, bool)) or v is None or isinstance(v, np.generic)


def _to_numpy(v: Any) -> np.ndarray:
    """
    Convert supported array-like values to np.ndarray.
    Supports: np.ndarray, torch.Tensor (without importing torch), array-likes.
    """
    if isinstance(v, np.ndarray):
        return v

    # torch.Tensor without importing torch
    if hasattr(v, "detach") and hasattr(v, "cpu") and hasattr(v, "numpy"):
        return v.detach().cpu().numpy()

    # fallback: try numpy conversion
    return np.asarray(v)


def _safe_metric_dirname(name: str) -> str:
    """
    Metric key to folder name:
    - keeps alnum, '-', '_'
    - replaces others with '_'
    """
    s = str(name)
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _epoch_tag(epoch: int) -> str:
    return f"ep{int(epoch):04d}"


def _upsert_epoch_row(path: Path, payload: Dict[str, Any]) -> None:
    """
    Write one epoch row, replacing instead of appending.

    Rows at or after this epoch belong to an earlier run of the same model
    (a rerun with FORCE, a re-test) and are dropped first, so each file holds
    one row per epoch of the latest run. A resumed run (starting at epoch k)
    keeps its rows before k. The file is rewritten atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    epoch = int(payload["epoch"])
    kept = []
    if path.exists():
        # The rows carried forward are re-signed below: a changed row must stop
        # here instead of being sealed again (utils/seal.py).
        check_seal(path, cache=False)
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                row_epoch = int(row.get("epoch"))
            except (ValueError, TypeError, AttributeError):
                continue  # unreadable legacy line
            if row_epoch < epoch:
                kept.append(row)
    kept.append(payload)
    kept = [json.dumps(row, separators=(",", ":"), ensure_ascii=False) for row in seal_jsonl_rows(kept, path)]
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _drop_later_arrays(metrics_dir: Path, process: str, epoch: int) -> list[Path]:
    """
    The process's arrays of epochs after `epoch` (metrics/arrays/<process>/<metric>/epNNNN.npy): written by an
    earlier run of the same model (a FORCE rerun, a resumed run) whose history rows from `epoch` on
    _upsert_epoch_row drops, so the arrays follow the rows: kept before `epoch`, rewritten at it, gone after it.
    Without this a reader taking the latest epoch's array (graph_manager array_epoch: latest) read the old run's.
    Returns the files removed.
    """
    removed: list[Path] = []
    root = metrics_dir / "arrays" / str(process)
    if not root.is_dir():
        return removed
    for metric_dir in root.iterdir():
        if not metric_dir.is_dir():
            continue
        for file in metric_dir.glob("ep*.npy"):
            stem = file.stem[2:]
            if stem.isdigit() and int(stem) > int(epoch):
                file.unlink(missing_ok=True)
                removed.append(file)
    return removed


def _save_array_artifact(
    *,
    metrics_dir: Path,
    process: str,
    metric_name: str,
    epoch: int,
    arr: np.ndarray,
) -> Dict[str, Any]:
    """
    Saves array as .npy under:
        metrics/arrays/<process>/<metric_name>/epXXXX.npy

    Returns JSON-serializable descriptor containing path relative to metrics_dir.
    """
    metric_folder = _safe_metric_dirname(metric_name)
    rel_path = Path("arrays") / str(process) / metric_folder / f"{_epoch_tag(epoch)}.npy"
    abs_path = metrics_dir / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as .npy (fast + preserves dtype), then sign it in a trailer np.load ignores
    np.save(str(abs_path), arr)
    seal_npy_file(abs_path)

    return {
        "_type": "ndarray",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        # store path relative to metrics/ for portability
        "path": str(rel_path).replace("\\", "/"),
    }


def _serialize_metric_value(
    *,
    metrics_dir: Path,
    process: str,
    epoch: int,
    key: str,
    value: Any,
) -> Any:
    """
    Scalars inline, arrays saved to artifacts folder with a pointer dict.
    """
    if _is_scalar(value):
        return value.item() if isinstance(value, np.generic) else value

    # Try treat as array-like
    try:
        arr = _to_numpy(value)
        if isinstance(arr, np.ndarray) and arr.ndim >= 1:
            return _save_array_artifact(
                metrics_dir=metrics_dir,
                process=process,
                metric_name=key,
                epoch=epoch,
                arr=arr,
            )
    except Exception as e:
        logger.warning("Failed to save array metric '%s': %s", key, e, exc_info=True)

    # Fallback: stringify non-serializable
    return str(value)


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------

def flat_history_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """A history row with its "dependencies" as columns (a metric of the same name wins)."""
    dependencies = row.get("dependencies")
    flat = {key: value for key, value in row.items() if key not in ("dependencies", SEAL_KEY)}
    if isinstance(dependencies, dict):
        for key, value in dependencies.items():
            flat.setdefault(key, value)
    return flat


def append_metrics_row(
    cfg: Dict[str, Any],
    results: Dict[str, Any],
    process: str,
    epoch: int,
) -> None:
    """
    Write one metrics record as a JSON line (one row per epoch; a rerun or
    re-test of the same model replaces its earlier rows, see _upsert_epoch_row).

    Output:
        metrics/scalars/{model_name}_metrics_{process}_history.jsonl

    Also saves arrays under:
        metrics/arrays/<process>/<metric>/epXXXX.npy

    and stores pointers in JSONL.
    """
    # Resolve metrics root directory: .../generated_models/<model>/metrics
    metrics_dir = Path(resolve_path(cfg["paths"]["metrics"], cfg))
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Scalars directory + file
    scalars_dir = metrics_dir / "scalars"
    scalars_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model_name"]
    out_file = scalars_dir / f"{model_name}_metrics_{process}_history.jsonl"

    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "timestamp": _now_iso(),
        "model_name": model_name,
        "model_hash": cfg.get("model_hash"),
        "process": str(process),
        "env": cfg.get("env"),
    }

    # Only what metrics_to_log names is a column; what those metrics depend
    # on goes under "dependencies". Arrays become pointers + artifacts.
    from sca.metrics import split_requested

    metrics, dependencies = split_requested(results or {}, cfg)

    def _serialized(values: Dict[str, Any]) -> Dict[str, Any]:
        return {
            str(key): _serialize_metric_value(
                metrics_dir=metrics_dir,
                process=process,
                epoch=epoch,
                key=str(key),
                value=value,
            )
            for key, value in values.items()
        }

    if not metrics and not dependencies:
        # No requested metric exists for this process at this epoch (e.g. only
        # rank metrics for the never-attacked training traces): no row.
        return
    payload.update(_serialized(metrics))
    if dependencies:
        payload["dependencies"] = _serialized(dependencies)

    stale = _drop_later_arrays(metrics_dir, str(process), int(epoch))     # the rows dropped below: their arrays too
    if stale:
        logger.info("Removed %d array file(s) of %s epochs after %d left by an earlier run", len(stale), process, epoch)
    _upsert_epoch_row(out_file, payload)

    logger.debug(
        "Metrics logged | model=%s process=%s epoch=%d → %s",
        model_name,
        process,
        epoch,
        out_file.name,
    )
