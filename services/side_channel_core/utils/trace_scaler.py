# services/side_channel_core/utils/trace_scaler.py
"""
Trace scalers fitted on the training traces (data[].trace_normalization.mode: fitted).

    trace_normalization:
      mode: fitted
      steps: [standard, minmax]      # applied in this order, per sample point (column)
      feature_range: [0, 1]          # of the minmax step

Each step is a scikit-learn scaler (sklearn.preprocessing.StandardScaler,
MinMaxScaler), chained in a sklearn Pipeline and fitted on the traces the model
trains on, never on the traces it is evaluated on:

  - a cross-validation fold: the fold's training rows; its validation rows are
    transformed with that fit (utils/data_split.get_cross_validation_loaders);
  - the refit (CV) or the training split (no CV): all of train.h5; this is the
    run's final scaler, stored sealed as scaler.json in the run's data folder,
    and the test (utils/data_split.get_test_loader, src/sca/test.py) reads it.

A fold's fit is not stored: fold models keep no weights, and the fit follows
from the fold's recorded rows. The traces stay unscaled in the generated files;
the scaler is applied when a dataset loads them (ASCADDataset.set_trace_scaler).

Author: Shraddha Mishra
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.atomic_io import atomic_write_text
from utils.logger import get_logger

logger = get_logger("trace_scaler")

SCALER_FILE = "scaler.json"
FORMAT_VERSION = 1
STEPS = ("standard", "minmax")
# The fitted attributes each sklearn scaler's transform reads, stored to rebuild it exactly.
_FITTED = {
    "standard": ("mean_", "var_", "scale_", "n_samples_seen_", "n_features_in_"),
    "minmax": ("min_", "scale_", "data_min_", "data_max_", "data_range_", "n_samples_seen_", "n_features_in_"),
}


def fitted_policy(cfg: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """The active entry's fitted-scaling policy, or None when its traces are not fitted-scaled."""
    from sca.ascad_data_loader import _get_trace_normalization_cfg

    if not isinstance(cfg, Mapping) or cfg.get("data") is None:
        return None                      # no data entry: nothing is scaled
    policy = _get_trace_normalization_cfg(cfg)
    return policy if policy.get("mode") == "fitted" else None


def make_pipeline(policy: Mapping[str, Any]) -> Pipeline:
    steps = []
    for name in policy["steps"]:
        if name == "standard":
            steps.append(("standard", StandardScaler()))
        else:
            steps.append(("minmax", MinMaxScaler(feature_range=tuple(policy["feature_range"]))))
    return Pipeline(steps)


def fit(policy: Mapping[str, Any], traces: np.ndarray) -> Pipeline:
    """The policy's sklearn Pipeline fitted on these (training) traces."""
    return make_pipeline(policy).fit(np.asarray(traces, dtype=np.float64))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"array": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _restored(value: Any) -> Any:
    if isinstance(value, dict) and "array" in value:
        return np.asarray(value["array"], dtype=value["dtype"])
    return value


def to_record(pipeline: Pipeline, *, policy: Mapping[str, Any], fitted_on: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "policy": dict(policy),
        "fitted_on": dict(fitted_on),
        "steps": [
            {"name": name, "class": type(step).__name__, "params": step.get_params(),
             "fitted": {attr: _jsonable(getattr(step, attr)) for attr in _FITTED[name]}}
            for name, step in pipeline.steps
        ],
    }


def from_record(record: Mapping[str, Any]) -> Pipeline:
    """The stored scaler rebuilt as the same sklearn objects with their fitted attributes."""
    if record.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"scaler format_version {record.get('format_version')!r} is not {FORMAT_VERSION}")
    pipeline = make_pipeline(record["policy"])
    for (name, step), stored in zip(pipeline.steps, record["steps"]):
        if stored["name"] != name:
            raise ValueError(f"scaler step {stored['name']!r} does not match the policy's {name!r}")
        step.set_params(**stored["params"])
        for attr, value in stored["fitted"].items():
            setattr(step, attr, _restored(value))
    return pipeline


def scaler_path(cfg: Mapping[str, Any]) -> Path:
    """scaler.json in the run's data folder (paths.data), beside train.h5 and test.h5."""
    from utils.path_utils import resolve_path

    folder = resolve_path(cfg["paths"]["data"], dict(cfg), model_name=str(cfg["model_name"]))
    return Path(folder) / SCALER_FILE


def save(pipeline: Pipeline, path: Path, *, policy: Mapping[str, Any], fitted_on: Mapping[str, Any]) -> Path:
    record = to_record(pipeline, policy=policy, fitted_on=fitted_on)
    return atomic_write_text(path, json.dumps(record, indent=1) + "\n")


def load(path: Path) -> Pipeline:
    from utils.seal import check as check_seal

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing: this run's traces are scaled by a scaler fitted on its training "
            "traces (trace_normalization.mode: fitted), and its test needs that scaler"
        )
    check_seal(path)
    return from_record(json.loads(path.read_text(encoding="utf-8")))


def scale_training(cfg: Mapping[str, Any], fit_on, rows: Optional[Sequence[int]], apply_to: Sequence,
                   *, final: bool, label: str) -> Optional[Pipeline]:
    """Fit the policy on fit_on's traces (rows of it, or all), attach it to every dataset in
    apply_to; the final fit (refit or no-CV training split) is also stored as scaler.json."""
    policy = fitted_policy(cfg)
    if policy is None:
        return None
    traces = fit_on.unscaled_traces()
    selected = traces if rows is None else traces[np.asarray(rows, dtype=np.int64)]
    pipeline = fit(policy, selected)
    for dataset in apply_to:
        dataset.set_trace_scaler(pipeline)
    logger.info("Trace scaler %s fitted on %d %s traces (%s)", "+".join(policy["steps"]),
                len(selected), label, "stored" if final else "fold, not stored")
    if final:
        save(pipeline, scaler_path(cfg), policy=policy, fitted_on={"split": label, "rows": int(len(selected))})
    return pipeline


def scale_test(cfg: Mapping[str, Any], dataset) -> Optional[Pipeline]:
    """Attach the run's stored final scaler to a test dataset."""
    if fitted_policy(cfg) is None:
        return None
    pipeline = load(scaler_path(cfg))
    dataset.set_trace_scaler(pipeline)
    return pipeline


__all__ = ["SCALER_FILE", "STEPS", "fit", "fitted_policy", "from_record", "load", "make_pipeline", "save",
           "scale_test", "scale_training", "scaler_path", "to_record"]
