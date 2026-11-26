# services/mlops/schemas/visualization.py
# ============================================================
# Visualization schema
# - Validates auto_mlops.yaml: visualization.*
# - TensorBoard + static report outputs (open-source only)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mlops.schemas.common import (
    forbid_unknown_keys,
    normalize_bool,
    normalize_dict,
    normalize_list,
    normalize_str,
)


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

_ALLOWED_REPORT_FORMATS = {"html", "csv", "json"}
_ALLOWED_PLOT_TYPES = {
    "metrics_curves",
    "pareto_front",
    "robustness_matrix",
    "calibration_optional",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _non_empty_str(v: Any, *, ctx: str, default: str = "") -> str:
    s = normalize_str(v, default=default).strip()
    if not s:
        raise ValueError(f"[schemas/visualization] {ctx} must be non-empty string")
    return s


def _as_optional_int(v: Any, *, ctx: str, min_value: Optional[int] = None) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"[schemas/visualization] {ctx} must be int (got {type(v)})")
    if min_value is not None and v < min_value:
        raise ValueError(f"[schemas/visualization] {ctx} must be >= {min_value} (got {v})")
    return int(v)


def _validate_formats(v: Any, *, ctx: str) -> List[str]:
    items = normalize_list(v)
    if not items:
        raise ValueError(f"[schemas/visualization] {ctx} must be non-empty list")
    out: List[str] = []
    for i, x in enumerate(items):
        if not isinstance(x, str) or not x.strip():
            raise ValueError(f"[schemas/visualization] {ctx}[{i}] must be non-empty string")
        f = x.strip().lower()
        if f not in _ALLOWED_REPORT_FORMATS:
            raise ValueError(
                f"[schemas/visualization] {ctx}[{i}]='{f}' invalid. Allowed: {sorted(_ALLOWED_REPORT_FORMATS)}"
            )
        out.append(f)
    # de-dup preserve order
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _validate_plots(v: Any, *, ctx: str) -> List[str]:
    items = normalize_list(v)
    if not items:
        raise ValueError(f"[schemas/visualization] {ctx} must be non-empty list")
    out: List[str] = []
    for i, x in enumerate(items):
        if not isinstance(x, str) or not x.strip():
            raise ValueError(f"[schemas/visualization] {ctx}[{i}] must be non-empty string")
        p = x.strip()
        if p not in _ALLOWED_PLOT_TYPES:
            raise ValueError(
                f"[schemas/visualization] {ctx}[{i}]='{p}' invalid. Allowed: {sorted(_ALLOWED_PLOT_TYPES)}"
            )
        out.append(p)
    # de-dup preserve order
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


# ------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------

@dataclass(frozen=True)
class TensorboardSpec:
    enabled: bool = False
    log_dir: Optional[str] = None
    flush_secs: Optional[int] = None
    record_hparams: Optional[bool] = None
    record_model_graph: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StaticReportsSpec:
    enabled: bool = False
    formats: List[str] = field(default_factory=list)
    include_plots: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualizationSpec:
    tensorboard: TensorboardSpec = field(default_factory=TensorboardSpec)
    static_reports: StaticReportsSpec = field(default_factory=StaticReportsSpec)
    extra: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------
# Validators
# ------------------------------------------------------------

def validate_tensorboard_block(block: Any, *, ctx: str = "visualization.tensorboard", strict: bool = False) -> TensorboardSpec:
    d = normalize_dict(block)

    allowed = {"enabled", "log_dir", "flush_secs", "record_hparams", "record_model_graph"}
    if strict:
        forbid_unknown_keys(d, allowed, ctx=ctx)

    enabled = normalize_bool(d.get("enabled"), default=False)

    log_dir = d.get("log_dir", None)
    if log_dir is not None:
        log_dir = _non_empty_str(log_dir, ctx=f"{ctx}.log_dir")

    flush_secs = d.get("flush_secs", None)
    flush_secs_i = _as_optional_int(flush_secs, ctx=f"{ctx}.flush_secs", min_value=1) if "flush_secs" in d else None

    rh = d.get("record_hparams", None)
    record_hparams = normalize_bool(rh, default=None) if "record_hparams" in d else None

    rmg = d.get("record_model_graph", None)
    record_model_graph = normalize_bool(rmg, default=None) if "record_model_graph" in d else None

    extra = {k: v for k, v in d.items() if k not in allowed}

    return TensorboardSpec(
        enabled=enabled,
        log_dir=log_dir,
        flush_secs=flush_secs_i,
        record_hparams=record_hparams,
        record_model_graph=record_model_graph,
        extra=extra,
    )


def validate_static_reports_block(block: Any, *, ctx: str = "visualization.static_reports", strict: bool = False) -> StaticReportsSpec:
    d = normalize_dict(block)

    allowed = {"enabled", "formats", "include_plots"}
    if strict:
        forbid_unknown_keys(d, allowed, ctx=ctx)

    enabled = normalize_bool(d.get("enabled"), default=False)

    formats: List[str] = []
    if "formats" in d and d["formats"] is not None:
        formats = _validate_formats(d["formats"], ctx=f"{ctx}.formats")

    include_plots: List[str] = []
    ip = d.get("include_plots", None)
    if ip is not None:
        ipd = normalize_dict(ip)
        # include_plots:
        #   - metrics_curves
        #   - pareto_front
        # ... OR legacy:
        # include_plots: [ ... ]
        if isinstance(ip, list):
            include_plots = _validate_plots(ip, ctx=f"{ctx}.include_plots")
        else:
            # your YAML uses include_plots: { - list } (actually still list in YAML)
            # but normalize_dict would coerce to {} if not dict; so handle both
            # If user wrote a dict with keys, accept keys that are true-ish.
            out: List[str] = []
            for k, v in ipd.items():
                if v:
                    out.append(str(k))
            if out:
                include_plots = _validate_plots(out, ctx=f"{ctx}.include_plots")
            else:
                # if empty dict, treat as empty
                include_plots = []

    extra = {k: v for k, v in d.items() if k not in allowed}

    return StaticReportsSpec(
        enabled=enabled,
        formats=formats,
        include_plots=include_plots,
        extra=extra,
    )


def validate_visualization_block(block: Any, *, ctx: str = "visualization", strict: bool = False) -> VisualizationSpec:
    """
    Validates:
      visualization:
        tensorboard: {...}
        static_reports: {...}
    """
    d = normalize_dict(block)

    allowed = {"tensorboard", "static_reports"}
    if strict:
        forbid_unknown_keys(d, allowed, ctx=ctx)

    tb = validate_tensorboard_block(d.get("tensorboard", {}) or {}, ctx=f"{ctx}.tensorboard", strict=strict)
    sr = validate_static_reports_block(d.get("static_reports", {}) or {}, ctx=f"{ctx}.static_reports", strict=strict)

    extra = {k: v for k, v in d.items() if k not in allowed}

    return VisualizationSpec(
        tensorboard=tb,
        static_reports=sr,
        extra=extra,
    )
