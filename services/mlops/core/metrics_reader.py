# services/mlops/core/metrics_reader.py
# ============================================================
# Metrics Reader (CORE / PURE LOGIC)
# - Normalize + validate training outputs → MetricsSnapshot
# - No file I/O. No global state. Deterministic.
#
# Design goals:
# - Accept metrics from arbitrary trainers (PyTorch loops, ClearML exports,
#   custom CSV readers, etc.) as in-memory objects.
# - Provide strict, schema-safe extraction for objectives + pruning + registry.
# - Never crash on "extra metrics"; optionally tolerate missing metrics.
# ============================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict,  List, Mapping, Optional, Sequence, Tuple, Union
from services.side_channel_core.utils.timestamps import now_iso

# ------------------------------------------------------------------
# Types
# ------------------------------------------------------------------

MetricName = str
MetricValue = Union[int, float]
MetricDict = Dict[MetricName, MetricValue]

Direction = str  # "minimize" | "maximize"


# ------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------

class MetricsError(ValueError):
    pass


class MetricsCoercionError(MetricsError):
    pass


class MetricsValidationError(MetricsError):
    pass


# ------------------------------------------------------------------
# Helpers (pure)
# ------------------------------------------------------------------

def _now_iso() -> str:
    return now_iso()


def _is_number(x: Any) -> bool:
    # bool is a subclass of int -> reject explicitly
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_nan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _is_inf(x: float) -> bool:
    return isinstance(x, float) and (math.isinf(x))


def _coerce_scalar_number(x: Any, *, ctx: str) -> MetricValue:
    """
    Coerce common scalar-like metric values into float/int.

    Accepts:
      - int / float (not bool)
      - numeric strings "0.123"
      - numpy scalar types (duck-typed via float(x))
      - torch scalar types (duck-typed via float(x))

    Rejects:
      - None, dict, list (handled elsewhere), bool, non-numeric strings
    """
    if x is None:
        raise MetricsCoercionError(f"[metrics_reader] {ctx}: value is None")
    if isinstance(x, bool):
        raise MetricsCoercionError(f"[metrics_reader] {ctx}: bool is not a metric number")
    if _is_number(x):
        # keep ints as ints, floats as floats
        return int(x) if isinstance(x, int) else float(x)

    if isinstance(x, str):
        s = x.strip()
        if not s:
            raise MetricsCoercionError(f"[metrics_reader] {ctx}: empty string is not a number")
        try:
            # allow "1e-3"
            f = float(s)
            # keep "2" as int? metrics often float anyway; preserve float.
            return float(f)
        except Exception as e:
            raise MetricsCoercionError(f"[metrics_reader] {ctx}: cannot parse '{x}' as float: {e}") from e

    # duck-typed numpy/torch scalar
    try:
        f = float(x)  # type: ignore[arg-type]
        return float(f)
    except Exception as e:
        raise MetricsCoercionError(f"[metrics_reader] {ctx}: unsupported metric type {type(x)}: {e}") from e


def _reduce_sequence(
    seq: Sequence[Any],
    *,
    strategy: str,
    ctx: str,
) -> MetricValue:
    """
    Reduce list/tuple-like metric outputs into a single scalar.

    strategy:
      - "last": use last element
      - "mean": mean of numeric elements
      - "min": min of numeric elements
      - "max": max of numeric elements
    """
    if not seq:
        raise MetricsCoercionError(f"[metrics_reader] {ctx}: empty sequence")
    nums: List[float] = []
    for i, v in enumerate(seq):
        try:
            nums.append(float(_coerce_scalar_number(v, ctx=f"{ctx}[{i}]")))
        except MetricsCoercionError:
            # ignore non-numeric elements rather than failing hard; caller controls strictness
            continue

    if not nums:
        raise MetricsCoercionError(f"[metrics_reader] {ctx}: no numeric elements in sequence")

    if strategy == "last":
        return float(nums[-1])
    if strategy == "mean":
        return float(sum(nums) / len(nums))
    if strategy == "min":
        return float(min(nums))
    if strategy == "max":
        return float(max(nums))

    raise MetricsCoercionError(f"[metrics_reader] {ctx}: unknown reduce strategy '{strategy}'")


def _coerce_metric_value(
    v: Any,
    *,
    ctx: str,
    reduce_sequence: str = "last",
) -> MetricValue:
    """
    Coerce a metric value into scalar number.
    Supports sequences by reduction strategy.
    """
    if _is_number(v) or isinstance(v, str):
        return _coerce_scalar_number(v, ctx=ctx)

    if isinstance(v, (list, tuple)):
        return _reduce_sequence(v, strategy=reduce_sequence, ctx=ctx)

    # allow numpy arrays / torch tensors by duck-typing .tolist()
    tolist = getattr(v, "tolist", None)
    if callable(tolist):
        try:
            vv = tolist()
            if isinstance(vv, (list, tuple)):
                return _reduce_sequence(vv, strategy=reduce_sequence, ctx=ctx)
            return _coerce_scalar_number(vv, ctx=ctx)
        except Exception as e:
            raise MetricsCoercionError(f"[metrics_reader] {ctx}: cannot coerce tolist() value: {e}") from e

    return _coerce_scalar_number(v, ctx=ctx)


def _normalize_metric_name(name: str) -> str:
    # We keep metric names stable and case-sensitive by default.
    # We only trim whitespace.
    return name.strip()


def _extract_metrics_mapping(raw: Any) -> Dict[str, Any]:
    """
    Extract a "metrics dict" from common trainer outputs.

    Accepts:
      - dict of metric->value
      - dict containing {"metrics": {...}} (common)
      - dict containing {"results": {...}} (common)
      - dict containing {"val": {...}} or {"train": {...}} (we accept but don't guess which)
    """
    if isinstance(raw, Mapping):
        d = dict(raw)

        # common nesting patterns
        for k in ("metrics", "results"):
            if k in d and isinstance(d[k], Mapping):
                return dict(d[k])

        # sometimes metric dict is top-level
        return d

    raise MetricsCoercionError(f"[metrics_reader] raw metrics must be mapping, got {type(raw)}")


def _validate_no_nan_inf(metrics: MetricDict, *, disallow_nan: bool, disallow_inf: bool) -> None:
    if not (disallow_nan or disallow_inf):
        return
    bad: List[str] = []
    for k, v in metrics.items():
        fv = float(v)
        if disallow_nan and _is_nan(fv):
            bad.append(f"{k}=NaN")
        if disallow_inf and _is_inf(fv):
            bad.append(f"{k}=Inf")
    if bad:
        raise MetricsValidationError(f"[metrics_reader] invalid metrics: {', '.join(bad)}")


def _sorted_objective_keys(directions: Mapping[str, Direction]) -> List[str]:
    # deterministic order for stable snapshots / hashing
    return sorted(list(directions.keys()))


# ------------------------------------------------------------------
# Snapshot
# ------------------------------------------------------------------

@dataclass(frozen=True)
class MetricsSnapshot:
    """
    Canonical normalized metrics snapshot used across MLSCA MLOps.

    - metrics: normalized scalar numbers only
    - missing_required: computed against "directions" if provided
    - violations: non-fatal issues (unless strict=True during build)
    """
    ts: str

    # provenance
    project_name: Optional[str] = None
    model_name: Optional[str] = None
    hypothesis_id: Optional[str] = None
    trial_id: Optional[int] = None

    # step identifiers (epoch, step, etc.)
    epoch: Optional[int] = None
    step: Optional[int] = None

    # normalized metrics
    metrics: MetricDict = field(default_factory=dict)

    # objective schema hints (optional; useful for judging)
    directions: Dict[str, Direction] = field(default_factory=dict)
    primary: Optional[str] = None
    tie_breakers: List[str] = field(default_factory=list)

    # diagnostics
    missing_required: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)

    def get(self, key: str, default: Optional[MetricValue] = None) -> Optional[MetricValue]:
        return self.metrics.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "project_name": self.project_name,
            "model_name": self.model_name,
            "hypothesis_id": self.hypothesis_id,
            "trial_id": self.trial_id,
            "epoch": self.epoch,
            "step": self.step,
            "metrics": dict(self.metrics),
            "directions": dict(self.directions),
            "primary": self.primary,
            "tie_breakers": list(self.tie_breakers),
            "missing_required": list(self.missing_required),
            "violations": list(self.violations),
        }


# ------------------------------------------------------------------
# Public API (pure)
# ------------------------------------------------------------------

def build_metrics_snapshot(
    raw: Any,
    *,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    project_name: Optional[str] = None,
    model_name: Optional[str] = None,
    hypothesis_id: Optional[str] = None,
    trial_id: Optional[int] = None,
    directions: Optional[Mapping[str, Direction]] = None,
    primary: Optional[str] = None,
    tie_breakers: Optional[Sequence[str]] = None,
    reduce_sequence: str = "last",
    disallow_nan: bool = False,
    disallow_inf: bool = True,
    strict: bool = False,
    allow_extra_metrics: bool = True,
) -> MetricsSnapshot:
    """
    Normalize and validate raw training outputs -> MetricsSnapshot.

    raw: mapping-like or common nested mapping. Examples:
      - {"accuracy": 0.9, "cross_entropy": 0.4}
      - {"metrics": {"guessing_entropy": 42, "mean_rank": 18}}
      - {"results": {...}}

    strict behavior:
      - strict=True raises on:
          - missing required objective metrics (directions keys)
          - non-coercible metric values
          - NaN/Inf (if disallow flags)
      - strict=False records violations + drops bad metrics.
    """
    violations: List[str] = []

    # ---- extract candidate dict ----
    try:
        raw_metrics = _extract_metrics_mapping(raw)
    except MetricsError:
        raise
    except Exception as e:
        raise MetricsCoercionError(f"[metrics_reader] cannot extract metrics mapping: {e}") from e

    # ---- normalize + coerce values ----
    out: MetricDict = {}
    for k, v in raw_metrics.items():
        if not isinstance(k, str):
            if strict:
                raise MetricsCoercionError(f"[metrics_reader] metric key must be str, got {type(k)}")
            violations.append(f"non_string_metric_key:{type(k)}")
            continue

        name = _normalize_metric_name(k)
        if not name:
            if strict:
                raise MetricsCoercionError("[metrics_reader] empty metric name is invalid")
            violations.append("empty_metric_name")
            continue

        try:
            out[name] = _coerce_metric_value(v, ctx=f"metric[{name}]", reduce_sequence=reduce_sequence)
        except MetricsCoercionError as e:
            if strict:
                raise
            violations.append(f"coerce_fail:{name}:{e}")
            continue

    # ---- objective requirements ----
    dir_map: Dict[str, Direction] = {}
    missing_required: List[str] = []

    if directions is not None:
        for mk, dv in directions.items():
            if not isinstance(mk, str) or not mk.strip():
                raise MetricsValidationError("[metrics_reader] directions keys must be non-empty strings")
            dvv = str(dv).strip().lower()
            if dvv not in ("minimize", "maximize"):
                raise MetricsValidationError(f"[metrics_reader] invalid direction for {mk}: {dv}")
            dir_map[mk.strip()] = dvv  # normalized

        for req in _sorted_objective_keys(dir_map):
            if req not in out:
                missing_required.append(req)

        if missing_required and strict:
            raise MetricsValidationError(f"[metrics_reader] missing required objective metrics: {missing_required}")

    # ---- primary & tie breakers ----
    primary_n = primary.strip() if isinstance(primary, str) else None
    if primary_n and directions is not None and primary_n not in dir_map:
        # schema mismatch: primary must be in directions
        if strict:
            raise MetricsValidationError(f"[metrics_reader] primary '{primary_n}' not found in objectives.directions")
        violations.append(f"primary_not_in_directions:{primary_n}")

    tb: List[str] = []
    if tie_breakers:
        for t in tie_breakers:
            if not isinstance(t, str) or not t.strip():
                continue
            tb.append(t.strip())

    # ---- extra metric policy ----
    if not allow_extra_metrics and directions is not None:
        # keep only those in direction keys
        allowed = set(dir_map.keys())
        out = {k: v for k, v in out.items() if k in allowed}

    # ---- NaN/Inf checks ----
    try:
        _validate_no_nan_inf(out, disallow_nan=disallow_nan, disallow_inf=disallow_inf)
    except MetricsValidationError as e:
        if strict:
            raise
        violations.append(str(e))

    return MetricsSnapshot(
        ts=_now_iso(),
        project_name=project_name,
        model_name=model_name,
        hypothesis_id=hypothesis_id,
        trial_id=trial_id,
        epoch=epoch,
        step=step,
        metrics=out,
        directions=dir_map,
        primary=primary_n,
        tie_breakers=tb,
        missing_required=missing_required,
        violations=violations,
    )


def extract_best_snapshot(
    snapshots: Sequence[MetricsSnapshot],
    *,
    directions: Mapping[str, Direction],
    primary: str,
    tie_breakers: Optional[Sequence[str]] = None,
) -> Optional[MetricsSnapshot]:
    """
    Deterministically select the best snapshot by:
      1) primary objective direction
      2) tie breakers in order (if provided)
      3) stable fallback: earlier ts wins (deterministic)
    """
    if not snapshots:
        return None

    primary = primary.strip()
    if not primary:
        raise MetricsValidationError("[metrics_reader] primary objective cannot be empty")

    d_primary = str(directions.get(primary, "")).strip().lower()
    if d_primary not in ("minimize", "maximize"):
        raise MetricsValidationError(f"[metrics_reader] primary '{primary}' missing or invalid in directions")

    tbs = [t.strip() for t in (tie_breakers or []) if isinstance(t, str) and t.strip()]

    def _key(s: MetricsSnapshot) -> Tuple:
        # missing primary -> worst
        pv = s.metrics.get(primary, None)
        if pv is None:
            primary_rank = (1, 0.0)  # worse bucket
        else:
            x = float(pv)
            # for minimize: smaller is better; for maximize: larger is better
            score = x if d_primary == "minimize" else -x
            primary_rank = (0, score)

        tb_ranks: List[Tuple[int, float]] = []
        for t in tbs:
            dv = str(directions.get(t, "")).strip().lower()
            tv = s.metrics.get(t, None)
            if tv is None or dv not in ("minimize", "maximize"):
                tb_ranks.append((1, 0.0))
                continue
            x = float(tv)
            score = x if dv == "minimize" else -x
            tb_ranks.append((0, score))

        # stable fallback: ts (earlier is better), then epoch/step
        # ISO strings sort lexicographically for UTC timestamps from our generator
        return (primary_rank, *tb_ranks, s.ts, s.epoch if s.epoch is not None else -1, s.step if s.step is not None else -1)

    return sorted(snapshots, key=_key)[0]


def normalize_metric_names(
    metrics: Mapping[str, Any],
    *,
    strip: bool = True,
) -> Dict[str, Any]:
    """
    Optional helper if upstream has whitespace/casing issues.
    Kept separate to avoid silently changing semantics in core builder.
    """
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        if not isinstance(k, str):
            continue
        kk = k.strip() if strip else k
        if not kk:
            continue
        out[kk] = v
    return out
