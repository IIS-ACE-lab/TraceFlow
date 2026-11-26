# services/mlops/core/pruning.py
# ============================================================
# Pruning Engine (PURE LOGIC)
# - Executes pruning rules against MetricsSnapshot + compiled context
# - Zero side effects (NO file writes, NO training control)
# - Returns deterministic decisions for the orchestrator to "halt" training
# ============================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict,  List, Mapping, Optional, Sequence, Tuple, Union
from services.side_channel_core.utils.timestamps import now_iso


# ============================================================
# Types
# ============================================================

Number = Union[int, float]


class PruneAction(str, Enum):
    """
    What the orchestrator should do *now*.

    NOTE:
      - HALT means: stop training early (no deletion). This is what you want.
      - PRUNE means: stop this trial early (no deletion implied here).
      - FAIL means: stop this trial due to invalid state (NaNs, etc.).
      - CONTINUE means: keep going.
    """
    CONTINUE = "continue"
    HALT = "halt"
    PRUNE = "prune"
    FAIL = "fail"


class PruneSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class MetricsSnapshot:
    """
    Minimal normalized metrics view used by pruning.

    This is intentionally small and schema-safe.
    Your metrics_reader.py can output this (or a superset).
    """
    epoch: int
    metrics: Dict[str, Any]  # typically {metric_name: float/int/None}
    ts: str = field(default_factory=now_iso)

    # Optional auxiliary context (if available)
    step: Optional[int] = None
    phase: Optional[str] = None  # "train" | "val" | "test" | "benchmark"
    trial_id: Optional[int] = None
    hypothesis_id: Optional[str] = None


@dataclass(frozen=True)
class PruneDecision:
    """
    Deterministic prune decision (no side effects).
    """
    action: PruneAction
    rule_name: Optional[str] = None
    reason: Optional[str] = None
    severity: PruneSeverity = PruneSeverity.INFO

    metric: Optional[str] = None
    metric_value: Optional[Any] = None
    epoch: Optional[int] = None

    # Extra payload for logging / registry
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_halt(self) -> bool:
        return self.action in {PruneAction.HALT, PruneAction.PRUNE, PruneAction.FAIL}


# ============================================================
# Rule model (matches your YAML style)
# ============================================================

@dataclass(frozen=True)
class MetricPredicate:
    """
    One metric predicate clause.

    Supported keys (YAML-like):
      - metric: "accuracy" / "guessing_entropy" / "rank_k_success@5" etc.
      - at_epoch: int
      - below: number
      - above: number
      - is_nan: bool
      - is_inf: bool
      - window: int
      - min_improvement: number
    """
    metric: str
    at_epoch: Optional[int] = None

    below: Optional[Number] = None
    above: Optional[Number] = None
    is_nan: Optional[bool] = None
    is_inf: Optional[bool] = None

    # plateau-like rule support
    window: Optional[int] = None
    min_improvement: Optional[Number] = None


@dataclass(frozen=True)
class PruneRule:
    """
    One pruning rule.

    YAML examples:
      - name: ge_plateau
        metric: guessing_entropy
        window: 3
        min_improvement: 0.01
        action: prune

      - name: early_accuracy_floor
        metric: accuracy
        at_epoch: 2
        below: 0.15
        action: prune

      - name: nan_guard
        if_any:
          - metric: cross_entropy
            is_nan: true
        action: fail
    """
    name: str
    action: PruneAction

    # Either a single predicate form OR if_any/if_all forms
    predicate: Optional[MetricPredicate] = None
    if_any: List[MetricPredicate] = field(default_factory=list)
    if_all: List[MetricPredicate] = field(default_factory=list)

    # optional human label
    description: Optional[str] = None


@dataclass(frozen=True)
class PruningSpec:
    """
    Normalized pruning spec consumed by this engine.

    Your schemas/pruning.py validator can output this shape,
    OR you can build it from raw dict.
    """
    rules: List[PruneRule] = field(default_factory=list)
    # default action to use if rule uses legacy "prune"/"fail" etc.
    default_halt_action: PruneAction = PruneAction.HALT


# ============================================================
# Public API
# ============================================================

def execute_pruning(
    pruning: Union[PruningSpec, Mapping[str, Any], None],
    *,
    snapshot: MetricsSnapshot,
    history: Sequence[MetricsSnapshot] = (),
    compiled: Optional[Mapping[str, Any]] = None,
) -> PruneDecision:
    """
    Evaluate pruning rules against the current snapshot and recent history.

    Inputs:
      - pruning: PruningSpec OR raw dict (from YAML) OR None
      - snapshot: current metrics
      - history: prior snapshots (ordered by epoch ascending) if available
      - compiled: optional compiled context (project/hypothesis/constraints/etc.)

    Output:
      - PruneDecision (CONTINUE if nothing fires)

    Determinism:
      - No file I/O, no randomness.
      - Same inputs => same output.
    """
    spec = _normalize_pruning(pruning)

    if not spec.rules:
        return PruneDecision(
            action=PruneAction.CONTINUE,
            rule_name=None,
            reason="no_pruning_rules",
            epoch=snapshot.epoch,
        )

    # Evaluate in rule order: first match wins (deterministic)
    for rule in spec.rules:
        decision = _evaluate_rule(
            rule,
            snapshot=snapshot,
            history=history,
            compiled=compiled or {},
            default_halt_action=spec.default_halt_action,
        )
        if decision is not None and decision.action != PruneAction.CONTINUE:
            return decision

    return PruneDecision(
        action=PruneAction.CONTINUE,
        rule_name=None,
        reason="no_rules_triggered",
        epoch=snapshot.epoch,
    )


# ============================================================
# Internals: normalization
# ============================================================

def _normalize_pruning(pruning: Union[PruningSpec, Mapping[str, Any], None]) -> PruningSpec:
    if pruning is None:
        return PruningSpec(rules=[])

    if isinstance(pruning, PruningSpec):
        return pruning

    if not isinstance(pruning, Mapping):
        raise TypeError(f"[mlops/core/pruning] pruning must be PruningSpec|mapping|None, got {type(pruning)}")

    # Accept either:
    #   pruning: { rules: [...] }
    # or older:
    #   pruning: { ...single rule? }
    raw_rules = pruning.get("rules", None)
    if raw_rules is None and "name" in pruning:
        raw_rules = [pruning]
    if raw_rules is None:
        raw_rules = []

    if not isinstance(raw_rules, list):
        raise ValueError("[mlops/core/pruning] pruning.rules must be a list")

    rules: List[PruneRule] = []
    for i, rr in enumerate(raw_rules):
        if not isinstance(rr, Mapping):
            raise ValueError(f"[mlops/core/pruning] pruning.rules[{i}] must be a mapping")
        rules.append(_normalize_rule(rr, idx=i))

    # default halt action: prefer HALT so you never delete/cleanup inside pruning
    return PruningSpec(rules=rules, default_halt_action=PruneAction.HALT)


def _normalize_rule(raw: Mapping[str, Any], *, idx: int) -> PruneRule:
    name = str(raw.get("name") or f"rule_{idx}").strip()
    if not name:
        name = f"rule_{idx}"

    action_raw = str(raw.get("action") or "prune").strip().lower()

    # Map YAML semantics onto our action space
    if action_raw in {"halt"}:
        action = PruneAction.HALT
    elif action_raw in {"prune", "early_stop", "stop"}:
        action = PruneAction.PRUNE
    elif action_raw in {"fail", "error"}:
        action = PruneAction.FAIL
    elif action_raw in {"continue", "noop"}:
        action = PruneAction.CONTINUE
    else:
        raise ValueError(f"[mlops/core/pruning] invalid action '{action_raw}' in rule '{name}'")

    desc = raw.get("description")
    description = str(desc).strip() if isinstance(desc, str) and desc.strip() else None

    if_any = raw.get("if_any")
    if_all = raw.get("if_all")

    # Single predicate shortcut: metric + fields at top-level
    if if_any is None and if_all is None and "metric" in raw:
        pred = _normalize_predicate(raw, ctx=f"pruning.rules[{idx}]")
        return PruneRule(name=name, action=action, predicate=pred, description=description)

    any_preds = _normalize_predicate_list(if_any, ctx=f"pruning.rules[{idx}].if_any") if if_any is not None else []
    all_preds = _normalize_predicate_list(if_all, ctx=f"pruning.rules[{idx}].if_all") if if_all is not None else []

    return PruneRule(
        name=name,
        action=action,
        predicate=None,
        if_any=any_preds,
        if_all=all_preds,
        description=description,
    )


def _normalize_predicate_list(v: Any, *, ctx: str) -> List[MetricPredicate]:
    if not isinstance(v, list) or not v:
        raise ValueError(f"[mlops/core/pruning] {ctx} must be a non-empty list")
    out: List[MetricPredicate] = []
    for i, item in enumerate(v):
        if not isinstance(item, Mapping):
            raise ValueError(f"[mlops/core/pruning] {ctx}[{i}] must be a mapping")
        out.append(_normalize_predicate(item, ctx=f"{ctx}[{i}]"))
    return out


def _normalize_predicate(raw: Mapping[str, Any], *, ctx: str) -> MetricPredicate:
    metric = raw.get("metric")
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError(f"[mlops/core/pruning] {ctx}.metric must be a non-empty string")

    def _as_int(x: Any, k: str) -> Optional[int]:
        if x is None:
            return None
        if isinstance(x, bool) or not isinstance(x, int):
            raise ValueError(f"[mlops/core/pruning] {ctx}.{k} must be int")
        return int(x)

    def _as_num(x: Any, k: str) -> Optional[Number]:
        if x is None:
            return None
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(f"[mlops/core/pruning] {ctx}.{k} must be number")
        return x

    def _as_bool(x: Any, k: str) -> Optional[bool]:
        if x is None:
            return None
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)) and x in (0, 1):
            return bool(x)
        if isinstance(x, str) and x.strip().lower() in {"true", "false"}:
            return x.strip().lower() == "true"
        raise ValueError(f"[mlops/core/pruning] {ctx}.{k} must be bool")

    return MetricPredicate(
        metric=metric.strip(),
        at_epoch=_as_int(raw.get("at_epoch"), "at_epoch"),
        below=_as_num(raw.get("below"), "below"),
        above=_as_num(raw.get("above"), "above"),
        is_nan=_as_bool(raw.get("is_nan"), "is_nan"),
        is_inf=_as_bool(raw.get("is_inf"), "is_inf"),
        window=_as_int(raw.get("window"), "window"),
        min_improvement=_as_num(raw.get("min_improvement"), "min_improvement"),
    )


# ============================================================
# Internals: evaluation
# ============================================================

def _evaluate_rule(
    rule: PruneRule,
    *,
    snapshot: MetricsSnapshot,
    history: Sequence[MetricsSnapshot],
    compiled: Mapping[str, Any],
    default_halt_action: PruneAction,
) -> Optional[PruneDecision]:
    # Determine if rule triggers
    if rule.predicate is not None:
        ok, detail = _predicate_holds(rule.predicate, snapshot=snapshot, history=history)
        if ok:
            return _decision_from_trigger(rule, snapshot=snapshot, detail=detail, compiled=compiled, default_halt_action=default_halt_action)
        return None

    if rule.if_any:
        for pred in rule.if_any:
            ok, detail = _predicate_holds(pred, snapshot=snapshot, history=history)
            if ok:
                return _decision_from_trigger(rule, snapshot=snapshot, detail=detail, compiled=compiled, default_halt_action=default_halt_action)
        return None

    if rule.if_all:
        all_details: List[Dict[str, Any]] = []
        for pred in rule.if_all:
            ok, detail = _predicate_holds(pred, snapshot=snapshot, history=history)
            if not ok:
                return None
            all_details.append(detail)
        # If all hold, include merged details
        merged = {"all": all_details}
        return _decision_from_trigger(rule, snapshot=snapshot, detail=merged, compiled=compiled, default_halt_action=default_halt_action)

    # If misconfigured rule (no predicate forms), ignore deterministically
    return None


def _decision_from_trigger(
    rule: PruneRule,
    *,
    snapshot: MetricsSnapshot,
    detail: Dict[str, Any],
    compiled: Mapping[str, Any],
    default_halt_action: PruneAction,
) -> PruneDecision:
    # Your policy: do NOT delete data; "halt" is the orchestrator behavior.
    # If rule asked for prune/fail, we keep that action but orchestrator can map PRUNE->HALT if desired.
    action = rule.action
    if action == PruneAction.PRUNE and default_halt_action == PruneAction.HALT:
        # Optional policy: treat PRUNE as HALT at runtime (stop cleanly).
        # Keeps semantics but matches your "halt poor training" behavior.
        effective = PruneAction.HALT
    else:
        effective = action

    severity = PruneSeverity.WARNING
    if action == PruneAction.FAIL:
        severity = PruneSeverity.CRITICAL
    elif action in {PruneAction.PRUNE, PruneAction.HALT}:
        severity = PruneSeverity.WARNING
    else:
        severity = PruneSeverity.INFO

    metric = detail.get("metric")
    metric_value = detail.get("value")

    reason = detail.get("reason") or "rule_triggered"

    payload = {
        "rule": rule.name,
        "action": str(action.value),
        "effective_action": str(effective.value),
        "detail": detail,
        "epoch": snapshot.epoch,
        "trial_id": snapshot.trial_id,
        "hypothesis_id": snapshot.hypothesis_id,
    }

    # Include minimal compiled identity if available
    proj = compiled.get("project_name") or compiled.get("project", {}).get("name")
    if proj:
        payload["project"] = proj

    return PruneDecision(
        action=effective,
        rule_name=rule.name,
        reason=reason,
        severity=severity,
        metric=metric,
        metric_value=metric_value,
        epoch=snapshot.epoch,
        data=payload,
    )


def _predicate_holds(
    pred: MetricPredicate,
    *,
    snapshot: MetricsSnapshot,
    history: Sequence[MetricsSnapshot],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (holds, detail).
    """
    metric = pred.metric
    epoch = snapshot.epoch

    # at_epoch gating
    if pred.at_epoch is not None and epoch != pred.at_epoch:
        return False, {"metric": metric, "reason": "epoch_mismatch", "expected": pred.at_epoch, "epoch": epoch}

    value = snapshot.metrics.get(metric, None)

    # is_nan / is_inf checks (treat missing metric as not-trigger unless explicitly checking NaN/Inf? still false)
    if pred.is_nan is True:
        if _is_nan(value):
            return True, {"metric": metric, "value": value, "reason": "is_nan"}
        return False, {"metric": metric, "value": value, "reason": "not_nan"}

    if pred.is_inf is True:
        if _is_inf(value):
            return True, {"metric": metric, "value": value, "reason": "is_inf"}
        return False, {"metric": metric, "value": value, "reason": "not_inf"}

    # below / above checks
    if pred.below is not None:
        if _as_float(value) is None:
            return False, {"metric": metric, "value": value, "reason": "missing_or_non_numeric"}
        if float(value) < float(pred.below):
            return True, {"metric": metric, "value": float(value), "threshold": float(pred.below), "reason": "below"}
        return False, {"metric": metric, "value": float(value), "threshold": float(pred.below), "reason": "not_below"}

    if pred.above is not None:
        if _as_float(value) is None:
            return False, {"metric": metric, "value": value, "reason": "missing_or_non_numeric"}
        if float(value) > float(pred.above):
            return True, {"metric": metric, "value": float(value), "threshold": float(pred.above), "reason": "above"}
        return False, {"metric": metric, "value": float(value), "threshold": float(pred.above), "reason": "not_above"}

    # plateau / min_improvement over window (generic)
    # Interpretation:
    #   - For minimization metrics: improvement = old - new
    #   - For maximization metrics: improvement = new - old
    # We do NOT guess direction here. Instead:
    #   - If metric decreases -> treat as improvement
    #   - If metric increases -> treat as improvement
    # That is: improvement = abs(delta)?? No.
    # We choose a safe semantic for "plateau prune":
    #   - Compute best_delta = max(|delta|) over last window vs first in window
    #   - If best_delta < min_improvement -> plateau.
    # This avoids needing objective directions here.
    if pred.window is not None and pred.min_improvement is not None:
        win = pred.window
        if win <= 0:
            return False, {"metric": metric, "reason": "invalid_window", "window": win}

        series = _metric_series(metric, history=history, current=snapshot)
        if len(series) < win + 1:
            return False, {"metric": metric, "reason": "insufficient_history", "need": win + 1, "have": len(series)}

        # last (win+1) points
        tail = series[-(win + 1) :]
        # Use first->last absolute change magnitude as "progress"
        first = tail[0]
        last = tail[-1]
        if first is None or last is None:
            return False, {"metric": metric, "reason": "non_numeric_in_window"}

        progress = abs(float(last) - float(first))
        if progress < float(pred.min_improvement):
            return True, {
                "metric": metric,
                "value": float(snapshot.metrics.get(metric)) if _as_float(snapshot.metrics.get(metric)) is not None else snapshot.metrics.get(metric),
                "window": win,
                "progress": progress,
                "min_improvement": float(pred.min_improvement),
                "reason": "plateau",
            }
        return False, {
            "metric": metric,
            "window": win,
            "progress": progress,
            "min_improvement": float(pred.min_improvement),
            "reason": "progress_ok",
        }

    # If predicate had no checks, it never triggers
    return False, {"metric": metric, "reason": "empty_predicate"}


# ============================================================
# Helpers
# ============================================================

def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except Exception:
            return None
    return None


def _is_nan(v: Any) -> bool:
    x = _as_float(v)
    return x is not None and math.isnan(x)


def _is_inf(v: Any) -> bool:
    x = _as_float(v)
    return x is not None and math.isinf(x)


def _metric_series(metric: str, *, history: Sequence[MetricsSnapshot], current: MetricsSnapshot) -> List[Optional[float]]:
    """
    Extract numeric metric values across snapshots.
    Keeps None for missing/non-numeric.
    """
    out: List[Optional[float]] = []
    for snap in history:
        out.append(_as_float(snap.metrics.get(metric)))
    out.append(_as_float(current.metrics.get(metric)))
    return out
