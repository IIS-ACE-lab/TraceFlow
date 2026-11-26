# services/mlops/core/trial_judge.py
# ============================================================
# Trial Judge (PURE LOGIC)
# - Converts metrics + pruning output + objectives/benchmark gates into decisions
# - accept/reject/qualify/promote flags + reason codes
# - ZERO side effects (no file writes, no training control)
# ============================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional,  Tuple, Union

from services.mlops.core.pruning import MetricsSnapshot, PruneAction, PruneDecision
from services.side_channel_core.utils.timestamps import now_iso

# ============================================================
# Enums / Types
# ============================================================

Number = Union[int, float]


class TrialStage(str, Enum):
    INTERMEDIATE = "intermediate"   # during training (per epoch snapshots)
    FINAL = "final"                 # end-of-trial (final metrics)


class TrialDecision(str, Enum):
    CONTINUE = "continue"           # keep training
    HALT = "halt"                   # stop training early (clean stop)
    REJECT = "reject"               # completed but not acceptable / not qualified
    ACCEPT = "accept"               # completed and acceptable (not necessarily best)
    QUALIFY = "qualify"             # meets qualify thresholds (candidate for promotion)
    PROMOTE = "promote"             # selected for promotion (usually decided later by promotion policy)


class Direction(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


# ============================================================
# Reason Codes (stable audit keys)
# ============================================================

RC_PRUNED = "PRUNED"
RC_FAILED = "FAILED"

RC_METRIC_MISSING = "METRIC_MISSING"
RC_METRIC_NAN = "METRIC_NAN"
RC_METRIC_INF = "METRIC_INF"

RC_CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
RC_NOT_QUALIFIED = "NOT_QUALIFIED"
RC_QUALIFIED = "QUALIFIED"

RC_ACCEPTED = "ACCEPTED"
RC_REJECTED = "REJECTED"

RC_PROMOTION_ELIGIBLE = "PROMOTION_ELIGIBLE"


# ============================================================
# Data Models
# ============================================================

@dataclass(frozen=True)
class ObjectiveSpec:
    directions: Dict[str, Direction]
    primary: Optional[str] = None
    tie_breakers: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrialScore:
    """
    Deterministic ranking tuple, lexicographic:
      (primary, tie1, tie2, ...)

    Values are normalized so that "higher is better" for comparison,
    regardless of original direction. (minimize -> inverted sign)
    """
    keys: List[str]
    values: List[Optional[float]]


@dataclass(frozen=True)
class TrialVerdict:
    ts: str
    stage: TrialStage

    decision: TrialDecision
    reason_code: str
    reason: str

    # qualifiers
    acceptable: bool = False
    qualified: bool = False
    promotion_eligible: bool = False

    # scoring / comparison aid
    score: Optional[TrialScore] = None

    # useful payload
    data: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Public API
# ============================================================

def judge_trial(
    *,
    snapshot: MetricsSnapshot,
    pruning: Optional[PruneDecision] = None,
    compiled: Mapping[str, Any],
    stage: TrialStage = TrialStage.INTERMEDIATE,
) -> TrialVerdict:
    """
    Core entry point.

    Inputs:
      - snapshot: normalized metrics snapshot
      - pruning: output from core/pruning.py (optional)
      - compiled: deterministic compiled context (from core/context.py)
      - stage: INTERMEDIATE or FINAL

    Output:
      - TrialVerdict (no side effects)

    Expectations about compiled:
      compiled["objectives"]["directions"]   => {metric: "minimize"/"maximize"}
      compiled["objectives"]["primary"]      => str (optional)
      compiled["objectives"]["tie_breakers"] => [str, ...]
      compiled.get("constraints", {})        => optional hard gates
      compiled.get("benchmarking", {})       => optional qualify_thresholds
      compiled.get("trial_judge", {})        => optional behavior flags
    """
    ts = _now_iso()

    # 1) If pruning tells us to stop, reflect it deterministically.
    if pruning is not None and pruning.should_halt:
        if pruning.action == PruneAction.FAIL:
            return TrialVerdict(
                ts=ts,
                stage=stage,
                decision=TrialDecision.HALT,
                reason_code=RC_FAILED,
                reason=pruning.reason or "trial_failed",
                acceptable=False,
                qualified=False,
                promotion_eligible=False,
                score=None,
                data={"pruning": pruning.data},
            )
        # PRUNE / HALT => halt cleanly (no deletion)
        return TrialVerdict(
            ts=ts,
            stage=stage,
            decision=TrialDecision.HALT,
            reason_code=RC_PRUNED,
            reason=pruning.reason or "trial_pruned",
            acceptable=False,
            qualified=False,
            promotion_eligible=False,
            score=None,
            data={"pruning": pruning.data},
        )

    # 2) Build objective spec + score
    obj = _read_objectives(compiled)
    score = _compute_trial_score(snapshot.metrics, obj)

    # 3) Hard gates / constraints
    ok_constraints, c_reason = _check_constraints(snapshot.metrics, compiled)
    if not ok_constraints:
        # During intermediate stage: usually HALT to stop wasting compute.
        # During final: reject.
        return TrialVerdict(
            ts=ts,
            stage=stage,
            decision=TrialDecision.HALT if stage == TrialStage.INTERMEDIATE else TrialDecision.REJECT,
            reason_code=RC_CONSTRAINT_VIOLATION,
            reason=c_reason,
            acceptable=False,
            qualified=False,
            promotion_eligible=False,
            score=score,
            data={"constraints": compiled.get("constraints", {})},
        )

    # 4) Qualification gates (benchmarking thresholds etc.)
    qualified, q_reason_code, q_reason, q_data = _check_qualification(snapshot.metrics, compiled)

    # 5) Decision by stage
    if stage == TrialStage.INTERMEDIATE:
        # keep training unless something requires stop
        return TrialVerdict(
            ts=ts,
            stage=stage,
            decision=TrialDecision.CONTINUE,
            reason_code=RC_ACCEPTED,
            reason="continue_training",
            acceptable=True,
            qualified=qualified,
            promotion_eligible=False,
            score=score,
            data={"qualification": q_data},
        )

    # FINAL stage
    if qualified:
        # eligible for promotion selection (promotion policy applied later)
        return TrialVerdict(
            ts=ts,
            stage=stage,
            decision=TrialDecision.QUALIFY,
            reason_code=RC_QUALIFIED,
            reason=q_reason,
            acceptable=True,
            qualified=True,
            promotion_eligible=True,
            score=score,
            data={"qualification": q_data, "promotion_hint": compiled.get("promotion", {})},
        )

    # Completed but not qualified
    accept_without_qualify = bool(compiled.get("trial_judge", {}).get("accept_without_qualify", False))
    if accept_without_qualify:
        return TrialVerdict(
            ts=ts,
            stage=stage,
            decision=TrialDecision.ACCEPT,
            reason_code=RC_ACCEPTED,
            reason=q_reason or "accepted_without_qualify",
            acceptable=True,
            qualified=False,
            promotion_eligible=False,
            score=score,
            data={"qualification": q_data},
        )

    return TrialVerdict(
        ts=ts,
        stage=stage,
        decision=TrialDecision.REJECT,
        reason_code=q_reason_code,
        reason=q_reason,
        acceptable=True,
        qualified=False,
        promotion_eligible=False,
        score=score,
        data={"qualification": q_data},
    )


# ============================================================
# Objectives / Scoring
# ============================================================

def _read_objectives(compiled: Mapping[str, Any]) -> ObjectiveSpec:
    o = compiled.get("objectives", {}) or {}
    dirs_raw = o.get("directions", {}) or {}
    if not isinstance(dirs_raw, Mapping):
        raise TypeError("[mlops/core/trial_judge] compiled.objectives.directions must be a mapping")

    directions: Dict[str, Direction] = {}
    for k, v in dirs_raw.items():
        if not isinstance(k, str) or not k.strip():
            continue
        vv = str(v).strip().lower()
        if vv == "minimize":
            directions[k.strip()] = Direction.MINIMIZE
        elif vv == "maximize":
            directions[k.strip()] = Direction.MAXIMIZE
        else:
            raise ValueError(f"[mlops/core/trial_judge] invalid objective direction '{v}' for '{k}'")

    primary = o.get("primary")
    if primary is not None:
        primary = str(primary).strip() or None

    tie = o.get("tie_breakers") or []
    tie_breakers: List[str] = []
    if isinstance(tie, list):
        for x in tie:
            if isinstance(x, str) and x.strip():
                tie_breakers.append(x.strip())

    # Defensive: if primary is set but not in directions, still allow scoring,
    # but it will likely be missing => None score component.
    return ObjectiveSpec(directions=directions, primary=primary, tie_breakers=tie_breakers)


def _compute_trial_score(metrics: Mapping[str, Any], obj: ObjectiveSpec) -> TrialScore:
    """
    Builds a lexicographic score with "higher is better".

    Rule:
      - If metric is maximize: value stays as-is
      - If metric is minimize: value becomes (-value)
      - Missing / non-numeric becomes None
    """
    order: List[str] = []

    if obj.primary:
        order.append(obj.primary)
    for tb in obj.tie_breakers:
        if tb not in order:
            order.append(tb)

    # If primary not set, fall back to deterministic order of directions keys
    if not order:
        order = sorted(obj.directions.keys())

    keys: List[str] = []
    values: List[Optional[float]] = []

    for k in order:
        keys.append(k)
        v = _as_float(metrics.get(k))
        if v is None:
            values.append(None)
            continue
        direction = obj.directions.get(k, Direction.MAXIMIZE)
        values.append(v if direction == Direction.MAXIMIZE else -v)

    return TrialScore(keys=keys, values=values)


# ============================================================
# Constraints / Qualification
# ============================================================

def _check_constraints(metrics: Mapping[str, Any], compiled: Mapping[str, Any]) -> Tuple[bool, str]:
    """
    Hard gates. If violated => not acceptable.

    Supported:
      - constraints.disallow_nan (default True)
      - constraints.max_latency_ms, max_memory_mb (if metrics provide)
      - constraints.min_accuracy_floor (if accuracy metric exists)
    """
    c = compiled.get("constraints", {}) or {}
    disallow_nan = bool(c.get("disallow_nan", True))

    if disallow_nan:
        for k, v in metrics.items():
            fv = _as_float(v)
            if fv is None:
                continue
            if math.isnan(fv):
                return False, f"{RC_METRIC_NAN}:{k}"
            if math.isinf(fv):
                return False, f"{RC_METRIC_INF}:{k}"

    # Optional explicit floors
    acc_floor = c.get("min_accuracy_floor", None)
    if acc_floor is not None:
        floor = _as_float(acc_floor)
        acc = _as_float(metrics.get("accuracy"))
        if floor is not None and acc is not None and acc < floor:
            return False, f"{RC_CONSTRAINT_VIOLATION}:accuracy<{floor}"

    # Optional resource constraints if metric exists
    latency_max = _as_float(c.get("max_latency_ms"))
    if latency_max is not None:
        lat = _as_float(metrics.get("latency_ms"))
        if lat is not None and lat > latency_max:
            return False, f"{RC_CONSTRAINT_VIOLATION}:latency_ms>{latency_max}"

    mem_max = _as_float(c.get("max_memory_mb"))
    if mem_max is not None:
        mem = _as_float(metrics.get("memory_mb"))
        if mem is not None and mem > mem_max:
            return False, f"{RC_CONSTRAINT_VIOLATION}:memory_mb>{mem_max}"

    return True, "ok"


def _check_qualification(
    metrics: Mapping[str, Any],
    compiled: Mapping[str, Any],
) -> Tuple[bool, str, str, Dict[str, Any]]:
    """
    Qualification thresholds (usually from benchmarking.qualify_thresholds).

    Supports patterns:
      qualify_thresholds:
        guessing_entropy_max: 40
        rank_k_success@5_min: 0.60

    Rule mapping:
      *_max => metric <= threshold
      *_min => metric >= threshold
    """
    b = compiled.get("benchmarking", {}) or {}
    qt = b.get("qualify_thresholds", {}) or {}
    if not isinstance(qt, Mapping) or not qt:
        return False, RC_NOT_QUALIFIED, "no_qualify_thresholds", {"qualify_thresholds": {}}

    failures: List[str] = []
    checked: Dict[str, Any] = {}

    for k, thr in qt.items():
        if not isinstance(k, str) or not k.strip():
            continue

        thr_f = _as_float(thr)
        if thr_f is None:
            continue

        if k.endswith("_max"):
            metric_name = k[: -len("_max")]
            val = _as_float(metrics.get(metric_name))
            checked[k] = {"metric": metric_name, "op": "<=", "threshold": thr_f, "value": val}
            if val is None:
                failures.append(f"{RC_METRIC_MISSING}:{metric_name}")
            elif val > thr_f:
                failures.append(f"{metric_name}>{thr_f}")

        elif k.endswith("_min"):
            metric_name = k[: -len("_min")]
            val = _as_float(metrics.get(metric_name))
            checked[k] = {"metric": metric_name, "op": ">=", "threshold": thr_f, "value": val}
            if val is None:
                failures.append(f"{RC_METRIC_MISSING}:{metric_name}")
            elif val < thr_f:
                failures.append(f"{metric_name}<{thr_f}")

        else:
            # Unknown contract; ignore deterministically
            continue

    if failures:
        return (
            False,
            RC_NOT_QUALIFIED,
            "qualify_failed:" + ";".join(failures),
            {"qualify_thresholds": checked, "failures": failures},
        )

    return True, RC_QUALIFIED, "qualified", {"qualify_thresholds": checked, "failures": []}


# ============================================================
# Utilities
# ============================================================

def _now_iso() -> str:
    return now_iso()


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
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None
