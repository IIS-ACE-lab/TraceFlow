# services/mlops/core/hypothesis_selector.py
# ============================================================
# Hypothesis selection + switching + allocation policies
# - Pure logic (no file I/O)
# - Works off compiled config view + in-memory registries
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Literal

from services.mlops.core.registry import HypothesisRegistry, TrialRegistry, TrialRecord

Json = Dict[str, Any]

AllocationPolicyName = Literal[
    "uniform",
    "round_robin",
    "profile_weighted",
    "adaptive_bandit",
]

TrialStatus = Literal["created", "running", "completed", "pruned", "failed", "promoted"]


# ============================================================
# Errors
# ============================================================

class HypothesisSelectionError(RuntimeError):
    pass


# ============================================================
# Data models
# ============================================================

@dataclass(frozen=True)
class SelectionRequest:
    """
    Selection request.
    """
    project_name: str
    policy: AllocationPolicyName
    min_trials_per_hypothesis: int = 0
    exploration_fraction: float = 0.0

    # Optional allow/deny sets
    allow_hypotheses: Optional[Sequence[str]] = None
    deny_hypotheses: Optional[Sequence[str]] = None

    # Optional signals
    dataset_profile: Optional[Json] = None  # e.g. {"snr": 0.7, ...}
    now_trial_index: Optional[int] = None   # monotonic counter if caller has one


@dataclass(frozen=True)
class SelectionResult:
    project_name: str
    hypothesis_id: str
    reason: str
    debug: Json


# ============================================================
# Public API
# ============================================================

def select_next_hypothesis(
    compiled: Json,
    *,
    trial_registry: TrialRegistry,
    hypothesis_registry: HypothesisRegistry,
    project_name: str,
    dataset_profile: Optional[Json] = None,
    now_trial_index: Optional[int] = None,
) -> SelectionResult:
    """
    Determine the next hypothesis to run a trial for, using per-project allocation policy.

    Inputs:
      - compiled: deterministic compiled config (from core/context.py)
      - trial_registry/hypothesis_registry: in-memory registries
      - project_name: project in compiled["project"]["name"]
      - dataset_profile: optional signals for profile_weighted/adaptive_bandit
      - now_trial_index: optional integer to make policy deterministic across calls

    Output:
      - SelectionResult (hypothesis_id + reason + debug)
    """
    p = _get_project_view(compiled, project_name=project_name)

    allocation = (p.get("allocation") or {})
    policy = str(allocation.get("policy") or "uniform").strip()
    if policy not in {"uniform", "round_robin", "profile_weighted", "adaptive_bandit"}:
        raise HypothesisSelectionError(f"[hypothesis_selector] invalid allocation.policy: {policy}")

    req = SelectionRequest(
        project_name=project_name,
        policy=policy,  # type: ignore[arg-type]
        min_trials_per_hypothesis=int(allocation.get("min_trials_per_hypothesis") or 0),
        exploration_fraction=float(allocation.get("exploration_fraction") or 0.0),
        allow_hypotheses=allocation.get("allow_hypotheses"),
        deny_hypotheses=allocation.get("deny_hypotheses"),
        dataset_profile=dataset_profile,
        now_trial_index=now_trial_index,
    )

    hypothesis_ids = _list_project_hypothesis_ids(p, req=req)
    if not hypothesis_ids:
        raise HypothesisSelectionError(f"[hypothesis_selector] no hypotheses available for project={project_name}")

    # Ensure registries have baseline entries
    for hid in hypothesis_ids:
        hypothesis_registry.ensure(project_name=project_name, hypothesis_id=hid)

    # Collect counts by hypothesis (from trial registry)
    counts = _trial_counts_by_hypothesis(
        trial_registry.list(project_name=project_name),
        hypothesis_ids=hypothesis_ids,
    )

    # Phase 1: enforce min_trials_per_hypothesis (deterministic fill)
    under = [hid for hid in hypothesis_ids if counts[hid]["started"] < req.min_trials_per_hypothesis]
    if under:
        hid = _tie_break_lowest_started_then_id(under, counts)
        return SelectionResult(
            project_name=project_name,
            hypothesis_id=hid,
            reason="min_trials_per_hypothesis_fill",
            debug={"policy": req.policy, "min_trials_per_hypothesis": req.min_trials_per_hypothesis, "counts": counts},
        )

    # Phase 2: policy selection
    if req.policy == "uniform":
        hid = _pick_uniform(hypothesis_ids, counts, now_trial_index=req.now_trial_index)
        return SelectionResult(project_name=project_name, hypothesis_id=hid, reason="uniform", debug={"counts": counts})

    if req.policy == "round_robin":
        hid = _pick_round_robin(hypothesis_ids, counts)
        return SelectionResult(project_name=project_name, hypothesis_id=hid, reason="round_robin", debug={"counts": counts})

    if req.policy == "profile_weighted":
        weights = _compute_profile_weights(p, hypothesis_ids, dataset_profile=req.dataset_profile)
        hid = _pick_weighted_deterministic(
            hypothesis_ids,
            weights=weights,
            counts=counts,
            exploration_fraction=req.exploration_fraction,
            now_trial_index=req.now_trial_index,
        )
        return SelectionResult(
            project_name=project_name,
            hypothesis_id=hid,
            reason="profile_weighted",
            debug={"counts": counts, "weights": weights, "exploration_fraction": req.exploration_fraction},
        )

    # adaptive_bandit (placeholder but deterministic + sane)
    # Strategy:
    #  - exploitation score from best_primary_metric (if available) + completion rate
    #  - exploration fraction ensures coverage
    weights = _compute_bandit_weights(
        hypothesis_registry=hypothesis_registry,
        project_name=project_name,
        hypothesis_ids=hypothesis_ids,
        counts=counts,
    )
    hid = _pick_weighted_deterministic(
        hypothesis_ids,
        weights=weights,
        counts=counts,
        exploration_fraction=req.exploration_fraction,
        now_trial_index=req.now_trial_index,
    )
    return SelectionResult(
        project_name=project_name,
        hypothesis_id=hid,
        reason="adaptive_bandit",
        debug={"counts": counts, "weights": weights, "exploration_fraction": req.exploration_fraction},
    )


def switch_hypothesis(
    *,
    compiled: Json,
    project_name: str,
    current_hypothesis_id: Optional[str],
    requested_hypothesis_id: Optional[str],
    trial_registry: TrialRegistry,
    hypothesis_registry: HypothesisRegistry,
    dataset_profile: Optional[Json] = None,
    now_trial_index: Optional[int] = None,
) -> SelectionResult:
    """
    Decide whether to switch hypotheses.

    Rules:
      - If requested_hypothesis_id is provided: validate and return it.
      - Else: select_next_hypothesis based on allocation policy.
      - If current_hypothesis_id is invalid -> always select_next_hypothesis.

    Pure: does not mutate registries; caller mutates on actual trial start.
    """
    p = _get_project_view(compiled, project_name=project_name)
    available = set(_list_project_hypothesis_ids(p, req=SelectionRequest(project_name=project_name, policy="uniform")))
    if requested_hypothesis_id is not None:
        hid = str(requested_hypothesis_id).strip()
        if hid not in available:
            raise HypothesisSelectionError(
                f"[hypothesis_selector] requested hypothesis '{hid}' not in project '{project_name}' hypotheses"
            )
        return SelectionResult(
            project_name=project_name,
            hypothesis_id=hid,
            reason="requested",
            debug={"requested_hypothesis_id": hid},
        )

    if current_hypothesis_id is not None and str(current_hypothesis_id).strip() in available:
        # We still may switch based on policy; keep simple:
        # If policy selects same -> no switch, else switch.
        nxt = select_next_hypothesis(
            compiled,
            trial_registry=trial_registry,
            hypothesis_registry=hypothesis_registry,
            project_name=project_name,
            dataset_profile=dataset_profile,
            now_trial_index=now_trial_index,
        )
        if nxt.hypothesis_id == str(current_hypothesis_id).strip():
            return SelectionResult(
                project_name=project_name,
                hypothesis_id=nxt.hypothesis_id,
                reason="keep_current",
                debug={"current_hypothesis_id": current_hypothesis_id, **(nxt.debug or {})},
            )
        return SelectionResult(
            project_name=project_name,
            hypothesis_id=nxt.hypothesis_id,
            reason="switch_by_policy",
            debug={"current_hypothesis_id": current_hypothesis_id, **(nxt.debug or {})},
        )

    # current invalid or None -> select
    return select_next_hypothesis(
        compiled,
        trial_registry=trial_registry,
        hypothesis_registry=hypothesis_registry,
        project_name=project_name,
        dataset_profile=dataset_profile,
        now_trial_index=now_trial_index,
    )


# ============================================================
# Internals
# ============================================================

def _get_project_view(compiled: Json, *, project_name: str) -> Json:
    pj = (compiled.get("project") or {})
    name = (pj.get("name") or "")
    if str(name).strip() != str(project_name).strip():
        # compiled is per-project view in your design; enforce
        raise HypothesisSelectionError(
            f"[hypothesis_selector] compiled project mismatch: compiled.project.name={name} vs requested={project_name}"
        )
    return pj


def _list_project_hypothesis_ids(project_view: Json, *, req: SelectionRequest) -> List[str]:
    hyps = project_view.get("hypotheses") or []
    if not isinstance(hyps, list):
        raise HypothesisSelectionError("[hypothesis_selector] project.hypotheses must be a list")
    ids: List[str] = []
    for i, h in enumerate(hyps):
        if not isinstance(h, dict):
            continue
        hid = h.get("hypothesis_id") or h.get("id")
        if not isinstance(hid, str) or not hid.strip():
            raise HypothesisSelectionError(f"[hypothesis_selector] hypothesis[{i}] missing hypothesis_id")
        if h.get("active", True) is False:
            continue
        ids.append(hid.strip())

    allow = set([x.strip() for x in (req.allow_hypotheses or []) if isinstance(x, str) and x.strip()])
    deny = set([x.strip() for x in (req.deny_hypotheses or []) if isinstance(x, str) and x.strip()])

    if allow:
        ids = [x for x in ids if x in allow]
    if deny:
        ids = [x for x in ids if x not in deny]

    # deterministic order
    ids = sorted(set(ids))
    return ids


def _trial_counts_by_hypothesis(trials: List[TrialRecord], *, hypothesis_ids: Sequence[str]) -> Json:
    counts: Json = {
        hid: {"started": 0, "completed": 0, "pruned": 0, "failed": 0, "promoted": 0, "running": 0}
        for hid in hypothesis_ids
    }
    for t in trials:
        hid = t.hypothesis_id
        if hid not in counts:
            continue
        counts[hid]["started"] += 1
        st = str(t.status or "created")
        if st == "running":
            counts[hid]["running"] += 1
        elif st == "completed":
            counts[hid]["completed"] += 1
        elif st == "pruned":
            counts[hid]["pruned"] += 1
        elif st == "failed":
            counts[hid]["failed"] += 1
        elif st == "promoted":
            counts[hid]["promoted"] += 1
    return counts


def _tie_break_lowest_started_then_id(ids: Sequence[str], counts: Json) -> str:
    return sorted(ids, key=lambda hid: (counts[hid]["started"], hid))[0]


def _pick_uniform(hypothesis_ids: Sequence[str], counts: Json, *, now_trial_index: Optional[int]) -> str:
    """
    Uniform selection but deterministic:
      - pick by (started, hypothesis_id) to balance and avoid randomness
      - if now_trial_index is provided, rotate tie breaks deterministically
    """
    ordered = sorted(hypothesis_ids, key=lambda hid: (counts[hid]["started"], hid))
    if now_trial_index is None:
        return ordered[0]
    # deterministic rotation among equal-started
    min_started = counts[ordered[0]]["started"]
    bucket = [hid for hid in ordered if counts[hid]["started"] == min_started]
    return bucket[now_trial_index % len(bucket)]


def _pick_round_robin(hypothesis_ids: Sequence[str], counts: Json) -> str:
    """
    Round-robin implemented as "least started" selection (deterministic).
    Equivalent to rotating if caller increments started on selection.
    """
    return sorted(hypothesis_ids, key=lambda hid: (counts[hid]["started"], hid))[0]


def _compute_profile_weights(project_view: Json, hypothesis_ids: Sequence[str], *, dataset_profile: Optional[Json]) -> Dict[str, float]:
    """
    Compute deterministic weights using dataset_profile signals (if available).

    If no dataset_profile: fall back to equal weights.

    Schema expectation (optional):
      project_view may include:
        data_profile:
          signals: [...]
      hypothesis may include:
        allocation_hints:
          signal_weights:
            snr: 1.2
            temporal_dependency: 0.7

    We keep it simple: dot(signal_weight, normalized_signal_value) + 1e-6.
    """
    if not dataset_profile or not isinstance(dataset_profile, dict):
        return {hid: 1.0 for hid in hypothesis_ids}

    # normalize signals to [0,1] roughly by clamping numeric values
    def norm(v: Any) -> float:
        if isinstance(v, bool):
            return 0.0
        if isinstance(v, (int, float)):
            x = float(v)
            if x != x:  # nan
                return 0.0
            if x < 0:
                return 0.0
            # if already 0..1 use it; else compress
            if x <= 1.0:
                return x
            return 1.0
        return 0.0

    # map hypothesis_id -> block
    hyp_blocks: Dict[str, Json] = {}
    for h in project_view.get("hypotheses") or []:
        if isinstance(h, dict):
            hid = (h.get("hypothesis_id") or h.get("id") or "").strip()
            if hid:
                hyp_blocks[hid] = h

    weights: Dict[str, float] = {}
    for hid in hypothesis_ids:
        hb = hyp_blocks.get(hid, {})
        hints = hb.get("allocation_hints") or {}
        sw = (hints.get("signal_weights") or {}) if isinstance(hints, dict) else {}
        score = 0.0
        if isinstance(sw, dict) and sw:
            for sig, w in sw.items():
                if not isinstance(sig, str):
                    continue
                if not isinstance(w, (int, float)) or isinstance(w, bool):
                    continue
                score += float(w) * norm(dataset_profile.get(sig))
        else:
            # default: weak uniform bias
            score = 1.0
        weights[hid] = max(1e-6, float(score))
    return weights


def _compute_bandit_weights(
    *,
    hypothesis_registry: HypothesisRegistry,
    project_name: str,
    hypothesis_ids: Sequence[str],
    counts: Json,
) -> Dict[str, float]:
    """
    Deterministic bandit-ish weights (no randomness).

    Uses:
      - best_primary_metric if available (higher is better proxy; unknown -> neutral)
      - completion rate signal
      - penalize high failure rate slightly

    Note: objectives direction is not imported here; core/trial_judge will
    handle directionality. This is just allocation.
    """
    weights: Dict[str, float] = {}
    for hid in hypothesis_ids:
        rec = hypothesis_registry.get(project_name=project_name, hypothesis_id=hid)
        best = rec.best_primary_metric
        # completion rate (completed / started)
        started = max(1, int(counts[hid]["started"]))
        completed = int(counts[hid]["completed"])
        failed = int(counts[hid]["failed"])
        pruned = int(counts[hid]["pruned"])

        completion_rate = completed / started
        failure_rate = failed / started
        prune_rate = pruned / started

        base = 1.0
        if isinstance(best, (int, float)) and not isinstance(best, bool):
            # compress best into a stable bounded contribution
            # (works regardless of actual metric scale)
            b = float(best)
            if b == b:
                base += (b / (abs(b) + 1.0))  # in (-1,1)
        base += 0.5 * completion_rate
        base -= 0.25 * failure_rate
        base -= 0.10 * prune_rate

        weights[hid] = max(1e-6, float(base))
    return weights


def _pick_weighted_deterministic(
    hypothesis_ids: Sequence[str],
    *,
    weights: Dict[str, float],
    counts: Json,
    exploration_fraction: float,
    now_trial_index: Optional[int],
) -> str:
    """
    Deterministic "weighted" picker:
      - With probability replaced by deterministic schedule:
        If exploration_fraction > 0 and now_trial_index hits an exploration slot,
        pick least-started (exploration).
      - Else pick max(weight / (1 + started)) to balance exploration/exploitation.

    No randomness; fully reproducible given counts + now_trial_index.
    """
    ef = float(exploration_fraction or 0.0)
    if ef < 0.0:
        ef = 0.0
    if ef > 0.9:
        ef = 0.9

    if now_trial_index is not None and ef > 0.0:
        # deterministic exploration cadence: every k-th trial is exploration
        k = max(1, int(round(1.0 / ef)))
        if now_trial_index % k == 0:
            return sorted(hypothesis_ids, key=lambda hid: (counts[hid]["started"], hid))[0]

    # exploit: maximize normalized weight with anti-starvation factor
    def score(hid: str) -> float:
        w = float(weights.get(hid, 1.0))
        started = float(counts[hid]["started"])
        return w / (1.0 + started)

    ranked = sorted(hypothesis_ids, key=lambda hid: (-score(hid), counts[hid]["started"], hid))
    return ranked[0]
# ============================================================
