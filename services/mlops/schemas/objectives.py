# services/mlops/schemas/objectives.py
# ============================================================
# Objectives schema (multi-objective directions + primary + pareto)
# Matches YAML shape:
#
# objectives:
#   directions:
#     minimize: [guessing_entropy, ...]
#     maximize: [{"rank_k_success": {"k": 5}}, accuracy, ...]
#   primary: guessing_entropy
#   tie_breakers: [rank_k_success@5, mean_rank, accuracy]
#   pareto:
#     enabled: true
#     export_front: true
#     selection_policy:
#       name: weighted_tchebycheff
#       weights: {guessing_entropy: 0.55, rank_k_success@5: 0.30, accuracy: 0.15}
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from services.mlops.schemas.common import (
    ensure_non_empty,
    ensure_one_of,
    normalize_bool,
    normalize_list,
    normalize_float,
    require_keys,
)

# ----------------------------
# Data models
# ----------------------------

@dataclass(frozen=True)
class ParetoSelectionPolicy:
    name: str
    weights: Dict[str, float]


@dataclass(frozen=True)
class ParetoSpec:
    enabled: bool
    export_front: bool
    selection_policy: Optional[ParetoSelectionPolicy] = None


@dataclass(frozen=True)
class ObjectivesSpec:
    # directions -> metric token strings (normalized)
    minimize: List[str]
    maximize: List[str]

    primary: str
    tie_breakers: List[str]

    pareto: ParetoSpec


# ----------------------------
# Normalization helpers
# ----------------------------

def _metric_token_from_item(item: Any, *, ctx: str) -> str:
    """
    Normalize objective items into a stable metric token string.

    Allowed YAML forms:
      - "accuracy"
      - {"rank_k_success": {"k": 5}}  -> "rank_k_success@5"
      - {"rank_k_success": {"k": "5"}}-> "rank_k_success@5"
    """
    # string metric name
    if isinstance(item, str):
        s = item.strip()
        if not s:
            raise ValueError(f"[{ctx}] empty metric name")
        return s

    # dict metric with args
    if isinstance(item, dict):
        if len(item) != 1:
            raise ValueError(f"[{ctx}] metric dict must have exactly 1 key (got {list(item.keys())})")
        name, args = next(iter(item.items()))
        name = str(name).strip()
        if not name:
            raise ValueError(f"[{ctx}] metric dict key must be a non-empty string")

        # allow args to be None / empty
        if args is None:
            return name

        if not isinstance(args, dict):
            raise ValueError(f"[{ctx}] metric args must be a dict (got {type(args).__name__})")

        # special normalization: rank_k_success + k -> rank_k_success@k
        if name == "rank_k_success":
            k = args.get("k")
            if k is None:
                raise ValueError(f"[{ctx}] rank_k_success requires args.k")
            try:
                k_int = int(k)
            except Exception:
                raise ValueError(f"[{ctx}] rank_k_success args.k must be int-like (got {k!r})")
            return f"{name}@{k_int}"

        # generic: name@key=value,key2=value2 (stable sorted)
        parts: List[str] = []
        for kk in sorted(args.keys(), key=lambda x: str(x)):
            vv = args[kk]
            parts.append(f"{kk}={vv}")
        suffix = ",".join(parts)
        return f"{name}@{suffix}" if suffix else name

    raise ValueError(f"[{ctx}] objective item must be str or single-key dict (got {type(item).__name__})")


def _parse_directions(directions: Dict[str, Any], *, ctx: str) -> Tuple[List[str], List[str]]:
    if not isinstance(directions, dict):
        raise ValueError(f"[{ctx}] directions must be a mapping")

    minimize_raw = normalize_list(directions.get("minimize"))
    maximize_raw = normalize_list(directions.get("maximize"))

    ensure_non_empty(minimize_raw + maximize_raw, ctx=f"{ctx}.directions")

    minimize: List[str] = []
    maximize: List[str] = []

    for it in minimize_raw:
        minimize.append(_metric_token_from_item(it, ctx=f"{ctx}.directions.minimize"))

    for it in maximize_raw:
        maximize.append(_metric_token_from_item(it, ctx=f"{ctx}.directions.maximize"))

    return minimize, maximize


def _validate_pareto_block(pareto_cfg: Any, *, ctx: str) -> ParetoSpec:
    if pareto_cfg is None:
        return ParetoSpec(enabled=False, export_front=False, selection_policy=None)

    if not isinstance(pareto_cfg, dict):
        raise ValueError(f"[{ctx}] pareto must be a mapping")

    enabled = normalize_bool(pareto_cfg.get("enabled"), default=False)
    export_front = normalize_bool(pareto_cfg.get("export_front"), default=False)

    policy_cfg = pareto_cfg.get("selection_policy")
    if not enabled:
        # if pareto disabled, ignore policy
        return ParetoSpec(enabled=False, export_front=export_front, selection_policy=None)

    if policy_cfg is None:
        # enabled but no policy -> still OK, selection can be done elsewhere
        return ParetoSpec(enabled=True, export_front=export_front, selection_policy=None)

    if not isinstance(policy_cfg, dict):
        raise ValueError(f"[{ctx}] selection_policy must be a mapping")

    require_keys(policy_cfg, ["name"], ctx=f"{ctx}.selection_policy")
    name = str(policy_cfg["name"]).strip()
    ensure_one_of(
        name,
        ["weighted_tchebycheff", "epsilon_constraint", "knee_point", "max_hypervolume"],
        ctx=f"{ctx}.selection_policy.name",
    )

    weights_cfg = policy_cfg.get("weights") or {}
    if not isinstance(weights_cfg, dict):
        raise ValueError(f"[{ctx}] selection_policy.weights must be a mapping")

    weights: Dict[str, float] = {}
    for k, v in weights_cfg.items():
        kk = str(k).strip()
        if not kk:
            raise ValueError(f"[{ctx}] selection_policy.weights contains empty key")
        w = normalize_float(v)
        if w is None:
            raise ValueError(f"[{ctx}] selection_policy.weights[{kk}] must be a number")
        weights[kk] = float(w)

    return ParetoSpec(
        enabled=True,
        export_front=export_front,
        selection_policy=ParetoSelectionPolicy(name=name, weights=weights),
    )


# ----------------------------
# Public validator
# ----------------------------

def validate_objectives_block(cfg: Dict[str, Any], *, ctx: str = "objectives") -> ObjectivesSpec:
    """
    Validate 'objectives' block with this YAML shape:

      objectives:
        directions:
          minimize: [<metric>, ...]
          maximize: [<metric or {metric: {args}}>, ...]
        primary: <metric-token-string>
        tie_breakers: [<metric-token-string>, ...]
        pareto:
          enabled: bool
          export_front: bool
          selection_policy:
            name: weighted_tchebycheff | epsilon_constraint | knee_point | max_hypervolume
            weights:
              <metric-token-string>: <float>
              ...

    Metric token normalization:
      - "accuracy" -> "accuracy"
      - {"rank_k_success": {"k": 5}} -> "rank_k_success@5"
      - For other dict-metrics: "<name>@k=v,k2=v2" (stable sorted keys)
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"[{ctx}] objectives must be a mapping")

    require_keys(cfg, ["directions", "primary"], ctx=ctx)

    minimize, maximize = _parse_directions(cfg["directions"], ctx=ctx)

    primary = str(cfg.get("primary", "")).strip()
    if not primary:
        raise ValueError(f"[{ctx}] primary must be a non-empty string")

    tie_raw = normalize_list(cfg.get("tie_breakers"))
    tie_breakers: List[str] = [str(x).strip() for x in tie_raw if str(x).strip()]

    pareto = _validate_pareto_block(cfg.get("pareto"), ctx=ctx)

    # Optional: warn-like checks (we keep them as hard errors only when clearly invalid)
    all_metrics = set(minimize + maximize)
    if primary not in all_metrics:
        raise ValueError(f"[{ctx}] primary '{primary}' must appear in directions minimize/maximize")

    # If weights exist, ensure they reference known metrics (common expectation)
    if pareto.selection_policy and pareto.selection_policy.weights:
        unknown = [k for k in pareto.selection_policy.weights.keys() if k not in all_metrics and k != primary]
        if unknown:
            raise ValueError(f"[{ctx}] pareto.weights contains unknown metrics: {unknown}")

    return ObjectivesSpec(
        minimize=minimize,
        maximize=maximize,
        primary=primary,
        tie_breakers=tie_breakers,
        pareto=pareto,
    )
