# services/mlops/schemas/registry.py
# ============================================================
# Registry schema (minimal)
# - Validates auto_mlops.yaml: registry.*
# - Contracts for: auto_run_map / leaderboard / pareto_front
#
# NOTE:
# Registry is for "what summaries to persist" (not filenames).
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from services.mlops.schemas.common import (
    forbid_unknown_keys,
    normalize_bool,
    normalize_dict,
    normalize_list,
    normalize_int,
)


# ------------------------------------------------------------
# Data model
# ------------------------------------------------------------

@dataclass(frozen=True)
class AutoRunMapSpec:
    """
    Controls the content of the per-trial "run map" summary:
    - metric snapshots at selected epochs
    - prune reason
    - exception trace
    """
    include_metrics_snapshot_epochs: List[int] = field(default_factory=list)
    include_prune_reason: bool = True
    include_exception_trace: bool = True


@dataclass(frozen=True)
class LeaderboardSpec:
    """
    Controls the content of the ranked leaderboard summary.
    """
    top_k: int = 50


@dataclass(frozen=True)
class ParetoFrontSpec:
    """
    Controls whether dominated points are included in pareto exports.
    """
    include_dominated: bool = False


@dataclass(frozen=True)
class RegistrySpec:
    auto_run_map: AutoRunMapSpec = field(default_factory=AutoRunMapSpec)
    leaderboard: LeaderboardSpec = field(default_factory=LeaderboardSpec)
    pareto_front: ParetoFrontSpec = field(default_factory=ParetoFrontSpec)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _as_int_list(v: Any, *, ctx: str) -> List[int]:
    items = normalize_list(v)
    out: List[int] = []
    for i, x in enumerate(items):
        # normalize_int already rejects bool-ish cases via int(True)=1 ambiguity
        xi = normalize_int(x)
        if xi is None:
            raise ValueError(f"[{ctx}] item {i} must be an int (got {x!r})")
        out.append(int(xi))
    # stable + dedup
    return sorted(set(out))


def _as_pos_int(v: Any, *, ctx: str, default: int) -> int:
    n = normalize_int(v, default=default)
    if n is None:
        raise ValueError(f"[{ctx}] must be int (got {v!r})")
    if n <= 0:
        raise ValueError(f"[{ctx}] must be > 0 (got {n})")
    return int(n)


# ------------------------------------------------------------
# Validators
# ------------------------------------------------------------

def validate_auto_run_map_block(block: Any, *, ctx: str, strict: bool) -> AutoRunMapSpec:
    d = normalize_dict(block)

    allowed = {
        "include_metrics_snapshot_epochs",
        "include_prune_reason",
        "include_exception_trace",
    }
    if strict:
        forbid_unknown_keys(d, allowed=allowed, ctx=ctx)

    include_epochs = _as_int_list(
        d.get("include_metrics_snapshot_epochs", []),
        ctx=f"{ctx}.include_metrics_snapshot_epochs",
    )

    include_prune_reason = normalize_bool(d.get("include_prune_reason"), default=True)
    include_exception_trace = normalize_bool(d.get("include_exception_trace"), default=True)

    return AutoRunMapSpec(
        include_metrics_snapshot_epochs=include_epochs,
        include_prune_reason=include_prune_reason,
        include_exception_trace=include_exception_trace,
    )


def validate_leaderboard_block(block: Any, *, ctx: str, strict: bool) -> LeaderboardSpec:
    d = normalize_dict(block)

    allowed = {"top_k"}
    if strict:
        forbid_unknown_keys(d, allowed=allowed, ctx=ctx)

    top_k = _as_pos_int(d.get("top_k", 50), ctx=f"{ctx}.top_k", default=50)
    return LeaderboardSpec(top_k=top_k)


def validate_pareto_front_block(block: Any, *, ctx: str, strict: bool) -> ParetoFrontSpec:
    d = normalize_dict(block)

    allowed = {"include_dominated"}
    if strict:
        forbid_unknown_keys(d, allowed=allowed, ctx=ctx)

    include_dominated = normalize_bool(d.get("include_dominated"), default=False)
    return ParetoFrontSpec(include_dominated=include_dominated)


def validate_registry_block(block: Any, *, ctx: str = "registry", strict: bool = False) -> RegistrySpec:
    """
    Validates (all optional):
      registry:
        auto_run_map: {...}
        leaderboard: {...}
        pareto_front: {...}

    Returns defaults if registry is missing/empty.
    """
    d = normalize_dict(block)

    allowed = {"auto_run_map", "leaderboard", "pareto_front"}
    if strict:
        forbid_unknown_keys(d, allowed=allowed, ctx=ctx)

    arm = validate_auto_run_map_block(
        d.get("auto_run_map", {}),
        ctx=f"{ctx}.auto_run_map",
        strict=strict,
    )
    lb = validate_leaderboard_block(
        d.get("leaderboard", {}),
        ctx=f"{ctx}.leaderboard",
        strict=strict,
    )
    pf = validate_pareto_front_block(
        d.get("pareto_front", {}),
        ctx=f"{ctx}.pareto_front",
        strict=strict,
    )

    return RegistrySpec(auto_run_map=arm, leaderboard=lb, pareto_front=pf)
