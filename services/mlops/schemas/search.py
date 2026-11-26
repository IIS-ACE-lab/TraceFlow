# services/mlops/schemas/search.py
# ============================================================
# Search schema (Optuna-style)
# - Validates auto_mlops.yaml: search.*
# - Strict on shape/types; flexible on extra keys for forward-compat.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mlops.schemas.common import (
    forbid_unknown_keys,
    normalize_dict,
    normalize_list,
    normalize_str,
    require_keys,
)

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

_ALLOWED_ENGINES = {"optuna"}

_ALLOWED_SAMPLERS = {"tpe", "random", "grid", "cmaes"}
_ALLOWED_PRUNERS = {"asha", "hyperband", "none"}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _as_int(v: Any, *, ctx: str, min_value: Optional[int] = None) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"[schemas/search] {ctx} must be int (got {type(v)})")
    if min_value is not None and v < min_value:
        raise ValueError(f"[schemas/search] {ctx} must be >= {min_value} (got {v})")
    return int(v)


def _as_optional_int(v: Any, *, ctx: str, min_value: Optional[int] = None) -> Optional[int]:
    if v is None:
        return None
    return _as_int(v, ctx=ctx, min_value=min_value)


def _as_bool(v: Any, *, ctx: str) -> bool:
    if not isinstance(v, bool):
        raise ValueError(f"[schemas/search] {ctx} must be bool (got {type(v)})")
    return v


def _as_optional_bool(v: Any, *, ctx: str) -> Optional[bool]:
    if v is None:
        return None
    return _as_bool(v, ctx=ctx)


def _as_float(v: Any, *, ctx: str, min_value: Optional[float] = None) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"[schemas/search] {ctx} must be number (got {type(v)})")
    fv = float(v)
    if min_value is not None and fv < min_value:
        raise ValueError(f"[schemas/search] {ctx} must be >= {min_value} (got {fv})")
    return fv


def _as_optional_float(v: Any, *, ctx: str, min_value: Optional[float] = None) -> Optional[float]:
    if v is None:
        return None
    return _as_float(v, ctx=ctx, min_value=min_value)


def _non_empty_str(v: Any, *, ctx: str) -> str:
    s = normalize_str(v, default="").strip()
    if not s:
        raise ValueError(f"[schemas/search] {ctx} must be non-empty string")
    return s


def _validate_model_tag_from(v: Any, *, ctx: str) -> List[str]:
    items = normalize_list(v)
    if not items:
        raise ValueError(f"[schemas/search] {ctx} must be non-empty list[str]")
    out: List[str] = []
    for i, x in enumerate(items):
        if not isinstance(x, str) or not x.strip():
            raise ValueError(f"[schemas/search] {ctx}[{i}] must be non-empty string")
        out.append(x.strip())
    return out


# ------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------

@dataclass(frozen=True)
class SamplerSpec:
    name: str
    seed: Optional[int] = None
    startup_trials: Optional[int] = None
    multivariate: Optional[bool] = None
    group: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrunerSpec:
    name: str
    min_resource: Optional[int] = None
    reduction_factor: Optional[int] = None
    min_early_stopping_rate: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetSpec:
    max_trials: int
    parallel_trials: int = 1
    max_gpu_hours: Optional[float] = None
    max_wall_hours: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialNamingSpec:
    pattern: str
    model_tag_from: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchSpec:
    engine: str
    sampler: SamplerSpec
    pruner: PrunerSpec
    budget: BudgetSpec
    trial_naming: TrialNamingSpec
    extra: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------
# Validators
# ------------------------------------------------------------

def validate_sampler_block(block: Any, *, ctx: str = "search.sampler", strict: bool = False) -> SamplerSpec:
    d = normalize_dict(block)
    require_keys(d, ["name"], ctx=ctx)

    if strict:
        forbid_unknown_keys(
            d,
            {"name", "seed", "startup_trials", "multivariate", "group"},
            ctx=ctx,
        )

    name = _non_empty_str(d.get("name"), ctx=f"{ctx}.name")
    if name not in _ALLOWED_SAMPLERS:
        raise ValueError(f"[schemas/search] {ctx}.name='{name}' invalid. Allowed: {sorted(_ALLOWED_SAMPLERS)}")

    seed = _as_optional_int(d.get("seed"), ctx=f"{ctx}.seed") if "seed" in d else None
    startup_trials = _as_optional_int(d.get("startup_trials"), ctx=f"{ctx}.startup_trials", min_value=0) if "startup_trials" in d else None
    multivariate = _as_optional_bool(d.get("multivariate"), ctx=f"{ctx}.multivariate") if "multivariate" in d else None
    group = _as_optional_bool(d.get("group"), ctx=f"{ctx}.group") if "group" in d else None

    allowed = {"name", "seed", "startup_trials", "multivariate", "group"}
    extra = {k: v for k, v in d.items() if k not in allowed}

    return SamplerSpec(
        name=name,
        seed=seed,
        startup_trials=startup_trials,
        multivariate=multivariate,
        group=group,
        extra=extra,
    )


def validate_pruner_block(block: Any, *, ctx: str = "search.pruner", strict: bool = False) -> PrunerSpec:
    d = normalize_dict(block)
    require_keys(d, ["name"], ctx=ctx)

    if strict:
        forbid_unknown_keys(
            d,
            {"name", "min_resource", "reduction_factor", "min_early_stopping_rate"},
            ctx=ctx,
        )

    name = _non_empty_str(d.get("name"), ctx=f"{ctx}.name")
    if name not in _ALLOWED_PRUNERS:
        raise ValueError(f"[schemas/search] {ctx}.name='{name}' invalid. Allowed: {sorted(_ALLOWED_PRUNERS)}")

    # If pruner is "none", still allow fields (but they are ignored downstream).
    min_resource = _as_optional_int(d.get("min_resource"), ctx=f"{ctx}.min_resource", min_value=0) if "min_resource" in d else None
    reduction_factor = _as_optional_int(d.get("reduction_factor"), ctx=f"{ctx}.reduction_factor", min_value=1) if "reduction_factor" in d else None
    min_early = _as_optional_int(d.get("min_early_stopping_rate"), ctx=f"{ctx}.min_early_stopping_rate", min_value=0) if "min_early_stopping_rate" in d else None

    allowed = {"name", "min_resource", "reduction_factor", "min_early_stopping_rate"}
    extra = {k: v for k, v in d.items() if k not in allowed}

    return PrunerSpec(
        name=name,
        min_resource=min_resource,
        reduction_factor=reduction_factor,
        min_early_stopping_rate=min_early,
        extra=extra,
    )


def validate_budget_block(block: Any, *, ctx: str = "search.budget", strict: bool = False) -> BudgetSpec:
    d = normalize_dict(block)
    require_keys(d, ["max_trials"], ctx=ctx)

    if strict:
        forbid_unknown_keys(
            d,
            {"max_trials", "parallel_trials", "max_gpu_hours", "max_wall_hours"},
            ctx=ctx,
        )

    max_trials = _as_int(d.get("max_trials"), ctx=f"{ctx}.max_trials", min_value=1)
    parallel_trials = _as_int(d.get("parallel_trials", 1), ctx=f"{ctx}.parallel_trials", min_value=1)

    max_gpu_hours = _as_optional_float(d.get("max_gpu_hours"), ctx=f"{ctx}.max_gpu_hours", min_value=0.0) if "max_gpu_hours" in d else None
    max_wall_hours = _as_optional_float(d.get("max_wall_hours"), ctx=f"{ctx}.max_wall_hours", min_value=0.0) if "max_wall_hours" in d else None

    allowed = {"max_trials", "parallel_trials", "max_gpu_hours", "max_wall_hours"}
    extra = {k: v for k, v in d.items() if k not in allowed}

    return BudgetSpec(
        max_trials=max_trials,
        parallel_trials=parallel_trials,
        max_gpu_hours=max_gpu_hours,
        max_wall_hours=max_wall_hours,
        extra=extra,
    )


def validate_trial_naming_block(block: Any, *, ctx: str = "search.trial_naming", strict: bool = False) -> TrialNamingSpec:
    d = normalize_dict(block)
    require_keys(d, ["pattern"], ctx=ctx)

    if strict:
        forbid_unknown_keys(d, {"pattern", "model_tag_from"}, ctx=ctx)

    pattern = _non_empty_str(d.get("pattern"), ctx=f"{ctx}.pattern")

    model_tag_from: List[str] = []
    if "model_tag_from" in d and d["model_tag_from"] is not None:
        model_tag_from = _validate_model_tag_from(d["model_tag_from"], ctx=f"{ctx}.model_tag_from")

    allowed = {"pattern", "model_tag_from"}
    extra = {k: v for k, v in d.items() if k not in allowed}

    return TrialNamingSpec(
        pattern=pattern,
        model_tag_from=model_tag_from,
        extra=extra,
    )


def validate_search_block(block: Any, *, ctx: str = "search", strict: bool = False) -> SearchSpec:
    """
    Validates:
      search:
        engine: optuna
        sampler: {...}
        pruner: {...}
        budget: {...}
        trial_naming: {...}
    """
    d = normalize_dict(block)
    require_keys(d, ["engine", "sampler", "pruner", "budget", "trial_naming"], ctx=ctx)

    if strict:
        forbid_unknown_keys(d, {"engine", "sampler", "pruner", "budget", "trial_naming"}, ctx=ctx)

    engine = _non_empty_str(d.get("engine"), ctx=f"{ctx}.engine")
    if engine not in _ALLOWED_ENGINES:
        raise ValueError(f"[schemas/search] {ctx}.engine='{engine}' invalid. Allowed: {sorted(_ALLOWED_ENGINES)}")

    sampler = validate_sampler_block(d.get("sampler"), ctx=f"{ctx}.sampler", strict=strict)
    pruner = validate_pruner_block(d.get("pruner"), ctx=f"{ctx}.pruner", strict=strict)
    budget = validate_budget_block(d.get("budget"), ctx=f"{ctx}.budget", strict=strict)
    trial_naming = validate_trial_naming_block(d.get("trial_naming"), ctx=f"{ctx}.trial_naming", strict=strict)

    allowed = {"engine", "sampler", "pruner", "budget", "trial_naming"}
    extra = {k: v for k, v in d.items() if k not in allowed}

    return SearchSpec(
        engine=engine,
        sampler=sampler,
        pruner=pruner,
        budget=budget,
        trial_naming=trial_naming,
        extra=extra,
    )
