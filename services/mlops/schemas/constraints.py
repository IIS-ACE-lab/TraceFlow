# services/mlops/schemas/constraints.py
# ============================================================
# Constraints schema (future-proof): hard/soft/budget/resources/stages
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from services.mlops.schemas.common import (
    normalize_int,
    normalize_float,
    normalize_bool,
    forbid_unknown_keys,
)

# ------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------

@dataclass(frozen=True)
class HardTrainingConstraints:
    max_epochs: Optional[int] = None
    max_params: Optional[int] = None
    disallow_nan: bool = False
    min_accuracy_floor: Optional[float] = None


@dataclass(frozen=True)
class HardSystemConstraints:
    max_latency_ms: Optional[float] = None
    max_memory_mb: Optional[int] = None
    max_disk_mb: Optional[int] = None


@dataclass(frozen=True)
class IntegrityConstraints:
    disallow_inf: bool = True
    require_finite_metrics: bool = True
    abort_on_metric_missing: bool = False


@dataclass(frozen=True)
class HardConstraints:
    training: HardTrainingConstraints
    system: HardSystemConstraints
    integrity: IntegrityConstraints


@dataclass(frozen=True)
class SoftTrainingConstraints:
    target_epochs: Optional[int] = None
    prefer_mixed_precision: bool = False


@dataclass(frozen=True)
class SoftSystemConstraints:
    target_latency_ms: Optional[float] = None
    target_memory_mb: Optional[int] = None


@dataclass(frozen=True)
class SoftPenaltySpec:
    enabled: bool = False
    weights: Optional[Dict[str, float]] = None  # latency_ms, memory_mb, params, etc.


@dataclass(frozen=True)
class SoftConstraints:
    training: Optional[SoftTrainingConstraints] = None
    system: Optional[SoftSystemConstraints] = None
    penalty: Optional[SoftPenaltySpec] = None


@dataclass(frozen=True)
class PerTrialBudget:
    max_wall_minutes: Optional[float] = None
    max_gpu_minutes: Optional[float] = None


@dataclass(frozen=True)
class BudgetSpec:
    max_trials: Optional[int] = None
    parallel_trials: Optional[int] = None
    max_wall_hours: Optional[float] = None
    max_gpu_hours: Optional[float] = None
    per_trial: Optional[PerTrialBudget] = None


@dataclass(frozen=True)
class GPUResourcePolicy:
    allow_multi_gpu: bool = False
    max_gpus: Optional[int] = None
    min_vram_mb: Optional[int] = None


@dataclass(frozen=True)
class CPUResourcePolicy:
    max_workers: Optional[int] = None


@dataclass(frozen=True)
class MemoryResourcePolicy:
    max_ram_mb: Optional[int] = None


@dataclass(frozen=True)
class ResourcePolicy:
    gpu: Optional[GPUResourcePolicy] = None
    cpu: Optional[CPUResourcePolicy] = None
    memory: Optional[MemoryResourcePolicy] = None


@dataclass(frozen=True)
class StageOverride:
    hard: Optional[HardConstraints] = None
    soft: Optional[SoftConstraints] = None
    budget: Optional[BudgetSpec] = None
    resources: Optional[ResourcePolicy] = None


@dataclass(frozen=True)
class ConstraintsSpec:
    hard: Optional[HardConstraints] = None
    soft: Optional[SoftConstraints] = None
    budget: Optional[BudgetSpec] = None
    resources: Optional[ResourcePolicy] = None
    stages: Optional[Dict[str, StageOverride]] = None


# ------------------------------------------------------------
# Internal validators
# ------------------------------------------------------------

def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _validate_hard_training(cfg: Dict[str, Any], *, ctx: str) -> HardTrainingConstraints:
    forbid_unknown_keys(
        cfg,
        allowed=["max_epochs", "max_params", "disallow_nan", "min_accuracy_floor"],
        ctx=ctx,
        strict=True,
    )
    return HardTrainingConstraints(
        max_epochs=normalize_int(cfg.get("max_epochs")),
        max_params=normalize_int(cfg.get("max_params")),
        disallow_nan=normalize_bool(cfg.get("disallow_nan"), default=False),
        min_accuracy_floor=normalize_float(cfg.get("min_accuracy_floor")),
    )


def _validate_hard_system(cfg: Dict[str, Any], *, ctx: str) -> HardSystemConstraints:
    forbid_unknown_keys(
        cfg,
        allowed=["max_latency_ms", "max_memory_mb", "max_disk_mb"],
        ctx=ctx,
        strict=True,
    )
    return HardSystemConstraints(
        max_latency_ms=normalize_float(cfg.get("max_latency_ms")),
        max_memory_mb=normalize_int(cfg.get("max_memory_mb")),
        max_disk_mb=normalize_int(cfg.get("max_disk_mb")),
    )


def _validate_integrity(cfg: Dict[str, Any], *, ctx: str) -> IntegrityConstraints:
    forbid_unknown_keys(
        cfg,
        allowed=["disallow_inf", "require_finite_metrics", "abort_on_metric_missing"],
        ctx=ctx,
        strict=True,
    )
    return IntegrityConstraints(
        disallow_inf=normalize_bool(cfg.get("disallow_inf"), default=True),
        require_finite_metrics=normalize_bool(cfg.get("require_finite_metrics"), default=True),
        abort_on_metric_missing=normalize_bool(cfg.get("abort_on_metric_missing"), default=False),
    )


def _validate_hard(cfg: Dict[str, Any], *, ctx: str) -> HardConstraints:
    forbid_unknown_keys(cfg, allowed=["training", "system", "integrity"], ctx=ctx, strict=True)
    training = _validate_hard_training(_as_dict(cfg.get("training")), ctx=f"{ctx}.training")
    system = _validate_hard_system(_as_dict(cfg.get("system")), ctx=f"{ctx}.system")
    integrity = _validate_integrity(_as_dict(cfg.get("integrity")), ctx=f"{ctx}.integrity")
    return HardConstraints(training=training, system=system, integrity=integrity)


def _validate_soft_training(cfg: Dict[str, Any], *, ctx: str) -> SoftTrainingConstraints:
    forbid_unknown_keys(cfg, allowed=["target_epochs", "prefer_mixed_precision"], ctx=ctx, strict=True)
    return SoftTrainingConstraints(
        target_epochs=normalize_int(cfg.get("target_epochs")),
        prefer_mixed_precision=normalize_bool(cfg.get("prefer_mixed_precision"), default=False),
    )


def _validate_soft_system(cfg: Dict[str, Any], *, ctx: str) -> SoftSystemConstraints:
    forbid_unknown_keys(cfg, allowed=["target_latency_ms", "target_memory_mb"], ctx=ctx, strict=True)
    return SoftSystemConstraints(
        target_latency_ms=normalize_float(cfg.get("target_latency_ms")),
        target_memory_mb=normalize_int(cfg.get("target_memory_mb")),
    )


def _validate_soft_penalty(cfg: Dict[str, Any], *, ctx: str) -> SoftPenaltySpec:
    forbid_unknown_keys(cfg, allowed=["enabled", "weights"], ctx=ctx, strict=True)
    enabled = normalize_bool(cfg.get("enabled"), default=False)

    weights_raw = cfg.get("weights")
    weights: Optional[Dict[str, float]] = None
    if isinstance(weights_raw, dict):
        weights = {}
        for k, v in weights_raw.items():
            weights[str(k)] = float(v)
    elif weights_raw is None:
        weights = None
    else:
        raise ValueError(f"[{ctx}] penalty.weights must be a mapping if provided")

    return SoftPenaltySpec(enabled=enabled, weights=weights)


def _validate_soft(cfg: Dict[str, Any], *, ctx: str) -> SoftConstraints:
    forbid_unknown_keys(cfg, allowed=["training", "system", "penalty"], ctx=ctx, strict=True)

    training = None
    if cfg.get("training") is not None:
        training = _validate_soft_training(_as_dict(cfg.get("training")), ctx=f"{ctx}.training")

    system = None
    if cfg.get("system") is not None:
        system = _validate_soft_system(_as_dict(cfg.get("system")), ctx=f"{ctx}.system")

    penalty = None
    if cfg.get("penalty") is not None:
        penalty = _validate_soft_penalty(_as_dict(cfg.get("penalty")), ctx=f"{ctx}.penalty")

    return SoftConstraints(training=training, system=system, penalty=penalty)


def _validate_budget(cfg: Dict[str, Any], *, ctx: str) -> BudgetSpec:
    forbid_unknown_keys(
        cfg,
        allowed=["max_trials", "parallel_trials", "max_wall_hours", "max_gpu_hours", "per_trial"],
        ctx=ctx,
        strict=True,
    )

    per_trial = None
    if cfg.get("per_trial") is not None:
        pt = _as_dict(cfg.get("per_trial"))
        forbid_unknown_keys(pt, allowed=["max_wall_minutes", "max_gpu_minutes"], ctx=f"{ctx}.per_trial", strict=True)
        per_trial = PerTrialBudget(
            max_wall_minutes=normalize_float(pt.get("max_wall_minutes")),
            max_gpu_minutes=normalize_float(pt.get("max_gpu_minutes")),
        )

    return BudgetSpec(
        max_trials=normalize_int(cfg.get("max_trials")),
        parallel_trials=normalize_int(cfg.get("parallel_trials")),
        max_wall_hours=normalize_float(cfg.get("max_wall_hours")),
        max_gpu_hours=normalize_float(cfg.get("max_gpu_hours")),
        per_trial=per_trial,
    )


def _validate_resources(cfg: Dict[str, Any], *, ctx: str) -> ResourcePolicy:
    forbid_unknown_keys(cfg, allowed=["gpu", "cpu", "memory"], ctx=ctx, strict=True)

    gpu = None
    if cfg.get("gpu") is not None:
        g = _as_dict(cfg.get("gpu"))
        forbid_unknown_keys(g, allowed=["allow_multi_gpu", "max_gpus", "min_vram_mb"], ctx=f"{ctx}.gpu", strict=True)
        gpu = GPUResourcePolicy(
            allow_multi_gpu=normalize_bool(g.get("allow_multi_gpu"), default=False),
            max_gpus=normalize_int(g.get("max_gpus")),
            min_vram_mb=normalize_int(g.get("min_vram_mb")),
        )

    cpu = None
    if cfg.get("cpu") is not None:
        c = _as_dict(cfg.get("cpu"))
        forbid_unknown_keys(c, allowed=["max_workers"], ctx=f"{ctx}.cpu", strict=True)
        cpu = CPUResourcePolicy(max_workers=normalize_int(c.get("max_workers")))

    memory = None
    if cfg.get("memory") is not None:
        m = _as_dict(cfg.get("memory"))
        forbid_unknown_keys(m, allowed=["max_ram_mb"], ctx=f"{ctx}.memory", strict=True)
        memory = MemoryResourcePolicy(max_ram_mb=normalize_int(m.get("max_ram_mb")))

    return ResourcePolicy(gpu=gpu, cpu=cpu, memory=memory)


def _validate_stage_override(cfg: Dict[str, Any], *, ctx: str) -> StageOverride:
    forbid_unknown_keys(cfg, allowed=["hard", "soft", "budget", "resources"], ctx=ctx, strict=True)

    hard = _validate_hard(_as_dict(cfg.get("hard")), ctx=f"{ctx}.hard") if cfg.get("hard") is not None else None
    soft = _validate_soft(_as_dict(cfg.get("soft")), ctx=f"{ctx}.soft") if cfg.get("soft") is not None else None
    budget = _validate_budget(_as_dict(cfg.get("budget")), ctx=f"{ctx}.budget") if cfg.get("budget") is not None else None
    resources = _validate_resources(_as_dict(cfg.get("resources")), ctx=f"{ctx}.resources") if cfg.get("resources") is not None else None

    return StageOverride(hard=hard, soft=soft, budget=budget, resources=resources)


# ------------------------------------------------------------
# Public validator
# ------------------------------------------------------------

def validate_constraints_block(cfg: Dict[str, Any], *, ctx: str = "constraints") -> ConstraintsSpec:
    """
    Validate constraints section (future-proof).

    Accepts:

      constraints:
        hard:
          training: { max_epochs, max_params, disallow_nan, min_accuracy_floor }
          system: { max_latency_ms, max_memory_mb, max_disk_mb }
          integrity: { disallow_inf, require_finite_metrics, abort_on_metric_missing }
        soft:
          training: { target_epochs, prefer_mixed_precision }
          system: { target_latency_ms, target_memory_mb }
          penalty: { enabled, weights: { latency_ms, memory_mb, params, ... } }
        budget:
          max_trials, parallel_trials, max_wall_hours, max_gpu_hours,
          per_trial: { max_wall_minutes, max_gpu_minutes }
        resources:
          gpu: { allow_multi_gpu, max_gpus, min_vram_mb }
          cpu: { max_workers }
          memory: { max_ram_mb }
        stages:
          train|test|deploy|...:
            hard|soft|budget|resources: (same shapes as above)

    Notes:
    - Unknown keys are rejected (strict hygiene) so you don’t silently drift.
    - Each sub-block is optional, but at least one should exist for usefulness.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"[{ctx}] must be a mapping")

    forbid_unknown_keys(cfg, allowed=["hard", "soft", "budget", "resources", "stages"], ctx=ctx, strict=True)

    hard = _validate_hard(_as_dict(cfg.get("hard")), ctx=f"{ctx}.hard") if cfg.get("hard") is not None else None
    soft = _validate_soft(_as_dict(cfg.get("soft")), ctx=f"{ctx}.soft") if cfg.get("soft") is not None else None
    budget = _validate_budget(_as_dict(cfg.get("budget")), ctx=f"{ctx}.budget") if cfg.get("budget") is not None else None
    resources = _validate_resources(_as_dict(cfg.get("resources")), ctx=f"{ctx}.resources") if cfg.get("resources") is not None else None

    stages = None
    if cfg.get("stages") is not None:
        raw = _as_dict(cfg.get("stages"))
        stages = {}
        for stage_name, stage_cfg in raw.items():
            if not isinstance(stage_cfg, dict):
                raise ValueError(f"[{ctx}.stages] stage '{stage_name}' must be a mapping")
            stages[str(stage_name)] = _validate_stage_override(stage_cfg, ctx=f"{ctx}.stages.{stage_name}")

    if hard is None and soft is None and budget is None and resources is None and stages is None:
        raise ValueError(f"[{ctx}] at least one of hard/soft/budget/resources/stages must be provided")

    return ConstraintsSpec(hard=hard, soft=soft, budget=budget, resources=resources, stages=stages)
