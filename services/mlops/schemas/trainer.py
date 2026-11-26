# services/mlops/schemas/trainer.py
# ============================================================
# Trainer schema
# - Validates hypothesis.trainer blocks inside auto_mlops.yaml
# - Handles both fixed scalars and search-space declarations
# - Keeps validation strict on types/shape, but does NOT assume
#   downstream trainer implementation details.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from services.mlops.schemas.common import (
    forbid_unknown_keys,
    is_number,
    normalize_bool,
    normalize_dict,
    require_keys,
)


# ============================================================
# Search-space primitives
# ============================================================

Scalar = Union[str, int, float, bool]
ScalarSpace = Union[Scalar, List[Scalar]]

Number = Union[int, float]
NumberSpace = Union[Number, List[Number]]


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool))


def _as_scalar_space(v: Any, *, ctx: str) -> ScalarSpace:
    """
    Accept scalar or list of scalars.
    """
    if _is_scalar(v):
        return v  # type: ignore[return-value]
    if isinstance(v, list):
        if not v:
            raise ValueError(f"[schemas/trainer] {ctx} cannot be an empty list")
        for i, x in enumerate(v):
            if not _is_scalar(x):
                raise ValueError(f"[schemas/trainer] {ctx}[{i}] must be scalar (got {type(x)})")
        return v  # type: ignore[return-value]
    raise ValueError(f"[schemas/trainer] {ctx} must be scalar or list of scalars (got {type(v)})")


def _as_number_space(v: Any, *, ctx: str, allow_int: bool = True, allow_float: bool = True) -> NumberSpace:
    """
    Accept number or list of numbers. Enforces numeric types.
    """
    def _ok(x: Any) -> bool:
        if isinstance(x, bool):
            return False
        if isinstance(x, int) and allow_int:
            return True
        if isinstance(x, float) and allow_float:
            return True
        return False

    if _ok(v):
        return v  # type: ignore[return-value]
    if isinstance(v, list):
        if not v:
            raise ValueError(f"[schemas/trainer] {ctx} cannot be an empty list")
        for i, x in enumerate(v):
            if not _ok(x):
                raise ValueError(f"[schemas/trainer] {ctx}[{i}] must be number (got {type(x)})")
        return v  # type: ignore[return-value]
    raise ValueError(f"[schemas/trainer] {ctx} must be number or list of numbers (got {type(v)})")


def _as_int_space(v: Any, *, ctx: str) -> Union[int, List[int]]:
    if isinstance(v, bool):
        raise ValueError(f"[schemas/trainer] {ctx} must be int, not bool")
    if isinstance(v, int):
        return v
    if isinstance(v, list):
        if not v:
            raise ValueError(f"[schemas/trainer] {ctx} cannot be an empty list")
        out: List[int] = []
        for i, x in enumerate(v):
            if isinstance(x, bool) or not isinstance(x, int):
                raise ValueError(f"[schemas/trainer] {ctx}[{i}] must be int (got {type(x)})")
            out.append(x)
        return out
    raise ValueError(f"[schemas/trainer] {ctx} must be int or list[int] (got {type(v)})")


def _validate_dict_of_dicts(v: Any, *, ctx: str) -> Dict[str, Dict[str, Any]]:
    """
    Used for scheduler.args.{SchedulerName: {...}} style.
    """
    d = normalize_dict(v)
    out: Dict[str, Dict[str, Any]] = {}
    for k, sub in d.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"[schemas/trainer] {ctx} keys must be non-empty strings")
        out[k] = normalize_dict(sub)
    return out


def _as_optional_bool(v: Any, *, ctx: str) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    # accept 0/1 and "true"/"false" if you want; currently strict:
    raise ValueError(f"[schemas/trainer] {ctx} must be bool (got {type(v)})")

# ============================================================
# Data model
# ============================================================

@dataclass(frozen=True)
class OptimizerSpec:
    type: ScalarSpace
    lr: Optional[NumberSpace] = None
    weight_decay: Optional[NumberSpace] = None
    betas: Optional[List[List[Number]]] = None
    eps: Optional[NumberSpace] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SchedulerSpec:
    type: ScalarSpace
    args: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GradientSpec:
    clip_norm: Optional[NumberSpace] = None
    accumulation_steps: Optional[Union[int, List[int]]] = None
    mixed_precision: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainerSpec:
    """
    Canonical schema for hypothesis.trainer in auto_mlops.yaml.
    """
    monitor_metric: Optional[str] = None
    epochs: Optional[int] = None
    batch_size: Optional[Union[int, List[int]]] = None

    early_stopping: Optional[bool] = None
    patience: Optional[Union[int, List[int]]] = None
    min_delta: Optional[NumberSpace] = None

    loss_reduction: Optional[str] = None

    optimizer: Optional[OptimizerSpec] = None
    scheduler: Optional[SchedulerSpec] = None
    gradient: Optional[GradientSpec] = None

    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Validators
# ============================================================

def _validate_betas(v: Any, *, ctx: str) -> List[List[Number]]:
    """
    betas: [ [0.9, 0.999], [0.9, 0.98] ] style.
    """
    if not isinstance(v, list) or not v:
        raise ValueError(f"[schemas/trainer] {ctx} must be a non-empty list of [beta1, beta2] pairs")
    out: List[List[Number]] = []
    for i, pair in enumerate(v):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"[schemas/trainer] {ctx}[{i}] must be a list of length 2")
        b1, b2 = pair[0], pair[1]
        if not is_number(b1) or not is_number(b2):
            raise ValueError(f"[schemas/trainer] {ctx}[{i}] values must be numbers")
        out.append([float(b1), float(b2)])
    return out


def validate_optimizer_block(opt_block: Any, *, ctx: str = "trainer.optimizer", strict: bool = False) -> OptimizerSpec:
    d = normalize_dict(opt_block)

    require_keys(d, ["type"], ctx=ctx)

    allowed = {
        "type", "lr", "weight_decay", "betas", "eps",
        # allow any extra optimizer knobs (momentum, amsgrad, etc.)
    }
    if strict:
        forbid_unknown_keys(d, allowed, ctx=ctx)

    opt_type = _as_scalar_space(d["type"], ctx=f"{ctx}.type")

    lr = d.get("lr", None)
    wd = d.get("weight_decay", None)
    betas = d.get("betas", None)
    eps = d.get("eps", None)

    lr_space = _as_number_space(lr, ctx=f"{ctx}.lr") if lr is not None else None
    wd_space = _as_number_space(wd, ctx=f"{ctx}.weight_decay") if wd is not None else None
    betas_list = _validate_betas(betas, ctx=f"{ctx}.betas") if betas is not None else None
    eps_space = _as_number_space(eps, ctx=f"{ctx}.eps") if eps is not None else None

    extra = {k: v for k, v in d.items() if k not in {"type", "lr", "weight_decay", "betas", "eps"}}

    return OptimizerSpec(
        type=opt_type,
        lr=lr_space,
        weight_decay=wd_space,
        betas=betas_list,
        eps=eps_space,
        extra=extra,
    )


def validate_scheduler_block(sch_block: Any, *, ctx: str = "trainer.scheduler", strict: bool = False) -> SchedulerSpec:
    d = normalize_dict(sch_block)
    require_keys(d, ["type"], ctx=ctx)

    allowed = {"type", "args"}
    if strict:
        forbid_unknown_keys(d, allowed, ctx=ctx)

    sch_type = _as_scalar_space(d["type"], ctx=f"{ctx}.type")

    args_raw = d.get("args", {}) or {}
    args = _validate_dict_of_dicts(args_raw, ctx=f"{ctx}.args") if args_raw is not None else {}

    extra = {k: v for k, v in d.items() if k not in {"type", "args"}}
    return SchedulerSpec(type=sch_type, args=args, extra=extra)


def validate_gradient_block(grad_block: Any, *, ctx: str = "trainer.gradient", strict: bool = False) -> GradientSpec:
    d = normalize_dict(grad_block)

    allowed = {"clip_norm", "accumulation_steps", "mixed_precision"}
    if strict:
        forbid_unknown_keys(d, allowed, ctx=ctx)

    clip_norm = d.get("clip_norm", None)
    accumulation_steps = d.get("accumulation_steps", None)
    mixed_precision = d.get("mixed_precision", None)

    clip_space = _as_number_space(clip_norm, ctx=f"{ctx}.clip_norm") if clip_norm is not None else None
    acc_space = _as_int_space(accumulation_steps, ctx=f"{ctx}.accumulation_steps") if accumulation_steps is not None else None
    mp = normalize_bool(mixed_precision, default=None) if mixed_precision is not None else None

    extra = {k: v for k, v in d.items() if k not in {"clip_norm", "accumulation_steps", "mixed_precision"}}
    return GradientSpec(clip_norm=clip_space, accumulation_steps=acc_space, mixed_precision=mp, extra=extra)


def validate_trainer_block(
    trainer_block: Any,
    *,
    ctx: str = "hypothesis.trainer",
    strict: bool = False,
) -> TrainerSpec:
    d = normalize_dict(trainer_block)

    allowed = {
        "monitor_metric",
        "epochs",
        "batch_size",
        "loss_reduction",
        "early_stopping",
        "patience",
        "min_delta",
        "optimizer",
        "scheduler",
        "gradient",
    }
    if strict:
        forbid_unknown_keys(d, allowed, ctx=ctx)

    monitor_metric = d.get("monitor_metric")
    if monitor_metric is not None:
        if not isinstance(monitor_metric, str) or not monitor_metric.strip():
            raise ValueError(f"[schemas/trainer] {ctx}.monitor_metric must be non-empty string")
        monitor_metric = monitor_metric.strip()

    epochs = d.get("epochs")
    if epochs is not None:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(f"[schemas/trainer] {ctx}.epochs must be positive int")
        epochs = int(epochs)

    batch_size = _as_int_space(d["batch_size"], ctx=f"{ctx}.batch_size") if "batch_size" in d else None
    if isinstance(batch_size, int) and batch_size <= 0:
        raise ValueError(f"[schemas/trainer] {ctx}.batch_size must be > 0")
    if isinstance(batch_size, list) and any(x <= 0 for x in batch_size):
        raise ValueError(f"[schemas/trainer] {ctx}.batch_size list entries must be > 0")

    loss_reduction = d.get("loss_reduction")
    if loss_reduction is not None:
        if not isinstance(loss_reduction, str) or not loss_reduction.strip():
            raise ValueError(f"[schemas/trainer] {ctx}.loss_reduction must be non-empty string")
        loss_reduction = loss_reduction.strip()

    early_stopping = _as_optional_bool(d.get("early_stopping", None), ctx=f"{ctx}.early_stopping")

    patience = _as_int_space(d["patience"], ctx=f"{ctx}.patience") if "patience" in d else None
    if isinstance(patience, int) and patience < 0:
        raise ValueError(f"[schemas/trainer] {ctx}.patience must be >= 0")
    if isinstance(patience, list) and any(x < 0 for x in patience):
        raise ValueError(f"[schemas/trainer] {ctx}.patience list entries must be >= 0")

    min_delta = _as_number_space(d["min_delta"], ctx=f"{ctx}.min_delta") if "min_delta" in d else None

    optimizer = validate_optimizer_block(d["optimizer"], ctx=f"{ctx}.optimizer", strict=strict) if "optimizer" in d else None
    scheduler = validate_scheduler_block(d["scheduler"], ctx=f"{ctx}.scheduler", strict=strict) if "scheduler" in d else None
    gradient = validate_gradient_block(d["gradient"], ctx=f"{ctx}.gradient", strict=strict) if "gradient" in d else None

    extra = {k: v for k, v in d.items() if k not in allowed}

    return TrainerSpec(
        monitor_metric=monitor_metric,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        loss_reduction=loss_reduction,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient=gradient,
        extra=extra,
    )
