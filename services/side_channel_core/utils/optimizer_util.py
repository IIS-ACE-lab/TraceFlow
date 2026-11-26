#services/side_channel_core/utils/optimizer_util.py
"""
YAML-driven PyTorch optimizer construction.

Canonical configuration
-----------------------
trainer:
  optimizer:
    type: RMSprop
    learning_rate: 1.0e-5
    weight_decay: 0.0
    args:
      alpha: 0.9
      eps: 1.0e-7
      momentum: 0.0
      centered: false
    scheduler:              # optional; omitted or type none disables it
      type: StepLR
      args:
        step_size: 15
        gamma: 0.5

Supported optimizers: Adadelta, Adafactor, Adagrad, Adam, AdamW, Adamax,
ASGD, LBFGS, NAdam, RAdam, RMSprop, Rprop, SGD, and SparseAdam.

Adafactor requires availability in the installed PyTorch version.
LBFGS requires a closure-enabled training loop.
SparseAdam requires sparse gradients.
Optimizer names are case-insensitive.
Responsibilities
----------------
- Define every optimizer type TraceFlow advertises.
- Bind every advertised name directly to its ``torch.optim`` class.
- Report an unavailable optional optimizer when it is selected.
- Validate the shared optimizer hierarchy and optimizer-specific arguments.
- Reject obsolete flat trainer fields and unresolved multi-value sweeps.
- Reject settings unsupported by the installed torch version.
- Build the optional learning-rate scheduler nested under the optimizer.
- Return serializable resolved metadata for provenance and tracking.

This module configures PyTorch optimizers; it does not reimplement their
optimization algorithms.

Author: Shraddha Mishra
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch


Validator = Callable[[Any, str], Any]


@dataclass(frozen=True)
class OptimizerSpec:
    """TraceFlow contract for one torch optimizer."""

    optimizer_class: type[torch.optim.Optimizer] | None
    arg_validators: Mapping[str, Validator]
    requires_closure: bool = False


def _resolved_scalar(value: Any, field: str) -> Any:
    """Return one value and reject an unresolved sweep dimension."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                f"{field} must be resolved to one value before optimizer "
                f"construction; got {value!r}"
            )
        return value[0]
    return value


def _finite_float(value: Any, field: str) -> float:
    value = _resolved_scalar(value, field)
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {result!r}")
    return result


def _positive_float(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be > 0, got {result}")
    return result


def _nonnegative_float(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be >= 0, got {result}")
    return result


def _left_closed_unit_float(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if not 0.0 <= result < 1.0:
        raise ValueError(f"{field} must be in [0, 1), got {result}")
    return result


def _nonpositive_float(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if result > 0.0:
        raise ValueError(f"{field} must be <= 0, got {result}")
    return result


def _positive_int(value: Any, field: str) -> int:
    value = _resolved_scalar(value, field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{field} must be > 0, got {value}")
    return value


def _optional_positive_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field)


def _bool(value: Any, field: str) -> bool:
    value = _resolved_scalar(value, field)
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be true or false, got {value!r}")
    return value


def _optional_bool(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    return _bool(value, field)


def _betas(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{field} must contain exactly two beta values")
    return (
        _left_closed_unit_float(value[0], f"{field}[0]"),
        _left_closed_unit_float(value[1], f"{field}[1]"),
    )


def _rprop_etas(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{field} must contain exactly two values")
    decrease = _positive_float(value[0], f"{field}[0]")
    increase = _positive_float(value[1], f"{field}[1]")
    if decrease >= 1.0:
        raise ValueError(f"{field}[0] must be < 1, got {decrease}")
    if increase <= 1.0:
        raise ValueError(f"{field}[1] must be > 1, got {increase}")
    return decrease, increase


def _ordered_positive_pair(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{field} must contain exactly two values")
    lower = _positive_float(value[0], f"{field}[0]")
    upper = _positive_float(value[1], f"{field}[1]")
    if lower > upper:
        raise ValueError(f"{field}[0] must be <= {field}[1]")
    return lower, upper


def _adafactor_eps(value: Any, field: str) -> tuple[float | None, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{field} must contain exactly two values")
    first = None if value[0] is None else _nonnegative_float(value[0], f"{field}[0]")
    second = _positive_float(value[1], f"{field}[1]")
    return first, second


def _line_search(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if value != "strong_wolfe":
        raise ValueError(f"{field} must be null or 'strong_wolfe', got {value!r}")
    return value


_COMMON_EXECUTION_ARGS: dict[str, Validator] = {
    "foreach": _optional_bool,
    "maximize": _bool,
    "capturable": _bool,
    "differentiable": _bool,
    "fused": _optional_bool,
}


def _args(**specific: Validator) -> Mapping[str, Validator]:
    """Combine optimizer-specific validators with common torch flags."""
    return {**specific, **_COMMON_EXECUTION_ARGS}


# Explicit contract for every optimizer type advertised by TraceFlow.
OPTIMIZER_SPECS: dict[str, OptimizerSpec] = {
    "Adadelta": OptimizerSpec(
        optimizer_class=torch.optim.Adadelta,
        arg_validators=_args(rho=_left_closed_unit_float, eps=_positive_float),
    ),
    "Adafactor": OptimizerSpec(
        optimizer_class=(torch.optim.Adafactor if hasattr(torch.optim, "Adafactor") else None),
        arg_validators=_args(
            beta2_decay=_nonpositive_float,
            eps=_adafactor_eps,
            d=_positive_float,
        ),
    ),
    "Adagrad": OptimizerSpec(
        optimizer_class=torch.optim.Adagrad,
        arg_validators=_args(
            lr_decay=_nonnegative_float,
            initial_accumulator_value=_nonnegative_float,
            eps=_positive_float,
        ),
    ),
    "Adam": OptimizerSpec(
        optimizer_class=torch.optim.Adam,
        arg_validators=_args(
            betas=_betas,
            eps=_positive_float,
            amsgrad=_bool,
            decoupled_weight_decay=_bool,
        ),
    ),
    "AdamW": OptimizerSpec(
        optimizer_class=torch.optim.AdamW,
        arg_validators=_args(betas=_betas, eps=_positive_float, amsgrad=_bool),
    ),
    "Adamax": OptimizerSpec(
        optimizer_class=torch.optim.Adamax,
        arg_validators=_args(betas=_betas, eps=_positive_float),
    ),
    "ASGD": OptimizerSpec(
        optimizer_class=torch.optim.ASGD,
        arg_validators=_args(
            lambd=_nonnegative_float,
            alpha=_nonnegative_float,
            t0=_nonnegative_float,
        ),
    ),
    "LBFGS": OptimizerSpec(
        optimizer_class=torch.optim.LBFGS,
        arg_validators={
            "max_iter": _positive_int,
            "max_eval": _optional_positive_int,
            "tolerance_grad": _nonnegative_float,
            "tolerance_change": _nonnegative_float,
            "history_size": _positive_int,
            "line_search_fn": _line_search,
        },
        requires_closure=True,
    ),
    "NAdam": OptimizerSpec(
        optimizer_class=torch.optim.NAdam,
        arg_validators=_args(
            betas=_betas,
            eps=_positive_float,
            momentum_decay=_nonnegative_float,
            decoupled_weight_decay=_bool,
        ),
    ),
    "RAdam": OptimizerSpec(
        optimizer_class=torch.optim.RAdam,
        arg_validators=_args(
            betas=_betas,
            eps=_positive_float,
            decoupled_weight_decay=_bool,
        ),
    ),
    "RMSprop": OptimizerSpec(
        optimizer_class=torch.optim.RMSprop,
        arg_validators=_args(
            alpha=_left_closed_unit_float,
            eps=_positive_float,
            momentum=_nonnegative_float,
            centered=_bool,
        ),
    ),
    "Rprop": OptimizerSpec(
        optimizer_class=torch.optim.Rprop,
        arg_validators=_args(etas=_rprop_etas, step_sizes=_ordered_positive_pair),
    ),
    "SGD": OptimizerSpec(
        optimizer_class=torch.optim.SGD,
        arg_validators=_args(
            momentum=_nonnegative_float,
            dampening=_nonnegative_float,
            nesterov=_bool,
        ),
    ),
    "SparseAdam": OptimizerSpec(
        optimizer_class=torch.optim.SparseAdam,
        arg_validators=_args(betas=_betas, eps=_positive_float),
    ),
}

OPTIMIZER_REGISTRY: dict[str, type[torch.optim.Optimizer]] = {
    name: spec.optimizer_class
    for name, spec in OPTIMIZER_SPECS.items()
    if spec.optimizer_class is not None
}

_OPTIMIZER_NAME_LOOKUP = {name.casefold(): name for name in OPTIMIZER_REGISTRY}

_OPTIMIZER_FIELDS = {"type", "learning_rate", "weight_decay", "args"}
_OPTIONAL_OPTIMIZER_FIELDS = {"scheduler"}
_LEGACY_TRAINER_FIELDS = {"learning_rate", "weight_decay", "optimizer_args", "scheduler"}
_SCHEDULER_FIELDS = {"type", "args", "interval"}
_DISABLED_SCHEDULERS = {"", "none", "null", "off"}
# Schedulers defined per optimizer step (per batch). Everything else in
# torch.optim.lr_scheduler is defined per epoch; ReduceLROnPlateau is per
# epoch and additionally consumes the validation objective.
_STEP_LEVEL_SCHEDULERS = frozenset({"OneCycleLR", "CyclicLR"})
_SCHEDULER_INTERVALS = frozenset({"epoch", "step"})
# torch OneCycleLR's length: total_steps, or epochs and steps_per_epoch. Without any of them the
# training loop gives it its own length (build_loop_scheduler), as PyTorch's documentation does:
# OneCycleLR(optimizer, max_lr, steps_per_epoch=len(data_loader), epochs=...).
_ONE_CYCLE_LENGTH_ARGS = ("total_steps", "epochs", "steps_per_epoch")
_RESERVED_ARGS = {"params", "lr", "weight_decay"}


def _canonical_optimizer_name(value: Any) -> str:
    value = _resolved_scalar(value, "trainer.optimizer.type")
    if not isinstance(value, str) or not value.strip():
        raise TypeError("trainer.optimizer.type must be a non-empty string")

    canonical = _OPTIMIZER_NAME_LOOKUP.get(value.strip().casefold())
    if canonical is None:
        if any(name.casefold() == value.strip().casefold() for name in OPTIMIZER_SPECS):
            raise ValueError(f"Optimizer {value!r} is unavailable in this PyTorch build")
        raise ValueError(
            f"Unsupported optimizer {value!r}; available: "
            f"{sorted(OPTIMIZER_REGISTRY)}"
        )
    return canonical


def _normalize_specific_args(
    optimizer_name: str,
    raw_args: Mapping[str, Any],
    optimizer_cls: type[torch.optim.Optimizer],
) -> dict[str, Any]:
    """Validate arguments against both TraceFlow and installed torch contracts."""
    spec = OPTIMIZER_SPECS[optimizer_name]
    args = dict(raw_args)

    duplicates = sorted(_RESERVED_ARGS.intersection(args))
    if duplicates:
        raise ValueError(
            "trainer.optimizer.args must not redefine canonical fields: "
            f"{duplicates}"
        )

    unknown_contract_args = sorted(set(args) - set(spec.arg_validators))
    if unknown_contract_args:
        raise ValueError(
            f"Unsupported TraceFlow arguments for {optimizer_name}: "
            f"{unknown_contract_args}; allowed: {sorted(spec.arg_validators)}"
        )

    signature = inspect.signature(optimizer_cls.__init__)
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_var_kwargs:
        unavailable_args = sorted(set(args) - set(signature.parameters))
        if unavailable_args:
            raise ValueError(
                f"The installed torch version does not support these "
                f"{optimizer_name} arguments: {unavailable_args}"
            )

    normalized: dict[str, Any] = {}
    for name, value in args.items():
        field = f"trainer.optimizer.args.{name}"
        normalized[name] = spec.arg_validators[name](value, field)

    if optimizer_name == "SGD" and normalized.get("nesterov", False):
        if normalized.get("momentum", 0.0) <= 0.0:
            raise ValueError("SGD nesterov=true requires momentum > 0")
        if normalized.get("dampening", 0.0) != 0.0:
            raise ValueError("SGD nesterov=true requires dampening = 0")

    return normalized


def resolve_optimizer_config(trainer_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ``trainer.optimizer`` and return one canonical configuration."""
    if not isinstance(trainer_cfg, Mapping):
        raise TypeError("cfg['trainer'] must be a mapping")

    legacy = sorted(_LEGACY_TRAINER_FIELDS.intersection(trainer_cfg))
    if legacy:
        raise ValueError(
            "Legacy trainer-level optimizer fields are forbidden: "
            f"{legacy}. Move them under trainer.optimizer."
        )

    optimizer_cfg = trainer_cfg.get("optimizer")
    if not isinstance(optimizer_cfg, Mapping):
        raise TypeError(
            "cfg['trainer']['optimizer'] must be a mapping containing "
            "type, learning_rate, weight_decay, and args"
        )

    allowed_fields = _OPTIMIZER_FIELDS | _OPTIONAL_OPTIMIZER_FIELDS
    unknown_fields = sorted(set(optimizer_cfg) - allowed_fields)
    if unknown_fields:
        raise ValueError(
            f"Unknown trainer.optimizer fields: {unknown_fields}; "
            f"allowed: {sorted(allowed_fields)}"
        )

    missing = sorted(_OPTIMIZER_FIELDS - set(optimizer_cfg))
    if missing:
        raise KeyError(f"trainer.optimizer is missing required fields: {missing}")

    optimizer_name = _canonical_optimizer_name(optimizer_cfg["type"])
    optimizer_cls = OPTIMIZER_REGISTRY[optimizer_name]
    learning_rate = _positive_float(
        optimizer_cfg["learning_rate"],
        "trainer.optimizer.learning_rate",
    )
    weight_decay = _nonnegative_float(
        optimizer_cfg["weight_decay"],
        "trainer.optimizer.weight_decay",
    )

    raw_args = optimizer_cfg["args"]
    if not isinstance(raw_args, Mapping):
        raise TypeError("trainer.optimizer.args must be a mapping")
    args = _normalize_specific_args(optimizer_name, raw_args, optimizer_cls)

    signature = inspect.signature(optimizer_cls.__init__)
    supports_weight_decay = "weight_decay" in signature.parameters
    if weight_decay != 0.0 and not supports_weight_decay:
        raise ValueError(
            f"{optimizer_name} does not support weight_decay, but "
            f"trainer.optimizer.weight_decay={weight_decay}"
        )

    return {
        "type": optimizer_name,
        "class": optimizer_cls,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "supports_weight_decay": supports_weight_decay,
        "requires_closure": OPTIMIZER_SPECS[optimizer_name].requires_closure,
        "args": args,
        "scheduler": resolve_scheduler_config(optimizer_cfg.get("scheduler")),
    }


def resolve_scheduler_config(scheduler_cfg: Any) -> dict[str, Any] | None:
    """Validate ``trainer.optimizer.scheduler``; return None when disabled.

    The returned spec carries ``interval``: ``"step"`` means the training
    loop calls ``scheduler.step()`` after every ``optimizer.step()``
    (OneCycleLR, CyclicLR); ``"epoch"`` means once per epoch after the
    epoch's optimisation passes (all other schedulers; ReduceLROnPlateau
    receives the validation objective). ``interval`` may be written
    explicitly but must agree with the scheduler's definition.
    """
    if scheduler_cfg is None:
        return None
    if not isinstance(scheduler_cfg, Mapping):
        raise TypeError("trainer.optimizer.scheduler must be a mapping or null")

    unknown = sorted(set(scheduler_cfg) - _SCHEDULER_FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown trainer.optimizer.scheduler fields: {unknown}; "
            f"allowed: {sorted(_SCHEDULER_FIELDS)}"
        )

    scheduler_type = _resolved_scalar(
        scheduler_cfg.get("type", "none"),
        "trainer.optimizer.scheduler.type",
    )
    if scheduler_type is None:
        return None
    if not isinstance(scheduler_type, str):
        raise TypeError("trainer.optimizer.scheduler.type must be a string")
    scheduler_type = scheduler_type.strip()
    if scheduler_type.casefold() in _DISABLED_SCHEDULERS:
        return None

    scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_type, None)
    if not (
        inspect.isclass(scheduler_cls)
        and scheduler_type[0].isupper()
        and hasattr(scheduler_cls, "step")
    ):
        raise ValueError(
            f"Unsupported torch learning-rate scheduler: {scheduler_type!r}"
        )

    args = scheduler_cfg.get("args") or {}
    if not isinstance(args, Mapping):
        raise TypeError("trainer.optimizer.scheduler.args must be a mapping")
    if "optimizer" in args:
        raise ValueError("trainer.optimizer.scheduler.args must not define optimizer")

    natural_interval = "step" if scheduler_type in _STEP_LEVEL_SCHEDULERS else "epoch"
    interval_raw = _resolved_scalar(
        scheduler_cfg.get("interval"), "trainer.optimizer.scheduler.interval"
    )
    if interval_raw is None:
        interval = natural_interval
    else:
        if not isinstance(interval_raw, str):
            raise TypeError("trainer.optimizer.scheduler.interval must be a string")
        interval = interval_raw.strip().lower()
        if interval not in _SCHEDULER_INTERVALS:
            raise ValueError(
                "trainer.optimizer.scheduler.interval must be 'epoch' or 'step', "
                f"got {interval_raw!r}"
            )
        if interval != natural_interval:
            raise ValueError(
                f"{scheduler_type} is defined per {natural_interval}; "
                f"trainer.optimizer.scheduler.interval={interval!r} would "
                "mis-step it. Remove the field or set it to "
                f"{natural_interval!r}"
            )

    return {
        "type": scheduler_type,
        "class": scheduler_cls,
        "args": dict(args),
        "interval": interval,
    }


def resolve_scheduler_interval(
    trainer_cfg: Mapping[str, Any],
    scheduler: Any | None,
) -> str | None:
    """Return ``"step"`` / ``"epoch"`` for the scheduler the loop received."""
    if scheduler is None:
        return None
    spec = resolve_optimizer_config(trainer_cfg)["scheduler"]
    if spec is None:
        raise ValueError(
            "A scheduler instance was supplied but trainer.optimizer.scheduler "
            "resolves to none"
        )
    if type(scheduler).__name__ != spec["type"]:
        raise ValueError(
            f"Scheduler instance {type(scheduler).__name__} does not match "
            f"trainer.optimizer.scheduler.type={spec['type']!r}"
        )
    return str(spec["interval"])


def validate_scheduler_for_loop(
    scheduler: Any | None,
    interval: str | None,
    *,
    epochs: int,
    steps_per_epoch: int,
) -> None:
    """Reject a per-step scheduler whose horizon disagrees with the loop.

    OneCycleLR fixes its schedule to ``total_steps``; when that is not
    ``epochs * steps_per_epoch`` the learning rate either never completes the
    cycle or the scheduler raises mid-training. ``steps_per_epoch`` must be
    ``len(train_loader)`` (which already accounts for ``drop_last``).
    """
    if scheduler is None:
        return
    if interval not in _SCHEDULER_INTERVALS:
        raise ValueError(f"scheduler interval must be 'epoch' or 'step', got {interval!r}")
    epochs = int(epochs)
    steps_per_epoch = int(steps_per_epoch)
    if epochs < 1 or steps_per_epoch < 1:
        raise ValueError("epochs and steps_per_epoch must be >= 1")
    if interval == "step" and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        expected = epochs * steps_per_epoch
        actual = int(getattr(scheduler, "total_steps"))
        if actual != expected:
            raise ValueError(
                f"OneCycleLR.total_steps={actual} but the loop performs "
                f"{expected} optimizer steps ({epochs} epochs x {steps_per_epoch} "
                "batches). Set args.total_steps to that value, or args.epochs "
                "and args.steps_per_epoch to match trainer.epochs and the "
                "training loader length"
            )


def build_optimizer_from_config(
    trainer_cfg: Mapping[str, Any],
    parameters: Iterable[torch.nn.Parameter],
    *,
    allow_closure_optimizer: bool = False,
) -> tuple[torch.optim.Optimizer, Any | None, dict[str, Any]]:
    """
    Construct the configured optimizer and its optional nested scheduler.

    Returns ``(optimizer, scheduler_or_None, serializable_metadata)``. This is
    the only place TraceFlow builds either object.
    Supported optimizers: Adadelta, Adafactor, Adagrad, Adam, AdamW, Adamax,
    ASGD, LBFGS, NAdam, RAdam, RMSprop, Rprop, SGD, and SparseAdam.

    Adafactor requires availability in the installed PyTorch version.
    LBFGS requires a closure-enabled training loop.
    SparseAdam requires sparse gradients.
    Optimizer names are case-insensitive.
    """
    resolved = resolve_optimizer_config(trainer_cfg)

    if resolved["requires_closure"] and not allow_closure_optimizer:
        raise ValueError(
            f"{resolved['type']} requires optimizer.step(closure). The active "
            "training loop has not declared closure support."
        )

    constructor_kwargs: dict[str, Any] = {
        "lr": resolved["learning_rate"],
        **resolved["args"],
    }
    if resolved["supports_weight_decay"]:
        constructor_kwargs["weight_decay"] = resolved["weight_decay"]

    optimizer_cls = resolved["class"]
    optimizer = optimizer_cls(parameters, **constructor_kwargs)

    scheduler_spec = resolved["scheduler"]
    # A OneCycleLR without a length is built by the training loop, which knows it
    # (build_loop_scheduler); every other scheduler is built here, as before.
    scheduler = (
        None
        if scheduler_spec is None or takes_loop_length(scheduler_spec)
        else scheduler_spec["class"](optimizer, **scheduler_spec["args"])
    )

    metadata = {
        "type": resolved["type"],
        "learning_rate": resolved["learning_rate"],
        "weight_decay": resolved["weight_decay"],
        "requires_closure": resolved["requires_closure"],
        "args": dict(resolved["args"]),
        "scheduler": (
            None
            if scheduler_spec is None
            else {
                "type": scheduler_spec["type"],
                "args": dict(scheduler_spec["args"]),
                "interval": scheduler_spec["interval"],
                **({"length": "trainer.epochs x len(train_loader) of each training loop"}
                   if takes_loop_length(scheduler_spec) else {}),
            }
        ),
    }
    return optimizer, scheduler, metadata


def takes_loop_length(scheduler_spec: Mapping[str, Any] | None) -> bool:
    """True for a OneCycleLR whose args give no length: the training loop sets it."""
    return (
        scheduler_spec is not None
        and scheduler_spec["type"] == "OneCycleLR"
        and not any(key in scheduler_spec["args"] for key in _ONE_CYCLE_LENGTH_ARGS)
    )


def build_loop_scheduler(
    optimizer: torch.optim.Optimizer,
    trainer_cfg: Mapping[str, Any],
    *,
    epochs: int,
    steps_per_epoch: int,
) -> Any | None:
    """The OneCycleLR of one training loop (a fold, the refit, a hold-out training), or None.

    Built with torch's own epochs and steps_per_epoch: epochs is the loop's trainer.epochs,
    steps_per_epoch is len(train_loader) (one optimizer step per batch, drop_last counted), so
    every loop completes its cycle. None when the configured scheduler is anything else; those
    are built with the optimizer (build_optimizer_from_config).
    """
    spec = resolve_optimizer_config(trainer_cfg)["scheduler"]
    if not takes_loop_length(spec):
        return None
    epochs, steps_per_epoch = int(epochs), int(steps_per_epoch)
    if epochs < 1 or steps_per_epoch < 1:
        raise ValueError("OneCycleLR needs epochs >= 1 and a training loader with at least one batch")
    return spec["class"](optimizer, **spec["args"], epochs=epochs, steps_per_epoch=steps_per_epoch)


def check_scheduler_training(trainer_cfg: Mapping[str, Any]) -> None:
    """Refuse early stopping with OneCycleLR: stopping before the last epoch cuts the cycle, and the
    learning rate never reaches its annealed floor. Holds in every training mode."""
    spec = resolve_optimizer_config(trainer_cfg)["scheduler"]
    if spec is not None and spec["type"] == "OneCycleLR" and trainer_cfg.get("early_stopping", False) is not False:
        raise ValueError(
            "trainer.early_stopping cannot be used with OneCycleLR: a run stopped before its last epoch "
            "leaves the cycle unfinished, its learning rate never annealed. Hint: set "
            "trainer.early_stopping: false (the cycle runs over trainer.epochs), or use another scheduler"
        )
