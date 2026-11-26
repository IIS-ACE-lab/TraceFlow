# services/side_channel_core/utils/type_coerce.py
"""
Strict configuration type coercion for the Side-Channel Analysis system.

This module normalizes scalar runtime settings while preserving list/tuple
containers used as search spaces. It also validates the evaluation checkpoint
policy introduced for validation-selected model testing.

Canonical evaluation configuration
----------------------------------

evaluation:
  validation:
    checkpointing: true
  test:
    load_model: best_validation  # best_validation | final_epoch

Responsibilities
----------------
- Coerce trainer, model, test, data, and evaluation settings.
- Preserve search-space containers rather than collapsing them.
- Reject ambiguous boolean strings and invalid enum values.
- Coerce nested trainer.optimizer, trainer.loss, and trainer.evaluation settings.
- Reject obsolete flat optimizer/loss fields rather than silently migrating them.
- Keep checkpoint fallback out of this module. Runtime fallback belongs in
  ``sca/test.py``, where checkpoint availability can be verified.

The input configuration dictionary is modified in place and returned.

Author: Shraddha Mishra
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_TRUE_STRINGS = frozenset({"true", "1", "yes", "y", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "n", "off"})

_SUPPORTED_TEST_MODEL_POLICIES = frozenset(
    {
        "best_validation",
        "final_epoch",
    }
)

_SUPPORTED_TRACE_NORMALIZATION_MODES = frozenset(
    {
        "auto",
        "none",
        "zscore",
        "minmax",
        "per_trace_standard",
        "per_trace_minmax",
        "fitted",
    }
)

# trainer.loss.args.reduction (also the config panel's menu)
LOSS_REDUCTIONS = frozenset({"mean", "sum"})

_SINGLE_DATASET_KEYS = frozenset(
    {
        "name",
        "database",
        "profiling_range",
        "attack_range",
        "validation_range",
        "test_range",
        "target_range",
        "trace_normalization",
    }
)


# =============================================================================
# Scalar coercion helpers
# =============================================================================
def coerce_scalar(value: Any) -> Any:
    """
    Coerce one scalar-like value.

    Important
    ---------
    - Lists and tuples are intentionally not collapsed here.
    - Only numeric strings are converted.
    - Boolean strings are handled only by :func:`coerce_bool`, because
      interpreting arbitrary strings through ``bool(value)`` is unsafe.
    """
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    try:
        return int(text)
    except ValueError:
        try:
            result = float(text)
        except ValueError:
            return value

        return result if math.isfinite(result) else value


def coerce_bool(value: Any, *, field: str) -> bool:
    """Return a strict boolean value or raise a descriptive error."""
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise TypeError(f"{field} must be boolean; integer {value!r} is invalid")

    if isinstance(value, float):
        if value in (0.0, 1.0):
            return bool(int(value))
        raise TypeError(f"{field} must be boolean; float {value!r} is invalid")

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False

    allowed = sorted(_TRUE_STRINGS | _FALSE_STRINGS)
    raise TypeError(
        f"{field} must be a boolean or one of {allowed}, got {value!r}"
    )


def _coerce_int(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return a strict integer without silently truncating fractional values."""
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, got boolean {value!r}")

    normalized = coerce_scalar(value)

    if isinstance(normalized, float):
        if not math.isfinite(normalized) or not normalized.is_integer():
            raise TypeError(f"{field} must be an integer, got {value!r}")
        result = int(normalized)
    else:
        try:
            result = int(normalized)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{field} must be an integer, got {value!r}") from exc

    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be <= {maximum}, got {result}")

    return result


def _coerce_float(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    """Return one finite floating-point value with optional range checks."""
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric, got boolean {value!r}")

    normalized = coerce_scalar(value)
    try:
        result = float(normalized)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric, got {value!r}") from exc

    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {result!r}")

    if minimum is not None:
        if minimum_inclusive:
            invalid = result < minimum
            relation = ">="
        else:
            invalid = result <= minimum
            relation = ">"
        if invalid:
            raise ValueError(f"{field} must be {relation} {minimum}, got {result}")

    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be <= {maximum}, got {result}")

    return result


def _coerce_choice(value: Any, *, field: str, choices: frozenset[str]) -> str:
    """Normalize and validate one lower-case string enum."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string, got {type(value).__name__}")

    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(
            f"{field} must be one of {sorted(choices)}, got {value!r}"
        )
    return normalized


def _coerce_scalar_or_sequence(
    value: Any,
    *,
    field: str,
    coercer: Callable[..., T],
    **kwargs: Any,
) -> T | list[T] | tuple[T, ...]:
    """
    Coerce a scalar or each item in a list/tuple while preserving its container.

    This is required because configuration lists may represent search spaces.
    """
    if isinstance(value, list):
        return [
            coercer(item, field=f"{field}[{index}]", **kwargs)
            for index, item in enumerate(value)
        ]

    if isinstance(value, tuple):
        return tuple(
            coercer(item, field=f"{field}[{index}]", **kwargs)
            for index, item in enumerate(value)
        )

    return coercer(value, field=field, **kwargs)


def _coerce_dict_values_in_place(values: MutableMapping[str, Any]) -> None:
    """
    Recursively coerce numeric-string leaves while preserving containers.

    Boolean strings are intentionally left unchanged unless their owning field
    has an explicit boolean contract elsewhere in this module.
    """
    for key, value in list(values.items()):
        if isinstance(value, MutableMapping):
            _coerce_dict_values_in_place(value)
            continue

        if isinstance(value, list):
            normalized: list[Any] = []
            for item in value:
                if isinstance(item, MutableMapping):
                    _coerce_dict_values_in_place(item)
                    normalized.append(item)
                else:
                    normalized.append(coerce_scalar(item))
            values[key] = normalized
            continue

        if isinstance(value, tuple):
            normalized_items: list[Any] = []
            for item in value:
                if isinstance(item, MutableMapping):
                    _coerce_dict_values_in_place(item)
                    normalized_items.append(item)
                else:
                    normalized_items.append(coerce_scalar(item))
            values[key] = tuple(normalized_items)
            continue

        values[key] = coerce_scalar(value)


# =============================================================================
# Trainer coercion
# =============================================================================
def _coerce_optional_bool(value: Any, *, field: str) -> bool | None:
    """Preserve PyTorch's automatic foreach/fused selection represented by null."""
    return None if value is None else coerce_bool(value, field=field)


def coerce_trainer(trainer_cfg: MutableMapping[str, Any]) -> None:
    """Coerce and validate common trainer hyperparameters."""
    legacy = {"learning_rate", "lr", "weight_decay", "optimizer_args", "loss_reduction"}
    obsolete = sorted(legacy.intersection(trainer_cfg))
    if obsolete:
        raise ValueError(
            f"Obsolete trainer fields: {obsolete}. Use trainer.optimizer "
            "and trainer.loss.args.reduction."
        )
    integer_fields = {
        "batch_size": {"minimum": 1},
        "epochs": {"minimum": 1},
        "patience": {"minimum": 1},
    }

    for key, constraints in integer_fields.items():
        if key in trainer_cfg:
            trainer_cfg[key] = _coerce_scalar_or_sequence(
                trainer_cfg[key],
                field=f"trainer.{key}",
                coercer=_coerce_int,
                **constraints,
            )

    float_fields = {
        "min_delta": {"minimum": 0.0},
        "grad_clip_norm": {"minimum": 0.0, "minimum_inclusive": False},
    }

    for key, constraints in float_fields.items():
        if key in trainer_cfg and trainer_cfg[key] is not None:
            trainer_cfg[key] = _coerce_scalar_or_sequence(
                trainer_cfg[key],
                field=f"trainer.{key}",
                coercer=_coerce_float,
                **constraints,
            )

    if "precision" in trainer_cfg and trainer_cfg["precision"] is not None:
        from utils.precision import PRECISIONS

        raw = trainer_cfg["precision"]
        value = raw.strip().lower() if isinstance(raw, str) else None
        if value not in PRECISIONS:
            raise ValueError(f"trainer.precision must be one of {PRECISIONS}, got {raw!r}")
        trainer_cfg["precision"] = value

    for key in ("early_stopping", "checkpointing"):
        if key in trainer_cfg:
            trainer_cfg[key] = _coerce_scalar_or_sequence(
                trainer_cfg[key],
                field=f"trainer.{key}",
                coercer=coerce_bool,
            )

    optimizer = trainer_cfg.get("optimizer")
    if "optimizer" in trainer_cfg:
        if not isinstance(optimizer, MutableMapping):
            raise TypeError("trainer.optimizer must be a mapping")
        for key, inclusive in (("learning_rate", False), ("weight_decay", True)):
            if key in optimizer:
                optimizer[key] = _coerce_scalar_or_sequence(
                    optimizer[key], field=f"trainer.optimizer.{key}",
                    coercer=_coerce_float, minimum=0.0,
                    minimum_inclusive=inclusive,
                )
        if "args" in optimizer:
            args = optimizer["args"]
            if not isinstance(args, MutableMapping):
                raise TypeError("trainer.optimizer.args must be a mapping")
            _coerce_dict_values_in_place(args)
            for key in (
                "centered", "nesterov", "amsgrad", "maximize", "capturable",
                "differentiable", "decoupled_weight_decay", "foreach", "fused",
            ):
                if key in args:
                    args[key] = _coerce_scalar_or_sequence(
                        args[key], field=f"trainer.optimizer.args.{key}",
                        coercer=_coerce_optional_bool if key in {"foreach", "fused"} else coerce_bool,
                    )

    if "loss" in trainer_cfg:
        loss = trainer_cfg["loss"]
        if not isinstance(loss, MutableMapping):
            raise TypeError("trainer.loss must be a mapping")
        if "args" in loss:
            args = loss["args"]
            if not isinstance(args, MutableMapping):
                raise TypeError("trainer.loss.args must be a mapping")
            if "reduction" in args:
                args["reduction"] = _coerce_scalar_or_sequence(
                    args["reduction"], field="trainer.loss.args.reduction",
                    coercer=_coerce_choice, choices=LOSS_REDUCTIONS,
                )
            if "label_smoothing" in args:
                args["label_smoothing"] = _coerce_scalar_or_sequence(
                    args["label_smoothing"], field="trainer.loss.args.label_smoothing",
                    coercer=_coerce_float, minimum=0.0, maximum=1.0,
                )
            # FocalLoss (torchvision.ops.sigmoid_focal_loss): alpha in [0, 1] or -1 (no weighting), checked by
            # utils/loss_util.resolve_loss_config; gamma >= 0
            if "alpha" in args:
                args["alpha"] = _coerce_scalar_or_sequence(
                    args["alpha"], field="trainer.loss.args.alpha", coercer=_coerce_float,
                )
            if "gamma" in args:
                args["gamma"] = _coerce_scalar_or_sequence(
                    args["gamma"], field="trainer.loss.args.gamma", coercer=_coerce_float, minimum=0.0,
                )

    if "evaluation" in trainer_cfg:
        evaluation = trainer_cfg["evaluation"]
        if not isinstance(evaluation, MutableMapping):
            raise TypeError("trainer.evaluation must be a mapping")
        if "output_transformation" in evaluation:
            transform = evaluation["output_transformation"]
            if not isinstance(transform, MutableMapping):
                raise TypeError("trainer.evaluation.output_transformation must be a mapping")
            if "args" in transform:
                args = transform["args"]
                if not isinstance(args, MutableMapping):
                    raise TypeError("trainer.evaluation.output_transformation.args must be a mapping")
                if "dim" in args:
                    args["dim"] = _coerce_scalar_or_sequence(
                        args["dim"], field="trainer.evaluation.output_transformation.args.dim",
                        coercer=_coerce_int, minimum=1, maximum=1,
                    )

    # Canonical location is trainer.optimizer.scheduler; the flat legacy
    # location is still coerced so the optimizer builder can reject it clearly.
    optimizer = trainer_cfg.get("optimizer")
    candidates = [("trainer.scheduler", trainer_cfg.get("scheduler"))]
    if isinstance(optimizer, MutableMapping):
        candidates.append(("trainer.optimizer.scheduler", optimizer.get("scheduler")))

    for field, scheduler in candidates:
        if scheduler is None:
            continue
        if not isinstance(scheduler, MutableMapping):
            raise TypeError(f"{field} must be a mapping")

        args = scheduler.get("args")
        if args is None:
            args = {}
        if not isinstance(args, MutableMapping):
            raise TypeError(f"{field}.args must be a mapping")

        _coerce_dict_values_in_place(args)
        scheduler["args"] = args


# =============================================================================
# Model coercion
# =============================================================================
def _iter_model_entries(cfg: Mapping[str, Any]) -> Iterator[MutableMapping[str, Any]]:
    """Yield every model definition rather than silently coercing only the first."""
    model_cfg = cfg.get("model")

    if model_cfg is None:
        return

    if isinstance(model_cfg, MutableMapping):
        yield model_cfg
        return

    if isinstance(model_cfg, list):
        if not model_cfg:
            raise ValueError("cfg['model'] is an empty list")

        for index, entry in enumerate(model_cfg):
            if not isinstance(entry, MutableMapping):
                raise TypeError(
                    f"cfg['model'][{index}] must be a mapping, "
                    f"got {type(entry).__name__}"
                )
            yield entry
        return

    raise TypeError(
        f"cfg['model'] must be a mapping or list, got {type(model_cfg).__name__}"
    )


def coerce_model(cfg: MutableMapping[str, Any]) -> None:
    """Coerce numeric leaves in every configured model definition."""
    for index, model_entry in enumerate(_iter_model_entries(cfg)):
        if "layers" in model_entry:
            layers = model_entry["layers"]
            if not isinstance(layers, list):
                raise TypeError(f"model entry {index}.layers must be a list")
            for layer_index, layer in enumerate(layers):
                if not isinstance(layer, MutableMapping):
                    raise TypeError(
                        f"model entry {index}.layers[{layer_index}] must be a mapping"
                    )
                _coerce_dict_values_in_place(layer)

        if "nodes" in model_entry:
            nodes = model_entry["nodes"]
            if not isinstance(nodes, list):
                raise TypeError(f"model entry {index}.nodes must be a list")
            for node_index, node in enumerate(nodes):
                if not isinstance(node, MutableMapping):
                    raise TypeError(
                        f"model entry {index}.nodes[{node_index}] must be a mapping"
                    )
                _coerce_dict_values_in_place(node)

        if "architecture_args" in model_entry:
            architecture_args = model_entry["architecture_args"]
            if not isinstance(architecture_args, MutableMapping):
                raise TypeError(
                    f"model entry {index}.architecture_args must be a mapping"
                )
            _coerce_dict_values_in_place(architecture_args)


# =============================================================================
# Test coercion
# =============================================================================
def coerce_test(test_cfg: MutableMapping[str, Any]) -> None:
    """Coerce ordinary test runtime settings."""
    if "batch_size" in test_cfg:
        test_cfg["batch_size"] = _coerce_scalar_or_sequence(
            test_cfg["batch_size"],
            field="test.batch_size",
            coercer=_coerce_int,
            minimum=1,
        )

    if "num_workers" in test_cfg:
        test_cfg["num_workers"] = _coerce_scalar_or_sequence(
            test_cfg["num_workers"],
            field="test.num_workers",
            coercer=_coerce_int,
            minimum=0,
        )

    for key in ("pin_memory", "debug_h5_inspect"):
        if key in test_cfg:
            test_cfg[key] = _coerce_scalar_or_sequence(
                test_cfg[key],
                field=f"test.{key}",
                coercer=coerce_bool,
            )


# =============================================================================
# Evaluation-policy coercion
# =============================================================================
def coerce_evaluation(cfg: MutableMapping[str, Any]) -> None:
    """
    Normalize the validation-checkpoint and test-model-selection policy.

    This function validates the requested policy only. It does not decide
    whether a best-validation checkpoint exists and does not silently perform
    fallback. Runtime fallback belongs in ``sca/test.py``.
    """
    evaluation = cfg.get("evaluation")
    if evaluation is None:
        evaluation = {}
        cfg["evaluation"] = evaluation
    if not isinstance(evaluation, MutableMapping):
        raise TypeError("cfg['evaluation'] must be a mapping")

    validation = evaluation.get("validation")
    if validation is None:
        validation = {}
        evaluation["validation"] = validation
    if not isinstance(validation, MutableMapping):
        raise TypeError("cfg['evaluation']['validation'] must be a mapping")

    test_policy = evaluation.get("test")
    if test_policy is None:
        test_policy = {}
        evaluation["test"] = test_policy
    if not isinstance(test_policy, MutableMapping):
        raise TypeError("cfg['evaluation']['test'] must be a mapping")

    trainer = cfg.get("trainer") or {}
    if not isinstance(trainer, MutableMapping):
        raise TypeError("cfg['trainer'] must be a mapping when present")

    new_checkpointing_present = "checkpointing" in validation
    legacy_checkpointing_present = "checkpointing" in trainer

    if new_checkpointing_present:
        checkpointing = coerce_bool(
            validation["checkpointing"],
            field="evaluation.validation.checkpointing",
        )
    elif legacy_checkpointing_present:
        checkpointing = coerce_bool(
            trainer["checkpointing"],
            field="trainer.checkpointing",
        )
        logger.warning(
            "trainer.checkpointing is deprecated; use "
            "evaluation.validation.checkpointing"
        )
    else:
        checkpointing = True

    validation["checkpointing"] = checkpointing

    # Temporary migration bridge: epoch_utils.py in the current codebase still
    # reads trainer.checkpointing. Keep an existing legacy field synchronized
    # until that reader is refactored, but do not create the field when absent.
    if legacy_checkpointing_present:
        if trainer["checkpointing"] != checkpointing:
            logger.warning(
                "evaluation.validation.checkpointing overrides the conflicting "
                "deprecated trainer.checkpointing value"
            )
        trainer["checkpointing"] = checkpointing

    raw_policy = test_policy.get("load_model")

    if raw_policy is None:
        legacy_test_cfg = cfg.get("test") or {}
        if not isinstance(legacy_test_cfg, Mapping):
            raise TypeError("cfg['test'] must be a mapping when present")

        if "load_model" in legacy_test_cfg:
            raw_policy = legacy_test_cfg["load_model"]
            logger.warning(
                "test.load_model is deprecated; use evaluation.test.load_model"
            )
        elif "test_checkpoint_policy" in evaluation:
            raw_policy = evaluation["test_checkpoint_policy"]
            logger.warning(
                "evaluation.test_checkpoint_policy is deprecated; use "
                "evaluation.test.load_model"
            )
        else:
            raw_policy = "best_validation"

    test_policy["load_model"] = _coerce_choice(
        raw_policy,
        field="evaluation.test.load_model",
        choices=_SUPPORTED_TEST_MODEL_POLICIES,
    )


# =============================================================================
# Data coercion
# =============================================================================
def _iter_data_entries(cfg: Mapping[str, Any]) -> Iterator[MutableMapping[str, Any]]:
    """Yield dataset configurations from list, direct-map, or keyed-map schemas."""
    data = cfg.get("data")
    if data is None:
        return

    if isinstance(data, list):
        if not data:
            raise ValueError("cfg['data'] is an empty list")
        for index, entry in enumerate(data):
            if not isinstance(entry, MutableMapping):
                raise TypeError(
                    f"cfg['data'][{index}] must be a mapping, "
                    f"got {type(entry).__name__}"
                )
            yield entry
        return

    if not isinstance(data, MutableMapping):
        raise TypeError(
            f"cfg['data'] must be a mapping or list, got {type(data).__name__}"
        )

    if _SINGLE_DATASET_KEYS.intersection(data.keys()):
        yield data
        return

    yielded = False
    for dataset_name, entry in data.items():
        if not isinstance(entry, MutableMapping):
            continue
        entry.setdefault("name", str(dataset_name))
        yielded = True
        yield entry

    if not yielded and data:
        raise TypeError(
            "cfg['data'] does not match a supported dataset schema"
        )


def _coerce_range(entry: MutableMapping[str, Any], key: str) -> None:
    """Coerce an integer range/point-list while preserving list or tuple type."""
    if key not in entry:
        return

    value = entry[key]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"data.{key} must be a list or tuple of integers")

    normalized = [
        _coerce_int(item, field=f"data.{key}[{index}]", minimum=0)
        for index, item in enumerate(value)
    ]

    if len(normalized) < 2:
        raise ValueError(f"data.{key} must contain at least two integers")

    entry[key] = tuple(normalized) if isinstance(value, tuple) else normalized


def coerce_data(cfg: MutableMapping[str, Any]) -> None:
    """Coerce dataset controls, provenance display, ranges, and normalization."""
    for entry in _iter_data_entries(cfg):
        if "show_contract" in entry:
            entry["show_contract"] = coerce_bool(
                entry["show_contract"],
                field=f"data.{entry.get('name', '<unnamed>')}.show_contract",
            )

        for key in (
            "profiling_desync",
            "attack_desync",
            "multilabel",
            "chunk_size",
        ):
            if key in entry:
                entry[key] = _coerce_int(
                    entry[key],
                    field=f"data.{entry.get('name', '<unnamed>')}.{key}",
                    minimum=0,
                )

        for range_key in (
            "profiling_range",
            "attack_range",
            "validation_range",
            "test_range",
            "target_range",
        ):
            _coerce_range(entry, range_key)

        normalization = entry.get("trace_normalization")
        if normalization is None:
            continue
        if not isinstance(normalization, MutableMapping):
            raise TypeError(
                f"data.{entry.get('name', '<unnamed>')}.trace_normalization "
                "must be a mapping"
            )

        raw_mode = normalization.get("mode", "none")
        normalization["mode"] = _coerce_choice(
            raw_mode,
            field=(
                f"data.{entry.get('name', '<unnamed>')}"
                ".trace_normalization.mode"
            ),
            choices=_SUPPORTED_TRACE_NORMALIZATION_MODES,
        )

        if "feature_range" in normalization:
            bounds = normalization["feature_range"]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise TypeError(
                    f"data.{entry.get('name', '<unnamed>')}.trace_normalization.feature_range "
                    f"must be [low, high], got {bounds!r}"
                )
            normalization["feature_range"] = [
                _coerce_float(bound, field=f"data.{entry.get('name', '<unnamed>')}.trace_normalization.feature_range")
                for bound in bounds
            ]
        if "steps" in normalization and isinstance(normalization["steps"], (list, tuple)):
            normalization["steps"] = [str(step).strip().lower() for step in normalization["steps"]]

        for key in ("mean", "std", "min", "max"):
            if key in normalization:
                normalization[key] = _coerce_float(
                    normalization[key],
                    field=(
                        f"data.{entry.get('name', '<unnamed>')}"
                        f".trace_normalization.{key}"
                    ),
                )

        if "eps" in normalization:
            normalization["eps"] = _coerce_float(
                normalization["eps"],
                field=(
                    f"data.{entry.get('name', '<unnamed>')}"
                    ".trace_normalization.eps"
                ),
                minimum=0.0,
                minimum_inclusive=False,
            )


# =============================================================================
# Public API
# =============================================================================
def coerce_types(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply all supported coercions and return the same configuration object."""
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg).__name__}")

    trainer = cfg.get("trainer")
    if trainer is not None:
        if not isinstance(trainer, MutableMapping):
            raise TypeError("cfg['trainer'] must be a mapping")
        coerce_trainer(trainer)

    if "model" in cfg:
        coerce_model(cfg)

    test_cfg = cfg.get("test")
    if test_cfg is not None:
        if not isinstance(test_cfg, MutableMapping):
            raise TypeError("cfg['test'] must be a mapping")
        coerce_test(test_cfg)

    if "data" in cfg:
        coerce_data(cfg)

    coerce_evaluation(cfg)
    return cfg


__all__ = [
    "coerce_bool",
    "coerce_data",
    "coerce_evaluation",
    "coerce_model",
    "coerce_scalar",
    "coerce_test",
    "coerce_trainer",
    "coerce_types",
]
