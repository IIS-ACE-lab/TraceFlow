# services/side_channel_core/utils/cross_validation.py
"""
Cross-Validation Utilities for Side-Channel Analysis
====================================================

This module implements the infrastructure contract for deterministic T-fold
cross-validation over SCA profiling data.

Supported strategies
--------------------
- k_fold
- stratified_k_fold

Scientific protocol
-------------------
1. Cross-validation operates only on profiling data.
2. Attack-derived validation and test partitions are never folded.
3. Every profiling sample appears in exactly one validation fold.
4. Every fold uses a fresh model, optimizer, scheduler, and DataLoaders.
5. Exact fold indices are always persisted for reproducibility.
6. Fold artifacts are isolated under paths.cross_validation.
7. Fold runs are assessed only; the final model is the refit.
8. After successful CV, a fresh model is always refitted on the complete
   profiling population for trainer.epochs.
9. Individual fold models are never promoted as final production models.
10. Folds are assessed at the final configured epoch only; the refit uses
    the universal seed and the folds the next n_folds seeds in fold order,
    so no fold shares the refit's weight init and data order.
11. CV requires evaluation.validation.checkpointing=false: fold assessment is
    weightless and the refit has no validation data to select a checkpoint.

Configuration contract
----------------------
cross_validation:
  enabled: false
  n_folds: 10
  strategy: k_fold
  shuffle: true
  report_aggregate: false

paths:
  cross_validation:
    "artifacts/{{env}}/generated_models/{{model_name}}/cross_validation"

The global top-level cfg["seed"] is the seed source of the training (the refit takes it, the
folds the next n_folds seeds). The fold partition takes cross_validation.partition_seed when it is
set, else seed: with it, a sweep over seed keeps the folds and varies only the training
randomness, and a sweep over partition_seed varies only the folds (Wouters et al.'s ten repeats
change both at once).

This module does not train models. Training orchestration remains the
responsibility of src/sca/train.py, while one-model epoch execution remains
the responsibility of utils/epoch_utils.py.

Author: Shraddha Mishra
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from utils.logger import get_logger
from utils.metrics_logger import flat_history_row
from utils.optimizer_util import resolve_scheduler_config, takes_loop_length
from core.framework_switch import resolve_framework
from utils.path_utils import resolve_path, resolve_paths_dict
from utils.timestamps import now_iso
from utils.seal import check as check_seal, seal_text, seal_zip_file
from utils.seed import run_seed
from src.sca.dataset_types import random_profiling_keys, require_cross_validation_settings


logger = get_logger("CrossValidation")


SCHEMA_VERSION = 2

SUPPORTED_STRATEGIES = {
    "k_fold",
    "stratified_k_fold",
}

# Bookkeeping that must never be averaged as an assessment metric: epoch/seed
# fields and the raw numerator/denominator counts behind accuracy and SR.
_NON_ASSESSMENT_METRICS = {"train_loss_reduction", "epoch", "seed"}
_COUNT_SUFFIXES = ("_correct", "_total", "_successes", "_trials")


# Every per-fold trace-count crossing (stable and first; GE, SR and the single-run
# time_to_recover). A fold's crossing is None when it never gets there, so a mean
# over folds would only count the folds that did; the headline is the crossing of
# the fold-mean curve (_headline: traces_to_ge_1_mean_curve).
_TRACES_TO_GE = re.compile(
    r"^(first_)?(traces_to_(ge|sr)_(\d+|k)|time_to_recover(@\d+)?)$"
)


def _is_assessment_scalar(name: Any, value: Any) -> bool:
    name = str(name)
    return (
        _is_finite_number(value)
        and name not in _NON_ASSESSMENT_METRICS
        and not name.endswith(_COUNT_SUFFIXES)
        # None when a fold never reaches k; aggregated on the mean curve instead.
        and not _TRACES_TO_GE.match(name)
    )


def partition_sha256(folds: Sequence["FoldPlan"], n_samples: int) -> str:
    """Hash the fold assignment of every profiling row (fold id per row)."""
    assignment = np.full(int(n_samples), -1, dtype="<i8")
    for fold in folds:
        assignment[np.asarray(fold.validation_indices, dtype=np.int64)] = fold.fold_index
    return hashlib.sha256(assignment.tobytes()).hexdigest()


class CrossValidationError(RuntimeError):
    """A CV run failed; the message names the phase and fold."""


# =============================================================================
# Data contracts
# =============================================================================
@dataclass(frozen=True)
class CrossValidationSpec:
    """
    Validated cross-validation configuration.

    All values are concrete runtime values. No YAML parsing or fallback logic
    should be needed after this object has been created.
    """

    enabled: bool
    n_folds: int
    strategy: str
    shuffle: bool
    seed: int
    # cross_validation.refit.enabled: the final training on all profiling traces (default true).
    refit: bool = True
    # cross_validation.partition_seed: the seed that shuffles the traces into folds; None (absent)
    # means seed, as for every run before the key. The training seeds stay seed's (FOLD_SEED_RULE).
    partition_seed: Optional[int] = None

    @property
    def partition(self) -> int:
        """The seed of the fold partition: partition_seed when set, else seed."""
        return self.seed if self.partition_seed is None else self.partition_seed


@dataclass(frozen=True)
class FoldPlan:
    """
    Exact train/validation indices for one fold.

    fold_index is zero-based internally.
    fold_number is one-based for logs and artifact names.
    """

    fold_index: int
    n_folds: int
    train_indices: np.ndarray
    validation_indices: np.ndarray

    @property
    def fold_number(self) -> int:
        return int(self.fold_index) + 1

    @property
    def tag(self) -> str:
        return f"fold_{self.fold_number:02d}"


@dataclass(frozen=True)
class CrossValidationPaths:
    """
    Resolved artifact paths for one cross-validation run.
    """

    root: Path
    folds: Path
    aggregate: Path
    manifest: Path
    fold_indices: Path
    fold_results: Path
    scalar_statistics: Path
    summary: Path
    diagnostics: Path
    # <artifacts_root>/metrics/cross_validation: per-fold metric outputs.
    fold_metrics: Path


@dataclass(frozen=True)
class CrossValidationPlan:
    """
    Complete deterministic CV plan created before any fold begins training.
    """

    spec: CrossValidationSpec
    paths: CrossValidationPaths
    folds: Tuple[FoldPlan, ...]


# =============================================================================
# Generic helpers
# =============================================================================
def _now_iso() -> str:
    return now_iso()


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _require_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be true or false, got {value!r}")
    return value


def _require_positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got boolean {value!r}")

    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc

    if out < int(minimum):
        raise ValueError(f"{name} must be >= {minimum}, got {out}")

    return out


class RankCurvePair(list):
    """One ``[trace_count, mean_rank]`` point, written on a single YAML line."""


def rank_curve_pairs(curve: Any) -> List[RankCurvePair]:
    """Return a ``[trace_count, mean_rank]`` curve as ``[int, float]`` pairs."""
    arr = np.asarray(curve, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"rank curve must have shape [n, 2], got {arr.shape}")
    traces = arr[:, 0]
    if not np.all(np.isfinite(traces)) or np.any(traces != np.floor(traces)):
        raise ValueError("rank curve trace counts must be whole numbers")
    return [RankCurvePair([int(t), float(r)]) for t, r in arr.tolist()]


def _to_serializable(value: Any) -> Any:
    """
    Convert runtime values into JSON/YAML-safe representations.

    Large arrays are represented by descriptors rather than expanded into
    potentially huge JSON/YAML payloads.
    """
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()

        return {
            "_type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, Mapping):
        return {
            str(key): _to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]

    if isinstance(value, RankCurvePair):
        return RankCurvePair(_to_serializable(item) for item in value)

    if isinstance(value, list):
        return [_to_serializable(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _deep_merge(base: Dict[str, Any], patch: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge patch into base and return base.
    """
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)

    return base


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = seal_text(path, text)  # utils/seal.py: YAML, JSON and JSONL carry their own seal

    # Plain exclusive creation: the file mode follows the process umask.
    # tempfile.mkstemp() would force 0600 and os.replace() would keep it.
    temp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"

    try:
        with temp_path.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _atomic_write_json(path: Path, payload: Any) -> None:
    serializable = _to_serializable(payload)
    text = json.dumps(
        serializable,
        indent=2,
        ensure_ascii=False,
        sort_keys=False,
    )
    _atomic_write_text(Path(path), text + "\n")


class _CurveDumper(yaml.SafeDumper):
    pass


_CurveDumper.add_representer(
    RankCurvePair,
    lambda dumper, pair: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", list(pair), flow_style=True
    ),
)


def _atomic_write_yaml(path: Path, payload: Any) -> None:
    serializable = _to_serializable(payload)
    text = yaml.dump(
        serializable,
        Dumper=_CurveDumper,
        sort_keys=False,
        allow_unicode=True,
    )
    _atomic_write_text(Path(path), text)


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Not NamedTemporaryFile: it creates 0600 files and os.replace() keeps that.
    temp_path = path.parent / f".{path.stem}.{uuid.uuid4().hex}.npz"

    try:
        np.savez_compressed(
            str(temp_path),
            **{
                str(key): np.asarray(value)
                for key, value in arrays.items()
            },
        )
        seal_zip_file(temp_path, scope_path=path)
        os.replace(temp_path, path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not Path(path).exists():
        return {}

    check_seal(path)
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise TypeError(
            f"Expected YAML mapping in {path}, got {type(loaded).__name__}"
        )

    return loaded


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False

    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return bool(np.isfinite(float(value)))
        except Exception:
            return False

    return False


# =============================================================================
# Configuration validation
# =============================================================================
def validate_cross_validation_config(
    cfg: dict,
    *,
    n_samples: Optional[int] = None,
    labels: Optional[Sequence[Any]] = None,
) -> CrossValidationSpec:
    """
    Validate and normalize cfg["cross_validation"].

    Parameters
    ----------
    cfg:
        Complete runtime configuration.

    n_samples:
        Optional profiling-population size. When supplied, n_folds is checked
        against it.

    labels:
        Optional one-dimensional profiling labels. Required for validating
        stratified_k_fold class counts.

    Returns
    -------
    CrossValidationSpec
        Immutable validated runtime contract.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg).__name__}")

    raw_cv = cfg.get("cross_validation")
    cv = _require_mapping(raw_cv, name="cfg['cross_validation']")

    enabled = _require_bool(
        cv.get("enabled", False),
        name="cross_validation.enabled",
    )
    if enabled:
        require_cross_validation_settings(cfg)   # ASCADr: folds unattacked (sca/dataset_types.py)

    n_folds = _require_positive_int(
        cv.get("n_folds", 10),
        name="cross_validation.n_folds",
        minimum=2,
    )

    strategy = str(cv.get("strategy", "k_fold")).strip().lower()
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            "cross_validation.strategy must be one of "
            f"{sorted(SUPPORTED_STRATEGIES)}, got {strategy!r}"
        )

    shuffle = _require_bool(
        cv.get("shuffle", True),
        name="cross_validation.shuffle",
    )

    if "epoch_policy" in cv:
        raise ValueError(
            "CV refit epochs are bound internally to trainer.epochs; "
            "remove cross_validation.epoch_policy"
        )
    refit = refit_enabled(cfg)
    partition_seed = None
    if cv.get("partition_seed") is not None:            # empty (the template's blank key) = absent
        partition_seed = _require_positive_int(
            cv["partition_seed"], name="cross_validation.partition_seed", minimum=0,
        )
        if not shuffle:
            raise ValueError(
                "cross_validation.partition_seed is set but cross_validation.shuffle is false: unshuffled "
                "folds are contiguous blocks of the profiling traces and take no seed. Hint: set "
                "cross_validation.shuffle: true, or remove partition_seed"
            )

    # The partition is a function of the top-level seed only. Fold configs
    # carry fold_seed() for weight init/data order, so read the universal
    # seed they record instead of their own seed.
    runtime = cfg.get("cross_validation_runtime")
    seed = _require_positive_int(
        (
            runtime["universal_seed"]
            if isinstance(runtime, Mapping) and runtime.get("universal_seed") is not None
            else run_seed(cfg)
        ),
        name="seed",
        minimum=0,
    )

    if enabled:
        paths_cfg = _require_mapping(
            cfg.get("paths"),
            name="cfg['paths']",
        )

        cv_path = paths_cfg.get("cross_validation")
        if not isinstance(cv_path, str) or not cv_path.strip():
            raise KeyError(
                "paths.cross_validation is required when "
                "cross_validation.enabled=true"
            )
        _validate_cv_training_contract(cfg)

    if n_samples is not None:
        sample_count = _require_positive_int(
            n_samples,
            name="profiling sample count",
            minimum=2,
        )

        if n_folds > sample_count:
            raise ValueError(
                "cross_validation.n_folds cannot exceed the number of "
                f"profiling samples: n_folds={n_folds}, samples={sample_count}"
            )

    if labels is not None:
        y = np.asarray(labels)

        if y.ndim != 1:
            raise ValueError(
                f"Profiling labels must be one-dimensional, got shape={y.shape}"
            )

        if y.size == 0:
            raise ValueError("Profiling labels are empty")

        if n_samples is not None and int(y.shape[0]) != int(n_samples):
            raise ValueError(
                "Profiling label count does not match profiling sample count: "
                f"labels={int(y.shape[0])}, samples={int(n_samples)}"
            )

        if strategy == "stratified_k_fold":
            classes, counts = np.unique(y, return_counts=True)
            insufficient = [
                (str(cls), int(count))
                for cls, count in zip(classes, counts)
                if int(count) < n_folds
            ]

            if insufficient:
                detail = ", ".join(
                    f"class={cls} count={count}"
                    for cls, count in insufficient
                )
                raise ValueError(
                    "stratified_k_fold requires every represented class to "
                    f"contain at least n_folds={n_folds} samples. "
                    f"Insufficient classes: {detail}"
                )

    return CrossValidationSpec(
        enabled=enabled,
        n_folds=n_folds,
        strategy=strategy,
        shuffle=shuffle,
        seed=seed,
        refit=refit,
        partition_seed=partition_seed,
    )


def check_refit_against_test(cfg: Mapping[str, Any], spec: CrossValidationSpec) -> None:
    """The refit and the test go together: the test attacks the refit's model and nothing else.

    Refused, before any fold trains: no refit while the test is attacked. Warned: a refit while the
    test is not attacked (perform_attack.test.attack_times: 0), a model no process will use. The
    test counts as attacked exactly when the orchestrator does not skip it
    (utils/test_registry.unattacked_test_reason).
    """
    from utils.test_registry import unattacked_test_reason

    test_skipped = unattacked_test_reason(cfg) is not None
    if not spec.refit and not test_skipped:
        message = (
            "cross_validation.refit.enabled: false trains no model on all profiling traces, but the "
            "test attacks that model (perform_attack.test.attack_times > 0). Hint: set "
            "cross_validation.refit.enabled: true to test the refit, or perform_attack.test."
            "attack_times: 0 to keep the CV results only"
        )
        logger.error("CV plan rejected before any fold is trained: %s", message)
        raise CrossValidationError(message)
    if spec.refit and test_skipped:
        logger.warning(
            "cross_validation.refit.enabled: true trains a final model on all profiling traces, but "
            "perform_attack.test.attack_times: 0 never attacks it, so the refit is used by no result. "
            "Set cross_validation.refit.enabled: false to skip it (it does not change the model_hash), "
            "or perform_attack.test.attack_times >= 1 to test it."
        )


def require_refit_model(cfg: Mapping[str, Any], model_name: Any) -> None:
    """Refuse to test a CV run made with cross_validation.refit.enabled: false: it has no model.

    Called before the checkpoint is looked for (src/sca/test.py, src/sca/test_request.py), so the
    reason is this one, not a missing checkpoint manifest. A run made before the key existed had
    its refit (absent means true).
    """
    cv = cfg.get("cross_validation") or {}
    if not (isinstance(cv, Mapping) and cv.get("enabled") is True) or refit_enabled(cfg):
        return
    raise ValueError(
        f"{model_name} cannot be tested: {REFIT_SKIPPED_REASON}. Hint: its CV results are in "
        "cross_validation/aggregate. cross_validation.refit is not part of the model_hash, and a "
        "repeat of a model_hash runs from its own snapshot, so to test this configuration train it "
        "without cross-validation (a new model_hash)"
    )


def refit_enabled(cfg: Mapping[str, Any]) -> bool:
    """cross_validation.refit.enabled: whether the final training on all profiling traces runs.

    An internal setting (owner ruling 2026-10-22): absent means true, which is what every run made
    before the key existed did, and it never joins the model_hash (utils/model_naming.py), since
    the folds and their CV results are the same either way. Only the refit gives the test a model.
    """
    cv = cfg.get("cross_validation") or {}
    block = cv.get("refit") if isinstance(cv, Mapping) else None
    if block is None:
        return True
    if not isinstance(block, Mapping):
        raise TypeError(f"cross_validation.refit must be a mapping {{enabled: true|false}}, got {block!r}")
    unknown = sorted(set(block) - {"enabled"})
    if unknown:
        raise ValueError(f"cross_validation.refit has unknown fields {unknown}; allowed: ['enabled']")
    return _require_bool(block.get("enabled", True), name="cross_validation.refit.enabled")


def _validate_cv_training_contract(cfg: Mapping[str, Any]) -> None:
    """Reject settings that contradict fixed-epoch CV plus validation-free refit.

    Runs before any fold trains so an incompatible configuration fails in
    seconds instead of after every fold has completed.
    """
    evaluation = cfg.get("evaluation") or {}
    validation = evaluation.get("validation") if isinstance(evaluation, Mapping) else None
    if not isinstance(validation, Mapping) or validation.get("checkpointing") is not False:
        raise ValueError(
            "cross_validation.enabled=true requires "
            "evaluation.validation.checkpointing=false: folds are assessed at "
            "the final configured epoch without persisting weights, and the "
            "refit has no validation data from which to select a checkpoint"
        )

    trainer = _require_mapping(cfg.get("trainer"), name="cfg['trainer']")
    if trainer.get("early_stopping", False) is not False:
        raise ValueError("cross_validation.enabled=true requires trainer.early_stopping=false")

    optimizer = trainer.get("optimizer")
    # Keras schedules (deep_learning.framework: keras) are functions of the
    # optimizer's step count, every one of them (decay_steps, boundaries,
    # first_decay_steps). A fold and the refit take different numbers of steps
    # per epoch, so the refit would not follow the learning rates the folds were
    # assessed with: refused, as OneCycleLR is below.
    if isinstance(optimizer, Mapping) and resolve_framework(cfg).is_keras:
        scheduler = optimizer.get("scheduler")
        kind = scheduler.get("type", "none") if isinstance(scheduler, Mapping) else scheduler
        if kind is not None and str(kind).strip().lower() not in {"", "none", "null", "off"}:
            import inspect

            import torch

            from core.framework_switch import keras_schedules

            name = str(kind).strip()
            if name not in keras_schedules() and inspect.isclass(getattr(torch.optim.lr_scheduler, name, None)):
                # A torch.optim.lr_scheduler class (OneCycleLR, ...): Keras has no such schedule.
                message = (
                    f"trainer.optimizer.scheduler.type={kind!r} is not a keras.optimizers.schedules class: "
                    "it is a torch.optim.lr_scheduler class, available under deep_learning.framework: "
                    "pytorch. Hint: set deep_learning.framework: pytorch, or the scheduler to none"
                )
                logger.error("CV configuration rejected before any fold is trained: %s", message)
                raise ValueError(message)
            message = (
                f"trainer.optimizer.scheduler.type={kind} cannot be used with "
                "cross_validation.enabled=true under deep_learning.framework: keras: "
                "every keras.optimizers.schedules class advances per optimizer step, "
                "and the folds and the refit take different numbers of steps per epoch "
                "(e.g. 225 per fold and 250 for the refit with 45,000 and 50,000 traces "
                "at batch size 200), so the refit would train on another learning-rate "
                "path than the folds were assessed on. Hint: set the scheduler to none, "
                "or train with deep_learning.framework: pytorch and a per-epoch scheduler "
                "such as StepLR, MultiStepLR, ExponentialLR or CosineAnnealingLR."
            )
            logger.error("CV configuration rejected before any fold is trained: %s", message)
            raise ValueError(message)
    if isinstance(optimizer, Mapping) and not resolve_framework(cfg).is_keras:
        scheduler = resolve_scheduler_config(optimizer.get("scheduler"))
        if scheduler is not None and scheduler["type"] == "ReduceLROnPlateau":
            raise ValueError(
                "ReduceLROnPlateau needs validation data and cannot drive the "
                "validation-free CV refit; choose an epoch-based scheduler"
            )
        if scheduler is not None and scheduler["type"] == "OneCycleLR" and not takes_loop_length(scheduler):
            # A length in the config is one number for every loop: folds and refit differ in size.
            given = sorted(k for k in ("total_steps", "epochs", "steps_per_epoch") if k in scheduler["args"])
            message = (
                f"trainer.optimizer.scheduler.args {given} cannot be used with "
                "cross_validation.enabled=true: OneCycleLR's length (total_steps = "
                "epochs x steps_per_epoch) would be one number, but the folds and the refit "
                "train on different numbers of traces (e.g. 45,000 per fold and 50,000 for "
                "the refit with 50,000 profiling traces and 10 folds), so one total_steps "
                "cannot fit both. Hint: remove total_steps, epochs and steps_per_epoch from "
                "the args: each training loop then gives OneCycleLR its own length "
                "(trainer.epochs x its batches per epoch) and completes the cycle. Or use "
                "none, or a per-epoch scheduler such as StepLR, MultiStepLR, ExponentialLR "
                "or CosineAnnealingLR."
            )
            logger.error("CV configuration rejected before any fold is trained: %s", message)
            raise ValueError(message)
        if scheduler is not None and scheduler["type"] == "CyclicLR":
            # step_size_up / step_size_down count optimizer steps (iterations).
            message = (
                "trainer.optimizer.scheduler.type=CyclicLR cannot be used with "
                "cross_validation.enabled=true: its step_size_up and step_size_down count "
                "optimizer steps, and the folds and the refit take different numbers of steps "
                "per epoch (e.g. 225 per fold and 250 for the refit with 45,000 and 50,000 "
                "traces at batch size 200), so the refit would end at another point of the "
                "cycle than the folds were assessed at. Hint: use OneCycleLR without a length "
                "(each loop gives it its own), none, or a per-epoch scheduler such as StepLR "
                "or CosineAnnealingLR; CyclicLR stays available without cross-validation."
            )
            logger.error("CV configuration rejected before any fold is trained: %s", message)
            raise ValueError(message)


# =============================================================================
# Path resolution
# =============================================================================
def resolve_cross_validation_paths(
    cfg: dict,
    *,
    create: bool = True,
) -> CrossValidationPaths:
    """
    Resolve the single configured cross-validation root and its derived paths.
    """
    raw_paths = _require_mapping(cfg.get("paths"), name="cfg['paths']")

    raw_root = raw_paths.get("cross_validation")
    if not isinstance(raw_root, str) or not raw_root.strip():
        raise KeyError("cfg['paths']['cross_validation'] is required")

    root = Path(
        resolve_path(
            raw_root,
            cfg,
            strict=True,
            inject=True,
        )
    )

    paths = CrossValidationPaths(
        root=root,
        folds=root / "folds",
        aggregate=root / "aggregate",
        manifest=root / "manifest.yaml",
        fold_indices=root / "fold_indices.npz",
        fold_results=root / "aggregate" / "fold_results.json",
        scalar_statistics=root / "aggregate" / "scalar_statistics.json",
        summary=root / "aggregate" / "summary.yaml",
        diagnostics=root / "aggregate" / "cross_validation_diagnostics.json",
        fold_metrics=(
            Path(resolve_path(raw_paths["artifacts_root"], cfg, strict=True, inject=True))
            if raw_paths.get("artifacts_root")
            else root.parent
        ) / "metrics" / "cross_validation",
    )

    if create:
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.folds.mkdir(parents=True, exist_ok=True)
        paths.aggregate.mkdir(parents=True, exist_ok=True)

    return paths


def get_fold_artifact_paths(
    paths: CrossValidationPaths,
    *,
    fold_index: int,
    n_folds: int,
    create: bool = True,
) -> Dict[str, Path]:
    """
    Derive the output paths of one fold.

    Only the fold result lives under paths.cross_validation/folds/fold_NN.
    Fold metrics (including key-recovery reports) go to
    <artifacts_root>/metrics/cross_validation/fold_NN; checkpoints and models
    use the run's standard paths.checkpoints / paths.model_out with a
    fold-numbered file name (see fold_artifact_name()).

    fold_index is zero-based.
    """
    fold_index = int(fold_index)
    n_folds = int(n_folds)

    if fold_index < 0 or fold_index >= n_folds:
        raise ValueError(
            f"fold_index must be in [0, {n_folds - 1}], got {fold_index}"
        )

    fold_number = fold_index + 1
    root = paths.folds / f"fold_{fold_number:02d}"

    metrics = paths.fold_metrics / f"fold_{fold_number:02d}"
    out = {
        "root": root,
        "metrics": metrics,
        "key_recovery": metrics / "key_recovery",
        "result": root / "fold_result.yaml",
    }

    if create:
        root.mkdir(parents=True, exist_ok=True)
        metrics.mkdir(parents=True, exist_ok=True)

    return out


def fold_artifact_name(model_name: str, fold_number: int) -> str:
    """Standard checkpoint/model file stem for one fold, e.g. <model>_fold_01."""
    return f"{model_name}_fold_{int(fold_number):02d}"


# Fold 1 must not reuse the refit's seed: its weight init and data order
# would then equal the refit's.
FOLD_SEED_RULE = "refit uses seed; folds use the next n_folds seeds in fold order"


def fold_seed(universal_seed: int, fold_index: int) -> int:
    """Weight-init/data-order seed of one fold (FOLD_SEED_RULE); fold_index is zero-based."""
    return int(universal_seed) + int(fold_index) + 1


def build_fold_runtime_config(
    cfg: dict,
    *,
    paths: CrossValidationPaths,
    fold_index: int,
    n_folds: int,
) -> dict:
    """
    Create an isolated runtime configuration for one fold.

    Only fold output paths are replaced. Dataset paths remain unchanged, so all
    folds read from the same profiling HDF5 file without duplicating data.

    The runtime context tells epoch_utils that:
    - this is a fold run;
    - fold artifacts must remain isolated.
    """
    fold_cfg = copy.deepcopy(cfg)

    fold_paths = get_fold_artifact_paths(
        paths,
        fold_index=fold_index,
        n_folds=n_folds,
        create=True,
    )

    model_name = str(fold_cfg.get("model_name") or "").strip()
    if not model_name:
        raise ValueError(
            "cfg['model_name'] must be resolved before building a fold config"
        )

    fold_cfg.setdefault("paths", {})

    # Only metric outputs are fold-specific. Checkpoints, models, logs,
    # graphs and the config snapshot keep the run's standard paths; fold
    # checkpoint/model files carry the fold number in their name instead.
    fold_cfg["paths"]["metrics"] = str(
        fold_paths["metrics"]
    )
    # key_recovery takes precedence over metrics in key_recovery_paths();
    # without this override every fold would write into the refit's reports.
    fold_cfg["paths"]["key_recovery"] = str(
        fold_paths["key_recovery"]
    )

    universal_seed = _require_positive_int(
        run_seed(cfg), name="seed", minimum=0
    )
    seed = fold_seed(universal_seed, fold_index)
    fold_cfg["seed"] = seed

    fold_cfg["cross_validation_runtime"] = {
        "active": True,
        "is_fold": True,
        "is_refit": False,
        "fold_index": int(fold_index),
        "fold_number": int(fold_index) + 1,
        "n_folds": int(n_folds),
        "universal_seed": universal_seed,
        "seed": seed,
        "fold_root": str(fold_paths["root"]),
        "fold_result_path": str(fold_paths["result"]),
    }

    fold_cfg["paths_abs"] = resolve_paths_dict(
        fold_cfg["paths"],
        fold_cfg,
        strict=True,
        mkdir=False,
        inject=True,
    )

    return fold_cfg


def build_refit_runtime_config(
    cfg: dict,
    *,
    resolved_epochs: int,
) -> dict:
    """
    Create the final full-profiling refit configuration.

    Unlike fold configs, the normal configured artifact paths remain active.
    """
    refit_cfg = copy.deepcopy(cfg)

    resolved_epochs = _require_positive_int(
        resolved_epochs,
        name="resolved refit epochs",
        minimum=1,
    )

    trainer_cfg = refit_cfg.setdefault("trainer", {})
    if not isinstance(trainer_cfg, dict):
            raise TypeError(
                "cfg['trainer'] must be a mapping"
            )

    trainer_cfg["epochs"] = resolved_epochs

    # Refit duration is internally bound to the configured epoch count.
    # Early stopping must not shorten the full-profiling refit.
    trainer_cfg["early_stopping"] = False

    universal_seed = _require_positive_int(
        run_seed(refit_cfg), name="seed", minimum=0
    )
    refit_cfg["cross_validation_runtime"] = {
        "active": True,
        "is_fold": False,
        "is_refit": True,
        "fold_index": None,
        "fold_number": None,
        "n_folds": int(
            (refit_cfg.get("cross_validation") or {}).get("n_folds", 0)
        ),
        "universal_seed": universal_seed,
        "seed": universal_seed,
        "resolved_refit_epochs": resolved_epochs,
    }

    refit_cfg["paths_abs"] = resolve_paths_dict(
        refit_cfg["paths"],
        refit_cfg,
        strict=True,
        mkdir=False,
        inject=True,
    )

    return refit_cfg


def resolve_cross_validation_runtime(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the execution contract consumed by the epoch loop."""
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")

    raw = cfg.get("cross_validation_runtime", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("cross_validation_runtime must be a mapping")

    active = _require_bool(
        raw.get("active", False),
        name="cross_validation_runtime.active",
    )
    is_fold = _require_bool(
        raw.get("is_fold", False),
        name="cross_validation_runtime.is_fold",
    )
    is_refit = _require_bool(
        raw.get("is_refit", False),
        name="cross_validation_runtime.is_refit",
    )

    if (is_fold and is_refit) or active != (is_fold or is_refit):
        raise ValueError(
            "cross_validation_runtime.active must be true for exactly one "
            "of is_fold or is_refit"
        )

    fold_index = fold_number = n_folds = refit_epochs = None
    if is_fold:
        fold_index = _require_positive_int(
            raw.get("fold_index"),
            name="cross_validation_runtime.fold_index",
            minimum=0,
        )
        fold_number = _require_positive_int(
            raw.get("fold_number"),
            name="cross_validation_runtime.fold_number",
        )
        n_folds = _require_positive_int(
            raw.get("n_folds"),
            name="cross_validation_runtime.n_folds",
            minimum=2,
        )
        if fold_number != fold_index + 1 or fold_number > n_folds:
            raise ValueError("Invalid cross-validation fold coordinates")
        tag = f"fold_{fold_number:02d}"
        description = f"CV fold {fold_number}/{n_folds}"
    elif is_refit:
        refit_epochs = _require_positive_int(
            raw.get("resolved_refit_epochs"),
            name="cross_validation_runtime.resolved_refit_epochs",
        )
        n_folds = _require_positive_int(
            raw.get("n_folds"),
            name="cross_validation_runtime.n_folds",
            minimum=2,
        )
        tag = "refit"
        description = "CV final full-profiling refit"
    else:
        tag = "standard"
        description = "standard training"

    return {
        "active": active,
        "is_fold": is_fold,
        "is_refit": is_refit,
        "fold_index": fold_index,
        "fold_number": fold_number,
        "n_folds": n_folds,
        "resolved_refit_epochs": refit_epochs,
        "tag": tag,
        "description": description,
    }


# =============================================================================
# Fold generation
# =============================================================================
def _import_splitters():
    """
    Import scikit-learn only when cross-validation is actually requested.
    """
    try:
        from sklearn.model_selection import KFold, StratifiedKFold
    except ImportError as exc:
        raise ImportError(
            "Cross-validation requires scikit-learn. "
            "Install it in the active environment before enabling CV."
        ) from exc

    return KFold, StratifiedKFold


def _validate_generated_folds(
    folds: Sequence[FoldPlan],
    *,
    n_samples: int,
    n_folds: int,
) -> None:
    """
    Validate exclusivity, completeness, and exact validation coverage.
    """
    if len(folds) != int(n_folds):
        raise ValueError(
            f"Expected {n_folds} folds, generated {len(folds)}"
        )

    all_indices = np.arange(int(n_samples), dtype=np.int64)
    expected_set = set(all_indices.tolist())
    validation_seen: List[int] = []

    for fold in folds:
        train_idx = np.asarray(fold.train_indices, dtype=np.int64)
        val_idx = np.asarray(fold.validation_indices, dtype=np.int64)

        if train_idx.ndim != 1 or val_idx.ndim != 1:
            raise ValueError(
                f"{fold.tag}: fold indices must be one-dimensional"
            )

        if train_idx.size == 0:
            raise ValueError(f"{fold.tag}: training indices are empty")

        if val_idx.size == 0:
            raise ValueError(f"{fold.tag}: validation indices are empty")

        if np.unique(train_idx).size != train_idx.size:
            raise ValueError(f"{fold.tag}: duplicate training indices detected")

        if np.unique(val_idx).size != val_idx.size:
            raise ValueError(
                f"{fold.tag}: duplicate validation indices detected"
            )

        if np.intersect1d(train_idx, val_idx).size != 0:
            raise ValueError(
                f"{fold.tag}: training and validation indices overlap"
            )

        fold_union = set(train_idx.tolist()) | set(val_idx.tolist())
        if fold_union != expected_set:
            raise ValueError(
                f"{fold.tag}: train/validation indices do not cover the "
                "complete profiling population"
            )

        if train_idx.min() < 0 or train_idx.max() >= n_samples:
            raise ValueError(
                f"{fold.tag}: training index outside [0, {n_samples - 1}]"
            )

        if val_idx.min() < 0 or val_idx.max() >= n_samples:
            raise ValueError(
                f"{fold.tag}: validation index outside [0, {n_samples - 1}]"
            )

        validation_seen.extend(val_idx.tolist())

    validation_seen_array = np.asarray(
        validation_seen,
        dtype=np.int64,
    )

    if validation_seen_array.size != n_samples:
        raise ValueError(
            "Combined validation folds do not contain exactly one entry per "
            f"profiling sample: entries={validation_seen_array.size}, "
            f"samples={n_samples}"
        )

    unique, counts = np.unique(
        validation_seen_array,
        return_counts=True,
    )

    if unique.size != n_samples or not np.all(counts == 1):
        raise ValueError(
            "Every profiling sample must appear in exactly one validation fold"
        )


def create_fold_plans(
    labels: Sequence[Any],
    cfg: dict,
) -> Tuple[FoldPlan, ...]:
    """
    Create deterministic fold assignments from profiling labels.

    Plain K-fold ignores labels.
    Stratified K-fold preserves leakage-class proportions.
    """
    y = np.asarray(labels)

    if y.ndim != 1:
        raise ValueError(
            f"Profiling labels must be one-dimensional, got shape={y.shape}"
        )

    n_samples = int(y.shape[0])

    spec = validate_cross_validation_config(
        cfg,
        n_samples=n_samples,
        labels=y,
    )

    if not spec.enabled:
        raise ValueError(
            "create_fold_plans() called while cross_validation.enabled=false"
        )

    KFold, StratifiedKFold = _import_splitters()

    sample_indices = np.arange(
        n_samples,
        dtype=np.int64,
    )

    random_state = spec.partition if spec.shuffle else None   # the partition's own seed (W1)

    if spec.strategy == "k_fold":
        splitter = KFold(
            n_splits=spec.n_folds,
            shuffle=spec.shuffle,
            random_state=random_state,
        )
        raw_splits = splitter.split(sample_indices)

    elif spec.strategy == "stratified_k_fold":
        splitter = StratifiedKFold(
            n_splits=spec.n_folds,
            shuffle=spec.shuffle,
            random_state=random_state,
        )
        raw_splits = splitter.split(sample_indices, y)

    else:
        raise RuntimeError(
            f"Unsupported validated CV strategy: {spec.strategy}"
        )

    folds: List[FoldPlan] = []

    for fold_index, (train_idx, val_idx) in enumerate(raw_splits):
        folds.append(
            FoldPlan(
                fold_index=int(fold_index),
                n_folds=spec.n_folds,
                train_indices=np.asarray(
                    train_idx,
                    dtype=np.int64,
                ),
                validation_indices=np.asarray(
                    val_idx,
                    dtype=np.int64,
                ),
            )
        )

    _validate_generated_folds(
        folds,
        n_samples=n_samples,
        n_folds=spec.n_folds,
    )

    return tuple(folds)


# =============================================================================
# Index persistence
# =============================================================================
def save_fold_indices(
    path: Path,
    *,
    spec: CrossValidationSpec,
    folds: Sequence[FoldPlan],
    n_samples: int,
) -> None:
    """
    Persist exact fold indices as a compressed NPZ artifact.
    """
    arrays: Dict[str, np.ndarray] = {
        "schema_version": np.asarray(
            [SCHEMA_VERSION],
            dtype=np.int64,
        ),
        "n_samples": np.asarray(
            [int(n_samples)],
            dtype=np.int64,
        ),
        "n_folds": np.asarray(
            [spec.n_folds],
            dtype=np.int64,
        ),
        "seed": np.asarray(
            [spec.partition],       # the seed that drew this partition
            dtype=np.int64,
        ),
        "strategy": np.asarray(
            [spec.strategy],
            dtype=np.str_,
        ),
        "shuffle": np.asarray(
            [spec.shuffle],
            dtype=np.bool_,
        ),
        "partition_sha256": np.asarray(
            [partition_sha256(folds, int(n_samples))],
            dtype=np.str_,
        ),
    }

    for fold in folds:
        arrays[f"{fold.tag}_train"] = np.asarray(
            fold.train_indices,
            dtype=np.int64,
        )
        arrays[f"{fold.tag}_validation"] = np.asarray(
            fold.validation_indices,
            dtype=np.int64,
        )

    _atomic_write_npz(Path(path), arrays)

    # logger.info(
    #     "Saved exact CV fold indices: %s",
    #     str(path),
    # )


def load_fold_indices(path: Path) -> Tuple[FoldPlan, ...]:
    """
    Load previously persisted exact fold assignments.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Fold-index artifact not found: {path}")

    check_seal(path)
    with np.load(str(path), allow_pickle=False) as data:
        n_folds = int(np.asarray(data["n_folds"]).reshape(-1)[0])

        folds: List[FoldPlan] = []

        for fold_index in range(n_folds):
            tag = f"fold_{fold_index + 1:02d}"
            train_key = f"{tag}_train"
            val_key = f"{tag}_validation"

            if train_key not in data or val_key not in data:
                raise KeyError(
                    f"Fold-index artifact is missing {train_key!r} "
                    f"or {val_key!r}"
                )

            folds.append(
                FoldPlan(
                    fold_index=fold_index,
                    n_folds=n_folds,
                    train_indices=np.asarray(
                        data[train_key],
                        dtype=np.int64,
                    ),
                    validation_indices=np.asarray(
                        data[val_key],
                        dtype=np.int64,
                    ),
                )
            )

        n_samples = int(
            np.asarray(data["n_samples"]).reshape(-1)[0]
        )

    _validate_generated_folds(
        folds,
        n_samples=n_samples,
        n_folds=n_folds,
    )

    return tuple(folds)


# =============================================================================
# Manifest handling
# =============================================================================
def _class_distribution(labels: Sequence[Any]) -> Dict[str, int]:
    y = np.asarray(labels)

    if y.ndim != 1:
        raise ValueError(
            f"Labels must be one-dimensional, got shape={y.shape}"
        )

    values, counts = np.unique(y, return_counts=True)

    return {
        str(value.item() if isinstance(value, np.generic) else value): int(count)
        for value, count in zip(values, counts)
    }


def build_initial_manifest(
    cfg: dict,
    *,
    spec: CrossValidationSpec,
    paths: CrossValidationPaths,
    folds: Sequence[FoldPlan],
    labels: Sequence[Any],
) -> Dict[str, Any]:
    """
    Construct the initial cross-validation manifest.
    """
    model_name = str(cfg.get("model_name") or "").strip()
    if not model_name:
        raise ValueError(
            "cfg['model_name'] must be resolved before CV initialization"
        )

    dataset_alias = cfg.get("dataset_alias") or cfg.get("dataset")
    if not dataset_alias:
        data_cfg = cfg.get("data")
        if isinstance(data_cfg, Mapping):
            dataset_alias = data_cfg.get("alias") or data_cfg.get("name")
        elif (
            isinstance(data_cfg, list)
            and len(data_cfg) == 1
            and isinstance(data_cfg[0], Mapping)
        ):
            dataset_alias = data_cfg[0].get("alias") or data_cfg[0].get("name")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "initialized",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "env": cfg.get("env"),
        "project": (cfg.get("project") or {}).get("name"),
        "model_name": model_name,
        "model_hash": cfg.get("model_hash"),
        "dataset_alias": dataset_alias,
        "seed": spec.seed,
        "profiling_samples": int(len(labels)),
        "class_distribution": _class_distribution(labels),
        "contract": {
            "enabled": spec.enabled,
            "n_folds": spec.n_folds,
            "strategy": spec.strategy,
            "shuffle": spec.shuffle,
            "refit": (
                {"enabled": True, "epochs_source": "trainer.epochs"} if spec.refit else {"enabled": False}
            ),
        },
        "paths": {
            "root": str(paths.root),
            "fold_indices": str(paths.fold_indices),
            "folds": str(paths.folds),
            "aggregate": str(paths.aggregate),
        },
        "folds": [
            {
                "fold_index": fold.fold_index,
                "fold_number": fold.fold_number,
                "tag": fold.tag,
                "status": "pending",
                "train_samples": int(fold.train_indices.size),
                "validation_samples": int(
                    fold.validation_indices.size
                ),
            }
            for fold in folds
        ],
        "refit": (
            {"enabled": True, "epochs_source": "trainer.epochs", "resolved_epochs": None, "status": "pending"}
            if spec.refit
            else {"enabled": False, "status": "disabled"}
        ),
    }


def write_manifest(
    paths: CrossValidationPaths,
    manifest: Mapping[str, Any],
) -> None:
    _atomic_write_yaml(paths.manifest, manifest)

    logger.info(
        "Cross-validation manifest written: %s",
        str(paths.manifest),
    )


def update_manifest(
    paths: CrossValidationPaths,
    patch: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Atomically update the CV manifest using a recursive patch.
    """
    manifest = _load_yaml_mapping(paths.manifest)
    _deep_merge(manifest, patch)
    manifest["updated_at"] = _now_iso()

    _atomic_write_yaml(paths.manifest, manifest)
    return manifest


def mark_fold_status(
    paths: CrossValidationPaths,
    *,
    fold_index: int,
    status: str,
    result: Optional[Mapping[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update one fold's status in the manifest.
    """
    manifest = _load_yaml_mapping(paths.manifest)
    folds = manifest.get("folds")

    if not isinstance(folds, list):
        raise ValueError(
            f"Manifest has malformed folds section: {paths.manifest}"
        )

    fold_index = int(fold_index)

    if fold_index < 0 or fold_index >= len(folds):
        raise IndexError(
            f"fold_index={fold_index} outside manifest fold range"
        )

    entry = folds[fold_index]

    if not isinstance(entry, dict):
        raise TypeError(
            f"Manifest fold entry {fold_index} is not a mapping"
        )

    entry["status"] = str(status)
    entry["updated_at"] = _now_iso()

    if result is not None:
        entry["result"] = _to_serializable(result)

    if error is not None:
        entry["error"] = str(error)

    manifest["updated_at"] = _now_iso()
    _atomic_write_yaml(paths.manifest, manifest)

    return manifest


# =============================================================================
# Plan initialization
# =============================================================================
def prepare_cross_validation_plan(
    cfg: dict,
    *,
    labels: Sequence[Any],
    create_directories: bool = True,
) -> CrossValidationPlan:
    """
    Validate configuration, create exact folds, persist their indices, and
    initialize the manifest.

    This function must run before any fold creates a model or optimizer.
    """
    y = np.asarray(labels)

    if y.ndim != 1:
        raise ValueError(
            f"Profiling labels must be one-dimensional, got shape={y.shape}"
        )

    spec = validate_cross_validation_config(
        cfg,
        n_samples=int(y.shape[0]),
        labels=y,
    )

    if not spec.enabled:
        raise ValueError(
            "prepare_cross_validation_plan() called while "
            "cross_validation.enabled=false"
        )
    check_refit_against_test(cfg, spec)

    paths = resolve_cross_validation_paths(
        cfg,
        create=create_directories,
    )

    # create_fold_plans() asserts pairwise-disjoint held-out parts, full
    # coverage and train/held-out separation before returning.
    folds = create_fold_plans(y, cfg)
    n_profiling = int(y.shape[0])
    held_out_sizes = sorted({int(fold.validation_indices.size) for fold in folds})
    if len(held_out_sizes) > 1:
        # Fold rank curves end at the held-out size; unequal folds cannot be
        # averaged point-wise (Eq. 7), and that would only surface after every
        # fold had trained. Fail here, before any training, with a hint.
        largest_multiple = (n_profiling // spec.n_folds) * spec.n_folds
        message = (
            f"Unequal held-out fold sizes {held_out_sizes}: {n_profiling} profiling "
            f"traces are not divisible by cross_validation.n_folds={spec.n_folds}, so "
            "the fold rank curves would end at different trace counts and could not "
            "be averaged into the fold-mean curve. Hint: choose n_folds dividing "
            f"{n_profiling}, or shorten data.profiling_range to a multiple of "
            f"n_folds (e.g. {largest_multiple} traces)."
        )
        logger.error("CV plan rejected before any fold is trained: %s", message)
        raise CrossValidationError(message)
    # Disjoint validation attacks split each held-out fold exactly (TODO.md §22): checked here, before
    # any fold trains, not per fold after training.
    try:
        validation_traces_per_attack(cfg, held_out_sizes[0])
    except CrossValidationError as exc:
        logger.error("CV plan rejected before any fold is trained: %s", exc)
        raise
    digest = partition_sha256(folds, n_profiling)
    logger.info(
        "CV partition check passed: %d disjoint held-out parts (sizes %s) cover "
        "all %d profiling rows; train/held-out disjoint per fold | "
        "partition_seed=%d | partition_sha256=%s",
        len(folds),
        sorted({int(fold.validation_indices.size) for fold in folds}),
        int(y.shape[0]),
        spec.partition,
        digest,
    )

    save_fold_indices(
        paths.fold_indices,
        spec=spec,
        folds=folds,
        n_samples=int(y.shape[0]),
    )

    manifest = build_initial_manifest(
        cfg,
        spec=spec,
        paths=paths,
        folds=folds,
        labels=y,
    )
    manifest["partition"] = {
        "seed": spec.partition,
        "seed_source": "cross_validation.partition_seed" if spec.partition_seed is not None else "seed",
        "sha256": digest,
        "fold_indices": str(paths.fold_indices),
        "fold_seed_rule": f"{FOLD_SEED_RULE} (weight init and data order only)",
    }

    write_manifest(paths, manifest)

    logger.info(
        "Prepared %d-fold CV plan | strategy=%s shuffle=%s seed=%d",
        spec.n_folds,
        spec.strategy,
        spec.shuffle,
        spec.seed,
    )

    return CrossValidationPlan(
        spec=spec,
        paths=paths,
        folds=folds,
    )


# =============================================================================
# Fold-result persistence
# =============================================================================
def write_fold_result(
    paths: CrossValidationPaths,
    *,
    fold_index: int,
    n_folds: int,
    result: Mapping[str, Any],
) -> Path:
    """
    Save one fold's compact result summary.

    Array-valued metrics are represented by descriptors. Their complete values
    remain in the fold-local metrics artifacts written by metrics_logger.py.
    """
    fold_paths = get_fold_artifact_paths(
        paths,
        fold_index=fold_index,
        n_folds=n_folds,
        create=True,
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "fold_index": int(fold_index),
        "fold_number": int(fold_index) + 1,
        "saved_at": _now_iso(),
        **dict(result),
    }

    result_path = fold_paths["result"]
    _atomic_write_yaml(result_path, payload)

    # logger.info(
    #     "Saved fold result: %s",
    #     str(result_path),
    # )

    return result_path


# =============================================================================
# Result aggregation
# =============================================================================
def _sample_statistics(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1 or array.size == 0:
        raise ValueError("Cannot summarize an empty metric sequence")

    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": (
            float(np.std(array, ddof=1))
            if array.size > 1
            else 0.0
        ),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate_fold_results(
    fold_results: Sequence[Mapping[str, Any]],
    *,
    expected_folds: int,
) -> Dict[str, Any]:
    """
    Aggregate final-epoch assessment scalars across completed folds.

    Only ``final_train.metrics`` and ``final_validation.metrics`` contribute,
    reported as ``train.<metric>`` and ``validation.<metric>``. Epoch counts,
    seeds, fold identity and trajectories are never aggregated; trajectories
    stay in fold_results.json for diagnostics.

    All expected folds are required. Partial cross-validation results are not
    accepted as complete scientific results.
    """
    expected_folds = _require_positive_int(
        expected_folds,
        name="expected_folds",
        minimum=2,
    )

    if len(fold_results) != expected_folds:
        raise ValueError(
            "Cannot aggregate incomplete cross-validation results: "
            f"expected={expected_folds}, received={len(fold_results)}"
        )

    seen_fold_numbers: set[int] = set()
    metric_values: Dict[str, List[float]] = {}

    for position, result in enumerate(fold_results):
        if not isinstance(result, Mapping):
            raise TypeError(
                f"Fold result at position {position} must be a mapping"
            )

        fold_number_raw = result.get(
            "fold_number",
            position + 1,
        )

        fold_number = _require_positive_int(
            fold_number_raw,
            name=f"fold_results[{position}].fold_number",
            minimum=1,
        )

        if fold_number > expected_folds:
            raise ValueError(f"fold_number must be in 1..{expected_folds}, got {fold_number}")
        if fold_number in seen_fold_numbers:
            raise ValueError(
                f"Duplicate fold_number in results: {fold_number}"
            )

        seen_fold_numbers.add(fold_number)

        for split in ("final_train", "final_validation"):
            snapshot = result.get(split)
            metrics = snapshot.get("metrics") if isinstance(snapshot, Mapping) else None
            if not isinstance(metrics, Mapping):
                raise KeyError(
                    f"fold_results[{position}].{split}.metrics is required"
                )
            prefix = split.removeprefix("final_")
            for name, value in metrics.items():
                # Arrays, strings and bookkeeping fields are not averaged.
                if not _is_assessment_scalar(name, value):
                    continue
                metric_values.setdefault(f"{prefix}.{name}", []).append(float(value))

    statistics: Dict[str, Dict[str, Any]] = {}

    for metric_name, values in sorted(metric_values.items()):
        # Only report metrics present in every fold.
        if len(values) != expected_folds:
            logger.warning(
                "Skipping partially available CV scalar '%s': "
                "%d/%d folds",
                metric_name,
                len(values),
                expected_folds,
            )
            continue

        statistics[metric_name] = _sample_statistics(values)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "n_folds_expected": expected_folds,
        "n_folds_completed": len(fold_results),
        "all_folds_successful": True,
        "scalar_metrics": statistics,
    }


# =============================================================================
# Refit epoch resolution
# =============================================================================
def validation_traces_per_attack(cfg: Mapping[str, Any], held_out: int) -> int:
    """The trace count a fold's validation rank curve ends at (TODO.md §22): the held-out size;
    perform_attack.validation.attack_trace_budget when set; for attack_distribution: disjoint
    without a budget, held-out size / attack_times (the curve is the mean over the fold's disjoint
    attacks of that many traces each). Raises CrossValidationError, before any fold trains, when
    the held-out traces cannot hold the attacks (sca/perform_attack.traces_per_attack)."""
    from sca.perform_attack import attack_plan, traces_per_attack

    plan = attack_plan(cfg, "validation")
    times = int(plan["attack_times"])
    if times == 0:
        return int(held_out)
    if plan["attack_trace_budget"] is None and plan["attack_distribution"] == "disjoint" and held_out % times:
        raise CrossValidationError(
            f"perform_attack.validation.attack_times={times} does not divide the held-out fold size "
            f"{held_out}: the disjoint attacks of a fold must split its held-out traces exactly. Hint: choose "
            f"attack_times dividing {held_out}, set attack_trace_budget, or attack_distribution: full_population."
        )
    try:
        return traces_per_attack(plan, int(held_out), field="perform_attack.validation")
    except ValueError as exc:
        raise CrossValidationError(
            f"{exc} (the held-out fold size). Hint: choose attack_times / attack_trace_budget that fit "
            f"{held_out} held-out traces, or attack_distribution: full_population without a budget."
        ) from exc


def attack_plan_of_validation(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """perform_attack.validation as the folds attacked: attack_times, shuffle, attack_distribution."""
    from sca.perform_attack import attack_plan

    return dict(attack_plan(cfg, "validation"))


def _aggregate_final_fold_curves(
    plan: CrossValidationPlan,
    cfg: dict,
    fold_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Average matching final-epoch rank curves, never best epochs or refits.

    Input ranks retain the configured internal convention. The separately
    labelled paper curve subtracts that base to make the best key rank zero.
    Curves already average trace-order repetitions within each fold; here
    every fold receives equal weight. No interpolation or truncation is used.
    """
    snapshots = [result.get("final_validation") or {} for result in fold_results]
    curves = [snapshot.get("metrics", {}).get("mean_attack_rank_curve") for snapshot in snapshots]
    if all(curve is None for curve in curves):
        requested = cfg.get("metrics_to_log", cfg.get("metrics", [])) or []
        if random_profiling_keys(cfg):
            return {"status": "not_available",
                    "reason": "random profiling keys: the folds are not attacked; rank curves come from the test"}
        if "mean_attack_rank_curve" in requested:
            raise ValueError("Configured mean_attack_rank_curve is missing from all final fold results")
        return {"status": "not_available", "reason": "mean_attack_rank_curve was not recorded"}
    if any(curve is None for curve in curves):
        raise ValueError("Every fold must provide its final mean_attack_rank_curve")
    if len(fold_results) != plan.spec.n_folds:
        raise ValueError("Final curve aggregation requires every planned fold")

    from sca.rank_curve import PUBLIC_RANK_BASE

    rank_base = (cfg.get("sca") or {}).get("rank_base", PUBLIC_RANK_BASE)    # an old snapshot may still name it
    if isinstance(rank_base, bool) or rank_base != PUBLIC_RANK_BASE:
        raise ValueError(f"sca.rank_base must be {PUBLIC_RANK_BASE}: TraceFlow ranks the keys from 1 (rank_curve)")
    epochs = _require_positive_int(cfg["trainer"]["epochs"], name="trainer.epochs")
    planned = {fold.fold_index: fold for fold in plan.folds}
    seen = set()
    trace_counts = None
    fold_ranks = []
    provenance = []
    for result, snapshot, raw_curve in zip(fold_results, snapshots, curves):
        index = result.get("fold_index")
        if isinstance(index, bool) or not isinstance(index, int) or index not in planned or index in seen:
            raise ValueError(f"Invalid or duplicate fold_index in final curves: {index!r}")
        seen.add(index)
        fold = planned[index]
        if result.get("fold_number") != index + 1 or result.get("n_folds") != plan.spec.n_folds:
            raise ValueError(f"Fold {index}: result identity differs from the fold plan")
        if result.get("seed") != fold_seed(plan.spec.seed, fold.fold_index):
            raise ValueError(
                f"Fold {index}: seed {result.get('seed')!r} differs from the "
                f"fold rule {FOLD_SEED_RULE} = "
                f"{fold_seed(plan.spec.seed, fold.fold_index)}"
            )
        if any(value != epochs for value in (
            snapshot.get("epoch"), result.get("epochs_completed"), result.get("epochs_configured"),
        )):
            raise ValueError(f"Fold {index}: fixed-epoch curve requires completion of all {epochs} epochs")
        if snapshot.get("population_source") != "Profiling_traces":
            raise ValueError(f"Fold {index}: curve must use held-out profiling traces")
        if not np.array_equal(snapshot.get("validation_indices"), fold.validation_indices):
            raise ValueError(f"Fold {index}: validation indices differ from the saved plan")
        population_count = int(fold.train_indices.size + fold.validation_indices.size)
        if snapshot.get("profiling_count") != population_count:
            raise ValueError(f"Fold {index}: profiling population size differs from the plan")

        curve = np.asarray(raw_curve, dtype=np.float64)
        if curve.ndim != 2 or curve.shape[1] != 2 or curve.shape[0] == 0 or not np.isfinite(curve).all():
            raise ValueError(f"Fold {index}: expected a finite non-empty [trace_count, mean_rank] curve")
        x, ranks = curve[:, 0], curve[:, 1]
        if np.any(x != np.floor(x)) or x[0] < 1 or np.any(np.diff(x) <= 0):
            raise ValueError(f"Fold {index}: trace counts must be increasing positive integers")
        traces_per_attack = validation_traces_per_attack(cfg, int(fold.validation_indices.size))
        if x[-1] != traces_per_attack:
            raise ValueError(
                f"Fold {index}: curve must end at {traces_per_attack} traces (the held-out population"
                + (" per attack: perform_attack.validation attack_trace_budget or disjoint attacks"
                   if traces_per_attack != fold.validation_indices.size else "")
                + f"), got {int(x[-1])}"
            )
        if np.any(ranks < rank_base) or np.any(ranks > rank_base + 255):
            raise ValueError(f"Fold {index}: ranks are outside the configured 256-key rank convention")
        if trace_counts is None:
            trace_counts = x
        elif not np.array_equal(trace_counts, x):
            raise ValueError("Fold curves have different trace counts; refusing to truncate or interpolate")
        fold_ranks.append(ranks)
        provenance.append({"fold_index": index, "fold_number": index + 1,
                           "epoch": epochs, "validation_count": int(fold.validation_indices.size),
                           "traces_per_attack": int(traces_per_attack)})

    stacked = np.stack(fold_ranks)
    mean_rank = np.mean(stacked, axis=0)
    std_rank = np.std(stacked, axis=0, ddof=1)
    # Reported next to the mean, which one outlying fold can pull (Wu, Perin
    # and Picek 2021).
    median_rank = np.median(stacked, axis=0)
    # The spread around the median: the 25th and 75th percentile curves (numpy's linear rule).
    q25_rank, q75_rank = np.percentile(stacked, [25, 75], axis=0)
    # logger.info(
    #     "CV fold-mean curve (Eq. 7, %d folds, epoch %d): GE@start=%.4f "
    #     "GE@mid=%.4f GE@final=%.4f (traces %d/%d/%d)",
    #     len(fold_ranks), epochs,
    #     mean_rank[0], mean_rank[len(mean_rank) // 2], mean_rank[-1],
    #     int(trace_counts[0]), int(trace_counts[len(trace_counts) // 2]), int(trace_counts[-1]),
    # )

    # Same rule as the per-fold metric, applied to the mean curve: the headline
    # is the STABLE crossing (mean rank stays <= k to the end of the fold
    # population); the first, possibly transient, crossing is kept alongside.
    from src.sca.perform_attack import _get_ge_thresholds, _threshold_trace_points

    mean_curve = np.column_stack((trace_counts, mean_rank))
    # The fold-median curve's crossings beside the mean's: Wu, Perin and Picek (COSADE 2022) report
    # the median over independent trainings, which one outlying fold cannot pull.
    median_curve = np.column_stack((trace_counts, median_rank))
    # perform_attack.ge_thresholds (always with 1) sets the headline
    # thresholds, whatever metrics_to_log records per fold.
    traces_to_ge = {}
    for k in _get_ge_thresholds(cfg):
        first_trace, stable_trace = _threshold_trace_points(mean_curve, threshold=k)
        median_first, median_stable = _threshold_trace_points(median_curve, threshold=k)
        # A fold reaches k when its own curve stays <= k to the end (stable).
        reached = sum(
            _threshold_trace_points(np.column_stack((trace_counts, ranks)), threshold=k)[1]
            is not None
            for ranks in fold_ranks
        )
        traces_to_ge[str(k)] = {
            "mean_curve_trace": stable_trace,
            "mean_curve_stable_trace": stable_trace,
            "mean_curve_first_trace": first_trace,
            "median_curve_stable_trace": median_stable,
            "median_curve_first_trace": median_first,
            "rule": "stable_crossing_headline",
            "folds_reached": int(reached),
            "n_folds": len(fold_ranks),
        }
        # logger.info(
        #     "CV traces_to_ge_%d: mean curve=%s | folds reached %d/%d",
        #     k,
        #     first_trace if first_trace is not None else "not reached",
        #     reached,
        #     len(fold_ranks),
        # )

    return {
        "status": "completed",
        "traces_to_ge": traces_to_ge,
        "fold_rank_curves": {
            "trace_counts": trace_counts.tolist(),
            "fold_numbers": [item["fold_number"] for item in provenance],
            "ranks": stacked.tolist(),
        },
        "population_source": "held_out_profiling_folds",
        # traces per attack: the held-out size, or held-out / attack_times for disjoint attacks
        "traces_per_attack": int(trace_counts[-1]),
        "attack_distribution": attack_plan_of_validation(cfg),
        "selection": "final_configured_epoch",
        "epoch": epochs,
        "n_folds": len(fold_ranks),
        "aggregation": "pointwise_arithmetic_mean_across_folds",
        "internal_rank_base": rank_base,
        "paper_rank_base": 0,
        "mean_attack_rank_curve": rank_curve_pairs(np.column_stack((trace_counts, mean_rank))),
        "paper_mean_attack_rank_curve": rank_curve_pairs(np.column_stack((trace_counts, mean_rank - rank_base))),
        "sample_std_rank_curve": rank_curve_pairs(np.column_stack((trace_counts, std_rank))),
        "median_rank_curve": rank_curve_pairs(np.column_stack((trace_counts, median_rank))),
        "q25_rank_curve": rank_curve_pairs(np.column_stack((trace_counts, q25_rank))),
        "q75_rank_curve": rank_curve_pairs(np.column_stack((trace_counts, q75_rank))),
        "std_ddof": 1,
        "fold_indices_path": str(plan.paths.fold_indices),
        "folds": provenance,
        "universal_seed": plan.spec.seed,
        "partition_seed": plan.spec.partition,
        "fold_seed_rule": FOLD_SEED_RULE,
        "sca": copy.deepcopy(cfg.get("sca", {})),
        "perform_attack": copy.deepcopy(cfg.get("perform_attack", {})),
    }


def resolve_refit_epochs(
    cfg: dict,
    *,
    fold_results: Sequence[Mapping[str, Any]],
) -> int:
    """Return the configured fixed epoch count for the mandatory refit."""
    validate_cross_validation_config(cfg)
    trainer = _require_mapping(cfg.get("trainer"), name="cfg['trainer']")
    return _require_positive_int(
        trainer.get("epochs"),
        name="trainer.epochs",
        minimum=1,
    )


# =============================================================================
# Final aggregation and manifest completion
# =============================================================================
REFIT_SKIPPED_REASON = (
    "cross_validation.refit.enabled: false: the folds and their CV results are complete; no final "
    "model is trained on all profiling traces, so there is no model to test"
)


def _refit_section(spec: CrossValidationSpec, resolved_epochs: Optional[int]) -> Dict[str, Any]:
    """The refit block of the CV summary and manifest once the folds are aggregated."""
    if not spec.refit:
        return {"enabled": False, "status": "disabled", "reason": REFIT_SKIPPED_REASON}
    return {
        "enabled": True,
        "epochs_source": "trainer.epochs",
        "resolved_epochs": resolved_epochs,
        "status": "ready",
    }


def save_aggregate_results(
    paths: CrossValidationPaths,
    *,
    fold_results: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    resolved_refit_epochs: Optional[int],
    spec: CrossValidationSpec,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Save fold summaries, scalar statistics, and a separate numeric rank curve.
    """
    _atomic_write_json(
        paths.fold_results,
        {
            "schema_version": SCHEMA_VERSION,
            "fold_results": list(fold_results),
        },
    )

    _atomic_write_json(
        paths.scalar_statistics,
        {key: value for key, value in aggregate.items() if key != "rank_curve"},
    )
    rank_curve_path = paths.aggregate / "mean_rank_curve.json"
    _atomic_write_json(rank_curve_path, aggregate["rank_curve"])
    stacked = aggregate["rank_curve"].get("fold_rank_curves")
    if stacked:
        _atomic_write_npz(
            paths.aggregate / "fold_rank_curves.npz",
            {
                "trace_counts": np.asarray(stacked["trace_counts"]),
                "fold_numbers": np.asarray(stacked["fold_numbers"]),
                "fold_ranks": np.asarray(stacked["ranks"]),
                "mean_ranks": np.asarray(aggregate["rank_curve"]["mean_attack_rank_curve"])[:, 1],
            },
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "folds_completed",
        "completed_at": _now_iso(),
        "n_folds": spec.n_folds,
        "strategy": spec.strategy,
        "shuffle": spec.shuffle,
        "seed": spec.seed,
        "partition_seed": spec.partition,
        "all_folds_successful": True,
        "rank_curve": {
            "path": str(rank_curve_path),
            "status": aggregate["rank_curve"]["status"],
        },
        "diagnostics": dict(diagnostics or {
            "requested": False,
            "status": "disabled",
            "path": None,
            "lifecycle_authority": False,
        }),
        "refit": _refit_section(spec, resolved_refit_epochs),
        "headline": dict(aggregate.get("headline") or {}),
        "scalar_metrics": aggregate.get(
            "scalar_metrics",
            {},
        ),
    }
    _atomic_write_yaml(paths.summary, summary)

    # logger.info(
    #     "Saved CV aggregate artifacts under: %s",
    #     str(paths.aggregate),
    # )


def _run_optional_aggregate_diagnostics(
    *,
    plan: CrossValidationPlan,
    cfg: dict,
    fold_results: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run supplemental diagnostics without affecting CV control flow."""
    cv_cfg = cfg.get("cross_validation")
    requested = (
        isinstance(cv_cfg, Mapping)
        and cv_cfg.get("report_aggregate") is True
    )
    state: Dict[str, Any] = {
        "requested": requested,
        "status": "disabled" if not requested else "unavailable",
        "path": None,
        "lifecycle_authority": False,
    }
    if not requested:
        return state

    try:
        # Local import keeps this optional observer outside the core CV import
        # and lifecycle path.
        from utils.cross_validation_diagnostics import (
            create_cross_validation_diagnostic_report,
        )

        report = create_cross_validation_diagnostic_report(
            cfg=cfg,
            fold_results=fold_results,
            aggregate=aggregate,
            report_path=plan.paths.diagnostics,
            logger=logger,
        )
        if report is not None:
            state.update({
                "status": "completed",
                "path": str(plan.paths.diagnostics),
            })
    except Exception as exc:
        # A diagnostic can never prevent aggregation, refit, or official test.
        logger.warning(
            "CV aggregate diagnostics unavailable; experiment will continue: %s",
            exc,
        )
    return state


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


# Trajectory column of each process that holds its per-epoch accuracy.
_ACCURACY_TRAJECTORY = {"train": "training_accuracy", "validation": "validation_accuracy"}


def mean_accuracy_requested(cfg: Mapping[str, Any]) -> bool:
    """Return whether metrics_to_log lists mean_accuracy."""
    names = cfg.get("metrics_to_log", cfg.get("metrics", [])) or []
    return any(
        str(next(iter(item)) if isinstance(item, Mapping) and item else item).strip().lower()
        == "mean_accuracy"
        for item in names
    )


def fold_mean_accuracy(
    fold_results: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Mean accuracy over the folds, per process and per epoch.

    Every fold has the same weight. ``mean_accuracy_std`` is the sample
    standard deviation over the folds and ``accuracy`` holds the values the
    mean was taken from, fold 1 first. A process is left out when a fold did
    not record its accuracy.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for process, column in _ACCURACY_TRAJECTORY.items():
        series = []
        for result in fold_results:
            trajectory = result.get("trajectory") or {}
            values = trajectory.get(column) or []
            epochs = trajectory.get("epochs") or []
            if len(values) != len(epochs) or not values or not all(_is_finite_number(v) for v in values):
                series = []
                break
            series.append((list(epochs), [float(v) for v in values]))
        if not series:
            continue
        epochs = series[0][0]
        if any(fold_epochs != epochs for fold_epochs, _values in series):
            raise ValueError("Folds recorded different epochs; cannot average their accuracy")
        rows = []
        for index, epoch in enumerate(epochs):
            fold_values = [values[index] for _epochs, values in series]
            stats = _stats(fold_values)
            rows.append({
                "epoch": int(epoch),
                "process": process,
                "accuracy": fold_values,
                "mean_accuracy": stats["mean"],
                "mean_accuracy_std": stats["std"],
                "n_folds": len(series),
            })
        out[process] = rows
    return out


_FIT_TIME_COLUMNS = ("fit_time_s", "fit_time_h", "fit_time_per_epoch")


def requested_fit_time_columns(cfg: Mapping[str, Any]) -> List[str]:
    """
    Fit-time columns whose mean over the folds metrics_to_log asks for.

    mean_fit_time_{s} -> fit_time_s. fit_time_{s} alone records the fit time
    of every run and asks for no mean.
    """
    names = cfg.get("metrics_to_log", cfg.get("metrics", [])) or []
    plain = {
        str(next(iter(item)) if isinstance(item, Mapping) and item else item)
        .strip().lower().replace("{", "").replace("}", "")
        for item in names
    }
    if "mean_fit_time" in plain:
        plain.add("mean_fit_time_s")
    return [column for column in _FIT_TIME_COLUMNS if f"mean_{column}" in plain]


def fold_mean_fit_time(
    paths: CrossValidationPaths,
    *,
    model_name: str,
    n_folds: int,
    columns: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Mean fit time over the folds, per process and per epoch.

    The paper's mean fit time (Sect. 2.4.2) is the ``train`` value: the time
    to train one fold model, averaged over the folds. Read from the history
    rows of every fold; ``mean_<column>_std`` is the sample standard deviation.
    ``<column>`` holds the values the mean was taken from, fold 1 first.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for process in ("train", "validation"):
        per_fold = []
        for fold_number in range(1, int(n_folds) + 1):
            path = (
                paths.fold_metrics / f"fold_{fold_number:02d}" / "scalars"
                / f"{model_name}_metrics_{process}_history.jsonl"
            )
            if not path.is_file():
                per_fold = []
                break
            rows = {}
            check_seal(path)
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    # A fit time requested only through its mean is a dependency of the row.
                    row = flat_history_row(json.loads(line))
                    rows[int(row["epoch"])] = row  # the latest record of an epoch
            per_fold.append(rows)
        if not per_fold:
            continue
        epochs = sorted(set.intersection(*(set(rows) for rows in per_fold)))
        process_rows = []
        for epoch in epochs:
            row: Dict[str, Any] = {"epoch": epoch, "process": process, "n_folds": len(per_fold)}
            for column in columns:
                values = [rows[epoch].get(column) for rows in per_fold]
                if all(_is_finite_number(v) for v in values):
                    stats = _stats([float(v) for v in values])
                    row[column] = [float(v) for v in values]
                    row[f"mean_{column}"] = stats["mean"]
                    row[f"mean_{column}_std"] = stats["std"]
            if len(row) > 3:
                process_rows.append(row)
        if process_rows:
            out[process] = process_rows
    return out


def merge_fold_means(
    *tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """One row per process and epoch from several fold-mean tables."""
    merged: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for table in tables:
        for process, rows in table.items():
            for row in rows:
                merged.setdefault(process, {}).setdefault(int(row["epoch"]), {}).update(row)
    return {process: [rows[e] for e in sorted(rows)] for process, rows in merged.items()}


def save_fold_mean_accuracy(
    paths: CrossValidationPaths,
    *,
    model_name: str,
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Path]:
    """
    Write the fold means next to the fold metrics, one history file per process.

    Like every history row: epoch, process, the requested means as columns,
    and what they were computed from (the folds' values, their standard
    deviation, the number of folds) under "dependencies".
    """

    def _history_row(row: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"epoch": row["epoch"], "process": row["process"]}
        dependencies: Dict[str, Any] = {}
        for key, value in row.items():
            if key in out:
                continue
            is_mean = str(key).startswith("mean_") and not str(key).endswith("_std")
            (out if is_mean else dependencies)[key] = value
        if dependencies:
            out["dependencies"] = dependencies
        return out

    written = []
    for process, process_rows in rows.items():
        path = (
            paths.fold_metrics / "mean" / "scalars"
            / f"{model_name}_metrics_{process}_history.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, "".join(json.dumps(_history_row(row)) + "\n" for row in process_rows))
        written.append(path)
    return written


def build_cv_headline(
    plan: CrossValidationPlan,
    *,
    cfg: Mapping[str, Any],
    fold_results: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> Dict[str, Any]:
    """Headline CV numbers for the summary, registry and final report."""
    curve = aggregate.get("rank_curve") or {}
    ge_1 = (curve.get("traces_to_ge") or {}).get("1") or {}
    accuracy = [
        float(result["final_validation"]["metrics"]["accuracy"])
        for result in fold_results
        if _is_finite_number(
            (result.get("final_validation") or {}).get("metrics", {}).get("accuracy")
        )
    ]
    times = [
        float(result["training_time_s"])
        for result in fold_results
        if _is_finite_number(result.get("training_time_s"))
    ]
    accuracy_stats, time_stats = _stats(accuracy), _stats(times)
    mean_curve = curve.get("mean_attack_rank_curve")
    end = lambda name: float(curve[name][-1][1]) if curve.get(name) else None   # a curve's last point
    epoch = curve.get("epoch")
    if epoch is None and random_profiling_keys(cfg):
        # No fold rank curve (random profiling keys, ASCADr): the epoch every fold was assessed at.
        assessed = {(result.get("final_validation") or {}).get("epoch") for result in fold_results}
        epoch = assessed.pop() if len(assessed) == 1 else None
    return {
        "model_name": cfg.get("model_name"),
        "model_hash": cfg.get("model_hash"),
        "n_folds": plan.spec.n_folds,
        "epoch": epoch,
        "aggregate_path": str(plan.paths.aggregate),
        "mean_curve_final_ge": float(mean_curve[-1][1]) if mean_curve else None,
        "mean_curve_traces": int(mean_curve[-1][0]) if mean_curve else None,
        # Stable crossing of the fold-mean curve (headline); first crossing kept.
        "traces_to_ge_1_mean_curve": ge_1.get("mean_curve_stable_trace"),
        "first_traces_to_ge_1_mean_curve": ge_1.get("mean_curve_first_trace"),
        "folds_reaching_ge_1": (
            f"{ge_1['folds_reached']}/{ge_1['n_folds']}" if ge_1 else None
        ),
        # The median over the folds (independent trainings) and its spread at the last trace,
        # beside the mean (Wu, Perin and Picek, COSADE 2022): the 25th / 75th percentiles and
        # the sample standard deviation of the fold ranks.
        "median_curve_final_rank": end("median_rank_curve"),
        "traces_to_ge_1_median_curve": ge_1.get("median_curve_stable_trace"),
        "first_traces_to_ge_1_median_curve": ge_1.get("median_curve_first_trace"),
        "final_rank_q25": end("q25_rank_curve"),
        "final_rank_q75": end("q75_rank_curve"),
        "final_rank_std": end("sample_std_rank_curve"),
        "accuracy_mean": accuracy_stats["mean"],
        "accuracy_std": accuracy_stats["std"],
        "training_time_mean_s": time_stats["mean"],
        "training_time_std_s": time_stats["std"],
    }


def finalize_cross_validation(
    plan: CrossValidationPlan,
    *,
    cfg: dict,
    fold_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate successful folds and prepare the mandatory full-data refit.
    """
    aggregate = aggregate_fold_results(
        fold_results,
        expected_folds=plan.spec.n_folds,
    )
    aggregate["rank_curve"] = _aggregate_final_fold_curves(plan, cfg, fold_results)

    resolved_refit_epochs = resolve_refit_epochs(
        cfg,
        fold_results=fold_results,
    )

    diagnostics = _run_optional_aggregate_diagnostics(
        plan=plan,
        cfg=cfg,
        fold_results=fold_results,
        aggregate=aggregate,
    )
    aggregate["headline"] = build_cv_headline(
        plan, cfg=cfg, fold_results=fold_results, aggregate=aggregate
    )
    accuracy_means: Dict[str, List[Dict[str, Any]]] = {}
    if mean_accuracy_requested(cfg):
        accuracy_means = fold_mean_accuracy(fold_results)
        if "validation" not in accuracy_means:
            raise ValueError(
                "mean_accuracy is requested but the folds did not record the "
                "accuracy of their held-out traces"
            )
        for process, rows in accuracy_means.items():
            # The fold values the mean comes from: stored beside the headline.
            aggregate.setdefault("headline_dependencies", {})[f"accuracy_{process}"] = rows[-1]["accuracy"]
            aggregate["headline"][f"mean_accuracy_{process}"] = rows[-1]["mean_accuracy"]
            aggregate["headline"][f"mean_accuracy_{process}_std"] = rows[-1]["mean_accuracy_std"]

    time_means: Dict[str, List[Dict[str, Any]]] = {}
    fit_time_columns = requested_fit_time_columns(cfg)
    if fit_time_columns:
        time_means = fold_mean_fit_time(
            plan.paths,
            model_name=str(cfg.get("model_name")),
            n_folds=plan.spec.n_folds,
            columns=fit_time_columns,
        )
        # The paper's mean fit time: training of one fold model, mean over the folds.
        for column in fit_time_columns:
            last = (time_means.get("train") or [{}])[-1]
            if f"mean_{column}" in last:
                aggregate.setdefault("headline_dependencies", {})[column] = last[column]
                aggregate["headline"][f"mean_{column}"] = last[f"mean_{column}"]
                aggregate["headline"][f"mean_{column}_std"] = last[f"mean_{column}_std"]

    fold_means = merge_fold_means(accuracy_means, time_means)
    if fold_means:
        save_fold_mean_accuracy(
            plan.paths, model_name=str(cfg.get("model_name")), rows=fold_means
        )
    # logger.info(
    #     "CV headline: final GE (mean curve)=%s | traces_to_ge_1 (mean curve)=%s | "
    #     "folds reaching GE<=1: %s | accuracy=%s +/- %s | training time=%s +/- %s s",
    #     headline["mean_curve_final_ge"],
    #     headline["traces_to_ge_1_mean_curve"],
    #     headline["folds_reaching_ge_1"],
    #     headline["accuracy_mean"],
    #     headline["accuracy_std"],
    #     headline["training_time_mean_s"],
    #     headline["training_time_std_s"],
    # )

    save_aggregate_results(
        plan.paths,
        fold_results=fold_results,
        aggregate=aggregate,
        resolved_refit_epochs=resolved_refit_epochs,
        spec=plan.spec,
        diagnostics=diagnostics,
    )
    manifest_patch: Dict[str, Any] = {
        "status": "folds_completed",
        "folds_completed_at": _now_iso(),
        "aggregate": {
            "fold_results": str(plan.paths.fold_results),
            "scalar_statistics": str(plan.paths.scalar_statistics),
            "summary": str(plan.paths.summary),
            "rank_curve": str(plan.paths.aggregate / "mean_rank_curve.json"),
            "diagnostics": diagnostics,
            "fold_rank_curves": str(plan.paths.aggregate / "fold_rank_curves.npz"),
            "headline": aggregate["headline"],
        },
        "refit": _refit_section(plan.spec, resolved_refit_epochs),
    }

    update_manifest(
        plan.paths,
        manifest_patch,
    )
    return {
        "aggregate": aggregate,
        "diagnostics": diagnostics,
        "resolved_refit_epochs": resolved_refit_epochs,
        "headline": aggregate["headline"],
    }


def mark_cross_validation_completed(
    paths: CrossValidationPaths,
    *,
    refit_result: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Mark the complete CV pipeline as successful.

    This must be called only after:
    - every fold completed;
    - fold aggregation completed;
    - the final refit completed successfully, or was skipped by
      cross_validation.refit.enabled: false (refit_result None).
    """
    completed_at = _now_iso()

    manifest = _load_yaml_mapping(
        paths.manifest
    )

    refit_section = manifest.get("refit") or {}

    if not isinstance(refit_section, dict):
        raise TypeError(
            "CV manifest refit section must be a mapping"
        )

    refit_on = refit_section.get("enabled")
    if refit_on is True and refit_result is None:
        raise ValueError("CV completion requires a successful refit result")
    if refit_on is False and refit_result is not None:
        raise ValueError("CV completion got a refit result although cross_validation.refit.enabled is false")
    if refit_on not in (True, False):
        raise ValueError("CV completion requires the refit contract (manifest refit.enabled)")

    patch: Dict[str, Any] = {
        "status": "completed",
        "completed_at": completed_at,
    }

    patch["refit"] = (
        {
            "status": "completed",
            "completed_at": completed_at,
            "result": _to_serializable(refit_result),
        }
        if refit_on
        else {"status": "skipped", "completed_at": completed_at, "reason": REFIT_SKIPPED_REASON}
    )

    updated_manifest = update_manifest(
        paths,
        patch,
    )

    if paths.summary.exists():
        summary = _load_yaml_mapping(
            paths.summary
        )

        summary["status"] = "completed"
        summary["completed_at"] = completed_at

        summary_refit = summary.setdefault(
            "refit",
            {},
        )

        if not isinstance(summary_refit, dict):
            raise TypeError(
                "CV summary refit section must be a mapping"
            )

        summary_refit["status"] = "completed" if refit_on else "skipped"
        summary_refit["completed_at"] = completed_at

        _atomic_write_yaml(
            paths.summary,
            summary,
        )

    return updated_manifest

def mark_cross_validation_failed(
    paths: CrossValidationPaths,
    *,
    error: BaseException | str,
    fold_index: Optional[int] = None,
    phase: Optional[str] = None,
    interrupted: bool = False,
) -> Dict[str, Any]:
    """
    Mark the complete CV run as failed or interrupted, preserving artifacts.

    A refit that had started is marked with the same terminal status; one
    that never began is "blocked".
    """
    status = "interrupted" if interrupted else "failed"
    patch: Dict[str, Any] = {
        "status": status,
        f"{status}_at": _now_iso(),
        "phase": phase,
        "error": repr(error) if isinstance(error, BaseException) else str(error),
        "refit": {
            "status": status if phase == "refit" else "blocked",
        },
    }

    if fold_index is not None:
        patch["failed_fold"] = {
            "fold_index": int(fold_index),
            "fold_number": int(fold_index) + 1,
        }

    return update_manifest(paths, patch)


def fold_manifest_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact manifest entry; the full record lives in fold_result.yaml."""
    def scalars(snapshot: Any) -> Dict[str, float]:
        metrics = snapshot.get("metrics") if isinstance(snapshot, Mapping) else None
        if not isinstance(metrics, Mapping):
            return {}
        return {
            str(key): float(value)
            for key, value in metrics.items()
            if _is_assessment_scalar(key, value)
        }

    return {
        "universal_seed": result.get("universal_seed"),
        "seed": result.get("seed"),
        "selection": result.get("selection"),
        "assessed_epoch": (result.get("final_validation") or {}).get("epoch"),
        "final_train": scalars(result.get("final_train")),
        "final_validation": scalars(result.get("final_validation")),
    }


__all__ = [
    "CrossValidationError",
    "CrossValidationSpec",
    "FoldPlan",
    "CrossValidationPaths",
    "CrossValidationPlan",
    "SUPPORTED_STRATEGIES",
    "validate_cross_validation_config",
    "resolve_cross_validation_paths",
    "get_fold_artifact_paths",
    "fold_artifact_name",
    "FOLD_SEED_RULE",
    "fold_seed",
    "build_fold_runtime_config",
    "build_refit_runtime_config",
    "resolve_cross_validation_runtime",
    "create_fold_plans",
    "save_fold_indices",
    "load_fold_indices",
    "build_initial_manifest",
    "write_manifest",
    "update_manifest",
    "mark_fold_status",
    "prepare_cross_validation_plan",
    "write_fold_result",
    "aggregate_fold_results",
    "resolve_refit_epochs",
    "save_aggregate_results",
    "finalize_cross_validation",
    "mark_cross_validation_completed",
    "mark_cross_validation_failed",
    "fold_manifest_summary",
    "build_cv_headline",
    "partition_sha256",
]
