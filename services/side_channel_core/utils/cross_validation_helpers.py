# services/side_channel_core/utils/cross_validation_helpers.py
"""Shared runtime helpers for standard training and SCA cross-validation.

CV folds and the mandatory refit receive independent model, optimizer, and
scheduler instances. The caller supplies the runtime seed: the universal seed
for the final refit and, for the folds, the next seeds in fold order
(cross_validation.FOLD_SEED_RULE).

This module prepares data but never evaluates ``test.h5``. In CV mode it omits
``val.h5`` because fold validation comes from complementary profiling indices.

Author: Shraddha Mishra
"""

from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.model import build_model
from utils.data_split import (
    _cv_enabled,
    _get_attack_byte_from_cfg,
    get_train_val_loaders,
    perform_and_save_data_split,
)
from utils.cross_validation import FOLD_SEED_RULE, fold_seed, rank_curve_pairs
from utils.device import initialize_device_and_model
from utils.label_checks import check_label_integrity, infer_label_model
from src.sca.dataset_types import label_target
from core.framework_switch import build_optimizer
from utils.path_utils import resolve_path
from utils.seed import set_seed


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    cfg: dict,
    logger,
) -> tuple[torch.optim.Optimizer, Any | None]:
    """Build a fresh optimizer and nested scheduler for one model."""
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    optimizer, scheduler, metadata = build_optimizer(cfg, model)
    # logger.info("Fresh training stack | optimizer=%s", metadata)
    return optimizer, scheduler


def build_fresh_training_stack(
    cfg: dict,
    *,
    model_id: str | None,
    logger,
):
    """Seed first, then construct an independent model and optimizer stack."""
    seed_used = set_seed(cfg)
    runtime = cfg.get("cross_validation_runtime")
    if isinstance(runtime, Mapping) and runtime.get("seed") is not None:
        expected_seed = int(runtime["seed"])
        if seed_used != expected_seed:
            raise ValueError(
                "cfg.seed conflicts with cross_validation_runtime.seed: "
                f"{seed_used} != {expected_seed}"
            )

    model, device = initialize_device_and_model(
        cfg,
        build_model,
        logger,
        model_id=model_id,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, logger)
    return seed_used, model, device, optimizer, scheduler


def close_dataset_chain(dataset: Any) -> None:
    """Close a dataset and any wrapped base datasets once each."""
    seen: set[int] = set()
    while dataset is not None and id(dataset) not in seen:
        seen.add(id(dataset))
        close = getattr(dataset, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        dataset = getattr(dataset, "dataset", None)


def close_loader(loader: Any) -> None:
    """Shut down DataLoader workers and close its dataset chain."""
    if loader is None:
        return

    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass
    try:
        loader._iterator = None
    except Exception:
        pass
    close_dataset_chain(getattr(loader, "dataset", None))


def cleanup_training_runtime(*loaders: Any) -> None:
    """Release workers, HDF5 handles, CUDA cache, and unreachable objects."""
    for loader in loaders:
        close_loader(loader)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _require_file(name: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} was not created at {path}")


def _log_inferred_label_model(path: Path, name: str, logger, leakage_model: str) -> None:
    # Diagnostic only; check_label_integrity() below is the hard gate.
    try:
        guess = infer_label_model(str(path), leakage_model=leakage_model, n=500)
        logger.info("[LabelModel] %s (sca.leakage_model=%s): %s", name, leakage_model, guess)
    except Exception as exc:
        logger.warning("infer_label_model failed on %s: %s", name, exc)


def prepare_and_validate_data(
    cfg: dict,
    *,
    model_name: str,
    seed: int,
    logger,
) -> dict[str, Any]:
    """Prepare split artifacts without performing model evaluation.

    CV creates ``train.h5`` from profiling traces and reserves ``test.h5``.
    Standard training additionally creates ``val.h5`` from validation_range.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    trainer_cfg = cfg.get("trainer")
    paths_cfg = cfg.get("paths")
    if not isinstance(trainer_cfg, dict):
        raise TypeError("cfg['trainer'] must be a mapping")
    if not isinstance(paths_cfg, dict):
        raise TypeError("cfg['paths'] must be a mapping")

    train_loader = validation_loader = None
    cv_enabled = _cv_enabled(cfg)
    try:
        if cv_enabled:
            perform_and_save_data_split(cfg, model_name=model_name)
        else:
            train_loader, validation_loader = get_train_val_loaders(
                cfg,
                trainer_cfg,
                seed=int(seed),
            )

        data_dir = Path(resolve_path(paths_cfg["data"], cfg, model_name=model_name))
        copied_dataset = Path(
            resolve_path(paths_cfg["data_out"], cfg, model_name=model_name)
        )
        train_h5 = data_dir / "train.h5"
        validation_h5 = None if cv_enabled else data_dir / "val.h5"
        test_h5 = data_dir / "test.h5"

        _require_file("copied dataset", copied_dataset)
        _require_file("train.h5", train_h5)
        _require_file("test.h5", test_h5)
        if validation_h5 is not None:
            _require_file("val.h5", validation_h5)

        attack_byte = _get_attack_byte_from_cfg(cfg)
        leakage_model = str((cfg.get("sca") or {}).get("leakage_model", "ID"))
        _log_inferred_label_model(copied_dataset, "copied dataset", logger, leakage_model)
        _log_inferred_label_model(train_h5, "train.h5", logger, leakage_model)

        checks = [(train_h5, "Profiling_traces")]
        if validation_h5 is not None:
            checks.append((validation_h5, "Attack_traces"))
        checks.append((test_h5, "Attack_traces"))
        for path, group in checks:
            check_label_integrity(
                str(path),
                attack_byte=attack_byte,
                leakage_model=leakage_model,
                n=10,
                target=label_target(cfg),
                group=group,
            )

        # logger.info(
        #     "SCA data gates passed | attack_byte=%d leakage_model=%s cv=%s",
        #     attack_byte,
        #     leakage_model,
        #     cv_enabled,
        # )
        return {
            "model_name": model_name,
            "seed": int(seed),
            "attack_byte": int(attack_byte),
            "leakage_model": leakage_model,
            "data_dir": str(data_dir),
            "copied_dataset": str(copied_dataset),
            "train_h5": str(train_h5),
            "validation_h5": str(validation_h5) if validation_h5 else None,
            "test_h5": str(test_h5),
        }
    finally:
        cleanup_training_runtime(train_loader, validation_loader)


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{field} must be >= 1")
    return parsed


def _final_snapshot(training_result: Mapping[str, Any], key: str, epochs: int) -> dict:
    snapshot = training_result.get(key)
    if not isinstance(snapshot, Mapping):
        raise KeyError(f"training_result.{key} is required")
    epoch = _positive_int(snapshot.get("epoch"), f"training_result.{key}.epoch")
    if epoch != epochs:
        raise RuntimeError(
            f"Fold {key} must come from the final configured epoch: "
            f"epoch={epoch}, configured={epochs}"
        )
    metrics = snapshot.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise TypeError(f"training_result.{key}.metrics must be a non-empty mapping")
    return {"epoch": epoch, "metrics": dict(metrics)}


def compact_fold_result(
    *,
    fold,
    seed_used: int,
    universal_seed: int,
    training_result: Mapping[str, Any],
    validation_indices: Any,
    profiling_count: int,
) -> dict[str, Any]:
    """Build the authoritative assessment record for one fixed-epoch fold.

    The record carries final-epoch training and validation assessments, the
    per-epoch trajectory, and the fold seed. It has no best-epoch semantics
    and, because fold assessment is weightless, no model/checkpoint fields.
    """
    if not isinstance(training_result, Mapping):
        raise TypeError("training_result must be a mapping")

    execution = training_result.get("execution") or {}
    if execution.get("type") != "fold" or execution.get("fold_index") != fold.fold_index:
        raise RuntimeError(
            f"{fold.tag}: training result belongs to execution {execution!r}"
        )
    if training_result.get("selection") != "final_configured_epoch":
        raise RuntimeError(f"{fold.tag}: fold result must use final-epoch assessment")
    if (training_result.get("checkpointing") or {}).get("model_persisted") is not False:
        raise RuntimeError(f"{fold.tag}: CV fold assessment must not persist model weights")

    seed_used = int(seed_used)
    universal_seed = int(universal_seed)
    if seed_used != fold_seed(universal_seed, fold.fold_index):
        raise RuntimeError(
            f"{fold.tag}: fold seed {seed_used} != "
            f"{fold_seed(universal_seed, fold.fold_index)} ({FOLD_SEED_RULE}, "
            f"seed={universal_seed})"
        )

    configured = _positive_int(
        training_result.get("epochs_configured"),
        "training_result.epochs_configured",
    )
    completed = _positive_int(
        training_result.get("epochs_completed"),
        "training_result.epochs_completed",
    )
    if completed != configured:
        raise RuntimeError(
            "CV folds must complete the configured fixed epoch count: "
            f"completed={completed}, configured={configured}"
        )

    final_train = _final_snapshot(training_result, "final_train", configured)
    final_validation = _final_snapshot(training_result, "final_validation", configured)

    curve = final_validation["metrics"].get("mean_attack_rank_curve")
    if curve is not None:
        # Persisted YAML/JSON reduces ndarrays to shape descriptors; the
        # aggregate rank curve needs the actual values.
        final_validation["metrics"]["mean_attack_rank_curve"] = rank_curve_pairs(curve)

    indices = np.asarray(validation_indices, dtype=np.int64)
    if not np.array_equal(indices, fold.validation_indices):
        raise RuntimeError(f"{fold.tag}: validation indices differ from the saved fold plan")
    final_validation.update({
        "population_source": "Profiling_traces",
        "validation_indices": indices.tolist(),
        "profiling_count": int(profiling_count),
    })

    trajectory = training_result.get("trajectory")
    if not isinstance(trajectory, Mapping):
        raise KeyError("training_result.trajectory is required")
    trajectory = {key: list(values) for key, values in trajectory.items()}
    if trajectory.get("epochs") != list(range(1, configured + 1)):
        raise RuntimeError(
            f"{fold.tag}: trajectory must cover epochs 1..{configured} contiguously"
        )

    return {
        "fold_index": int(fold.fold_index),
        "fold_number": int(fold.fold_number),
        "n_folds": int(fold.n_folds),
        "universal_seed": universal_seed,
        "seed": seed_used,
        "selection": "final_configured_epoch",
        "epochs_configured": configured,
        "epochs_completed": completed,
        "training_time_s": float(training_result.get("training_time_s", float("nan"))),
        "train_samples": int(fold.train_indices.size),
        "validation_samples": int(fold.validation_indices.size),
        "final_train": final_train,
        "final_validation": final_validation,
        "trajectory": trajectory,
    }


__all__ = [
    "build_optimizer_and_scheduler",
    "build_fresh_training_stack",
    "close_dataset_chain",
    "close_loader",
    "cleanup_training_runtime",
    "prepare_and_validate_data",
    "compact_fold_result",
]
