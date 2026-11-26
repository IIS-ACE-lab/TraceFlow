# services/side_channel_core/utils/cross_validation_pipeline.py
"""
High-Level Cross-Validation Training Pipeline
=============================================

This module owns the complete cross-validation execution path.

The main src/sca/train.py module should only select between:

    if cv_enabled:
        run_cross_validation_pipeline(...)
    else:
        existing standard training pipeline

Responsibilities
----------------
- validate the complete CV configuration;
- verify consistency with the seed resolved by train.py;
- prepare reproducibility artifacts;
- create and validate profiling and reserved test HDF5 splits;
- create deterministic profiling-data fold assignments;
- construct a fresh model, optimizer, scheduler, and DataLoaders per fold;
- execute and persist every fold;
- aggregate successful fold results;
- resolve the final refit epoch count from trainer.epochs;
- refit a fresh model on the complete profiling population, unless
  cross_validation.refit.enabled is false (the CV results only, no model to test);
- refit for the resolved epoch count without a validation loader;
- update CV manifest state;
- release DataLoader, HDF5, CPU, and GPU resources.

Non-responsibilities
--------------------
- the original non-CV training path;
- official test execution;
- ClearML task closure;
- run-level final artifact upload;
- global trial-runtime cleanup.

Those remain owned by train.py and test.py.

Scientific protocol
-------------------
1. CV folds operate only on profiling data.
2. Attack-derived validation and test data are never folded.
3. Every fold receives a fresh deterministic training stack.
4. Fold models are assessed only: they leave no model weights.
5. All folds must succeed before aggregation and refitting.
6. No fold model is promoted as the final model.
7. Final refitting starts from a fresh model.
8. Final refitting uses the complete profiling population.
9. Official testing remains separate from this training pipeline.

Author: Shraddha Mishra
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union


from utils.cross_validation import (
    FOLD_SEED_RULE,
    REFIT_SKIPPED_REASON,
    CrossValidationError,
    CrossValidationPlan,
    FoldPlan,
    build_fold_runtime_config,
    build_refit_runtime_config,
    finalize_cross_validation,
    fold_seed,
    mark_cross_validation_completed,
    mark_cross_validation_failed,
    fold_manifest_summary,
    mark_fold_status,
    prepare_cross_validation_plan,
    update_manifest,
    validate_cross_validation_config,
    write_fold_result,
)
from utils.cross_validation_helpers import (
    build_fresh_training_stack,
    cleanup_training_runtime,
    compact_fold_result,
    prepare_and_validate_data,
)
from utils.data_reproducibility import prepare_artifacts_data
from utils.data_split import (
    get_cross_validation_labels,
    get_cross_validation_loaders,
    get_full_profiling_eval_loader,
    get_full_profiling_loader,
)
from utils.epoch_utils import run_training_loop
from utils.timestamps import now_iso


# =============================================================================
# Generic helpers
# =============================================================================
def _now_iso() -> str:
    """Current time, ISO 8601 in the configured timezone (utils/timestamps.py)."""
    return now_iso()


def _tracker_is_enabled(tracker: Any) -> bool:
    """Return whether the supplied tracking adapter is active."""
    return bool(
        tracker
        and getattr(
            tracker,
            "enabled",
            False,
        )
    )


def _validate_pipeline_inputs(
    cfg: dict,
    *,
    model_name: str,
    env: str,
    base_dir: Union[str, Path],
    seed_used: int,
):
    """
    Validate pipeline-level inputs and return the validated CV specification.
    """
    if not isinstance(cfg, dict):
        raise TypeError(
            f"cfg must be a dict, got {type(cfg).__name__}"
        )

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            "model_name must be a non-empty string"
        )

    configured_model_name = str(
        cfg.get("model_name")
        or ""
    ).strip()

    if configured_model_name and configured_model_name != model_name:
        raise ValueError(
            "CV model-name mismatch: "
            f"argument={model_name!r}, "
            f"cfg.model_name={configured_model_name!r}"
        )

    if not isinstance(env, str) or not env.strip():
        raise ValueError(
            "env must be a non-empty string"
        )

    resolved_base_dir = Path(base_dir)

    if not resolved_base_dir.is_absolute():
        resolved_base_dir = resolved_base_dir.resolve()

    try:
        resolved_seed = int(seed_used)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"seed_used must be an integer, got {seed_used!r}"
        ) from exc

    spec = validate_cross_validation_config(
        cfg
    )

    if not spec.enabled:
        raise RuntimeError(
            "run_cross_validation_pipeline() was called while "
            "cross_validation.enabled=false"
        )

    if resolved_seed != int(spec.seed):
        raise ValueError(
            "CV seed mismatch: "
            f"train.py resolved seed={resolved_seed}, "
            f"CV configuration resolved seed={spec.seed}"
        )

    return spec, resolved_base_dir, resolved_seed


def _connect_tracker(
    tracker: Any,
    *,
    plan: CrossValidationPlan,
    logger,
) -> None:
    """
    Connect CV metadata to ClearML without making tracking a training
    dependency.

    A tracking failure must not invalidate an otherwise valid experiment.
    """
    if not _tracker_is_enabled(tracker):
        return

    try:
        tracker.connect_parameters(
            {
                "pipeline": "cross_validation",
                "enabled": True,
                "n_folds": plan.spec.n_folds,
                "strategy": plan.spec.strategy,
                "shuffle": plan.spec.shuffle,
                "universal_seed": plan.spec.seed,
                "fold_seed_rule": FOLD_SEED_RULE,
                "refit_seed": plan.spec.seed,
                "cross_validation_root": str(
                    plan.paths.root
                ),
            },
            name="cross_validation_pipeline",
        )

    except Exception as exc:
        logger.warning(
            "Failed to connect CV metadata to tracker: %s",
            exc,
            exc_info=True,
        )


# =============================================================================
# One-fold execution
# =============================================================================
def _run_single_fold(
    cfg: dict,
    *,
    plan: CrossValidationPlan,
    fold: FoldPlan,
    logger,
    model_name: str,
    model_id: Optional[str],
    env: str,
    task: Any,
    tracker: Any,
) -> Dict[str, Any]:
    """
    Execute one complete profiling-data fold.

    Every invocation constructs and destroys its own model, optimizer,
    scheduler, and DataLoaders.
    """
    fold_train_loader = None
    fold_validation_loader = None

    fold_model = None
    fold_optimizer = None
    fold_scheduler = None

    expected_seed = fold_seed(plan.spec.seed, fold.fold_index)

    mark_fold_status(
        plan.paths,
        fold_index=fold.fold_index,
        status="running",
        result={
            "universal_seed": plan.spec.seed,
            "fold_seed": expected_seed,
        },
    )

    logger.info(
        "════════════════════════════════════════════════════"
    )
    logger.info(
        "Starting CV fold %d/%d | train=%d held_out=%d | "
        "partition_seed=%d (fixed plan) init/data-order seed=%d",
        fold.fold_number,
        fold.n_folds,
        int(fold.train_indices.size),
        int(fold.validation_indices.size),
        plan.spec.partition,
        expected_seed,
    )
    logger.info(
        "════════════════════════════════════════════════════"
    )

    try:
        fold_cfg = build_fold_runtime_config(
            cfg,
            paths=plan.paths,
            fold_index=fold.fold_index,
            n_folds=fold.n_folds,
        )

        (
            runtime_seed,
            fold_model,
            fold_device,
            fold_optimizer,
            fold_scheduler,
        ) = build_fresh_training_stack(
            fold_cfg,
            model_id=model_id,
            logger=logger,
        )

        if int(runtime_seed) != expected_seed:
            raise RuntimeError(
                f"{fold.tag}: runtime seed mismatch: "
                f"expected={expected_seed}, actual={runtime_seed}"
            )

        (
            fold_train_loader,
            fold_validation_loader,
        ) = get_cross_validation_loaders(
            fold_cfg,
            fold_cfg["trainer"],
            train_indices=fold.train_indices,
            validation_indices=fold.validation_indices,
            seed=runtime_seed,
        )

        fold_runtime = fold_cfg["cross_validation_runtime"]

        # The validation Subset retains the order consumed by its sequential
        # loader. Publish those exact profiling-row indices for attack checks.
        validation_dataset = fold_validation_loader.dataset
        fold_runtime["validation_indices"] = list(validation_dataset.indices)
        fold_runtime["profiling_count"] = len(validation_dataset.dataset)

        training_result = run_training_loop(
            model=fold_model,
            device=fold_device,
            optimizer=fold_optimizer,
            scheduler=fold_scheduler,
            train_loader=fold_train_loader,
            val_loader=fold_validation_loader,
            cfg=fold_cfg,
            logger=logger,
            base_dir=Path(fold_runtime["fold_root"]),
            model_name=model_name,
            env=env,
            task=task,
            tracker=tracker,
        )

        fold_result = compact_fold_result(
            fold=fold,
            seed_used=runtime_seed,
            universal_seed=plan.spec.seed,
            training_result=training_result,
            validation_indices=fold_runtime["validation_indices"],
            profiling_count=fold_runtime["profiling_count"],
        )

        write_fold_result(
            plan.paths,
            fold_index=fold.fold_index,
            n_folds=fold.n_folds,
            result=fold_result,
        )

        mark_fold_status(
            plan.paths,
            fold_index=fold.fold_index,
            status="completed",
            result=fold_manifest_summary(fold_result),
        )

        logger.info(
            "CV fold %d/%d completed | assessed at final epoch %d/%d | seed=%d",
            fold.fold_number,
            fold.n_folds,
            fold_result["final_validation"]["epoch"],
            fold_result["epochs_configured"],
            runtime_seed,
        )

        return fold_result

    except BaseException as exc:
        mark_fold_status(
            plan.paths,
            fold_index=fold.fold_index,
            status=(
                "interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "failed"
            ),
            error=repr(exc),
        )

        logger.error(
            "CV fold %d/%d failed: %s",
            fold.fold_number,
            fold.n_folds,
            exc,
            exc_info=True,
        )

        raise

    finally:
        # Remove model references before clearing the CUDA cache.
        fold_scheduler = None
        fold_optimizer = None
        fold_model = None

        cleanup_training_runtime(
            fold_train_loader,
            fold_validation_loader,
        )


# =============================================================================
# Final full-profiling refit
# =============================================================================
def _run_final_refit(
    cfg: dict,
    *,
    plan: CrossValidationPlan,
    finalized: Dict[str, Any],
    logger,
    model_name: str,
    model_id: Optional[str],
    env: str,
    base_dir: Path,
    task: Any,
    tracker: Any,
) -> Dict[str, Any]:
    """
    Train one fresh final model on the complete profiling population.

    No validation loader is constructed. The final epoch is retained, and
    official testing remains separate.
    """
    resolved_epochs = finalized.get(
        "resolved_refit_epochs"
    )

    if resolved_epochs is None:
        raise RuntimeError(
            "CV aggregation did not resolve final refit epochs"
        )

    try:
        resolved_epochs = int(
            resolved_epochs
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "resolved_refit_epochs must be an integer"
        ) from exc

    if resolved_epochs < 1:
        raise ValueError(
            "resolved_refit_epochs must be >= 1, "
            f"got {resolved_epochs}"
        )

    update_manifest(
        plan.paths,
        {
            "status": "refitting",
            "refit": {
                "status": "running",
                "started_at": _now_iso(),
                "resolved_epochs": resolved_epochs,
            },
        },
    )

    # Final training: a normal training loop over 100% of the profiling traces
    # for trainer.epochs with fresh weights and the top-level seed; the
    # final-epoch weights go to paths.model_out and are tested next.
    logger.info("\n%s", "\n".join(["=" * 80, "FINAL TRAINING", "=" * 80]))

    refit_cfg = build_refit_runtime_config(
        cfg,
        resolved_epochs=resolved_epochs,
    )

    trainer_cfg = refit_cfg["trainer"]

    refit_train_loader = None
    refit_eval_loader = None

    refit_model = None
    refit_optimizer = None
    refit_scheduler = None

    try:
        (
            refit_seed,
            refit_model,
            refit_device,
            refit_optimizer,
            refit_scheduler,
        ) = build_fresh_training_stack(
            refit_cfg,
            model_id=model_id,
            logger=logger,
        )

        if int(refit_seed) != int(plan.spec.seed):
            raise RuntimeError(
                "Final refit runtime seed mismatch: "
                f"expected={plan.spec.seed}, actual={refit_seed}"
            )

        refit_train_loader = get_full_profiling_loader(
            refit_cfg,
            trainer_cfg,
            seed=refit_seed,
        )
        # Eval-mode accuracy and loss on the training traces only. The refit
        # never attacks its own training traces: rank metrics come from
        # held-out data (the CV folds, written as process 'validation', and
        # the test).
        refit_eval_loader = get_full_profiling_eval_loader(
            refit_cfg,
            trainer_cfg,
            seed=refit_seed,
        )

        refit_result = run_training_loop(
            model=refit_model,
            device=refit_device,
            optimizer=refit_optimizer,
            scheduler=refit_scheduler,
            train_loader=refit_train_loader,
            val_loader=refit_eval_loader,
            cfg=refit_cfg,
            logger=logger,
            base_dir=base_dir,
            model_name=model_name,
            env=env,
            task=task,
            tracker=tracker,
        )

        # The CV results are the aggregate under paths.cross_validation
        # (summary.yaml, mean_rank_curve.json, the diagnostics report) and the
        # fold metrics; model_judge reads them from there.
        mark_cross_validation_completed(
            plan.paths,
            refit_result=refit_result,
        )
        return refit_result

    finally:
        refit_scheduler = None
        refit_optimizer = None
        refit_model = None

        cleanup_training_runtime(
            refit_train_loader,
            refit_eval_loader,
        )


# =============================================================================
# Complete CV pipeline
# =============================================================================
def run_cross_validation_pipeline(
    cfg: dict,
    *,
    logger,
    model_name: str,
    model_id: Optional[str],
    env: str,
    base_dir: Union[str, Path],
    seed_used: int,
    task: Any = None,
    tracker: Any = None,
) -> Dict[str, Any]:
    """
    Execute the complete cross-validation training pipeline.

    Parameters
    ----------
    cfg:
        Fully resolved runtime configuration.

    logger:
        Active run logger created by train.py.

    model_name:
        Final resolved model/run name.

    model_id:
        Optional dynamic model identifier.

    env:
        Active runtime environment.

    base_dir:
        Normal run artifact root. The final refitted model is written under
        the ordinary configured model paths.

    seed_used:
        Global seed already resolved by train.py.

    task, tracker:
        Existing ClearML objects. This module uses but does not close them.

    Returns
    -------
    dict
        data_artifacts:
            Validated split-artifact metadata.

        cross_validation:
            Fold aggregate and resolved refit metadata.

        fold_results:
            Compact result from every fold.

        refit:
            Final full-profiling refit result; None when
            cross_validation.refit.enabled is false.
    """
    (
        spec,
        resolved_base_dir,
        resolved_seed,
    ) = _validate_pipeline_inputs(
        cfg,
        model_name=model_name,
        env=env,
        base_dir=base_dir,
        seed_used=seed_used,
    )

    plan: Optional[CrossValidationPlan] = None
    active_fold_index: Optional[int] = None
    execution_phase = "initialization"

    logger.info(
        "Entering dedicated CV pipeline | "
        "folds=%d strategy=%s shuffle=%s universal_seed=%d "
        "fold_seed_rule='%s' refit=%s",
        spec.n_folds,
        spec.strategy,
        spec.shuffle,
        spec.seed,
        FOLD_SEED_RULE,
        "on" if spec.refit else "off (cross_validation.refit.enabled: false)",
    )

    try:
        # ------------------------------------------------------------------
        # Reproducibility and profiling/reserved test artifacts
        # ------------------------------------------------------------------
        execution_phase = "data_preparation"

        prepare_artifacts_data(
            cfg
        )

        data_artifacts = prepare_and_validate_data(
            cfg,
            model_name=model_name,
            seed=resolved_seed,
            logger=logger,
        )

        # ------------------------------------------------------------------
        # Deterministic fold plan
        # ------------------------------------------------------------------
        execution_phase = "plan_creation"

        profiling_labels = get_cross_validation_labels(
            cfg
        )

        plan = prepare_cross_validation_plan(
            cfg,
            labels=profiling_labels,
        )

        update_manifest(
            plan.paths,
            {
                "status": "running",
                "started_at": _now_iso(),
                "data_artifacts": data_artifacts,
                "seed_contract": {
                    "universal_seed": spec.seed,
                    "fold_seed_rule": FOLD_SEED_RULE,
                    "refit_seed": spec.seed,
                },
            },
        )

        _connect_tracker(
            tracker,
            plan=plan,
            logger=logger,
        )

        # ------------------------------------------------------------------
        # Fold training
        # ------------------------------------------------------------------
        execution_phase = "fold_training"

        fold_results = []

        for fold in plan.folds:
            active_fold_index = int(
                fold.fold_index
            )

            fold_result = _run_single_fold(
                cfg,
                plan=plan,
                fold=fold,
                logger=logger,
                model_name=model_name,
                model_id=model_id,
                env=env,
                task=task,
                tracker=tracker,
            )

            fold_results.append(
                fold_result
            )

        active_fold_index = None

        # ------------------------------------------------------------------
        # Fold aggregation and refit-epoch resolution
        # ------------------------------------------------------------------
        execution_phase = "aggregation"

        finalized = finalize_cross_validation(
            plan,
            cfg=cfg,
            fold_results=fold_results,
        )

        # ------------------------------------------------------------------
        # Final full-profiling refit (cross_validation.refit.enabled, default true)
        # ------------------------------------------------------------------
        execution_phase = "refit"

        if plan.spec.refit:
            refit_result = _run_final_refit(
                cfg,
                plan=plan,
                finalized=finalized,
                logger=logger,
                model_name=model_name,
                model_id=model_id,
                env=env,
                base_dir=resolved_base_dir,
                task=task,
                tracker=tracker,
            )
        else:
            # The CV results are complete; no model is trained for a test (the plan refused an
            # attacked test with the refit off).
            logger.info("Final training skipped: %s", REFIT_SKIPPED_REASON)
            mark_cross_validation_completed(plan.paths, refit_result=None)
            refit_result = None

        logger.info(
            "Dedicated CV pipeline completed successfully"
        )

        return {
            "data_artifacts": data_artifacts,
            "cross_validation": finalized,
            "fold_results": fold_results,
            "refit": refit_result,
        }

    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        location = execution_phase
        if active_fold_index is not None:
            location += f" (fold {active_fold_index + 1}/{spec.n_folds})"

        if plan is not None:
            try:
                mark_cross_validation_failed(
                    plan.paths,
                    error=exc,
                    fold_index=active_fold_index,
                    phase=execution_phase,
                    interrupted=interrupted,
                )
            except Exception as manifest_exc:
                logger.error(
                    "Failed to persist CV failure state: %s",
                    manifest_exc,
                    exc_info=True,
                )

        if interrupted:
            logger.warning("CV pipeline interrupted during %s", location)
            raise

        logger.error(
            "Dedicated CV pipeline failed during %s: %s",
            location,
            exc,
            exc_info=True,
        )
        if isinstance(exc, Exception):
            # The lifecycle layer records str(exc) as the FAILED reason.
            raise CrossValidationError(
                f"cross-validation failed during {location}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        raise


__all__ = [
    "run_cross_validation_pipeline",
]
