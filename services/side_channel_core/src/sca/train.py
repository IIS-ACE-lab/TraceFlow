#  services/side_channel_core/src/sca/train.py
"""

train.py – Side-Channel Analysis Model Trainer
==============================================

Introduction
------------
This module provides a fully-featured, reproducible training pipeline for
Side-Channel Analysis (SCA) models.  It supports both command-line (CLI)
execution and ClearML-instrumented (“UI”) runs, seamlessly merging your
master.yaml, params.yaml, and any extra configs to drive PyTorch training
loops, checkpointing, metric logging, and artifact management.

Module Functionality
--------------------
- Parses and coerces hyperparameters from YAML (learning rate, weight decay, scheduler, etc.).
- Generates a deterministic “funny name” for each run (model_name + hash).
- Creates all artifact directories:
    artifacts/generated_models/{model_name}/[logs, graphs, model, checkpoints, data, metrics]
- Writes the run’s resolved config to config/{model_name}_params.yaml (paths.params_out)
  and builds its derived dataset (paths.data_out + train/val/test.h5) in data/;
  neither the raw dataset nor the live params.yaml is copied.
- Seeds all RNGs (Python, NumPy, PyTorch) for determinism.
- Builds model, optimizer, and optional scheduler via your config.
- Loads deterministic train, validation, and test HDF5 artifacts generated from
the configured profiling_range, validation_range, and test_range.
- Loads training and validation DataLoaders from HDF5 or index splits.
- Runs epoch loops:
    • Forward/backward on training batches
    • Optional learning_rate stepping
    • Memory cleanup (gc + torch.cuda.empty_cache)
    • Validation pass to compute Guessing Entropy, Accuracy, SR@k, etc.
    • Live CSV logging of per-epoch metrics for external monitoring
    • Early stopping and best-metric checkpointing
- After training:
    • Records the model in artifacts/done_model_names.yaml
    • Closes ClearML Task (if used)

Features
--------
• **YAML-driven**: master.yaml + params.yaml → unified `cfg` dict  
• **ClearML integration**: real-time UI dashboards, artifact upload, config tracking  
• **Reproducible**: seeds, deterministic run names, resolved config snapshot, derived data  
• **Robust logging**: structured file + console logs, exception catching  
• **Live Monitoring**: epoch-by-epoch CSV logs with fsync for tail-follower scripts  
• **Benchmarking**: central registry of all runs for easy comparison

Gotchas & Requirements
----------------------
- Your project must have:
    - `<project_root>/config/master.yaml`
    - `<project_root>/config/params.yaml`
    - A valid dataset path in `params.yaml` (HDF5 or index ranges)
- Early stopping patience, monitor_metric, and scheduler args must be set in your trainer config.
- The `paths` block in your config must define Jinja2 templates for all artifact locations.
- Run names must be unique; collisions will overwrite existing artifacts.

Usage
-----
From your repository root:
```bash
python services/side_channel_core/src/sca/train.py \
    --project DLSCA_ASCAD \
    --dataset ascad \

#TODO: Use Cross-Device / Cross-Key Testing:

    Train on traces from device A, test on B.

    This gives more meaningful insights into model generalization — a core goal of your system

Author: Shraddha Mishra
"""



from pathlib import Path
import argparse
from services.side_channel_core.utils.graph_config_sync import append_model_name
from core.model import build_model
from services.side_channel_core.src.sca.leakage_models import resolve_runtime_leakage_model
from utils.error_handler import catch_exceptions
from utils.config_loader import get_project_config
from utils.data_reproducibility import prepare_artifacts_data
from utils.data_split import (
    _get_attack_byte_from_cfg,
    get_train_val_loaders,
)
from utils.logger import setup_logging,  get_logger, pretty_config
from utils.sweep_engine import get_pending_runs, update_status, determine_run_identity, is_done
from services.visualization.clearml.clearml_tracker import build_tracker_from_run_context
from utils.epoch_utils import run_training_loop
from utils.model_import import is_imported as is_imported_run
from core.framework_switch import build_optimizer
from utils.type_coerce import coerce_types
from utils.label_checks import check_label_integrity
from services.side_channel_core.src.sca.dataset_types import label_target
from utils.device import initialize_device_and_model, require_device
from utils.path_utils import resolve_path
from utils.artifact_utils import prepare_run_dir
from utils.seed import set_seed
from utils.precision import apply_precision
from utils.device_record import write_device_record
from utils.path_utils import resolve_paths_dict
from services.mlops.utils.zombie_reaper import cleanup_trial_runtime
from services.visualization.utils.ensure import _ensure_visualization_cfg
from core.model_contract_resolver import resolve_model_contract
from services.side_channel_core.utils.resolver_mode import decide_resolver_action
from services.side_channel_core.utils.model_preview import (
    save_resolved_model_preview,
    print_resolved_model_preview,
)
logger = get_logger("Train")



@catch_exceptions

def _register_model_for_graphs(cfg: dict, model_name: str, logger) -> None:
    """Record a generated model in artifacts/done_model_names.yaml."""
    project_root = (cfg.get("project") or {}).get("root")
    if not project_root:
        return
    try:
        append_model_name(project_root, model_name)
    except Exception as exc:
        # Bookkeeping must never fail a finished training run.
        logger.warning("artifacts/done_model_names.yaml not updated: %s", exc)

def train(cfg: dict, exec_process: str = "train", override_model_name: str = None, model_id: str = None,):
    # grab our logger for this function
    # 1) Set up root logger for this run
    #configure_global_logging(cfg)
    logger = get_logger(__name__)
    # 2) Get training-specific logger that logs to its own file too
    
    task = None
    tracker = None
    try:


        # after loading or selecting your cfg Coerce trainer, model, test hyperparams 
        coerce_types(cfg)
        #Make the space of the metrics to be logged
        if "metrics" not in cfg and "metrics_to_log" in cfg:
            cfg["metrics"] = cfg["metrics_to_log"]
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # B) Choose a reproducible run name
        # ── Determine run identity: name and hash ───────────────────────────────────
        model_name, model_hash = determine_run_identity(cfg, override_model_name)
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # CRITICAL: inject identity BEFORE any resolve_path() / paths_abs creation
        cfg["model_name"] = model_name
        cfg["model_hash"] = model_hash

        # if you use cfg["run"]["model_name"] elsewhere
        cfg.setdefault("run", {})
        if isinstance(cfg["run"], dict):
            cfg["run"]["model_name"] = model_name

        cfg["paths_abs"] = resolve_paths_dict(cfg["paths"], cfg, inject=True, mkdir=False)
        project_name = cfg["project"]["name"]
        project_root = Path(cfg["project"]["root"])  # use project metadata

        base_dir = Path(resolve_path(cfg["paths"]["artifacts_root"], cfg, inject=True))

            # C) Create artifact directories
        prepare_run_dir(cfg=cfg,
                        project_root=project_root,
                        model_hash = model_hash,
                        model_name = model_name,
                    )
        # # C.1) dump params.yaml into run_dir/config/params.yaml
        # params_file = cfg["paths"]["params_out"]
        # with open(base_dir/"config"/f"{model_name}_params.yaml", "w") as f:
        #     yaml.safe_dump(convert_paths_to_strings(cfg), f)
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # D) Local logging first
        logger, task, safe_cfg = setup_logging(cfg=cfg,project_name=project_name,model_name=model_name,base_dir=base_dir,process_name="Training")
        # D.1) Central ClearML tracker

        cfg = _ensure_visualization_cfg(cfg, project_name)
        #logger.info("ClearML debug: visualization present=%s", bool(cfg.get("visualization")))

        tracker = None
        try:
            tracker = build_tracker_from_run_context(
                cfg=cfg,
                process_name="train",
                model_name=model_name,
                model_hash=cfg.get("model_hash"),
                extra_tags=[
                    cfg.get("env", "dev"),
                    str(project_name),
                    exec_process,
                ],
                task_type="training",
            )
            logger.info("ClearML tracker enabled=%s failure_reason=%s",
                getattr(tracker, "enabled", False),
                getattr(tracker, "disabled_reason", None))
        except Exception as e:
            logger.warning("ClearML tracker init failed: %s", e)
            tracker = None

        # Keep backward compatibility with existing epoch loop code that expects `task`
        if tracker and tracker.enabled:
            task = tracker.task

            tracker.connect_config(cfg)
            tracker.connect_parameters(
            {
                "group_id": model_name,
                "group_tag": f"group:{model_name}",
                "task_name": f"{model_name}_train",
                "model_name": model_name,
                "model_hash": model_hash,
                "project_name": project_name,
                "process_name": "train",
                "env": cfg.get("env", "dev"),
                "artifacts_root": str(base_dir),
                },
                name="run_context",
            )

            tracker.set_comment(
                f"SCA train run grouped under {model_name} for project={project_name}"
            )

        # Pretty-print config with color for terminal output
        pretty_config(
            safe_cfg=safe_cfg,
            model_name=model_name,
            project_name=project_name,
            process_name="Training",
            model_hash=model_hash,
        )
        # ────────────────────────────────────────────────────────────────────────────────────────────────

        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # F) Seed all RNGs
        seed_used = set_seed(cfg)
        apply_precision(cfg)   # trainer.precision: fp32 | tf32 | default
        # The requested GPU must be usable before anything is written: never a CPU fallback
        # (utils/device.py; the run fails with status_reason "GPU FAILED: ...").
        require_device(cfg)
        # device.yaml in paths.device (default the run folder): the GPU, library
        # versions, precision switches and a timing probe (graph job relative:).
        # The probe restores the RNG state.
        try:
            write_device_record(cfg, logger=logger)
        except Exception as e:
            logger.warning("Device record not written: %s", e, exc_info=True)
        #logger.info("Using seed=%d", seed_used)
        if tracker and tracker.enabled:
            tracker.connect_parameters(
                {
                    "seed_used": seed_used,
                },
                name="runtime_seed",
            )

        # ─────────────────────────────────────────────────────────────────────────────
        # Resolve dynamic model contract before build_model()
        # ─────────────────────────────────────────────────────────────────────────────

        # Auto-search / bridge path may inject model_id into cfg instead of function arg.
        if model_id is None:
            cfg_model_id = cfg.get("model_id")
            if isinstance(cfg_model_id, str) and cfg_model_id.strip():
                model_id = cfg_model_id.strip()

        resolver_decision = decide_resolver_action(cfg, model_id, logger)

        env = cfg.get("env", "dev")

        if resolver_decision.should_resolve:
            resolved_cfg = resolve_model_contract(
                cfg,
                mode=resolver_decision.mode,
                model_id=model_id,
                sampled=None,
                trial=None,
            )
        else:
            resolved_cfg = cfg

        # Important:
        # From this point onward, training must use the resolved config.
        # This ensures trainer.optimizer, model_id, and architecture_args.layers
        # match the model that was saved into resolved_model.yaml.
        cfg = resolved_cfg
        # Reassert run identity after model-contract resolution.
        # The resolver may return a copied runtime configuration.
        cfg["model_name"] = model_name
        cfg["model_hash"] = model_hash

        cfg.setdefault("run", {})

        if not isinstance(cfg["run"], dict):
            raise TypeError("cfg['run'] must be a mapping")

        cfg["run"]["model_name"] = model_name

        # Refresh resolved paths against the final runtime configuration.
        cfg["paths_abs"] = resolve_paths_dict(
            cfg["paths"],
            cfg,
            inject=True,
            mkdir=False,
        )
        resolve_runtime_leakage_model(cfg)
        #logger.info("Resolved runtime leakage_model=%s", runtime_lm)
        # Save the concrete resolved model BEFORE build_model().
        # This records the exact dynamic architecture produced by resolve_model_contract().
        try:
            print_resolved_model_preview(cfg, logger=logger)
            save_resolved_model_preview(
                cfg,
                env=cfg.get("env", env),
                model_name=model_name,
                logger=logger,
                save_json=True,
            )
        except Exception as e:
            logger.warning("Failed to save resolved_model preview: %s", e, exc_info=True)

        # =====================================================================
        # Select the training execution pipeline
        # =====================================================================
        raw_cv_cfg = cfg.get("cross_validation")

        if raw_cv_cfg is None:
            cv_cfg = {}
        elif not isinstance(raw_cv_cfg, dict):
            raise TypeError(
                "cfg['cross_validation'] must be a mapping, "
                f"got {type(raw_cv_cfg).__name__}"
            )
        else:
            cv_cfg = raw_cv_cfg

        cv_enabled = cv_cfg.get(
            "enabled",
            False,
        )

        if not isinstance(cv_enabled, bool):
            raise TypeError(
                "cross_validation.enabled must be true or false, "
                f"got {cv_enabled!r}"
            )

        training_result = None

        # =====================================================================
        # CROSS-VALIDATION PIPELINE
        # =====================================================================
        # CV: fixed-epoch folds on profiling data only, then the mandatory
        # full-profiling refit, which publishes the single official test model.
        # Attack-derived validation/test data is never touched here.
        if cv_enabled:
            # Lazy import means the CV pipeline and sklearn dependencies are
            # loaded only when cross-validation is actually enabled.
            from utils.cross_validation_pipeline import (
                run_cross_validation_pipeline,
            )

            # logger.info(
            #     "Cross-validation enabled: invoking the dedicated "
            #     "cross-validation pipeline"
            # )

            training_result = run_cross_validation_pipeline(
                cfg,
                logger=logger,
                model_name=model_name,
                model_id=model_id,
                env=env,
                base_dir=base_dir,
                seed_used=seed_used,
                task=task,
                tracker=tracker,
            )

        # =====================================================================
        # MODEL IMPORT: a released model, not trained (utils/model_import.py)
        # =====================================================================
        # The snapshot's model_import record (make import): TraceFlow builds the
        # converted model, the released weights are copied into it and checked
        # against the released model (the equality gate), and the checkpoint is
        # published as a trained one is; the standard test follows.
        elif is_imported_run(cfg):
            from utils.model_import_weights import import_released_weights

            model, device = initialize_device_and_model(cfg, build_model, logger, model_id=model_id)
            prepare_artifacts_data(cfg)
            training_result = import_released_weights(cfg, model, device, logger=logger, seed=seed_used)

        # =====================================================================
        # ORIGINAL STANDARD TRAINING PIPELINE
        # =====================================================================
        else:
            #logger.info("Cross-validation disabled: using the original " "standard training pipeline")

            # -------------------------------------------------------------
            # Initialize device and model using the resolved configuration.
            # -------------------------------------------------------------
            model, device = initialize_device_and_model(
                cfg,
                build_model,
                logger,
                model_id=model_id,
            )

            # -------------------------------------------------------------
            # Write the params_out snapshot and build/verify the derived dataset (paths.data_out).
            # -------------------------------------------------------------
            prepare_artifacts_data(cfg)

            if tracker and tracker.enabled:
                tracker.connect_parameters(
                    {
                        "device": str(device),
                        "model_id": model_id,
                    },
                    name="runtime_device",
                )

            # -------------------------------------------------------------
            # Construct optimizer and its nested scheduler.
            # -------------------------------------------------------------
            try:
                # torch.optim or keras.optimizers, by deep_learning.framework.
                optimizer, scheduler, resolved_optimizer_cfg = build_optimizer(cfg, model)
            except Exception as e:
                logger.error(
                    "Model/optimizer setup failed: %s",
                    e,
                    exc_info=True,
                )
                raise

            if tracker and tracker.enabled:
                tracker.connect_parameters(
                    {"optimizer": resolved_optimizer_cfg},
                    name="optimizer_config",
                )

            # -------------------------------------------------------------
            # Create standard train and validation loaders.
            # -------------------------------------------------------------
            try:
                train_loader, val_loader = get_train_val_loaders(
                    cfg,
                    cfg["trainer"],
                    seed=seed_used,
                )

                data_dir = Path(
                    resolve_path(
                        cfg["paths"]["data"],
                        cfg,
                        model_name=model_name,
                    )
                )

                train_h5 = data_dir / "train.h5"
                val_h5 = data_dir / "val.h5"
                test_h5 = data_dir / "test.h5"

                copied_dataset_path = Path(
                    resolve_path(
                        cfg["paths"]["data_out"],
                        cfg,
                        model_name=model_name,
                    )
                )

                if not copied_dataset_path.exists():
                    raise FileNotFoundError(
                        "Copied dataset missing at "
                        f"{copied_dataset_path} "
                        "(prepare_artifacts_data should create it)"
                    )

                if not train_h5.exists():
                    raise FileNotFoundError(
                        f"train.h5 not found at {train_h5} "
                        "(split generation should have created it)"
                    )

                if not val_h5.exists():
                    raise FileNotFoundError(
                        f"val.h5 not found at {val_h5} "
                        "(split generation should have created it)"
                    )

                if not test_h5.exists():
                    raise FileNotFoundError(
                        f"test.h5 not found at {test_h5} "
                        "(split generation should have created it)"
                    )

                cfg_attack_byte = _get_attack_byte_from_cfg(
                    cfg
                )

                # try:
                #     logger.info(
                #         "[LabelModel] copied dataset: %s",
                #         infer_label_model(
                #             str(copied_dataset_path),
                #             leakage_model=(cfg.get("sca") or {}).get("leakage_model", "ID"),
                #             n=500,
                #         ),
                #     )
                # except Exception as e:
                #     logger.warning(
                #         "infer_label_model failed on copied dataset: %s",
                #         e,
                #     )

                leakage_model = (
                    (cfg.get("sca") or {}).get(
                        "leakage_model",
                        "ID",
                    )
                )

                # logger.info(
                #     "[LabelModel] train.h5: %s",
                #     infer_label_model(
                #         str(train_h5),
                #         leakage_model=leakage_model,
                #         n=500,
                #     ),
                # )

                check_label_integrity(
                    str(train_h5),
                    attack_byte=cfg_attack_byte,
                    leakage_model=leakage_model,
                    n=10,
                    target=label_target(cfg),
                    group="Profiling_traces",
                )

                check_label_integrity(
                    str(val_h5),
                    attack_byte=cfg_attack_byte,
                    leakage_model=leakage_model,
                    n=10,
                    target=label_target(cfg),
                    group="Attack_traces",
                )

                check_label_integrity(
                    str(test_h5),
                    attack_byte=cfg_attack_byte,
                    leakage_model=leakage_model,
                    n=10,
                    target=label_target(cfg),
                    group="Attack_traces",
                )

                # logger.info(
                #     "SCA gates passed: "
                #     "attack_byte=%d and labels verified",
                #     cfg_attack_byte,
                # )

            except Exception as e:
                logger.error(
                    "Data loading failed: %s",
                    e,
                    exc_info=True,
                )
                raise

            # -------------------------------------------------------------
            # Run the original standard epoch loop.
            # -------------------------------------------------------------
            try:
                training_result = run_training_loop(
                    model=model,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    cfg=cfg,
                    logger=logger,
                    base_dir=base_dir,
                    model_name=model_name,
                    env=cfg.get("env", "dev"),
                    task=task,
                    tracker=tracker,
                )

            except Exception as e:
                logger.error(
                    "Training loop failed: %s",
                    e,
                    exc_info=True,
                )
                raise
        #Tracker settings
        if tracker and tracker.enabled:
            try:
                paths_abs = cfg.get("paths_abs", {}) or {}

                for artifact_name in (
                    "params_out",
                    "log_file",
                    "metrics_out",
                ):
                    artifact_path = paths_abs.get(artifact_name)
                    if artifact_path:
                        tracker.upload_file(artifact_name, artifact_path)
            except Exception as e:
                logger.warning("ClearML artifact upload skipped/failed: %s", e)

        logger.info(
            "Training completed for model: %s",
            model_name,
        )
        _register_model_for_graphs(cfg, model_name, logger)
        # Closes the training stage (CV or standard) before testing starts.
        logger.info("\n%s", "=" * 80)

        return training_result

    finally:
        try:
            if tracker and getattr(
                tracker,
                "enabled",
                False,
            ):
                tracker.close()

            elif task:
                task.close()

        except Exception as close_exc:
            logger.warning(
                "Failed to close training tracker/task: %s",
                close_exc,
                exc_info=True,
            )

        finally:
            cleanup_trial_runtime()
def main():
    p = argparse.ArgumentParser(description="Side-Channel Core Training")
    p.add_argument("--project", "-p", required=True, help="Which project (as named in master.yaml)")
    p.add_argument("--dataset", "-d", default="ascad", help="Which dataset (by name in params.yaml)")
    p.add_argument("--model-id", type=str, default=None, help="Model ID from params.yaml model block")
    args = p.parse_args()

    cfg = get_project_config(args.project, validate_data=True)
    cfg = _ensure_visualization_cfg(cfg, args.project)
    project_root = cfg["project"]["root"]

    if isinstance(cfg.get("data"), list):
        cfg["data"] = {ds["name"]: ds for ds in cfg["data"]}

    if args.dataset not in cfg["data"]:
        raise ValueError(f"Dataset '{args.dataset}' not in config. Available: {list(cfg['data'].keys())}")

    selected_dataset = cfg["data"][args.dataset]

    for model_hash, sweep_cfg, run_dir, sweep_model_name in get_pending_runs(cfg=cfg, project_root=project_root):
        if is_done(model_id=model_hash, project_root=project_root):
            logger.info("Skipping completed run %s: %s", model_hash, sweep_model_name)
            continue

        sweep_cfg["data"] = selected_dataset
        sweep_cfg["model_name"] = sweep_model_name
        sweep_cfg["model_hash"] = model_hash
        sweep_cfg = _ensure_visualization_cfg(sweep_cfg, args.project)
        logger.info("▶  Starting run %s: %s", model_hash, sweep_model_name)
        try:
            train(
                sweep_cfg,
                exec_process="train",
                override_model_name=sweep_model_name,
                model_id=args.model_id,
            )
            update_status(model_id=model_hash, status="done", project_root=project_root)
            logger.info("Finished run %s: %s", model_hash, sweep_model_name)
        except Exception:
            update_status(model_id=model_hash, status="failed", project_root=project_root)
            logger.error('Train run "%s" failed', model_hash, exc_info=True)
            raise
