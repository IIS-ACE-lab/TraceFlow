# services/side_channel_core/src/sca/test.py
"""
test.py – Side-Channel Analysis Model Tester (SCA-correct)

Runs a one-off test pass on a trained SCA model.

SCA-correct protocol (ASCAD fixed-key style):
- train comes from Profiling_traces
- evaluation (val/test) comes from Attack_traces

Therefore:
test.h5 MUST be evaluated using Attack_traces (split="attack")
metadata MUST be preserved (for GE / rank-based metrics)

This script:
- Resolves the run’s artifact paths using cfg + model_name
- Ensures test.h5 exists (generates SCA split if missing)
- Loads checkpoint weights into the model (critical!)
- Runs run_test_loop() on attack-derived test split
- Logs metrics / updates registry via run_test_loop()

Author: Shraddha Mishra
"""

from __future__ import annotations

import argparse
import copy
from numbers import Integral
from pathlib import Path
from typing import Mapping, Optional

import torch
import yaml
from services.visualization.clearml.clearml_tracker import build_tracker_from_run_context
from utils.error_handler import catch_exceptions
from utils.seal import check as check_seal
from utils.logger import setup_logging, get_logger
from utils.config_loader import get_project_config
from utils.path_utils import resolve_path
from utils.h5_utils import inspect_and_log_h5
from utils.device import initialize_device_and_model
from utils.epoch_utils import run_test_loop
from utils.checkpoint_policy import resolve_test_checkpoint
from utils.seed import set_seed
from utils.precision import apply_precision
from services.mlops.utils.zombie_reaper import cleanup_trial_runtime
from utils.path_utils import resolve_paths_dict
from services.visualization.utils.ensure import _ensure_visualization_cfg
from services.side_channel_core.src.sca.leakage_models import resolve_runtime_leakage_model
try:
    from utils.seed import make_worker_init_fn
except ImportError:
    make_worker_init_fn = None
from core.model import build_model
from utils.trace_scaler import scale_test
from services.side_channel_core.src.sca.ascad_data_loader import load_split_dataset

logger = get_logger("Test")


def _load_seed_from_params(params_file: Path, *, fallback: Optional[int] = 42) -> tuple[Optional[int], Optional[int]]:
    """
    The seed stored inside the run's params_out artifact (for strict reproducibility), else the fallback (cfg['seed']),
    as (seed, seed_drawn). seed: null (an unseeded run, utils/seed.run_seed): (None, the seed the run drew), which its
    snapshot must hold.
    """
    if fallback is not None and (isinstance(fallback, bool) or not isinstance(fallback, Integral)):
        raise TypeError(f"fallback seed must be an integer or None, got {fallback!r}")
    if not params_file.exists():
        return (None if fallback is None else int(fallback)), None

    try:
        artifact_params = yaml.safe_load(params_file.read_text()) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Cannot read the saved run parameters from {params_file}"
        ) from exc

    if not isinstance(artifact_params, Mapping):
        raise TypeError(
            f"Saved run parameters must be a mapping, got "
            f"{type(artifact_params).__name__}"
        )

    seed = artifact_params.get("seed", fallback)
    if seed is None:                                     # an unseeded run: the seed it drew
        drawn = artifact_params.get("seed_drawn")
        if isinstance(drawn, bool) or not isinstance(drawn, Integral):
            raise TypeError(
                f"Saved run has seed: null but no integer seed_drawn (the seed it drew), got {drawn!r} in {params_file}"
            )
        return None, int(drawn)
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError(
            f"Saved run seed must be an integer, got {seed!r} in {params_file}"
        )
    return int(seed), None


def _load_model_weights(model: torch.nn.Module, ckpt_path: Path, device: torch.device, logger) -> None:
    """
    Load checkpoint weights into model.

    Supports:
    - raw state_dict
    - dict with 'state_dict'
    """
    check_seal(ckpt_path)  # utils/seal.py: refuses a model changed after training
    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    logger.info("Loaded weights (strict=True) from: %s", str(ckpt_path))


@catch_exceptions
def test(
    cfg: dict,
    *,
    exec_process: str = "test",
    override_model_name: Optional[str] = None,
    model_id: Optional[str] = None,
):
    """
    Test one trained model on attack-derived test split (test.h5).

    Contract:
    - `exec_process`: kept for uniform dispatcher/sweep interface ("test" here)
    - `override_model_name`: accepted for uniform interface (not used; test targets cfg['test']['model_name'])
    """
    try:
        # Intentional: interface compatibility with sweep/orchestrator
        #_ = exec_process
        _ = override_model_name

        logger = get_logger(__name__)

        # Single source of truth for downstream modules like ClearML adapter
        test_opts = cfg.get("test") or {}
        model_name = test_opts.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise KeyError("cfg['test']['model_name'] is required for test()")
        model_name = model_name.strip()

        # Work on a run-local cfg to avoid mutating caller
        run_cfg = copy.deepcopy(cfg)
        run_cfg["model_name"] = model_name
        run_cfg["env"] = run_cfg.get("env", "dev")
        runtime_lm = resolve_runtime_leakage_model(run_cfg)
        logger.info("Resolved runtime leakage_model=%s", runtime_lm)
        run_cfg = _ensure_visualization_cfg(run_cfg, project_name=(run_cfg.get("project") or {}).get("name", "DLSCA_ASCAD"))
        if "paths" in run_cfg:
            run_cfg["paths_abs"] = resolve_paths_dict(run_cfg["paths"], run_cfg, inject=True, mkdir=False)
        # Resolve base_dir for this run
        base_dir = Path(resolve_path(run_cfg["paths"]["artifacts_root"], run_cfg, model_name=model_name))
        base_dir.mkdir(parents=True, exist_ok=True)

        project_name = test_opts.get("project_name") or (run_cfg.get("project") or {}).get("name")

        # A CV run made with cross_validation.refit.enabled: false trained no final model.
        from utils.cross_validation import require_refit_model

        require_refit_model(run_cfg, model_name)

        # Canonical test checkpoint selection.
        #
        # The checkpoint manifest written by training is the authority. The
        # requested policy comes from:
        #
        #   evaluation.test.load_model: best_validation | final_epoch
        #
        # resolve_test_checkpoint() verifies the canonical path and SHA-256
        # before the model is loaded. Its provenance is passed unchanged to
        # run_test_loop(), which verifies it again at the evaluation boundary.
        checkpoint_resolution = resolve_test_checkpoint(
            run_cfg,
            model_name=model_name,
        )
        checkpoint_provenance = checkpoint_resolution.to_dict()
        ckpt_path = checkpoint_resolution.checkpoint_path

        logger, task, _ = setup_logging(
            cfg=run_cfg,
            project_name=project_name,
            model_name=model_name,
            base_dir=base_dir,
            process_name="Testing",
        )
        #logger.info("ClearML debug: visualization present=%s", bool(run_cfg.get("visualization")))
        tracker = None
        try:
            tracker = build_tracker_from_run_context(
                cfg=run_cfg,
                process_name="test",
                model_name=model_name,
                model_hash=run_cfg.get("model_hash"),
                extra_tags=[
                    run_cfg.get("env", "dev"),
                    str(project_name),
                    exec_process,
                ],
                task_type="testing",
            )
            # logger.info("ClearML tracker enabled=%s failure_reason=%s",
            #     getattr(tracker, "enabled", False),
            #     getattr(tracker, "disabled_reason", None))
        except Exception as e:
            logger.warning("ClearML tracker init failed: %s", e)
            tracker = None

        # Backward compatibility with existing loop code that expects `task`
        if tracker and tracker.enabled:
            task = tracker.task

            tracker.connect_config(run_cfg)
            tracker.connect_parameters(
                {
                    "group_id": model_name,
                    "group_tag": f"group:{model_name}",
                    "task_name": f"{model_name}_test",
                    "model_name": model_name,
                    "model_hash": run_cfg.get("model_hash"),
                    "project_name": project_name,
                    "process_name": "test",
                    "env": run_cfg.get("env", "dev"),
                    "checkpoint_policy_requested": checkpoint_resolution.requested_policy,
                    "checkpoint_policy_effective": checkpoint_resolution.effective_policy,
                    "checkpoint_epoch": checkpoint_resolution.checkpoint_epoch,
                    "checkpoint_path": str(ckpt_path),
                    "artifacts_root": str(base_dir),
                },
                name="run_context",
            )

            tracker.set_comment(
                f"SCA test run grouped under {model_name} for project={project_name}"
            )

        # SCA-correct: test dataset must be attack-derived "test.h5"
        data_dir = Path(resolve_path(run_cfg["paths"]["data"], run_cfg, model_name=model_name))
        test_h5 = data_dir / "test.h5"

        if not test_h5.exists():
            raise FileNotFoundError(
            f"test.h5 missing for model {model_name}. "
            f"Training phase did not complete properly."
        )

        if bool(test_opts.get("debug_h5_inspect", True)) and test_h5.exists():
            try:
                logger.info("[Before Testing]:")
                inspect_and_log_h5(str(test_h5), label="test.h5 (Attack_traces)")
            except Exception as exc:
                logger.warning(
                    "Optional test.h5 inspection failed; the dataset loader will "
                    "still enforce the data contract: %s",
                    exc,
                    exc_info=True,
                )
        params_file = Path(resolve_path(run_cfg["paths"]["params_out"], run_cfg, model_name=model_name))
        if tracker and tracker.enabled:
            tracker.connect_parameters(
                {
                    "test_h5": str(test_h5),
                    "params_file": str(params_file),
                    "batch_size": int(test_opts.get("batch_size", 128)),
                },
                name="test_data_context",
            )
        # Seed: prefer run artifact params_out if present, else cfg seed

        fallback = run_cfg.get("seed", 42)
        seed, drawn = _load_seed_from_params(params_file, fallback=None if fallback is None else int(fallback))
        run_cfg["seed"] = seed  # ensure consistency everywhere (None: an unseeded run, its drawn seed below)
        if seed is None:
            if drawn is None:
                drawn = run_cfg.get("seed_drawn")
            if drawn is not None:
                run_cfg["seed_drawn"] = int(drawn)
        seed_used = set_seed(run_cfg)
        apply_precision(run_cfg)   # trainer.precision: fp32 | tf32 | default
        logger.info("Using seed=%d", seed_used)
        if tracker and tracker.enabled:
            tracker.connect_parameters(
                {
                    "seed_used": seed_used,
                },
                name="runtime_seed",
            )
        # Dataset with metadata preserved for GE/rank metrics
        # IMPORTANT: call filename positionally to avoid keyword-arg mismatch across versions.
        test_ds = load_split_dataset(run_cfg, "test.h5", split="attack", return_metadata=True)
        # trace_normalization.mode fitted: the run's final scaler, fitted on its training traces.
        scale_test(run_cfg, test_ds)

        # Prefer cfg-controlled dataloader knobs (consistent with your training loaders)
        dl_cfg = (run_cfg.get("dataloader") or {})
        nw_cfg = (run_cfg.get("num_workers") or {})
        num_workers = int(nw_cfg.get("val", 1))                 # the training loaders' default (dataloader_utils)

        worker_init = make_worker_init_fn(seed_used) if (make_worker_init_fn and num_workers > 0) else None

        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=int(test_opts.get("batch_size", 128)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(dl_cfg.get("pin_memory", True)),
            persistent_workers=bool(dl_cfg.get("persistent_workers", True)) and (num_workers > 0),
            prefetch_factor=int(dl_cfg.get("prefetch_factor", 4)) if (num_workers > 0) else None,   # one default with the training loaders (dataloader_utils)
            worker_init_fn=worker_init,
        )

            # Build model/device
        model, device = initialize_device_and_model(run_cfg, build_model, logger, model_id=model_id)
        if tracker and tracker.enabled:
            tracker.connect_parameters(
                {
                    "device": str(device),
                    "model_id": model_id,
                },
                name="runtime_device",
            )
        # Load the checkpoint selected and cryptographically verified by
        # checkpoint_policy.resolve_test_checkpoint().
        logger.info(
            "Loading resolved test checkpoint: policy=%s epoch=%d path=%s",
            checkpoint_resolution.effective_policy.upper(),
            checkpoint_resolution.checkpoint_epoch,
            str(ckpt_path),
        )
        _load_model_weights(model, ckpt_path, device, logger)

        if tracker and tracker.enabled:
            tracker.connect_parameters(
                {
                    "requested_policy": checkpoint_resolution.requested_policy,
                    "effective_policy": checkpoint_resolution.effective_policy,
                    "validation_checkpointing": checkpoint_resolution.validation_checkpointing,
                    "checkpoint_epoch": checkpoint_resolution.checkpoint_epoch,
                    "checkpoint_path": str(ckpt_path),
                    "checkpoint_sha256": checkpoint_resolution.checkpoint_sha256,
                    "manifest_path": str(checkpoint_resolution.manifest_path),
                    "fallback_reason": checkpoint_resolution.fallback_reason,
                    "selection_metric": checkpoint_resolution.selection_metric,
                    "selection_mode": checkpoint_resolution.selection_mode,
                    "selection_score": checkpoint_resolution.selection_score,
                },
                name="checkpoint_context",
            )
            try:
                tracker.upload_file("test_checkpoint", ckpt_path)
            except Exception as exc:
                logger.warning(
                    "Optional checkpoint upload failed; local testing will continue: %s",
                    exc,
                    exc_info=True,
                )
        # Run test loop
        try:
            test_results = run_test_loop(
                model=model,
                device=device,
                test_loader=test_loader,
                cfg=run_cfg,
                logger=logger,
                base_dir=base_dir,
                model_name=model_name,
                env=run_cfg.get("env", "dev"),
                task=task,
                tracker=tracker,
                checkpoint_provenance=checkpoint_provenance,
            )
            logger.info("Testing completed for %s", model_name)
            return test_results
        except Exception as e:
            logger.error("Testing loop failed: %s", e, exc_info=True)
            raise
        finally:
            if tracker and tracker.enabled:
                try:
                    paths_abs = run_cfg.get("paths_abs", {}) or {}

                    for artifact_name in (
                        "params_out",
                        "log_file",
                        "metrics_out",
                    ):
                        artifact_path = paths_abs.get(artifact_name)
                        if artifact_path:
                            tracker.upload_file(artifact_name, artifact_path)
                except Exception as e:
                    logger.warning("ClearML artifact upload skipped/failed during test close: %s", e)

                tracker.close()
            elif task:
                task.close()

    finally:
        cleanup_trial_runtime()


def main():
    p = argparse.ArgumentParser(description="Side-Channel Core Testing")
    p.add_argument("--project", "-p", required=True, help="Which project (as named in master.yaml)")
    p.add_argument("--model-id", type=str, default=None, help="Model ID from params.yaml model block")
    args = p.parse_args()

    cfg = get_project_config(args.project, validate_data=True)
    cfg = _ensure_visualization_cfg(cfg, args.project)
    # `test()` expects cfg['test']['model_name'] to be present in the composed config.
    test(cfg, exec_process="test", override_model_name=None, model_id=args.model_id)


if __name__ == "__main__":
    main()
