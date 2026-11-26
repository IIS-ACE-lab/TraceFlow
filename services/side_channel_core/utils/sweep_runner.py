#services/side_channel_core/utils/sweep_runner.py
"""
This module runs a sweep of experiments based on a parameter grid defined in params.yaml.
It executes a command function for each combination of parameters,
allowing for retries of failed runs, resuming interrupted runs, and filtering by model names.
It also manages a registry of runs to track their status and names.
It adds the model to candidates if it has a training status of DONE.
It uses a command function to execute the runs and manages the run registry to track the status of each model.

Author: Shraddha Mishra
"""

from typing      import List, Optional
from pathlib     import Path
from utils.logger import get_logger, configure_global_logging

from utils.sweep_engine import get_pending_runs,  update_name, update_status, count_all_param_combinations
from services.side_channel_core.utils.run_lifecycle_logger import (
    complete_run_map_entry,
    fail_run_map_entry,
    interrupt_run_map_entry,
)
import copy
import yaml
logger = get_logger("SweepRunner")



def run_sweep(
    cmd_fn,
    exec_process: str,
    cfg: dict,
    project_root: Path,
    include_model_name: List[str],
    retry_failed: bool = False,
    resume_interrupted: bool = False,
    alias: Optional[str] = None,
    force: bool = False,
    include_hashes: list[str] | None = None,
):
    """
    Stream SCC runs one at a time.

    HARD RULE:
    Do NOT list(get_pending_runs()).
    get_pending_runs() writes IN_PROGRESS, so materializing it will mark
    every eligible SCC combination before training starts.
    """
    logger = get_logger(__name__)

    params_path = Path(cfg["project"]["root"]) / cfg["project"]["params"]
    base_cfg = yaml.safe_load(params_path.open())
    raw_total = count_all_param_combinations(base_cfg)

    logger.info(f"{raw_total} total grid combinations defined in params.yaml")
    logger.info("Streaming SCC runs one-by-one. No batch pre-marking.")

    executed = 0

    for model_hash, run_cfg, run_dir, model_name in get_pending_runs(
        cfg=cfg,
        project_root=project_root,
        retry_failed=retry_failed,
        resume_interrupted=resume_interrupted,
        force=force,
    ):
        if include_hashes and str(model_hash) not in {str(x) for x in include_hashes}:
            continue
        if include_model_name and model_name not in include_model_name and model_hash not in include_model_name:
            continue

        executed += 1
        run_cfg["exec_process"] = exec_process

        get_logger(name="Training", log_dir=run_dir, model_hash=model_hash)
        configure_global_logging(cfg=run_cfg, log_dir=run_dir, model_hash=model_hash)

        override = alias if (alias and executed == 1) else model_name

        try:
            update_name(project_root=project_root, model_id=model_hash, new_name=override)

            logger.info(f"Stay Patient. Running SCC Experiment {executed} …")
            logger.info(f"→ Running {exec_process.title()} of {model_hash} aka ({override})…")

            assert run_cfg.get("model_id"), (
                "model_id missing before train cmd_fn"
            )

            resolved_cfg = copy.deepcopy(run_cfg)

            #
            # get_pending_runs() already performs:
            #
            #   sweep expansion
            #   -> SCA conditional resolution
            #   -> resolve_model_contract()
            #   -> identity/hash freezing
            #
            # Do not resolve or sweep the configuration again here.
            #

            resolved_cfg["model_hash"] = str(model_hash)
            resolved_cfg["model_name"] = str(override)

            resolved_cfg.setdefault("run", {})
            resolved_cfg["run"]["model_name"] = str(override)

            sca_cfg = (
                resolved_cfg.get("sca")
                or {}
            )

            key_enumeration = (
                sca_cfg.get("key_enumeration")
                or {}
            )

            logger.info(
                "→ Executing concrete SCC run "
                "%s aka (%s) | model_id=%s "
                "attack_byte=%s leakage_model=%s "
                "candidate_ranking=%s key_enumeration=%s",
                model_hash,
                override,
                resolved_cfg.get("model_id"),
                sca_cfg.get("attack_byte"),
                sca_cfg.get("leakage_model"),
                (
                    sca_cfg.get("candidate_ranking")
                    or {}
                ).get("mode"),
                key_enumeration.get("mode"),
            )

            cmd_fn(
                cfg=resolved_cfg,
                exec_process=exec_process,
                override_model_name=override,
            )

            update_status(project_root=project_root, model_id=model_hash, new_status="DONE")

            complete_run_map_entry(
                cfg=run_cfg,
                model_hash=model_hash,
                model_name=override,
                model_id=run_cfg.get("model_id"),
                mode="scc",
                run_dir=run_dir,
                trial_index=(run_cfg.get("runtime") or {}).get("trial_index"),
                total_trials=(run_cfg.get("runtime") or {}).get("trial_total"),
                selection_source="scc_sweep",
            )

            logger.info(f"Done SCC Experiment {executed}: {model_hash} aka ({override})")

        except KeyboardInterrupt as e:
            update_status(project_root=project_root, model_id=model_hash, new_status="INTERRUPTED",
                          reason="keyboard_interrupt")

            interrupt_run_map_entry(
                    cfg=run_cfg,
                    model_hash=model_hash,
                    model_name=override,
                    model_id=run_cfg.get("model_id"),
                    mode="scc",
                    run_dir=run_dir,
                    trial_index=(run_cfg.get("runtime") or {}).get("trial_index"),
                    total_trials=(run_cfg.get("runtime") or {}).get("trial_total"),
                    selection_source="scc_sweep",
                    reason="keyboard_interrupt",
                )

            logger.info(f"Interrupted SCC Experiment {executed}: {model_hash} aka ({override})")
            raise e

        except Exception as e:
            update_status(project_root=project_root, model_id=model_hash, new_status="FAILED",
                          reason=f"scc_training_exception: {type(e).__name__}: {e}"[:500])

            fail_run_map_entry(
                cfg=run_cfg,
                model_hash=model_hash,
                model_name=override,
                model_id=run_cfg.get("model_id"),
                mode="scc",
                run_dir=run_dir,
                trial_index=(run_cfg.get("runtime") or {}).get("trial_index"),
                total_trials=(run_cfg.get("runtime") or {}).get("trial_total"),
                selection_source="scc_sweep",
                reason=f"scc_training_exception:{e}",
            )

            logger.exception(f"Failed SCC Experiment {executed}: {model_hash} aka ({override})")
            continue

    if executed == 0:
        logger.info(
            "No scoped SCC experiments to run. Either all are DONE, or current "
            "MODEL_ID/ENV/MODELS scope has no eligible pending run."
        )
