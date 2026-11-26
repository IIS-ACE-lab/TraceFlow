# services/side_channel_core/utils/cli_helpers.py
"""
Helpers of the orchestrator and the CLI: plan or resolve the exact config of
one model_hash, and run it as train -> test.

functions : resolve_registered_run_cfg, train_and_test, comma_split
orchestrator (also train-cmd -n <registered model>, see core/orchestrator.py)
 └─ resolve_registered_run_cfg   (params_out snapshot of a registered hash)
 └─ train_and_test
      └─ train in process (sca.train)
      └─ subprocess test-cmd -n <hash>   (SCC_SINGLE_HASH=<hash>)

Author: Shraddha Mishra
"""
from __future__ import annotations

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Optional
import re
from click import Context, Option as TyperOption
from services.side_channel_core.utils.run_lifecycle_logger import (
    start_run_map_entry,
    complete_run_map_entry,
    fail_run_map_entry,
    interrupt_run_map_entry,
)
from services.side_channel_core.src.sca.train import train as scc_train
from services.side_channel_core.utils.key_recovery_helper import (
    _log_completed_cycle_key_recovery_summary,
)
from utils import heartbeat
from utils.logger import get_logger
from utils.sweep_engine import update_status, get_pending_runs
from utils.sweep_test_runner import load_run_snapshot, verify_snapshot_identity
from utils.data_reproducibility import SNAPSHOT_DIGEST_KEY
from utils.test_registry import skip_unattacked_test, test_attack_settings, update_test_status
from utils.run_utils import _norm, _status
from services.side_channel_core.utils.model_naming import make_experiment_hash
from services.side_channel_core.src.core.model_contract_resolver import resolve_model_contract
from services.side_channel_core.utils.type_coerce import coerce_types
logger = get_logger("-n training")


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge override into base (dicts recurse, other types overwrite).
    """
    out = deepcopy(base)

    def rec(dst, src):
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                rec(dst[k], v)
            else:
                dst[k] = v

    rec(out, override or {})
    return out


def _ensure_train_safe_cfg(run_cfg: dict, *, model_id: str | None) -> dict:
    run_cfg = coerce_types(run_cfg)

    # A concrete run has one learning rate; a list is an unexpanded sweep
    # dimension of raw params.yaml (coerce_types already rejects a list of
    # optimizers: trainer.optimizer must be a mapping).
    lr = ((run_cfg.get("trainer") or {}).get("optimizer") or {}).get("learning_rate")
    if isinstance(lr, list):
        raise RuntimeError(
            "BUG: train_and_test received unresolved SCC config: "
            f"trainer.optimizer.learning_rate={lr!r}. "
            "This path must use the concrete run_cfg from the sweep engine or the "
            "params_out snapshot (resolve_registered_run_cfg), not raw params.yaml."
        )

    return resolve_model_contract(
        cfg=run_cfg,
        mode="scc",
        model_id=model_id or run_cfg.get("model_id"),
    )

def extract_hash_from_name(name: str) -> str:
    name = str(name).strip()

    # STRICT: match "-<hash>" at end
    match = re.search(r"-([a-f0-9]{8,})$", name)

    if match:
        return match.group(1)

    # if already a hash, accept it
    if re.fullmatch(r"[a-f0-9]{8,}", name):
        return name

    return name
def _plan_scc_models_from_sweep_engine(
    *,
    cfg,
    project_root,
    retry_failed=False,
    resume_interrupted=False,
    force=False,
    limit: int = 1,
):
    """
    Pull only `limit` executable SCC identities from sweep_engine.

    Critical:
    - Do NOT list(get_pending_runs()).
    - get_pending_runs() writes registry/run_map entries as it yields.
    - Materializing all candidates creates thousands of queued/in_progress plans
      before train/test gets a chance to run.
    """
    models = []

    for model_hash, run_cfg, run_dir, model_name in get_pending_runs(
        cfg=cfg,
        project_root=project_root,
        retry_failed=retry_failed,
        resume_interrupted=resume_interrupted,
        force=force,
    ):
        models.append((
            model_hash,
            {
                "model_name": model_name,
                "model_hash": model_hash,
                "run_cfg": run_cfg,
                "run_dir": str(run_dir),
            }
        ))

        if len(models) >= limit:
            break

    return models

def _replan_single_hash(
    *,
    cfg,
    project_root,
    model_hash: str,
    prior_status: str,
    force: bool = False,
):
    """
    Re-plan exactly one registered hash through the sweep engine.

    get_pending_runs() honours SCC_SINGLE_RUN/SCC_SINGLE_HASH (the same lock
    train_and_test() uses for its children), so only this hash can be yielded
    and the planner stops right after it. Every retry/resume status is allowed
    because the caller already selected the hash from the registry; force is
    needed when the entry is DONE.
    """
    saved = {k: os.environ.get(k) for k in ("SCC_SINGLE_RUN", "SCC_SINGLE_HASH")}
    os.environ["SCC_SINGLE_RUN"] = "1"
    os.environ["SCC_SINGLE_HASH"] = str(model_hash)
    try:
        planned = _plan_scc_models_from_sweep_engine(
            cfg=cfg,
            project_root=project_root,
            retry_failed=True,
            resume_interrupted=True,
            force=bool(force or prior_status == "done"),
            limit=1,
        )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    for planned_hash, planned_info in planned:
        if str(planned_hash) == str(model_hash) and isinstance(planned_info.get("run_cfg"), dict):
            return planned_info["run_cfg"]
    return None


def resolve_registered_run_cfg(
    *,
    cfg,
    project_root,
    model_hash: str,
    info: dict,
    registry_before: dict,
    force: bool = False,
):
    """
    Resolve the exact config to (re)run one selected hash with.

    Returns (run_cfg, source) where source is one of:

        "snapshot"  - paths.params_out snapshot of the registered run: the
                      resolved config it was trained with. Mandatory for any
                      repeat (retry/resume/force/--model-name/run_map
                      recovery) of a hash already in run_registry.json.
        "replanned" - snapshot missing (the run failed before
                      data_reproducibility wrote it): the hash was re-planned
                      through the sweep engine and make_experiment_hash()
                      was asserted against the registry hash.
        "planner"   - hash not registered before this pass: the planner's
                      fresh config (first creation).

    model_hash and model_name stay frozen in every case. Raises with a clear
    message when neither source can reproduce the registered hash. A snapshot
    that exists but is truncated or damaged (digest, required sections,
    model_hash) or no longer hashes to model_hash raises too: it is never
    silently replaced by a re-planned config. So does a snapshot written
    before 2026-09-29 (no snapshot_sha256): its hash cannot be reproduced.
    """
    prior = registry_before.get(str(model_hash)) if isinstance(registry_before, dict) else None
    prior_status = _status(prior)
    # An entry registered from a shipped snapshot (utils/reproduce.py) is trained from that
    # snapshot from its first run on, like an established one.
    established = isinstance(prior, dict) and (prior_status not in {"", "new"} or bool(prior.get("imported_snapshot")))
    planned = info.get("run_cfg") if isinstance(info.get("run_cfg"), dict) else None
    train_info = prior if isinstance(prior, dict) else info

    model_name = _norm(info.get("model_name")) or _norm((prior or {}).get("model_name"))
    if not model_name:
        raise RuntimeError(f"Registry entry for hash={model_hash} has no model_name; cannot resolve its config.")

    if established:
        try:
            # Only a missing snapshot falls back; a damaged one raises (ValueError).
            run_cfg = load_run_snapshot(cfg, train_info, model_name=model_name)
        except FileNotFoundError as exc:
            logger.warning(
                "[orchestrator] %s Falling back to re-planning hash=%s through the sweep engine.",
                exc, model_hash,
            )
        else:
            # A pre-2026-09-29 snapshot (no digest) cannot reproduce its
            # path-dependent hash; re-training it would write a snapshot its
            # own test then rejects. It can still be tested (make test).
            if SNAPSHOT_DIGEST_KEY not in run_cfg:
                message = (
                    f"hash={model_hash} ({model_name}) was registered before 2026-09-29, when "
                    "model_hash still included absolute paths; it cannot be re-run under that "
                    "hash. Re-run it as a new experiment: make delete_database ENV=<env> "
                    f"PROJECT=<project> MODEL={model_name} CONFIRM=1, then make orchestrate."
                )
                logger.error("[orchestrator] %s", message)
                raise RuntimeError(message)
            verify_snapshot_identity(run_cfg, train_info, model_hash=str(model_hash), model_name=model_name)
            # Recomputed by the orchestrator against the live project root.
            run_cfg.pop("paths_abs", None)
            return run_cfg, "snapshot"

    source = "replanned" if established else "planner"
    if planned is None:
        planned = _replan_single_hash(
            cfg=cfg,
            project_root=project_root,
            model_hash=str(model_hash),
            prior_status=prior_status,
            force=force,
        )
    if planned is None:
        raise RuntimeError(
            f"Cannot reproduce the config of registered hash={model_hash} ({model_name}): "
            "no params_out snapshot exists and the current params.yaml sweep does not "
            "produce this hash any more. Delete the stale registry/run_map entries or "
            "restore the snapshot."
        )

    recomputed = make_experiment_hash(planned)
    if str(recomputed) != str(model_hash):
        raise RuntimeError(
            f"Re-planned config for hash={model_hash} ({model_name}) hashes to "
            f"{recomputed}; refusing to run it under a foreign identity."
        )

    planned["model_hash"] = str(model_hash)
    planned["model_name"] = model_name
    planned["frozen_identity"] = {"model_name": model_name, "model_hash": str(model_hash)}
    return planned, source


def comma_split(
    ctx: Context,
    param: TyperOption,
    value: Optional[str] | Optional[List[str]],
) -> List[str]:
    """
    Turn a comma-separated string or list of comma-strings
    into a flat List[str].
    """
    if not value:
        return []
    parts: List[str] = []
    if isinstance(value, list):
        for chunk in value:
            parts += [v.strip() for v in chunk.split(",") if v.strip()]
    else:
        parts = [v.strip() for v in value.split(",") if v.strip()]
    return parts


def train_and_test(
    cfg,
    env,
    override_name=None,
    retry_failed=False,
    resume_interrupted=False,
    force=False,
    model_id=None,
):
    """
    Execute exactly ONE selected SCC run as:

        train-cmd(model_hash) -> test-cmd(model_hash)

    Keeps -r/-c/-f, but locks child sweep scope to this model_hash.
    """
    logger = get_logger("Orchestrator")

    project_root = Path(cfg["project"]["root"]).resolve()

    model_hash = str(
        cfg.get("model_hash")
        or (cfg.get("runtime") or {}).get("model_hash")
        or ""
    ).strip()

    model_name = str(
        cfg.get("model_name")
        or (cfg.get("runtime") or {}).get("model_name")
        or override_name
        or ""
    ).strip()

    if not model_hash:
        raise ValueError("train_and_test requires cfg['model_hash'] for paired SCC execution")

    if not model_name:
        raise ValueError("train_and_test requires cfg['model_name'] for paired SCC execution")

    effective_model_id = str(
        model_id
        or cfg.get("model_id")
        or os.environ.get("MODEL_ID")
        or ""
    ).strip()

    env_vars = os.environ.copy()
    env_vars["ENV"] = env
    env_vars["MODEL_ID"] = effective_model_id

    # Critical: flags may exist, but child sweep must see only this hash.
    env_vars["SCC_SINGLE_RUN"] = "1"
    env_vars["SCC_SINGLE_HASH"] = model_hash
    env_vars["MODEL_NAME"] = model_hash
    env_vars["MODELS"] = model_hash

    cmd_train = [
        sys.executable, "-m", "core.cli", "train-cmd",
        "--project", cfg["project"]["name"],
        "--dataset", cfg["data"]["name"],
        "--env", env,
        "-n", model_hash,
    ]

    cmd_test = [
        sys.executable, "-m", "core.cli", "test-cmd",
        "--project", cfg["project"]["name"],
        "--dataset", cfg["data"]["name"],
        "--env", env,
        "-n", model_hash,
    ]

    if effective_model_id:
        cmd_train += ["--model-id", effective_model_id]
        cmd_test += ["--model-id", effective_model_id]

    if override_name:
        cmd_train += ["--alias", override_name]

    # Bring back self-healing flags.
    if retry_failed:
        cmd_train += ["-r"]
        cmd_test += ["-r"]

    if resume_interrupted:
        cmd_train += ["-c"]
        cmd_test += ["-c"]

    if force:
        cmd_train += ["-f"]
        cmd_test += ["-f"]
    #################################################################
    #train fails → raise error → test does not run
    #train succeeds → test runs
    #################################################################
    safe_cfg = None
    #safe_cfg is the correct configuration for report path resolution because it contains the resolved SCC contract.
    try:
        logger.info(
            "[train_and_test] HASH-LOCKED TRAIN: hash=%s name=%s cmd=%s",
            model_hash,
            model_name,
            " ".join(cmd_train),
        )
        start_run_map_entry(
            cfg=cfg,
            model_hash=model_hash,
            model_name=model_name,
            model_id=effective_model_id,
            mode="scc",
            run_dir=Path(str((cfg.get("paths") or {}).get("artifacts_root") or "")),
            trial_index=(cfg.get("runtime") or {}).get("trial_index"),
            total_trials=(cfg.get("runtime") or {}).get("trial_total"),
            selection_source=(cfg.get("runtime") or {}).get("selection_source", "scc_sweep"),
        )
        # logger.info(
        #     "[train_and_test] DIRECT HASH-LOCKED TRAIN: hash=%s name=%s model_id=%s",
        #     model_hash,
        #     model_name,
        #     effective_model_id,
        # )

        try:
            safe_cfg = _ensure_train_safe_cfg(
                cfg,
                model_id=effective_model_id or cfg.get("model_id"),
            )
            # Preserve the exact run identity needed to resolve persisted reports.
            safe_cfg["model_hash"] = model_hash
            safe_cfg["model_name"] = model_name
            safe_cfg["model_id"] = effective_model_id

            safe_cfg.setdefault("runtime", {})
            if not isinstance(safe_cfg["runtime"], dict):
                safe_cfg["runtime"] = {}

            safe_cfg["runtime"]["model_hash"] = model_hash
            safe_cfg["runtime"]["model_name"] = model_name
            safe_cfg["runtime"]["model_id"] = effective_model_id
            scc_train(
                cfg=safe_cfg,
                exec_process="train",
                override_model_name=model_name,
                model_id=safe_cfg.get("model_id"),
            )

        except Exception as e:
            logger.error(
                "[train_and_test] TRAIN failed. Skipping TEST for hash=%s name=%s error=%s",
                model_hash,
                model_name,
                e,
                exc_info=True,
            )

            fail_run_map_entry(
                cfg=cfg,
                model_hash=model_hash,
                model_name=model_name,
                model_id=effective_model_id,
                mode="scc",
                run_dir=Path(str((cfg.get("paths") or {}).get("artifacts_root") or "")),
                trial_index=(cfg.get("runtime") or {}).get("trial_index"),
                total_trials=(cfg.get("runtime") or {}).get("trial_total"),
                selection_source=(cfg.get("runtime") or {}).get("selection_source", "scc_sweep"),
                reason=f"train_failed_{type(e).__name__}",
            )

            raise
        update_status(
            project_root=project_root,
            model_id=model_hash,
            new_status="DONE",
        )

        # perform_attack.test.attack_times: 0 leaves the test traces unattacked:
        # there is nothing to test, so no test process is started. The test
        # is recorded as SKIPPED with the reason, and the summary below says so.
        test_cfg = safe_cfg if isinstance(safe_cfg, dict) else cfg
        # what the test is attacked with, before the decision below reads it (a skipped test starts no
        # test process, so this line is its record)
        logger.info("[train_and_test] Test attack settings: %s", test_attack_settings(test_cfg))
        skip_reason = skip_unattacked_test(project_root, model_hash, test_cfg)
        if skip_reason is not None:
            logger.info("\n%s", "\n".join(["=" * 80, "TESTING SKIPPED", "=" * 80]))
            logger.info(
                "[train_and_test] TEST skipped: hash=%s name=%s: %s. Set "
                "perform_attack.test.attack_times >= 1 to test the model.",
                model_hash,
                model_name,
                skip_reason,
            )
        else:
            update_test_status(
                project_root=project_root,
                model_id=model_hash,
                new_status="queued",
            )
            logger.info(
                "[train_and_test] HASH-LOCKED TEST: hash=%s name=%s cmd=%s",
                model_hash,
                model_name,
                " ".join(cmd_test),
            )
            logger.info("\n%s", "\n".join(["=" * 80, "TESTING", "=" * 80]))
            # TraceFlow Monitor: the test runs in a subprocess; this process beats for it.
            heartbeat.update(phase="test")

            try:
                subprocess.run(
                    cmd_test,
                    env=env_vars,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                logger.error(
                    "[train_and_test] TEST failed: hash=%s name=%s exit=%s",
                    model_hash,
                    model_name,
                    exc.returncode,
                )
                fail_run_map_entry(
                    cfg=cfg,
                    model_hash=model_hash,
                    model_name=model_name,
                    model_id=effective_model_id,
                    mode="scc",
                    run_dir=Path(str((cfg.get("paths") or {}).get("artifacts_root") or "")),
                    trial_index=(cfg.get("runtime") or {}).get("trial_index"),
                    total_trials=(cfg.get("runtime") or {}).get("trial_total"),
                    selection_source=(cfg.get("runtime") or {}).get("selection_source", "scc_sweep"),
                    reason=f"test_failed_exit_{exc.returncode}",
                )
                raise
        # The complete model lifecycle has succeeded:
        # training -> validation artifact generation -> final testing (or its
        # documented skip).
        #
        # Use the resolved training configuration because it contains the concrete
        # leakage model, resolved paths, model identity, and reporting contract.
        summary_cfg = safe_cfg if isinstance(safe_cfg, dict) else cfg
        try:
            summary_complete = _log_completed_cycle_key_recovery_summary(
                cfg=summary_cfg,
                logger=logger,
            )

            if not summary_complete:
                logger.warning(
                    "[train_and_test] Final key-recovery summary incomplete: "
                    "hash=%s name=%s",
                    model_hash,
                    model_name,
                )

        except Exception as summary_exc:
            # Reporting is deliberately non-fatal. A successful train/test lifecycle
            # must not be converted into a failed run by terminal reporting.
            logger.warning(
                "[train_and_test] Final key-recovery reporting failed: "
                "hash=%s name=%s error=%s",
                model_hash,
                model_name,
                summary_exc,
                exc_info=True,
            )
        complete_run_map_entry(
            cfg=cfg,
            model_hash=model_hash,
            model_name=model_name,
            model_id=effective_model_id,
            mode="scc",
            run_dir=Path(str((cfg.get("paths") or {}).get("artifacts_root") or "")),
            trial_index=(cfg.get("runtime") or {}).get("trial_index"),
            total_trials=(cfg.get("runtime") or {}).get("trial_total"),
            selection_source=(cfg.get("runtime") or {}).get("selection_source", "scc_sweep"),
        )
    except KeyboardInterrupt:
        logger.info(
            "[train_and_test] INTERRUPTED: hash=%s name=%s",
            model_hash,
            model_name,
        )
        interrupt_run_map_entry(
            cfg=cfg,
            model_hash=model_hash,
            model_name=model_name,
            model_id=effective_model_id,
            mode="scc",
            run_dir=Path(str((cfg.get("paths") or {}).get("artifacts_root") or "")),
            trial_index=(cfg.get("runtime") or {}).get("trial_index"),
            total_trials=(cfg.get("runtime") or {}).get("trial_total"),
            selection_source=(cfg.get("runtime") or {}).get("selection_source", "scc_sweep"),
            reason="keyboard_interrupt",
        )
        raise


def select_data_entry(cfg: dict, dataset: Optional[str], command: str) -> None:
    """cfg["data"] becomes the one entry DATASET names. No default: a run never assumes an entry
    (the old default "ascad" failed every project without one: make reproduce on ZvW; audit 2026-10-23)."""
    entries = list((cfg.get("data") or {}).keys())
    if not dataset:
        raise ValueError(f"{command}: DATASET is required; the data entries of params.yaml: {entries}")
    if dataset not in cfg["data"]:
        raise ValueError(f"{command}: dataset '{dataset}' not defined in params.yaml; its entries: {entries}")
    cfg["data"] = cfg["data"][dataset]
