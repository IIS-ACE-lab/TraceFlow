# services/side_channel_core/utils/sweep_test_runner.py
"""
Robust test sweep runner for Side-Channel Analysis.

Train registry is the source of truth for test eligibility; the resolved
training config comes from the run's paths.params_out snapshot. The master config only selects test scope.
run_map is advisory metadata only: if present, it must not contradict scope.

Author: Shraddha Mishra
"""
from __future__ import annotations

import copy
import os

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import yaml

from utils.data_reproducibility import SNAPSHOT_DIGEST_KEY, read_params_snapshot
from utils.logger import configure_global_logging, get_logger
from utils.path_utils import resolve_path
from services.side_channel_core.utils.model_naming import make_experiment_hash
from .sweep_engine import load_registry as load_train_registry
from .device import failure_reason
from .test_registry import (
    FINISHED_TEST_STATUSES,
    SKIPPED,
    load_test_registry,
    update_test_registry,
    test_attack_settings,
    unattacked_test_reason,
)
from services.side_channel_core.utils.run_map import load_run_map
from services.side_channel_core.utils.run_lifecycle_logger import sync_run_map_before_planning

logger = get_logger("TestSweep")


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _status(value: Any) -> str:
    return _norm(value).upper()


def _matches_scope(
    *, model_hash: str, train_info: dict, run_map_info: dict | None,
    env: str, model_id: str, include_models: list[str],
) -> bool:
    if not isinstance(train_info, dict):
        return False
    run_map_info = run_map_info if isinstance(run_map_info, dict) else {}
    train_env = _norm(train_info.get("env"))
    map_env = _norm(run_map_info.get("env"))
    if env:
        if train_env and train_env != env:
            return False
        if map_env and map_env != env:
            return False
    train_model_id = _norm(train_info.get("model_id"))
    map_model_id = _norm(run_map_info.get("model_id"))
    if model_id:
        if train_model_id and train_model_id != model_id:
            return False
        if map_model_id and map_model_id != model_id:
            return False
        if not train_model_id and not map_model_id:
            return False
    if include_models:
        model_name = _norm(train_info.get("model_name"))
        allowed = {_norm(x) for x in include_models if _norm(x)}
        return model_hash in allowed or model_name in allowed
    return True


def _should_test(
    *, train_status: str, test_status: str, include_models: list[str],
    force: bool, retry_failed: bool, resume_interrupted: bool,
) -> bool:
    if train_status != "DONE":
        return False
    if include_models or force:
        return True
    if retry_failed:
        return test_status == "FAILED"
    if resume_interrupted:
        return test_status in {"IN_PROGRESS", "INTERRUPTED", "RUNNING", "IN_PROCESS"}
    # SKIPPED (perform_attack.test.attack_times: 0) is a finished test too.
    return test_status not in FINISHED_TEST_STATUSES


def load_run_snapshot(cfg: dict, train_info: dict, *, model_name: str) -> dict:
    """Load the resolved config training dumped to paths.params_out.

    Shared by testing (below) and by the orchestrator's repeat paths
    (retry/resume/force/--model-name): once a model_hash exists in
    run_registry.json, every repeat of that experiment uses this snapshot.
    """
    lookup = copy.deepcopy(cfg)
    lookup["model_name"] = model_name
    lookup["env"] = _norm(train_info.get("env")) or _norm(cfg.get("env")) or "dev"
    path = Path(resolve_path(lookup["paths"]["params_out"], lookup))
    if not path.is_file():
        raise FileNotFoundError(
            f"Resolved training config snapshot not found for {model_name}: {path}. "
            "Testing needs the exact configuration the model was trained with."
        )
    # Rejects a truncated/damaged snapshot (digest, required sections, model_hash).
    saved = read_params_snapshot(path)
    # The snapshot stores portable (relative) project paths; use the live ones.
    # Other relative paths are re-rooted by resolve_path() where they are used.
    saved["project"] = copy.deepcopy(cfg["project"])
    return saved


# Backward-compatible private name.
_load_run_snapshot = load_run_snapshot


def verify_snapshot_identity(
    run_cfg: dict, train_info: dict, *, model_hash: str, model_name: str,
) -> dict:
    """Pin a loaded snapshot to the registry identity; reject conflicting records.

    model_hash/model_name are frozen to the registry values; env/model_id must
    agree with the registry entry when both sides record them.

    A snapshot with snapshot_sha256 (written since 2026-09-29, layout-
    independent hashing) must also reproduce model_hash: make_experiment_hash
    is recomputed from it (with the live project block) and a mismatch
    raises. An older snapshot cannot reproduce its hash (the hash then
    included absolute paths); it keeps the registry hash with a WARNING.
    """
    for field, expected in (("model_hash", model_hash), ("model_name", model_name)):
        recorded = _norm(run_cfg.get(field))
        if recorded and recorded != expected:
            raise ValueError(f"Saved run_cfg.{field}={recorded!r} conflicts with registry {expected!r}")
        run_cfg[field] = expected
    if run_cfg.pop(SNAPSHOT_DIGEST_KEY, None) is not None:
        recomputed = make_experiment_hash(run_cfg)
        if recomputed != model_hash:
            message = (
                f"Snapshot of {model_name} hashes to {recomputed}, not to its registered "
                f"model_hash {model_hash}: the snapshot was edited, master.yaml changed "
                "this project's scripts since training, or the run was re-trained under a "
                "hash registered before 2026-09-29. Restore the snapshot / master.yaml "
                "scripts, or re-run the experiment as a new run."
            )
            logger.error(message)
            raise ValueError(message)
    else:
        logger.warning(
            "Snapshot of %s predates 2026-09-29 (no %s): model_hash %s is taken from "
            "the registry without recomputation.",
            model_name, SNAPSHOT_DIGEST_KEY, model_hash,
        )
    for field in ("env", "model_id"):
        registered = _norm(train_info.get(field))
        recorded = _norm(run_cfg.get(field))
        if registered and recorded and registered != recorded:
            raise ValueError(f"Saved run_cfg.{field} conflicts with the training registry")
        if registered:
            run_cfg[field] = registered
    run_cfg["frozen_identity"] = {"model_name": model_name, "model_hash": model_hash}
    return run_cfg


def data_budget(run_cfg: Mapping[str, Any]) -> Optional[int]:
    """data[].attack_trace_budget of the run's data entry (owner 2026-10-27: read by a Budget test, out of the hash):
    the project's params.yaml entry of that name as it is now, so a run trained before the key gets it too; else the
    snapshot's copy of the entry. None when neither sets one (a Budget test is then refused: test_request._apply_kind)."""
    entry = run_cfg.get("data") if isinstance(run_cfg.get("data"), Mapping) else {}
    name, budget = entry.get("name"), entry.get("attack_trace_budget")
    project = run_cfg.get("project") or {}
    params = Path(str(project.get("root") or "")) / str(project.get("params") or "config/params.yaml")
    if name and params.is_file():
        try:
            current = yaml.safe_load(params.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            current = {}
        for item in current.get("data") or []:
            if isinstance(item, Mapping) and item.get("name") == name:
                if item.get("attack_trace_budget") is not None:      # empty there: the snapshot's copy
                    budget = item["attack_trace_budget"]
                break
    if budget is None:
        return None
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError(f"data entry {name!r}: attack_trace_budget must be an integer >= 1 (the traces of one test attack), got {budget!r}")
    return budget


def _build_test_cfg(train_info: dict, *, model_hash: str, model_name: str, cfg: dict) -> dict:
    """Use the trained run's snapshot; never reconstruct it from sweep lists."""
    run_cfg = load_run_snapshot(cfg, train_info, model_name=model_name)
    verify_snapshot_identity(run_cfg, train_info, model_hash=model_hash, model_name=model_name)
    sca = run_cfg.get("sca")
    if not isinstance(sca, dict) or not isinstance(sca.get("leakage_model"), str):
        raise ValueError("Saved training run_cfg.sca.leakage_model must be a resolved string")
    if type(sca.get("attack_byte")) is not int:
        raise ValueError("Saved training run_cfg.sca.attack_byte must be a resolved integer")

    test_cfg = run_cfg.setdefault("test", {})
    if not isinstance(test_cfg, dict):
        raise TypeError("Saved run_cfg.test must be a mapping")
    per_model = test_cfg.get("models", {}).get(model_name, {})
    if isinstance(per_model, dict):
        test_cfg.update(copy.deepcopy(per_model))
    test_cfg["model_name"] = model_name
    # test.py resolves the actual checkpoint through the training manifest.
    for key, value in list((run_cfg.get("paths") or {}).items()):
        run_cfg["paths"][key] = resolve_path(value, run_cfg)
    return run_cfg


# test_registry.json fields a re-test keeps: the batch labels and the external tests.
KEPT_TEST_FIELDS = ("batches", "batch_ids", "external_tests")


def run_test_sweep(
    cmd_fn, cfg: dict, exec_process: str, project_root: Path,
    include_models: list, retry_failed: bool, resume_interrupted: bool,
    force: bool = False,
    registry_extra: dict | None = None,
):
    """
    Execute eligible tests.

    registry_extra: optional fields stored in each test_registry.json entry
    (make test records its request there).

    Eligibility:
    - train registry entry must be DONE
    - env/model_id/include filters must match
    - test status policy decides whether to run
    """
    project_root = Path(project_root).resolve()
    sync_run_map_before_planning(project_root)
    train_reg = load_train_registry(project_root)
    test_reg = load_test_registry(project_root)
    run_map = load_run_map(project_root)
    requested_env = _norm(cfg.get("env") or os.environ.get("ENV") or "dev") or "dev"
    requested_model_id = _norm(cfg.get("model_id") or os.environ.get("MODEL_ID"))
    include_models = [_norm(x) for x in include_models if _norm(x)]
    logger.info(
        "TestSweep scope: env=%s model_id=%s include=%s force=%s retry_failed=%s resume_interrupted=%s train_entries=%d test_entries=%d",
        requested_env, requested_model_id or "<unset>", include_models,
        force, retry_failed, resume_interrupted, len(train_reg), len(test_reg),
    )
    candidates: list[tuple[str, dict]] = []
    for model_hash, train_info in train_reg.items():
        if not _matches_scope(
            model_hash=model_hash, train_info=train_info,
            run_map_info=run_map.get(model_hash), env=requested_env,
            model_id=requested_model_id, include_models=include_models,
        ):
            continue
        if _should_test(
            train_status=_status(train_info.get("status")),
            test_status=_status((test_reg.get(model_hash) or {}).get("status")),
            include_models=include_models, force=force, retry_failed=retry_failed,
            resume_interrupted=resume_interrupted,
        ):
            candidates.append((model_hash, train_info))
    logger.info("TestSweep selected %d run(s)", len(candidates))
    failed: list[str] = []
    for idx, (model_hash, train_info) in enumerate(candidates, start=1):
        model_name = _norm(train_info.get("model_name"))
        if not model_name:
            logger.warning("Skipping %s: missing model_name", model_hash)
            continue
        # A new test of the recorded kind: its outcome fields start afresh, while the entry's
        # batch labels and its external tests (test_request.run_external_test) stay.
        previous = test_reg.get(model_hash) or {}
        test_reg[model_hash] = {
            "model_name": model_name,
            "model_id": _norm(train_info.get("model_id") or requested_model_id),
            "status": "in_progress", "path": _norm(train_info.get("path")),
            "env": _norm(train_info.get("env") or requested_env),
            **{key: copy.deepcopy(previous[key]) for key in KEPT_TEST_FIELDS if key in previous},
            **copy.deepcopy(registry_extra or {}),
        }
        update_test_registry(project_root, lambda reg: reg.__setitem__(model_hash, test_reg[model_hash]))
        # Config construction and logging can fail too; record those failures.
        try:
            run_cfg = _build_test_cfg(
                train_info, model_hash=model_hash, model_name=model_name, cfg=cfg
            )
            runtime = run_cfg.setdefault("runtime", {})
            runtime["model_hash"] = model_hash
            runtime["model_name"] = model_name
            effective_model_id = _norm(run_cfg.get("model_id"))
            if effective_model_id:
                runtime["model_id"] = effective_model_id
            log_dir = configure_global_logging(cfg=run_cfg, model_hash=model_hash)
            testing_logger = get_logger(name="Testing", log_dir=str(log_dir))
            # what the test is attacked with, in the Testing log before the decision below reads it
            testing_logger.info("Test attack settings: %s", test_attack_settings(run_cfg))
            # The model's own config leaves the test traces unattacked: there
            # is nothing to test. Recorded as SKIPPED, with the reason.
            skip_reason = unattacked_test_reason(run_cfg)
            if skip_reason is not None:
                test_reg[model_hash]["status"] = SKIPPED
                test_reg[model_hash]["status_reason"] = skip_reason
                logger.info(
                    "[%d/%d] Test SKIPPED hash=%s name=%s: %s. Set perform_attack.test.attack_times "
                    ">= 1 (a new experiment) to test a model.",
                    idx, len(candidates), model_hash, model_name, skip_reason,
                )
                continue
            logger.info("[%d/%d] Testing hash=%s name=%s", idx, len(candidates), model_hash, model_name)
            cmd_fn(run_cfg, exec_process=exec_process,
                   override_model_name=model_name, model_id=effective_model_id)
            test_reg[model_hash]["status"] = "DONE"
            test_reg[model_hash].pop("status_reason", None)
            logger.info("Test DONE hash=%s name=%s", model_hash, model_name)
        except KeyboardInterrupt:
            test_reg[model_hash]["status"] = "INTERRUPTED"
            test_reg[model_hash]["status_reason"] = "keyboard_interrupt: stopped by the user during the test"
            logger.warning("Test INTERRUPTED hash=%s name=%s", model_hash, model_name)
            raise
        except Exception as exc:
            test_reg[model_hash]["status"] = "FAILED"
            test_reg[model_hash]["status_reason"] = failure_reason(exc)   # utils/device.py
            failed.append(f"{model_hash}: {type(exc).__name__}: {exc}")
            logger.exception("Test FAILED hash=%s name=%s error=%s", model_hash, model_name, exc)
        finally:
            update_test_registry(project_root, lambda reg: reg.__setitem__(model_hash, test_reg[model_hash]))
    if include_models and not candidates:
        raise RuntimeError(
            f"No trained run eligible for testing matched {include_models} "
            f"(env={requested_env}, model_id={requested_model_id or '<unset>'})"
        )
    if failed:
        # Remaining runs were still attempted; the caller must see the failure.
        raise RuntimeError("Test failed for: " + "; ".join(failed))
