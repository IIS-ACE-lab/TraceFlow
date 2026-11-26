# services/side_channel_core/utils/run_lifecycle_logger.py
"""
run_lifecycle_logger.py
=======================

Shared run lifecycle logger for SCC, AutoSearch, and AutoMLOps.

Purpose
-------
This module provides a single high-level API for updating the project-global
run_map.jsonl during model execution.

The low-level run_map.jsonl storage rules live in:

    services/side_channel_core/utils/run_map.py

This module intentionally acts as a thin lifecycle wrapper around run_map.py so
that different execution modes can log run progress consistently:

    - SCC grid/sweep mode
    - AutoSearch / OptunaLite mode
    - AutoMLOps mode
    - future distributed or dashboard-triggered execution modes

Why This Module Exists
----------------------
Before this module, detailed run_map.jsonl updates were mostly embedded inside
AutoSearch / OptunaLite. That made AutoSearch resumable and observable, but
plain SCC and AutoMLOps modes could miss the same detailed mapping updates.

This module fixes that by giving every orchestrator the same lifecycle calls:

    - sync_run_map_before_planning()
    - start_run_map_entry()
    - complete_run_map_entry()
    - fail_run_map_entry()
    - interrupt_run_map_entry()
    - verify_run_map_status()

Design Contract
---------------
- model_hash is the durable identity.
- model_name is the human-readable run name.
- sampled is the compact reproducibility fingerprint for the run.
- run_map.jsonl is project-global:

      <project_root>/artifacts/run_map.jsonl

- This module does not replace run_registry.json or test_registry.json.
- This module does not decide training success/failure.
- This module only records lifecycle state into run_map.jsonl.

Status Contract
---------------
The lifecycle status values written here are normalized uppercase strings:

    IN_PROGRESS
    DONE
    FAILED
    INTERRUPTED

These are compatible with run_map.py, which normalizes statuses internally for
resume planning and retry prioritization.

Circular Import Rule
--------------------
Do not import sweep_engine.py here.

sweep_engine.py will later import this module. If this module imports
sweep_engine.py, SCC will risk circular imports.

For this reason, this file reads run_registry.json directly when needed for
syncing run_map.jsonl before planning.

Author: Shraddha Mishra
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from services.side_channel_core.utils.run_map import (
    DEFAULT_ARTIFACTS_ROOT_TEMPLATE,
    load_run_map,
    resolve_artifacts_root_template_from_cfg,
    resolve_run_map_path,
    sync_run_map_status_from_run_registry,
    upsert_run_map_entry,
)
from services.side_channel_core.utils.seal import strip as strip_seal

Json = Dict[str, Any]


RUN_REGISTRY_FILENAME = "run_registry.json"


# =============================================================================
# Basic helpers
# =============================================================================


def _norm(value: Any) -> str:
    """
    Normalize arbitrary values into stripped strings.
    """
    return str(value or "").strip()


def _norm_env(value: Any) -> str:
    """
    Normalize runtime environment.

    Defaults to ENV from the shell, then dev.
    """
    env = _norm(value)
    if env:
        return env

    env = _norm(os.environ.get("ENV"))
    if env:
        return env

    return "dev"


def _project_root_from_cfg(cfg: Mapping[str, Any]) -> Path:
    """
    Resolve project root from a config mapping.

    Expected:
        cfg["project"]["root"]

    Returns:
        Absolute project root path.
    """
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")

    project = cfg.get("project")
    if not isinstance(project, Mapping):
        raise ValueError("cfg['project'] must be a mapping")

    root = project.get("root")
    if not root:
        raise ValueError("cfg['project']['root'] is required")

    return Path(str(root)).expanduser().resolve()


def _env_from_cfg(cfg: Mapping[str, Any]) -> str:
    """
    Resolve environment from config, falling back to ENV/dev.
    """
    return _norm_env(cfg.get("env"))


def _model_id_from_cfg_or_value(
    cfg: Mapping[str, Any],
    model_id: Optional[Any],
) -> Optional[str]:
    """
    Prefer explicit model_id, then cfg['model_id'].

    Returns None if no usable model_id exists.
    """
    if model_id is not None and _norm(model_id):
        return _norm(model_id)

    cfg_model_id = cfg.get("model_id")
    if cfg_model_id is not None and _norm(cfg_model_id):
        return _norm(cfg_model_id)

    return None


def _artifacts_root_template_from_cfg(cfg: Mapping[str, Any]) -> str:
    """
    Resolve artifacts_root template from config.

    Falls back to run_map.py's default template.
    """
    try:
        return resolve_artifacts_root_template_from_cfg(cfg)
    except Exception:
        return DEFAULT_ARTIFACTS_ROOT_TEMPLATE


def _read_json_file(path: Path) -> Any:
    """
    Safely read a JSON document.

    Returns None on missing/invalid files.
    """
    if not path.is_file():
        return None

    from services.side_channel_core.utils.seal import check as check_seal

    check_seal(path)  # utils/seal.py: a changed registry raises TamperedArtifactError, never read as data
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None

    if not text:
        return None

    try:
        return strip_seal(json.loads(text))  # utils/seal.py: no seal key among the entries
    except Exception:
        return None


def _load_run_registry_direct(project_root: str | Path) -> Dict[str, Any]:
    """
    Load run_registry.json without importing sweep_engine.py.

    This avoids circular imports because sweep_engine.py will later import this
    lifecycle logger.

    Returns:
        Dict registry if present and valid, otherwise {}.
    """
    root = Path(project_root).expanduser().resolve()
    path = root / "artifacts" / RUN_REGISTRY_FILENAME

    data = _read_json_file(path)
    if isinstance(data, dict):
        return data

    return {}


def _compact_mapping(
    value: Any,
    *,
    max_depth: Optional[int] = 3,
) -> Any:
    """
    Create a compact, JSON-safe representation.

    max_depth=None means unlimited depth. Use this only for trusted
    reproducibility-critical config blocks like SCC model architecture.
    """
    if max_depth is not None and max_depth <= 0:
        if isinstance(value, Mapping):
            return "{...}"
        if isinstance(value, list):
            return "[...]"
        return copy.deepcopy(value)

    next_depth = None if max_depth is None else max_depth - 1

    if isinstance(value, Mapping):
        out: Json = {}
        for key, child in value.items():
            key_s = str(key)

            if key_s in {
                "paths_abs",
                "visualization",
                "logging",
                "runtime",
                "runtime_contract",
                "data",
                "dataset",
                "dataset_name",
                "clearml",
                "secrets",
                "credentials",
            }:
                continue

            out[key_s] = _compact_mapping(child, max_depth=next_depth)

        return out

    if isinstance(value, list):
        # Do not truncate architecture layers. They are reproducibility metadata.
        return [
            _compact_mapping(item, max_depth=next_depth)
            for item in value
        ]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return copy.deepcopy(value)

    return str(value)


# =============================================================================
# Sampled fingerprint builder
# =============================================================================


def build_sampled_fingerprint_from_cfg(
    cfg: Mapping[str, Any],
    *,
    mode: str,
    sampled: Optional[Mapping[str, Any]] = None,
    include_model_block: bool = True,
    include_trainer_block: bool = True,
    include_sca_block: bool = True,
    include_dataset_alias: bool = True,
) -> Json:
    """
    Build a compact sampled fingerprint for run_map.jsonl.

    AutoSearch should pass its actual sampled point.

    SCC and AutoMLOps may not always have a clean sampled point, so this helper
    builds a compact fingerprint from the effective run config.

    Important
    ---------
    Do not store the full cfg in run_map.jsonl.

    Full configs can be large, unstable, and may include absolute paths,
    runtime-only fields, secrets, dashboard config, or dataset metadata.

    Parameters
    ----------
    cfg:
        Effective run config.

    mode:
        Execution mode label, for example:
            "auto_search"
            "scc"
            "auto_mlops"

    sampled:
        Optional explicit sampled parameter mapping. If provided, this is copied
        into the fingerprint under "sampled".

    include_model_block:
        Include compact cfg["model"].

    include_trainer_block:
        Include compact cfg["trainer"].

    include_sca_block:
        Include compact cfg["sca"].

    include_dataset_alias:
        Include dataset/dataset_name/dataset_alias if present.

    Returns
    -------
    dict
        Compact deterministic fingerprint suitable for run_map.sampled.
    """
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")

    out: Json = {
        "mode": str(mode),
    }

    model_id = cfg.get("model_id")
    if model_id is not None and _norm(model_id):
        out["model_id"] = _norm(model_id)

    if include_dataset_alias:
        for key in ("dataset", "dataset_name", "dataset_alias"):
            value = cfg.get(key)
            if value is not None and _norm(value):
                out[key] = _norm(value)

    if sampled is not None:
        if not isinstance(sampled, Mapping):
            raise TypeError("sampled must be a mapping when provided")
        out["sampled"] = _compact_mapping(dict(sampled), max_depth=None)

    # Every block at full depth: the fingerprint is the run's sampled_key, and a
    # level cut to "{...}" (trainer.optimizer.scheduler.args) would give runs that
    # differ there one key.
    if include_trainer_block and isinstance(cfg.get("trainer"), Mapping):
        out["trainer"] = _compact_mapping(cfg["trainer"], max_depth=None)

    if include_sca_block and isinstance(cfg.get("sca"), Mapping):
        out["sca"] = _compact_mapping(cfg["sca"], max_depth=None)

    if include_model_block and "model" in cfg:
        out["model"] = _compact_mapping(cfg.get("model"), max_depth=None)

    return out


# =============================================================================
# Sync and verification
# =============================================================================


def sync_run_map_before_planning(
    project_root: str | Path,
    *,
    run_registry: Optional[Mapping[str, Any]] = None,
) -> Dict[str, int]:
    """
    Sync run_map.jsonl statuses from run_registry.json before planning runs.

    This is useful before AutoSearch/SCC/AutoMLOps decides what to run next.

    Example
    -------
    If the previous process died while a run was IN_PROGRESS, startup repair or
    registry inspection may mark the run as INTERRUPTED in run_registry.json.
    This function copies that status into run_map.jsonl, making the sampled point
    retryable.

    Parameters
    ----------
    project_root:
        Project root path.

    run_registry:
        Optional already-loaded run registry. If omitted, this function loads
        <project_root>/artifacts/run_registry.json directly.

    Returns
    -------
    dict
        Example:
            {"changed": 2}
    """
    root = Path(project_root).expanduser().resolve()

    if run_registry is None:
        registry_dict = _load_run_registry_direct(root)
    else:
        registry_dict = dict(run_registry)

    return sync_run_map_status_from_run_registry(root, registry_dict)


def verify_run_map_status(
    *,
    project_root: str | Path,
    model_hash: str,
    expected_status: str,
    raise_on_mismatch: bool = True,
) -> bool:
    """
    Verify that run_map.jsonl contains the expected status for model_hash.

    This is useful after writing IN_PROGRESS, because failed persistence should
    be detected before launching expensive training.

    Parameters
    ----------
    project_root:
        Project root path.

    model_hash:
        Durable model hash.

    expected_status:
        Expected status string. Comparison is case-insensitive.

    raise_on_mismatch:
        If True, raise RuntimeError when verification fails.

    Returns
    -------
    bool
        True if the status matches, False otherwise.
    """
    root = Path(project_root).expanduser().resolve()
    model_hash_s = _norm(model_hash)
    expected = _norm(expected_status).upper()

    run_map = load_run_map(root)
    rec = run_map.get(model_hash_s) or {}

    actual = _norm(rec.get("status")).upper()

    ok = actual == expected

    if not ok and raise_on_mismatch:
        raise RuntimeError(
            "run_map status verification failed: "
            f"model_hash={model_hash_s!r}, "
            f"expected={expected!r}, actual={actual!r}, "
            f"path={resolve_run_map_path(root)}"
        )

    return ok


# =============================================================================
# Lifecycle write API
# =============================================================================


def write_run_map_entry(
    *,
    cfg: Mapping[str, Any],
    model_hash: str,
    model_name: str,
    status: str,
    model_id: Optional[Any] = None,
    sampled: Optional[Mapping[str, Any]] = None,
    mode: str = "unknown",
    run_dir: Optional[str | Path] = None,
    trial_index: Optional[int] = None,
    total_trials: Optional[int] = None,
    total_remaining_trials: Optional[int] = None,
    run_session_id: Optional[str] = None,
    selection_source: Optional[str] = None,
    reason: Optional[str] = None,
    verify: bool = False,
) -> Json:
    """
    Generic lifecycle writer for run_map.jsonl.

    Most callers should prefer the semantic wrappers:

        start_run_map_entry()
        complete_run_map_entry()
        fail_run_map_entry()
        interrupt_run_map_entry()

    Parameters
    ----------
    cfg:
        Effective run config containing cfg["project"]["root"].

    model_hash:
        Durable run identity.

    model_name:
        Human-readable model/run name.

    status:
        Lifecycle status to write.

    model_id:
        Optional configured model_id.

    sampled:
        Compact sampled fingerprint. If omitted, a compact fingerprint is built
        from cfg and mode.

    mode:
        Execution mode label used only when sampled is omitted.

    run_dir:
        Optional concrete run artifact directory.

    trial_index:
        Optional 1-based progress index.

    total_trials:
        Optional total planned trials/runs.

    selection_source:
        Source planner label, for example:
            finite_bo
            finite_rs
            scc_sweep
            auto_mlops

    reason:
        Optional failure/interruption reason.

    verify:
        If True, reload run_map.jsonl and assert the written status exists.

    Returns
    -------
    dict
        The written run_map record.
    """
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")

    root = _project_root_from_cfg(cfg)

    model_hash_s = _norm(model_hash)
    model_name_s = _norm(model_name)
    status_s = _norm(status).upper()

    if not model_hash_s:
        raise ValueError("model_hash is required")

    if not model_name_s:
        raise ValueError("model_name is required")

    if not status_s:
        raise ValueError("status is required")

    sampled_payload: Mapping[str, Any]
    if sampled is not None:
        if not isinstance(sampled, Mapping):
            raise TypeError("sampled must be a mapping when provided")
        sampled_payload = dict(sampled)
    elif isinstance(cfg.get("sampled"), Mapping):
        # The run's sweep point as the planner recorded it (sweep_engine._scc_sampled_payload,
        # kept in the params_out snapshot): one sampled_key for every record of the run.
        sampled_payload = dict(cfg["sampled"])
    else:
        sampled_payload = build_sampled_fingerprint_from_cfg(
            cfg,
            mode=mode,
            sampled=None,
        )
    runtime = cfg.get("runtime") or {}
    if not isinstance(runtime, Mapping):
        runtime = {}

    if total_remaining_trials is None:
        total_remaining_trials = runtime.get("total_remaining_trials")
    if run_session_id is None:
        run_session_id = runtime.get("run_session_id")
    rec = upsert_run_map_entry(
        root,
        model_hash=model_hash_s,
        model_name=model_name_s,
        model_id=_model_id_from_cfg_or_value(cfg, model_id),
        sampled=sampled_payload,
        status=status_s,
        env=_env_from_cfg(cfg),
        artifacts_root_template=_artifacts_root_template_from_cfg(cfg),
        run_dir=run_dir,
        trial_index=trial_index,
        total_trials=total_trials,
        total_remaining_trials=total_remaining_trials,
        run_session_id=run_session_id,
        selection_source=selection_source,
        reason=reason,
    )

    if verify:
        verify_run_map_status(
            project_root=root,
            model_hash=model_hash_s,
            expected_status=status_s,
            raise_on_mismatch=True,
        )

    return rec
def start_run_map_entry(
    *,
    cfg: Mapping[str, Any],
    model_hash: str,
    model_name: str,
    model_id: Optional[Any] = None,
    sampled: Optional[Mapping[str, Any]] = None,
    mode: str = "unknown",
    run_dir: Optional[str | Path] = None,
    trial_index: Optional[int] = None,
    total_remaining_trials: Optional[int] = None,
    run_session_id: Optional[str] = None,
    total_trials: Optional[int] = None,
    selection_source: Optional[str] = None,
    verify: bool = True,
) -> Json:
    """
    Mark a run as IN_PROGRESS in run_map.jsonl.

    Use this immediately before launching training/testing work.

    verify defaults to True because an expensive training job should not start
    if run_map.jsonl cannot confirm that the run was recorded.
    """
    return write_run_map_entry(
        cfg=cfg,
        model_hash=model_hash,
        model_name=model_name,
        model_id=model_id,
        sampled=sampled,
        status="IN_PROGRESS",
        mode=mode,
        run_dir=run_dir,
        trial_index=trial_index,
        total_remaining_trials=total_remaining_trials,
        run_session_id=run_session_id,
        total_trials=total_trials,
        selection_source=selection_source,
        reason=None,
        verify=verify,
    )


def complete_run_map_entry(
    *,
    cfg: Mapping[str, Any],
    model_hash: str,
    model_name: str,
    model_id: Optional[Any] = None,
    sampled: Optional[Mapping[str, Any]] = None,
    mode: str = "unknown",
    run_dir: Optional[str | Path] = None,
    trial_index: Optional[int] = None,
    total_remaining_trials: Optional[int] = None,
    run_session_id: Optional[str] = None,
    total_trials: Optional[int] = None,
    selection_source: Optional[str] = None,
    verify: bool = False,
) -> Json:
    """
    Mark a run as DONE in run_map.jsonl.

    Use this only after the training/testing/orchestration command completed
    successfully.
    """
    return write_run_map_entry(
        cfg=cfg,
        model_hash=model_hash,
        model_name=model_name,
        model_id=model_id,
        sampled=sampled,
        status="DONE",
        mode=mode,
        run_dir=run_dir,
        trial_index=trial_index,
        total_remaining_trials=total_remaining_trials,
        run_session_id=run_session_id,
        total_trials=total_trials,
        selection_source=selection_source,
        reason=None,
        verify=verify,
    )


def fail_run_map_entry(
    *,
    cfg: Mapping[str, Any],
    model_hash: str,
    model_name: str,
    reason: str,
    model_id: Optional[Any] = None,
    sampled: Optional[Mapping[str, Any]] = None,
    mode: str = "unknown",
    run_dir: Optional[str | Path] = None,
    trial_index: Optional[int] = None,
    total_remaining_trials: Optional[int] = None,
    run_session_id: Optional[str] = None,
    total_trials: Optional[int] = None,
    selection_source: Optional[str] = None,
    verify: bool = False,
) -> Json:
    """
    Mark a run as FAILED in run_map.jsonl.

    Use this inside exception handlers after the authoritative registry has also
    been updated, or when the run driver detects an execution failure.
    """
    reason_s = _norm(reason) or "unknown_failure"

    return write_run_map_entry(
        cfg=cfg,
        model_hash=model_hash,
        model_name=model_name,
        model_id=model_id,
        sampled=sampled,
        status="FAILED",
        mode=mode,
        run_dir=run_dir,
        trial_index=trial_index,
        total_remaining_trials=total_remaining_trials,
        run_session_id=run_session_id,
        total_trials=total_trials,
        selection_source=selection_source,
        reason=reason_s,
        verify=verify,
    )


def interrupt_run_map_entry(
    *,
    cfg: Mapping[str, Any],
    model_hash: str,
    model_name: str,
    reason: str = "interrupted",
    model_id: Optional[Any] = None,
    sampled: Optional[Mapping[str, Any]] = None,
    mode: str = "unknown",
    run_dir: Optional[str | Path] = None,
    trial_index: Optional[int] = None,
    total_remaining_trials: Optional[int] = None,
    run_session_id: Optional[str] = None,
    total_trials: Optional[int] = None,
    selection_source: Optional[str] = None,
    verify: bool = False,
) -> Json:
    """
    Mark a run as INTERRUPTED in run_map.jsonl.

    This is useful for graceful signal handling or startup repair.
    """
    reason_s = _norm(reason) or "interrupted"

    return write_run_map_entry(
        cfg=cfg,
        model_hash=model_hash,
        model_name=model_name,
        model_id=model_id,
        sampled=sampled,
        status="INTERRUPTED",
        mode=mode,
        run_dir=run_dir,
        trial_index=trial_index,
        total_remaining_trials=total_remaining_trials,
        run_session_id=run_session_id,
        total_trials=total_trials,
        selection_source=selection_source,
        reason=reason_s,
        verify=verify,
    )
