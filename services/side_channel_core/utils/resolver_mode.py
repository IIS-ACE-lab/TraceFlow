# services/side_channel_core/utils/resolver_mode.py
"""
resolver_mode.py
================

Resolver-mode selection helpers for Side-Channel Core.

Purpose
-------
This module centralizes the logic that decides whether a run should resolve its
model contract in:

    - "auto_search" mode
    - "scc" mode
    - no mode at all because the contract was already resolved upstream

Why This Module Exists
----------------------
AutoSearch / OptunaLite may resolve architecture_args.search_space into concrete
architecture_args.layers before SCC training starts.

After that happens, the model block may look like a normal SCC model because it
has concrete layers. If train.py calls the resolver-mode selector again without
checking whether the contract was already resolved upstream, it can incorrectly
log/select SCC mode even though the run came from AutoSearch.

This module prevents that by separating two decisions:

    1. Should train.py skip resolve_model_contract() entirely?
    2. If not skipped, which resolver mode should be used?

Design Contract
---------------
- AutoSearch owns materialization of architecture_args.search_space.
- SCC training consumes an already-resolved contract when
  cfg["runtime_contract"]["resolved"] is True.
- Normal SCC runs still use "scc" when concrete architecture_args.layers exist.
- Configs that contain neither enabled search_space nor concrete layers fail
  loudly.

Author: Shraddha Mishra
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from utils.sweep_engine import load_registry
from services.side_channel_core.utils.run_map import load_run_map


@dataclass(frozen=True)
class ResolverDecision:
    """
    Resolver decision returned to train.py.

    Attributes
    ----------
    mode:
        Resolver mode to use when should_resolve=True.
        Usually "auto_search" or "scc".

    should_resolve:
        If False, train.py should skip resolve_model_contract() and use cfg as-is.

    reason:
        Human-readable reason for logging/debugging.
    """

    mode: str
    should_resolve: bool
    reason: str


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _norm(value).lower()


def _find_model_entry(cfg: Mapping[str, Any], model_id: Optional[str]) -> Mapping[str, Any]:
    """
    Find the selected model entry from cfg["model"].

    Supports:
    - cfg["model"] as a list of model entries
    - cfg["model"] as a single dict model entry

    Selection order:
    1. explicit model_id
    2. cfg["model_id"]
    3. if only one model exists, select it
    """
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")

    selected_id = _norm(model_id) or _norm(cfg.get("model_id"))
    model_block = cfg.get("model")

    if isinstance(model_block, Mapping):
        return model_block

    if not isinstance(model_block, list) or not model_block:
        raise ValueError("cfg['model'] must be a non-empty dict or list")

    candidates = [entry for entry in model_block if isinstance(entry, Mapping)]

    if not candidates:
        raise ValueError("cfg['model'] does not contain any valid model entries")

    if selected_id:
        for entry in candidates:
            if _norm(entry.get("model_id")) == selected_id:
                return entry

        available = [_norm(entry.get("model_id")) for entry in candidates]
        raise ValueError(
            f"model_id={selected_id!r} not found in cfg['model']. "
            f"Available model_id values: {available}"
        )

    if len(candidates) == 1:
        return candidates[0]

    available = [_norm(entry.get("model_id")) for entry in candidates]
    raise ValueError(
        "Multiple model entries exist but no model_id was selected. "
        f"Available model_id values: {available}"
    )


def _model_has_enabled_search_space(model_entry: Mapping[str, Any]) -> bool:
    """
    Return True if selected model has architecture_args.search_space.enabled=true.
    """
    if not isinstance(model_entry, Mapping):
        return False

    arch_args = model_entry.get("architecture_args") or {}
    if not isinstance(arch_args, Mapping):
        return False

    search_space = arch_args.get("search_space") or {}
    if not isinstance(search_space, Mapping):
        return False

    return bool(search_space.get("enabled", False))


def _model_has_concrete_layers(model_entry: Mapping[str, Any]) -> bool:
    """
    Return True if selected model has concrete architecture_args.layers.
    """
    if not isinstance(model_entry, Mapping):
        return False

    arch_args = model_entry.get("architecture_args") or {}
    if not isinstance(arch_args, Mapping):
        return False

    layers = arch_args.get("layers")

    return isinstance(layers, list) and len(layers) > 0


def is_auto_search_runtime(cfg: Mapping[str, Any]) -> bool:
    """
    Detect whether the active run came from AutoSearch / OptunaLite.

    This checks runtime metadata injected by the AutoSearch bridge and the
    AUTO_SEARCH environment variable.
    """
    runtime = cfg.get("runtime") or {}
    if not isinstance(runtime, Mapping):
        runtime = {}

    return (
        _lower(runtime.get("execution_source")) == "auto_search"
        or _lower(runtime.get("driver")) == "optuna_lite"
        or _norm(os.environ.get("AUTO_SEARCH")) == "1"
    )


def get_runtime_contract(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Return cfg["runtime_contract"] as a mapping.
    """
    runtime_contract = cfg.get("runtime_contract") or {}
    if isinstance(runtime_contract, Mapping):
        return runtime_contract
    return {}


def choose_resolver_mode(
    cfg: Mapping[str, Any],
    model_id: Optional[str],
    logger,
) -> str:
    """
    Choose resolver mode based on the selected model block.

    Rules
    -----
    1. If selected model has architecture_args.search_space.enabled=true:
       use auto_search.
    2. Else if selected model already has concrete architecture_args.layers:
       use scc.
    3. Else fail with a clear config error.

    Important
    ---------
    This function chooses a mode only. It does not decide whether train.py should
    skip resolution entirely. For that, use decide_resolver_action().
    """
    selected_model = _find_model_entry(cfg, model_id)

    selected_model_id = selected_model.get("model_id", model_id)
    has_search_space = _model_has_enabled_search_space(selected_model)
    has_layers = _model_has_concrete_layers(selected_model)

    if has_search_space:
        logger.info(
            "Resolver mode selected: auto_search "
            "(model_id=%s has architecture_args.search_space.enabled=true)",
            selected_model_id,
        )
        return "auto_search"

    if has_layers:
        logger.info(
            "Resolver mode selected: scc "
            "(model_id=%s already has concrete architecture_args.layers)",
            selected_model_id,
        )
        return "scc"

    global_auto = cfg.get("auto_search") or {}
    hard_replace = None
    if isinstance(global_auto, Mapping):
        hard_replace = global_auto.get("hard_replace_params")

    raise ValueError(
        f"Cannot resolve model_id={selected_model_id!r}. "
        "Selected model has neither enabled architecture_args.search_space "
        "nor concrete architecture_args.layers. "
        f"auto_search.hard_replace_params={hard_replace!r}. "
        "Fix config generation so each selected model has either "
        "search_space.enabled=true or concrete layers."
    )


def decide_resolver_action(
    cfg: Mapping[str, Any],
    model_id: Optional[str],
    logger,
) -> ResolverDecision:
    """
    Decide whether train.py should resolve the model contract.

    This is the function train.py should call.

    Behavior
    --------
    - If runtime_contract.resolved is True:
        skip resolve_model_contract(); use cfg as-is.

    - Else if runtime metadata indicates AutoSearch:
        force mode="auto_search".

    - Else:
        choose mode from the selected model block.

    Returns
    -------
    ResolverDecision
        Small immutable decision object.
    """
    runtime_contract = get_runtime_contract(cfg)

    if runtime_contract.get("resolved") is True:
        upstream_mode = _norm(
            runtime_contract.get("mode")
            or runtime_contract.get("resolver_mode")
            or (cfg.get("runtime") or {}).get("execution_source")
            or "upstream"
        )

        reason = (
            f"Model contract already resolved upstream by {upstream_mode}; "
            "skipping second resolve_model_contract()."
        )

        logger.info(reason)

        return ResolverDecision(
            mode=upstream_mode if upstream_mode else "upstream",
            should_resolve=False,
            reason=reason,
        )

    if is_auto_search_runtime(cfg):
        reason = (
            "Resolver mode selected: auto_search "
            "(runtime execution_source/driver indicates OptunaLite AutoSearch)"
        )

        logger.info(reason)

        return ResolverDecision(
            mode="auto_search",
            should_resolve=True,
            reason=reason,
        )

    mode = choose_resolver_mode(cfg, model_id, logger)

    return ResolverDecision(
        mode=mode,
        should_resolve=True,
        reason=f"Resolver mode selected by model config: {mode}",
    )
def _scoped_run_map_records(project_root: Path, *, env: str, model_id: str | None) -> dict:
    run_map = load_run_map(project_root)

    out = {}
    for h, rec in run_map.items():
        if not isinstance(rec, dict):
            continue

        if env and str(rec.get("env") or "").strip() != str(env).strip():
            continue

        if model_id and str(rec.get("model_id") or "").strip() != str(model_id).strip():
            continue

        out[str(h)] = rec

    return out


def _status_upper(rec: dict) -> str:
    return str((rec or {}).get("status") or "").strip().upper()


def _derive_attempt_window(
    project_root: Path,
    *,
    env: str,
    model_id: str | None,
    run_session_id: str | None = None,
) -> tuple[int, int, int]:
    """
    Returns:
        next_trial_index:
            Global progress index = DONE + recoverable/in-progress + 1 for next new run.

        total_remaining_trials:
            Full sweep total minus DONE count.

        total_trials:
            Immutable full SCC sweep size.
    """
    records = _scoped_run_map_records(project_root, env=env, model_id=model_id)

    done = 0
    active_or_recoverable = 0
    total_trials = 0

    for rec in records.values():
        st = _status_upper(rec)

        if st == "DONE":
            done += 1
        elif st in {"FAILED", "INTERRUPTED", "IN_PROGRESS", "IN_PROCESS", "RUNNING", "QUEUED"}:
            active_or_recoverable += 1

        for key in ("total_trials", "trial_total"):
            try:
                total_trials = max(total_trials, int(rec.get(key) or 0))
            except Exception:
                pass

    if total_trials <= 0:
        train_reg = load_registry(project_root)
        for rec in train_reg.values():
            if not isinstance(rec, dict):
                continue
            if env and str(rec.get("env") or "").strip() != str(env).strip():
                continue
            if model_id and str(rec.get("model_id") or "").strip() != str(model_id).strip():
                continue

            for key in ("total_trials", "trial_total"):
                try:
                    total_trials = max(total_trials, int(rec.get(key) or 0))
                except Exception:
                    pass

    def _next_session_trial_index() -> int:
        if not run_session_id:
            return 1

        max_session_idx = 0

        for rec in records.values():
            if str(rec.get("run_session_id") or "").strip() != str(run_session_id).strip():
                continue

            try:
                max_session_idx = max(max_session_idx, int(rec.get("trial_index") or 0))
            except Exception:
                pass

        return max_session_idx + 1


    if total_trials <= 0:
        return int(_next_session_trial_index()), 0, 0

    next_trial_index = _next_session_trial_index()
    total_remaining_trials = max(total_trials - done, 0)

    return int(next_trial_index), int(total_remaining_trials), int(total_trials)
