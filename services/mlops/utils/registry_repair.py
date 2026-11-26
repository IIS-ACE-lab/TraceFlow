# services/mlops/utils/registry_repair.py
"""
registry_repair.py
==================

Startup/interrupt repair utility for AutoMLOps registry state.

Purpose
-------
This module repairs stale registry entries left behind when Docker/Compose/host
termination kills the Python training process before it can cleanly update
status.

Rules
-----
- run_registry.json:
    NEW / queued / running / in_progress / in_process -> INTERRUPTED

- test_registry.json:
    Only update existing stale test entries.
    Do NOT create fake test entries if testing never started.

This module is intentionally outside:
- scc_bridge_lite.py
- optuna_lite.py

Those files must stay fast and focused.

Author: Shraddha Mishra
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from services.side_channel_core.utils.sweep_engine import (
    load_registry,
    save_registry,
)
from services.side_channel_core.utils.test_registry import (
    load_test_registry,
    save_test_registry,
)
from services.side_channel_core.utils.run_map import (
    sync_run_map_status_from_run_registry,
)

STALE_STATUSES = {
    "new",
    "queued",
    "pending",
    "running",
    "in_progress",
    "in_process",
}


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _status(value: Any) -> str:
    return _norm(value).lower()


def _env_matches(entry: Dict[str, Any], env: Optional[str]) -> bool:
    if not env:
        return True

    entry_env = _norm(entry.get("env"))
    return entry_env == _norm(env)


def _mark_interrupted(entry: Dict[str, Any], reason: str) -> Dict[str, Any]:
    entry["status"] = "INTERRUPTED"
    entry["interrupt_reason"] = str(reason or "stale_repair")
    entry["status_reason"] = entry["interrupt_reason"]
    return entry


def repair_stale_auto_search_runs(
    *,
    project_root: str | Path,
    env: Optional[str] = None,
    reason: str = "startup_repair",
) -> Dict[str, int]:
    """
    Repair stale AutoMLOps registry entries.

    Parameters
    ----------
    project_root:
        Project root containing artifacts/run_registry.json and
        artifacts/test_registry.json.

    env:
        Optional env filter. If provided, only entries matching this env are repaired.

    reason:
        Human-readable reason stored on repaired entries.

    Returns
    -------
    dict
        Counts of repaired run/test entries.
    """
    root = Path(project_root).expanduser().resolve()

    run_registry = load_registry(root)
    test_registry = load_test_registry(root)

    run_changed = 0
    test_changed = 0

    # Repair train/run entries.
    for model_hash, entry in list(run_registry.items()):
        if not isinstance(entry, dict):
            continue

        if not _env_matches(entry, env):
            continue

        if _status(entry.get("status")) not in STALE_STATUSES:
            continue

        run_registry[model_hash] = _mark_interrupted(entry, reason)
        run_changed += 1

    # Repair only existing test entries.
    # Do NOT create test entries here; testing may never have started.
    for model_hash, entry in list(test_registry.items()):
        if not isinstance(entry, dict):
            continue

        if not _env_matches(entry, env):
            continue

        if _status(entry.get("status")) not in STALE_STATUSES:
            continue

        test_registry[model_hash] = _mark_interrupted(entry, reason)
        test_changed += 1

    if run_changed:
        save_registry(run_registry, root)

    if test_changed:
        save_test_registry(test_registry, root)

    # Always sync run_map from run_registry.
    # Important: this must run even if run_changed == 0 because another repair
    # call may already have updated run_registry, while run_map is still stale.
    run_map_sync = sync_run_map_status_from_run_registry(
        project_root=root,
        run_registry=run_registry,
    )

    return {
        "run_changed": run_changed,
        "test_changed": test_changed,
        "run_map_changed": int(run_map_sync.get("changed", 0)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair stale AutoMLOps run/test registry entries."
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="Project root containing artifacts/run_registry.json.",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Optional env filter, e.g. dev/staging/prod.",
    )
    parser.add_argument(
        "--reason",
        default="startup_repair",
        help="Reason stored in repaired registry entries.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    result = repair_stale_auto_search_runs(
        project_root=args.project_root,
        env=args.env,
        reason=args.reason,
    )

    print(
        json.dumps(
            {
                "event": "registry_repair",
                "project_root": str(Path(args.project_root).expanduser().resolve()),
                "env": args.env,
                "reason": args.reason,
                **result,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
