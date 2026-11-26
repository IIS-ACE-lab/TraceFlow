# services/side_channel_core/utils/lifecycle_outcome.py
"""
Key-recovery outcome of a run in the lifecycle registries.

artifacts/run_registry.json carries the outcome of the training run's
validation and artifacts/test_registry.json the outcome of the test, so both
answer "was the key recovered, and by what" without reading the run folder.
The fields are patched in once the key-recovery reports are final
(key_recovery_helper._record_key_recovery_registries).

Author: Shraddha Mishra
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from utils.atomic_io import atomic_write_text
from utils.logger import get_logger
from utils.seal import load_json

logger = get_logger("LifecycleOutcome")

# Resolved-config echoes that belong to the run's params_out snapshot.
RETIRED_LIFECYCLE_FIELDS = (
    "candidate_ranking_mode",
    "key_enumeration_mode",
)

LIFECYCLE_REGISTRY_FIELDS = (
    "key_recovery",
    "key_recovered_through",
    "key_rank_reported",
    "attack_byte",
    "key_recovered",
    "true_key",
    "recovered_key",
    "recovered_key_rank",
    # Cross-validation runs: the outcome of every fold (held-out part, final epoch).
    "fold_key_recovery",
)


def begin_attempt(entry: dict, started_at: str | None) -> None:
    """
    A new attempt of the run (a first run, a retry, a resume, a FORCE retrain):
    its own start time (none yet for a test that is only queued), no finish
    time, and nothing an earlier attempt recorded (its key-recovery outcome,
    the legacy `reason`), so an attempt that fails never shows the outcome of
    the one before.
    """
    if started_at:
        entry["started_at"] = started_at
    else:
        entry.pop("started_at", None)
    entry.pop("finished_at", None)
    entry.pop("reason", None)
    for field in LIFECYCLE_REGISTRY_FIELDS:
        entry.pop(field, None)


def _coerce_hash(value: Any) -> str:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value):08x}"
    out = "" if value is None else str(value).strip()
    if out.lower() in {"", "nan", "none", "null", "na"}:
        raise ValueError("the lifecycle registries are keyed by a non-empty model_hash")
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return str(value)


def update_lifecycle_registry_outcome(
    *,
    registry_path: Path,
    model_hash: Any,
    fields: dict,
) -> None:
    """Patch the key-recovery outcome into artifacts/<name>_registry.json."""
    registry_path = Path(registry_path)
    model_hash = _coerce_hash(model_hash)
    outcome = {
        key: _jsonable(value)
        for key, value in (fields or {}).items()
        if key in LIFECYCLE_REGISTRY_FIELDS
    }

    if not outcome or not registry_path.is_file():
        return

    try:
        entries = load_json(registry_path)  # utils/seal.py: verified, seal key dropped
    except Exception as exc:
        logger.warning("Could not read %s: %s", registry_path, exc)
        return

    entry = entries.get(model_hash) if isinstance(entries, dict) else None

    if not isinstance(entry, dict):
        logger.warning("No %s entry for hash=%s", registry_path.name, model_hash)
        return

    entry.update(outcome)

    for retired in RETIRED_LIFECYCLE_FIELDS:
        entry.pop(retired, None)

    atomic_write_text(registry_path, json.dumps(entries, indent=2) + "\n")


__all__ = ["LIFECYCLE_REGISTRY_FIELDS", "begin_attempt", "update_lifecycle_registry_outcome"]
