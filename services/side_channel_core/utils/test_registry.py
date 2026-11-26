# utils/test_registry.py
"""
Test Registry (stable env + atomic writes + timestamps + ordering)

Loads/saves a test registry JSON under:
  <project_root>/artifacts/test_registry.json

CONTRACT:
- Every entry must have a stable "env" field.
- If env is missing, infer it from path like "DLSCA_ASCAD/artifacts/<env>/..."
- Every entry carries the run's "model_name" and "path" (as in
  run_registry.json), filled in from the run when missing.
- Every entry carries the test's key-recovery outcome: key_recovered,
  true_key, recovered_key, recovered_key_rank; null until a finished test
  records them (key_recovery_helper._record_key_recovery_registries).
- A test of a model whose config leaves the test traces unattacked
  (perform_attack.test.attack_times: 0) is not run: its entry is SKIPPED, a
  finished status, with the reason (skip_unattacked_test).
- FAILED / INTERRUPTED / SKIPPED entries carry "status_reason"; it is dropped once the
  status moves on.
- Writes are atomic (utils/atomic_io.py).
- Per-entry timestamps:
    started_at, finished_at (local timezone ISO8601 with offset)
- JSON order:
    RUNNING first, QUEUED second, then everything else; latest-first inside each group.

Author: Shraddha Mishra
"""
from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from re import sub as re_sub
from typing import Any, Dict, Optional
from services.side_channel_core.utils.atomic_io import atomic_write_text
from services.side_channel_core.utils.run_map import load_run_map
from utils.batch import merge_batches
from utils.lifecycle_outcome import begin_attempt
from utils.logger import get_logger
from services.side_channel_core.utils.timestamps import now_iso
from services.side_channel_core.utils.seal import load_json

logger = get_logger("TestRegistry")

TEST_REGISTRY_FILENAME = "test_registry.json"


# -----------------------------------------------------------------------------
# Time + status helpers
# -----------------------------------------------------------------------------

def _now_local_iso() -> str:
    return now_iso()


def _status_norm(s: str) -> str:
    return (s or "").strip().lower()


def _is_running_status(s: str) -> bool:
    return _status_norm(s) in {"in_progress", "running"}


def _is_queued_status(s: str) -> bool:
    return _status_norm(s) in {"new", "queued", "pending"}


def _is_terminal_status(s: str) -> bool:
    return _status_norm(s) in {"done", "skipped", "failed", "interrupted", "cancelled", "canceled"}


def _needs_reason(s: str) -> bool:
    return _status_norm(s) in {"failed", "interrupted", "cancelled", "canceled", "skipped"}


def _group_rank(status: str) -> int:
    if _is_running_status(status):
        return 0
    if _is_queued_status(status):
        return 1
    return 2


def _entry_latest_ts(entry: Dict[str, Any]) -> str:
    # ISO8601 sorts lexicographically for our use-case
    return str(entry.get("finished_at") or entry.get("started_at") or "").strip()


def _reorder_registry_for_write(registry: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Returns a NEW dict with insertion order:
      - running first (latest first)
      - queued second (latest first)
      - then everything else (latest first)
    """
    items = [(k, v) for k, v in registry.items() if isinstance(v, dict)]

    # stable multi-pass sort
    items.sort(key=lambda kv: _entry_latest_ts(kv[1]), reverse=True)
    items.sort(key=lambda kv: _group_rank(str(kv[1].get("status") or "")))

    out: Dict[str, Dict[str, Any]] = {}
    for k, v in items:
        out[k] = v
    return out


# -----------------------------------------------------------------------------
# Env helpers
# -----------------------------------------------------------------------------

def _env_from_path(p: str) -> str:
    """
    Infer env from paths like:
      - "DLSCA_ASCAD/artifacts/dev/generated_models/..."
      - "/app/projects/DLSCA_ASCAD/artifacts/dev/generated_models/..."
      - "artifacts/dev/generated_models/..."
    """
    if not isinstance(p, str):
        return ""

    marker = "/artifacts/"
    if marker in p:
        rest = p.split(marker, 1)[1]
        return rest.split("/", 1)[0].strip()

    marker2 = "artifacts/"
    if marker2 in p:
        rest = p.split(marker2, 1)[1]
        return rest.split("/", 1)[0].strip()

    return ""


def _canonical_env(
    *,
    env: Optional[str] = None,
    entry: Optional[Dict[str, Any]] = None,
    default: str = "dev",
) -> str:
    """
    Deterministic env resolution (never None):
      1) explicit env argument
      2) entry["env"]
      3) os.getenv("ENV")
      4) infer from entry["path"]
      5) default ("dev")
    """
    v = str(env or "").strip()
    if v:
        return v

    if isinstance(entry, dict):
        v = str(entry.get("env") or "").strip()
        if v:
            return v

    v = str(os.getenv("ENV") or "").strip()
    if v:
        return v

    if isinstance(entry, dict):
        p = entry.get("path")
        if isinstance(p, (str, Path)):
            v = _env_from_path(str(p))
            if v:
                return v

    return default


# -----------------------------------------------------------------------------
# Registry I/O (atomic + normalization)
# -----------------------------------------------------------------------------

def _path_for(project_root: Path) -> Path:
    art = Path(project_root) / "artifacts"
    art.mkdir(exist_ok=True, parents=True)
    return art / TEST_REGISTRY_FILENAME


KEY_RECOVERY_FIELDS = ("key_recovered", "true_key", "recovered_key", "recovered_key_rank")


def _missing_run_fields(entry: Dict[str, Any]) -> bool:
    return any(not str(entry.get(k) or "").strip() for k in ("model_name", "path"))


def _fill_from_run(entry: Dict[str, Any], run: Any) -> None:
    """Fill an empty model_name / path from the hash's run_registry.json entry."""
    if not isinstance(run, dict):
        return
    for key in ("model_name", "path"):
        if not str(entry.get(key) or "").strip() and str(run.get(key) or "").strip():
            entry[key] = run[key]


def _train_registry(project_root: Path) -> Dict[str, Any]:
    from utils.sweep_engine import load_registry as load_train_registry

    return load_train_registry(Path(project_root), heal=False)


def _normalize_entry(info: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """
    In-place normalization:
    - ensure 'path' is project-root-relative when possible
    - ensure 'env' exists (backfill from path / ENV / default)
    - ensure timestamps:
        * running => started_at exists
        * terminal => finished_at exists
    """
    # normalize path to project-relative if possible
    p = info.get("path")
    if isinstance(p, (str, Path)):
        pth = Path(str(p))
        if pth.is_absolute():
            try:
                info["path"] = str(pth.relative_to(project_root))
            except ValueError:
                logger.warning(f"TestRegistry path is absolute and outside project_root: {pth}")

    # enforce env permanence (derive from path/ENV if missing)
    env_val = str(info.get("env") or "").strip()
    if not env_val:
        env_val = (
            _env_from_path(str(info.get("path", "")))
            or str(os.getenv("ENV") or "").strip()
            or "dev"
        )
        info["env"] = env_val

    for key in KEY_RECOVERY_FIELDS:
        info.setdefault(key, None)

    # timestamps based on status
    status = str(info.get("status") or "")

    if _is_running_status(status) and not str(info.get("started_at") or "").strip():
        info["started_at"] = _now_local_iso()

    if _is_terminal_status(status) and not str(info.get("finished_at") or "").strip():
        info["finished_at"] = _now_local_iso()

    return info


def load_test_registry(project_root: Path, *, heal: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Load FULL test_registry.json.
    Returns {} if missing; raises if unreadable, so a damaged registry is
    never rewritten with only the entries of the current run.

    If heal=True, normalize env/path/timestamps and rewrite registry if needed.
    """
    p = _path_for(project_root)
    if not p.exists():
        return {}

    try:
        data = load_json(p)  # utils/seal.py: verified, seal key dropped
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{p} is not valid JSON ({e}); fix it before running. The test "
            "registry is not rewritten while it is damaged."
        ) from e

    if not isinstance(data, dict):
        raise ValueError(f"{p} must hold a JSON object keyed by model_hash")

    if not heal:
        return data  # type: ignore[return-value]

    before = json.dumps(data, sort_keys=True, default=str)

    runs = (
        _train_registry(project_root)
        if any(isinstance(v, dict) and _missing_run_fields(v) for v in data.values())
        else {}
    )
    cleaned: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        v = dict(v)
        _fill_from_run(v, runs.get(k))
        cleaned[k] = _normalize_entry(v, Path(project_root))

    cleaned = _reorder_registry_for_write(cleaned)

    after = json.dumps(cleaned, sort_keys=True, default=str)

    if after != before:
        save_test_registry(cleaned, project_root)
        return cleaned

    return data  # type: ignore[return-value]


def save_test_registry(registry: Dict[str, Dict[str, Any]], project_root: Path) -> None:
    """
    Save registry atomically with env/path/timestamp normalization and ordering.

    IMPORTANT:
    This normalizes EVERY entry, so env gets backfilled for staging/prod too,
    even if only one model_id was updated.
    """
    project_root = Path(project_root)
    p = _path_for(project_root)

    cleaned: Dict[str, Dict[str, Any]] = {}
    for k, v in registry.items():
        if not isinstance(v, dict):
            continue
        cleaned[k] = _normalize_entry(v, project_root)

    cleaned = _reorder_registry_for_write(cleaned)

    try:
        atomic_write_text(p, json.dumps(cleaned, indent=2))  # keep insertion order; do NOT sort_keys
    except Exception as e:
        logger.error(f"Failed to save {p}: {e}")
        raise


@contextmanager
def test_registry_lock(project_root: Path):
    """
    One writer of test_registry.json at a time (an exclusive flock on artifacts/.test_registry.json.lock).
    Writers overlap when a queue's test phase, make test and the panel's Retest run together: each wrote
    back a copy loaded minutes earlier and the other's update was lost (audit 2026-10-23). A lock file
    that cannot be opened (owned by another user) is reported once and the write goes on unlocked.
    """
    p = _path_for(Path(project_root))
    lock_path = p.with_name(f".{p.name}.lock")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        lock = open(lock_path, "a")
    except OSError as exc:
        logger.warning("test_registry: no lock (%s: %s); writing unlocked", lock_path, exc)
        yield
        return
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def update_test_registry(project_root: Path, change, *, heal: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Read-modify-write under the lock: the registry is loaded fresh inside it, ``change(registry)``
    edits it in place, and it is saved before the lock is released. Returns the registry written.
    Every writer goes through this, so no writer saves a copy another one has moved on from.
    """
    with test_registry_lock(project_root):
        registry = load_test_registry(project_root, heal=heal)
        change(registry)
        save_test_registry(registry, project_root)
        return registry


# -----------------------------------------------------------------------------
# Updates
# -----------------------------------------------------------------------------

def update_test_status(
    project_root: Path,
    model_id: str,
    new_status: str,
    env: str | None = None,
    reason: str | None = None,
) -> None:
    """
    Update test registry status.

    Note:
    - Here `model_id` is the registry key/model_hash. A new entry is created
      only for a hash of run_registry.json; any other key (a model_name)
      raises instead of adding a junk entry.
    - The architecture model_id is backfilled only when the entry has none,
      from run_map.jsonl[model_hash]["model_id"], else os.environ["MODEL_ID"].
      An existing model_id is never overwritten.
    - An empty model_name / path is filled from run_registry.json[model_hash].
    - FAILED / INTERRUPTED / SKIPPED records `reason` as status_reason
      ("unspecified", with a warning, when the caller gives none); any other
      status drops it.
    """
    with test_registry_lock(project_root):
        _update_test_status_locked(project_root, model_id, new_status, env=env, reason=reason)


def _update_test_status_locked(
    project_root: Path,
    model_id: str,
    new_status: str,
    env: str | None = None,
    reason: str | None = None,
) -> None:
    """update_test_status's body, inside the registry lock: loaded, changed and saved as one step."""
    registry = load_test_registry(project_root)
    runs = _train_registry(project_root)

    if model_id not in registry or not isinstance(registry.get(model_id), dict):
        if model_id not in runs:
            message = (
                f"update_test_status: {model_id!r} is not a model_hash of run_registry.json; "
                "test_registry.json is keyed by model_hash, never by model_name. Pass the "
                "hash (artifacts/model_name_map.json maps names to hashes)."
            )
            logger.error(message)
            raise KeyError(message)
        registry[model_id] = {
            "model_name": "",
            "status": "NEW",
            "path": "",
            "env": _canonical_env(env=env, entry={}, default="dev"),
        }

    entry = registry[model_id]
    if not isinstance(entry, dict):
        logger.error(f"Registry entry for {model_id} is not a dict")
        return
    # Batches (utils/batch.py) follow the run's run_registry.json entry.
    merge_batches(entry, runs.get(model_id))

    _fill_from_run(entry, runs.get(model_id))

    previous = str(entry.get("status") or "")
    ns = str(new_status)
    entry["status"] = ns
    if _needs_reason(ns):
        if not str(reason or "").strip():
            logger.warning(
                "update_test_status: %s set to %s without a reason; status_reason=unspecified",
                model_id, ns,
            )
        entry["status_reason"] = str(reason or "").strip() or "unspecified"
    else:
        entry.pop("status_reason", None)
    if _status_norm(ns) in {"done", "failed"}:
        # FORCE queued it (orchestrator._force_queue); it has now been retested.
        entry.pop("force_previous_status", None)

    # Backfill the architecture model_id only when the entry has none; an
    # existing model_id belongs to the hash, not to the calling process.
    if not str(entry.get("model_id") or "").strip():
        map_rec = load_run_map(project_root).get(str(model_id)) or {}
        backfill = str(map_rec.get("model_id") or os.getenv("MODEL_ID") or "").strip()
        if backfill:
            entry["model_id"] = backfill

    env_val = _canonical_env(env=env, entry=entry, default="dev")
    entry["env"] = env_val

    # The legacy failure field; the registry's reason is status_reason.
    entry.pop("reason", None)

    # Timestamps belong to the attempt (as in sweep_engine.update_status): a
    # finished test queued again (or queued by FORCE, force_previous_status)
    # is a new attempt with no earlier times or outcome (begin_attempt);
    # entering in_progress starts one with its own start time, and entering a
    # terminal status ends it. A repeated status keeps its time.
    if _is_queued_status(ns) and (_is_terminal_status(previous) or entry.get("force_previous_status")):
        begin_attempt(entry, None)
    if _is_running_status(ns) and (
        not _is_running_status(previous) or not str(entry.get("started_at") or "").strip()
    ):
        begin_attempt(entry, _now_local_iso())

    if _is_terminal_status(ns) and (
        not _is_terminal_status(previous) or not str(entry.get("finished_at") or "").strip()
    ):
        entry["finished_at"] = _now_local_iso()

    if env is not None:
        pth = entry.get("path")
        if isinstance(pth, (str, Path)):
            pth_s = str(pth)
            entry["path"] = re_sub(r"(artifacts/)([^/]+)/", rf"\1{env_val}/", pth_s)

    save_test_registry(registry=registry, project_root=project_root)


# -----------------------------------------------------------------------------
# A test that attacks nothing is not run
# -----------------------------------------------------------------------------

SKIPPED = "SKIPPED"
# Statuses of a test that needs no further run.
FINISHED_TEST_STATUSES = frozenset({"DONE", SKIPPED})


def test_attack_settings(cfg: Any) -> str:
    """The settings the test is attacked with, as logged before the run decides whether to test:
    'perform_attack.test: attack_times=100, shuffle=true, attack_distribution=full_population;
    perform_attack: enabled=true, attack_step=1, ge_thresholds=[1]'."""
    plan = (cfg or {}).get("perform_attack") if isinstance(cfg, dict) else None
    plan = plan if isinstance(plan, dict) else {}
    block = plan.get("test") if isinstance(plan.get("test"), dict) else {}

    def text(value: Any) -> str:
        return json.dumps(value) if isinstance(value, (bool, list)) or value is None else str(value)

    test = ", ".join(f"{k}={text(block.get(k))}" for k in ("attack_times", "shuffle", "attack_distribution"))
    if block.get("attack_trace_budget") is not None:          # empty: every trace, not printed
        test += f", attack_trace_budget={block['attack_trace_budget']}"
    shared = ", ".join(f"{k}={text(plan.get(k))}" for k in ("enabled", "attack_step", "ge_thresholds"))
    return f"perform_attack.test: {test}; perform_attack: {shared}"


def unattacked_test_reason(cfg: Any) -> Optional[str]:
    """
    Why the test of this config is skipped, or None when it must run.

    perform_attack.test.attack_times: 0 leaves the test traces unattacked:
    there is nothing to measure them against, so the test process is not run.
    """
    plan = (cfg or {}).get("perform_attack") if isinstance(cfg, dict) else None
    block = plan.get("test") if isinstance(plan, dict) else None
    if isinstance(block, dict) and block.get("attack_times") == 0:
        return (
            "perform_attack.test.attack_times is 0: the test traces are not "
            "attacked, so the test process is skipped"
        )
    return None


def skip_unattacked_test(
    project_root: Path,
    model_hash: str,
    cfg: Any,
    env: str | None = None,
) -> Optional[str]:
    """
    Mark the model's test SKIPPED when its config leaves the test traces
    unattacked. Returns the reason, or None when the test must run (the
    registry is then not touched).
    """
    reason = unattacked_test_reason(cfg)
    if reason is not None:
        update_test_status(project_root, model_id=model_hash, new_status=SKIPPED, env=env, reason=reason)
    return reason
