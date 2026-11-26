# services/side_channel_core/utils/run_map.py
"""
run_map.py
==========

Durable AutoSearch run map for OptunaLite / AutoMLOps.

Purpose
-------
run_registry.json tells us model_hash -> execution status.

run_map.jsonl tells us:
    sampled parameters -> model_hash -> model_name -> status -> path

This is what allows AutoSearch to resume instead of starting from trial 1
after an interruption.

File Location
-------------
The run map is project-global:

    <project_root>/artifacts/run_map.jsonl

Example:

    /app/projects/DLSCA_ASCAD/artifacts/run_map.jsonl

The per-model artifact path stored inside each record follows:

    artifacts/{{env}}/generated_models/{{model_name}}

Example record path:

    artifacts/dev/generated_models/garlic_zigzag

Storage Format
--------------
JSON Lines: one compact JSON object per line, the latest record per
model_hash. A line that is not valid JSON is an error (never skipped), so a
damaged file cannot be rewritten with records missing. A run map saved by
older code as run_map.json is read when run_map.jsonl does not exist yet.

This module intentionally does NOT write run_map.csv.

Author: Shraddha Mishra
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from services.side_channel_core.utils.atomic_io import atomic_write_text
from services.side_channel_core.utils.timestamps import now_iso as _configured_now_iso
from services.side_channel_core.utils.batch import merge_batches, registry_batches
from services.side_channel_core.utils.seal import check as check_seal, strip as strip_seal


RUN_MAP_FILENAME = "run_map.jsonl"
LEGACY_RUN_MAP_FILENAME = "run_map.json"

DEFAULT_ARTIFACTS_ROOT_TEMPLATE = "artifacts/{{env}}/generated_models/{{model_name}}"

DONE_STATUSES = {"done"}
RETRYABLE_STATUSES = {
    "failed",
    "interrupted",
    "in_progress",
    "running",
    "in_process",
}
STALE_STATUSES = {
    "new",
    "queued",
    "pending",
    "running",
    "in_progress",
    "in_process",
}


def now_iso() -> str:
    return _configured_now_iso()


def _norm(value: Any) -> str:
    return str(value or "").strip()


def norm_status(value: Any) -> str:
    return _norm(value).lower()


def sample_key(sampled: Mapping[str, Any]) -> str:
    """
    Stable key for one sampled AutoSearch point.

    This key is independent of trial_id. It is based only on the sampled
    parameter content.
    """
    return json.dumps(
        sampled,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def resolve_project_artifacts_dir(project_root: str | Path) -> Path:
    """
    Resolve project-level artifacts directory.

    Example:
        /app/projects/DLSCA_ASCAD/artifacts
    """
    return Path(project_root).expanduser().resolve() / "artifacts"


def resolve_run_map_path(project_root: str | Path) -> Path:
    """
    Resolve project-global run map path.

    Example:
        /app/projects/DLSCA_ASCAD/artifacts/run_map.jsonl
    """
    return resolve_project_artifacts_dir(project_root) / RUN_MAP_FILENAME


def resolve_generated_model_dir(
    *,
    project_root: str | Path,
    env: str,
    model_name: str,
    artifacts_root_template: str = DEFAULT_ARTIFACTS_ROOT_TEMPLATE,
) -> Path:
    """
    Resolve per-model generated artifact directory from template.

    Template:
        artifacts/{{env}}/generated_models/{{model_name}}

    Example:
        /app/projects/DLSCA_ASCAD/artifacts/dev/generated_models/garlic_zigzag
    """
    root = Path(project_root).expanduser().resolve()
    env = _norm(env) or "dev"
    model_name = _norm(model_name)

    if not model_name:
        raise ValueError("model_name is required")

    rel = (
        str(artifacts_root_template or DEFAULT_ARTIFACTS_ROOT_TEMPLATE)
        .replace("{{env}}", env)
        .replace("{{model_name}}", model_name)
    )

    path = root / rel

    # Avoid accidental duplicated model_name:
    # artifacts/dev/generated_models/oxen_oboe/oxen_oboe
    parts = path.parts
    if len(parts) >= 2 and parts[-1] == model_name and parts[-2] == model_name:
        path = Path(*parts[:-1])

    return path


def resolve_artifacts_root_template_from_cfg(cfg: Mapping[str, Any]) -> str:
    """
    Read artifacts_root template from cfg when available.

    If cfg["paths"]["artifacts_root"] is absent, fallback to:
        artifacts/{{env}}/generated_models/{{model_name}}
    """
    paths = cfg.get("paths") or {}

    if isinstance(paths, Mapping):
        value = paths.get("artifacts_root")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return DEFAULT_ARTIFACTS_ROOT_TEMPLATE


def _relative_to_project_root(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def project_relative_path(path: str | Path, project_root: str | Path) -> str:
    """
    A run folder as run_map.jsonl stores it: relative to the project
    ('artifacts/dev/generated_models/x'). Takes an absolute path, or the
    run_registry.json form, which starts with the project's name
    ('DLSCA_ASCAD/artifacts/dev/generated_models/x',
    artifact_utils._canonical_registry_path).
    """
    root = Path(project_root).expanduser().resolve()
    p = Path(str(path)).expanduser()
    if p.is_absolute():
        return _relative_to_project_root(p.resolve(), root)
    parts = p.parts
    if len(parts) > 1 and parts[0] == root.name and not (root / parts[0]).exists():
        return str(Path(*parts[1:]))
    return str(p)


def _usable_record(rec: Any) -> bool:
    """A record needs a real model_hash (not the "sampled" placeholder) and a model_name."""
    return (
        isinstance(rec, dict)
        and _norm(rec.get("model_hash")) not in ("", "sampled")
        and bool(_norm(rec.get("model_name")))
    )


def load_run_map(project_root: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Load run_map.jsonl: one JSON object per line, latest record per model_hash.

    Raises ValueError on a line that is not valid JSON, naming the line.
    Falls back to a legacy run_map.json (JSON Lines, or one dict/list JSON
    document) only when run_map.jsonl does not exist.
    """
    path = resolve_run_map_path(project_root)
    if not path.exists():
        legacy = path.with_name(LEGACY_RUN_MAP_FILENAME)
        if not legacy.exists():
            return {}
        path = legacy

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        check_seal(path)  # utils/seal.py: a changed line raises TamperedArtifactError
    except json.JSONDecodeError:
        pass  # a damaged line is reported below, by line number

    records: list[Any] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(strip_seal(json.loads(line)))
        except json.JSONDecodeError as exc:
            if path.name != LEGACY_RUN_MAP_FILENAME:
                raise ValueError(
                    f"{path}:{lineno} is not valid JSON ({exc.msg}); fix or remove "
                    "that line. The run map is not rewritten while it is damaged."
                ) from exc
            # Legacy run_map.json may be one pretty-printed dict/list document.
            try:
                document = json.loads(text)
            except json.JSONDecodeError:
                raise ValueError(f"{path}:{lineno} is not valid JSON ({exc.msg})") from exc
            document = strip_seal(document)
            if isinstance(document, dict):
                records = [
                    {**value, "model_hash": _norm(value.get("model_hash") or key)}
                    for key, value in document.items()
                    if isinstance(value, dict)
                ]
            else:
                records = list(document) if isinstance(document, list) else []
            break

    return {
        _norm(rec["model_hash"]): rec for rec in records if _usable_record(rec)
    }


def _status_rank(status: Any) -> int:
    s = norm_status(status)

    if s in {"in_progress", "running", "in_process"}:
        return 0

    if s in {"interrupted"}:
        return 1

    if s in {"failed"}:
        return 2

    if s in {"new", "queued", "pending"}:
        return 3

    if s in {"done"}:
        return 4

    return 9

        
def save_run_map(
    project_root: str | Path,
    run_map: Dict[str, Dict[str, Any]],
) -> None:
    """
    Save run_map.jsonl: one compact JSON object per line.
    """
    path = resolve_run_map_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    def sort_key(item: tuple[str, Dict[str, Any]]) -> tuple[int, str, str]:
        model_hash, rec = item
        return (
            _status_rank(rec.get("status")),
            _norm(rec.get("updated_at")),
            _norm(model_hash),
        )

    lines = []

    for model_hash, rec in sorted(run_map.items(), key=sort_key):
        if not isinstance(rec, dict):
            continue

        clean = dict(rec)
        clean["model_hash"] = _norm(clean.get("model_hash") or model_hash)

        if not clean["model_hash"] or clean["model_hash"] == "sampled":
            continue

        if not _norm(clean.get("model_name")):
            continue

        if "sampled" not in clean or not isinstance(clean.get("sampled"), dict):
            continue

        lines.append(
            json.dumps(
                clean,
                ensure_ascii=False,
                sort_keys=False,
                separators=(",", ":"),
                default=str,
            )
        )

    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def upsert_run_map_entry(
    project_root: str | Path,
    *,
    model_hash: str,
    model_name: str,
    model_id: Optional[str],
    sampled: Mapping[str, Any],
    status: str,
    env: str,
    artifacts_root_template: str = DEFAULT_ARTIFACTS_ROOT_TEMPLATE,
    run_dir: Optional[str | Path] = None,
    trial_index: Optional[int] = None,
    total_remaining_trials: Optional[int] = None,
    run_session_id: Optional[str] = None,
    total_trials: Optional[int] = None,
    selection_source: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Insert/update one AutoSearch mapping record.

    This is the central function OptunaLite should call before/after each trial.
    """
    root = Path(project_root).expanduser().resolve()

    model_hash = _norm(model_hash)
    model_name = _norm(model_name)
    env = _norm(env) or "dev"

    if not model_hash:
        raise ValueError("model_hash is required")

    if not model_name:
        raise ValueError("model_name is required")

    if run_dir is not None:
        # The registry's form (<project>/artifacts/...) is rooted as artifacts/...
        model_dir = root / project_relative_path(run_dir, root)

        parts = model_dir.parts
        if len(parts) >= 2 and parts[-1] == model_name and parts[-2] == model_name:
            model_dir = Path(*parts[:-1])
    else:
        model_dir = resolve_generated_model_dir(
            project_root=root,
            env=env,
            model_name=model_name,
            artifacts_root_template=artifacts_root_template,
        )
    run_map = load_run_map(root)
    sampled_key = sample_key(sampled)

    rec = run_map.get(model_hash)

    if not isinstance(rec, dict):
        rec = {}

        # stale_hashes = [
        #     existing_hash
        #     for existing_hash, existing_rec in run_map.items()
        #     if isinstance(existing_rec, dict)
        #     and _norm(existing_rec.get("sampled_key")) == sampled_key
        #     and _norm(existing_hash) != model_hash
        # ]

        # for stale_hash in stale_hashes:
        #     run_map.pop(stale_hash, None)

    status_norm = _norm(status).upper()

    rec["env"] = env
    rec["model_name"] = model_name
    rec["model_hash"] = model_hash
    rec["model_id"] = str(model_id) if model_id is not None else None
    rec["status"] = status_norm
    rec["path"] = _relative_to_project_root(model_dir, root)
    rec["sampled_key"] = sampled_key
    rec["sampled"] = dict(sampled)
    rec["updated_at"] = now_iso()

    if trial_index is not None:
        rec["trial_index"] = int(trial_index)
    
    if total_remaining_trials is not None:
        rec["total_remaining_trials"] = int(total_remaining_trials)

    if total_trials is not None:
        rec["total_trials"] = int(total_trials)
    if run_session_id:
        rec["run_session_id"] = str(run_session_id)
    if selection_source:
        rec["selection_source"] = str(selection_source)

    if reason:
        rec["reason"] = str(reason)
    elif status_norm in {"IN_PROGRESS", "DONE"}:
        rec.pop("reason", None)

    # Batches (utils/batch.py) follow the run's run_registry.json entry.
    merge_batches(rec, registry_batches(root, model_hash))

    # State-snapshot semantics:
    # one latest record per model_hash.
    run_map[model_hash] = rec
    save_run_map(root, run_map)

    return rec

def sync_run_map_status_from_run_registry(
    project_root: str | Path,
    run_registry: Dict[str, Any],
) -> Dict[str, int]:
    """
    Sync run_map statuses from run_registry.json.

    Only existing run_map records are repaired; no new records are created from
    run_registry alone because sampled parameters cannot be reconstructed
    safely. The run map is saved once, after all repairs.
    """
    root = Path(project_root).expanduser().resolve()
    run_map = load_run_map(root)

    changed = 0

    for model_hash, reg_entry in run_registry.items():
        if not isinstance(reg_entry, dict):
            continue

        model_hash_s = _norm(model_hash)
        if not model_hash_s:
            continue

        rec = run_map.get(model_hash_s)
        if not isinstance(rec, dict):
            continue

        reg_status = _norm(reg_entry.get("status"))
        if not reg_status:
            continue

        reg_status_upper = reg_status.upper()

        # load_run_map() gives latest collapsed state. If it already matches,
        # do not append duplicate repair events.
        if norm_status(rec.get("status")) == reg_status.lower():
            continue

        repaired = dict(rec)
        repaired["status"] = reg_status_upper
        repaired["updated_at"] = now_iso()

        # status_reason is the registry's reason (sweep_engine.update_status);
        # the other names are only in entries written before 2026-10-20.
        reason = (
            reg_entry.get("status_reason")
            or reg_entry.get("interrupt_reason")
            or reg_entry.get("reason")
            or reg_entry.get("failure_reason")
        )

        if reason:
            repaired["reason"] = str(reason)
        elif reg_status_upper in {"IN_PROGRESS", "DONE"}:
            repaired.pop("reason", None)

        # Preserve/repair useful registry fields when present.
        if _norm(reg_entry.get("model_name")):
            repaired["model_name"] = _norm(reg_entry.get("model_name"))

        if _norm(reg_entry.get("model_id")):
            repaired["model_id"] = _norm(reg_entry.get("model_id"))

        if _norm(reg_entry.get("env")):
            repaired["env"] = _norm(reg_entry.get("env"))

        if _norm(reg_entry.get("path")):
            repaired["path"] = project_relative_path(_norm(reg_entry.get("path")), root)

        run_map[model_hash_s] = repaired
        changed += 1

    if changed:
        save_run_map(root, run_map)
    return {"changed": changed}


def prioritize_sample_points(
    points: Iterable[Dict[str, Any]],
    run_map: Dict[str, Dict[str, Any]],
    *,
    force: bool = False,
) -> list[Dict[str, Any]]:
    """
    Return AutoSearch points in execution order.

    FORCE=0:
        - skip DONE
        - retry FAILED first
        - retry INTERRUPTED second
        - retry IN_PROGRESS/RUNNING/IN_PROCESS third
        - then run fresh points

    FORCE=1:
        - return all points
    """
    points = list(points)

    if force:
        return points

    # sampled_key -> status
    status_by_key: Dict[str, str] = {}

    for rec in run_map.values():
        if not isinstance(rec, dict):
            continue

        key = _norm(rec.get("sampled_key"))
        if not key:
            continue

        status_by_key[key] = norm_status(rec.get("status"))

    failed: list[Dict[str, Any]] = []
    interrupted: list[Dict[str, Any]] = []
    running: list[Dict[str, Any]] = []
    fresh: list[Dict[str, Any]] = []

    for point in points:
        key = sample_key(point)
        status = status_by_key.get(key, "")

        if status in DONE_STATUSES:
            continue

        if status == "failed":
            failed.append(point)
        elif status == "interrupted":
            interrupted.append(point)
        elif status in {"in_progress", "running", "in_process"}:
            running.append(point)
        else:
            fresh.append(point)

    return failed + interrupted + running + fresh
