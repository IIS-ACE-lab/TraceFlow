# services/side_channel_core/utils/batch.py
"""
Batches: a common label for the runs made together, e.g. for one figure.

    make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig6     # BATCH=fig6, one BATCH_ID
    make orchestrate ... MODEL_ID=ASCAD_CNNbase BATCH=fig6     # by hand
    make batch_label ENV=dev PROJECT=DLSCA_ASCAD BATCH=fig6    # label existing runs

A run's run_registry.json entry carries two lists (the source of truth; its
test_registry.json entry and its run_map.jsonl record carry the same lists):

    batches    the batch names it belongs to         ["fig5", "fig6"]
    batch_ids  the executions it was part of         ["fig6-20261006T083000Z"]

A run joins a batch when the batch trains it, and also when the batch finds it
already DONE in its scope and skips it: the CNNbase reference run of Fig. 5 is
then in batch fig6 as well. Neither list is part of the model hash. Find the
runs of a batch with make find_model and

    - path: batches
      contains: fig6

Author: Shraddha Mishra
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, MutableMapping, Optional, Tuple
from services.side_channel_core.utils.seal import check as check_seal, load_json

BATCH_ENV = "BATCH"
BATCH_ID_ENV = "BATCH_ID"
BATCHES_KEY = "batches"
BATCH_IDS_KEY = "batch_ids"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _check(value: str, field: str) -> str:
    if not _NAME_RE.match(value):
        raise ValueError(
            f"{field}={value!r} is not a valid batch label: use letters, digits, '_', '.', '-' "
            "(starting with a letter or digit), e.g. BATCH=fig6"
        )
    return value


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def new_batch_id(batch: str) -> str:
    """One execution of a batch: <batch>-<UTC start time>."""
    return f"{_check(batch, BATCH_ENV)}-{utc_stamp()}"


def current_batch(environ: Optional[MutableMapping[str, str]] = None) -> Optional[Tuple[str, str]]:
    """
    (batch, batch_id) of this process, or None when no BATCH is set.

    BATCH_ID defaults to <batch>-<orchestrator session>, so the passes of one
    make orchestrate call share it.
    """
    env = os.environ if environ is None else environ
    batch = str(env.get(BATCH_ENV) or "").strip()
    batch_id = str(env.get(BATCH_ID_ENV) or "").strip()
    if not batch:
        if batch_id:
            raise ValueError(f"{BATCH_ID_ENV}={batch_id!r} is set without {BATCH_ENV}")
        return None
    _check(batch, BATCH_ENV)
    if not batch_id:
        session = str(env.get("ORCHESTRATOR_SESSION_ID") or "").strip()
        batch_id = f"{batch}-{session}" if session else new_batch_id(batch)
    return batch, _check(batch_id, BATCH_ID_ENV)


def held_by_another_batch(entry: Optional[MutableMapping[str, Any]], batch: Optional[Tuple[str, str]]) -> bool:
    """The run belongs to a batch other than `batch` ((BATCH, BATCH_ID), None outside any batch): another
    figure's run. FORCE keeps such a run when it is DONE (sweep_engine, orchestrator._force_queue)."""
    labels = (entry or {}).get(BATCHES_KEY) if isinstance(entry, dict) else None
    labels = labels if isinstance(labels, (list, tuple)) else ([labels] if labels else [])
    name = batch[0] if batch else None
    return any(str(label) != name for label in labels)


def add_batch(entry: MutableMapping[str, Any], batch: str, batch_id: str) -> bool:
    """Add batch and batch_id to a registry entry's lists. True when it changed."""
    changed = False
    for key, value in ((BATCHES_KEY, batch), (BATCH_IDS_KEY, batch_id)):
        values = entry.get(key)
        values = list(values) if isinstance(values, (list, tuple)) else ([values] if values else [])
        if value not in values:
            values.append(value)
            changed = True
        entry[key] = values
    return changed


def merge_batches(entry: MutableMapping[str, Any], source: Optional[MutableMapping[str, Any]]) -> bool:
    """Add the batches / batch_ids of `source` (a run_registry entry) to `entry`. True when it changed."""
    if not isinstance(source, dict):
        return False
    changed = False
    for key in (BATCHES_KEY, BATCH_IDS_KEY):
        new = source.get(key)
        if not isinstance(new, (list, tuple)) or not new:
            continue
        values = entry.get(key)
        values = list(values) if isinstance(values, (list, tuple)) else ([values] if values else [])
        for value in new:
            if value not in values:
                values.append(value)
                changed = True
        entry[key] = values
    return changed


def registry_batches(project_root: str | Path, model_hash: str) -> Optional[dict]:
    """The run_registry.json entry of `model_hash`, read only (None when absent)."""
    path = Path(project_root) / "artifacts" / "run_registry.json"
    try:
        entry = load_json(path).get(str(model_hash))  # utils/seal.py
    except (OSError, ValueError, AttributeError):
        return None
    return entry if isinstance(entry, dict) else None


def stamp_registry(
    project_root: str | Path,
    model_hashes: Iterable[str],
    *,
    batch: Optional[Tuple[str, str]] = None,
) -> list[str]:
    """
    Add the current batch (or `batch`) to these hashes in run_registry.json,
    and to their entries in test_registry.json and run_map.jsonl.

    Hashes without a run_registry entry are left alone; a test entry or run_map
    record that does not exist yet copies the lists from run_registry.json when
    it is written (update_test_status, upsert_run_map_entry). Returns the hashes
    whose run_registry entry changed. Each file is written only when it changed.
    """
    batch = batch if batch is not None else current_batch()
    hashes = [str(h).strip() for h in model_hashes if str(h or "").strip()]
    if batch is None or not hashes:
        return []
    # Imported here: these modules import this one.
    from utils.run_map import load_run_map, save_run_map
    from utils.sweep_engine import load_registry, save_registry
    from utils.test_registry import update_test_registry

    root = Path(project_root)
    registry = load_registry(root)
    hashes = [h for h in hashes if isinstance(registry.get(h), dict)]
    changed = [h for h in hashes if add_batch(registry[h], *batch)]
    if changed:
        save_registry(registry, root)

    # A list, not any(): every entry must be labelled, not only the first. Under the registry's
    # lock: loaded, labelled and saved as one step (utils/test_registry.py update_test_registry).
    update_test_registry(root, lambda tests: [
        h for h in hashes if isinstance(tests.get(h), dict) and add_batch(tests[h], *batch)
    ])

    run_map = load_run_map(root)
    if [h for h in hashes if isinstance(run_map.get(h), dict) and add_batch(run_map[h], *batch)]:
        save_run_map(root, run_map)

    # The run's snapshot carries the label too (outside its digest), so it still names its batch
    # when shipped alone (utils/reproduce.py). A run without a snapshot yet gets it when written.
    from utils.data_reproducibility import add_snapshot_batch
    from utils.reproduce import snapshot_path
    for h in hashes:
        entry = registry.get(h) or {}
        env, name = str(entry.get("env") or "").strip(), str(entry.get("model_name") or "").strip()
        snap = snapshot_path(root, env, name) if env and name else None
        if snap is not None and snap.is_file():
            try:
                add_snapshot_batch(snap, batch[0])
            except (OSError, ValueError):
                pass                                   # a damaged snapshot is reported when it is used
    return changed


# =============================================================================
# make batch_label: label existing runs
# =============================================================================
def label_models(
    project_root: str | Path,
    *,
    batch: str,
    env: Optional[str] = None,
    models_file: Optional[str | Path] = None,
) -> list[tuple[str, str]]:
    """
    Add `batch` to the runs listed in a make find_model output
    (artifacts/filtered_models.yaml by default). Returns [(model_hash, model_name)].

    The file's details carry each model's hash; a bare `models:` list is matched
    by model_name (and env) in run_registry.json. A name that is not a run of
    the registry stops the labelling before anything is written.
    """
    import yaml

    from utils.sweep_engine import load_registry

    project_root = Path(project_root)
    path = Path(models_file) if models_file else project_root / "artifacts" / "filtered_models.yaml"
    if not path.is_absolute() and not path.exists():
        path = project_root / path
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run make find_model first, or pass FILE=")
    check_seal(path)  # utils/seal.py (only a file under artifacts/ is checked)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = [str(n).strip() for n in (data.get("models") or []) if str(n).strip()]
    if not names:
        raise ValueError(f"{path} lists no models")
    hash_of = {str(d.get("model_name")): str(d.get("model_hash"))
               for d in (data.get("details") or []) if isinstance(d, dict) and d.get("model_hash")}

    registry = load_registry(project_root)
    targets: list[tuple[str, str]] = []
    unknown: list[str] = []
    for name in names:
        h = hash_of.get(name)
        if not h:
            found = [k for k, v in registry.items() if isinstance(v, dict)
                     and v.get("model_name") == name and (env is None or v.get("env") == env)]
            h = found[0] if len(found) == 1 else None
        if not h or not isinstance(registry.get(h), dict):
            unknown.append(name)
        else:
            targets.append((h, name))
    if unknown:
        raise ValueError(f"not runs of {project_root / 'artifacts' / 'run_registry.json'}"
                         f"{f' (env {env})' if env else ''}: {unknown}")
    _check(batch, BATCH_ENV)
    stamp_registry(project_root, [h for h, _ in targets], batch=(batch, f"{batch}-label-{utc_stamp()}"))
    return targets


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Add a batch label to existing runs (make batch_label).")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--env", default=None)
    parser.add_argument("--file", default=None, help="a make find_model output; default artifacts/filtered_models.yaml")
    args = parser.parse_args()
    try:
        targets = label_models(args.project_root, batch=args.batch, env=args.env, models_file=args.file)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"batch_label: {exc}") from exc
    for h, name in targets:
        print(f"  {name} ({h}): batches += {args.batch}")
    print(f"batch_label: {len(targets)} run(s) in batch {args.batch}. "
          f"Find them with make find_model and `path: batches, contains: {args.batch}`.")


if __name__ == "__main__":
    main()
