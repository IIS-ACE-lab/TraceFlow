# services/side_channel_core/utils/data_reproducibility.py

'''
This module handles the preparation of artifacts for data reproducibility in machine learning projects.
It writes the run's resolved config snapshot (paths.params_out) and copies the
enabled parent-config YAML files into the run's data directory, and then
produces (data.generate_data_out: true) or verifies (false) paths.data_out and
its split files: train/test with cross_validation.enabled, train/val/test without.

The files are stored in the following structure:

{Project_Root}/
    ├── artifacts/
    .   └── {env}/generated_models/
            └── {model_name}/
                ├── config/
                │   └── {model_name}_params.yaml    # paths.params_out: the run's config
                └── data/
                    ├── {parent_name}.yaml          # only for enabled parent config blocks
                    └── {dataset_alias}.h5, train/val/test.h5

The live project params.yaml (the whole sweep grid) is not copied: the run's
own config is config/{model_name}_params.yaml.

params_out snapshot (2026-09-29, SYS-05/SYS-28)
------------------------------------------------
- Paths are stored relative to the project root or the workspace root
  (path_utils.normalize_paths_for_dump); resolve_path() re-roots them on load
  to the same file. A path no relative form maps back to stays absolute.
- The file is written atomically (temp file, fsync, os.replace) and starts with
  ``snapshot_sha256``: the SHA-256 of the canonical YAML dump of every other
  key. read_params_snapshot() rejects a file whose content no longer matches
  it (truncated or edited), or that lacks a required section or model_hash.
- Snapshots written before 2026-09-29 have no digest: they are accepted with
  a WARNING when they parse and hold every required section and model_hash.
- ``batches`` (2026-10-12): the batch labels of the run (BATCH= / the queue key),
  so a snapshot shipped alone still says which figure it belongs to
  (utils/reproduce.py, make reproduce BATCH=). A label, not part of the
  experiment: outside the digest and the model hash, so labelling a run later
  (make batch_label, a queue finding it DONE) adds to it without breaking the seal.

Author: Shraddha Mishra
'''
from __future__ import annotations

import copy
import hashlib
import os
import shutil
import yaml
from pathlib import Path

from utils.config_view import resolved_view
from utils.logger import get_logger
from utils.path_utils import resolve_path
from utils.path_utils import normalize_paths_for_dump, convert_paths_to_strings, to_portable_path
from utils.seal import TamperedArtifactError, check as check_seal, seal_yaml_text

logger = get_logger("DataReproducibility")

SNAPSHOT_DIGEST_KEY = "snapshot_sha256"
SNAPSHOT_BATCHES_KEY = "batches"
# Keys outside the digest: the digest itself and the batch labels (a label, not the experiment).
_UNSEALED_KEYS = (SNAPSHOT_DIGEST_KEY, SNAPSHOT_BATCHES_KEY)
# Sections every retry/test of a run needs (make_experiment_hash, paths, SCA
# contract) plus its identity; a truncated dump loses its last keys.
SNAPSHOT_REQUIRED_KEYS = (
    "project", "data", "model", "trainer", "seed", "sca", "paths",
    "model_hash", "model_name",
)


def _is_enabled_like(value) -> bool:
    """
    Generic truthy parser for YAML-ish flags.
    Supports bools, ints, and common string forms.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return False


def _iter_enabled_parent_yaml_blocks(cfg: dict):
    """
    Yield (parent_key, parent_block) for top-level parent config blocks that:
      - are dicts
      - have enabled: truthy
      - expose a yaml payload and/or config path
    This is intentionally generic and does not hardcode field names.
    """
    if not isinstance(cfg, dict):
        return

    for parent_key, parent_val in cfg.items():
        if not isinstance(parent_val, dict):
            continue
        if not _is_enabled_like(parent_val.get("enabled")):
            continue

        has_yaml_payload = isinstance(parent_val.get("yaml"), dict)
        has_config_ref = bool(parent_val.get("config_abs") or parent_val.get("config"))

        if has_yaml_payload or has_config_ref:
            yield parent_key, parent_val


def _normalize_snapshot_yaml_inclusion(cfg: dict) -> dict:
    """
    Generic snapshot normalization.

    Rule 1:
      If a top-level parent block is disabled, remove its nested `yaml` payload
      from the snapshot.

    Rule 2:
      If a top-level parent block is enabled and contains a nested `yaml` dict,
      remove any duplicated top-level keys that are already represented inside
      that nested yaml payload.

    This is fully generic and does not hardcode parent names or child keys.
    """
    out = copy.deepcopy(cfg)

    if not isinstance(out, dict):
        return out

    keys_to_remove = set()

    for parent_key, parent_val in out.items():
        if not isinstance(parent_val, dict):
            continue

        raw_yaml = parent_val.get("yaml")

        # Disabled parent -> remove nested yaml payload entirely
        if not _is_enabled_like(parent_val.get("enabled")):
            if "yaml" in parent_val:
                parent_val.pop("yaml", None)
            continue

        # Enabled parent with nested yaml payload -> prune duplicate root keys
        if not isinstance(raw_yaml, dict):
            continue

        for child_key, child_val in raw_yaml.items():
            if child_key == parent_key:
                continue
            if child_key not in out:
                continue
            # Retries and tests rebuild the run from the snapshot alone.
            if child_key in SNAPSHOT_REQUIRED_KEYS:
                continue

            # Remove only exact duplicates from snapshot root
            if out.get(child_key) == child_val:
                keys_to_remove.add(child_key)

    for k in keys_to_remove:
        out.pop(k, None)

    return out


def _copy_enabled_parent_yaml_files(cfg: dict, artifacts_data_dir: Path, project_root: Path) -> None:
    """
    Generic config-file copier.

    For every enabled parent block exposing config_abs/config, copy the backing YAML file
    into the artifact data directory as <parent_key>.yaml.

    No hardcoded field names.
    """
    for parent_key, parent_block in _iter_enabled_parent_yaml_blocks(cfg):
        cfg_src = parent_block.get("config_abs") or parent_block.get("config")
        if not cfg_src:
            continue

        src = Path(cfg_src)
        if not src.is_absolute():
            src = (project_root / src).resolve()

        dst = artifacts_data_dir / f"{parent_key}.yaml"

        if src.is_file():
            shutil.copy(src, dst)
            # logger.info(
            #     "Active %s config copied to %s",
            #     parent_key,
            #     to_portable_path(dst),
            # )
        else:
            logger.warning(
                "%s enabled but config file not found at: %s",
                parent_key,
                to_portable_path(src),
            )


def snapshot_digest(snapshot: dict) -> str:
    """SHA-256 of the canonical YAML dump of a snapshot without its digest and batch labels."""
    body = {k: v for k, v in snapshot.items() if k not in _UNSEALED_KEYS}
    return hashlib.sha256(yaml.safe_dump(body, sort_keys=True).encode("utf-8")).hexdigest()


def read_params_snapshot(path: Path) -> dict:
    """
    Load a params_out snapshot; reject a truncated or damaged one.

    Raises ValueError (after log.error) when the file does not parse, is not
    a mapping, lacks a SNAPSHOT_REQUIRED_KEYS section or model_hash, or no
    longer matches its snapshot_sha256. A snapshot without the digest
    (written before 2026-09-29) is accepted with a WARNING. The digest key is
    returned with the mapping; verify_snapshot_identity() consumes it.
    """
    path = Path(path)
    hint = (
        "Restore it from a backup, or re-run the experiment from the current "
        "params.yaml as a new run (make delete_database ... MODEL=<model_name>, "
        "then make orchestrate)."
    )

    try:
        check_seal(path)  # utils/seal.py: a signed snapshot whose bytes changed is a damaged one
    except TamperedArtifactError as exc:
        logger.error("params_out snapshot %s fails its seal. %s", path, hint)
        raise ValueError(f"params_out snapshot {path} fails its seal ({exc}). {hint}") from exc
    try:
        snapshot = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        logger.error("params_out snapshot %s does not parse (%s). %s", path, exc, hint)
        raise ValueError(f"params_out snapshot {path} does not parse ({exc}). {hint}") from exc

    if not isinstance(snapshot, dict) or not snapshot:
        logger.error("params_out snapshot %s is empty or not a mapping. %s", path, hint)
        raise ValueError(f"params_out snapshot {path} is empty or not a mapping. {hint}")

    missing = [key for key in SNAPSHOT_REQUIRED_KEYS if key not in snapshot]
    if not str(snapshot.get("model_hash") or "").strip() and "model_hash" not in missing:
        missing.append("model_hash")
    if missing:
        logger.error("params_out snapshot %s is truncated or damaged: missing %s. %s", path, missing, hint)
        raise ValueError(f"params_out snapshot {path} is truncated or damaged: missing {missing}. {hint}")

    recorded = snapshot.get(SNAPSHOT_DIGEST_KEY)
    if recorded is None:
        logger.warning(
            "params_out snapshot %s has no %s (written before 2026-09-29); "
            "its integrity cannot be verified.",
            path, SNAPSHOT_DIGEST_KEY,
        )
    elif str(recorded) != snapshot_digest(snapshot):
        logger.error(
            "params_out snapshot %s does not match its %s (truncated or edited). %s",
            path, SNAPSHOT_DIGEST_KEY, hint,
        )
        raise ValueError(
            f"params_out snapshot {path} does not match its {SNAPSHOT_DIGEST_KEY} "
            f"(truncated or edited). {hint}"
        )
    return snapshot


def write_params_snapshot(cfg: dict, params_out_file: Path) -> None:
    """
    Write the run's resolved config to paths.params_out, atomically.

    Relative paths only, snapshot_sha256 first (a truncated file keeps the
    digest and fails it). Temp file in the same directory, flush + fsync,
    os.replace; the written file is read back and verified.
    """
    params_out_file = Path(params_out_file)
    params_out_file.parent.mkdir(parents=True, exist_ok=True)

    dump_cfg = normalize_paths_for_dump(
        convert_paths_to_strings(cfg),
        project_root=(cfg.get("project") or {}).get("root"),
    )
    dump_cfg = _normalize_snapshot_yaml_inclusion(dump_cfg)
    # What the run uses only: no graph_config.yaml, only the selected variant of each
    # discriminated union (utils/config_view.py; the model hash is unchanged by it).
    dump_cfg = resolved_view(dump_cfg)
    dump_cfg.pop(SNAPSHOT_DIGEST_KEY, None)       # a reloaded snapshot's old digest
    # The run's batch labels: those it already carries (a reloaded snapshot) and the current BATCH.
    from utils.batch import current_batch
    labels = [str(b) for b in (dump_cfg.pop(SNAPSHOT_BATCHES_KEY, None) or []) if str(b).strip()]
    batch = current_batch()
    if batch and batch[0] not in labels:
        labels.append(batch[0])
    missing = [key for key in SNAPSHOT_REQUIRED_KEYS if key not in dump_cfg]
    if missing:
        raise ValueError(
            f"Run config lacks {missing}; the params_out snapshot {params_out_file} "
            "must hold every section a retry or test rebuilds the run from."
        )
    snapshot = {SNAPSHOT_DIGEST_KEY: snapshot_digest(dump_cfg), **({SNAPSHOT_BATCHES_KEY: labels} if labels else {}), **dump_cfg}

    tmp = params_out_file.with_name(f".{params_out_file.name}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            # utils/seal.py: signed by its writer, whose key id travels with it
            f.write(seal_yaml_text(yaml.safe_dump(snapshot, sort_keys=False), params_out_file))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, params_out_file)
    finally:
        tmp.unlink(missing_ok=True)

    read_params_snapshot(params_out_file)


def add_snapshot_batch(path: Path, batch: str) -> bool:
    """Add a batch label to a written snapshot (outside the digest: the seal stays valid). True when changed."""
    path = Path(path)
    snapshot = read_params_snapshot(path)
    labels = [str(b) for b in (snapshot.get(SNAPSHOT_BATCHES_KEY) or []) if str(b).strip()]
    if batch in labels:
        return False
    labels.append(batch)
    ordered = {SNAPSHOT_DIGEST_KEY: snapshot.pop(SNAPSHOT_DIGEST_KEY)} if SNAPSHOT_DIGEST_KEY in snapshot else {}
    snapshot.pop(SNAPSHOT_BATCHES_KEY, None)
    ordered[SNAPSHOT_BATCHES_KEY] = labels
    ordered.update(snapshot)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(seal_yaml_text(yaml.safe_dump(ordered, sort_keys=False), path))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    read_params_snapshot(path)
    return True


def prepare_artifacts_data(cfg: dict):
    project_root = Path(cfg["project"]["root"])

    artifacts_data_dir = resolve_path(cfg["paths"]["data"], cfg, inject=True, mkdir=True)
    params_out_file = resolve_path(cfg["paths"]["params_out"], cfg, inject=True, mkdir=True)

    artifacts_data_dir.mkdir(parents=True, exist_ok=True)

    # The live params.yaml (whole sweep grid, possibly edited since) is not
    # copied: nothing reads it, and the run's own config is params_out.

    # Generic copy of active parent config YAMLs (config files: from the project's own root)
    _copy_enabled_parent_yaml_files(cfg, artifacts_data_dir, Path(cfg["project"].get("config_root") or project_root))

    write_params_snapshot(cfg, params_out_file)

    # logger.info("Runtime config dumped to %s", to_portable_path(params_out_file))

    prepare_dataset_artifacts(cfg)


def prepare_dataset_artifacts(cfg: dict) -> None:
    """Produce or verify paths.data_out and its split files.

    data.generate_data_out (default true) allows building missing or stale
    artifacts; false only verifies existing ones. The split set follows
    cross_validation.enabled: train.h5 + test.h5 with CV (fold validation
    comes from profiling rows), train.h5 + val.h5 + test.h5 without.
    """
    # Local import keeps this module importable without the HDF5 stack.
    from utils.data_split import _split_paths, perform_and_save_data_split

    model_name = str(cfg.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("cfg['model_name'] must be resolved before preparing data artifacts")

    perform_and_save_data_split(cfg, model_name=model_name)

    train_path, validation_path, test_path = _split_paths(cfg, model_name=model_name)
    # logger.info(
    #     "Dataset artifacts ready: data_out=%s | splits=%s",
    #     to_portable_path(resolve_path(cfg["paths"]["data_out"], cfg, inject=True)),
    #     ", ".join(path.name for path in produced),
    # )
