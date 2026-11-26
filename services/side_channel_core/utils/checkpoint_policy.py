#services/side_channel_core/utils/checkpoint_policy.py
"""
Validation-checkpoint and test-model selection for Side-Channel Analysis.

This module is the single source of truth for the lifecycle:

    training
        -> optional best-validation checkpoint
        -> mandatory final-epoch checkpoint
        -> checkpoint manifest
        -> test-time checkpoint resolution

Canonical configuration
-----------------------

evaluation:
  validation:
    checkpointing: true
  test:
    load_model: best_validation  # best_validation | final_epoch

Scientific contract
-------------------
- ``best_validation`` may be used only when validation checkpointing produced a
  valid selected checkpoint.
- When validation checkpointing is disabled, a requested ``best_validation``
  policy falls back explicitly to ``final_epoch`` and records the reason.
- When checkpointing is enabled but the selected checkpoint or manifest is
  missing/corrupt, test fails. It never silently falls back.
- Every canonical checkpoint is fingerprinted with SHA-256 and verified before
  test loading.

Author: Shraddha Mishra
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Optional
from uuid import uuid4

from utils.logger import get_logger
from utils.path_utils import resolve_path
from utils.seal import archive_digest, check as check_seal, seal_text, strip as strip_seal

logger = get_logger("checkpoint_policy")

CHECKPOINT_MANIFEST_SCHEMA_VERSION = "1.0"
SUPPORTED_TEST_MODEL_POLICIES = frozenset({"best_validation", "final_epoch"})


class CheckpointPolicyError(RuntimeError):
    """Raised when checkpoint policy or provenance is invalid."""


@dataclass(frozen=True)
class CheckpointPaths:
    """Canonical checkpoint artifact paths for one model run."""

    checkpoint_dir: Path
    model_out_dir: Path
    best_validation: Path
    final_epoch: Path
    manifest: Path

    def epoch_checkpoint(self, epoch: int) -> Path:
        epoch = _strict_int(epoch, field="epoch", minimum=1)
        return self.checkpoint_dir / f"{self.best_validation.stem.removesuffix('_best_validation')}_ep{epoch}.pt"


@dataclass(frozen=True)
class CheckpointResolution:
    """Fully validated test-time checkpoint decision."""

    requested_policy: str
    effective_policy: str
    validation_checkpointing: bool
    checkpoint_path: Path
    checkpoint_epoch: int
    checkpoint_sha256: str
    manifest_path: Path
    fallback_reason: Optional[str]
    selection_metric: Optional[str]
    selection_mode: Optional[str]
    selection_score: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoint_path"] = str(self.checkpoint_path)
        payload["manifest_path"] = str(self.manifest_path)
        return payload


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping, got {type(value).__name__}")
    return value


def _strict_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be true or false, got {value!r}")
    return value


def _strict_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer, got {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {result}")
    return result


def _optional_finite_float(value: Any, *, field: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric or null, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric or null, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {result!r}")
    return result


def _normalize_policy(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string, got {type(value).__name__}")
    policy = value.strip().lower()
    if policy not in SUPPORTED_TEST_MODEL_POLICIES:
        raise ValueError(
            f"{field} must be one of {sorted(SUPPORTED_TEST_MODEL_POLICIES)}, "
            f"got {value!r}"
        )
    return policy


def get_validation_checkpointing(cfg: Mapping[str, Any]) -> bool:
    """Return canonical ``evaluation.validation.checkpointing``."""
    _require_mapping(cfg, field="cfg")
    evaluation = _require_mapping(cfg.get("evaluation") or {}, field="cfg['evaluation']")
    validation = _require_mapping(
        evaluation.get("validation") or {},
        field="cfg['evaluation']['validation']",
    )

    if "checkpointing" not in validation:
        raise KeyError(
            "cfg['evaluation']['validation']['checkpointing'] is required. "
            "Move the setting from deprecated trainer.checkpointing."
        )

    canonical = _strict_bool(
        validation["checkpointing"],
        field="evaluation.validation.checkpointing",
    )

    trainer = cfg.get("trainer") or {}
    if trainer:
        trainer = _require_mapping(trainer, field="cfg['trainer']")
        if "checkpointing" in trainer:
            legacy = _strict_bool(
                trainer["checkpointing"],
                field="trainer.checkpointing",
            )
            if legacy != canonical:
                raise CheckpointPolicyError(
                    "Conflicting checkpoint settings: "
                    f"evaluation.validation.checkpointing={canonical} but "
                    f"deprecated trainer.checkpointing={legacy}. Remove the legacy field."
                )
            logger.warning(
                "trainer.checkpointing is deprecated and ignored; remove it and use "
                "evaluation.validation.checkpointing only"
            )

    return canonical


def get_requested_test_policy(cfg: Mapping[str, Any]) -> str:
    """Return canonical ``evaluation.test.load_model``."""
    _require_mapping(cfg, field="cfg")
    evaluation = _require_mapping(cfg.get("evaluation") or {}, field="cfg['evaluation']")
    test_cfg = _require_mapping(
        evaluation.get("test") or {},
        field="cfg['evaluation']['test']",
    )
    if "load_model" not in test_cfg:
        raise KeyError(
            "cfg['evaluation']['test']['load_model'] is required and must be "
            "best_validation or final_epoch"
        )
    return _normalize_policy(
        test_cfg["load_model"],
        field="evaluation.test.load_model",
    )


def checkpoint_paths(cfg: Mapping[str, Any], *, model_name: str) -> CheckpointPaths:
    """Resolve canonical checkpoint paths for one run."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    model_name = model_name.strip()

    paths_cfg = _require_mapping(cfg.get("paths") or {}, field="cfg['paths']")
    for key in ("checkpoints", "model_out"):
        if not paths_cfg.get(key):
            raise KeyError(f"cfg['paths']['{key}'] is required")

    checkpoint_dir = Path(
        resolve_path(paths_cfg["checkpoints"], cfg, model_name=model_name)
    )
    model_out_dir = Path(
        resolve_path(paths_cfg["model_out"], cfg, model_name=model_name)
    )

    return CheckpointPaths(
        checkpoint_dir=checkpoint_dir,
        model_out_dir=model_out_dir,
        best_validation=model_out_dir / f"{model_name}_best_validation.pt",
        final_epoch=model_out_dir / f"{model_name}_trained.pt",
        manifest=model_out_dir / f"{model_name}_checkpoint_manifest.json",
    )


def checkpoint_content_sha256(path: Path) -> str:
    """The checkpoint's content: its archive members, without the seal in the zip comment
    (utils/seal.py), which binds each copy to its own path; a non-zip file's whole bytes."""
    import zipfile

    return archive_digest(path) if zipfile.is_zipfile(path) else file_sha256(path)


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")
    chunk_size = _strict_int(chunk_size, field="chunk_size", minimum=1)

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = json.dumps(
        dict(payload),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    serialized = seal_text(path, serialized)  # utils/seal.py

    # tempfile.mkstemp() forces mode 0600, and os.replace() would preserve that
    # restrictive mode on the published manifest. Exclusive normal file
    # creation instead follows the process umask, just like an ordinary JSON
    # file, without embedding chmod/permission policy in application code.
    temp_path = path.parent / (
        f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )

    try:
        with temp_path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return path


def clear_checkpoint_artifacts(cfg: Mapping[str, Any], *, model_name: str) -> CheckpointPaths:
    """Remove stale canonical artifacts before a fresh training run. paths.checkpoints is
    created only when evaluation.validation.checkpointing is on: nothing is written there
    otherwise."""
    paths = checkpoint_paths(cfg, model_name=model_name)
    if get_validation_checkpointing(cfg):
        paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths.model_out_dir.mkdir(parents=True, exist_ok=True)

    for path in (paths.best_validation, paths.final_epoch, paths.manifest):
        path.unlink(missing_ok=True)

    for path in paths.checkpoint_dir.glob(f"{model_name}_ep*.pt"):
        if path.is_file():
            path.unlink()

    return paths


def write_checkpoint_manifest(
    *,
    cfg: Mapping[str, Any],
    model_name: str,
    model_hash: Optional[str],
    validation_checkpointing: bool,
    monitor_metric: Optional[str],
    monitor_mode: Optional[str],
    best_epoch: Optional[int],
    best_score: Optional[float],
    epoch_checkpoint: Optional[Path],
    best_validation_checkpoint: Optional[Path],
    final_epoch: int,
    final_checkpoint: Path,
    epochs_configured: int,
    execution: Mapping[str, Any],
) -> Path:
    """Write the authoritative training/checkpoint manifest."""
    paths = checkpoint_paths(cfg, model_name=model_name)
    validation_checkpointing = _strict_bool(
        validation_checkpointing,
        field="validation_checkpointing",
    )
    final_epoch = _strict_int(final_epoch, field="final_epoch", minimum=1)
    epochs_configured = _strict_int(
        epochs_configured,
        field="epochs_configured",
        minimum=1,
    )
    if final_epoch > epochs_configured:
        raise CheckpointPolicyError(
            "final_epoch cannot exceed epochs_configured: "
            f"final_epoch={final_epoch}, epochs_configured={epochs_configured}"
        )

    if not isinstance(model_hash, str) or not model_hash.strip():
        raise CheckpointPolicyError(
            "model_hash must be a non-empty string before publishing a manifest"
        )
    model_hash = model_hash.strip()

    final_checkpoint = Path(final_checkpoint).resolve()
    if final_checkpoint != paths.final_epoch.resolve():
        raise CheckpointPolicyError(
            "Final checkpoint path is not canonical: "
            f"expected={paths.final_epoch}, got={final_checkpoint}"
        )
    final_digest = file_sha256(final_checkpoint)

    selection: Optional[dict[str, Any]]
    if best_epoch is None:
        if epoch_checkpoint is not None or best_validation_checkpoint is not None:
            raise CheckpointPolicyError(
                "Checkpoint paths cannot be present when best_epoch is null"
            )
        selection = None
    else:
        best_epoch = _strict_int(best_epoch, field="best_epoch", minimum=1)
        if best_epoch > final_epoch:
            raise CheckpointPolicyError(
                f"best_epoch={best_epoch} cannot exceed final_epoch={final_epoch}"
            )
        if not isinstance(monitor_metric, str) or not monitor_metric.strip():
            raise CheckpointPolicyError(
                "best_epoch requires a non-empty monitor_metric"
            )
        if not isinstance(monitor_mode, str):
            raise CheckpointPolicyError(
                "best_epoch requires monitor_mode='minimize' or 'maximize'"
            )
        monitor_mode = monitor_mode.strip().lower()
        if monitor_mode not in {"minimize", "maximize"}:
            raise CheckpointPolicyError(
                "monitor_mode must be 'minimize' or 'maximize', "
                f"got {monitor_mode!r}"
            )
        normalized_score = _optional_finite_float(best_score, field="best_score")
        if normalized_score is None:
            raise CheckpointPolicyError("best_epoch requires a finite best_score")
        selection = {
            "epoch": best_epoch,
            "metric": monitor_metric.strip(),
            "mode": monitor_mode,
            "score": normalized_score,
            "checkpointing_enabled": validation_checkpointing,
            "epoch_checkpoint": None,
            "epoch_checkpoint_sha256": None,
            "best_validation_checkpoint": None,
            "best_validation_checkpoint_sha256": None,
        }

        if validation_checkpointing:
            if epoch_checkpoint is None or best_validation_checkpoint is None:
                raise CheckpointPolicyError(
                    "Validation checkpointing is enabled but selected checkpoint paths are missing"
                )
            epoch_checkpoint = Path(epoch_checkpoint).resolve()
            best_validation_checkpoint = Path(best_validation_checkpoint).resolve()
            if best_validation_checkpoint != paths.best_validation.resolve():
                raise CheckpointPolicyError(
                    "Best-validation checkpoint path is not canonical: "
                    f"expected={paths.best_validation}, got={best_validation_checkpoint}"
                )
            expected_epoch_path = paths.epoch_checkpoint(best_epoch).resolve()
            if epoch_checkpoint != expected_epoch_path:
                raise CheckpointPolicyError(
                    "Selected epoch checkpoint path does not match selected epoch: "
                    f"expected={expected_epoch_path}, got={epoch_checkpoint}"
                )
            epoch_digest = file_sha256(epoch_checkpoint)
            best_digest = file_sha256(best_validation_checkpoint)
            # The same archive members; the files differ only in their seals.
            if checkpoint_content_sha256(epoch_checkpoint) != checkpoint_content_sha256(best_validation_checkpoint):
                raise CheckpointPolicyError(
                    "Selected epoch checkpoint and canonical best-validation "
                    "checkpoint do not hold the same content"
                )
            selection.update(
                {
                    "epoch_checkpoint": str(epoch_checkpoint),
                    "epoch_checkpoint_sha256": epoch_digest,
                    "best_validation_checkpoint": str(best_validation_checkpoint),
                    "best_validation_checkpoint_sha256": best_digest,
                }
            )

    payload = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "model": {
            "name": model_name,
            "hash": model_hash,
        },
        "validation_checkpointing": validation_checkpointing,
        "selection": selection,
        "final": {
            "epoch": final_epoch,
            "checkpoint": str(final_checkpoint),
            "checkpoint_sha256": final_digest,
        },
        "training": {
            "epochs_configured": epochs_configured,
            "epochs_completed": final_epoch,
        },
        "execution": dict(execution),
    }

    return _atomic_write_json(paths.manifest, payload)


def load_checkpoint_manifest(
    cfg: Mapping[str, Any],
    *,
    model_name: str,
) -> dict[str, Any]:
    """Load and structurally validate one checkpoint manifest."""
    paths = checkpoint_paths(cfg, model_name=model_name)
    if not paths.manifest.is_file():
        raise FileNotFoundError(f"Checkpoint manifest not found: {paths.manifest}")

    check_seal(paths.manifest)
    try:
        payload = strip_seal(json.loads(paths.manifest.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise CheckpointPolicyError(
            f"Invalid checkpoint manifest JSON: {paths.manifest}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise CheckpointPolicyError("Checkpoint manifest must contain a JSON object")
    if payload.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
        raise CheckpointPolicyError(
            "Unsupported checkpoint manifest schema: "
            f"{payload.get('schema_version')!r}"
        )

    model = _require_mapping(payload.get("model") or {}, field="manifest.model")
    if model.get("name") != model_name:
        raise CheckpointPolicyError(
            f"Checkpoint manifest model mismatch: expected={model_name!r}, "
            f"found={model.get('name')!r}"
        )

    configured_hash = cfg.get("model_hash")
    manifest_hash = model.get("hash")
    if not isinstance(manifest_hash, str) or not manifest_hash.strip():
        raise CheckpointPolicyError("manifest.model.hash must be a non-empty string")
    if configured_hash is not None and str(configured_hash) != manifest_hash:
        raise CheckpointPolicyError(
            "Checkpoint manifest model-hash mismatch: "
            f"configuration={configured_hash!r}, manifest={manifest_hash!r}"
        )

    manifest_checkpointing = _strict_bool(
        payload.get("validation_checkpointing"),
        field="manifest.validation_checkpointing",
    )
    configured_checkpointing = get_validation_checkpointing(cfg)
    if manifest_checkpointing != configured_checkpointing:
        raise CheckpointPolicyError(
            "Checkpoint manifest/config mismatch: "
            f"manifest validation_checkpointing={manifest_checkpointing}, "
            f"configuration={configured_checkpointing}"
        )

    final = _require_mapping(payload.get("final") or {}, field="manifest.final")
    final_epoch = _strict_int(
        final.get("epoch"), field="manifest.final.epoch", minimum=1
    )
    if not isinstance(final.get("checkpoint"), str) or not final["checkpoint"]:
        raise CheckpointPolicyError("manifest.final.checkpoint must be a path string")
    if not isinstance(final.get("checkpoint_sha256"), str):
        raise CheckpointPolicyError("manifest.final.checkpoint_sha256 must be a string")

    training = _require_mapping(
        payload.get("training") or {}, field="manifest.training"
    )
    epochs_configured = _strict_int(
        training.get("epochs_configured"),
        field="manifest.training.epochs_configured",
        minimum=1,
    )
    epochs_completed = _strict_int(
        training.get("epochs_completed"),
        field="manifest.training.epochs_completed",
        minimum=1,
    )
    if epochs_completed != final_epoch:
        raise CheckpointPolicyError(
            "manifest.training.epochs_completed must equal manifest.final.epoch"
        )
    if final_epoch > epochs_configured:
        raise CheckpointPolicyError(
            "manifest.final.epoch cannot exceed training.epochs_configured"
        )
    execution = _require_mapping(
        payload.get("execution") or {}, field="manifest.execution"
    )
    # With CV enabled, only the full-profiling refit is an official test
    # model. Fold assessment never publishes a manifest, and a standard-run
    # manifest must not stand in for a CV result (or vice versa).
    cv_cfg = cfg.get("cross_validation") or {}
    cv_enabled = isinstance(cv_cfg, Mapping) and cv_cfg.get("enabled") is True
    execution_type = execution.get("type")
    if cv_enabled and execution_type != "refit":
        raise CheckpointPolicyError(
            "cross_validation.enabled=true requires the CV full-profiling refit "
            f"model, but the manifest execution type is {execution_type!r}"
        )
    # A model import (utils/model_import.py) publishes the released weights, never a
    # trained model: its manifest says so, and only an imported run's config accepts it.
    imported = isinstance(cfg.get("model_import"), Mapping)
    if imported and execution_type != "model_import":
        raise CheckpointPolicyError(
            "an imported run's checkpoint must come from its model import, but the manifest "
            f"execution type is {execution_type!r}"
        )
    if not cv_enabled and not imported and execution_type not in (None, "standard"):
        raise CheckpointPolicyError(
            "cross_validation.enabled=false requires a standard training model, "
            f"but the manifest execution type is {execution_type!r}"
        )

    selection = payload.get("selection")
    if selection is not None:
        selection = _require_mapping(selection, field="manifest.selection")
        selected_epoch = _strict_int(
            selection.get("epoch"), field="manifest.selection.epoch", minimum=1
        )
        if selected_epoch > final_epoch:
            raise CheckpointPolicyError(
                "manifest.selection.epoch cannot exceed manifest.final.epoch"
            )
        metric = selection.get("metric")
        if not isinstance(metric, str) or not metric.strip():
            raise CheckpointPolicyError(
                "manifest.selection.metric must be a non-empty string"
            )
        mode = selection.get("mode")
        if not isinstance(mode, str) or mode.strip().lower() not in {
            "minimize",
            "maximize",
        }:
            raise CheckpointPolicyError(
                "manifest.selection.mode must be 'minimize' or 'maximize'"
            )
        score = _optional_finite_float(
            selection.get("score"), field="manifest.selection.score"
        )
        if score is None:
            raise CheckpointPolicyError(
                "manifest.selection.score must be a finite number"
            )
        selected_checkpointing = _strict_bool(
            selection.get("checkpointing_enabled"),
            field="manifest.selection.checkpointing_enabled",
        )
        if selected_checkpointing != manifest_checkpointing:
            raise CheckpointPolicyError(
                "manifest.selection.checkpointing_enabled conflicts with "
                "manifest.validation_checkpointing"
            )

    return payload


def _same_run_file(named: Path, expected: Path) -> bool:
    """Whether a manifest's path names this run's file under another checkout root: the same parts from the last
    `projects` folder on (projects/<project>/artifacts/<env>/generated_models/<run>/...)."""
    parts = expected.resolve().parts
    if "projects" not in parts:
        return False
    tail = parts[len(parts) - 1 - parts[::-1].index("projects"):]
    return len(named.parts) >= len(tail) and tuple(named.parts[-len(tail):]) == tail


def _verify_manifest_checkpoint(
    *,
    path: Path,
    expected_path: Path,
    expected_sha256: str,
    field: str,
) -> tuple[Path, str]:
    resolved = Path(path).resolve()
    if resolved != expected_path.resolve():
        # The manifest names the file where training wrote it: the container's /app checkout. On another root (a
        # host test, a copied checkout) the same run file is the one at the same place under projects/; its
        # SHA-256 below is the guarantee (as trace_view._checkpoint reads it).
        if not _same_run_file(Path(path), expected_path):
            raise CheckpointPolicyError(
                f"{field} path is not canonical: expected={expected_path}, got={resolved}"
            )
        resolved = expected_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} does not exist: {resolved}")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise CheckpointPolicyError(f"{field} manifest SHA-256 is invalid")
    observed = file_sha256(resolved)
    if observed != expected_sha256:
        raise CheckpointPolicyError(
            f"{field} SHA-256 mismatch: expected={expected_sha256}, observed={observed}"
        )
    return resolved, observed


def resolve_test_checkpoint(
    cfg: Mapping[str, Any],
    *,
    model_name: str,
) -> CheckpointResolution:
    """Resolve and cryptographically verify the checkpoint to test."""
    requested = get_requested_test_policy(cfg)
    validation_checkpointing = get_validation_checkpointing(cfg)
    paths = checkpoint_paths(cfg, model_name=model_name)
    manifest = load_checkpoint_manifest(cfg, model_name=model_name)

    final = _require_mapping(manifest["final"], field="manifest.final")

    if requested == "best_validation" and validation_checkpointing:
        selection = manifest.get("selection")
        if not isinstance(selection, Mapping):
            raise CheckpointPolicyError(
                "best_validation was requested with validation checkpointing enabled, "
                "but the manifest contains no selected validation checkpoint"
            )
        checkpoint_raw = selection.get("best_validation_checkpoint")
        digest_raw = selection.get("best_validation_checkpoint_sha256")
        if not isinstance(checkpoint_raw, str) or not checkpoint_raw:
            raise CheckpointPolicyError(
                "Manifest is missing selection.best_validation_checkpoint"
            )
        checkpoint_path, digest = _verify_manifest_checkpoint(
            path=Path(checkpoint_raw),
            expected_path=paths.best_validation,
            expected_sha256=str(digest_raw),
            field="best-validation checkpoint",
        )
        epoch = _strict_int(
            selection.get("epoch"),
            field="manifest.selection.epoch",
            minimum=1,
        )
        selection_score = _optional_finite_float(
            selection.get("score"),
            field="manifest.selection.score",
        )
        selection_metric_raw = selection.get("metric")
        selection_mode_raw = selection.get("mode")
        selection_metric = (
            str(selection_metric_raw)
            if selection_metric_raw is not None
            else None
        )
        selection_mode = (
            str(selection_mode_raw)
            if selection_mode_raw is not None
            else None
        )

        return CheckpointResolution(
            requested_policy=requested,
            effective_policy="best_validation",
            validation_checkpointing=True,
            checkpoint_path=checkpoint_path,
            checkpoint_epoch=epoch,
            checkpoint_sha256=digest,
            manifest_path=paths.manifest.resolve(),
            fallback_reason=None,
            selection_metric=selection_metric,
            selection_mode=selection_mode,
            selection_score=selection_score,
        )

    checkpoint_raw = final.get("checkpoint")
    digest_raw = final.get("checkpoint_sha256")
    if not isinstance(checkpoint_raw, str) or not checkpoint_raw:
        raise CheckpointPolicyError("Manifest is missing final.checkpoint")
    checkpoint_path, digest = _verify_manifest_checkpoint(
        path=Path(checkpoint_raw),
        expected_path=paths.final_epoch,
        expected_sha256=str(digest_raw),
        field="final-epoch checkpoint",
    )
    epoch = _strict_int(final.get("epoch"), field="manifest.final.epoch", minimum=1)

    fallback_reason = None
    if requested == "best_validation":
        fallback_reason = "validation checkpointing is disabled; no best-validation checkpoint was created"

    return CheckpointResolution(
        requested_policy=requested,
        effective_policy="final_epoch",
        validation_checkpointing=validation_checkpointing,
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=epoch,
        checkpoint_sha256=digest,
        manifest_path=paths.manifest.resolve(),
        fallback_reason=fallback_reason,
        selection_metric=None,
        selection_mode=None,
        selection_score=None,
    )


__all__ = [
    "CHECKPOINT_MANIFEST_SCHEMA_VERSION",
    "SUPPORTED_TEST_MODEL_POLICIES",
    "CheckpointPaths",
    "CheckpointPolicyError",
    "CheckpointResolution",
    "checkpoint_paths",
    "clear_checkpoint_artifacts",
    "file_sha256",
    "checkpoint_content_sha256",
    "get_requested_test_policy",
    "get_validation_checkpointing",
    "load_checkpoint_manifest",
    "resolve_test_checkpoint",
    "write_checkpoint_manifest",
]
