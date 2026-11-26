# services/mlops/schemas/retention.py
# ============================================================
# Retention schema: trial retention + artifact cleanup rules
# (future-proof, ctx-compatible)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mlops.schemas.common import (
    require_keys,
    normalize_bool,
    forbid_unknown_keys,
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}

def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else ([] if x is None else [x])

# ------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactPolicy:
    """
    Fine-grained artifact retention toggles for a trial outcome class.
    """
    keep_auto_params: bool = True
    keep_checkpoints: bool = True
    keep_weights: bool = True
    keep_metrics: bool = True
    keep_decision_trace: bool = True
    keep_logs: bool = True
    keep_reports: bool = True


@dataclass(frozen=True)
class RetentionKeep:
    """
    Coarse keep flags by trial outcome.
    Expandable without schema breakage.
    """
    promoted: bool = True
    qualified: bool = True
    completed: bool = True
    pruned: bool = True
    failed: bool = True
    interrupted: bool = True
    invalid_config: bool = True
    oom: bool = True
    timeout: bool = True


@dataclass(frozen=True)
class CheckpointPolicy:
    enabled: bool = False
    keep_last_k: int = 1
    keep_best: bool = True
    keep_every_n_epochs: Optional[int] = None


@dataclass(frozen=True)
class CompressionPolicy:
    enabled: bool = False
    formats: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetentionSpec:
    cleanup_enabled: bool = True
    protected_files: List[str] = field(default_factory=list)

    keep: RetentionKeep = field(default_factory=RetentionKeep)

    default_artifact_policy: ArtifactPolicy = field(default_factory=ArtifactPolicy)
    artifact_policy: Dict[str, ArtifactPolicy] = field(default_factory=dict)

    checkpoint_policy: Optional[CheckpointPolicy] = None
    compression: Optional[CompressionPolicy] = None

    def policy_for(self, outcome: str) -> ArtifactPolicy:
        o = (outcome or "").strip()
        return self.artifact_policy.get(o, self.default_artifact_policy)


# ------------------------------------------------------------
# Validators
# ------------------------------------------------------------

_ALLOWED_KEEP_KEYS = {
    "promoted", "qualified", "completed", "pruned", "failed",
    "interrupted", "invalid_config", "oom", "timeout",
}

_ALLOWED_ARTIFACT_POLICY_KEYS = {
    "keep_auto_params",
    "keep_checkpoints",
    "keep_weights",
    "keep_metrics",
    "keep_decision_trace",
    "keep_logs",
    "keep_reports",
}

_ALLOWED_TOP_KEYS = {
    "cleanup_enabled",
    "protected_files",
    "keep",
    "default_artifact_policy",
    "artifact_policy",
    "checkpoint_policy",
    "compression",
}

_ALLOWED_CHECKPOINT_POLICY_KEYS = {"enabled", "keep_last_k", "keep_best", "keep_every_n_epochs"}
_ALLOWED_COMPRESSION_KEYS = {"enabled", "formats"}


def _parse_keep_block(cfg: Dict[str, Any], *, ctx: str) -> RetentionKeep:
    forbid_unknown_keys(cfg, allowed=list(_ALLOWED_KEEP_KEYS), ctx=ctx, strict=True)
    return RetentionKeep(
        promoted=normalize_bool(cfg.get("promoted"), default=True),
        qualified=normalize_bool(cfg.get("qualified"), default=True),
        completed=normalize_bool(cfg.get("completed"), default=True),
        pruned=normalize_bool(cfg.get("pruned"), default=True),
        failed=normalize_bool(cfg.get("failed"), default=True),
        interrupted=normalize_bool(cfg.get("interrupted"), default=True),
        invalid_config=normalize_bool(cfg.get("invalid_config"), default=True),
        oom=normalize_bool(cfg.get("oom"), default=True),
        timeout=normalize_bool(cfg.get("timeout"), default=True),
    )


def _parse_artifact_policy_block(cfg: Dict[str, Any], *, ctx: str) -> ArtifactPolicy:
    forbid_unknown_keys(cfg, allowed=list(_ALLOWED_ARTIFACT_POLICY_KEYS), ctx=ctx, strict=True)
    return ArtifactPolicy(
        keep_auto_params=normalize_bool(cfg.get("keep_auto_params"), default=True),
        keep_checkpoints=normalize_bool(cfg.get("keep_checkpoints"), default=True),
        keep_weights=normalize_bool(cfg.get("keep_weights"), default=True),
        keep_metrics=normalize_bool(cfg.get("keep_metrics"), default=True),
        keep_decision_trace=normalize_bool(cfg.get("keep_decision_trace"), default=True),
        keep_logs=normalize_bool(cfg.get("keep_logs"), default=True),
        keep_reports=normalize_bool(cfg.get("keep_reports"), default=True),
    )


def _parse_checkpoint_policy(cfg: Dict[str, Any], *, ctx: str) -> CheckpointPolicy:
    forbid_unknown_keys(cfg, allowed=list(_ALLOWED_CHECKPOINT_POLICY_KEYS), ctx=ctx, strict=True)
    enabled = normalize_bool(cfg.get("enabled"), default=False)
    keep_last_k = int(cfg.get("keep_last_k", 1))
    if keep_last_k < 0:
        raise ValueError(f"[{ctx}] keep_last_k must be >= 0")
    keep_best = normalize_bool(cfg.get("keep_best"), default=True)
    keep_every_n_epochs = cfg.get("keep_every_n_epochs")
    if keep_every_n_epochs is not None:
        keep_every_n_epochs = int(keep_every_n_epochs)
        if keep_every_n_epochs <= 0:
            raise ValueError(f"[{ctx}] keep_every_n_epochs must be > 0")
    return CheckpointPolicy(
        enabled=enabled,
        keep_last_k=keep_last_k,
        keep_best=keep_best,
        keep_every_n_epochs=keep_every_n_epochs,
    )


def _parse_compression(cfg: Dict[str, Any], *, ctx: str) -> CompressionPolicy:
    forbid_unknown_keys(cfg, allowed=list(_ALLOWED_COMPRESSION_KEYS), ctx=ctx, strict=True)
    enabled = normalize_bool(cfg.get("enabled"), default=False)
    formats = [str(x) for x in _as_list(cfg.get("formats"))]
    return CompressionPolicy(enabled=enabled, formats=formats)


def validate_retention_block(cfg: Dict[str, Any], *, ctx: str = "retention") -> RetentionSpec:
    """
    Validate retention policy.

    Minimal accepted:
      retention:
        keep: { promoted, qualified, completed, pruned, failed }

    Recommended full:
      retention:
        cleanup_enabled: true
        protected_files: [...]
        keep: {... + interrupted/oom/timeout/invalid_config }
        default_artifact_policy: {...}
        artifact_policy: { pruned: {...}, failed: {...}, completed_nonqualified: {...} }
        checkpoint_policy: {...}
        compression: {...}
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"[{ctx}] must be a mapping")

    forbid_unknown_keys(cfg, allowed=list(_ALLOWED_TOP_KEYS), ctx=ctx, strict=True)

    # keep is required (because it controls trial retention safety)
    require_keys(cfg, ["keep"], ctx=ctx)

    cleanup_enabled = normalize_bool(cfg.get("cleanup_enabled"), default=True)
    protected_files = [str(x) for x in _as_list(cfg.get("protected_files"))]

    keep_raw = _as_dict(cfg.get("keep"))
    keep = _parse_keep_block(keep_raw, ctx=f"{ctx}.keep")

    default_policy = ArtifactPolicy()
    if cfg.get("default_artifact_policy") is not None:
        default_policy = _parse_artifact_policy_block(
            _as_dict(cfg.get("default_artifact_policy")),
            ctx=f"{ctx}.default_artifact_policy",
        )

    artifact_policy: Dict[str, ArtifactPolicy] = {}
    policy_raw = _as_dict(cfg.get("artifact_policy"))
    for outcome, block in policy_raw.items():
        if not isinstance(block, dict):
            raise ValueError(f"[{ctx}.artifact_policy] '{outcome}' must be a mapping")
        artifact_policy[str(outcome)] = _parse_artifact_policy_block(
            block, ctx=f"{ctx}.artifact_policy.{outcome}"
        )

    checkpoint_policy = None
    if cfg.get("checkpoint_policy") is not None:
        checkpoint_policy = _parse_checkpoint_policy(
            _as_dict(cfg.get("checkpoint_policy")),
            ctx=f"{ctx}.checkpoint_policy",
        )

    compression = None
    if cfg.get("compression") is not None:
        compression = _parse_compression(
            _as_dict(cfg.get("compression")),
            ctx=f"{ctx}.compression",
        )

    return RetentionSpec(
        cleanup_enabled=cleanup_enabled,
        protected_files=protected_files,
        keep=keep,
        default_artifact_policy=default_policy,
        artifact_policy=artifact_policy,
        checkpoint_policy=checkpoint_policy,
        compression=compression,
    )
