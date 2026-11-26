# services/mlops/schemas/learning.py
# ============================================================
# Learning schema: learning method + training strategy flags
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from services.mlops.schemas.common import (
    ensure_one_of,
    normalize_bool,
)

# ============================================================
# Learning Methods (strings by design)
# ============================================================

class LearningMethod:
    """
    Supported learning methods for hypothesis definitions.

    Kept as strings to allow downstream systems (SCA/Finance)
    to extend without schema breakage.
    """
    SUPERVISED = "supervised"
    SELF_SUPERVISED = "self_supervised"
    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi_supervised"
    TRANSFER = "transfer"
    FINE_TUNE = "fine_tune"
    ZERO_SHOT = "zero_shot"


# ============================================================
# Schema objects
# ============================================================

@dataclass(frozen=True)
class TrainingStrategy:
    """
    Flags controlling HOW training is executed.
    """
    distributed: bool = False
    mixed_precision: bool = False
    gradient_accumulation: bool = False
    checkpointing: bool = True
    resume_from_checkpoint: bool = False
    freeze_backbone: bool = False


@dataclass(frozen=True)
class LearningSpec:
    """
    Canonical learning specification for a hypothesis.

    Supports your YAML shape:

      learning:
        method: supervised
        label_space: key_byte
        loss:
          type: cross_entropy
          args: { ... }

    While also supporting the older/alternate shape:

      learning:
        mode: supervised

    Unknown keys are preserved in `extra` for forward-compatibility.
    """
    method: str
    strategy: TrainingStrategy = field(default_factory=TrainingStrategy)
    label_space: Optional[str] = None
    loss: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Validators
# ============================================================

def validate_learning_block(cfg: Dict[str, Any]) -> LearningSpec:
    """
    Validate learning section.

    Accepts:
      - learning.method (preferred; matches your auto_mlops.yaml)
      - learning.mode   (legacy compatibility)

    Also accepts optional strategy flags:
      - distributed, mixed_precision, gradient_accumulation,
        checkpointing, resume_from_checkpoint, freeze_backbone

    Preserves unknown keys into LearningSpec.extra.
    """
    if not isinstance(cfg, dict):
        raise ValueError("learning must be a dict")

    # Your YAML uses 'method'. Some older code used 'mode'.
    method = cfg.get("method", None)
    if method is None:
        method = cfg.get("mode", None)

    if method is None:
        # keep error message crisp (this is what users must fix)
        raise ValueError("[learning] missing required key: 'method' (or legacy 'mode')")

    method = str(method).strip()
    ensure_one_of(
        method,
        [
            LearningMethod.SUPERVISED,
            LearningMethod.SELF_SUPERVISED,
            LearningMethod.UNSUPERVISED,
            LearningMethod.SEMI_SUPERVISED,
            LearningMethod.TRANSFER,
            LearningMethod.FINE_TUNE,
            LearningMethod.ZERO_SHOT,
        ],
        ctx="learning.method",
    )

    # Strategy flags (optional)
    strategy = TrainingStrategy(
        distributed=normalize_bool(cfg.get("distributed")),
        mixed_precision=normalize_bool(cfg.get("mixed_precision")),
        gradient_accumulation=normalize_bool(cfg.get("gradient_accumulation")),
        checkpointing=normalize_bool(cfg.get("checkpointing"), default=True),
        resume_from_checkpoint=normalize_bool(cfg.get("resume_from_checkpoint")),
        freeze_backbone=normalize_bool(cfg.get("freeze_backbone")),
    )

    label_space = cfg.get("label_space", None)
    if label_space is not None:
        label_space = str(label_space).strip()

    loss = cfg.get("loss", None)
    if loss is not None and not isinstance(loss, dict):
        raise ValueError("[learning.loss] must be a dict if provided")

    # Preserve unknown / future fields (forward-compatible)
    known = {
        "method", "mode",
        "distributed", "mixed_precision", "gradient_accumulation",
        "checkpointing", "resume_from_checkpoint", "freeze_backbone",
        "label_space", "loss",
    }
    extra = {k: v for k, v in cfg.items() if k not in known}

    return LearningSpec(
        method=method,
        strategy=strategy,
        label_space=label_space,
        loss=loss,
        extra=extra,
    )


# ============================================================
# End of learning schema
# ============================================================
