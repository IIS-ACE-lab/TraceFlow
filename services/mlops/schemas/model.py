# services/mlops/schemas/model.py
# ============================================================
# Model schema: family, architecture hints, weight inheritance
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mlops.schemas.common import (
    require_keys,
    ensure_one_of,
    normalize_bool,
    normalize_dict,
    normalize_list,
    normalize_str,
)


# ============================================================
# Model Families (v1 baseline)
# ============================================================

class ModelFamily:
    CNN = "cnn"
    TRANSFORMER = "transformer"
    HYBRID = "hybrid"
    RNN = "rnn"
    GRAPH = "graph"


class ModelGenerator:
    YAML_SEQUENTIAL = "yaml_sequential"
    YAML_GRAPH = "yaml_graph"
    NAMED_ARCHITECTURE = "named_architecture"


# ============================================================
# Weight Inheritance / Transfer
# ============================================================

@dataclass(frozen=True)
class WeightInheritance:
    """
    Declarative instruction to inherit weights from another project/run.

    This does NOT execute transfer. It only describes intent; the training
    launcher / pretraining resolver consumes it.
    """
    enabled: bool = False
    source_project: Optional[str] = None
    selector: Optional[str] = None  # promoted_best | by_name | by_hash | custom
    source_ref: Optional[str] = None  # name/hash if selector requires it
    freeze_policy: Optional[str] = None  # full | partial | none
    layers_regex: List[str] = field(default_factory=list)
    unfreeze_after_epoch: Optional[int] = None


# ============================================================
# Model Spec
# ============================================================

@dataclass(frozen=True)
class ModelSpec:
    family: str
    generator: str
    # Hints commonly used by builder:
    architecture: Optional[str] = None        # for named_architecture generator
    arch_params: Dict[str, Any] = field(default_factory=dict)
    # Search space blob (validated elsewhere in hypothesis schema):
    search_space: Dict[str, Any] = field(default_factory=dict)
    # Weight inheritance (transfer flags):
    inheritance: WeightInheritance = field(default_factory=WeightInheritance)


# ============================================================
# Validators
# ============================================================

def _validate_inheritance(cfg: Optional[Dict[str, Any]]) -> WeightInheritance:
    if not cfg:
        return WeightInheritance(enabled=False)

    enabled = normalize_bool(cfg.get("enabled"), default=True)

    layers_regex = normalize_list(cfg.get("layers_regex"))
    layers_regex = [normalize_str(x, ctx="model.inheritance.layers_regex") for x in layers_regex]

    return WeightInheritance(
        enabled=enabled,
        source_project=cfg.get("project") or cfg.get("source_project"),
        selector=cfg.get("model_selector") or cfg.get("selector"),
        source_ref=cfg.get("source_ref"),
        freeze_policy=(cfg.get("freeze") or {}).get("policy") if isinstance(cfg.get("freeze"), dict) else cfg.get("freeze_policy"),
        layers_regex=layers_regex,
        unfreeze_after_epoch=(cfg.get("adaptation") or {}).get("unfreeze_after_epoch")
        if isinstance(cfg.get("adaptation"), dict)
        else cfg.get("unfreeze_after_epoch"),
    )


def validate_model_block(cfg: Dict[str, Any]) -> ModelSpec:
    """
    Validate a model block.

    Expected shapes (v1):
      model:
        family: cnn
        generator: yaml_sequential
        search_space: {...}

      model:
        family: transformer
        generator: named_architecture
        search_space:
          architecture: [TraceTransformer, ...]
          arch_params: {...}

    Weight inheritance is optional:
      learning:
        source: {...}
      OR
      model:
        inheritance: {...}
    """
    require_keys(cfg, ["family", "generator"], ctx="model")

    family = normalize_str(cfg["family"], ctx="model.family")
    ensure_one_of(
        family,
        [
            ModelFamily.CNN,
            ModelFamily.TRANSFORMER,
            ModelFamily.HYBRID,
            ModelFamily.RNN,
            ModelFamily.GRAPH,
        ],
        ctx="model.family",
    )

    generator = normalize_str(cfg["generator"], ctx="model.generator")
    ensure_one_of(
        generator,
        [
            ModelGenerator.YAML_SEQUENTIAL,
            ModelGenerator.YAML_GRAPH,
            ModelGenerator.NAMED_ARCHITECTURE,
        ],
        ctx="model.generator",
    )

    search_space = normalize_dict(cfg.get("search_space"))
    # convenience for named_architecture: allow architecture/arch_params at top-level too
    architecture = cfg.get("architecture") or search_space.get("architecture")
    arch_params = cfg.get("arch_params") or search_space.get("arch_params") or {}

    if architecture is not None:
        architecture = normalize_str(architecture, ctx="model.architecture") if not isinstance(architecture, list) else None
        # If list, hypothesis schema should handle sampling; we don't coerce here.

    arch_params = normalize_dict(arch_params)

    inheritance = _validate_inheritance(cfg.get("inheritance"))

    return ModelSpec(
        family=family,
        generator=generator,
        architecture=architecture,
        arch_params=arch_params,
        search_space=search_space,
        inheritance=inheritance,
    )


# ============================================================
# End of model schema
# ============================================================
