# services/mlops/schemas/pruning.py
# ============================================================
# Pruning schema: early stop rules & prune decisions
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mlops.schemas.common import (
    Decision,
    require_keys,
    ensure_one_of,
    normalize_float,
    normalize_int,
    normalize_list,
)


# ============================================================
# Early Stop Rule
# ============================================================

@dataclass(frozen=True)
class EarlyStopRule:
    """
    Declarative early stopping rule.

    Examples:
      - stop if loss > threshold after N epochs
      - stop if accuracy < threshold at step K
    """
    metric: str
    operator: str                # "<", "<=", ">", ">=", "=="
    threshold: float
    min_step: Optional[int] = None
    description: Optional[str] = None


# ============================================================
# Pruning Policy
# ============================================================

@dataclass(frozen=True)
class PruningSpec:
    """
    Pruning configuration.

    decision:
      - prune
      - retry
      - reject
    """
    decision: Decision = Decision.PRUNE
    rules: List[EarlyStopRule] = field(default_factory=list)
    grace_steps: Optional[int] = None
    enabled: bool = True


# ============================================================
# Validators
# ============================================================

def _validate_rule(cfg: Dict[str, Any]) -> EarlyStopRule:
    require_keys(cfg, ["metric", "operator", "threshold"], ctx="pruning.rule")

    op = cfg["operator"]
    if op not in ("<", "<=", ">", ">=", "=="):
        raise ValueError(f"[pruning.rule] invalid operator '{op}'")

    return EarlyStopRule(
        metric=str(cfg["metric"]),
        operator=op,
        threshold=normalize_float(cfg["threshold"]),
        min_step=normalize_int(cfg.get("min_step")),
        description=cfg.get("description"),
    )


def validate_pruning_block(cfg: Dict[str, Any]) -> PruningSpec:
    """
    Validate pruning section.

    Example:
      pruning:
        enabled: true
        decision: prune
        grace_steps: 5
        rules:
          - metric: loss
            operator: ">"
            threshold: 2.0
            min_step: 10
    """
    enabled = bool(cfg.get("enabled", True))

    decision = cfg.get("decision", Decision.PRUNE.value)
    ensure_one_of(decision, [d.value for d in Decision], ctx="pruning.decision")

    rules_cfg = normalize_list(cfg.get("rules"))
    rules: List[EarlyStopRule] = []

    for r in rules_cfg:
        if not isinstance(r, dict):
            raise ValueError("[pruning.rules] each rule must be a mapping")
        rules.append(_validate_rule(r))

    return PruningSpec(
        decision=Decision(decision),
        rules=rules,
        grace_steps=normalize_int(cfg.get("grace_steps")),
        enabled=enabled,
    )


# ============================================================
# End of pruning schema
# ============================================================
