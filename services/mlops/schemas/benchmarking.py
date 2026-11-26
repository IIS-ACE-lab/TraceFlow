# services/mlops/schemas/benchmarking.py
# ============================================================
# Benchmarking schema: benchmark comparison contract + leaderboard semantics
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.mlops.schemas.common import (
    normalize_bool,
    normalize_dict,
    normalize_list,
    normalize_str,
)


# ============================================================
# Dataclasses
# ============================================================

@dataclass(frozen=True)
class BenchmarkTestSpec:
    """
    A single benchmark test definition.

    Examples:
      - standard_attack
      - desync_robustness (with desync_values)
      - noise_robustness (with noise_levels)
      - cross_device_eval
    """
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualifyThresholds:
    """
    Canonical qualification thresholds for a hypothesis.
    These are hypothesis-specific; global objectives remain separate.
    """
    guessing_entropy_max: Optional[float] = None
    mean_rank_max: Optional[float] = None
    cross_entropy_max: Optional[float] = None

    accuracy_min: Optional[float] = None
    rank_k_success_at5_min: Optional[float] = None  # YAML: rank_k_success@5_min


@dataclass(frozen=True)
class BenchmarkingSpec:
    enabled: bool
    tests: List[BenchmarkTestSpec]
    qualify: QualifyThresholds


# ============================================================
# Parsing helpers
# ============================================================

def _parse_test(item: Dict[str, Any], ctx: str) -> BenchmarkTestSpec:
    item = normalize_dict(item)
    name = normalize_str(item.get("name"), default="")
    if not name:
        raise ValueError(f"[schemas/benchmarking] missing test.name in {ctx}")

    enabled = normalize_bool(item.get("enabled"), default=True)

    # Keep arbitrary knobs under params (everything except name/enabled)
    params = {k: v for k, v in item.items() if k not in ("name", "enabled")}
    return BenchmarkTestSpec(name=name, enabled=enabled, params=params)


def _coerce_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except Exception:
        return None


def _parse_qualify_thresholds(raw: Dict[str, Any]) -> QualifyThresholds:
    raw = normalize_dict(raw)

    # Support your YAML keys exactly (including rank_k_success@5_min)
    rk5 = raw.get("rank_k_success@5_min", raw.get("rank_k_success_at5_min"))

    return QualifyThresholds(
        guessing_entropy_max=_coerce_float(raw.get("guessing_entropy_max")),
        mean_rank_max=_coerce_float(raw.get("mean_rank_max")),
        cross_entropy_max=_coerce_float(raw.get("cross_entropy_max")),
        accuracy_min=_coerce_float(raw.get("accuracy_min")),
        rank_k_success_at5_min=_coerce_float(rk5),
    )


# ============================================================
# Public validator
# ============================================================

def validate_benchmarking_block(cfg: Dict[str, Any], *, ctx: str = "hypothesis.benchmarking") -> BenchmarkingSpec:
    """
    Validates hypothesis-level benchmarking block:

      benchmarking:
        enabled: true
        tests:
          - name: standard_attack
            enabled: true
          - name: desync_robustness
            enabled: true
            desync_values: [0,25,50,100]
        qualify_thresholds:
          guessing_entropy_max: 40
          rank_k_success@5_min: 0.60

    Returns BenchmarkingSpec.
    """
    cfg = normalize_dict(cfg)

    enabled = normalize_bool(cfg.get("enabled"), default=False)

    tests_raw = normalize_list(cfg.get("tests", []))
    tests: List[BenchmarkTestSpec] = []
    for i, t in enumerate(tests_raw):
        if not isinstance(t, dict):
            raise ValueError(f"[schemas/benchmarking] tests[{i}] must be dict in {ctx}")
        tests.append(_parse_test(t, f"{ctx}.tests[{i}]"))

    qualify_raw = normalize_dict(cfg.get("qualify_thresholds", {}))
    qualify = _parse_qualify_thresholds(qualify_raw)

    return BenchmarkingSpec(enabled=enabled, tests=tests, qualify=qualify)


# ============================================================
# Leaderboard semantics (used by registry writer)
# ============================================================

# In v1, leaderboard row is computed from:
# - objectives.primary + tie breakers (global)
# - qualify pass/fail (hypothesis)
# - benchmark metrics (hypothesis tests)
#
# This helper defines the "canonical" column names we expect.

LEADERBOARD_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "project",
    "env",
    "hypothesis_id",
    "sota_trial",
    "trial_id",
    "status",           # COMPLETED / PRUNED / FAILED / PROMOTED / QUALIFIED / NONQUALIFIED
    "primary_metric",   # objectives.primary name
    "primary_value",
    "score",            # scalar score used for sorting (e.g., weighted selection policy output)
    "qualified",        # bool
    "promoted",         # bool
)

# Optional columns frequently useful
LEADERBOARD_OPTIONAL_COLUMNS: Tuple[str, ...] = (
    "started_at",
    "ended_at",
    "duration_sec",
    "params_hash",
    "model_family",
    "model_generator",
    "learning_method",
    "best_epoch",
)
