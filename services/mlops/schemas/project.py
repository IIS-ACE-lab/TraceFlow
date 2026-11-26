# services/mlops/schemas/project.py
# ============================================================
# Project schema: project list (like master.yaml) + naming contract
#
# Validates top-level:
#   project:
#     - name: DLSCA_ASCAD
#       root: "DLSCA_ASCAD/"
#       overrides: {...}           # optional
#       allocation: {...}          # optional
#       hypotheses: [ {...}, ... ] # required non-empty
#       promotion: {...}           # optional
#
# Each hypothesis validates:
#   - hypothesis_id (required)
#   - learning/model/trainer/pruning/benchmarking (optional blocks)
#
# IMPORTANT:
# - This schema is for auto_mlops.yaml, not per-project params.yaml.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mlops.schemas.common import (
    normalize_bool,
    normalize_dict,
    normalize_list,
    normalize_str,
    normalize_number,
    require_keys,
)

from services.mlops.schemas.learning import validate_learning_block, LearningSpec
from services.mlops.schemas.model import validate_model_block, ModelSpec
from services.mlops.schemas.trainer import validate_trainer_block, TrainerSpec
from services.mlops.schemas.pruning import validate_pruning_block, PruningSpec
from services.mlops.schemas.benchmarking import validate_benchmarking_block, BenchmarkingSpec


# ============================================================
# Constants
# ============================================================

_ALLOWED_ALLOCATION_POLICIES = ("uniform", "profile_weighted", "round_robin", "adaptive_bandit")


# ============================================================
# Dataclasses
# ============================================================

@dataclass(frozen=True)
class AllocationSpec:
    policy: str = "uniform"
    min_trials_per_hypothesis: int = 1
    exploration_fraction: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromotionSpec:
    top_k: int = 1
    require_qualify: bool = True
    require_robustness_pass: bool = False
    export_only_promoted: bool = True
    export_include: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HypothesisSpec:
    hypothesis_id: str

    learning: Optional[LearningSpec] = None
    model: Optional[ModelSpec] = None
    trainer: Optional[TrainerSpec] = None
    pruning: Optional[PruningSpec] = None
    benchmarking: Optional[BenchmarkingSpec] = None

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectEntrySpec:
    name: str
    root: str

    overrides: Dict[str, Any] = field(default_factory=dict)
    allocation: AllocationSpec = field(default_factory=AllocationSpec)
    hypotheses: List[HypothesisSpec] = field(default_factory=list)
    promotion: Optional[PromotionSpec] = None

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectSpec:
    """
    Wrapper for the top-level `project:` list.
    """
    projects: List[ProjectEntrySpec] = field(default_factory=list)


# ============================================================
# Validators
# ============================================================

def _validate_allocation_block(v: Any, *, ctx: str) -> AllocationSpec:
    d = normalize_dict(v)

    policy = normalize_str(d.get("policy"), default="uniform").strip()
    if policy not in _ALLOWED_ALLOCATION_POLICIES:
        raise ValueError(f"[schemas/project] {ctx}.policy '{policy}' invalid. Allowed: {list(_ALLOWED_ALLOCATION_POLICIES)}")

    m = d.get("min_trials_per_hypothesis", 1)
    if isinstance(m, bool) or not isinstance(m, int) or m <= 0:
        raise ValueError(f"[schemas/project] {ctx}.min_trials_per_hypothesis must be positive int")

    ef = d.get("exploration_fraction", 0.0)
    ef = float(normalize_number(ef, default=0.0))
    if ef < 0.0 or ef > 1.0:
        raise ValueError(f"[schemas/project] {ctx}.exploration_fraction must be in [0, 1]")

    extra = {k: val for k, val in d.items() if k not in {"policy", "min_trials_per_hypothesis", "exploration_fraction"}}
    return AllocationSpec(policy=policy, min_trials_per_hypothesis=int(m), exploration_fraction=float(ef), extra=extra)


def _validate_promotion_block(v: Any, *, ctx: str) -> PromotionSpec:
    d = normalize_dict(v)

    top_k = d.get("top_k", 1)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"[schemas/project] {ctx}.top_k must be positive int")

    require_qualify = normalize_bool(d.get("require_qualify", True), default=True)
    require_robustness_pass = normalize_bool(d.get("require_robustness_pass", False), default=False)
    export_only_promoted = normalize_bool(d.get("export_only_promoted", True), default=True)

    export_include_raw = d.get("export_include", []) or []
    export_include = []
    for x in normalize_list(export_include_raw):
        s = normalize_str(x, default="").strip()
        if s:
            export_include.append(s)

    extra = {
        k: val
        for k, val in d.items()
        if k not in {"top_k", "require_qualify", "require_robustness_pass", "export_only_promoted", "export_include"}
    }

    return PromotionSpec(
        top_k=int(top_k),
        require_qualify=bool(require_qualify),
        require_robustness_pass=bool(require_robustness_pass),
        export_only_promoted=bool(export_only_promoted),
        export_include=export_include,
        extra=extra,
    )


def _validate_hypothesis_entry(v: Any, *, ctx: str) -> HypothesisSpec:
    d = normalize_dict(v)

    # We allow lots of keys because hypotheses are research-heavy.
    # We still validate known blocks strictly within their own validators.
    require_keys(d, ["hypothesis_id"], ctx=ctx)

    hid = normalize_str(d.get("hypothesis_id"), default="").strip()
    if not hid:
        raise ValueError(f"[schemas/project] {ctx}.hypothesis_id must be non-empty string")

    learning = d.get("learning", None)
    model = d.get("model", None)
    trainer = d.get("trainer", None)
    pruning = d.get("pruning", None)
    benchmarking = d.get("benchmarking", None)

    learning_spec = validate_learning_block(learning, ctx=f"{ctx}.learning") if learning is not None else None
    model_spec = validate_model_block(model, ctx=f"{ctx}.model") if model is not None else None
    trainer_spec = validate_trainer_block(trainer, ctx=f"{ctx}.trainer") if trainer is not None else None
    pruning_spec = validate_pruning_block(pruning, ctx=f"{ctx}.pruning") if pruning is not None else None
    benchmarking_spec = validate_benchmarking_block(benchmarking, ctx=f"{ctx}.benchmarking") if benchmarking is not None else None

    known = {"hypothesis_id", "learning", "model", "trainer", "pruning", "benchmarking"}
    extra = {k: val for k, val in d.items() if k not in known}

    return HypothesisSpec(
        hypothesis_id=hid,
        learning=learning_spec,
        model=model_spec,
        trainer=trainer_spec,
        pruning=pruning_spec,
        benchmarking=benchmarking_spec,
        extra=extra,
    )


def _validate_project_entry(v: Any, *, ctx: str) -> ProjectEntrySpec:
    d = normalize_dict(v)

    allowed = {"name", "root", "overrides", "allocation", "hypotheses", "promotion"}
    # allow future project keys but keep structure sane
    # (do NOT forbid unknown keys here; keep them as extra)
    name = normalize_str(d.get("name"), default="").strip()
    root = normalize_str(d.get("root"), default="").strip()
    if not name:
        raise ValueError(f"[schemas/project] {ctx}.name must be non-empty string")
    if not root:
        raise ValueError(f"[schemas/project] {ctx}.root must be non-empty string")

    overrides = normalize_dict(d.get("overrides", {}) or {})
    allocation = _validate_allocation_block(d.get("allocation", {}) or {}, ctx=f"{ctx}.allocation")

    hyps_raw = d.get("hypotheses", None)
    if hyps_raw is None:
        raise ValueError(f"[schemas/project] {ctx}.hypotheses missing (must be non-empty list)")
    hyps_list = normalize_list(hyps_raw)
    if not hyps_list:
        raise ValueError(f"[schemas/project] {ctx}.hypotheses must be non-empty list")

    hypotheses: List[HypothesisSpec] = []
    seen: set[str] = set()
    for i, h in enumerate(hyps_list):
        hs = _validate_hypothesis_entry(h, ctx=f"{ctx}.hypotheses[{i}]")
        if hs.hypothesis_id in seen:
            raise ValueError(f"[schemas/project] {ctx}.hypotheses duplicate hypothesis_id '{hs.hypothesis_id}'")
        seen.add(hs.hypothesis_id)
        hypotheses.append(hs)

    promotion_raw = d.get("promotion", None)
    promotion = _validate_promotion_block(promotion_raw, ctx=f"{ctx}.promotion") if promotion_raw is not None else None

    extra = {k: val for k, val in d.items() if k not in allowed}
    return ProjectEntrySpec(
        name=name,
        root=root,
        overrides=overrides,
        allocation=allocation,
        hypotheses=hypotheses,
        promotion=promotion,
        extra=extra,
    )


def validate_project_block(cfg: Any, *, ctx: str = "project") -> ProjectSpec:
    """
    Validates the top-level `project:` list and returns ProjectSpec.
    """
    projects_raw = normalize_list(cfg)
    if not projects_raw:
        raise ValueError(f"[schemas/project] {ctx} must be a non-empty list")

    projects: List[ProjectEntrySpec] = []
    seen_names: set[str] = set()
    for i, p in enumerate(projects_raw):
        ps = _validate_project_entry(p, ctx=f"{ctx}[{i}]")
        if ps.name in seen_names:
            raise ValueError(f"[schemas/project] {ctx} duplicate project name '{ps.name}'")
        seen_names.add(ps.name)
        projects.append(ps)

    return ProjectSpec(projects=projects)
