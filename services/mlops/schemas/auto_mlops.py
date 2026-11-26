# services/mlops/schemas/auto_mlops.py
# ============================================================
# TOP-LEVEL schema for services/mlops/config/auto_mlops.yaml
# - Validates auto_mlops.yaml (v1)
# - Delegates project/hypotheses validation to schemas/project.py
#
# IMPORTANT DESIGN RULES:
# - Schemas are PURE validation: no MLOpsConfig, no path resolving, no mkdir/touch.
# - Project-specific hypothesis blocks (learning/model/trainer/pruning/benchmarking)
#   are validated inside schemas/project.py
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from services.mlops.schemas.common import (
    forbid_unknown_keys,
    normalize_dict,
    normalize_list,
    normalize_str,
    require_keys,
)
from services.mlops.schemas.project import validate_project_block, ProjectSpec
from services.mlops.schemas.objectives import validate_objectives_block, ObjectivesSpec
from services.mlops.schemas.constraints import validate_constraints_block, ConstraintsSpec
from services.mlops.schemas.retention import validate_retention_block, RetentionSpec
from services.mlops.schemas.search import validate_search_block, SearchSpec
from services.mlops.schemas.visualization import validate_visualization_block, VisualizationSpec
from services.mlops.schemas.registry import validate_registry_block, RegistrySpec


# ============================================================
# Constants
# ============================================================

ALLOWED_TOP_MODES = ("auto_sota", "evaluate_only", "replay")

# Keys expected at the TOP level of auto_mlops.yaml.
# NOTE: hypothesis-specific blocks live under project[].hypotheses[] and are validated there.
TOP_LEVEL_KEYS = {
    "version",
    "mode",
    "paths",
    "visualization",
    "search",
    "data_profile",   # optional (present in your YAML)
    "objectives",
    "constraints",
    "retention",
    "registry",
    "project",        # list[project entries]
    "notes",
}

# Required plugin-owned path keys (minimal hard contract for v1 runner + logger).
# You can expand this later; extra string keys are allowed and preserved.
_PATH_KEYS_REQUIRED = (
    "auto_params",
    "decisions_log_file",
    "pruning_log_file",
    "exceptions_log_file",
    "events_jsonl_file",
    "auto_run_map_benchmark",
    "leaderboard_benchmark",
    "pareto_front_benchmark",
    "auto_trial_registry",
    "auto_hypothesis_registry",
    "auto_sota_registry",
)


# ============================================================
# Dataclass
# ============================================================

@dataclass(frozen=True)
class AutoMLOpsSpec:
    version: str
    mode: str

    # Top-level blocks
    paths: Dict[str, str]
    objectives: ObjectivesSpec
    constraints: ConstraintsSpec
    retention: RetentionSpec
    registry: RegistrySpec
    visualization: VisualizationSpec
    search: SearchSpec

    # Project list (each project contains hypotheses etc.)
    project: ProjectSpec

    # Optional blocks
    data_profile: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    # Raw dict retained for pass-through / forward compat
    raw: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Helpers
# ============================================================

def _validate_paths_block(paths_any: Any, *, ctx: str = "paths") -> Dict[str, str]:
    d = normalize_dict(paths_any)
    require_keys(d, list(_PATH_KEYS_REQUIRED), ctx=ctx)

    out: Dict[str, str] = {}
    for k in _PATH_KEYS_REQUIRED:
        v = normalize_str(d.get(k), default="").strip()
        if not v:
            raise ValueError(f"[schemas/auto_mlops] {ctx}.{k} must be a non-empty string")
        out[k] = v

    # Allow extra future path keys as long as they're non-empty strings
    for k, v in d.items():
        if k in out:
            continue
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()

    return out


def _validate_data_profile_block(v: Any, *, ctx: str = "data_profile") -> Dict[str, Any]:
    # Keep it permissive for v1: it guides allocation but should not block runs.
    # Still ensure it is a dict if present.
    if v is None:
        return {}
    return normalize_dict(v)


# ============================================================
# Public validator
# ============================================================

def validate_auto_mlops_yaml(auto_mlops_cfg: Dict[str, Any]) -> AutoMLOpsSpec:
    """
    Validate full auto_mlops.yaml (TOP-LEVEL contract).

    Expected skeleton (v1):

      version: 1.0
      mode: auto_sota | evaluate_only | replay
      paths: {...}                   # required keys in _PATH_KEYS_REQUIRED
      visualization: {...}
      search: {...}
      data_profile: {...}            # optional
      objectives: {...}
      constraints: {...}
      retention: {...}
      registry: {...}
      project:
        - name: ...
          root: ...
          overrides: ...
          allocation: ...
          hypotheses: [...]
          promotion: ...

    Notes:
      - learning/model/trainer/pruning/benchmarking blocks are validated inside
        project[].hypotheses[] entries by schemas/project.py (not top-level).
    """
    cfg = normalize_dict(auto_mlops_cfg)

    # Strict top-level hygiene (expand allowlist as you add blocks)
    forbid_unknown_keys(cfg,  allowed=TOP_LEVEL_KEYS, where="auto_mlops")

    # Required top-level keys
    require_keys(
        cfg,
        ["version", "mode", "paths", "visualization", "search", "objectives", "constraints", "retention", "registry", "project"],
        ctx="auto_mlops",
    )

    version = normalize_str(cfg.get("version"), default="").strip()
    if not version:
        raise ValueError("[schemas/auto_mlops] missing/empty top-level 'version'")

    mode = normalize_str(cfg.get("mode"), default="").strip()
    if mode not in ALLOWED_TOP_MODES:
        raise ValueError(f"[schemas/auto_mlops] invalid mode '{mode}'. Allowed: {list(ALLOWED_TOP_MODES)}")

    # ---- Paths (plugin-owned) ----
    paths = _validate_paths_block(cfg.get("paths", {}), ctx="paths")

    # ---- Objectives / Constraints / Retention / Registry ----
    objectives = validate_objectives_block(cfg.get("objectives", {}), ctx="objectives")
    constraints = validate_constraints_block(cfg.get("constraints", {}), ctx="constraints")
    retention = validate_retention_block(cfg.get("retention", {}), ctx="retention")
    registry = validate_registry_block(cfg.get("registry", {}), ctx="registry")

    # ---- Visualization / Search ----
    visualization = validate_visualization_block(cfg.get("visualization", {}), ctx="visualization")
    search = validate_search_block(cfg.get("search", {}), ctx="search")

    # ---- Data profile (optional, permissive) ----
    data_profile = _validate_data_profile_block(cfg.get("data_profile", None), ctx="data_profile")

    # ---- Project list ----
    project_list = normalize_list(cfg.get("project", []))
    project = validate_project_block(project_list, ctx="project")

    notes = cfg.get("notes")
    if notes is not None:
        notes = normalize_str(notes, default="").strip() or None

    return AutoMLOpsSpec(
        version=version,
        mode=mode,
        paths=paths,
        objectives=objectives,
        constraints=constraints,
        retention=retention,
        registry=registry,
        visualization=visualization,
        search=search,
        project=project,
        data_profile=data_profile,
        notes=notes,
        raw=cfg,
    )
