# services/mlops/core/context.py
# ============================================================
# Pure Context Compiler (NO side effects)
# - Loads + merges:
#     auto_mlops.yaml (already loaded as dict)
#     project.overrides
#     selected hypothesis blocks
# - Produces a deterministic "compiled view" used by every module
# - Pure dict in -> dict out (no file writes, no env mutation)
#
# Author: Shraddha Mishra
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast
from copy import deepcopy

# NOTE:
# We intentionally keep this module pure.
# It must not import anything that writes to disk, reads env files, etc.
#
# Validation is optional (pure: validate does not mutate).
try:
    from services.mlops.schemas.auto_mlops import validate_auto_mlops_yaml, AutoMLOpsSpec  # type: ignore
except Exception:  # pragma: no cover
    validate_auto_mlops_yaml = None  # type: ignore
    AutoMLOpsSpec = None  # type: ignore


Json = Dict[str, Any]


# ============================================================
# Exceptions
# ============================================================

class ContextError(ValueError):
    pass


class ProjectNotFound(ContextError):
    pass


class HypothesisNotFound(ContextError):
    pass


# ============================================================
# Deterministic deep merge
# ============================================================

def _is_mapping(x: Any) -> bool:
    return isinstance(x, dict)


def deep_merge(
    base: Any,
    override: Any,
    *,
    list_policy: str = "replace",   # "replace" | "concat_unique"
) -> Any:
    """
    Deterministic deep merge:
      - dict + dict: merge keys recursively
      - list + list:
          * replace (default): override replaces base
          * concat_unique: base + items from override not already present (by equality)
      - otherwise: override wins (including None)
    """
    if _is_mapping(base) and _is_mapping(override):
        out: Dict[str, Any] = {}
        # Deterministic ordering: first base keys in sorted order, then new override keys sorted.
        base_keys = sorted(base.keys())
        override_keys = sorted(override.keys())
        seen = set()

        for k in base_keys:
            seen.add(k)
            if k in override:
                out[k] = deep_merge(base[k], override[k], list_policy=list_policy)
            else:
                out[k] = deepcopy(base[k])

        for k in override_keys:
            if k in seen:
                continue
            out[k] = deepcopy(override[k])

        return out

    if isinstance(base, list) and isinstance(override, list):
        if list_policy == "concat_unique":
            out_list = list(deepcopy(base))
            for item in override:
                if item not in out_list:
                    out_list.append(deepcopy(item))
            return out_list
        # default: replace
        return deepcopy(override)

    # Scalars / type mismatch: override wins
    return deepcopy(override)


# ============================================================
# Selection helpers
# ============================================================

def _normalize_str(x: Any, *, ctx: str) -> str:
    if not isinstance(x, str) or not x.strip():
        raise ContextError(f"[context] {ctx} must be non-empty string")
    return x.strip()


def _ensure_list_of_dicts(x: Any, *, ctx: str) -> List[Dict[str, Any]]:
    if x is None:
        return []
    if not isinstance(x, list):
        raise ContextError(f"[context] {ctx} must be a list")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(x):
        if not isinstance(item, dict):
            raise ContextError(f"[context] {ctx}[{i}] must be a mapping")
        out.append(item)
    return out


def find_project_block(auto_mlops_cfg: Json, *, project_name: str) -> Dict[str, Any]:
    """
    auto_mlops.yaml project block is a LIST in your current YAML.
    Example:
      project:
        - name: DLSCA_ASCAD
          root: "DLSCA_ASCAD/"
          overrides: {...}
          hypotheses: [...]
    """
    projects = _ensure_list_of_dicts(auto_mlops_cfg.get("project"), ctx="project")
    for p in projects:
        if str(p.get("name", "")).strip() == project_name:
            return p
    raise ProjectNotFound(f"[context] project '{project_name}' not found in auto_mlops.project[]")


def find_hypothesis_block(project_block: Dict[str, Any], *, hypothesis_id: str) -> Dict[str, Any]:
    hyps = _ensure_list_of_dicts(project_block.get("hypotheses"), ctx=f"project[{project_block.get('name')}].hypotheses")
    for h in hyps:
        if str(h.get("hypothesis_id", "")).strip() == hypothesis_id:
            return h
    raise HypothesisNotFound(
        f"[context] hypothesis '{hypothesis_id}' not found under project '{project_block.get('name')}'"
    )


# ============================================================
# Compiled view
# ============================================================

@dataclass(frozen=True)
class CompiledContext:
    """
    Canonical compiled view (pure).
    Everything downstream consumes this.
    """
    project_name: str
    project_root: str
    hypothesis_id: str

    # Deterministic merged config
    compiled: Json

    # Convenience derived blocks
    paths: Dict[str, Any]
    search: Dict[str, Any]
    visualization: Dict[str, Any]
    registry: Dict[str, Any]

    objectives: Dict[str, Any]
    constraints: Dict[str, Any]
    retention: Dict[str, Any]

    learning: Dict[str, Any]
    model: Dict[str, Any]
    trainer: Dict[str, Any]
    pruning: Dict[str, Any]
    benchmarking: Dict[str, Any]

    # Optional "core config" (e.g., params.yaml) snapshot pass-through
    core_cfg: Optional[Json] = None

    # Render context used by path resolver (NO side effects)
    render_ctx: Dict[str, str] = None  # type: ignore[assignment]


def _build_render_ctx(
    *,
    core_cfg: Optional[Json],
    project_name: str,
    project_root: str,
    hypothesis_id: str,
    compiled: Json,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Build the context used for path rendering ({{...}} / ${...})
    without touching env vars.

    Common placeholders supported:
      - env, model_name, dataset_alias, sota_trial, hypothesis_id, project_name
    """
    ctx: Dict[str, str] = {}

    # core_cfg (params.yaml) is authoritative for env/model_name if present.
    if core_cfg:
        if core_cfg.get("env") is not None:
            ctx["env"] = str(core_cfg["env"])
        # In MLSCA you typically use model_name in params.yaml
        if core_cfg.get("model_name") is not None:
            ctx["model_name"] = str(core_cfg["model_name"])

        # dataset_alias might live in params.yaml under data/dataset
        # We keep it best-effort (pure).
        ds_alias = (
            core_cfg.get("dataset_alias")
            or (core_cfg.get("dataset") or {}).get("alias")
            or (core_cfg.get("data") or {}).get("dataset_alias")
        )
        if ds_alias is not None:
            ctx["dataset_alias"] = str(ds_alias)

    # AutoMLOps may have fields that matter for rendering
    # e.g. sota_trial is often runtime-provided; default to "trial0000" style caller supplies.
    if "sota_trial" in compiled:
        ctx["sota_trial"] = str(compiled["sota_trial"])

    ctx["hypothesis_id"] = hypothesis_id
    ctx["project_name"] = project_name
    ctx["project_root"] = project_root

    # Reasonable fallback defaults (only if still missing)
    ctx.setdefault("env", "dev")
    ctx.setdefault("model_name", "unnamed_model")

    # Allow caller to override/add placeholders deterministically
    if extra:
        for k, v in extra.items():
            if v is None:
                continue
            ctx[str(k)] = str(v)

    return ctx


# ============================================================
# Compiler
# ============================================================

_GLOBAL_TOP_BLOCKS = (
    "version",
    "mode",
    "paths",
    "visualization",
    "search",
    "data_profile",
    "objectives",
    "constraints",
    "retention",
    "registry",
    # NOTE: "project" is handled separately (list)
)

# Hypothesis-level blocks that can override global defaults.
# These are the blocks we "lift" from a hypothesis and merge into the compiled view.
_HYP_BLOCKS = (
    "learning",
    "model",
    "trainer",
    "pruning",
    "benchmarking",
)

# Project-level blocks used by core logic
_PROJECT_BLOCKS = (
    "allocation",
    "promotion",
)

def compile_context(
    auto_mlops_cfg: Json,
    *,
    project_name: str,
    hypothesis_id: str,
    core_cfg: Optional[Json] = None,
    # runtime knobs
    sota_trial: Optional[str] = None,
    extra_render_ctx: Optional[Dict[str, Any]] = None,
    # validation
    validate_schema: bool = True,
    list_merge_policy: str = "replace",
) -> CompiledContext:
    """
    Pure compiler:
      - Selects project + hypothesis
      - Applies project overrides over global defaults
      - Applies hypothesis blocks over merged globals
      - Returns a deterministic compiled view
    """
    if not isinstance(auto_mlops_cfg, dict):
        raise ContextError("[context] auto_mlops_cfg must be a dict (already YAML-loaded)")

    # Optional schema validation (pure)
    if validate_schema and validate_auto_mlops_yaml is not None:
        # This returns a dataclass spec; we only use it as a validator.
        validate_auto_mlops_yaml(auto_mlops_cfg)

    project_name = _normalize_str(project_name, ctx="project_name")
    hypothesis_id = _normalize_str(hypothesis_id, ctx="hypothesis_id")

    project_block = find_project_block(auto_mlops_cfg, project_name=project_name)
    hyp_block = find_hypothesis_block(project_block, hypothesis_id=hypothesis_id)

    project_root = str(project_block.get("root", "")).strip()
    if not project_root:
        raise ContextError(f"[context] project '{project_name}' missing/empty 'root'")

    # 1) Start with deterministic global skeleton (only known top blocks)
    compiled: Json = {}
    for k in _GLOBAL_TOP_BLOCKS:
        if k in auto_mlops_cfg:
            compiled[k] = deepcopy(auto_mlops_cfg[k])

    # Add runtime-provided knobs into compiled view (pure)
    if sota_trial is not None:
        compiled["sota_trial"] = _normalize_str(sota_trial, ctx="sota_trial")

    # 2) Apply project overrides on top of global compiled
    overrides = project_block.get("overrides") or {}
    if overrides:
        if not isinstance(overrides, dict):
            raise ContextError(f"[context] project '{project_name}'.overrides must be a mapping")
        compiled = cast(Json, deep_merge(compiled, overrides, list_policy=list_merge_policy))

    # 3) Merge hypothesis blocks (learning/model/trainer/pruning/benchmarking)
    for blk in _HYP_BLOCKS:
        if blk in hyp_block:
            if not isinstance(hyp_block[blk], dict):
                raise ContextError(f"[context] hypothesis.{blk} must be a mapping")
            compiled[blk] = cast(Json, deep_merge(compiled.get(blk, {}), hyp_block[blk], list_policy=list_merge_policy))

    # 4) Lift project-level policies (allocation/promotion) into compiled for convenience
    for blk in _PROJECT_BLOCKS:
        if blk in project_block:
            compiled[blk] = deepcopy(project_block[blk])

    # 5) Stamp identity metadata into compiled (kept explicit and deterministic)
    compiled["identity"] = {
        "project_name": project_name,
        "project_root": project_root,
        "hypothesis_id": hypothesis_id,
    }

    # 6) Prepare convenience sub-blocks (best-effort; never KeyError)
    def _get_block(name: str) -> Dict[str, Any]:
        v = compiled.get(name, {}) or {}
        if not isinstance(v, dict):
            raise ContextError(f"[context] compiled.{name} must be a mapping")
        return cast(Dict[str, Any], v)

    paths = _get_block("paths")
    search = _get_block("search")
    visualization = _get_block("visualization")
    registry = _get_block("registry")

    objectives = _get_block("objectives")
    constraints = _get_block("constraints")
    retention = _get_block("retention")

    learning = _get_block("learning")
    model = _get_block("model")
    trainer = _get_block("trainer")
    pruning = _get_block("pruning")
    benchmarking = _get_block("benchmarking")

    render_ctx = _build_render_ctx(
        core_cfg=core_cfg,
        project_name=project_name,
        project_root=project_root,
        hypothesis_id=hypothesis_id,
        compiled=compiled,
        extra=extra_render_ctx,
    )

    return CompiledContext(
        project_name=project_name,
        project_root=project_root,
        hypothesis_id=hypothesis_id,
        compiled=compiled,
        paths=paths,
        search=search,
        visualization=visualization,
        registry=registry,
        objectives=objectives,
        constraints=constraints,
        retention=retention,
        learning=learning,
        model=model,
        trainer=trainer,
        pruning=pruning,
        benchmarking=benchmarking,
        core_cfg=deepcopy(core_cfg) if core_cfg else None,
        render_ctx=render_ctx,
    )


# ============================================================
# Convenience: compile from an AutoConfig instance (still pure)
# ============================================================

def compile_context_from_autoconfig(
    cfg: Any,
    *,
    project_name: str,
    hypothesis_id: str,
    core_cfg: Optional[Json] = None,
    sota_trial: Optional[str] = None,
    extra_render_ctx: Optional[Dict[str, Any]] = None,
    validate_schema: bool = True,
    list_merge_policy: str = "replace",
) -> CompiledContext:
    """
    Helper for your integration layer:
      from services.mlops.utils.auto_config import AutoConfig
      ac = AutoConfig(...)
      auto_cfg = ac.get_map("auto_mlops")  # dict
    """
    # We do not import AutoConfig at module import time to keep this core file pure.
    if not hasattr(cfg, "get_map"):
        raise ContextError("[context] cfg must provide get_map() (AutoConfig-like)")

    auto_mlops_cfg = cfg.get_map("auto_mlops")  # type: ignore[attr-defined]
    if not isinstance(auto_mlops_cfg, dict):
        raise ContextError("[context] cfg.get_map('auto_mlops') must return a dict")

    return compile_context(
        auto_mlops_cfg,
        project_name=project_name,
        hypothesis_id=hypothesis_id,
        core_cfg=core_cfg,
        sota_trial=sota_trial,
        extra_render_ctx=extra_render_ctx,
        validate_schema=validate_schema,
        list_merge_policy=list_merge_policy,
    )
