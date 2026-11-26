# services/side_channel_core/utils/config_loader.py
"""
config_loader.py

Centralized loader for unified configuration in the Project MLSCA system.

This loader composes the final runtime configuration (`cfg`) from:
- Global settings:              services/side_channel_core/config/master.yaml
- Project settings:             <Project_X>/config/params.yaml
- Optional env overrides:       <Project_X>/config/env.<env>.yaml (via master.yaml mapping)
- Optional extra configs:       per-project extras declared in master.yaml (graph/test/etc.)
- Distributed settings:         master.yaml + optional per-project distributed_override
- Script path resolution:       resolves script paths to absolute filesystem locations
                                (runtime only: model_hash and the params_out snapshot
                                store them relative to the project/workspace root)

AutoMLOps integration (plugin-owned, conditional):
- master.yaml can declare:
    auto_mlops:
        enabled: true|false
        config: "services/mlops/config/auto_mlops.yaml"
        hard_replace_params: true|false

Policy:
1) auto_mlops.enabled == False
   - DO NOT read/parse/validate auto_mlops.yaml at all.
   - merged["auto_mlops"] remains metadata only (enabled/config/hard_replace_params/config_abs).

2) auto_mlops.enabled == True
   - Load + validate auto_mlops.yaml via services.mlops.schemas.auto_mlops.validate_auto_mlops_yaml
   - Store plugin config in merged["auto_mlops"]["spec"] (validated summary) and optionally raw yaml
   - Apply overrides ONLY for top-level keys in OVERRIDE_KEYS (default: ("model","trainer")):

   2a) hard_replace_params == True  (STRICT)
       - merged["model"]   = auto_mlops_yaml["model"]   (if present)
       - merged["trainer"] = auto_mlops_yaml["trainer"] (if present)
       => params.yaml sections for these keys are ignored.

   2b) hard_replace_params == False (LENIENT)
       - deep-merge auto_mlops sections on top of params.yaml
       => params.yaml supplies missing values.

Usage:
    from utils.config_loader import get_project_config
    cfg = get_project_config("DLSCA_ASCAD")

Author: Shraddha Mishra
"""

from __future__ import annotations

import ast
import difflib
import os
from pathlib import Path
from typing import Dict, Tuple, List
import copy
import yaml
from services.side_channel_core.utils.logger import get_logger
from services.side_channel_core.utils.path_utils import _get_workspace_root, to_portable_path
from services.side_channel_core.src.sca.dataset_types import registered, validate_entry
from services.side_channel_core.utils.config_errors import ConfigError
from services.side_channel_core.utils import config_schema
from services.side_channel_core.utils.config_freeze import config_path, frozen_dir

logger = get_logger("Config Loader")


# ---------- Path autodetection (container & host safe) ----------

def _detect_scc_root() -> Path:
    env_root = os.environ.get("SCC_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if (p / "config").exists():
            return p

    p = Path("/app/services/side_channel_core")
    if (p / "config").exists():
        return p

    p = Path(__file__).resolve()
    for _ in range(10):
        if (p / "config").exists() and (p / "src").exists():
            return p
        p = p.parent

    return Path("/app/services/side_channel_core")


def _detect_projects_root(scc_root: Path) -> Path:
    env_proj = os.environ.get("PROJECTS_ROOT")
    if env_proj:
        return Path(env_proj).resolve()

    # Prefer explicit /app/projects if present (common container layout)
    p = Path("/app/projects")
    if p.exists():
        return p.resolve()

    # Otherwise walk upwards looking for a "projects" dir
    p = scc_root
    for _ in range(10):
        cand = p / "projects"
        if cand.exists():
            return cand.resolve()
        p = p.parent

    # Fallback: /app
    return Path("/app").resolve()

def _render_like_resolve_path(template: str, cfg: dict) -> str:
    """
    Lightweight render similar to utils.path_utils.resolve_path() but without importing jinja2 here.
    Only handles {{env}}, {{project}}, etc if present in cfg via a minimal replace.
    (Database paths usually don't need templating; this is a safe helper.)
    """
    if not isinstance(template, str):
        return str(template)

    rendered = template
    # minimal common placeholders (extend if you ever need more)
    if "{{env}}" in rendered and isinstance(cfg.get("env"), str):
        rendered = rendered.replace("{{env}}", cfg["env"])
    return rendered


def _candidate_dataset_paths(
    raw_path: str,
    *,
    project_root: Path,
    projects_root: Path,
    scc_root: Path,
    workspace_root: Path,
) -> List[Path]:
    """
    Deterministic candidate list for resolving dataset database paths.

    Accepts:
      - absolute paths: /app/... or /home/...
      - repo-relative: services/side_channel_core/data/...
      - scc-data-relative: data/ASCAD/...
      - legacy relative: ../services/side_channel_core/data/...

    Relative paths are tried against the detected roots only (host checkout
    or container /app alike), never a hard-coded mount point.
    """
    p = Path(raw_path)

    # absolute is a single candidate
    if p.is_absolute():
        return [p]

    # relative candidates (ordered)
    return [
        project_root / p,                     # relative to project (rare, but valid)
        projects_root / p,                    # relative to projects root
        workspace_root / p,                   # repo root (host checkout or /app)
        scc_root / p,                         # relative to SCC root
        (scc_root / "data") / p,              # allow "data/..." shorthand
    ]


def _resolve_dataset_database_path(
    raw_path: str,
    *,
    cfg: dict,
    project_name: str,
    project_root: Path,
    projects_root: Path,
    scc_root: Path,
    ds_name: str,
) -> str:
    """
    Resolve cfg.data[ds].database into an absolute path that exists.

    Raises a rich FileNotFoundError listing all candidates tried.
    """
    if not raw_path or not isinstance(raw_path, str):
        raise ValueError(f"Dataset '{ds_name}' has invalid database path: {raw_path!r}")

    rendered = _render_like_resolve_path(raw_path, cfg).strip()
    workspace_root = _get_workspace_root()

    tried = []
    for cand in _candidate_dataset_paths(
        rendered,
        project_root=project_root,
        projects_root=projects_root,
        scc_root=scc_root,
        workspace_root=workspace_root,
    ):
        tried.append(str(cand))
        if cand.exists():
            resolved = str(cand.resolve())
            # model_hash and the params_out snapshot store it relative when a
            # relative form maps back to this file (path_utils.to_portable_path).
            if Path(to_portable_path(resolved, project_root=project_root)).is_absolute():
                logger.warning(
                    "Dataset '%s' database %s has no location-independent form: model_hash "
                    "and the params_out snapshot keep this absolute path, so the same config "
                    "hashes differently in another checkout or in the container (/app). "
                    "Place the file under services/side_channel_core/data/ (or the project "
                    "folder) to share runs across checkouts.",
                    ds_name, resolved,
                )
            return resolved

    # A registered database (src/sca/dataset_types.py) this checkout does not hold yet: its place,
    # where the run that selects it fetches it (sca/ascad_data_loader.check_file_exists ->
    # utils/dataset_fetch.py). Loading the configuration downloads nothing.
    if registered(rendered):
        relative = rendered.replace("\\", "/").split("side_channel_core/data/", 1)[-1]
        target = (scc_root / "data" / relative).resolve()
        logger.info("Dataset '%s': %s is not here yet; a run that selects it fetches it", ds_name, target)
        return str(target)

    # rich error (deterministic + debuggable)
    msg = (
        f"Dataset file not found for project='{project_name}' dataset='{ds_name}'.\n"
        f"  raw:      {raw_path}\n"
        f"  rendered: {rendered}\n"
        f"  project_root:  {project_root}\n"
        f"  projects_root: {projects_root}\n"
        f"  scc_root:      {scc_root}\n"
        f"  workspace_root: {workspace_root}\n"
        f"  candidates tried:\n    - " + "\n    - ".join(tried) + "\n"
        "  hint: give data.database relative to the repo root "
        "(services/side_channel_core/data/...) or set WORKSPACE_ROOT to the checkout."
    )
    raise FileNotFoundError(msg)


SCC_ROOT: Path = _detect_scc_root()
SCC_CONFIG_DIR: Path = Path(os.environ.get("SCC_CONFIG_DIR", str(SCC_ROOT / "config"))).resolve()
PROJECTS_ROOT: Path = _detect_projects_root(SCC_ROOT)


# ---------- Small helpers ----------

def _maybe_eval(v):
    if isinstance(v, str) and any(op in v for op in "+-*/"):
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    return v


def _walk_and_eval(cfg):
    if isinstance(cfg, dict):
        return {k: _walk_and_eval(_maybe_eval(v)) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_walk_and_eval(_maybe_eval(v)) for v in cfg]
    return _maybe_eval(cfg)


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict, override: dict) -> dict:
    """
    Recursive dict merge (override wins).
    """
    base = base or {}
    override = override or {}
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base

def _is_empty(value) -> bool:
    """No value given: null, an empty string, list or mapping (0 and false are values)."""
    return value is None or (isinstance(value, (str, list, tuple, dict)) and len(value) == 0)


def _shape(value) -> str:
    return "mapping" if isinstance(value, dict) else "list" if isinstance(value, (list, tuple)) else "value"


def drop_empty_keys(node):
    """`node` without its empty (None) mapping values, at every depth (list items kept: a null item of a
    sweep list is a value its reader refuses). An empty key is the code's default (6b in get_project_config).

    A block whose keys are all empty is absent too (owner 2026-10-23): the blueprint's `args: {dim: }` is no
    `args`, so a run planned from it keeps the hash of a run made before the block was written. A block written
    empty, `{}`, is a value and stays: runs were hashed with it."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if value is None:
                continue
            kept = drop_empty_keys(value)
            if isinstance(value, dict) and value and not kept:
                continue                                 # every key of the block was empty
            out[key] = kept
        return out
    if isinstance(node, list):
        return [drop_empty_keys(v) for v in node]
    return node


def explicit_null_seed(path) -> bool:
    """Whether the file writes `seed: null` (or `~`): an unseeded run (utils/seed.run_seed: a seed drawn from the
    operating system per run). An empty `seed:` is no setting, as every empty key (6b); only a written null keeps
    the key, as null, through drop_empty_keys."""
    import yaml as _yaml

    try:
        node = _yaml.compose(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, _yaml.YAMLError):
        return False
    if not isinstance(node, _yaml.MappingNode):
        return False
    for key, value in node.value:
        if isinstance(key, _yaml.ScalarNode) and key.value == "seed":
            return (isinstance(value, _yaml.ScalarNode) and value.tag == "tag:yaml.org,2002:null"
                    and value.value in ("null", "Null", "NULL", "~"))
    return False


def canonical_form(cfg):
    """The config as the readers and the run hash take it: empty keys dropped (drop_empty_keys), and the
    written forms of params.yaml folded into the one form the code reads (canonical_trainer). Keeps every
    hash made before a written form existed: the canonical form is the form those runs were hashed in."""
    out = drop_empty_keys(cfg)
    if isinstance(out, dict) and isinstance(out.get("trainer"), dict):
        out["trainer"] = canonical_trainer(out["trainer"])
    if isinstance(out, dict) and isinstance(out.get("deep_learning"), dict):
        if isinstance(out["deep_learning"].get("framework"), list):   # owner 2026-10-23: one framework per file
            raise ConfigError("params.yaml: deep_learning.framework takes one framework (pytorch | keras), not a list: "
                              "a project runs under one framework at a time (the other framework's run is another "
                              "pass of the file with framework changed)")
        out = canonical_grad_clip(out)
    return out


def check_seed_spacing(cfg: dict) -> None:
    """Repeats of a cross-validated run (`seed: [42, 53, 64]`) must not share a training seed: a run trains the
    refit with its seed and the folds with the next n_folds seeds (utils/cross_validation), so two seeds closer than
    n_folds + 1 give two runs some identical fold trainings. Refused by name with the spacing that works (owner
    2026-10-23; the panel says the same beside the seed list as it is typed)."""
    seeds = cfg.get("seed") if isinstance(cfg, dict) else None
    cv = cfg.get("cross_validation") if isinstance(cfg, dict) else None
    if not isinstance(seeds, list) or len(seeds) < 2 or not isinstance(cv, dict) or cv.get("enabled") is not True:
        return
    folds = cv.get("n_folds")
    folds = max(int(f) for f in folds) if isinstance(folds, list) and folds else int(folds or 0)
    gap = folds + 1
    values = sorted(int(s) for s in seeds if isinstance(s, (int, float)) and not isinstance(s, bool))
    for a, b in zip(values, values[1:]):
        if b - a < gap:
            raise ConfigError(f"params.yaml: seed {values}: seeds {a} and {b} are {b - a} apart, but a run trains with "
                              f"its seed and the next {folds} (cross_validation.n_folds {folds}: the folds), so two seeds "
                              f"closer than {gap} share fold trainings. Space the repeats by at least {gap}, e.g. "
                              f"{', '.join(str(values[0] + k * gap) for k in range(len(values)))}")


def canonical_grad_clip(cfg: dict) -> dict:
    """params.yaml writes gradient clipping under the framework it belongs to (2026-10-22):
    deep_learning.pytorch.grad_clip_norm (torch.nn.utils.clip_grad_norm_ every optimizer step) and
    deep_learning.keras.grad_clip_norm (the optimizer's global_clipnorm). The code reads trainer.grad_clip_norm
    (core/framework_switch.gradient_clipper, build_optimizer): the chosen framework's value is folded there,
    and the keys leave the blocks. trainer.grad_clip_norm written flat still works (a snapshot, an older file)."""
    dl = cfg["deep_learning"]
    framework = dl.get("framework")
    framework = framework[0] if isinstance(framework, list) and framework else framework
    chosen = str(framework or "pytorch")
    blocks = {k: v for k, v in dl.items() if isinstance(v, dict) and "grad_clip_norm" in v}
    if not blocks:
        return cfg
    out = dict(cfg)
    out["deep_learning"] = {k: ({kk: vv for kk, vv in v.items() if kk != "grad_clip_norm"} if k in blocks else v) for k, v in dl.items()}
    if chosen in blocks:
        out["trainer"] = {**(out.get("trainer") or {}), "grad_clip_norm": blocks[chosen]["grad_clip_norm"]}
    return out


def canonical_trainer(trainer: dict) -> dict:
    """trainer as the code reads it. params.yaml writes early stopping as one block (2026-10-22):

        early_stopping:
          enabled: false
          monitor_metric: guessing_entropy    # the metric early stopping and validation checkpointing watch
          patience: 7                         # read with enabled: true
          min_delta: 0.0

    The code (utils/epoch_utils, utils/monitor_metric, utils/checkpoint_policy, utils/cross_validation) reads
    the flat keys the runs were hashed with: early_stopping (true / false), monitor_metric, patience,
    min_delta. The block is folded into them here: enabled -> early_stopping; monitor_metric always;
    patience and min_delta only with enabled: true, since they are read only then (a run that never
    stops early hashes alike with or without them). The flat keys still work (a snapshot, an older file).
    trainer.monitor_mode is retired: the direction follows the metric (utils/monitor_metric.get_monitor_mode)."""
    out = dict(trainer)
    out.pop("monitor_mode", None)
    block = out.get("early_stopping")
    if isinstance(block, dict):
        enabled = block.get("enabled", False)
        out["early_stopping"] = enabled
        if block.get("monitor_metric") is not None:
            out["monitor_metric"] = block["monitor_metric"]
        on = enabled is True or (isinstance(enabled, list) and True in enabled)   # a swept enabled keeps them too
        for key in ("patience", "min_delta"):
            if on and block.get(key) is not None:
                out[key] = block[key]
        for key in block:
            if key not in ("enabled", "monitor_metric", "patience", "min_delta"):
                raise ConfigError(f"params.yaml: trainer.early_stopping.{key} is not a key TraceFlow reads; "
                                  "keys here: enabled, monitor_metric, patience, min_delta")
    return out


def apply_env_overrides(base: dict, override, *, source: str, _path: str = "") -> dict:
    """An ENV file (master.yaml env_configs: dev/staging/prod.yaml) over params.yaml and the project's
    extra configs (owner ruling 2026-10-18):

    - only a key that exists there is overridden, and only the key named: a nested mapping
      overrides its own keys, the others keep their params.yaml values;
    - an empty value (null, "", [], {}) overrides nothing; 0 and false are values;
    - a key that does not exist, or a mapping for a key that is not one (or the reverse), raises
      ConfigError naming the file and the key path.

    A list or a value is replaced whole; one value for a list is a one-value sweep, as in params.yaml.
    A key that params.yaml leaves empty takes the override as it is."""
    if override is None:
        return base
    if not isinstance(override, dict):
        raise ConfigError(f"{source} must hold a mapping of params.yaml keys, got {type(override).__name__}")
    for key, value in override.items():
        path = f"{_path}.{key}" if _path else str(key)
        if key not in base:
            close = difflib.get_close_matches(str(key), [str(k) for k in base], n=1)
            hint = f"; did you mean {_path + '.' if _path else ''}{close[0]}?" if close else ""
            raise ConfigError(
                f"{source}: {path} is not a key of params.yaml or of the project's extra configs: an "
                f"override replaces an existing key only{hint}"
            )
        current = base[key]
        if isinstance(value, dict) and isinstance(current, dict):
            apply_env_overrides(current, value, source=source, _path=path)
            continue
        if _is_empty(value):
            continue
        if current is not None and isinstance(value, dict) != isinstance(current, dict):
            raise ConfigError(
                f"{source}: {path} is a {_shape(current)} in params.yaml; its override must be one too "
                f"(got a {_shape(value)})"
            )
        base[key] = value
    return base


def _make_paths_project_relative(paths: dict, *, project_root: Path) -> dict:
    """
    If a path is absolute AND under project_root, convert it to a project-relative string.
    Otherwise leave it as-is.
    """
    if not isinstance(paths, dict):
        return paths

    out = {}
    for k, v in paths.items():
        if not isinstance(v, str):
            out[k] = v
            continue

        p = Path(v)
        if p.is_absolute():
            try:
                rel = p.resolve().relative_to(project_root.resolve())
                out[k] = str(rel)
                continue
            except Exception:
                pass

        out[k] = v
    return out
def deep_override(base: dict, override: dict) -> dict:
    """
    Recursive override merge.

    Any field present in override wins.
    Dicts recurse.
    Lists/scalars replace wholesale.
    """
    base = copy.deepcopy(base or {})
    override = override or {}

    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_override(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def deep_fill_missing(base: dict, overlay: dict) -> dict:
    """
    Conservative merge:
    only fill missing fields from overlay.
    Existing scalar/list values in base are preserved.
    """
    base = copy.deepcopy(base or {})
    overlay = overlay or {}

    for k, v in overlay.items():
        if k not in base:
            base[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_fill_missing(base[k], v)
        else:
            # keep base[k] as-is
            pass

    return base


def _apply_auto_search_overrides(
    merged: dict,
    *,
    auto_search_yaml: dict,
    hard_replace_params: bool,
) -> dict:
    """
    Apply auto_search.yaml onto merged SCC config.

    hard_replace_params=True:
      Any field present in auto_search.yaml overrides merged cfg.

    hard_replace_params=False:
      auto_search.yaml only fills missing fields.
    """
    if not isinstance(auto_search_yaml, dict):
        raise TypeError("auto_search.yaml must parse into a dict")

    if hard_replace_params:
        return deep_override(merged, auto_search_yaml)
    return deep_fill_missing(merged, auto_search_yaml)

# The core entry points a project runs when its master.yaml entry names none (TODO.md §26: an entry of
# name, root and params alone trained, then every test failed with KeyError 'test'). A project's own
# scripts block overrides any of them.
DEFAULT_SCRIPTS = {
    "train": "src/sca/train.py",
    "model": "src/core/model.py",
    "test": "src/sca/test.py",
    "data_loader": "src/sca/ascad_data_loader.py",
}


def _resolve_scripts_dict(project_block: dict, project_root: Path) -> Dict[str, str]:
    scripts = {**DEFAULT_SCRIPTS, **(project_block.get("scripts", {}) or {})}
    code_dir_name = project_block.get("code_dir", "src")
    use_core_src = project_block.get("use_core_src", True)

    bases = []
    if use_core_src:
        bases.append(SCC_ROOT)
    bases.append(project_root)

    def candidate_paths(rel: Path):
        cands = []
        for base in bases:
            cands.append(base / rel)
            cands.append(base / code_dir_name / rel)
            if rel.parts and rel.parts[0] == code_dir_name:
                cands.append(base / rel)

        seen, out = set(), []
        for c in cands:
            s = str(c)
            if s not in seen:
                seen.add(s)
                out.append(c)
        return out

    resolved: Dict[str, str] = {}
    for key, value in scripts.items():
        rel = Path(value)

        if rel.is_absolute():
            resolved[key] = str(rel)
            continue

        found = None
        for cand in candidate_paths(rel):
            if cand.exists():
                found = cand.resolve()
                break

        if not found:
            if not (rel.parts and rel.parts[0] == code_dir_name):
                found = (SCC_ROOT / code_dir_name / rel).resolve()
            else:
                found = (SCC_ROOT / rel).resolve()
            logger.warning("Script '%s' not found. Using fallback: %s", key, found)

        resolved[key] = str(found)

    return resolved


# ---------- AutoMLOps merge policy ----------

_AUTO_MLOPS_OVERRIDE_KEYS_DEFAULT: Tuple[str, ...] = ("model", "trainer")


def _resolve_auto_mlops_path(cfg_path: str) -> Path:
    """
    Resolve plugin config path relative to PROJECTS_ROOT unless absolute.
    """
    p = Path(cfg_path)
    return p.resolve() if p.is_absolute() else (PROJECTS_ROOT / p).resolve()


def _apply_auto_mlops_overrides(
    merged: dict,
    *,
    auto_mlops_yaml: dict,
    hard_replace_params: bool,
    override_keys: Tuple[str, ...] = _AUTO_MLOPS_OVERRIDE_KEYS_DEFAULT,
) -> None:
    """
    Apply selected top-level overrides from auto_mlops.yaml onto SCC merged config.

    - hard_replace_params=True:
        merged[k] = auto_mlops_yaml[k]     (k in override_keys)
      => params.yaml is ignored for those sections.

    - hard_replace_params=False:
        merged[k] = deep_update(merged[k], auto_mlops_yaml[k])
      => params.yaml supplies missing values.

    Note:
      This function does NOT decide whether auto_mlops.yaml is loaded.
      That decision is made by get_project_config() based on master.yaml.
    """
    if not isinstance(auto_mlops_yaml, dict):
        raise TypeError("auto_mlops.yaml must parse into a dict")

    for k in override_keys:
        if k not in auto_mlops_yaml:
            continue

        if hard_replace_params:
            merged[k] = auto_mlops_yaml[k]
        else:
            if isinstance(auto_mlops_yaml[k], dict) and isinstance(merged.get(k), dict):
                merged[k] = deep_update(merged.get(k, {}), auto_mlops_yaml[k])
            else:
                merged[k] = auto_mlops_yaml[k]


def _validate_auto_mlops_yaml(auto_yaml: dict) -> dict:
    """
    Validate auto_mlops.yaml using plugin schema.

    Returns a lightweight dict summary you can safely keep in SCC cfg
    without pulling schema dataclasses deep into SCC runtime.

    Raises ValueError with schema context on any contract violation.
    """
    # Import locally so SCC runs fine even when plugin is not installed,
    # *as long as auto_mlops.enabled is false*.
    from services.mlops.schemas.auto_mlops import validate_auto_mlops_yaml as _v

    spec = _v(auto_yaml)  # AutoMLOpsSpec dataclass

    # Keep a small summary to avoid accidental heavy objects in cfg dumps
    return {
        "version": spec.version,
        "mode": spec.mode,
        "paths": dict(spec.paths),
        "notes": spec.notes,
        # keep raw for forward-compat and auditing (you can turn this off if you want)
        "raw": dict(spec.raw) if isinstance(spec.raw, dict) else {},
    }


# ---------- Main loader ----------

def get_project_config(project_name: str, validate_data: bool = True) -> dict:
    """
    Build the final SCC runtime config for `project_name`.

    Key guarantee:
      If master.yaml sets auto_mlops.enabled=false, auto_mlops.yaml is NEVER loaded.
    """

    # 1) Load master.yaml
    master_path = SCC_CONFIG_DIR / "master.yaml"
    if not master_path.exists():
        raise FileNotFoundError(f"master.yaml not found at {master_path}")
    master_cfg = load_yaml(master_path)
    config_schema.check_keys(master_cfg, config_schema.MASTER, source="master.yaml")

    # 2) Pick project block
    project_list = master_cfg.get("project", [])
    project_block = next((p for p in project_list if p.get("name") == project_name), None)
    if not project_block:
        raise ValueError(f"No project named '{project_name}' found in master.yaml")
    config_schema.check_keys(project_block, config_schema.MASTER_PROJECT,
                             source=f"master.yaml (project {project_name})")

    project_root = PROJECTS_ROOT / project_block["root"]
    # A run reads its frozen copy of config/ (utils/config_freeze.py: make experiment freezes at
    # queue start, a lone make orchestrate at its own), so an edit made meanwhile reaches the next run.
    frozen = frozen_dir(project_name, project_root)
    if frozen is not None:
        logger.info("project '%s': config files read from the run's frozen copy %s", project_name, to_portable_path(frozen))

    def cfg_file(rel) -> Path:
        return config_path(project_root / rel, project_root, frozen)

    params_path = cfg_file(project_block["params"])
    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found at {params_path}")

    # 3) Load base params.yaml (baseline SCC truth); a key no code reads is refused (utils/config_schema.py)
    merged = load_yaml(params_path)
    config_schema.check_keys(merged, config_schema.PARAMS, source=params_path.name)
    config_schema.refuse_moved(merged, source=params_path.name)

    # 4) Merge extras (graph, test, etc.)
    for name, rel in (project_block.get("extra_configs", {}) or {}).items():
        extra = cfg_file(rel)
        if not extra.exists():
            raise FileNotFoundError(f"Extra config '{name}' not found at {extra}")
        merged[name] = load_yaml(extra)
        if name == "graph":
            config_schema.check_keys(merged[name], config_schema.GRAPH, source=extra.name)

    # 5) Set ENV
    env = os.environ.get("ENV", "dev")
    merged["env"] = env

    # 6) env.yaml over base + extras: existing keys only, given values only (apply_env_overrides)
    env_cfgs = project_block.get("env_configs", {}) or {}
    if env_cfgs and env in env_cfgs:
        env_cfg_file = cfg_file(env_cfgs[env])
        if not env_cfg_file.exists():
            raise FileNotFoundError(f"ENV config file for '{env}' not found at: {env_cfg_file}")

        try:
            env_cfg = load_yaml(env_cfg_file)
        except Exception as e:
            raise RuntimeError(f"Failed to parse ENV config {env_cfg_file}: {e}") from e

        merged = apply_env_overrides(merged, env_cfg, source=env_cfg_file.name)

    # 6b) An empty key (`patience:`, YAML null) is not a setting: the code's default applies, as when the
    # key is absent. The blueprint starts a project with every key present and empty, and the panel's
    # "(empty)" means this; a required key left empty is refused by its reader as a missing one. Dropped
    # after the ENV overrides (an override may fill an empty key) and before the hash (an empty key would
    # join it as null).
    merged = canonical_form(merged)
    # 6b') `seed: null` written: an unseeded run (owner 2026-10-24; utils/seed.run_seed), kept as null (an empty
    # `seed:` was dropped above as no setting)
    if "seed" not in merged and explicit_null_seed(params_path):
        merged["seed"] = None
    check_seed_spacing(merged)                           # 6c) repeats of a cross-validated run share no training seed

    # 7) Flatten data
    if isinstance(merged.get("data"), list):
        for index, ds in enumerate(merged["data"]):
            if not isinstance(ds, dict) or not ds.get("name"):
                raise ConfigError(f"{params_path.name}: data entry #{index + 1} has no name; give it one (DATASET=<name> selects it)")
        merged["data"] = {ds["name"]: ds for ds in merged["data"]}
    if "data" not in merged or not isinstance(merged["data"], dict):
        raise ValueError("Final config is missing or has malformed `data:` block.")

    # 7b) Every entry agrees with its database type (src/sca/dataset_types.py): an unknown
    # type, an unknown key or a missing required key stops here, before a run is planned.
    for ds_block in merged["data"].values():
        validate_entry(ds_block)

    if validate_data:
        for ds_name, ds_block in merged["data"].items():
            if "database" not in ds_block:
                raise KeyError(f"Dataset '{ds_name}' is missing `database:` key in final config.")

            ds_block["database"] = _resolve_dataset_database_path(
                ds_block["database"],
                cfg=merged,
                project_name=project_name,
                project_root=project_root,
                projects_root=PROJECTS_ROOT,
                scc_root=SCC_ROOT,
                ds_name=ds_name,
            )
    else:
        logger.info(
            "Skipping dataset database validation for project='%s' because validate_data=False",
            project_name,
        )
    # 8) AutoMLOps metadata + CONDITIONAL loading/applying
    auto_mlops_block = project_block.get("auto_mlops", {}) or {}
    auto_enabled = bool(auto_mlops_block.get("enabled", False))
    auto_cfg = auto_mlops_block.get("config")  # path to auto_mlops.yaml
    hard_replace_params = bool(auto_mlops_block.get("hard_replace_params", False))

    merged["auto_mlops"] = {
        "enabled": auto_enabled,
        "config": auto_cfg,
        "hard_replace_params": hard_replace_params,
    }

    # Important: even if config is declared, we do NOT load it unless enabled.
    if auto_cfg:
        auto_abs = _resolve_auto_mlops_path(auto_cfg)
        merged["auto_mlops"]["config_abs"] = str(auto_abs)

        if auto_enabled:
            if not auto_abs.exists():
                raise FileNotFoundError(f"auto_mlops.yaml enabled but not found at: {auto_abs}")

            auto_yaml = load_yaml(auto_abs)

            # schema validation (plugin contract)
            merged["auto_mlops"]["spec"] = _validate_auto_mlops_yaml(auto_yaml)

            # optional: keep full yaml for traceability (comment out if you want smaller cfg dumps)
            merged["auto_mlops"]["yaml"] = auto_yaml

            # apply overrides (model/trainer)
            _apply_auto_mlops_overrides(
                merged,
                auto_mlops_yaml=auto_yaml,
                hard_replace_params=hard_replace_params,
                override_keys=_AUTO_MLOPS_OVERRIDE_KEYS_DEFAULT,
            )
    # 8.5) AutoSearch (paper reproduction / BO / RS)
    auto_search_block = project_block.get("auto_search", {}) or {}
    auto_search_enabled = bool(auto_search_block.get("enabled", False))
    auto_search_cfg = auto_search_block.get("config")
    auto_search_hard = bool(auto_search_block.get("hard_replace_params", False))

    merged["auto_search"] = {
        "enabled": auto_search_enabled,
        "config": auto_search_cfg,
        "hard_replace_params": auto_search_hard,
    }

    if auto_search_cfg:
        auto_search_abs = cfg_file(auto_search_cfg).resolve()
        merged["auto_search"]["config_abs"] = str(auto_search_abs)

        if auto_search_enabled:
            if not auto_search_abs.exists():
                raise FileNotFoundError(f"auto_search.yaml not found at: {auto_search_abs}")

            auto_search_yaml = load_yaml(auto_search_abs)
            if not isinstance(auto_search_yaml, dict):
                raise TypeError(f"auto_search.yaml must parse into a dict: {auto_search_abs}")
            config_schema.check_keys(auto_search_yaml, config_schema.AUTO_SEARCH, source=auto_search_abs.name)

            # Preserve metadata first so it survives the override application.
            auto_search_meta = copy.deepcopy(merged["auto_search"])

            # Raw spec for debugging / traceability
            auto_search_meta["yaml"] = copy.deepcopy(auto_search_yaml)

            # Optional safety: prevent recursive self-overwrite of metadata block
            auto_search_payload = copy.deepcopy(auto_search_yaml)
            auto_search_payload.pop("auto_search", None)

            merged = _apply_auto_search_overrides(
                merged,
                auto_search_yaml=auto_search_payload,
                hard_replace_params=auto_search_hard,
            )

            # Restore/attach loader-owned metadata block
            merged["auto_search"] = auto_search_meta
            merged = canonical_form(merged)              # 6b again: an empty key or a written form of the overlay
    # 9) Inject final project metadata AFTER env overrides
    merged["env"] = env
    # TRACEFLOW_OUTPUT_ROOT (make reproduce REPRODUCE_ROOT=<folder>, utils/reproduce.py): the run's outputs
    # (every paths: template, the registries, the heartbeat) go under that folder instead of the project's
    # root; the config files above were read from the project's root, kept as config_root. Only for the
    # project TRACEFLOW_OUTPUT_PROJECT names, when it is set.
    output_root = os.environ.get("TRACEFLOW_OUTPUT_ROOT", "").strip()
    output_project = os.environ.get("TRACEFLOW_OUTPUT_PROJECT", "").strip()
    if output_root and (not output_project or output_project == project_name):
        run_root = Path(output_root).expanduser().resolve()
    else:
        run_root = Path(project_root)
    merged["project"] = {
        "name": project_name,
        "root": str(run_root),
        "config_root": str(project_root),
        "params": project_block["params"],
        "code_dir": project_block.get("code_dir", "src"),
        "logs": project_block.get("logs", "artifacts/logs/"),
        "models": project_block.get("models", "artifacts/models/"),
        "metrics": project_block.get("metrics", "artifacts/metrics/"),
        "use_core_src": project_block.get("use_core_src", True),
        "scripts": project_block.get("scripts", {}),
        "env_configs": project_block.get("env_configs", {}),
        # Carried here so logger.pretty_config never has to re-read master.yaml
        # from a fixed mount point (a host run has no /app).
        "print_config": project_block.get("print_config", True),
    }

    merged["project"]["scripts"] = _resolve_scripts_dict(project_block, project_root)
    merged["project"]["core_src_dir"] = str((SCC_ROOT / "src").resolve())
    merged["project"]["project_src_dir"] = str((project_root / merged["project"]["code_dir"]).resolve())
    # 9.1) Preserve project-level visualization metadata from master.yaml.
    # This includes graph_manager.enabled/config/ui and dashboard port settings.
    merged["visualization"] = copy.deepcopy(project_block.get("visualization", {}) or {})

    graph_manager_block = (
        (merged.get("visualization") or {})
        .get("graph_manager", {}) or {}
    )

    if graph_manager_block:
        graph_rel = graph_manager_block.get("config")
        if graph_rel:
            graph_abs = cfg_file(graph_rel).resolve()
            merged["visualization"]["graph_manager"]["config_abs"] = str(graph_abs)

            # Defensive consistency check:
            # If graph_manager points to a graph config, it should match loaded extra config.
            if not graph_abs.exists():
                raise FileNotFoundError(
                    f"visualization.graph_manager.config points to missing file: {graph_abs}"
                )

        # If graph_manager is enabled, require loaded cfg["graph"].
        if bool(graph_manager_block.get("enabled", False)) and "graph" not in merged:
            raise KeyError(
                "graph_manager.enabled=true but cfg['graph'] was not loaded. "
                "Add extra_configs.graph: 'config/graph_config.yaml' in master.yaml."
            )
    # 9.5) Normalize paths for human-facing cfg (remove /app noise)
    # Keep cfg["paths"] project-relative; also provide cfg["paths_abs"] for execution.
    if isinstance(merged.get("paths"), dict):
        merged["paths"] = _make_paths_project_relative(merged["paths"], project_root=project_root)

        # runtime absolute map (deterministic)
        try:
            from utils.path_utils import resolve_path
            merged["paths_abs"] = {k: str(resolve_path(v, merged)) if isinstance(v, str) else v
                                for k, v in merged["paths"].items()}
        except Exception:
            # never crash config load because of convenience map
            merged["paths_abs"] = dict(merged["paths"])

    # 10) Apply distributed patch
    dist_cfg = master_cfg.get("distributed", {}) or {}
    override = project_block.get("distributed_override", {}) or {}
    if override.get("use"):
        dist_cfg.update({k: v for k, v in override.items() if k != "use"})
    merged["distributed"] = dist_cfg

    # 11) Core distributed script paths
    core_dir = SCC_ROOT
    for key in ("ddp_script", "horovod_script", "multi_worker_script"):
        rel = merged["distributed"].get(key)
        if isinstance(rel, str):
            merged["distributed"][key] = str((core_dir / rel).resolve())

    # 12) Optional model_config.yaml
    model_cfg = cfg_file("config/model_config.yaml")
    if model_cfg.exists():
        merged.setdefault("extras", {})["model_config"] = load_yaml(model_cfg)

    # 13) Evaluate numeric expressions
    merged = _walk_and_eval(merged)
    # Defensive boundary: downstream code MUST NOT be able to mutate the loader's result
    return copy.deepcopy(merged)
