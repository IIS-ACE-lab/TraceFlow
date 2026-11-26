# services/mlops/utils/auto_config.py
"""
Config loader and manager for MLOps (production-grade).
======================================================

Keeps **usage identical**:

    from services.mlops.utils.auto_config import MLOpsConfig
    cfg = MLOpsConfig()
    auto_params_path = cfg.get_map(
        "auto_mlops",
        project_name="MyProject",
        key_map="paths.auto_params",
        resolve_as_path=True,
    )

Key design goals
----------------
- Federation-style `get_map()` API (dot-map lookup + optional path resolve)
- Strong validation for auto_mlops.yaml via `services/mlops/schemas/auto_mlops.py`
- Stable singleton behavior in staging/prod, hot-reload in dev (ENV=dev)
- Safe caching (mtime-aware), no silent failures, clear errors
- Tracking registry resolves file paths relative to registry file location

Notes
-----
- Schemas are PURE validation; this module does config loading + optional
  project overrides deep-merge + path resolution.
- If `core_cfg` is not provided, we attempt to pull it from your Federation
  singleton accessor (if available). If not available, we fail loudly with
  an actionable error.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

try:
    from yaml import CSafeLoader as SafeLoader
except Exception:
    from yaml import SafeLoader  # type: ignore

from services.mlops.utils.path_resolver import resolve_path

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    # Only for type-checkers (mypy/pyright); never executed at runtime
    from services.mlops.schemas.auto_mlops import AutoMLOpsSpec as _AutoMLOpsSpec
else:
    _AutoMLOpsSpec = Any  # runtime fallback type alias


# ---- Schema (v1) ----
try:
    from services.mlops.schemas.auto_mlops import validate_auto_mlops_yaml
except Exception:  # pragma: no cover
    validate_auto_mlops_yaml = None  # type: ignore[assignment]


log = logging.getLogger(__name__)


# =============================================================================
# File cache (mtime-aware)
# =============================================================================

@dataclass(frozen=True)
class _FileCacheEntry:
    mtime_ns: int
    text: str


class _FileCache:
    """
    Thread-safe file cache with mtime invalidation.
    - If hot_reload=True: never cache (always read from disk).
    """

    def __init__(self, *, hot_reload: bool = False) -> None:
        self._hot_reload = hot_reload
        self._lock = threading.RLock()
        self._text_cache: Dict[str, _FileCacheEntry] = {}

    def read_text(self, path: Path) -> str:
        p = path.resolve()
        if self._hot_reload:
            return p.read_text(encoding="utf-8")

        key = str(p)
        st = p.stat()
        with self._lock:
            ent = self._text_cache.get(key)
            if ent and ent.mtime_ns == st.st_mtime_ns:
                return ent.text
            txt = p.read_text(encoding="utf-8")
            self._text_cache[key] = _FileCacheEntry(mtime_ns=st.st_mtime_ns, text=txt)
            return txt

    def clear(self) -> None:
        with self._lock:
            self._text_cache.clear()


# =============================================================================
# Low-level config loaders
# =============================================================================

def _yaml_load_text(txt: str) -> Dict[str, Any]:
    obj = yaml.load(txt, Loader=SafeLoader)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("[mlops] YAML root must be a mapping/dict")
    return obj


def _json_load_text(txt: str) -> Dict[str, Any]:
    obj = json.loads(txt or "{}")
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("[mlops] JSON root must be a mapping/dict")
    return obj


def _load_config_file(path: Path, *, cache: _FileCache) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"[mlops] config file missing: {path}")

    suf = path.suffix.lower()
    txt = cache.read_text(path)

    if suf in (".yaml", ".yml"):
        return _yaml_load_text(txt)
    if suf == ".json":
        return _json_load_text(txt)

    raise ValueError(f"[mlops] unsupported config extension: {suf} ({path})")


# =============================================================================
# Deep helpers
# =============================================================================

def _deep_get(root: Dict[str, Any], key_map: str) -> Any:
    if not key_map:
        return root
    node: Any = root
    for part in key_map.split("."):
        if not isinstance(node, dict):
            raise KeyError(f"[mlops/get_map] cannot descend into non-dict at '{part}' (map='{key_map}')")
        if part not in node:
            raise KeyError(f"[mlops/get_map] key '{part}' not found (map='{key_map}')")
        node = node[part]
    return node


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge override into base (dicts recursively).
    Lists/scalars are replaced (override wins).
    """
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


# =============================================================================
# Tracking registry
# =============================================================================

def load_tracking_registry(registry_path: str, *, cache: _FileCache) -> Dict[str, Path]:
    """
    Load config_tracking_registry.yaml/json and return mapping config_key -> absolute Path.

    Important:
    - tracked file paths are resolved relative to the registry file's directory
      if they are not absolute.
    """
    rp = Path(registry_path).expanduser().resolve()
    reg = _load_config_file(rp, cache=cache)

    files = reg.get("tracked_files", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("[mlops] tracking registry must contain non-empty tracked_files mapping")

    base_dir = rp.parent
    out: Dict[str, Path] = {}
    for key, rel in files.items():
        if not isinstance(rel, str) or not rel.strip():
            raise ValueError(f"[mlops] tracked_files['{key}'] must be a non-empty string path")
        p = Path(rel).expanduser()
        if not p.is_absolute():
            p = (base_dir / p)
        out[key] = p.resolve()
    return out


# =============================================================================
# Project selection
# =============================================================================

@dataclass(frozen=True)
class ProjectSelection:
    project_block: Dict[str, Any]
    selected_index: int


# =============================================================================
# Core config acquisition (optional federation bridge)
# =============================================================================

def _try_load_core_cfg_from_federation() -> Optional[Dict[str, Any]]:
    """
    Best-effort: pull core cfg via your Federation singleton accessor if present.

    This keeps `MLOpsConfig()` usable without args in your ecosystem.
    If federation isn't available in some unit test context, return None.
    """
    try:
        # You said you use:
        #   from Brahman.utils.administration.federation_config_loader import get_federation
        #   _FED = get_federation()#TODO: Build a config loader
        from utils.administration.federation_config_loader import get_federation  # type: ignore

        fed = get_federation()
        # We don't hardcode your federation keypaths here; we just ask for a global
        # dict-like config map if your FederationConfig exposes it.
        #
        # If your FederationConfig uses get_map(namespace, key, ...), you can export
        # a single "core_cfg" map there and it will be picked up.
        if hasattr(fed, "get_map"):
            # Convention: allow ENV override for the federation key.
            # Example:
            #   export MLOPS_CORE_CFG_FED_KEY="side_channel_core.runtime.core_cfg"
            k = os.environ.get("MLOPS_CORE_CFG_FED_KEY", "").strip()
            if k:
                val = fed.get_map("b_fed", k, os_name=os.name)  # type: ignore[attr-defined]
                if isinstance(val, dict):
                    return val
        return None
    except Exception:
        return None


# =============================================================================
# Singleton facade (usage stays: cfg = MLOpsConfig())
# =============================================================================

_mlops_instance: Optional["MLOpsConfig"] = None
_mlops_lock = threading.RLock()


class MLOpsConfig:
    """
    Federation-like config reader for MLSCA (MLOps plugin).

    Usage stays identical:
        cfg = MLOpsConfig()
        cfg.get_map(...)

    Singleton rules:
      - DEV (ENV=dev): hot_reload=True => always re-read files & re-validate.
      - STAGING/PROD (ENV!=dev): singleton instance (stable), mtime caching.

    Key features:
      - Loads configs by key via config_tracking_registry.yaml
      - Dot-map lookup: cfg.get_map("auto_mlops", "paths.auto_params")
      - If project_name is provided and config has project[]:
          - selects project by name
          - for auto_mlops: applies selected project's overrides via deep merge
          - attaches selected project at compiled["_project"]
      - auto_mlops: schema validation via validate_auto_mlops_yaml()
          - attaches parsed spec at compiled["_spec"] for downstream use
      - Optional path resolution via resolve_path()
      - Optional touch_file for resolved file paths
    """

    # ---------------------------
    # Singleton constructor
    # ---------------------------

    def __new__(cls, *args: Any, **kwargs: Any) -> "MLOpsConfig":
        env = os.environ.get("ENV", "dev").strip().lower()
        if env == "dev":
            # Dev: always a fresh instance (hot reload).
            return super().__new__(cls)

        # Non-dev: singleton
        global _mlops_instance
        with _mlops_lock:
            if _mlops_instance is None:
                _mlops_instance = super().__new__(cls)
            return _mlops_instance

    def __init__(
        self,
        *,
        core_cfg: Optional[Dict[str, Any]] = None,
        registry_path: str = "services/mlops/config/config_tracking_registry.yaml",
    ) -> None:
        # Prevent re-init of singleton in non-dev
        if getattr(self, "_initialized", False) and os.environ.get("ENV", "dev").strip().lower() != "dev":
            return

        env = os.environ.get("ENV", "dev").strip().lower()
        self.hot_reload = env == "dev"

        self._cache = _FileCache(hot_reload=self.hot_reload)
        self.registry_path = registry_path

        # Core cfg (needed for resolve_path templating)
        if core_cfg is None:
            core_cfg = _try_load_core_cfg_from_federation()

        if core_cfg is None:
            raise RuntimeError(
                "[mlops] MLOpsConfig requires core_cfg, but none was provided and federation fallback was not available.\n"
                "Fix options:\n"
                "  1) Pass core_cfg explicitly: MLOpsConfig(core_cfg=core_cfg)\n"
                "  2) Export MLOPS_CORE_CFG_FED_KEY to a federation map key that returns a dict\n"
                "     e.g. export MLOPS_CORE_CFG_FED_KEY='record_keeper.freedom_runtime.core_cfg'\n"
            )
        if not isinstance(core_cfg, dict):
            raise TypeError(f"[mlops] core_cfg must be a dict; got {type(core_cfg)}")

        self.core_cfg = core_cfg

        # registry (mtime-aware)
        self.registry = load_tracking_registry(self.registry_path, cache=self._cache)

        # caches
        self._lock = threading.RLock()
        self._raw_cache: Dict[str, Dict[str, Any]] = {}
        self._compiled_cache: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
        self._auto_spec_cache: Dict[Tuple[str, Optional[str]], _AutoMLOpsSpec] = {}

        self._initialized = True

    # ---------------------------
    # Loaders
    # ---------------------------

    def load(self, config_key: str) -> Dict[str, Any]:
        """
        Load raw config dict for a given registry key.
        In dev: always reload.
        In non-dev: cache by key.
        """
        if config_key not in self.registry:
            raise KeyError(f"[mlops] unknown config key '{config_key}' (not in tracking registry)")

        if self.hot_reload:
            return _load_config_file(self.registry[config_key], cache=self._cache)

        with self._lock:
            if config_key in self._raw_cache:
                return self._raw_cache[config_key]
            obj = _load_config_file(self.registry[config_key], cache=self._cache)
            self._raw_cache[config_key] = obj
            return obj

    def _select_project_block(self, cfg_obj: Dict[str, Any], project_name: str) -> ProjectSelection:
        projects = cfg_obj.get("project")
        if not isinstance(projects, list):
            raise KeyError("[mlops] project_name provided but config has no 'project' list")

        for i, p in enumerate(projects):
            if isinstance(p, dict) and p.get("name") == project_name:
                return ProjectSelection(project_block=p, selected_index=i)

        names = [p.get("name") for p in projects if isinstance(p, dict)]
        raise KeyError(f"[mlops] project '{project_name}' not found. Found: {names}")

    def _validate_auto_mlops(self, raw: Dict[str, Any]) -> _AutoMLOpsSpec:
        if validate_auto_mlops_yaml is None:
            raise RuntimeError(
                "[mlops] auto_mlops schema validator import failed. "
                "Expected: services.mlops.schemas.auto_mlops.validate_auto_mlops_yaml"
            )
        spec = validate_auto_mlops_yaml(raw)
        return cast(_AutoMLOpsSpec, spec)


    def compile(self, config_key: str, project_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Compile config:
        - auto_mlops: schema validate
        - if project_name is provided:
            - select project
            - for auto_mlops: deep-merge overrides into raw
            - attach selected project block at _project
            - attach parsed schema at _spec (auto_mlops only)
        """
        ck = (config_key, project_name)

        if self.hot_reload:
            raw = self.load(config_key)
            return self._compile_uncached(config_key, raw, project_name)

        with self._lock:
            if ck in self._compiled_cache:
                return self._compiled_cache[ck]
            raw = self.load(config_key)
            out = self._compile_uncached(config_key, raw, project_name)
            self._compiled_cache[ck] = out
            return out

    def _compile_uncached(self, config_key: str, raw: Dict[str, Any], project_name: Optional[str]) -> Dict[str, Any]:
        # Validate auto_mlops and keep spec for downstream
        auto_spec: Optional[_AutoMLOpsSpec] = None
        if config_key == "auto_mlops":
            auto_spec = self._validate_auto_mlops(raw)

        # No project selection
        if not project_name:
            out = deepcopy(raw)
            if auto_spec is not None:
                out["_spec"] = auto_spec
            return out

        # Select project
        sel = self._select_project_block(raw, project_name)

        if config_key == "auto_mlops":
            overrides = sel.project_block.get("overrides") or {}
            if overrides and not isinstance(overrides, dict):
                raise TypeError(f"[mlops] project '{project_name}' overrides must be a dict if present")
            out = _deep_merge(raw, overrides)
            out["_project"] = deepcopy(sel.project_block)
            if auto_spec is not None:
                # Spec validated the *raw*; overrides may change effective values.
                # We re-validate the merged result to avoid silent drift.
                merged_spec = self._validate_auto_mlops(out)
                out["_spec"] = merged_spec
            return out

        # Non-auto configs: attach project only
        out = deepcopy(raw)
        out["_project"] = deepcopy(sel.project_block)
        return out

    # ---------------------------
    # get_map (Federation-style)
    # ---------------------------

    def get_map(
        self,
        config_key: str,
        key_map: Optional[str] = None,
        *,
        project_name: Optional[str] = None,
        resolve_as_path: bool = False,
        mkdir: bool = True,
        strict: bool = True,
        extra: Optional[Dict[str, Any]] = None,
        touch_file: bool = False,
    ) -> Any:
        """
        Example:
          cfg.get_map(
              "auto_mlops",
              "paths.auto_params",
              project_name="DLSCA_ASCAD",
              resolve_as_path=True,
              extra={"sota_trial": "..."},
          )

        - If resolve_as_path=True and looked-up value is a string:
            - resolve via services/mlops/utils/path_resolver.resolve_path()
            - mkdir as per resolver option
            - optionally touch the file if it looks like a file path
        """
        compiled = self.compile(config_key, project_name=project_name)

        value: Any = compiled if not key_map else _deep_get(compiled, key_map)

        if not resolve_as_path:
            return value

        if not isinstance(value, str):
            raise TypeError(
                f"[mlops/get_map] resolve_as_path=True requires a string path template; got {type(value)}"
            )

        merged_extra = dict(extra or {})
        if project_name:
            merged_extra.setdefault("project_name", project_name)

        p = resolve_path(
            value,
            cfg=self.core_cfg,
            extra=merged_extra,
            mkdir=mkdir,
            strict=strict,
        )

        # Touch file only if it *looks* like a file path (has a suffix)
        if touch_file and p.suffix:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch(exist_ok=True)

        return p
