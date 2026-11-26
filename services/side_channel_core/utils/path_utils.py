# utils/path_utils.py
"""
Path Utilities — SideChannel Core (Authoritative Resolver)

Fixes:
- Ensures runtime identifiers (env/model_name/dataset_alias) exist BEFORE resolving templates.
- Deterministically generates model_name if missing (stable hash over cfg fields).
- resolve_paths_dict() defaults to inject=True so paths_abs is always concrete.
- Supports cfg["data"] as dict OR list[dict].
- Strict mode prevents silently persisting unresolved templates.

Author: Shraddha Mishra

"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import StrictUndefined, Template, Undefined

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Root detection helpers
# ---------------------------------------------------------------------


def _get_workspace_root(cfg: dict | None = None) -> Path:
    """
    Repo/workspace root.
    Priority:
      1) WORKSPACE_ROOT / REPO_ROOT
      2) infer from SCC_ROOT (..../services/side_channel_core -> workspace is before 'services')
      3) infer from PROJECTS_ROOT (/app/projects -> /app)
      4) the checkout holding this file (<root>/services/side_channel_core/utils/)
      5) /app if exists
      6) cwd
    """
    env_root = (os.environ.get("WORKSPACE_ROOT") or os.environ.get("REPO_ROOT") or "").strip()
    if env_root:
        return Path(env_root).resolve()

    scc = (os.environ.get("SCC_ROOT") or "").strip()
    if scc:
        p = Path(scc).resolve()
        parts = list(p.parts)
        if "services" in parts:
            idx = parts.index("services")
            return Path(*parts[:idx]).resolve()
        return p.parent.parent.resolve()

    pr = (os.environ.get("PROJECTS_ROOT") or "").strip()
    if pr:
        return Path(pr).resolve().parent

    # Host runs without the env vars: the checkout this module belongs to,
    # wherever it is cloned (the working directory is not the repo root).
    here = Path(__file__).resolve().parents[3]
    if (here / "services" / "side_channel_core").is_dir():
        return here

    if Path("/app").exists():
        return Path("/app").resolve()

    return Path.cwd().resolve()


def _get_scc_root(cfg: dict | None = None) -> Path:
    """
    SCC root.
    Priority:
      1) SCC_ROOT
      2) WORKSPACE_ROOT/services/side_channel_core
    """
    v = (os.environ.get("SCC_ROOT") or "").strip()
    if v:
        return Path(v).resolve()
    return (_get_workspace_root(cfg) / "services/side_channel_core").resolve()


def _get_project_root(cfg: dict | None = None) -> Path | None:
    if not cfg:
        return None
    proj = cfg.get("project")
    if isinstance(proj, dict):
        root = proj.get("root")
        if root:
            return Path(root).resolve()
    return None


def _ws_root_str() -> str:
    # The detected workspace root (host checkout or container /app), not a
    # hard-coded /app: host runs must strip their own prefix too.
    return str(_get_workspace_root()).replace("\\", "/").rstrip("/")


def to_portable_path(p: str | Path, *, project_root: str | Path | None = None) -> str:
    """
    Pure string-based conversion to a machine-independent relative path that
    resolve_path() maps back to the same location (same project_root):
    '<project_root>/artifacts/...' -> 'artifacts/...'          (project-relative)
    '<workspace>/services/...'     -> 'services/...'           (workspace-relative)
    '<project_root>/data/...'      -> 'projects/<P>/data/...'  ('data/...' is SCC-scoped)
    The project root itself stays workspace-relative ('projects/<P>').
    A path no relative form maps back to (outside both roots, or e.g.
    '<workspace>/datasets/...', which would resolve under the project) is
    returned unchanged. Works even if path does not exist.
    """
    ws = _ws_root_str()
    s = str(p).replace("\\", "/")

    def _maps_back(rel: str, scopes: tuple[str, ...]) -> bool:
        # resolve_path() normalises the relative path, then routes it by prefix.
        return bool(rel) and _normalize_rel(rel) == rel and _scope(rel) in scopes

    if project_root:
        proj = str(project_root).replace("\\", "/").rstrip("/")
        if proj and s.startswith(proj + "/") and _maps_back(s[len(proj) + 1 :], ("project",)):
            return s[len(proj) + 1 :]

    if s == ws and not project_root:
        return "."
    if s.startswith(ws + "/"):
        # Without a project root, resolve_path() roots project-scoped paths at the workspace.
        scopes = ("workspace",) if project_root else ("workspace", "project")
        if _maps_back(s[len(ws) + 1 :], scopes):
            return s[len(ws) + 1 :]
    return s


def normalize_paths_for_dump(obj: Any, *, project_root: str | Path | None = None) -> Any:
    """
    Deep transform:
    - Converts Path -> portable string
    - Converts any string under project_root / the workspace root to portable
      (see to_portable_path); resolve_path() re-roots them on load.
    """
    def _walk(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_walk(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_walk(v) for v in x)
        if isinstance(x, Path):
            return to_portable_path(x, project_root=project_root)
        if isinstance(x, str) and x.replace("\\", "/").startswith("/"):
            # Only absolute paths are rewritten; relative strings are already portable.
            return to_portable_path(x, project_root=project_root)
        return x

    return _walk(obj)


# ---------------------------------------------------------------------
# Routing rules
# ---------------------------------------------------------------------

_PROJECT_SCOPED_PREFIXES = ("artifacts/", "config/")
_WORKSPACE_SCOPED_PREFIXES = ("services/", "deployment/", "infra/", "projects/")
_SCC_SCOPED_PREFIXES = ("data/", "side_channel_core/")


def _normalize_rel(s: str) -> str:
    rel = s.strip().lstrip("./")
    return rel.replace("\\", "/")


def _scope(rel_norm: str) -> str:
    """The root resolve_path() joins a relative path to: 'project', 'workspace' or 'scc'."""
    if rel_norm.startswith(_PROJECT_SCOPED_PREFIXES):
        return "project"
    if rel_norm.startswith(_WORKSPACE_SCOPED_PREFIXES):
        return "workspace"
    if rel_norm.startswith(_SCC_SCOPED_PREFIXES):
        return "scc"
    return "project"


# ---------------------------------------------------------------------
# Template vars + context selection
# ---------------------------------------------------------------------

_TEMPLATE_VAR_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


def _template_vars(s: str) -> set[str]:
    return set(_TEMPLATE_VAR_RE.findall(s))


def _pick_data_block(cfg: dict) -> Optional[dict]:
    """
    cfg["data"] may be dict OR list[dict].
    If list, select by dataset/dataset_name/dataset_alias or env DATASET. A
    selector that matches no entry raises; with several entries and no
    selector nothing is selected (None), so no dataset is chosen silently.
    """
    data = cfg.get("data")

    if isinstance(data, dict):
        return data

    if isinstance(data, list) and data:
        blocks = [d for d in data if isinstance(d, dict)]
        wanted = (
            cfg.get("dataset")
            or cfg.get("dataset_name")
            or cfg.get("dataset_alias")
            or os.environ.get("DATASET")
        )
        if wanted:
            for d in blocks:
                # match across common keys
                if wanted in (d.get("name"), d.get("alias"), d.get("output_name")):
                    return d
            available = [d.get("alias") or d.get("name") for d in blocks]
            raise ValueError(
                f"Dataset {wanted!r} (dataset/dataset_alias or DATASET) matches no "
                f"entry of cfg['data']; configured datasets: {available}"
            )
        if len(blocks) == 1:
            return blocks[0]

    return None


# ---------------------------------------------------------------------
# Deterministic runtime identity generation (the real fix)
# ---------------------------------------------------------------------


def _stable_json(obj: Any) -> str:
    """
    Stable JSON encoding for hashing.
    - drop Path to str
    - sort keys
    """
    def _default(o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Not JSON serializable: {type(o)}")

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def _derive_model_name(cfg: dict) -> str:
    """
    Deterministically derive a model_name when not provided yet.

    We hash only *stable* identifiers (not timestamps) so that:
    - same cfg => same model_name
    - different dataset/seed/project => different model_name

    If you later inject a 'funny name' generator, do it BEFORE calling resolve_paths_dict(),
    or set MODEL_NAME/RUN_NAME env vars.
    """
    project = (cfg.get("project") or {})
    project_name = cfg.get("project_name") or project.get("name") or project.get("id") or "project"
    env = cfg.get("env") or os.environ.get("ENV", "dev")

    data_block = _pick_data_block(cfg) or {}
    dataset_alias = (
        cfg.get("dataset_alias")
        or data_block.get("output_name")
        or data_block.get("alias")
        or data_block.get("name")
        or cfg.get("dataset")
        or os.environ.get("DATASET")
        or "dataset"
    )

    seed = None
    if isinstance(cfg.get("experiment"), dict):
        seed = cfg["experiment"].get("seed")
    seed = seed if seed is not None else cfg.get("seed")

    payload = {
        "project": str(project_name),
        "env": str(env),
        "dataset_alias": str(dataset_alias),
        "seed": seed,
    }

    h = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:10]
    # short + readable; consistent with production logs
    return f"{project_name}-{dataset_alias}-{env}-{h}"


def ensure_runtime_identifiers(cfg: dict, *, force: bool = False) -> dict:
    """
    Mutates cfg in-place (intentionally) to ensure:
      - cfg["env"]
      - cfg["dataset_alias"]
      - cfg["model_name"] (or cfg["run"]["model_name"])

    This is the missing "inject at the right time" step.
    Call it once right after loading cfg OR let resolve_path(inject=True) do it.

    force=True will overwrite existing values.
    """
    if not isinstance(cfg, dict):
        return cfg

    # env
    if force or not cfg.get("env"):
        cfg["env"] = cfg.get("env") or os.environ.get("ENV", "dev")
    frozen = cfg.get("frozen_identity") or {}
    if isinstance(frozen, dict):
        frozen_name = str(frozen.get("model_name") or "").strip()
        frozen_hash = str(frozen.get("model_hash") or "").strip()

        if frozen_name and frozen_hash:
            cfg["model_name"] = frozen_name
            cfg["model_hash"] = frozen_hash
            cfg.setdefault("run", {})
            cfg["run"]["model_name"] = frozen_name
            return cfg
    # dataset_alias
    data_block = _pick_data_block(cfg) or {}
    ds = (
        cfg.get("dataset_alias")
        or data_block.get("output_name")
        or data_block.get("alias")
        or data_block.get("name")
        or cfg.get("dataset")
        or os.environ.get("DATASET")
    )
    if (force or not cfg.get("dataset_alias")) and ds:
        cfg["dataset_alias"] = str(ds)

    # model_name: prefer explicit or env, else derived
    existing = (
        cfg.get("model_name")
        or (cfg.get("run") or {}).get("model_name")
        or (cfg.get("experiment") or {}).get("name")
        or cfg.get("name")
        or os.environ.get("MODEL_NAME")
        or os.environ.get("RUN_NAME")
    )

    if force or not existing or (isinstance(existing, str) and existing.strip() == ""):
        generated = _derive_model_name(cfg)
        # store in both places without breaking older code
        cfg["model_name"] = generated
        cfg.setdefault("run", {})
        if isinstance(cfg["run"], dict):
            cfg["run"].setdefault("model_name", generated)

    return cfg


def _finalize_runtime_context(cfg: dict | None, overrides: dict, *, inject: bool) -> Dict[str, Any]:
    """
    Finalize templating context at call time.
    If inject=True, ensures runtime identifiers exist in cfg first.
    """
    ctx: Dict[str, Any] = {}

    if isinstance(cfg, dict):
        if inject:
            ensure_runtime_identifiers(cfg, force=False)

        # shallow merge for templating keys
        ctx.update(cfg)

        # minimal fallbacks (in case inject=False)
        ctx.setdefault("env", cfg.get("env") or os.environ.get("ENV", "dev"))

        ctx.setdefault(
            "model_name",
            cfg.get("model_name")
            or (cfg.get("run") or {}).get("model_name")
            or (cfg.get("experiment") or {}).get("name")
            or cfg.get("name")
            or os.environ.get("MODEL_NAME")
            or os.environ.get("RUN_NAME"),
        )

        if cfg.get("dataset_alias"):
            ctx.setdefault("dataset_alias", cfg.get("dataset_alias"))
        else:
            data_block = _pick_data_block(cfg)
            if data_block:
                ds = data_block.get("output_name") or data_block.get("alias") or data_block.get("name")
                if ds:
                    ctx.setdefault("dataset_alias", str(ds))

        ctx.setdefault("dataset_alias", cfg.get("dataset") or os.environ.get("DATASET"))

    # explicit overrides win
    ctx.update(overrides or {})
    return ctx


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def resolve_path(
    template: str | Path,
    cfg: dict | None = None,
    *,
    strict: bool = True,
    mkdir: bool = False,
    inject: bool = False,
    **overrides: Any,
) -> Path:
    """
    Render + resolve a path template.

    strict=True:
      - if template contains {{var}} and ctx[var] missing/None/"" -> raises KeyError
      - if rendering leaves '{{' behind -> raises ValueError (prevents persisting templates)

    inject=True:
      - ensures cfg has env/dataset_alias/model_name BEFORE rendering (mutates cfg)
      - This is what you want when generating paths_abs early.

    mkdir:
      - best-effort create directories
    """
    if isinstance(template, Path):
        template = str(template)

    tpl = str(template).strip()
    if not tpl:
        raise ValueError("resolve_path(): empty template")

    ctx = _finalize_runtime_context(cfg, overrides, inject=inject)

    vars_in_tpl = _template_vars(tpl)
    if vars_in_tpl:
        missing: list[str] = []
        for k in sorted(vars_in_tpl):
            v = ctx.get(k, None)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                missing.append(k)

        if missing:
            msg = f"resolve_path(): missing {missing} for template={tpl!r}"
            if strict:
                raise KeyError(msg)
            _log.warning("%s -> will render empty", msg)

    rendered = (
        Template(tpl, undefined=StrictUndefined).render(**ctx).strip()
        if strict
        else Template(tpl, undefined=Undefined).render(**ctx).strip()
    )

    rendered = rendered.replace("\\", "/")
    if not rendered.startswith("/"):
        rendered = re.sub(r"/{2,}", "/", rendered)

    # hard guard: never allow unresolved templates to slip into paths_abs
    if strict and "{{" in rendered:
        raise ValueError(f"resolve_path(): unresolved template after render: {rendered!r} from {tpl!r}")

    p = Path(rendered)

    if p.is_absolute():
        out = p
    else:
        rel_norm = _normalize_rel(rendered)

        project_root = _get_project_root(cfg)
        workspace_root = _get_workspace_root(cfg)
        scc_root = _get_scc_root(cfg)

        # One routing rule (_scope) for resolve_path() and to_portable_path().
        scope = _scope(rel_norm)
        if scope == "workspace":
            out = (workspace_root / rel_norm).resolve()
        elif scope == "scc":
            out = (scc_root / rel_norm).resolve()
        else:
            base = project_root or workspace_root
            out = (base / rel_norm).resolve()

    if mkdir:
        try:
            if out.suffix:
                out.parent.mkdir(parents=True, exist_ok=True)
            else:
                out.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    return out


def resolve_paths_dict(
    paths: Dict[str, str | Path],
    cfg: dict | None = None,
    *,
    strict: bool = True,
    mkdir: bool = False,
    inject: bool = True,
    **overrides: Any,
) -> Dict[str, str]:
    """
    Resolve a dict of templates into absolute strings.

    IMPORTANT:
    - inject=True by default because this is typically used to build cfg["paths_abs"].
    """
    out: Dict[str, str] = {}
    for k, v in (paths or {}).items():
        out[k] = str(resolve_path(v, cfg, strict=strict, mkdir=mkdir, inject=inject, **overrides))
    return out


def convert_paths_to_strings(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: convert_paths_to_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_paths_to_strings(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def get_metrics_dir(cfg: dict) -> Path:
    metrics_tpl = (cfg.get("paths") or {}).get("metrics")
    if not metrics_tpl:
        raise KeyError("paths.metrics not found in cfg")

    metrics_dir = resolve_path(metrics_tpl, cfg, strict=True, mkdir=True, inject=True)
    if not metrics_dir.is_dir():
        raise ValueError(f"Metrics directory {metrics_dir} is not a valid directory.")
    return metrics_dir
