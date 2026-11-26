# services/mlops/utils/path_resolver.py
from __future__ import annotations

import os
import re
import json

from pathlib import Path
from typing import Any, Dict, Optional

class PathResolutionError(RuntimeError):
    """
    Raised when a template contains unresolved placeholders after expansion.
    Treat as CONFIG BUG (fail fast).
    """
    def __init__(self, *, template: str, expanded: str, missing: list[str], context: Dict[str, Any]) -> None:
        msg = (
            "\n\nCRITICAL: Unresolved placeholders in path template \n"
            f"Template: {template}\n"
            f"Expanded:  {expanded}\n"
            f"Missing:   {missing}\n"
            f"Context:   {json.dumps(context, indent=2, ensure_ascii=False)}\n"
            "System HALTED. Fix configuration or runtime context.\n"
        )
        super().__init__(msg)
        self.template = template
        self.expanded = expanded
        self.missing = missing
        self.context = context


# Support BOTH syntaxes:
#   - Jinja style: {{ env }}
#   - Bash style:  ${ENV}
_JINJA_RE = re.compile(r"\{\{\s*([^}\s]+)\s*\}\}")
_DOLLAR_RE = re.compile(r"\$\{([^}]+)\}")


_FILE_EXTS = (".json", ".jsonl", ".yaml", ".yml", ".csv", ".txt", ".log", ".pth", ".pt", ".ckpt", ".hmac")


def _repo_root_from_here(start: Path) -> Path:
    """
    MLSCA-safe repo root detection:
    - Prefer env var SCA_ROOT if set
    - Else walk up until 'services/' exists
    - Else fallback to cwd
    """
    env = os.environ.get("SCA_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "services").exists():
            return parent.resolve()
    return Path.cwd().resolve()


REPO_ROOT = _repo_root_from_here(Path(__file__))


def build_context(cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Minimal MLSCA context for rendering auto_mlops paths.

    Required keys typically:
      - env
      - model_name
      - sota_trial

    We also include common fallbacks.
    """
    ctx: Dict[str, str] = {}

    # From cfg
    for k in ("env", "model_name"):
        if k in cfg and cfg[k] is not None:
            ctx[k] = str(cfg[k])

    # Optional nested
    hyp = cfg.get("auto_mlops", {}).get("hypothesis_id")
    if hyp:
        ctx["hypothesis_id"] = str(hyp)

    # Environment variables as fallback (non-authoritative)
    ctx.setdefault("env", os.environ.get("ENV", "dev"))
    ctx.setdefault("model_name", os.environ.get("MODEL_NAME", ctx.get("model_name", "unnamed_model")))

    # Extra overrides (authoritative for render)
    if extra:
        for k, v in extra.items():
            if v is None:
                continue
            ctx[k] = str(v)

    return ctx


def _expand_jinja(template: str, ctx: Dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        return ctx.get(key, f"{{{{{key}}}}}")  # leave unresolved for strict check
    return _JINJA_RE.sub(repl, template)


def _expand_dollar(template: str, ctx: Dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        return ctx.get(key, f"${{{key}}}")
    return _DOLLAR_RE.sub(repl, template)


def _find_unresolved(expanded: str) -> list[str]:
    missing = set()
    for m in _JINJA_RE.finditer(expanded):
        missing.add(m.group(1))
    for m in _DOLLAR_RE.finditer(expanded):
        missing.add(m.group(1))
    return sorted(missing)


def resolve_path(
    template: str,
    cfg: Dict[str, Any],
    *,
    extra: Optional[Dict[str, Any]] = None,
    mkdir: bool = True,
    strict: bool = True,
    anchor: Optional[Path] = None,
) -> Path:
    """
    Resolve a path template into an absolute path under repo root (or custom anchor).

    - Supports {{key}} and ${key}
    - strict=True: unresolved placeholders -> raise PathResolutionError
    - mkdir=True: creates parent dir for files, dir for directories
    """
    ctx = build_context(cfg, extra=extra)

    expanded = _expand_jinja(template, ctx)
    expanded = _expand_dollar(expanded, ctx)

    missing = _find_unresolved(expanded)
    if missing and strict:
        raise PathResolutionError(template=template, expanded=expanded, missing=missing, context=ctx)

    base = (anchor or REPO_ROOT).resolve()
    p = (base / expanded).expanduser().resolve()

    if mkdir:
        if p.suffix.lower() in _FILE_EXTS:
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)

    return p
