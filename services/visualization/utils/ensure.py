# services/visualization/utils/ensure.py
""" 
Visualization Config Ensurer
- Ensures visualization config in run_cfg is always up-to-date with master.yaml
- Authoritative source of truth is master.yaml's visualization block for the project
- Used by train/test/rerun paths to refresh visualization settings like ClearML scope/profile

Author: Shraddha Mishra
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _resolve_master_yaml() -> Path:
    env_master = os.getenv("SCC_MASTER_YAML")
    if env_master:
        p = Path(env_master)
        if p.exists():
            return p

    repo_root = Path(__file__).resolve().parents[3]
    master_path = repo_root / "services" / "side_channel_core" / "config" / "master.yaml"
    if not master_path.exists():
        raise FileNotFoundError(f"master.yaml not found at expected path: {master_path}")
    return master_path


def _load_master_cfg() -> dict:
    master_path = _resolve_master_yaml()
    return yaml.safe_load(master_path.read_text()) or {}


def _ensure_visualization_cfg(run_cfg: Dict[str, Any], project_name: str) -> Dict[str, Any]:
    """
    Always refresh visualization block from authoritative master.yaml.

    This intentionally overwrites stale run_cfg['visualization'] so scope/profile
    changes in master.yaml are respected by train/test/rerun paths.
    """
    master = _load_master_cfg()

    projects = master.get("project") or []
    if isinstance(projects, dict):
        projects = [projects]

    for proj in projects:
        if not isinstance(proj, dict):
            continue
        if str(proj.get("name", "")).strip() == str(project_name).strip():
            viz = proj.get("visualization") or {}
            run_cfg["visualization"] = viz
            return run_cfg

    # fallback: keep existing if project not found
    return run_cfg
