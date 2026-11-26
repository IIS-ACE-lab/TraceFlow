# services/mlops/utils/logger.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from rich.console import Console

from services.mlops.utils.auto_config import MLOpsConfig
from services.side_channel_core.utils.timestamps import now_str


def mlops_logger(
    cfg: str,
    key_map: str,
    msg: str,
    mirror_to_terminal: bool = True,
    color: str = "cyan",
) -> Path:
    """
    Minimal MLOps logger (single-call).

    Writes a timestamped line to a file path resolved via:
        MLOpsConfig().get_map(cfg, key_map=..., project_name=..., resolve_as_path=True, extra=...)

    Parameters
    ----------
    cfg : str
        Config key registered in config_tracking_registry.yaml.
        Example: "auto_mlops"

    key_map : str
        Dot-map path to the log file template inside cfg.
        Example: "paths.decisions_log_file" | "paths.pruning_log_file" | "paths.exceptions_log_file"

    msg : str
        The message line to log.

    mirror_to_terminal : bool
        If True, also print to terminal via Rich.

    color : str
        Rich color name used for terminal print.
        Allowed colors (safe defaults):
            - "cyan"
            - "green"
            - "yellow"
            - "red"
            - "magenta"
            - "blue"
            - "white"

    Environment variables used (so we don't require extra args)
    ----------------------------------------------------------
    - ENV (default: "dev")
    - AUTO_MLOPS_PROJECT or PROJECT_NAME (required)
    - AUTO_MLOPS_MODEL_NAME or MODEL_NAME (required)
    - SOTA_TRIAL or AUTO_MLOPS_SOTA_TRIAL (optional)

    Returns
    -------
    Path
        The resolved log file path that was written to.
    """

    # ---- terminal styling ----
    color_map: Dict[str, str] = {
        "cyan": "cyan",
        "green": "green",
        "yellow": "yellow",
        "red": "red",
        "magenta": "magenta",
        "blue": "blue",
        "white": "white",
    }
    c = color_map.get(str(color).strip().lower(), "cyan")

    # ---- infer context (no extra args allowed) ----
    env = (os.environ.get("ENV") or "dev").strip() or "dev"
    project_name = (os.environ.get("AUTO_MLOPS_PROJECT") or os.environ.get("PROJECT_NAME") or "").strip()
    model_name = (os.environ.get("AUTO_MLOPS_MODEL_NAME") or os.environ.get("MODEL_NAME") or "").strip()
    sota_trial = (os.environ.get("AUTO_MLOPS_SOTA_TRIAL") or os.environ.get("SOTA_TRIAL") or "").strip()

    if not project_name:
        raise RuntimeError(
            "[mlops_logger] missing project name. Set AUTO_MLOPS_PROJECT or PROJECT_NAME."
        )
    if not model_name:
        raise RuntimeError(
            "[mlops_logger] missing model name. Set AUTO_MLOPS_MODEL_NAME or MODEL_NAME."
        )

    # ---- resolve path via MLOpsConfig (authoritative) ----
    cfg_obj = MLOpsConfig()
    extra = {"env": env, "model_name": model_name}
    if sota_trial:
        extra["sota_trial"] = sota_trial

    # NOTE: strict is handled internally by MLOpsConfig/resolve_path.
    # We do not expose strict_paths anymore.
    log_path = cfg_obj.get_map(
        cfg,
        key_map=key_map,
        project_name=project_name,
        resolve_as_path=True,
        mkdir=True,
        strict=True,
        extra=extra,
        touch_file=True,
    )
    if not isinstance(log_path, Path):
        log_path = Path(str(log_path))

    # ---- write line ----
    ts = now_str()
    line = f"{ts} | {msg}".rstrip() + "\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)

    # ---- terminal mirror ----
    if mirror_to_terminal:
        console = Console(force_terminal=True, width=200, color_system="truecolor")
        console.print(f"[{c}]{ts} | {msg}[/{c}]")

    return log_path
