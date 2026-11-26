# services/mlops/utils/runner.py
"""
services/mlops/utils/runner.py

Production-grade AutoMLOps runner (MLOps-side).
----------------------------------------------

This module is the *implementation* behind the stable entrypoint:

    python -m services.mlops.core.auto_mlops_runner_entry --master-yaml ...

Hard rules (per your requirement):
- Do NOT load master.yaml here.
- Do NOT read auto_mlops.yaml directly here.
- ONLY use MLOpsConfig (services/mlops/utils/auto_config.py) to resolve any
  AutoMLOps fields/paths.

Responsibilities:
1) Resolve env + project_name (from CLI override or env)
2) Use MLOpsConfig.get_map("auto_mlops", ...) to:
   - validate + load compiled auto_mlops config (dict)
   - read engine, budget, project list, hypotheses
3) Dispatch to the engine runner (OptunaRunner in v1)
4) Produce explicit logs + deterministic exit codes (no silent success)

Exit codes:
  0 = success (or dry-run success)
  2 = config/validation/usage error
  3 = unexpected/unhandled exception

Author: Shraddha Mishra
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.mlops.utils.auto_config import MLOpsConfig
from services.mlops.utils.logger import mlops_logger


# ----------------------------
# Errors (explicit + readable)
# ----------------------------

class AutoMLOpsConfigError(RuntimeError):
    """Raised when auto_mlops config is missing/invalid/incomplete for execution."""
    pass


class RunnerUsageError(RuntimeError):
    """Raised when the runner is invoked without required context (e.g., project name)."""
    pass


# ----------------------------
# Helpers
# ----------------------------

def _resolve_env(env_override: Optional[str]) -> str:
    env = (env_override or os.environ.get("ENV") or "dev").strip() or "dev"
    os.environ["ENV"] = env  # keep SCC + MLOps consistent
    return env


def _resolve_project_name(project_override: Optional[str]) -> str:
    """
    We are not allowed to read master.yaml here, so project_name must come from:
      - CLI override (preferred)
      - env AUTO_MLOPS_PROJECT / PROJECT_NAME
    """
    if project_override and str(project_override).strip():
        return str(project_override).strip()

    env_name = (os.environ.get("AUTO_MLOPS_PROJECT") or os.environ.get("PROJECT_NAME") or "").strip()
    if env_name:
        return env_name

    raise RunnerUsageError(
        "project_name not provided. Provide --project <name> "
        "or set AUTO_MLOPS_PROJECT / PROJECT_NAME env var."
    )


def _safe_get(d: Any, path: str, default: Any = None) -> Any:
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return cur if cur is not None else default


def _select_project_block(auto_cfg: Dict[str, Any], project_name: str) -> Dict[str, Any]:
    projects = auto_cfg.get("project") or []
    if not isinstance(projects, list):
        raise AutoMLOpsConfigError("auto_mlops: 'project' must be a list")
    for p in projects:
        if isinstance(p, dict) and str(p.get("name", "")).strip() == project_name:
            return p
    raise AutoMLOpsConfigError(f"auto_mlops: no project block named '{project_name}'")


def _extract_hypothesis_ids(project_block: Dict[str, Any]) -> List[str]:
    hyps = project_block.get("hypotheses") or []
    if not isinstance(hyps, list) or not hyps:
        raise AutoMLOpsConfigError("auto_mlops: project.hypotheses must be a non-empty list")

    out: List[str] = []
    for h in hyps:
        if not isinstance(h, dict):
            continue
        hid = str(h.get("hypothesis_id", "")).strip()
        if hid:
            out.append(hid)

    if not out:
        raise AutoMLOpsConfigError("auto_mlops: hypotheses[] entries missing hypothesis_id")
    return out


def _resolve_trials(cli_max_trials: Optional[int], yaml_max_trials: int) -> Tuple[int, str]:
    if cli_max_trials is not None:
        try:
            v = int(cli_max_trials)
        except Exception:
            raise ValueError("max_trials must be an int")
        if v < 0:
            raise ValueError("max_trials must be >= 0")
        return v, "cli"
    return int(yaml_max_trials), "auto_mlops.yaml"


def _set_env_context_for_paths(*, project_name: str, env: str) -> None:
    """
    Ensures your path templates have variables available without passing extra args everywhere.
    These are harmless if already set.
    """
    os.environ.setdefault("ENV", env)
    os.environ.setdefault("AUTO_MLOPS_PROJECT", project_name)
    os.environ.setdefault("PROJECT_NAME", project_name)


# ----------------------------
# Public entry (called by auto_mlops_runner_entry.py)
# ----------------------------

def run_auto_mlops_from_master(
    *,
    master_yaml_path: Path,  # accepted for entrypoint compatibility; intentionally not used here
    project_override: Optional[str] = None,
    env_override: Optional[str] = None,
    dry_run: bool = False,
    resume: bool = False,
    max_trials: Optional[int] = None,
) -> int:
    """
    Executes AutoMLOps using ONLY MLOpsConfig for all auto_mlops-related reads.

    IMPORTANT:
    - master_yaml_path is intentionally NOT parsed here.
      It remains part of the stable entrypoint interface.
    - 'resume' is accepted for interface stability; OptunaRunner handles persistence.
    """
    # Stage-0: resolve core context (must happen before any path templating)
    try:
        env = _resolve_env(env_override)
        project_name = _resolve_project_name(project_override)
        _set_env_context_for_paths(project_name=project_name, env=env)
    except Exception as e:
        # we can still print to terminal safely without any resolved log path
        try:
            mlops_logger("auto_mlops", "paths.exceptions_log_file", f"BOOT_FAIL | {type(e).__name__}: {e}", True, "red")
        except Exception:
            pass
        return 2

    # Stage-1: load + validate auto_mlops via MLOpsConfig (no direct YAML reads)
    try:
        cfg = MLOpsConfig()

        auto_cfg = cfg.get_map(
            "auto_mlops",
            project_name=project_name,
            resolve_as_path=False,
            mkdir=True,
            strict=True,
            extra={"env": env},
        )
        if not isinstance(auto_cfg, dict):
            raise AutoMLOpsConfigError("MLOpsConfig.get_map('auto_mlops') must return a dict")

        # Write an explicit start line
        mlops_logger(
            "auto_mlops",
            "paths.decisions_log_file",
            f"AUTO_MLOPS_START | env={env} project={project_name} dry_run={dry_run} resume={resume} max_trials_cli={max_trials}",
            True,
            "green",
        )

        engine = str(_safe_get(auto_cfg, "search.engine", "")).strip().lower()
        if engine != "optuna":
            raise AutoMLOpsConfigError(f"Only search.engine=optuna is supported in v1 (got {engine!r})")

        yaml_max_trials_any = _safe_get(auto_cfg, "search.budget.max_trials", 0)
        try:
            yaml_max_trials = int(yaml_max_trials_any or 0)
        except Exception:
            raise AutoMLOpsConfigError("search.budget.max_trials must be an int")

        resolved_trials, trials_src = _resolve_trials(max_trials, yaml_max_trials)

        proj_block = _select_project_block(auto_cfg, project_name)
        hypothesis_ids = _extract_hypothesis_ids(proj_block)

        mlops_logger(
            "auto_mlops",
            "paths.decisions_log_file",
            f"CONFIG_OK | engine={engine} max_trials={resolved_trials} (src={trials_src}) hypotheses={hypothesis_ids}",
            True,
            "cyan",
        )

        if resolved_trials == 0:
            mlops_logger(
                "auto_mlops",
                "paths.decisions_log_file",
                "NOOP | max_trials=0 → nothing to run, exiting cleanly",
                True,
                "yellow",
            )
            return 0

        if dry_run:
            mlops_logger(
                "auto_mlops",
                "paths.decisions_log_file",
                "DRY_RUN_OK | validated config only, exiting",
                True,
                "yellow",
            )
            return 0

    except (RunnerUsageError, AutoMLOpsConfigError, ValueError, FileNotFoundError, KeyError, RuntimeError) as e:
        mlops_logger(
            "auto_mlops",
            "paths.exceptions_log_file",
            f"CONFIG_FAIL | {type(e).__name__}: {e}",
            True,
            "red",
        )
        return 2
    except Exception as e:
        mlops_logger(
            "auto_mlops",
            "paths.exceptions_log_file",
            f"UNHANDLED_CONFIG_FAIL | {type(e).__name__}: {e}",
            True,
            "red",
        )
        mlops_logger(
            "auto_mlops",
            "paths.exceptions_log_file",
            traceback.format_exc(),
            False,
            "red",
        )
        return 3

    # Stage-2: dispatch to Optuna runner
    try:
        from services.mlops.core.optuna_runner import OptunaRunner

        runner = OptunaRunner(env=env)

        failures = 0
        for hid in hypothesis_ids:
            mlops_logger(
                "auto_mlops",
                "paths.decisions_log_file",
                f"RUN_HYPOTHESIS_START | hypothesis_id={hid} max_trials={resolved_trials}",
                True,
                "blue",
            )

            try:
                # IMPORTANT: runner.run returns (results, summary) in your current design.
                results, summary = runner.run(
                    project_name=project_name,
                    hypothesis_id=hid,
                    mode="train",
                    max_trials_override=resolved_trials,
                )
                mlops_logger(
                    "auto_mlops",
                    "paths.decisions_log_file",
                    f"RUN_HYPOTHESIS_DONE | hypothesis_id={hid} attempted={summary.attempted} done={summary.done} pruned={summary.pruned} failed={summary.failed} study={summary.study_name}",
                    True,
                    "green",
                )
                # If the hypothesis had hard failures, count them (do not crash whole plugin)
                if summary.failed > 0:
                    failures += 1

            except Exception as e:
                failures += 1
                mlops_logger(
                    "auto_mlops",
                    "paths.exceptions_log_file",
                    f"RUN_HYPOTHESIS_FAIL | hypothesis_id={hid} | {type(e).__name__}: {e}",
                    True,
                    "red",
                )
                mlops_logger(
                    "auto_mlops",
                    "paths.exceptions_log_file",
                    traceback.format_exc(),
                    False,
                    "red",
                )

        if failures > 0:
            mlops_logger(
                "auto_mlops",
                "paths.exceptions_log_file",
                f"AUTO_MLOPS_DONE_WITH_ERRORS | failures={failures} hypotheses={hypothesis_ids}",
                True,
                "yellow",
            )
            # keep deterministic but non-crashing exit behavior:
            return 2

        mlops_logger(
            "auto_mlops",
            "paths.decisions_log_file",
            f"AUTO_MLOPS_DONE | hypotheses={hypothesis_ids}",
            True,
            "green",
        )
        return 0

    except (RunnerUsageError, AutoMLOpsConfigError, ValueError, FileNotFoundError, KeyError, RuntimeError) as e:
        mlops_logger(
            "auto_mlops",
            "paths.exceptions_log_file",
            f"DISPATCH_FAIL | {type(e).__name__}: {e}",
            True,
            "red",
        )
        return 2
    except Exception as e:
        mlops_logger(
            "auto_mlops",
            "paths.exceptions_log_file",
            f"UNHANDLED_DISPATCH_FAIL | {type(e).__name__}: {e}",
            True,
            "red",
        )
        mlops_logger(
            "auto_mlops",
            "paths.exceptions_log_file",
            traceback.format_exc(),
            False,
            "red",
        )
        return 3
