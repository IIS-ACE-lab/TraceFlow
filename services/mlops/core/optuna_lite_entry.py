# services/mlops/core/optuna_lite_entry.py
"""
optuna_lite_entry.py
====================

Stable entrypoint for lean auto-search runs using OptunaLiteRunner.

Responsibilities
----------------
- Parse CLI arguments
- Resolve project name from CLI or master.yaml default_project
- Run OptunaLiteRunner
- Print a concise summary
- Return process exit code

This entrypoint is intentionally thin and stable so shell/orchestrator
contracts do not need to change when internal runner logic evolves.

Usage
-----
python -m services.mlops.core.optuna_lite_entry --project DLSCA_ASCAD
python -m services.mlops.core.optuna_lite_entry --project DLSCA_ASCAD --env dev
python -m services.mlops.core.optuna_lite_entry --project DLSCA_ASCAD --mode train --max-trials 10

Notes
-----
- This is for the lightweight auto_search.yaml path only.
- Full hypothesis-driven AutoMLOps should continue using auto_mlops_runner_entry.py.

Author: Shraddha Mishra
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

from services.mlops.core.optuna_lite import OptunaLiteRunner


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _load_master_yaml(master_yaml: str) -> dict:
    p = Path(master_yaml).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"master.yaml not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"master.yaml must parse to a dict: {p}")
    return data


def _resolve_project_name(cli_project: Optional[str], master_yaml: str) -> str:
    if cli_project and str(cli_project).strip():
        return str(cli_project).strip()

    master = _load_master_yaml(master_yaml)

    default_project = master.get("default_project")
    if isinstance(default_project, str) and default_project.strip():
        return default_project.strip()

    projects = master.get("project") or []
    if isinstance(projects, dict):
        projects = [projects]

    if isinstance(projects, list):
        for pr in projects:
            if isinstance(pr, dict):
                name = pr.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()

    raise ValueError(
        "Could not resolve project name. Pass --project or set default_project in master.yaml."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lean Optuna entrypoint for auto_search.yaml-driven project search."
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project name as declared in services/side_channel_core/config/master.yaml",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Runtime environment override (defaults to ENV env var or 'dev')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "orchestrate"],
        help="Execution mode forwarded to SCC bridge. Default: train",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional override for search.budget.max_trials",
    )
    parser.add_argument(
        "--master-yaml",
        type=str,
        default="/app/services/side_channel_core/config/master.yaml",
        help="Path to master.yaml. Default: /app/services/side_channel_core/config/master.yaml",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        env = (args.env or os.environ.get("ENV") or "dev").strip()
        os.environ["ENV"] = env

        project_name = _resolve_project_name(args.project, args.master_yaml)

        runner = OptunaLiteRunner(env=env)
        results, summary = runner.run(
            project_name=project_name,
            mode=args.mode,
            max_trials_override=args.max_trials,
        )

        print(
            (
                f"[optuna_lite_entry] OK "
                f"project={project_name} env={env} mode={args.mode} "
                f"study={summary.study_name} "
                f"attempted={summary.attempted} done={summary.done} failed={summary.failed}"
            ),
            flush=True,
        )

        # Non-zero only if nothing succeeded and at least one failed.
        if summary.done == 0 and summary.failed > 0:
            return 1
        return 0

    except KeyboardInterrupt:
        _eprint("[optuna_lite_entry] interrupted by user")
        return 130
    except Exception as e:
        _eprint(f"[optuna_lite_entry] fatal: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
