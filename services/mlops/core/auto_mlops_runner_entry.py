# services/mlops/core/auto_mlops_runner_entry.py

"""  
The stable CLI/module entrypoint that your orchestrator executes:
python -m services.mlops.core.auto_mlops_runner_entry --master-yaml ...


So the architecture should stay:

- auto_mlops_runner_entry.py = entrypoint + arg parsing + exit codes

- services/mlops/utils/runner.py = implementation (run_auto_mlops_from_master())

That separation is good engineering and keeps your orchestrator contract stable even if you refactor internals.

Author: Shraddha Mishra
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master-yaml", required=True)
    p.add_argument("--project", default=None)
    p.add_argument("--env", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-trials", type=int, default=None)
    return p.parse_args(argv)



def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    master = Path(args.master_yaml).resolve()
    if not master.exists():
        print(f"[auto_mlops_runner_entry] FATAL: master.yaml not found: {master}", file=sys.stderr)
        return 2

    # IMPORTANT: use services.* namespace
    from services.mlops.utils.runner import run_auto_mlops_from_master

    return int(run_auto_mlops_from_master(
        master_yaml_path=master,
        project_override=args.project,
        env_override=args.env,
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
        max_trials=args.max_trials,
    ))

if __name__ == "__main__":
    raise SystemExit(main())
