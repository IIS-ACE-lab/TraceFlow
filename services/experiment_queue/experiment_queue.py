# services/experiment_queue/experiment_queue.py
"""
Experiment queue: run the commands lined up under one key, one after another.

    make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig6 [FORCE=1]

Reads projects/<P>/config/experiment_queue.yaml, a mapping of queue name to a
list of `make ...` commands, and runs the commands of QUEUE in the order they
are listed. Each command runs to completion before the next one starts.

Without FORCE the queue resumes: the orchestrator skips every experiment that
is already DONE and continues the FAILED and INTERRUPTED ones, so a queue that
stopped half way picks up at the first unfinished command. With FORCE=1 every
command gets FORCE=1 and its experiments are trained again.

The queue stops at the first command that fails; the commands after it are not
started. The project's config folder is frozen at queue start
(services/side_channel_core/utils/config_freeze.py): every command reads
params.yaml, the ENV overrides, test_config.yaml, auto_search.yaml and
model_config.yaml as they were when the queue started, so an edit made while
the queue runs reaches the next queue, not this one. The copy is removed when
the queue ends.

Every command also gets BATCH=<queue> and one BATCH_ID=<queue>-<UTC start> for
this execution, so the runs of the queue can be found together later
(run_registry.json "batches" / "batch_ids"; make find_model with
`path: batches, contains: fig6`). See services/side_channel_core/utils/batch.py.

Author: Shraddha Mishra
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from services.side_channel_core.utils.seal import load_json
from services.side_channel_core.utils.config_freeze import ENV_DIR, ENV_PROJECT, freeze, project_root_of, release

logger = logging.getLogger("experiment_queue")

# make hands its own command line to child makes through these; a queued
# command must run with the variables written on its line and nothing else.
# IMAGE_TAG is pinned per make call (Makefile): each queued command computes
# its own, so a commit made between two commands reaches the next image build.
_MAKE_ENV = ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES", "IMAGE_TAG")
# The run selectors the Makefile exports (its `export` lines) and declares with
# ?=: an exported value reaches a child make as its default, so `make experiment
# QUEUE=fig3 MODELS=x` (or REPEAT=3, or a MODEL_ID left in the shell) would
# narrow or repeat every queued line. None reaches a queued command; the queue
# passes the ones it owns on the line itself (ENV, PROJECT, BATCH, BATCH_ID, FORCE).
_RUN_VARIABLES = (
    "ENV", "HY", "FORCE", "MLOPS", "PROJECT", "DATASET", "MODE", "ALIAS", "MODELS", "MODEL_ID",
    "BATCH", "BATCH_ID", "GRAPH_ID", "FACE_ID", "RERUN_EXPERIMENT", "REPEAT", "REPRODUCE_MODELS",
    "REPRODUCE_ROOT", "QUEUE", "TEST_MODEL", "TEST_CONFIG", "TEST_EXTERNAL",
)


# A queue name is also the batch label of its runs (utils/batch.py).
_QUEUE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ExperimentQueueError(Exception):
    """The queue file or one of its commands cannot be used."""


def load_queue(config: Path, queue: str) -> List[str]:
    """Return the commands listed under `queue`, as written."""
    config = Path(config)
    if not config.is_file():
        raise ExperimentQueueError(f"missing queue file: {config}")
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ExperimentQueueError(f"{config} must be a mapping of queue name to a list of commands")
    if queue not in data:
        raise ExperimentQueueError(
            f"queue {queue!r} is not in {config}; available: {', '.join(str(k) for k in data)}"
        )
    commands = data[queue]
    if not isinstance(commands, list) or not commands or not all(isinstance(c, str) and c.strip() for c in commands):
        raise ExperimentQueueError(f"queue {queue!r} in {config} must be a non-empty list of commands")
    return [c.strip() for c in commands]


def build_argv(command: str, *, project: str, env: Optional[str], force: bool) -> List[str]:
    """
    Turn one queued line into the argument list to execute.

    Only `make` commands of this project and environment are accepted, and
    FORCE belongs to `make experiment`, not to a queued line.
    """
    argv = shlex.split(command)
    if not argv or argv[0] != "make":
        raise ExperimentQueueError(f"only make commands can be queued: {command!r}")
    variables = dict(a.split("=", 1) for a in argv[1:] if "=" in a)
    for owned in ("FORCE", "BATCH", "BATCH_ID"):
        if owned in variables:
            raise ExperimentQueueError(f"{owned} is set by make experiment, not in the queue: {command!r}")
    if variables.get("PROJECT", project) != project:
        raise ExperimentQueueError(f"command is for PROJECT={variables['PROJECT']}, not {project}: {command!r}")
    if env and variables.get("ENV", env) != env:
        raise ExperimentQueueError(f"command is for ENV={variables['ENV']}, not {env}: {command!r}")
    # The queue's ENV and PROJECT on a line that leaves them out: they no longer
    # reach it through the environment (_RUN_VARIABLES).
    owned = ([f"ENV={env}"] if env and "ENV" not in variables else []) + \
            ([f"PROJECT={project}"] if "PROJECT" not in variables else [])
    return argv + owned + (["FORCE=1"] if force else [])


def batch_id_for(queue: str) -> str:
    """One execution of a queue: <queue>-<UTC start time>, e.g. fig6-20261006T083000Z."""
    return f"{queue}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def batch_experiments(project_root: Path, batch_id: str) -> Dict[str, int]:
    """
    Experiments of one execution of a queue, counted by status from
    run_registry.json (a queued make orchestrate may train several: every
    combination of the sweep lists of params.yaml). {} when there is none yet.
    """
    path = Path(project_root) / "artifacts" / "run_registry.json"
    try:
        registry = load_json(path)  # utils/seal.py: verified, seal key dropped
    except (OSError, ValueError):
        return {}
    counts: Dict[str, int] = {}
    for entry in registry.values() if isinstance(registry, dict) else ():
        if isinstance(entry, dict) and batch_id in (entry.get("batch_ids") or []):
            status = str(entry.get("status") or "unknown").upper()
            counts[status] = counts.get(status, 0) + 1
    return counts


def planned_experiments(project: str, dataset: Optional[str] = None) -> Optional[int]:
    """
    Experiments one queued make orchestrate will plan: the number of
    combinations of the sweep lists of the project's params.yaml, counted by
    the sweep engine itself. None when the engine cannot be imported here.
    The count does not depend on MODEL_ID. It depends on DATASET only through
    the entry's profiling_split.random_state list (one run per split): with
    `dataset` the count is that entry's line's; without it, a line whose
    entry has no such list.
    """
    try:
        from utils.config_loader import get_project_config
        from utils.sweep_engine import count_all_param_combinations
    except ImportError:
        return None
    try:
        cfg = get_project_config(project)
        if dataset and isinstance(cfg.get("data"), dict) and dataset in cfg["data"]:
            cfg["data"] = cfg["data"][dataset]          # the line's entry, as orchestrate selects it
        return int(count_all_param_combinations(cfg))
    except Exception as exc:                     # a damaged config: the orchestrator will say so
        logger.warning("could not count the experiments of %s: %s", project, exc)
        return None


def _experiments(counts: Dict[str, int]) -> str:
    if not counts:
        return "no experiment registered yet"
    total = sum(counts.values())
    by_status = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    return f"{total} experiment(s): {by_status}"


def run_queue(
    config: Path,
    *,
    project: str,
    queue: str,
    env: Optional[str] = None,
    force: bool = False,
    cwd: Optional[Path] = None,
) -> int:
    """Run the queue in order. Returns 0, or the exit code of the command that failed."""
    commands = load_queue(config, queue)
    if not _QUEUE_NAME_RE.match(queue):
        raise ExperimentQueueError(
            f"queue name {queue!r} cannot label a batch: use letters, digits, '_', '.', '-'"
        )
    batch_id = batch_id_for(queue)
    # Every line is checked before the first one runs.
    argvs = [
        build_argv(c, project=project, env=env, force=force) + [f"BATCH={queue}", f"BATCH_ID={batch_id}"]
        for c in commands
    ]

    # Frozen once, here: the count below, every command and the container read this copy.
    frozen_root = project_root_of(project)
    frozen = freeze(frozen_root, batch_id)
    before = {k: os.environ.get(k) for k in (ENV_DIR, ENV_PROJECT)}
    os.environ[ENV_DIR], os.environ[ENV_PROJECT] = frozen, project
    logger.info("queue %s: config frozen at queue start in %s/%s", queue, project, frozen)
    try:
        return _run_commands(argvs, project=project, queue=queue, force=force, cwd=cwd, batch_id=batch_id)
    finally:
        release(frozen_root, frozen)
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_commands(argvs: List[List[str]], *, project: str, queue: str, force: bool,
                  cwd: Optional[Path], batch_id: str) -> int:
    """The queue's commands in order, each with the queue's frozen config; 0 or the failed command's exit code."""
    child_env: Dict[str, Any] = {k: v for k, v in os.environ.items() if k not in _MAKE_ENV + _RUN_VARIABLES}
    project_root = Path(cwd or os.getcwd()) / "projects" / project
    total = len(argvs)
    per_command = planned_experiments(project)
    if per_command is None:
        planned = "experiments counted after each command"
    else:
        planned = (f"{per_command * total} experiment(s) planned by params.yaml, {per_command} per command "
                   "(every combination of its sweep lists); DONE ones are skipped")
    logger.info("queue %s: %d command(s), %s, %s, batch_id %s",
                queue, total, planned, "FORCE=1 (retrain)" if force else "resume", batch_id)

    for index, argv in enumerate(argvs, start=1):
        line = shlex.join(argv)
        span = ("" if per_command is None else
                f" (experiments {(index - 1) * per_command + 1}-{index * per_command} of {per_command * total})")
        logger.info("queue %s command %d/%d%s: %s", queue, index, total, span, line)
        try:
            rc = subprocess.run(argv, cwd=cwd, env=child_env, check=False).returncode
        except KeyboardInterrupt:
            rc = 130
        experiments = _experiments(batch_experiments(project_root, batch_id))
        if rc != 0:
            logger.error(
                "queue %s stopped at command %d/%d (exit code %d): %s", queue, index, total, rc, line,
            )
            logger.error(
                "commands finished: %d, not started: %d; this batch so far: %s (batch_id %s). "
                "Run the same make experiment command without FORCE to resume from this command.",
                index - 1, total - index, experiments, batch_id,
            )
            return rc
        logger.info("queue %s command %d/%d done; this batch so far: %s", queue, index, total, experiments)

    logger.info("queue %s finished: %d/%d command(s), %s (batch_id %s)", queue, total, total,
                experiments, batch_id)
    return 0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the experiments lined up under one queue.")
    parser.add_argument("--project", "-p", required=True)
    parser.add_argument("--queue", "-q", required=True, help="top-level key of the queue file, e.g. fig6")
    parser.add_argument("--env", "-e", default=None)
    parser.add_argument("--force", action="store_true", help="add FORCE=1 to every command")
    parser.add_argument("--config", default=None, help="default: projects/<project>/config/experiment_queue.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = args.config or f"projects/{args.project}/config/experiment_queue.yaml"
    try:
        rc = run_queue(Path(config), project=args.project, queue=args.queue, env=args.env, force=args.force)
    except ExperimentQueueError as exc:
        raise SystemExit(f"experiment_queue: {exc}") from exc
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
