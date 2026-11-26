# services/side_channel_core/utils/config_freeze.py
"""
The frozen configuration of a run: the project's config folder, copied once when
the run starts, read by every command of the run instead of the files themselves.

    make experiment   freezes at queue start (experiment_queue.py): every line of the
                      queue, every orchestrator pass, model and test reads the copy
    make orchestrate  run alone, freezes at its own start (Makefile orchestrate)

So params.yaml, the ENV overrides, test_config.yaml, auto_search.yaml, model_config.yaml
and the extra configs may be edited while a run goes on: the edit reaches the next run,
not this one. The copy is config/.frozen/<label>/ under the project's root, named to the
run by TRACEFLOW_FROZEN_CONFIG (relative to the project's root, so the host and the
container, which mounts projects/ at /app/projects, name the same folder) for the project
TRACEFLOW_FROZEN_PROJECT names. config_loader.get_project_config reads through
config_path(); the run's own record stays its sealed params_out snapshot. The copy is
removed when the run that made it ends.

    python -m services.side_channel_core.utils.config_freeze freeze  --project P --label L
    python -m services.side_channel_core.utils.config_freeze release --project P --dir REL

Author: Shraddha Mishra
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

import yaml

ENV_DIR = "TRACEFLOW_FROZEN_CONFIG"
ENV_PROJECT = "TRACEFLOW_FROZEN_PROJECT"
FROZEN = ".frozen"

_SCC = Path(__file__).resolve().parents[1]


def project_root_of(project: str) -> Path:
    """The project's root as master.yaml maps it (project: root, under PROJECTS_ROOT)."""
    master = Path(os.environ.get("SCC_CONFIG_DIR", str(_SCC / "config"))) / "master.yaml"
    projects = Path(os.environ.get("PROJECTS_ROOT", str(_SCC.parents[1] / "projects")))
    data = yaml.safe_load(master.read_text(encoding="utf-8")) or {}
    block = next((p for p in data.get("project") or [] if isinstance(p, dict) and p.get("name") == project), None)
    if block is None:
        raise ValueError(f"No project named '{project}' found in {master}")
    return (projects / block["root"]).resolve()


def freeze(project_root: Path, label: str) -> str:
    """Copy <project_root>/config (without .frozen) to config/.frozen/<label>/; its path relative to the root."""
    rel = Path("config") / FROZEN / label
    shutil.copytree(Path(project_root) / "config", Path(project_root) / rel,
                    ignore=shutil.ignore_patterns(FROZEN))           # an existing copy of the label raises
    return rel.as_posix()


def release(project_root: Path, rel: str) -> None:
    """Remove a copy freeze() made (only a folder directly under config/.frozen/)."""
    root = (Path(project_root) / "config" / FROZEN).resolve()
    target = (Path(project_root) / rel).resolve()
    if target.parent == root:
        shutil.rmtree(target, ignore_errors=True)


def frozen_dir(project_name: str, project_root: Path) -> Optional[Path]:
    """The frozen copy this process reads for the project, or None (it reads the config folder)."""
    rel = os.environ.get(ENV_DIR, "").strip()
    if not rel or os.environ.get(ENV_PROJECT, "").strip() != project_name:
        return None
    folder = (Path(project_root) / rel).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"{ENV_DIR}={rel}: the frozen configuration of this run is missing at {folder}")
    return folder


def config_path(path: Path, project_root: Path, frozen: Optional[Path]) -> Path:
    """A config file's path, in the frozen copy when there is one (a file outside config/ as given)."""
    if frozen is None:
        return Path(path)
    try:
        rel = Path(path).resolve().relative_to((Path(project_root) / "config").resolve())
    except ValueError:
        return Path(path)
    return frozen / rel


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze", help="copy the project's config folder; prints its path relative to the root")
    f.add_argument("--project", required=True)
    f.add_argument("--label", required=True)
    r = sub.add_parser("release", help="remove a frozen copy")
    r.add_argument("--project", required=True)
    r.add_argument("--dir", required=True)
    args = parser.parse_args(argv)
    root = project_root_of(args.project)
    if args.cmd == "freeze":
        print(freeze(root, args.label))
    else:
        release(root, args.dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
