# services/side_channel_core/utils/reproduce.py
"""
Reproduce runs from their params_out snapshots alone.

A run's snapshot is the complete resolved config it was trained with, at
``paths.params_out`` (``artifacts/{{env}}/generated_models/{{model_name}}/config/
{{model_name}}_params.yaml``), sealed by ``snapshot_sha256`` and carrying its
``model_hash``, ``model_name`` and the ``batches`` it was made for. Shipping an
experiment means shipping those files: the receiver copies them to the same
place in their project (the run folder with only ``config/`` inside) and runs

    make snapshots ENV=dev PROJECT=DLSCA_ASCAD                  # what is on disk, and its state
    make reproduce ENV=dev PROJECT=DLSCA_ASCAD BATCH=fig26      # register them, then train

The project is found by its name in master.yaml (project: - name / root): the
snapshots are looked for at <projects>/<root>/ + paths.params_out of its
params.yaml, e.g. projects/DLSCA_ASCAD/artifacts/dev/generated_models/
cake_bacon/config/cake_bacon_params.yaml.

``list`` finds every snapshot under ``artifacts/<env>/generated_models/`` and
checks it the way a re-run would: it parses, is complete, matches its digest,
sits in the folder of its own model_name, and its config still hashes to its
model_hash (make_experiment_hash with this project's live project block).

``register`` takes the snapshots of one ENV that match ``--batch`` (the
snapshot's ``batches``) and/or ``--model`` (names or hashes), refuses any that
fails a check or whose model_name belongs to another hash here, and enters the
unregistered ones in run_registry.json (status NEW, ``imported_snapshot: true``,
the batch label) and model_name_map.json under their own name and hash. It
prints the names still to train (also imported runs left unfinished: resume),
one comma-separated line last, for ``make rerun RERUN_EXPERIMENT=``. The
orchestrator then trains each from its snapshot (cli_helpers.
resolve_registered_run_cfg: an imported entry is resolved from its snapshot
like an established one), so name, hash and every setting are the sender's.

``register`` from the command line (make reproduce) first refuses while an
experiment trains on this machine (running_experiments): let the current run
finish. One training stack per machine, and make orchestrate starts with make
stack-down, which would stop it.

Author: Shraddha Mishra
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml
from services.side_channel_core.utils.seal import load_json

# Running experiments, as the TraceFlow panel sees them (vscode_monitor/commands.js BUSY_PATTERN): a make
# that trains, or the trainer itself (the container's process, visible from the host).
BUSY_PATTERN = re.compile(
    r"^((\S*/)?make( \S+)* (orchestrate|experiment|test|reproduce|import)( |$)"
    r"|(\S*/)?python[0-9.]* -m core\.cli (orchestrate|test|train)-cmd( |$))"
)
LIVE_BEAT_S = 60        # a heartbeat saying running and younger than this: a run is training

DEFAULT_PARAMS_OUT = "artifacts/{{env}}/generated_models/{{model_name}}/config/{{model_name}}_params.yaml"
ENVS = ("dev", "staging", "prod")
IMPORTED_KEY = "imported_snapshot"


def _norm(value: Any) -> str:
    return str(value or "").strip()


def params_out_template(project_root: Path) -> str:
    """paths.params_out of the project's params.yaml (the snapshot location), else the default."""
    try:
        data = yaml.safe_load((Path(project_root) / "config" / "params.yaml").read_text(encoding="utf-8")) or {}
        template = _norm((data.get("paths") or {}).get("params_out"))
    except (OSError, yaml.YAMLError, AttributeError):
        template = ""
    return template or DEFAULT_PARAMS_OUT


def snapshot_path(project_root: Path, env: str, model_name: str, template: Optional[str] = None) -> Path:
    t = template or params_out_template(project_root)
    rel = re.sub(r"\{\{\s*env\s*\}\}", env, t)
    rel = re.sub(r"\{\{\s*model_name\s*\}\}", model_name, rel)
    return Path(project_root) / rel


def resolve_project(name: str) -> Path:
    """The project's root folder, as master.yaml registers it (config_loader: projects/<root>)."""
    from utils.config_loader import get_project_config

    return Path(get_project_config(project_name=name, validate_data=False)["project"]["root"]).resolve()


def scan_pattern(project_root: Path, repo_root: Optional[Path] = None) -> str:
    """Where the snapshots are looked for: the project root joined with paths.params_out."""
    root = Path(project_root).resolve()
    try:
        root = root.relative_to(Path(repo_root).resolve()) if repo_root else root
    except ValueError:
        pass
    t = params_out_template(project_root)
    t = re.sub(r"\{\{\s*env\s*\}\}", "<ENV>", t)
    t = re.sub(r"\{\{\s*model_name\s*\}\}", "<model_name>", t)
    return f"{root.as_posix()}/{t}"


def _live_project_block(project_root: Path, project_name: Optional[str] = None) -> dict:
    """The project block a re-run uses (config_loader), for make_experiment_hash."""
    from utils.config_loader import get_project_config

    name = project_name or Path(project_root).name
    return copy.deepcopy(get_project_config(project_name=name, validate_data=False)["project"])


def _check(snapshot_file: Path, env: Optional[str], project_block: Optional[dict], expected_name: Optional[str] = None) -> dict:
    """One snapshot and what a re-run would say about it.

    env: the ENV folder it was found under (a project scan), or None to take the snapshot's own (a
    custom folder). expected_name: the name it must carry (its run folder; a custom file's name).
    """
    from utils.data_reproducibility import SNAPSHOT_DIGEST_KEY, read_params_snapshot
    from utils.model_naming import make_experiment_hash

    folder = expected_name or snapshot_file.parent.parent.name
    row: dict[str, Any] = {"env": env or "", "folder": folder, "path": str(snapshot_file), "problem": ""}
    try:
        snap = read_params_snapshot(snapshot_file)
    except ValueError as exc:
        row.update(model_name=folder, model_hash="", verified=False, problem=str(exc).split(". ")[0])
        return row
    trainer = snap.get("trainer") or {}
    data = snap.get("data")
    data = data[0] if isinstance(data, list) and data else data
    sca = snap.get("sca") or {}
    row.update(
        model_name=_norm(snap.get("model_name")), model_hash=_norm(snap.get("model_hash")),
        model_id=_norm(snap.get("model_id")), dataset=_norm((data or {}).get("name")) if isinstance(data, dict) else "",
        leakage_model=_norm(sca.get("leakage_model")), epochs=trainer.get("epochs"), batch_size=trainer.get("batch_size"),
        batches=[_norm(b) for b in (snap.get("batches") or []) if _norm(b)],
        snapshot_env=_norm(snap.get("env")),
        snapshot_project=_norm((snap.get("project") or {}).get("name")),
    )
    if env is None:                                     # a custom folder: the snapshot says its ENV
        row["env"] = row["snapshot_env"]
    if SNAPSHOT_DIGEST_KEY not in snap:
        row["problem"] = "written before 2026-09-29 (no snapshot_sha256): its hash cannot be reproduced"
    elif row["model_name"] != folder:
        row["problem"] = (f"model_name {row['model_name']!r} is not its folder {folder!r}" if expected_name is None
                          else f"model_name {row['model_name']!r} does not match its file name ({folder}_params.yaml)")
    elif row["env"] not in ENVS:
        row["problem"] = f"its env {row['env']!r} is not dev, staging or prod"
    elif env is not None and row["snapshot_env"] and row["snapshot_env"] != env:
        row["problem"] = f"recorded env {row['snapshot_env']!r}, found under artifacts/{env}/"
    elif project_block is not None:
        body = {k: v for k, v in snap.items() if k != SNAPSHOT_DIGEST_KEY}
        body["project"] = copy.deepcopy(project_block)
        try:
            recomputed = make_experiment_hash(body)
        except Exception as exc:                            # an incomplete or foreign config
            row["problem"] = f"cannot be hashed here: {exc}"
        else:
            if recomputed != row["model_hash"]:
                row["problem"] = (f"hashes to {recomputed} here, not to its model_hash {row['model_hash']} "
                                  "(edited, or this project's master.yaml scripts differ)")
    row["verified"] = not row["problem"]
    return row


def discover(project_root: Path, env: Optional[str] = None, *, verify_hash: bool = True,
             project_name: Optional[str] = None, config_root: Optional[Path] = None) -> list[dict]:
    """Every snapshot under artifacts/<env>/generated_models/, checked, with its registry state.

    project_root: the outputs' root (the project's own, or a chosen folder: REPRODUCE_ROOT);
    config_root: where its params.yaml (paths.params_out) is read, by default project_root.
    """
    from utils.model_naming import _load_mapping
    from utils.sweep_engine import load_registry

    project_root = Path(project_root)
    template = params_out_template(Path(config_root) if config_root else project_root)
    registry = load_registry(project_root, heal=False)
    name_map = _load_mapping(project_root)
    project_block = _live_project_block(project_root, project_name) if verify_hash else None
    rows: list[dict] = []
    for e in ([env] if env else ENVS):
        root = project_root / "artifacts" / e / "generated_models"
        if not root.is_dir():
            continue
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            snap = snapshot_path(project_root, e, run_dir.name, template)
            if not snap.is_file():
                continue
            row = _check(snap, e, project_block)
            row["project"] = project_name or project_root.name
            row["origin"] = row["project"]
            _registry_state(row, registry, name_map)
            rows.append(row)
    return rows


def _registry_state(row: dict, registry: dict, name_map: dict) -> None:
    """The row's state in the project it would run in: registered, status, conflicts."""
    entry = registry.get(row["model_hash"]) if row["model_hash"] else None
    row["registered"] = isinstance(entry, dict)
    row["status"] = _norm(entry.get("status")) if isinstance(entry, dict) else ""
    row["imported"] = bool(isinstance(entry, dict) and entry.get(IMPORTED_KEY))
    if isinstance(entry, dict) and _norm(entry.get("model_name")) not in ("", row["model_name"]):
        row["problem"] = row["problem"] or f"hash registered here as {entry.get('model_name')!r}"
    owner = next((h for h, n in name_map.items() if _norm(n) == row["model_name"] and h != row["model_hash"]), None)
    if owner:
        row["problem"] = row["problem"] or f"model_name already belongs to hash {owner} here"
    row["verified"] = not row["problem"]
    if isinstance(entry, dict) and not row["batches"]:
        row["batches"] = [_norm(b) for b in (entry.get("batches") or []) if _norm(b)]


def check_output_root(folder: Path) -> Path:
    """A chosen folder that is the root of a run's outputs (every paths: template under it). Refused
    when it holds the repository: the code and data paths would then hash relative to it."""
    folder = Path(folder).expanduser().resolve()
    repo = Path(__file__).resolve().parents[3]
    if repo == folder or folder in repo.parents:
        raise ValueError(f"{folder} holds the repository {repo}: choose a folder outside it (or inside projects/)")
    return folder


def discover_folder(folder: Path) -> list[dict]:
    """Every snapshot (<model_name>_params.yaml) anywhere under a chosen folder (the CUSTOM panel).

    The folder is the root of the reproduced runs: every paths: template of the snapshot applies inside
    it (artifacts/<ENV>/generated_models/<model_name>/..., the registries under its artifacts/). Each
    snapshot is checked against the project it names (registered in master.yaml: its scripts and config
    files), its ENV is the snapshot's, `target` is its params_out path inside the folder and the
    registry state is the folder's own."""
    from utils.model_naming import _load_mapping
    from utils.sweep_engine import load_registry

    folder = check_output_root(folder)
    if not folder.is_dir():
        raise ValueError(f"not a folder: {folder}")
    registry, name_map = load_registry(folder, heal=False), _load_mapping(folder)
    projects: dict[str, Any] = {}
    rows: list[dict] = []
    for f in sorted(folder.rglob("*_params.yaml")):
        if f.name.startswith(".") or not f.is_file():
            continue
        try:
            head = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            name = _norm((head.get("project") or {}).get("name")) if isinstance(head, dict) else ""
        except (OSError, yaml.YAMLError):
            name = ""
        if name and name not in projects:
            try:
                root = resolve_project(name)
                projects[name] = (root, _live_project_block(root, name), params_out_template(root))
            except Exception as exc:                    # not registered here
                projects[name] = exc
        ctx = projects.get(name)
        row = _check(f, None, ctx[1] if isinstance(ctx, tuple) else None, expected_name=f.name[: -len("_params.yaml")])
        row["origin"] = folder.name
        row["project"] = name
        row["target"] = ""
        row["placed"] = False
        if not name:
            row["problem"] = row["problem"] or "names no project (project.name)"
        elif not isinstance(ctx, tuple):
            row["problem"] = row["problem"] or f"project {name!r} is not registered in master.yaml here"
        else:
            target = snapshot_path(folder, row["env"], row["model_name"], ctx[2]) if row["env"] and row["model_name"] else None
            row["target"] = str(target) if target else ""
            if target is not None and target.is_file():
                row["placed"] = target.resolve() == f.resolve() or target.read_bytes() == f.read_bytes()
                if not row["placed"]:
                    row["problem"] = row["problem"] or f"a different snapshot already sits at {target}"
            _registry_state(row, registry, name_map)
        row["verified"] = not row["problem"]
        rows.append(row)
    return rows


def place(snapshot_file: Path, root: Optional[Path] = None, source: Optional[Path] = None) -> dict:
    """Copy a snapshot to its params_out path inside `root` (the destination folder; by default the
    file's own folder), refused when it fails a check or a different snapshot is there. `source` is
    the folder it was scanned in (default `root`): a snapshot of a project, or of another folder, is
    checked there and copied to a new destination; the row then carries the destination's registry
    state."""
    import shutil

    snapshot_file = Path(snapshot_file).expanduser().resolve()
    root = Path(root).expanduser().resolve() if root else snapshot_file.parent
    source = Path(source).expanduser().resolve() if source else root
    rows = [r for r in discover_folder(source) if Path(r["path"]).resolve() == snapshot_file]
    if not rows:
        raise ValueError(f"{snapshot_file} is not a snapshot (<model_name>_params.yaml)")
    row = rows[0]
    if source != root and row["verified"]:
        from utils.model_naming import _load_mapping
        from utils.sweep_engine import load_registry

        check_output_root(root)
        target = snapshot_path(root, row["env"], row["model_name"], params_out_template(resolve_project(row["project"])))
        row["target"] = str(target)
        row["placed"] = target.is_file() and target.read_bytes() == snapshot_file.read_bytes()
        if target.is_file() and not row["placed"]:
            row["problem"] = f"a different snapshot already sits at {target}"
        row.pop("status", None)
        _registry_state(row, load_registry(root, heal=False), _load_mapping(root))
        row["verified"] = not row["problem"]
    if not row["verified"]:
        raise ValueError(f"refused: {row['model_name'] or row['folder']}: {row['problem']}")
    target = Path(row["target"])
    if not row["placed"]:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot_file, target)
        row["placed"] = True
    return row


def select(rows: Iterable[dict], *, env: str, batches: Iterable[str] = (), models: Iterable[str] = ()) -> list[dict]:
    """The snapshots of `env` in scope: in any of `batches` and/or named in `models` (both given: both)."""
    batches = {b for b in (_norm(x) for x in batches) if b}
    models = {m for m in (_norm(x) for x in models) if m}
    out = []
    for r in rows:
        if r["env"] != env:
            continue
        if batches and not (set(r.get("batches") or []) & batches):
            continue
        if models and r["model_name"] not in models and r["model_hash"] not in models:
            continue
        out.append(r)
    return out


def register(project_root: Path, *, env: str, batch: str = "", models: Iterable[str] = (),
             project_name: Optional[str] = None, force: bool = False, config_root: Optional[Path] = None) -> list[str]:
    """Enter the in-scope, unregistered snapshots in the registries; the names still to train."""
    from utils.batch import add_batch, utc_stamp, _check as check_label
    from utils.model_naming import _load_mapping, _save_mapping
    from utils.sweep_engine import load_registry, save_registry

    project_root = Path(project_root)
    if env not in ENVS:
        raise ValueError(f"env must be one of {ENVS}, got {env!r}")
    models = [m for m in (_norm(x) for x in models) if m]
    if not batch and not models:
        raise ValueError("give BATCH (the snapshots' batch label) and/or MODEL (names or hashes)")
    if batch:
        check_label(batch, "BATCH")
    scope = select(discover(project_root, env, project_name=project_name, config_root=config_root), env=env,
                   batches=[batch] if batch else [], models=models)
    if not scope:
        raise ValueError(f"no snapshot of env={env} matches BATCH={batch or '-'} MODEL={','.join(models) or '-'} "
                         f"under {project_root / 'artifacts' / env / 'generated_models'}")
    bad = [r for r in scope if not r["verified"]]
    if bad:
        raise ValueError("refused, nothing registered: " + "; ".join(f"{r['model_name'] or r['folder']}: {r['problem']}" for r in bad))

    registry = load_registry(project_root)
    name_map = _load_mapping(project_root)
    label = (batch, f"{batch}-import-{utc_stamp()}") if batch else None
    project = project_root.name
    for r in scope:
        if r["registered"]:
            continue
        entry = {
            "model_hash": r["model_hash"], "model_name": r["model_name"], "model_id": r.get("model_id", ""),
            "env": env, "status": "NEW", "leakage_model": r.get("leakage_model", ""),
            "path": f"{project}/artifacts/{env}/generated_models/{r['model_name']}",
            "selection_source": "snapshot_import", IMPORTED_KEY: True,
        }
        for b in r.get("batches") or []:
            add_batch(entry, b, f"{b}-import-{utc_stamp()}")
        registry[r["model_hash"]] = entry
        name_map[r["model_hash"]] = r["model_name"]
    if label:
        for r in scope:
            if isinstance(registry.get(r["model_hash"]), dict):
                add_batch(registry[r["model_hash"]], *label)
    save_registry(registry, project_root)
    _save_mapping(project_root, name_map)
    # force: DONE runs too (train them again from their snapshots; their results are replaced)
    return [r["model_name"] for r in scope if force or _norm(registry[r["model_hash"]].get("status")).lower() != "done"]


def _ancestors(pid: int) -> set[int]:
    """pid and every parent above it (from /proc): make reproduce's own chain is not "another run"."""
    out: set[int] = set()
    while pid > 1 and pid not in out:
        out.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return out


def _processes() -> list[tuple[int, str]]:
    procs = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            except OSError:
                continue
            if cmd:
                procs.append((int(d.name), cmd))
    return procs


def running_experiments(repo_root: Path, *, processes: Optional[list[tuple[int, str]]] = None,
                        own: Optional[set[int]] = None, now_ns: Optional[int] = None,
                        extra_roots: Iterable[Path] = ()) -> list[str]:
    """What is training on this machine now: matching processes outside our own chain, and every
    project's heartbeat that says running and beat less than LIVE_BEAT_S ago."""
    import os
    import time

    own = own if own is not None else _ancestors(os.getpid())
    found = []
    for pid, cmd in (processes if processes is not None else _processes()):
        if pid not in own and BUSY_PATTERN.search(cmd):
            found.append(cmd)
    now_ns = now_ns if now_ns is not None else time.time_ns()
    beats = sorted(Path(repo_root).glob("projects/*/artifacts/*/heartbeat.json"))
    for root in extra_roots:                            # a chosen output root (REPRODUCE_ROOT) beats there
        beats += sorted(Path(root).glob("artifacts/*/heartbeat.json"))
    for hb_file in dict.fromkeys(beats):
        try:
            hb = load_json(hb_file)  # utils/seal.py
            age = (now_ns - int(hb["written_ns"])) / 1e9
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if hb.get("status") == "running" and age < LIVE_BEAT_S:
            project, env = hb_file.parent.parent.parent.name, hb_file.parent.name
            found.append(f"{project} ENV={env}: {hb.get('model_name') or hb.get('model_hash')} training "
                         f"({hb.get('phase') or '?'}, last beat {max(0, round(age))} s ago)")
    return list(dict.fromkeys(found))


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("no snapshot found")
        return
    for r in rows:
        state = r["status"] or "not registered"
        flag = "ok" if r["verified"] else f"REFUSED: {r['problem']}"
        print(f"{r['env']:8s} {r['model_name']:24s} {r['model_hash'] or '-':9s} {r.get('model_id', ''):22s} "
              f"{r.get('dataset', ''):18s} {','.join(r.get('batches') or []) or '-':10s} {state:12s} {flag}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="the snapshots on disk and their state")
    ls.add_argument("--project", default="", help="project name registered in master.yaml (its root is scanned)")
    ls.add_argument("--project-root", default="", help="or the project folder itself")
    ls.add_argument("--folder", default="", help="or any folder holding <model_name>_params.yaml snapshots (CUSTOM)")
    ls.add_argument("--env", default="")
    ls.add_argument("--json", action="store_true")
    rg = sub.add_parser("register", help="register the snapshots of a batch / of named runs; print the names to train")
    rg.add_argument("--project", default="", help="project name registered in master.yaml")
    rg.add_argument("--project-root", default="", help="or the project folder itself")
    rg.add_argument("--env", required=True)
    rg.add_argument("--batch", default="")
    rg.add_argument("--model", default="", help="comma-separated model names or hashes")
    rg.add_argument("--force", action="store_true", help="DONE runs too: train them again from their snapshots")
    rg.add_argument("--output-root", default="", help="a chosen folder holding the runs' outputs (REPRODUCE_ROOT)")
    pl = sub.add_parser("place", help="copy a snapshot to its params_out path inside a chosen folder")
    pl.add_argument("--file", required=True)
    pl.add_argument("--root", default="", help="the chosen folder (default: the file's folder)")
    args = parser.parse_args(argv)

    def quiet(fn):
        """fn() with stdout sent to stderr (the config loader logs to stdout); stdout keeps the JSON."""
        sys.stdout.flush()
        saved = os.dup(1)
        os.dup2(2, 1)
        try:
            return fn()
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)

    if args.cmd == "place":
        try:
            row = quiet(lambda: place(Path(args.file), Path(args.root) if _norm(args.root) else None))
        except ValueError as exc:
            raise SystemExit(f"[reproduce] {exc}")
        print(json.dumps(row))
        return 0
    if args.cmd == "list" and _norm(args.folder):
        try:
            rows = quiet(lambda: discover_folder(Path(args.folder))) if args.json else discover_folder(Path(args.folder))
        except ValueError as exc:
            raise SystemExit(f"[reproduce] {exc}")
        if args.json:
            print(json.dumps({"folder": str(Path(args.folder).expanduser().resolve()), "snapshots": rows}))
        else:
            _print_table(rows)
        return 0
    name = _norm(args.project)
    if not name and not _norm(args.project_root):
        raise SystemExit("[reproduce] give --project (the name in master.yaml) or --project-root")

    def scan():
        root = resolve_project(name) if name else Path(args.project_root).resolve()
        if not (root / "config").is_dir():
            raise SystemExit(f"[reproduce] not a project root: {root}")
        return root, (discover(root, _norm(args.env) or None, project_name=name or None) if args.cmd == "list" else None)

    if args.cmd == "list":
        if args.json:                       # stdout carries the JSON only: the config loader logs to stdout
            sys.stdout.flush()
            saved = os.dup(1)
            os.dup2(2, 1)
            try:
                root, rows = scan()
            finally:
                sys.stdout.flush()
                os.dup2(saved, 1)
                os.close(saved)
            print(json.dumps({"project": name or root.name, "root": str(root), "pattern": scan_pattern(root), "snapshots": rows}))
        else:
            root, rows = scan()
            print(f"scanned: {scan_pattern(root, Path.cwd())}")
            _print_table(rows)
        return 0
    root, _ = scan()
    config_root = None
    if _norm(getattr(args, "output_root", "")):                 # outputs in a chosen folder, configs from the project
        try:
            config_root, root = root, check_output_root(Path(args.output_root))
        except ValueError as exc:
            raise SystemExit(f"[reproduce] {exc}")
    # One experiment at a time: make reproduce goes on with make orchestrate, which starts with make
    # stack-down and would stop the run that is training now (one training stack per machine).
    busy = running_experiments(Path(__file__).resolve().parents[3], extra_roots=[root])
    if busy:
        raise SystemExit(
            "[reproduce] Let the current run finish: an experiment is running on this machine:\n  "
            + "\n  ".join(busy)
            + "\nThis machine trains one experiment at a time (one training stack; make orchestrate starts "
            "with make stack-down, which would stop it). Nothing was registered; start again when it is done."
        )
    try:
        names = register(root, env=_norm(args.env), batch=_norm(args.batch), models=args.model.split(","),
                         project_name=name or None, force=args.force, config_root=config_root)
    except ValueError as exc:
        raise SystemExit(f"[reproduce] {exc}")
    if not names:
        print("[reproduce] every run in scope is DONE: nothing to train")
        print("")
        return 0
    print(f"[reproduce] to train: {', '.join(names)}")
    print(",".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
