# services/visualization/vscode_monitor/layout.py
"""
What the TraceFlow Monitor needs to know about the repository, as JSON.

    python -m services.visualization.vscode_monitor.layout export
    python -m services.visualization.vscode_monitor.layout scaffold
    python -m services.visualization.vscode_monitor.layout plan --project NAME
    python -m services.visualization.vscode_monitor.layout trials --project NAME
    python -m services.visualization.vscode_monitor.layout snapshots --project NAME
    python -m services.visualization.vscode_monitor.layout snapshots --folder PATH
    python -m services.visualization.vscode_monitor.layout import-describe --project NAME --file PATH
    python -m services.visualization.vscode_monitor.layout import-add --project NAME --file PATH --source S --data D --leakage-model L

``export`` prints one JSON object: every project declared in
services/side_channel_core/config/master.yaml with the paths the panel reads
(artifacts/, run registry, graph jobs), its config files in blueprint order,
its queues (experiment_queue.yaml, each with the ENV its lines name), its graph jobs
(graph_config.yaml) and the model ids and datasets of its params.yaml. A file
that is missing or unreadable is reported under ``errors``; it never stops
the export.

``scaffold`` creates projects/<root>/config/ for every declared project whose
entry is complete (name, root and params given; root one plain folder name) and
copies into it each blueprint file it lacks, of the eleven project templates
(SCAFFOLD_FILES: every key, empty values, commented examples). The blueprint's
older templates (auto_search.yaml, model_config.yaml) are not copied: the
config loader reads a project's model_config.yaml when it exists, so copying
it would change what a project's runs load. A file the project has is never
touched; the blueprint is only read.

``trials`` prints ``{"trials": [{"position", "epochs", "passes"}]}``: every
combination of the project's sweep in its order (the trial number a run gets),
with its epochs and how many times they are trained (n_folds + 1 with
cross-validation: the folds and the refit; else 1). The panel's batch ETA adds
up the trials still to come.

``snapshots`` prints ``{"snapshots": [...]}``: every params_out snapshot of the
project under artifacts/<env>/generated_models/, checked as a re-run checks it,
with its batch labels and registry state (utils/reproduce.py discover); the
panel's Snapshots section lists them and reproduces a batch (make reproduce).

``import-describe`` and ``import-add`` serve the panel's Import a Model (utils/model_import.py): what a
released model file says about itself (read with h5py only) with the project's data entries that fit its
input, and the import added to the project's import_config.yaml (the file copied into the project's
released/ folder when it lies outside the project; import_config.yaml created from the blueprint when
missing).

``plan`` prints ``{"per_command": N}``: the experiments one queued
``make orchestrate`` of the project plans, counted by the sweep engine (the
count ``make experiment`` logs); null when it cannot be counted.

Author: Shraddha Mishra
"""
from __future__ import annotations

import contextlib
import os

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from services.graph_manager import job_paths

REPO = Path(__file__).resolve().parents[3]
MASTER = Path("services/side_channel_core/config/master.yaml")
BLUEPRINT = Path("services/side_channel_core/config/blueprint")
PROJECTS = Path("projects")
CONFIG_ORDER = (
    "params.yaml", "experiment_queue.yaml", "single_runs.yaml", "graph_config.yaml", "model_filter.yaml",
    "model_judge.yaml", "test_config.yaml", "import_config.yaml", "dev.yaml", "staging.yaml", "prod.yaml",
    "auto_search.yaml", "model_config.yaml",
)
# The eleven project templates of the blueprint (make_blueprint.FILES and the hand-written single_runs.yaml and
# import_config.yaml, which master.yaml's extra_configs.import names); only these are scaffolded.
SCAFFOLD_FILES = CONFIG_ORDER[:11]
_MAKE_VAR = re.compile(r"\b(MODEL_ID|DATASET|ENV)=(\S+)")
# RUN SINGLE EXPERIMENT: a line of single_runs.yaml is `make orchestrate` and these variables, each once
_SINGLE_VARS = ("ENV", "PROJECT", "DATASET", "MODEL_ID")


def _load(path: Path, errors: list[str]) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"missing: {path}")
    except Exception as exc:                                  # unreadable or invalid YAML
        errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return None


def blueprint_files(root: Path) -> list[str]:
    """The blueprint's YAML files, in panel order."""
    present = {p.name for p in (root / BLUEPRINT).glob("*.yaml")}
    return [f for f in CONFIG_ORDER if f in present] + sorted(present - set(CONFIG_ORDER))


_FOLDER = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


def declared_projects(root: Path, errors: list[str]) -> list[dict[str, Any]]:
    """name and root folder of every project in master.yaml; complete: name, root and params given,
    root one plain folder name (an entry still being typed is not complete)."""
    master = _load(root / MASTER, errors) or {}
    out = []
    for entry in master.get("project") or []:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        folder = str(entry.get("root") or entry["name"]).strip().strip("/")
        complete = bool(entry.get("root") and entry.get("params")) and bool(_FOLDER.match(folder)) and folder not in (".", "..")
        out.append({"name": str(entry["name"]), "folder": folder, "complete": complete})
    return out


def _queues(data: Any) -> list[dict[str, Any]]:
    queues = []
    for key, commands in (data or {}).items() if isinstance(data, dict) else []:
        lines = [c for c in commands or [] if isinstance(c, str)]
        parsed = [dict(_MAKE_VAR.findall(line), line=line) for line in lines]
        # The ENV the queue runs in is the one its lines name (make experiment refuses a line naming
        # another): env is that ENV when every line agrees, envs every ENV named.
        envs = sorted({c["ENV"] for c in parsed if c.get("ENV")})
        queues.append({
            "key": str(key),
            "commands": parsed,
            "env": envs[0] if len(envs) == 1 and all(c.get("ENV") for c in parsed) else None,
            "envs": envs,
        })
    return queues


def _single_runs(data: Any, project: str) -> list[dict[str, Any]]:
    """single_runs.yaml's runs, in order: each line with its variables, and why it cannot run ('' when it can:
    `make orchestrate` and ENV, PROJECT (this project when left out; the extension checks the one named),
    DATASET, MODEL_ID, each once, nothing else)."""
    lines = (data or {}).get("runs") if isinstance(data, dict) else None
    out = []
    for i, line in enumerate(lines or [] if isinstance(lines, list) else []):
        text = str(line if line is not None else "").strip()
        words = text.split()
        why = ""
        vars_: dict[str, str] = {}
        if words[:2] != ["make", "orchestrate"]:
            why = "not a make orchestrate line"
        for word in words[2:] if not why else []:
            key, eq, value = word.partition("=")
            if not eq or key not in _SINGLE_VARS or key in vars_ or not value:
                why = f"{word}: a line takes ENV, PROJECT, DATASET, MODEL_ID once each"
                break
            vars_[key] = value
        vars_.setdefault("PROJECT", project)
        if not why and not all(vars_.get(k) for k in ("ENV", "DATASET", "MODEL_ID")):
            why = "ENV, DATASET and MODEL_ID are needed"
        out.append({"index": i, "line": text, **vars_, "error": why})
    return out


def _name_map(proj: Path) -> dict[str, str]:
    """artifacts/model_name_map.json (model_hash -> model_name), without its seal; {} when absent or unreadable."""
    try:
        data = json.loads((proj / "artifacts" / "model_name_map.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items() if k != "_tfseal"} if isinstance(data, dict) else {}


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _exported_ids(text: str) -> set[int]:
    """The jobs graph_export.py made from finished runs (Export, Overlay Graph): the header it writes
    above their `- id:` says "Exported from the finished runs"."""
    out = set()
    for block in re.split(r"\n\s*-\s*id:", text):
        heads = re.findall(r"#\s*JOB\s+(\d+):", block)
        if heads and "Exported from the finished runs" in block.split("# JOB")[-1]:
            out.add(int(heads[-1]))
    return out


def _graph_jobs(data: Any, name_map: Optional[dict[str, str]] = None, exported: Optional[set[int]] = None) -> list[dict[str, Any]]:
    jobs = (data or {}).get("jobs") if isinstance(data, dict) else None
    out = []
    for job in jobs or []:
        if isinstance(job, dict) and job.get("id") is not None:
            try:
                job_id = int(job["id"])
            except (TypeError, ValueError):
                continue
            out.append({
                "id": job_id,
                "name": str(job.get("name") or ""),
                "metric": job.get("metric"),
                "source": str(job.get("source") or "attack_metrics"),
                "processes": [str(x) for x in job.get("processes") or [] if isinstance(x, (str, int))],
                "models": [str(m) for m in job.get("models") or [] if isinstance(m, (str, int))],
                # its runs as names (job 0: top_models): one run draws into that run's graphs folder
                "runs": job_paths.run_names(job, name_map),
                "enabled": job.get("enabled", True) is not False,
                # made by graph_export.py from finished runs: its metric is theirs, shown, not edited
                "exported": job_id in (exported or set()),
                # its figure folder, exactly: a folder of the same id with another name (left by a deleted
                # job) is not this job's
                "dir": job_paths.folder_name({**job, "id": job_id}),
            })
    return out


def _graph_facets(data: Any) -> list[dict[str, Any]]:
    facets = (data or {}).get("faceting") if isinstance(data, dict) else None
    out = []
    for facet in facets or []:
        if isinstance(facet, dict) and facet.get("id") is not None:
            try:
                # its output folder, exactly (graphs/facets/facet_NNN_<name>, faceting.py)
                out.append({"id": int(facet["id"]), "name": str(facet.get("name") or ""),
                            "dir": job_paths.facet_folder_name(facet)})
            except (TypeError, ValueError):
                continue
    return out


def _params_lists(data: Any) -> tuple[list[str], list[str], str]:
    if not isinstance(data, dict):
        return [], [], "graphs"
    models = [str(m.get("model_id")) for m in data.get("model") or [] if isinstance(m, dict) and m.get("model_id")]
    datasets = [str(d.get("name")) for d in data.get("data") or [] if isinstance(d, dict) and d.get("name")]
    graphs = str(((data.get("paths") or {}).get("graphs")) or "graphs")
    return models, datasets, graphs


def _judge_output(proj: Path, data: Any) -> str:
    """model_judge.yaml's selected_model file: <output_dir>/<outputs.selected_model>."""
    judge = (data or {}).get("model_judge") if isinstance(data, dict) else None
    judge = judge if isinstance(judge, dict) else {}
    outputs = judge.get("outputs") if isinstance(judge.get("outputs"), dict) else {}
    return str(proj / str(judge.get("output_dir") or "model_selection") / str(outputs.get("selected_model") or "selected_model.yaml"))


def project_layout(root: Path, name: str, folder: str) -> dict[str, Any]:
    """One project's paths and settings. No ENV here: the panel follows the ENV of the commands
    (the running run's, a queue's lines', the latest run's), so it reads artifacts/<env>/ itself."""
    errors: list[str] = []
    proj = root / PROJECTS / folder
    config = proj / "config"
    files = [f for f in blueprint_files(root) if (config / f).exists()]
    files += sorted(p.name for p in config.glob("*.yaml") if p.name not in files) if config.is_dir() else []
    queue_file = config / "experiment_queue.yaml"
    graph_file = config / "graph_config.yaml"
    params = _load(config / "params.yaml", errors)
    models, datasets, graphs = _params_lists(params)
    return {
        "name": name,
        "folder": folder,
        "path": str(proj),
        "artifacts": str(proj / "artifacts"),
        "config_dir": str(config),
        "configs": [{"file": f, "name": f.rsplit(".", 1)[0], "path": str(config / f)} for f in files],
        "registry": str(proj / "artifacts" / "run_registry.json"),
        "graph_jobs_dir": str(proj / graphs / "jobs"),
        "filtered_models": str(proj / "artifacts" / "filtered_models.yaml"),
        "test_registry": str(proj / "artifacts" / "test_registry.json"),
        "model_selection": str(proj / "model_selection"),
        "judge_output": _judge_output(proj, _load(config / "model_judge.yaml", errors) if (config / "model_judge.yaml").exists() else None),
        "queues": _queues(_load(queue_file, errors)) if queue_file.exists() else [],
        "single_runs": _single_runs(_load(config / "single_runs.yaml", errors), name) if (config / "single_runs.yaml").exists() else [],
        "single_runs_file": str(config / "single_runs.yaml"),
        "graph_jobs": _graph_jobs(_load(graph_file, errors), _name_map(proj), _exported_ids(_text(graph_file)))
        if graph_file.exists() else [],
        "graph_facets": _graph_facets(_load(graph_file, errors)) if graph_file.exists() else [],
        "graphs_dir": str(proj / graphs),
        # a graph job of one run draws into this folder of the run (services/graph_manager/job_paths.py)
        "model_graphs": str(((params or {}).get("paths") or {}).get("model_graphs") or job_paths.DEFAULT_MODEL_GRAPHS)
        if isinstance(params, dict) else job_paths.DEFAULT_MODEL_GRAPHS,
        "model_ids": models,
        "datasets": datasets,
        "errors": errors,
    }


def orphan_projects(root: Path, declared: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The project folders under projects/ that master.yaml does not declare (its entry removed, or never
    written): each with what is still there, so the panel can say "master.yaml integration missing" and
    whether the project root, its config/ and its artifacts/ are present or deleted."""
    folders = {p["folder"] for p in declared}
    base = root / PROJECTS
    out = []
    for path in sorted(base.iterdir()) if base.is_dir() else []:
        if not path.is_dir() or path.name.startswith((".", "_")) or path.name in folders:
            continue
        if not (path / "config").is_dir() and not (path / "artifacts").is_dir():
            continue                                         # not a project folder (notes, data, ...)
        out.append({"name": path.name, "root": str(path), "present": True,
                    "config": (path / "config").is_dir(), "artifacts": (path / "artifacts").is_dir()})
    return out


def export(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    projects = []
    declared = declared_projects(root, errors)
    for p in declared:
        if p["complete"]:
            projects.append(p)
        else:
            errors.append(f"master.yaml project {p['name']!r}: not shown until it has root (one folder name) and params")
    return {
        "root": str(root),
        "master": str(root / MASTER),
        "blueprint": str(root / BLUEPRINT),
        "projects": [project_layout(root, p["name"], p["folder"]) for p in projects],
        "orphans": orphan_projects(root, declared),
        "timezone": _timezone(),
        "errors": errors,
    }


PROJECT_README = "PROJECT_README.md"             # the project's README template: shipped as projects/<P>/README.md


def scaffold(root: Path) -> list[str]:
    """Copy each blueprint file a declared project lacks, and the project README template as the project's
    README.md (owner 2026-10-23: the figure index that organises a manual, per-queue workflow); returns the
    files created. complete_projects does the keys (layout.py scaffold prints both)."""
    errors: list[str] = []
    created = []
    for project in declared_projects(root, errors):
        if not project["complete"]:                   # e.g. a half-typed entry saved by autosave
            continue
        config = root / PROJECTS / project["folder"] / "config"
        config.mkdir(parents=True, exist_ok=True)
        for name in [f for f in blueprint_files(root) if f in SCAFFOLD_FILES]:
            target = config / name
            if not target.exists():
                shutil.copyfile(root / BLUEPRINT / name, target)
                created.append(str(target.relative_to(root)))
        readme = config.parent / "README.md"
        if not readme.exists() and (root / BLUEPRINT / PROJECT_README).is_file():
            shutil.copyfile(root / BLUEPRINT / PROJECT_README, readme)
            created.append(str(readme.relative_to(root)))
    return created


def complete_projects(root: Path) -> dict[str, dict[str, list[str]]]:
    """Every declared project's config files given the blueprint keys they lack (a project from an older
    template, or a blueprint grown since): make_blueprint.complete_project, the hash-safe way (empty keys
    only; the rest is listed as left, for the panel's Load Blueprint). Returns, per
    projects/<P>/config/<file>, {added, left}."""
    from services.visualization.vscode_monitor.make_blueprint import complete_project

    out = {}
    errors: list[str] = []
    for project in declared_projects(root, errors):
        if not project["complete"]:
            continue
        config = root / PROJECTS / project["folder"] / "config"
        if not config.is_dir():
            continue
        try:
            report = complete_project(config, hash_safe=True)
        except Exception as exc:  # noqa: BLE001 - one project's odd file never stops the others
            out[str(config.relative_to(root))] = {"added": [], "left": [], "error": f"{type(exc).__name__}: {exc}"}
            continue
        for name, r in report.items():
            out[str((config / name).relative_to(root))] = r
    return out


def trials(project: str) -> dict[str, Any]:
    """Every combination of the project's sweep, in order, with its epochs and passes (needs side_channel_core on the path)."""
    from utils.config_loader import get_project_config
    from utils.sweep_engine import generate_param_combinations

    cfg = get_project_config(project)
    # sca.attack_byte left empty follows the data entry of each run (dataset_types.resolve_attack_byte, at
    # the run's start); the file's empty list would be refused by the sweep, and the trials need only each
    # combination's epochs and passes, so one placeholder byte stands in (as count_all_param_combinations does).
    sca = cfg.get("sca")
    if isinstance(sca, dict) and sca.get("attack_byte") in (None, []):
        cfg = {**cfg, "sca": {**sca, "attack_byte": [0]}}
    out = []
    for i, combo in enumerate(generate_param_combinations(cfg), start=1):
        cv = combo.get("cross_validation") or {}
        folds = int(cv.get("n_folds") or 0) if isinstance(cv, dict) and cv.get("enabled") is True else 0
        # the refit is one more pass, unless cross_validation.refit.enabled is false (the folds only)
        refit = 0 if folds and isinstance(cv.get("refit"), dict) and cv["refit"].get("enabled") is False else 1
        epochs = (combo.get("trainer") or {}).get("epochs")
        out.append({"position": i, "epochs": int(epochs) if isinstance(epochs, (int, float)) else None,
                    "passes": folds + refit})
    return {"trials": out}


def _timezone() -> str:
    """master.yaml's timezone as an IANA name (the system's own resolver), else UTC."""
    try:
        from services.side_channel_core.utils.timestamps import timezone_name

        return timezone_name()
    except Exception:                                   # unreadable master.yaml, unknown city: still export
        return "UTC"


def check_master(root: Path) -> dict[str, Any]:
    """After a Master Control write: every project of master.yaml loads as a run loads it
    (config_loader.get_project_config, the schema included), and the timezone resolves; an error per
    project ('' when it loads), the zone or its error."""
    from services.side_channel_core.utils.timestamps import resolve_timezone
    from utils.config_loader import get_project_config

    errors: list[str] = []
    master = _load(root / MASTER, errors) or {}
    out: dict[str, Any] = {"projects": {}, "errors": errors}
    for entry in master.get("project") or [] if isinstance(master, dict) else []:
        name = str((entry or {}).get("name") or "") if isinstance(entry, dict) else ""
        if not name:
            errors.append("master.yaml: a project entry has no name")
            continue
        try:
            get_project_config(name, validate_data=False)
            out["projects"][name] = ""
        except Exception as exc:                        # noqa: BLE001 - shown per project, as the run would stop
            out["projects"][name] = f"{type(exc).__name__}: {exc}"
    zone = (master.get("timezone") or {}) if isinstance(master, dict) else {}
    try:
        out["timezone"] = resolve_timezone(zone.get("current_location"), zone.get("current_country")) if zone else ""
    except Exception as exc:                            # noqa: BLE001 - shown, the image clock reads it
        out["timezone_error"] = str(exc)
    default = master.get("default_project") if isinstance(master, dict) else None
    if default and default not in out["projects"]:
        errors.append(f"default_project {default!r} is not a project of master.yaml")
    return out


def project_template(root: Path, name: str) -> dict[str, Any]:
    """Add a Project: blueprint/master_project.yaml under the name given (root <name>/, no description)."""
    entry = yaml.safe_load((root / BLUEPRINT / "master_project.yaml").read_text(encoding="utf-8")) or {}
    return {**entry, "name": name, "description": "", "root": f"{name}/"}


def plan(project: str, dataset: str | None = None) -> dict[str, Any]:
    """Experiments one `make orchestrate` of the project plans (needs services/side_channel_core{,/src} on the path);
    with the line's DATASET, its entry's profiling_split.random_state list (one run per split) counts too."""
    from services.experiment_queue.experiment_queue import planned_experiments

    return {"per_command": planned_experiments(project, dataset)}


def snapshots(project: str, root: Path) -> dict[str, Any]:
    """The project's params_out snapshots and their state (needs side_channel_core on the path)."""
    from utils.reproduce import discover, resolve_project, scan_pattern

    project_root = resolve_project(project)                 # master.yaml's root for this name
    return {"project": project, "root": str(project_root), "pattern": scan_pattern(project_root, root),
            "snapshots": discover(project_root, project_name=project)}


def _at(node: Any, parts: list[str]) -> list[Any]:
    """Every value at a dotted path: a list takes an index (data.0.name), or any of its items (data.name)."""
    if not parts:
        return [node]
    head, rest = parts[0], parts[1:]
    if isinstance(node, dict):
        return _at(node[head], rest) if head in node else []
    if isinstance(node, list):
        if head.isdigit():
            return _at(node[int(head)], rest) if int(head) < len(node) else []
        return [v for item in node for v in _at(item, parts)]
    return []


def _same(found: Any, wanted: str) -> bool:
    """A snapshot value equals a typed one: numbers by value (800 = 800.0, 1e-3 = 0.001), true/false/null
    as YAML spells them, any other text ignoring case."""
    want = wanted.strip()
    if isinstance(found, bool) or found is None:
        return want.lower() == {True: "true", False: "false", None: "null"}[found] or (found is None and want.lower() in ("none", "~"))
    if isinstance(found, (int, float)):
        try:
            return float(want) == float(found)
        except ValueError:
            return False
    return isinstance(found, str) and found.lower() == want.lower()


def param_filter(where: str, files: list[str]) -> dict[str, Any]:
    """Housekeeping's PARAM filter over run snapshots: `key.path=value[,value…]`, conditions joined by `;`
    (all must hold), or the dotted form key.path.value (the longest key path a snapshot has wins).
    Returns {match: {file: bool}, resolved: ["trainer.epochs=800"], error}."""
    docs: dict[str, Any] = {}
    for f in files:
        try:
            docs[f] = yaml.safe_load(Path(f).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            docs[f] = None                                    # no snapshot: never matches
    read = [d for d in docs.values() if d is not None]
    conds, errors = [], []
    for raw in (c.strip() for c in where.split(";")):
        if not raw:
            continue
        if "=" in raw:
            key, value = (x.strip() for x in raw.split("=", 1))
            parts = [x for x in key.split(".") if x]
            if parts[:1] == ["params"] and not any(isinstance(d, dict) and "params" in d for d in read):
                parts = parts[1:]                             # params.trainer.epochs: params is the snapshot
            if not parts or not value:
                errors.append(f"{raw}: write key.path=value, e.g. trainer.epochs=800")
                continue
        else:
            parts_all = [x for x in raw.split(".") if x]
            if parts_all[:1] == ["params"] and not any(isinstance(d, dict) and "params" in d for d in read):
                parts_all = parts_all[1:]
            found = next(((parts_all[:k], ".".join(parts_all[k:])) for k in range(len(parts_all) - 1, 0, -1)
                          if any(_at(d, parts_all[:k]) for d in read)), None)
            if not found:
                errors.append(f"{raw}: no snapshot has such a key; write key.path=value, e.g. trainer.epochs=800")
                continue
            parts, value = found
        if not any(_at(d, parts) for d in read):
            errors.append(f"{'.'.join(parts)}: no snapshot has this key ({len(read)} read)")
            continue
        conds.append((parts, [v.strip() for v in value.split(",") if v.strip()]))
    if errors or not conds:
        return {"match": {}, "resolved": [], "error": "PARAM " + ("; ".join(errors) or "is empty")}
    match = {
        f: d is not None and all(any(_same(v, w) for v in _at(d, parts) for w in values) for parts, values in conds)
        for f, d in docs.items()
    }
    return {"match": match, "resolved": [f"{'.'.join(p)}={','.join(v)}" for p, v in conds], "error": ""}


def external(project: str, model: str) -> dict[str, Any]:
    """One model's external test settings from the project's test_config.yaml (test_request.py), or why not."""
    from services.side_channel_core.src.sca.test_request import TestRequestError, external_settings, test_config_path

    try:
        path = test_config_path(project)
        return {"settings": external_settings(path, model), "config": str(path), "error": ""}
    except TestRequestError as exc:
        return {"error": str(exc)}


# ── Import a Model (utils/model_import.py) ──────────────────────────────────
RELEASED = "released"          # where Import a Model copies a file from outside the project (the run sees projects/ only)


def _project_folder(root: Path, project: str) -> Path:
    errors: list[str] = []
    for p in declared_projects(root, errors):
        if p["name"] == project and p["complete"]:
            return root / PROJECTS / p["folder"]
    raise ValueError(f"no complete project named {project!r} in master.yaml")


def _target_lengths(params: Any) -> dict[str, Optional[int]]:
    """Each data entry of a params.yaml and the trace length its target_range gives (None: not stated)."""
    out: dict[str, Optional[int]] = {}
    for entry in (params or {}).get("data") or [] if isinstance(params, dict) else []:
        if isinstance(entry, dict) and entry.get("name"):
            window = entry.get("target_range")
            out[str(entry["name"])] = (int(window[1]) - int(window[0])
                                       if isinstance(window, (list, tuple)) and len(window) >= 2 else None)
    return out


def import_describe(root: Path, project: str, file: str) -> dict[str, Any]:
    """Import a Model: what a released file says about itself (model_import.describe: h5py only, nothing
    loaded) and the project's data entries, those whose traces fit its input first (`fitting`)."""
    from utils.model_import import describe

    folder = _project_folder(root, project)
    out = describe(Path(file))
    lengths = _target_lengths(_load(folder / "config" / "params.yaml", []))
    width = (out.get("input") or [None, None])[1]
    out["datasets"] = list(lengths)
    out["fitting"] = [name for name, n in lengths.items() if width is not None and n == width]
    out["inside"] = Path(file).resolve().is_relative_to(folder.resolve())
    return out


def import_add(root: Path, project: str, file: str, *, source: str, data: str, leakage_model: str,
               name: str = "", fit_on: str = "") -> dict[str, Any]:
    """Import a Model: one import added to the project's import_config.yaml (created from the blueprint when
    missing; comments kept). A file outside the project is copied into its released/ folder first (the run's
    container sees projects/ only); one already there is never overwritten."""
    from services.visualization.vscode_monitor import yamledit
    from utils.model_import import import_config_path

    folder = _project_folder(root, project)
    src = Path(file).resolve()
    if not src.is_file():
        raise ValueError(f"{file} is not a file")
    if src.is_relative_to(folder.resolve()):
        rel = src.relative_to(folder.resolve())
        copied = None
    else:
        target = folder / RELEASED / src.name
        if target.exists():
            raise ValueError(f"{target} exists already: rename the file, or import that copy")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        rel, copied = target.relative_to(folder), str(target)
    config = import_config_path(project)
    if not config.exists():
        config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / BLUEPRINT / "import_config.yaml", config)
    entry: dict[str, Any] = {**({"name": name} if name else {}), "file": str(rel), "source": source, "data": data,
                             "sca": {"leakage_model": leakage_model}, **({"fit_on": fit_on} if fit_on else {})}
    text = config.read_text(encoding="utf-8")
    current = (yaml.safe_load(text) or {}).get("model_import") or {}
    if not isinstance(current.get("models"), list):
        text = yamledit.edit_text(text, "set", ["model_import", "models"], "[]")
    text = yamledit.edit_text(text, "add", ["model_import", "models"], json.dumps(entry))
    config.write_text(text, encoding="utf-8")
    return {"config": str(config), "file": str(rel), "copied": copied, "entry": entry}


@contextlib.contextmanager
def _stdout_to_stderr():
    """Everything written to stdout (fd 1) goes to stderr meanwhile: the project code logs INFO lines
    to stdout (a console that captured sys.stdout at import), and the panel reads stdout as JSON."""
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def folder_snapshots(folder: str) -> dict[str, Any]:
    """The snapshots anywhere under a chosen folder (the CUSTOM panel; utils/reproduce.py discover_folder)."""
    from utils.reproduce import discover_folder

    path = Path(folder).expanduser().resolve()
    return {"folder": str(path), "origin": path.name, "snapshots": discover_folder(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TraceFlow Monitor layout")
    parser.add_argument("command", choices=("export", "scaffold", "plan", "trials", "snapshots", "place", "params", "external",
                                            "check-master", "project-template", "import-describe", "import-add"))
    parser.add_argument("--root", default=str(REPO))
    parser.add_argument("--project", help="plan, trials, snapshots: the project name in master.yaml")
    parser.add_argument("--dataset", help="plan: the line's DATASET (its profiling_split.random_state list counts)")
    parser.add_argument("--folder", help="snapshots: any folder holding <model_name>_params.yaml (CUSTOM)")
    parser.add_argument("--file", help="place: a snapshot to copy to its params_out path inside --into")
    parser.add_argument("--into", help="place: the chosen folder, root of the reproduced runs")
    parser.add_argument("--from", dest="source", help="place: the folder the snapshot was scanned in, when --into is another")
    parser.add_argument("--where", help="params: Housekeeping's PARAM filter, key.path=value[;…]")
    parser.add_argument("--snapshot", action="append", default=[], help="params: a run's snapshot (repeatable)")
    parser.add_argument("--model", help="external: the model whose external test settings to show")
    parser.add_argument("--source", help="import-add: the release's citation, or repository URL @ commit")
    parser.add_argument("--data", help="import-add: the data entry the released model's traces come from")
    parser.add_argument("--leakage-model", help="import-add: the labels it predicts (ID, HW, MSB)")
    parser.add_argument("--name", default="", help="import-add: the run's name (empty: a generated one)")
    parser.add_argument("--fit-on", default="", help="import-add: a fitted scaler's fit set (training | profiling)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.command == "export":
        print(json.dumps(export(root)))
    elif args.command == "scaffold":
        print(json.dumps({"created": scaffold(root), "completed": complete_projects(root)}))
    elif args.command == "check-master":
        with _stdout_to_stderr():
            result = check_master(root)
        print(json.dumps(result))
    elif args.command == "project-template":
        print(json.dumps(project_template(root, str(args.project or ""))))
    elif args.command == "trials":
        print(json.dumps(trials(args.project)))
    elif args.command == "place":
        from utils.reproduce import place

        with _stdout_to_stderr():
            try:
                result = {"placed": place(Path(args.file), Path(args.into) if args.into else None,
                                          Path(args.source) if args.source else None)}
            except ValueError as exc:
                result = {"error": str(exc)}
        print(json.dumps(result))
    elif args.command == "external":
        with _stdout_to_stderr():
            result = external(args.project, args.model)
        print(json.dumps(result))
    elif args.command in ("import-describe", "import-add"):
        with _stdout_to_stderr():
            try:
                result = (import_describe(root, str(args.project or ""), str(args.file or "")) if args.command == "import-describe"
                          else import_add(root, str(args.project or ""), str(args.file or ""), source=str(args.source or ""),
                                          data=str(args.data or ""), leakage_model=str(args.leakage_model or ""),
                                          name=args.name, fit_on=args.fit_on))
            except Exception as exc:                         # the panel shows why
                result = {"error": str(exc)}
        print(json.dumps(result))
    elif args.command == "params":
        print(json.dumps(param_filter(args.where or "", args.snapshot)))
    elif args.command == "snapshots":
        with _stdout_to_stderr():
            result = folder_snapshots(args.folder) if args.folder else snapshots(args.project, root)
        print(json.dumps(result))
    else:
        print(json.dumps(plan(args.project, args.dataset)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
