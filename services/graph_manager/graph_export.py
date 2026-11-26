# services/graph_manager/graph_export.py
"""
From finished runs to graph jobs: the step between a sweep and its figure.

    python -m services.graph_manager.graph_export choices --project P --env E --models a,b
    python -m services.graph_manager.graph_export export  --project P --env E --models a,b,c \\
        --source cross_validation --process validation --metric mean_attack_rank_curve
    python -m services.graph_manager.graph_export prune   --project P --models a,b
    python -m services.graph_manager.graph_export delete-job --project P --id 35
    python -m services.graph_manager.graph_export renumber --project P

choices   what each run recorded (graph_manager/metric_catalog.py): the menus of the
          panel's Done list.
export    the runs as one graph job, written the way a job is written by hand:
          models:
            # ASCAD_CNNbase on ascad, 100 epochs, batch 200, learning rate 1e-05 (batch fig10)
            - fish_shadow        # Adam
          legend.labels the same values, source/metric/processes as chosen, a header comment
          above `- id:` saying what the figure is. The shared settings and what differs come
          from the runs' snapshots (their `sampled` sweep choices, dataset and model id; a
          changed optimizer `type` stands for the arguments that change with it). A job of the
          same source, metric and processes over the same batch and dataset gets the new
          models added; otherwise a new job is appended, styled like the latest job of the
          same metric (else job 0), with the next id.
remove    the panel's Delete from Graph Config: one run leaves the jobs drawing a source, metric
          and process; a job left with no model is deleted and the later ids move down.
prune     a deleted run leaves every job: models, legend.labels, job 0's top_models
          (utils/delete_database.py calls it).
close-gap  the same for a job already removed by hand (the panel offers it when a job id
          disappears from graph_config.yaml).
delete-job  removes a job and moves every later job down by one id: its figure folder
          (graphs/jobs/job_NNN_<name>), the faceting addresses into it and the `JOB N` /
          `GRAPH_ID=N` lines of its header follow. A figure's sources.json is not rewritten
          (it records how that figure was drawn); files elsewhere that still name an old id
          (README, paper, queue) are listed, not edited.
renumber  closes every gap the same way (job 0 stays 0).
new-job   a job at the next id named --title, no models yet (enabled: false until they are listed).
new-facet a facet at the next id named --title, no panels yet (enabled: false until they are listed).
study-plan / study  the finished runs of a queue's study (the queues named <study>_..., all DONE)
          as graph jobs: its axes, the legend axes as a figure's lines, each value of the other axes
          a figure, one job per metric ticked (the panel's QUEUES → Overlay Graph).
export-facet  drawn graph files (addresses under graphs/) as one new facet: panels numbered in
          the order given, a row for up to three, a grid of rows of three beyond, a header
          comment naming each panel's job and the FACE_ID make line.
delete-facet / facet-close-gap  a facet removed (here or by hand): the later facets move down
          one id with their FACET / FACE_ID header lines and output folders; its output goes to
          graphs/facets/_deleted/. A deleted job's panels leave every facet; a facet left with
          fewer than two panels is removed the same way.

graph_config.yaml is edited one entry at a time with the panel's own editor
(services/visualization/vscode_monitor/yamledit.py): every other byte, comment and
blank line stays, and each write is parsed back and must be exactly the intended change.
Each command prints one JSON object.

Author: Shraddha Mishra
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import yaml

from services.visualization.vscode_monitor import yamledit, yamlspan
from services.graph_manager.graph_job_validator import GROUP_BANDS

# Snapshot parts that identify a run, not an experiment choice.
_IDENTITY = {"frozen_identity", "model_hash", "model_name", "paths", "paths_abs", "run", "runtime",
             "snapshot_sha256", "batches", "batch_ids", "project", "env", "sampled"}
# A job's processes when the panel names none: the figure's population for each source.
DEFAULT_PROCESS = {"cross_validation": "validation", "attack_metrics": "test", "key_recovery": "test"}
_RULE = "  # " + "=" * 64


class ExportError(ValueError):
    """A request the graph config cannot take as asked."""


# ── project files ─────────────────────────────────────────────────────────────

class Project:
    def __init__(self, name: str, env: str = "dev") -> None:
        from services.side_channel_core.utils.config_loader import get_project_config

        self.name, self.env = name, env
        self.cfg = get_project_config(name, validate_data=False)
        self.root = Path(self.cfg["project"]["root"]).resolve()
        rel = ((self.cfg.get("visualization") or {}).get("graph_manager") or {}).get("config", "config/graph_config.yaml")
        self.graph_config = (self.root / rel).resolve()
        graphs = Path(str((self.cfg.get("paths") or {}).get("graphs", "graphs")))
        self.graphs = graphs if graphs.is_absolute() else self.root / graphs

    def registry(self) -> dict[str, Any]:
        from services.side_channel_core.utils.sweep_engine import load_registry

        return load_registry(self.root, heal=False)

    def run_dir(self, model_name: str) -> Path:
        return self.root / "artifacts" / self.env / "generated_models" / model_name

    def snapshot(self, model_name: str) -> dict[str, Any]:
        found = sorted((self.run_dir(model_name) / "config").glob("*_params.yaml"))
        if not found:
            raise ExportError(f"{model_name}: no config snapshot in {self.run_dir(model_name) / 'config'}")
        return yaml.safe_load(found[0].read_text(encoding="utf-8")) or {}


def _read(path: Path) -> str:
    return yamledit.read(path)


def _write(path: Path, text: str) -> None:
    yamledit._atomic_write(path, text)


def _jobs(text: str) -> list[dict[str, Any]]:
    return list((yamlspan.load(text) or {}).get("jobs") or [])


def _int_id(entry: Mapping[str, Any]) -> Optional[int]:
    """A job's or facet's id as an integer; None for an entry without one (a template's empty job, a
    job written by hand with no id yet): such an entry is never matched, counted or moved by id."""
    try:
        return int(entry["id"]) if entry.get("id") is not None and str(entry["id"]).strip() != "" else None
    except (TypeError, ValueError):
        return None


def _ids(entries: Sequence[Mapping[str, Any]]) -> list[int]:
    return [i for i in (_int_id(e) for e in entries) if i is not None]


# ── choices ───────────────────────────────────────────────────────────────────

def choices(project: Project, models: Sequence[str]) -> dict[str, list[dict[str, str]]]:
    """Per run: every (source, process, metric) it recorded, as the Done list's menu."""
    from services.graph_manager.metric_catalog import catalog

    out: dict[str, list[dict[str, str]]] = {}
    for name in models:
        rows = []
        for (source, process), (scalars, arrays, _empty) in sorted(catalog(project.run_dir(name) / "metrics").items()):
            for kind, metrics in (("curve", arrays), ("per_epoch", scalars)):
                rows += [{"source": source, "process": process, "metric": m, "kind": kind} for m in sorted(metrics)]
        out[name] = rows
    return out


# ── what the runs share and what differs ──────────────────────────────────────

def _flat(value: Any, prefix: tuple = ()) -> Iterable[tuple[tuple, Any]]:
    """Leaf keys: a mapping by key, a list of blocks (the model's layers) by position, a list of values whole."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _flat(item, prefix + (str(key),))
    elif isinstance(value, list) and any(isinstance(item, (Mapping, list)) for item in value):
        for index, item in enumerate(value):
            yield from _flat(item, prefix + (str(index),))
    else:
        yield prefix, value


def _choices_of(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """The experiment choices of a run: its sweep sample, dataset and model id."""
    out = {".".join(k): v for k, v in _flat(snapshot.get("sampled") or {})}
    out["dataset"] = (snapshot.get("data") or {}).get("name")
    out["model_id"] = snapshot.get("model_id")
    return out


def split_of(snapshot: Mapping[str, Any]) -> Optional[int]:
    """The run's split of a profiling_split.random_state list (one run per split; Wouters et al. TCHES 2020(3)
    Sect. 3.2: ten trainings of each configuration), else None. Runs differing only by it are repeats of one
    configuration: a study pools them (graph job groups), never draws them as lines of their own."""
    split = (snapshot.get("data") or {}).get("profiling_split") or {}
    state = split.get("random_state") if isinstance(split, Mapping) and split.get("enabled") is True else None
    return state if isinstance(state, int) and not isinstance(state, bool) else None


def _same(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _axis_rank(key: str) -> tuple:
    """Which of two keys that follow from each other names the axis: what is set (dataset, model_id,
    trainer, sca) before the model's layers, a shorter path before a longer one."""
    return (key.startswith("model."), key.count("."), key)


def _differing(picks: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The keys the runs set to two values or more (a key unset in some runs and one value in the
    others is not: sca.variant, which the dataset type gives since 2026-10-16)."""
    keys = sorted(set().union(*map(set, picks.values())))
    values = {k: {_same(p.get(k)) for p in picks.values() if p.get(k) is not None} for k in keys}
    return [k for k in keys if len(values[k]) > 1]


def differing(snapshots: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Every experiment choice the runs do not share, axis or not: what a figure's header leaves out."""
    return _differing({name: _choices_of(s) for name, s in snapshots.items()})


def differences(snapshots: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """The study's axes, and each run's values on them. A key differs when the runs set it to two
    values or more (_differing). Where a block's `type` differs (an optimizer, a scheduler), an
    argument changing only with it is dropped (rho of RMSprop, beta_1 of Adam); one that also differs
    between runs of one type stays an axis (the learning rate of two RMSprop runs beside Adam and
    SGD). A key with one value for each value of an axis follows from that axis and is no axis itself
    (the last layer's out_features from sca.leakage_model: 256 for ID, 9 for HW; the epochs that each
    model_id of fig14 has)."""
    picks = {name: _choices_of(s) for name, s in snapshots.items()}
    varying = _differing(picks)
    typed = {k.rsplit(".", 1)[0] for k in varying if k.endswith(".type")}

    def within_one_type(key: str, block: str) -> bool:
        seen: dict[str, set[str]] = {}
        for p in picks.values():
            if p.get(key) is not None:
                seen.setdefault(_same(p.get(block + ".type")), set()).add(_same(p.get(key)))
        return any(len(v) > 1 for v in seen.values())

    varying = [k for k in varying
               if k.endswith(".type") or all(within_one_type(k, block) for block in typed if k.startswith(block + "."))]

    def follows(axis: str, key: str) -> bool:
        seen: dict[str, str] = {}
        return all(seen.setdefault(_same(p.get(axis)), _same(p.get(key))) == _same(p.get(key)) for p in picks.values())

    axes: list[str] = []
    for key in sorted(varying, key=_axis_rank):
        if not any(follows(axis, key) for axis in axes):
            axes.append(key)
    return axes, {name: {k: p.get(k) for k in axes} for name, p in picks.items()}


def _show(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return "none" if value is None else str(value)


def label(values: Mapping[str, Any]) -> str:
    """One run's legend label: the value alone when one setting differs, else key=value pairs."""
    if len(values) == 1:
        return _show(next(iter(values.values())))
    return ", ".join(f"{k.rsplit('.', 1)[-1]}={_show(v)}" for k, v in values.items()) or "run"


def shared_line(snapshot: Mapping[str, Any], varying: Sequence[str], batches: Sequence[str]) -> str:
    """'ASCAD_CNNbase on ascad, 100 epochs, batch 200, learning rate 1e-05 (batch fig10)' without
    the settings that differ (`varying`: the axes and every other key the runs do not share,
    differing())."""
    trainer = snapshot.get("trainer") or {}
    optimizer = trainer.get("optimizer") or {}
    parts = []
    if "model_id" not in varying:
        parts.append(str(snapshot.get("model_id")))
    head = " on ".join(parts + ([str((snapshot.get("data") or {}).get("name"))] if "dataset" not in varying else []))
    facts = []
    for key, text in (("trainer.epochs", f"{trainer.get('epochs')} epochs"),
                      ("trainer.batch_size", f"batch {trainer.get('batch_size')}"),
                      ("trainer.optimizer.learning_rate", f"learning rate {_show(optimizer.get('learning_rate'))}")):
        if key not in varying and text.split()[0] != "None":
            facts.append(text)
    tail = f" (batch {', '.join(batches)})" if batches else ""
    return ", ".join([head] + facts) + tail


# ── writing a job ─────────────────────────────────────────────────────────────

class _IndentDumper(yaml.SafeDumper):
    """Block lists indented under their key, as graph_config.yaml is written."""

    def increase_indent(self, flow: bool = False, indentless: bool = False):  # noqa: D401
        return super().increase_indent(flow, False)


class _Flow(list):
    """A list written on one line ([[1, 2], [3]]): a facet's arrangement.order."""


_IndentDumper.add_representer(_Flow, lambda dumper, value: dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True))


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _job_name(batches: Sequence[str], dataset: Optional[str], metric: str) -> str:
    return _safe("_".join([*(batches or ["runs"]), metric] + ([dataset] if dataset else [])))


def _header(job_id: int, title: str, lines: Sequence[str], project: str) -> list[str]:
    body = [f"  # JOB {job_id}: {title}"] + [f"  # {line}" for line in lines]
    return [_RULE, *body, f"  # Run: make graph ENV=dev PROJECT={project} GRAPH_ID={job_id}", _RULE]


def _job_text(job: Mapping[str, Any], header: Sequence[str], model_notes: Mapping[str, str],
              models_comment: str) -> str:
    """One `- id:` item of the jobs list, in the file's indentation, with its comments."""
    dumped = yaml.dump(dict(job), Dumper=_IndentDumper, sort_keys=False, allow_unicode=True,
                       default_flow_style=False, width=1000)
    lines = []
    in_models = False
    for i, line in enumerate(dumped.splitlines()):
        text = ("  - " if i == 0 else "    ") + line
        if line == "models:":
            in_models = True
            lines.append(text)
            lines.append(f"      # {models_comment}")
            continue
        if in_models and line.startswith("  - "):
            name = line[4:].strip().strip("'\"")
            text = f"{text}  # {model_notes[name]}" if model_notes.get(name) else text
        elif in_models and not line.startswith(" "):
            in_models = False
        lines.append(text)
    return "\n".join([*header, *lines]) + "\n"


def _jobs_end(text: str) -> int:
    """Where a new job goes: after the last job, before the comments and key that follow `jobs:`."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if re.match(r"^jobs:\s*(#.*)?$", l)), None)
    if start is None:
        raise ExportError("graph_config.yaml has no top-level jobs: list")
    nxt = next((i for i in range(start + 1, len(lines)) if re.match(r"^[^\s#]", lines[i])), len(lines))
    j = nxt
    while j > start + 1 and (lines[j - 1].startswith("#") or not lines[j - 1].strip()):
        j -= 1
    return sum(len(l) for l in lines[:j])


def _insert_job(text: str, chunk: str, new_job: Mapping[str, Any]) -> str:
    at = _jobs_end(text)
    new = text[:at] + ("" if text[:at].endswith("\n\n") else "\n") + chunk + "\n" + text[at:]
    before, after = yamlspan.load(text), yamlspan.load(new)
    expected = copy.deepcopy(before)
    expected["jobs"] = list(expected.get("jobs") or []) + [dict(new_job)]
    if after != expected:
        raise ExportError("the new job does not read back as exactly that job; graph_config.yaml not written")
    return new


def _metric_names(job: Mapping[str, Any]) -> set[str]:
    """The job's metric(s) as recorded: a list or one name, mean_fit_time_{s} as mean_fit_time_s."""
    raw = job.get("metric")
    return {_LOGGED_BRACES.sub("", str(m)) for m in (raw if isinstance(raw, list) else [raw]) if m is not None}


def _template(jobs: Sequence[Mapping[str, Any]], source: str, metric: str, relative: bool = False) -> dict[str, Any]:
    """The job a new one is styled like: the latest of the same metric (and source; a relative one
    first among the relative ones), else job 0."""
    ranked = sorted((j for j in jobs if _int_id(j) not in (None, 0)), key=_int_id, reverse=True)
    same = lambda j: metric in _metric_names(j)
    for test in (lambda j: same(j) and j.get("source", "attack_metrics") == source and bool(j.get("relative")) == relative,
                 lambda j: same(j) and bool(j.get("relative")) == relative,
                 lambda j: same(j) and j.get("source", "attack_metrics") == source,
                 same):
        found = next((j for j in ranked if test(j)), None)
        if found:
            return copy.deepcopy(dict(found))
    zero = next((j for j in jobs if _int_id(j) == 0), None)
    return copy.deepcopy(dict(zero or {"type": "model_overlay", "mode": "auto", "plot_type": "line"}))


def _matching_job(project: Project, jobs: Sequence[Mapping[str, Any]], source: str, metric: str, process: str,
                  batches: set[str], dataset: Optional[str], registry: Mapping[str, Any]) -> Optional[int]:
    """The index of a job over the same source, metric, process, batch and dataset, if one exists."""
    by_name = {str(e.get("model_name")): e for e in registry.values() if isinstance(e, Mapping)}
    for index, job in enumerate(jobs):
        if _int_id(job) in (None, 0) or job.get("type") == "model_explanation":
            continue
        if (job.get("metric"), job.get("source", "attack_metrics")) != (metric, source):
            continue
        if list(job.get("processes") or []) != [process]:
            continue
        for name in job.get("models") or []:
            entry = by_name.get(str(name))
            if not entry or not batches & set(entry.get("batches") or []):
                continue
            try:
                if (project.snapshot(str(name)).get("data") or {}).get("name") == dataset:
                    return index
            except ExportError:
                continue
    return None


def export(project: Project, models: Sequence[str], *, source: str, metric: str,
           process: Optional[str] = None) -> dict[str, Any]:
    if not models:
        raise ExportError("no models to export")
    process = process or DEFAULT_PROCESS.get(source, "test")
    registry = project.registry()
    entries = {str(e.get("model_name")): e for e in registry.values() if isinstance(e, Mapping)}
    missing = [m for m in models if m not in entries]
    if missing:
        raise ExportError(f"not in artifacts/run_registry.json: {missing}")
    available = choices(project, models)
    lacking = [m for m in models if not any((r["source"], r["process"], r["metric"]) == (source, process, metric)
                                            for r in available[m])]
    if lacking:
        raise ExportError(f"{metric} ({source}, {process}) is not recorded by: {lacking}")
    snapshots = {m: project.snapshot(m) for m in models}
    batches = sorted(set().union(*(set(entries[m].get("batches") or []) for m in models)))
    datasets = {(s.get("data") or {}).get("name") for s in snapshots.values()}
    dataset = next(iter(datasets)) if len(datasets) == 1 else None
    # What differs is read across the batch: every finished run of the same batch and dataset, so a
    # run exported alone (or added to its figure later) is still labelled by its batch's parameter.
    peers = dict(snapshots)
    for entry in entries.values():
        name = str(entry.get("model_name"))
        if name in peers or str(entry.get("status")) != "DONE" or not set(batches) & set(entry.get("batches") or []):
            continue
        if str(entry.get("env") or project.env) != project.env:
            continue
        try:
            snap = project.snapshot(name)
        except ExportError:
            continue
        if dataset is None or (snap.get("data") or {}).get("name") == dataset:
            peers[name] = snap
    varying, values = differences(peers)
    labels = {m: label(values[m]) for m in models}
    first = snapshots[models[0]]
    models_comment = shared_line(first, [*varying, *differing(peers)], batches)

    text = _read(project.graph_config)
    jobs = _jobs(text)
    index = _matching_job(project, jobs, source, metric, process, set(batches), dataset, registry)
    job = jobs[index] if index is not None else {}
    shown = {**{str(k): str(v) for k, v in ((job.get("legend") or {}).get("labels") or {}).items()}, **labels}
    twins = sorted(m for m in shown if list(shown.values()).count(shown[m]) > 1)
    if twins:
        raise ExportError(f"runs {', '.join(twins)} would share one legend label ({shown[twins[0]]}): "
                          "nothing in their experiment choices tells them apart")
    if index is not None:
        added = [m for m in models if m not in (job.get("models") or [])]
        before = text
        for name in added:
            text = yamledit.edit_text(text, "add", ["jobs", index, "models"], name)
            text = _comment_item(text, index, name, labels[name])
            text = yamledit.edit_text(text, "put", ["jobs", index, "legend", "labels"], labels[name], key=name)
        text, relocated = _relocated(project, before, text)
        _write(project.graph_config, text)
        return {"ok": True, "updated": int(job["id"]), "added": added, "labels": labels,
                **({"relocated": relocated} if relocated else {})}

    title = f"{', '.join(batches) or 'runs'}: {metric} ({source}, {process})" + (f", dataset {dataset}" if dataset else "")
    text, new_id, name = _new_job(project, text, models, labels=labels, varying=varying, shared=models_comment,
                                  source=source, metric=metric, process=process,
                                  name=_job_name(batches, dataset, metric), title=title,
                                  figure=f"{', '.join(batches) or 'runs'}: {dataset or 'several datasets'}")
    _write(project.graph_config, text)
    return {"ok": True, "created": new_id, "name": name, "labels": labels, "differ": varying}


def _new_job(project: Project, text: str, models: Sequence[str], *, labels: Mapping[str, str], varying: Sequence[str],
             shared: str, source: str, metric: str, process: str, name: str, title: str,
             figure: str, relative: Optional[Mapping[str, Any]] = None,
             groups: Optional[Sequence[Mapping[str, Any]]] = None) -> tuple[str, int, str]:
    """`text` with a new job appended (the next id), written the way a job is written by hand; styled
    like the latest job of the metric (else job 0). Every exported job reads each run's last epoch
    (array_epoch: latest), whatever the epochs of the runs. Returns (text, its id, its name)."""
    jobs = _jobs(text)
    new_id = max(_ids(jobs) + [0]) + 1
    job = _template(jobs, source, metric, relative=bool(relative))
    for key in ("top_models", "relative"):
        job.pop(key, None)
    job.update(id=new_id, name=name, enabled=True, source=source, metric=metric, models=list(models),
               processes=[process])
    name = _free_name(project, job)       # a folder left by a deleted job of this id keeps its name: acc_1
    job.update(name=name, output_file_name=name, array_epoch={"mode": "latest"})
    order = ["id", "name", "type", "mode", "enabled", "source", "metric", "plot_type", "models", "processes"]
    job = {**{k: job[k] for k in order if k in job}, **{k: v for k, v in job.items() if k not in order}}
    from services.graph_manager.metric_router import plot_types_for

    if groups:                                # repeats of one line pooled: thin runs, bold mean, the band
        job["type"] = "model_overlay"
        job["groups"] = [dict(g) for g in groups]
    else:
        job.pop("groups", None)
    if relative:                              # each run's value over the reference's, one bar per run
        job["relative"] = dict(relative)
        job["plot_type"] = "barh"
    if job.get("plot_type", "auto") not in plot_types_for(job.get("type", "model_overlay"), [metric], "latest"):
        job["plot_type"] = "auto"                 # the copied job's plot_type does not draw this metric
    legend = dict(job.get("legend") or {})
    legend.update(enabled=True, mode="custom", labels=dict(labels))
    job["legend"] = legend
    # A title rule names its own figure: the copied one is replaced by what this figure shows.
    style = dict(job.get("style") or {})
    rules = []
    for rule in style.get("text_rules") or []:
        if isinstance(rule, Mapping) and rule.get("target") == "title":
            rule = {**rule, "text": figure}
        elif relative and isinstance(rule, Mapping) and rule.get("target") in ("x_label", "y_label"):
            # bars along x: the value axis says what each bar is over ({relative_to}: the reference line)
            what = "fit time" if "fit_time" in metric else metric
            rule = {**rule, "text": f"{what} relative to {{relative_to}}" if rule["target"] == "x_label" else ""}
        rules.append(rule)
    if rules:
        style["text_rules"] = rules
        job["style"] = style
    differ = ", ".join(varying) if varying else "nothing (one run, or identical settings)"
    header = _header(new_id, title, [f"{shared}; models differ by {differ}.",
                                     f"Exported from the finished runs on {date.today().isoformat()} "
                                     "(graph_export.py: models, legend.labels from their snapshots)."], project.name)
    chunk = _job_text(job, header, labels, shared)
    return _insert_job(text, chunk, job), new_id, name


def _comment_item(text: str, job_index: int, name: str, note: str) -> str:
    """Append `# note` to the models item `name` of job `job_index` (comments are not data)."""
    root = yamlspan.compose(text)
    jobs_node = next(v for k, v in root.value if k.value == "jobs")
    job_node = jobs_node.value[job_index]
    models = next(v for k, v in job_node.value if k.value == "models")
    item = next(n for n in models.value if n.value == name)
    end = item.end_mark.index
    line_end = text.find("\n", end)
    line_end = len(text) if line_end < 0 else line_end
    if "#" in text[end:line_end]:
        return text
    return text[:line_end] + f"  # {note}" + text[line_end:]


# ── prune ─────────────────────────────────────────────────────────────────────

def prune(project: Project, models: Sequence[str]) -> dict[str, Any]:
    """Drop deleted runs from every job's models, legend.labels and job 0's top_models."""
    if not project.graph_config.is_file():
        return {"ok": True, "jobs": []}
    gone = {str(m) for m in models}
    text = _read(project.graph_config)
    before = text
    touched: set[int] = set()
    changed = True
    while changed:
        changed = False
        for index, job in enumerate(_jobs(text)):
            for field in ("models", "top_models"):
                for position, name in enumerate(job.get(field) or []):
                    if str(name) in gone:
                        text = yamledit.edit_text(text, "remove", ["jobs", index, field, position])
                        touched.add(int(job["id"]))
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                for name in list(((job.get("legend") or {}).get("labels") or {})):
                    if str(name) in gone:
                        text = yamledit.edit_text(text, "delete", ["jobs", index, "legend", "labels", name])
                        touched.add(int(job["id"]))
                        changed = True
                        break
            if changed:
                break
    if touched:
        text, _ = _relocated(project, before, text)
        _write(project.graph_config, text)
    return {"ok": True, "jobs": sorted(touched)}


def remove_from_jobs(project: Project, model: str, *, source: str, metric: str, process: Optional[str] = None) -> dict[str, Any]:
    """The panel's Delete from Graph Config: the run leaves every job drawing this source, metric
    and process (models, legend.labels); a job left without models is deleted, the later ids move
    down (a graph job never lists no trained run)."""
    process = process or DEFAULT_PROCESS.get(source, "test")
    text = _read(project.graph_config)
    before = text
    updated, emptied = [], []
    for index, job in enumerate(_jobs(text)):
        if _int_id(job) in (None, 0) or model not in [str(m) for m in job.get("models") or []]:
            continue
        if (job.get("metric"), job.get("source", "attack_metrics"), list(job.get("processes") or [])) != (metric, source, [process]):
            continue
        position = [str(m) for m in job["models"]].index(model)
        text = yamledit.edit_text(text, "remove", ["jobs", index, "models", position])
        if model in ((job.get("legend") or {}).get("labels") or {}):
            text = yamledit.edit_text(text, "delete", ["jobs", index, "legend", "labels", model])
        (emptied if len(job["models"]) == 1 else updated).append(int(job["id"]))
    if not updated and not emptied:
        raise ExportError(f"{model} is in no graph job drawing {metric} ({source}, {process})")
    text, _ = _relocated(project, before, text)
    _write(project.graph_config, text)
    deleted = []
    for job_id in sorted(emptied, reverse=True):            # highest first: the lower ids stay valid
        deleted.append(delete_job(project, job_id))
    return {"ok": True, "updated": updated, "deleted": sorted(emptied), "moved": [d.get("moved") for d in deleted]}


# ── a job added by hand (the panel's GRAPH_CONFIG.YAML → jobs → Add a Job) ──────────────────────────

def new_job(project: Project, name: str) -> dict[str, Any]:
    """A new job at the next id, named `name`, styled like the latest rank-curve job (else job 0): no
    models yet, so written enabled: false (make graph skips it) until its models are listed in the
    panel; array_epoch: latest. Its header says so."""
    name = _safe(name) if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(name or "")) else str(name)
    if not name:
        raise ExportError("a job needs a name")
    text = _read(project.graph_config)
    jobs = _jobs(text)
    new_id = max(_ids(jobs) + [0]) + 1
    job = _template(jobs, "cross_validation", "mean_attack_rank_curve")
    for key in ("top_models", "relative", "models"):
        job.pop(key, None)
    job.update(id=new_id, name=name, enabled=False, source="cross_validation", metric="mean_attack_rank_curve",
               processes=["validation"], models=[])
    name = _free_name(project, job)
    job.update(name=name, output_file_name=name, array_epoch={"mode": "latest"})
    legend = dict(job.get("legend") or {})
    legend.update(mode="auto", labels={})
    job["legend"] = legend
    order = ["id", "name", "type", "mode", "enabled", "source", "metric", "plot_type", "models", "processes"]
    job = {**{k: job[k] for k in order if k in job}, **{k: v for k, v in job.items() if k not in order}}
    header = _header(new_id, name, [f"Added in the panel on {date.today().isoformat()}: list its models, pick its source, "
                                    "metric and processes, then set enabled: true."], project.name)
    text = _insert_job(text, _job_text(job, header, {}, ""), job)
    _write(project.graph_config, text)
    return {"ok": True, "created": new_id, "name": name}


# ── a study: the finished runs of a queue (or of the queues of one figure) as its graph jobs ──────────
# A study is named by the part of a queue key before its first "_": fig14_cnnbest, fig14_vgg16 and
# fig14_mlpbest are study fig14, fig26 is study fig26. Where a queue of that very name exists (fig10
# beside fig10_optimizers), every queue is a study of its own. It exists once every queue of it is DONE (the
# runs of its latest batch_id all DONE, as many as its sweep times its commands: the panel's
# state.js queueStates): no job is made for a run that is not born. Its axes are what differs
# between its runs (differences()); the legend axes are a figure's lines, every value of the other
# axes is a figure of its own; each figure gets one job per metric chosen.

STUDY_FAILED = "This data can't be studied automatically; do it yourself in graph_config.yaml."
_LOGGED_BRACES = re.compile(r"[{}]")


class StudyError(ExportError):
    """The one failure of a study: TraceFlow cannot make its jobs; `detail` says what it met."""

    def __init__(self, detail: str) -> None:
        super().__init__(STUDY_FAILED)
        self.detail = detail


def _queue_file(project: Project) -> Path:
    return project.root / "config" / "experiment_queue.yaml"


def study_name(queue: str, queues: Iterable[str] = ()) -> str:
    """fig14_cnnbest -> fig14; a queue's own key when a queue is named by that stem (fig10_optimizers
    beside fig10)."""
    stem = str(queue).split("_", 1)[0]
    return str(queue) if stem != str(queue) and stem in set(map(str, queues)) else stem


def _latest_batch_id(entries: Sequence[Mapping[str, Any]], queue: str) -> str:
    """The queue's newest batch_id <queue>-<UTC start>; '' when none."""
    own = re.compile(rf"^{re.escape(queue)}-\d{{8}}T\d{{6}}Z$")
    ids = [str(i) for e in entries for i in (e.get("batch_ids") or []) if own.match(str(i))]
    return max(ids) if ids else ""


def _latest_batch(entries: Sequence[Mapping[str, Any]], queue: str) -> list[Mapping[str, Any]]:
    """The runs of the queue's latest execution (its newest batch_id <queue>-<UTC start>); runs made
    before batch ids existed: those labelled with the queue."""
    latest = _latest_batch_id(entries, queue)
    if not latest:
        return [e for e in entries if queue in (e.get("batches") or [])]
    return [e for e in entries if latest in (e.get("batch_ids") or [])]


def _batch_start(batch_id: str) -> Optional[datetime]:
    """The UTC start in a batch id (<queue>-YYYYMMDDTHHMMSSZ); None when it has none."""
    m = re.search(r"-(\d{8}T\d{6})Z$", batch_id or "")
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc) if m else None


def _started(r: Mapping[str, Any]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(r.get("started_at")))
    except (TypeError, ValueError):
        return None


def _taken_since(r: Mapping[str, Any], since: Optional[datetime]) -> bool:
    """A later execution of another queue trained the run again after `since` (this queue's batch start): its
    start, sweep size and status are now that execution's (fig20's FORCE retraining fig24's haberdashery_flock:
    total_trials 5). state.js takenSince does the same."""
    started = _started(r)
    if since is None or started is None:
        return False
    starts = [_batch_start(str(i)) for i in r.get("batch_ids") or []]
    return any(at is not None and at > since and started >= at for at in starts)


def _trained_since(runs: Sequence[Mapping[str, Any]], batch_id: str) -> list[Mapping[str, Any]]:
    """The runs a queue's execution trained itself: started after its batch began (the UTC start in the
    batch id). A run it skipped as already DONE joined the batch but carries the sweep it was trained in
    (fig10's ketchup_computer: 5 trials, from fig10_optimizers); [] when the id has no start."""
    since = _batch_start(batch_id)
    if since is None:
        return []
    return [r for r in runs if (started := _started(r)) is not None and started >= since]


def _queue_done(runs: Sequence[Mapping[str, Any]], commands: int, batch_id: str = "", queue: str = "") -> bool:
    """Every planned run of the queue's latest execution is DONE. The sweep size of one line comes from the
    queue's own runs only (state.js queueStates does the same): those the execution trained, else those the
    queue trained first (a retry that trained nothing yet); a run it skipped as DONE that another queue
    trained, or one another queue trained again later, carries that queue's sweep and never sizes it. With no
    run of its own: one run per line, never what params.yaml plans now (another queue's settings)."""
    def origin(r: Mapping[str, Any]) -> str:
        return str(((r.get("batch_ids") or []) + (r.get("batches") or []) + [""])[0])
    held = [r for r in runs if not _taken_since(r, _batch_start(batch_id))]
    own = [r for r in held if queue and (origin(r) == queue or origin(r).startswith(f"{queue}-"))]
    trained = _trained_since(held, batch_id) or own
    per_command = max([int(r.get("total_trials") or 0) for r in trained] + [0])
    planned = max(per_command * max(commands, 1), len(runs)) if per_command else max(commands, len(runs))
    return bool(runs) and all(str(r.get("status")) == "DONE" for r in runs) and len(runs) >= planned


def study_plan(project: Project, queue: str) -> dict[str, Any]:
    """The study of `queue`: its queues, runs, axes (each with its values), the default legend (the
    axis of most values; a tie: a numeric one, then by name) and the metrics every run recorded
    (those of metrics_to_log marked)."""
    queues = yaml.safe_load(_read(_queue_file(project))) or {}
    if queue not in queues:
        raise ExportError(f"no queue {queue} in {_queue_file(project)}")
    name = study_name(queue, queues)
    group = [k for k in queues if study_name(k, queues) == name]
    entries = [e for e in project.registry().values() if isinstance(e, Mapping)]
    unborn = [k for k in group
              if not _queue_done(_latest_batch(entries, k), len(queues[k] or []), _latest_batch_id(entries, k), k)]
    if unborn:
        raise ExportError(f"study {name} is not DONE ({', '.join(unborn)}): a study is made of finished runs only")
    runs: list[str] = []
    for k in group:
        runs += [str(e["model_name"]) for e in _latest_batch(entries, k) if str(e.get("model_name")) not in runs]
    if len(runs) < 2:
        raise StudyError(f"study {name} has one run: a solo figure, not an overlay (Done list → Export)")
    snapshots = {r: project.snapshot(r) for r in runs}
    axes, values = differences(snapshots)
    numeric = {a: all(isinstance(values[r][a], (int, float)) for r in runs if values[r][a] is not None) for a in axes}
    shown = {a: [_show(v) for v in sorted({_same(values[r][a]): values[r][a] for r in runs}.values(),
                                           key=lambda v: (v is None, v if numeric[a] else str(v)))] for a in axes}
    legend = [min(axes, key=lambda a: (-len(shown[a]), not numeric[a], a))] if axes else []
    available = choices(project, runs)
    common = set.intersection(*({(c["source"], c["process"], c["metric"], c["kind"]) for c in available[r]} for r in runs))
    logged = {_LOGGED_BRACES.sub("", str(m)) for s in snapshots.values() for m in s.get("metrics_to_log") or []}
    # each metric of metrics_to_log ticked once: where the figures read it (cross-validation's
    # validation folds; a fit time, the time of training, its train folds as job 4 reads it), else the
    # first place it is recorded
    ticked = {}
    home = lambda m: ("cross_validation", "train") if "fit_time" in m else ("cross_validation", "validation")
    for s, pr, m, _k in sorted(common, key=lambda c: ((c[0], c[1]) != home(c[2]), c)):
        if m in logged:
            ticked.setdefault(m, (s, pr))
    # repeats: runs of one configuration on several splits (split_of): pooled, so the band is asked
    configs: dict[str, set] = {}
    for r in runs:
        configs.setdefault(_same({a: values[r][a] for a in axes}), set()).add(split_of(snapshots[r]))
    repeats = max((len(s) for s in configs.values() if None not in s), default=1)
    return {"ok": True, "study": name, "queues": group, "runs": runs, "repeats": repeats,
            "axes": [{"key": a, "values": shown[a]} for a in axes], "legend": legend,
            "values": {r: {a: values[r][a] for a in axes} for r in runs},
            "metrics": [{"source": s, "process": pr, "metric": m, "kind": k, "logged": ticked.get(m) == (s, pr),
                         "relative": k == "per_epoch" and "fit_time" in m and ticked.get(m) == (s, pr)}
                        for s, pr, m, k in sorted(common)]}


def _filter_path(axis: str) -> str:
    """The model_filter path of an axis (relative.where): dataset is data.name in the run's snapshot,
    and a run's one model block is `model` there (model.0.x -> model.x)."""
    if axis == "dataset":
        return "data.name"
    return re.sub(r"^model\.0\.", "model.", axis)


def _relative_block(reference: Mapping[str, Any]) -> dict[str, Any]:
    """relative: of a job whose values are divided by the reference line's (graph_job_validator.py)."""
    return {"rule": "run", "where": [{"path": _filter_path(a), "equals": v} for a, v in reference.items()],
            "label": label(dict(reference))}


def study(project: Project, queue: str, *, legend: Sequence[str], metrics: Sequence[tuple[str, str, str]],
          relative: Sequence[tuple[str, str, str]] = (), reference: Optional[Mapping[str, Any]] = None,
          band: Optional[str] = None) -> dict[str, Any]:
    """The study's graph jobs: one per figure (each value of the axes not in `legend`) and metric; a
    job already listing exactly those runs for that source, metric and process is kept, not doubled.
    A metric of `relative` is drawn divided by the reference line of each figure (`reference`: its
    legend values, e.g. {trainer.optimizer.type: RMSprop}), as bars. Nothing is written unless every
    job can be made."""
    plan = study_plan(project, queue)
    axes = [a["key"] for a in plan["axes"]]
    if not metrics and not relative:
        raise ExportError("tick one metric at least")
    reference = dict(reference or {})
    if relative and (not reference or set(reference) - set(legend)):
        raise ExportError(f"a relative metric needs the reference line: a value of each legend axis {list(legend)}")
    unknown = [a for a in legend if a not in axes]
    if unknown:
        raise ExportError(f"not an axis of study {plan['study']}: {unknown} (axes: {axes})")
    split = [a for a in axes if a not in legend]
    runs = plan["runs"]
    snapshots = {r: project.snapshot(r) for r in runs}
    _, values = differences(snapshots)
    available = {(m["source"], m["process"], m["metric"]) for m in plan["metrics"]}
    kind_of = {(m["source"], m["process"], m["metric"]): m["kind"] for m in plan["metrics"]}
    splits = {r: split_of(snapshots[r]) for r in runs}
    missing = [m for m in [*metrics, *relative] if tuple(m) not in available]
    if missing:
        raise StudyError(f"not recorded by every run of study {plan['study']}: "
                         + ", ".join(f"{m} ({s}, {pr})" for s, pr, m in missing))
    figures: dict[tuple, list[str]] = {}
    for r in runs:
        figures.setdefault(tuple(_same(values[r][a]) for a in split), []).append(r)
    # a figure's lines in the order of their legend values (20, 50, 100 epochs; Adam, SGD)
    order = lambda r: tuple((values[r][a] is None, isinstance(values[r][a], str), values[r][a] if isinstance(values[r][a], (int, float)) else str(values[r][a]))
                            for a in legend)
    figures = {k: sorted(models, key=order) for k, models in figures.items()}
    text = _read(project.graph_config)
    created, kept = [], []
    for models in figures.values():
        labels = {r: label({a: values[r][a] for a in legend}) if legend else "run" for r in models}
        # runs of one line on several splits (profiling_split.random_state, one run each) are repeats: a curve
        # pools them (groups: thin runs, bold mean, the band), a per-epoch scalar draws each with its split
        pooled: dict[str, list[str]] = {}
        for r in models:
            pooled.setdefault(labels[r], []).append(r)
        repeats = {lab: rs for lab, rs in pooled.items() if len(rs) > 1
                   and None not in {splits[r] for r in rs} and len({splits[r] for r in rs}) == len(rs)}
        twins = [r for r in models if list(labels.values()).count(labels[r]) > 1 and labels[r] not in repeats]
        if twins:
            raise StudyError(f"runs {', '.join(twins)} would share one legend label ({labels[twins[0]]}): "
                             "nothing tells them apart on the axes ticked")
        where = {a: values[models[0]][a] for a in split}
        place = ", ".join(f"{a.rsplit('.', 1)[-1]}={_show(v)}" for a, v in where.items())
        datasets = {(snapshots[r].get("data") or {}).get("name") for r in models}
        dataset = next(iter(datasets)) if len(datasets) == 1 else None
        if relative and not any(all(_same(values[r][a]) == _same(v) for a, v in reference.items()) for r in models):
            raise StudyError(f"the reference {label(reference)} is not a line of the figure {place or 'of the study'}")
        for (source, process, metric), rel in [*((m, None) for m in metrics), *((m, _relative_block(reference)) for m in relative)]:
            groups, job_labels = None, dict(labels)
            if repeats and kind_of.get((source, process, metric)) == "curve" and not rel:
                if band not in GROUP_BANDS:
                    raise ExportError(f"the lines of study {plan['study']} repeat on {max(map(len, repeats.values()))} splits "
                                     f"each: pick the band their pooled curve is drawn with ({', '.join(GROUP_BANDS)})")
                groups = [{"label": lab, "models": sorted(rs, key=lambda r: splits[r]), "band": band}
                          for lab, rs in repeats.items()]
            elif repeats:
                job_labels = {r: f"{labels[r]}, split {splits[r]}" if labels[r] in repeats else labels[r] for r in models}
            same = next((j for j in _jobs(text) if _int_id(j) not in (None, 0) and set(map(str, j.get("models") or [])) == set(models)
                         and (j.get("source", "attack_metrics"), j.get("metric"), list(j.get("processes") or [])) == (source, metric, [process])
                         and _same(j.get("relative") or None) == _same(rel)), None)
            if same is not None:
                kept.append(int(same["id"]))
                continue
            parts = [plan["study"], *[_show(v) for v in where.values()], *(["relative"] if rel else []), metric]
            if dataset and "dataset" not in split:
                parts.append(dataset)
            figure = f"{plan['study']}: {place or dataset or 'runs'}"
            text, new_id, name = _new_job(
                project, text, models, labels=job_labels, varying=axes, groups=groups,
                shared=shared_line(snapshots[models[0]], [*axes, *differing({r: snapshots[r] for r in models})],
                                   plan["queues"]), source=source, metric=metric,
                process=process, name=_safe("_".join(parts)), figure=figure, relative=rel,
                title=f"study {plan['study']}{f' ({place})' if place else ''}: {metric}"
                      + (f" relative to {rel['label']}" if rel else "") + f" ({source}, {process})"
                      + (f", dataset {dataset}" if dataset and "dataset" not in split else ""))
            created.append({"id": new_id, "name": name, "models": list(models), "metric": metric, "relative": bool(rel)})
    if created:
        _write(project.graph_config, text)
    return {"ok": True, "study": plan["study"], "legend": list(legend), "split": split,
            "figures": len(figures), "created": created, "kept": sorted(set(kept))}


# ── ids ───────────────────────────────────────────────────────────────────────

def _model_graphs(project: Project) -> Optional[str]:
    """paths.model_graphs of the project (job_paths.DEFAULT_MODEL_GRAPHS when not set)."""
    return ((getattr(project, "cfg", None) or {}).get("paths") or {}).get("model_graphs")


def _job_dir(project: Project, job: Mapping[str, Any]) -> Path:
    """The job's figure folder (services/graph_manager/job_paths.py): its one run's graphs folder, else graphs/jobs/."""
    from services.graph_manager import job_paths
    from services.side_channel_core.utils.model_naming import _load_mapping

    return job_paths.job_dir(job, project_root=project.root, graphs_root=project.graphs, env=project.env,
                             model_graphs=_model_graphs(project),
                             name_map=_load_mapping(project.root))


def _free_name(project: Project, job: Mapping[str, Any]) -> str:
    """The job's name, or name_1, name_2, ... while that folder exists (job_paths.free_job_name)."""
    from services.graph_manager import job_paths
    from services.side_channel_core.utils.model_naming import _load_mapping

    return job_paths.free_job_name(job, project_root=project.root, graphs_root=project.graphs, env=project.env,
                                   model_graphs=_model_graphs(project), name_map=_load_mapping(project.root))


def _renamed(project: Project, text: str, index: int, job: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """The job, renamed in `text` (name, and output_file_name when it was the name) to its next free
    name when its folder exists: the folder there is not this job's and keeps its name."""
    name = _free_name(project, job)
    if name == str(job.get("name") or ""):
        return text, dict(job)
    text = yamledit.edit_text(text, "set", ["jobs", index, "name"], name)
    if "output_file_name" in job and str(job["output_file_name"]) == str(job.get("name")):
        text = yamledit.edit_text(text, "set", ["jobs", index, "output_file_name"], name)
    return text, {**job, "name": name}


def _address(project: Project, folder: Path) -> str:
    """The facet address prefix of a job folder (graphs/-relative, or artifacts/... for a run's)."""
    from services.graph_manager import job_paths

    return job_paths.file_address(folder, project_root=project.root, graphs_root=project.graphs) or folder.name


def _free_path(path: Path) -> Path:
    """path, or path_1, path_2, ... (a folder moved aside twice on one day)."""
    from services.graph_manager import job_paths

    return job_paths.free_path(path)


def _move_folder(src: Path, dst: Path) -> None:
    """Move a figure folder TraceFlow made, keeping its seals. A seal is bound to its file's path, so
    each JSON / JSONL file whose seal holds at the old path is signed again at the new one; a file
    whose seal is already broken moves as it is and stays reported (make verify)."""
    from services.side_channel_core.utils import seal
    from services.side_channel_core.utils.atomic_io import atomic_write_text

    intact = [f.relative_to(src) for f in src.rglob("*")
              if f.is_file() and seal.kind_of(f) in ("json", "jsonl") and seal.verify(f).state == "ok"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    for rel in intact:
        path = dst / rel
        text = path.read_text(encoding="utf-8")
        if seal.kind_of(path) == "json":
            plain = json.dumps(seal.strip(json.loads(text)), indent=2, ensure_ascii=False, default=str) + "\n"
        else:
            plain = "".join(json.dumps(seal.strip(json.loads(line)), separators=(",", ":"), ensure_ascii=False) + "\n"
                            for line in text.splitlines() if line.strip())
        atomic_write_text(path, plain)                    # seals at the new path


def _relocated(project: Project, before: str, text: str) -> tuple[str, list[list[str]]]:
    """A job whose run count crossed one (Export of a second run, Delete from Graph Config of all
    but one) changes folder (job_paths): its drawn folder follows, and facet addresses into it."""
    old = {_int_id(j): j for j in _jobs(before) if _int_id(j) is not None}
    moved = []
    for index, job in enumerate(_jobs(text)):
        prev = old.get(_int_id(job))
        if prev is None:
            continue
        src, dst = _job_dir(project, prev), _job_dir(project, job)
        if src == dst or not src.is_dir():
            continue
        if dst.exists():                  # not this job's: it keeps its name, the job takes the next one
            text, job = _renamed(project, text, index, job)
            dst = _job_dir(project, job)
        a, b = _address(project, src), _address(project, dst)
        _move_folder(src, dst)
        text = text.replace(f"{a}/", f"{b}/")
        moved.append([str(src), str(dst)])
    return text, moved


def _header_span(text: str, root, job_index: int, section: str = "jobs") -> tuple[int, int]:
    """The comment lines right above item `job_index`'s `- id:` line in `section` (its header)."""
    jobs_node = next(v for k, v in root.value if k.value == section)
    start = jobs_node.value[job_index].start_mark.index
    line_start = text.rfind("\n", 0, start) + 1
    top = line_start
    while top > 0:
        prev = text.rfind("\n", 0, top - 1) + 1
        if text[prev:top].lstrip().startswith("#"):
            top = prev
        else:
            break
    return top, line_start


def _move(project: Project, text: str, moves: Mapping[int, int]) -> tuple[str, dict[str, Any]]:
    """Give jobs their new ids: the id entries, their header lines, their figure folders and the
    faceting addresses into those folders."""
    jobs = _jobs(text)
    renamed, facets = [], 0
    for index, job in enumerate(jobs):
        old = _int_id(job)
        if old is None or old not in moves:
            continue
        new = moves[old]
        text = yamledit.edit_text(text, "set", ["jobs", index, "id"], str(new))
        root = yamlspan.compose(text)
        top, bottom = _header_span(text, root, index)
        block = text[top:bottom]
        block = re.sub(rf"(#\s*JOB\s+){old}\b", rf"\g<1>{new}", block)
        block = re.sub(rf"GRAPH_ID={old}\b", f"GRAPH_ID={new}", block)
        text = text[:top] + block + text[bottom:]
        before, after = _job_dir(project, job), _job_dir(project, {**job, "id": new})
        if before.is_dir():
            if after.exists():            # left by a job deleted by hand: it keeps its name, this job takes the next
                text, renamed_job = _renamed(project, text, index, {**job, "id": new})
                after = _job_dir(project, renamed_job)
            _move_folder(before, after)
            renamed.append([before.name, after.name])
        a, b = _address(project, before), _address(project, after)
        count = text.count(f"{a}/")
        if count:
            text = text.replace(f"{a}/", f"{b}/")
            facets += count
    return text, {"folders": renamed, "facet_addresses": facets}


def _old_references(project: Project, moves: Mapping[int, int]) -> list[str]:
    """Places that still name a moved id, listed for the user, not edited: other files, and prose
    in graph_config.yaml's comments ("the bars of job 55")."""
    patterns = [re.compile(rf"GRAPH_ID={old}\b|job_{old:03d}_") for old in moves]
    hits = []
    prose = [re.compile(rf"\bjob\s+{old}\b", re.IGNORECASE) for old in moves]
    for number, line in enumerate(_read(project.graph_config).splitlines(), 1):
        if line.lstrip().startswith("#") and not re.search(r"#\s*JOB\s+\d+:", line) and any(p.search(line) for p in prose):
            hits.append(f"{project.graph_config}:{number}")
    for path in [project.root / "README.md", project.root / "config" / "experiment_queue.yaml",
                 *sorted((project.root.parent.parent / "paper").rglob("*.tex"))]:
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if any(p.search(line) for p in patterns):
                hits.append(f"{path}:{number}")
    return hits


def delete_job(project: Project, job_id: int) -> dict[str, Any]:
    """Remove job `job_id` (with its header) and move every later job down by one id."""
    if int(job_id) == 0:
        raise ExportError("job 0 is the model judge's job; it is never deleted")
    text = _read(project.graph_config)
    jobs = _jobs(text)
    index = next((i for i, j in enumerate(jobs) if _int_id(j) == int(job_id)), None)
    if index is None:
        raise ExportError(f"no graph job id={job_id}")
    root = yamlspan.compose(text)
    top, bottom = _header_span(text, root, index)
    header = text[top:bottom]
    text = yamledit.edit_text(text, "delete", ["jobs", index])
    # The deleted job's own header (comments are not data, so the delete keeps them): removed when
    # it is still right there and is that job's (it names JOB <id>); the next job's header stays.
    if header and text[top:top + len(header)] == header and re.search(rf"#\s*JOB\s+{int(job_id)}\b", header):
        text = text[:top] + text[top + len(header):]
    _write(project.graph_config, text)
    return {"deleted": int(job_id), **close_gap(project, int(job_id), name=str(jobs[index]["name"]), job=jobs[index])}


def close_gap(project: Project, job_id: int, *, name: Optional[str] = None,
              job: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Job `job_id` is gone from graph_config.yaml (deleted here or by hand): its figures move aside
    (graphs/jobs/_deleted/, nothing removed) and every later job moves down by one id."""
    text = _read(project.graph_config)
    jobs = _jobs(text)
    if any(_int_id(j) == int(job_id) for j in jobs):
        raise ExportError(f"graph job id={job_id} is still in graph_config.yaml")
    if job is not None:
        folders = [_job_dir(project, job)]
    else:                           # deleted by hand: its folder in graphs/jobs/ or in a run's graphs folder
        from services.graph_manager import job_paths

        kept = {_job_dir(project, j) for j in jobs}
        runs = job_paths.run_graphs_dir(project.root, project.env, "*", _model_graphs(project))
        pattern = f"job_{int(job_id):03d}_*"
        found = list((project.graphs / "jobs").glob(pattern))
        try:
            found += [d for d in project.root.glob(str(runs.relative_to(project.root) / pattern))]
        except ValueError:          # paths.model_graphs outside the project: no run folders to search
            pass
        folders = [d for d in found if d.is_dir() and d not in kept]
    # Facets that showed the removed job lose those panels (a facet left with fewer than two goes).
    facets = {"facets_removed": [], "facet_panels_removed": 0}
    for folder in folders or [None]:
        text, dropped = _drop_job_from_facets(project, text, folder.name if folder is not None else f"job_{int(job_id):03d}_")
        facets["facets_removed"] += dropped["facets_removed"]
        facets["facet_panels_removed"] += dropped["facet_panels_removed"]
    jobs = _jobs(text)
    aside = []
    for folder in folders:
        if folder.is_dir():
            target = _free_path(project.graphs / "jobs" / "_deleted" / f"{folder.name}_{date.today().isoformat()}")
            _move_folder(folder, target)
            aside.append(str(target))
    moves = {_int_id(j): _int_id(j) - 1 for j in jobs if _int_id(j) is not None and _int_id(j) > int(job_id)}
    text, moved = _move(project, text, moves)
    _write(project.graph_config, text)
    return {"ok": True, "figures_moved_aside": aside, "moved": {str(k): v for k, v in moves.items()}, **moved,
            **facets, "old_references": _old_references(project, moves)}


# ── facets ────────────────────────────────────────────────────────────────────

def _facets(text: str) -> list[dict[str, Any]]:
    return list((yamlspan.load(text) or {}).get("faceting") or [])


def _facet_folder(project: Project, facet_id: int, name: str) -> Path:
    from services.graph_manager import job_paths

    return project.graphs / "facets" / job_paths.facet_folder_name({"id": facet_id, "name": name})


def _without_ids(order: Any, gone: set[int]) -> Any:
    if any(isinstance(x, list) for x in order or []):
        rows = [[i for i in row if int(i) not in gone] for row in order]
        return [row for row in rows if row]
    return [i for i in order or [] if int(i) not in gone]


def _delete_facet_item(text: str, index: int, facet_id: int) -> str:
    """Remove facet item `index` and its own header (a header that names FACET <id>)."""
    top, bottom = _header_span(text, yamlspan.compose(text), index, "faceting")
    header = text[top:bottom]
    text = yamledit.edit_text(text, "delete", ["faceting", index])
    if header and text[top:top + len(header)] == header and re.search(rf"#\s*FACET\s+{int(facet_id)}\b", header):
        text = text[:top] + text[top + len(header):]
    return text


def _shift_facets(project: Project, text: str, gone: int) -> str:
    """Facets after the removed id `gone` move down one id: the id, the FACET / FACE_ID lines of
    their header, their output folder."""
    for index, facet in enumerate(_facets(text)):
        old = _int_id(facet)
        if old is None or old <= gone:
            continue
        new = old - 1
        text = yamledit.edit_text(text, "set", ["faceting", index, "id"], str(new))
        top, bottom = _header_span(text, yamlspan.compose(text), index, "faceting")
        block = re.sub(rf"(#\s*FACET\s+){old}\b", rf"\g<1>{new}", text[top:bottom])
        block = re.sub(rf"FACE_ID={old}\b", f"FACE_ID={new}", block)
        text = text[:top] + block + text[bottom:]
        name = str(facet.get("name") or f"facet_{old}")
        before, after = _facet_folder(project, old, name), _facet_folder(project, new, name)
        if before.is_dir():
            if after.exists():
                raise ExportError(f"cannot move {before.name}: {after.name} exists")
            before.rename(after)
    return text


def _remove_facet(project: Project, text: str, index: int) -> tuple[str, int]:
    facet = _facets(text)[index]
    facet_id = int(facet["id"])
    text = _delete_facet_item(text, index, facet_id)
    folder = _facet_folder(project, facet_id, str(facet.get("name") or f"facet_{facet_id}"))
    if folder.is_dir():
        target = _free_path(project.graphs / "facets" / "_deleted" / f"{folder.name}_{date.today().isoformat()}")
        target.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(target)
    return _shift_facets(project, text, facet_id), facet_id


def _points_into(address: str, job_folder: str) -> bool:
    """The address names a file of the job folder `job_folder` (a full name, or the job_NNN_ prefix)."""
    from services.graph_manager import job_paths

    found = job_paths.address_job(address)
    return bool(found) and (found[0] == job_folder or (job_folder.endswith("_") and found[0].startswith(job_folder)))


def _drop_job_from_facets(project: Project, text: str, job_folder: str) -> tuple[str, dict[str, Any]]:
    """Panels showing files of the job folder `job_folder` (a full folder name, or the
    `job_NNN_` prefix) leave their facets; a facet left with fewer than two panels is removed."""
    removed, panels_removed = [], 0
    index = len(_facets(text)) - 1
    while index >= 0:
        facet = _facets(text)[index]
        panels = list(facet.get("panels") or [])
        hit = [k for k, p in enumerate(panels) if _points_into(str(p.get("address", "")), job_folder)]
        if hit:
            if len(panels) - len(hit) < 2:
                text, facet_id = _remove_facet(project, text, index)
                removed.append(facet_id)
            else:
                gone = {int(panels[k]["file_id"]) for k in hit}
                for k in sorted(hit, reverse=True):
                    text = yamledit.edit_text(text, "delete", ["faceting", index, "panels", k])
                arrangement = facet.get("arrangement") or {}
                if arrangement.get("order") is not None:
                    order = _without_ids(arrangement["order"], gone)
                    text = yamledit.edit_text(text, "set", ["faceting", index, "arrangement", "order"],
                                              json.dumps(order))
                panels_removed += len(hit)
        index -= 1
    return text, {"facets_removed": sorted(removed), "facet_panels_removed": panels_removed}


def delete_facet(project: Project, facet_id: int) -> dict[str, Any]:
    text = _read(project.graph_config)
    index = next((i for i, f in enumerate(_facets(text)) if _int_id(f) == int(facet_id)), None)
    if index is None:
        raise ExportError(f"no facet id={facet_id}")
    text, _ = _remove_facet(project, text, index)
    _write(project.graph_config, text)
    return {"ok": True, "deleted": int(facet_id)}


def facet_close_gap(project: Project, facet_id: int) -> dict[str, Any]:
    """Facet `facet_id` was removed by hand: the later facets move down one id."""
    text = _read(project.graph_config)
    if any(_int_id(f) == int(facet_id) for f in _facets(text)):
        raise ExportError(f"facet id={facet_id} is still in graph_config.yaml")
    for folder in (project.graphs / "facets").glob(f"facet_{int(facet_id):03d}_*") if (project.graphs / "facets").is_dir() else []:
        target = _free_path(project.graphs / "facets" / "_deleted" / f"{folder.name}_{date.today().isoformat()}")
        target.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(target)
    text = _shift_facets(project, text, int(facet_id))
    _write(project.graph_config, text)
    return {"ok": True, "closed": int(facet_id)}


def _facet_comments(lines: list[str]) -> list[str]:
    """Each key of a facet that takes a known set says its options (graph_manager/faceting.py),
    so a facet in the file shows how faceting is written."""
    from services.graph_manager import faceting as F

    notes = {
        ("panels", "orientation"): " | ".join(sorted(F.VALID_ORIENTATIONS)),
        ("arrangement", "mode"): " | ".join(sorted(F.VALID_ARRANGEMENTS)) + " (grid: order as rows, e.g. [[1, 2], [3]])",
        ("arrangement", "order"): "the panels' file_ids in place order",
        ("legend", "enabled"): "true draws the legend; entry i takes the colour of a job's i-th model",
        ("legend", "position"): " | ".join(sorted(F.VALID_LEGEND_POSITIONS)),
        ("legend", "columns"): "entries per legend row",
        ("legend", "handle"): "[shape, thickness pt]: line | patch | a Matplotlib marker | one character",
        ("legend", "mode"): "auto: the entries of the first panel's graph job | custom: legend.entries",
        ("export", "formats"): " | ".join(sorted(F.VALID_FACET_FORMATS)),
        ("export", "margin"): "white border in inches",
        ("png", "dpi"): "the png, and a pdf / svg that embeds png panels",
        (None, "output_file_name"): "graphs/facets/facet_<id>_<name>/<output_file_name>.<format>",
    }
    stack: list[tuple[int, str]] = []
    out = []
    for line in lines:
        m = re.match(r"^(\s*)(- )?([A-Za-z_]+):", line)
        if m:
            indent = len(m.group(1)) + (2 if m.group(2) else 0)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1] if stack else None
            note = notes.get((parent, m.group(3)))
            if note and "#" not in line:
                line = f"{line}   # {note}"
            stack.append((indent, m.group(3)))
        out.append(line)
    return out


def _facet_layout(count: int) -> dict[str, Any]:
    """One row for up to three panels, else a grid of rows of three."""
    ids = list(range(1, count + 1))
    if count <= 3:
        return {"mode": "horizontal", "order": ids}
    return {"mode": "grid", "order": [ids[i:i + 3] for i in range(0, count, 3)]}


def export_facet(project: Project, addresses: Sequence[str], *, title: Optional[str] = None) -> dict[str, Any]:
    """Drawn graph files composed as one new facet; each address is relative to graphs/, or artifacts/...
    for a figure in a run's graphs folder (a job of one run: job_paths.py)."""
    from services.graph_manager.faceting import validate_facet

    from services.graph_manager import job_paths

    addresses = [str(a).strip().lstrip("/") for a in addresses if str(a).strip()]
    if len(addresses) < 2:
        raise ExportError("a facet composes at least two figure files")
    missing = [a for a in addresses
               if not job_paths.address_file(a, project_root=project.root, graphs_root=project.graphs).is_file()]
    if missing:
        raise ExportError(f"not drawn yet (no such file under graphs/ or a run's graphs folder): {missing}")
    text = _read(project.graph_config)
    jobs = {_int_id(j): j for j in _jobs(text) if _int_id(j) is not None}
    sources = []
    for address in addresses:
        found = job_paths.address_job(address)
        sources.append(jobs.get(found[1]) if found else None)
    facets = _facets(text)
    new_id = max(_ids(facets) + [0]) + 1
    names = [str(j["name"]) if j else Path(a).stem for j, a in zip(sources, addresses)]
    name = _safe("_and_".join(dict.fromkeys(f"job_{int(j['id'])}" if j else Path(a).stem for j, a in zip(sources, addresses))))
    orientation = lambda j: str((((j or {}).get("style") or {}).get("figure") or {}).get("orientation", "landscape"))
    facet = {
        "id": new_id, "name": name, "enabled": True, "output_file_name": name,
        "panels": [{"file_id": k, "address": a, "orientation": orientation(j)}
                   for k, (a, j) in enumerate(zip(addresses, sources), 1)],
        "arrangement": {**_facet_layout(len(addresses)), "order": _Flow(_facet_layout(len(addresses))["order"])},
        "legend": {"enabled": False, "position": "bottom", "columns": 4, "handle": _Flow(["line", 1.5]), "mode": "auto"},
        "export": {"formats": ["svg"], "margin": 0.05, "png": {"dpi": 300}},
        "text_rules": [
            {"target": "title", "text": title or " | ".join(dict.fromkeys(names)), "bold": True},
            {"target": "panel_label", "text": "({letter})"}],
    }
    validate_facet(facet)
    lines = [f"job {int(j['id'])} ({j['name']})" if j else a for j, a in zip(sources, addresses)]
    header = [_RULE, f"  # FACET {new_id}: {title or ' | '.join(dict.fromkeys(names))}",
              *[f"  #   panel {k}: {line}" for k, line in enumerate(lines, 1)],
              "  #   Draw the graph jobs first; arrangement.order places the file_ids; legend mode auto names the colours as panel 1's job.",
              f"  #   Run: make ENV=dev PROJECT={project.name} FACE_ID={new_id} graph", _RULE]
    _write(project.graph_config, _insert_facet(text, header, facet))
    return {"ok": True, "created": new_id, "name": name, "panels": len(addresses)}


def _insert_facet(text: str, header: Sequence[str], facet: Mapping[str, Any]) -> str:
    """`text` with the facet written last under faceting: (made when missing), every key commented
    with its options; refused unless the file reads back as exactly that facet added."""
    dumped = yaml.dump(dict(facet), Dumper=_IndentDumper, sort_keys=False, allow_unicode=True, default_flow_style=False, width=1000)
    body = [("  - " if i == 0 else "    ") + l for i, l in enumerate(dumped.splitlines())]
    chunk = "\n".join([*header, *_facet_comments(body)]) + "\n"
    empty = re.search(r"^faceting:\s*\[\s*\][^\n]*\n?", text, flags=re.M)
    if empty:
        new = text[:empty.start()] + "faceting:\n\n" + chunk + text[empty.end():]
    elif re.search(r"^faceting:", text, flags=re.M):
        new = text.rstrip("\n") + "\n\n" + chunk
    else:
        new = text.rstrip("\n") + "\n\nfaceting:\n\n" + chunk
    before, after = yamlspan.load(text) or {}, yamlspan.load(new) or {}
    expected = copy.deepcopy(before)
    expected["faceting"] = list(expected.get("faceting") or []) + [dict(facet)]
    if after != expected:
        raise ExportError("the new facet does not read back as exactly that facet; graph_config.yaml not written")
    return new


def new_facet(project: Project, name: str) -> dict[str, Any]:
    """A facet at the next id, named `name`, with no panels yet (the panel's GRAPH_CONFIG.YAML →
    faceting → Add a Facet): written enabled: false until its panels are listed (+ add a panel, then
    Find Figure); they are drawn in a row as listed (no arrangement.order) until a layout is picked."""
    from services.graph_manager.faceting import validate_facet

    name = str(name or "").strip()
    name = name if re.fullmatch(r"[A-Za-z0-9_.-]+", name) else _safe(name)
    if not name:
        raise ExportError("a facet needs a name")
    text = _read(project.graph_config)
    new_id = max(_ids(_facets(text)) + [0]) + 1
    facet = {
        "id": new_id, "name": name, "enabled": False, "output_file_name": name, "panels": [],
        "arrangement": {"mode": "horizontal"},
        "legend": {"enabled": False, "position": "bottom", "columns": 4, "handle": _Flow(["line", 1.5]), "mode": "auto"},
        "export": {"formats": ["svg"], "margin": 0.05, "png": {"dpi": 300}},
        "text_rules": [{"target": "title", "text": name, "bold": True}, {"target": "panel_label", "text": "({letter})"}],
    }
    # every key as graph_manager reads it, with one panel standing in for those still to be listed
    validate_facet({**facet, "enabled": True, "panels": [{"file_id": 1, "address": "jobs/job_001_x/x.png"}]})
    header = [_RULE, f"  # FACET {new_id}: {name}",
              f"  #   Added in the panel on {date.today().isoformat()}: + add a panel under panels (Find Figure picks its file),",
              "  #   pick the layout, then set enabled: true. Draw the graph jobs first.",
              f"  #   Run: make ENV=dev PROJECT={project.name} FACE_ID={new_id} graph", _RULE]
    _write(project.graph_config, _insert_facet(text, header, facet))
    return {"ok": True, "created": new_id, "name": name}


def renumber(project: Project) -> dict[str, Any]:
    """Close every gap: the jobs keep their order by id and become 0, 1, 2, ..."""
    text = _read(project.graph_config)
    ids = sorted(_ids(_jobs(text)))
    moves = {old: new for new, old in enumerate(ids) if old != new and old != 0}
    if not moves:
        return {"ok": True, "moved": {}}
    text, moved = _move(project, text, moves)
    _write(project.graph_config, text)
    return {"ok": True, "moved": {str(k): v for k, v in moves.items()}, **moved,
            "old_references": _old_references(project, moves)}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Finished runs into graph jobs (graph_config.yaml).")
    parser.add_argument("command", choices=["choices", "export", "remove", "prune", "delete-job", "close-gap", "renumber",
                                            "export-facet", "delete-facet", "facet-close-gap", "study-plan", "study", "new-job", "new-facet"])
    parser.add_argument("--project", required=True)
    parser.add_argument("--env", default="dev")
    parser.add_argument("--models", default="")
    parser.add_argument("--source", default="cross_validation")
    parser.add_argument("--process", default=None)
    parser.add_argument("--metric", default="mean_attack_rank_curve")
    parser.add_argument("--id", type=int, default=None)
    parser.add_argument("--files", default="", help="export-facet: graph files under graphs/, comma-separated")
    parser.add_argument("--title", default=None)
    parser.add_argument("--queue", default="", help="study-plan / study: a queue of the study (experiment_queue.yaml key)")
    parser.add_argument("--legend", default="", help="study: the axes drawn as lines, comma-separated")
    parser.add_argument("--metrics", default="", help="study: source|process|metric, comma-separated")
    parser.add_argument("--relative", default="", help="study: source|process|metric drawn relative to --reference")
    parser.add_argument("--reference", default="", help='study: the reference line as JSON, {"<legend axis>": value}')
    parser.add_argument("--band", default="", help="study: min_max | std | none, the band of lines repeated on splits")
    args = parser.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    try:
        project = Project(args.project, args.env)
        if args.command == "choices":
            out: Any = choices(project, models)
        elif args.command == "export":
            out = export(project, models, source=args.source, metric=args.metric, process=args.process)
        elif args.command == "remove":
            if len(models) != 1:
                raise ExportError("remove takes one model (--models NAME)")
            out = remove_from_jobs(project, models[0], source=args.source, metric=args.metric, process=args.process)
        elif args.command == "prune":
            out = prune(project, models)
        elif args.command == "delete-job":
            if args.id is None:
                raise ExportError("delete-job needs --id")
            out = delete_job(project, args.id)
        elif args.command == "export-facet":
            out = export_facet(project, [f for f in args.files.split(",") if f.strip()], title=args.title)
        elif args.command in ("delete-facet", "facet-close-gap"):
            if args.id is None:
                raise ExportError(f"{args.command} needs --id")
            out = delete_facet(project, args.id) if args.command == "delete-facet" else facet_close_gap(project, args.id)
        elif args.command == "new-job":
            out = new_job(project, args.title or "")
        elif args.command == "new-facet":
            out = new_facet(project, args.title or "")
        elif args.command == "study-plan":
            out = study_plan(project, args.queue)
        elif args.command == "study":
            picked = [tuple(m.split("|")) for m in args.metrics.split(",") if m.strip()]
            ratios = [tuple(m.split("|")) for m in args.relative.split(",") if m.strip()]
            if any(len(m) != 3 for m in [*picked, *ratios]):
                raise ExportError("--metrics and --relative take source|process|metric, comma-separated")
            try:
                reference = json.loads(args.reference) if args.reference.strip() else None
            except ValueError as exc:
                raise ExportError(f"--reference is not JSON: {exc}") from None
            out = study(project, args.queue, legend=[a for a in args.legend.split(",") if a.strip()], metrics=picked,
                        relative=ratios, reference=reference, band=args.band.strip() or None)
        elif args.command == "close-gap":
            if args.id is None:
                raise ExportError("close-gap needs --id (the id that is gone)")
            out = close_gap(project, args.id)
        else:
            out = renumber(project)
    except StudyError as exc:
        print(json.dumps({"error": str(exc), "study_failed": True, "detail": exc.detail,
                          "graph_config": str(project.graph_config)}))
        return 1
    except (ExportError, yamledit.EditError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
