# services/graph_manager/sources.py
"""
sources.json: the provenance of a figure, written beside it and signed
(services/side_channel_core/utils/seal.py).

A graph job records the runs its figures were drawn from: every run by name,
hash, env, status and the key its device.yaml declares, with every file of the
run the job read and the seal state that file had when it was read
(utils/seal.py recording()). Files read outside a run folder (the registries)
are listed too. The figures it wrote are listed with their SHA-256, so the
signed sources.json covers them as well: `make verify` reports a figure that no
longer matches (utils/seal.py _changed_figures).

A facet records the figure files it composed, their SHA-256, and the
sources.json of each graph job they came from (its runs and its own SHA-256).

    {"figure": {"kind": "graph job" | "facet", "id", "name", "definition"},
     "drawn_at", "code": {commit, ...},
     "runs": [{"model_name", "model_hash", "env", "status", "folder", "key",
               "seal": {"ok": n, ...}, "files": [{"path", "seal", "key"[, "sha256"]}]}],
     "other_files": [{"path", "seal", "key"[, "sha256"]}],
     "outputs": [{"file", "sha256"}]}

A file read without a seal (written before seals existed) carries its SHA-256
here, so a later change to it is still visible.

Author: Shraddha Mishra
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from services.side_channel_core.utils.atomic_io import atomic_write_text
from services.side_channel_core.utils.seal import SealStatus, judge, run_folder_of, run_key
from services.side_channel_core.utils.timestamps import now_iso

SOURCES_FILE = "sources.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _file_entry(path: Path, status: SealStatus, root: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {"path": _relative(path, root), "seal": status.state}
    if status.key:
        entry["key"] = status.key
    if status.state != "ok" and path.is_file():
        entry["sha256"] = file_sha256(path)
    return entry


def _outputs(outputs: Iterable[Path], folder: Path) -> list[dict[str, str]]:
    """The figures written, by their path inside the figure folder (where sources.json sits)."""
    return [{"file": _relative(Path(p), Path(folder)), "sha256": file_sha256(Path(p))}
            for p in outputs if Path(p).is_file()]


def _code() -> dict[str, Any]:
    from services.side_channel_core.utils.device_record import code_version

    return code_version()


def job_sources(
    *,
    job: Mapping[str, Any],
    bound: Iterable[Mapping[str, Any]],
    reads: Mapping[str, SealStatus],
    outputs: Iterable[Path],
    project_root: Path,
    env: str,
    folder: Path,
) -> dict[str, Any]:
    """The sources.json of one graph job: its runs, the files it read, the figures it wrote."""
    project_root = Path(project_root)
    runs: dict[str, dict[str, Any]] = {}
    for run in bound:
        runs[str(run["model_name"])] = {
            "model_name": str(run["model_name"]),
            "model_hash": str(run["model_hash"]),
            "env": env,
            "status": str(run.get("status")),
            "folder": None,
            "key": None,
            "seal": {},
            "files": [],
        }
    other: list[dict[str, Any]] = []
    for raw, status in sorted(reads.items()):
        path = Path(raw)
        run_folder = run_folder_of(path)
        run = runs.get(run_folder.name) if run_folder is not None else None
        if run is None:
            other.append(_file_entry(path, status, project_root))
            continue
        run["folder"] = run["folder"] or _relative(run_folder, project_root)
        run["key"] = run["key"] or run_key(path)
        run["files"].append(_file_entry(path, status, run_folder))
        run["seal"][status.state] = run["seal"].get(status.state, 0) + 1
    return {
        "figure": {"kind": "graph job", "id": job.get("id"), "name": job.get("name"), "definition": dict(job)},
        "drawn_at": now_iso(),
        "code": _code(),
        "runs": list(runs.values()),
        "other_files": other,
        "outputs": _outputs(outputs, folder),
    }


def facet_sources(
    *,
    facet: Mapping[str, Any],
    panels: Iterable[Path],
    outputs: Iterable[Path],
    graph_root: Path,
    folder: Path,
) -> dict[str, Any]:
    """The sources.json of one facet: the panel figures and the graph-job sources.json behind each."""
    graph_root = Path(graph_root)
    panel_entries = []
    for figure in panels:
        figure = Path(figure)
        entry: dict[str, Any] = {"file": _relative(figure, graph_root), "sha256": file_sha256(figure)}
        # The graph job's sources.json sits in its job folder, above the figure's subfolders.
        job_file = next((d / SOURCES_FILE for d in figure.parents
                         if (d / SOURCES_FILE).is_file() and graph_root.resolve() in d.resolve().parents), None)
        if job_file is not None:
            status = judge(job_file)
            entry["job_sources"] = {"file": _relative(job_file, graph_root), "sha256": file_sha256(job_file),
                                    "seal": status.state}
            try:
                runs = json.loads(job_file.read_text(encoding="utf-8")).get("runs") or []
            except (OSError, ValueError):
                runs = []
            entry["runs"] = [{"model_name": r.get("model_name"), "model_hash": r.get("model_hash")} for r in runs]
        else:
            entry["job_sources"] = None   # drawn before sources.json existed
        panel_entries.append(entry)
    return {
        "figure": {"kind": "facet", "id": facet.get("id"), "name": facet.get("name"), "definition": dict(facet)},
        "drawn_at": now_iso(),
        "code": _code(),
        "panels": panel_entries,
        "outputs": _outputs(outputs, folder),
    }


def write_sources(folder: Path, sources: Mapping[str, Any]) -> Path:
    """Write sources.json into the figure folder, signed (atomic_write_text seals JSON)."""
    path = Path(folder) / SOURCES_FILE
    atomic_write_text(path, json.dumps(sources, indent=2, ensure_ascii=False, default=str) + "\n")
    return path

