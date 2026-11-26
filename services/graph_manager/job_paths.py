# services/graph_manager/job_paths.py
"""
Where a graph job's figures go (owner ruling 2026-10-18):

    one run in the job        <project>/<paths.model_graphs of that run>/job_NNN_<name>/
                              (artifacts/<env>/generated_models/<run>/graphs/job_NNN_<name>/)
    several runs, or none     <project>/<paths.graphs>/jobs/job_NNN_<name>/

A job's runs are its `models` (job 0: its `top_models`, written by the model judge), each a
model_name or a model_hash; a hash counts as its run's name (artifacts/model_name_map.json).

A job's folder is job_NNN_<name>. When that folder already exists and is not the job's (a deleted job
left it, or a job moving to a new id or a new place meets one), the job takes the next free name, the
way a file manager names a second copy: acc, acc_1, acc_2 (free_job_name); the folder there keeps its
name and its files. A folder moved aside twice on one day is indexed the same way (free_path).

Every reader and writer of a job's folder asks here: graph_manager (drawing), graph_export (delete-job,
close-gap and the id shift, and a job whose run count crosses one), faceting (panel addresses) and the
panel's layout (FIGURES).

Author: Shraddha Mishra
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Optional

DEFAULT_JOB_ID = 0
DEFAULT_MODEL_GRAPHS = "artifacts/{{env}}/generated_models/{{model_name}}/graphs"
_PLACEHOLDER = re.compile(r"\{\{\s*(env|model_name)\s*\}\}")


def safe_name(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unnamed"


def render_template(text: Any, context: Mapping[str, Any] | None = None) -> str:
    """
    Render simple placeholders used by graph_config.yaml.

    Supports both forms:
      {{process}}
      {process}

    Missing keys are left unchanged instead of crashing.
    """

    out = "" if text is None else str(text)
    context = context or {}

    for key, value in context.items():
        value_text = "" if value is None else str(value)
        out = out.replace("{{" + str(key) + "}}", value_text)
        out = out.replace("{" + str(key) + "}", value_text)

    return out


def folder_name(job: Mapping[str, Any]) -> str:
    """job_NNN_<name>: the job's name with its {{id}}, {{name}}, {{metric}}, {{type}} filled."""
    context = {"id": job.get("id"), "name": job.get("name"), "metric": job.get("metric"), "type": job.get("type")}
    return f"job_{int(job['id']):03d}_{safe_name(render_template(str(job.get('name') or ''), context))}"


def run_names(job: Mapping[str, Any], name_map: Optional[Mapping[str, str]] = None) -> list[str]:
    """The job's runs as names, in order, each once (job 0: its top_models)."""
    raw = job.get("top_models") if str(job.get("id")) == str(DEFAULT_JOB_ID) else job.get("models")
    names: list[str] = []
    for ref in raw or []:
        text = str(ref).strip()
        name = str((name_map or {}).get(text, text))
        if text and name not in names:
            names.append(name)
    return names


def run_graphs_dir(project_root: Path, env: str, run: str, model_graphs: Optional[str] = None) -> Path:
    """The run's paths.model_graphs folder (artifacts/<env>/generated_models/<run>/graphs by default)."""
    rel = _PLACEHOLDER.sub(lambda m: env if m.group(1) == "env" else run, str(model_graphs or DEFAULT_MODEL_GRAPHS))
    path = Path(rel)
    return path if path.is_absolute() else Path(project_root) / path


def job_dir(job: Mapping[str, Any], *, project_root: Path, graphs_root: Path, env: str,
            model_graphs: Optional[str] = None, name_map: Optional[Mapping[str, str]] = None) -> Path:
    """The folder of the job's figures: its one run's graphs folder (not with folder: project), else the project's
    graphs/jobs/."""
    runs = run_names(job, name_map)
    if len(runs) == 1 and str(job.get("folder") or "run") != "project":
        return run_graphs_dir(project_root, env, runs[0], model_graphs) / folder_name(job)
    return Path(graphs_root) / "jobs" / folder_name(job)


def indexed(name: str, taken) -> str:
    """`name`, or the first of name_1, name_2, ... that `taken` refuses, the way a file manager saves a
    second accuracy.pdf as accuracy_1.pdf: a folder already there (a deleted job's, say) keeps its name."""
    if not taken(name):
        return name
    index = 1
    while taken(f"{name}_{index}"):
        index += 1
    return f"{name}_{index}"


def free_job_name(job: Mapping[str, Any], *, exists=lambda path: Path(path).exists(), **where: Any) -> str:
    """The job's name, indexed (indexed()) while its folder (job_dir(), `where` its arguments) exists."""
    base = str(job.get("name") or "")
    return indexed(base, lambda name: exists(job_dir({**job, "name": name}, **where)))


def free_path(path: Path) -> Path:
    """`path`, or path_1, path_2, ... when it exists (a folder moved aside twice on one day)."""
    path = Path(path)
    return path.with_name(indexed(path.name, lambda name: path.with_name(name).exists()))


# ── facet folders ───────────────────────────────────────────────────────────────
# A facet draws into <project>/<paths.graphs>/facets/facet_NNN_<name>/ (faceting.py render_facet).

def facet_safe(value: Any) -> str:
    text = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value).strip())
    return text.strip("_") or "facet"


def facet_folder_name(facet: Mapping[str, Any]) -> str:
    """facet_NNN_<name>: its name as faceting.py validate_facet reads it (facet_<id> when it has none)."""
    facet_id = int(facet["id"])
    name = str(facet.get("name", f"facet_{facet_id}")).strip() or f"facet_{facet_id}"
    return f"facet_{facet_id:03d}_{facet_safe(name)}"


# ── facet addresses ─────────────────────────────────────────────────────────────
# A facet panel's address names a drawn file: relative to <project>/graphs/ for a job of several runs
# (jobs/job_NNN_<name>/...), relative to the project for a job of one run, whose files are in the run's
# folder (artifacts/<env>/generated_models/<run>/graphs/job_NNN_<name>/...).
RUN_ADDRESS = "artifacts/"
_JOB_IN_ADDRESS = re.compile(r"(?:^|/)(job_(\d{3})_[^/]*)/")


def address_file(address: str, *, project_root: Path, graphs_root: Path) -> Path:
    """The file a facet address names."""
    text = str(address).strip().lstrip("/")
    return (Path(project_root) if text.startswith(RUN_ADDRESS) else Path(graphs_root)) / text


def file_address(file: Path, *, project_root: Path, graphs_root: Path) -> Optional[str]:
    """The facet address of a drawn file; None outside both roots."""
    file = Path(file).resolve()
    for root in (Path(graphs_root).resolve(), Path(project_root).resolve() / RUN_ADDRESS.rstrip("/")):
        try:
            rel = file.relative_to(root).as_posix()
        except ValueError:
            continue
        return rel if root == Path(graphs_root).resolve() else f"{RUN_ADDRESS}{rel}"
    return None


def address_job(address: str) -> Optional[tuple[str, int]]:
    """(job folder name, job id) of the job folder an address points into, if any."""
    m = _JOB_IN_ADDRESS.search(str(address))
    return (m.group(1), int(m.group(2))) if m else None


__all__ = ["DEFAULT_MODEL_GRAPHS", "RUN_ADDRESS", "address_file", "address_job", "file_address", "folder_name",
           "facet_folder_name", "facet_safe", "free_job_name", "free_path", "indexed", "job_dir", "render_template", "run_graphs_dir", "run_names",
           "safe_name"]
