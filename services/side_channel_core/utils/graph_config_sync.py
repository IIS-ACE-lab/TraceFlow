# services/side_channel_core/utils/graph_config_sync.py
"""
Generated-model bookkeeping outside of the metric artifacts:

    projects/<project>/artifacts/done_model_names.yaml
        every generated model, appended when its training completes

    projects/<project>/config/graph_config.yaml
        jobs: - id: 0 - top_models: the model-judge ranking (make judge)

Only the one list block of graph job 0 is rewritten, line by line, so the rest
of the file (comments, other jobs) is left untouched. Each edit is verified by
re-parsing before an atomic replace, and serialized with a lock file so
concurrent runs cannot lose each other's entries.

Author: Shraddha Mishra
"""

from __future__ import annotations

import fcntl
import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from services.side_channel_core.utils.atomic_io import atomic_write_text
from services.side_channel_core.utils.seal import load_yaml


DEFAULT_JOB_ID = 0
_JOB_START = re.compile(r"^(\s*)-\s+id:\s*(\S+)\s*(#.*)?$")


def _default_job_span(lines: list[str]) -> tuple[int, int, int]:
    """(start, end, key_indent) of job id 0; end is exclusive."""
    for i, line in enumerate(lines):
        match = _JOB_START.match(line)
        if not match or str(match.group(2)).strip("'\"") != str(DEFAULT_JOB_ID):
            continue
        dash_indent = len(match.group(1))
        end = len(lines)
        for j in range(i + 1, len(lines)):
            text = lines[j]
            if not text.strip() or text.lstrip().startswith("#"):
                continue
            indent = len(text) - len(text.lstrip())
            if indent <= dash_indent:
                end = j
                break
        # Drop trailing blank/comment lines: they belong to the next job.
        while end > i + 1 and (not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")):
            end -= 1
        return i, end, dash_indent + 2
    raise KeyError(f"graph job id={DEFAULT_JOB_ID} not found")


def _replace_list(lines: list[str], key: str, values: list[str]) -> list[str]:
    start, end, indent = _default_job_span(lines)
    pad = " " * indent
    key_re = re.compile(rf"^{pad}{re.escape(key)}:\s*(\[\s*\])?\s*(#.*)?$")
    block = (
        [f"{pad}{key}:"] + [f'{pad}  - "{v}"' for v in values]
        if values else [f"{pad}{key}: []"]
    )
    for k in range(start + 1, end):
        if key_re.match(lines[k]):
            # Consume the existing items: "- x" at key indent or deeper, and
            # anything indented deeper (commented-out items included).
            stop = k + 1
            while stop < end and lines[stop].strip():
                ind = len(lines[stop]) - len(lines[stop].lstrip())
                if ind > indent or (ind == indent and lines[stop].lstrip().startswith("- ")):
                    stop += 1
                else:
                    break
            return lines[:k] + block + lines[stop:]
    return lines[:end] + block + lines[end:]


def _default_job(text: str) -> dict[str, Any]:
    for job in (yaml.safe_load(text) or {}).get("jobs") or []:
        if isinstance(job, dict) and str(job.get("id")) == str(DEFAULT_JOB_ID):
            return job
    raise KeyError(f"graph job id={DEFAULT_JOB_ID} not found")


def update_default_job_models(
    path: str | Path,
    key: str,
    names: Iterable[str],
    *,
    replace: bool,
) -> list[str]:
    """
    Append (replace=False) or set (replace=True) job 0's `key` list.

    Returns the resulting list.
    """
    path = Path(path)
    names = [str(n).strip() for n in names if str(n).strip()]
    lock_path = path.with_name(f".{path.name}.lock")
    with open(lock_path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = path.read_text(encoding="utf-8")
        current = [str(m) for m in (_default_job(text).get(key) or [])]
        values = names if replace else current + [n for n in names if n not in current]
        if values == current:
            return current
        lines = _replace_list(text.splitlines(), key, list(dict.fromkeys(values)))
        new_text = "\n".join(lines) + "\n"
        written = [str(m) for m in (_default_job(new_text).get(key) or [])]
        if written != list(dict.fromkeys(values)):
            raise RuntimeError(f"graph_config.yaml edit of job 0 '{key}' did not verify: {written}")
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, path)
        return written


MODEL_NAMES_FILE = "done_model_names.yaml"


def append_model_name(project_root: str | Path, model_name: str) -> list[str]:
    """Add a trained model to artifacts/done_model_names.yaml; returns the list."""
    path = Path(project_root) / "artifacts" / MODEL_NAMES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    model_name = str(model_name).strip()
    lock_path = path.with_name(f".{path.name}.lock")
    with open(lock_path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        names = []
        if path.is_file():
            names = [str(n).strip() for n in (load_yaml(path) or []) if str(n).strip()]
        if model_name and model_name not in names:
            names.append(model_name)
            atomic_write_text(
                path,
                "# Models of this project whose training completed; pick the ones a graph job should plot.\n"
                + "".join(f"- {name}\n" for name in names),
            )
        return names


__all__ = ["append_model_name", "update_default_job_models"]
