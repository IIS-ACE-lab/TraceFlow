# utils/atomic_io.py
"""
Atomic text writes for the project records under artifacts/:

    run_registry.json, test_registry.json, run_map.jsonl,
    model_name_map.json, done_model_names.yaml

The text goes to a unique temp file next to the target (so concurrent writers
never share one), is fsynced, and is published with os.replace; the directory
is fsynced after the rename. A reader sees the old file or the new one, never
a partial one, and a crash before the rename leaves the old file and no temp
file behind. A YAML, JSON or JSONL file is sealed on the way (utils/seal.py);
a file the user edits by hand, such as a config file, passes seal=False.

Author: Shraddha Mishra
"""
from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from .seal import seal_text


def atomic_write_text(path: str | Path, text: str, mode: int | None = None, *, seal: bool = True) -> Path:
    """Write text to path atomically. mode: the permission bits the file is published with
    (set before the rename, so it never exists with other bits); None keeps the umask.
    seal: sign a YAML, JSON or JSONL file in its own text (utils/seal.py)."""
    path = Path(path)
    if seal:
        text = seal_text(path, text)
    path.parent.mkdir(parents=True, exist_ok=True)
    # open("x") follows the umask; tempfile.mkstemp() would publish mode 0600.
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            if mode is not None:
                os.fchmod(handle.fileno(), mode)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass
    return path


__all__ = ["atomic_write_text"]
