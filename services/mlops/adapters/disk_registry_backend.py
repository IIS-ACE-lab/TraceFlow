# services/mlops/adapters/disk_registry_backend.py
"""# ============================================================
# Disk backend (atomic writes, lock, corruption-safe)
# Matches services/mlops/core/registry.py::RegistryBackend
Author: Shraddha Mishra
# ============================================================"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from services.mlops.core.registry import RegistryBackend, Json
from services.side_channel_core.utils.seal import check as check_seal, seal_text, strip as strip_seal



@dataclass(frozen=True)
class DiskRegistryBackend(RegistryBackend):
    """
    Persistence for registries (Phase 3):
      - namespace -> file mapping: <base_dir>/<namespace>.json
      - atomic writes: temp -> fsync -> replace
      - backup rotation: <file>.bak
      - corruption-safe read: main -> .bak -> None
      - cross-process best-effort lock: <base_dir>/.registry.lock

    Contract with core:
      - load(namespace) returns Optional[Json]; None if missing.
      - save(namespace, state) persists full snapshot dict.
    """

    base_dir: Path

    # filesystem + safety
    lock_filename: str = ".registry.lock"
    file_ext: str = ".json"

    lock_timeout_s: float = 30.0
    lock_poll_s: float = 0.05

    # payload wrapper (helps future migrations)
    wrapper_schema: str = "mlops.registry.disk.v1"

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # RegistryBackend API
    # ---------------------------

    def load(self, *, namespace: str) -> Optional[Json]:
        ns = self._sanitize_namespace(namespace)
        path = self._path_for(ns)

        with self._lock():
            # 1) main
            data = self._try_read(path)
            if data is not None:
                return data

            # 2) backup
            bak = self._bak_path_for(path)
            data = self._try_read(bak)
            if data is not None:
                return data

            # 3) not found / unreadable
            return None

    def save(self, *, namespace: str, state: Json) -> None:
        ns = self._sanitize_namespace(namespace)
        path = self._path_for(ns)

        if not isinstance(state, dict):
            raise TypeError(f"[DiskRegistryBackend] state must be dict, got {type(state)}")

        self.base_dir.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "schema": self.wrapper_schema,
            "ts": time.time(),
            "namespace": ns,
            "state": state,
        }

        # utils/seal.py: the state file carries its own seal
        encoded = seal_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

        tmp = self._tmp_path_for(path)
        bak = self._bak_path_for(path)

        with self._lock():
            # 1) write temp + fsync file
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="utf-8") as f:
                f.write(encoded)
                f.flush()
                os.fsync(f.fileno())

            # 2) move current -> .bak (best-effort; never blocks save)
            try:
                if path.exists():
                    os.replace(path, bak)
            except Exception:
                pass

            # 3) promote temp -> final (atomic replace)
            os.replace(tmp, path)

            # 4) fsync directory entry (best-effort)
            self._fsync_dir(path.parent)

    # ---------------------------
    # Paths / names
    # ---------------------------

    def _sanitize_namespace(self, namespace: str) -> str:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("[DiskRegistryBackend] namespace must be non-empty string")
        ns = namespace.strip()

        # harden: namespace must be a safe filename stem
        # allow letters, numbers, _, -, .
        for ch in ns:
            ok = ch.isalnum() or ch in ("_", "-", ".")
            if not ok:
                raise ValueError(f"[DiskRegistryBackend] invalid namespace '{namespace}' (bad char: {ch!r})")
        return ns

    def _path_for(self, namespace: str) -> Path:
        return (self.base_dir / f"{namespace}{self.file_ext}").resolve()

    def _bak_path_for(self, path: Path) -> Path:
        return path.with_suffix(path.suffix + ".bak")

    def _tmp_path_for(self, path: Path) -> Path:
        return path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")

    # ---------------------------
    # Reading (corruption safe)
    # ---------------------------

    def _try_read(self, path: Path) -> Optional[Json]:
        if not path.exists():
            return None
        if path.suffix == ".json":
            # A changed state file is refused, not replaced by its backup.
            check_seal(path, cache=False)
        try:
            raw = path.read_text(encoding="utf-8")
            obj = strip_seal(json.loads(raw))

            # preferred: wrapper payload
            if isinstance(obj, dict) and "state" in obj and isinstance(obj["state"], dict):
                return obj["state"]

            # legacy: direct dict
            if isinstance(obj, dict):
                return obj

            return None
        except Exception:
            return None

    # ---------------------------
    # Locking (best-effort cross-process)
    # ---------------------------

    @contextmanager
    def _lock(self):
        """
        Best-effort lock with lock file.

        - Unix: fcntl.flock
        - Windows: msvcrt.locking
        - Fallback: proceed without OS lock (still serialized per-process by context manager)
        """
        lock_path = (self.base_dir / self.lock_filename).resolve()
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        deadline = time.time() + float(self.lock_timeout_s)
        fh = None
        locked = False

        while True:
            try:
                fh = lock_path.open("a+", encoding="utf-8")
                locked = self._try_os_lock(fh)
                if locked:
                    break
            except Exception:
                locked = False

            if time.time() >= deadline:
                # do not deadlock system; proceed without lock
                break
            time.sleep(float(self.lock_poll_s))

        try:
            yield
        finally:
            try:
                if fh is not None and locked:
                    self._try_os_unlock(fh)
                if fh is not None:
                    fh.close()
            except Exception:
                pass

    @staticmethod
    def _try_os_lock(fh) -> bool:
        # Unix
        try:
            import fcntl  # type: ignore

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except Exception:
            pass

        # Windows
        try:
            import msvcrt  # type: ignore

            # lock 1 byte (advisory)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except Exception:
            pass

        # No OS locking available; treat as locked (best-effort)
        return True

    @staticmethod
    def _try_os_unlock(fh) -> None:
        # Unix
        try:
            import fcntl  # type: ignore

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            return
        except Exception:
            pass

        # Windows
        try:
            import msvcrt  # type: ignore

            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            return
        except Exception:
            pass

    @staticmethod
    def _fsync_dir(dir_path: Path) -> None:
        """
        Best-effort fsync of directory entry to reduce risk of losing rename on crash.
        """
        try:
            fd = os.open(str(dir_path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            return
