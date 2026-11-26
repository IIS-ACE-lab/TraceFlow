# services/side_channel_core/utils/heartbeat.py
"""
Heartbeat of the running experiment, for the TraceFlow Monitor.

One file per project and env, ``projects/<P>/artifacts/<env>/heartbeat.json``,
rewritten atomically every few seconds by a daemon thread while a run trains
and tests. The panel lights green while the file is fresh, amber when a beat
is late and red when a running experiment has stopped beating, so a stuck or
killed process shows even though nothing reported it.

    begin(cfg, project_root, env)   the orchestrator, before train_and_test()
    update(**fields)                the training loop and the test (cheap: a
                                    dict update; the thread does the writing)
    finish(status)                  the orchestrator: done / failed / interrupted

``progress`` is the fraction of the run's work done: every fold's epochs, the
refit's epochs and the test, each epoch counted by its finished batches.
A write that fails is logged once and never reaches the training.

Author: Shraddha Mishra
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from utils.atomic_io import atomic_write_text
from utils.logger import get_logger

logger = get_logger("Heartbeat")

INTERVAL_S = 5.0
FILE_NAME = "heartbeat.json"


def heartbeat_path(project_root: str | Path, env: str) -> Path:
    return Path(project_root) / "artifacts" / str(env or "dev") / FILE_NAME


def progress_fraction(fields: Mapping[str, Any]) -> float:
    """Work done / work planned, in [0, 1].

    Units are epochs: n_folds x fold epochs + refit epochs + 1 for the test
    (without cross-validation: epochs + 1; no refit epochs when refit is false). The current epoch counts its
    finished batches. The refit's epochs are known only when it starts; until
    then they are taken as one fold's.
    """
    if fields.get("status") == "done":
        return 1.0
    n_folds = int(fields.get("n_folds") or 0)
    fold_epochs = int(fields.get("fold_epochs") or fields.get("epochs") or 0)
    refit_epochs = int(fields.get("refit_epochs") or fields.get("epochs") or (fold_epochs if n_folds else 0))
    if n_folds and fields.get("refit") is False:
        refit_epochs = 0
    if fold_epochs <= 0 and refit_epochs <= 0:
        return 0.0
    total = n_folds * fold_epochs + refit_epochs + 1
    epoch = int(fields.get("epoch") or 0)
    batch = int(fields.get("batch") or 0)
    per_epoch = int(fields.get("batches_per_epoch") or 0)
    current = max(epoch - 1, 0) + (min(batch, per_epoch) / per_epoch if per_epoch > 0 and epoch > 0 else 0.0)
    phase = fields.get("phase")
    if phase == "fold":
        done = (max(int(fields.get("fold") or 1), 1) - 1) * fold_epochs + current
    elif phase in ("refit", "train"):
        done = n_folds * fold_epochs + current
    elif phase == "test":
        done = n_folds * fold_epochs + refit_epochs
    else:
        done = 0.0
    return max(0.0, min(done / total, 1.0))


class Heartbeat:
    """The beating file of one run."""

    def __init__(self, path: Path, fields: Mapping[str, Any], interval: float = INTERVAL_S):
        self.path = Path(path)
        self.interval = float(interval)
        self.fields: dict[str, Any] = dict(fields)
        self.fields.setdefault("status", "running")
        self.fields.setdefault("started_ns", time.time_ns())
        self.fields["pid"] = os.getpid()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = False
        self._progress = 0.0                          # never shown going back (the refit estimate may be high)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = dict(self.fields)
        self._progress = max(self._progress, round(progress_fraction(data), 6))
        data["progress"] = self._progress
        data["written_ns"] = time.time_ns()
        return data

    def write(self) -> None:
        try:
            # Read-only for everyone (0444): editors open it read-only and a user cannot save
            # over it; only this writer replaces it, by rename (root can still override).
            atomic_write_text(self.path, json.dumps(self.snapshot(), sort_keys=True), mode=0o444)
        except Exception as exc:                       # never reaches the training
            if not self._warned:
                logger.warning("Heartbeat not written to %s: %s", self.path, exc)
                self._warned = True

    def _beat(self) -> None:
        while not self._stop.wait(self.interval):
            self.write()

    def start(self) -> "Heartbeat":
        self.write()
        self._thread = threading.Thread(target=self._beat, name="traceflow-heartbeat", daemon=True)
        self._thread.start()
        return self

    def update(self, **fields: Any) -> None:
        with self._lock:
            self.fields.update(fields)

    def finish(self, status: str) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1.0)
        self.update(status=status)
        self.write()


_active: Heartbeat | None = None


def begin(cfg: Mapping[str, Any], project_root: str | Path, env: str, **extra: Any) -> Heartbeat:
    """Start the heartbeat of one run (ends any previous one as 'interrupted')."""
    global _active
    if _active is not None:
        _active.finish("interrupted")
    cv = cfg.get("cross_validation") or {}
    n_folds = int(cv.get("n_folds") or 0) if isinstance(cv, Mapping) and cv.get("enabled") is True else 0
    data = cfg.get("data") or {}
    project = cfg.get("project") or {}
    fields = {
        "project": project.get("name") if isinstance(project, Mapping) else None,
        "env": env,
        "model_name": cfg.get("model_name"),
        "model_hash": cfg.get("model_hash"),
        "model_id": cfg.get("model_id"),
        "dataset": data.get("name") if isinstance(data, Mapping) else None,
        "n_folds": n_folds,
        # cross_validation.refit.enabled: false trains no final model, so its epochs are not planned
        "refit": not (n_folds and isinstance(cv.get("refit"), Mapping) and cv["refit"].get("enabled") is False),
        # early stopping may end the run before its planned epochs: an ETA is then an upper bound
        "early_stopping": bool((cfg.get("trainer") or {}).get("early_stopping")) if isinstance(cfg.get("trainer"), Mapping) else False,
        "phase": "starting",
        **extra,
    }
    _active = Heartbeat(heartbeat_path(project_root, env), fields).start()
    return _active


def update(**fields: Any) -> None:
    """Update the active run's state; a no-op when no heartbeat runs."""
    if _active is not None:
        _active.update(**fields)


def finish(status: str) -> None:
    """Final beat with ``status``; a no-op when no heartbeat runs."""
    global _active
    if _active is not None:
        _active.finish(status)
        _active = None
