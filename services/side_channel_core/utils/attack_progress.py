# services/side_channel_core/utils/attack_progress.py
"""
The test's progress while it attacks: attack i of n, and the attack metrics as they stand after i attacks.

The test attacks its traces perform_attack.test.attack_times times; with 22,500 traces (AES_HD, AES_RD) that
takes a while, and nothing else moves meanwhile. While epoch_utils computes the test's metrics it activates
a writer (reporting()); sca/perform_attack.perform_attacks reports after each test attack (report()), and the
writer rewrites artifacts/<env>/attack_progress.json beside the heartbeat, sealed as every artifact JSON:

    {"model_name", "model_hash", "attack": 37, "attacks": 100, "running": true, "started_ns", "written_ns",
     "metrics": {"mean_attack_rank_curve": 61.2, "traces_to_ge_1": null,
                 "per_attack_traces_to_ge_1": {"value": 342.0, "reached": 30, "of": 37}, ...}}

core/monitor.py shows it under the running run (the TraceFlow Monitor, the status page, the terminal); started_ns,
the time the test began, gives the panel's ETA the test's own speed (state.js eta).

The live values are the run's own attack metrics over the attacks done so far, from the same per-attack rank
curves and the same stable-crossing rule as sca/metrics.py, so after the last attack they are the values the
test records: mean_attack_rank_curve and guessing_entropy the mean final rank, final_ranks their mean,
traces_to_ge_1 the stable crossing of the mean curve, per_attack_traces_to_ge_1 the mean of each attack's own
crossing over the attacks that reach rank 1. Only the metrics in metrics_to_log are reported.

A display only: no result, no history row and no hash reads it; a failed write is logged once and ignored.

Author: Shraddha Mishra
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np

from utils.atomic_io import atomic_write_text
from utils.logger import get_logger

logger = get_logger("attack_progress")

FILE_NAME = "attack_progress.json"
LIVE_METRICS = ("mean_attack_rank_curve", "guessing_entropy", "final_ranks", "traces_to_ge_1",
                "per_attack_traces_to_ge_1")


def progress_path(run_dir: str | Path) -> Path:
    """artifacts/<env>/attack_progress.json for a run folder artifacts/<env>/generated_models/<name>."""
    return Path(run_dir).parent.parent / FILE_NAME


class _Writer:
    """The running sums of one test's attacks and the file they are written to."""

    def __init__(self, path: Path, *, model_name: Any, model_hash: Any, metrics: Sequence[str]):
        self.path = path
        self.head = {"model_name": model_name, "model_hash": model_hash, "started_ns": time.time_ns()}
        self.metrics = [m for m in LIVE_METRICS if m in set(metrics)]
        self.sum: Optional[np.ndarray] = None          # the summed rank curve of the attacks so far
        self.crossings: list[Optional[float]] = []     # each attack's own stable crossing of rank 1
        self.finals: list[float] = []
        self._warned = False

    def add(self, curve: np.ndarray) -> None:
        from sca.metrics import _stable_threshold_trace

        curve = np.asarray(curve, dtype=np.float64)
        self.sum = curve[:, 1].copy() if self.sum is None else self.sum + curve[:, 1]
        self.axis = curve[:, 0]
        self.finals.append(float(curve[-1, 1]))
        if "per_attack_traces_to_ge_1" in self.metrics:
            self.crossings.append(_stable_threshold_trace(curve, threshold=1.0, direction="le"))

    def values(self) -> dict[str, Any]:
        from sca.metrics import _stable_threshold_trace

        k = len(self.finals)
        out: dict[str, Any] = {}
        mean_final = float(np.mean(self.finals))
        for name in self.metrics:
            if name in ("mean_attack_rank_curve", "guessing_entropy", "final_ranks"):
                out[name] = mean_final
            elif name == "traces_to_ge_1":
                mean_curve = np.column_stack([self.axis, self.sum / k])
                out[name] = _stable_threshold_trace(mean_curve, threshold=1.0, direction="le")
            elif name == "per_attack_traces_to_ge_1":
                reached = [float(c) for c in self.crossings if c is not None]
                out[name] = {"value": float(np.mean(reached)) if reached else None, "reached": len(reached), "of": k}
        return out

    def write(self, done: int, total: int) -> None:
        body = {**self.head, "attack": int(done), "attacks": int(total), "running": done < total,
                "written_ns": time.time_ns(), "metrics": self.values()}
        try:
            atomic_write_text(self.path, json.dumps(body, sort_keys=True), mode=0o444)
        except Exception as exc:                       # a display: it never stops the test
            if not self._warned:
                logger.warning("Test progress not written to %s: %s", self.path, exc)
                self._warned = True


_active: Optional[_Writer] = None


@contextmanager
def reporting(cfg: Mapping[str, Any], run_dir: str | Path) -> Iterator[None]:
    """Report the test attacks made inside this block (epoch_utils, around the test's compute_metrics)."""
    global _active
    if Path(run_dir).parent.name != "generated_models":     # not a run folder (artifacts/<env>/generated_models/<name>)
        yield
        return
    metrics = cfg.get("metrics_to_log") or cfg.get("metrics") or []
    _active = _Writer(progress_path(run_dir), model_name=cfg.get("model_name"), model_hash=cfg.get("model_hash"),
                      metrics=[str(m) for m in metrics])
    try:
        yield
    finally:
        _active = None


def report(process: Optional[str], done: int, total: int, curve: np.ndarray) -> None:
    """One test attack done (sca/perform_attack.perform_attacks): its rank curve joins the sums and the file
    is rewritten. A no-op outside reporting() and for any process but the test."""
    if _active is None or process != "test":
        return
    _active.add(curve)
    _active.write(done, total)


__all__ = ["FILE_NAME", "LIVE_METRICS", "progress_path", "reporting", "report"]
