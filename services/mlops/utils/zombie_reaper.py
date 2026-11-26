# services/mlops/utils/zombie_reaper.py
"""
Runtime hygiene for long experiment chains.

One orchestrator process trains and tests hundreds of models in sequence.
Anything a finished (or crashed) trial leaves behind degrades the next one:
exited children that were never reaped, DataLoader workers that outlived
their loader, HDF5 handles that still hold file locks, cached CUDA blocks,
or stale GPU contexts of a crashed run. This module removes those leftovers
without ever pausing the chain: every step is bounded and non-blocking.

Two entry points:

    prepare_trial_runtime(logger=...)   before a trial (orchestrator loop)
    cleanup_trial_runtime(logger=...)   after a trial (train/test ``finally``)

Contract: never raises, never waits for the GPU, and only kills processes
that this process created or that are provably orphaned TraceFlow runs.
Without ``logger`` the module is silent; every function returns a small
stats dict.

Author: Shraddha Mishra
"""

from __future__ import annotations

import gc
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # optional; the /proc fallback below covers containers without it
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

# Children that legitimately outlive a trial and must be left alone.
_KEEP_CHILD_MARKERS = ("multiprocessing.resource_tracker",)
# Only these command lines may be killed when found orphaned on the GPU.
_TRACEFLOW_MARKERS = ("core.cli", "side_channel_core", "orchestrate-cmd", "train-cmd", "test-cmd")

LEAK_WARN_MIB = 64
_NVSMI_TIMEOUT_S = 10


def _log(logger, level: str, msg: str, *args) -> None:
    if logger is None:
        return
    try:
        getattr(logger, level)(msg, *args)
    except Exception:
        pass


# =============================================================================
# Process helpers
# =============================================================================
def reap_zombie_children(*, max_reaps: int = 1024) -> int:
    """Reap exited children without blocking. Returns the number reaped."""
    reaped = 0
    for _ in range(int(max_reaps)):
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except (ChildProcessError, OSError):
            break
        if pid == 0:
            break
        reaped += 1
    return reaped


def _proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except Exception:
        return ""


def _proc_ppid(pid: int) -> Optional[int]:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except Exception:
        return None
    return None


def _descendant_pids() -> List[int]:
    """PIDs of every live descendant of this process (children first)."""
    me = os.getpid()
    if psutil is not None:
        try:
            return [p.pid for p in psutil.Process(me).children(recursive=True)]
        except Exception:
            pass
    by_parent: Dict[int, List[int]] = {}
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            ppid = _proc_ppid(int(entry))
            if ppid is not None:
                by_parent.setdefault(ppid, []).append(int(entry))
    except Exception:
        return []
    out: List[int] = []
    stack = [me]
    while stack:
        for child in by_parent.get(stack.pop(), []):
            out.append(child)
            stack.append(child)
    return out


def _ancestor_pids() -> set:
    out, pid = set(), os.getppid()
    while pid and pid > 1 and pid not in out:
        out.add(pid)
        pid = _proc_ppid(pid) or 0
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie answers kill(0); check its state
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("State:"):
                return "Z" not in line.split()[1]
    except Exception:
        return False
    return True


def _terminate(pids: List[int], *, grace_s: float, logger, what: str) -> Dict[str, int]:
    stats = {"terminated": 0, "killed": 0}
    targets = [p for p in pids if _alive(p)]
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
            stats["terminated"] += 1
            _log(logger, "warning", "[reaper] SIGTERM %s pid=%d :: %s", what, pid, _proc_cmdline(pid)[:160])
        except Exception:
            pass
    deadline = time.time() + grace_s
    while time.time() < deadline and any(_alive(p) for p in targets):
        time.sleep(0.2)
    for pid in targets:
        if _alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                stats["killed"] += 1
                _log(logger, "warning", "[reaper] SIGKILL %s pid=%d", what, pid)
            except Exception:
                pass
    reap_zombie_children()
    return stats


def terminate_orphaned_workers(*, grace_s: float = 5.0, logger=None) -> Dict[str, int]:
    """
    Kill children of this process that outlived their trial.

    After a trial no DataLoader worker (or other helper child) of this process
    should be alive; trials never run as concurrent child processes. The
    multiprocessing resource tracker is kept.
    """
    victims = [
        pid for pid in _descendant_pids()
        if _alive(pid) and not any(m in _proc_cmdline(pid) for m in _KEEP_CHILD_MARKERS)
    ]
    if not victims:
        return {"terminated": 0, "killed": 0}
    return _terminate(victims, grace_s=grace_s, logger=logger, what="orphaned worker")


# =============================================================================
# HDF5 and CUDA hygiene
# =============================================================================
def close_leaked_hdf5_handles(*, logger=None) -> int:
    """Close h5py files still open in this process (their locks block the next split writer)."""
    h5py = sys.modules.get("h5py")
    if h5py is None:
        return 0
    closed = 0
    for obj in list(gc.get_objects()):
        try:
            # type(obj), not isinstance: isinstance reads obj.__class__, which makes
            # deprecated module attributes (torch.distributed.reduce_op) warn.
            if issubclass(type(obj), h5py.File) and obj.id.valid:
                name = obj.filename
                obj.close()
                closed += 1
                _log(logger, "warning", "[reaper] closed leaked HDF5 handle: %s", name)
        except Exception:
            pass
    return closed


def release_cuda(*, logger=None) -> Dict[str, Any]:
    """Return cached CUDA blocks to the driver and report what is still allocated."""
    stats: Dict[str, Any] = {"cuda_cleaned": False, "allocated_mib": 0.0, "reserved_mib": 0.0}
    torch = sys.modules.get("torch")
    if torch is None:
        try:
            import torch  # type: ignore
        except Exception:
            return stats
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return stats
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        stats["cuda_cleaned"] = True
        stats["allocated_mib"] = torch.cuda.memory_allocated() / 2**20
        stats["reserved_mib"] = torch.cuda.memory_reserved() / 2**20
        torch.cuda.reset_peak_memory_stats()
        if stats["allocated_mib"] > LEAK_WARN_MIB:
            _log(
                logger, "warning",
                "[reaper] %.0f MiB of CUDA tensors are still referenced after cleanup; "
                "a previous trial leaked model/optimizer state",
                stats["allocated_mib"],
            )
    except Exception as exc:
        _log(logger, "warning", "[reaper] CUDA cleanup skipped: %s", exc)
    return stats


# =============================================================================
# GPU status
# =============================================================================
def _nvidia_smi(args: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=_NVSMI_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return out.stdout if out.returncode == 0 else None


def gpu_status() -> Dict[str, Any]:
    """
    One dict per call: responsive, temperature_c, used_mib, total_mib,
    utilization_pct, throttle_reasons (hex), compute_pids. Unknown fields are None.
    """
    status: Dict[str, Any] = {
        "responsive": False, "temperature_c": None, "used_mib": None, "total_mib": None,
        "utilization_pct": None, "throttle_reasons": None, "compute_pids": [],
    }
    raw = _nvidia_smi([
        "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu,clocks_throttle_reasons.active",
        "--format=csv,noheader,nounits",
    ])
    if raw is None:
        return status
    first = raw.strip().splitlines()[0] if raw.strip() else ""
    parts = [p.strip() for p in first.split(",")]
    if len(parts) >= 5:
        def num(v):
            try:
                return float(v)
            except ValueError:
                return None
        status.update({
            "responsive": True,
            "temperature_c": num(parts[0]), "used_mib": num(parts[1]), "total_mib": num(parts[2]),
            "utilization_pct": num(parts[3]), "throttle_reasons": parts[4],
        })
    apps = _nvidia_smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]) or ""
    for line in apps.strip().splitlines():
        pid = line.split(",")[0].strip()
        if pid.isdigit():
            status["compute_pids"].append(int(pid))
    return status


def kill_stale_gpu_processes(*, grace_s: float = 5.0, logger=None) -> Dict[str, int]:
    """
    Kill GPU contexts left by crashed TraceFlow runs.

    A process is killed only when nvidia-smi lists it on the GPU, it is not
    this process, an ancestor or a descendant, its parent is init (orphaned),
    and its command line is a TraceFlow entry point. Anything else on the GPU
    (another user's job, a notebook) is reported, never touched.
    """
    me, ancestors, descendants = os.getpid(), _ancestor_pids(), set(_descendant_pids())
    stale, foreign = [], []
    for pid in gpu_status()["compute_pids"]:
        if pid == me or pid in ancestors or pid in descendants or not _alive(pid):
            continue
        cmd = _proc_cmdline(pid)
        if _proc_ppid(pid) == 1 and any(m in cmd for m in _TRACEFLOW_MARKERS):
            stale.append(pid)
        else:
            foreign.append(pid)
    if foreign:
        _log(logger, "warning", "[reaper] other processes hold the GPU and are left alone: %s", foreign)
    if not stale:
        return {"terminated": 0, "killed": 0}
    return _terminate(stale, grace_s=grace_s, logger=logger, what="stale GPU context")


# =============================================================================
# Entry points
# =============================================================================
def cleanup_trial_runtime(*, cuda: bool = True, reap: bool = True, logger=None) -> Dict[str, Any]:
    """Per-trial cleanup after training or testing. Never raises."""
    stats: Dict[str, Any] = {
        "reaped_children": 0, "gc_collected": 0, "cuda_cleaned": False,
        "workers_terminated": 0, "workers_killed": 0, "hdf5_closed": 0,
    }
    try:
        stats["hdf5_closed"] = close_leaked_hdf5_handles(logger=logger)
    except Exception:
        pass
    try:
        stats["gc_collected"] = int(gc.collect())
    except Exception:
        pass
    if cuda:
        stats.update(release_cuda(logger=logger))
    if reap:
        try:
            worker_stats = terminate_orphaned_workers(logger=logger)
            stats["workers_terminated"] = worker_stats["terminated"]
            stats["workers_killed"] = worker_stats["killed"]
        except Exception:
            pass
        stats["reaped_children"] = reap_zombie_children()
    return stats


def prepare_trial_runtime(*, cuda: bool = True, logger=None) -> Dict[str, Any]:
    """
    Make the process and the GPU ready for the next trial. Never raises.

    Order: reap and kill leftovers of the previous trial, close leaked HDF5
    handles, kill orphaned TraceFlow GPU contexts, release cached CUDA memory,
    then log one line with the GPU state. Nothing here waits.
    """
    stats: Dict[str, Any] = {}
    try:
        stats.update(cleanup_trial_runtime(cuda=cuda, reap=True, logger=logger))
        gpu_stats = kill_stale_gpu_processes(logger=logger)
        stats["stale_gpu_terminated"] = gpu_stats["terminated"]
        stats["stale_gpu_killed"] = gpu_stats["killed"]
        if cuda:
            status = gpu_status()
            stats["gpu"] = status
            if not status["responsive"]:
                _log(logger, "error", "[reaper] nvidia-smi did not answer; starting the trial anyway")
            _log(
                logger, "info",
                "[reaper] GPU state: temperature=%s C used=%s/%s MiB utilization=%s%% | "
                "reaped=%d workers_killed=%d stale_gpu_killed=%d hdf5_closed=%d",
                status.get("temperature_c"), status.get("used_mib"), status.get("total_mib"),
                status.get("utilization_pct"),
                stats["reaped_children"], stats["workers_terminated"] + stats["workers_killed"],
                stats["stale_gpu_terminated"] + stats["stale_gpu_killed"], stats["hdf5_closed"],
            )
    except Exception as exc:
        _log(logger, "warning", "[reaper] pre-trial preparation incomplete: %s", exc)
    return stats


__all__ = [
    "cleanup_trial_runtime",
    "prepare_trial_runtime",
    "reap_zombie_children",
    "terminate_orphaned_workers",
    "close_leaked_hdf5_handles",
    "release_cuda",
    "gpu_status",
    "kill_stale_gpu_processes",
]
