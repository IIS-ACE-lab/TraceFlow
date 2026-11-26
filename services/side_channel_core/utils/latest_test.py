# services/side_channel_core/utils/latest_test.py
"""
The latest test of a run, by kind (owner 2026-10-27): what the panel, the model judge, the graph manager and the
P2 report read, never an earlier test.

A run's tests are its training's recorded test (kind all: every test trace), in the run folder's metrics/, and
every DONE test appended after it (src/sca/test_request.py: Retest / Test, All / Budget), each under
<run>/test/<model_hash>_et_<index>/metrics/ and test_registry.json[<hash>].external_tests[<id>]. A record of before
the kinds existed is a budget test when its attack has an attack_trace_budget. The latest of a kind is the one that
finished last; a test that did not finish (FAILED, in progress, interrupted) is never read.

    test_sources(run_dir, entry)        every test of the run: [{id, kind, source, finished_at, time, metrics_dir,
                                        key_recovery_dir}], the recorded test first
    latest_test(run_dir, entry, kind)   the latest of a kind ("all" | "budget"), or None

Author: Shraddha Mishra
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

KINDS = ("all", "budget")


def _when(text: Any) -> float:
    """An ISO time as seconds; 0 when absent or unreadable."""
    try:
        return datetime.datetime.fromisoformat(str(text)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _kind(record: Mapping[str, Any]) -> str:
    if record.get("kind") in KINDS:
        return str(record["kind"])
    budget = (((record.get("settings") or {}).get("perform_attack") or {}).get("test") or {}).get("attack_trace_budget")
    return "budget" if budget is not None else "all"


def test_sources(run_dir: Path, entry: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every test of the run (module docstring): the recorded test when its metrics exist, then each DONE appended
    test whose folder holds metrics."""
    run_dir = Path(run_dir)
    entry = entry if isinstance(entry, Mapping) else {}
    out: list[dict[str, Any]] = []
    recorded = run_dir / "metrics" / "attack_metrics"
    histories = sorted((recorded / "scalars").glob("*_test_history.jsonl")) if recorded.is_dir() else []
    if histories:
        out.append({"id": "recorded", "kind": "all", "source": "training", "finished_at": entry.get("finished_at"),
                    "time": _when(entry.get("finished_at")) or histories[0].stat().st_mtime,
                    "metrics_dir": recorded, "key_recovery_dir": run_dir / "metrics" / "key_recovery"})
    for key, record in (entry.get("external_tests") or {}).items():
        if not isinstance(record, Mapping) or str(record.get("status") or "").upper() != "DONE":
            continue
        folder = run_dir / "test" / str(key) / "metrics"
        if not (folder / "attack_metrics").is_dir():
            continue
        out.append({"id": str(key), "kind": _kind(record), "source": str(record.get("source") or "test"),
                    "finished_at": record.get("finished_at"), "time": _when(record.get("finished_at")),
                    "metrics_dir": folder / "attack_metrics", "key_recovery_dir": folder / "key_recovery"})
    return out


def latest_test(run_dir: Path, entry: Optional[Mapping[str, Any]], kind: str = "all") -> Optional[dict[str, Any]]:
    """The run's test of `kind` that finished last (module docstring), or None when it has none."""
    if kind not in KINDS:
        raise ValueError(f"a test's kind is one of {list(KINDS)}, got {kind!r}")
    tests = [t for t in test_sources(run_dir, entry) if t["kind"] == kind]
    return max(tests, key=lambda t: t["time"]) if tests else None


__all__ = ["KINDS", "latest_test", "test_sources"]
