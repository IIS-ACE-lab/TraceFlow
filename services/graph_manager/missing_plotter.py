# services/graph_manager/missing_plotter.py
"""
Missing metric terminal reporter for TraceFlow GraphManager.

Current policy:
- Do not render placeholder images.
- If metric data is missing, print a clear terminal warning.
- Continue graph execution safely.

Author: Shraddha Mishra
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MissingMetric:
    metric: str
    model: str | None = None
    process: str | None = None
    reason: str | None = None
    expected_paths: tuple[str, ...] = ()


def report_missing_metric(
    *,
    metric: str,
    model: str | None = None,
    process: str | None = None,
    reason: str | None = None,
    expected_paths: list[str] | tuple[str, ...] | None = None,
) -> MissingMetric:
    item = MissingMetric(
        metric=str(metric),
        model=str(model) if model is not None else None,
        process=str(process) if process is not None else None,
        reason=str(reason) if reason else "metric data not found",
        expected_paths=tuple(str(p) for p in (expected_paths or ())),
    )

    print("[GraphManager][MISSING]")
    print(f"  metric : {item.metric}")

    if item.model:
        print(f"  model  : {item.model}")

    if item.process:
        print(f"  process: {item.process}")

    print(f"  reason : {item.reason}")

    if item.expected_paths:
        print("  checked paths:")
        for p in item.expected_paths:
            print(f"    - {p}")

    return item


def missing_to_dict(item: MissingMetric) -> dict[str, Any]:
    return {
        "metric": item.metric,
        "model": item.model,
        "process": item.process,
        "reason": item.reason,
        "expected_paths": list(item.expected_paths),
    }
