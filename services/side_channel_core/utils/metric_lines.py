# services/side_channel_core/utils/metric_lines.py
"""
Metrics side by side: one line per metric, a column per process, the | separators in one row.

    mean_accuracy           train: 0.0412 ▲ | validation: 0.0038 ▼
    mean_attack_rank_curve                  | validation:    107 ▼ | test: 36

Every value of a process stands under the others, its text right-aligned to the widest; a metric a process does
not record leaves that column blank, and a line stops after its last value. core/monitor.py (the live view) and
utils/logger.py (each epoch's block in the run's logs) both draw with it, so the two read alike.

A row is (name, {process: (label, text, mark)}): the label ("train", "validation (CV)", "test"), the value as
text, and a one-character mark (▲ better, ▼ worse than the epoch before, or a space; it may carry colour codes,
which take no width).

Author: Shraddha Mishra
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

PROCESSES = ("train", "validation", "test")

Row = tuple[str, Mapping[str, tuple[str, str, str]]]


def columns(rows: Sequence[Row]) -> list[str]:
    """The processes that hold a value on some line, in train, validation, test order."""
    return [p for p in PROCESSES if any(p in cells for _, cells in rows)]


def widths(rows: Sequence[Row]) -> dict[str, tuple[int, int, int]]:
    """Per column: (label width with its colon, text width, the whole cell's width)."""
    out = {}
    for p in columns(rows):
        label = max(len(cells[p][0]) for _, cells in rows if p in cells) + 1
        text = max(len(cells[p][1]) for _, cells in rows if p in cells)
        out[p] = (label, text, label + text + 3)
    return out


def lines(rows: Sequence[Row], *, indent: str = "  ", name_width: Optional[int] = None) -> list[str]:
    """The rows as aligned lines (name_width: at least this wide, for a row drawn beside them)."""
    if not rows:
        return []
    cols, w = columns(rows), widths(rows)
    name_width = max([len(name) for name, _ in rows] + [name_width or 0])
    out = []
    for name, cells in rows:
        shown = []
        for p in cols[:max(cols.index(p) for p in cells) + 1]:
            if p not in cells:
                shown.append(" " * w[p][2])
                continue
            label, text, mark = cells[p]
            shown.append(f"{(label + ':').ljust(w[p][0])} {text.rjust(w[p][1])} {mark}")
        out.append((f"{indent}{name.ljust(name_width)}  " + " | ".join(shown)).rstrip())
    return out


__all__ = ["PROCESSES", "columns", "widths", "lines"]
