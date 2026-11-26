# services/graph_manager/trace_budget.py
"""
A graph job's attack trace budget and test (owner 2026-10-26).

trace_budget: B
    The attack curves of the job drawn at a budget of B traces per attack. A curve
    point at n traces is the mean (or rate) over the attacks of their first n traces,
    so the curves up to B are the curves of B-trace attacks: each run's attack
    order is fixed by its seed, and an attack of B traces is the first B traces of
    the attack drawn without a budget (perform_attack.py). The curves are cut at B
    and final_ranks is read at B from per_attack_curves. B must be a recorded point
    of each run's curve (perform_attack.attack_step), no further than its last.

test: latest | budget | recorded | <index>
    Which test the test process is drawn from (owner 2026-10-27: the latest, never an earlier one):
    latest (empty too) the latest All test, the training's recorded test or a later Retest All / Test All;
    budget the latest Budget test (its data entry's attack_trace_budget per attack); recorded the training's
    own test; <index> the run's test <index> (<run>/test/<model_hash>_et_<index>/, sca/test_request.py).
    utils/latest_test.py finds the latest of a kind.

Author: Shraddha Mishra
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# The metrics a budget cuts: every point is an n-trace attack statistic.
CURVE_METRICS = frozenset({"mean_attack_rank_curve", "per_attack_curves", "full_ranks", "sr_curve"})
BUDGET_METRICS = CURVE_METRICS | {"final_ranks"}
RECORDED_TEST = "recorded"
LATEST_TEST, BUDGET_TEST = "latest", "budget"
TEST_CHOICES = (LATEST_TEST, BUDGET_TEST, RECORDED_TEST)


class TraceBudgetError(ValueError):
    """A trace budget the run's curves cannot be read at."""


def is_budget_metric(canonical: str) -> bool:
    canonical = str(canonical)
    return canonical in BUDGET_METRICS or canonical.startswith("sr_curve_")


def _cut_table(table: np.ndarray, budget: int, where: str) -> np.ndarray:
    """A [M, 2] (trace count, value) curve up to and including `budget` traces."""
    x = table[:, 0]
    if budget > x[-1]:
        raise TraceBudgetError(f"{where}: trace_budget {budget} is beyond the run's {int(x[-1])} recorded traces")
    if not np.any(x == budget):
        raise TraceBudgetError(
            f"{where}: trace_budget {budget} is not a recorded point of the curve (it records every "
            f"{int(x[1] - x[0]) if x.size > 1 else '?'} traces from {int(x[0])}); choose one of them"
        )
    return table[x <= budget]


def cut_curve(data: Any, budget: int, where: str) -> Any:
    """`data` (a [M, 2] curve, [R, M, 2] per-attack curves, or a mapping of them) cut at `budget` traces."""
    if isinstance(data, dict):
        return {key: cut_curve(value, budget, f"{where} {key}") for key, value in data.items()}
    array = np.asarray(data, dtype=float)
    if array.ndim == 2 and array.shape[1] == 2:
        return _cut_table(array, budget, where)
    if array.ndim == 3 and array.shape[2] == 2:
        return np.stack([_cut_table(curve, budget, where) for curve in array])
    raise TraceBudgetError(f"{where}: a trace budget needs (trace count, value) curves, got shape {array.shape}")


def final_ranks_at(per_attack_curves: Any, budget: int, where: str) -> np.ndarray:
    """Every attack's rank after `budget` traces: its final rank as a `budget`-trace attack."""
    curves = cut_curve(per_attack_curves, budget, where)
    if not (isinstance(curves, np.ndarray) and curves.ndim == 3):
        raise TraceBudgetError(f"{where}: final_ranks at a budget needs the run's per_attack_curves")
    return curves[:, -1, 1].astype(np.int64)


def test_metrics_dir(model_root: Path, entry: Any, choice: Any) -> Path:
    """The metrics/attack_metrics folder the test process of a job reads (module docstring: test), from the run's
    test_registry.json entry. A job choosing budget for a run without a Budget test raises FileNotFoundError."""
    from services.side_channel_core.utils.latest_test import latest_test

    model_root = Path(model_root)
    if isinstance(choice, int):
        return external_test_dir(model_root, choice)
    if choice == RECORDED_TEST:
        return model_root / "metrics" / "attack_metrics"
    kind = "budget" if choice == BUDGET_TEST else "all"
    latest = latest_test(model_root, entry, kind)
    if latest is None:
        if kind == "budget":
            raise FileNotFoundError(f"{model_root.name} has no Budget test yet (make test MODEL={model_root.name} BUDGET=1)")
        return model_root / "metrics" / "attack_metrics"
    return latest["metrics_dir"]


def external_test_dir(model_root: Path, index: int) -> Path:
    """The metrics/attack_metrics folder of external test `index` of the run at `model_root`."""
    matches = sorted((Path(model_root) / "test").glob(f"*_et_{int(index)}"))
    matches = [m for m in matches if m.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"external test {index} of {Path(model_root).name}: "
            + ("not run (make test EXTERNAL=1 writes it under test/<model_hash>_et_<index>/)" if not matches
               else f"{len(matches)} folders match {[m.name for m in matches]}")
        )
    return matches[0] / "metrics" / "attack_metrics"
