# services/graph_manager/attack_plotters.py
"""
SCA attack-specific plotters for TraceFlow GraphManager.

Handles paper-grade SCA curves:
- full_ranks              -> traces vs key rank
- mean_attack_rank_curve  -> traces vs guessing entropy
- sr_curve / sr_curve_k   -> traces vs success rate
- per_attack_curves       -> traces vs key rank overlays
- final_ranks             -> attack index vs final rank
- LDD                     -> key guess vs leakage distribution difference
- fold_rank_curves        -> one rank curve per CV fold plus the fold mean
- key_candidate_scores    -> score of every key hypothesis, true key marked

Author: Shraddha Mishra
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from .plot_style import apply_matplotlib_style, get_line_kwargs, on_x_grid
from .plotters import (
    _finalize_axes,
    plot_curve,
    plot_curve_overlay,
    plot_distribution,
    save_figure,
)


def _extract_npz_metric(data: Any, metric: str) -> Any:
    if isinstance(data, dict):
        if metric in data:
            return data[metric]

        low = metric.lower()
        for key, value in data.items():
            if str(key).lower() == low:
                return value

        if len(data) == 1:
            return next(iter(data.values()))

    return data


def _coerce_multi_curves(data: Any) -> list[np.ndarray]:
    """
    Convert per_attack_curves-like data into list of [x,y] arrays.
    Accepts:
    - [num_attacks, num_points, 2]
    - [num_attacks, num_points]
    - list of curves
    """

    if isinstance(data, pd.DataFrame):
        arr = data.to_numpy(dtype=float)
    else:
        arr = np.asarray(data)

    arr = np.squeeze(arr)

    curves: list[np.ndarray] = []

    if arr.ndim == 3:
        for i in range(arr.shape[0]):
            c = np.asarray(arr[i], dtype=float)
            if c.ndim == 2 and c.shape[1] >= 2:
                curves.append(c[:, :2])
            elif c.ndim == 1:
                x = np.arange(1, len(c) + 1, dtype=float)
                curves.append(np.stack([x, c], axis=1))

    elif arr.ndim == 2:
        # If already one curve [N,2]
        if arr.shape[1] >= 2:
            curves.append(arr[:, :2].astype(float))
        else:
            for i in range(arr.shape[0]):
                y = arr[i].astype(float)
                x = np.arange(1, len(y) + 1, dtype=float)
                curves.append(np.stack([x, y], axis=1))

    elif arr.ndim == 1:
        y = arr.astype(float)
        x = np.arange(1, len(y) + 1, dtype=float)
        curves.append(np.stack([x, y], axis=1))

    return curves


def plot_full_ranks(
    *,
    data: Any,
    job: Mapping[str, Any],
    out_base: Path,
    model: str,
    process: str,
) -> list[Path]:
    return plot_curve(
        data=data,
        job=job,
        out_base=out_base,
        metric="full_ranks",
        model=model,
        process=process,
        x_label="Attack Traces",
        y_label="Key Rank",
        title=None,
    )


def plot_mean_attack_rank_curve(
    *,
    data: Any,
    job: Mapping[str, Any],
    out_base: Path,
    model: str,
    process: str,
) -> list[Path]:
    return plot_curve(
        data=data,
        job=job,
        out_base=out_base,
        metric="mean_attack_rank_curve",
        model=model,
        process=process,
        x_label="Attack Traces",
        y_label="Guessing Entropy",
        title=None,
    )


def plot_sr_curve(
    *,
    data: Any,
    job: Mapping[str, Any],
    out_base: Path,
    model: str,
    process: str,
    metric: str = "sr_curve",
) -> list[Path]:
    return plot_curve(
        data=data,
        job=job,
        out_base=out_base,
        metric=metric,
        model=model,
        process=process,
        x_label="Attack Traces",
        y_label="Success Rate",
        title=None,
    )


def plot_per_attack_curves(
    *,
    data: Any,
    job: Mapping[str, Any],
    out_base: Path,
    model: str,
    process: str,
) -> list[Path]:
    curves = _coerce_multi_curves(data)

    if not curves:
        print(
            f"[GraphManager][WARN] No per-attack curves found: "
            f"model={model} process={process}"
        )
        return []

    series = {
        f"attack_{idx + 1}": curve
        for idx, curve in enumerate(curves)
    }

    return plot_curve_overlay(
        series=series,
        job=job,
        out_base=out_base,
        metric="per_attack_curves",
        x_label="Attack Traces",
        y_label="Key Rank",
        title=f"Per-Attack Rank Curves: {model} / {process}",
    )


def plot_final_ranks(
    *,
    data: Any,
    job: Mapping[str, Any],
    out_base: Path,
    model: str,
    process: str,
) -> list[Path]:
    return plot_distribution(
        data=data,
        job=job,
        out_base=out_base,
        metric="final_ranks",
        model=model,
        process=process,
        x_label="Attack Index",
        y_label="Final Key Rank",
        title=None,
    )


def plot_ldd(
    *,
    data: Any,
    job: Mapping[str, Any],
    out_base: Path,
    model: str,
    process: str,
) -> list[Path]:
    data = _extract_npz_metric(data, "LDD")

    return plot_distribution(
        data=data,
        job=job,
        out_base=out_base,
        metric="LDD",
        model=model,
        process=process,
        x_label="Key Guess",
        y_label="Leakage Distribution Difference",
        title=None,
    )


def plot_fold_rank_curves(
    *,
    folds: Mapping[str, Any],
    job: Mapping[str, Any],
    out_base: Path,
    model: str,
    process: str,
) -> list[Path]:
    """
    Cross-validation rank curves: every fold in thin gray, the fold mean
    (the CV headline curve, Eq. 7) on top, and the GE = 1 recovery line.
    """
    apply_matplotlib_style(job)
    curves = {name: np.asarray(c, dtype=float) for name, c in folds.items()}
    ref = next(iter(curves.values()))
    if any(c.shape != ref.shape or not np.array_equal(c[:, 0], ref[:, 0]) for c in curves.values()):
        print(f"[GraphManager][WARN] Fold curves of {model} use different trace grids; not plotted")
        return []

    line = get_line_kwargs(job)
    width = float(line.get("linewidth") or 1.0)
    fig, ax = plt.subplots()
    for i, curve in enumerate(curves.values()):
        ax.plot(*on_x_grid(job, curve[:, 0], curve[:, 1]), color="0.65", linewidth=0.6 * width,
                label=f"folds (n={len(curves)})" if i == 0 else None)
    mean = np.mean(np.stack([c[:, 1] for c in curves.values()]), axis=0)
    ax.plot(*on_x_grid(job, ref[:, 0], mean), color="C0", linewidth=1.8 * width, label="mean over folds")
    ax.axhline(1.0, color="0.3", linewidth=0.6 * width, linestyle="--", label="GE = 1 (key recovered)")

    _finalize_axes(
        ax,
        job=job,
        context={"metric": "fold_rank_curves", "model": model, "model_name": model, "process": process},
        default_title=f"Cross-validation rank curves: {model}",
        default_x_label="Attack Traces",
        default_y_label="Guessing Entropy",
    )
    return save_figure(fig, out_base, job)


def plot_key_candidate_scores(
    *,
    payload: Mapping[str, Any],
    job: Mapping[str, Any],
    out_base: Path,
    model: str,
    process: str,
) -> list[Path]:
    """
    Score of every key hypothesis relative to the top guess (0 = best).
    The true key is drawn in a second colour and named in the legend, so it
    is identifiable without colour.
    """
    apply_matplotlib_style(job)
    keys = np.asarray(payload["key_guess"], dtype=int)
    rel = np.asarray(payload["score"], dtype=float)
    rel = rel - np.nanmax(rel)
    true_key = payload.get("true_key")
    rank = payload.get("true_key_rank")

    fig, ax = plt.subplots()
    ax.bar(keys, rel, width=0.8, color="C0", label="key hypotheses")
    if true_key is not None and int(true_key) in set(keys.tolist()):
        idx = int(np.where(keys == int(true_key))[0][0])
        ax.bar([keys[idx]], [rel[idx]], width=2.0, color="C3",
               label=f"true key 0x{int(true_key):02X}" + (f" (rank {rank})" if rank is not None else ""))

    _finalize_axes(
        ax,
        job=job,
        context={"metric": "key_candidate_scores", "model": model, "model_name": model, "process": process},
        default_title=f"Key hypothesis scores: {model} / {process}",
        default_x_label="Key Guess",
        default_y_label="Log-likelihood relative to top guess",
    )
    return save_figure(fig, out_base, job)


def plot_attack_metric(
    *,
    metric: str,
    data: Any,
    job: Mapping[str, Any],
    out_base: Path,
    model: str,
    process: str,
) -> list[Path]:
    """
    Dispatch one attack/research metric to the correct SCA plotter.

    This function is schema-light by design:
    - no output_file_name handling
    - no legend handling
    - no plot_type handling
    - no export handling

    Those are handled by graph_manager.py and plotters.py.
    """

    canonical = str(metric).strip()
    canonical_lower = canonical.lower()

    if canonical_lower == "full_ranks":
        return plot_full_ranks(
            data=data,
            job=job,
            out_base=out_base,
            model=model,
            process=process,
        )

    if canonical_lower == "mean_attack_rank_curve":
        return plot_mean_attack_rank_curve(
            data=data,
            job=job,
            out_base=out_base,
            model=model,
            process=process,
        )

    if canonical_lower == "sr_curve" or canonical_lower.startswith("sr_curve_"):
        return plot_sr_curve(
            data=data,
            job=job,
            out_base=out_base,
            model=model,
            process=process,
            metric=canonical,
        )

    if canonical_lower == "per_attack_curves":
        return plot_per_attack_curves(
            data=data,
            job=job,
            out_base=out_base,
            model=model,
            process=process,
        )

    if canonical_lower == "final_ranks":
        return plot_final_ranks(
            data=data,
            job=job,
            out_base=out_base,
            model=model,
            process=process,
        )

    if canonical_lower == "ldd":
        return plot_ldd(
            data=data,
            job=job,
            out_base=out_base,
            model=model,
            process=process,
        )

    print(f"[GraphManager][WARN] No attack plotter for metric={canonical}")
    return []
