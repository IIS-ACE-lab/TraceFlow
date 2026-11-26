# services/graph_manager/metric_router.py
"""
Metric router for TraceFlow GraphManager.

Purpose:
- Decide how a metric should be loaded and plotted.
- Keep graph_config.yaml simple:
    mode: "auto"
    metric: "key_rank"

Important:
- This file does not render plots (plotters.py does).
- plot_types_for(): the plot_type values that change the drawing of a job's
  metric; the validator refuses any other, the config panel offers only these.
- CLI logic belongs to graph_manager.py.

Author: Shraddha Mishra
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CLASSIFICATION_METRICS = {
    "cross_entropy",
    "loss",
    "accuracy",
    "mean_accuracy",
    "top_k_accuracy",
    "top_5_accuracy",
    "top_10_accuracy",
}

RANK_SCALAR_METRICS = {
    "key_rank",
    "mean_rank",
    "variance_rank",
    "median_rank",
    "rank_percentile",
    "normalized_mean_rank",
    "normalized_variance_rank",
    "auc_ge",
    "auc_sr",
    "rank_k_success",
    "time_to_recover",
    "ge_reduction_rate",
    "rank_at_traces",
    "ge_at_traces",
}

RANK_CURVE_METRICS = {
    "full_ranks",
    "sr_curve",
}

REPEATED_ATTACK_SCALAR_METRICS = {
    "guessing_entropy",
    "success_rate",
    "traces_to_ge_1",
    "traces_to_ge_k",
    "per_attack_traces_to_ge_1",       # the papers' N̄tGE: the mean of the per-attack crossings
}

REPEATED_ATTACK_CURVE_METRICS = {
    "mean_attack_rank_curve",
    "per_attack_curves",
    "final_ranks",
}

RESEARCH_METRICS = {
    "LDD",
    "Lm",
}

# source: cross_validation -> one rank curve per fold plus their mean (Eq. 7).
FOLD_CURVE_METRICS = {
    "fold_rank_curves",
}

# source: key_recovery -> score of every key hypothesis, true key marked.
KEY_RECOVERY_ARRAY_METRICS = {
    "key_candidate_scores",
}

ARRAY_METRICS = {
    "full_ranks",
    "sr_curve",
    "mean_attack_rank_curve",
    "per_attack_curves",
    "final_ranks",
    "LDD",
}

SCALAR_METRICS = (
    CLASSIFICATION_METRICS
    | RANK_SCALAR_METRICS
    | REPEATED_ATTACK_SCALAR_METRICS
    | {"Lm"}
)


@dataclass(frozen=True)
class MetricRoute:
    metric: str
    canonical_metric: str
    family: str
    storage: str
    plotter: str
    x_axis: str
    y_axis: str


METRIC_ALIASES = {
    "val_loss": "cross_entropy",
    "validation_loss": "cross_entropy",
    "ce": "cross_entropy",
    "categorical_cross_entropy": "cross_entropy",
    "rank": "key_rank",
    "ge": "guessing_entropy",
    "guessing_entropy_curve": "mean_attack_rank_curve",
    "success_rate_curve": "sr_curve",
    "ldd": "LDD",
    "lm": "Lm",
}


def canonical_metric_name(metric: Any) -> str:
    m = str(metric).strip()

    low = m.lower()

    if low in METRIC_ALIASES:
        return METRIC_ALIASES[low]

    if low.startswith("top_") and low.endswith("_accuracy"):
        return low

    if low.startswith("sr_curve_"):
        return low

    if m in {"LDD", "Lm"}:
        return m

    return low


def is_sr_curve_variant(metric: str) -> bool:
    return str(metric).startswith("sr_curve_")


def is_topk_accuracy_variant(metric: str) -> bool:
    m = str(metric)
    return m.startswith("top_") and m.endswith("_accuracy")


def route_metric(metric: Any, *, mode: str = "auto") -> MetricRoute:
    """
    Route a metric to loader + plotter behavior.

    storage:
      - scalar_history
      - array_curve
      - array_table
      - key_scores

    plotter:
      - scalar_line
      - attack_curve
      - distribution
      - fold_curves
      - key_scores
    """

    if str(mode).strip().lower() != "auto":
        raise ValueError(f"Unsupported graph job mode={mode!r}. Only 'auto' is supported.")

    canonical = canonical_metric_name(metric)

    if canonical in CLASSIFICATION_METRICS or is_topk_accuracy_variant(canonical):
        return MetricRoute(
            metric=str(metric),
            canonical_metric=canonical,
            family="classification",
            storage="scalar_history",
            plotter="scalar_line",
            x_axis="epoch",
            y_axis=canonical,
        )

    if canonical in RANK_SCALAR_METRICS:
        return MetricRoute(
            metric=str(metric),
            canonical_metric=canonical,
            family="rank",
            storage="scalar_history",
            plotter="scalar_line",
            x_axis="epoch",
            y_axis=canonical,
        )

    if canonical == "full_ranks":
        return MetricRoute(
            metric=str(metric),
            canonical_metric=canonical,
            family="rank",
            storage="array_curve",
            plotter="attack_curve",
            x_axis="traces",
            y_axis="key_rank",
        )

    if canonical == "sr_curve" or is_sr_curve_variant(canonical):
        return MetricRoute(
            metric=str(metric),
            canonical_metric=canonical,
            family="rank",
            storage="array_curve",
            plotter="attack_curve",
            x_axis="traces",
            y_axis="success_rate",
        )

    if canonical in REPEATED_ATTACK_SCALAR_METRICS:
        return MetricRoute(
            metric=str(metric),
            canonical_metric=canonical,
            family="repeated_attack",
            storage="scalar_history",
            plotter="scalar_line",
            x_axis="epoch",
            y_axis=canonical,
        )

    if canonical == "mean_attack_rank_curve":
        return MetricRoute(
            metric=str(metric),
            canonical_metric=canonical,
            family="repeated_attack",
            storage="array_curve",
            plotter="attack_curve",
            x_axis="traces",
            y_axis="guessing_entropy",
        )

    if canonical == "per_attack_curves":
        return MetricRoute(
            metric=str(metric),
            canonical_metric=canonical,
            family="repeated_attack",
            storage="array_curve",
            plotter="attack_curve",
            x_axis="traces",
            y_axis="key_rank",
        )

    if canonical == "final_ranks":
        return MetricRoute(
            metric=str(metric),
            canonical_metric=canonical,
            family="repeated_attack",
            storage="array_table",
            plotter="distribution",
            x_axis="attack_index",
            y_axis="final_rank",
        )

    if canonical == "LDD":
        return MetricRoute(
            metric=str(metric),
            canonical_metric=canonical,
            family="research",
            storage="array_table",
            plotter="distribution",
            x_axis="key_guess",
            y_axis="LDD",
        )

    if canonical == "fold_rank_curves":
        return MetricRoute(
            metric=str(metric),
            canonical_metric=canonical,
            family="cross_validation",
            storage="array_curve",
            plotter="fold_curves",
            x_axis="traces",
            y_axis="guessing_entropy",
        )

    if canonical == "key_candidate_scores":
        return MetricRoute(
            metric=str(metric),
            canonical_metric=canonical,
            family="key_recovery",
            storage="key_scores",
            plotter="key_scores",
            x_axis="key_guess",
            y_axis="score_relative_to_top_guess",
        )

    if canonical == "Lm":
        return MetricRoute(
            metric=str(metric),
            canonical_metric=canonical,
            family="research",
            storage="scalar_history",
            plotter="scalar_line",
            x_axis="epoch",
            y_axis="Lm",
        )

    # Any other recorded scalar (training_objective, traces_to_ge_5,
    # final_true_key_rank, top1_top2_score_margin, ...) is a per-epoch column.
    return MetricRoute(
        metric=str(metric),
        canonical_metric=canonical,
        family="scalar",
        storage="scalar_history",
        plotter="scalar_line",
        x_axis="epoch",
        y_axis=canonical,
    )


# ---------------------------------------------------------------------------
# plot_type: the values that change the drawing of a metric
# ---------------------------------------------------------------------------
#
# What each drawer reads (plotters.py, attack_plotters.py):
#   per-epoch scalar (storage scalar_history)
#     solo     plot_scalar_line -> _draw_xy: line, line_scatter, scatter, step, bar
#              (barh falls back to line)
#     overlay  plot_scalar_overlay -> _draw_xy as above; one value per model draws
#              bars, barh horizontal ones (with array_epoch latest: the last epoch)
#   curve (plotter attack_curve: full_ranks, sr_curve*, mean_attack_rank_curve,
#   per_attack_curves)
#     overlay, or solo with array_epoch.mode overlay
#              plot_curve_overlay -> _draw_xy without bar: line, line_scatter, scatter, step
#     solo     plot_attack_metric: plot_type is not read
#   fixed (final_ranks, LDD, fold_rank_curves, key_candidate_scores): their drawers
#     do not read plot_type
#   model_explanation does not read plot_type.
#   heatmap (overlay of a key_recovery scalar of cross-validated runs): GraphManager.run_fold_heatmap,
#              a model per row, a fold per column (graph_job_validator.check_heatmap_job)
#   strip (overlay of the test's key-recovery score margin): plotters.plot_strip, a dot per run on
#              log10(1 + margin), a row per group (graph_job_validator.check_strip_job)
# auto is always allowed (a line where one is drawn). histogram and distribution
# are reached by no job (plot_distribution is not dispatched to).
XY_PLOT_TYPES = ("line", "line_scatter", "scatter", "step")
PLOT_TYPES: dict[str, dict[str, tuple[str, ...]]] = {
    "scalar": {"solo": XY_PLOT_TYPES + ("bar",), "overlay": XY_PLOT_TYPES + ("strip", "bar", "barh", "heatmap")},
    "curve": {"solo": (), "overlay": XY_PLOT_TYPES},
    "fixed": {"solo": (), "overlay": ()},
}
OVERLAY_JOB_TYPES = ("process_overlay", "model_overlay")


def metric_kind(metric: Any) -> str:
    """scalar | curve | fixed: which drawer a metric reaches (see PLOT_TYPES)."""
    route = route_metric(metric)
    if route.storage == "scalar_history":
        return "scalar"
    return "curve" if route.plotter == "attack_curve" else "fixed"


def plot_types_for(job_type: Any, metrics: Any, array_epoch_mode: Any = "latest") -> list[str]:
    """The plot_type values that change the drawing of these metrics in a job of this type."""
    job_type = str(job_type).strip()
    if job_type == "model_explanation":
        return ["auto"]
    names = [metrics] if isinstance(metrics, str) else list(metrics or [])
    allowed: set[str] | None = None
    for name in names:
        kind = metric_kind(name)
        overlay = job_type in OVERLAY_JOB_TYPES or (kind == "curve" and str(array_epoch_mode) == "overlay")
        types = set(PLOT_TYPES[kind]["overlay" if overlay else "solo"])
        allowed = types if allowed is None else allowed & types
    order = ("auto",) + XY_PLOT_TYPES + ("strip", "bar", "barh", "heatmap")
    return [t for t in order if t == "auto" or t in (allowed or set())]
