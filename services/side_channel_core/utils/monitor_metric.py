# utils/monitor_metric.py
"""
monitor_metric.py

Utility functions to determine which metric to monitor during training and whether a higher or lower value is 'better'.

Functions:
- is_better_score(current, best, metric, mode=None, min_delta=0.0):
    Compare a new metric score against the current best and decide if it's an improvement.

- get_monitor_metric_and_mode(cfg):
    Extract the metric name and mode ('minimize' or 'maximize') from a training config dict, with sensible defaults.

- get_monitor_mode(metric_name):
    Infer whether to minimize or maximize a given metric based on its name.

- monitorable_metrics(cfg), check_monitor_metric(cfg, metric, ...):
    The metrics a run of cfg can monitor (its metrics_to_log that a validation pass writes as one
    number, and validation_objective), and the refusal of any other when early stopping or
    validation checkpointing reads the monitor (the panel's trainer.monitor_metric menu lists them).

These helpers are intended for use in training loops, checkpointing, and early stopping logic.

Author: Shraddha Mishra
"""
import re


DEFAULT_METRIC = "guessing_entropy"
DEFAULT_MODE = "minimize"


def is_better_score(current: float, best: float, metric: str, mode: str = None, min_delta: float = 0.0) -> bool:
    """
    Determine if the `current` score is a meaningful improvement over `best`, given a monitoring mode and tolerance.

    Args:
        current: The newly observed metric value.
        best: The best metric value seen so far.
        metric: The name of the metric being monitored.
        mode: One of 'minimize' or 'maximize'. If None, will be inferred from metric name.
        min_delta: Minimum change required for an improvement (positive float).

    Returns:
        True if `current` is better than `best` by at least `min_delta`, according to the chosen `mode`.

    Raises:
        ValueError: If `mode` is not 'minimize' or 'maximize'.
    """
    mode = mode or get_monitor_mode(metric)

    if mode == "minimize":
        return current < best - min_delta
    elif mode == "maximize":
        return current > best + min_delta
    else:
        raise ValueError(f"Invalid monitor_mode '{mode}'. Use 'minimize' or 'maximize'.")


def get_monitor_metric_and_mode(cfg: dict) -> tuple[str | None, str]:
    """
    Extract the desired monitoring metric name and its mode from a configuration dict.

    Reads `cfg['trainer']['monitor_metric']` and `cfg['trainer']['monitor_mode']`,
    applies defaults if missing or disabled, and logs any fallbacks.

    Args:
        cfg: A configuration dictionary (typically loaded from master + params).

    Returns:
        A tuple (metric_name, mode) where:
            - metric_name is a lowercase string, the metric to track, or None
              when monitor_metric is none/na (monitor disabled).
            - mode is either 'minimize' or 'maximize'.
    """
    trainer = cfg.get("trainer", {})
    raw = trainer.get("monitor_metric", DEFAULT_METRIC)
    metric = "" if raw is None else str(raw).strip().lower()

    # "none" disables the monitor; it is never replaced by a default.
    if metric in ("na", "none", ""):
        return None, trainer.get("monitor_mode") or DEFAULT_MODE

    mode = trainer.get("monitor_mode") or get_monitor_mode(metric)
    return metric, mode


VALIDATION_OBJECTIVE = "validation_objective"   # the loss on the validation traces, written every epoch


def monitorable_metrics(cfg: dict) -> list[str]:
    """The metrics early stopping and validation checkpointing can monitor in a run of `cfg`: each
    metric of metrics_to_log that a validation pass computes as one number per epoch (as
    compute_metrics names it in the epoch's results), then validation_objective, which every epoch
    writes. In metrics_to_log order, each once."""
    from sca.metrics import (FOLD_MEAN_METRICS, NON_SCALAR_METRICS, VALID_METRIC_PROCESSES,
                             _canonical_metric_name, _parse_metrics_list)

    out: list[str] = []
    for name, _options in _parse_metrics_list(cfg):
        canonical = _canonical_metric_name(name)
        computed = FOLD_MEAN_METRICS.get(canonical, canonical)
        if computed in NON_SCALAR_METRICS or "validation" not in VALID_METRIC_PROCESSES.get(computed, []):
            continue
        for key in (canonical, computed):            # mean_accuracy is also written as accuracy
            if key not in out:
                out.append(key)
    return out + ([VALIDATION_OBJECTIVE] if VALIDATION_OBJECTIVE not in out else [])


def check_monitor_metric(cfg: dict, metric: str | None, *, early_stopping: bool, checkpointing: bool) -> None:
    """A monitor that early stopping or validation checkpointing reads must be a metric the run
    records each epoch (monitorable_metrics); otherwise they would wait for it forever. A monitor
    nothing reads is not checked."""
    from services.side_channel_core.utils.config_errors import ConfigError

    if not metric or not (early_stopping or checkpointing):
        return
    allowed = monitorable_metrics(cfg)
    if metric not in allowed:
        users = " and ".join(n for n, on in (("early_stopping", early_stopping), ("validation checkpointing", checkpointing)) if on)
        raise ConfigError(
            f"trainer.monitor_metric {metric!r} is not recorded by this run, and {users} reads it every epoch; "
            f"pick one of {allowed} (metrics_to_log's metrics a validation pass writes as one number), "
            "or add it to metrics_to_log"
        )


def get_monitor_mode(metric_name: str) -> str:
    """
    Infer whether a metric should be minimized or maximized based on its name.

    Args:
        metric_name: Name of the metric (e.g. 'accuracy', 'cross_entropy').

    Returns:
        'minimize' for loss‐type metrics, 'maximize' for performance metrics, or DEFAULT_MODE otherwise.
    """
    return "maximize" if metric_direction(metric_name) == "max" else DEFAULT_MODE


# Higher is better: accuracies, success rates, SR area, GE reduction rate (the
# negated early GE slope), Lm (agreement of the model's key ranking with the
# leakage ranking, ldd.py) and the CV aggregate's fold fractions; η_t, the learning
# rate at iteration or epoch t, has no better or worse of its own, and its rise is drawn
# ▲ (owner 2026-10-25; train rows only, so never a monitor). Everything else is a rank,
# entropy, loss, trace count, time or spread: lower is better.
_MAXIMIZE = re.compile(
    r"accuracy|top_\d+_accuracy|top_k_accuracy|success_rate|sr@\w+|sr_at_\d+"
    r"|rank_k_success(@\d+)?|auc_sr(@\d+)?|ge_reduction_rate(@\d+)?|lm"
    r"|folds_reaching_ge_\d+_fraction|folds_better_than_chance|normalized_rank_advantage|η_t"
)


def metric_direction(metric: str) -> str:
    """'max' when higher is better, else 'min': the one rule of the training monitor and
    model_judge. Read through the names the pipeline builds: the fold mean (mean_accuracy,
    accuracy_mean), the process (accuracy_validation, mean_accuracy_train) and final_
    (final_sr_at_1) keep the metric's direction; a spread (accuracy_std,
    mean_accuracy_validation_std, training_time_std_s) is lower-is-better whatever it spreads."""
    name = str(metric).lower()
    if re.search(r"(^|_)std(_|$)", name):
        return "min"
    name = re.sub(r"_(train|validation|test)$", "", name)
    name = re.sub(r"_mean$", "", name)
    name = re.sub(r"^(mean|final)_", "", name)
    return "max" if _MAXIMIZE.fullmatch(name) else "min"
