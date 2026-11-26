# services/side_channel_core/src/sca/metrics.py
"""
Metric computation for Side-Channel Analysis (SCA).

Two metric families:

(A) Classification metrics:
    - cross_entropy
    - accuracy
    - top_k_accuracy (configurable: {"top_k_accuracy": {"k": 5}})

(B) SCA key-rank metrics:
    Single-run rank-curve metrics:
      - full_ranks
      - key_rank
      - mean_rank / variance_rank / median_rank / rank_percentile
      - normalized_mean_rank / normalized_variance_rank
      - rank_k_success
      - time_to_recover
      - ge_reduction_rate
      - rank_at_traces / ge_at_traces

    Repeated-attack metrics:
      - guessing_entropy
      - mean_attack_rank_curve
      - success_rate / sr_curve / auc_sr
      - auc_ge / ge_at_traces
      - traces_to_ge_1
      - traces_to_ge_k
      - traces_to_sr_1
      - per_attack_traces_to_ge_1   (the papers' N̄tGE: the mean of the per-attack crossings)

    Research metrics:
      - LDD
      - Lm

Notes
-----
- Single-run rank metrics come from core.rank_curve.full_ranks_curve().
- GE/SR metrics come only from repeated curves produced by
  sca.perform_attack.perform_attacks().
- LDD/Lm come from sca.ldd.compute_ldd_lm().
- Non-scalar metrics (e.g. full_ranks, sr_curve, mean_attack_rank_curve, LDD)
  are returned in the results dict and should be saved as artifacts by the caller.

Author: Shraddha Mishra
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from core.framework_switch import cross_entropy_per_trace, resolve_framework
from utils.output_transformation_util import build_output_transformation_from_config
from utils.logger import get_logger
from sca.dataset_types import label_target, resolved_variant
from sca.rank_curve import full_ranks_curve
from sca.perform_attack import (
    endpoint_tie_seed,
    KEY_RECOVERY_EVIDENCE_KEY,
    _get_attack_step,
    attack_plan,
    perform_attacks,
)
from sca.ldd import compute_ldd_lm
from sca.leakage_models import (
    normalize_leakage_model,
    validate_num_classes_for_leakage,
)

logger = get_logger("metrics")

MetricSpec = Union[str, Dict[str, Dict[str, Any]]]

_RANK_METRICS = {
    "full_ranks",
    "key_rank",
    "mean_rank",
    "variance_rank",
    "median_rank",
    "rank_percentile",
    "normalized_mean_rank",
    "normalized_variance_rank",
    "rank_k_success",
    "time_to_recover",
    "rank_at_traces",
}

_REPEATED_ATTACK_METRICS = {
    "guessing_entropy",
    "mean_attack_rank_curve",
    "sr_curve",
    "auc_ge",
    "auc_sr",
    "ge_at_traces",
    "success_rate",
    "traces_to_ge_1",
    "traces_to_ge_k",
    "traces_to_sr_1",
    "ge_reduction_rate",
    "per_attack_curves",
    "final_ranks",
    "per_attack_traces_to_ge_1",
}

_LDD_METRICS = {
    "LDD",
    "Lm",
}

# Time measured by the training loop itself, not computed from predictions.
#   fit_time_s          cumulative time up to the epoch, in seconds
#   fit_time_h          the same in hours
#   fit_time_per_epoch  mean time of one epoch so far, in seconds
# In the train rows it is the time of the optimisation passes (the paper's
# fit time); in the validation rows it is the time of the evaluation on the
# held-out traces (inference and attack metrics) that follows each epoch.
#
# mean_fit_time_s / _h / _per_epoch ask for the mean over the folds. The run
# then records its own fit time under both names, and the cross-validation
# aggregate writes the mean, its standard deviation and the fold values.
FIT_TIME_METRICS = ("fit_time_s", "fit_time_h", "fit_time_per_epoch")
MEAN_FIT_TIME_METRICS = tuple(f"mean_{name}" for name in FIT_TIME_METRICS)
_FIT_TIME_NAMES = set(FIT_TIME_METRICS) | set(MEAN_FIT_TIME_METRICS)

# Counted from the model by the training loop, written to its train rows:
#   trainable_parameters  the model's trainable weights (utils/parameter_count.py)
MODEL_SIZE_METRICS = ("trainable_parameters",)
# Read from the optimizer at the end of each epoch, written to its train row:
#   η_t  the learning rate at iteration or epoch t (eta_t below): the rate the epoch's last optimisation
#        step used (param group 0), the record of a schedule (a completed one-cycle ends at
#        max / (div x final_div)). Not trainer.optimizer.learning_rate, the setting (under a schedule, its
#        maximum).
LOOP_STATE_METRICS = ("η_t",)
# Metric names that were renamed: metrics_to_log refuses the old name and names the new one.
RENAMED_METRICS = {"learning_rate": "η_t", "used_learning_rate": "η_t"}
LOOP_PROVIDED_METRICS = _FIT_TIME_NAMES | set(MODEL_SIZE_METRICS) | set(LOOP_STATE_METRICS) | {"training_objective", "validation_objective"}


def canonical_fit_time_name(name: Any) -> Optional[str]:
    """
    Canonical name of a fit-time metric, or None for any other metric.

    ``fit_time_{s}``, ``fit_time_{h}`` and ``fit_time_{per_epoch}`` are the
    config spellings of fit_time_s, fit_time_h and fit_time_per_epoch, and
    ``mean_fit_time_{s}`` ... of mean_fit_time_s ...; ``fit_time`` and
    ``mean_fit_time`` alone mean seconds.
    """
    text = str(name).strip().lower().replace("{", "").replace("}", "")
    if text in ("fit_time", "mean_fit_time"):
        return f"{text}_s"
    return text if text in _FIT_TIME_NAMES else None


# mean_accuracy: written to the rows of every process (Eq. 8: correct
# predictions over traces). In the validation and test rows it is an
# evaluation pass (eval mode, the epoch's final weights). In the train row it
# is the running accuracy of the epoch's own optimisation passes: the logits of
# each batch as it was trained, in train mode, the weights changing within the
# epoch (Keras fit's "acc"), not a separate evaluation of the training traces.
# With cross-validation the aggregate also averages it over the folds, the
# paper's mean accuracy.
FOLD_MEAN_METRICS = {"mean_accuracy": "accuracy"}


# Base metric a requested mean is computed from, first, with the fields it
# writes; the group moves only when the base metric itself is not requested.
MEAN_DEPENDENCIES = {
    "mean_accuracy": ("accuracy", "accuracy_correct", "accuracy_total"),
    "mean_fit_time_s": ("fit_time_s",),
    "mean_fit_time_h": ("fit_time_h",),
    "mean_fit_time_per_epoch": ("fit_time_per_epoch",),
}


def split_mean_dependencies(results: Mapping[str, Any], cfg: dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    (requested, dependencies) of one process's results.

    A base metric recorded only because its mean was requested goes to
    ``dependencies``; everything else stays with the process.
    """
    requested = {_canonical_metric_name(name) for name, _opts in _parse_metrics_list(cfg)}
    moved = {
        dep
        for mean, deps in MEAN_DEPENDENCIES.items()
        if mean in requested and deps[0] not in requested
        for dep in deps
    }
    return (
        {k: v for k, v in results.items() if k not in moved},
        {k: v for k, v in results.items() if k in moved},
    )


# The loss the run optimises, as trainer.loss defines it (type and arguments), per epoch:
#   training_objective    the mean loss of the epoch's optimisation passes (train rows)
#   validation_objective  the same loss over the held-out traces after the epoch (validation rows)
# Requestable in metrics_to_log; a row that holds one names the loss beside it (loss_type).
OBJECTIVE_METRICS = ("training_objective", "validation_objective")


def loss_signature(cfg: Mapping[str, Any]) -> str:
    """trainer.loss as one string, e.g. CrossEntropyLoss(reduction=mean, from_logits=True): the loss_type
    column written beside a recorded objective, so the row says which loss the number is."""
    loss = ((cfg.get("trainer") or {}).get("loss") or {}) if isinstance(cfg, Mapping) else {}
    kind = str(loss.get("type") or "unknown")
    args = loss.get("args") if isinstance(loss.get("args"), Mapping) else {}
    inner = ", ".join(f"{k}={v}" for k, v in args.items() if v is not None and v != "")
    return f"{kind}({inner})" if inner else kind


# Written by the training loop for its own bookkeeping (the loss it optimises
# when nobody asked for it, its reduction, counters of the evaluation pass).
# They are not metrics of metrics_to_log and no file stores them; the
# cross-validation diagnostics read the losses from the in-memory trajectory.
LOOP_ONLY_COLUMNS = frozenset({
    "training_objective",
    "validation_objective",
    "loss_type",
    "train_loss_reduction",
    "running_train_accuracy",
    "evaluated_traces",
})

# The metrics a metrics_to_log entry gives a value, {name: {option: value}}: the option compute_metrics
# reads and its default when the entry is a bare name (requested_column names the column after it).
METRIC_OPTIONS: Dict[str, Tuple[str, Any]] = {
    "top_k_accuracy": ("k", 5),          # correct when the true label is among the k most probable
    "rank_k_success": ("k", 1),          # 1 when the final key rank is <= k
    "traces_to_ge_k": ("k", 1),          # first trace count with GE <= k (stably)
    "sr_curve": ("k", 1),                # success = the key ranked within the first k
    "auc_sr": ("k", 1),
    "time_to_recover": ("k", 1),         # first trace count with rank <= k
    "ge_at_traces": ("traces", 2000),    # GE after this many attack traces
    "rank_at_traces": ("traces", 2000),  # rank after this many attack traces
    "rank_percentile": ("p", 50),        # the p-th percentile of the rank curve
    "ge_reduction_rate": ("window", 5),  # the GE slope over this many points
}


def requested_column(name: Any, opts: Optional[Mapping[str, Any]] = None) -> str:
    """
    Column a metrics_to_log entry is recorded under (compute_metrics and the
    training loop write exactly these names).
    """
    opts = dict(opts or {})
    metric = _canonical_metric_name(str(name))

    def _budget() -> int:
        return int(opts.get("traces", opts.get("t", opts.get("n", 2000))))

    has_budget = any(key in opts for key in ("traces", "t", "n"))
    if metric == "top_k_accuracy":
        return f"top_{int(opts.get('k', 5))}_accuracy"
    if metric in ("sr_curve", "auc_sr"):
        return f"{metric}@{int(opts['k'])}" if "k" in opts else metric
    if metric == "ge_at_traces":
        return f"ge@{_budget()}" if has_budget else metric
    if metric == "rank_at_traces":
        return f"rank@{_budget()}" if has_budget else metric
    if metric == "traces_to_ge_k":
        return f"traces_to_ge_{int(opts['k'])}" if "k" in opts else metric
    if metric == "rank_percentile":
        return f"rank_{int(float(opts['p']))}_percentile" if "p" in opts else metric
    if metric == "rank_k_success":
        return f"rank_k_success@{int(opts.get('k', 1))}"
    if metric == "time_to_recover":
        return f"time_to_recover@{int(opts['k'])}" if "k" in opts else metric
    if metric == "ge_reduction_rate":
        return f"ge_reduction_rate@{int(opts['window'])}" if "window" in opts else metric
    return metric


def requested_columns(cfg: dict) -> List[Tuple[str, str]]:
    """(column, canonical metric) of every metrics_to_log entry, in config order."""
    columns: List[Tuple[str, str]] = []
    for name, opts in _parse_metrics_list(cfg):
        item = (requested_column(name, opts), _canonical_metric_name(str(name)))
        if item not in columns:
            columns.append(item)
    return columns


def split_requested(results: Mapping[str, Any], cfg: dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    (metrics, dependencies) of one process's results, for the data files.

    ``metrics`` holds exactly the columns metrics_to_log names, in config
    order. ``dependencies`` holds every other recorded value a requested
    metric is derived from or read with: the base of a requested mean, exact
    counts, first crossings, selected_epoch, the key-recovery outcome. The
    training loop's own bookkeeping (LOOP_ONLY_COLUMNS) is stored nowhere.
    """
    requested = [column for column, _metric in requested_columns(cfg)]
    metrics = {column: results[column] for column in requested if column in results}
    # The loss the run optimises is bookkeeping until a row records it: then loss_type (the loss's
    # name and arguments) is what that number depends on, and goes with it.
    loop_only = set(LOOP_ONLY_COLUMNS)
    if any(name in metrics for name in OBJECTIVE_METRICS):
        loop_only.discard("loss_type")
    dependencies = {
        key: value
        for key, value in results.items()
        if key not in metrics and key not in loop_only
    }
    return metrics, dependencies


def requested_fit_time_metrics(cfg: dict) -> List[str]:
    """
    Fit-time columns a run records, in canonical order.

    A requested mean also records the fit time it is the mean of.
    """
    requested = {
        canonical_fit_time_name(name) for name, _opts in _parse_metrics_list(cfg)
    }
    columns: List[str] = []
    for name in FIT_TIME_METRICS:
        if name in requested or f"mean_{name}" in requested:
            columns.append(name)
        if f"mean_{name}" in requested:
            columns.append(f"mean_{name}")
    return columns


def fit_time_values(requested: List[str], *, seconds: float, epoch: int) -> Dict[str, float]:
    """
    Values of the requested fit-time columns after ``epoch`` epochs.

    For one run mean_<name> equals <name>; the mean over the folds is written
    by the cross-validation aggregate.
    """
    seconds = float(seconds)
    values = {
        "fit_time_s": seconds,
        "fit_time_h": seconds / 3600.0,
        "fit_time_per_epoch": seconds / max(int(epoch), 1),
    }
    values.update({f"mean_{name}": value for name, value in list(values.items())})
    return {name: values[name] for name in requested}


def requested_model_size_metrics(cfg: dict) -> List[str]:
    """Model-size columns a run records (MODEL_SIZE_METRICS), in canonical order."""
    requested = {_canonical_metric_name(name) for name, _opts in _parse_metrics_list(cfg)}
    return [name for name in MODEL_SIZE_METRICS if name in requested]


def requested_loop_state_metrics(cfg: dict) -> List[str]:
    """Loop-state columns a run records (LOOP_STATE_METRICS), in canonical order."""
    requested = {_canonical_metric_name(name) for name, _opts in _parse_metrics_list(cfg)}
    return [name for name in LOOP_STATE_METRICS if name in requested]


def loop_state_values(requested: List[str], *, learning_rate: Optional[float]) -> Dict[str, Optional[float]]:
    """Values of the requested loop-state columns. η_t (eta_t) is the rate the epoch's last optimizer
    step used (read from the optimizer before the scheduler advanced), so a completed one-cycle ends
    at max / (div_factor x final_div_factor); the scheduler's state after its last step is one past
    the cycle and is never written."""
    if not requested:
        return {}
    rate = None if learning_rate is None else float(learning_rate)
    return {name: rate for name in requested if name == "η_t"}


def eta_t(optimizer: Any) -> Optional[float]:
    """η_t - the learning rate at iteration or epoch t: the rate the optimizer's first parameter group holds
    now (the rate its next or last step uses); None when the optimizer has none. The training loop reads it
    after each step, before the scheduler moves on, and the metric η_t records the epoch's last step.

    Who writes it so. Loshchilov and Hutter, SGDR (ICLR 2017), define their cosine schedule as
    η_t = η_min + ½ (η_max − η_min) (1 + cos(π T_cur / T_i)), T_cur the epochs since the last restart,
    advanced at every batch; PyTorch documents its CosineAnnealingLR scheduler with the same η_t. Other texts
    give the quantity another letter: ε_k, the learning-rate schedule ε_1, ε_2, ... of SGD in Goodfellow,
    Bengio and Courville, Deep Learning (2016, Algorithm 8.1); α_k, the step size of Bottou, Curtis and
    Nocedal, Optimization Methods for Large-Scale Machine Learning (SIAM Review 2018).

    Why the subscript. Under a schedule the learning rate is a function of the step, not one number:
    trainer.optimizer.learning_rate is the setting (under OneCycleLR the maximum, Smith and Topin's max_lr)
    and η_t is the value the schedule gives step t. Without a schedule η_t is the setting at every t.

    Its purpose. The record shows the schedule the run followed, against the one the paper states: one-cycle
    (Smith and Topin, Super-Convergence, 2019; the policy Zaid et al. TCHES 2020(1) train with) rises from
    max / div_factor to max and falls to max / (div_factor x final_div_factor), so the last epoch's η_t is
    the evidence that the cycle completed (projects/ZvW D1a). Large early rates move fast across the loss
    surface, small late ones let stochastic gradient descent settle: stochastic approximation (Robbins and
    Monro, 1951) converges only when the step sizes decrease (their sum diverges, the sum of their squares
    does not)."""
    groups = getattr(optimizer, "param_groups", None) or []
    return float(groups[0]["lr"]) if groups and "lr" in groups[0] else None


def model_size_values(requested: List[str], model: torch.nn.Module) -> Dict[str, int]:
    """Values of the requested model-size columns of ``model``."""
    if not requested:
        return {}
    from utils.parameter_count import trainable_parameter_count

    values = {"trainable_parameters": trainable_parameter_count(model)}
    return {name: values[name] for name in requested}


_NON_SCALAR_WARNED = False
# Metrics written as arrays (artifacts), not as one number per epoch: they cannot be a monitor.
NON_SCALAR_METRICS = frozenset({"sr_curve", "full_ranks", "mean_attack_rank_curve", "LDD"})
VALID_METRIC_PROCESSES = {
    # Time of the optimisation passes (train) and of the evaluation (validation).
    "fit_time_s": ["train", "validation"],
    "fit_time_h": ["train", "validation"],
    "fit_time_per_epoch": ["train", "validation"],
    "mean_fit_time_s": ["train", "validation"],
    "mean_fit_time_h": ["train", "validation"],
    "mean_fit_time_per_epoch": ["train", "validation"],

    # Size of the trained model (train rows only).
    "trainable_parameters": ["train"],
    # The learning rate the epoch's last optimizer step used (train rows only).
    "η_t": ["train"],
    # The loss trainer.loss defines, per epoch: the training passes' mean (train rows), the held-out
    # traces' (validation rows); loss_type beside it names the loss.
    "training_objective": ["train"],
    "validation_objective": ["validation"],

    # Classification / training-safe metrics
    "cross_entropy": ["train", "validation", "test"],
    "accuracy": ["train", "validation", "test"],
    "mean_accuracy": ["train", "validation", "test"],
    "top_k_accuracy": ["train", "validation", "test"],
    "top_5_accuracy": ["train", "validation", "test"],
    "train_loss_reduction": ["train"],

    # Single-run SCA rank metrics
    "full_ranks": ["validation", "test"],
    "key_rank": ["validation", "test"],
    "mean_rank": ["validation", "test"],
    "variance_rank": ["validation", "test"],
    "median_rank": ["validation", "test"],
    "rank_percentile": ["validation", "test"],
    "normalized_mean_rank": ["validation", "test"],
    "normalized_variance_rank": ["validation", "test"],
    "rank_k_success": ["validation", "test"],
    "time_to_recover": ["validation", "test"],
    "ge_reduction_rate": ["validation", "test"],
    "rank_at_traces": ["validation", "test"],

    # Repeated-attack metrics
    "guessing_entropy": ["validation", "test"],
    "mean_attack_rank_curve": ["validation", "test"],
    "sr_curve": ["validation", "test"],
    "auc_ge": ["validation", "test"],
    "auc_sr": ["validation", "test"],
    "ge_at_traces": ["validation", "test"],
    "success_rate": ["validation", "test"],
    "traces_to_ge_1": ["validation", "test"],
    "traces_to_ge_k": ["validation", "test"],
    "traces_to_sr_1": ["validation", "test"],
    "per_attack_curves": ["validation", "test"],
    "final_ranks": ["validation", "test"],
    # The papers' N̄tGE (Zaid Sect. 5): each attack's own stable crossing of rank 1, averaged.
    "per_attack_traces_to_ge_1": ["validation", "test"],
    # LDD metrics
    "LDD": ["validation", "test"],
    "Lm": ["validation", "test"],
}
# =============================================================================
# Config helpers
# =============================================================================
def _metrics_cfg(cfg: dict) -> dict:
    mc = cfg.get("metrics_cfg") or {}
    return mc if isinstance(mc, dict) else {}
def canonical_process_name(process: str) -> str:
    p = str(process).strip().lower()
    aliases = {
        "validate": "validation",
        "val": "validation",
        "valid": "validation",
    }
    return aliases.get(p, p)


def is_metric_valid_for_process(metric: str, process: str) -> bool:
    metric = _canonical_metric_name(metric)
    process = canonical_process_name(process)

    # materialized variants
    if metric.startswith("top_") and metric.endswith("_accuracy"):
        metric = "top_k_accuracy"

    if metric.startswith("sr_curve_"):
        metric = "sr_curve"

    valid = VALID_METRIC_PROCESSES.get(metric)

    # Unknown/custom metrics are allowed everywhere unless explicitly registered.
    if valid is None:
        return True

    return process in valid


def valid_processes_for_metric(metric: str) -> list[str]:
    metric = _canonical_metric_name(metric)

    if metric.startswith("top_") and metric.endswith("_accuracy"):
        metric = "top_k_accuracy"

    if metric.startswith("sr_curve_"):
        metric = "sr_curve"

    return VALID_METRIC_PROCESSES.get(metric, ["train", "validation", "test"])

def _get_attack_byte(cfg: dict) -> int:
    sca = cfg.get("sca") or {}
    if not isinstance(sca, dict):
        raise TypeError("cfg['sca'] must be a mapping")
    if "attack_byte" not in sca:
        raise KeyError(
            "Rank-based SCA metrics require cfg['sca']['attack_byte'] (0..15)."
        )
    b = sca["attack_byte"]
    if isinstance(b, (bool, list, tuple, set, dict, np.ndarray)):
        raise TypeError(
            "sca.attack_byte must be one resolved scalar integer, got "
            f"{type(b).__name__}: {b!r}"
        )
    try:
        numeric = float(b)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"sca.attack_byte must be an integer, got {b!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise TypeError(f"sca.attack_byte must be an integer, got {b!r}")
    b = int(numeric)

    if b < 0 or b > 15:
        raise ValueError(f"attack_byte must be 0..15, got {b}")
    return b

def _topk_rank_threshold(*, k: int, rank_base: int) -> float:
    """Return the top-k threshold under TraceFlow's one-based rank contract."""
    k = int(k)
    if not 1 <= k <= 256:
        raise ValueError(f"top-k metrics require k in [1,256], got {k}")

    rank_base = int(rank_base)
    if rank_base != 1:
        raise ValueError(
            f"Unsupported rank_base={rank_base}. TraceFlow requires sca.rank_base: 1."
        )

    return float(k)


def _get_rank_step(cfg: dict, default: int = 10) -> int:
    v = cfg.get("validation") or {}
    t = cfg.get("trainer") or {}
    raw = (
        v["rank_step"]
        if isinstance(v, dict) and "rank_step" in v
        else t.get("rank_step", default) if isinstance(t, dict) else default
    )
    if isinstance(raw, (bool, list, tuple, set, dict, np.ndarray)):
        raise TypeError(f"rank_step must be one scalar integer, got {raw!r}")
    try:
        step = int(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"rank_step must be an integer, got {raw!r}") from exc
    if step <= 0:
        raise ValueError(f"rank_step must be > 0, got {step}")
    return step


def _get_ce_eps(cfg: dict, default: float = 1e-12) -> float:
    raw = _metrics_cfg(cfg).get("ce_eps", default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"metrics_cfg.ce_eps must be numeric, got {raw!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"metrics_cfg.ce_eps must be finite and > 0, got {raw!r}")
    return value


def _get_rank_log_eps(cfg: dict, default: float = 1e-40) -> float:
    raw = _metrics_cfg(cfg).get("rank_log_eps", default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"metrics_cfg.rank_log_eps must be numeric, got {raw!r}"
        ) from exc
    # 0 disables the per-trace floor (exact float64 log-likelihood).
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            f"metrics_cfg.rank_log_eps must be finite and in [0, 1], got {raw!r}"
        )
    return value


def _get_rank_base(cfg: dict) -> int:
    sca = cfg.get("sca") or {}
    rb = int(sca.get("rank_base", 1))
    if rb != 1:
        raise ValueError(
            f"Unsupported sca.rank_base={rb}. TraceFlow exposes one-based ranks only; "
            "set sca.rank_base: 1."
        )
    return rb


def pop_key_recovery_evidence(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Remove and return the reserved key-recovery evidence payload.

    Call this immediately after ``compute_metrics`` and before generic metric
    persistence, ClearML logging, or terminal metric formatting.
    """
    if not isinstance(results, dict):
        raise TypeError(f"results must be dict, got {type(results)}")

    evidence = results.pop(KEY_RECOVERY_EVIDENCE_KEY, None)
    if evidence is None:
        return None
    if not isinstance(evidence, dict):
        raise TypeError(
            f"{KEY_RECOVERY_EVIDENCE_KEY} must be dict, got {type(evidence)}"
        )
    return evidence


def _get_leakage_model(cfg: dict) -> str:
    sca = cfg.get("sca") or {}
    if not isinstance(sca, dict):
        raise TypeError("cfg['sca'] must be a mapping")
    if "leakage_model" not in sca:
        raise KeyError("Rank-based SCA metrics require cfg['sca']['leakage_model']")
    raw = sca["leakage_model"]
    if isinstance(raw, (list, tuple, set, dict)):
        raise TypeError(
            "sca.leakage_model must be one resolved scalar value, got "
            f"{type(raw).__name__}: {raw!r}"
        )
    if isinstance(raw, np.ndarray):
        if raw.shape != ():
            raise TypeError(
                "sca.leakage_model must be one resolved scalar value, got "
                f"array shape {raw.shape}"
            )
        raw = raw.item()
    return normalize_leakage_model(raw)


def _get_strict_prob_check(cfg: dict, default: bool = True) -> bool:
    value = _metrics_cfg(cfg).get("strict_prob_check", default)
    if not isinstance(value, bool):
        raise TypeError(
            f"metrics_cfg.strict_prob_check must be true or false, got {value!r}"
        )
    return value


def _get_fail_on_metric_error(cfg: dict, default: bool = True) -> bool:
    """Fail the evaluation pass when a requested metric cannot be computed."""
    value = _metrics_cfg(cfg).get("fail_on_metric_error", default)
    if not isinstance(value, bool):
        raise TypeError(
            f"metrics_cfg.fail_on_metric_error must be true or false, got {value!r}"
        )
    return value


def _get_prob_atol(cfg: dict, default: float = 1e-3) -> float:
    v = _metrics_cfg(cfg).get("prob_atol", default)
    try:
        v = float(v)
    except Exception:
        v = float(default)
    if not np.isfinite(v) or v < 0.0:
        v = float(default)
    return float(v)


def _get_prob_rtol(cfg: dict, default: float = 1e-3) -> float:
    v = _metrics_cfg(cfg).get("prob_rtol", default)
    try:
        v = float(v)
    except Exception:
        v = float(default)
    if not np.isfinite(v) or v < 0.0:
        v = float(default)
    return float(v)


def _sanitize_reduction(reduction: Any, default: str = "mean") -> str:
    r = str(default if reduction is None else reduction).lower().strip()
    if r not in ("none", "mean", "sum"):
        raise ValueError(
            "trainer.loss.args.reduction must be 'none', 'mean', or 'sum', "
            f"got {reduction!r}"
        )
    return r


# =============================================================================
# Rank curve path
# =============================================================================
def _get_or_compute_rank_curve(
    *,
    probs: torch.Tensor,
    labels: torch.Tensor,
    metadata: Any,
    cfg: dict,
    cached: Optional[np.ndarray],
    log_probs: torch.Tensor,
    process: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Lazily compute single-run rank curve only when a rank metric is requested.
    Evaluation rank metrics require metadata and fail if it is unavailable.
    The attack consumes float64 ``log_probs``; ``probs`` is used for checks.
    """
    if cached is not None:
        return cached

    if metadata is None:
        raise ValueError("Rank-based evaluation metrics require metadata")

    _validate_rank_curve_inputs(probs, metadata, cfg)

    rank_base = _get_rank_base(cfg)
    leakage_model = _get_leakage_model(cfg)
    tb = _get_attack_byte(cfg)
    step = _get_rank_step(cfg, default=10)

    sca = cfg.get("sca") or {}
    # The key mode is the database type's (sca/dataset_types.py); the sca.variable_key alias
    # still counts. Before database types this read the alias only.
    variable_key = bool(sca.get("variable_key", False)) or str(resolved_variant(cfg) or "fixed_key") != "fixed_key"
    aggregation = str(sca.get("variable_key_aggregation", "weighted")).lower()
    if aggregation not in ("weighted", "equal"):
        raise ValueError(
            f"Invalid sca.variable_key_aggregation={aggregation!r}. "
            "Expected 'weighted' or 'equal'."
        )

    rank_eps = _get_rank_log_eps(cfg, default=1e-40)
    strict_prob_check = _get_strict_prob_check(cfg, default=True)
    prob_atol = _get_prob_atol(cfg, default=1e-3)
    prob_rtol = _get_prob_rtol(cfg, default=1e-3)

    curve = full_ranks_curve(
        preds=log_probs,
        metadata=metadata,
        target_byte=tb,
        step=step,
        eps=rank_eps,
        strict_prob_check=strict_prob_check,
        prob_atol=prob_atol,
        prob_rtol=prob_rtol,
        check_fixed_key=not variable_key,
        variable_key=variable_key,
        variable_key_aggregation=aggregation,
        rank_base=rank_base,
        leakage_model=leakage_model,
        target=label_target(cfg),
        preds_are_log_probs=True,
        # the endpoint's tie-break stream of this process (perform_attack): a tied endpoint ranks the same here
        tie_break_seed=endpoint_tie_seed(cfg, process),
    )

    if not isinstance(curve, np.ndarray) or curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError("full_ranks_curve returned invalid curve shape; expected (M,2).")

    return curve


def _single_run_curve_equals_first_attack(cfg: dict, process: str) -> bool:
    """True when perform_attacks() repetition 1 is the single-run rank curve."""
    try:
        sca = cfg.get("sca") or {}
        if bool(sca.get("variable_key", False)) or str(resolved_variant(cfg) or "fixed_key") != "fixed_key":
            return False
        plan = attack_plan(cfg, process)
        if (plan["shuffle"] or plan["attack_distribution"] != "full_population"
                or plan["attack_trace_budget"] is not None):
            return False
        return int(_get_attack_step(cfg)) == int(_get_rank_step(cfg, default=10))
    except Exception:
        return False


# =============================================================================
# Metric list parsing + canonicalization
# =============================================================================
def _parse_metrics_list(cfg: dict) -> List[Tuple[str, Dict[str, Any]]]:
    metrics = cfg.get("metrics_to_log", None)
    if not isinstance(metrics, list):
        metrics = cfg.get("metrics", [])
    if metrics is None:
        metrics = []
    if not isinstance(metrics, list):
        raise TypeError(f"cfg['metrics_to_log'] / cfg['metrics'] must be list, got {type(metrics)}")

    parsed: List[Tuple[str, Dict[str, Any]]] = []
    for m in metrics:
        renamed = RENAMED_METRICS.get(m if isinstance(m, str) else next(iter(m), None) if isinstance(m, dict) else None)
        if renamed:
            raise ValueError(f"metrics_to_log: {m!r} is now {renamed!r} (the rate the epoch's last optimizer step "
                             "used; trainer.optimizer.learning_rate is the setting)")
        if isinstance(m, str):
            parsed.append((m, {}))
        elif isinstance(m, dict) and len(m) == 1:
            name, opts = next(iter(m.items()))
            if opts is None:
                opts = {}
            if not isinstance(opts, dict):
                raise TypeError(f"Metric opts must be dict for '{name}', got {type(opts)}")
            parsed.append((str(name), opts))
        else:
            raise TypeError(f"Invalid metric specification: {m!r}")
    return parsed


def _canonical_metric_name(name: str) -> str:
    n = str(name).strip()

    fit_time = canonical_fit_time_name(n)
    if fit_time is not None:
        return fit_time

    if n == "top_5_accuracy":
        return "top_k_accuracy"

    if n in ("ge", "final_ge"):
        return "guessing_entropy"
    if n == "auc_guessing_entropy":
        return "auc_ge"
    if n == "success_rate_curve":
        return "sr_curve"
    if n == "auc_success_rate":
        return "auc_sr"
    if n == "rank_success":
        return "rank_k_success"

    return n


# =============================================================================
# Pred handling
# =============================================================================
def _preds_are_probs(cfg: dict) -> bool:
    value = _metrics_cfg(cfg).get("preds_are_probs", False)
    if not isinstance(value, bool):
        raise TypeError(
            "metrics_cfg.preds_are_probs must be true or false, "
            f"got {value!r}"
        )
    return value


# The model outputs _ensure_logits_and_probs turns into probabilities and log-probabilities.
_ACCEPTED_REPRESENTATIONS = ("logits", "probabilities", "log_probabilities")


def _ensure_logits_and_probs(
    preds: torch.Tensor,
    cfg: dict,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, bool]:
    """
    Return (original_logits_or_none, probabilities, log_probabilities,
    input_was_probabilities).

    Evaluation supports Identity, Softmax, and LogSoftmax. Convert their
    respective outputs to the probabilities required by downstream metrics;
    preserve original logits for cross-entropy. Explicit probability inputs
    are validated and passed through without a second transformation.

    Numeric contract: ``probabilities`` and ``log_probabilities`` are
    ``float64``. For a logits (Identity) head, ``log_probabilities`` is
    ``log_softmax`` evaluated in float64, so no class can underflow; the
    rank/GE/LDD attacks consume this tensor. Cross-entropy still uses the
    original logits (same dtype as the training loss). A Softmax head or
    ``preds_are_probs=true`` cannot recover what float32 already lost:
    ``log_probabilities`` is then ``log(p)`` and may contain ``-inf`` where
    ``p == 0``; the attack floors it at ``log(rank_log_eps)``.
    """
    if not isinstance(preds, torch.Tensor):
        raise TypeError(f"preds must be torch.Tensor, got {type(preds)}")
    if preds.dim() != 2:
        raise ValueError(f"preds must have shape [N,C], got {tuple(preds.shape)}")

    if int(preds.shape[0]) <= 0 or int(preds.shape[1]) <= 0:
        raise ValueError(f"preds must be non-empty, got {tuple(preds.shape)}")
    if not bool(torch.isfinite(preds).all().item()):
        raise ValueError("preds contains NaN or infinite values")

    evaluation_cfg = cfg["trainer"]["evaluation"]
    if not isinstance(evaluation_cfg, Mapping):
        raise TypeError("trainer.evaluation must be a mapping")
    transform_cfg = evaluation_cfg.get("output_transformation")
    if not isinstance(transform_cfg, Mapping):
        raise TypeError("trainer.evaluation.output_transformation must be a resolved mapping")
    # What this function accepts, stated here and not read from the YAML: each
    # representation has its probability adapter below. The builder refuses a
    # transformation producing anything else and reports the one it produces.
    transform = build_output_transformation_from_config(
        evaluation_cfg, expected_representation=_ACCEPTED_REPRESENTATIONS,
    ).to(preds.device)
    representation = transform.representation
    if not _preds_are_probs(cfg):
        # The model/training contract supplies raw logits. Never reinterpret a
        # probability-looking tensor heuristically, because that changes CE.
        # float64 before the transform: a float32 softmax underflows below
        # ~1.4e-45 and is subnormal below ~1.2e-38, which distorts per-trace
        # log-likelihoods before they are summed.
        transformed = transform(preds.detach().to(dtype=torch.float64))
        scores64 = transformed
        if representation == "logits":
            log_probs = torch.log_softmax(scores64, dim=1)
            probs = torch.exp(log_probs)
        elif representation == "log_probabilities":
            log_probs = scores64
            probs = torch.exp(log_probs)
        else:
            probs = scores64
            log_probs = torch.log(probs)
        return preds, probs, log_probs, False

    atol = _get_prob_atol(cfg)
    rtol = _get_prob_rtol(cfg)
    if bool((preds < -atol).any().item()) or bool((preds > 1.0 + atol).any().item()):
        raise ValueError("Configured probability predictions contain values outside [0,1]")
    row_sum = preds.sum(dim=1)
    if not torch.allclose(
        row_sum,
        torch.ones_like(row_sum),
        atol=atol,
        rtol=rtol,
    ):
        raise ValueError("Configured probability predictions do not sum to 1 per row")
    probs = preds.detach().to(dtype=torch.float64)
    return None, probs, torch.log(probs), True


# =============================================================================
# Metric groups + validation
# =============================================================================
def _validate_rank_curve_inputs(probs: torch.Tensor, metadata: Any, cfg: dict) -> None:
    if metadata is None:
        raise ValueError(
            "Rank-based SCA metrics requested but metadata is None. "
            "Make sure your loader is called with return_metadata=True and propagates metadata through the DataLoader."
        )
    if probs.dim() != 2:
        raise ValueError(f"probs must be [N,C], got {tuple(probs.shape)}")

    leakage_model = _get_leakage_model(cfg)
    validate_num_classes_for_leakage(int(probs.shape[1]), leakage_model)

    n = int(probs.shape[0])
    if not hasattr(metadata, "__len__"):
        raise TypeError("metadata must be sized (len(metadata) must work).")
    if len(metadata) != n:
        raise ValueError(f"metadata length {len(metadata)} does not match N={n} predictions.")


# =============================================================================
# CE from probabilities
# =============================================================================
def _cross_entropy_from_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    eps: float,
    reduction: str,
) -> torch.Tensor:
    if probs.dim() != 2:
        raise ValueError(f"probs must be [N,C], got {tuple(probs.shape)}")
    n = int(probs.shape[0])
    if labels.dim() != 1 or int(labels.shape[0]) != n:
        raise ValueError(f"labels must be [N], got {tuple(labels.shape)} for N={n}")

    p = probs.clamp_min(float(eps))
    idx = torch.arange(n, device=p.device)
    ce = -torch.log(p[idx, labels])

    if reduction == "none":
        return ce
    if reduction == "sum":
        return ce.sum()
    return ce.mean()


# =============================================================================
# Rank curve helpers
# =============================================================================
def _slope_over_first_points(rank_curve: np.ndarray, k: int) -> float:
    if not isinstance(rank_curve, np.ndarray) or rank_curve.ndim != 2 or rank_curve.shape[1] != 2:
        return float("nan")

    n = int(rank_curve.shape[0])
    k = int(k)
    if n < 2 or k < 2:
        return float("nan")

    k = min(k, n)
    x = rank_curve[:k, 0].astype(np.float64)
    y = rank_curve[:k, 1].astype(np.float64)

    vx = np.var(x)
    if vx <= 0:
        return float("nan")

    slope = np.cov(x, y, ddof=0)[0, 1] / vx
    return float(slope)


def _rank_at_trace_budget(rank_curve: np.ndarray, budget: int) -> float:
    """Rank at exactly ``budget`` traces; the budget must be a curve point."""
    rank_curve = np.asarray(rank_curve, dtype=np.float64)
    if rank_curve.ndim != 2 or rank_curve.shape[1] != 2 or rank_curve.shape[0] == 0:
        raise ValueError(f"rank curve must be non-empty [M,2], got {rank_curve.shape}")

    b = int(budget)
    if b <= 0:
        raise ValueError("trace budget must be > 0")

    x = rank_curve[:, 0].astype(np.int64)
    hits = np.flatnonzero(x == b)
    if hits.size != 1:
        # A clamped or rounded budget would publish GE@b for another trace count.
        raise ValueError(
            f"trace budget {b} is not a point of the rank curve (points "
            f"{int(x[0])}, {int(x[1]) if x.size > 1 else int(x[0])}, ..., "
            f"{int(x[-1])}); use a multiple of perform_attack.attack_step "
            "no larger than the attacked population"
        )
    return float(rank_curve[int(hits[0]), 1])


def _warn_if_non_scalar_requested(requested):
    global _NON_SCALAR_WARNED

    if _NON_SCALAR_WARNED:
        return

    names = {n for n, _ in requested}
    triggered = names.intersection(NON_SCALAR_METRICS)

    if triggered:
        logger.info(
            "Non-scalar metrics detected (saved as artifacts, not scalar logs): %s",
            sorted(triggered),
        )
        _NON_SCALAR_WARNED = True


def _validated_repeated_attack_summary(summary: Any) -> Dict[str, Any]:
    """Validate repeated-attack evidence and return canonical derived curves."""
    if not isinstance(summary, dict):
        raise TypeError(
            f"perform_attacks must return a dict, got {type(summary).__name__}"
        )

    raw_curves = summary.get("per_attack_curves")
    try:
        curves = np.asarray(raw_curves, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("per_attack_curves must be a rectangular numeric array") from exc
    if curves.ndim != 3 or curves.shape[2] != 2:
        raise ValueError(
            "per_attack_curves must have shape [repetitions, points, 2], "
            f"got {curves.shape}"
        )
    repetitions, points, _columns = curves.shape
    if repetitions <= 0 or points <= 0 or not np.all(np.isfinite(curves)):
        raise ValueError("per_attack_curves must be non-empty and finite")

    trace_axes = curves[:, :, 0]
    ranks = curves[:, :, 1]
    if not np.allclose(trace_axes, trace_axes[0], rtol=0.0, atol=0.0):
        raise ValueError("Repeated attacks returned inconsistent trace axes")
    traces = trace_axes[0]
    if np.any(traces <= 0) or (points > 1 and np.any(np.diff(traces) <= 0)):
        raise ValueError("Repeated-attack trace axis must be positive and increasing")
    if not np.all(traces == np.floor(traces)):
        raise ValueError("Repeated-attack trace counts must be integers")
    if np.any(ranks < 1.0) or np.any(ranks > 256.0):
        raise ValueError("Repeated-attack ranks must use the one-based range [1,256]")
    if not np.all(ranks == np.floor(ranks)):
        raise ValueError("Each per-attack rank must be an integer")

    configured_repetitions = int(summary.get("attack_repetitions", repetitions))
    if configured_repetitions != repetitions:
        raise ValueError(
            "attack_repetitions disagrees with per_attack_curves: "
            f"{configured_repetitions} != {repetitions}"
        )

    final_ranks = np.asarray(summary.get("final_ranks"), dtype=np.float64)
    if final_ranks.shape != (repetitions,) or not np.all(np.isfinite(final_ranks)):
        raise ValueError(
            f"final_ranks must have shape {(repetitions,)}, got {final_ranks.shape}"
        )
    if not np.allclose(final_ranks, ranks[:, -1], rtol=0.0, atol=0.0):
        raise ValueError("final_ranks do not match per_attack_curves endpoints")

    mean_curve = np.stack([traces, np.mean(ranks, axis=0)], axis=1)
    reported_mean_curve = np.asarray(
        summary.get("mean_attack_rank_curve"), dtype=np.float64
    )
    if reported_mean_curve.shape != mean_curve.shape or not np.allclose(
        reported_mean_curve, mean_curve, rtol=1e-12, atol=1e-12
    ):
        raise ValueError(
            "mean_attack_rank_curve is missing or inconsistent with "
            "per_attack_curves"
        )

    guessing_entropy = float(np.mean(final_ranks))
    reported_ge = float(summary.get("guessing_entropy"))
    if not math.isclose(reported_ge, guessing_entropy, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("guessing_entropy does not equal mean(final_ranks)")

    # Endpoint trials: every attack, or the one whole-population endpoint
    # however many orderings were drawn (perform_attack.py). A record written
    # before attack_trace_budget has no endpoint_rank_order_invariant: its
    # endpoint is order-invariant exactly when it is not disjoint.
    protocol = summary.get("attack_protocol") or {}
    invariant = protocol.get("endpoint_rank_order_invariant")
    if invariant is None:
        invariant = protocol.get("attack_distribution") != "disjoint"
    trial_ranks = final_ranks[:1] if invariant else final_ranks
    sr_trials = int(trial_ranks.size)
    if int(summary.get("sr_trials", -1)) != sr_trials:
        raise ValueError("sr_trials disagrees with the attack distribution")
    successful_attacks = int(np.count_nonzero(trial_ranks <= 1.0))
    success_rate = float(successful_attacks / sr_trials)
    if int(summary.get("successful_attacks")) != successful_attacks:
        raise ValueError("successful_attacks disagrees with final_ranks")
    reported_sr = float(summary.get("success_rate"))
    if not math.isclose(reported_sr, success_rate, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("success_rate does not equal rank-1 successes/trials")

    canonical = dict(summary)
    canonical["per_attack_curves"] = curves
    canonical["mean_attack_rank_curve"] = mean_curve
    canonical["final_ranks"] = final_ranks.astype(np.int64).tolist()
    canonical["attack_repetitions"] = repetitions
    canonical["successful_attacks"] = successful_attacks
    canonical["sr_trials"] = sr_trials
    canonical["guessing_entropy"] = guessing_entropy
    canonical["success_rate"] = success_rate
    canonical["_rank_matrix"] = ranks
    canonical["_trace_axis"] = traces
    return canonical


def _repeated_sr_curve(summary: Dict[str, Any], *, k: int) -> np.ndarray:
    threshold = _topk_rank_threshold(k=k, rank_base=1)
    ranks = np.asarray(summary["_rank_matrix"], dtype=np.float64)
    traces = np.asarray(summary["_trace_axis"], dtype=np.float64)
    sr = np.mean(ranks <= threshold, axis=0, dtype=np.float64)
    return np.stack([traces, sr], axis=1)


def _first_threshold_trace(
    curve: np.ndarray,
    *,
    threshold: float,
    direction: str,
) -> Optional[int]:
    """Trace count of the first curve point satisfying the threshold; None if never."""
    values = np.asarray(curve[:, 1], dtype=np.float64)
    if direction == "le":
        mask = values <= threshold
    elif direction == "ge":
        mask = values >= threshold
    else:
        raise ValueError(f"Unsupported threshold direction {direction!r}")
    return int(curve[int(np.argmax(mask)), 0]) if np.any(mask) else None


def _stable_threshold_trace(
    curve: np.ndarray,
    *,
    threshold: float,
    direction: str,
) -> Optional[int]:
    """Earliest trace count from which the curve satisfies the threshold
    at every later point (the headline "traces to" statistic). None when the
    threshold is never held to the end of the evaluated population."""
    values = np.asarray(curve[:, 1], dtype=np.float64)
    if direction == "le":
        mask = values <= threshold
    elif direction == "ge":
        mask = values >= threshold
    else:
        raise ValueError(f"Unsupported threshold direction {direction!r}")
    if not np.any(mask):
        return None
    suffix_all = np.logical_and.accumulate(mask[::-1])[::-1]
    stable = np.flatnonzero(suffix_all)
    return int(curve[int(stable[0]), 0]) if stable.size else None


def _threshold_message(
    *, threshold: int, stable: Optional[int], first: Optional[int], population: int
) -> str:
    if stable is not None:
        return (
            f"GE stably <= {threshold} from {stable} traces "
            f"(first crossing at {first})"
        )
    if first is not None:
        return (
            f"GE touched <= {threshold} at {first} traces but did not stay "
            f"there through {population} traces (not stable)"
        )
    return f"GE never reached <= {threshold} within the evaluated trace population"


# =============================================================================
# Main API
# =============================================================================
def compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    cfg: dict,
    metadata: Any = None,
    *,
    process: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute metrics for one evaluation pass.

    Contract:
      - preds: [N,C] logits or probs (cfg['metrics_cfg']['preds_are_probs'] controls)
      - labels: [N] class labels
      - metadata: sized sequence len N (REQUIRED for rank-based metrics; train may pass None)
      - process: train | validation | test. Required when repeated-attack metrics are requested.
    """
    results: Dict[str, Any] = {}

    requested = _parse_metrics_list(cfg)
    if not requested:
        return results

    requested = [(_canonical_metric_name(n), (o or {})) for (n, o) in requested]
    requested = [(n, o) for (n, o) in requested if n not in LOOP_PROVIDED_METRICS]
    wants_mean_accuracy = any(n == "mean_accuracy" for n, _o in requested)
    requested = [(FOLD_MEAN_METRICS.get(n, n), o) for (n, o) in requested]
    requested = [
        item for index, item in enumerate(requested)
        if item[0] not in FOLD_MEAN_METRICS.values()
        or item[0] not in [earlier[0] for earlier in requested[:index]]
    ]
    if not requested:
        return results
    _warn_if_non_scalar_requested(requested)

    normalized_process = (
        canonical_process_name(process)
        if process is not None
        else None
    )
    if normalized_process is not None and normalized_process not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            "process must be train, validation, or test, "
            f"got {process!r}"
        )
    if normalized_process is not None:
        requested = [
            (name, opts)
            for name, opts in requested
            if is_metric_valid_for_process(name, normalized_process)
        ]
        if not requested:
            return results
    if any(name in _REPEATED_ATTACK_METRICS for name, _opts in requested):
        if normalized_process not in {"validation", "test"}:
            raise ValueError(
                "Repeated-attack metrics require "
                "compute_metrics(..., process='validation'|'test')"
            )

    fail_on_metric_error = _get_fail_on_metric_error(cfg, default=True)

    logits, probs, log_probs, treated_as_probs = _ensure_logits_and_probs(preds, cfg)
    n, c = probs.shape

    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    labels = labels.to(device=preds.device, dtype=torch.long)
    if labels.dim() != 1 or int(labels.shape[0]) != int(n):
        raise ValueError(f"labels must have shape [N], got {tuple(labels.shape)} for N={n}")
    if int(labels.numel()) <= 0:
        raise ValueError("labels must not be empty")
    if bool((labels < 0).any().item()) or bool((labels >= int(c)).any().item()):
        raise ValueError(f"labels must be in [0,{int(c) - 1}]")

    c_int = int(c)

    raw_reduction = cfg["trainer"]["loss"]["args"].get("reduction", "mean")
    reduction = _sanitize_reduction(raw_reduction, default="mean")
    ce_eps = _get_ce_eps(cfg, default=1e-12)
    keras_cross_entropy = resolve_framework(cfg).is_keras   # False: PyTorch cross-entropy

    rank_curve: Optional[np.ndarray] = None
    attack_summary: Optional[Dict[str, Any]] = None
    ldd_summary: Optional[Dict[str, Any]] = None
    reuse_unshuffled_attack = (
        metadata is not None
        and normalized_process in {"validation", "test"}
        and any(name in _REPEATED_ATTACK_METRICS for name, _opts in requested)
        and _single_run_curve_equals_first_attack(cfg, normalized_process)
    )

    def _need_rank_curve(name: str) -> bool:
        return name in _RANK_METRICS

    def _need_repeated_attack(name: str) -> bool:
        return name in _REPEATED_ATTACK_METRICS

    def _need_ldd(name: str) -> bool:
        return name in _LDD_METRICS

    def _ensure_rank_curve_cached() -> Optional[np.ndarray]:
        nonlocal rank_curve
        if rank_curve is None and reuse_unshuffled_attack:
            # An unshuffled repeated attack consumes traces 0..N-1 in order,
            # exactly like the single-run curve; the first repetition IS that
            # curve (same eps, byte, leakage model, step), so do not attack twice.
            summary = _ensure_attack_summary_cached()
            if summary is not None:
                rank_curve = np.array(summary["per_attack_curves"][0], dtype=np.float64)
        if rank_curve is None:
            rank_curve = _get_or_compute_rank_curve(
                probs=probs,
                labels=labels,
                metadata=metadata,
                cfg=cfg,
                cached=rank_curve,
                log_probs=log_probs,
                process=process,
            )
        return rank_curve

    def _ensure_attack_summary_cached() -> Optional[Dict[str, Any]]:
        nonlocal attack_summary
        if attack_summary is not None:
            return attack_summary

        if metadata is None:
            return None

        _validate_rank_curve_inputs(probs, metadata, cfg)
        if process is None:
            raise ValueError(
                "Repeated-attack metrics require compute_metrics(..., process='validation'|'test')"
            )
        attack_summary = _validated_repeated_attack_summary(
            perform_attacks(
                preds=log_probs,
                metadata=metadata,
                cfg=cfg,
                process=normalized_process,
                preds_are_log_probs=True,
            )
        )
        return attack_summary

    def _ensure_ldd_summary_cached() -> Optional[Dict[str, Any]]:
        nonlocal ldd_summary
        if ldd_summary is not None:
            return ldd_summary

        if metadata is None:
            return None

        _validate_rank_curve_inputs(probs, metadata, cfg)
        ldd_summary = compute_ldd_lm(
            preds=log_probs, metadata=metadata, cfg=cfg, preds_are_log_probs=True
        )
        return ldd_summary

    for name, opts in requested:
        try:
            if name == "cross_entropy" and keras_cross_entropy:
                # deep_learning.framework: keras: the metric is
                # keras.losses.categorical_crossentropy, the function the Keras
                # loss trains on, so the reported cross_entropy is the
                # objective's own quantity.
                ce_t = cross_entropy_per_trace(cfg, logits=logits, probabilities=probs, targets=labels)
                if reduction != "none":
                    ce_t = ce_t.sum() if reduction == "sum" else ce_t.mean()
                results["cross_entropy"] = (
                    ce_t.detach().cpu().numpy() if reduction == "none" else float(ce_t.item())
                )

            elif name == "cross_entropy":
                if logits is not None:
                    if reduction == "none":
                        results["cross_entropy"] = (
                            F.cross_entropy(logits, labels, reduction=reduction)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    else:
                        results["cross_entropy"] = float(
                            F.cross_entropy(logits, labels, reduction=reduction).item()
                        )
                else:
                    ce_t = _cross_entropy_from_probs(probs, labels, eps=ce_eps, reduction=reduction)
                    if reduction == "none":
                        results["cross_entropy"] = ce_t.detach().cpu().numpy()
                    else:
                        results["cross_entropy"] = float(ce_t.item())
                    if treated_as_probs:
                        logger.warning(
                            "cross_entropy computed from probabilities; ensure your model outputs "
                            "are probabilities (or set metrics_cfg.preds_are_probs accordingly)."
                        )

            elif name == "accuracy":
                pred1 = logits.argmax(dim=1) if logits is not None else probs.argmax(dim=1)
                correct = int((pred1 == labels).sum().item())
                total_count = int(labels.numel())
                results["accuracy"] = float(correct / total_count)
                # Exact count provenance allows presentation as, for example,
                # 0.28% (14 / 5000; fraction=0.002800) without altering the
                # canonical machine-readable [0,1] accuracy value.
                results["accuracy_correct"] = correct
                results["accuracy_total"] = total_count
                if wants_mean_accuracy:
                    results["mean_accuracy"] = results["accuracy"]

            elif name == "top_k_accuracy":
                k = int(opts.get("k", 5))
                if k <= 0:
                    raise ValueError("top_k_accuracy requires k>0")
                if k > c_int:
                    raise ValueError(f"top_k_accuracy requires k<=C (C={c_int}), got k={k}")
                topk = probs.argsort(dim=1, descending=True)[:, :k]
                # An exact count ratio, as accuracy: 29 of 500 is 0.058, not float32's 0.0579999983.
                hits = int((topk == labels.unsqueeze(1)).any(dim=1).sum().item())
                results[f"top_{k}_accuracy"] = float(hits / int(labels.numel()))

            elif _need_repeated_attack(name):
                summary = _ensure_attack_summary_cached()
                if summary is None:
                    raise ValueError(f"Repeated-attack metric {name!r} requires metadata")

                if name == "guessing_entropy":
                    results["guessing_entropy"] = float(summary["guessing_entropy"])

                elif name == "mean_attack_rank_curve":
                    results["mean_attack_rank_curve"] = summary["mean_attack_rank_curve"]

                elif name == "sr_curve":
                    k = int(opts.get("k", 1))
                    payload = _repeated_sr_curve(summary, k=k)
                    if "k" in opts:
                        results[f"sr_curve@{k}"] = payload
                    else:
                        results["sr_curve"] = payload

                elif name == "auc_ge":
                    ge_curve = np.asarray(summary["mean_attack_rank_curve"])
                    results["auc_ge"] = float(
                        np.trapezoid(ge_curve[:, 1], x=ge_curve[:, 0])
                    )

                elif name == "auc_sr":
                    k = int(opts.get("k", 1))
                    sr_curve = _repeated_sr_curve(summary, k=k)
                    value = float(np.trapezoid(sr_curve[:, 1], x=sr_curve[:, 0]))
                    if "k" in opts:
                        results[f"auc_sr@{k}"] = value
                    else:
                        results["auc_sr"] = value

                elif name == "ge_at_traces":
                    budget = int(opts.get("traces", opts.get("t", opts.get("n", 2000))))
                    value = float(
                        _rank_at_trace_budget(summary["mean_attack_rank_curve"], budget)
                    )
                    if any(key in opts for key in ("traces", "t", "n")):
                        results[f"ge@{budget}"] = value
                    else:
                        results["ge_at_traces"] = value

                elif name == "ge_reduction_rate":
                    # The slope of GE (the mean over the attacks), not of one attack's curve.
                    window = int(opts.get("window", 5))
                    if window < 2:
                        raise ValueError("ge_reduction_rate requires window>=2")
                    value = float(-_slope_over_first_points(
                        np.asarray(summary["mean_attack_rank_curve"], dtype=np.float64), k=window
                    ))
                    if "window" in opts:
                        results[f"ge_reduction_rate@{window}"] = value
                    else:
                        results["ge_reduction_rate"] = value

                elif name == "per_attack_curves":
                    results["per_attack_curves"] = summary["per_attack_curves"]

                elif name == "final_ranks":
                    results["final_ranks"] = summary["final_ranks"]
                elif name == "success_rate":
                    results["success_rate"] = float(summary["success_rate"])
                    results["sr_at_1_successes"] = int(summary["successful_attacks"])
                    results["sr_at_1_trials"] = int(summary["sr_trials"])

                elif name in ("traces_to_ge_1", "traces_to_ge_k"):
                    # Headline = STABLE crossing (rank stays <= k to the end);
                    # first_traces_to_ge_k keeps the first, possibly transient,
                    # crossing. Same rule as the key-recovery "recovered" flag.
                    k = 1 if name == "traces_to_ge_1" else int(opts.get("k", 1))
                    _topk_rank_threshold(k=k, rank_base=1)
                    curve = summary["mean_attack_rank_curve"]
                    value = _stable_threshold_trace(
                        curve, threshold=float(k), direction="le"
                    )
                    first_value = _first_threshold_trace(
                        curve, threshold=float(k), direction="le"
                    )
                    suffix = str(k) if (name == "traces_to_ge_1" or "k" in opts) else "k"
                    results[f"traces_to_ge_{suffix}"] = value
                    results[f"first_traces_to_ge_{suffix}"] = first_value
                    if k == 1:
                        results["ge_threshold_status"] = (
                            "reached" if value is not None else "not_reached"
                        )
                        results["traces_to_ge_1_message"] = _threshold_message(
                            threshold=1,
                            stable=value,
                            first=first_value,
                            population=int(np.asarray(curve)[-1, 0]),
                        )

                elif name == "per_attack_traces_to_ge_1":
                    # The papers' N̄tGE (Zaid Sect. 5: "100 NtGE are computed to give the average
                    # value"): every attack's own stable crossing of rank 1, then the mean over the
                    # attacks that reach it. Not traces_to_ge_1, which crosses the mean curve.
                    curves = np.asarray(summary["per_attack_curves"], dtype=np.float64)
                    crossings = [
                        _stable_threshold_trace(curve, threshold=1.0, direction="le") for curve in curves
                    ]
                    reached = [float(c) for c in crossings if c is not None]
                    results["per_attack_traces_to_ge_1"] = float(np.mean(reached)) if reached else None
                    results["per_attack_traces_to_ge_1_std"] = float(np.std(reached)) if reached else None
                    results["per_attack_traces_to_ge_1_reached"] = int(len(reached))
                    results["per_attack_traces_to_ge_1_unreached"] = int(len(crossings) - len(reached))

                elif name == "traces_to_sr_1":
                    sr_curve = _repeated_sr_curve(summary, k=1)
                    results["traces_to_sr_1"] = _stable_threshold_trace(
                        sr_curve, threshold=1.0, direction="ge"
                    )
                    results["first_traces_to_sr_1"] = _first_threshold_trace(
                        sr_curve, threshold=1.0, direction="ge"
                    )

            elif _need_ldd(name):
                summary = _ensure_ldd_summary_cached()
                if summary is None:
                    raise ValueError(f"Research metric {name!r} requires metadata")

                if name == "LDD":
                    results["LDD"] = summary["LDD"]

                elif name == "Lm":
                    results["Lm"] = float(summary["Lm"])

            elif _need_rank_curve(name):
                rc = _ensure_rank_curve_cached()

                if name == "full_ranks":
                    results["full_ranks"] = rc

                elif name == "key_rank":
                    results["key_rank"] = int(rc[-1, 1])

                elif name == "rank_at_traces":
                    budget = int(opts.get("traces", opts.get("t", opts.get("n", 2000))))
                    value = int(_rank_at_trace_budget(rc, budget))
                    if any(k in opts for k in ("traces", "t", "n")):
                        results[f"rank@{budget}"] = value
                    else:
                        results["rank_at_traces"] = value

                elif name == "mean_rank":
                    results["mean_rank"] = float(np.mean(rc[:, 1]))

                elif name == "variance_rank":
                    results["variance_rank"] = float(np.var(rc[:, 1], ddof=0))

                elif name == "median_rank":
                    results["median_rank"] = float(np.median(rc[:, 1]))

                elif name == "rank_percentile":
                    p = float(opts.get("p", 50))
                    if not (0.0 <= p <= 100.0):
                        raise ValueError("rank_percentile requires p in [0,100]")
                    value = float(np.percentile(rc[:, 1], p))
                    if "p" in opts:
                        results[f"rank_{int(p)}_percentile"] = value
                    else:
                        results["rank_percentile"] = value

                elif name == "normalized_mean_rank":
                    rank_base = float(_get_rank_base(cfg))
                    denom = 255.0
                    results["normalized_mean_rank"] = float(
                        np.mean((rc[:, 1] - rank_base) / denom)
                    )

                elif name == "normalized_variance_rank":
                    rank_base = float(_get_rank_base(cfg))
                    denom = 255.0
                    results["normalized_variance_rank"] = float(
                        np.var((rc[:, 1] - rank_base) / denom, ddof=0)
                    )

                elif name == "rank_k_success":
                    k = int(opts.get("k", 1))
                    if k <= 0:
                        raise ValueError("rank_k_success requires k>0")
                    if k > 256:
                        raise ValueError(f"rank_k_success requires k<=256, got k={k}")

                    rank_base = _get_rank_base(cfg)
                    final_rank = float(rc[-1, 1])
                    thr = _topk_rank_threshold(k=k, rank_base=rank_base)
                    results[f"rank_k_success@{k}"] = float(1.0 if final_rank <= thr else 0.0)

                elif name == "time_to_recover":
                    k = int(opts.get("k", 1))
                    if k <= 0:
                        raise ValueError("time_to_recover requires k>0")
                    if k > 256:
                        raise ValueError(f"time_to_recover requires k<=256, got k={k}")

                    rank_base = _get_rank_base(cfg)
                    thr = float(_topk_rank_threshold(k=k, rank_base=rank_base))
                    # Stable crossing is the headline; first crossing kept too.
                    value = _stable_threshold_trace(rc, threshold=thr, direction="le")
                    first_value = _first_threshold_trace(rc, threshold=thr, direction="le")

                    suffix = f"@{k}" if "k" in opts else ""
                    results[f"time_to_recover{suffix}"] = value
                    results[f"first_time_to_recover{suffix}"] = first_value

            else:
                raise ValueError(f"Unsupported metric {name!r}")

        except Exception as e:
            logger.error("Failed to compute '%s': %s", name, e, exc_info=True)
            if fail_on_metric_error:
                raise

    reporting_cfg = cfg.get("key_recovery_reporting") or {}
    reporting_enabled = (
        isinstance(reporting_cfg, dict)
        and bool(reporting_cfg.get("enabled", False))
    )
    requested_rank_evaluation = any(
        name in _RANK_METRICS or name in _REPEATED_ATTACK_METRICS
        for name, _opts in requested
    )

    if reporting_enabled and metadata is not None and requested_rank_evaluation:
        summary = _ensure_attack_summary_cached()
        if summary is None:
            raise ValueError(
                "key_recovery_reporting is enabled but repeated-attack evidence "
                "could not be computed"
            )
        evidence = summary.get(KEY_RECOVERY_EVIDENCE_KEY)
        if evidence is None:
            raise ValueError(
                "key_recovery_reporting is enabled but perform_attacks returned no "
                "fixed-key recovery evidence"
            )
        if not isinstance(evidence, dict):
            raise TypeError(
                f"{KEY_RECOVERY_EVIDENCE_KEY} must be dict, got {type(evidence)}"
            )
        results[KEY_RECOVERY_EVIDENCE_KEY] = evidence

    return results
