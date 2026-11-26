#services/side_channel_core/utils/cross_validation_diagnostics.py

"""
Non-authoritative diagnostics for completed SCA cross-validation folds.

The report is created only when ``cross_validation.report_aggregate`` is the
literal boolean ``True``. Diagnostics are observational: failures are logged
and suppressed, and this module never updates lifecycle state, checkpoints,
registries, manifests, refit decisions, or official test results.

Chance references and fold-stability thresholds come from one profile per
``sca.leakage_model`` (ID, HW, MSB); the training trend is judged against the
best epoch of the validation loss.

Author: Shraddha Mishra
"""

from __future__ import annotations

import json
import math
import os
import statistics
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.sca.leakage_models import (
    apply_leakage_model,
    get_num_classes,
    normalize_leakage_model,
)
from utils.cross_validation import FOLD_SEED_RULE
from utils.timestamps import now_iso
from utils.seal import seal_text


REPORT_SCHEMA_VERSION = 2
_CONSISTENT_FOLD_FRACTION = 0.90
_MEANINGFUL_RANK_ADVANTAGE = 0.10
_ORDER_CORRELATION_WARNING = 0.70

# One diagnostics profile per sca.leakage_model. The label-space chance
# references come from the label prior of a uniform AES intermediate byte:
# the accuracy of always guessing the majority class, and the prior entropy
# (cross-entropy of always predicting the prior), in nats.
#   ID : 1/256 = 0.0039          ln 256 = 5.5452
#   HW : 70/256 = 0.2734 (HW 4)  H(Binomial(8, 1/2)) = 1.7635
#   MSB: 1/2                     ln 2 = 0.6931
# The key rank is over 256 key hypotheses whatever the labelling, so the rank
# chance (rank_base + 127.5) is the same for every profile.
#
# Fold stability, on the endpoint rank of each held-out fold:
#   stable          : at least _STABLE_RECOVERED_FRACTION of the folds end at
#                     GE <= best rank (rank 1 under rank_base 1) and the
#                     endpoint-rank sample std is at most stable_rank_std ranks
#   variable        : endpoint-rank sample std at most variable_rank_std ranks
#   highly_variable : otherwise
# Stability is agreement between folds, so the recovered-fold fraction is the
# same for every profile (90%, as _CONSISTENT_FOLD_FRACTION): half the folds
# missing rank 1 is never stable. The leakage model changes how many traces an
# attack needs: an ID label carries at most 8 bits of the intermediate per
# trace, a HW label 2.54 bits (1.7635 nats) and an MSB label 1 bit, so a fold
# still converging at the end of the fold attack budget ends further from
# rank 1 under HW or MSB. Only the absolute spread scales with that ratio, from
# ID's 2 ranks (HW 6 ~ 2 x 8/2.54, MSB 16 = 2 x 8/1); with 9/10 folds at rank 1
# the tenth may then end at about rank 7 (ID), 20 (HW) or 52 (MSB). Every
# variable ceiling stays well below the ~74-rank std of chance endpoints
# (uniform over 256 keys).
_STABLE_RECOVERED_FRACTION = _CONSISTENT_FOLD_FRACTION
_STABILITY_THRESHOLDS = {
    # leakage model: (stable_rank_std, variable_rank_std)
    "ID": (2.0, 16.0),
    "HW": (6.0, 24.0),
    "MSB": (16.0, 32.0),
}

# Training trend, relative to the best epoch of the validation objective: a
# fold's validation loss rises after its minimum when final - min exceeds
# 0.5% of the profile's chance cross-entropy (ID 0.0277, HW 0.0088, MSB
# 0.0035 nats), an allowance so that epoch-to-epoch jitter of the held-out
# loss is not reported as overfitting. The tail slope is a least-squares fit
# over the last _TREND_TAIL_EPOCHS epochs, in nats per epoch.
_OVERFIT_TOLERANCE_FRACTION = 0.005
_TREND_TAIL_EPOCHS = 5


@dataclass(frozen=True)
class DiagnosticsProfile:
    """Chance references and thresholds of one sca.leakage_model."""

    name: str
    class_count: int
    chance_accuracy: float
    chance_cross_entropy: float
    recovered_fraction: float
    stable_rank_std: float
    variable_rank_std: float
    overfit_tolerance: float


def _now_iso() -> str:
    return now_iso()


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "median": float(statistics.median(values)),
        "q1": _percentile(values, 0.25),
        "q3": _percentile(values, 0.75),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _pearson(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 3:
        return None
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values)
    )
    x_mass = sum((x - x_mean) ** 2 for x in x_values)
    y_mass = sum((y - y_mean) ** 2 for y in y_values)
    denominator = math.sqrt(x_mass * y_mass)
    return float(numerator / denominator) if denominator else None


def _diagnostics_profile(cfg: Mapping[str, Any]) -> DiagnosticsProfile:
    """Profile of the run's sca.leakage_model; chance references from its label prior."""
    value = (cfg.get("sca") or {}).get("leakage_model")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 1:
        value = value[0]
    try:
        name = normalize_leakage_model(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CV diagnostics need one resolved sca.leakage_model, got {value!r} ({exc}). "
            "Hint: set sca.leakage_model to one of ID, HW or MSB in params.yaml."
        ) from exc
    class_count = get_num_classes(name)
    labels = apply_leakage_model(np.arange(256, dtype=np.uint8), name)
    prior = np.bincount(labels, minlength=class_count) / 256.0
    chance_cross_entropy = float(-sum(p * math.log(p) for p in prior if p > 0.0))
    stable_rank_std, variable_rank_std = _STABILITY_THRESHOLDS[name]
    return DiagnosticsProfile(
        name=name,
        class_count=class_count,
        chance_accuracy=float(prior.max()),
        chance_cross_entropy=chance_cross_entropy,
        recovered_fraction=_STABLE_RECOVERED_FRACTION,
        stable_rank_std=stable_rank_std,
        variable_rank_std=variable_rank_std,
        overfit_tolerance=_OVERFIT_TOLERANCE_FRACTION * chance_cross_entropy,
    )


def _rank_contract(cfg: Mapping[str, Any]) -> tuple[int, float, int]:
    rank_base = (cfg.get("sca") or {}).get("rank_base", 1)
    if isinstance(rank_base, bool) or rank_base not in (0, 1):
        rank_base = 1
    return rank_base, rank_base + 127.5, rank_base + 255


def _configured_thresholds(cfg: Mapping[str, Any], best_rank: int) -> list[float]:
    raw = (cfg.get("perform_attack") or {}).get("ge_thresholds", best_rank)
    values = raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else [raw]
    thresholds = sorted({value for item in values if (value := _finite_float(item)) is not None})
    return thresholds or [float(best_rank)]


def _final_metrics(result: Mapping[str, Any]) -> Mapping[str, Any]:
    snapshot = result.get("final_validation")
    if isinstance(snapshot, Mapping) and isinstance(snapshot.get("metrics"), Mapping):
        return snapshot["metrics"]
    metrics = result.get("validation")
    return metrics if isinstance(metrics, Mapping) else {}


def _rank_curve(result: Mapping[str, Any]) -> list[tuple[float, float]] | None:
    raw = _final_metrics(result).get("mean_attack_rank_curve")
    if raw is None:
        return None
    try:
        curve = []
        for point in raw:
            if len(point) != 2:
                return None
            trace_count = _finite_float(point[0])
            rank = _finite_float(point[1])
            if trace_count is None or rank is None:
                return None
            curve.append((trace_count, rank))
    except (TypeError, ValueError):
        return None
    if not curve or any(
        current[0] <= previous[0]
        for previous, current in zip(curve, curve[1:])
    ):
        return None
    return curve


def _normalized_rank_auc(
    curve: Sequence[tuple[float, float]],
    *,
    best_rank: int,
    chance_rank: float,
) -> float:
    if len(curve) == 1:
        mean_rank = curve[0][1]
    else:
        area = sum(
            (right_x - left_x) * (left_y + right_y) / 2.0
            for (left_x, left_y), (right_x, right_y) in zip(curve, curve[1:])
        )
        mean_rank = area / (curve[-1][0] - curve[0][0])
    return float((mean_rank - best_rank) / (chance_rank - best_rank))


def _first_trace_at_or_below(
    curve: Sequence[tuple[float, float]],
    threshold: float,
) -> int | None:
    for trace_count, rank in curve:
        if rank <= threshold:
            return int(trace_count)
    return None


def _stable_trace_at_or_below(
    curve: Sequence[tuple[float, float]],
    threshold: float,
) -> int | None:
    """Earliest trace count from which every later rank stays <= threshold."""
    stable = None
    for trace_count, rank in reversed(list(curve)):
        if rank <= threshold:
            stable = int(trace_count)
        else:
            break
    return stable


def _index_range(result: Mapping[str, Any]) -> tuple[int, int] | None:
    snapshot = result.get("final_validation")
    indices = snapshot.get("validation_indices") if isinstance(snapshot, Mapping) else None
    if indices is None:
        return None
    try:
        values = [int(value) for value in indices]
    except (TypeError, ValueError):
        return None
    return (min(values), max(values)) if values else None


def _fold_evidence(
    fold_results: Sequence[Mapping[str, Any]],
    *,
    cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    best_rank, chance_rank, worst_rank = _rank_contract(cfg)
    thresholds = _configured_thresholds(cfg, best_rank)
    evidence = []
    limitations = []

    for position, result in enumerate(fold_results):
        fold_number = int(result.get("fold_number", position + 1))
        curve = _rank_curve(result)
        if curve is None:
            limitations.append(f"Fold {fold_number} has no usable final rank curve")
            continue
        index_range = _index_range(result)
        evidence.append({
            "fold_number": fold_number,
            "seed": result.get("seed"),
            "endpoint_rank": float(curve[-1][1]),
            "trace_count": int(curve[-1][0]),
            "chance_normalized_rank_auc": _normalized_rank_auc(
                curve,
                best_rank=best_rank,
                chance_rank=chance_rank,
            ),
            # Headline: stable crossing (same rule as traces_to_ge_k and the
            # key-recovery "recovered" flag); first crossing kept alongside.
            "traces_to_threshold": {
                str(threshold): _stable_trace_at_or_below(curve, threshold)
                for threshold in thresholds
            },
            "first_traces_to_threshold": {
                str(threshold): _first_trace_at_or_below(curve, threshold)
                for threshold in thresholds
            },
            "validation_index_min": index_range[0] if index_range else None,
            "validation_index_max": index_range[1] if index_range else None,
            "validation_index_midpoint": (
                statistics.fmean(index_range) if index_range else None
            ),
            "rank_bounds": [best_rank, worst_rank],
        })
    return evidence, limitations


def _outlier_folds(evidence: Sequence[Mapping[str, Any]]) -> list[int]:
    values = [float(item["endpoint_rank"]) for item in evidence]
    if len(values) < 3:
        return []
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    if mad == 0:
        return [
            int(item["fold_number"])
            for item in evidence
            if float(item["endpoint_rank"]) != median
        ]
    return [
        int(item["fold_number"])
        for item in evidence
        if abs(0.6745 * (float(item["endpoint_rank"]) - median) / mad) > 3.5
    ]


def _epoch_points(raw: Any) -> list[tuple[int, float]]:
    """Finite (epoch, value) pairs of one per-epoch series; epochs start at 1."""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [
        (epoch, value)
        for epoch, item in enumerate(raw, start=1)
        if (value := _finite_float(item)) is not None
    ]


def _validation_minimum(points: Sequence[tuple[int, float]]) -> dict[str, Any]:
    """Validation loss relative to its best epoch (earliest epoch wins a tie)."""
    epoch_of_min, minimum = min(points, key=lambda point: (point[1], point[0]))
    tail = points[-_TREND_TAIL_EPOCHS:]
    return {
        "epoch_of_min": epoch_of_min,
        "min": minimum,
        "first_minus_min": points[0][1] - minimum,
        "final_minus_min": points[-1][1] - minimum,
        "tail_slope": statistics.linear_regression(
            [epoch for epoch, _ in tail],
            [value for _, value in tail],
        ).slope,
    }


def _trajectory_analysis(
    fold_results: Sequence[Mapping[str, Any]],
    *,
    profile: DiagnosticsProfile,
) -> dict[str, Any]:
    keys = (
        "training_objective",
        "validation_objective",
        "training_accuracy",
        "validation_accuracy",
    )
    changes: dict[str, list[float]] = {key: [] for key in keys}
    minima: list[dict[str, Any]] = []
    available_folds = 0

    for position, result in enumerate(fold_results):
        trajectory = result.get("trajectory")
        if not isinstance(trajectory, Mapping):
            continue
        available_folds += 1
        for key in keys:
            points = _epoch_points(trajectory.get(key))
            if len(points) >= 2:
                changes[key].append(points[-1][1] - points[0][1])
        points = _epoch_points(trajectory.get("validation_objective"))
        if len(points) >= 2:
            minima.append({
                "fold_number": int(result.get("fold_number", position + 1)),
                **_validation_minimum(points),
            })

    summaries = {
        key: _summary(values)
        for key, values in changes.items()
        if values
    }
    tolerance = profile.overfit_tolerance
    validation_minimum = None
    if minima:
        validation_minimum = {
            "first_minus_min": _summary([item["first_minus_min"] for item in minima]),
            "final_minus_min": _summary([item["final_minus_min"] for item in minima]),
            "tail_slope_per_epoch": _summary([item["tail_slope"] for item in minima]),
            "tail_epochs": _TREND_TAIL_EPOCHS,
            "rise_tolerance": tolerance,
            "folds_rising_after_min": sum(
                item["final_minus_min"] > tolerance for item in minima
            ),
            "folds": minima,
        }
    train_delta = summaries.get("training_objective", {}).get("mean")
    validation_delta = summaries.get("validation_objective", {}).get("mean")
    if train_delta is None or validation_minimum is None:
        status = "insufficient_evidence"
    elif train_delta < 0.0 and validation_minimum["final_minus_min"]["mean"] > tolerance:
        # Judged against the best validation epoch, so a U-shaped validation
        # loss that still ends below epoch 1 is not called an improvement.
        status = "possible_overfitting"
    elif train_delta < 0.0 and validation_delta < 0.0:
        status = "objectives_improved"
    elif train_delta >= 0.0 and validation_delta >= 0.0:
        status = "no_objective_improvement"
    else:
        status = "mixed"
    return {
        "status": status,
        "folds_with_trajectory": available_folds,
        "epoch_of_min": (
            _summary([item["epoch_of_min"] for item in minima]) if minima else None
        ),
        "validation_objective_minimum": validation_minimum,
        "mean_first_to_final_change": summaries,
    }


def _rank_analysis(
    evidence: Sequence[Mapping[str, Any]],
    *,
    cfg: Mapping[str, Any],
    profile: DiagnosticsProfile,
) -> dict[str, Any]:
    best_rank, chance_rank, _ = _rank_contract(cfg)
    endpoints = [float(item["endpoint_rank"]) for item in evidence]
    if not endpoints:
        return {"status": "insufficient_evidence"}

    endpoint_stats = _summary(endpoints)
    better_count = sum(value < chance_rank for value in endpoints)
    normalized_advantage = (
        (chance_rank - float(endpoint_stats["mean"]))
        / (chance_rank - best_rank)
    )
    reached_count = sum(value <= best_rank for value in endpoints)
    reached_fraction = reached_count / len(endpoints)
    rank_std = float(endpoint_stats["sample_std"])
    if reached_fraction >= profile.recovered_fraction and rank_std <= profile.stable_rank_std:
        stability = "stable"
    elif rank_std <= profile.variable_rank_std:
        stability = "variable"
    else:
        stability = "highly_variable"

    auc_values = [float(item["chance_normalized_rank_auc"]) for item in evidence]
    mean_auc = statistics.fmean(auc_values)
    efficiency = "high" if mean_auc <= 0.25 else "moderate" if mean_auc <= 0.60 else "low"

    thresholds = _configured_thresholds(cfg, best_rank)
    threshold_results = {}
    for threshold in thresholds:
        key = str(threshold)
        reached = [
            int(item["traces_to_threshold"][key])
            for item in evidence
            if item["traces_to_threshold"][key] is not None
        ]
        threshold_results[key] = {
            "folds_reached": len(reached),
            "folds_total": len(evidence),
            "traces": _summary(reached) if reached else None,
        }

    return {
        "status": "completed",
        "rank_base": best_rank,
        "chance_expected_rank": chance_rank,
        "endpoint_rank": endpoint_stats,
        "folds_better_than_chance": better_count,
        "folds_total": len(evidence),
        "better_than_chance_fraction": better_count / len(evidence),
        "normalized_rank_advantage": float(normalized_advantage),
        "folds_reaching_best_rank": reached_count,
        "reaching_best_rank_fraction": reached_fraction,
        "stability": stability,
        "stability_rule": {
            "profile": profile.name,
            "recovered_fraction": profile.recovered_fraction,
            "stable_rank_std": profile.stable_rank_std,
            "variable_rank_std": profile.variable_rank_std,
        },
        "rank_curve_efficiency": efficiency,
        "chance_normalized_rank_auc": _summary(auc_values),
        "thresholds": threshold_results,
        "outlier_folds": _outlier_folds(evidence),
    }


def _order_analysis(
    evidence: Sequence[Mapping[str, Any]],
    *,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    cv_cfg = cfg.get("cross_validation") or {}
    if cv_cfg.get("shuffle") is not False:
        return {"status": "not_applicable", "reason": "cross_validation.shuffle is not false"}
    positioned = [
        item for item in evidence
        if item.get("validation_index_midpoint") is not None
    ]
    correlation = _pearson(
        [float(item["validation_index_midpoint"]) for item in positioned],
        [float(item["endpoint_rank"]) for item in positioned],
    )
    if correlation is None:
        return {"status": "insufficient_evidence"}
    return {
        "status": (
            "possible_order_effect"
            if abs(correlation) >= _ORDER_CORRELATION_WARNING
            else "no_strong_linear_order_effect"
        ),
        "endpoint_rank_correlation_with_validation_position": correlation,
        "interpretation": (
            "Association only; it does not prove acquisition drift or causality."
        ),
    }


def _verdict(
    rank: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> tuple[str, str]:
    if rank.get("status") != "completed":
        return "insufficient_evidence", "Final fold rank curves were unavailable."

    fraction = float(rank["better_than_chance_fraction"])
    advantage = float(rank["normalized_rank_advantage"])
    stability = rank["stability"]
    overfitting = trajectory.get("status") == "possible_overfitting"
    learning = fraction >= _CONSISTENT_FOLD_FRACTION and advantage > _MEANINGFUL_RANK_ADVANTAGE
    # Only rank evidence that is reliable (the best rank in enough folds, a small
    # spread) outweighs a validation loss rising after its minimum: in SCA the
    # loss and the rank can part ways. Below that, beating chance in most folds
    # does not hide overfitting (jewel_bangles, 2026-10-20: 10/10 folds better
    # than chance, 0/10 at the best rank, validation loss rising in every fold).
    if learning and stability == "stable":
        return (
            "reliable_learning",
            "Held-out profiling ranks reach the best rank in enough folds "
            "with a small endpoint spread"
            + ("; validation loss rose after its minimum." if overfitting else "."),
        )
    if overfitting:
        return (
            "possible_overfitting",
            "Training loss improved while validation loss rose after its minimum.",
        )
    if learning:
        rule = rank["stability_rule"]
        return (
            "learning_but_fold_sensitive",
            f"Learning is evident, but {rank['folds_reaching_best_rank']}/"
            f"{rank['folds_total']} folds reach GE <= {rank['rank_base']} and the "
            f"endpoint std is {rank['endpoint_rank']['sample_std']:.2f} ranks "
            f"(stable needs >= {100.0 * rule['recovered_fraction']:.0f}% and "
            f"<= {rule['stable_rank_std']:g}).",
        )
    if advantage <= 0.0:
        return (
            "no_learning_signal",
            "Mean held-out profiling rank does not improve over the chance expectation.",
        )
    if fraction >= _CONSISTENT_FOLD_FRACTION:
        return (
            "consistently_weak",
            "Most folds improve over chance, but the mean rank advantage is small.",
        )
    return (
        "mixed_fold_behavior",
        "Some profiling folds improve over chance while others do not.",
    )


def _findings(
    rank: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    order: Mapping[str, Any],
) -> list[dict[str, str]]:
    findings = []
    if rank.get("status") == "completed":
        findings.append({
            "code": "rank_learning_signal",
            "level": "positive" if rank["normalized_rank_advantage"] > 0 else "warning",
            "message": (
                f"{rank['folds_better_than_chance']}/{rank['folds_total']} folds "
                "finished better than the random-rank expectation."
            ),
        })
        if rank["stability"] != "stable":
            findings.append({
                "code": "fold_sensitivity",
                "level": "warning",
                "message": (
                    "Fold dispersion reflects profiling-subset and initialization "
                    "sensitivity together."
                ),
            })
        if rank["outlier_folds"]:
            findings.append({
                "code": "fold_outliers",
                "level": "warning",
                "message": f"Endpoint-rank outliers: {rank['outlier_folds']}",
            })
    if trajectory.get("status") == "possible_overfitting":
        minimum = trajectory["validation_objective_minimum"]
        findings.append({
            "code": "possible_overfitting",
            "level": "warning",
            "message": (
                "Training loss improved while validation loss rose "
                f"{minimum['final_minus_min']['mean']:.4f} nats after its minimum "
                f"(median epoch {trajectory['epoch_of_min']['median']:g}) in "
                f"{minimum['folds_rising_after_min']}/{len(minimum['folds'])} folds."
            ),
        })
    if order.get("status") == "possible_order_effect":
        findings.append({
            "code": "possible_order_effect",
            "level": "warning",
            "message": "Endpoint rank is strongly associated with profiling position.",
        })
    return findings


def _build_report(
    *,
    cfg: Mapping[str, Any],
    fold_results: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    profile = _diagnostics_profile(cfg)
    evidence, limitations = _fold_evidence(fold_results, cfg=cfg)
    rank = _rank_analysis(evidence, cfg=cfg, profile=profile)
    trajectory = _trajectory_analysis(fold_results, profile=profile)
    order = _order_analysis(evidence, cfg=cfg)
    verdict_code, verdict_summary = _verdict(rank, trajectory)

    if trajectory["status"] == "insufficient_evidence":
        limitations.append(
            "Fold trajectories were unavailable; convergence and generalization-gap "
            "judgements were not made"
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "status": "completed",
        "scope": "profiling_cross_validation",
        "official_test_affected": False,
        "lifecycle_authority": False,
        "verdict": {"code": verdict_code, "summary": verdict_summary},
        "experiment": {
            "n_folds": len(fold_results),
            "leakage_model": profile.name,
            "diagnostics_profile": profile.name,
            "class_count": profile.class_count,
            # Label prior of a uniform AES intermediate byte: majority-class
            # accuracy and prior entropy (nats); the key-rank chance is
            # label-independent.
            "chance_accuracy": profile.chance_accuracy,
            "chance_cross_entropy": profile.chance_cross_entropy,
            "chance_expected_rank": _rank_contract(cfg)[1],
            "fold_seed_rule": FOLD_SEED_RULE,
            "fold_sensitivity_confounding": (
                "Profiling partition and initialization seed vary together."
            ),
        },
        "analyses": {
            "rank_learning_and_stability": rank,
            "trajectory_and_generalization": trajectory,
            "profiling_order_effect": order,
        },
        "fold_evidence": evidence,
        "findings": _findings(rank, trajectory, order),
        "limitations": limitations,
        "aggregate_provenance": {
            "schema_version": aggregate.get("schema_version"),
            "n_folds_expected": aggregate.get("n_folds_expected"),
            "n_folds_completed": aggregate.get("n_folds_completed"),
            "all_folds_successful": aggregate.get("all_folds_successful"),
        },
    }


def _row(label: str, value: Any) -> str:
    return f"{label:<26}: {value}"


def _upper(value: Any) -> str:
    return str(value or "unknown").replace("_", " ").upper()


def format_cross_validation_diagnostic_report(
    report: Mapping[str, Any],
    *,
    report_path: str | Path | None = None,
) -> str:
    """Render the diagnostic as one framed block (same layout as the data contract)."""
    verdict = report.get("verdict") or {}
    experiment = report.get("experiment") or {}
    analyses = report.get("analyses") or {}
    rank = analyses.get("rank_learning_and_stability") or {}
    trajectory = analyses.get("trajectory_and_generalization") or {}
    order = analyses.get("profiling_order_effect") or {}

    lines = [
        "=" * 80,
        "CROSS-VALIDATION DIAGNOSTIC — PROFILING DATA ONLY",
        "=" * 80,
        _row("Verdict", _upper(verdict.get("code"))),
        _row("Interpretation", verdict.get("summary", "Unavailable")),
        _row(
            "Folds",
            f"{experiment.get('n_folds', 'n/a')} | leakage={experiment.get('leakage_model')} "
            f"| classes={experiment.get('class_count')} "
            f"| profile={experiment.get('diagnostics_profile')}",
        ),
        _row(
            "Chance (label prior)",
            f"accuracy {experiment.get('chance_accuracy', float('nan')):.4f} "
            f"| cross-entropy {experiment.get('chance_cross_entropy', float('nan')):.4f} nats",
        ),
        "-" * 80,
    ]

    if rank.get("status") == "completed":
        endpoint = rank["endpoint_rank"]
        auc = rank["chance_normalized_rank_auc"]
        rule = rank["stability_rule"]
        lines += [
            _row(
                "Final rank (endpoint GE)",
                f"mean {endpoint['mean']:.2f} ± {endpoint['sample_std']:.2f} "
                f"| median {endpoint['median']:.1f} "
                f"| range {endpoint['min']:.0f}–{endpoint['max']:.0f}",
            ),
            _row(
                "Chance reference",
                f"expected rank {rank['chance_expected_rank']:.1f} "
                f"(rank base {rank['rank_base']})",
            ),
            _row(
                "Better than chance",
                f"{rank['folds_better_than_chance']}/{rank['folds_total']} folds "
                f"({100.0 * rank['better_than_chance_fraction']:.0f}%) "
                f"| advantage {100.0 * rank['normalized_rank_advantage']:+.1f}%",
            ),
            _row(
                "Fold stability",
                f"{_upper(rank['stability'])} "
                f"({rank['folds_reaching_best_rank']}/{rank['folds_total']} folds at "
                f"GE <= {rank['rank_base']}, endpoint std {endpoint['sample_std']:.2f} ranks; "
                f"{rule['profile']} stable: >= {100.0 * rule['recovered_fraction']:.0f}% "
                f"and <= {rule['stable_rank_std']:g})",
            ),
            _row(
                "Curve efficiency",
                f"{_upper(rank['rank_curve_efficiency'])} "
                f"(normalized rank AUC {auc['mean']:.3f} ± {auc['sample_std']:.3f})",
            ),
        ]
        for threshold, item in (rank.get("thresholds") or {}).items():
            traces = item.get("traces")
            reached = (
                f"median {traces['median']:.0f} traces" if traces else "not reached"
            )
            lines.append(
                _row(
                    f"GE <= {float(threshold):g} reached",
                    f"{item['folds_reached']}/{item['folds_total']} folds | {reached}",
                )
            )
        outliers = rank.get("outlier_folds") or []
        lines.append(_row("Outlier folds", ", ".join(map(str, outliers)) or "none"))
    else:
        lines.append(_row("Rank analysis", _upper(rank.get("status"))))

    lines.append("-" * 80)
    changes = trajectory.get("mean_first_to_final_change") or {}
    lines.append(
        _row(
            "Training trend",
            f"{_upper(trajectory.get('status'))} "
            f"({trajectory.get('folds_with_trajectory', 0)} folds with trajectories)",
        )
    )
    minimum = trajectory.get("validation_objective_minimum")
    epoch_of_min = trajectory.get("epoch_of_min")
    if minimum and epoch_of_min:
        rise = minimum["final_minus_min"]
        slope = minimum["tail_slope_per_epoch"]
        lines += [
            _row(
                "Val loss minimum",
                f"epoch median {epoch_of_min['median']:g} "
                f"(range {epoch_of_min['min']:.0f}–{epoch_of_min['max']:.0f})",
            ),
            _row(
                "Val loss final − min",
                f"{rise['mean']:+.6f} ± {rise['sample_std']:.6f} nats "
                f"| {minimum['folds_rising_after_min']}/{len(minimum['folds'])} folds "
                f"> {minimum['rise_tolerance']:.4f}",
            ),
            _row(
                f"Val loss slope (last {minimum['tail_epochs']})",
                f"{slope['mean']:+.6f} ± {slope['sample_std']:.6f} nats/epoch",
            ),
        ]
    for key, label in (
        ("training_objective", "Δ train loss (first→last)"),
        ("validation_objective", "Δ val loss (first→last)"),
        ("training_accuracy", "Δ train accuracy"),
        ("validation_accuracy", "Δ val accuracy"),
    ):
        if key in changes:
            lines.append(
                _row(label, f"{changes[key]['mean']:+.6f} ± {changes[key]['sample_std']:.6f}")
            )
    lines.append(
        _row(
            "Profiling order effect",
            _upper(order.get("status"))
            + (f" ({order['reason']})" if order.get("reason") else ""),
        )
    )

    findings = report.get("findings") or []
    limitations = report.get("limitations") or []
    if findings or limitations:
        lines.append("-" * 80)
        for finding in findings:
            marker = "+" if finding.get("level") == "positive" else "!"
            lines.append(f"  {marker} {finding.get('message')}")
        for limitation in limitations:
            lines.append(f"  - {limitation}")

    lines += [
        "-" * 80,
        _row("Official test affected", "no (diagnostic only)"),
    ]
    if report_path is not None:
        lines.append(_row("Report file", report_path))
    lines.append("=" * 80)
    return "\n".join(lines)


def log_cross_validation_diagnostic_report(
    report: Mapping[str, Any],
    logger,
    *,
    report_path: str | Path | None = None,
) -> None:
    """Log the supplemental judgement once; never change experiment state."""
    logger.info(
        "\n%s",
        format_cross_validation_diagnostic_report(report, report_path=report_path),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Plain exclusive creation so the report is readable per the process umask
    # (NamedTemporaryFile would force mode 0600).
    temporary_name = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary_name.open("x", encoding="utf-8") as handle:
            text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            handle.write(seal_text(path, text))  # utils/seal.py
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except Exception:
                pass
        raise


def create_cross_validation_diagnostic_report(
    *,
    cfg: Mapping[str, Any],
    fold_results: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    report_path: str | Path,
    logger,
) -> dict[str, Any] | None:
    """Create and log the optional report without affecting experiment flow.

    Returns the report on success and ``None`` when disabled or unavailable.
    Every ordinary diagnostic or persistence exception is suppressed so the
    caller can continue to mandatory refit and official testing.
    """
    cv_cfg = cfg.get("cross_validation") if isinstance(cfg, Mapping) else None
    if not isinstance(cv_cfg, Mapping) or cv_cfg.get("report_aggregate") is not True:
        return None

    try:
        report = _build_report(
            cfg=cfg,
            fold_results=fold_results,
            aggregate=aggregate,
        )
        path = Path(report_path)
        _write_json(path, report)
        log_cross_validation_diagnostic_report(report, logger, report_path=path)
        return report
    except Exception as exc:
        logger.warning(
            "CV aggregate diagnostics unavailable; experiment will continue: %s",
            exc,
        )
        return None


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "create_cross_validation_diagnostic_report",
    "format_cross_validation_diagnostic_report",
    "log_cross_validation_diagnostic_report",
]
