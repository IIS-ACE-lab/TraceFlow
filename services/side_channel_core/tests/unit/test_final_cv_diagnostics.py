# services/side_channel_core/tests/unit/test_final_cv_diagnostics.py
"""Codebase finalisation 2026-09-29: CV diagnostics (lane C2).

STAT-11 one diagnostics profile per sca.leakage_model (label-prior chance
references; one recovered-fold fraction, per-profile rank-spread
thresholds); STAT-12 training trend judged
against the best validation epoch; fold_seed_rule follows FOLD_SEED_RULE.
"""

import json
import logging
import math
import statistics

import pytest

import utils.cross_validation_diagnostics as diagnostics


def _cfg(leakage_model="ID", **cv):
    return {
        "sca": {"leakage_model": leakage_model, "rank_base": 1},
        "perform_attack": {"ge_thresholds": 1},
        "cross_validation": {"report_aggregate": True, "shuffle": True, **cv},
    }


def _fold(number, endpoint, *, validation=(5.55, 5.40, 5.30), training=None):
    training = training or [5.55 - 0.1 * epoch for epoch in range(len(validation))]
    return {
        "fold_number": number,
        "seed": 42 + number,
        "final_validation": {"metrics": {
            "mean_attack_rank_curve": [[10, 200.0], [5000, float(endpoint)]],
        }},
        "trajectory": {
            "epochs": list(range(1, len(validation) + 1)),
            "training_objective": list(training),
            "validation_objective": list(validation),
            "training_accuracy": [0.0] * len(validation),
            "validation_accuracy": [0.0] * len(validation),
        },
    }


def _report(leakage_model, endpoints, **fold_kwargs):
    folds = [_fold(i + 1, rank, **fold_kwargs) for i, rank in enumerate(endpoints)]
    return diagnostics._build_report(cfg=_cfg(leakage_model), fold_results=folds, aggregate={})


# ---------------------------------------------------------------------------
# STAT-11: chance references from the label prior, one profile per model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("leakage_model, classes, accuracy, cross_entropy", [
    ("ID", 256, 1 / 256, math.log(256)),
    ("HW", 9, 70 / 256, 1.7635033290792477),
    ("MSB", 2, 0.5, math.log(2)),
])
def test_profile_chance_references_follow_the_label_prior(
    leakage_model, classes, accuracy, cross_entropy
):
    # A one-element list (the params.yaml sweep form) selects the same profile.
    for value in (leakage_model, [leakage_model.lower()]):
        profile = diagnostics._diagnostics_profile({"sca": {"leakage_model": value}})
        assert profile.name == leakage_model and profile.class_count == classes
        assert profile.chance_accuracy == pytest.approx(accuracy, abs=1e-12)
        assert profile.chance_cross_entropy == pytest.approx(cross_entropy, abs=1e-12)
        assert profile.overfit_tolerance == pytest.approx(0.005 * cross_entropy)

    experiment = _report(leakage_model, [1] * 10)["experiment"]
    assert experiment["diagnostics_profile"] == leakage_model
    assert experiment["chance_accuracy"] == pytest.approx(accuracy, abs=1e-12)
    assert experiment["chance_cross_entropy"] == pytest.approx(cross_entropy, abs=1e-12)
    # The key rank is over 256 keys whatever the labelling.
    assert experiment["chance_expected_rank"] == 128.5


@pytest.mark.parametrize("value", [None, "LSB", ["ID", "HW"]])
def test_profile_rejects_an_unresolved_leakage_model_with_a_hint(value):
    with pytest.raises(ValueError, match="set sca.leakage_model to one of ID, HW or MSB"):
        diagnostics._diagnostics_profile({"sca": {"leakage_model": value}})


def test_stable_needs_recovered_folds_and_a_small_absolute_spread():
    all_recovered = _report("ID", [1] * 10)
    rank = all_recovered["analyses"]["rank_learning_and_stability"]
    assert rank["stability"] == "stable" and rank["folds_reaching_best_rank"] == 10
    assert all_recovered["verdict"]["code"] == "reliable_learning"

    # Audit case: std 11.9 ranks passed the old std/255 <= 0.05 rule, so
    # 5/10 recovering folds were called reliable_learning. Half the folds
    # missing rank 1 is not stable under any profile.
    endpoints = [1, 1, 1, 1, 1, 20, 22, 24, 25, 26]
    assert statistics.stdev(endpoints) / 255 <= 0.05
    for leakage_model, stable_std, variable_std in (
        ("ID", 2.0, 16.0), ("HW", 6.0, 24.0), ("MSB", 16.0, 32.0),
    ):
        report = _report(leakage_model, endpoints)
        rank = report["analyses"]["rank_learning_and_stability"]
        assert rank["stability"] == "variable"
        assert rank["reaching_best_rank_fraction"] == 0.5
        assert rank["stability_rule"] == {
            "profile": leakage_model, "recovered_fraction": 0.9,
            "stable_rank_std": stable_std, "variable_rank_std": variable_std,
        }
        assert report["verdict"]["code"] == "learning_but_fold_sensitive"
        assert "5/10 folds reach GE <= 1" in report["verdict"]["summary"]
        assert f"(stable needs >= 90% and <= {stable_std:g})" in report["verdict"]["summary"]
        assert "normalized_endpoint_dispersion" not in rank

    # Verifier cases: 5/10 or 8/10 folds at rank 1 with a spread inside the
    # MSB/HW allowance are still not stable.
    for leakage_model, endpoints in (
        ("MSB", [1] * 5 + [30] * 5), ("HW", [1] * 8 + [12, 12]), ("HW", [1] * 8 + [3, 3]),
    ):
        report = _report(leakage_model, endpoints)
        assert report["analyses"]["rank_learning_and_stability"]["stability"] == "variable"
        assert report["verdict"]["code"] == "learning_but_fold_sensitive"

    # Same 9/10 recovered folds; only the spread allowance scales per profile.
    straggler = {
        # tenth fold's endpoint: (ID, HW, MSB) stability
        7: ("stable", "stable", "stable"),        # std 1.90
        15: ("variable", "stable", "stable"),     # std 4.43
        40: ("variable", "variable", "stable"),   # std 12.33
    }
    for last, expected in straggler.items():
        for leakage_model, stability in zip(("ID", "HW", "MSB"), expected):
            report = _report(leakage_model, [1] * 9 + [last])
            assert report["analyses"]["rank_learning_and_stability"]["stability"] == stability
            assert report["verdict"]["code"] == (
                "reliable_learning" if stability == "stable" else "learning_but_fold_sensitive"
            )

    # Folds that disagree by tens of ranks are highly variable in every profile.
    spread = [1, 1, 1, 1, 1, 1, 1, 1, 100, 120]
    for leakage_model in ("ID", "HW", "MSB"):
        rank = _report(leakage_model, spread)["analyses"]["rank_learning_and_stability"]
        assert rank["stability"] == "highly_variable"


# ---------------------------------------------------------------------------
# STAT-12: training trend relative to the best validation epoch
# ---------------------------------------------------------------------------
def test_u_shaped_validation_loss_is_possible_overfitting():
    # Ends below epoch 1, so the old first->final rule said objectives_improved.
    u_shape = (5.55, 5.30, 5.20, 5.25, 5.35, 5.45)
    report = _report("ID", [1] * 10, validation=u_shape)
    trend = report["analyses"]["trajectory_and_generalization"]
    assert trend["status"] == "possible_overfitting"
    assert trend["mean_first_to_final_change"]["validation_objective"]["mean"] < 0
    assert trend["epoch_of_min"]["median"] == 3
    minimum = trend["validation_objective_minimum"]
    assert minimum["final_minus_min"]["mean"] == pytest.approx(0.25)
    assert minimum["first_minus_min"]["mean"] == pytest.approx(0.35)
    assert minimum["rise_tolerance"] == pytest.approx(0.005 * math.log(256))
    assert minimum["folds_rising_after_min"] == 10
    # Least-squares slope over epochs 2..6.
    assert minimum["tail_slope_per_epoch"]["mean"] == pytest.approx(0.045)
    assert minimum["folds"][0]["epoch_of_min"] == 3
    assert any(f["code"] == "possible_overfitting" for f in report["findings"])


def test_rise_within_the_profile_tolerance_is_not_overfitting():
    # Final is 0.01 nats above the minimum: below the ID tolerance (0.0277)
    # but above the MSB tolerance (0.0035).
    validation = (0.70, 0.60, 0.50, 0.51)
    trend = _report("ID", [1] * 10, validation=validation)["analyses"]["trajectory_and_generalization"]
    assert trend["status"] == "objectives_improved"
    assert trend["validation_objective_minimum"]["folds_rising_after_min"] == 0
    trend = _report("MSB", [1] * 10, validation=validation)["analyses"]["trajectory_and_generalization"]
    assert trend["status"] == "possible_overfitting"


def test_monotone_trend_labels_and_earliest_minimum():
    decreasing = _report("ID", [1] * 10, validation=(5.5, 5.4, 5.3))
    trend = decreasing["analyses"]["trajectory_and_generalization"]
    assert trend["status"] == "objectives_improved" and trend["epoch_of_min"]["median"] == 3
    # Rising from epoch 1: the minimum is epoch 1.
    rising = _report("ID", [1] * 10, validation=(5.5, 5.6, 5.7))
    trend = rising["analyses"]["trajectory_and_generalization"]
    assert trend["status"] == "possible_overfitting" and trend["epoch_of_min"]["median"] == 1
    # A tie keeps the earliest epoch.
    tied = _report("ID", [1] * 10, validation=(5.5, 5.3, 5.4, 5.3))
    assert tied["analyses"]["trajectory_and_generalization"]["epoch_of_min"]["median"] == 2


def test_overfitting_decides_the_verdict_unless_the_rank_evidence_is_reliable():
    u_shape = (5.55, 5.20, 5.45)
    # 8/10 folds better than chance (< 90%).
    report = _report("ID", [1] * 8 + [200, 220], validation=u_shape)
    assert report["verdict"]["code"] == "possible_overfitting"
    assert "rose after its minimum" in report["verdict"]["summary"]
    # jewel_bangles (2026-10-20): every fold better than chance, none at the best rank,
    # validation loss rising in every fold. It read "Learning is evident".
    report = _report("ID", [40, 55, 60, 70, 75, 80, 90, 95, 100, 110], validation=u_shape)
    assert report["verdict"]["code"] == "possible_overfitting"
    # Reliable rank evidence still wins, and says the loss rose.
    report = _report("ID", [1] * 10, validation=u_shape)
    assert report["verdict"]["code"] == "reliable_learning"
    assert "validation loss rose after its minimum" in report["verdict"]["summary"]


# ---------------------------------------------------------------------------
# Report file, printed block and the fields other code reads
# ---------------------------------------------------------------------------
def test_report_exposes_profile_chance_and_the_fold_seed_rule(tmp_path, caplog):
    logger = logging.getLogger("test_final_cv_diagnostics")
    folds = [_fold(i + 1, 1, validation=(1.76, 1.60, 1.55, 1.58)) for i in range(10)]
    path = tmp_path / "aggregate" / "cross_validation_diagnostics.json"
    with caplog.at_level(logging.INFO, logger=logger.name):
        report = diagnostics.create_cross_validation_diagnostic_report(
            cfg=_cfg("HW"), fold_results=folds, aggregate={}, report_path=path, logger=logger,
        )
    assert report is not None
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["schema_version"] == 2
    experiment = written["experiment"]
    assert experiment["fold_seed_rule"] == "refit uses seed; folds use the next n_folds seeds in fold order"
    assert experiment["diagnostics_profile"] == "HW" and experiment["class_count"] == 9

    # Fields model_judge reads from the diagnostics report (run_reader._cv_metrics).
    rank = written["analyses"]["rank_learning_and_stability"]
    for key in ("endpoint_rank", "folds_better_than_chance", "normalized_rank_advantage",
                "stability", "rank_curve_efficiency", "outlier_folds"):
        assert key in rank
    assert written["analyses"]["trajectory_and_generalization"]["status"] == "possible_overfitting"
    assert written["verdict"]["code"] == "reliable_learning"

    text = caplog.text
    assert "profile=HW" in text
    assert "accuracy 0.2734 | cross-entropy 1.7635 nats" in text
    assert "10/10 folds at GE <= 1" in text
    assert "Val loss minimum" in text and "epoch median 3" in text


def test_unknown_leakage_model_leaves_the_experiment_running(tmp_path, caplog):
    logger = logging.getLogger("test_final_cv_diagnostics")
    path = tmp_path / "diagnostics.json"
    with caplog.at_level(logging.WARNING, logger=logger.name):
        report = diagnostics.create_cross_validation_diagnostic_report(
            cfg=_cfg(None), fold_results=[_fold(1, 1)], aggregate={},
            report_path=path, logger=logger,
        )
    assert report is None and not path.exists()
    assert "set sca.leakage_model to one of ID, HW or MSB" in caplog.text
