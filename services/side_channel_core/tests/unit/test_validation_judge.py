# services/side_channel_core/tests/unit/test_validation_judge.py
"""utils/validation_judge.py: the per-epoch verdict on train and validation accuracy against chance, chance
taken from each split's own labels and predictions (any database type, leakage model or split size)."""

import numpy as np
import pytest
import torch

from services.side_channel_core.utils.config_errors import ConfigError
import test_one_cycle_loop
from utils.validation_judge import Reading, ValidationJudge, guessing_entropy, judge_log_enabled

loop = test_one_cycle_loop.loop        # the training-loop harness fixture (artifact writers stubbed)

HW = np.array([bin(v).count("1") for v in range(256)])


def _split(labels, correct, classes, seed=0):
    """(one-hot logits, labels): the first `correct` traces predicted right, the others as another class."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    predicted = labels.copy()
    wrong = (labels[correct:] + rng.integers(1, classes, labels.size - correct)) % classes
    predicted[correct:] = wrong
    return np.eye(classes)[predicted], labels


def _uniform(n, correct, seed=0):
    return _split(np.random.default_rng(seed + 1).integers(0, 256, n), correct, 256, seed)


def test_judge_log_is_off_unless_set_true_and_refuses_anything_but_a_bool():
    assert judge_log_enabled({}) is False
    assert judge_log_enabled({"evaluation": {"validation": {"judge_log": None}}}) is False
    assert judge_log_enabled({"evaluation": {"validation": {"judge_log": True}}}) is True
    with pytest.raises(ConfigError, match="judge_log must be true or false"):
        judge_log_enabled({"evaluation": {"validation": {"judge_log": "yes"}}})


def test_train_climbing_above_chance_with_validation_at_chance_is_memorising():
    judge = ValidationJudge()
    judge.line(epoch=1, epochs=75, train=_uniform(45_000, 450), validation=_uniform(5_000, 20))
    line = judge.line(epoch=2, epochs=75, train=_uniform(45_000, 7_847), validation=_uniform(5_000, 17))
    assert line.startswith("Validation judge, epoch 2/75: memorising its training traces: train accuracy 0.1744 above "
                           "chance 0.00")
    assert ", climbing; validation 0.0034 at chance 0.00" in line and "(17 of 5,000, p = " in line


def test_validation_above_chance_is_generalising_and_both_at_chance_is_not_learning():
    judge = ValidationJudge()
    assert ": not learning yet: train accuracy 0.0040 at chance" in judge.line(
        epoch=1, epochs=10, train=_uniform(45_000, 180), validation=_uniform(5_000, 20))
    line = judge.line(epoch=2, epochs=10, train=_uniform(45_000, 900), validation=_uniform(5_000, 100))
    assert ": generalising: validation 0.0200 above chance" in line and "(100 of 5,000, p < 0.001)" in line


def test_chance_follows_the_labels_hw_class_4_always_predicted_is_not_learning():
    """HW labels: class 4 holds 70 of the 256 values, so predicting it for every trace is right 27 % of the
    time. Chance from the labels and predictions says so; a uniform 1/9 would call it learning."""
    labels = HW[np.random.default_rng(3).integers(0, 256, 5_000)]
    always_4 = np.eye(9)[np.full(labels.size, 4)]
    reading = Reading(always_4, labels)
    assert reading.accuracy == pytest.approx(70 / 256, abs=0.02)
    assert reading.chance == reading.accuracy and not reading.above
    assert ": not learning yet: " in ValidationJudge().line(epoch=1, epochs=1, train=(always_4, labels),
                                                           validation=(always_4, labels))


def test_a_fixed_key_over_few_traces_and_a_small_validation_set_are_judged_on_their_own_counts():
    """Few traces under one key hold a few label values unevenly; 50 validation traces (ZvW DPAv4's 10 % of 500)
    are tested exactly: one hit is chance, five are not."""
    labels = np.random.default_rng(4).choice([3, 3, 3, 7, 7, 11, 200], 50)          # an uneven label set
    majority = np.eye(256)[np.full(50, 3)]                                          # the most common label always
    assert not Reading(majority, labels).above                                      # 3/7 of 50 right, at chance
    assert not Reading(*_uniform(50, 1)).above
    five = Reading(*_uniform(50, 5))
    assert five.above and five.evidence() == "(5 of 50, p < 0.001)"


def test_a_rising_validation_objective_under_a_falling_training_objective_is_overfitting():
    judge = ValidationJudge()
    objectives = [(5.5, 5.50), (5.4, 5.52), (5.3, 5.55)]
    for epoch, (train, validation) in enumerate(objectives, 1):
        assert "overfitting" not in judge.line(epoch=epoch, epochs=9, train=_uniform(4_500, 900),
                                               validation=_uniform(500, 2), train_objective=train,
                                               validation_objective=validation)
    assert judge.line(epoch=4, epochs=9, train=_uniform(4_500, 950), validation=_uniform(500, 2), train_objective=5.2,
                      validation_objective=5.60).endswith(
        "; validation objective rising 3 epochs while the training objective falls: overfitting")


def test_an_attacked_epoch_adds_its_guessing_entropy_because_the_rank_decides():
    curve = np.array([[1.0, 128.0], [5_000.0, 6.0]])
    assert guessing_entropy({"mean_attack_rank_curve": curve}) == 6.0
    assert guessing_entropy({"final_ranks": [1, 3, 5]}) == 3.0
    assert guessing_entropy({"guessing_entropy": 2.5, "final_ranks": [9]}) == 2.5
    assert guessing_entropy({"mean_accuracy": 0.004}) is None
    line = ValidationJudge().line(epoch=75, epochs=75, train=_uniform(45_000, 38_559), validation=_uniform(5_000, 20),
                                  validation_results={"mean_attack_rank_curve": curve})
    assert line.endswith("; validation guessing entropy 6 after all traces (random 128.5)")


def test_torch_tensors_are_read_as_numpy_arrays():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    reading = Reading(logits, torch.tensor([1, 1, 1]))
    assert (reading.correct, reading.n) == (2, 3)
    assert reading.chance == pytest.approx(2 / 3)                 # predicted class 1 twice, labelled 1 always


def test_the_training_loop_logs_one_judge_line_per_validated_epoch_only_when_asked(loop, tmp_path, caplog):
    """A two-epoch hold-out loop on CPU (test_one_cycle_loop's harness): a line after each epoch's validation
    with judge_log true, none without it."""
    import copy
    import logging

    from test_final_cv import _loop_cfg
    from test_one_cycle_loop import _train

    cfg = _loop_cfg(["accuracy"], epochs=2)
    with caplog.at_level(logging.INFO):
        _train(loop, copy.deepcopy(cfg), 24, tmp_path)
    assert not [r for r in caplog.records if "Validation judge" in r.getMessage()]
    caplog.clear()
    cfg["evaluation"] = {"validation": {"judge_log": True}}
    with caplog.at_level(logging.INFO):
        _train(loop, cfg, 24, tmp_path)
    lines = [r.getMessage() for r in caplog.records if "Validation judge" in r.getMessage()]
    assert [line.split(":")[0] for line in lines] == ["Validation judge, epoch 1/2", "Validation judge, epoch 2/2"]
    assert all(any(v in line for v in ("generalising", "memorising", "not learning yet")) for line in lines)
