# services/side_channel_core/tests/unit/test_recorded_only_listed.py
"""Owner 2026-10-27: a run records only what metrics_to_log names and its dependencies, in every record: the
history rows (utils/metrics_logger), the training result that fold_result.yaml and the CV aggregate are made of,
and its trajectory (utils/epoch_utils._recorded, _recorded_trajectory)."""

from utils.epoch_utils import _recorded, _recorded_trajectory

CFG = {"metrics_to_log": ["mean_attack_rank_curve", "mean_accuracy", "mean_fit_time_{s}"]}
TRAIN = {"accuracy": 0.03, "accuracy_correct": 1296, "accuracy_total": 45000, "mean_accuracy": 0.03,
         "train_loss_reduction": "mean", "training_objective": 5.17, "loss_type": "CrossEntropyLoss",
         "fit_time_s": 630.1, "mean_fit_time_s": 630.1}
VALIDATION = {"accuracy": 0.004, "mean_accuracy": 0.004, "validation_objective": 5.68, "loss_type": "CrossEntropyLoss",
              "fit_time_s": 76.8, "mean_fit_time_s": 76.8}
TRAJECTORY = {"epochs": [1, 2], "training_objective": [5.5, 5.4], "validation_objective": [5.6, 5.7],
              "training_accuracy": [0.01, 0.02], "validation_accuracy": [0.003, 0.004]}


def test_an_unlisted_loss_is_kept_nowhere():
    train, validation = _recorded(TRAIN, CFG), _recorded(VALIDATION, CFG)
    assert train == {"mean_accuracy": 0.03, "mean_fit_time_s": 630.1,                       # the listed columns
                     "accuracy": 0.03, "accuracy_correct": 1296, "accuracy_total": 45000, "fit_time_s": 630.1}
    assert set(validation) == {"mean_accuracy", "mean_fit_time_s", "accuracy", "fit_time_s"}
    assert set(_recorded_trajectory(TRAJECTORY, train, validation)) == {"epochs", "training_accuracy", "validation_accuracy"}


def test_a_listed_loss_is_recorded_with_its_loss_type():
    cfg = {"metrics_to_log": CFG["metrics_to_log"] + ["training_objective", "validation_objective"]}
    train, validation = _recorded(TRAIN, cfg), _recorded(VALIDATION, cfg)
    assert (train["training_objective"], train["loss_type"]) == (5.17, "CrossEntropyLoss")
    assert (validation["validation_objective"], validation["loss_type"]) == (5.68, "CrossEntropyLoss")
    assert "train_loss_reduction" not in train                                              # the loop's bookkeeping
    assert _recorded_trajectory(TRAJECTORY, train, validation) == TRAJECTORY
    assert set(_recorded_trajectory(TRAJECTORY, train, {})) == {"epochs", "training_objective", "training_accuracy"}  # a refit
