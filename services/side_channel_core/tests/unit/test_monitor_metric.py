# services/side_channel_core/tests/unit/test_monitor_metric.py
"""trainer.monitor_metric: what a run can monitor, and the refusal of a monitor it does not record."""

import pytest

from services.side_channel_core.utils.config_errors import ConfigError
from utils.monitor_metric import check_monitor_metric, monitorable_metrics

CFG = {"metrics_to_log": ["mean_attack_rank_curve", "mean_accuracy", "mean_fit_time_{s}", "guessing_entropy"]}


def test_a_run_monitors_the_metrics_a_validation_pass_writes_as_one_number():
    # the rank curve is an array; mean_accuracy is also written as accuracy; validation_objective always
    assert monitorable_metrics(CFG) == ["mean_accuracy", "accuracy", "mean_fit_time_s", "guessing_entropy", "validation_objective"]
    assert monitorable_metrics({"metrics_to_log": ["ge", "train_loss_reduction"]}) == ["guessing_entropy", "validation_objective"]


def test_a_monitor_the_run_does_not_record_is_refused_only_when_something_reads_it():
    cfg = {"metrics_to_log": ["mean_accuracy"]}
    check_monitor_metric(cfg, "guessing_entropy", early_stopping=False, checkpointing=False)      # inert: not checked
    check_monitor_metric(cfg, "accuracy", early_stopping=True, checkpointing=False)
    with pytest.raises(ConfigError, match="'guessing_entropy' is not recorded by this run, and early_stopping"):
        check_monitor_metric(cfg, "guessing_entropy", early_stopping=True, checkpointing=False)
    with pytest.raises(ConfigError, match="validation checkpointing reads it"):
        check_monitor_metric(cfg, "key_rank", early_stopping=False, checkpointing=True)


def test_keras_has_only_the_cross_entropy_loss():
    from core.framework_switch import KERAS_LOSSES, build_loss

    assert KERAS_LOSSES == ("CrossEntropyLoss",)
    with pytest.raises(ValueError, match="has no Keras counterpart; under framework: keras use CrossEntropyLoss"):
        build_loss({"deep_learning": {"framework": "keras"}, "trainer": {"loss": {"type": "NLLLoss", "args": {}}}})


@pytest.mark.parametrize("metric", ["accuracy", "mean_accuracy", "top_5_accuracy", "success_rate", "lm"])
def test_higher_is_better_metrics_are_maximised_by_the_monitor(metric):
    from utils.monitor_metric import get_monitor_mode, is_better_score

    assert get_monitor_mode(metric) == "maximize"
    assert is_better_score(0.010, 0.004, metric)          # mean_accuracy rising is an improvement


@pytest.mark.parametrize("metric", ["guessing_entropy", "validation_objective", "cross_entropy", "mean_rank"])
def test_ranks_and_losses_are_minimised_by_the_monitor(metric):
    from utils.monitor_metric import get_monitor_mode

    assert get_monitor_mode(metric) == "minimize"


def test_the_monitor_and_model_judge_share_one_direction_rule():
    from services.model_judge.model_judge import metric_direction as judge
    from services.side_channel_core.utils.monitor_metric import metric_direction

    assert judge is metric_direction
