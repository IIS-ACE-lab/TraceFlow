# services/side_channel_core/tests/unit/test_per_attack_ntge.py
"""per_attack_traces_to_ge_1: the papers' N̄tGE (Zaid Sect. 5: 100 attacks, each with its own NtGE, averaged),
apart from traces_to_ge_1, the stable crossing of the mean curve; and η_t, the learning rate at iteration or epoch t, the optimizer's rate
written into the train row (the record of a schedule, D1a of projects/ZvW)."""
import numpy as np
import pytest
import torch

from sca import metrics as M


def _summary(curves):
    reps = np.asarray(curves, dtype=np.float64)
    return {
        "per_attack_curves": reps,
        "mean_attack_rank_curve": np.column_stack([reps[0, :, 0], reps[:, :, 1].mean(axis=0)]),
        "final_ranks": reps[:, -1, 1].astype(int).tolist(),
    }


def test_the_papers_ntge_is_the_mean_of_the_per_attack_crossings_not_the_crossing_of_the_mean():
    traces = np.arange(1, 11)
    a = np.column_stack([traces, [9, 5, 1, 1, 1, 1, 1, 1, 1, 1]])    # stays at rank 1 from trace 3
    b = np.column_stack([traces, [9, 9, 9, 9, 9, 9, 1, 1, 1, 1]])    # from trace 7
    c = np.column_stack([traces, [9, 9, 9, 9, 9, 9, 9, 9, 9, 2]])    # never
    summary = _summary([a, b, c])
    per_attack = [M._stable_threshold_trace(x, threshold=1.0, direction="le") for x in (a, b, c)]
    assert per_attack == [3, 7, None]
    mean_curve = summary["mean_attack_rank_curve"]
    assert M._stable_threshold_trace(mean_curve, threshold=1.0, direction="le") is None   # the mean never reaches 1
    # the branch itself, as compute_metrics runs it
    results = {}
    curves = np.asarray(summary["per_attack_curves"], dtype=np.float64)
    crossings = [M._stable_threshold_trace(curve, threshold=1.0, direction="le") for curve in curves]
    reached = [float(x) for x in crossings if x is not None]
    results["per_attack_traces_to_ge_1"] = float(np.mean(reached)) if reached else None
    assert results["per_attack_traces_to_ge_1"] == 5.0 and len(crossings) - len(reached) == 1


def test_the_metric_is_registered_as_a_repeated_attack_metric_on_validation_and_test():
    assert "per_attack_traces_to_ge_1" in M._REPEATED_ATTACK_METRICS
    assert M.VALID_METRIC_PROCESSES["per_attack_traces_to_ge_1"] == ["validation", "test"]
    from utils.monitor_metric import metric_direction
    assert metric_direction("per_attack_traces_to_ge_1") == "min"
    assert metric_direction("per_attack_traces_to_ge_1_std") == "min"


def test_eta_t_is_a_train_row_metric_read_from_the_optimizer():
    cfg = {"metrics_to_log": ["accuracy", "η_t", "trainable_parameters"]}
    assert M.requested_loop_state_metrics(cfg) == ["η_t"]
    assert M.requested_loop_state_metrics({"metrics_to_log": ["accuracy"]}) == []
    assert M.VALID_METRIC_PROCESSES["η_t"] == ["train"]
    assert "η_t" in M.LOOP_PROVIDED_METRICS
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-3, total_steps=10, pct_start=0.4, div_factor=10.0,
                                                final_div_factor=100.0, anneal_strategy="linear", three_phase=True,
                                                cycle_momentum=False)
    used = None
    for _ in range(10):                                   # the loop: optimizer step, read the rate it used, then the scheduler
        opt.step(); used = M.eta_t(opt); sched.step()
    got = M.loop_state_values(["η_t"], learning_rate=used)
    assert set(got) == {"η_t"}
    assert abs(got["η_t"] - 5e-3 / (10.0 * 100.0)) < 1e-12      # a completed cycle ends at max / (div x final_div)
    assert M.eta_t(opt) < 0                   # one step past the cycle: never the value written
    assert M.loop_state_values([], learning_rate=used) == {} and M.loop_state_values(["η_t"], learning_rate=None) == {"η_t": None}


def test_the_loss_the_run_optimises_is_recorded_when_asked_for_with_its_name():
    """training_objective / validation_objective are columns when metrics_to_log names them, and the row
    then carries loss_type (trainer.loss's type and arguments) as their dependency; unrequested, both the
    objective and loss_type stay the loop's bookkeeping and no file stores them."""
    cfg = {"metrics_to_log": ["accuracy", "training_objective"],
           "trainer": {"loss": {"type": "CrossEntropyLoss", "args": {"reduction": "mean", "from_logits": True, "label_smoothing": None}}}}
    assert M.loss_signature(cfg) == "CrossEntropyLoss(reduction=mean, from_logits=True)"
    results = {"accuracy": 0.5, "training_objective": 1.23, "loss_type": M.loss_signature(cfg), "train_loss_reduction": "mean"}
    metrics, deps = M.split_requested(results, cfg)
    assert metrics == {"accuracy": 0.5, "training_objective": 1.23}
    assert deps == {"loss_type": "CrossEntropyLoss(reduction=mean, from_logits=True)"}
    metrics, deps = M.split_requested(results, {"metrics_to_log": ["accuracy"], "trainer": cfg["trainer"]})
    assert metrics == {"accuracy": 0.5} and deps == {}
    assert M.VALID_METRIC_PROCESSES["training_objective"] == ["train"]
    assert M.VALID_METRIC_PROCESSES["validation_objective"] == ["validation"]
    from utils.monitor_metric import metric_direction
    assert metric_direction("training_objective") == "min" and metric_direction("validation_objective") == "min"


def test_the_old_metric_names_are_refused_and_name_eta_t():
    """learning_rate is the trainer's setting (trainer.optimizer.learning_rate); the recorded rate is η_t."""
    for old in ("learning_rate", "used_learning_rate"):
        with pytest.raises(ValueError, match=rf"'{old}' is now 'η_t'"):
            M._parse_metrics_list({"metrics_to_log": ["accuracy", old]})


def test_eta_t_rises_as_a_better_arrow_and_is_never_a_monitor():
    """The log and the panel's live lines draw η_t's rise ▲ (metric_direction 'max'); a train-row metric, it is
    no monitor, so early stopping and the judge never read that direction."""
    from utils.monitor_metric import metric_direction, monitorable_metrics

    assert metric_direction("η_t") == "max"
    assert "η_t" not in monitorable_metrics({"metrics_to_log": ["η_t", "mean_accuracy"]})
