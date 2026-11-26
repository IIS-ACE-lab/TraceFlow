# services/side_channel_core/tests/unit/test_trainable_parameters.py
"""trainable_parameters (P2 plan Phase 1, 2026-10-22): the training loop writes the
model's trainable parameter count to its train rows, counted by one helper
(utils/parameter_count.py) that the panel's model check shares.

Zaid et al.'s ASCAD N0 network (gabzai repository, ASCAD/N0=0/cnn_architecture.py)
has 16,952 trainable parameters; the 16,960 of its Table 4 ("Complexity (trainable
parameters)") and of Wouters et al.'s Table 1 is Keras's count_params(), which adds
the 8 moving statistics of its BatchNormalization.
"""
import torch
from torch import nn

from test_final_cv import _loop_cfg, _run, _stack, harness  # noqa: F401  (the training-loop fixture)


def _zaid_ascad_n0_torch():
    return nn.Sequential(nn.Conv1d(1, 4, 1), nn.BatchNorm1d(4), nn.AvgPool1d(2, 2), nn.Flatten(),
                         nn.Linear(1400, 10), nn.SELU(), nn.Linear(10, 10), nn.SELU(), nn.Linear(10, 256))


def test_counts_the_trainable_parameters_of_a_pytorch_model():
    from utils.parameter_count import trainable_parameter_count

    model = _zaid_ascad_n0_torch()
    assert trainable_parameter_count(model) == 16952          # BatchNorm's running statistics are buffers
    for p in model[0].parameters():
        p.requires_grad_(False)
    assert trainable_parameter_count(model) == 16952 - 8      # a frozen layer is not trainable


def test_an_uninitialised_lazy_layer_is_not_counted():
    from utils.parameter_count import trainable_parameter_count

    model = nn.Sequential(nn.LazyLinear(4), nn.Linear(4, 2))
    assert trainable_parameter_count(model) == 10             # LazyLinear has no shape before a forward pass
    model(torch.zeros(1, 3))
    assert trainable_parameter_count(model) == 26


def test_a_keras_model_counts_its_trainable_weights_not_count_params():
    from core.framework_switch import KerasModel, keras_module
    from utils.parameter_count import trainable_parameter_count

    keras = keras_module()
    L = keras.layers
    net = keras.Sequential([keras.Input((700, 1)), L.Conv1D(4, 1, padding="same", activation="selu"),
                            L.BatchNormalization(), L.AveragePooling1D(2, strides=2), L.Flatten(),
                            L.Dense(10, activation="selu"), L.Dense(10, activation="selu"),
                            L.Dense(256, activation="softmax")])
    assert net.count_params() == 16960                        # the published "trainable parameters"
    assert trainable_parameter_count(KerasModel(net)) == 16952
    assert trainable_parameter_count(KerasModel(net)) == trainable_parameter_count(_zaid_ascad_n0_torch())


def test_the_metric_is_a_train_metric_the_loop_provides():
    from sca import metrics as M

    assert M.VALID_METRIC_PROCESSES["trainable_parameters"] == ["train"]
    assert "trainable_parameters" in M.LOOP_PROVIDED_METRICS
    assert M.canonical_fit_time_name("trainable_parameters") is None
    assert M.canonical_fit_time_name("fit_time_{s}") == "fit_time_s"
    assert M.requested_model_size_metrics({"metrics_to_log": ["accuracy", "trainable_parameters"]}) == [
        "trainable_parameters"]
    assert M.requested_model_size_metrics({"metrics_to_log": ["accuracy"]}) == []
    assert M.model_size_values([], _zaid_ascad_n0_torch()) == {}
    assert M.model_size_values(["trainable_parameters"], _zaid_ascad_n0_torch()) == {"trainable_parameters": 16952}


def test_the_judge_reads_it_as_a_cost_lower_is_better():
    from services.model_judge.model_judge import _TRAIN_COST
    from utils.monitor_metric import metric_direction

    assert _TRAIN_COST.fullmatch("trainable_parameters")
    assert metric_direction("trainable_parameters") == "min"


def test_every_train_row_carries_it_and_no_validation_row(harness, tmp_path):
    eu, rec = harness
    cfg = _loop_cfg(["accuracy", "trainable_parameters"])
    model, optimizer, scheduler = _stack(cfg)
    _run(eu, cfg, model, optimizer, scheduler, tmp_path)
    expected = sum(p.numel() for p in model.parameters())
    train = [row for process, _epoch, row in rec.rows if process == "train"]
    validation = [row for process, _epoch, row in rec.rows if process == "validation"]
    assert train and all(row["trainable_parameters"] == expected for row in train)
    assert validation and all("trainable_parameters" not in row for row in validation)


def test_without_the_metric_no_row_carries_it(harness, tmp_path):
    eu, rec = harness
    cfg = _loop_cfg(["accuracy"])
    model, optimizer, scheduler = _stack(cfg)
    _run(eu, cfg, model, optimizer, scheduler, tmp_path)
    assert rec.rows and all("trainable_parameters" not in row for _process, _epoch, row in rec.rows)
