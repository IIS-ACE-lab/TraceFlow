# services/side_channel_core/tests/unit/test_framework_switch.py
"""deep_learning.framework: every deep-learning function comes from the library named.

The decisive check is test_keras_run_equals_keras_own_training_step: a Keras
model driven by TraceFlow's loop (KerasModel, build_loss, build_optimizer,
zero_grad / backward / step, as utils/epoch_utils.py does) ends with the same
weights, bit for bit, as the same Keras model trained by Keras itself
(model.compile + train_on_batch).

Run from the repository root:
    PYTHONPATH="$PWD:$PWD/services/side_channel_core:$PWD/services/side_channel_core/src" \
        CUDA_VISIBLE_DEVICES="" conda run -n sca python -m pytest \
        services/side_channel_core/tests/unit/test_framework_switch.py -q
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.framework_switch import (
    build_loss,
    build_optimizer,
    gradient_clipper,
    keras_module,
    resolve_framework,
)
from core.model import build_model
from utils.model_naming import make_experiment_hash

keras = keras_module()


def _cfg(framework="keras", activation="relu", *, layers=None, optimizer=None, loss=None, **dl):
    layers = layers or [
        {"Flatten": {}},
        {"Linear": {"in_features": 700, "out_features": 50, "activation": activation}},
        {"Linear": {"in_features": 50, "out_features": 50, "activation": activation}},
        {"Linear": {"in_features": 50, "out_features": "auto"}},
    ]
    return {
        "seed": 42,
        "deep_learning": {"framework": framework, **dl},
        "sca": {"leakage_model": "HW"},
        "model": [{"model_id": "m", "architecture": "MLP", "architecture_args": {"layers": layers}}],
        "trainer": {
            "loss": loss or {"type": "CrossEntropyLoss", "args": {"reduction": "mean"}},
            "optimizer": optimizer or {
                "type": "RMSprop", "learning_rate": 1e-3, "weight_decay": 0.0,
                "args": {"rho": 0.9, "epsilon": 1e-7, "momentum": 0.0, "centered": False},
            },
        },
    }


def _build(cfg, seed=0):
    torch.manual_seed(seed)
    return build_model(cfg, model_id="m")


def _batch(n=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 1, 700, generator=g), torch.randint(0, 9, (n,), generator=g)


# ── deep_learning block ─────────────────────────────────────────────────────

def test_no_block_is_pytorch_with_pytorch_defaults_and_no_hash_entry():
    spec = resolve_framework({})
    assert (spec.framework, spec.weight_init.name, spec.bias_init.name, spec.declared) == ("pytorch", "default", "default", False)
    assert spec.hash_payload() is None


def test_keras_block_defaults_are_keras_layer_defaults():
    spec = resolve_framework({"deep_learning": {"framework": "keras", "keras": {"weight_init": None}}})
    assert (spec.weight_init.name, spec.bias_init.name) == ("glorot_uniform", "zeros")
    assert spec.hash_payload() == {
        "framework": "keras",
        "keras": {"weight_init": {"name": "glorot_uniform", "args": {}}, "bias_init": {"name": "zeros", "args": {}}},
    }


@pytest.mark.parametrize("block, message", [
    ({"framework": "jax"}, "must be one of"),
    ({"framework": ["pytorch", "keras"]}, "resolved to one value"),
    ({"framework": "keras", "tf_keras": {}}, "Unknown deep_learning key"),
    ({"framework": "keras", "keras": {"weight_init": "glorot_uniform", "weight_init_args": {"seed": 42}}}, "run's seed"),
    ({"framework": "keras", "keras": {"weight_init": "xavier_uniform"}}, "not a keras.initializers name"),
    ({"framework": "pytorch", "pytorch": {"weight_init": "lecun_normal"}}, "not a torch.nn.init function"),
    ({"framework": "keras", "keras": {"weight_init": "random_normal", "weight_init_args": {"std": 1}}}, "has no argument"),
])
def test_bad_blocks_fail_loudly(block, message):
    with pytest.raises((ValueError, TypeError), match=message):
        resolve_framework({"deep_learning": block})


def test_top_level_framework_key_is_rejected():
    with pytest.raises(ValueError, match="deep_learning.framework"):
        resolve_framework({"framework": "keras"})


def _hash_cfg(**extra):
    return {
        "seed": 42,
        "project": {"name": "P", "root": "P", "scripts": {}},
        "data": {"name": "ascad", "database": "data/ascad.h5"},
        "model": [{"model_id": "m", "architecture": "MLP", "architecture_args": {"layers": []}}],
        "trainer": {"epochs": 1, "batch_size": 1},
        "sca": {"variant": "fixed_key", "attack_byte": 2, "leakage_model": "ID"},
        "metrics_to_log": ["mean_attack_rank_curve"],
        **extra,
    }


def test_hash_joins_the_active_framework_only():
    without = make_experiment_hash(_hash_cfg())
    pytorch = make_experiment_hash(_hash_cfg(deep_learning={"framework": "pytorch"}))
    keras_ = make_experiment_hash(_hash_cfg(deep_learning={"framework": "keras"}))
    assert len({without, pytorch, keras_}) == 3
    # editing the inactive block leaves the identity alone
    edited = _hash_cfg(deep_learning={"framework": "pytorch", "keras": {"weight_init": "he_normal"}})
    assert make_experiment_hash(edited) == pytorch
    # the active block's initialiser does change it
    xavier = _hash_cfg(deep_learning={"framework": "pytorch", "pytorch": {"weight_init": "xavier_uniform"}})
    assert make_experiment_hash(xavier) != pytorch


# ── Keras model ─────────────────────────────────────────────────────────────

def test_keras_model_uses_keras_layers_activations_and_initialisers():
    model = _build(_cfg(activation="hard_sigmoid"))
    assert model.framework == "keras" and isinstance(model, nn.Module)
    kinds = [type(layer).__name__ for layer in model.keras_model.layers]
    assert kinds == ["Flatten", "Dense", "Activation", "Dense", "Activation", "Dense"]
    act = model.keras_model.layers[2]
    assert act.activation is keras.activations.hard_sigmoid          # the library's own function
    dense = model.keras_model.layers[1]
    assert type(dense.kernel_initializer).__name__ == "GlorotUniform"
    assert type(dense.bias_initializer).__name__ == "Zeros"
    assert torch.count_nonzero(dense.bias.value) == 0
    # the model's parameters are the Keras variables
    assert {id(p) for p in model.parameters()} == {id(v.value) for v in model.keras_model.trainable_variables}


def test_keras_forward_is_the_keras_function_composition():
    model = _build(_cfg(activation="hard_sigmoid"))
    x, _ = _batch(4)
    model.eval()
    with torch.no_grad():
        out = model(x)
        layers = model.keras_model.layers
        h = x.reshape(4, -1)
        for dense in (layers[1], layers[3]):
            h = keras.activations.hard_sigmoid(keras.ops.matmul(h, dense.kernel.value) + dense.bias.value)
        ref = keras.ops.matmul(h, layers[5].kernel.value) + layers[5].bias.value
    assert torch.equal(out, ref)
    # Keras 3 hard_sigmoid = relu6(x + 3) / 6
    assert torch.allclose(keras.activations.hard_sigmoid(torch.tensor([-1.0, 0.0, 1.0])), torch.tensor([1 / 3, 0.5, 2 / 3]))


def test_keras_weights_follow_the_run_seed():
    a, b, c = _build(_cfg(), 3), _build(_cfg(), 3), _build(_cfg(), 4)
    wa, wb, wc = (m.keras_model.layers[3].kernel.value for m in (a, b, c))
    assert torch.equal(wa, wb) and not torch.equal(wa, wc)
    # two layers of one shape draw different weights (each initialiser has its own seed)
    assert not torch.equal(wa, a.keras_model.layers[3].kernel.value * 0 + a.keras_model.layers[3].kernel.value.roll(1, 0))


def test_keras_conv_padding_pooling_and_widths():
    layers = [
        {"Conv1d": {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"}},
        {"Pooling": {"dim": 1, "type": "avg", "kernel_size": 2, "stride": 2}},
        {"Conv1d": {"in_channels": 4, "out_channels": 4, "kernel_size": 3, "padding": 0}},
        {"Pooling": {"dim": 1, "type": "max", "kernel_size": 2, "stride": 2}},
        {"Flatten": {}},
        {"Linear": {"in_features": 4 * 174, "out_features": "auto"}},
    ]
    model = _build(_cfg(layers=layers))
    convs = [layer for layer in model.keras_model.layers if type(layer).__name__ == "Conv1D"]
    assert [c.padding for c in convs] == ["same", "valid"] and all(c.data_format == "channels_first" for c in convs)
    x, _ = _batch(2)
    assert model(x).shape == (2, 9)

    wrong = copy.deepcopy(layers)
    wrong[5]["Linear"]["in_features"] = 700
    with pytest.raises(ValueError, match="configured input width 700, but the previous layer gives 696"):
        _build(_cfg(layers=wrong))


@pytest.mark.parametrize("layers, message", [
    ([{"Flatten": {}}, {"Linear": {"in_features": 700, "out_features": 9, "activation": "rrelu"}}], "Keras has no activation"),
    ([{"Pooling": {"dim": 1, "type": "lp", "kernel_size": 2}}, {"Flatten": {}}], "has no Keras layer"),
    ([{"Conv1d": {"in_channels": 1, "out_channels": 2, "kernel_size": 3, "padding": 2}}], "no Keras Conv1D equivalent"),
    ([{"LSTM": {"input_size": 1, "hidden_size": 2}}], "has no Keras mapping"),
])
def test_keras_rejects_what_keras_does_not_define(layers, message):
    with pytest.raises(ValueError, match=message):
        _build(_cfg(layers=layers))


def test_keras_rejects_the_named_pytorch_classes():
    cfg = _cfg()
    cfg["model"] = [{"model_id": "m", "architecture": "ResNet50_1D", "architecture_args": {"in_channels": 1, "num_classes": 9}}]
    with pytest.raises(ValueError, match="PyTorch class .* has no Keras build"):
        build_model(cfg, model_id="m")


def test_keras_graph_with_add_concatenate_padding_and_activations():
    nodes = [
        {"name": "pad", "type": "ZeroPad1d", "inputs": ["input"], "params": {"padding": [1, 2]}},
        {"name": "c1", "type": "Conv1d", "inputs": ["pad"], "params": {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "padding": 1}},
        {"name": "c2", "type": "Conv1d", "inputs": ["pad"], "params": {"in_channels": 1, "out_channels": 4, "kernel_size": 1, "bias": False}},
        {"name": "bn", "type": "BatchNorm1d", "inputs": ["c2"], "params": {"num_features": 4, "scale": False, "activation": "relu"}},
        {"name": "sum", "type": "Add", "inputs": ["c1", "bn"], "params": {"activation": "relu"}},
        {"name": "cat", "type": "Concatenate", "inputs": ["sum", "c1"]},
        {"name": "avg", "type": "Pooling", "inputs": ["cat"], "params": {"dim": 1, "type": "avg", "kernel_size": 3, "stride": 1, "padding": 1, "count_include_pad": False}},
        {"name": "gap", "type": "Pooling", "inputs": ["avg"], "params": {"dim": 1, "type": "global_avg"}},
        {"name": "flat", "type": "Flatten", "inputs": ["gap"]},
        {"name": "head", "type": "Linear", "inputs": ["flat"], "params": {"in_features": 8, "out_features": "auto"}},
    ]
    built = {}
    for framework in ("pytorch", "keras"):
        cfg = _cfg(framework)
        cfg["model"] = [{"model_id": "m", "architecture": "CNN", "architecture_args": {"nodes": nodes, "outputs": ["head"]}}]
        built[framework] = _build(cfg, 3).eval()
    torch_graph, keras_graph = built["pytorch"], built["keras"]
    assert keras_graph.framework == "keras"
    trainable = [sum(p.numel() for p in m.parameters() if p.requires_grad) for m in (torch_graph, keras_graph)]
    assert trainable[0] == trainable[1]
    for name in ("c1", "c2"):
        src = torch_graph.layers[name]
        keras_graph.keras_model.get_layer(name).kernel.assign(src.weight.detach().permute(2, 1, 0).contiguous())
        if src.bias is not None:
            keras_graph.keras_model.get_layer(name).bias.assign(src.bias.detach())
    for p_ in torch_graph.layers["bn"].module.parameters():
        assert p_.requires_grad == (p_ is torch_graph.layers["bn"].module.bias)
    torch_graph.layers["bn"].module.eps = 1e-3
    head = torch_graph.layers["head"]
    keras_graph.keras_model.get_layer("head").kernel.assign(head.weight.detach().T.contiguous())
    keras_graph.keras_model.get_layer("head").bias.assign(head.bias.detach())
    x, _ = _batch(2)
    with torch.no_grad():
        a, b = torch_graph(x), keras_graph(x)
    assert a.shape == (2, 9) and torch.allclose(a, b, rtol=1e-5, atol=1e-5)


def test_keras_stride_pooling_is_the_first_value_of_each_window():
    layers = [{"Pooling": {"dim": 1, "type": "stride", "kernel_size": 2, "stride": 2}}, {"Flatten": {}}]
    x = torch.randn(3, 1, 175)
    cfg = _cfg(layers=layers)
    cfg["data"] = {"target_range": [0, 175]}
    model = _build(cfg)
    model.eval()
    with torch.no_grad():
        out = model(x)
    reference = x[..., :174:2].flatten(1)          # the first value of each whole window
    assert out.shape == (3, 87) and torch.equal(out, reference)


def test_keras_dropout_follows_train_and_eval_mode():
    layers = [{"Flatten": {}}, {"Dropout": {"p": 0.5}}, {"Linear": {"in_features": 700, "out_features": 9}}]
    model = _build(_cfg(layers=layers))
    x, _ = _batch(8)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), model(x))
    model.train()
    with torch.no_grad():
        assert not torch.equal(model(x), model(x))


def test_keras_state_dict_round_trip():
    a, b = _build(_cfg(), 5), _build(_cfg(), 6)
    b.load_state_dict(a.state_dict())
    x, _ = _batch(4)
    a.eval(), b.eval()
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


def test_legacy_weight_init_without_the_block_maps_to_the_same_torch_function():
    # Run snapshots written before deep_learning existed name Keras modes per model.
    layers = [{"Flatten": {}}, {"Linear": {"in_features": 700, "out_features": 200, "activation": "relu"}},
              {"Linear": {"in_features": 200, "out_features": "auto"}}]
    cfg = _cfg(layers=layers)
    del cfg["deep_learning"]
    cfg["model"][0]["architecture_args"]["weight_init"] = "glorot_uniform"
    model = _build(cfg)
    limit = (6.0 / 900) ** 0.5
    assert float(model[1].weight.abs().max()) <= limit + 1e-7 and torch.count_nonzero(model[1].bias) == 0
    cfg["model"][0]["architecture_args"]["weight_init"] = "kaiming_uniform"     # "PyTorch's default"
    assert torch.count_nonzero(_build(cfg)[1].bias) > 0
    cfg["model"][0]["architecture_args"]["weight_init"] = "lecun_normal"        # no torch.nn.init equivalent
    with pytest.raises(ValueError, match="needs a deep_learning block"):
        _build(cfg)


# ── loss, optimizer, clipping ───────────────────────────────────────────────

def test_keras_loss_is_keras_categorical_crossentropy():
    cfg = _cfg(loss={"type": "CrossEntropyLoss", "args": {"reduction": "mean", "label_smoothing": 0.1}})
    criterion = build_loss(cfg)
    logits, targets = torch.randn(16, 9), torch.randint(0, 9, (16,))
    ref = keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)(F.one_hot(targets, 9).float(), logits)
    assert torch.equal(criterion(logits, targets), ref)
    with pytest.raises(ValueError, match="no Keras counterpart"):
        build_loss(_cfg(loss={"type": "NLLLoss", "args": {"reduction": "mean"}}))
    assert isinstance(build_loss(_cfg("pytorch")), nn.CrossEntropyLoss)


def test_keras_optimizer_takes_keras_arguments_and_schedules():
    model = _build(_cfg())
    with pytest.raises(ValueError, match="rho is PyTorch's alpha"):
        build_optimizer(_cfg(optimizer={"type": "RMSprop", "learning_rate": 1e-3, "weight_decay": 0.0,
                                        "args": {"alpha": 0.9}}), model)
    with pytest.raises(ValueError, match="not a keras.optimizers.schedules class"):
        build_optimizer(_cfg(optimizer={"type": "RMSprop", "learning_rate": 1e-3, "weight_decay": 0.0, "args": {},
                                        "scheduler": {"type": "StepLR", "args": {"step_size": 1}}}), model)
    optimizer, scheduler, meta = build_optimizer(_cfg(optimizer={
        "type": "Adam", "learning_rate": 1e-3, "weight_decay": 0.0, "args": {"amsgrad": True},
        "scheduler": {"type": "ExponentialDecay", "args": {"decay_steps": 10, "decay_rate": 0.5}}}), model)
    assert scheduler is None and meta["scheduler"]["interval"] == "step"
    assert isinstance(optimizer.keras_optimizer, keras.optimizers.Adam) and optimizer.keras_optimizer.amsgrad
    assert isinstance(optimizer.keras_optimizer._learning_rate, keras.optimizers.schedules.ExponentialDecay)


def test_gradient_clipping_is_the_framework_own():
    cfg = _cfg("pytorch")
    cfg["trainer"]["grad_clip_norm"] = 1.0
    assert callable(gradient_clipper(cfg))
    cfg = _cfg()
    cfg["trainer"]["grad_clip_norm"] = 1.0
    assert gradient_clipper(cfg) is None                          # the Keras optimizer clips itself
    _, _, meta = build_optimizer(cfg, _build(cfg))
    assert meta["global_clipnorm"] == 1.0


@pytest.mark.parametrize("optimizer", [
    {"type": "RMSprop", "learning_rate": 1e-3, "weight_decay": 0.0,
     "args": {"rho": 0.9, "epsilon": 1e-7, "momentum": 0.0, "centered": False}},
    {"type": "Adam", "learning_rate": 1e-3, "weight_decay": 0.01, "args": {"amsgrad": True},
     "scheduler": {"type": "CosineDecay", "args": {"decay_steps": 4, "alpha": 0.1}}},
    {"type": "SGD", "learning_rate": 1e-2, "weight_decay": 0.0, "args": {"momentum": 0.9, "nesterov": True},
     "scheduler": {"type": "ExponentialDecay", "args": {"decay_steps": 2, "decay_rate": 0.5}}},
])
def test_keras_run_equals_keras_own_training_step(optimizer):
    cfg = _cfg(optimizer=optimizer, loss={"type": "CrossEntropyLoss", "args": {"reduction": "mean", "label_smoothing": 0.05}})
    cfg["trainer"]["grad_clip_norm"] = 0.5
    ours, theirs = _build(cfg, 11), _build(cfg, 11)

    # TraceFlow's loop, as utils/epoch_utils.py runs it.
    opt, _, meta = build_optimizer(cfg, ours)
    criterion = build_loss(cfg)
    ours.train()
    for step in range(5):
        x, y = _batch(seed=step)
        opt.zero_grad(set_to_none=True)
        criterion(ours(x), y).backward()
        opt.step()

    # Keras' own training step on the same Keras model and the same settings.
    k = theirs.keras_model
    k.compile(
        optimizer=type(opt.keras_optimizer).from_config(opt.keras_optimizer.get_config()),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05),
    )
    for step in range(5):
        x, y = _batch(seed=step)
        k.train_on_batch(x, F.one_hot(y, 9).float())

    for a, b in zip(ours.keras_model.trainable_variables, k.trainable_variables):
        assert torch.equal(a.value, b.value), a.path
    assert int(opt.keras_optimizer.iterations.numpy()) == 5


# ── the cross_entropy metric ────────────────────────────────────────────────

def test_cross_entropy_metric_is_the_framework_function():
    from sca.metrics import compute_metrics

    logits = torch.randn(6, 256)
    labels = torch.randint(0, 256, (6,))
    base = {
        "metrics_to_log": ["cross_entropy"],
        "metrics_cfg": {"preds_are_probs": False, "ce_eps": 1e-12},
        "trainer": {"loss": {"type": "CrossEntropyLoss", "args": {"reduction": "mean"}},
                    "evaluation": {"output_transformation": {"type": "Softmax"}}},
        "sca": {"leakage_model": "ID"},
    }
    k = compute_metrics(logits, labels, {**base, "deep_learning": {"framework": "keras"}}, process="train")
    p = compute_metrics(logits, labels, {**base, "deep_learning": {"framework": "pytorch"}}, process="train")
    ref_k = keras.losses.categorical_crossentropy(F.one_hot(labels, 256).float(), logits, from_logits=True).mean()
    assert k["cross_entropy"] == pytest.approx(float(ref_k), rel=1e-6)
    assert p["cross_entropy"] == pytest.approx(float(F.cross_entropy(logits, labels)), rel=1e-6)



def test_every_layer_and_pooling_type_keras_layers_lists_builds_under_keras():
    from core.framework_switch import KERAS_LAYERS, KERAS_POOLING

    one = {"Conv1d": {"in_channels": 1, "out_channels": 2, "kernel_size": 3, "padding": 1},
           "Pooling": {"dim": 1, "type": "max", "kernel_size": 2}, "BatchNorm1d": {"num_features": 1},
           "Dropout": {"p": 0.1}, "ZeroPad1d": {"padding": 1}, "Flatten": {},
           "Linear": {"in_features": 700, "out_features": 9}}
    assert set(one) == set(KERAS_LAYERS)                     # the panel's list is what the builder maps
    for name in KERAS_LAYERS:
        layers = [{"Flatten": {}}, {name: one[name]}] if name == "Linear" else [{name: one[name]}, {"Flatten": {}}]
        assert _build(_cfg(layers=layers)).framework == "keras", name
    for kind in KERAS_POOLING:
        _build(_cfg(layers=[{"Pooling": {"dim": 1, "type": kind, "kernel_size": 2}}, {"Flatten": {}}]))


def test_keras_takes_the_schedules_keras_schedules_lists_and_no_torch_scheduler():
    from core.framework_switch import _keras_schedule, keras_schedules

    assert {"ExponentialDecay", "CosineDecay"} <= set(keras_schedules()) and "LearningRateSchedule" not in keras_schedules()
    assert _keras_schedule({"type": "ExponentialDecay", "args": {"decay_steps": 10, "decay_rate": 0.9}}, 1e-3) is not None
    with pytest.raises(ValueError, match="is not a keras.optimizers.schedules class"):
        _keras_schedule({"type": "StepLR", "args": {}}, 1e-3)


def test_a_refit_built_after_ten_folds_loads_into_a_freshly_built_test_model():
    """Keras numbers the Dropout SeedGenerator by a process-wide counter: the CV refit (the 11th
    model of its process) saved seed_generator_10/... and the test process expected seed_generator/..."""
    from core.framework_switch import InitSpec, build_keras_model

    layers = [{"Flatten": {}}, {"Linear": {"in_features": 20, "out_features": 32, "activation": "relu"}},
              {"Dropout": {"p": 0.35}}, {"Linear": {"in_features": 32, "out_features": 256}}]

    def build():
        return build_keras_model(layers, input_shape=(1, 20), weight=InitSpec("glorot_uniform"),
                                 bias=InitSpec("zeros"), resolve_out_features=lambda v, i: v, where="t")

    for _fold in range(10):
        build()
    refit = build().state_dict()
    test_model = build()
    assert list(test_model.state_dict()) == list(refit)
    test_model.load_state_dict(refit, strict=True)
