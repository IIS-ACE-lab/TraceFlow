# services/side_channel_core/tests/unit/test_model.py
"""
Unit tests for the dynamic model builder and its PyTorch initialisation
(deep_learning.pytorch, torch.nn.init).

Run from the repository root:
    PYTHONPATH="$PWD:$PWD/services/side_channel_core:$PWD/services/side_channel_core/src" \
        conda run -n sca python -m pytest services/side_channel_core/tests/unit/test_model.py -q
"""
import math

import pytest
import torch
import torch.nn as nn

from core.model import build_model
from core.framework_switch import InitSpec, apply_torch_init, has_lazy_parameters

XAVIER = {"weight_init": "xavier_uniform", "weight_init_args": {"gain": 1.0}, "bias_init": "zeros"}


def _cnn_cfg(pytorch_block=None, *, lazy=False, leakage="ID", target_range=(0, 700)):
    first_dense = (
        {"LazyLinear": {"out_features": 32, "activation": "relu"}}
        if lazy
        else {"Linear": {"in_features": 8 * 350, "out_features": 32, "activation": "relu"}}
    )
    cfg = {
        "sca": {"leakage_model": leakage},
        "data": {"target_range": list(target_range)},
        "model": [
            {
                "model_id": "cnn",
                "architecture": "CNN",
                "architecture_args": {
                    "layers": [
                        {"Conv1d": {"in_channels": 1, "out_channels": 8, "kernel_size": 3,
                                    "stride": 1, "padding": 1, "activation": "relu"}},
                        {"Pooling": {"dim": 1, "type": "avg", "kernel_size": 2, "stride": 2}},
                        {"Flatten": {}},
                        first_dense,
                        {"Linear": {"in_features": 32, "out_features": "auto"}},
                    ],
                },
            }
        ],
    }
    if pytorch_block is not None:
        cfg["deep_learning"] = {"framework": "pytorch", "pytorch": pytorch_block}
    return cfg


def _mlp_cfg(pytorch_block=None, leakage="ID"):
    cfg = {
        "sca": {"leakage_model": leakage},
        "model": [
            {
                "model_id": "mlp",
                "architecture": "MLP",
                "architecture_args": {
                    "layers": [
                        {"Flatten": {}},
                        {"Linear": {"in_features": 700, "out_features": 200, "activation": "relu"}},
                        {"Linear": {"in_features": 200, "out_features": "auto"}},
                    ],
                },
            }
        ],
    }
    if pytorch_block is not None:
        cfg["deep_learning"] = {"framework": "pytorch", "pytorch": pytorch_block}
    return cfg


def _kernel_layers(model):
    return [m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]


def _xavier_limit(weight):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
    return math.sqrt(6.0 / (fan_in + fan_out))


# ---------------------------------------------------------------- building --

def test_mlp_builds_expected_sequence_and_head():
    model = build_model(_mlp_cfg(XAVIER), model_id="mlp")
    names = [type(m).__name__ for m in model]
    assert names == ["Flatten", "Linear", "ReLU", "Linear"]
    assert model[-1].out_features == 256
    assert model(torch.zeros(3, 1, 700)).shape == (3, 256)


def test_cnn_builds_and_head_follows_leakage_model():
    model = build_model(_cnn_cfg(XAVIER, leakage="HW"), model_id="cnn")
    names = [type(m).__name__ for m in model]
    assert names == ["Conv1d", "ReLU", "Pooling", "Flatten", "Linear", "ReLU", "Linear"]
    assert model[-1].out_features == 9
    assert model(torch.zeros(2, 1, 700)).shape == (2, 9)


def test_activation_key_never_reaches_the_layer_constructor():
    # nn.Conv1d would raise on an unknown 'activation' kwarg; it must be popped.
    model = build_model(_cnn_cfg(), model_id="cnn")
    assert isinstance(model[0], nn.Conv1d) and isinstance(model[1], nn.ReLU)


def test_hard_sigmoid_is_the_torch_function_under_pytorch():
    cfg = _mlp_cfg()
    cfg["model"][0]["architecture_args"]["layers"][1]["Linear"]["activation"] = "hard_sigmoid"
    model = build_model(cfg, model_id="mlp")
    assert isinstance(model[2], nn.Hardsigmoid)


# ------------------------------------------------------- torch.nn.init path --

def test_xavier_uniform_bounds_and_zero_bias():
    torch.manual_seed(0)
    model = build_model(_cnn_cfg(XAVIER), model_id="cnn")
    for layer in _kernel_layers(model):
        limit = _xavier_limit(layer.weight)
        w = layer.weight.detach()
        assert float(w.abs().max()) <= limit + 1e-7
        assert float(w.abs().max()) > 0.9 * limit
        assert torch.count_nonzero(layer.bias) == 0


def test_default_leaves_pytorch_init_untouched():
    torch.manual_seed(1)
    for cfg in (_cnn_cfg(), _cnn_cfg({"weight_init": "default", "bias_init": "default"})):
        model = build_model(cfg, model_id="cnn")
        # PyTorch's default bias is U(-1/sqrt(fan_in), 1/sqrt(fan_in)), never all zero.
        assert all(torch.count_nonzero(l.bias) > 0 for l in _kernel_layers(model))


def test_same_seed_same_weights_and_framework_args_reach_torch():
    block = {"weight_init": "kaiming_normal", "weight_init_args": {"nonlinearity": "linear"},
             "bias_init": "constant", "bias_init_args": {"val": 0.25}}
    torch.manual_seed(9)
    a = build_model(_mlp_cfg(block), model_id="mlp")
    torch.manual_seed(9)
    b = build_model(_mlp_cfg(block), model_id="mlp")
    assert torch.equal(a[1].weight, b[1].weight)
    assert torch.all(a[1].bias == 0.25)
    # kaiming_normal_(nonlinearity="linear"): std 1/sqrt(fan_in)
    assert abs(float(a[1].weight.std()) * math.sqrt(700) - 1.0) < 0.03


def test_unknown_torch_argument_is_rejected():
    with pytest.raises(ValueError, match="has no argument"):
        build_model(_mlp_cfg({"weight_init": "xavier_uniform", "weight_init_args": {"scale": 2.0}}), model_id="mlp")


def test_xavier_uniform_with_lazy_linear_materialises_then_initialises():
    torch.manual_seed(2)
    model = build_model(_cnn_cfg(XAVIER, lazy=True, target_range=(0, 700)), model_id="cnn")
    assert not has_lazy_parameters(model)
    lazy = model[4]
    assert isinstance(lazy, nn.Linear) and lazy.in_features == 8 * 350
    limit = _xavier_limit(lazy.weight)
    assert float(lazy.weight.detach().abs().max()) <= limit + 1e-7
    assert float(lazy.weight.detach().abs().max()) > 0.9 * limit
    assert torch.count_nonzero(lazy.bias) == 0
    # the model is back in train mode and works on real batches
    assert model.training
    assert model(torch.randn(4, 1, 700)).shape == (4, 256)


def test_lazy_layer_uses_configured_trace_length():
    cfg = _cnn_cfg(XAVIER, lazy=True, target_range=(100, 300))
    model = build_model(cfg, model_id="cnn")
    assert model[4].in_features == 8 * 100


def test_apply_torch_init_refuses_lazy_without_example_input():
    model = nn.Sequential(nn.Flatten(), nn.LazyLinear(4))
    with pytest.raises(ValueError, match="example input"):
        apply_torch_init(model, InitSpec("xavier_uniform"), InitSpec("zeros"), example_input=None)


# ------------------------------------------------- a layer's own initialiser --

HE = {"weight_init": "kaiming_uniform", "weight_init_args": {"nonlinearity": "relu"}, "bias_init": "zeros"}


def _he_limit(weight):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    return math.sqrt(6.0 / fan_in)


def test_layer_weight_init_overrides_the_model_for_that_layer_only():
    # Keras gives a layer its code names no kernel_initializer glorot_uniform (Zaid's and Wouters' output
    # layers), the rest he_uniform: kaiming_uniform (relu) model-wide, xavier_uniform on the output layer
    cfg = _cnn_cfg(HE)
    cfg["model"][0]["architecture_args"]["layers"][4]["Linear"]["weight_init"] = "xavier_uniform"
    torch.manual_seed(0)
    conv, hidden, out = _kernel_layers(build_model(cfg, model_id="cnn"))
    for layer in (conv, hidden):
        assert float(layer.weight.detach().abs().max()) <= _he_limit(layer.weight) + 1e-7
        assert float(layer.weight.detach().abs().max()) > 0.9 * _he_limit(layer.weight)
    assert float(out.weight.detach().abs().max()) <= _xavier_limit(out.weight) + 1e-7
    assert float(out.weight.detach().abs().max()) > 0.9 * _xavier_limit(out.weight)
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in (conv, hidden, out))


def test_layer_init_takes_name_and_args_and_refuses_other_layers():
    cfg = _cnn_cfg(HE)
    cfg["model"][0]["architecture_args"]["layers"][4]["Linear"]["weight_init"] = {"name": "xavier_uniform", "args": {"gain": 1.0}}
    assert build_model(cfg, model_id="cnn") is not None
    cfg = _cnn_cfg(HE)
    cfg["model"][0]["architecture_args"]["layers"][1]["Pooling"]["weight_init"] = "xavier_uniform"
    with pytest.raises(ValueError, match="Conv and Linear layers only"):
        build_model(cfg, model_id="cnn")
    cfg = _cnn_cfg(None)                                   # no deep_learning block: no layer initialiser
    cfg["model"][0]["architecture_args"]["layers"][4]["Linear"]["weight_init"] = "xavier_uniform"
    with pytest.raises(ValueError, match="needs a deep_learning block"):
        build_model(cfg, model_id="cnn")
