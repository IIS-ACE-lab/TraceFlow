# services/side_channel_core/tests/unit/test_final_models.py
"""
Codebase finalisation (2026-09-29): model construction and weight init.

DL-01  named architectures (ResNet50_1D, InceptionV3_1D) apply the
       deep_learning.pytorch initialisers (torch.nn.init).
DL-08  unknown keys, initialiser names of the wrong framework and swallowed
       activation args fail loudly with a hint.
CLAIM-16  ResNet50_1D / InceptionV3_1D follow Keras 2.1.1 keras.applications
       layer by layer (positions, stride placement, BN, bias, 7-tap mapping).

Run from the repository root:
    PYTHONPATH="$PWD:$PWD/services/side_channel_core:$PWD/services/side_channel_core/src" \
        CUDA_VISIBLE_DEVICES="" conda run -n sca python -m pytest \
        services/side_channel_core/tests/unit/test_final_models.py -q
"""
import copy
import inspect
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from core.model import build_model
from model_architectures.public_cnn_1d import InceptionV3_1D, ResNet50_1D, _TFSamePad, _avg_pool_same
from services.side_channel_core.src.core.model_contract_resolver import (
    ModelContractError,
    resolve_model_contract,
)
from services.side_channel_core.utils.model_preview import (
    build_resolved_model_preview_text,
    extract_resolved_model_payload,
)

FLAT_SIZE = Path(__file__).resolve().parents[2] / "utils" / "flat_size.py"
PARAMS_YAML = Path(__file__).resolve().parents[4] / "projects" / "DLSCA_ASCAD" / "config" / "params.yaml"


def _named_cfg(architecture, **arch_args):
    return {
        "sca": {"leakage_model": "ID"},
        "model": [{"model_id": "named", "architecture": architecture, "architecture_args": arch_args}],
    }


def _mlp_cfg(activation="relu", **arch_args):
    args = {
        "layers": [
            {"Flatten": {}},
            {"Linear": {"in_features": 700, "out_features": 200, "activation": activation}},
            {"Linear": {"in_features": 200, "out_features": "auto"}},
        ],
        **arch_args,
    }
    return {
        "sca": {"leakage_model": "ID"},
        "model": [{"model_id": "mlp", "architecture": "MLP", "architecture_args": args}],
    }


# ------------------------------------------------ DL-01: named architectures --

def _with_framework(cfg, **block):
    """cfg with deep_learning: {framework: pytorch, pytorch: block}."""
    return {**cfg, "deep_learning": {"framework": "pytorch", "pytorch": block}}


@pytest.mark.parametrize("architecture", ["ResNet50_1D", "InceptionV3_1D"])
def test_named_architecture_applies_xavier_uniform_and_zero_bias(architecture):
    torch.manual_seed(0)
    cfg = _with_framework(
        _named_cfg(architecture, in_channels=1, num_classes=256),
        weight_init="xavier_uniform", weight_init_args={"gain": 1.0}, bias_init="zeros",
    )
    model = build_model(cfg, model_id="named")

    layers = [m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    assert layers
    for layer in layers:
        w = layer.weight.detach()
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(w)
        limit = math.sqrt(6.0 / (fan_in + fan_out))      # torch.nn.init.xavier_uniform_, gain 1
        assert float(w.abs().max()) <= limit + 1e-7
        if w.numel() >= 10_000:
            assert abs(float(w.std()) / (limit / math.sqrt(3.0)) - 1.0) < 0.1
        if layer.bias is not None:
            assert torch.count_nonzero(layer.bias) == 0

    norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    assert norms and all(torch.all(m.weight == 1) and torch.all(m.bias == 0) for m in norms)

    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 1, 700)).shape == (2, 256)


def test_named_architecture_without_deep_learning_keeps_pytorch_default():
    torch.manual_seed(1)
    model = build_model(_named_cfg("ResNet50_1D", in_channels=1, num_classes=9), model_id="named")
    # PyTorch's own Linear bias is U(-1/sqrt(fan_in), 1/sqrt(fan_in)), never all zero.
    assert torch.count_nonzero(model.fc.bias) > 0


def test_model_weight_init_needs_the_deep_learning_block():
    cfg = _named_cfg("ResNet50_1D", in_channels=1, num_classes=9, weight_init="xavier_uniform")
    with pytest.raises(ValueError, match="needs a deep_learning block"):
        build_model(cfg, model_id="named")


def test_named_architecture_rejects_a_keras_name_under_pytorch():
    cfg = _with_framework(_named_cfg("InceptionV3_1D", in_channels=1, num_classes=256), weight_init="glorot_uniform")
    with pytest.raises(ValueError, match="is not a torch.nn.init function") as info:
        build_model(cfg, model_id="named")
    assert "'xavier_uniform'" in str(info.value) and "lecun" in str(info.value)


def test_named_architecture_unknown_constructor_key_hints():
    cfg = _named_cfg("ResNet50_1D", in_channels=1, num_class=256)
    with pytest.raises(ValueError, match=r"'num_class' \(did you mean 'num_classes'\?\)") as info:
        build_model(cfg, model_id="named")
    assert "weight_init" in str(info.value)


def test_resolver_and_preview_accept_weight_init_for_named_architecture():
    cfg = {
        "deep_learning": {"framework": "pytorch"},
        "trainer": {"optimizer": {"learning_rate": 1e-5}},
        "sca": {"leakage_model": "HW"},
        "model": [{
            "model_id": "resnet",
            "architecture": "ResNet50_1D",
            "architecture_args": {"in_channels": 1, "weight_init": "xavier_uniform", "num_classes": "auto"},
        }],
    }
    out = resolve_model_contract(cfg, mode="scc", model_id="resnet")
    args = out["model"][0]["architecture_args"]
    assert args == {"in_channels": 1, "weight_init": "xavier_uniform", "num_classes": 9}

    text = build_resolved_model_preview_text(out)
    assert "model weight_init=xavier_uniform" in text
    assert "framework       : pytorch" in text
    assert "ResNet50_1D class, built from architecture_args" in text
    payload = extract_resolved_model_payload(out, env="dev", model_name="m")
    assert "layers" not in payload["resolved_model"]["architecture_args"]

    bad = {**cfg, "model": [{**cfg["model"][0], "architecture_args": {**args, "weight_init": "banana"}}]}
    with pytest.raises(ModelContractError, match="is not a torch.nn.init function"):
        resolve_model_contract(bad, mode="scc", model_id="resnet")


# --------------------------------------------- DL-08: loud config failures --

def test_misspelled_architecture_args_key_names_the_closest_key():
    cfg = _mlp_cfg(strict_graph_args=False, weight_int="xavier_uniform")
    with pytest.raises(ValueError, match=r"'weight_int' \(did you mean 'weight_init'\?\)") as info:
        build_model(cfg, model_id="mlp")
    assert "Allowed:" in str(info.value) and "'strict_graph_args'" in str(info.value)


def test_layer_model_without_initialiser_keeps_pytorch_default():
    torch.manual_seed(4)
    model = build_model(_mlp_cfg(), model_id="mlp")
    assert torch.count_nonzero(model[1].bias) > 0


def test_model_level_initialiser_per_framework():
    torch.manual_seed(5)
    cfg = _with_framework(_mlp_cfg(
        "selu",
        weight_init={"pytorch": {"name": "kaiming_normal", "args": {"nonlinearity": "linear"}},
                     "keras": {"name": "lecun_normal"}},
    ))
    first = build_model(cfg, model_id="mlp")[1]
    # torch.nn.init.kaiming_normal_(nonlinearity="linear"): N(0, 1/fan_in), not truncated
    assert abs(float(first.weight.detach().std()) * math.sqrt(700) - 1.0) < 0.03
    assert float(first.weight.detach().abs().max()) > 2.5 / math.sqrt(700)


def test_softmax_activation_honours_dim_and_rejects_unknown_args():
    cfg = _mlp_cfg({"name": "softmax", "args": {"dim": -1}})
    model = build_model(cfg, model_id="mlp")
    assert isinstance(model[2], nn.Softmax) and model[2].dim == -1

    cfg = _mlp_cfg({"name": "softmax", "args": {"dimm": 1}})
    with pytest.raises(TypeError, match=r"'dimm' \(did you mean 'dim'\?\)"):
        build_model(cfg, model_id="mlp")


@pytest.mark.parametrize("name", ["identity", "linear", "none", ""])
def test_identity_activations_reject_arguments(name):
    model = build_model(_mlp_cfg({"name": name}), model_id="mlp")
    assert isinstance(model[2], nn.Identity)

    cfg = _mlp_cfg({"name": name, "args": {"slope": 3}})
    with pytest.raises(TypeError, match="unknown argument\\(s\\) 'slope'.*Allowed: none"):
        build_model(cfg, model_id="mlp")


@pytest.mark.parametrize("name, cls", [("logsoftmax", nn.LogSoftmax), ("softmin", nn.Softmin)])
def test_logsoftmax_and_softmin_honour_dim(name, cls):
    model = build_model(_mlp_cfg(name), model_id="mlp")
    assert isinstance(model[2], cls) and model[2].dim == 1

    model = build_model(_mlp_cfg({"name": name, "args": {"dim": -1}}), model_id="mlp")
    assert isinstance(model[2], cls) and model[2].dim == -1

    cfg = _mlp_cfg({"name": name, "args": {"dimm": 1}})
    with pytest.raises(TypeError, match=r"'dimm' \(did you mean 'dim'\?\)"):
        build_model(cfg, model_id="mlp")


def test_logsoftmax2d_is_not_offered():
    # torch has no nn.LogSoftmax2d; the name used to fail with an AttributeError at build time
    with pytest.raises(ValueError, match="Unknown activation 'logsoftmax2d'"):
        build_model(_mlp_cfg("logsoftmax2d"), model_id="mlp")


def test_layer_kwarg_typo_names_the_closest_constructor_arg():
    cfg = _mlp_cfg()
    cfg["model"][0]["architecture_args"]["layers"][1]["Linear"]["out_feature"] = 200
    with pytest.raises(TypeError, match=r"layer 'Linear' at index 1.*'out_feature' \(did you mean 'out_features'\?\)"):
        build_model(cfg, model_id="mlp")


# -------------------------------------------- flat_size: no absolute paths --

def test_flat_size_runs_from_any_checkout_on_a_given_yaml(tmp_path):
    arch = tmp_path / "arch.yaml"
    arch.write_text(yaml.safe_dump({"model": {"layers": [
        {"Conv1d": {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "padding": 1, "activation": "relu"}},
        {"Pooling": {"dim": 1, "type": "avg", "kernel_size": 2, "stride": 2}},
        {"Flatten": {}},
        {"Linear": {"in_features": "TBD", "out_features": 256}},
    ]}}))
    result = subprocess.run(
        [sys.executable, str(FLAT_SIZE), str(arch)],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "Flattened feature size before Linear: 1400" in result.stdout
    saved = yaml.safe_load(arch.read_text())
    assert saved["model"]["layers"][3]["Linear"]["in_features"] == 1400
    assert "weight_init" not in saved["model"]


# ------------------------- CLAIM-16: Keras 2.1.1 ResNet-50 / Inception-v3 ports --

def _trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _total(model):
    return sum(p.numel() for p in model.parameters())


def _convs(module):
    return [m for m in module.modules() if isinstance(m, nn.Conv1d)]


def _norms(module):
    return [m for m in module.modules() if isinstance(m, nn.BatchNorm1d)]


def test_resnet50_1d_lengths_channels_and_stride_placement():
    model = ResNet50_1D().eval()
    stages = (model.layer1, model.layer2, model.layer3, model.layer4)
    with torch.no_grad():
        x = model.stem(torch.randn(2, 1, 700))
        shapes = [tuple(x.shape[1:])]
        x = model.pool(x)
        shapes.append(tuple(x.shape[1:]))
        for stage in stages:
            x = stage(x)
            shapes.append(tuple(x.shape[1:]))
    assert shapes == [(64, 350), (64, 174), (256, 174), (512, 87), (1024, 44), (2048, 22)]
    assert [len(stage) for stage in stages] == [3, 4, 6, 3]

    # Keras v1: (conv1, conv2, shortcut) strides of each stage's first block;
    # the shortcut exists on the first block of every stage, stage 2 included.
    strides = []
    for stage in stages:
        first, rest = stage[0], list(stage)[1:]
        strides.append((first.conv1[0].stride[0], first.conv2[0].stride[0], first.downsample[0].stride[0]))
        assert all(b.downsample is None and b.conv1[0].stride[0] == 1 for b in rest)
    assert strides == [(1, 1, 1), (2, 1, 2), (2, 1, 2), (2, 1, 2)]


@pytest.mark.parametrize("num_classes, n_params", [(256, 16_505_344), (9, 15_999_241)])
def test_resnet50_1d_conv_bias_bn_defaults_and_parameter_count(num_classes, n_params):
    model = ResNet50_1D(num_classes=num_classes)
    convs, norms = _convs(model), _norms(model)
    assert len(convs) == 53 and all(c.bias is not None for c in convs)
    assert len(norms) == 53
    assert all(n.eps == 1e-3 and n.momentum == 0.01 and n.weight.requires_grad for n in norms)
    assert model.fc.in_features == 2048
    assert _total(model) == _trainable(model) == n_params

    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 1, 700)).shape == (2, num_classes)


@pytest.mark.parametrize("length, amounts", [(700, (2, 3)), (701, (3, 3))])
def test_tf_same_pad_matches_tensorflow(length, amounts):
    padded = _TFSamePad(7, 2)(torch.ones(1, 1, length))[0, 0]
    real = padded.nonzero().flatten()
    assert (int(real[0]), len(padded) - 1 - int(real[-1])) == amounts
    # TensorFlow SAME: a stride-2 convolution returns ceil(L / 2) positions
    assert nn.Conv1d(1, 1, 7, stride=2)(padded.view(1, 1, -1)).shape[-1] == math.ceil(length / 2)


def test_inception_v3_1d_lengths_channels():
    model = InceptionV3_1D().eval()
    with torch.no_grad():
        x = torch.randn(2, 1, 700)
        stem, blocks = [], []
        for layer in model.stem:
            x = layer(x)
            stem.append(x.shape[-1])
        for block in model.blocks:
            x = block(x)
            blocks.append(tuple(x.shape[1:]))
    assert stem == [349, 347, 347, 173, 173, 171, 85]
    assert blocks == [(256, 85), (288, 85), (288, 85)] + [(768, 42)] * 5 + [(1280, 20), (2048, 20), (2048, 20)]
    assert model.fc.in_features == 2048


def test_inception_factorised_kernels_follow_keras():
    blocks = list(InceptionV3_1D().blocks)

    # mixed4-7: 1x7, 7x1 and 7x1, 1x7, 7x1, 1x7 each become one 7-tap conv
    for block, c7 in zip(blocks[4:8], (128, 160, 160, 192)):
        assert [c.kernel_size[0] for c in _convs(block.branch7)] == [1, 7, 7]
        dbl = _convs(block.branch7dbl)
        assert [c.kernel_size[0] for c in dbl] == [1, 7, 7, 7, 7]
        assert [c.out_channels for c in dbl] == [c7, c7, c7, c7, 192]
        assert all(c.padding == (3,) for c in dbl[1:])

    # mixed8: 1x1, 1x7, 7x1, 3x3 stride 2
    d7 = _convs(blocks[8].branch7)
    assert [(c.kernel_size[0], c.stride[0]) for c in d7] == [(1, 1), (7, 1), (7, 1), (3, 2)]

    # mixed9-10: 1x3 || 3x1 stay two parallel 384-channel 3-tap convs per pair
    for block in blocks[9:]:
        for a, b in ((block.branch3_2a, block.branch3_2b), (block.branch3dbl_2a, block.branch3dbl_2b)):
            assert a[0] is not b[0]
            for conv in (a[0], b[0]):
                assert (conv.in_channels, conv.out_channels, conv.kernel_size, conv.padding) == (384, 384, (3,), (1,))
        assert block.out_channels == 2048


@pytest.mark.parametrize(
    "num_classes, trainable, total", [(256, 18_310_816, 18_328_032), (9, 17_804_713, 17_821_929)]
)
def test_inception_bn_has_no_scale_and_keras_defaults(num_classes, trainable, total):
    model = InceptionV3_1D(num_classes=num_classes)
    convs, norms = _convs(model), _norms(model)
    assert len(convs) == 94 and all(c.bias is None for c in convs)
    assert len(norms) == 94
    for n in norms:
        assert torch.all(n.weight == 1) and not n.weight.requires_grad and n.bias.requires_grad
        assert n.eps == 1e-3 and n.momentum == 0.01
    pools = [m for m in model.modules() if isinstance(m, nn.AvgPool1d)]
    assert len(pools) == 9 and all(p.count_include_pad is False for p in pools)
    # the frozen gammas are the only non-trainable parameters
    assert _trainable(model) == trainable and _total(model) == total
    assert total - trainable == sum(n.num_features for n in norms) == 17_216

    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 1, 700)).shape == (2, num_classes)


def test_same_average_pooling_leaves_padding_out_of_the_mean():
    out = _avg_pool_same()(torch.arange(1.0, 6.0).view(1, 1, 5))
    # TensorFlow SAME: the edge windows average 2 real samples, not 3 with a zero
    assert out.flatten().tolist() == [1.5, 2.0, 3.0, 4.0, 4.5]


def test_inception_gamma_stays_one_after_an_optimizer_step():
    torch.manual_seed(5)
    model = InceptionV3_1D(num_classes=9).train()
    # as train.py builds it: every parameter goes to the optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5, alpha=0.9, eps=1e-7)
    before = model.fc.weight.detach().clone()
    loss = nn.functional.cross_entropy(model(torch.randn(4, 1, 700)), torch.randint(0, 9, (4,)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.equal(model.fc.weight.detach(), before)
    assert all(n.weight.grad is None and torch.all(n.weight == 1) for n in _norms(model))


@pytest.mark.parametrize(
    "cls, names", [(ResNet50_1D, ["in_channels", "num_classes", "layers"]), (InceptionV3_1D, ["in_channels", "num_classes"])]
)
def test_constructors_take_no_var_kwargs(cls, names):
    # build_model checks architecture_args against this signature and applies
    # weight_init itself, so the classes take neither **kwargs nor weight_init
    params = list(inspect.signature(cls.__init__).parameters.values())[1:]
    assert [p.name for p in params] == names
    assert all(p.kind is not inspect.Parameter.VAR_KEYWORD for p in params)


def _concrete_params(leakage_model):
    cfg = yaml.safe_load(PARAMS_YAML.read_text())
    cfg["sca"]["leakage_model"] = leakage_model
    cfg["sca"]["attack_byte"] = 2
    cfg["sca"]["candidate_ranking"] = {"mode": "none", "top_k": 5}
    cfg["sca"]["key_enumeration"] = {"enabled": False}
    optimizer = dict(cfg["trainer"]["optimizer"][0])
    optimizer["learning_rate"] = optimizer["learning_rate"][0]
    cfg["trainer"]["optimizer"] = optimizer
    cfg["trainer"]["epochs"] = cfg["trainer"]["epochs"][0]
    cfg["trainer"]["batch_size"] = cfg["trainer"]["batch_size"][0]
    return cfg


@pytest.mark.parametrize("model_id", ["ASCAD_ResNet50", "ASCAD_InceptionV3"])
@pytest.mark.parametrize("leakage_model, num_classes", [("ID", 256), ("HW", 9)])
def test_params_yaml_named_models_build_with_xavier_uniform(model_id, leakage_model, num_classes):
    # ResNet-50 and Inception-v3 are graph models in params.yaml; under framework
    # pytorch they take deep_learning.pytorch (xavier_uniform gain 1, zero biases).
    torch.manual_seed(6)
    params = _concrete_params(leakage_model)
    params["deep_learning"]["framework"] = "pytorch"
    cfg = resolve_model_contract(params, mode="scc", model_id=model_id)
    args = next(m for m in cfg["model"] if m["model_id"] == model_id)["architecture_args"]
    head = next(n for n in args["nodes"] if n["name"] == args["outputs"][0])
    assert head["params"]["out_features"] == num_classes
    model = build_model(cfg, model_id=model_id)

    for layer in (m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        assert float(layer.weight.detach().abs().max()) <= math.sqrt(6.0 / (fan_in + fan_out)) + 1e-7
        assert layer.bias is None or torch.count_nonzero(layer.bias) == 0
    assert all(torch.all(n.weight == 1) and torch.all(n.bias == 0) for n in _norms(model))

    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 1, 700)).shape == (2, num_classes)


@pytest.mark.parametrize("framework", ["pytorch", "keras"])
@pytest.mark.parametrize("leakage_model, num_classes", [("ID", 256), ("HW", 9)])
def test_params_yaml_snn_builds_standard_normal(framework, leakage_model, num_classes):
    # Sect. C.3.1 p. 184: "the initialization of the weights is performed from a
    # standard normal distribution": N(0, 1), not truncated; zero biases.
    params = _concrete_params(leakage_model)
    params["deep_learning"]["framework"] = framework
    snn = next(m for m in params["model"] if m["model_id"] == "ASCAD_SNN")["architecture_args"]
    assert snn["weight_init"] == {
        "pytorch": {"name": "normal", "args": {"mean": 0.0, "std": 1.0}},
        "keras": {"name": "random_normal", "args": {"mean": 0.0, "stddev": 1.0}},
    }

    torch.manual_seed(7)
    cfg = resolve_model_contract(params, mode="scc", model_id="ASCAD_SNN")
    model = build_model(cfg, model_id="ASCAD_SNN")
    weights = [p.detach() for p in model.parameters() if p.ndim == 2]
    biases = [p.detach() for p in model.parameters() if p.ndim == 1]
    assert len(weights) == 6
    for w in weights:
        assert abs(float(w.std()) - 1.0) < 0.05 and abs(float(w.mean())) < 0.05
    assert max(float(w.abs().max()) for w in weights) > 4.0   # not cut at 2 std
    assert all(torch.count_nonzero(b) == 0 for b in biases)

    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 1, 700)).shape == (2, num_classes)


@pytest.mark.parametrize("leakage_model, num_classes", [("ID", 256), ("HW", 9)])
def test_lecun_normal_under_keras_is_keras_lecunnormal(leakage_model, num_classes):
    # keras.initializers.LecunNormal (the rule of the SELU article), on the ASCAD_SNN network.
    params = _concrete_params(leakage_model)
    snn = next(m for m in params["model"] if m["model_id"] == "ASCAD_SNN")
    twin = copy.deepcopy(snn)
    twin["model_id"] = "snn_lecun"
    twin["architecture_args"]["weight_init"] = {"keras": {"name": "lecun_normal"}}
    params["model"].append(twin)

    torch.manual_seed(7)
    cfg = resolve_model_contract(params, mode="scc", model_id="snn_lecun")
    model = build_model(cfg, model_id="snn_lecun")
    assert getattr(model, "framework", None) == "keras"

    dense = [layer for layer in model.keras_model.layers if type(layer).__name__ == "Dense"]
    kernel = dense[0].kernel.value.detach()
    sigma = math.sqrt(1.0 / 700)
    assert tuple(kernel.shape) == (700, 200)              # Keras kernel layout (in, out)
    # keras.initializers.LecunNormal: truncated at 2 x sigma / 0.8796, realised std sigma.
    # Keras' torch-backend truncated_normal draws 4 candidates per weight and keeps
    # the first inside +-2 std, or the first candidate when none is (probability
    # 0.0455^4 = 4e-6), so a rare weight lies beyond the cut.
    assert abs(float(kernel.std()) / sigma - 1.0) < 0.05
    beyond = (kernel.abs() > 2.0 * sigma / 0.87962566103423978 + 1e-7).float().mean()
    assert float(beyond) < 1e-4
    assert all(torch.count_nonzero(layer.bias.value) == 0 for layer in dense)

    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 1, 700)).shape == (2, num_classes)


_EQ9_CNNS = [
    "ASCAD_CNNbest", "ASCAD_CNNbase", "ASCAD_CNNbase_blocks2", "ASCAD_CNNbase_blocks3",
    "ASCAD_CNNbase_blocks5", "ASCAD_CNNbase_conv0", "ASCAD_CNNbase_conv2", "ASCAD_CNNbase_conv3",
    "ASCAD_CNNbase_filters16", "ASCAD_CNNbase_filters32", "ASCAD_CNNbase_filters128",
    "ASCAD_CNNbase_kernel6", "ASCAD_CNNbase_kernel11", "ASCAD_CNNbase_dense0", "ASCAD_CNNbase_dense1",
    "ASCAD_CNNbase_dense3", "ASCAD_CNNbase_sigmoid", "ASCAD_CNNbase_tanh", "ASCAD_CNNbase_avgpool",
    "ASCAD_CNNbase_validpad", "ASCAD_CNNbase_stridepool",
]


def test_params_yaml_cnns_follow_eq9():
    # Eq. (9) p. 171, s o [lambda]^ndense o [delta o [alpha o gamma]^nconv]^nblocks:
    # gamma a CONV with no activation, alpha = BN then ACT; every hidden FC layer
    # lambda applies ReLU (Sect. 3.3.4 p. 176: the dense layers simulate "the QDA part";
    # the released cnn_best), and the output layer none, the softmax s following it.
    blocks = {m["model_id"]: m for m in _concrete_params("ID")["model"]}
    assert set(_EQ9_CNNS) <= set(blocks)
    for model_id in _EQ9_CNNS:
        layers = [next(iter(layer.items())) for layer in blocks[model_id]["architecture_args"]["layers"]]
        names = [name for name, _ in layers]
        for i, (name, args) in enumerate(layers):
            if name == "Conv1d":
                assert "activation" not in args, model_id
                bn_name, bn = layers[i + 1]
                assert bn_name == "BatchNorm1d" and bn["num_features"] == args["out_channels"], model_id
                assert bn["activation"] in ("relu", "sigmoid", "tanh"), model_id
        dense = [args for name, args in layers if name == "Linear"]
        assert all(d.get("activation") == "relu" for d in dense[:-1]), model_id
        assert "activation" not in dense[-1], model_id
        assert names.count("BatchNorm1d") == names.count("Conv1d"), model_id
        assert names.count("Pooling") == {"ASCAD_CNNbest": 5, "ASCAD_CNNbase_blocks2": 2,
                                          "ASCAD_CNNbase_blocks3": 3, "ASCAD_CNNbase_blocks5": 5}.get(model_id, 4), model_id


@pytest.mark.parametrize("framework", ["pytorch", "keras"])
def test_params_yaml_cnnbest_eq9_weights(framework):
    # 66,652,544 conv and dense weights plus gamma and beta of the five BN layers
    # (2 x (64 + 128 + 256 + 512 + 512)); Keras also lists the moving mean and variance.
    params = _concrete_params("ID")
    params["deep_learning"]["framework"] = framework
    cfg = resolve_model_contract(params, mode="scc", model_id="ASCAD_CNNbest")
    model = build_model(cfg, model_id="ASCAD_CNNbest")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 66_652_544 + 2 * 1472
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 1, 700)).shape == (2, 256)


# ------------- ResNet-50 / Inception-v3 as params.yaml graphs (2026-10-10) --

def _bn_eps(model, eps):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eps = eps


@pytest.mark.parametrize("model_id, port", [("ASCAD_ResNet50", ResNet50_1D), ("ASCAD_InceptionV3", InceptionV3_1D)])
def test_params_yaml_graph_equals_the_class_port(model_id, port):
    """The YAML graph is the class port layer for layer: with the port's weights
    (and its Keras BatchNorm epsilon) it gives exactly the port's outputs."""
    torch.manual_seed(11)
    params = _concrete_params("ID")
    params["deep_learning"]["framework"] = "pytorch"
    graph = build_model(resolve_model_contract(params, mode="scc", model_id=model_id), model_id=model_id).eval()
    reference = port(1, 256).eval()
    with torch.no_grad():
        for module in reference.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.running_mean.uniform_(-0.5, 0.5)
                module.running_var.uniform_(0.5, 2.0)
                module.bias.uniform_(-0.2, 0.2)
    source = [v for k, v in reference.state_dict().items() if not k.endswith("num_batches_tracked")]
    keys = [k for k in graph.state_dict() if not k.endswith("num_batches_tracked")]
    assert [v.shape for v in source] == [graph.state_dict()[k].shape for k in keys]
    state = graph.state_dict()
    state.update(dict(zip(keys, source)))
    graph.load_state_dict(state)
    _bn_eps(graph, 1e-3)
    assert _trainable(graph) == _trainable(reference)
    x = torch.randn(2, 1, 700) * 30
    with torch.no_grad():
        assert torch.equal(graph(x), reference(x))


def test_params_yaml_resnet_keras_graph_computes_the_same_function():
    """Under keras the same nodes build keras.layers; with the same weights the
    output agrees with the PyTorch graph to float32 rounding."""
    params = _concrete_params("HW")
    built = {}
    for framework in ("pytorch", "keras"):
        cfg = dict(params, deep_learning=dict(params["deep_learning"], framework=framework))
        torch.manual_seed(12)
        built[framework] = build_model(resolve_model_contract(cfg, mode="scc", model_id="ASCAD_ResNet50"),
                                       model_id="ASCAD_ResNet50").eval()
    torch_graph, keras_graph = built["pytorch"], built["keras"]
    assert keras_graph.framework == "keras" and _trainable(keras_graph) == _trainable(torch_graph)
    _bn_eps(torch_graph, 1e-3)
    with torch.no_grad():
        for name, module in torch_graph.layers.items():
            layer = getattr(module, "module", module)
            target = keras_graph.keras_model.get_layer(name)
            if isinstance(layer, nn.Conv1d):
                target.kernel.assign(layer.weight.permute(2, 1, 0).contiguous())
                target.bias.assign(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                target.gamma.assign(layer.weight)
                target.beta.assign(layer.bias)
                target.moving_mean.assign(layer.running_mean)
                target.moving_variance.assign(layer.running_var)
            elif isinstance(layer, nn.Linear):
                target.kernel.assign(layer.weight.T.contiguous())
                target.bias.assign(layer.bias)
        x = torch.randn(2, 1, 700) * 30
        a, b = torch_graph(x), keras_graph(x)
    assert a.shape == (2, 9)
    assert float((a - b).abs().max()) <= 1e-5 * float(a.abs().max())


def test_stride_pooling_is_library_layers_and_the_first_value_of_each_window():
    from core.model import Pooling

    x = torch.randn(3, 4, 175)
    pool = Pooling(type="stride", dim=1, kernel_size=2, stride=2)
    assert [type(m).__name__ for m in pool.pool] == ["ConstantPad1d", "MaxPool1d"]
    assert torch.equal(pool(x), x[..., :174:2])          # whole windows: 175 -> 87


def test_graph_wiring_is_never_a_sweep_dimension():
    """inputs / outputs / padding pairs of a graph are structure: a sweep over them
    would rewire the network (found by the end-to-end run of 2026-10-10)."""
    from utils.sweep_engine import _find_sweep_paths

    cfg = yaml.safe_load(PARAMS_YAML.read_text())
    assert [p for p, _ in _find_sweep_paths(cfg) if p and p[0] == "model"] == []
    graph = {"model": [{"architecture_args": {"nodes": [
        {"name": "a", "type": "ZeroPad1d", "inputs": ["input"], "params": {"padding": [2, 3]}},
        {"name": "b", "type": "Add", "inputs": ["a", "input"]}], "outputs": ["b"],
        "layers": [{"Conv1d": {"kernel_size": [3, 11]}}]}}]}
    # a primitive list inside a layer list is still a sweep (kernel_size: [3, 11])
    assert [p for p, _ in _find_sweep_paths(graph)] == [("model", 0, "architecture_args", "layers", 0, "Conv1d", "kernel_size")]
