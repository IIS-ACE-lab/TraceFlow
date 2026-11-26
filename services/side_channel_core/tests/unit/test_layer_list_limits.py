# services/side_channel_core/tests/unit/test_layer_list_limits.py
"""core.model.LAYER_LIST_LIMITS: the layers a params.yaml layer list cannot build on 1-D traces, each with
why. The panel's menus show them disabled with that reason (choices.model_forms). The probe below builds
each layer the way a run does (core.model.build_model) between a Conv1d and the dense head, on one batch of
zero traces, so the list always matches what TraceFlow builds."""

import warnings

import pytest
import torch

from core.model import LAYER_LIST_LIMITS, _LAYER_MAP, build_model

NAMED = {"ResNetV1", "ResNet50_1D", "InceptionV3_1D", "MultiTest", "MultiResNet", "MultiResNetNoPermind"}

# each layer's smallest arguments, after Conv1d(1 -> 8): 8 channels x 700 samples
MINIMAL_ARGS = {
    "Conv1d": dict(in_channels=8, out_channels=8, kernel_size=3, padding=1),
    "Conv2d": dict(in_channels=8, out_channels=8, kernel_size=3, padding=1),
    "Conv3d": dict(in_channels=8, out_channels=8, kernel_size=3, padding=1),
    "ConvTranspose1d": dict(in_channels=8, out_channels=8, kernel_size=3, padding=1),
    "ConvTranspose2d": dict(in_channels=8, out_channels=8, kernel_size=3, padding=1),
    "ConvTranspose3d": dict(in_channels=8, out_channels=8, kernel_size=3, padding=1),
    "Pooling": dict(dim=1, type="max", kernel_size=2, stride=2),
    "Linear": dict(in_features=700, out_features=8),
    "LazyLinear": dict(out_features=8),
    "Embedding": dict(num_embeddings=10, embedding_dim=4),
    "Flatten": {},
    "View": dict(shape=[-1, 5600]),
    "Dropout": dict(p=0.1), "Dropout1d": dict(p=0.1), "Dropout2d": dict(p=0.1), "Dropout3d": dict(p=0.1),
    "AlphaDropout": dict(p=0.1),
    "BatchNorm1d": dict(num_features=8), "BatchNorm2d": dict(num_features=8), "BatchNorm3d": dict(num_features=8),
    "InstanceNorm1d": dict(num_features=8), "InstanceNorm2d": dict(num_features=8), "InstanceNorm3d": dict(num_features=8),
    "LayerNorm": dict(normalized_shape=700),
    "GroupNorm": dict(num_groups=2, num_channels=8),
    "LocalResponseNorm": dict(size=2),
    "Upsample": dict(scale_factor=2),
    "PixelShuffle": dict(upscale_factor=2),
    "RNN": dict(input_size=700, hidden_size=8, batch_first=True),
    "LSTM": dict(input_size=700, hidden_size=8, batch_first=True),
    "GRU": dict(input_size=700, hidden_size=8, batch_first=True),
    "MultiheadAttention": dict(embed_dim=700, num_heads=1),
    "TransformerEncoderLayer": dict(d_model=700, nhead=1),
    "TransformerDecoderLayer": dict(d_model=700, nhead=1),
    "SPP": dict(levels=[1, 2]),
    "SpectralPool1d": dict(keep_ratio=0.5),
    "SpectralPool2d": dict(keep_ratio=0.5),
    "ZeroPad1d": dict(padding=2),
    "Add": {}, "Concatenate": {}, "Identity": {},
}

# builds, but does not do what a layer list means: with batch_first=False it attends across the batch
BUILDS_BUT_WRONG = {"TransformerEncoderLayer"}


def _probe_builds(name: str) -> bool:
    layers = [{"Conv1d": dict(in_channels=1, out_channels=8, kernel_size=3, padding=1)}, {name: MINIMAL_ARGS[name]},
              {"Flatten": {}}, {"LazyLinear": dict(out_features=4)}]
    cfg = {"model_id": "P", "sca": {"leakage_model": "ID"},
           "model": [{"model_id": "P", "architecture": "CNN", "architecture_args": {"layers": layers}}]}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return tuple(build_model(cfg, "P")(torch.zeros(2, 1, 700)).shape) == (2, 4)
    except Exception:  # noqa: BLE001 - any failure is "does not build"
        return False


def test_every_layer_of_the_map_is_probed():
    assert set(MINIMAL_ARGS) == set(_LAYER_MAP) - NAMED


def test_the_listed_layers_are_the_ones_a_layer_list_cannot_build():
    failing = {n for n in MINIMAL_ARGS if not _probe_builds(n)}
    assert set(LAYER_LIST_LIMITS) == failing | BUILDS_BUT_WRONG
    assert {"LSTM", "GRU", "RNN", "MultiheadAttention", "TransformerEncoderLayer"} <= set(LAYER_LIST_LIMITS)


@pytest.mark.parametrize("name", sorted(LAYER_LIST_LIMITS))
def test_every_limit_says_why(name):
    assert isinstance(LAYER_LIST_LIMITS[name], str) and len(LAYER_LIST_LIMITS[name]) > 20
