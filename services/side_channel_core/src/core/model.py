# services/side_channel_core/src/model.py
'''Dynamic model builder for PyTorch.
This module provides a unified way to create various layers in PyTorch, including convolutions, pooling, and custom layers.
It allows you to define a model architecture in a configuration file and build it dynamically. With this in place, your build_model will recognize all the above layer names, giving you complete flexibility to craft any CNN, RNN, Transformer, or hybrid architecture purely via YAML.


Extend _LAYER_MAP at any time by adding:
_LAYER_MAP["MyCustomLayer"] = MyCustomLayerClass
Then, in your model config, use:
    - MyCustomLayer: {param1: value1, param2: value2}  
This will automatically build the layer with the given parameters.

# ─── Supported layer types ────────────────────────────────────────────────────────────────────────────────
    # Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
    # Linear, Embedding
    # Flatten, View
    # Dropout, Dropout1d, Dropout2d, Dropout3d, AlphaDropout
    # BatchNorm1d, BatchNorm2d, BatchNorm3d
    # InstanceNorm1d, InstanceNorm2d, InstanceNorm3d
    # LayerNorm, GroupNorm, LocalResponseNorm
    # Pooling         ← unified pooling (all variants below)
        #   - max, avg, dilated_max, dilated_avg, global_max, global_avg, adaptive_max, adaptive_avg, adaptive, lp, lppool, fractional_max, fractional, spp, spatial_pyramid, spectral
    # SPP (SpatialPyramidPooling)
    # SpectralPool1d, SpectralPool2d
    # Upsample, PixelShuffle
    # RNN, LSTM, GRU
    # MultiheadAttention, TransformerEncoderLayer, TransformerDecoderLayer
    # Activation functions: relu, softmax, sigmoid, tanh, leaky_relu, prelu
    # (and more, see _ACTIVATION_MAP)
    # ────────────────────────────────────────────────────────────────────────────────────────────────────

Author: Shraddha Mishra
#
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import difflib
import inspect
from typing import List
from utils.logger import get_logger
from model_architectures.resnet import ResNet1D
from model_architectures.public_cnn_1d import InceptionV3_1D, ResNet50_1D
from model_architectures.multi_test import MultiTest
from model_architectures.multi_resnet import MultiResNet, MultiResNetNoPermind
from .framework_switch import (
    LAYER_INIT_ATTR,
    FrameworkSpec,
    _KERNEL_LAYERS,
    apply_torch_init,
    build_keras_graph,
    build_keras_model,
    has_lazy_parameters,
    layer_init_overrides,
    keras_input_shape,
    model_init_specs,
    resolve_framework,
)

logger = get_logger("Model")

# === Helper classes for special pooling ===

class SpectralPooling1d(nn.Module):
    def __init__(self, keep_ratio: float = 0.5):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L)
        freq = torch.fft.rfft(x, dim=-1)
        cutoff = int(freq.size(-1) * self.keep_ratio)
        freq[..., cutoff:] = 0
        return torch.fft.irfft(freq, n=x.size(-1), dim=-1)

class SpectralPooling2d(nn.Module):
    def __init__(self, keep_ratio: float = 0.5):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        freq = torch.fft.rfftn(x, dim=(-2, -1))
        H, Wf = freq.size(-2), freq.size(-1)
        cut_H = int(H * self.keep_ratio)
        cut_W = int(Wf * self.keep_ratio)
        freq[..., cut_H:, :] = 0
        freq[..., :, cut_W:] = 0
        return torch.fft.irfftn(freq, s=x.shape[-2:], dim=(-2, -1))

class SpatialPyramidPooling(nn.Module):
    """
    Splits feature maps into multiple adaptive pooled grids and concatenates.
    levels: list of ints, e.g. [1,2,4]
    pool_type: "max" or "avg"
    """
    def __init__(self, levels: List[int], pool_type: str = "max"):
        super().__init__()
        self.levels = levels
        self.pool_type = pool_type.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        N, C, H, W = x.size()
        outputs = []
        for lvl in self.levels:
            size = (lvl, lvl)
            if self.pool_type == "max":
                tensor = F.adaptive_max_pool2d(x, size)
            else:
                tensor = F.adaptive_avg_pool2d(x, size)
            outputs.append(tensor.view(N, -1))
        return torch.cat(outputs, dim=1)


class Add(nn.Module):
    """Takes two or more tensors and returns their element-wise sum."""
    def forward(self, *inputs):
        return sum(inputs)


class Concatenate(nn.Module):
    """Concatenates two or more (N, C, L) tensors along the channel axis, in input order."""
    def forward(self, *inputs):
        return torch.cat(inputs, dim=1)


def _batch_norm_1d(
    num_features: int,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = True,
    track_running_stats: bool = True,
    scale: bool = True,
) -> nn.BatchNorm1d:
    """``nn.BatchNorm1d``; ``scale: false`` keeps gamma at 1 (not trained), as
    Keras BatchNormalization(scale=False) has no gamma (Inception-v3's conv2d_bn)."""
    bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine,
                        track_running_stats=track_running_stats)
    if not scale:
        if not affine:
            raise ValueError("BatchNorm1d scale: false needs affine: true (beta is still trained)")
        bn.weight.requires_grad_(False)
    return bn

def _softmax(dim: int = 1) -> nn.Softmax:
    """Softmax over the channel/class axis unless ``args: {dim: ...}`` says otherwise."""
    return nn.Softmax(dim=dim)


def _log_softmax(dim: int = 1) -> nn.LogSoftmax:
    """LogSoftmax over the channel/class axis unless ``args: {dim: ...}`` says otherwise."""
    return nn.LogSoftmax(dim=dim)


def _softmin(dim: int = 1) -> nn.Softmin:
    """Softmin over the channel/class axis unless ``args: {dim: ...}`` says otherwise."""
    return nn.Softmin(dim=dim)


def _identity() -> nn.Identity:
    """
    No activation. It takes no arguments, so a configured ``args`` block is
    rejected instead of being swallowed by ``nn.Identity(*args, **kwargs)``.
    """
    return nn.Identity()


class Identity(nn.Module):
    """A no-op layer for explicit skip connections."""
    def forward(self, x):
        return x

class View(nn.Module):
    """
    True reshape/view layer.

    YAML example:
      - View: {shape: [256, 10]}   # => (N, 256, 10)

    Notes:
    - Keeps batch dimension N.
    - Uses .view(), so input must be contiguous. If not, we call contiguous().
    """
    def __init__(self, shape):
        super().__init__()
        if isinstance(shape, int):
            shape = [shape]
        if not isinstance(shape, (list, tuple)) or len(shape) == 0:
            raise ValueError("View requires 'shape' as non-empty list/tuple or int")
        self.shape = tuple(int(s) for s in shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        return x.view(x.size(0), *self.shape)

# === Main Pooling class ===

# Every name Pooling(type=...) builds, in the order of its branches (the config panel's menu;
# tests/unit/test_option_lists.py builds each one).
POOLING_TYPES = ("max", "avg", "dilated_max", "dilated_avg", "stride", "stride_pool", "global_max", "global_avg",
                 "adaptive_max", "adaptive_avg", "adaptive", "lp", "lppool", "fractional_max", "fractional",
                 "spp", "spatial_pyramid", "spectral")
# The ones that build on 1-D traces (dim=1), which the panel offers. Not these, each raising today:
# dilated_avg (AvgPool1d has no dilation), fractional_max / fractional (torch has no
# FractionalMaxPool1d), spp / spatial_pyramid (SpatialPyramidPooling takes 4-D input). TODO.md §25.
POOLING_TYPES_1D = tuple(t for t in POOLING_TYPES
                         if t not in {"dilated_avg", "fractional_max", "fractional", "spp", "spatial_pyramid"})


class Pooling(nn.Module):
    """
    Unified pooling layer. Supported `type` values:

        # fixed-window
        - max, avg, dilated_max, dilated_avg
        # global
        - global_max, global_avg
        # adaptive
        - adaptive_max, adaptive_avg, adaptive
        # Lp-norm
        - lp, lppool
        # fractional max
        - fractional_max, fractional
        # spatial pyramid
        - spp, spatial_pyramid
        # spectral
        - spectral

    Common parameters:
        dim: 1 or 2
        kernel_size: for fixed, dilated, lp, fractional
        stride: optional
        padding: optional
        dilation: for dilated
        output_size: for adaptive (or fractional)
        p: float (for lp-norm)
        return_indices: bool (for fractional)
        ceil_mode: bool (for fixed or lp)
        levels: list[int] (for spp)
        pool_type: "max" or "avg" (for spp)
        keep_ratio: float in (0,1] (for spectral)
    """
    def __init__(
        self,
        type: str = "max",
        dim: int = 2,
        kernel_size=None,
        stride=None,
        padding=0,
        dilation=1,
        output_size=None,
        p: float = 2.0,
        return_indices: bool = False,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        levels: List[int] = None,
        pool_type: str = "max",
        keep_ratio: float = 0.5,
    ):
        super().__init__()
        t = type.lower()
        if dim == 1 and t in POOLING_TYPES and t not in POOLING_TYPES_1D:
            # TODO.md §25: refused here, with the reason, not by a torch error at build or at the first batch
            why = {"dilated_avg": "average pooling has no dilation in torch (AvgPool1d)",
                   "fractional_max": "torch has no FractionalMaxPool1d (2-D and 3-D only)",
                   "fractional": "torch has no FractionalMaxPool1d (2-D and 3-D only)",
                   "spp": "SpatialPyramidPooling takes 4-D (image) input",
                   "spatial_pyramid": "SpatialPyramidPooling takes 4-D (image) input"}[t]
            raise ValueError(f"Pooling type {type!r} cannot pool 1-D traces (dim=1): {why}; "
                             f"use one of {', '.join(POOLING_TYPES_1D)}")

        # 1) Fixed-window pooling
        if t in ("max", "avg", "dilated_max", "dilated_avg"):
            if kernel_size is None:
                raise ValueError(f"{type} pooling requires kernel_size")
            is_max = "max" in t
            cls = (nn.MaxPool1d if dim == 1 else nn.MaxPool2d) if is_max else (nn.AvgPool1d if dim == 1 else nn.AvgPool2d)
            # build only the args that the underlying class actually supports
            params = {
                "kernel_size": kernel_size,
                "stride": stride or kernel_size,
                "padding": padding,
                "ceil_mode": ceil_mode,
            }
            # only inject dilation if you asked for a dilated variant
            if "dilated" in t:
                params["dilation"] = dilation
            if not is_max:
                # False leaves the padded zeros out of the mean (Keras/TF 'same').
                params["count_include_pad"] = bool(count_include_pad)
            self.pool = cls(**{k: v for k, v in params.items() if v is not None})

        # 1b) Stride pooling: ASCAD footnote 24 p. 186, benchmarked in Fig. 31.
        elif t in ("stride", "stride_pool"):
            if dim != 1:
                raise ValueError("stride pooling is implemented for 1-D traces only")
            if kernel_size is None:
                raise ValueError("stride pooling requires kernel_size")
            # ASCAD footnote 24 p. 186: "taking the first value on each input window
            # defined by the stride", on the window grid of max/avg pooling (whole
            # windows only, 175 -> 87). torch.nn has no stride-pooling layer; this is
            # torch.nn layers only: ConstantPad1d with a negative right padding crops
            # the last kernel_size - 1 samples, MaxPool1d(1, stride) then keeps
            # positions 0, s, 2s, ... (Keras: Cropping1D + MaxPooling1D(pool_size=1)).
            kernel = int(kernel_size)
            self.pool = nn.Sequential(
                nn.ConstantPad1d((0, -(kernel - 1)), 0.0),
                nn.MaxPool1d(kernel_size=1, stride=int(stride or kernel)),
            )

        # 2) Global pooling
        elif t in ("global_max", "global_avg"):
            size = 1 if dim == 1 else (1,1)
            if "max" in t:
                cls = nn.AdaptiveMaxPool1d if dim==1 else nn.AdaptiveMaxPool2d
            else:
                cls = nn.AdaptiveAvgPool1d if dim==1 else nn.AdaptiveAvgPool2d
            self.pool = cls(size)

        # 3) Adaptive pooling
        elif t in ("adaptive_max", "adaptive_avg", "adaptive"):
            if "max" in t:
                cls = nn.AdaptiveMaxPool1d if dim==1 else nn.AdaptiveMaxPool2d
            else:
                cls = nn.AdaptiveAvgPool1d if dim==1 else nn.AdaptiveAvgPool2d
            size = output_size if output_size is not None else (1 if dim==1 else (1,1))
            self.pool = cls(size)

        # 4) Lp-norm pooling
        elif t in ("lp", "lppool"):
            if kernel_size is None:
                raise ValueError("Lp pooling requires kernel_size and p")
            cls = nn.LPPool1d if dim==1 else nn.LPPool2d
            params = {
                "norm_type": p,
                "kernel_size": kernel_size,
                "stride": stride or kernel_size,
                "ceil_mode": ceil_mode,
            }
            self.pool = cls(**params)

        # 5) Fractional max pooling
        elif t in ("fractional_max", "fractional"):
            if kernel_size is None:
                raise ValueError("FractionalMaxPool requires kernel_size")
            cls = nn.FractionalMaxPool1d if dim==1 else nn.FractionalMaxPool2d
            params = {
                "kernel_size": kernel_size,
                "output_size": output_size,
                "return_indices": return_indices,
            }
            self.pool = cls(**{k:v for k,v in params.items() if v is not None})

        # 6) Spatial Pyramid Pooling
        elif t in ("spp", "spatial_pyramid"):
            if not levels:
                raise ValueError("SPP requires a list of pyramid levels `levels`")
            self.pool = SpatialPyramidPooling(levels, pool_type)

        # 7) Spectral pooling
        elif t in ("spectral",):
            if keep_ratio <= 0 or keep_ratio > 1:
                raise ValueError("keep_ratio must be in (0,1]")
            self.pool = SpectralPooling1d(keep_ratio) if dim==1 else SpectralPooling2d(keep_ratio)

        else:
            raise ValueError(f"Unknown pooling type '{type}'; one of {', '.join(POOLING_TYPES)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool(x)
        # Fractional and some return (tensor, indices)
        return out[0] if isinstance(out, tuple) else out

# === Dynamic model builder ===

_LAYER_MAP = {
    # ─── Convolutions ───────────────────────────────────────────────
    "Conv1d":            nn.Conv1d,
    "Conv2d":            nn.Conv2d,
    "Conv3d":            nn.Conv3d,
    "ConvTranspose1d":   nn.ConvTranspose1d,
    "ConvTranspose2d":   nn.ConvTranspose2d,
    "ConvTranspose3d":   nn.ConvTranspose3d,

    # ─── Pooling (your unified class) ──────────────────────────────
    "Pooling":           Pooling,

    # ─── Linear / Dense ────────────────────────────────────────────
    "Linear":            nn.Linear,
    "LazyLinear": nn.LazyLinear,

    # ─── Embedding ─────────────────────────────────────────────────
    "Embedding":         nn.Embedding,

    # ─── Flatten / Reshape ─────────────────────────────────────────
    "Flatten":           nn.Flatten,
    "View":              View,  # or custom view layers

    # ─── Dropout ──────────────────────────────────────────────────
    "Dropout":           nn.Dropout,
    "Dropout1d":         nn.Dropout1d,
    "Dropout2d":         nn.Dropout2d,
    "Dropout3d":         nn.Dropout3d,
    "AlphaDropout":      nn.AlphaDropout,

    # ─── Normalization ────────────────────────────────────────────
    "BatchNorm1d":       _batch_norm_1d,   # nn.BatchNorm1d (+ scale: false)
    "BatchNorm2d":       nn.BatchNorm2d,
    "BatchNorm3d":       nn.BatchNorm3d,
    "InstanceNorm1d":    nn.InstanceNorm1d,
    "InstanceNorm2d":    nn.InstanceNorm2d,
    "InstanceNorm3d":    nn.InstanceNorm3d,
    "LayerNorm":         nn.LayerNorm,
    "GroupNorm":         nn.GroupNorm,
    "LocalResponseNorm": nn.LocalResponseNorm,

    # ─── Upsampling / Interpolation ─────────────────────────────────
    "Upsample":          nn.Upsample,
    "PixelShuffle":      nn.PixelShuffle,

    # ─── RNNs / Sequence ────────────────────────────────────────────
    "RNN":               nn.RNN,
    "LSTM":              nn.LSTM,
    "GRU":               nn.GRU,

    # ─── Attention / Transformer ───────────────────────────────────
    "MultiheadAttention":        nn.MultiheadAttention,
    "TransformerEncoderLayer":   nn.TransformerEncoderLayer,
    "TransformerDecoderLayer":   nn.TransformerDecoderLayer,

    # ─── Special pooling variants ──────────────────────────────────
    "SPP":               SpatialPyramidPooling,
    "SpectralPool1d":    SpectralPooling1d,
    "SpectralPool2d":    SpectralPooling2d,

    # ─── Padding ──────────────────────────────────────────────────
    "ZeroPad1d":         nn.ZeroPad1d,

    # Graph‐merge ops
    "Add":      Add,
    "Concatenate": Concatenate,
    "Identity": Identity,
    
    #
    "ResNetV1":             ResNet1D,
    "ResNet50_1D":          ResNet50_1D,
    "InceptionV3_1D":       InceptionV3_1D,
    "MultiTest":            MultiTest,
    "MultiResNet":          MultiResNet,
    "MultiResNetNoPermind": MultiResNetNoPermind,
}

# Layers of _LAYER_MAP that a params.yaml layer list cannot build on 1-D traces (batch, channels, length),
# each with why. The TraceFlow Monitor shows them disabled with this reason, in every layer menu and as the
# model kinds they make (choices.model_forms); tests/unit/test_layer_list_limits.py probes every layer as a
# run builds it, so the list stays what TraceFlow builds. A layer leaves the list when core.model learns to
# build it in a layer list.
_TWO_D = "a 2-D layer: a trace is 1-D (channels x length), read as one unbatched image it fails"
_THREE_D = "a 3-D layer: it takes 4-D or 5-D input, a trace batch is 3-D (batch, channels, length)"
LAYER_LIST_LIMITS = {
    "Conv2d": _TWO_D,
    "Conv3d": _THREE_D,
    "ConvTranspose2d": _TWO_D,
    "ConvTranspose3d": _THREE_D,
    "BatchNorm2d": "a 2-D layer: it takes 4-D input, a trace batch is 3-D (batch, channels, length)",
    "BatchNorm3d": _THREE_D,
    "InstanceNorm3d": _THREE_D,
    "PixelShuffle": "a 2-D layer: it rearranges image channels into height and width; a trace has no height",
    "SPP": "spatial pyramid pooling takes 4-D (image) input; a trace batch is 3-D",
    "Embedding": "it looks up integer indices; a trace holds real-valued samples",
    "RNN": "returns (output, hidden), a tuple: the next layer of the list cannot take it",
    "LSTM": "returns (output, (hidden, cell)), a tuple: the next layer of the list cannot take it",
    "GRU": "returns (output, hidden), a tuple: the next layer of the list cannot take it",
    "MultiheadAttention": "needs query, key and value; a layer list passes one input, and it returns a tuple",
    "TransformerEncoderLayer": "builds only when d_model equals the trace length, and with batch_first=False "
                               "it attends across the batch, not along the trace",
    "TransformerDecoderLayer": "needs the encoder's memory as a second input; a layer list passes one input",
}

_ACTIVATION_MAP = {
    "identity": _identity,
    "linear": _identity,
    "none": _identity,
    "": _identity,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "softmax": _softmax,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "leakyrelu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "elu": nn.ELU,
    "celu": nn.CELU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "swish": lambda **kw: nn.SiLU(**kw),
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
    "logsigmoid": nn.LogSigmoid,
    "logsoftmax": _log_softmax,
    "threshold": nn.Threshold,
    "rrelu": nn.RReLU,
    "hardshrink": nn.Hardshrink,
    "hardtanh": nn.Hardtanh,
    "hardsigmoid": nn.Hardsigmoid,
    "hard_sigmoid": nn.Hardsigmoid,   # PyTorch definition: clip(x/6 + 0.5, 0, 1)
    "hardswish": nn.Hardswish,
    "glu": nn.GLU,
    "softshrink": nn.Softshrink,
    "tanhshrink": nn.Tanhshrink,
    "softmin": _softmin,
    "softmax2d": lambda **kw: nn.Softmax2d(**kw),
    # no "logsoftmax2d": torch has no nn.LogSoftmax2d; use logsoftmax with args {dim: -3}
    # add others as needed
}
# Every official torch.nn activation class resolves by its own name too:
# build_model lower-cases the configured name, so "ReLU6", "LeakyReLU",
# "SiLU", "Hardsigmoid", ... reach the entries above.


def _activation_factory(name: str, where: str):
    """The torch.nn module factory of a configured activation name (framework: pytorch).

    Under framework: keras build_model hands the whole layer list to
    core.framework_switch, which takes the activations from keras.activations.
    """
    factory = _ACTIVATION_MAP.get(name)
    if factory is None:
        raise ValueError(f"Unknown activation {name!r} at {where}")
    return factory

def _activation_module(raw_activation, where: str) -> nn.Module:
    """The torch.nn activation module of a configured activation (a name, or {name, args})."""
    if isinstance(raw_activation, str):
        activation_name = raw_activation.strip().lower()
        activation_args = {}
    elif isinstance(raw_activation, dict):
        activation_name = str(raw_activation.get("name") or "").strip().lower()
        activation_args = dict(raw_activation.get("args") or {})
    else:
        raise ValueError(f"Invalid activation {raw_activation!r} at {where}")

    activation_factory = _activation_factory(activation_name, where)
    activation_kwargs = _filter_kwargs_for_factory(
        activation_factory,
        activation_args,
        strict=True,
        where=f"activation {activation_name!r} at {where}",
    )
    try:
        return activation_factory(**activation_kwargs)
    except Exception as exc:
        raise TypeError(
            f"Error creating activation {activation_name!r} at {where} with "
            f"{activation_kwargs}: {exc}"
        ) from exc


_LEAKAGE_CLASS_COUNTS = {
    "ID": 256,
    "HW": 9,
    "MSB": 2,
}


def _get_concrete_leakage_model(root_cfg: dict) -> str:
    """Return the single leakage model attached to one concrete SCC run."""
    if not isinstance(root_cfg, dict):
        raise TypeError(
            "out_features='auto' requires the complete runtime config as a dict"
        )

    sca_cfg = root_cfg.get("sca") or {}
    if not isinstance(sca_cfg, dict):
        raise TypeError("cfg['sca'] must be a dictionary")

    leakage_model = sca_cfg.get("leakage_model")

    if isinstance(leakage_model, (list, tuple)):
        raise RuntimeError(
            "build_model received an unresolved leakage-model sweep: "
            f"sca.leakage_model={leakage_model!r}. Expected exactly one "
            "concrete value after SCC sweep expansion."
        )

    leakage_key = str(leakage_model or "").strip().upper()
    if leakage_key not in _LEAKAGE_CLASS_COUNTS:
        raise ValueError(
            "Cannot resolve out_features='auto' from "
            f"sca.leakage_model={leakage_model!r}. Supported values: "
            f"{sorted(_LEAKAGE_CLASS_COUNTS)}"
        )

    return leakage_key


def _resolve_output_features(
    value,
    *,
    root_cfg: dict,
    layer_index,
) -> int:
    """Resolve a Linear/LazyLinear output width from YAML."""
    if isinstance(value, bool):
        raise TypeError(
            f"Linear layer {layer_index}: out_features cannot be boolean"
        )

    if isinstance(value, int):
        if value <= 0:
            raise ValueError(
                f"Linear layer {layer_index}: out_features must be positive, "
                f"got {value}"
            )
        return value

    if not isinstance(value, str):
        raise TypeError(
            f"Linear layer {layer_index}: out_features must be a positive "
            f"integer, numeric string, or 'auto'; got {value!r}"
        )

    normalized = value.strip().lower()
    if normalized != "auto":
        try:
            resolved = int(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Linear layer {layer_index}: unsupported "
                f"out_features={value!r}; use a positive integer or 'auto'"
            ) from exc

        if resolved <= 0:
            raise ValueError(
                f"Linear layer {layer_index}: out_features must be positive, "
                f"got {resolved}"
            )
        return resolved

    leakage_key = _get_concrete_leakage_model(root_cfg)
    resolved = _LEAKAGE_CLASS_COUNTS[leakage_key]

    # logger.info(
    #     "[model-build] Resolved out_features='auto': "
    #     "leakage_model=%s -> out_features=%d",
    #     leakage_key,
    #     resolved,
    # )
    return resolved


def _unknown_args_message(unknown, allowed, where: str) -> str:
    """Name every unknown key, the closest allowed key and the allowed list."""
    allowed = sorted(str(k) for k in allowed)
    named = []
    for key in sorted(str(k) for k in unknown):
        near = difflib.get_close_matches(key, allowed, n=1)
        named.append(f"{key!r} (did you mean {near[0]!r}?)" if near else repr(key))
    return (
        f"{where}: unknown argument(s) {', '.join(named)}. "
        f"Allowed: {allowed if allowed else 'none'}. Fix the key in the model config."
    )


def _factory_args(factory) -> set | None:
    """Keyword arguments ``factory`` accepts; None when it takes **kwargs."""
    sig = inspect.signature(factory.__init__ if inspect.isclass(factory) else factory)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return None
    return set(sig.parameters.keys()) - {"self"}


def _filter_kwargs_for_factory(factory, params: dict, strict: bool = False, where: str | None = None) -> dict:
    allowed = _factory_args(factory)
    if allowed is None:
        return dict(params)

    unknown = set(params.keys()) - allowed
    if strict and unknown:
        label = where or getattr(factory, "__name__", str(factory))
        raise TypeError(_unknown_args_message(unknown, allowed, label))

    return {k: v for k, v in params.items() if k in allowed}


class _NodeWithActivation(nn.Module):
    """A graph node followed by its params.activation; passes every input to the node."""

    def __init__(self, module: nn.Module, activation: nn.Module):
        super().__init__()
        self.module = module
        self.activation = activation

    def forward(self, *inputs):
        return self.activation(self.module(*inputs))


def _own_init(module: nn.Module, own: tuple, where: str) -> nn.Module:
    """Mark a Conv/Linear layer with its own (weight, bias) initialisers (framework_switch.layer_init_overrides),
    which apply_torch_init applies over the model's."""
    if any(own):
        if not isinstance(module, _KERNEL_LAYERS):
            raise ValueError(f"{where}: weight_init / bias_init are read on Conv and Linear layers only")
        setattr(module, LAYER_INIT_ATTR, own)
    return module


def _build_from_graph(cfg: dict, *, root_cfg: dict, framework: FrameworkSpec | None = None) -> nn.Module:
    """Build a graph/DAG model and resolve automatic classifier widths."""
    nodes = cfg["nodes"]
    outputs = cfg["outputs"]

    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Graph model requires a non-empty 'nodes' list")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Graph model requires a non-empty 'outputs' list")

    modules = nn.ModuleDict()
    known_names = {"input"}
    strict = bool(cfg.get("strict_graph_args", True))

    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise TypeError(f"Graph node at index {idx} must be a dictionary")

        node_name = str(node.get("name") or "").strip()
        layer_type = str(node.get("type") or "").strip()
        if not node_name:
            raise ValueError(f"Graph node at index {idx} is missing 'name'")
        if node_name in known_names:
            raise ValueError(f"Duplicate graph node name: {node_name!r}")
        if not layer_type:
            raise ValueError(f"Graph node {node_name!r} is missing 'type'")

        factory = _LAYER_MAP.get(layer_type)
        if factory is None:
            raise ValueError(
                f"Unknown layer type {layer_type!r} in node {node_name!r}"
            )

        params = dict(node.get("params") or {})
        raw_activation = params.pop("activation", None)
        node_where = f"graph node {node_name!r} ({layer_type})"
        own = layer_init_overrides(framework, params, node_where) if framework is not None else (None, None)
        if layer_type in {"Linear", "LazyLinear"} and "out_features" in params:
            params["out_features"] = _resolve_output_features(
                params["out_features"],
                root_cfg=root_cfg,
                layer_index=f"graph:{node_name}",
            )

        inputs = node.get("inputs", ["input"])
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(
                f"Graph node {node_name!r} requires a non-empty inputs list"
            )
        missing_inputs = [name for name in inputs if name not in known_names]
        if missing_inputs:
            raise ValueError(
                f"Graph node {node_name!r} references unavailable inputs: "
                f"{missing_inputs}"
            )

        valid_args = _filter_kwargs_for_factory(
            factory, params, strict=strict, where=f"graph node {node_name!r} ({layer_type})"
        )
        try:
            module = factory(**valid_args)
        except Exception as exc:
            raise TypeError(
                f"Error creating graph node {node_name!r} of type "
                f"{layer_type!r} with {valid_args}: {exc}"
            ) from exc
        module = _own_init(module, own, node_where)
        if raw_activation is not None:
            module = _NodeWithActivation(module, _activation_module(raw_activation, f"graph node {node_name!r}"))
        modules[node_name] = module

        known_names.add(node_name)

    missing_outputs = [name for name in outputs if name not in known_names]
    if missing_outputs:
        raise ValueError(
            f"Graph outputs reference unavailable nodes: {missing_outputs}"
        )

    class GraphModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = modules
            self.node_defs = nodes
            self.out_names = outputs

        def forward(self, x):
            tensor_map = {"input": x}
            for node in self.node_defs:
                name = node["name"]
                inputs = node.get("inputs", ["input"])
                if isinstance(inputs, str):
                    inputs = [inputs]
                input_tensors = [tensor_map[input_name] for input_name in inputs]
                tensor_map[name] = (
                    self.layers[name](*input_tensors)
                    if len(input_tensors) > 1
                    else self.layers[name](input_tensors[0])
                )

            result = [tensor_map[name] for name in self.out_names]
            return result[0] if len(result) == 1 else tuple(result)

    return GraphModel()


def _select_model_entry(model_cfg: dict, model_id: str | None = None) -> dict:
    """
    Accept either:
      A) old shape:
         {"layers": [...]} or {"architecture": "...", ...}

      B) new shape:
         {
           "model": [
             {
               "model_id": "CNN1",
               "architecture": "CNN",
               "architecture_args": {...}
             },
             ...
           ]
         }

      C) direct list shape:
         [
           {"model_id": "CNN1", ...},
           {"model_id": "MLP1", ...}
         ]

    Returns the selected single model entry or the original direct config.
    """
    # Old direct single-model config
    if isinstance(model_cfg, dict) and (
        "layers" in model_cfg
        or "nodes" in model_cfg
        or "outputs" in model_cfg
        or "architecture" in model_cfg
    ):
        # But avoid mistaking a root wrapper {"model": [...]} for a direct config
        if "model" not in model_cfg or any(
            k in model_cfg for k in ("layers", "nodes", "outputs", "architecture_args")
        ):
            return model_cfg

    entries = None

    # Root wrapper: {"model": [...]}
    if isinstance(model_cfg, dict) and "model" in model_cfg:
        entries = model_cfg.get("model")

    # Direct list shape
    elif isinstance(model_cfg, list):
        entries = model_cfg

    if entries is None:
        raise TypeError(
            "build_model expected either a single model config dict, "
            "or a root config containing 'model: [...]', or a direct model list."
        )

    if not isinstance(entries, list) or not entries:
        raise ValueError("'model' must be a non-empty list of model entries")

    # If only one model is present, allow implicit selection
    if model_id is None:
        if len(entries) == 1:
            entry = entries[0]
            if not isinstance(entry, dict):
                raise TypeError("Selected model entry must be a dict")
            return entry
        raise ValueError(
            "Multiple model entries found. Please provide model_id to select one."
        )

    wanted = str(model_id).strip()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("model_id", "")).strip() == wanted:
            return entry

    available = [
        str(e.get("model_id", ""))
        for e in entries
        if isinstance(e, dict) and "model_id" in e
    ]
    raise KeyError(f"model_id '{wanted}' not found. Available: {available}")


# architecture_args keys a layer-built model (CNN/MLP/TRANSFORMER/HYBRID) reads.
# search_space is AutoSearch recipe input that the contract resolver strips.
# Any other key, e.g. a misspelled 'weight_int', would be ignored silently.
LAYER_MODEL_ARGS = ("layers", "nodes", "outputs", "strict_graph_args", "weight_init", "bias_init", "search_space")


def _normalize_model_cfg(model_cfg: dict, model_id: str | None = None) -> dict:
    """
    Normalize your chosen model-entry schema into one of the existing builder shapes:
      - {"layers": [...]}
      - {"nodes": [...], "outputs": [...]}
      - {"architecture": "...", "arch_params": {...}}

    Special rule:
      architecture: "CNN" with architecture_args.layers -> treat as sequential YAML
      architecture: "MLP" with architecture_args.layers -> treat as sequential YAML
    """
    entry = _select_model_entry(model_cfg, model_id=model_id)

    if not isinstance(entry, dict):
        raise TypeError("Selected model entry must be a dict")

    # Old-style already-normalized configs still work
    if "layers" in entry or ("nodes" in entry and "outputs" in entry):
        return entry

    arch = entry.get("architecture")
    arch_args = entry.get("architecture_args", {}) or {}

    if arch is None:
        raise ValueError("Selected model entry is missing 'architecture'")

    if not isinstance(arch_args, dict):
        raise TypeError("'architecture_args' must be a dict")

    arch_name = str(arch).strip()
    arch_upper = arch_name.upper()

    # Your YAML-defined CNN/MLP/etc should be treated as config-driven models,
    # not imported Python class names.
    if arch_upper in {"CNN", "MLP", "TRANSFORMER", "HYBRID"}:
        unknown = set(arch_args) - set(LAYER_MODEL_ARGS)
        if unknown:
            raise ValueError(
                _unknown_args_message(
                    unknown,
                    LAYER_MODEL_ARGS,
                    f"model_id={entry.get('model_id')!r} ({arch_name}) architecture_args",
                )
            )
        if "nodes" in arch_args and "outputs" in arch_args:
            return arch_args
        if "layers" in arch_args:
            return arch_args
        raise ValueError(
            f"architecture '{arch_name}' requires 'architecture_args.layers' "
            f"or graph-style 'architecture_args.nodes' + 'architecture_args.outputs'"
        )

    # True named architecture path
    return {
        "architecture": arch_name,
        "arch_params": arch_args,
    }


def _input_length(root_cfg: dict) -> int:
    """Trace length from data.target_range; 700 (the ASCAD window) otherwise."""
    data = root_cfg.get("data") if isinstance(root_cfg, dict) else None
    target_range = None
    if isinstance(data, dict):
        if "target_range" in data:
            target_range = data.get("target_range")
        else:
            first = next(iter(data.values()), {})
            target_range = first.get("target_range") if isinstance(first, dict) else None
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        target_range = data[0].get("target_range")
    if isinstance(target_range, (list, tuple)) and len(target_range) >= 2:
        return int(target_range[1]) - int(target_range[0])
    return 700


def _example_input(model: nn.Module, root_cfg: dict) -> torch.Tensor | None:
    """
    Build the dummy trace batch that materialises lazy layers.

    Only needed when the model holds a LazyLinear/LazyConv. The channel count
    comes from the first Conv1d; a 1-D trace model without one takes 1 channel.
    """
    if not has_lazy_parameters(model):
        return None
    channels = 1
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            channels = int(module.in_channels)
            break
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            raise ValueError(
                "weight_init with lazy layers is supported for 1-D trace models "
                "only; replace the lazy layer with a concrete one"
            )
    return torch.zeros(2, channels, _input_length(root_cfg))


def _apply_init(model: nn.Module, spec: FrameworkSpec, arch_args: dict, *, root_cfg: dict, where: str) -> nn.Module:
    """
    Initialise every Conv/Linear layer with the torch.nn.init functions the
    run names (framework: pytorch).

    The model's architecture_args.weight_init / bias_init override
    deep_learning.pytorch.weight_init / bias_init; "default" keeps PyTorch's
    own reset_parameters() (Kaiming uniform with a=sqrt(5), uniform bias).
    Layer models and named architectures (ResNet50_1D, InceptionV3_1D, ...)
    share this routine; normalisation layers keep their own initialisation.
    Lazy layers are materialised with one dummy forward pass first.
    """
    weight, bias = model_init_specs(spec, arch_args, where)
    touched = apply_torch_init(model, weight, bias, example_input=_example_input(model, root_cfg))
    logger.info(
        "[model-build] %s: weight_init=%s%s bias_init=%s%s (torch.nn.init) on %d Conv/Linear layers",
        where, weight.name, dict(weight.args) or "", bias.name, dict(bias.args) or "", touched,
    )
    return model


def build_model(model_cfg: dict, model_id: str | None = None) -> nn.Module:
    """
    Build a PyTorch model from a complete runtime config or model-only config.

    For ``out_features: auto``, callers must pass the complete concrete runtime
    configuration so the builder can resolve ``sca.leakage_model``:

        ID  -> 256
        HW  -> 9
        MSB -> 2
    """
    if not isinstance(model_cfg, (dict, list)):
        raise TypeError(
            "build_model expected a config dictionary or direct model list, "
            f"got {type(model_cfg).__name__}"
        )

    # Preserve the complete runtime config. Normalization intentionally strips
    # runtime sections and returns only the selected architecture definition.
    root_cfg = model_cfg if isinstance(model_cfg, dict) else {}

    effective_model_id = model_id
    if effective_model_id is None and isinstance(root_cfg, dict):
        configured_model_id = root_cfg.get("model_id")
        if configured_model_id is not None:
            effective_model_id = str(configured_model_id).strip() or None

    leakage_model = (
        (root_cfg.get("sca") or {}).get("leakage_model")
        if isinstance(root_cfg.get("sca"), dict)
        else None
    )

    logger.info(
        "[model-build] Received config: model_id=%s leakage_model=%r "
        "model_hash=%s model_name=%s",
        effective_model_id,
        leakage_model,
        root_cfg.get("model_hash"),
        root_cfg.get("model_name"),
    )

    normalized_cfg = _normalize_model_cfg(
        model_cfg,
        model_id=effective_model_id,
    )
    where = f"model_id={effective_model_id!r}"
    framework = resolve_framework(root_cfg)

    if "architecture" in normalized_cfg:
        architecture = normalized_cfg["architecture"]
        params = dict(normalized_cfg.get("arch_params") or {})
        where = f"{where} ({architecture})"

        try:
            module = __import__(
                "model_architectures." + architecture,
                fromlist=[architecture],
            )
            architecture_class = getattr(module, architecture)
        except (ImportError, AttributeError):
            architecture_class = _LAYER_MAP.get(architecture)

        if architecture_class is None:
            raise ValueError(f"Unknown architecture {architecture!r}")

        if framework.is_keras:
            raise ValueError(
                f"{where} is a PyTorch class (model_architectures/) and has no Keras build; "
                "under deep_learning.framework: keras describe the network in params.yaml as "
                "architecture_args.layers or nodes/outputs (ASCAD_ResNet50 and ASCAD_InceptionV3 are graph models)"
            )
        # weight_init / bias_init are builder settings, not constructor
        # arguments: the class is built with its own init, then re-initialised
        # like a layer model.
        allowed = _factory_args(architecture_class)
        unknown = set(params) - {"weight_init", "bias_init"} - (allowed or set())
        if allowed is not None and unknown:
            raise ValueError(
                _unknown_args_message(
                    unknown,
                    [*allowed, "weight_init", "bias_init"],
                    f"{where} architecture_args",
                )
            )
        init_args = {k: params.pop(k) for k in ("weight_init", "bias_init") if k in params}

        try:
            model = architecture_class(**params)
        except Exception as exc:
            raise TypeError(
                f"Error creating architecture {architecture!r} with "
                f"parameters {params}: {exc}"
            ) from exc
        return _apply_init(model, framework, init_args, root_cfg=root_cfg, where=where)

    if "nodes" in normalized_cfg and "outputs" in normalized_cfg:
        if framework.is_keras:
            weight, bias = model_init_specs(framework, normalized_cfg, where)
            nodes = normalized_cfg["nodes"]
            model = build_keras_graph(
                nodes,
                list(normalized_cfg["outputs"]),
                input_shape=keras_input_shape(nodes, _input_length(root_cfg)),
                weight=weight,
                bias=bias,
                resolve_out_features=lambda value, idx: _resolve_output_features(value, root_cfg=root_cfg, layer_index=idx),
                where=where,
                spec=framework,
            )
            logger.info(
                "[model-build] %s: Keras 3 graph model (torch backend), kernel_initializer=%s%s "
                "bias_initializer=%s%s (keras.initializers)",
                where, weight.name, dict(weight.args) or "", bias.name, dict(bias.args) or "",
            )
            return model
        return _apply_init(
            _build_from_graph(normalized_cfg, root_cfg=root_cfg, framework=framework),
            framework,
            normalized_cfg,
            root_cfg=root_cfg,
            where=where,
        )

    layer_defs = normalized_cfg.get("layers")
    if not isinstance(layer_defs, list) or not layer_defs:
        raise ValueError("Sequential model requires a non-empty 'layers' list")

    if framework.is_keras:
        weight, bias = model_init_specs(framework, normalized_cfg, where)
        model = build_keras_model(
            layer_defs,
            input_shape=keras_input_shape(layer_defs, _input_length(root_cfg)),
            weight=weight,
            bias=bias,
            resolve_out_features=lambda value, idx: _resolve_output_features(value, root_cfg=root_cfg, layer_index=idx),
            where=where,
            spec=framework,
        )
        logger.info(
            "[model-build] %s: Keras 3 model (torch backend), kernel_initializer=%s%s "
            "bias_initializer=%s%s (keras.initializers)",
            where, weight.name, dict(weight.args) or "", bias.name, dict(bias.args) or "",
        )
        return model

    layers = []
    for idx, layer_def in enumerate(layer_defs):
        if not isinstance(layer_def, dict) or len(layer_def) != 1:
            raise ValueError(
                f"Layer at index {idx} must be a single-key dictionary"
            )

        name, raw_params = next(iter(layer_def.items()))
        params = dict(raw_params or {})
        raw_activation = params.pop("activation", None)
        own = layer_init_overrides(framework, params, f"{where} layer {idx} ({name})")

        if name in {"Linear", "LazyLinear"} and "out_features" in params:
            params["out_features"] = _resolve_output_features(
                params["out_features"],
                root_cfg=root_cfg,
                layer_index=idx,
            )

        factory = _LAYER_MAP.get(name)
        if factory is None:
            raise ValueError(f"Unknown layer {name!r} at index {idx}")

        valid_kwargs = _filter_kwargs_for_factory(
            factory, params, strict=True, where=f"layer {name!r} at index {idx}"
        )
        try:
            layer = factory(**valid_kwargs)
        except Exception as exc:
            raise TypeError(
                f"Error creating layer {name!r} at index {idx} with "
                f"{valid_kwargs}: {exc}"
            ) from exc
        layers.append(_own_init(layer, own, f"layer {name!r} at index {idx}"))

        if raw_activation is None:
            continue
        layers.append(_activation_module(raw_activation, f"layer {idx}"))

    return _apply_init(nn.Sequential(*layers), framework, normalized_cfg, root_cfg=root_cfg, where=where)
