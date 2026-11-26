# services/side_channel_core/src/model_architectures/public_cnn_1d.py
"""
One-dimensional ports of the two public image models ASCAD benchmarks.

Benadjila et al. (JCEN 2020) compare VGG-16, ResNet-50 and Inception-v3 on the
ASCAD traces (Sect. 3.3.1 Fig. 2, p. 172) and report that only VGG-16 extracts
key-dependent information. Footnote 18 (p. 172) says the three were
"straightforwardly customized to apply on 1-dimensional inputs of 700 units and
outputs of 256 units" and gives no further detail. The paper's code is Keras
2.1.1 (Sect. 2.4.3, p. 168), so the ports follow keras.applications.resnet50
and keras.applications.inception_v3 of that release layer by layer, with one
mapping rule, which is this reproduction's interpretation:

- every Keras layer becomes one 1-D layer of the same kind and arguments; a
  2-D kernel or pool window (a, b) becomes a window of max(a, b) taps, so
  Inception's factorised 1x7 / 7x1 and 1x3 / 3x1 convolutions each keep their
  own 7- or 3-tap layer and every block keeps its published layers and widths;
- strides (s, s) become s; 'valid' is no padding and 'same' is TensorFlow's
  SAME padding (average pooling leaves the padded zeros out of the mean);
- the pooling that feeds the classifier is global average pooling (Inception's
  GlobalAveragePooling2D; ResNet's 7x7 AveragePooling2D, which is global at the
  224-pixel design size), followed by one linear layer of ``num_classes`` units;
- BatchNormalization keeps the Keras defaults: eps 1e-3, momentum 0.99
  (PyTorch momentum 0.01); Inception's has no scale (gamma fixed at 1);
- ResNet-50 is Keras' v1: the stride of a stage sits on the first 1x1
  convolution of its first block, and every convolution has a bias;
- Inception-v3 has no auxiliary classifier, as in keras.applications.

A 700-sample trace reaches ResNet-50's first stage with 174 positions
(700 -> 350 -> 174 -> 87 -> 44 -> 22) and Inception-v3's first block with 85
(700 -> 349 -> 347 -> 347 -> 173 -> 173 -> 171 -> 85 -> 42 -> 20).
Weights are initialised by ``architecture_args.weight_init`` in core/model.py.

Author: Shraddha Mishra
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bn(channels: int, scale: bool = True) -> nn.BatchNorm1d:
    """Keras BatchNormalization defaults; ``scale=False`` fixes gamma at 1."""
    bn = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
    if not scale:
        bn.weight.requires_grad_(False)
    return bn


def _conv_bn(in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0) -> nn.Sequential:
    """Inception's conv2d_bn: convolution without bias, BN without scale, ReLU."""
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
        _bn(out_ch, scale=False),
        nn.ReLU(inplace=True),
    )


def _avg_pool_same() -> nn.AvgPool1d:
    """Keras AveragePooling2D((3, 3), strides=(1, 1), padding='same')."""
    return nn.AvgPool1d(3, stride=1, padding=1, count_include_pad=False)


class _TFSamePad(nn.Module):
    """TensorFlow SAME padding for a strided convolution (more on the right)."""

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        out = -(-length // self.stride)
        total = max((out - 1) * self.stride + self.kernel_size - length, 0)
        return F.pad(x, (total // 2, total - total // 2))


class _Bottleneck(nn.Module):
    """Keras conv_block / identity_block: 1x1 (stride) -> 3x3 -> 1x1, 4x expansion."""

    expansion = 4

    def __init__(self, in_ch: int, width: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        out_ch = width * self.expansion
        self.conv1 = nn.Sequential(nn.Conv1d(in_ch, width, 1, stride=stride), _bn(width), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv1d(width, width, 3, padding=1), _bn(width), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv1d(width, out_ch, 1), _bn(out_ch))
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        return self.relu(out + identity)


class ResNet50_1D(nn.Module):
    """
    ResNet-50 v1 (He et al. 2016; Keras 2.1.1 resnet50) over 1-D traces.

    ASCAD usage: Sect. 3.3.1 Fig. 2 (p. 172), where it fails to extract
    key-dependent information from the masked implementation.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 256, layers: List[int] | None = None):
        super().__init__()
        layers = list(layers or [3, 4, 6, 3])
        self.in_ch = 64

        self.stem = nn.Sequential(
            _TFSamePad(7, 2), nn.Conv1d(in_channels, 64, 7, stride=2), _bn(64), nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool1d(3, stride=2)
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * _Bottleneck.expansion, int(num_classes))

    def _make_layer(self, width: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        out_ch = width * _Bottleneck.expansion
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_ch, out_ch, 1, stride=stride),
                _bn(out_ch),
            )
        blocks_list = [_Bottleneck(self.in_ch, width, stride=stride, downsample=downsample)]
        self.in_ch = out_ch
        blocks_list += [_Bottleneck(self.in_ch, width) for _ in range(1, blocks)]
        return nn.Sequential(*blocks_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.stem(x))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


class _InceptionA(nn.Module):
    """mixed0-2: 1x1 | 1x1 -> 5 | 1x1 -> 3 -> 3 | avgpool -> 1x1."""

    def __init__(self, in_ch: int, pool_features: int):
        super().__init__()
        self.branch1 = _conv_bn(in_ch, 64, 1)
        self.branch5 = nn.Sequential(_conv_bn(in_ch, 48, 1), _conv_bn(48, 64, 5, padding=2))
        self.branch3 = nn.Sequential(
            _conv_bn(in_ch, 64, 1), _conv_bn(64, 96, 3, padding=1), _conv_bn(96, 96, 3, padding=1)
        )
        self.branch_pool = nn.Sequential(_avg_pool_same(), _conv_bn(in_ch, pool_features, 1))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch5(x), self.branch3(x), self.branch_pool(x)], 1)


class _InceptionB(nn.Module):
    """mixed3: 3 stride 2 | 1x1 -> 3 -> 3 stride 2 | maxpool stride 2."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.branch3 = _conv_bn(in_ch, 384, 3, stride=2)
        self.branch3dbl = nn.Sequential(
            _conv_bn(in_ch, 64, 1), _conv_bn(64, 96, 3, padding=1), _conv_bn(96, 96, 3, stride=2)
        )
        self.pool = nn.MaxPool1d(3, stride=2)

    def forward(self, x):
        return torch.cat([self.branch3(x), self.branch3dbl(x), self.pool(x)], 1)


class _InceptionC(nn.Module):
    """mixed4-7: every 1x7 and 7x1 convolution becomes one 7-tap convolution."""

    def __init__(self, in_ch: int, channels_7: int):
        super().__init__()
        c7 = channels_7
        self.branch1 = _conv_bn(in_ch, 192, 1)
        self.branch7 = nn.Sequential(                        # 1x1, 1x7, 7x1
            _conv_bn(in_ch, c7, 1), _conv_bn(c7, c7, 7, padding=3), _conv_bn(c7, 192, 7, padding=3)
        )
        self.branch7dbl = nn.Sequential(                     # 1x1, 7x1, 1x7, 7x1, 1x7
            _conv_bn(in_ch, c7, 1),
            _conv_bn(c7, c7, 7, padding=3),
            _conv_bn(c7, c7, 7, padding=3),
            _conv_bn(c7, c7, 7, padding=3),
            _conv_bn(c7, 192, 7, padding=3),
        )
        self.branch_pool = nn.Sequential(_avg_pool_same(), _conv_bn(in_ch, 192, 1))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch7(x), self.branch7dbl(x), self.branch_pool(x)], 1)


class _InceptionD(nn.Module):
    """mixed8: 1x1 -> 3 stride 2 | 1x1 -> 7 -> 7 -> 3 stride 2 | maxpool stride 2."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.branch3 = nn.Sequential(_conv_bn(in_ch, 192, 1), _conv_bn(192, 320, 3, stride=2))
        self.branch7 = nn.Sequential(                        # 1x1, 1x7, 7x1, 3x3 stride 2
            _conv_bn(in_ch, 192, 1),
            _conv_bn(192, 192, 7, padding=3),
            _conv_bn(192, 192, 7, padding=3),
            _conv_bn(192, 192, 3, stride=2),
        )
        self.pool = nn.MaxPool1d(3, stride=2)

    def forward(self, x):
        return torch.cat([self.branch3(x), self.branch7(x), self.pool(x)], 1)


class _InceptionE(nn.Module):
    """mixed9-10: the parallel 1x3 / 3x1 pairs become two parallel 3-tap convolutions."""

    #: 320 + 2 * 384 + 2 * 384 + 192, as in the 2-D model.
    out_channels = 2048

    def __init__(self, in_ch: int):
        super().__init__()
        self.branch1 = _conv_bn(in_ch, 320, 1)
        self.branch3_1 = _conv_bn(in_ch, 384, 1)
        self.branch3_2a = _conv_bn(384, 384, 3, padding=1)   # 1x3
        self.branch3_2b = _conv_bn(384, 384, 3, padding=1)   # 3x1
        self.branch3dbl_1 = nn.Sequential(_conv_bn(in_ch, 448, 1), _conv_bn(448, 384, 3, padding=1))
        self.branch3dbl_2a = _conv_bn(384, 384, 3, padding=1)  # 1x3
        self.branch3dbl_2b = _conv_bn(384, 384, 3, padding=1)  # 3x1
        self.branch_pool = nn.Sequential(_avg_pool_same(), _conv_bn(in_ch, 192, 1))

    def forward(self, x):
        b3 = self.branch3_1(x)
        bd = self.branch3dbl_1(x)
        return torch.cat(
            [
                self.branch1(x),
                self.branch3_2a(b3), self.branch3_2b(b3),
                self.branch3dbl_2a(bd), self.branch3dbl_2b(bd),
                self.branch_pool(x),
            ],
            1,
        )


class InceptionV3_1D(nn.Module):
    """
    Inception-v3 (Szegedy et al. 2016; Keras 2.1.1 inception_v3) over 1-D traces.

    ASCAD usage: Sect. 3.3.1 Fig. 2 (p. 172), where it fails like ResNet-50.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            _conv_bn(in_channels, 32, 3, stride=2),
            _conv_bn(32, 32, 3),
            _conv_bn(32, 64, 3, padding=1),
            nn.MaxPool1d(3, stride=2),
            _conv_bn(64, 80, 1),
            _conv_bn(80, 192, 3),
            nn.MaxPool1d(3, stride=2),
        )
        blocks = (
            _InceptionA(192, pool_features=32),
            _InceptionA(256, pool_features=64),
            _InceptionA(288, pool_features=64),
            _InceptionB(288),
            _InceptionC(768, channels_7=128),
            _InceptionC(768, channels_7=160),
            _InceptionC(768, channels_7=160),
            _InceptionC(768, channels_7=192),
            _InceptionD(768),
            _InceptionE(1280),
            _InceptionE(_InceptionE.out_channels),
        )
        self.blocks = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(_InceptionE.out_channels, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x))
        return self.fc(torch.flatten(self.avgpool(x), 1))


__all__ = ["InceptionV3_1D", "ResNet50_1D"]
