#services/side_channel_core/utils/loss_util.py
"""Losses for single-label SCA classification (PyTorch framework).

Supported options (case-insensitive): CrossEntropyLoss, NLLLoss, FocalLoss.
Every criterion accepts raw logits [batch, classes] and integer targets
[batch]; NLLLoss is composed with torch.nn.LogSoftmax internally. This
preserves one model-output contract across training and attacks.

Supported arguments: reduction ('mean' or 'sum'); CrossEntropyLoss also
supports label_smoothing. Class weights and ignored labels are not exposed.

FocalLoss is torchvision.ops.sigmoid_focal_loss (Lin et al. 2017, RetinaNet),
the official implementation, called with the one-hot targets of the labels:
a one-vs-all sigmoid focal loss over the classes, not a softmax one.

    alpha   weight of the true class against the others, in [0, 1], or -1
            for none (torchvision's default 0.25)
    gamma   focusing exponent of (1 - p_t), >= 0 (default 2; 0 gives the
            alpha-weighted binary cross-entropy)

'mean' averages over every trace x class element, as torchvision does. The
attacks still rank keys from the log-softmax of the logits: a trace's class
order is the model's, but the logits are per-class sigmoid scores, not a
calibrated softmax. PyTorch only (core/framework_switch.KERAS_LOSSES).

Under deep_learning.framework: keras the same trainer.loss block
(CrossEntropyLoss, reduction, label_smoothing) builds
keras.losses.CategoricalCrossentropy instead (core/framework_switch.py);
resolve_loss_config() validates the block for both frameworks.

trainer.loss.args.from_logits (CrossEntropyLoss, Keras's own argument):

    true    default; current runs and hashes unchanged. The cross-entropy
            from the logits, no clipping.
    false   the paper's loss: the softmax probabilities clipped to
            [1e-7, 1 - 1e-7] (keras.config.epsilon()) before the logarithm,
            as Keras 2.1.1 did; Keras 3 does the same on the torch backend.

    It is accepted only under deep_learning.framework: keras; PyTorch has no
    such loss (build_loss_from_config rejects false under pytorch). The other
    defaults stay reduction: mean and label_smoothing: 0.0. The cross_entropy
    metric follows the same setting (framework_switch.cross_entropy_per_trace).
    tests/unit/test_keras_clipped_loss.py confirms that false equals Keras
    2.1.1's formula and that a trace below the clip gets no gradient.

Author: Shraddha Mishra
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torchvision.ops import sigmoid_focal_loss


LOSS_REGISTRY = {
    "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
    "NLLLoss": torch.nn.NLLLoss,
    "FocalLoss": sigmoid_focal_loss,
}


class _FocalFromLabels(torch.nn.Module):
    """torchvision.ops.sigmoid_focal_loss on the logits and the one-hot targets of the integer labels."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits, targets):
        one_hot = torch.nn.functional.one_hot(targets.long(), logits.shape[1]).to(logits.dtype)
        return sigmoid_focal_loss(logits, one_hot, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)


class _NLLFromLogits(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.criterion = criterion

    def forward(self, logits, targets):
        return self.criterion(self.log_softmax(logits), targets)


def resolve_loss_config(trainer_cfg: Mapping) -> tuple[str, dict]:
    """Validate trainer.loss: {type, args}; return (canonical type, args)."""
    if not isinstance(trainer_cfg, Mapping):
        raise TypeError("trainer must be a mapping")
    if "loss_reduction" in trainer_cfg:
        raise ValueError("Move trainer.loss_reduction to trainer.loss.args.reduction")
    cfg = trainer_cfg.get("loss")
    if not isinstance(cfg, Mapping) or set(cfg) != {"type", "args"}:
        raise ValueError("trainer.loss must contain exactly type and args")
    name = cfg["type"]
    if not isinstance(name, str):
        raise TypeError("trainer.loss.type must be a resolved string")
    canonical = next((key for key in LOSS_REGISTRY if key.casefold() == name.strip().casefold()), None)
    if canonical is None:
        raise ValueError(f"Unsupported loss {name!r}; supported: {tuple(LOSS_REGISTRY)}")
    if not isinstance(cfg["args"], Mapping):
        raise TypeError("trainer.loss.args must be a mapping")
    args = dict(cfg["args"])
    allowed = {"reduction"}
    if canonical == "CrossEntropyLoss":
        allowed |= {"label_smoothing", "from_logits"}
    if canonical == "FocalLoss":
        allowed |= {"alpha", "gamma"}
    if set(args) - allowed:
        raise ValueError(f"Unsupported {canonical} arguments: {set(args) - allowed}")
    reduction = args.get("reduction", "mean")
    if reduction not in ("mean", "sum"):
        raise ValueError("loss reduction must be mean or sum for scalar backward()")
    args["reduction"] = reduction
    if "label_smoothing" in args:
        value = args["label_smoothing"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("label_smoothing must be a resolved number")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("label_smoothing must lie in [0, 1]")
        args["label_smoothing"] = float(value)
    for key in ("alpha", "gamma"):
        if key in args and (isinstance(args[key], bool) or not isinstance(args[key], (int, float))
                            or not math.isfinite(args[key])):
            raise TypeError(f"trainer.loss.args.{key} must be a resolved number")
    if "alpha" in args and args["alpha"] != -1 and not 0 <= args["alpha"] <= 1:
        raise ValueError("trainer.loss.args.alpha must lie in [0, 1], or be -1 for no weighting")
    if "gamma" in args and args["gamma"] < 0:
        raise ValueError("trainer.loss.args.gamma must be >= 0")
    if "from_logits" in args and not isinstance(args["from_logits"], bool):
        raise TypeError("trainer.loss.args.from_logits must be true or false")
    return canonical, args


def build_loss_from_config(trainer_cfg: Mapping) -> torch.nn.Module:
    """Build the torch.nn loss from trainer.loss: {type: CrossEntropyLoss, args: {...}}.

    Configuration must already be resolved to scalar settings. The returned
    scalar criterion can be called as criterion(logits, targets).
    """
    canonical, args = resolve_loss_config(trainer_cfg)
    if not args.pop("from_logits", True):
        raise ValueError(
            "trainer.loss.args.from_logits: false (the clipped cross-entropy of softmax "
            "probabilities) is keras.losses.CategoricalCrossentropy's; set "
            "deep_learning.framework: keras, or drop from_logits (PyTorch has no such loss)"
        )
    if canonical == "FocalLoss":
        return _FocalFromLabels(**{k: v if k == "reduction" else float(v) for k, v in args.items()})
    criterion = LOSS_REGISTRY[canonical](**args)
    if canonical == "NLLLoss":
        return _NLLFromLogits(criterion)
    return criterion
