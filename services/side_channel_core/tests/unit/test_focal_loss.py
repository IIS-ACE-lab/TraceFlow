# services/side_channel_core/tests/unit/test_focal_loss.py
"""trainer.loss.type: FocalLoss is torchvision.ops.sigmoid_focal_loss on the one-hot targets (utils/loss_util.py)."""

import pytest
import torch
from torchvision.ops import sigmoid_focal_loss

from utils.loss_util import build_loss_from_config, resolve_loss_config


def _trainer(**args):
    return {"loss": {"type": "FocalLoss", "args": args}}


def _batch():
    torch.manual_seed(0)
    return torch.randn(16, 256, requires_grad=True), torch.randint(0, 256, (16,))


def test_focal_loss_is_torchvisions_on_the_one_hot_labels():
    logits, labels = _batch()
    one_hot = torch.nn.functional.one_hot(labels, 256).float()
    for args, kw in (({}, {"alpha": 0.25, "gamma": 2.0}), ({"alpha": 0.5, "gamma": 1, "reduction": "sum"},
                                                          {"alpha": 0.5, "gamma": 1.0, "reduction": "sum"})):
        expected = sigmoid_focal_loss(logits, one_hot, **{"reduction": "mean", **kw})
        assert torch.equal(build_loss_from_config(_trainer(**args))(logits, labels), expected)


def test_gamma_zero_without_alpha_is_pytorchs_binary_cross_entropy_of_the_logits():
    logits, labels = _batch()
    one_hot = torch.nn.functional.one_hot(labels, 256).float()
    focal = build_loss_from_config(_trainer(alpha=-1, gamma=0))(logits, labels)
    assert torch.allclose(focal, torch.nn.functional.binary_cross_entropy_with_logits(logits, one_hot))


def test_focal_loss_trains():
    logits, labels = _batch()
    build_loss_from_config(_trainer())(logits, labels).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


@pytest.mark.parametrize("args, error, message", [
    ({"alpha": 2.0}, ValueError, "alpha must lie in"),
    ({"gamma": -1.0}, ValueError, "gamma must be >= 0"),
    ({"gamma": "2"}, TypeError, "gamma must be a resolved number"),
    ({"label_smoothing": 0.1}, ValueError, "Unsupported FocalLoss arguments"),
])
def test_focal_loss_refuses_bad_arguments(args, error, message):
    with pytest.raises(error, match=message):
        resolve_loss_config(_trainer(**args))


def test_focal_loss_is_pytorchs_alone():
    from core.framework_switch import build_loss

    with pytest.raises(ValueError, match="has no Keras counterpart"):
        build_loss({"deep_learning": {"framework": "keras"}, "trainer": _trainer()})
