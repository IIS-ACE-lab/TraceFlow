"""trainer.loss.args.from_logits under deep_learning.framework: keras (core/framework_switch.py)."""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import pytest
import torch
import torch.nn.functional as F

from core.framework_switch import build_loss
from utils.loss_util import build_loss_from_config, resolve_loss_config


def _cfg(framework="keras", **args):
    return {"deep_learning": {"framework": framework},
            "trainer": {"loss": {"type": "CrossEntropyLoss", "args": {"reduction": "mean", **args}}}}


def _keras_2_1_1(logits, targets, eps=1e-7):
    """Keras 2.1.1 categorical_crossentropy(from_logits=False) on softmax(logits), as written there."""
    p = torch.softmax(logits, dim=-1)
    p = p / p.sum(-1, keepdim=True)
    p = torch.clamp(p, eps, 1 - eps)
    return -(F.one_hot(targets, logits.shape[1]).to(p.dtype) * torch.log(p)).sum(-1).mean()


def test_from_logits_false_is_the_clipped_cross_entropy_of_the_softmax():
    logits = torch.randn(64, 9, dtype=torch.float32) * 6
    logits[0, 0] = -40.0                                   # trace 0: its true class far below 1e-7
    targets = torch.randint(0, 9, (64,)); targets[0] = 0
    loss = build_loss(_cfg(from_logits=False))
    assert torch.allclose(loss(logits, targets), _keras_2_1_1(logits, targets), atol=1e-5)


def test_a_trace_below_epsilon_gets_no_gradient():
    logits = torch.zeros(2, 9, dtype=torch.float32); logits[0, 0] = -40.0
    logits.requires_grad_(True)
    targets = torch.tensor([0, 1])
    build_loss(_cfg(from_logits=False))(logits, targets).backward()
    assert float(logits.grad[0].abs().max()) == 0.0      # clipped: dead
    assert float(logits.grad[1].abs().max()) > 0.0
    logits.grad = None
    build_loss(_cfg())(logits, targets).backward()          # default from_logits: true
    assert float(logits.grad[0].abs().max()) > 0.0


def test_default_is_unchanged_and_pytorch_refuses_the_clipped_form():
    logits = torch.randn(16, 9, dtype=torch.float32); targets = torch.randint(0, 9, (16,))
    assert torch.allclose(build_loss(_cfg())(logits, targets), F.cross_entropy(logits, targets), atol=1e-5)
    assert torch.allclose(build_loss(_cfg(from_logits=True))(logits, targets), F.cross_entropy(logits, targets), atol=1e-5)
    with pytest.raises(ValueError, match="framework: keras"):
        build_loss_from_config(_cfg("pytorch", from_logits=False)["trainer"])
    build_loss_from_config(_cfg("pytorch", from_logits=True)["trainer"])         # true is PyTorch's own form
    with pytest.raises(TypeError):
        resolve_loss_config(_cfg(from_logits="no")["trainer"])
