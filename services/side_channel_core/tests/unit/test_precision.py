# services/side_channel_core/tests/unit/test_precision.py
"""trainer.precision: fp32 | tf32 | default.

PyTorch runs convolutions (and, when allowed, matmuls) in TF32 on NVIDIA
GPUs of compute capability 8.0 and later. fp32 switches that off. It changes
what is trained, so it joins the experiment hash, like the loss type.
(The framework's own functions, deep_learning.framework, are tested in
test_framework_switch.py.)
"""
from __future__ import annotations

import copy

import pytest
import torch

from utils.model_naming import make_experiment_hash
from utils.precision import apply_precision, resolve_precision
from utils.type_coerce import coerce_trainer


# ── trainer.precision ──────────────────────────────────────────────────────
@pytest.fixture
def restore_backend_flags():
    saved = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
             torch.get_float32_matmul_precision())
    yield
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved[0], saved[1]
    torch.set_float32_matmul_precision(saved[2])


def test_precision_fp32_turns_tf32_off_everywhere(restore_backend_flags):
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    assert apply_precision({"trainer": {"precision": "fp32"}}) == "fp32"
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"


def test_precision_tf32_turns_tf32_on_everywhere(restore_backend_flags):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    assert apply_precision({"trainer": {"precision": "tf32"}}) == "tf32"
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cuda.matmul.allow_tf32 is True


def test_precision_default_and_absent_leave_the_backend_alone(restore_backend_flags):
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    assert apply_precision({"trainer": {"precision": "default"}}) == "default"
    assert apply_precision({"trainer": {}}) == "default"
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_precision_rejects_unknown_values():
    with pytest.raises(ValueError):
        resolve_precision({"trainer": {"precision": "fp16"}})
    with pytest.raises(ValueError):
        resolve_precision({"trainer": {"precision": ["fp32", "tf32"]}})


def test_coerce_trainer_accepts_precision_and_rejects_bad_values():
    trainer = {"epochs": 1, "batch_size": 1, "precision": " FP32 "}
    coerce_trainer(trainer)
    assert trainer["precision"] == "fp32"
    with pytest.raises(ValueError):
        coerce_trainer({"epochs": 1, "batch_size": 1, "precision": "bf16"})


# ── experiment identity ───────────────────────────────────────────────────
def _identity_cfg() -> dict:
    return {
        "seed": 42,
        "project": {"name": "P", "root": "P", "scripts": {}},
        "data": {"name": "ascad", "database": "data/ascad.h5"},
        "model": [{"model_id": "m", "architecture": "CNN", "architecture_args": {"layers": []}}],
        "trainer": {"epochs": 1, "batch_size": 1,
                    "loss": {"type": "CrossEntropyLoss", "args": {"reduction": "mean"}},
                    "optimizer": {"type": "RMSprop", "learning_rate": 1e-5}},
        "sca": {"variant": "fixed_key", "attack_byte": 2, "leakage_model": "ID"},
        "metrics_to_log": ["mean_attack_rank_curve"],
    }


def test_loss_type_and_precision_change_the_experiment_hash():
    base = _identity_cfg()
    reference = make_experiment_hash(copy.deepcopy(base))

    nll = copy.deepcopy(base)
    nll["trainer"]["loss"] = {"type": "NLLLoss", "args": {"reduction": "mean"}}
    assert make_experiment_hash(nll) != reference

    fp32 = copy.deepcopy(base)
    fp32["trainer"]["precision"] = "fp32"
    assert make_experiment_hash(fp32) != reference
    assert make_experiment_hash(copy.deepcopy(fp32)) == make_experiment_hash(fp32)
