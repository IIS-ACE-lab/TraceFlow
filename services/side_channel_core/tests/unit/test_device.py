# services/side_channel_core/tests/unit/test_device.py
"""A requested GPU that cannot be used fails the run ("GPU FAILED"), never a CPU fallback (2026-10-15)."""

import pytest
import torch

from utils import device as dv


def _model(cfg, model_id=None):
    return torch.nn.Linear(2, 2)


def test_cuda_unavailable_fails_with_gpu_failed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(dv.GPUFailedError, match="^GPU FAILED: device cuda requested"):
        dv.initialize_device_and_model({"device": "cuda"}, _model, logger=None)


def test_a_model_that_cannot_move_to_the_gpu_fails_with_gpu_failed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class Stuck(torch.nn.Linear):
        def to(self, *a, **k):
            raise RuntimeError("CUDA error: no CUDA-capable device is detected")

    with pytest.raises(dv.GPUFailedError, match="^GPU FAILED: moving the model"):
        dv.initialize_device_and_model({"device": "cuda"}, lambda cfg, model_id=None: Stuck(2, 2), logger=None)


def test_cpu_requested_on_purpose_still_trains_on_the_cpu():
    class Log:
        def info(self, *a):
            pass

    model, device = dv.initialize_device_and_model({"device": "cpu"}, _model, logger=Log())
    assert device.type == "cpu" and next(model.parameters()).device.type == "cpu"


def test_the_status_reason_names_the_gpu_failure_through_wrapping_errors(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        try:
            dv.require_device({"device": "cuda"})
        except dv.GPUFailedError as inner:
            raise RuntimeError("fold 1 failed") from inner
    except RuntimeError as outer:
        assert dv.failure_reason(outer).startswith("GPU FAILED: device cuda requested")
    assert dv.failure_reason(ValueError("bad")) == "ValueError: bad"
