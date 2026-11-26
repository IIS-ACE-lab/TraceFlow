# services/side_channel_core/utils/device.py
'''
This module determines the compute device from cfg["device"], builds the model
and moves it to that device. A run never falls back to the CPU: when the
requested CUDA device cannot be used, it fails with GPUFailedError, whose
message (the run's status_reason) starts with "GPU FAILED". A benchmark of GPU
settings trained on a CPU is not that benchmark; set device: cpu to train on
the CPU on purpose.


Author: Shraddha Mishra
'''

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Tuple

import torch

GPU_FAILED = "GPU FAILED"


class GPUFailedError(RuntimeError):
    """cfg['device'] asks for CUDA and this process cannot use it."""

    states_its_reason = True   # its message is the run's status_reason (failure_reason)


def requested_device(cfg: Mapping[str, Any]) -> torch.device:
    """cfg['device'] as a torch.device (default cuda)."""
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")
    requested_raw = cfg.get("device", "cuda")
    if not isinstance(requested_raw, str) or not requested_raw.strip():
        raise TypeError(
            "cfg['device'] must be a non-empty device string, "
            f"got {requested_raw!r}"
        )
    try:
        return torch.device(requested_raw.strip())
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Invalid cfg['device'] value: {requested_raw!r}"
        ) from exc


def require_device(cfg: Mapping[str, Any]) -> torch.device:
    """The requested device, or GPUFailedError when it is CUDA and CUDA is not available."""
    device = requested_device(cfg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise GPUFailedError(
            f"{GPU_FAILED}: device {device} requested, but CUDA is not available in this "
            "process; not falling back to the CPU. Check nvidia-smi inside the container "
            "(a systemd reload on the host, e.g. by an unattended upgrade, revokes a running "
            "container's GPU access: restart the container)."
        )
    return device


def failure_reason(exc: BaseException) -> str:
    """status_reason of a failed run: an error that states its own reason anywhere in the
    exception chain (GPUFailedError "GPU FAILED: ...", utils/dataset_fetch.DataFetchError
    "DATA FAILED: ...") gives its message, anything else "<type>: <message>"."""
    seen, current = set(), exc
    while current is not None and id(current) not in seen:
        if getattr(current, "states_its_reason", False):
            return str(current)[:500]
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return f"{type(exc).__name__}: {exc}"[:500]


def initialize_device_and_model(
    cfg: Mapping[str, Any],
    build_model: Callable[..., torch.nn.Module],
    logger,
    model_id: Optional[str] = None,
) -> Tuple[torch.nn.Module, torch.device]:
    """Build one model and move it to the requested device.

    CUDA that is unavailable, or a model that cannot be moved to it, raises
    GPUFailedError: never a fallback to the CPU. Model-construction failures
    are never disguised as device failures.
    """
    device = require_device(cfg)

    # Build exactly once. A model-contract or architecture RuntimeError must
    # propagate unchanged instead of being mislabeled as a device problem.
    model = build_model(cfg, model_id=model_id)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "build_model must return torch.nn.Module, got "
            f"{type(model).__name__}"
        )

    try:
        model.to(device)
    except RuntimeError as exc:
        if device.type == "cpu":
            raise
        raise GPUFailedError(
            f"{GPU_FAILED}: moving the model to {device} failed ({exc}); not falling back to the CPU."
        ) from exc

    logger.info("Using device: %s", device)
    return model, device
