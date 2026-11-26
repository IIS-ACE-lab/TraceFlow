# services/side_channel_core/utils/precision.py
"""Floating-point precision of training and attacks: trainer.precision.

    default   leave PyTorch's defaults. On NVIDIA GPUs of compute capability
              8.0 and later (Ampere, Ada, Blackwell) convolutions then run in
              TF32 (10-bit mantissa); matmuls run in fp32 unless the user
              changed torch's defaults.
    fp32      every float32 operation in full fp32: the arithmetic of a
              2018 Keras/TensorFlow training. Use it for every run compared
              with a published number.
    tf32      TF32 for convolutions and matmuls: fastest on the GPUs above.

The value joins the experiment hash with the rest of trainer. On hardware
without TF32 (CPU, GPUs before Ampere) fp32 and tf32 change nothing, which
apply_precision() reports.

Author: Shraddha Mishra
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from utils.logger import get_logger

logger = get_logger("utils.precision")

PRECISIONS = ("default", "fp32", "tf32")


def resolve_precision(cfg: Mapping[str, Any]) -> str:
    """trainer.precision as one of PRECISIONS; absent means default."""
    trainer = (cfg or {}).get("trainer") or {}
    if not isinstance(trainer, Mapping):
        raise TypeError("trainer must be a mapping")
    raw = trainer.get("precision", "default")
    if raw is None:
        return "default"
    if not isinstance(raw, str):
        raise ValueError(f"trainer.precision must be one resolved string of {PRECISIONS}, got {raw!r}")
    value = raw.strip().lower()
    if value not in PRECISIONS:
        raise ValueError(f"trainer.precision must be one of {PRECISIONS}, got {raw!r}")
    return value


def hardware_has_tf32() -> bool:
    """True on a CUDA device of compute capability 8.0 or later."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability() >= (8, 0)
    except Exception:  # no device, or a driver that cannot answer
        return False


def apply_precision(cfg: Mapping[str, Any]) -> str:
    """Set torch's TF32 switches from trainer.precision; return the resolved value."""
    precision = resolve_precision(cfg)
    if precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    elif precision == "tf32":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if precision != "default":
        logger.info(
            "trainer.precision=%s: cudnn.allow_tf32=%s cuda.matmul.allow_tf32=%s (%s)",
            precision, torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
            "the device has TF32" if hardware_has_tf32() else "no TF32 on this device, so the setting changes nothing",
        )
    return precision
