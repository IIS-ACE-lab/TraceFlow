# services/side_channel_core/src/sca/converters/__init__.py
"""
Converters: one module per dataset family, each with convert(inputs, output) that writes the
ASCAD layout the runtime reads (Profiling_traces / Attack_traces, each with `traces` and a
structured `metadata` holding plaintext, key and, where the source has it, ciphertext).
Called once per machine by utils/dataset_fetch.ensure(); registered in src/sca/dataset_types.py.

Author: Shraddha Mishra
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def metadata_dtype(*, ciphertext: bool, mask: bool = False) -> np.dtype:
    fields = [("plaintext", np.uint8, (16,)), ("key", np.uint8, (16,))]
    if ciphertext:
        fields.append(("ciphertext", np.uint8, (16,)))
    if mask:
        fields.append(("mask", np.uint8))      # the mask byte on the attacked byte (target sbox_masked)
    return np.dtype(fields)


def write_ascad_layout(output: Path, groups: dict[str, dict[str, np.ndarray]]) -> None:
    """groups: {"Profiling_traces": {"traces", "plaintext", "key"[, "ciphertext"]}, "Attack_traces": {...}}."""
    with h5py.File(output, "w") as handle:
        for group_name, arrays in groups.items():
            group = handle.create_group(group_name)
            group.create_dataset("traces", data=arrays["traces"])
            dtype = metadata_dtype(ciphertext="ciphertext" in arrays, mask="mask" in arrays)
            metadata = np.zeros(arrays["traces"].shape[0], dtype=dtype)
            for field in dtype.names:
                metadata[field] = arrays[field]
            group.create_dataset("metadata", data=metadata)
