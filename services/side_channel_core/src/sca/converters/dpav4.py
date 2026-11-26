# services/side_channel_core/src/sca/converters/dpav4.py
"""
DPA contest v4 as Zaid et al. 2020 distribute it (DPAv4_dataset.zip, gabzai/Methodology-for-
efficient-CNN-architectures-in-SCA) to the ASCAD layout, with each trace's mask byte.

The zip holds profiling (4,500) and attack (500) traces of 4,000 float64 samples, their
plaintexts, key.npy (the fixed key, AES-256 RSM; its first 16 bytes are the first-round key)
and mask.npy (the 16 RSM masks). The attack set has an offset per byte; the label is
Sbox[p[0] ^ k[0]] ^ M[(offset[0] + 1) % 16] (DPA-contest v4/exploit_pred.py:70). The profiling
set has no offsets: its mask is recovered as label ^ Sbox[p[0] ^ k[0]], which must be one of
the 16 RSM masks (owner ruling 2026-10-16: masks as recovered, documented as such). Both sets
are checked: every attack label must equal the formula under its offset's mask. Traces are
stored as float32, the precision the runtime reads them in.

Author: Shraddha Mishra
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

from sca.converters import write_ascad_layout
from utils.aes import AES_SBOX

SOURCE = "DPAv4_dataset.zip"
BYTE = 0


def _load(zipped: zipfile.ZipFile, name: str) -> np.ndarray:
    return np.load(io.BytesIO(zipped.read(f"DPAv4_dataset/{name}.npy")))


def convert(inputs: dict[str, Path], output: Path) -> None:
    with zipfile.ZipFile(inputs[SOURCE]) as zipped:
        arrays = {name: _load(zipped, name) for name in (
            "profiling_traces_dpav4", "profiling_plaintext_dpav4", "profiling_labels_dpav4",
            "attack_traces_dpav4", "attack_plaintext_dpav4", "attack_labels_dpav4", "attack_offset_dpav4",
            "key", "mask")}
    key = np.asarray(arrays["key"][:16], dtype=np.uint8)
    rsm = np.asarray(arrays["mask"], dtype=np.uint8)
    groups = {}
    for group, prefix in (("Profiling_traces", "profiling"), ("Attack_traces", "attack")):
        plaintext = np.asarray(arrays[f"{prefix}_plaintext_dpav4"], dtype=np.uint8)
        labels = np.asarray(arrays[f"{prefix}_labels_dpav4"], dtype=np.uint8).reshape(-1)
        unmasked = AES_SBOX[plaintext[:, BYTE] ^ key[BYTE]]
        if prefix == "attack":
            offset = np.asarray(arrays["attack_offset_dpav4"], dtype=np.int64)[:, BYTE]
            mask = rsm[(offset + 1) % 16]
            if not np.array_equal(unmasked ^ mask, labels):
                raise ValueError("DPAv4 attack labels do not equal Sbox[p0 ^ k0] ^ M[(offset0 + 1) % 16]")
        else:
            mask = unmasked ^ labels                      # recovered from Zaid's labels
            if not np.all(np.isin(mask, rsm)):
                raise ValueError("a recovered DPAv4 profiling mask is not one of the 16 RSM masks")
        n = plaintext.shape[0]
        groups[group] = {"traces": np.asarray(arrays[f"{prefix}_traces_dpav4"], dtype=np.float32),
                         "plaintext": plaintext, "key": np.tile(key, (n, 1)), "mask": mask.astype(np.uint8)}
    write_ascad_layout(output, groups)
