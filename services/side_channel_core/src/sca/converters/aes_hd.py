# services/side_channel_core/src/sca/converters/aes_hd.py
"""
AES_HD to the ASCAD layout: the official traces and labels (AESHD/AES_HD_Dataset, commit 8baa486)
with the two ciphertext bytes the last-round target needs from Zaid et al. 2020's copy
(AES_HD_dataset.zip), the owner's ruling of 2026-10-16.

The official release holds 100,000 traces of 1,250 integer samples (traces_1..5.csv, 20,000
each, space-separated) and one label per trace (labels.csv), but no ciphertext and no key, so
no key can be ranked from it alone. Zaid et al. use official traces 0-49,999 as profiling and
50,000-74,999 as attack, and add per-trace ciphertext columns of which only bytes 7 and 11 are
filled; their attack ranks key byte 0x00 against InvSbox[c[11] ^ k] ^ c[7]. Here the traces and
labels are the official ones, c[7] and c[11] come from Zaid's arrays row by row, and the key is
0x00 (their convention: the real last-round key is not published). Every official label must
equal InvSbox[c[11]] ^ c[7], so the two sources are checked against each other. Plaintexts are
not published: the field is zero.

Zaid's c[7] and c[11] are not measured ciphertext but a re-encoding of the label: for row i of
either group, c[11] = Sbox[i // 256] and c[7] = label ^ (i // 256). c[11] is constant over each
block of 256 rows, so every wrong key's hypothesis there is the true label XOR one constant. The
run settings this cannot support (MSB, unshuffled attacks) are refused in
sca/dataset_types.require_target_settings.

Author: Shraddha Mishra
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from sca.converters import write_ascad_layout
from utils.aes import AES_INV_SBOX

TRACES = [f"traces_{i}.csv" for i in range(1, 6)]
LABELS = "labels.csv"
ZAID = "AES_HD_dataset.zip"
PROFILING, ATTACK = 50_000, 25_000


def _read_csv(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=np.int64, engine="c").to_numpy()


def convert(inputs: dict[str, Path], output: Path) -> None:
    traces = np.concatenate([_read_csv(inputs[name]) for name in TRACES])[: PROFILING + ATTACK]
    labels = _read_csv(inputs[LABELS]).reshape(-1)[: PROFILING + ATTACK]
    if traces.min() < 0 or traces.max() > 255:
        raise ValueError("AES_HD samples are expected to be bytes")
    with zipfile.ZipFile(inputs[ZAID]) as zipped:
        ciphertext = np.concatenate([
            np.load(io.BytesIO(zipped.read(f"AES_HD_dataset/{name}_ciphertext_AES_HD.npy")))
            for name in ("profiling", "attack")])
    ciphertext = np.asarray(ciphertext, dtype=np.uint8)
    kept = np.zeros_like(ciphertext)
    kept[:, [7, 11]] = ciphertext[:, [7, 11]]            # the only filled bytes
    if not np.array_equal(AES_INV_SBOX[kept[:, 11]] ^ kept[:, 7], labels.astype(np.uint8)):
        raise ValueError("official AES_HD labels do not equal InvSbox[c[11]] ^ c[7] of Zaid's arrays")
    zeros = np.zeros((PROFILING + ATTACK, 16), np.uint8)
    traces = traces.astype(np.uint8)
    write_ascad_layout(output, {
        "Profiling_traces": {"traces": traces[:PROFILING], "plaintext": zeros[:PROFILING],
                             "key": zeros[:PROFILING], "ciphertext": kept[:PROFILING]},
        "Attack_traces": {"traces": traces[PROFILING:], "plaintext": zeros[PROFILING:],
                          "key": zeros[PROFILING:], "ciphertext": kept[PROFILING:]},
    })
