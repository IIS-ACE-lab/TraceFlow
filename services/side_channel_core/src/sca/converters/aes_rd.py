# services/side_channel_core/src/sca/converters/aes_rd.py
"""
AES_RD (ikizhvatov/randomdelays-traces, ctraces_fm16x4_2.mat) to the ASCAD layout.

The MATLAB v5 file holds CompressedTraces (3500, 50000) uint8, samples by traces, and
plaintext (50000, 16) uint8. The key is not in the file; the repository README gives it
(2b7e151628aed2a6abf7158809cf4f3c). The split is Zaid et al. 2020's, checked against their
published arrays: the first 25,000 traces profiling, the last 25,000 attack.

Author: Shraddha Mishra
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sca.converters import write_ascad_layout

SOURCE = "ctraces_fm16x4_2.mat"
KEY = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
PROFILING = 25_000


def convert(inputs: dict[str, Path], output: Path) -> None:
    from scipy.io import loadmat

    mat = loadmat(inputs[SOURCE])
    traces = np.ascontiguousarray(np.asarray(mat["CompressedTraces"], dtype=np.uint8).T)
    plaintext = np.asarray(mat["plaintext"], dtype=np.uint8)
    if traces.shape[0] != plaintext.shape[0] or plaintext.shape[1] != 16:
        raise ValueError(f"AES_RD traces {traces.shape} and plaintexts {plaintext.shape} do not agree")
    key = np.tile(np.frombuffer(KEY, dtype=np.uint8), (traces.shape[0], 1))
    write_ascad_layout(output, {
        "Profiling_traces": {"traces": traces[:PROFILING], "plaintext": plaintext[:PROFILING], "key": key[:PROFILING]},
        "Attack_traces": {"traces": traces[PROFILING:], "plaintext": plaintext[PROFILING:], "key": key[PROFILING:]},
    })
