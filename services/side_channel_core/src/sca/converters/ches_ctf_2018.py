# services/side_channel_core/src/sca/converters/ches_ctf_2018.py
"""
CHES CTF 2018, AISyLab copy (ches_ctf.h5, AISyLab/EnsembleSCA) to the ASCAD layout.

The source holds four flat datasets: profiling_traces (45000, 2200) and attacking_traces
(5000, 2200), float32 z-scored by the redistributor, and profiling_data / attacking_data
(N, 48) uint8 whose columns are the plaintext (0-15), the ciphertext (16-31) and the key
(32-47), as the AISyLab loader reads them (commons/load_datasets.py). The traces are kept as
they are; the trace normalisation of the data entry decides any further scaling.

Author: Shraddha Mishra
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from sca.converters import write_ascad_layout

SOURCE = "ches_ctf.h5"


def _group(traces, data) -> dict[str, np.ndarray]:
    data = np.asarray(data, dtype=np.uint8)
    if data.ndim != 2 or data.shape[1] != 48:
        raise ValueError(f"CHES CTF 2018 data rows must have 48 bytes, got {data.shape}")
    return {"traces": np.asarray(traces), "plaintext": data[:, 0:16], "ciphertext": data[:, 16:32], "key": data[:, 32:48]}


def convert(inputs: dict[str, Path], output: Path) -> None:
    with h5py.File(inputs[SOURCE], "r") as source:
        groups = {
            "Profiling_traces": _group(source["profiling_traces"][()], source["profiling_data"][()]),
            "Attack_traces": _group(source["attacking_traces"][()], source["attacking_data"][()]),
        }
    write_ascad_layout(output, groups)
