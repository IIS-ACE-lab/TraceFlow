# services/side_channel_core/utils/h5_utils.py
"""
HDF5 Utilities for Side-Channel Analysis (SCA)

This module provides:
- Structure inspection for debugging
- Robust sample counting across common ASCAD-style layouts
- Writing split datasets into canonical grouped HDF5

Scientific correctness:
- For SCA evaluation (GE / rank-based metrics), metadata must be preserved.
- This module therefore defaults to STRICT mode: no dummy metadata is allowed.

Auto-labeling:
- The _recompute_labels_in_h5() function can recompute labels from plaintext/key metadata using the active SCA leakage model (ID, HW, MSB):
    ID:
    h5_utils -> compute_leakage_label -> AES_SBOX[pt ^ key]

    HW:
    h5_utils -> compute_leakage_label -> AES_SBOX[pt ^ key] -> HW(...)

    MSB:
    h5_utils -> compute_leakage_label -> AES_SBOX[pt ^ key] -> MSB(...)

Author: Shraddha Mishra
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import h5py
import numpy as np

from utils.logger import get_logger
from src.sca.leakage_models import (
    compute_leakage_label,
    normalize_leakage_model,
)
logger = get_logger("h5_utils")

# ---------------------------------------------------------------------
# Import authoritative AES_SBOX (single source of truth)
# ---------------------------------------------------------------------

# if that import path fails in your runtime, adjust to match your PYTHONPATH

def _recompute_labels_in_h5(
    out_h5: Path,
    *,
    group_name: str,
    attack_byte: int,
    leakage_model: str = "ID",
) -> None:
    """
    Overwrite <group_name>/labels using metadata plaintext/key and the active
    SCA leakage model.

    Label contract:
      ID:
        label = SBOX(plaintext[byte] XOR key[byte])          -> 0..255

      HW:
        label = HW(SBOX(plaintext[byte] XOR key[byte]))      -> 0..8

      MSB:
        label = MSB(SBOX(plaintext[byte] XOR key[byte]))     -> 0..1

    Fixed-key override support:
      - If metadata has no 'key' field but file root attr 'sca_fixed_key_hex' exists,
        use that fixed key for all traces.
      - If metadata has 'key', use the per-trace key.
    """
    b = int(attack_byte)
    if b < 0 or b > 15:
        raise ValueError(f"attack_byte must be 0..15, got {b}")

    leakage_model = normalize_leakage_model(leakage_model)

    def _parse_fixed_key_hex(v) -> np.ndarray:
        s = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
        s = s.strip().lower().replace("0x", "").replace(" ", "")
        if len(s) != 32:
            raise ValueError(
                f"sca_fixed_key_hex must be 32 hex chars (16 bytes), got len={len(s)}"
            )
        return np.frombuffer(bytes.fromhex(s), dtype=np.uint8)

    with h5py.File(str(out_h5), "r+") as f:
        if group_name not in f:
            raise KeyError(f"Group {group_name} not found in {out_h5}")

        grp = f[group_name]

        if "metadata" not in grp or "labels" not in grp:
            raise KeyError(
                f"{out_h5} missing {group_name}/metadata or {group_name}/labels"
            )

        md = grp["metadata"]

        if md.dtype.fields is None or "plaintext" not in md.dtype.fields:
            raise KeyError(
                f"{out_h5} {group_name}/metadata missing 'plaintext'"
            )

        pt = np.asarray(md["plaintext"], dtype=np.uint8)
        if pt.ndim != 2 or pt.shape[1] != 16:
            raise ValueError(
                f"{out_h5} {group_name}/metadata['plaintext'] expected shape [N,16], "
                f"got {pt.shape}"
            )

        # --- key source selection ---
        fixed_key = None
        if "sca_fixed_key_hex" in f.attrs and f.attrs["sca_fixed_key_hex"]:
            fixed_key = _parse_fixed_key_hex(f.attrs["sca_fixed_key_hex"])
            if fixed_key.shape != (16,):
                raise ValueError(f"Fixed key parsed to wrong shape: {fixed_key.shape}")

        if md.dtype.fields is not None and "key" in md.dtype.fields:
            ky = np.asarray(md["key"], dtype=np.uint8)
            if ky.ndim != 2 or ky.shape[1] != 16:
                raise ValueError(
                    f"{out_h5} {group_name}/metadata['key'] expected shape [N,16], "
                    f"got {ky.shape}"
                )
        else:
            if fixed_key is None:
                raise KeyError(
                    f"{out_h5} {group_name}/metadata missing 'key' and no root attr "
                    "'sca_fixed_key_hex' present (cannot relabel)"
                )

            ky = np.tile(
                fixed_key.reshape(1, 16),
                (pt.shape[0], 1),
            ).astype(np.uint8, copy=False)

        new_labels = compute_leakage_label(
            pt[:, b],
            ky[:, b],
            leakage_model,
        ).astype(np.uint8, copy=False)

        labels_ds = grp["labels"]
        if labels_ds.shape != new_labels.shape:
            raise ValueError(
                f"{out_h5} {group_name}/labels shape mismatch: "
                f"stored={labels_ds.shape}, recomputed={new_labels.shape}"
            )

        labels_ds[...] = new_labels

        f.attrs["sca_attack_byte"] = int(b)
        f.attrs["sca_leakage_model"] = str(leakage_model)
        grp.attrs["sca_attack_byte"] = int(b)
        grp.attrs["sca_leakage_model"] = str(leakage_model)

    logger.info(
        "Recomputed labels in %s:%s using attack_byte=%d leakage_model=%s",
        str(out_h5),
        group_name,
        b,
        leakage_model,
    )

# ---------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------
def inspect_h5_structure(h5_file: str, label: str = "") -> None:
    """Logs the structure of an HDF5 file (groups, shapes, dtypes)."""
    try:
        if label:
            logger.info("Inspecting HDF5 structure (%s): %s", label, h5_file)
        else:
            logger.info("Inspecting HDF5 structure: %s", h5_file)

        with h5py.File(h5_file, "r") as f:

            def walk(name, node):
                if isinstance(node, h5py.Dataset):
                    logger.info("Dataset: %-45s shape=%s dtype=%s", name, node.shape, node.dtype)
                elif isinstance(node, h5py.Group):
                    logger.debug("Group:   %s", name)

            f.visititems(walk)

    except Exception as e:
        logger.error("Error inspecting %s: %s", h5_file, e, exc_info=True)
        raise


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _to_1d_trace_np(x: Any) -> np.ndarray:
    """Convert torch.Tensor / numpy array to 1D numpy trace [L]."""
    if hasattr(x, "detach"):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)

    x_np = np.squeeze(x_np)  # [1,L] -> [L]
    if x_np.ndim != 1:
        raise ValueError(f"Trace must be 1D after squeeze, got shape={x_np.shape}")
    return x_np


def _to_int_label(y: Any) -> int:
    """Convert torch.Tensor / numpy scalar / python int -> int."""
    if hasattr(y, "detach"):
        y_np = y.detach().cpu().numpy()
        if np.ndim(y_np) > 0:
            y_np = y_np.item()
        return int(y_np)
    return int(y)


def _unpack_item(item: Any) -> Tuple[np.ndarray, int, Any]:
    """
    Requires dataset[i] returning (x, y, meta).
    Returns (trace_1d_np, label_int, meta_object).
    """
    if not isinstance(item, (tuple, list)):
        raise TypeError(f"Dataset item must be tuple/list, got {type(item)}")

    if len(item) != 3:
        raise ValueError(
            "SCA split saving requires dataset items to be (x, y, meta). "
            f"Got len={len(item)}. Fix your Dataset/__getitem__ to include metadata."
        )

    x, y, meta = item
    if meta is None:
        raise ValueError("Metadata is None for a sample. This is not allowed for SCA evaluation.")

    return _to_1d_trace_np(x), _to_int_label(y), meta


def _meta_to_np(meta: Any) -> np.ndarray:
    """
    Convert metadata (numpy structured row, numpy array, dict-like, etc.) into a storable numpy object.

    Best case for ASCAD:
      - meta is a numpy.void (structured row) or numpy structured array row.
      - np.asarray(meta) keeps dtype fields.

    If meta is dict-like, np.asarray(meta) becomes object dtype; storable but not ideal.
    Prefer structured dtypes for interoperability.
    """
    try:
        return np.asarray(meta)
    except Exception as e:
        raise TypeError(f"Could not convert metadata to numpy array: {type(meta)} ({e})") from e


# ---------------------------------------------------------------------
# Counting / inspection helpers
# ---------------------------------------------------------------------
def count_h5_samples(h5_path: str | Path) -> int:
    """Counts samples from common ASCAD-style layouts."""
    h5_path = str(h5_path)
    with h5py.File(h5_path, "r") as f:
        # canonical grouped paths
        if "Profiling_traces" in f and "traces" in f["Profiling_traces"]:
            return int(f["Profiling_traces/traces"].shape[0])
        if "Attack_traces" in f and "traces" in f["Attack_traces"]:
            return int(f["Attack_traces/traces"].shape[0])

        # legacy top-level
        if "traces" in f:
            return int(f["traces"].shape[0])

        # fallback: first dataset anywhere
        def _find_first_dataset(group: h5py.Group) -> Optional[h5py.Dataset]:
            for _, item in group.items():
                if isinstance(item, h5py.Dataset):
                    return item
                if isinstance(item, h5py.Group):
                    ds = _find_first_dataset(item)
                    if ds is not None:
                        return ds
            return None

        ds0 = _find_first_dataset(f)
        if ds0 is None:
            raise RuntimeError(f"No dataset found in {h5_path}")
        return int(ds0.shape[0])


def inspect_and_log_h5(h5_path: str | Path, label: str) -> int:
    """Counts samples, logs it, then inspects structure."""
    h5_path = str(h5_path)
    n_samples = count_h5_samples(h5_path)
    logger.info("Inspecting %d samples at %s", n_samples, h5_path)
    inspect_h5_structure(h5_path, label=label)
    return n_samples
