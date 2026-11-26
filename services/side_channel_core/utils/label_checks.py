# utils/label_checks.py
"""
Label Integrity Checks — ASCAD / SCA-fit

Purpose
-------
Hard-stop validation that stored labels in an HDF5 split match the *intended*
label function. This prevents the classic failure mode where training converges
to ~ln(256) because labels don't correspond to the metadata.

Authoritative label function in this codebase (unmasked ASCAD):
    label = leakage_model( AES_SBOX[ plaintext[byte] ^ key[byte] ] )

AES_SBOX comes from utils/aes.py so there is only ONE S-box definition.

Masked labels (SBOX[p^k] ^ mask) are not accepted for the S-box target: masks are
unknown at attack time, so a split labelled that way could not be attacked by
rank_curve.py. infer_label_model() still reports a masked-label match as a
diagnostic. A database type with another label target (sca/dataset_types.py:
DPAv4's known mask, AES_HD's last round) records it in the file
(attr sca_label_target); the check then recomputes that target
(sca/leakage_models.target_labels), and a file whose recorded target is not
the run's is refused.

Optional:
- supports fixed_key override via H5 attr sca_fixed_key_hex
- prints concrete mismatch examples
- includes a diagnostic helper infer_label_model() to detect a wrong byte, leakage model (ID/HW/MSB) or masked labels.

Usage
-----
from utils.label_checks import check_label_integrity

check_label_integrity(str(val_h5), attack_byte=3, n=10)
check_label_integrity(str(test_h5), attack_byte=3, n=10)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from utils.aes import AES_SBOX
from src.sca.leakage_models import (
    apply_leakage_model,
    get_supported_leakage_models,
    normalize_leakage_model,
    target_labels,
)
import h5py
import numpy as np


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
_GROUP_BY_SPLIT = {
    "profiling": "Profiling_traces",
    "attack": "Attack_traces",
    "train": "Profiling_traces",
    "val": "Attack_traces",
    "test": "Attack_traces",
}
def _is_fixed_key_variant(f: h5py.File) -> bool:
    v = str(f.attrs.get("sca_variant", "")).strip().lower()
    return v == "fixed_key"

def _hex_to_key_bytes(hex_str: str) -> np.ndarray:
    s = str(hex_str).strip().lower().replace("0x", "").replace(" ", "")
    if len(s) != 32:
        raise ValueError(f"sca_fixed_key_hex must be 32 hex chars (16 bytes). Got len={len(s)}")
    b = bytes.fromhex(s)
    return np.frombuffer(b, dtype=np.uint8)


def _pick_group(f: h5py.File, group: Optional[str], split: Optional[str]) -> h5py.Group:
    if group:
        if group not in f:
            raise KeyError(f"Group '{group}' not found in {f.filename}. Keys={list(f.keys())}")
        return f[group]

    if split:
        g = _GROUP_BY_SPLIT.get(str(split).strip().lower())
        if not g:
            raise ValueError(f"Unknown split={split!r}. Use one of {list(_GROUP_BY_SPLIT)}")
        if g not in f:
            raise KeyError(f"Group '{g}' not found in {f.filename}. Keys={list(f.keys())}")
        return f[g]

    # default: prefer Attack_traces if present, else Profiling_traces
    if "Attack_traces" in f:
        return f["Attack_traces"]
    if "Profiling_traces" in f:
        return f["Profiling_traces"]
    raise KeyError(f"No known groups found in {f.filename}. Keys={list(f.keys())}")


def _read_meta_fields(grp: h5py.Group) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (plaintext[N,16], key[N,16] or None, masks[N,16] or None)
    """
    if "metadata" not in grp:
        raise KeyError(f"Missing {grp.name}/metadata; cannot run label integrity checks.")

    md = grp["metadata"]
    if not hasattr(md, "dtype") or md.dtype.fields is None:
        raise TypeError(f"{grp.name}/metadata is not structured; dtype={getattr(md, 'dtype', None)}")

    fields = md.dtype.fields
    if "plaintext" not in fields:
        raise KeyError(f"{grp.name}/metadata missing field 'plaintext'")
    pt = np.asarray(md["plaintext"], dtype=np.uint8)

    ky = None
    if "key" in fields:
        ky = np.asarray(md["key"], dtype=np.uint8)

    masks = None
    if "masks" in fields:
        masks = np.asarray(md["masks"], dtype=np.uint8)

    if pt.ndim != 2 or pt.shape[1] != 16:
        raise ValueError(f"plaintext expected shape [N,16], got {pt.shape}")
    if ky is not None and (ky.ndim != 2 or ky.shape[1] != 16):
        raise ValueError(f"key expected shape [N,16], got {ky.shape}")
    if masks is not None and (masks.ndim != 2 or masks.shape[1] != 16):
        # some datasets store a scalar mask or different layout; we keep it optional
        masks = None

    return pt, ky, masks


def _compute_unmasked_labels(pt: np.ndarray, ky: np.ndarray, byte_idx: int) -> np.ndarray:
    x = np.bitwise_xor(pt[:, byte_idx], ky[:, byte_idx]).astype(np.uint8, copy=False)
    return AES_SBOX[x].astype(np.uint8, copy=False)


def _compute_masked_labels_xor_output(unmasked: np.ndarray, masks: np.ndarray, byte_idx: int) -> np.ndarray:
    """
    Diagnostic only (infer_label_model and the mismatch hint below): a
    masked-label convention  label = SBOX(pt^key) ^ mask_byte.
    check_label_integrity() never accepts it.
    """
    m = np.asarray(masks[:, byte_idx], dtype=np.uint8)
    return np.bitwise_xor(unmasked, m).astype(np.uint8, copy=False)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def check_label_integrity(
    h5_path: str,
    *,
    attack_byte: int,
    leakage_model: str,
    n: int = 10,
    group: Optional[str] = None,
    split: Optional[str] = None,
    seed: int = 1337,
    verbose: bool = True,
    target: str = "sbox",
) -> None:
    """
    Hard-stop check on a random sample of ``n`` rows:
      stored_labels == leakage_model(AES_SBOX[pt[byte]^key[byte]])
    (with fixed-key override if present). Masked labels are rejected.
    ``target`` (the run's database type, sca/dataset_types.label_target): another
    label target is recomputed with sca/leakage_models.target_labels; the file must
    record the same target (attr sca_label_target, "sbox" when absent).

    Raises ValueError on failure.
    """
    p = Path(h5_path)
    if not p.exists():
        raise FileNotFoundError(f"H5 not found: {p}")

    b = int(attack_byte)
    if b < 0 or b > 15:
        raise ValueError(f"attack_byte must be 0..15, got {b}")

    with h5py.File(str(p), "r") as f:
        grp = _pick_group(f, group=group, split=split)
        raw_target = f.attrs.get("sca_label_target", "sbox")
        file_target = raw_target.decode() if isinstance(raw_target, bytes) else str(raw_target)
        if file_target != target:
            raise ValueError(
                f"Label integrity FAILED: {p} is labelled for target {file_target!r}, "
                f"the run's database type labels {target!r}"
            )

        if "labels" not in grp:
            raise KeyError(f"Missing {grp.name}/labels")
        stored = np.asarray(grp["labels"], dtype=np.uint8)
        N = int(stored.shape[0])
        if N <= 0:
            raise ValueError(f"{grp.name}/labels is empty")

        pt, ky, masks = _read_meta_fields(grp)

        # Fixed-key override from file attrs (if present)
        fk_hex = f.attrs.get("sca_fixed_key_hex", None)

        # Apply fixed-key override ONLY if:
        #   (a) metadata has no per-trace key, OR
        #   (b) file explicitly says it's fixed_key
        if fk_hex is not None and str(fk_hex).strip():
            if (ky is None) or _is_fixed_key_variant(f):
                fk = _hex_to_key_bytes(str(fk_hex))
                ky = np.tile(fk.reshape(1, 16), (N, 1)).astype(np.uint8, copy=False)

        if ky is None:
            raise KeyError(
                f"{grp.name}/metadata has no 'key' field and no sca_fixed_key_hex attr present; "
                "cannot recompute labels."
            )

        # sample indices
        rng = np.random.default_rng(int(seed))
        k = min(int(n), N)
        idx = rng.choice(N, size=k, replace=False)
        idx.sort()

        pt_s = pt[idx]
        ky_s = ky[idx]
        stored_s = stored[idx]

        lm = str(leakage_model).strip().upper()
        if target != "sbox":
            md = grp["metadata"]
            rows = {name: np.asarray(md[name])[idx] for name in md.dtype.names}
            rows["key"] = ky_s
            recomputed = target_labels(rows, target=target, byte=b, leakage_model=lm).astype(np.uint8)
            if np.all(recomputed == stored_s):
                if verbose:
                    print(f"[LabelCheck] OK {p} group={grp.name} attack_byte={b} checked={k} mode={target}")
                return
            raise ValueError("Label integrity FAILED: stored labels do not match expected label function "
                             f"(target {target}, {int(np.sum(recomputed != stored_s))} of {k} differ).")
        recomputed = _compute_unmasked_labels(pt_s, ky_s, b)
        if lm == "HW":
            recomputed = np.unpackbits(recomputed.reshape(-1, 1), axis=1).sum(axis=1).astype(np.uint8)
        elif lm == "MSB":
            recomputed = ((recomputed >> 7) & 1).astype(np.uint8)
        ok = np.all(recomputed == stored_s)

        if ok:
            if verbose:
                print(f"[LabelCheck] OK {p} group={grp.name} attack_byte={b} checked={k} mode=unmasked")
            return

        # report mismatches
        mism = np.where(recomputed != stored_s)[0]
        if verbose:
            print(
                f"[LabelCheck] {p} group={grp.name} attack_byte={b} checked={k} mismatches={len(mism)}"
            )
            # print up to 5 mismatch examples
            for j in mism[:5]:
                i = int(idx[j])
                print(f"  idx {i} stored {int(stored_s[j])} recomputed {int(recomputed[j])}")

            # extra hint if masks exist: masked labels are diagnosed, never accepted
            if masks is not None and lm == "ID":
                masked_try = _compute_masked_labels_xor_output(recomputed, masks[idx], b)
                if not np.any(masked_try != stored_s):
                    print("  hint: stored labels equal SBOX[p^k] ^ masks[byte] (masked labels); "
                          "regenerate the split with unmasked labels, the attack cannot use masks")

        raise ValueError("Label integrity FAILED: stored labels do not match expected label function.")


# ---------------------------------------------------------------------
# Diagnostics (optional but very useful)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class LabelModelGuess:
    byte: int
    mode: str  # "unmasked" or "masked_xor_output"
    leakage_model: str  # "ID", "HW" or "MSB"
    matches: int
    checked: int


def infer_label_model(
    h5_path: str,
    *,
    leakage_model: str,
    n: int = 200,
    group: Optional[str] = None,
    split: Optional[str] = None,
    seed: int = 1337,
) -> LabelModelGuess:
    """
    Try to infer which label model your file uses by brute-checking:
      - bytes 0..15
      - leakage models ID, HW and MSB, the active ``leakage_model`` first
        (it wins ties)
      - unmasked: L(SBOX(pt^key))
      - masked_xor_output: L(SBOX(pt^key) ^ masks[byte])   (only if masks exist)

    Returns best guess (max matches).
    """
    active = normalize_leakage_model(leakage_model)
    hypotheses = [active] + [lm for lm in get_supported_leakage_models() if lm != active]
    p = Path(h5_path)
    if not p.exists():
        raise FileNotFoundError(f"H5 not found: {p}")

    with h5py.File(str(p), "r") as f:
        grp = _pick_group(f, group=group, split=split)
        if "labels" not in grp:
            raise KeyError(f"Missing {grp.name}/labels")

        stored = np.asarray(grp["labels"], dtype=np.uint8)
        N = int(stored.shape[0])
        if N <= 0:
            raise ValueError(f"{grp.name}/labels is empty")

        pt, ky, masks = _read_meta_fields(grp)

        fk_hex = f.attrs.get("sca_fixed_key_hex", None)

        # Apply fixed-key override ONLY if:
        #   (a) metadata has no per-trace key, OR
        #   (b) file explicitly says it's fixed_key
        if fk_hex is not None and str(fk_hex).strip():
            if (ky is None) or _is_fixed_key_variant(f):
                fk = _hex_to_key_bytes(str(fk_hex))
                ky = np.tile(fk.reshape(1, 16), (N, 1)).astype(np.uint8, copy=False)

        if ky is None:
            raise KeyError(
                f"{grp.name}/metadata has no 'key' field and no sca_fixed_key_hex attr present; cannot infer."
            )

        rng = np.random.default_rng(int(seed))
        k = min(int(n), N)
        idx = rng.choice(N, size=k, replace=False)
        idx.sort()

        pt_s = pt[idx]
        ky_s = ky[idx]
        stored_s = stored[idx]
        masks_s = masks[idx] if masks is not None else None

        best = LabelModelGuess(byte=0, mode="unmasked", leakage_model=active, matches=-1, checked=k)

        for lm in hypotheses:
            for b in range(16):
                rec = _compute_unmasked_labels(pt_s, ky_s, b)
                m_unmasked = int(np.sum(apply_leakage_model(rec, lm) == stored_s))
                if m_unmasked > best.matches:
                    best = LabelModelGuess(
                        byte=b, mode="unmasked", leakage_model=lm, matches=m_unmasked, checked=k
                    )

                if masks_s is not None:
                    rec_m = _compute_masked_labels_xor_output(rec, masks_s, b)
                    m_masked = int(np.sum(apply_leakage_model(rec_m, lm) == stored_s))
                    if m_masked > best.matches:
                        best = LabelModelGuess(
                            byte=b, mode="masked_xor_output", leakage_model=lm,
                            matches=m_masked, checked=k,
                        )

        return best
