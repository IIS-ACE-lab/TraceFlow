# services/side_channel_core/src/sca/ldd.py
"""
Leakage Distribution Difference (LDD) and Lm for Side-Channel Analysis (SCA).

Purpose
-------
Implements AutoSCA-style research metrics:
- LDD: Leakage Distribution Difference
- Lm : correlation between ranked LDD and model-induced key ranking

Definitions
-----------
For each key hypothesis k in [0..255] and plaintext p in [0..255]:

  z_k(p) = leakage_model( SBOX(p XOR k) )

LDD for correct key k_c is summed over the attack traces, d_i being the
plaintext byte of trace i (Wu, Perin, Picek, "I Choose You", Eq. 8):
    LDD[k] = sum_i ( z_kc(d_i) - z_k(d_i) )^2

Lm (Eq. 10) is the Pearson correlation between:
- ranked LDD vector
- key-ranking vector of the attack: the keys ranked by the attack's own key
  scores (rank_curve.per_trace_key_scores, summed over the traces; for HW that
  is log p(z|t) - log p(z)), so Lm describes the ranking the attack reports

Notes
-----
- This module assumes AES byte attacks.
- Lm is only well-defined in the fixed-key setting in the same way as the
  AutoSCA paper/code. For variable-key datasets, this module raises.
- Inputs to key-log-prob computation must be probabilities, not logits.

Author: Shraddha Mishra
"""
from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Dict, Sequence

import numpy as np
import torch

from sca.dataset_types import label_target, resolved_variant
from sca.rank_curve import _check_scores, _log_floor, per_trace_key_scores
from sca.leakage_models import (
    normalize_leakage_model,
    build_label_table,
    resolve_runtime_leakage_model,
    validate_num_classes_for_leakage,
)


def _strict_int(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer, got {value!r}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be <= {maximum}, got {result}")
    return result


def _strict_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be true or false, got {value!r}")
    return value


def _finite_float(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be numeric, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite, got {result!r}")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            op = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{field} must be {op} {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be <= {maximum}, got {result}")
    return result


# =============================================================================
# Config helpers
# =============================================================================
def _get_attack_byte(cfg: dict) -> int:
    sca = cfg.get("sca") or {}
    b = sca.get("attack_byte", None)
    if b is None:
        raise KeyError("LDD/Lm requires cfg['sca']['attack_byte']")
    return _strict_int(b, field="sca.attack_byte", minimum=0, maximum=15)


def _get_leakage_model(cfg: dict) -> str:
    return str(
        resolve_runtime_leakage_model(cfg, default=None, mutate=False)
    )


def _get_rank_log_eps(cfg: dict) -> float:
    # 0 disables the per-trace floor (exact float64 log-likelihood; needs
    # log-probability input). See rank_curve.py "Numeric contract".
    mc = cfg.get("metrics_cfg") or {}
    return _finite_float(
        mc.get("rank_log_eps", 1e-40),
        field="metrics_cfg.rank_log_eps",
        minimum=0.0,
        maximum=1.0,
    )


# =============================================================================
# Metadata helpers
# =============================================================================
def _as_16_vector(v: Any, *, field: str) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()

    arr = np.asarray(v)
    arr = np.squeeze(arr)

    if arr.shape != (16,):
        raise ValueError(f"metadata['{field}'] must have shape (16,), got {arr.shape}")
    return arr


def _get_meta_byte(m: Any, field: str, tb: int) -> int:
    if not isinstance(m, dict):
        raise TypeError(f"metadata element must be dict, got {type(m)}")
    if field not in m:
        raise KeyError(f"metadata dict missing required field '{field}'")

    arr = _as_16_vector(m[field], field=field)
    x = arr[int(tb)]

    if isinstance(x, (bool, np.bool_)) or not isinstance(x, Real):
        raise TypeError(
            f"metadata['{field}'][{tb}] must be an integer byte, got {x!r}"
        )
    numeric = float(x)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(
            f"metadata['{field}'][{tb}] must be a finite integer byte, got {x!r}"
        )
    return _strict_int(
        int(numeric),
        field=f"metadata['{field}'][{tb}]",
        minimum=0,
        maximum=255,
    )


def _extract_target_bytes(metadata: Sequence[Any], tb: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(metadata)
    pt_b = np.empty(n, dtype=np.uint8)
    key_b = np.empty(n, dtype=np.uint8)
    for i in range(n):
        pt_b[i] = np.uint8(_get_meta_byte(metadata[i], "plaintext", tb))
        key_b[i] = np.uint8(_get_meta_byte(metadata[i], "key", tb))
    return pt_b, key_b


def _infer_fixed_correct_key(metadata: Sequence[Any], tb: int) -> int:
    if metadata is None or len(metadata) == 0:
        raise ValueError("metadata is empty; cannot infer correct key for LDD/Lm")

    _pt_b, key_b = _extract_target_bytes(metadata, tb)
    k0 = int(key_b[0])
    if np.any(key_b != np.uint8(k0)):
        raise ValueError(
            "Lm/LDD currently expects fixed-key metadata for paper-faithful evaluation. "
            "Detected multiple key-byte values in metadata."
        )
    return k0


# =============================================================================
# Probability helpers
# =============================================================================
def _to_numpy_probs(preds: torch.Tensor) -> np.ndarray:
    if not isinstance(preds, torch.Tensor):
        raise TypeError(f"preds must be torch.Tensor, got {type(preds)}")
    if preds.dim() != 2:
        raise ValueError(f"preds must be [N,C], got {tuple(preds.shape)}")
    if int(preds.shape[0]) < 1 or int(preds.shape[1]) < 1:
        raise ValueError(f"preds must have non-empty shape [N,C], got {tuple(preds.shape)}")

    probs = preds.detach().cpu().numpy()
    if probs.dtype != np.float64:
        probs = probs.astype(np.float64, copy=False)
    return probs


def _check_probs(
    probs: np.ndarray,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    min_tol: float = -1e-9,
    max_tol: float = 1.0 + 1e-6,
) -> None:
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape {probs.shape}")
    if probs.shape[0] < 1 or probs.shape[1] < 1:
        raise ValueError("LDD/Lm requires a non-empty probability matrix")
    if not np.all(np.isfinite(probs)):
        raise ValueError("LDD/Lm requires finite probabilities (no NaN/Inf)")

    pmin = float(np.min(probs))
    pmax = float(np.max(probs))
    if pmin < min_tol:
        raise ValueError(f"LDD/Lm requires non-negative probabilities. min={pmin:.3e}")
    if pmax > max_tol:
        raise ValueError(f"LDD/Lm requires probabilities <= 1. max={pmax:.3e}")

    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol, rtol=rtol):
        worst = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(
            f"LDD/Lm requires probabilities (rows sum to 1). max|sum-1|={worst:.3e} "
            f"(atol={atol}, rtol={rtol})"
        )


# =============================================================================
# Ranking helpers
# =============================================================================
def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """
    scipy.stats.rankdata(method='average') equivalent, ascending ranks in [1..N].
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    return ranks


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError(f"Pearson correlation requires same shape, got {x.shape} vs {y.shape}")

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x_c = x - x_mean
    y_c = y - y_mean

    denom = np.sqrt(np.sum(x_c * x_c) * np.sum(y_c * y_c))
    if denom <= 0.0 or not np.isfinite(denom):
        raise ValueError(
            "Lm is undefined because at least one ranked vector is constant"
        )

    corr = float(np.sum(x_c * y_c) / denom)
    if not np.isfinite(corr):
        raise ValueError("Lm correlation is not finite")
    if corr < -1.0 - 1e-12 or corr > 1.0 + 1e-12:
        raise ValueError(f"Lm correlation lies outside [-1,1]: {corr}")
    return float(np.clip(corr, -1.0, 1.0))


# =============================================================================
# Public LDD / key-score functions
# =============================================================================
def calculate_ldd(
    correct_key: int,
    leakage_model: str = "HW",
    *,
    plaintexts: Any,
) -> np.ndarray:
    """
    Compute LDD vector of length 256 for a given correct key hypothesis.

    ``plaintexts`` are the target plaintext bytes d_i of the attack traces
    (Eq. 8 sums over the traces; a plaintext value met several times counts
    each time).

    Returns:
      ldd[k] for all k in [0..255]
    """
    correct_key = _strict_int(
        correct_key, field="correct_key", minimum=0, maximum=255
    )

    leakage_model = normalize_leakage_model(leakage_model)

    plaintexts = np.asarray(plaintexts)
    if plaintexts.ndim != 1 or plaintexts.size == 0:
        raise ValueError("LDD requires the plaintext byte of at least one attack trace")
    if not np.issubdtype(plaintexts.dtype, np.integer):
        raise TypeError("LDD plaintexts must be integer byte values")
    if np.any(plaintexts < 0) or np.any(plaintexts > 255):
        raise ValueError("LDD plaintexts must lie in [0, 255]")

    table = build_label_table(leakage_model).astype(np.float64)  # [256,256], rows=pt, cols=key
    ref = table[:, correct_key][:, None]                         # [256,1]
    diffs = ref - table                                          # [256,256]
    # Sum over the traces = every plaintext value weighted by how often it occurs.
    occurrences = np.bincount(plaintexts.astype(np.int64), minlength=256).astype(np.float64)
    ldd = occurrences @ (diffs * diffs)                          # [256]

    if not np.all(np.isfinite(ldd)):
        raise ValueError("LDD contains non-finite values")
    return ldd.astype(np.float64, copy=False)


def compute_key_log_probs(
    preds: torch.Tensor,
    metadata: Sequence[Any],
    *,
    target_byte: int,
    leakage_model: str = "ID",
    eps: float = 1e-40,
    prob_atol: float = 1e-3,
    prob_rtol: float = 1e-3,
    preds_are_log_probs: bool = False,
) -> np.ndarray:
    """
    Compute the attack's key scores for the 256 key hypotheses: the sum over
    the traces of rank_curve.per_trace_key_scores, i.e. log p(z|t) and, for a
    leakage model with non-uniform classes (HW), log p(z|t) - log p(z). The
    same scores as the rank curve's final_key_scores.

    Args:
      preds:
        [N,C] float64 log-probabilities when ``preds_are_log_probs`` is true
        (preferred, exact); probabilities otherwise. Never logits.
      metadata:
        sequence of dict-like metadata with plaintext/key
      target_byte:
        attacked AES byte
      leakage_model:
        'ID', 'HW', or 'MSB'
      eps:
        lower clipping boundary used before logarithmic accumulation

    Returns:
      key_log_probs: np.ndarray shape [256]
    """
    tb = _strict_int(
        target_byte, field="target_byte", minimum=0, maximum=15
    )
    eps = _finite_float(eps, field="eps", minimum=0.0, maximum=1.0)
    prob_atol = _finite_float(prob_atol, field="prob_atol", minimum=0.0)
    prob_rtol = _finite_float(prob_rtol, field="prob_rtol", minimum=0.0)
    preds_are_log_probs = _strict_bool(
        preds_are_log_probs, field="preds_are_log_probs"
    )
    log_floor = _log_floor(eps)
    if log_floor is None and not preds_are_log_probs:
        raise ValueError(
            "rank_log_eps=0 (exact log-likelihood) requires log-probability input"
        )

    leakage_model = normalize_leakage_model(leakage_model)

    probs = _to_numpy_probs(preds)
    _check_scores(
        probs,
        scores_are_log=preds_are_log_probs,
        atol=prob_atol,
        rtol=prob_rtol,
        check_row_sums=True,
    )

    n, c = probs.shape
    validate_num_classes_for_leakage(int(c), leakage_model)

    if metadata is None:
        raise ValueError("metadata is required for key-log-prob computation")
    if len(metadata) != n:
        raise ValueError(f"metadata length {len(metadata)} does not match N={n}")

    plaintexts, _key_b = _extract_target_bytes(metadata, tb)
    table = build_label_table(leakage_model)  # [256,256], rows=pt, cols=key
    if not isinstance(table, np.ndarray) or table.shape != (256, 256):
        raise ValueError("build_label_table() must return shape [256,256]")

    per_trace = per_trace_key_scores(                            # [n, 256]
        probs,
        np.arange(n, dtype=np.int64),
        plaintexts,
        table,
        eps=eps,
        scores_are_log=preds_are_log_probs,
    )
    # cumsum keeps the sequential float64 addition order of the former loop.
    key_log_probs = np.array(
        np.cumsum(per_trace, axis=0, dtype=np.float64)[-1], dtype=np.float64
    )

    if not np.all(np.isfinite(key_log_probs)):
        raise ValueError("Computed key_log_probs contains non-finite values")
    return key_log_probs


def calculate_lm_from_key_log_probs(
    key_log_probs: np.ndarray,
    *,
    correct_key: int,
    leakage_model: str = "HW",
    plaintexts: Any,
) -> float:
    """
    Compute Lm from the attack's key scores (compute_key_log_probs) and the
    plaintext bytes of the attack traces.

    AutoSCA-style definition:
      ranked_LDD = rankdata(LDD(correct_key, plaintexts))
      key_rank_vector = 256 - rankdata(key_log_probs)
      Lm = corr(ranked_LDD, key_rank_vector)
    """
    key_log_probs = np.asarray(key_log_probs, dtype=np.float64)
    if key_log_probs.shape != (256,):
        raise ValueError(f"key_log_probs must have shape (256,), got {key_log_probs.shape}")
    if not np.all(np.isfinite(key_log_probs)):
        raise ValueError("key_log_probs contains non-finite values")

    leakage_model = normalize_leakage_model(leakage_model)
    correct_key = _strict_int(
        correct_key, field="correct_key", minimum=0, maximum=255
    )

    ldd = calculate_ldd(correct_key=correct_key, leakage_model=leakage_model, plaintexts=plaintexts)
    ranked_ldd = _rankdata_average(ldd)                   # ascending ranks 1..256
    key_rank_vector = 256.0 - _rankdata_average(key_log_probs)

    corr = _pearson_corr(ranked_ldd, key_rank_vector)
    return float(corr)


def compute_ldd_lm(
    preds: torch.Tensor,
    metadata: Sequence[Any],
    cfg: dict,
    *,
    preds_are_log_probs: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper for metrics.py / evaluation code.

    Returns:
      {
        "LDD": np.ndarray shape [256],
        "Lm":  float,
        "key_log_probs": np.ndarray shape [256],
        "correct_key": int,
      }
    """
    target = label_target(cfg)
    if target != "sbox":
        # LDD is defined on z_k(p) = leakage_model(SBOX(p XOR k)) (above); no other label target.
        raise ValueError(f"LDD/Lm are defined for the first-round S-box target only, not {target!r} "
                         "(the database type's label target, sca/dataset_types.py)")
    tb = _get_attack_byte(cfg)
    leakage_model = _get_leakage_model(cfg)
    eps = _get_rank_log_eps(cfg)

    mc = cfg.get("metrics_cfg") or {}
    if not isinstance(mc, dict):
        raise TypeError("cfg['metrics_cfg'] must be a mapping")
    prob_atol = _finite_float(
        mc.get("prob_atol", 1e-3), field="metrics_cfg.prob_atol", minimum=0.0
    )
    prob_rtol = _finite_float(
        mc.get("prob_rtol", 1e-3), field="metrics_cfg.prob_rtol", minimum=0.0
    )

    sca = cfg.get("sca") or {}
    if not isinstance(sca, dict):
        raise TypeError("cfg['sca'] must be a mapping")
    alias = sca.get("variable_key")
    if alias is not None:
        alias = _strict_bool(alias, field="sca.variable_key")
    variant_raw = resolved_variant(cfg)   # the database type gives it (sca/dataset_types.py)
    if variant_raw is None:
        variant = "variable_key" if alias else "fixed_key"
    elif not isinstance(variant_raw, str):
        raise TypeError("sca.variant must be a string")
    else:
        variant = variant_raw.strip().lower()
    if variant not in {"fixed_key", "variable_key"}:
        raise ValueError("sca.variant must be 'fixed_key' or 'variable_key'")
    if alias is not None and alias != (variant == "variable_key"):
        raise ValueError("sca.variant conflicts with sca.variable_key")
    if variant == "variable_key":
        raise ValueError("LDD/Lm is defined only for fixed-key evaluation")

    correct_key = _infer_fixed_correct_key(metadata, tb)
    key_log_probs = compute_key_log_probs(
        preds,
        metadata,
        target_byte=tb,
        leakage_model=leakage_model,
        eps=eps,
        prob_atol=prob_atol,
        prob_rtol=prob_rtol,
        preds_are_log_probs=preds_are_log_probs,
    )
    plaintexts, _key_bytes = _extract_target_bytes(metadata, tb)
    ldd = calculate_ldd(correct_key=correct_key, leakage_model=leakage_model, plaintexts=plaintexts)
    lm = calculate_lm_from_key_log_probs(
        key_log_probs,
        correct_key=correct_key,
        leakage_model=leakage_model,
        plaintexts=plaintexts,
    )

    return {
        "LDD": ldd,
        "Lm": float(lm),
        "key_log_probs": key_log_probs,
        "correct_key": int(correct_key),
    }
