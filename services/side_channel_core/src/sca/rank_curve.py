# services/side_channel_core/src/sca/rank_curve.py
"""
Key-rank curve computation for Side-Channel Analysis (SCA).

TraceFlow public rank contract
------------------------------
All ranks leaving this module are one-based:

- best AES-byte key rank: 1
- worst AES-byte key rank: 256
- target-byte recovery: rank == 1

NumPy array positions remain zero-based internally, but are converted at the
rank-computation boundary. ``sca.rank_base`` is therefore a contract guard,
not a tunable metric convention, and only the value ``1`` is accepted.

Standard ASCAD-style attack
---------------------------
The model predicts a leakage class derived from::

    SBOX(plaintext[target_byte] XOR key_guess)

For each trace and each key hypothesis, this module accumulates the logarithm
of the predicted probability for the corresponding leakage class. It supports
ID, HW, and MSB leakage through ``leakage_models.py`` as the single source of
truth.

Numeric contract
----------------
Scores are accumulated in ``float64``. Callers should pass ``float64``
log-probabilities (``log_softmax`` of the model logits computed in
``float64``) with ``preds_are_log_probs=True``; that path never underflows.
Probabilities are still accepted for compatibility and are converted with
``log(clip(p, eps, 1))``. In both paths ``eps`` is a lower floor on the
per-trace log-likelihood: ``eps > 0`` floors every term at ``log(eps)``
(default ``1e-40``, i.e. about -92 nats); ``eps == 0`` disables the floor and
requires log-probability input so the score is the exact log-likelihood.

For HW and MSB the per-trace score is ``log p(z|t) - log p(z)`` (the class
prior of Eq. 2); ID classes are equiprobable and use ``log p(z|t)``.
``score_rule`` below names the rule an attack used.

Ties
----
When the true key's score equals that of other keys, the true key takes a
uniformly random place among them, drawn from a seeded generator (the
``TIE_BREAK_POLICY``). The rank is then an integer whose expectation is the
mid-rank, so GE and SR are unbiased: an uninformative model gets GE 128.5 in
expectation, not 1. A tie is never counted as a recovered key
(``true_key_tie_count`` must be 1 at the endpoint). A key is tied when its
score differs from the true key's by at most ``tie_tolerance``; the difference
is computed once, so every key is exactly one of higher, tied or lower.

Author: Shraddha Mishra
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch

from sca.leakage_models import (
    build_label_table,
    normalize_leakage_model,
    target_hypotheses,
    validate_num_classes_for_leakage,
)

KEYSPACE_SIZE = 256
PUBLIC_RANK_BASE = 1
TIE_BREAK_POLICY = "uniform_random_among_tied_keys_seeded"
# Seed of the tie-break generator when the caller supplies none.
DEFAULT_TIE_BREAK_SEED = 0
# Keys whose scores are the same sum of per-trace terms in another order (HW,
# MSB) differ only by float64 rounding, at most ~n ulps of the largest score
# after n traces. Scores within TIE_TOLERANCE_ULPS * n ulps are ties.
TIE_TOLERANCE_ULPS = 64


def _strict_int(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return an integer without silently truncating floats or booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer, got {value!r}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be <= {maximum}, got {result}")
    return result


def _strict_bool(value: Any, *, field: str) -> bool:
    """Reject truthy strings and integers used as configuration booleans."""
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
    """Return one finite real scalar subject to explicit bounds."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite, got {result!r}")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{field} must be {operator} {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be <= {maximum}, got {result}")
    return result


# =============================================================================
# Prediction and metadata helpers
# =============================================================================
def _to_numpy_probs(preds: torch.Tensor) -> np.ndarray:
    if not isinstance(preds, torch.Tensor):
        raise TypeError(f"preds must be torch.Tensor, got {type(preds)}")
    if preds.dim() != 2:
        raise ValueError(f"preds must be [N,C], got {tuple(preds.shape)}")
    if int(preds.shape[0]) < 1:
        raise ValueError("preds must contain at least one trace")
    if int(preds.shape[1]) < 1:
        raise ValueError("preds must contain at least one leakage class")

    probs = preds.detach().cpu().numpy()
    if probs.dtype != np.float64:
        probs = probs.astype(np.float64, copy=False)
    return probs


def _as_16_vector(value: Any, *, field: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    arr = np.asarray(value)
    if arr.shape != (16,):
        raise ValueError(f"metadata['{field}'] must have shape (16,), got {arr.shape}")
    return arr


def _get_meta_byte(metadata_row: Any, field: str, target_byte: int) -> int:
    if not isinstance(metadata_row, dict):
        raise TypeError(f"metadata element must be dict, got {type(metadata_row)}")
    if field not in metadata_row:
        raise KeyError(f"metadata dict missing required field '{field}'")

    arr = _as_16_vector(metadata_row[field], field=field)
    value = arr[int(target_byte)]

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(
            f"metadata['{field}'][{target_byte}] must be an integer byte, "
            f"got {value!r}"
        )
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(
            f"metadata['{field}'][{target_byte}] must be a finite integer "
            f"byte, got {value!r}"
        )
    byte_value = int(numeric)
    if not 0 <= byte_value <= 255:
        raise ValueError(
            f"metadata['{field}'][{target_byte}] out of byte range [0,255]: "
            f"{byte_value}"
        )
    return byte_value


def _extract_target_bytes(
    metadata: Sequence[Any],
    target_byte: int,
) -> Tuple[np.ndarray, np.ndarray]:
    target_byte = _strict_int(
        target_byte, field="target_byte", minimum=0, maximum=15
    )

    count = len(metadata)
    fast = _extract_target_bytes_fast(metadata, target_byte)
    if fast is not None:
        return fast

    plaintext_bytes = np.empty(count, dtype=np.uint8)
    key_bytes = np.empty(count, dtype=np.uint8)

    for index in range(count):
        plaintext_bytes[index] = np.uint8(
            _get_meta_byte(metadata[index], "plaintext", target_byte)
        )
        key_bytes[index] = np.uint8(
            _get_meta_byte(metadata[index], "key", target_byte)
        )

    return plaintext_bytes, key_bytes


def _extract_target_bytes_fast(
    metadata: Sequence[Any],
    target_byte: int,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Vectorised path for the common case: dict rows holding uint8[16] arrays.

    Returns None whenever any row does not match, so the validating per-row
    path above produces its precise error message instead.
    """
    try:
        plaintexts = np.stack([row["plaintext"] for row in metadata])
        keys = np.stack([row["key"] for row in metadata])
    except Exception:
        return None
    for array in (plaintexts, keys):
        if array.dtype != np.uint8 or array.ndim != 2 or array.shape[1] != 16:
            return None
    return (
        np.ascontiguousarray(plaintexts[:, int(target_byte)], dtype=np.uint8),
        np.ascontiguousarray(keys[:, int(target_byte)], dtype=np.uint8),
    )


def _check_probs(
    probs: np.ndarray,
    *,
    atol: float,
    rtol: float,
    check_row_sums: bool,
) -> None:
    if probs.ndim != 2 or probs.shape[0] < 1 or probs.shape[1] < 1:
        raise ValueError(
            f"probabilities must have non-empty shape [N,C], got {probs.shape}"
        )
    if not np.all(np.isfinite(probs)):
        raise ValueError("NaN/Inf in probabilities")

    if np.min(probs) < 0.0 or np.max(probs) > 1.0 + atol:
        raise ValueError("Probabilities must lie in [0,1] within prob_atol")

    if check_row_sums:
        row_sums = probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=atol, rtol=rtol):
            raise ValueError("Rows do not sum to 1 (likely logits)")


def _check_log_probs(
    log_probs: np.ndarray,
    *,
    atol: float,
    rtol: float,
    check_row_sums: bool,
) -> None:
    """Validate float64 log-probabilities; ``-inf`` (exact zero) is allowed."""
    if log_probs.ndim != 2 or log_probs.shape[0] < 1 or log_probs.shape[1] < 1:
        raise ValueError(
            f"log-probabilities must have non-empty shape [N,C], got {log_probs.shape}"
        )
    if np.any(np.isnan(log_probs)) or np.any(np.isposinf(log_probs)):
        raise ValueError("NaN/+Inf in log-probabilities")
    if np.max(log_probs) > np.log1p(atol):
        raise ValueError("log-probabilities must be <= 0 within prob_atol")

    if check_row_sums:
        row_max = np.max(log_probs, axis=1, keepdims=True)
        row_sums = np.exp(row_max[:, 0]) * np.sum(np.exp(log_probs - row_max), axis=1)
        if not np.allclose(row_sums, 1.0, atol=atol, rtol=rtol):
            raise ValueError("exp(log-probabilities) rows do not sum to 1")


def _check_scores(
    scores: np.ndarray,
    *,
    scores_are_log: bool,
    atol: float,
    rtol: float,
    check_row_sums: bool,
) -> None:
    """Validate ``scores`` as probabilities or log-probabilities."""
    if scores_are_log:
        _check_log_probs(scores, atol=atol, rtol=rtol, check_row_sums=check_row_sums)
    else:
        _check_probs(scores, atol=atol, rtol=rtol, check_row_sums=check_row_sums)


def _log_floor(eps: float) -> float | None:
    """Return ``log(eps)`` as the per-trace floor, or None when ``eps == 0``."""
    return None if eps <= 0.0 else float(np.log(eps))


def _eps_arg(eps: Any) -> float:
    """``eps`` is a probability floor in [0, 1]; 0 disables the floor."""
    return _finite_float(eps, field="eps", minimum=0.0, maximum=1.0)


def _validate_rank_base(rank_base: int) -> int:
    normalized = _strict_int(rank_base, field="rank_base")
    if normalized != PUBLIC_RANK_BASE:
        raise ValueError(
            f"Unsupported rank_base={normalized}. TraceFlow exposes one-based ranks "
            "only: best rank=1 and worst AES-byte rank=256. Set sca.rank_base: 1."
        )
    return normalized


# =============================================================================
# Rank helpers
# =============================================================================
def _trace_endpoints(trace_count: int, step: int) -> List[int]:
    """Return evaluation endpoints and always include the final trace."""
    trace_count = _strict_int(trace_count, field="trace_count", minimum=0)
    step = _strict_int(step, field="step", minimum=1)

    if trace_count <= 0:
        return []
    endpoints = list(range(step, trace_count + 1, step))
    if not endpoints or endpoints[-1] != trace_count:
        endpoints.append(trace_count)
    return endpoints


def _descending_order(scores: np.ndarray) -> np.ndarray:
    """Deterministic descending order; stable handling of equal scores."""
    return np.argsort(-scores, kind="stable")


def tie_break_rng(seed: Any = None) -> np.random.Generator:
    """Generator for tie-breaking; ``seed`` is anything ``default_rng`` takes."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(DEFAULT_TIE_BREAK_SEED if seed is None else seed)


def tie_tolerance(scores: np.ndarray, n_traces: Any) -> np.ndarray:
    """Per-row absolute tie tolerance for scores summed over ``n_traces``."""
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    n = np.maximum(np.asarray(n_traces, dtype=np.float64).reshape(-1, 1), 1.0)
    scale = np.maximum(np.max(np.abs(scores), axis=1, keepdims=True), 1.0)
    return TIE_TOLERANCE_ULPS * n * np.finfo(np.float64).eps * scale


def _split_around_true_key(
    scores: np.ndarray,
    true_key: int,
    n_traces: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Masks ``(higher, tied, lower)`` of the keys in each row of ``scores``.

    ``d = s - t`` (key score minus true-key score) is computed once and every
    key lands in exactly one class: higher ``d > tol``, tied ``|d| <= tol``
    (the true key included) or lower ``d < -tol``, with ``tol`` from
    ``tie_tolerance``. Testing ``s > t + tol`` instead rounds ``t + tol`` and
    leaves a key at ``fl(t + tol)`` or ``fl(t - tol)`` in no class.
    """
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    diff = scores - scores[:, [int(true_key)]]
    tolerance = tie_tolerance(scores, n_traces)
    higher = diff > tolerance
    tied = np.abs(diff) <= tolerance
    lower = diff < -tolerance
    # The classes are disjoint (tol > 0); only a non-finite score escapes all.
    covered = (
        np.count_nonzero(higher) + np.count_nonzero(tied) + np.count_nonzero(lower)
    )
    if covered != diff.size:
        raise ValueError(
            "key scores must be finite to rank the true key; check the "
            "prediction representation and metrics_cfg.rank_log_eps"
        )
    return higher, tied, lower


def true_key_ranks(
    scores: np.ndarray,
    true_key: int,
    rng: np.random.Generator,
    n_traces: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """One-based rank of ``true_key`` in each row of ``scores`` ``[M, 256]``.

    ``n_traces`` (scalar or per row) is the number of traces summed into each
    row. Returns ``(ranks, tie_counts)``; ``tie_counts`` includes the true
    key. A tied true key takes a uniformly random place among its ties
    (module docstring, "Ties"), so the rank never depends on the key's value.
    """
    higher, tied, _ = _split_around_true_key(scores, true_key, n_traces)
    ties = np.count_nonzero(tied, axis=1)
    # integers(0, ties) draws one place in [0, ties - 1] per row.
    offsets = rng.integers(0, ties)
    return np.count_nonzero(higher, axis=1) + offsets + PUBLIC_RANK_BASE, ties


def order_with_true_key_at(
    scores: np.ndarray,
    true_key: int,
    rank: int,
    n_traces: int,
) -> np.ndarray:
    """Descending key order whose position ``rank`` holds ``true_key``.

    Keys above, tied with (within ``tie_tolerance``) and below the true key
    keep the stable descending order; the true key takes position ``rank``
    inside its tie group, so the top hypothesis agrees with the rank.
    """
    true_key = int(true_key)
    masks = [m[0] for m in _split_around_true_key(scores, true_key, n_traces)]
    stable = [int(k) for k in _descending_order(scores) if int(k) != true_key]
    higher, tied, lower = ([k for k in stable if mask[k]] for mask in masks)
    place = int(rank) - PUBLIC_RANK_BASE - len(higher)
    if not 0 <= place <= len(tied):
        raise ValueError(
            f"rank {rank} is not a tie-consistent position for key {true_key}"
        )
    return np.asarray(
        higher + tied[:place] + [true_key] + tied[place:] + lower, dtype=np.int64
    )


def score_rule(leakage_model: str) -> str:
    """Name of the per-trace score an attack under ``leakage_model`` uses."""
    counts = np.bincount(build_label_table(leakage_model)[0])
    counts = counts[counts > 0]
    return (
        "log_likelihood"
        if np.all(counts == counts[0])
        else "log_likelihood_minus_log_class_prior"
    )


def recovery_trace_points(
    curve: np.ndarray,
    *,
    recovery_rank: int = 1,
) -> Tuple[int | None, int | None]:
    """
    Return ``(first_recovery_trace, stable_recovery_trace)``.

    Stable recovery is the earliest evaluated trace count after which every
    remaining curve point stays at or below ``recovery_rank``.
    """
    if (
        not isinstance(curve, np.ndarray)
        or curve.ndim != 2
        or curve.shape[1] != 2
        or curve.shape[0] == 0
    ):
        return None, None

    threshold = _strict_int(
        recovery_rank,
        field="recovery_rank",
        minimum=PUBLIC_RANK_BASE,
        maximum=KEYSPACE_SIZE,
    )

    recovered = curve[:, 1].astype(np.float64) <= float(threshold)
    if not np.any(recovered):
        return None, None

    first_index = int(np.flatnonzero(recovered)[0])
    first_trace = int(curve[first_index, 0])

    # suffix_all[i] is true exactly when all points i..end are recovered.
    suffix_all = np.logical_and.accumulate(recovered[::-1])[::-1]
    stable_indices = np.flatnonzero(suffix_all)
    stable_trace = (
        int(curve[int(stable_indices[0]), 0]) if stable_indices.size else None
    )
    return first_trace, stable_trace


def _ranked_key_candidates(
    scores: np.ndarray,
    order: np.ndarray,
    n_traces: int,
) -> List[Dict[str, float | int]]:
    """Return the complete, deterministically ordered AES-byte keyspace."""
    if scores.shape != (KEYSPACE_SIZE,):
        raise ValueError(
            f"scores must have shape ({KEYSPACE_SIZE},), got {scores.shape}"
        )
    if order.shape != (KEYSPACE_SIZE,):
        raise ValueError(
            f"order must have shape ({KEYSPACE_SIZE},), got {order.shape}"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")

    ordered_keys = order.astype(np.int64, copy=False)
    if (
        np.any(ordered_keys < 0)
        or np.any(ordered_keys >= KEYSPACE_SIZE)
        or np.unique(ordered_keys).size != KEYSPACE_SIZE
    ):
        raise ValueError(
            "order must contain every AES-byte key hypothesis exactly once"
        )
    # Tied keys (within the rounding tolerance) may appear in any order.
    if np.any(np.diff(scores[ordered_keys]) > tie_tolerance(scores, n_traces)[0, 0]):
        raise ValueError("candidate order must be a descending order of scores")

    rows: List[Dict[str, float | int]] = []
    for rank_index, key_guess in enumerate(ordered_keys, start=PUBLIC_RANK_BASE):
        guess = int(key_guess)
        rows.append(
            {
                "rank": rank_index,
                "key_guess": guess,
                "score": float(scores[guess]),
            }
        )
    return rows


# =============================================================================
# Fixed-key attack computation
# =============================================================================
def _hypotheses(metadata: Any, target: str, target_byte: int, leakage_model: str) -> Optional[np.ndarray]:
    """``[N, 256]`` hypotheses of a non-S-box target; None for "sbox" (the table path, unchanged)."""
    if target == "sbox":
        return None
    return target_hypotheses(metadata, target=target, byte=int(target_byte), leakage_model=leakage_model)


def per_trace_key_scores(
    scores: np.ndarray,
    trace_indices: np.ndarray,
    plaintexts: np.ndarray,
    label_table: np.ndarray,
    *,
    eps: float,
    scores_are_log: bool,
    hypotheses: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score of every key hypothesis for each trace, ``[len(trace_indices), 256]``.

    ``scores`` is ``[N, C]`` float64 (log-probabilities when ``scores_are_log``,
    probabilities otherwise), ``plaintexts`` the target plaintext byte of each
    selected trace and ``label_table[plaintext, key_guess]`` the class. This is
    the one definition of the attack's per-trace score: the rank curve and the
    key vector of Lm (ldd.py) both sum it.
    """
    # The class of every key guess per trace: label_table[plaintext] for the first-round S-box,
    # else the database type's target (sca/leakage_models.target_hypotheses), in trace order.
    labels = label_table[plaintexts] if hypotheses is None else hypotheses
    per_trace = scores[trace_indices[:, None], labels]
    log_floor = _log_floor(eps)
    if scores_are_log:
        # float64 log-probabilities: exact, optionally floored.
        if log_floor is not None:
            per_trace = np.maximum(per_trace, log_floor)
    else:
        per_trace = np.log(np.clip(per_trace, eps, 1.0))
    # Eq. 2 divides by the prior of the class: log p(z|t) - log p(z). For HW
    # the classes are not equiprobable (C(8,z)/256); without this the ranking
    # favours keys that map to common classes (Benadjila et al. 2020, C.3.2;
    # Ito et al. 2021). Uniform priors (ID, MSB) shift every key by the same
    # constant and are skipped so their scores stay bit-identical.
    class_counts = np.bincount(label_table[0], minlength=scores.shape[1])
    if np.any(class_counts != class_counts[0]):
        log_prior = np.log(class_counts.astype(np.float64) / KEYSPACE_SIZE)
        per_trace = per_trace - log_prior[labels]
    return per_trace


def _fixed_key_attack_from_indices(
    scores: np.ndarray,
    plaintext_bytes: np.ndarray,
    key_bytes: np.ndarray,
    indices: np.ndarray,
    *,
    step: int,
    eps: float,
    rank_base: int,
    leakage_model: str,
    top_k_candidates: int = 5,
    scores_are_log: bool = False,
    tie_break_seed: Any = None,
    hypotheses: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Accumulate float64 per-key log-likelihoods over ``indices``.

    ``hypotheses`` (``[N, 256]``, sca/leakage_models.target_hypotheses) replaces the
    first-round S-box table for database types with another label target.

    ``scores`` is ``[N, C]`` float64: log-probabilities when
    ``scores_are_log`` is true, probabilities otherwise. See the module
    docstring for the ``eps`` floor contract and the tie rule;
    ``tie_break_seed`` seeds the tie-break draws.
    """
    # Retained for caller compatibility. Evidence always contains the complete
    # keyspace; report construction applies the configured candidate selection.
    _strict_int(
        top_k_candidates,
        field="top_k_candidates",
        minimum=1,
        maximum=KEYSPACE_SIZE,
    )
    _validate_rank_base(rank_base)

    if not isinstance(indices, np.ndarray) or indices.ndim != 1:
        raise TypeError("indices must be a one-dimensional NumPy array")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("indices must contain integers")

    count = int(indices.shape[0])
    if count == 0:
        raise ValueError("Cannot compute a key-rank curve from zero traces")
    if np.any(indices < 0) or np.any(indices >= scores.shape[0]):
        raise IndexError("indices contain a trace position outside preds")
    if np.unique(indices).size != count:
        raise ValueError("indices must not contain duplicate trace positions")

    step = _strict_int(step, field="step", minimum=1)
    eps = _eps_arg(eps)
    scores_are_log = _strict_bool(scores_are_log, field="scores_are_log")
    log_floor = _log_floor(eps)
    if scores.dtype != np.float64:
        raise TypeError(f"scores must be float64, got {scores.dtype}")
    if log_floor is None:
        if not scores_are_log:
            raise ValueError(
                "rank_log_eps=0 (exact log-likelihood) requires log-probability "
                "input; probabilities cannot represent log(0)"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "rank_log_eps=0 (exact log-likelihood) requires finite "
                "log-probabilities; a probability head underflowed to 0. Use a "
                "logits/LogSoftmax head or set metrics_cfg.rank_log_eps > 0"
            )
    true_key = int(key_bytes[int(indices[0])])

    label_table = build_label_table(leakage_model)
    if not isinstance(label_table, np.ndarray) or label_table.shape != (
        KEYSPACE_SIZE,
        KEYSPACE_SIZE,
    ):
        raise ValueError(
            "build_label_table() must return a [256,256] NumPy array "
            "indexed as [plaintext, key_guess]"
        )
    if not np.issubdtype(label_table.dtype, np.integer):
        raise TypeError("build_label_table() must return integer class indices")
    if np.any(label_table < 0) or np.any(label_table >= scores.shape[1]):
        raise ValueError("build_label_table() returned an out-of-range class index")
    # Per-trace score of every key hypothesis, in attack order: [count, 256].
    ordered_plaintexts = plaintext_bytes[indices]
    per_trace = per_trace_key_scores(
        scores,
        indices,
        ordered_plaintexts,
        label_table,
        eps=eps,
        scores_are_log=scores_are_log,
        hypotheses=None if hypotheses is None else hypotheses[indices],
    )
    # Running sum along the trace axis: the same sequential float64 additions
    # as accumulating one trace at a time, so scores are bit-identical.
    cumulative = np.cumsum(per_trace, axis=0, dtype=np.float64)
    if not np.all(np.isfinite(cumulative[-1])):
        raise FloatingPointError(
            "Accumulated key scores became non-finite; check the "
            "prediction representation and metrics_cfg.rank_log_eps"
        )

    # Ties are broken uniformly at random (module docstring, "Ties").
    endpoints = np.asarray(_trace_endpoints(count, step), dtype=np.int64)
    ranks, tie_counts = true_key_ranks(
        cumulative[endpoints - 1], true_key, tie_break_rng(tie_break_seed), endpoints
    )
    curve_rows: List[List[float]] = [
        [float(e), float(r)] for e, r in zip(endpoints, ranks)
    ]

    log_scores = np.array(cumulative[-1], dtype=np.float64)
    curve = np.asarray(curve_rows, dtype=np.float64)
    final_rank = int(ranks[-1])
    final_order = order_with_true_key_at(log_scores, true_key, final_rank, count)
    first_rank1, stable_rank1 = recovery_trace_points(curve, recovery_rank=1)

    margin = None
    if final_order.size >= 2:
        top1 = int(final_order[0])
        top2 = int(final_order[1])
        margin = float(log_scores[top1] - log_scores[top2])

    tied_with_true = int(tie_counts[-1])
    higher, _, _ = _split_around_true_key(log_scores, true_key, count)
    true_rank_best = int(np.count_nonzero(higher)) + PUBLIC_RANK_BASE
    true_rank_worst = true_rank_best + tied_with_true - 1

    return {
        "curve": curve,
        "true_key_byte": int(true_key),
        "top_key_hypothesis": int(final_order[0]),
        "final_true_key_rank": int(final_rank),
        "first_rank1_trace": first_rank1,
        "stable_rank1_trace": stable_rank1,
        "top_key_candidates": _ranked_key_candidates(
            log_scores,
            final_order,
            count,
        ),
        "top1_top2_score_margin": margin,
        "true_key_tie_count": tied_with_true,
        "true_key_rank_best_if_tied": true_rank_best,
        "true_key_rank_worst_if_tied": true_rank_worst,
        "tie_break_policy": TIE_BREAK_POLICY,
        "score_rule": score_rule(leakage_model),
        # Internal aggregation material for repeated attacks. The caller must
        # not put this raw vector into scalar metric logs.
        "final_key_scores": log_scores.copy(),
        "traces_used": int(curve[-1, 0]),
    }


def fixed_key_attack_evidence(
    preds: torch.Tensor,
    metadata: Sequence[Any],
    target_byte: int,
    step: int = 10,
    *,
    eps: float = 1e-40,
    strict_prob_check: bool = True,
    prob_atol: float = 1e-3,
    prob_rtol: float = 1e-3,
    check_fixed_key: bool = True,
    rank_base: Literal[1] = 1,
    leakage_model: Literal["ID", "HW", "MSB"] = "ID",
    top_k_candidates: int = 5,
    preds_are_log_probs: bool = False,
    tie_break_seed: Any = None,
    target: str = "sbox",
) -> Dict[str, Any]:
    """Compute one fixed-key attack curve together with final key evidence.

    ``target``: the database type's label intermediate (sca/dataset_types.py); "sbox" is the
    first-round S-box output, the others score every guess from sca/leakage_models.target_hypotheses.

    ``preds`` is ``[N, C]``: float64 log-probabilities when
    ``preds_are_log_probs`` is true (preferred, exact), probabilities otherwise.
    """
    _validate_rank_base(rank_base)
    leakage_model = normalize_leakage_model(leakage_model)
    preds_are_log_probs = _strict_bool(
        preds_are_log_probs, field="preds_are_log_probs"
    )

    step = _strict_int(step, field="step", minimum=1)
    strict_prob_check = _strict_bool(
        strict_prob_check, field="strict_prob_check"
    )
    check_fixed_key = _strict_bool(check_fixed_key, field="check_fixed_key")
    if not check_fixed_key:
        raise ValueError(
            "fixed_key_attack_evidence cannot disable fixed-key validation; "
            "use full_ranks_curve(..., variable_key=True) for changing keys"
        )
    eps = _eps_arg(eps)
    prob_atol = _finite_float(prob_atol, field="prob_atol", minimum=0.0)
    prob_rtol = _finite_float(prob_rtol, field="prob_rtol", minimum=0.0)

    probs = _to_numpy_probs(preds)
    _check_scores(
        probs,
        scores_are_log=preds_are_log_probs,
        atol=prob_atol,
        rtol=prob_rtol,
        check_row_sums=strict_prob_check,
    )
    validate_num_classes_for_leakage(probs.shape[1], leakage_model)

    if metadata is None:
        raise ValueError("fixed_key_attack_evidence requires metadata")
    if len(metadata) != int(probs.shape[0]):
        raise ValueError(
            f"metadata length {len(metadata)} does not match N={probs.shape[0]}"
        )
    if len(metadata) == 0:
        raise ValueError("fixed_key_attack_evidence requires at least one trace")

    plaintext_bytes, key_bytes = _extract_target_bytes(metadata, target_byte)
    key0 = int(key_bytes[0])
    if not np.all(key_bytes == key0):
        unique_keys = np.unique(key_bytes).tolist()
        raise ValueError(
            "Fixed-key attack received changing key bytes. "
            f"Observed key-byte values: {unique_keys[:16]}"
        )

    return _fixed_key_attack_from_indices(
        probs,
        plaintext_bytes,
        key_bytes,
        np.arange(int(probs.shape[0]), dtype=np.int64),
        step=step,
        eps=eps,
        rank_base=rank_base,
        leakage_model=leakage_model,
        top_k_candidates=top_k_candidates,
        scores_are_log=preds_are_log_probs,
        hypotheses=_hypotheses(metadata, target, target_byte, leakage_model),
        tie_break_seed=tie_break_seed,
    )


# =============================================================================
# Variable-key aggregation
# =============================================================================
def _aggregate_group_curves(
    group_curves: List[np.ndarray],
    group_sizes: List[int],
    *,
    mode: Literal["weighted", "equal"],
    step: int,
) -> np.ndarray:
    """Aggregate variable-key groups on a traces-per-key-group axis.

    This diagnostic curve is not a full-population fixed-key recovery curve and
    must not be passed to the repeated full-population attack protocol.
    """
    if not group_curves:
        return np.zeros((0, 2), dtype=np.float64)

    if len(group_curves) != len(group_sizes):
        raise ValueError("group_curves and group_sizes must have equal length")
    if mode not in {"weighted", "equal"}:
        raise ValueError("mode must be 'weighted' or 'equal'")
    step = _strict_int(step, field="step", minimum=1)
    if any(size < 1 for size in group_sizes):
        raise ValueError("every variable-key group must contain at least one trace")

    common_trace_count = min(int(curve[-1, 0]) for curve in group_curves)
    endpoints = np.asarray(
        _trace_endpoints(common_trace_count, step), dtype=np.int64
    )
    ranks = np.stack(
        [curve[endpoints - 1, 1] for curve in group_curves], axis=0
    )

    if mode == "equal":
        y = np.mean(ranks, axis=0)
    else:
        weights = np.asarray(group_sizes, dtype=np.float64)
        weights /= np.sum(weights)
        y = np.sum(ranks * weights[:, None], axis=0)

    return np.stack([endpoints.astype(np.float64), y], axis=1)


# =============================================================================
# Backward-compatible public curve API
# =============================================================================
def full_ranks_curve(
    preds: torch.Tensor,
    metadata: Any,
    target_byte: int,
    step: int = 10,
    *,
    eps: float = 1e-40,
    strict_prob_check: bool = True,
    prob_atol: float = 1e-3,
    prob_rtol: float = 1e-3,
    check_fixed_key: bool = True,
    variable_key: bool = False,
    variable_key_aggregation: Literal["weighted", "equal"] = "weighted",
    rank_base: Literal[1] = 1,
    leakage_model: Literal["ID", "HW", "MSB"] = "ID",
    target: str = "sbox",
    preds_are_log_probs: bool = False,
    tie_break_seed: Any = None,
) -> np.ndarray:
    """Return a one-based key-rank curve with shape ``[M, 2]``.

    ``preds_are_log_probs=True`` marks ``preds`` as float64 log-probabilities.
    """
    _validate_rank_base(rank_base)
    leakage_model = normalize_leakage_model(leakage_model)
    preds_are_log_probs = _strict_bool(
        preds_are_log_probs, field="preds_are_log_probs"
    )
    step = _strict_int(step, field="step", minimum=1)
    strict_prob_check = _strict_bool(
        strict_prob_check, field="strict_prob_check"
    )
    check_fixed_key = _strict_bool(check_fixed_key, field="check_fixed_key")
    variable_key = _strict_bool(variable_key, field="variable_key")
    if variable_key_aggregation not in {"weighted", "equal"}:
        raise ValueError(
            "variable_key_aggregation must be 'weighted' or 'equal'"
        )

    if not variable_key:
        return fixed_key_attack_evidence(
            preds=preds,
            metadata=metadata,
            target_byte=target_byte,
            step=step,
            eps=eps,
            strict_prob_check=strict_prob_check,
            prob_atol=prob_atol,
            prob_rtol=prob_rtol,
            check_fixed_key=check_fixed_key,
            rank_base=rank_base,
            leakage_model=leakage_model,
            top_k_candidates=1,
            preds_are_log_probs=preds_are_log_probs,
            tie_break_seed=tie_break_seed,
            target=target,
        )["curve"]

    probs = _to_numpy_probs(preds)
    eps = _eps_arg(eps)
    prob_atol = _finite_float(prob_atol, field="prob_atol", minimum=0.0)
    prob_rtol = _finite_float(prob_rtol, field="prob_rtol", minimum=0.0)
    _check_scores(
        probs,
        scores_are_log=preds_are_log_probs,
        atol=prob_atol,
        rtol=prob_rtol,
        check_row_sums=strict_prob_check,
    )
    validate_num_classes_for_leakage(probs.shape[1], leakage_model)

    if metadata is None:
        raise ValueError("variable-key rank evaluation requires metadata")
    if len(metadata) != int(probs.shape[0]):
        raise ValueError(
            f"metadata length {len(metadata)} does not match N={probs.shape[0]}"
        )

    plaintext_bytes, key_bytes = _extract_target_bytes(metadata, target_byte)
    groups: Dict[int, np.ndarray] = {}
    for key_value in range(KEYSPACE_SIZE):
        group_indices = np.where(key_bytes == key_value)[0]
        if group_indices.size:
            groups[key_value] = group_indices

    curves: List[np.ndarray] = []
    sizes: List[int] = []
    tie_rng = tie_break_rng(tie_break_seed)
    hypotheses = _hypotheses(metadata, target, target_byte, leakage_model)
    for group_indices in groups.values():
        evidence = _fixed_key_attack_from_indices(
            probs,
            plaintext_bytes,
            key_bytes,
            group_indices,
            # Unit-step curves allow exact alignment across unequal key groups.
            step=1,
            eps=eps,
            rank_base=rank_base,
            leakage_model=leakage_model,
            top_k_candidates=1,
            scores_are_log=preds_are_log_probs,
            tie_break_seed=tie_rng,
            hypotheses=hypotheses,
        )
        curves.append(evidence["curve"])
        sizes.append(int(group_indices.size))

    return _aggregate_group_curves(
        curves,
        sizes,
        mode=variable_key_aggregation,
        step=step,
    )
