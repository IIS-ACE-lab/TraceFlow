# services/side_channel_core/tests/unit/test_final_rank.py
"""
Tie classification around the true key (audit SCA-01): ``d = s - t`` is
computed once and every key is exactly one of higher, tied or lower, also at
the rounded tolerance boundaries ``fl(t + tol)`` and ``fl(t - tol)``.

Run from the repository root with
PYTHONPATH="$PWD:$PWD/services/side_channel_core:$PWD/services/side_channel_core/src".
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sca.leakage_models import build_label_table
from sca.rank_curve import (
    KEYSPACE_SIZE,
    _fixed_key_attack_from_indices,
    _split_around_true_key,
    order_with_true_key_at,
    tie_break_rng,
    tie_tolerance,
    true_key_ranks,
)

TRUE_KEY = 0x2A
BOUNDARY_KEY = 0x07
SCALE_KEY = 0xFF  # holds the largest |score|, so it sets the tolerance
PLAINTEXT = 0x5C


def _boundary_scores(side: int, outside: bool) -> np.ndarray:
    """One-trace scores with ``BOUNDARY_KEY`` at ``fl(t + side * tol)``.

    ``outside`` picks a case where the rounded boundary lies strictly outside
    the exact tolerance (the former gap) or on/inside it. All scores sit in
    [40, 64) in magnitude, so ``s - t`` is exact (Sterbenz) and the class can
    be checked in exact rational arithmetic.
    """
    rng = np.random.default_rng(3)
    for _ in range(10_000):
        t = float(rng.uniform(-60.0, -40.0))
        scores = np.full(KEYSPACE_SIZE, t - 5.0)
        scores[SCALE_KEY] = -float(rng.uniform(80.0, 99.0))
        scores[TRUE_KEY] = t
        tol = float(tie_tolerance(scores, 1)[0, 0])
        s = t + side * tol  # float64 rounding: fl(t +/- tol)
        if (abs(Fraction(s) - Fraction(t)) > Fraction(tol)) == outside:
            scores[BOUNDARY_KEY] = s
            return scores
    raise AssertionError("no boundary case found")


def _one_trace_attack(key_scores: np.ndarray, seed: int = 0) -> dict:
    """ID attack on one trace whose per-key log-likelihoods are ``key_scores``."""
    plain = np.array([PLAINTEXT], dtype=np.uint8)
    row = np.empty(KEYSPACE_SIZE, dtype=np.float64)
    row[build_label_table("ID")[PLAINTEXT]] = key_scores
    return _fixed_key_attack_from_indices(
        row[None, :], plain, np.array([TRUE_KEY], dtype=np.uint8), np.arange(1),
        step=1, eps=0.0, rank_base=1, leakage_model="ID",
        scores_are_log=True, tie_break_seed=seed,
    )


def _prefix_ranks(scores, true_key, rng, n_traces):
    """``true_key_ranks`` before SCA-01 (tests ``s > fl(t + tol)``)."""
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    true_scores = scores[:, [int(true_key)]]
    tolerance = tie_tolerance(scores, n_traces)
    higher = np.count_nonzero(scores > true_scores + tolerance, axis=1)
    ties = np.count_nonzero(np.abs(scores - true_scores) <= tolerance, axis=1)
    return higher + rng.integers(0, ties) + 1, ties


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("outside", [True, False])
def test_boundary_key_lands_in_exactly_one_class(side, outside):
    scores = _boundary_scores(side, outside)
    higher, tied, lower = _split_around_true_key(scores, TRUE_KEY, 1)
    assert np.all(higher.astype(int) + tied + lower == 1)
    expected = (higher if side > 0 else lower) if outside else tied
    assert expected[0, BOUNDARY_KEY]
    assert tied[0, TRUE_KEY]


def test_key_just_above_tolerance_counts_as_higher_mid_curve():
    above = _boundary_scores(1, outside=True)
    below = np.full(KEYSPACE_SIZE, -50.0)
    below[TRUE_KEY] = -40.0
    ranks, ties = true_key_ranks(np.stack([below, above]), TRUE_KEY, tie_break_rng(0), [1, 1])
    # The former rounded test missed the key at fl(t + tol): rank 1, not 2.
    assert ranks.tolist() == [1, 2]
    assert ties.tolist() == [1, 1]


@pytest.mark.parametrize("side,rank,top", [(1, 2, BOUNDARY_KEY), (-1, 1, TRUE_KEY)])
def test_boundary_endpoint_orders_all_256_keys(side, rank, top):
    # Formerly: ValueError "order must have shape (256,), got (255,)".
    scores = _boundary_scores(side, outside=True)
    evidence = _one_trace_attack(scores)
    assert np.array_equal(evidence["final_key_scores"], scores)
    guesses = [row["key_guess"] for row in evidence["top_key_candidates"]]
    assert sorted(guesses) == list(range(KEYSPACE_SIZE))
    assert evidence["final_true_key_rank"] == rank
    assert evidence["true_key_tie_count"] == 1
    assert evidence["true_key_rank_best_if_tied"] == rank == evidence["true_key_rank_worst_if_tied"]
    assert evidence["top_key_hypothesis"] == top
    assert guesses[:2] == ([BOUNDARY_KEY, TRUE_KEY] if side > 0 else [TRUE_KEY, BOUNDARY_KEY])


@pytest.mark.parametrize("rank", [1, 2])
def test_boundary_tie_orders_true_key_at_either_place(rank):
    scores = _boundary_scores(1, outside=False)
    order = order_with_true_key_at(scores, TRUE_KEY, rank, 1)
    assert sorted(order.tolist()) == list(range(KEYSPACE_SIZE))
    assert order[rank - 1] == TRUE_KEY
    assert set(order[:2].tolist()) == {TRUE_KEY, BOUNDARY_KEY}


@pytest.mark.parametrize("quantised", [False, True])
def test_ranks_unchanged_away_from_the_boundary(quantised):
    rng = np.random.default_rng(29)
    for _ in range(300):
        scores = rng.normal(-100.0, 30.0, size=(12, KEYSPACE_SIZE))
        if quantised:
            scores = np.round(scores / 4.0)  # exact ties between keys
        n_traces = rng.integers(1, 5000, size=12)
        key = int(rng.integers(0, KEYSPACE_SIZE))
        seed = int(rng.integers(0, 2**31))
        prefix_rng, new_rng = tie_break_rng(seed), tie_break_rng(seed)
        expected_ranks, expected_ties = _prefix_ranks(scores, key, prefix_rng, n_traces)
        ranks, ties = true_key_ranks(scores, key, new_rng, n_traces)
        assert np.array_equal(ranks, expected_ranks) and ranks.dtype == expected_ranks.dtype
        assert np.array_equal(ties, expected_ties)
        # Same tie-break draw count and order.
        assert prefix_rng.bit_generator.state == new_rng.bit_generator.state


@pytest.mark.parametrize("leakage_model,classes", [("ID", 256), ("HW", 9), ("MSB", 2)])
def test_attack_ranks_unchanged_for_each_leakage_model(leakage_model, classes):
    rng = np.random.default_rng(31)
    for trial in range(40):
        n = int(rng.integers(1, 50))
        logits = rng.normal(0.0, 1.0, size=(n, classes))
        if trial % 2:
            logits = np.round(logits)  # quantised: HW/MSB keys tie up to rounding
        log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        plain = rng.integers(0, 256, size=n, dtype=np.uint8)
        key = int(rng.integers(0, KEYSPACE_SIZE))
        evidence = _fixed_key_attack_from_indices(
            log_probs, plain, np.full(n, key, dtype=np.uint8), np.arange(n),
            step=n, eps=1e-40, rank_base=1,
            leakage_model=leakage_model, scores_are_log=True, tie_break_seed=trial,
        )
        final = evidence["final_key_scores"]
        ranks, ties = _prefix_ranks(final, key, tie_break_rng(trial), n)
        assert evidence["final_true_key_rank"] == int(ranks[0])
        assert evidence["true_key_tie_count"] == int(ties[0])
        best = int(np.count_nonzero(final > final[key] + tie_tolerance(final, n)[0, 0])) + 1
        assert evidence["true_key_rank_best_if_tied"] == best


def test_non_finite_score_is_rejected_with_a_hint():
    scores = np.full(KEYSPACE_SIZE, -50.0)
    scores[3] = np.nan
    with pytest.raises(ValueError, match="rank_log_eps"):
        true_key_ranks(scores, TRUE_KEY, tie_break_rng(0), 1)
