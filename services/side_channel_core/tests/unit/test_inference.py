# services/side_channel_core/tests/unit/test_inference.py
"""
Numeric contract of the attack path: float64 log-likelihood accumulation,
hand-computed key ranks, stable "traces to" statistics, and LR-scheduler
stepping rules.

Run from the repository root with
PYTHONPATH="$PWD:$PWD/services/side_channel_core:$PWD/services/side_channel_core/src".
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from sca.leakage_models import build_label_table
from sca.rank_curve import (
    KEYSPACE_SIZE,
    fixed_key_attack_evidence,
    full_ranks_curve,
    recovery_trace_points,
)
from sca.ldd import compute_key_log_probs
from sca.metrics import _first_threshold_trace, _stable_threshold_trace
from utils.optimizer_util import (
    resolve_scheduler_config,
    resolve_scheduler_interval,
    validate_scheduler_for_loop,
)

TARGET_BYTE = 2
TRUE_KEY = 0xE0


def _metadata(plaintexts: np.ndarray, key_byte: int) -> list[dict]:
    rows = []
    for p in plaintexts:
        pt = np.zeros(16, dtype=np.uint8)
        pt[TARGET_BYTE] = p
        key = np.zeros(16, dtype=np.uint8)
        key[TARGET_BYTE] = key_byte
        rows.append({"plaintext": pt, "key": key})
    return rows


def _reference_rank(log_probs: np.ndarray, plaintexts: np.ndarray, true_key: int) -> int:
    """Independent re-implementation: sum log p[label(p, k)] per key guess."""
    table = build_label_table("ID")
    scores = np.zeros(KEYSPACE_SIZE, dtype=np.float64)
    for i, p in enumerate(plaintexts):
        scores += log_probs[i, table[int(p)]]
    order = np.argsort(-scores, kind="stable")
    return int(np.flatnonzero(order == true_key)[0]) + 1


def _synthetic_logits(n: int, scale: float, seed: int = 7) -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    plaintexts = rng.integers(0, 256, size=n, dtype=np.uint8)
    table = build_label_table("ID")
    logits = rng.normal(0.0, 1.0, size=(n, KEYSPACE_SIZE))
    for i, p in enumerate(plaintexts):
        logits[i, table[int(p), TRUE_KEY]] += scale
    return torch.tensor(logits, dtype=torch.float32), plaintexts


def test_hand_computed_rank_three_traces():
    # Three traces; the correct class gets most of the mass only for key 0xE0.
    plaintexts = np.array([0x00, 0x7F, 0xFF], dtype=np.uint8)
    table = build_label_table("ID")
    probs = np.full((3, KEYSPACE_SIZE), 1e-3, dtype=np.float64)
    for i, p in enumerate(plaintexts):
        probs[i, table[int(p), TRUE_KEY]] = 0.745
    probs /= probs.sum(axis=1, keepdims=True)
    log_probs = torch.tensor(np.log(probs), dtype=torch.float64)

    evidence = fixed_key_attack_evidence(
        preds=log_probs,
        metadata=_metadata(plaintexts, TRUE_KEY),
        target_byte=TARGET_BYTE,
        step=1,
        preds_are_log_probs=True,
    )
    assert evidence["true_key_byte"] == TRUE_KEY
    assert evidence["final_true_key_rank"] == 1
    assert evidence["curve"].tolist() == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    expected_score = float(np.log(probs[np.arange(3), table[plaintexts, TRUE_KEY]]).sum())
    assert math.isclose(evidence["final_key_scores"][TRUE_KEY], expected_score, rel_tol=1e-12)


def test_log_domain_matches_probability_domain_when_nothing_underflows():
    logits, plaintexts = _synthetic_logits(n=300, scale=3.0)
    meta = _metadata(plaintexts, TRUE_KEY)
    log64 = torch.log_softmax(logits.double(), dim=1)
    prob64 = torch.exp(log64)
    curve_log = full_ranks_curve(
        preds=log64, metadata=meta, target_byte=TARGET_BYTE, step=10, preds_are_log_probs=True
    )
    curve_prob = full_ranks_curve(
        preds=prob64, metadata=meta, target_byte=TARGET_BYTE, step=10, preds_are_log_probs=False
    )
    assert np.array_equal(curve_log, curve_prob)
    assert int(curve_log[-1, 1]) == _reference_rank(log64.numpy(), plaintexts, TRUE_KEY)


def test_float32_softmax_underflow_is_avoided_in_float64_log_domain():
    # Very confident logits: float32 softmax underflows to exactly 0 for the
    # other classes, so the old probability path can only floor at log(eps),
    # while float64 log_softmax keeps the true per-trace log-likelihoods.
    logits, plaintexts = _synthetic_logits(n=50, scale=120.0)
    prob32 = torch.softmax(logits, dim=1)
    assert bool((prob32 == 0).any()), "test premise: float32 softmax underflows"
    log64 = torch.log_softmax(logits.double(), dim=1)
    assert bool(torch.isfinite(log64).all())
    meta = _metadata(plaintexts, TRUE_KEY)
    exact = fixed_key_attack_evidence(
        preds=log64, metadata=meta, target_byte=TARGET_BYTE, step=50,
        eps=0.0, preds_are_log_probs=True,
    )
    floored = fixed_key_attack_evidence(
        preds=prob32.double(), metadata=meta, target_byte=TARGET_BYTE, step=50,
        eps=1e-40, preds_are_log_probs=False,
    )
    wrong_key = (TRUE_KEY + 1) % 256
    # The floor caps every wrong-key term at -92.1 nats; the exact path is lower.
    assert exact["final_key_scores"][wrong_key] < floored["final_key_scores"][wrong_key]
    assert math.isclose(floored["final_key_scores"][wrong_key], 50 * math.log(1e-40), rel_tol=1e-6)
    assert exact["final_true_key_rank"] == 1 == floored["final_true_key_rank"]


def test_exact_mode_requires_log_probabilities():
    logits, plaintexts = _synthetic_logits(n=20, scale=2.0)
    meta = _metadata(plaintexts, TRUE_KEY)
    with pytest.raises(ValueError, match="requires log-probability input"):
        fixed_key_attack_evidence(
            preds=torch.softmax(logits.double(), dim=1), metadata=meta,
            target_byte=TARGET_BYTE, eps=0.0,
        )


def test_ldd_key_log_probs_agree_with_rank_scores():
    logits, plaintexts = _synthetic_logits(n=64, scale=2.5)
    meta = _metadata(plaintexts, TRUE_KEY)
    log64 = torch.log_softmax(logits.double(), dim=1)
    scores = compute_key_log_probs(
        log64, meta, target_byte=TARGET_BYTE, leakage_model="ID", preds_are_log_probs=True
    )
    evidence = fixed_key_attack_evidence(
        preds=log64, metadata=meta, target_byte=TARGET_BYTE, step=64, preds_are_log_probs=True
    )
    assert np.allclose(scores, evidence["final_key_scores"], rtol=0, atol=1e-9)


def test_stable_crossing_is_headline_first_crossing_is_kept():
    # rank touches 1 at 30 traces, bounces to 40, and only stays at 1 from 60.
    curve = np.array([[10, 90], [20, 12], [30, 1], [40, 40], [50, 3], [60, 1], [70, 1]], dtype=float)
    first = _first_threshold_trace(curve, threshold=1.0, direction="le")
    stable = _stable_threshold_trace(curve, threshold=1.0, direction="le")
    assert (first, stable) == (30, 60) and type(first) is int and type(stable) is int
    assert recovery_trace_points(curve, recovery_rank=1) == (30, 60)
    never = np.array([[10, 5], [20, 2]], dtype=float)
    assert _stable_threshold_trace(never, threshold=1.0, direction="le") is None
    transient = np.array([[10, 1], [20, 2]], dtype=float)
    assert _stable_threshold_trace(transient, threshold=1.0, direction="le") is None
    assert _first_threshold_trace(transient, threshold=1.0, direction="le") == 10


def _trainer(scheduler: dict) -> dict:
    return {
        "optimizer": {
            "type": "SGD",
            "learning_rate": 0.1,
            "weight_decay": 0.0,
            "args": {},
            "scheduler": scheduler,
        }
    }


def test_scheduler_interval_rules():
    assert resolve_scheduler_config({"type": "StepLR", "args": {"step_size": 5}})["interval"] == "epoch"
    assert resolve_scheduler_config({"type": "OneCycleLR", "args": {"max_lr": 0.1, "total_steps": 10}})["interval"] == "step"
    assert resolve_scheduler_config({"type": "CyclicLR", "args": {"base_lr": 0.01, "max_lr": 0.1}})["interval"] == "step"
    with pytest.raises(ValueError, match="defined per step"):
        resolve_scheduler_config({"type": "OneCycleLR", "args": {"max_lr": 0.1, "total_steps": 10}, "interval": "epoch"})
    with pytest.raises(ValueError, match="defined per epoch"):
        resolve_scheduler_config({"type": "StepLR", "args": {"step_size": 5}, "interval": "step"})
    assert resolve_scheduler_config({"type": "none"}) is None


def test_one_cycle_horizon_must_match_loop():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.1)
    trainer = _trainer({"type": "OneCycleLR", "args": {"max_lr": 0.1, "total_steps": 40}})
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=40)
    interval = resolve_scheduler_interval(trainer, scheduler)
    assert interval == "step"
    validate_scheduler_for_loop(scheduler, interval, epochs=4, steps_per_epoch=10)
    with pytest.raises(ValueError, match="OneCycleLR.total_steps=40"):
        validate_scheduler_for_loop(scheduler, interval, epochs=5, steps_per_epoch=10)
    assert resolve_scheduler_interval(trainer, None) is None


# ---------------------------------------------------------------------------
# Vectorised attack accumulation must be bit-identical to one-trace-at-a-time
# accumulation (the former implementation), for both input representations.
# ---------------------------------------------------------------------------
def _reference_attack(scores, plaintexts, true_key, *, step, eps, scores_are_log):
    table = build_label_table("ID")
    log_scores = np.zeros(KEYSPACE_SIZE, dtype=np.float64)
    rows = []
    floor = None if eps <= 0 else float(np.log(eps))
    endpoints = list(range(step, len(plaintexts) + 1, step))
    if not endpoints or endpoints[-1] != len(plaintexts):
        endpoints.append(len(plaintexts))
    previous = 0
    for endpoint in endpoints:
        for i in range(previous, endpoint):
            selected = scores[i, table[int(plaintexts[i])]]
            if scores_are_log:
                if floor is not None:
                    selected = np.maximum(selected, floor)
                log_scores += selected
            else:
                log_scores += np.log(np.clip(selected, eps, 1.0))
        previous = endpoint
        order = np.argsort(-log_scores, kind="stable")
        rows.append([float(endpoint), float(np.flatnonzero(order == true_key)[0]) + 1])
    return np.asarray(rows), log_scores


@pytest.mark.parametrize("scores_are_log,eps", [(True, 1e-40), (True, 0.0), (False, 1e-40), (False, 1e-6)])
def test_vectorised_attack_is_bit_identical_to_sequential(scores_are_log, eps):
    rng = np.random.default_rng(11)
    n = 101
    plaintexts = rng.integers(0, 256, size=n, dtype=np.uint8)
    logits = rng.normal(0, 3.0, size=(n, KEYSPACE_SIZE))
    log_probs = torch.log_softmax(torch.tensor(logits), dim=1).numpy()
    scores = log_probs if scores_are_log else np.exp(log_probs)
    meta = _metadata(plaintexts, TRUE_KEY)
    evidence = fixed_key_attack_evidence(
        preds=torch.tensor(scores), metadata=meta, target_byte=TARGET_BYTE, step=7,
        eps=eps, preds_are_log_probs=scores_are_log,
    )
    ref_curve, ref_scores = _reference_attack(
        scores, plaintexts, TRUE_KEY, step=7, eps=eps, scores_are_log=scores_are_log
    )
    assert np.array_equal(evidence["curve"], ref_curve)
    assert np.array_equal(evidence["final_key_scores"], ref_scores)


def test_ldd_vectorised_sum_is_bit_identical_to_sequential():
    rng = np.random.default_rng(5)
    n = 77
    plaintexts = rng.integers(0, 256, size=n, dtype=np.uint8)
    log_probs = torch.log_softmax(torch.tensor(rng.normal(0, 2.0, size=(n, KEYSPACE_SIZE))), dim=1).numpy()
    scores = compute_key_log_probs(
        torch.tensor(log_probs), _metadata(plaintexts, TRUE_KEY),
        target_byte=TARGET_BYTE, leakage_model="ID", preds_are_log_probs=True,
    )
    _curve, ref_scores = _reference_attack(log_probs, plaintexts, TRUE_KEY, step=n, eps=1e-40, scores_are_log=True)
    assert np.array_equal(scores, ref_scores)


def test_fast_metadata_extraction_matches_validated_path():
    from sca.rank_curve import _extract_target_bytes, _extract_target_bytes_fast, _get_meta_byte
    rng = np.random.default_rng(2)
    plaintexts = rng.integers(0, 256, size=40, dtype=np.uint8)
    meta = _metadata(plaintexts, TRUE_KEY)
    fast = _extract_target_bytes_fast(meta, TARGET_BYTE)
    assert fast is not None
    slow_pt = np.array([_get_meta_byte(m, "plaintext", TARGET_BYTE) for m in meta], dtype=np.uint8)
    slow_key = np.array([_get_meta_byte(m, "key", TARGET_BYTE) for m in meta], dtype=np.uint8)
    assert np.array_equal(fast[0], slow_pt) and np.array_equal(fast[1], slow_key)
    pt, key = _extract_target_bytes(meta, TARGET_BYTE)
    assert np.array_equal(pt, slow_pt) and np.array_equal(key, slow_key)
    # Rows that are not uint8[16] fall back to the validating path.
    odd = [{"plaintext": m["plaintext"].astype(np.int64), "key": m["key"]} for m in meta]
    assert _extract_target_bytes_fast(odd, TARGET_BYTE) is None
    pt2, _ = _extract_target_bytes(odd, TARGET_BYTE)
    assert np.array_equal(pt2, slow_pt)


def test_first_unshuffled_attack_equals_single_run_curve():
    from sca.perform_attack import perform_attacks
    logits, plaintexts = _synthetic_logits(n=120, scale=2.0)
    meta = np.empty(120, dtype=object)
    for i, row in enumerate(_metadata(plaintexts, TRUE_KEY)):
        meta[i] = row
    log64 = torch.log_softmax(logits.double(), dim=1)
    cfg = {
        "seed": 1, "sca": {"attack_byte": TARGET_BYTE, "leakage_model": "ID", "variant": "fixed_key", "rank_base": 1},
        "perform_attack": {"enabled": True, "attack_step": 10, "ge_thresholds": [1],
                           "validation": {"attack_times": 1, "shuffle": False, "attack_distribution": "full_population"}},
        "metrics_cfg": {"rank_log_eps": 1e-40},
        "data": {"name": "syn", "profiling_range": [0, 10], "attack_range": [0, 120], "validation_range": [0, 120], "test_range": [0, 120]},
    }
    summary = perform_attacks(preds=log64, metadata=meta, cfg=cfg, process="validation", preds_are_log_probs=True)
    single = full_ranks_curve(preds=log64, metadata=meta, target_byte=TARGET_BYTE, step=10, preds_are_log_probs=True)
    assert np.array_equal(np.asarray(summary["per_attack_curves"][0], dtype=np.float64), single)


# ---------------------------------------------------------------------------
# LDD / Lm (Wu, Perin, Picek, "I Choose You", Eq. 8-10)
# ---------------------------------------------------------------------------
def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    below = (values[None, :] < values[:, None]).sum(axis=1)
    not_above = (values[None, :] <= values[:, None]).sum(axis=1)
    return (below + not_above + 1) / 2.0


def _hw_posterior(plaintexts: np.ndarray, key: int, seed: int = 3) -> np.ndarray:
    """p(HW class | leak) for leak = HW(Sbox(p ^ key)) + noise, with the binomial class prior."""
    rng = np.random.default_rng(seed)
    labels = build_label_table("HW")[plaintexts, key].astype(np.float64)
    leak = labels + rng.normal(0.0, 2.0, size=len(plaintexts))
    likelihood = np.exp(-0.5 * ((leak[:, None] - np.arange(9)[None, :]) / 2.0) ** 2)
    posterior = likelihood * np.array([math.comb(8, h) for h in range(9)]) / 256.0
    return posterior / posterior.sum(axis=1, keepdims=True)


def _lm_cfg(leakage_model: str, **sca) -> dict:
    return {"sca": {"attack_byte": TARGET_BYTE, "leakage_model": leakage_model, "rank_base": 1, **sca},
            "metrics_cfg": {"rank_log_eps": 1e-40}}


@pytest.mark.parametrize("leakage_model", ["ID", "HW", "MSB"])
def test_ldd_sums_over_the_attack_traces(leakage_model):
    from sca.ldd import calculate_ldd

    # Eq. 8: sum over the traces' data values d_i, repeated plaintexts counted each time.
    plaintexts = np.array([7, 7, 7, 200, 13], dtype=np.uint8)
    table = build_label_table(leakage_model).astype(np.float64)
    expected = sum((table[int(p), TRUE_KEY] - table[int(p)]) ** 2 for p in plaintexts)

    ldd = calculate_ldd(TRUE_KEY, leakage_model, plaintexts=plaintexts)
    assert np.array_equal(ldd, expected) and ldd[TRUE_KEY] == 0.0
    # Not the sum over the 256 possible plaintext values.
    assert not np.array_equal(ldd, ((table[:, [TRUE_KEY]] - table) ** 2).sum(axis=0))


def test_ldd_requires_the_attack_plaintexts():
    from sca.ldd import calculate_ldd

    with pytest.raises(TypeError):
        calculate_ldd(TRUE_KEY, "HW")
    with pytest.raises(ValueError, match="at least one attack trace"):
        calculate_ldd(TRUE_KEY, "HW", plaintexts=np.array([], dtype=np.uint8))


@pytest.mark.parametrize("leakage_model", ["ID", "HW", "MSB"])
def test_lm_key_scores_are_the_attacks_key_scores(leakage_model):
    rng = np.random.default_rng(9)
    n = 120
    plaintexts = rng.integers(0, 256, size=n, dtype=np.uint8)
    classes = {"ID": 256, "HW": 9, "MSB": 2}[leakage_model]
    log_probs = torch.log_softmax(torch.tensor(rng.normal(0, 2.0, size=(n, classes))), dim=1)
    meta = _metadata(plaintexts, TRUE_KEY)

    scores = compute_key_log_probs(log_probs, meta, target_byte=TARGET_BYTE, leakage_model=leakage_model,
                                   preds_are_log_probs=True)
    evidence = fixed_key_attack_evidence(preds=log_probs, metadata=meta, target_byte=TARGET_BYTE, step=n,
                                         leakage_model=leakage_model, preds_are_log_probs=True)
    # HW: log p(z|t) - log p(z), the class prior the rank computation applies.
    assert np.array_equal(scores, evidence["final_key_scores"])


def test_lm_follows_the_paper_for_hw():
    from sca.ldd import compute_ldd_lm

    rng = np.random.default_rng(4)
    plaintexts = rng.integers(0, 256, size=300, dtype=np.uint8)
    posterior = _hw_posterior(plaintexts, TRUE_KEY)
    meta = _metadata(plaintexts, TRUE_KEY)
    out = compute_ldd_lm(torch.tensor(np.log(posterior)), meta, _lm_cfg("HW"), preds_are_log_probs=True)

    table = build_label_table("HW").astype(np.float64)
    ldd = ((table[plaintexts][:, [TRUE_KEY]] - table[plaintexts]) ** 2).sum(axis=0)     # Eq. 8
    attack = fixed_key_attack_evidence(preds=torch.tensor(np.log(posterior)), metadata=meta,
                                       target_byte=TARGET_BYTE, step=300, leakage_model="HW",
                                       preds_are_log_probs=True)["final_key_scores"]
    expected = np.corrcoef(_average_ranks(ldd), 256.0 - _average_ranks(attack))[0, 1]   # Eq. 10
    assert np.array_equal(out["LDD"], ldd)
    assert math.isclose(out["Lm"], expected, rel_tol=1e-12)
    assert out["Lm"] > 0.4          # 0.16 with the 256-value LDD and no class prior
