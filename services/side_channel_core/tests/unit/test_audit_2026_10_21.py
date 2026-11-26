# services/side_channel_core/tests/unit/test_audit_2026_10_21.py
"""Fixes of the DL-SCA assessment of 2026-10-21 (MET-03, MET-04, MET-05)."""

import numpy as np
import pytest
import torch

from utils.aes import AES_SBOX

_BYTE, _KEY, _N = 2, 0x4D, 400


def _attack_cfg(metrics):
    plan = {"attack_times": 20, "shuffle": True, "attack_distribution": "full_population"}
    return {
        "seed": 3,
        "metrics_to_log": metrics,
        "metrics_cfg": {"preds_are_probs": False, "rank_log_eps": 1e-40},
        "trainer": {"loss": {"type": "CrossEntropyLoss", "args": {"reduction": "mean"}},
                    "evaluation": {"output_transformation": {"type": "Identity"}}},
        "validation": {"rank_step": 10},
        "sca": {"attack_byte": _BYTE, "leakage_model": "ID", "variant": "fixed_key", "rank_base": 1},
        "perform_attack": {"enabled": True, "attack_step": 10, "ge_thresholds": [1, 5],
                           "validation": plan, "test": plan},
        "deep_learning": {"framework": "pytorch"},
        "data": {"name": "syn", "profiling_range": [0, 10], "attack_range": [0, _N],
                 "validation_range": [0, _N], "test_range": [0, _N]},
    }


def _weak_model_attack_set():
    rng = np.random.default_rng(0)
    plaintexts = rng.integers(0, 256, size=_N, dtype=np.uint8)
    labels = np.asarray(AES_SBOX, dtype=np.int64)[plaintexts ^ _KEY]
    logits = torch.tensor(rng.normal(0, 1, size=(_N, 256)))
    logits[np.arange(_N), labels] += 0.22
    metadata = np.empty(_N, dtype=object)
    for row, plaintext in enumerate(plaintexts):
        p, k = np.zeros(16, dtype=np.uint8), np.zeros(16, dtype=np.uint8)
        p[_BYTE], k[_BYTE] = plaintext, _KEY
        metadata[row] = {"plaintext": p, "key": k}
    return logits, torch.tensor(labels), metadata


def test_ge_reduction_rate_is_the_slope_of_ge_not_of_one_attack():
    from sca.metrics import compute_metrics

    logits, labels, metadata = _weak_model_attack_set()
    results = compute_metrics(logits, labels, _attack_cfg(["mean_attack_rank_curve", "ge_reduction_rate"]),
                              metadata=metadata, process="test")
    ge = np.asarray(results["mean_attack_rank_curve"], dtype=np.float64)[:5]
    expected = -np.cov(ge[:, 0], ge[:, 1], ddof=0)[0, 1] / np.var(ge[:, 0])
    assert results["ge_reduction_rate"] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("name", ["traces_to_ge_1", "traces_to_ge_5", "traces_to_ge_k", "first_traces_to_ge_1",
                                  "first_traces_to_ge_k", "traces_to_sr_1", "first_traces_to_sr_1",
                                  "time_to_recover", "time_to_recover@5", "first_time_to_recover@5"])
def test_per_fold_crossings_are_not_averaged_over_folds(name):
    from utils.cross_validation import _is_assessment_scalar

    assert not _is_assessment_scalar(name, 200)


def test_fold_quality_metrics_are_still_averaged():
    from utils.cross_validation import _is_assessment_scalar

    for name in ("guessing_entropy", "accuracy", "traces_to_ge_1_mean_curve", "ge_reduction_rate"):
        assert _is_assessment_scalar(name, 1.0)


def test_key_values_settings_and_the_score_margin_are_not_offered_for_ranking():
    from services.model_judge.model_judge import available_metrics

    test = {"final_true_key_rank": 3, "final_ge": 3.0, "stable_rank1_trace": 900, "true_key_byte": 224,
            "true_key_tie_count": 1, "top_key_hypothesis": 17, "recovery_rank": 1,
            "top1_top2_score_margin": 0.25}
    candidate = {"model_name": "m", "processes": {"test": test}}
    assert available_metrics([candidate]) == {"test": ["final_ge", "final_true_key_rank", "stable_rank1_trace"]}


@pytest.mark.parametrize("momentum", [0.1, 0.01, 0.5])
def test_batchnorm_momentum_and_eps_mean_the_same_in_both_frameworks(momentum):
    """TR-3: one BatchNorm1d entry gives the same running statistics under keras and pytorch."""
    from core import framework_switch as fs
    from core.model import _batch_norm_1d

    entry = {"BatchNorm1d": {"num_features": 1, "momentum": momentum, "eps": 1e-3}}
    keras_model = fs.build_keras_model(
        [entry, {"Flatten": {}}, {"Linear": {"in_features": 4, "out_features": 2}}], input_shape=(1, 4),
        weight=fs.InitSpec("glorot_uniform", {}), bias=fs.InitSpec("zeros", {}),
        resolve_out_features=lambda value, index: value, where="test")
    torch_bn = _batch_norm_1d(1, momentum=momentum, eps=1e-3)
    batch = torch.full((8, 1, 4), 10.0)
    keras_model.train()
    torch_bn.train()
    with torch.no_grad():
        keras_model(batch)
        torch_bn(batch)
    layer = next(v for v in keras_model.keras_model.layers if type(v).__name__ == "BatchNormalization")
    moving = next(v for v in keras_model.keras_model.variables if "moving_mean" in v.path)
    assert float(moving.value[0]) == pytest.approx(float(torch_bn.running_mean[0]), rel=1e-6)
    assert layer.epsilon == pytest.approx(torch_bn.eps)


def test_batchnorm_refuses_the_keras_spelling_of_eps():
    from core import framework_switch as fs

    with pytest.raises(ValueError, match="epsilon"):
        fs.build_keras_model(
            [{"BatchNorm1d": {"num_features": 1, "epsilon": 1e-3}}, {"Flatten": {}}], input_shape=(1, 4),
            weight=fs.InitSpec("glorot_uniform", {}), bias=fs.InitSpec("zeros", {}),
            resolve_out_features=lambda value, index: value, where="test")


_AES_HD = {"name": "aes_hd", "type": "AES_HD", "database": "services/side_channel_core/data/AES_HD/AES_HD_data/AES_HD.h5"}


def _aes_hd_cfg(leakage_model, shuffle, attack_times=1):
    return {"data": _AES_HD, "sca": {"attack_byte": [11], "leakage_model": leakage_model},
            "perform_attack": {"validation": {"attack_times": attack_times, "shuffle": shuffle},
                               "test": {"attack_times": 0, "shuffle": False}}}


def test_aes_hd_refuses_msb_whose_keys_tie_inside_each_block_of_256_traces():
    """LBL-01: AES_HD's c[11] = Sbox[row // 256] (Zaid et al.'s re-encoding of the label)."""
    from sca import dataset_types as dt

    dt.check_run_settings(_aes_hd_cfg(["ID", "HW"], True))
    with pytest.raises(dt.DatasetTypeError, match="MSB"):
        dt.check_run_settings(_aes_hd_cfg(["ID", "MSB"], True))


def test_aes_hd_refuses_an_attack_in_file_order_but_not_an_unattacked_process():
    from sca import dataset_types as dt

    with pytest.raises(dt.DatasetTypeError, match="perform_attack.validation.shuffle: true"):
        dt.check_run_settings(_aes_hd_cfg(["ID"], [True, False]))
    dt.check_run_settings(_aes_hd_cfg(["ID"], False, attack_times=0))     # test is not attacked either
