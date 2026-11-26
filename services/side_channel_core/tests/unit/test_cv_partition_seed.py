# services/side_channel_core/tests/unit/test_cv_partition_seed.py
"""cross_validation.partition_seed (W1 of the Zaid-Wouters critique): the fold partition has its own seed,
so a sweep over seed keeps the folds and varies only the training randomness, and a sweep over
partition_seed varies only the folds. Absent, the partition comes from seed, as before."""

import copy

import numpy as np
import pytest

from test_final_cv import _cv_cfg

N = 200


def _cfg(seed, partition_seed=None, shuffle=True):
    cfg = _cv_cfg({"type": "none"})
    cfg.update(seed=seed, paths={"cross_validation": "cv"})
    cfg["cross_validation"] = {"enabled": True, "n_folds": 5, "strategy": "k_fold", "shuffle": shuffle}
    if partition_seed is not None:
        cfg["cross_validation"]["partition_seed"] = partition_seed
    return cfg


def _held_out(cfg):
    from utils.cross_validation import create_fold_plans

    return [tuple(fold.validation_indices.tolist()) for fold in create_fold_plans(np.zeros(N, dtype=np.int64), cfg)]


def test_the_training_seed_no_longer_moves_the_folds_when_the_partition_has_its_own_seed():
    assert _held_out(_cfg(42, partition_seed=7)) == _held_out(_cfg(43, partition_seed=7))     # seed sweep: same folds
    assert _held_out(_cfg(42, partition_seed=7)) != _held_out(_cfg(42, partition_seed=8))     # partition sweep: new folds


def test_without_the_key_the_partition_is_the_one_seed_drew_before():
    assert _held_out(_cfg(42)) == _held_out(_cfg(42, partition_seed=42))
    assert _held_out(_cfg(42)) != _held_out(_cfg(43))                                          # as every run so far


def test_the_training_seeds_stay_the_seeds():
    from utils.cross_validation import fold_seed, validate_cross_validation_config

    spec = validate_cross_validation_config(_cfg(42, partition_seed=7))
    assert (spec.seed, spec.partition, spec.partition_seed) == (42, 7, 7)
    assert [fold_seed(spec.seed, i) for i in range(5)] == [43, 44, 45, 46, 47]
    assert validate_cross_validation_config(_cfg(42)).partition == 42


@pytest.mark.parametrize("value, error", [(-1, ValueError), (True, TypeError), ("x", TypeError)])
def test_a_partition_seed_is_a_non_negative_integer(value, error):
    from utils.cross_validation import validate_cross_validation_config

    with pytest.raises(error):
        validate_cross_validation_config(_cfg(42, partition_seed=value))


def test_a_partition_seed_without_shuffling_is_refused():
    from utils.cross_validation import validate_cross_validation_config

    with pytest.raises(ValueError, match="partition_seed is set but cross_validation.shuffle is false"):
        validate_cross_validation_config(_cfg(42, partition_seed=7, shuffle=False))


def test_the_key_joins_the_hash_only_when_set(tmp_path, monkeypatch):
    from test_final_identity import _cfg as identity_cfg
    from utils.model_naming import make_experiment_hash

    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    base = identity_cfg(tmp_path)
    base["cross_validation"] = {"enabled": True, "n_folds": 10, "strategy": "k_fold", "shuffle": True}
    before = make_experiment_hash(base)
    with_seed = copy.deepcopy(base)
    with_seed["cross_validation"]["partition_seed"] = 7
    other = copy.deepcopy(base)
    other["cross_validation"]["partition_seed"] = 8
    assert make_experiment_hash(copy.deepcopy(base)) == before                                 # absent: unchanged
    assert len({before, make_experiment_hash(with_seed), make_experiment_hash(other)}) == 3


def test_partition_seed_is_the_one_cross_validation_list_that_sweeps():
    from utils.sweep_engine import _find_sweep_paths

    cfg = {"seed": [42, 43], "cross_validation": {"enabled": True, "partition_seed": [1, 2, 3], "n_folds": [10],
                                                  "refit": {"enabled": [True]}}}
    assert dict(_find_sweep_paths(cfg)) == {("seed",): [42, 43], ("cross_validation", "partition_seed"): [1, 2, 3]}


def _four_folds(curves):
    """A 4-fold plan of 40 profiling traces (10 held out each) whose folds end at the given rank curves."""
    import types

    import utils.cross_validation as cv

    held, epochs = 10, 3
    folds = [types.SimpleNamespace(fold_index=i, validation_indices=np.arange(i * held, (i + 1) * held),
                                   train_indices=np.setdiff1d(np.arange(4 * held), np.arange(i * held, (i + 1) * held)))
             for i in range(4)]
    plan = types.SimpleNamespace(spec=types.SimpleNamespace(n_folds=4, seed=42, partition=42), folds=folds,
                                 paths=types.SimpleNamespace(aggregate="/a", fold_indices="/a/fold_indices.npz"))
    cfg = {"trainer": {"epochs": epochs}, "sca": {"attack_byte": 2}, "metrics_to_log": ["mean_attack_rank_curve"],
           "perform_attack": {"validation": {"attack_times": 1, "shuffle": False,
                                             "attack_distribution": "full_population"}, "ge_thresholds": 1}}
    results = [{"fold_index": f.fold_index, "fold_number": f.fold_index + 1, "n_folds": 4,
                "seed": cv.fold_seed(42, f.fold_index), "epochs_completed": epochs, "epochs_configured": epochs,
                "final_validation": {"epoch": epochs, "metrics": {"mean_attack_rank_curve": curve},
                                     "population_source": "Profiling_traces", "validation_indices": f.validation_indices,
                                     "profiling_count": 4 * held}} for f, curve in zip(folds, curves)]
    return cv, plan, cfg, results


def test_the_cv_headline_reports_the_fold_median_and_its_spread_beside_the_mean():
    """B1 of the critique: the median over the folds (independent trainings), its quartiles and std."""
    recovering = [[float(n), float(max(1, 50 - 10 * n))] for n in range(1, 11)]       # rank 1 from 5 traces
    outlier = [[float(n), 200.0] for n in range(1, 11)]                               # one fold never recovers
    cv, plan, cfg, results = _four_folds([outlier, recovering, recovering, recovering])
    aggregate = cv._aggregate_final_fold_curves(plan, cfg, results)
    ge_1 = aggregate["traces_to_ge"]["1"]
    assert ge_1["mean_curve_stable_trace"] is None                    # the outlier holds the mean above 1
    assert ge_1["median_curve_stable_trace"] == 5                     # the median follows the three folds
    headline = cv.build_cv_headline(plan, cfg=cfg, fold_results=results, aggregate={"rank_curve": aggregate})
    assert headline["median_curve_final_rank"] == 1.0 and headline["traces_to_ge_1_median_curve"] == 5
    assert (headline["final_rank_q25"], headline["final_rank_q75"]) == (1.0, 50.75)   # endpoints [200, 1, 1, 1]
    assert headline["final_rank_std"] == pytest.approx(99.5)          # sample std, ddof = 1
    assert headline["traces_to_ge_1_mean_curve"] is None              # the mean headline stays as it was


def test_an_empty_partition_seed_is_no_partition_seed(tmp_path, monkeypatch):
    """The template lists the key empty, so the panel shows it: empty changes neither the folds nor the hash."""
    from test_final_identity import _cfg as identity_cfg
    from utils.model_naming import make_experiment_hash

    empty = _cfg(42)
    empty["cross_validation"]["partition_seed"] = None
    assert _held_out(empty) == _held_out(_cfg(42))
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    base = identity_cfg(tmp_path)
    base["cross_validation"] = {"enabled": True, "n_folds": 10, "strategy": "k_fold", "shuffle": True}
    blank = copy.deepcopy(base)
    blank["cross_validation"]["partition_seed"] = None
    assert make_experiment_hash(blank) == make_experiment_hash(base)
