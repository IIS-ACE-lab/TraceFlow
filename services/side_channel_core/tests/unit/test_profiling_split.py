# services/side_channel_core/tests/unit/test_profiling_split.py
"""data[].profiling_split (2026-10-24): the profiling rows a run trains on, drawn from profiling_range by
scikit-learn's train_test_split as written when enabled (Wouters et al. TCHES 2020(3): test_size 0.1,
random_state 0). Absent or enabled: false, a run trains on all of profiling_range as before, and its hash does
not move."""

import copy
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from sca import dataset_types as dt
from sca.ascad_data_loader import SCADataContractError, profiling_indices

WOUTERS = {"enabled": True, "test_size": 0.1, "shuffle": True, "random_state": 0}


def _entry(**extra):
    from test_dataset_types import _entry as ascad_entry

    return ascad_entry(**extra)


def test_without_the_key_or_disabled_every_row_of_profiling_range_is_trained_on():
    np.testing.assert_array_equal(profiling_indices({"profiling_range": [0, 45000]}), np.arange(0, 45000))
    np.testing.assert_array_equal(profiling_indices({"profiling_range": [100, 200]}), np.arange(100, 200))
    for split in ({"enabled": False}, {**WOUTERS, "enabled": False}, {"enabled": False, "test_size": 7}):
        np.testing.assert_array_equal(profiling_indices({"profiling_range": [0, 300], "profiling_split": split}),
                                      np.arange(0, 300))


@pytest.mark.parametrize("pool, trained", [(50000, 45000), (4500, 4050), (25000, 22500)])
def test_the_rows_are_scikit_learns_training_part_ascending(pool, trained):
    rows = profiling_indices({"profiling_range": [0, pool], "profiling_split": WOUTERS})
    expected, _held_out = train_test_split(np.arange(pool), test_size=0.1, shuffle=True, random_state=0)
    assert rows.size == trained
    np.testing.assert_array_equal(rows, np.sort(expected))


def test_the_rows_are_the_ones_train_test_split_picks_from_the_traces_themselves():
    """Wouters' call splits the trace array; the rows it keeps depend on the row count and random_state only."""
    traces = np.random.default_rng(1).random((5000, 7))
    kept, _held_out = train_test_split(traces, test_size=0.1, random_state=0)
    rows = profiling_indices({"profiling_range": [0, 5000], "profiling_split": WOUTERS})
    np.testing.assert_array_equal(np.sort(kept, axis=0), np.sort(traces[rows], axis=0))


def test_shuffle_false_keeps_the_first_rows_and_an_offset_range_is_respected():
    first = profiling_indices({"profiling_range": [0, 1000], "profiling_split": {**WOUTERS, "shuffle": False}})
    np.testing.assert_array_equal(first, np.arange(0, 900))
    offset = profiling_indices({"profiling_range": [1000, 2000], "profiling_split": WOUTERS})
    assert offset.min() >= 1000 and offset.max() < 2000 and offset.size == 900


def test_an_entry_takes_the_key_enabled_with_all_three_arguments_or_disabled_alone():
    assert dt.validate_entry(_entry(profiling_split=WOUTERS)).name == "ASCADf"
    assert dt.validate_entry(_entry(profiling_split={**WOUTERS, "test_size": 5000})).name == "ASCADf"
    assert dt.validate_entry(_entry(profiling_split={"enabled": False})).name == "ASCADf"
    assert dt.validate_entry(_entry(profiling_split={**WOUTERS, "enabled": False, "test_size": 2.0})).name == "ASCADf"


@pytest.mark.parametrize("split, message", [
    ({}, "enabled must be true or false, not None"),
    ({"test_size": 0.1, "shuffle": True, "random_state": 0}, "enabled must be true or false"),
    ({"enabled": "yes"}, "enabled must be true or false"),
    ({"enabled": True}, r"missing \['test_size', 'shuffle'\]"),
    ({"enabled": True, "test_size": 0.1}, r"missing \['shuffle'\]"),
    ({"enabled": False, "train_size": 0.9}, r"does not take \['train_size'\]"),
    ({**WOUTERS, "train_size": 0.9}, r"does not take \['train_size'\]"),
    ({**WOUTERS, "test_size": 1.5}, "not a fraction in"),
    ({**WOUTERS, "test_size": 0}, "holds out no row"),
    ({**WOUTERS, "test_size": 50000}, "none is left to train on"),
    ({**WOUTERS, "test_size": "0.1"}, "a fraction in"),
    ({**WOUTERS, "shuffle": "yes"}, "true or false"),
    ({**WOUTERS, "random_state": -1}, "non-negative integer"),
    ({**WOUTERS, "random_state": True}, "non-negative integer"),
    ("0.9:0.1", "must be a mapping"),
])
def test_a_wrong_split_is_refused_when_the_configuration_loads(split, message):
    with pytest.raises(dt.DatasetTypeError, match=message):
        dt.validate_entry(_entry(profiling_split=split))
    with pytest.raises(SCADataContractError, match=message):
        profiling_indices({"profiling_range": [0, 50000], "profiling_split": split})


def test_cross_validation_of_an_entry_with_a_split_is_refused():
    entry = _entry(type="ASCADf", profiling_split=WOUTERS)
    cfg = {"data": entry, "sca": {}, "cross_validation": {"enabled": True}}
    with pytest.raises(dt.DatasetTypeError, match="cross-validation partitions profiling_range itself"):
        dt.require_cross_validation_settings(cfg)
    dt.require_cross_validation_settings({**cfg, "cross_validation": {"enabled": False}})
    disabled = {**cfg, "data": _entry(type="ASCADf", profiling_split={**WOUTERS, "enabled": False})}
    dt.require_cross_validation_settings(disabled)
    dt.require_cross_validation_settings({"data": _entry(type="ASCADf"), "sca": {}, "cross_validation": {"enabled": True}})


def test_the_split_joins_the_hash_only_when_enabled(monkeypatch):
    """Absent, enabled: false (the blueprint's form, its arguments empty or not), the entry hashes as before:
    every run made without the key keeps its hash. Enabled, each argument changes it."""
    from test_final_identity import _cfg
    from utils.config_loader import drop_empty_keys
    from utils.model_naming import make_experiment_hash

    monkeypatch.setenv("WORKSPACE_ROOT", "/app")
    base_cfg = _cfg(Path("/app"))
    base = make_experiment_hash(base_cfg)
    assert base == "2bb8d827"                    # pinned: the identity a run without the key has had all along

    for split in ({"enabled": False, "test_size": None, "shuffle": None, "random_state": None},
                  {"enabled": False}, {**WOUTERS, "enabled": False},
                  {"enabled": None, "test_size": None, "shuffle": None, "random_state": None}):
        off = copy.deepcopy(base_cfg)
        off["data"]["profiling_split"] = split
        assert make_experiment_hash(drop_empty_keys(off)) == base, split

    hashes = set()
    for split in (WOUTERS, {**WOUTERS, "random_state": 1}, {**WOUTERS, "test_size": 0.2}, {**WOUTERS, "shuffle": False}):
        cfg = copy.deepcopy(base_cfg)
        cfg["data"]["profiling_split"] = split
        hashes.add(make_experiment_hash(cfg))
    assert len(hashes) == 4 and base not in hashes


def test_random_state_null_cuts_unseeded_a_new_cut_every_run_the_run_seed_does_not_seed_it():
    """Zaid et al.'s scripts: random.shuffle without a seed. random_state null (or left out) draws from the operating
    system, so two draws differ even with NumPy's global generator seeded alike (utils/seed.set_seed)."""
    entry = {"profiling_range": [0, 50000], "profiling_split": {"enabled": True, "test_size": 5000, "shuffle": True, "random_state": None}}
    assert dt.validate_entry(_entry(profiling_split=entry["profiling_split"])).name == "ASCADf"
    assert not dt.profiling_split_reproducible(entry)
    np.random.seed(100)
    first = profiling_indices(entry)
    np.random.seed(100)
    second = profiling_indices(entry)
    assert first.size == second.size == 45000 and not np.array_equal(first, second)
    assert np.all(first[1:] > first[:-1]) and first.min() >= 0 and first.max() < 50000
    assert dt.profiling_split_reproducible({"profiling_split": WOUTERS})
    assert dt.profiling_split_reproducible({"profiling_split": {**WOUTERS, "random_state": None, "shuffle": False}})
    assert dt.profiling_split_reproducible({"profiling_split": {"enabled": False}})


def test_random_state_null_joins_the_hash_as_null(monkeypatch):
    from test_final_identity import _cfg
    from utils.config_loader import drop_empty_keys
    from utils.model_naming import make_experiment_hash

    monkeypatch.setenv("WORKSPACE_ROOT", "/app")
    hashes = []
    for random_state in (None, 0, 100):
        cfg = _cfg(Path("/app"))
        cfg["data"]["profiling_split"] = {**WOUTERS, "test_size": 5000, "random_state": random_state}
        hashes.append(make_experiment_hash(drop_empty_keys(cfg)))   # null: dropped as an empty key, then hashed as null
    assert len(set(hashes)) == 3


def test_a_random_state_list_is_ten_runs_one_split_each_and_nothing_else_under_data_sweeps():
    """Wouters et al. Sect. 3.2: every configuration trained on ten splits. random_state [0..9] makes a run per split
    (utils/sweep_engine: the one list under a data entry that sweeps); a run reads one value, never the list."""
    from utils.sweep_engine import SPLIT_SWEEP_PATH, _find_sweep_paths

    ten = {**WOUTERS, "random_state": list(range(10))}
    assert dt.validate_entry(_entry(profiling_split=ten)).name == "ASCADf"
    assert dt.profiling_split_reproducible({"profiling_split": ten})
    with pytest.raises(dt.DatasetTypeError, match="give distinct values"):
        dt.validate_entry(_entry(profiling_split={**WOUTERS, "random_state": [0, 0]}))
    with pytest.raises(dt.DatasetTypeError, match="or a list of them"):
        dt.validate_entry(_entry(profiling_split={**WOUTERS, "random_state": [0, -1]}))
    with pytest.raises(SCADataContractError, match="the sweep makes one run per value"):
        profiling_indices({"profiling_range": [0, 100], "profiling_split": ten})
    selected = {"seed": None, "data": _entry(profiling_split=ten, attack_range=[0, 10000], test_range=[0, 9000])}
    paths = dict(_find_sweep_paths(selected))
    assert paths == {SPLIT_SWEEP_PATH: list(range(10))}                 # ranges and other data lists stay values
    unselected = {"data": {"a": _entry(profiling_split=ten), "b": _entry()}}
    assert _find_sweep_paths(unselected) == []                          # the file's every entry: no data list sweeps


def test_the_planner_gives_each_run_its_own_split(monkeypatch, tmp_path):
    from utils import sweep_engine as se

    ten = {**WOUTERS, "random_state": [0, 1, 2]}
    cfg = {"project": {"name": "P", "root": str(tmp_path)}, "data": _entry(profiling_split=ten)}
    seen = []
    for sweep_cfg in [{"data": {**cfg["data"], "profiling_split": {**ten, "random_state": i}}} for i in (0, 1, 2)]:
        run_cfg = copy.deepcopy(sweep_cfg)
        chosen = se._get_path(sweep_cfg, se.SPLIT_SWEEP_PATH)
        written = se._get_path(cfg, se.SPLIT_SWEEP_PATH)
        assert chosen in (0, 1, 2) and written == [0, 1, 2]
        run_cfg["data"] = copy.deepcopy(cfg["data"])
        run_cfg["data"]["profiling_split"]["random_state"] = chosen
        seen.append(run_cfg["data"]["profiling_split"]["random_state"])
    assert seen == [0, 1, 2] and cfg["data"]["profiling_split"]["random_state"] == [0, 1, 2]   # the entry as written
