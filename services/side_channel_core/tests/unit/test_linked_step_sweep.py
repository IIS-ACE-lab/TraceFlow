"""perform_attack.attack_step and validation.rank_step sweep together (zipped), never as a product."""
import copy

import pytest

from utils.sweep_engine import _sweep_dimensions, count_all_param_combinations, generate_param_combinations


def _cfg(attack_step, rank_step, epochs=(50,)):
    return {"seed": 0, "perform_attack": {"attack_step": attack_step, "ge_thresholds": [1]},
            "validation": {"rank_step": rank_step, "rank_eval_interval": 1},
            "trainer": {"epochs": list(epochs), "batch_size": [100]}}


def _steps(cfg):
    return [(c["perform_attack"]["attack_step"], c["validation"]["rank_step"], c["trainer"]["epochs"])
            for c in generate_param_combinations(cfg)]


def test_lists_pair_up_and_cross_the_other_dimensions():
    cfg = _cfg([5, 10, 50], [5, 10, 50], epochs=(100, 200))
    got = _steps(cfg)
    assert sorted(got) == sorted((s, s, e) for s in (5, 10, 50) for e in (100, 200))     # 6 runs, not 18
    assert count_all_param_combinations(cfg) == len(got) == 6


def test_a_scalar_goes_with_every_entry_of_the_other_list():
    assert sorted(_steps(_cfg([1, 5], 5))) == [(1, 5, 50), (5, 5, 50)]
    assert sorted(_steps(_cfg(10, [10, 50]))) == [(10, 10, 50), (10, 50, 50)]


def test_scalars_and_one_entry_lists_are_one_point():
    assert _steps(_cfg(5, 5)) == [(5, 5, 50)]
    assert _steps(_cfg([5], [5])) == [(5, 5, 50)]                  # [5] runs as the scalar 5: same hash as before
    assert count_all_param_combinations(_cfg([5], [5])) == 1


def test_lists_of_different_lengths_are_refused():
    with pytest.raises(ValueError, match="linked sweep"):
        _sweep_dimensions(_cfg([5, 10], [5, 10, 50]))


def test_an_empty_attack_byte_counts_as_one_value():
    """sca.attack_byte left empty follows the data entry (dataset_types.resolve_attack_byte, before a run's sweep);
    the queue's planned count reads the file before any entry is selected and counts it as one value."""
    base = _cfg([5], [5])
    base["sca"] = {"attack_byte": [], "leakage_model": ["ID"], "candidate_ranking": {"mode": ["none"]}}
    assert count_all_param_combinations(base) == 1
    base["sca"]["leakage_model"] = ["ID", "HW"]
    assert count_all_param_combinations(base) == 2
    base["sca"]["attack_byte"] = [0, 2]
    assert count_all_param_combinations(base) == 4


def test_the_planner_sweeps_the_file_with_the_runs_resolved_attack_byte(monkeypatch):
    """The planner re-reads params.yaml (get_pending_runs, planned_hashes); its empty sca.attack_byte would be refused
    by the sweep, so the byte the run resolved from its data entry (dataset_types.resolve_attack_byte) is carried
    into the file's config before the sweep (owner 2026-10-23: the byte follows the entry; a 2026-10-23 review found
    no ZvW run could be planned without this)."""
    from utils import sweep_engine

    file_cfg = {**_cfg([1], [1]), "sca": {"attack_byte": [], "leakage_model": ["ID"], "candidate_ranking": {"mode": ["none"]}}}
    monkeypatch.setattr(sweep_engine, "get_project_config", lambda name: copy.deepcopy(file_cfg))
    run_cfg = {"project": {"name": "P"}, "sca": {"attack_byte": [2], "leakage_model": ["ID"]}}
    base = sweep_engine._base_cfg_for(run_cfg)
    assert base["sca"]["attack_byte"] == [2]
    combos = list(generate_param_combinations(base))
    assert len(combos) == 1 and combos[0]["sca"]["attack_byte"] == 2
    kept = {**file_cfg, "sca": {**file_cfg["sca"], "attack_byte": [0]}}
    monkeypatch.setattr(sweep_engine, "get_project_config", lambda name: copy.deepcopy(kept))
    assert sweep_engine._base_cfg_for(run_cfg)["sca"]["attack_byte"] == [0], "a byte the file sets is kept"


def test_a_list_under_a_data_entry_is_a_value_not_a_sweep():
    """A run takes one data entry as written (DATASET); a list in it (a range, target points, a
    normalization chain) is never a grid: counted as one it planned runs that could not be made."""
    cfg = _cfg([1], [1])
    cfg["data"] = [{"name": "ascad", "type": "ASCAD_V1_FIXED", "profiling_range": [0, 50000],
                    "target_points": [100, 200, 300], "trace_normalization": {"steps": ["zscore", "minmax"]}}]
    assert [p for p, _ in _sweep_dimensions(cfg) if isinstance(p, tuple) and p[0] == "data"] == []
    assert count_all_param_combinations(cfg) == 1
    assert len(list(generate_param_combinations(cfg))) == 1
