# services/side_channel_core/tests/unit/test_data_entry_budget.py
"""data[].attack_trace_budget (owner 2026-10-27): the traces of one test attack in a Budget test of a run on a data
entry (test_request._apply_kind), left out of the run hash. The project's params.yaml entry of that name as it is now
wins over the snapshot's copy (a run trained before the key gets it)."""

import copy
from pathlib import Path

import pytest
import yaml


def _run(tmp_path, entry_budget=None, params_budget=None, own=None):
    root = tmp_path / "P"
    (root / "config").mkdir(parents=True)
    entry = {"name": "e", "type": "ASCADf"}
    if params_budget is not None:
        entry["attack_trace_budget"] = params_budget
    (root / "config" / "params.yaml").write_text(yaml.safe_dump({"data": [entry, {"name": "other", "attack_trace_budget": 7}]}))
    test = {"attack_times": 100, "shuffle": True, "attack_distribution": "full_population"}
    if own is not None:
        test["attack_trace_budget"] = own
    data = {"name": "e"}
    if entry_budget is not None:
        data["attack_trace_budget"] = entry_budget
    return {"project": {"root": str(root), "params": "config/params.yaml"}, "data": data, "perform_attack": {"test": test}}



def test_the_project_entry_wins_over_the_snapshot_copy_and_the_copy_serves_when_it_is_empty(tmp_path):
    from utils.sweep_test_runner import data_budget

    assert data_budget(_run(tmp_path / "a", entry_budget=1000, params_budget=250)) == 250
    assert data_budget(_run(tmp_path / "b", entry_budget=1000)) == 1000
    assert data_budget(_run(tmp_path / "c")) is None



@pytest.mark.parametrize("bad", [0, -5, True, "250"])
def test_an_entry_budget_must_be_a_trace_count(tmp_path, bad):
    from utils.sweep_test_runner import data_budget

    with pytest.raises(ValueError, match="attack_trace_budget must be an integer"):
        data_budget(_run(tmp_path, params_budget=bad))


def test_every_database_type_takes_the_key_and_the_hash_leaves_it_out(monkeypatch):
    from services.side_channel_core.src.sca.dataset_types import TEST_TIME_KEYS, TYPES
    from test_final_identity import _cfg
    from utils.model_naming import make_experiment_hash

    assert "attack_trace_budget" in TEST_TIME_KEYS
    assert all("attack_trace_budget" in t.allowed for t in TYPES.values())
    monkeypatch.setenv("WORKSPACE_ROOT", "/app")
    base = _cfg(Path("/app"))
    budgeted = copy.deepcopy(base)
    budgeted["data"]["attack_trace_budget"] = 250
    assert make_experiment_hash(budgeted) == make_experiment_hash(base)
