# services/side_channel_core/tests/unit/test_attack_trace_budget.py
"""perform_attack.<process>.attack_trace_budget (owner 2026-10-26): the traces of one attack.

full_population: attack r takes the first B traces of its own seeded permutation, so its curve is
exactly the first B points of attack r drawn without a budget. disjoint: attack_times blocks of B
traces. The key evidence stays one attack on every trace. Empty / absent: every trace, the hash of
every run made before the key."""

import copy
from pathlib import Path

import numpy as np
import pytest

from test_audit_2026_09_28 import _attack_cfg, _leaky_attack

N = 400


def _plan(times=20, shuffle=True, distribution="full_population", budget=None):
    plan = {"attack_times": times, "shuffle": shuffle, "attack_distribution": distribution}
    if budget is not None:
        plan["attack_trace_budget"] = budget
    return plan


def _attack(plan, step=10, n=N):
    from services.side_channel_core.src.sca.perform_attack import perform_attacks

    preds, meta = _leaky_attack(n=n)
    return perform_attacks(preds, meta, _attack_cfg(n, plan, step=step), process="test", preds_are_log_probs=True)


# ------------------------------------------------------------------ the attack
@pytest.mark.parametrize("step, budget", [(10, 150), (1, 37), (10, 390)])
def test_a_budget_attack_is_the_first_b_points_of_the_attack_without_one(step, budget):
    full = _attack(_plan(), step=step)
    cut = _attack(_plan(budget=budget), step=step)
    for whole, part in zip(full["per_attack_curves"], cut["per_attack_curves"]):
        assert part[-1, 0] == budget
        np.testing.assert_array_equal(part, whole[whole[:, 0] <= budget])


def test_ge_and_sr_at_the_budget_are_over_the_attacks():
    result = _attack(_plan(budget=60))
    final = np.asarray([curve[-1, 1] for curve in result["per_attack_curves"]])
    assert result["guessing_entropy"] == pytest.approx(final.mean())
    assert result["sr_trials"] == 20 and result["successful_attacks"] == int((final <= 1).sum())
    protocol = result["attack_protocol"]
    assert protocol["attack_trace_budget"] == 60 and protocol["traces_per_attack"] == 60
    assert protocol["endpoint_rank_order_invariant"] is False
    assert protocol["sampling_mode"] == "random_permutation_first_budget_traces"
    assert protocol["endpoint_semantics"] == "mean_final_rank_over_overlapping_budget_attacks"
    assert 60 < protocol["traces_attacked"] <= N


def test_the_key_evidence_is_one_attack_on_every_trace():
    full = _attack(_plan())["__key_recovery_evidence__"]
    cut = _attack(_plan(budget=60))["__key_recovery_evidence__"]
    for field in ("final_true_key_rank", "top_key_hypothesis", "traces_used"):
        assert cut[field] == full[field]
    assert cut["traces_used"] == N
    assert cut["ranking_basis"] == "first_attack_order_on_all_traces"
    assert cut["final_ge"] == pytest.approx(np.mean(cut["final_ranks"]))            # per-attack, at B
    assert len(cut["final_ranks"]) == 20 and cut["sr_trials"] == 20


def test_a_budget_of_every_trace_is_the_attack_without_one():
    full = _attack(_plan())
    same = _attack(_plan(budget=N))
    np.testing.assert_array_equal(np.asarray(full["per_attack_curves"]), np.asarray(same["per_attack_curves"]))
    assert same["sr_trials"] == 1 and same["attack_protocol"]["endpoint_rank_order_invariant"] is True


def test_disjoint_blocks_of_b_leave_the_rest_unattacked():
    result = _attack(_plan(times=3, shuffle=False, distribution="disjoint", budget=100))
    protocol = result["attack_protocol"]
    assert protocol["traces_per_attack"] == 100 and protocol["traces_attacked"] == 300
    assert result["sr_trials"] == 3
    one = _attack(_plan(times=1, shuffle=False))["__key_recovery_evidence__"]
    evidence = result["__key_recovery_evidence__"]           # the partition in sequence: every trace, in order
    for field in ("final_true_key_rank", "top_key_hypothesis", "first_rank1_trace", "stable_rank1_trace"):
        assert evidence[field] == one[field]


@pytest.mark.parametrize("plan, match", [
    (_plan(budget=N + 1), "exceeds the 400 traces"),
    (_plan(times=5, distribution="disjoint", budget=81), "need 405 traces"),
    (_plan(budget=0), "attack_trace_budget"),
    (_plan(budget=True), "attack_trace_budget"),
])
def test_a_budget_the_population_cannot_hold_is_refused(plan, match):
    with pytest.raises((ValueError, TypeError), match=match):
        _attack(plan)


def test_the_flat_budget_stays_retired():
    cfg = _attack_cfg(N, _plan())
    cfg["perform_attack"]["attack_trace_budget"] = 100
    from services.side_channel_core.src.sca.perform_attack import attack_plan

    with pytest.raises(ValueError, match="Obsolete perform_attack"):
        attack_plan(cfg, "test")


# ------------------------------------------------------------------ the key-recovery report
def test_a_budget_report_builds_and_reads_back(tmp_path):
    from src.sca.key_recovery import build_key_recovery_report
    from src.sca.perform_attack import perform_attacks

    preds, meta = _leaky_attack()
    from src.sca.key_recovery import _validate_base_key_recovery_report

    cfg = _attack_cfg(N, None)
    cfg["perform_attack"].pop("test")
    cfg["perform_attack"]["validation"] = _plan(budget=60)
    cfg["data"] = {"name": "ascad", "validation_range": [0, N]}
    cfg["key_recovery_reporting"] = {"enabled": True, "recovery_rank": 1, "require_stable_recovery": True}
    evidence = perform_attacks(preds, meta, cfg, process="validation",
                               preds_are_log_probs=True)["__key_recovery_evidence__"]
    report = build_key_recovery_report(cfg=cfg, process="validation", evidence=evidence, epoch=1,
                                       model_name="m", model_hash="abcd1234")
    _validate_base_key_recovery_report(report)                 # the check a persisted report is read back with
    recovery = report["recovery"]
    assert recovery["attack_trace_budget"] == 60 and recovery["endpoint_rank_order_invariant"] is False
    assert recovery["final_ge"] == pytest.approx(np.mean(recovery["final_ranks"]))


# ------------------------------------------------------------------ cross-validation
def test_a_fold_validation_curve_ends_at_the_budget():
    from utils import cross_validation as cv

    cfg = {"perform_attack": {"enabled": True, "validation": _plan(times=10, budget=50)}}
    assert cv.validation_traces_per_attack(cfg, 500) == 50
    with pytest.raises(cv.CrossValidationError, match="exceeds the 40 traces"):
        cv.validation_traces_per_attack(cfg, 40)


# ------------------------------------------------------------------ identity
def test_no_budget_keeps_the_hash_and_a_budget_changes_it(monkeypatch):
    from test_final_identity import _cfg
    from utils.model_naming import make_experiment_hash

    monkeypatch.setenv("WORKSPACE_ROOT", "/app")
    base = _cfg(Path("/app"))
    base["perform_attack"]["test"] = _plan()
    before = make_experiment_hash(base)
    shown_empty = copy.deepcopy(base)
    shown_empty["perform_attack"]["test"]["attack_trace_budget"] = None       # a snapshot shows it empty
    budgeted = copy.deepcopy(base)
    budgeted["perform_attack"]["test"]["attack_trace_budget"] = 1500
    assert make_experiment_hash(shown_empty) == before != make_experiment_hash(budgeted)


# ------------------------------------------------------------------ the key's place: the data entry
def _request(tmp_path, test):
    import yaml

    path = tmp_path / "test_config.yaml"
    path.write_text(yaml.safe_dump({"test_request": {"model": ["m"], "perform_attack": {"test": test}}}))
    return path




def test_the_budget_is_the_data_entrys_never_perform_attacks():
    """Owner 2026-10-27: perform_attack.<process>.attack_trace_budget retired; empty (a file the panel completed)
    is accepted, a value is refused naming data[].attack_trace_budget."""
    from services.side_channel_core.utils.config_errors import ConfigError
    from services.side_channel_core.utils.config_schema import RETIRED, refuse_moved

    retired = RETIRED["params.yaml"]
    assert {"perform_attack.attack_trace_budget", "perform_attack.test.attack_trace_budget",
            "perform_attack.validation.attack_trace_budget"} <= retired
    refuse_moved({"perform_attack": {"test": {"attack_times": 1, "attack_trace_budget": None}}}, source="params.yaml")
    for process in ("test", "validation"):
        with pytest.raises(ConfigError, match=r"moved to data\[\]\.attack_trace_budget"):
            refuse_moved({"perform_attack": {process: {"attack_trace_budget": 250}}}, source="params.yaml")


def test_a_test_config_budget_is_refused(tmp_path):
    from src.sca.test_request import TestRequestError, external_settings

    with pytest.raises(TestRequestError, match="the budget is the data entry's"):
        external_settings(_request(tmp_path, {"attack_times": 10, "attack_trace_budget": 300}), "m")
    assert "attack_trace_budget" not in external_settings(_request(tmp_path, {"attack_times": 10}), "m")["test"]


def test_the_snapshot_carries_no_budget(tmp_path):
    import yaml
    from utils import data_reproducibility as dr
    from test_reproduce import _cfg

    cfg = _cfg("cake_bacon", "816fd431")
    cfg["perform_attack"] = {"enabled": True, "test": _plan()}
    out = tmp_path / "snap.yaml"
    dr.write_params_snapshot(cfg, out)
    assert "attack_trace_budget" not in yaml.safe_load(out.read_text())["perform_attack"]["test"]
