# tests/unit/test_unattacked_test.py
"""
perform_attack.test.attack_times: 0 (2026-09-30)

The test traces are not attacked, so the test process is not run at all:
test_registry.json gets status SKIPPED with the reason, and SKIPPED counts as
a finished test everywhere (orchestrator, signal handler, make test, judge).
With attack_times > 0 the test runs as before.

Cross-validation runs also record the key-recovery outcome of every fold
(held-out part, final epoch) in run_registry.json, as fold_key_recovery.
"""
from __future__ import annotations

import json
import logging
import signal
from pathlib import Path

import pytest

from test_final_orchestration import main_env, orch  # noqa: F401  (orchestrator fixtures)

REASON = "perform_attack.test.attack_times is 0: the test traces are not attacked, so the test process is skipped"
HASH, NAME = "3745aeec", "tulip_fish"


def _entry(status, **extra):
    return {"model_name": NAME, "status": status, "env": "dev", "model_id": "ASCAD_CNNbase",
            "path": f"P/artifacts/dev/generated_models/{NAME}", **extra}


def _registries(root, run_status="in_progress", test_status="queued"):
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "run_registry.json").write_text(json.dumps({HASH: _entry(run_status)}))
    (artifacts / "test_registry.json").write_text(json.dumps({HASH: _entry(test_status)}))


def _read(root, name):
    return json.loads((root / "artifacts" / name).read_text())[HASH]


def _cfg(root, attack_times):
    return {
        "project": {"root": str(root), "name": "P"}, "data": {"name": "ascad"},
        "model_hash": HASH, "model_name": NAME, "model_id": "ASCAD_CNNbase", "runtime": {},
        "paths": {"artifacts_root": f"artifacts/dev/generated_models/{NAME}"},
        "perform_attack": {"validation": {"attack_times": 1}, "test": {"attack_times": attack_times}},
    }


# ------------------------------------------------------------ the registry
def test_skipped_is_a_finished_test_status_with_its_reason(tmp_path):
    from utils.test_registry import skip_unattacked_test, unattacked_test_reason

    assert unattacked_test_reason(_cfg(tmp_path, 1)) is None
    assert unattacked_test_reason(_cfg(tmp_path, 0)) == REASON

    _registries(tmp_path)
    assert skip_unattacked_test(tmp_path, HASH, _cfg(tmp_path, 1)) is None
    assert _read(tmp_path, "test_registry.json")["status"] == "queued"

    assert skip_unattacked_test(tmp_path, HASH, _cfg(tmp_path, 0)) == REASON
    entry = _read(tmp_path, "test_registry.json")
    assert entry["status"] == "SKIPPED" and entry["status_reason"] == REASON
    assert entry["finished_at"]


# ---------------------------------------------------------- train_and_test
@pytest.fixture
def paired_run(tmp_path, monkeypatch):
    """cli_helpers.train_and_test with training and every subprocess recorded."""
    import utils.cli_helpers as helpers

    _registries(tmp_path)
    calls = {"train": 0, "subprocess": [], "summary": 0}
    monkeypatch.setattr(helpers, "start_run_map_entry", lambda **kw: None)
    monkeypatch.setattr(helpers, "complete_run_map_entry", lambda **kw: None)
    monkeypatch.setattr(helpers, "_ensure_train_safe_cfg", lambda cfg, model_id=None: dict(cfg))
    monkeypatch.setattr(helpers, "scc_train", lambda **kw: calls.__setitem__("train", calls["train"] + 1))
    monkeypatch.setattr(helpers.subprocess, "run", lambda cmd, **kw: calls["subprocess"].append(cmd))
    monkeypatch.setattr(helpers, "_log_completed_cycle_key_recovery_summary",
                        lambda **kw: calls.__setitem__("summary", calls["summary"] + 1) or True)
    return helpers, calls, tmp_path


def test_unattacked_test_is_skipped_without_starting_the_test_process(paired_run, caplog):
    helpers, calls, root = paired_run
    with caplog.at_level(logging.INFO):
        helpers.train_and_test(_cfg(root, 0), "dev")

    assert calls["train"] == 1 and calls["subprocess"] == []        # no test-cmd
    assert calls["summary"] == 1                                    # the summary still explains it
    assert _read(root, "run_registry.json")["status"] == "DONE"
    test = _read(root, "test_registry.json")
    assert test["status"] == "SKIPPED" and test["status_reason"] == REASON
    assert "TESTING SKIPPED" in caplog.text and "attack_times is 0" in caplog.text
    # the settings are logged before the decision reads them: a skipped test's only record of them
    settings = caplog.text.index("Test attack settings: perform_attack.test: attack_times=0")
    assert settings < caplog.text.index("TESTING SKIPPED")


def test_attacked_test_still_runs(paired_run):
    helpers, calls, root = paired_run
    helpers.train_and_test(_cfg(root, 1), "dev")

    assert len(calls["subprocess"]) == 1 and "test-cmd" in calls["subprocess"][0]
    assert _read(root, "test_registry.json")["status"] == "queued"   # the test process owns it from here


# ------------------------------------------------------------ orchestrator
def test_orchestrator_keeps_a_skipped_test_skipped(main_env, monkeypatch, orch):
    from utils.test_registry import skip_unattacked_test

    root, calls, run = main_env
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    runs = {"aaaa": {"model_name": "m-a", "status": "INTERRUPTED", "env": "dev", "model_id": "MLP1",
                     "path": "artifacts/dev/generated_models/m-a"}}
    (root / "artifacts" / "run_registry.json").write_text(json.dumps(runs))
    (root / "artifacts" / "test_registry.json").write_text(json.dumps({"aaaa": dict(runs["aaaa"], status="FAILED")}))
    monkeypatch.setattr(orch, "train_and_test", lambda run_cfg, env, **kw: skip_unattacked_test(
        root, "aaaa", {"perform_attack": {"test": {"attack_times": 0}}}))

    with pytest.raises(SystemExit) as exc:
        run()

    assert exc.value.code == 0
    assert json.loads((root / "artifacts" / "run_registry.json").read_text())["aaaa"]["status"] == "DONE"
    test = json.loads((root / "artifacts" / "test_registry.json").read_text())["aaaa"]
    assert test["status"] == "SKIPPED" and test["status_reason"] == REASON


def test_signal_after_a_skipped_test_finalizes_the_run(tmp_path, monkeypatch, orch):
    _registries(tmp_path, run_status="in_progress", test_status="SKIPPED")
    monkeypatch.setattr(orch, "complete_run_map_entry", lambda **kw: None)
    monkeypatch.setattr(orch, "interrupt_run_map_entry", lambda **kw: None)
    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(orch, "RUN_ENV", "dev")
    monkeypatch.setattr(orch, "SELECTED_HASHES", {HASH})

    with pytest.raises(SystemExit):
        orch.handle_signal(signal.SIGINT, None)

    # Train and (skipped) test were complete: the run is DONE, not INTERRUPTED.
    assert _read(tmp_path, "run_registry.json")["status"] == "DONE"
    assert _read(tmp_path, "test_registry.json")["status"] == "SKIPPED"


# ------------------------------------------------------ make test, the judge
def test_test_sweep_skips_an_unattacked_model(tmp_path, monkeypatch):
    import utils.sweep_test_runner as runner

    _registries(tmp_path, run_status="DONE", test_status="FAILED")
    monkeypatch.setattr(runner, "sync_run_map_before_planning", lambda root: None)
    monkeypatch.setattr(runner, "load_run_map", lambda root: {})
    monkeypatch.setattr(runner, "_build_test_cfg", lambda info, **kw: _cfg(tmp_path, 0))
    monkeypatch.setattr(runner, "configure_global_logging", lambda **kw: tmp_path)
    ran = []

    runner.run_test_sweep(lambda *a, **k: ran.append(1), {"env": "dev", "model_id": "ASCAD_CNNbase"}, "test",
                          tmp_path, include_models=[NAME], retry_failed=False, resume_interrupted=False)

    assert ran == []
    test = _read(tmp_path, "test_registry.json")
    assert test["status"] == "SKIPPED" and test["status_reason"] == REASON
    # A skipped test is finished: a later plain pass does not select it again.
    assert runner._should_test(train_status="DONE", test_status="SKIPPED", include_models=[], force=False,
                               retry_failed=False, resume_interrupted=False) is False


@pytest.mark.parametrize("test_status, rejected", [("DONE", False), ("SKIPPED", False), ("FAILED", True)])
def test_judge_counts_a_skipped_test_as_finished(test_status, rejected):
    from services.model_judge.model_judge import rejection_reason

    candidate = {"env": "dev", "train_status": "DONE", "test_status": test_status,
                 "processes": {"cross_validation": {"mean_curve_final_ge": 1.0}}}
    judge_cfg = {"ranking": {"primary": {"process": "cross_validation", "metric": "mean_curve_final_ge"}},
                 "filters": {"require_done": True}}
    assert (rejection_reason(candidate, judge_cfg, "dev") is not None) is rejected


def test_run_counters_count_a_skipped_test_as_finished():
    script = (Path(__file__).resolve().parents[2] / "scripts" / "orchestrator.sh").read_text()
    counts = script[script.index("_py_counts() {"):script.index("_py_status_count() {")]
    assert '("DONE", "SKIPPED")' in counts


# ------------------------------------ run_registry.json: the folds' outcome
def _report(recovered, rank, top, stable=None):
    return {"key": {"recovered": recovered, "final_true_key_rank": rank, "true_key_byte": 224,
                    "top_key_hypothesis": top}, "target": {"attack_byte": 2},
            "recovery": {"stable_rank1_trace": stable}, "trace_population": {"used": 5000}}


def _record(tmp_path, monkeypatch, **kwargs):
    from utils import key_recovery_helper

    _registries(tmp_path, run_status="DONE", test_status="SKIPPED")
    monkeypatch.setattr(key_recovery_helper, "_split_file", lambda cfg, name: tmp_path / "data" / name)
    cfg = {"project": {"root": str(tmp_path)}, "model_hash": HASH, "model_name": NAME, "runtime": {},
           "paths": {}, "cross_validation": {"enabled": True}, "sca": {"attack_byte": 2}}
    key_recovery_helper._record_key_recovery_registries(
        cfg=cfg, logger=logging.getLogger("t"), validation_report=None, test_report=None,
        test_attacked=False, **kwargs)
    return _read(tmp_path, "run_registry.json")


def test_cv_run_registry_lists_the_key_recovery_of_every_fold(tmp_path, monkeypatch):
    headline = {"fold_reports": [
        {"fold_number": 1, "report": _report(True, 1, 224, stable=150), "problem": None},
        {"fold_number": 2, "report": _report(False, 7, 13), "problem": None},
        {"fold_number": 3, "report": None, "problem": "report damaged (x.json): bad"},
    ]}
    run = _record(tmp_path, monkeypatch, cv_headline=headline)

    # The final fit has no validation attack of its own ...
    assert run["key_recovered"] == "not applicable in CV as no validation training in final fit."
    assert run["true_key"] == "224 (0xE0)"
    # ... the folds do: held-out part, final epoch.
    assert run["fold_key_recovery"] == [
        {"fold": 1, "key_recovered": True, "recovered_key": "224 (0xE0)", "recovered_key_rank": 1,
         "stable_rank_1": "@150 /5000"},
        {"fold": 2, "key_recovered": False, "recovered_key": "13 (0x0D)", "recovered_key_rank": 7,
         "stable_rank_1": "not reached /5000"},
        {"fold": 3, "key_recovered": "not available: report damaged (x.json): bad"},
    ]


def test_cv_run_registry_says_when_the_folds_were_not_attacked(tmp_path, monkeypatch):
    run = _record(tmp_path, monkeypatch, cv_headline=None, validation_attacked=False)
    assert run["fold_key_recovery"] == "not available: perform_attack.validation.attack_times is 0"


def test_the_attack_settings_line_names_every_value_the_decision_and_the_attack_read():
    from utils.test_registry import test_attack_settings

    cfg = {"perform_attack": {"enabled": True, "attack_step": 1, "ge_thresholds": [1],
                              "test": {"attack_times": 100, "shuffle": True, "attack_distribution": "full_population"}}}
    assert test_attack_settings(cfg) == ("perform_attack.test: attack_times=100, shuffle=true, "
                                         "attack_distribution=full_population; perform_attack: enabled=true, "
                                         "attack_step=1, ge_thresholds=[1]")
    assert test_attack_settings({}).startswith("perform_attack.test: attack_times=null")     # an absent block, said so
