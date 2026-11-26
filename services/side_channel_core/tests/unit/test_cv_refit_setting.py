# services/side_channel_core/tests/unit/test_cv_refit_setting.py
"""cross_validation.refit.enabled (owner ruling 2026-10-22): an internal setting, absent means true,
never part of the model_hash; false skips the final training and leaves no model to test."""

import copy
import logging
import shutil
import types

import pytest
import yaml


def _cfg(refit=None, test_attack_times=0):
    cv = {"enabled": True, "n_folds": 10, "strategy": "k_fold", "shuffle": True}
    if refit is not None:
        cv["refit"] = refit
    return {"cross_validation": cv,
            "perform_attack": {"enabled": True, "test": {"attack_times": test_attack_times, "shuffle": False,
                                                         "attack_distribution": "full_population"}}}


# ------------------------------------------------------------------ the setting
def test_absent_means_the_refit_runs_as_every_run_before_the_key():
    from utils.cross_validation import refit_enabled

    assert refit_enabled(_cfg()) is True
    assert refit_enabled(_cfg({"enabled": True})) is True and refit_enabled(_cfg({})) is True
    assert refit_enabled(_cfg({"enabled": False})) is False


@pytest.mark.parametrize("block, error", [
    ({"enabled": "false"}, TypeError), ({"enabled": [True, False]}, TypeError), ({"enabled": None}, TypeError),
    (False, TypeError), ({"enabled": False, "epochs": 3}, ValueError),
])
def test_only_a_true_or_false_enabled_is_accepted(block, error):
    from utils.cross_validation import refit_enabled

    with pytest.raises(error):
        refit_enabled(_cfg(block))


def test_the_setting_never_joins_the_model_hash(tmp_path, monkeypatch):
    from test_final_identity import _cfg as identity_cfg
    from utils.model_naming import make_experiment_hash

    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    base = identity_cfg(tmp_path)
    base["cross_validation"] = {"enabled": True, "n_folds": 10, "strategy": "k_fold", "shuffle": True}
    hashes = set()
    for refit in (None, {"enabled": True}, {"enabled": False}):
        cfg = copy.deepcopy(base)
        if refit is not None:
            cfg["cross_validation"]["refit"] = refit
        hashes.add(make_experiment_hash(cfg))
    assert len(hashes) == 1


def test_the_schema_accepts_the_key_and_refuses_another_under_it():
    from utils import config_schema

    ConfigError = config_schema.ConfigError                 # the class the schema raises (one import path)
    config_schema.check_keys({"cross_validation": {"enabled": True, "refit": {"enabled": False}}},
                             config_schema.PARAMS, source="params.yaml")
    with pytest.raises(ConfigError):
        config_schema.check_keys({"cross_validation": {"refit": {"enable": False}}},
                                 config_schema.PARAMS, source="params.yaml")


# ------------------------------------------------- the refit and the test go together
def test_no_refit_with_an_attacked_test_is_refused_before_any_fold(caplog):
    from utils.cross_validation import CrossValidationError, CrossValidationSpec, check_refit_against_test

    spec = CrossValidationSpec(enabled=True, n_folds=10, strategy="k_fold", shuffle=True, seed=42, refit=False)
    with caplog.at_level(logging.ERROR), pytest.raises(CrossValidationError) as error:
        check_refit_against_test(_cfg({"enabled": False}, test_attack_times=100), spec)
    assert "cross_validation.refit.enabled: true" in str(error.value)
    assert "perform_attack.test.attack_times: 0" in str(error.value)
    assert any("rejected before any fold" in r.getMessage() for r in caplog.records)
    check_refit_against_test(_cfg({"enabled": False}, test_attack_times=0), spec)       # CV results only


def test_a_refit_no_test_attacks_is_warned_naming_both_keys(caplog):
    from utils.cross_validation import CrossValidationSpec, check_refit_against_test

    spec = CrossValidationSpec(enabled=True, n_folds=10, strategy="k_fold", shuffle=True, seed=42)
    with caplog.at_level(logging.WARNING):
        check_refit_against_test(_cfg(None, test_attack_times=0), spec)
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1
    assert "cross_validation.refit.enabled: false" in warned[0] and "perform_attack.test.attack_times" in warned[0]
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        check_refit_against_test(_cfg(None, test_attack_times=100), spec)               # a refit that is tested
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ------------------------------------------------------------ the pipeline and its records
def _run_pipeline(monkeypatch, refit):
    import utils.cross_validation_pipeline as cvp
    from utils.cross_validation import CrossValidationSpec

    spec = CrossValidationSpec(enabled=True, n_folds=2, strategy="k_fold", shuffle=True, seed=42, refit=refit)
    plan = types.SimpleNamespace(spec=spec, folds=[types.SimpleNamespace(fold_index=0), types.SimpleNamespace(fold_index=1)],
                                 paths=types.SimpleNamespace())
    calls = {"refit": 0, "completed": []}
    monkeypatch.setattr(cvp, "_validate_pipeline_inputs", lambda cfg, **k: (spec, "/tmp/base", 42))
    monkeypatch.setattr(cvp, "prepare_artifacts_data", lambda cfg: None)
    monkeypatch.setattr(cvp, "prepare_and_validate_data", lambda cfg, **k: {})
    monkeypatch.setattr(cvp, "get_cross_validation_labels", lambda cfg: [0, 1, 0, 1])
    monkeypatch.setattr(cvp, "prepare_cross_validation_plan", lambda cfg, labels: plan)
    monkeypatch.setattr(cvp, "update_manifest", lambda paths, patch: None)
    monkeypatch.setattr(cvp, "_connect_tracker", lambda tracker, **k: None)
    monkeypatch.setattr(cvp, "_run_single_fold", lambda cfg, **k: {"fold": k["fold"].fold_index})
    monkeypatch.setattr(cvp, "finalize_cross_validation", lambda plan, **k: {"headline": {}})
    monkeypatch.setattr(cvp, "_run_final_refit", lambda cfg, **k: calls.update(refit=calls["refit"] + 1) or {"r": 1})
    monkeypatch.setattr(cvp, "mark_cross_validation_completed",
                        lambda paths, refit_result=None: calls["completed"].append(refit_result))
    result = cvp.run_cross_validation_pipeline({}, logger=logging.getLogger("t"), model_name="m", model_id="M",
                                               env="dev", base_dir="/tmp/base", seed_used=42)
    return result, calls


def test_the_pipeline_skips_the_refit_and_completes_the_cv(monkeypatch):
    result, calls = _run_pipeline(monkeypatch, refit=False)
    assert calls == {"refit": 0, "completed": [None]} and result["refit"] is None
    assert [f["fold"] for f in result["fold_results"]] == [0, 1]


def test_the_pipeline_runs_the_refit_by_default(monkeypatch):
    result, calls = _run_pipeline(monkeypatch, refit=True)
    assert calls == {"refit": 1, "completed": []} and result["refit"] == {"r": 1}   # the refit marks completion itself


def test_completion_records_a_skipped_refit_and_refuses_a_contradiction(tmp_path):
    from utils.cross_validation import CrossValidationSpec, _refit_section, mark_cross_validation_completed

    paths = types.SimpleNamespace(manifest=tmp_path / "manifest.yaml", summary=tmp_path / "missing.yaml")
    off = CrossValidationSpec(enabled=True, n_folds=2, strategy="k_fold", shuffle=True, seed=42, refit=False)
    paths.manifest.write_text(yaml.safe_dump({"status": "folds_completed", "refit": _refit_section(off, 5)}))
    with pytest.raises(ValueError, match="refit.enabled is false"):
        mark_cross_validation_completed(paths, refit_result={"r": 1})
    manifest = mark_cross_validation_completed(paths, refit_result=None)
    assert manifest["status"] == "completed"
    assert manifest["refit"]["enabled"] is False and manifest["refit"]["status"] == "skipped"
    assert "no model to test" in manifest["refit"]["reason"]

    on = CrossValidationSpec(enabled=True, n_folds=2, strategy="k_fold", shuffle=True, seed=42)
    assert _refit_section(on, 5) == {"enabled": True, "epochs_source": "trainer.epochs", "resolved_epochs": 5,
                                     "status": "ready"}                          # unchanged for every refit
    paths.manifest.write_text(yaml.safe_dump({"status": "folds_completed", "refit": _refit_section(on, 5)}))
    with pytest.raises(ValueError, match="successful refit result"):
        mark_cross_validation_completed(paths, refit_result=None)


def test_the_progress_bar_plans_no_refit_epochs_when_it_is_off():
    from utils.heartbeat import progress_fraction

    running = {"n_folds": 2, "fold_epochs": 10, "phase": "fold", "fold": 2, "epoch": 10, "batch": 4,
               "batches_per_epoch": 4}
    assert progress_fraction(running) == pytest.approx(20 / 31)                   # 2 x 10 + 10 refit + 1 test
    assert progress_fraction({**running, "refit": False}) == pytest.approx(20 / 21)  # 2 x 10 + 1 test


def test_a_cv_run_without_a_refit_is_still_a_judge_candidate(tmp_path):
    from test_model_judge import _candidates, _project

    root = _project(tmp_path, [("a", "aaaa0001", 1.0, 900, 6, 3.0, "DONE")])
    run = root / "artifacts" / "dev" / "generated_models" / "a"
    shutil.rmtree(run / "model")                                                 # no refit: no checkpoint manifest
    (run / "metrics" / "attack_metrics" / "scalars" / "a_metrics_train_history.jsonl").unlink()
    (run / "metrics" / "attack_metrics" / "scalars" / "a_metrics_test_history.jsonl").unlink()
    candidate = _candidates(root)["a"]
    assert set(candidate["processes"]) == {"cross_validation"}
    assert candidate["processes"]["cross_validation"]["traces_to_ge_1_mean_curve"] == 900


def test_a_run_without_a_refit_is_refused_by_every_test_before_its_checkpoint_is_looked_for():
    from utils.cross_validation import require_refit_model

    with pytest.raises(ValueError, match="m cannot be tested: cross_validation.refit.enabled: false"):
        require_refit_model(_cfg({"enabled": False}), "m")
    require_refit_model(_cfg(), "m")                                       # a run made before the key: its refit
    require_refit_model(_cfg({"enabled": True}), "m")
    require_refit_model({"cross_validation": {"enabled": False, "refit": {"enabled": False}}}, "m")   # no CV
