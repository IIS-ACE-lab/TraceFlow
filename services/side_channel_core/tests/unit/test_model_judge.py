# services/side_channel_core/tests/unit/test_model_judge.py
"""Model judge: joining, ranking alignment, directions, filters, outputs.

Run from the repository root:
    PYTHONPATH="$PWD" python -m pytest services/side_channel_core/tests/unit/test_model_judge.py
"""
import json

import pytest
import yaml

from services.model_judge.model_judge import list_choices, metric_direction, run_model_judge

CONFIG = {
    "model_judge": {
        "output_dir": "model_selection",
        "ranking": {
            "N_top_candidates": 10,
            "primary": {"process": "cross_validation", "metric": "mean_curve_final_ge"},
            "tie_breakers": [
                {"process": "cross_validation", "metric": "traces_to_ge_1_mean_curve"},
                {"process": "cross_validation", "metric": "folds_reaching_ge_1_fraction"},
            ],
        },
        "report_metrics": [{"process": "test", "metric": "guessing_entropy"}],
        "filters": {"require_done": True, "require_processes": ["cross_validation", "test"]},
    }
}


def _rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _run_folder(root, name, *, final_ge, t1, reached, test_ge, epochs=2):
    """What a finished run leaves in artifacts/dev/generated_models/<name>/."""
    run = root / "artifacts" / "dev" / "generated_models" / name
    (run / "model").mkdir(parents=True)
    (run / "model" / f"{name}_checkpoint_manifest.json").write_text(json.dumps(
        {"validation_checkpointing": False, "selection": None, "final": {"epoch": epochs}}))
    scalars = run / "metrics" / "attack_metrics" / "scalars"
    _rows(scalars / f"{name}_metrics_train_history.jsonl", [
        {"epoch": e, "process": "train", "model_name": name, "accuracy": 0.5 if e < epochs else 0.01,
         "dependencies": {"accuracy_total": 5}}
        for e in range(1, epochs + 1)
    ])
    _rows(scalars / f"{name}_metrics_test_history.jsonl",
          [{"epoch": 1, "process": "test", "guessing_entropy": test_ge}])
    if final_ge is not None:
        aggregate = run / "cross_validation" / "aggregate"
        aggregate.mkdir(parents=True)
        (aggregate / "summary.yaml").write_text(yaml.safe_dump({
            "headline": {"model_name": name, "aggregate_path": "/app/x", "n_folds": 10,
                         "mean_curve_final_ge": final_ge, "traces_to_ge_1_mean_curve": t1,
                         "folds_reaching_ge_1": f"{reached}/10", "accuracy_mean": 0.01},
            "scalar_metrics": {"train.fit_time_s": {"mean": 270.0}, "validation.fit_time_s": {"mean": 6.0}},
        }))
        (aggregate / "mean_rank_curve.json").write_text(json.dumps({"traces_to_ge": {"1": {
            "mean_curve_stable_trace": t1, "mean_curve_first_trace": t1, "folds_reached": reached, "n_folds": 10}}}))
        (aggregate / "cross_validation_diagnostics.json").write_text(json.dumps({
            "verdict": {"code": "reliable_learning"},
            "analyses": {"rank_learning_and_stability": {"folds_better_than_chance": reached,
                                                         "normalized_rank_advantage": 0.9,
                                                         "endpoint_rank": {"sample_std": 0.5}}}}))
    return run


def _project(tmp_path, models, config=CONFIG):
    root = tmp_path / "projects" / "P"
    (root / "artifacts").mkdir(parents=True)
    (root / "config").mkdir()
    (root / "config" / "model_judge.yaml").write_text(yaml.safe_dump(config))
    runs, tests = {}, {}
    for name, h, final_ge, t1, reached, test_ge, status in models:
        _run_folder(root, name, final_ge=final_ge, t1=t1, reached=reached, test_ge=test_ge)
        runs[h] = {"model_name": name, "status": status, "env": "dev", "model_id": "M",
                   "path": f"P/artifacts/dev/generated_models/{name}"}
        tests[h] = {"model_name": name, "status": "DONE", "env": "dev"}
    (root / "artifacts" / "run_registry.json").write_text(json.dumps(runs))
    (root / "artifacts" / "test_registry.json").write_text(json.dumps(tests))
    return root


def test_directions():
    assert metric_direction("mean_curve_final_ge") == "min"
    assert metric_direction("traces_to_ge_1_mean_curve") == "min"
    assert metric_direction("folds_reaching_ge_1_fraction") == "max"
    assert metric_direction("accuracy_mean") == "max"
    assert metric_direction("top_5_accuracy") == "max"
    assert metric_direction("key_rank") == "min"


@pytest.mark.parametrize("name", ["accuracy_train", "accuracy_validation", "mean_accuracy", "mean_accuracy_train",
                                  "mean_accuracy_validation", "top_5_accuracy_validation", "final_sr_at_1"])
def test_the_names_run_reader_builds_keep_their_metrics_direction(name):
    assert metric_direction(name) == "max"


@pytest.mark.parametrize("name", ["accuracy_std", "mean_accuracy_validation_std", "training_time_std_s",
                                  "mean_fit_time_s_train", "validation_objective_validation", "mean_rank"])
def test_spreads_times_losses_and_ranks_stay_lower_is_better(name):
    assert metric_direction(name) == "min"


def test_selected_epoch_is_bookkeeping_not_a_metric_to_rank_on():
    from services.model_judge.model_judge import available_metrics

    candidate = {"model_name": "m", "processes": {"train": {"accuracy": 0.5, "selected_epoch": 100}}}
    assert available_metrics([candidate]) == {"train": ["accuracy"]}


def test_ranking_alignment_filters_and_outputs(tmp_path):
    root = _project(tmp_path, [
        # name, hash, final GE, traces_to_ge_1, folds reached, test GE, train status
        ("a", "aaaa0001", 1.0, 900, 6, 1.0, "DONE"),
        ("b", "bbbb0002", 1.0, None, 9, 1.0, "DONE"),    # tie on GE, missing traces -> after "a"
        ("c", "cccc0003", 1.0, 400, 2, 99.0, "DONE"),    # best traces_to_ge_1 -> first
        ("d", "dddd0004", 3.0, 100, 10, 1.0, "DONE"),    # worse primary -> last despite tie-breakers
        ("e", "eeee0005", 0.5, 50, 10, 1.0, "FAILED"),   # best numbers but training not DONE
        ("f", "ffff0006", None, None, 0, 1.0, "DONE"),   # no cross-validation in its run folder
    ])
    result = run_model_judge(root / "config" / "model_judge.yaml", project="P", env="dev",
                             projects_root=tmp_path / "projects")
    assert [c["model_name"] for c in result["leaderboard"]] == ["c", "a", "b", "d"]
    assert {r["model_name"]: r["reason"] for r in result["rejected"]} == {
        "e": "train status FAILED", "f": "missing processes ['cross_validation']",
    }
    assert result["selection_uses_test_data"] is False
    selected = yaml.safe_load((root / "model_selection" / "selected_model.yaml").read_text())
    assert selected["model_name"] == "c"
    assert selected["reported_metrics"] == {"test.guessing_entropy": 99.0}
    assert selected["artifacts"]["final_model"].endswith("generated_models/c/model/c_trained.pt")
    assert json.loads((root / "model_selection" / "leaderboard.json").read_text())[0]["model_name"] == "c"


def test_ranking_on_test_is_flagged(tmp_path):
    config = json.loads(json.dumps(CONFIG))
    config["model_judge"]["ranking"]["tie_breakers"] = [{"process": "test", "metric": "guessing_entropy"}]
    root = _project(tmp_path, [("a", "aaaa0001", 1.0, 9, 1, 5.0, "DONE"), ("b", "bbbb0002", 1.0, 9, 1, 2.0, "DONE")], config)
    result = run_model_judge(root / "config" / "model_judge.yaml", project="P", env="dev",
                             projects_root=tmp_path / "projects")
    assert result["selection_uses_test_data"] is True
    assert result["selected"]["model_name"] == "b"


def test_no_eligible_model_explains_why(tmp_path):
    root = _project(tmp_path, [("a", "aaaa0001", None, None, 0, 1.0, "DONE")])
    with pytest.raises(RuntimeError, match="missing processes"):
        run_model_judge(root / "config" / "model_judge.yaml", project="P", env="dev",
                        projects_root=tmp_path / "projects")


def test_config_hints_list_available_choices(tmp_path):
    config = json.loads(json.dumps(CONFIG))
    config["model_judge"]["ranking"]["tie_breakers"] = [
        {"process": "cross_validation", "metric": "mean_final_ge"},      # typo
        {"process": "validation", "metric": "guessing_entropy"},         # process not in data
        {"process": "cross_validation", "metric": "traces_to_ge_1_mean_curve"},  # recorded but empty
    ]
    root = _project(tmp_path, [("a", "aaaa0001", 1.0, None, 0, 1.0, "DONE")], config)
    result = run_model_judge(root / "config" / "model_judge.yaml", project="P", env="dev",
                             projects_root=tmp_path / "projects")
    hints = result["config_hints"]
    assert any("'mean_final_ge' not found" in h and "mean_curve_final_ge" in h for h in hints)
    assert any("process 'validation' has no data" in h and "cross_validation" in h for h in hints)
    assert any("traces_to_ge_1_mean_curve: recorded but empty" in h for h in hints)
    assert "accuracy_total" not in str(result["available_metrics"])       # bookkeeping filtered
    assert result["selected"]["model_name"] == "a"


# ------------------------------------------------------- reading the run folders
def _candidates(root):
    from services.model_judge.run_reader import load_candidates

    return {c["model_name"]: c for c in load_candidates(root)}


def test_candidate_metrics_come_from_the_run_folder(tmp_path):
    root = _project(tmp_path, [("a", "aaaa0001", 1.0, 900, 6, 3.0, "DONE")])
    candidate = _candidates(root)["a"]

    assert not (root / "benchmark").exists()
    assert (candidate["model_hash"], candidate["env"], candidate["train_status"]) == ("aaaa0001", "dev", "DONE")
    # train: the row of the tested epoch (the final one), a dependency read like a metric
    assert candidate["processes"]["train"] == {"accuracy": 0.01, "accuracy_total": 5, "selected_epoch": 2}
    assert candidate["processes"]["test"] == {"guessing_entropy": 3.0}
    assert "validation" not in candidate["processes"]          # a CV run has no validation history
    cv = candidate["processes"]["cross_validation"]
    assert cv["mean_curve_final_ge"] == 1.0 and cv["traces_to_ge_1_mean_curve"] == 900
    assert cv["folds_reaching_ge_1_fraction"] == 0.6 and cv["endpoint_rank_std"] == 0.5
    assert (cv["fit_time_s"], cv["fit_time_s_train"], cv["fit_time_s_validation"]) == (270.0, 270.0, 6.0)
    assert "aggregate_path" not in cv and candidate["cv_verdict"] == "reliable_learning"


def test_train_and_validation_metrics_are_those_of_the_tested_epoch(tmp_path):
    root = _project(tmp_path, [("a", "aaaa0001", None, None, 0, 1.0, "DONE")])
    run = root / "artifacts" / "dev" / "generated_models" / "a"
    (run / "model" / "a_checkpoint_manifest.json").write_text(json.dumps(
        {"validation_checkpointing": True, "selection": {"epoch": 1}, "final": {"epoch": 2}}))
    (run / "config").mkdir()
    (run / "config" / "a_params.yaml").write_text(yaml.safe_dump({"evaluation": {"test": {"load_model": "best_validation"}}}))
    _rows(run / "metrics" / "attack_metrics" / "scalars" / "a_metrics_validation_history.jsonl",
          [{"epoch": 1, "guessing_entropy": 4.0}, {"epoch": 2, "guessing_entropy": 9.0}])

    processes = _candidates(root)["a"]["processes"]
    assert processes["train"]["accuracy"] == 0.5 and processes["train"]["selected_epoch"] == 1
    # the best-validation epoch is tested: its validation values chose it (selection-biased)
    assert processes["validation"] == {"guessing_entropy": 4.0, "selected_epoch": 1, "selection_biased": True}

    # final_epoch is tested: the best-validation epoch does not describe the tested model
    (run / "config" / "a_params.yaml").write_text(yaml.safe_dump({"evaluation": {"test": {"load_model": "final_epoch"}}}))
    assert _candidates(root)["a"]["processes"]["validation"] == {"guessing_entropy": 9.0, "selected_epoch": 2,
                                                                 "selection_biased": False}


def test_key_recovery_reports_are_rankable(tmp_path):
    root = _project(tmp_path, [("a", "aaaa0001", None, None, 0, 1.0, "DONE")])
    reports = root / "artifacts" / "dev" / "generated_models" / "a" / "metrics" / "key_recovery"
    reports.mkdir(parents=True)
    (reports / "test.json").write_text(json.dumps({
        "key": {"final_true_key_rank": 245, "recovered": False, "true_key_byte": 224},
        "recovery": {"final_ge": 245.0, "stable_rank1_trace": None},
        "target": {"attack_byte": 2},
        "enumeration": {"key_rank_reported": 162, "key_rank_source": "MODEL+KEY_ENUMERATION",
                        "key_recovery": "FAILED", "enumeration_position": 162},
        "diagnostics": {"top1_top2_score_margin": 0.25},
    }))
    test = _candidates(root)["a"]["processes"]["test"]
    assert test["guessing_entropy"] == 1.0                       # the history metric
    assert (test["final_true_key_rank"], test["key_rank_reported"], test["key_enumeration_position"]) == (245, 162, 162)
    assert test["key_recovery"] == "FAILED" and test["key_recovered_through"] == "MODEL+KEY_ENUMERATION"
    assert test["final_ge"] == 245.0 and test["top1_top2_score_margin"] == 0.25 and test["attack_byte"] == 2


def test_a_registered_run_without_a_folder_is_not_a_candidate(tmp_path):
    root = _project(tmp_path, [("a", "aaaa0001", 1.0, 9, 1, 5.0, "DONE")])
    runs = json.loads((root / "artifacts" / "run_registry.json").read_text())
    runs["bbbb0002"] = {"model_name": "b", "status": "INTERRUPTED", "env": "dev",
                        "path": "P/artifacts/dev/generated_models/b"}
    runs["cccc0003"] = {"model_name": "a2", "status": "DONE", "env": "dev",      # path relative to the project
                        "path": "artifacts/dev/generated_models/a"}
    (root / "artifacts" / "run_registry.json").write_text(json.dumps(runs))
    assert sorted(_candidates(root)) == ["a", "a2"]


def test_list_choices_names_what_the_runs_recorded_and_the_files_own_choice(tmp_path):
    """The panel's Judge asks with this: the processes and metrics the finished runs hold (read only),
    the file's primary marked as current. A shipped file naming a process no run has rejected every
    run (DLSCA_ASCAD: test.accuracy under cross-validation; audit 2026-10-23)."""
    root = _project(tmp_path, [("a", "aaaa0001", 1.0, 300, 7, 2.0, "DONE")])
    got = list_choices(root / "config" / "model_judge.yaml", project="P", env="dev", projects_root=tmp_path / "projects")
    assert got["candidates"] == 1
    assert "cross_validation" in got["available"]
    assert "mean_curve_final_ge" in got["available"]["cross_validation"]
    assert got["direction"]["mean_curve_final_ge"] == "min"
    assert got["current"] == {"process": CONFIG["model_judge"]["ranking"]["primary"]["process"],
                              "metric": CONFIG["model_judge"]["ranking"]["primary"]["metric"]}
    assert not (root / "model_selection").exists()                          # nothing ranked or written


def test_test_and_test_budget_are_the_latest_tests_of_their_kind(tmp_path):
    """Owner 2026-10-27: the judge reads the latest All test as test and the latest Budget test as test_budget."""
    root = _project(tmp_path, [("a", "aaaa0001", None, None, 0, 7.0, "DONE")])
    run = root / "artifacts" / "dev" / "generated_models" / "a"
    for key, ge in (("aaaa0001_et_1", 5.0), ("aaaa0001_et_2", 2.0), ("aaaa0001_et_3", 40.0), ("aaaa0001_et_4", 1.0)):
        _rows(run / "test" / key / "metrics" / "attack_metrics" / "scalars" / "a_metrics_test_history.jsonl",
              [{"epoch": 1, "process": "test", "guessing_entropy": ge}])
    rec = lambda i, kind, day, status="DONE": {"index": i, "kind": kind, "status": status,
                                               "finished_at": f"2026-10-{day:02d}T00:00:00+00:00"}
    (root / "artifacts" / "test_registry.json").write_text(json.dumps({"aaaa0001": {
        "model_name": "a", "status": "DONE", "env": "dev", "finished_at": "2026-10-01T00:00:00+00:00",
        "external_tests": {"aaaa0001_et_1": rec(1, "all", 5), "aaaa0001_et_2": rec(2, "all", 3),
                           "aaaa0001_et_3": rec(3, "budget", 4), "aaaa0001_et_4": rec(4, "all", 9, "FAILED")}}}))

    candidate = _candidates(root)["a"]
    assert candidate["processes"]["test"] == {"guessing_entropy": 5.0}            # et_1 finished last, not et_2's 2.0
    assert candidate["processes"]["test_budget"] == {"guessing_entropy": 40.0}
    assert candidate["tests"] == {"test": "aaaa0001_et_1", "test_budget": "aaaa0001_et_3"}


def test_without_appended_tests_the_recorded_test_is_the_latest_and_there_is_no_budget(tmp_path):
    candidate = _candidates(_project(tmp_path, [("a", "aaaa0001", None, None, 0, 7.0, "DONE")]))["a"]
    assert candidate["processes"]["test"] == {"guessing_entropy": 7.0} and "test_budget" not in candidate["processes"]
    assert candidate["tests"] == {"test": "recorded", "test_budget": None}
