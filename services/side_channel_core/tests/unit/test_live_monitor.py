# services/side_channel_core/tests/unit/test_live_monitor.py
"""core/monitor.snapshot: a running run's latest metrics per process, as the TraceFlow Monitor shows them."""

import json
import os
import re

import numpy as np
import yaml

from core.monitor import snapshot


def _history(folder, process, rows):
    scalars = folder / "scalars"
    scalars.mkdir(parents=True, exist_ok=True)
    head = {"model_name": "run", "model_hash": "abcd1234", "process": process, "env": "dev", "timestamp": "t"}
    (scalars / f"run_metrics_{process}_history.jsonl").write_text(
        "".join(json.dumps({**head, **row}) + "\n" for row in rows), encoding="utf-8")


def _curve(folder, epoch, rank):
    arrays = folder / "arrays" / "validation" / "mean_attack_rank_curve"
    arrays.mkdir(parents=True, exist_ok=True)
    np.save(arrays / f"ep{epoch:04d}.npy", np.array([[1.0, 128.0], [5000.0, rank]]))
    return {"_type": "ndarray", "shape": [2, 2], "dtype": "float64",
            "path": f"arrays/validation/mean_attack_rank_curve/ep{epoch:04d}.npy"}


def _run(tmp_path, attack_times=0):
    run = tmp_path / "dev" / "generated_models" / "run"           # as artifacts/<env>/generated_models/<name>
    (run / "config").mkdir(parents=True)
    (run / "config" / "run_params.yaml").write_text(yaml.safe_dump({"perform_attack": {"test": {"attack_times": attack_times}},
                                                                       "trainer": {"loss": {"type": "CrossEntropyLoss"}}}))
    fold = run / "metrics" / "cross_validation" / "fold_03"
    _history(fold, "train", [{"epoch": 1, "mean_accuracy": 0.004, "mean_fit_time_s": 5.0},
                             {"epoch": 2, "mean_accuracy": 0.006, "mean_fit_time_s": 10.0}])
    _history(fold, "validation", [{"epoch": 1, "mean_attack_rank_curve": _curve(fold, 1, 40.0), "mean_accuracy": 0.005},
                                  {"epoch": 2, "mean_attack_rank_curve": _curve(fold, 2, 12.0), "mean_accuracy": 0.004,
                                   "dependencies": {"accuracy": 0.004}}])
    _history(run / "metrics" / "cross_validation" / "mean", "validation", [{"epoch": 2, "mean_accuracy": 0.9}])
    return run


def _lines(snap):
    return {m["name"]: [(v["label"], v["value"], v["improved"]) for v in m["values"]] for m in snap["metrics"]}


def _touch(path, when):
    os.utime(path, (when, when))


def test_one_line_per_metric_its_value_per_process_from_the_fold_being_trained(tmp_path):
    snap = snapshot(_run(tmp_path))
    assert (snap["source"], snap["epoch"]) == ("fold 3", 2)
    assert _lines(snap) == {"mean_accuracy": [("train", 0.006, True), ("validation (CV)", 0.004, False)],
                            "mean_fit_time_s": [("train", 10.0, None)],          # a time only grows: never better or worse
                            "mean_attack_rank_curve": [("validation (CV)", 12.0, True)]}                  # the curve's last point
    accuracy = snap["metrics"][0]["values"][0]
    assert accuracy == {"process": "train", "label": "train", "value": 0.006, "previous": 0.004, "direction": "max",
                        "improved": True}
    assert snap["metrics"][1]["values"][0]["direction"] == "min"                                      # no test: attack_times 0


def test_the_refit_shows_its_own_train_rows_never_a_folds_validation_beside_them(tmp_path):
    run = _run(tmp_path)
    _history(run / "metrics" / "attack_metrics", "train", [{"epoch": 1, "mean_accuracy": 0.3}, {"epoch": 2, "mean_accuracy": 0.5}])
    for process in ("train", "validation"):
        _touch(run / "metrics" / "cross_validation" / "fold_03" / "scalars" / f"run_metrics_{process}_history.jsonl", 1_000)
    snap = snapshot(run)
    assert (snap["source"], snap["epoch"]) == ("refit", 2)
    assert _lines(snap) == {"mean_accuracy": [("train", 0.5, True)]}


def test_a_run_without_cross_validation_names_validation_plainly(tmp_path):
    run = tmp_path / "dev" / "generated_models" / "run"
    _history(run / "metrics" / "attack_metrics", "train", [{"epoch": 1, "mean_accuracy": 0.2}])
    _history(run / "metrics" / "attack_metrics", "validation", [{"epoch": 1, "mean_accuracy": 0.01}])
    snap = snapshot(run)
    assert (snap["source"], snap["epoch"]) == (None, 1)
    assert _lines(snap) == {"mean_accuracy": [("train", 0.2, None), ("validation", 0.01, None)]}     # the first epoch


def test_a_test_value_only_when_the_test_is_attacked(tmp_path):
    run = _run(tmp_path, attack_times=100)
    _history(run / "metrics" / "attack_metrics", "test", [{"epoch": 2, "mean_accuracy": 0.0051}])
    assert _lines(snapshot(run))["mean_accuracy"][-1] == ("test", 0.0051, None)


def test_a_loss_metrics_to_log_does_not_list_is_not_shown(tmp_path):
    """Owner 2026-10-27: only the metrics_to_log columns are lines; an old heartbeat's objective is not read."""
    run = _run(tmp_path)
    beat = {"model_name": "run", "status": "running",
            "objective": {"epoch": 2, "train": 5.51, "train_previous": 5.53, "validation": 5.58, "validation_previous": 5.55}}
    (tmp_path / "dev" / "heartbeat.json").write_text(json.dumps(beat))       # artifacts/<env>/heartbeat.json
    assert set(_lines(snapshot(run))) == {"mean_accuracy", "mean_fit_time_s", "mean_attack_rank_curve"}


def test_logged_objectives_join_the_loss_line_and_the_heartbeat_is_not_read(tmp_path):
    run = tmp_path / "dev" / "generated_models" / "run"
    (run / "config").mkdir(parents=True)
    (run / "config" / "run_params.yaml").write_text(yaml.safe_dump({"trainer": {"loss": {"type": "CrossEntropyLoss"}}}))
    _history(run / "metrics" / "attack_metrics", "train", [{"epoch": 1, "training_objective": 5.6}, {"epoch": 2, "training_objective": 5.5}])
    _history(run / "metrics" / "attack_metrics", "validation", [{"epoch": 2, "validation_objective": 5.7}])
    (tmp_path / "dev" / "heartbeat.json").write_text(json.dumps({"model_name": "run", "objective": {"train": 9.0}}))
    assert _lines(snapshot(run)) == {"CrossEntropyLoss": [("train", 5.5, True), ("validation", 5.7, None)]}


def test_a_run_without_metrics_yet_has_no_lines(tmp_path):
    assert snapshot(tmp_path / "new_run") == {"run": "new_run", "source": None, "epoch": None, "metrics": [],
                                              "attack": None, "key_recovery": None}


def test_the_terminal_lines_put_every_separator_in_one_column_and_numbers_right_aligned(capsys):
    from core.monitor import _print

    def value(process, label, v, improved):
        return {"process": process, "label": label, "value": v, "previous": None, "direction": "max", "improved": improved}
    _print({"run": "run", "source": None, "epoch": 40, "metrics": [
        {"name": "mean_accuracy", "values": [value("train", "train", 0.0412, True), value("validation", "validation", 0.0038, False),
                                             value("test", "test", 0.0041, None)]},
        {"name": "mean_attack_rank_curve", "values": [value("validation", "validation", 3, True), value("test", "test", 1, None)]},
        {"name": "CrossEntropyLoss", "values": [value("train", "train", 5.31, True), value("validation", "validation", 5.55, False)]}]})
    out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out).splitlines()
    assert out[0] == "run · epoch 40"
    assert out[1:4] == ["  mean_accuracy           train: 0.0412 ▲ | validation: 0.0038 ▼ | test: 0.0041",
                        "  mean_attack_rank_curve                  | validation:      3 ▲ | test:      1",
                        "  CrossEntropyLoss        train:   5.31 ▲ | validation:   5.55 ▼"]
    assert len({line.index("|") for line in out[1:4]}) == 1


def _testing(tmp_path, run, phase="test"):
    """This run's heartbeat in `phase`, and its attack progress (utils/attack_progress.py) at attack 37 of 100."""
    (tmp_path / "dev" / "heartbeat.json").write_text(json.dumps({"model_name": "run", "status": "running", "phase": phase}))
    (tmp_path / "dev" / "attack_progress.json").write_text(json.dumps({
        "model_name": "run", "model_hash": "abcd1234", "attack": 37, "attacks": 100, "running": True,
        "started_ns": 1_000_000_000, "written_ns": 38_000_000_000, "metrics": {"mean_attack_rank_curve": 61.24, "traces_to_ge_1": None,
                    "per_attack_traces_to_ge_1": {"value": 342.0, "reached": 30, "of": 37}}}))


def test_while_the_test_attacks_the_attack_row_and_the_test_values_so_far(tmp_path):
    run = _run(tmp_path, attack_times=100)
    _testing(tmp_path, run)
    snap = snapshot(run)
    assert snap["attack"] == {"attack": 37, "attacks": 100, "running": True,          # the times: the ETA's test speed
                              "started_ns": 1_000_000_000, "written_ns": 38_000_000_000}
    test = {m["name"]: next(v for v in m["values"] if v["process"] == "test") for m in snap["metrics"]
            if any(v["process"] == "test" for v in m["values"])}
    assert (test["mean_attack_rank_curve"]["value"], test["mean_attack_rank_curve"]["live"]) == (61.24, True)
    assert [v["process"] for v in next(m for m in snap["metrics"] if m["name"] == "mean_attack_rank_curve")["values"]] \
        == ["validation", "test"]                                       # beside the epoch's validation value
    assert (test["traces_to_ge_1"]["value"], test["traces_to_ge_1"]["note"]) == (None, "not reached yet")
    assert (test["per_attack_traces_to_ge_1"]["value"], test["per_attack_traces_to_ge_1"]["note"]) == (342.0, "30 of 37 reached")


def test_no_attack_row_outside_the_test_phase_or_for_another_runs_progress(tmp_path):
    run = _run(tmp_path, attack_times=100)
    _testing(tmp_path, run, phase="train")
    assert snapshot(run)["attack"] is None
    _testing(tmp_path, run)
    progress = json.loads((tmp_path / "dev" / "attack_progress.json").read_text())
    (tmp_path / "dev" / "attack_progress.json").write_text(json.dumps({**progress, "model_name": "another_run"}))
    assert snapshot(run)["attack"] is None


def test_the_key_recovery_line_is_the_last_validation_report_then_the_tests(tmp_path):
    run = _run(tmp_path)
    reports = run / "metrics" / "key_recovery"
    reports.mkdir(parents=True)
    report = {"key": {"recovered": False, "final_true_key_rank": 107, "top_key_hypothesis_hex": "0xFB"},
              "recovery": {"final_ge": 107.0, "final_sr_at_1": 0.0}}
    (reports / "validation_history.jsonl").write_text(
        "".join(json.dumps({**report, "epoch": e}) + "\n" for e in (19, 20)))
    assert snapshot(run)["key_recovery"] == {"source": "validation, epoch 20", "recovered": False, "true_key_rank": 107,
                                             "top_guess": "0xFB", "ge": 107.0, "sr1": 0.0}
    (reports / "test.json").write_text(json.dumps({"key": {"recovered": True, "final_true_key_rank": 1,
                                                           "top_key_hypothesis_hex": "0x00"},
                                                   "recovery": {"final_ge": 1.0, "final_sr_at_1": 1.0}}))
    assert snapshot(run)["key_recovery"] == {"source": "test", "recovered": True, "true_key_rank": 1, "top_guess": "0x00",
                                             "ge": 1.0, "sr1": 1.0}


def test_a_threshold_not_reached_yet_reads_not_reached_by_the_traces_attacked(tmp_path):
    """traces_to_ge_1 and N̄tGE recorded null (the attacks never stayed at rank 1): the line stays, "not reached by
    5,000 traces" as the run's log says it, 5,000 the last trace count of the epoch's rank curve."""
    run = tmp_path / "dev" / "generated_models" / "run"
    (run / "config").mkdir(parents=True)
    (run / "config" / "run_params.yaml").write_text(yaml.safe_dump({"perform_attack": {"test": {"attack_times": 0}}}))
    folder = run / "metrics" / "attack_metrics"
    _history(folder, "validation", [{"epoch": 1, "mean_attack_rank_curve": _curve(folder, 1, 3.0), "traces_to_ge_1": None,
                                     "per_attack_traces_to_ge_1": None, "mean_accuracy": 0.002}])
    values = {m["name"]: m["values"][0] for m in snapshot(run)["metrics"]}
    for name in ("traces_to_ge_1", "per_attack_traces_to_ge_1"):
        assert (values[name]["value"], values[name]["note"], values[name]["improved"]) == (None, "not reached by 5,000 traces", None)
    assert values["mean_attack_rank_curve"]["value"] == 3.0                    # the numbers as before
    assert "mean_accuracy" in values
