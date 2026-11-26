# tests/unit/test_registry_records.py
"""
Registry records of 2026-09-30:

- test_registry.json entries carry the run's model_name and path (from
  run_registry.json), also for entries created by a FAILED/INTERRUPTED marking;
- FAILED / INTERRUPTED entries carry a status_reason, dropped once the status
  moves on;
- artifacts/done_model_names.yaml lists only models whose training completed,
  never a name that was merely generated;
- run_registry.json, test_registry.json, run_map.jsonl, model_name_map.json and
  done_model_names.yaml are written atomically.
"""
from __future__ import annotations

import json
import logging
import os
import signal

import pytest
import yaml


def _write_registries(root, runs, tests=None):
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "run_registry.json").write_text(json.dumps(runs))
    (artifacts / "test_registry.json").write_text(json.dumps(tests or {}))


def _read(root, name):
    return json.loads((root / "artifacts" / name).read_text())


def _entry(name, status, env="dev", **extra):
    return {"model_name": name, "status": status, "env": env, "model_id": "ASCAD_CNNbase",
            "path": f"DLSCA_ASCAD/artifacts/{env}/generated_models/{name}", **extra}


def test_new_test_entry_takes_model_name_and_path_from_the_run(tmp_path):
    from utils.test_registry import update_test_status

    _write_registries(tmp_path, {"3745aeec": _entry("tulip_fish", "INTERRUPTED")})
    update_test_status(tmp_path, model_id="3745aeec", new_status="FAILED", reason="signal_2_interrupt")

    entry = _read(tmp_path, "test_registry.json")["3745aeec"]
    assert entry["model_name"] == "tulip_fish"
    assert entry["path"] == "DLSCA_ASCAD/artifacts/dev/generated_models/tulip_fish"
    assert entry["status"] == "FAILED" and entry["status_reason"] == "signal_2_interrupt"


def test_loading_heals_an_entry_without_model_name_or_path(tmp_path):
    from utils.test_registry import load_test_registry

    _write_registries(
        tmp_path,
        {"3745aeec": _entry("tulip_fish", "INTERRUPTED")},
        tests={"3745aeec": {"model_name": "", "status": "FAILED", "path": "", "env": "dev"}},
    )
    entry = load_test_registry(tmp_path)["3745aeec"]
    assert entry["model_name"] == "tulip_fish"
    assert entry["path"].endswith("generated_models/tulip_fish")
    assert _read(tmp_path, "test_registry.json")["3745aeec"]["model_name"] == "tulip_fish"


def test_status_reason_is_dropped_when_the_status_moves_on(tmp_path):
    from utils.sweep_engine import update_status
    from utils.test_registry import update_test_status

    _write_registries(tmp_path, {"aaaa": _entry("m-a", "in_progress")})
    update_status(tmp_path, model_id="aaaa", new_status="FAILED", reason="RuntimeError: boom")
    update_test_status(tmp_path, model_id="aaaa", new_status="FAILED", reason="train failed: RuntimeError: boom")
    assert _read(tmp_path, "run_registry.json")["aaaa"]["status_reason"] == "RuntimeError: boom"

    update_status(tmp_path, model_id="aaaa", new_status="queued")
    update_test_status(tmp_path, model_id="aaaa", new_status="queued")
    assert "status_reason" not in _read(tmp_path, "run_registry.json")["aaaa"]
    assert "status_reason" not in _read(tmp_path, "test_registry.json")["aaaa"]


OLD_START, OLD_END = "2026-10-19T21:55:41+08:00", "2026-10-19T21:55:51+08:00"
OLD_OUTCOME = {"key_recovered": True, "true_key": "224 (0xE0)", "recovered_key": "224 (0xE0)",
               "recovered_key_rank": 1, "fold_key_recovery": [{"fold": 1, "key_recovered": True}]}


def test_a_retrain_is_a_new_attempt_with_its_own_times_and_no_earlier_outcome(tmp_path):
    """elephant_donuts (2026-10-19): DONE with the first attempt's 10 s although the retry trained for hours."""
    from utils.sweep_engine import update_status

    _write_registries(tmp_path, {"aaaa": _entry("m-a", "DONE", started_at=OLD_START, finished_at=OLD_END,
                                                reason="GPU FAILED: lost", **OLD_OUTCOME)})
    update_status(tmp_path, model_id="aaaa", new_status="in_progress")
    entry = _read(tmp_path, "run_registry.json")["aaaa"]
    assert entry["started_at"] != OLD_START and "finished_at" not in entry and "reason" not in entry
    assert not set(OLD_OUTCOME) & set(entry)                     # a failing retrain shows no old outcome
    started = entry["started_at"]

    update_status(tmp_path, model_id="aaaa", new_status="in_progress")          # the same attempt
    assert _read(tmp_path, "run_registry.json")["aaaa"]["started_at"] == started

    update_status(tmp_path, model_id="aaaa", new_status="FAILED", reason="RuntimeError: boom")
    entry = _read(tmp_path, "run_registry.json")["aaaa"]
    assert entry["started_at"] == started and entry["finished_at"] not in (None, OLD_END)
    assert "fold_key_recovery" not in entry and entry["status_reason"] == "RuntimeError: boom"


def test_a_finished_or_force_queued_test_queued_again_loses_its_earlier_times_and_outcome(tmp_path):
    from utils.test_registry import update_test_status

    tests = {"aaaa": _entry("m-a", "SKIPPED", started_at=OLD_START, finished_at=OLD_END, **OLD_OUTCOME),
             "bbbb": _entry("m-b", "queued", force_previous_status="DONE", started_at=OLD_START,
                            finished_at=OLD_END, **OLD_OUTCOME)}
    _write_registries(tmp_path, {"aaaa": _entry("m-a", "in_progress"), "bbbb": _entry("m-b", "in_progress")},
                      tests=tests)
    for h in ("aaaa", "bbbb"):
        update_test_status(tmp_path, model_id=h, new_status="queued")
        entry = _read(tmp_path, "test_registry.json")[h]
        assert "started_at" not in entry and "finished_at" not in entry
        assert all(entry.get(k) != v for k, v in OLD_OUTCOME.items())   # the keys stay, emptied (load heals them)

    update_test_status(tmp_path, model_id="aaaa", new_status="SKIPPED", reason="perform_attack.test.attack_times is 0")
    entry = _read(tmp_path, "test_registry.json")["aaaa"]
    assert entry["finished_at"] != OLD_END and "started_at" not in entry


def test_a_recorded_failure_ends_the_attempt_without_the_legacy_reason(tmp_path):
    from core.orchestrator import _record_failed_attempt

    _write_registries(tmp_path, {"aaaa": _entry("m-a", "in_progress", started_at=OLD_START, reason="GPU FAILED: lost")})
    _record_failed_attempt(tmp_path, "aaaa", reason="run_cfg_unrecoverable: x")
    entry = _read(tmp_path, "run_registry.json")["aaaa"]
    assert entry["status"] == "FAILED" and entry["status_reason"] == "run_cfg_unrecoverable: x"
    assert entry.get("finished_at") and "reason" not in entry


def test_a_force_claimed_done_run_that_cannot_rerun_gets_its_attempt_back(tmp_path):
    from core.orchestrator import _restore_done_status
    from utils.sweep_engine import update_status

    before = _entry("m-a", "DONE", started_at=OLD_START, finished_at=OLD_END, **OLD_OUTCOME)
    _write_registries(tmp_path, {"aaaa": dict(before)})
    update_status(tmp_path, model_id="aaaa", new_status="in_progress")        # -f claims it
    _restore_done_status(tmp_path, "aaaa", before)
    entry = _read(tmp_path, "run_registry.json")["aaaa"]
    assert entry["status"] == "DONE" and (entry["started_at"], entry["finished_at"]) == (OLD_START, OLD_END)
    assert {k: entry[k] for k in OLD_OUTCOME} == OLD_OUTCOME


def test_run_map_sync_takes_the_registry_status_reason(tmp_path):
    from utils.run_map import load_run_map, save_run_map, sync_run_map_status_from_run_registry

    save_run_map(tmp_path, {"aaaa": {"model_hash": "aaaa", "model_name": "m-a", "status": "IN_PROGRESS",
                                     "reason": "GPU FAILED: lost", "sampled": {}}})
    sync_run_map_status_from_run_registry(
        tmp_path, {"aaaa": _entry("m-a", "INTERRUPTED", status_reason="signal 15 (SIGTERM)")})
    assert load_run_map(tmp_path)["aaaa"]["reason"] == "signal 15 (SIGTERM)"


def test_run_map_paths_stay_project_relative_whatever_form_a_writer_passes(tmp_path):
    """A registry path (<project>/artifacts/...) in run_map made make test look in projects/P/P/artifacts/..."""
    from utils.run_map import load_run_map, project_relative_path, sync_run_map_status_from_run_registry, upsert_run_map_entry

    root = tmp_path / "P"
    root.mkdir()
    want = "artifacts/dev/generated_models/m-a"
    assert project_relative_path("P/" + want, root) == want
    assert project_relative_path(root / want, root) == want
    assert project_relative_path(want, root) == want

    upsert_run_map_entry(root, model_hash="aaaa", model_name="m-a", model_id="M", sampled={}, status="in_progress",
                         env="dev", run_dir="P/" + want)
    assert load_run_map(root)["aaaa"]["path"] == want
    sync_run_map_status_from_run_registry(root, {"aaaa": _entry("m-a", "DONE", path="P/" + want)})
    assert load_run_map(root)["aaaa"]["path"] == want


def _scheduled_cfg(root, step_size):
    trainer = {"epochs": 10, "optimizer": {"type": "SGD", "learning_rate": 1e-3,
                                           "scheduler": {"type": "StepLR", "args": {"step_size": step_size}}}}
    sampled = {"mode": "scc", "model_id": "M", "trainer": trainer, "sca": {"leakage_model": "ID"}, "model": []}
    return {"project": {"root": str(root)}, "env": "dev", "model_id": "M", "trainer": trainer,
            "sca": {"leakage_model": "ID"}, "model": [], "sampled": sampled}


def test_runs_differing_only_in_scheduler_arguments_get_different_sampled_keys(tmp_path):
    from utils.run_lifecycle_logger import build_sampled_fingerprint_from_cfg
    from utils.run_map import sample_key

    a, b = (build_sampled_fingerprint_from_cfg(_scheduled_cfg(tmp_path, s), mode="scc") for s in (5, 10))
    assert a["trainer"]["optimizer"]["scheduler"]["args"] == {"step_size": 5}          # never "{...}"
    assert sample_key(a) != sample_key(b)


def test_every_run_map_record_of_a_run_carries_the_planners_sweep_point(tmp_path):
    from utils.run_lifecycle_logger import complete_run_map_entry, start_run_map_entry
    from utils.run_map import load_run_map, sample_key

    cfg = _scheduled_cfg(tmp_path, 5)
    start_run_map_entry(cfg=cfg, model_hash="aaaa", model_name="m-a", model_id="M", mode="scc")
    started = load_run_map(tmp_path)["aaaa"]
    assert started["sampled"] == cfg["sampled"] and started["sampled_key"] == sample_key(cfg["sampled"])
    complete_run_map_entry(cfg=cfg, model_hash="aaaa", model_name="m-a", model_id="M", mode="scc")
    assert load_run_map(tmp_path)["aaaa"]["sampled_key"] == started["sampled_key"]


def test_failed_without_a_reason_is_recorded_as_unspecified(tmp_path, caplog):
    from utils.test_registry import update_test_status

    _write_registries(tmp_path, {"aaaa": _entry("m-a", "DONE")})
    update_test_status(tmp_path, model_id="aaaa", new_status="FAILED")
    assert _read(tmp_path, "test_registry.json")["aaaa"]["status_reason"] == "unspecified"
    assert "without a reason" in caplog.text


def test_signal_records_why_train_and_test_stopped(tmp_path, monkeypatch):
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    from core import orchestrator as orch
    for sig, handler in saved.items():
        signal.signal(sig, handler)

    _write_registries(tmp_path, {"3745aeec": _entry("tulip_fish", "in_progress")})
    monkeypatch.setattr(orch, "interrupt_run_map_entry", lambda **kw: None)
    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(orch, "RUN_ENV", "dev")
    monkeypatch.setattr(orch, "SELECTED_HASHES", {"3745aeec"})

    with pytest.raises(SystemExit):
        orch.handle_signal(signal.SIGINT, None)

    run = _read(tmp_path, "run_registry.json")["3745aeec"]
    test = _read(tmp_path, "test_registry.json")["3745aeec"]
    assert run["status"] == "INTERRUPTED" and "signal 2" in run["status_reason"]
    assert test["status"] == "FAILED" and "signal 2" in test["status_reason"]
    assert test["model_name"] == "tulip_fish" and test["path"] == run["path"]


def test_only_trained_models_are_listed_in_done_model_names_yaml(tmp_path, monkeypatch):
    from utils import model_naming
    from utils.graph_config_sync import append_model_name

    names = iter(["model_1", "model_2"])
    monkeypatch.setattr(model_naming, "_generate_funny_name", lambda h=None: next(names))
    monkeypatch.setattr(model_naming, "make_experiment_hash", lambda cfg: cfg["seed"])
    for seed in ("aaaa0001", "aaaa0002"):
        model_naming.choose_run_name({"project": {"root": str(tmp_path)}, "seed": seed})
    assert not (tmp_path / "artifacts" / "done_model_names.yaml").exists()

    append_model_name(tmp_path, "model_1")   # train.py, once training completed
    append_model_name(tmp_path, "model_2")
    listed = yaml.safe_load((tmp_path / "artifacts" / "done_model_names.yaml").read_text())
    assert listed == ["model_1", "model_2"]
    assert not (tmp_path / "artifacts" / "model_names.yaml").exists()


@pytest.mark.parametrize("write", ["run_registry", "test_registry", "run_map", "name_map", "model_names"])
def test_registry_writes_are_atomic(tmp_path, monkeypatch, write):
    from utils import atomic_io
    from utils.graph_config_sync import append_model_name
    from utils.model_naming import _save_mapping
    from utils.run_map import save_run_map
    from utils.sweep_engine import save_registry
    from utils.test_registry import save_test_registry

    writers = {
        "run_registry": ("run_registry.json", lambda: save_registry({"aaaa": _entry("m-a", "DONE")}, tmp_path)),
        "test_registry": ("test_registry.json", lambda: save_test_registry({"aaaa": _entry("m-a", "DONE")}, tmp_path)),
        "run_map": ("run_map.jsonl", lambda: save_run_map(
            tmp_path, {"aaaa": {"model_name": "m-a", "status": "DONE", "sampled": {}}})),
        "name_map": ("model_name_map.json", lambda: _save_mapping(tmp_path, {"aaaa": "m-a"})),
        "model_names": ("done_model_names.yaml", lambda: append_model_name(tmp_path, "m-a")),
    }
    name, run = writers[write]
    target = tmp_path / "artifacts" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    original = "- ORIGINAL\n" if write == "model_names" else "ORIGINAL"
    target.write_text(original)

    # A crash between writing the temp file and publishing it leaves the old file.
    def crash(src, dst):
        raise OSError("crash before publish")

    monkeypatch.setattr(atomic_io.os, "replace", crash)
    with pytest.raises(OSError, match="crash before publish"):
        run()
    assert target.read_text() == original
    assert not [p.name for p in target.parent.iterdir() if ".tmp" in p.name]

    monkeypatch.undo()
    run()
    assert target.read_text() != original
    assert oct(os.stat(target).st_mode & 0o777) != oct(0o600)


# ------------------------------------------------ key-recovery outcome fields
# run_registry.json speaks for perform_attack.validation (the CV folds' held-out
# parts, or val.h5 without CV), test_registry.json for perform_attack.test
# (test.h5). true_key is always a value, "224 (0xE0)": read from the run's own
# data split (what the data loader wrote and the attack used), else from the
# report.
FIELDS = ("key_recovered", "true_key", "recovered_key", "recovered_key_rank")
CV_NOT_APPLICABLE = "not applicable in CV as no validation training in final fit."
TEST_NOT_ATTACKED = "not available: perform_attack.test.attack_times is 0"
VALIDATION_NOT_ATTACKED = "not available: perform_attack.validation.attack_times is 0"


def _report(process, *, recovered, rank, top, true_key=224, enumeration=None):
    report = {
        "process": process,
        "key": {"recovered": recovered, "final_true_key_rank": rank,
                "true_key_byte": true_key, "top_key_hypothesis": top},
        "target": {"attack_byte": 2},
    }
    if enumeration:
        report["enumeration"] = enumeration
    return report


def _split(tmp_path, name, group, key_bytes):
    import h5py
    import numpy as np

    dtype = np.dtype([("plaintext", "u1", (16,)), ("key", "u1", (16,))])
    metadata = np.zeros(len(key_bytes), dtype=dtype)
    metadata["key"][:, 2] = key_bytes
    path = tmp_path / "data" / name
    path.parent.mkdir(exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_group(group).create_dataset("metadata", data=metadata)


def _record(tmp_path, monkeypatch, *, cv, validation_report, test_report, cv_headline=None, **attacked):
    from utils import key_recovery_helper

    monkeypatch.setattr(key_recovery_helper, "_split_file", lambda cfg, name: tmp_path / "data" / name)
    _write_registries(tmp_path, {"3745aeec": _entry("tulip_fish", "DONE")},
                      tests={"3745aeec": _entry("tulip_fish", "DONE")})
    cfg = {"project": {"root": str(tmp_path)}, "model_hash": "3745aeec", "model_name": "tulip_fish",
           "runtime": {}, "paths": {},
           "cross_validation": {"enabled": cv}, "sca": {"attack_byte": 2}}
    key_recovery_helper._record_key_recovery_registries(
        cfg=cfg, logger=logging.getLogger("test_registry_records"),
        validation_report=validation_report, test_report=test_report, cv_headline=cv_headline,
        **attacked,
    )
    return _read(tmp_path, "run_registry.json")["3745aeec"], _read(tmp_path, "test_registry.json")["3745aeec"]


def _fields(entry):
    return {k: entry.get(k) for k in FIELDS}


def test_without_cv_both_registries_copy_their_key_recovery_outcome(tmp_path, monkeypatch):
    run, test = _record(
        tmp_path, monkeypatch, cv=False,
        validation_report=_report("validation", recovered=True, rank=1, top=224),
        test_report=_report("test", recovered=False, rank=7, top=13),
    )
    assert _fields(run) == {"key_recovered": True, "true_key": "224 (0xE0)", "recovered_key": "224 (0xE0)",
                            "recovered_key_rank": 1}
    assert _fields(test) == {"key_recovered": False, "true_key": "224 (0xE0)", "recovered_key": "13 (0x0D)",
                             "recovered_key_rank": 7}


def test_cv_run_registry_gives_the_cv_reason_and_the_fold_key(tmp_path, monkeypatch):
    fold = _report("validation", recovered=True, rank=1, top=224, true_key=77)
    run, test = _record(
        tmp_path, monkeypatch, cv=True, validation_report=None,
        test_report=_report("test", recovered=True, rank=1, top=224),
        cv_headline={"fold_reports": [{"report": None}, {"report": fold}]},
    )
    assert _fields(run) == {"key_recovered": CV_NOT_APPLICABLE, "true_key": "77 (0x4D)",
                            "recovered_key": CV_NOT_APPLICABLE, "recovered_key_rank": CV_NOT_APPLICABLE}
    assert test["key_recovered"] is True and test["recovered_key_rank"] == 1


def test_cv_with_unattacked_folds_reads_the_key_from_train_h5(tmp_path, monkeypatch):
    _split(tmp_path, "train.h5", "Profiling_traces", [224, 224, 224])
    run, _ = _record(
        tmp_path, monkeypatch, cv=True, validation_report=None, test_report=None,
        cv_headline=None, validation_attacked=False, test_attacked=False,
    )
    assert _fields(run) == {"key_recovered": CV_NOT_APPLICABLE, "true_key": "224 (0xE0)",
                            "recovered_key": VALIDATION_NOT_ATTACKED, "recovered_key_rank": VALIDATION_NOT_ATTACKED}


def test_unattacked_test_names_the_test_setting_and_reads_test_h5(tmp_path, monkeypatch):
    _split(tmp_path, "test.h5", "Attack_traces", [224, 224])
    fold = _report("validation", recovered=True, rank=1, top=224)
    run, test = _record(
        tmp_path, monkeypatch, cv=True, validation_report=None, test_report=None,
        cv_headline={"fold_reports": [{"report": fold}]}, test_attacked=False,
    )
    assert _fields(test) == {"key_recovered": TEST_NOT_ATTACKED, "true_key": "224 (0xE0)",
                             "recovered_key": TEST_NOT_ATTACKED, "recovered_key_rank": TEST_NOT_ATTACKED}
    assert "test.attack_times" not in json.dumps(run)   # run_registry speaks for validation only


def test_unattacked_validation_without_cv_reads_val_h5(tmp_path, monkeypatch):
    _split(tmp_path, "val.h5", "Attack_traces", [224])
    run, test = _record(
        tmp_path, monkeypatch, cv=False, validation_report=None, validation_attacked=False,
        test_report=_report("test", recovered=True, rank=1, top=224),
    )
    assert _fields(run) == {"key_recovered": VALIDATION_NOT_ATTACKED, "true_key": "224 (0xE0)",
                            "recovered_key": VALIDATION_NOT_ATTACKED, "recovered_key_rank": VALIDATION_NOT_ATTACKED}
    assert test["key_recovered"] is True


def test_true_key_of_a_variable_key_split_is_described(tmp_path, monkeypatch):
    _split(tmp_path, "test.h5", "Attack_traces", [1, 2, 3])
    _, test = _record(tmp_path, monkeypatch, cv=False, validation_report=None, test_report=None,
                      test_attacked=False)
    assert test["true_key"] == "variable key: 3 distinct values of key byte 2 in test.h5"


def test_true_key_without_a_readable_split_says_why(tmp_path, monkeypatch):
    _, test = _record(tmp_path, monkeypatch, cv=False, validation_report=None, test_report=None,
                      test_attacked=False)
    assert test["true_key"].startswith("not available: test.h5 unreadable")


def test_true_key_comes_from_the_data_split_before_the_report(tmp_path, monkeypatch, caplog):
    _split(tmp_path, "test.h5", "Attack_traces", [224, 224])
    _, test = _record(tmp_path, monkeypatch, cv=False, validation_report=None,
                      test_report=_report("test", recovered=True, rank=1, top=77, true_key=77))
    assert test["true_key"] == "224 (0xE0)"
    assert "true_key differs" in caplog.text


def test_make_test_leaves_run_registry_alone(tmp_path, monkeypatch):
    run, test = _record(
        tmp_path, monkeypatch, cv=True, validation_report=None,
        test_report=_report("test", recovered=True, rank=1, top=224),
    )   # run_test_key_recovery_summary: no cv_headline, validation not in question
    assert "key_recovered" not in run and test["key_recovered"] is True


def test_enumeration_outcome_wins_over_the_model_guess(tmp_path, monkeypatch):
    enumerated = _report("test", recovered=False, rank=245, top=13, enumeration={
        "key_recovery": "DONE", "key_rank_reported": 1, "key_rank_source": "MODEL+KEY_ENUMERATION",
        "recovered_key_byte": 224, "enumeration_position": 1})
    _, test = _record(tmp_path, monkeypatch, cv=False, validation_report=None, test_report=enumerated)
    assert (test["key_recovered"], test["recovered_key"], test["recovered_key_rank"]) == (True, "224 (0xE0)", 1)


def test_every_test_entry_carries_the_key_recovery_fields(tmp_path):
    from utils.test_registry import load_test_registry, update_test_status

    _write_registries(
        tmp_path,
        {"3745aeec": _entry("tulip_fish", "INTERRUPTED"), "bbbb": _entry("m-b", "DONE")},
        tests={"3745aeec": {"model_name": "", "status": "FAILED", "path": "", "env": "dev"},
               "cccc": _entry("m-c", "DONE", key_recovered=True, true_key=224, recovered_key=224,
                              recovered_key_rank=1)},
    )
    update_test_status(tmp_path, model_id="bbbb", new_status="queued")
    tests = load_test_registry(tmp_path)
    for h in ("3745aeec", "bbbb"):   # never tested: present, null
        assert _fields(tests[h]) == dict.fromkeys(FIELDS)
    assert _fields(tests["cccc"]) == {"key_recovered": True, "true_key": 224, "recovered_key": 224,
                                      "recovered_key_rank": 1}


def test_make_test_without_a_test_attack_records_it(tmp_path, monkeypatch):
    from utils import key_recovery_helper

    _split(tmp_path, "test.h5", "Attack_traces", [224])
    monkeypatch.setattr(key_recovery_helper, "_split_file", lambda cfg, name: tmp_path / "data" / name)
    monkeypatch.setattr(key_recovery_helper, "validate_key_recovery_contract", lambda cfg: {"enabled": True})
    monkeypatch.setattr(key_recovery_helper, "attack_plan", lambda cfg, process: {"attack_times": 0})
    _write_registries(tmp_path, {"3745aeec": _entry("tulip_fish", "DONE")},
                      tests={"3745aeec": _entry("tulip_fish", "DONE")})
    cfg = {"project": {"root": str(tmp_path)}, "model_hash": "3745aeec", "model_name": "tulip_fish",
           "runtime": {}, "paths": {}, "sca": {"attack_byte": 2}}
    assert key_recovery_helper.run_test_key_recovery_summary(cfg=cfg, logger=logging.getLogger("t"))
    test = _read(tmp_path, "test_registry.json")["3745aeec"]
    assert _fields(test) == {"key_recovered": TEST_NOT_ATTACKED, "true_key": "224 (0xE0)",
                             "recovered_key": TEST_NOT_ATTACKED, "recovered_key_rank": TEST_NOT_ATTACKED}
    assert "key_recovered" not in _read(tmp_path, "run_registry.json")["3745aeec"]


# ------------------------------------------------------------ FORCE=1 retrains
def test_force_takes_precedence_over_run_map_recovery_in_orchestrator_sh():
    from pathlib import Path

    script = (Path(__file__).resolve().parents[2] / "scripts" / "orchestrator.sh").read_text()
    loop = script[script.index("while true; do"):]
    force = loop.index('if [[ "${FORCE}" = "1" ]]; then')
    recovery = loop.index('elif [[ -n "${run_map_flag}" ]]; then')
    assert force < recovery   # an INTERRUPTED run_map record must not turn FORCE into a resume pass


@pytest.mark.parametrize("record, attempted", [
    ({"run_session_id": "S2", "status": "DONE"}, True),     # retrained in this session
    ({"run_session_id": "S2", "status": "FAILED"}, True),   # tried in this session: no endless retry
    ({"run_session_id": "S1", "status": "DONE"}, False),    # DONE in an earlier session: retrain
    ({"run_session_id": "S2", "status": "INTERRUPTED"}, False),
    ({}, False),
])
def test_force_attempts_each_hash_once_per_session(record, attempted):
    from utils.sweep_engine import _attempted_in_session

    assert _attempted_in_session([record, {}], "S2") is attempted


# ------------------------------------- summary: why a report is unavailable
TEST_REASON = "Because perform_attack.test.attack_times is 0, the test traces were not attacked."


def test_summary_section_gives_the_reason_a_report_is_unavailable():
    from sca.key_recovery import _summary_section

    lines = _summary_section("TEST - FINAL EVALUATION", None, key_display="full", unavailable_reason=TEST_REASON)
    assert lines[2] == "Report                  : unavailable"
    assert lines[3] == f"Reason                  : {TEST_REASON}"
    # No reason known: the line is not invented.
    assert "Reason" not in "\n".join(_summary_section("TEST - FINAL EVALUATION", None, key_display="full"))


@pytest.mark.parametrize("plan, expected", [
    ({"validation": {"attack_times": 1}, "test": {"attack_times": 0}},
     {"test": TEST_REASON, "validation": None}),
    ({"validation": {"attack_times": 0}, "test": {"attack_times": 1}},
     {"test": None,
      "validation": "Because perform_attack.validation.attack_times is 0, the validation traces were not attacked."}),
])
def test_logged_summary_names_the_unattacked_process(monkeypatch, plan, expected):
    from sca import key_recovery

    monkeypatch.setattr(key_recovery, "validate_key_recovery_contract", lambda cfg: {"key_display": "full"})
    logged = []
    logger = type("L", (), {"info": lambda self, fmt, *args: logged.append(fmt % args)})()
    key_recovery.log_combined_key_recovery_summary(
        logger, cfg={"perform_attack": plan, "sca": {}}, validation_report=None, test_report=None,
    )
    text = logged[0]
    validation, test = text.split("TEST - FINAL EVALUATION")
    for section, reason in ((test, expected["test"]), (validation, expected["validation"])):
        assert "Report                  : unavailable" in section
        if reason is None:
            assert "Reason" not in section
        else:
            assert f"Reason                  : {reason}" in section


def test_model_names_come_only_from_funny_words(monkeypatch):
    """No run_<hash> substitute: without funny-words no model_name is made."""
    from services.side_channel_core.utils import model_naming

    def missing(*a, **k):
        raise FileNotFoundError("funny-words")

    monkeypatch.setattr(model_naming.subprocess, "check_output", missing)
    with pytest.raises(RuntimeError, match="funny-words is not installed"):
        model_naming.generate_unique_model_name(set())
    monkeypatch.setattr(model_naming.subprocess, "check_output", lambda *a, **k: b"hold_hobbit\n")
    assert model_naming.generate_unique_model_name({"other_name"}) == "hold_hobbit"
    with pytest.raises(RuntimeError, match="unused by the other hashes"):
        model_naming.generate_unique_model_name({"hold_hobbit"}, attempts=3)


def test_test_registry_writers_do_not_lose_each_others_updates(tmp_path):
    """Every writer goes through update_test_registry: the file is loaded fresh under an exclusive
    lock, changed and saved before the lock is released, so a copy loaded earlier by another writer
    (a queue's test phase beside make test) is never written back over this one (audit 2026-10-23)."""
    import threading
    from utils.test_registry import load_test_registry, test_registry_lock, update_test_registry, update_test_status

    _write_registries(tmp_path, {f"h{i:07d}": _entry(f"run_{i}", "DONE") for i in range(20)})
    stale = load_test_registry(tmp_path)                                   # a writer's old copy
    stale["h0000000"] = {"model_name": "run_0", "status": "DONE", "env": "dev"}

    def writer(i):
        update_test_status(tmp_path, model_id=f"h{i:07d}", new_status="DONE", env="dev")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    update_test_registry(tmp_path, lambda reg: reg.update({"h0000000": stale["h0000000"]}))
    got = _read(tmp_path, "test_registry.json")
    assert {k for k in got if k.startswith("h")} == {f"h{i:07d}" for i in range(20)}
    assert all(got[f"h{i:07d}"]["status"] == "DONE" for i in range(20))
    assert (tmp_path / "artifacts" / ".test_registry.json.lock").exists()
    with test_registry_lock(tmp_path):
        pass                                                               # re-entrant from the same thread: released


def test_generated_model_names_are_shell_words(monkeypatch):
    """funny-words can return punctuation (sickish_o'clock): such a name is drawn again, never used."""
    from services.side_channel_core.utils import model_naming

    names = iter([b"sickish_o'clock\n", b"hooligan_peek-a-boo\n"])
    monkeypatch.setattr(model_naming.subprocess, "check_output", lambda *a, **k: next(names))
    assert model_naming.generate_unique_model_name(set()) == "hooligan_peek-a-boo"
