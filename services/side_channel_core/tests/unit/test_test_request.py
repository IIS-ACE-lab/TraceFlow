# services/side_channel_core/tests/unit/test_test_request.py
"""Unit tests for make test (re-test trained models by name)."""

import json
from pathlib import Path

import pytest
import yaml

from services.side_channel_core.utils.seal import strip as strip_seal  # rows and registries carry their "_tfseal" (utils/seal.py)
from services.side_channel_core.src.sca import test_request as tr
from services.side_channel_core.src.sca.test_request import (
    TestRequestError,
    load_test_request,
    resolve_models,
)

REPO = Path(__file__).resolve().parents[4]


def _write(tmp_path, block):
    path = tmp_path / "test_config.yaml"
    path.write_text(yaml.safe_dump({"test_request": block}))
    return path


@pytest.mark.parametrize(
    "path",
    [
        REPO / "projects/DLSCA_ASCAD/config/test_config.yaml",
        REPO / "services/side_channel_core/config/blueprint/test_config.yaml",
    ],
)
def test_shipped_files_are_valid(path):
    names = load_test_request(path)
    assert isinstance(names, list) and all(isinstance(n, str) and n for n in names)


@pytest.mark.parametrize(
    "model, expected",
    [
        (None, []),
        ("model_a", ["model_a"]),
        ("model_a, model_b", ["model_a", "model_b"]),
        (["model_a", "model_b", "model_a"], ["model_a", "model_b"]),
    ],
)
def test_model_accepts_a_name_or_a_list(tmp_path, model, expected):
    assert load_test_request(_write(tmp_path, {"model": model})) == expected


def test_only_model_and_the_external_attack_are_allowed(tmp_path):
    with pytest.raises(TestRequestError, match="Unknown key"):
        load_test_request(_write(tmp_path, {"model": "a", "runtime": {"batch_size": 1}}))
    with pytest.raises(TestRequestError, match="changes only"):
        load_test_request(_write(tmp_path, {"model": "a", "perform_attack": {"validation": {"attack_times": 2}}}))
    with pytest.raises(TestRequestError, match="unknown key"):
        load_test_request(_write(tmp_path, {"model": "a", "models": {"a": {"trainer": {}}}}))


def test_external_settings_merge_the_model_over_every_model(tmp_path):
    path = _write(tmp_path, {
        "model": ["a", "b"],
        "perform_attack": {"test": {"attack_times": 100, "shuffle": True, "attack_distribution": "full_population"},
                           "attack_step": [1], "ge_thresholds": 1},
        "models": {"b": {"perform_attack": {"test": {"attack_times": 10}, "attack_step": 5}}},
    })
    assert tr.external_settings(path, "a") == {
        "attack_step": 1, "ge_thresholds": 1,
        "test": {"attack_times": 100, "shuffle": True, "attack_distribution": "full_population"}}
    assert tr.external_settings(path, "b") == {
        "attack_step": 5, "ge_thresholds": 1,
        "test": {"attack_times": 10, "shuffle": True, "attack_distribution": "full_population"}}


@pytest.mark.parametrize("attack, message", [
    ({"test": {"attack_times": 0}}, "attacks the test traces"),
    ({}, "attacks the test traces"),
    ({"test": {"attack_times": 5}, "attack_step": [1, 10]}, "not a sweep list"),
    ({"test": {"attack_times": 5, "repeats": 3}}, "unknown key"),
])
def test_external_settings_refuse_what_attacks_nothing_or_sweeps(tmp_path, attack, message):
    with pytest.raises(TestRequestError, match=message):
        tr.external_settings(_write(tmp_path, {"model": "a", "perform_attack": attack}), "a")


@pytest.fixture
def project(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "model_name_map.json").write_text(
        json.dumps({"h1": "model_a", "h2": "model_b", "h3": "model_c"})
    )
    rows = [
        {"model_hash": "h1", "model_name": "model_a", "env": "dev", "status": "DONE", "path": "artifacts/dev/generated_models/model_a"},
        {"model_hash": "h2", "model_name": "model_b", "env": "dev", "status": "FAILED"},
        {"model_hash": "h3", "model_name": "model_c", "env": "staging", "status": "DONE"},
    ]
    (artifacts / "run_map.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(tr, "_project_block", lambda project: ({}, tmp_path))
    return tmp_path


def test_names_resolve_through_name_map_and_run_map(project):
    runs = resolve_models("P", ["model_a", "h1"], env="dev")
    assert [(r["model_name"], r["model_hash"]) for r in runs] == [("model_a", "h1"), ("model_a", "h1")]
    assert runs[0]["path"] == "artifacts/dev/generated_models/model_a"


@pytest.mark.parametrize(
    "names, message",
    [
        (["nope"], "nope: not in"),
        (["model_b"], "not DONE"),
        (["model_c"], "ENV=staging"),
        ([], "No model to test"),
    ],
)
def test_unusable_models_stop_everything(project, names, message):
    with pytest.raises(TestRequestError, match=message):
        resolve_models("P", names, env="dev")


# -----------------------------------------------------------------------------
# External tests: beside the recorded test, never over it
# -----------------------------------------------------------------------------
RECORDED = {  # a recorded test, as the training lifecycle wrote it
    "model_name": "model_a", "status": "SKIPPED", "path": "artifacts/dev/generated_models/model_a", "env": "dev",
    "batches": ["fig24"], "batch_ids": ["fig24-20261013T221049Z"], "model_id": "M",
    "key_recovered": "not available: perform_attack.test.attack_times is 0", "true_key": "224 (0xE0)",
    "recovered_key": "not available: perform_attack.test.attack_times is 0",
    "recovered_key_rank": "not available: perform_attack.test.attack_times is 0",
    "status_reason": "perform_attack.test.attack_times is 0", "finished_at": "2026-10-14T16:44:44+08:00",
}


@pytest.fixture
def external(project, monkeypatch):
    """project + a recorded test, a trained run, and the test path with its heavy parts stubbed."""
    import utils.checkpoint_policy as cp
    import utils.logger as lg
    import utils.sweep_engine as se
    import utils.sweep_test_runner as stw

    art = project / "artifacts"
    (art / "test_registry.json").write_text(json.dumps({"h1": RECORDED}))
    (project / "test_config.yaml").write_text(yaml.safe_dump({"test_request": {
        "model": ["model_a"],
        "perform_attack": {"test": {"attack_times": 100, "shuffle": True, "attack_distribution": "full_population"},
                           "attack_step": 1, "ge_thresholds": 1}}}))
    run_dir = project / "artifacts/dev/generated_models/model_a"
    monkeypatch.setattr(tr, "test_config_path", lambda p, o=None: project / "test_config.yaml")
    monkeypatch.setattr(tr, "load_snapshot", lambda p, run: {"env": "dev"})
    monkeypatch.setattr(se, "load_registry", lambda root, heal=False: {"h1": {"model_name": "model_a", "status": "DONE"}})
    snapshot_attack = {"test": {"attack_times": 0, "shuffle": False, "attack_distribution": "full_population"},
                       "validation": {"attack_times": 1}, "attack_step": 1, "ge_thresholds": 1}
    built = {}

    def build(info, *, model_hash, model_name, cfg):
        built["cfg"] = {"model_id": "M", "perform_attack": json.loads(json.dumps(snapshot_attack)),
                        "paths": {"artifacts_root": run_dir, "metrics": run_dir / "metrics/attack_metrics",
                                  "key_recovery": run_dir / "metrics/key_recovery", "logs": run_dir / "logs"},
                        "project": {"scripts": {"test": "src/sca/test.py"}},
                        "key_recovery_reporting": {"enabled": True}}
        return built["cfg"]

    monkeypatch.setattr(stw, "_build_test_cfg", build)
    sha = {"value": "a" * 64}
    monkeypatch.setattr(cp, "resolve_test_checkpoint", lambda cfg, model_name: type(
        "R", (), {"effective_policy": "final_epoch", "checkpoint_sha256": sha["value"]})())
    monkeypatch.setattr(lg, "configure_global_logging", lambda cfg=None, model_hash=None: str(cfg["paths"]["logs"]))
    import src.sca.key_enumeration as ke
    monkeypatch.setattr(ke, "validate_key_enumeration_contract", lambda cfg: {"enabled": False})
    tested = []
    module = type("M", (), {"test": staticmethod(lambda cfg, **kw: tested.append(json.loads(json.dumps(cfg, default=str))))})
    monkeypatch.setattr(tr, "_external_outcome", lambda cfg, name, log: {
        "key_recovered": True, "true_key": "224 (0xE0)", "recovered_key": "224 (0xE0)", "recovered_key_rank": 1})
    return {"root": project, "run_dir": run_dir, "tested": tested, "module": module, "sha": sha, "built": built}


def _run(external, models=None):
    return tr.run_external_test(project="P", models=models, env="dev", load_module=lambda s: external["module"])


def test_an_external_test_appends_beside_the_recorded_test(external):
    out = _run(external)
    registry = json.loads((external["root"] / "artifacts/test_registry.json").read_text())
    entry = registry["h1"]
    assert {k: entry[k] for k in RECORDED} == RECORDED                       # the recorded test is untouched
    (key, record), = entry["external_tests"].items()
    assert key == out[0]["key"] == "h1_et_1"
    assert record["index"] == 1 and record["runs"] == 1 and record["status"] == "DONE"
    assert record["recovered_key_rank"] == 1 and record["key_recovered"] is True
    assert record["status_reason"] == "completed: the key is recovered (true-key rank 1)"
    assert record["settings_id"] == tr.external_test_id(record["settings"])
    assert record["settings"] == {
        "perform_attack": {"test": {"attack_times": 100, "shuffle": True, "attack_distribution": "full_population"},
                           "attack_step": 1, "ge_thresholds": 1},
        "load_model": "final_epoch", "checkpoint_sha256": "a" * 64}
    assert record["report"] == "artifacts/dev/generated_models/model_a/test/h1_et_1"
    # the test ran with the external attack and wrote under test/<id>/, not over the recorded test's files
    cfg = external["tested"][0]
    assert cfg["perform_attack"]["test"]["attack_times"] == 100
    assert cfg["perform_attack"]["validation"] == {"attack_times": 1}
    out_dir = external["run_dir"] / "test" / "h1_et_1"
    assert cfg["paths"]["metrics"] == str(out_dir / "metrics/attack_metrics")
    assert cfg["paths"]["key_recovery"] == str(out_dir / "metrics/key_recovery")
    assert cfg["paths"]["logs"] == str(out_dir / "logs")
    assert json.loads((out_dir / "test_settings.json").read_text())["key"] == "h1_et_1"


def test_every_test_is_appended_and_the_same_settings_too(external):
    """Owner 2026-10-27: a test never replaces another; the same settings again take the next index, so the
    latest of each kind is the one read and an earlier, better result cannot be chosen over it."""
    first = _run(external)[0]["key"]
    again = _run(external)[0]["key"]
    external["sha"]["value"] = "b" * 64                                       # retrained: other weights, same hash
    retrained = _run(external)[0]["key"]
    tests = json.loads((external["root"] / "artifacts/test_registry.json").read_text())["h1"]["external_tests"]
    assert (first, again, retrained) == ("h1_et_1", "h1_et_2", "h1_et_3")
    assert [tests[t]["index"] for t in (first, again, retrained)] == [1, 2, 3]
    assert tests[first]["settings_id"] == tests[again]["settings_id"] != tests[retrained]["settings_id"]
    assert all(t["kind"] == "all" and t["source"] == "test" and t["runs"] == 1 for t in tests.values())


def test_a_retest_is_appended_with_the_snapshots_attack(external):
    out = tr.run_test_request(project="P", models=None, env="dev", load_module=lambda s: external["module"])
    (record,) = json.loads((external["root"] / "artifacts/test_registry.json").read_text())["h1"]["external_tests"].values()
    assert out[0]["key"] == "h1_et_1" and record["source"] == "retest" and record["kind"] == "all"
    assert record["settings"]["perform_attack"]["test"]["attack_times"] == 0  # the snapshot's, not test_config.yaml's 100


def test_a_budget_test_attacks_with_the_data_entrys_budget_and_is_refused_without_one(external, monkeypatch):
    import utils.sweep_test_runner as stw

    monkeypatch.setattr(stw, "data_budget", lambda cfg: 250)
    (out,) = tr.run_external_test(project="P", models=None, env="dev", load_module=lambda s: external["module"], kind="budget")
    record = json.loads((external["root"] / "artifacts/test_registry.json").read_text())["h1"]["external_tests"][out["key"]]
    assert record["kind"] == "budget" and record["settings"]["perform_attack"]["test"]["attack_trace_budget"] == 250
    assert external["tested"][-1]["perform_attack"]["test"]["attack_trace_budget"] == 250
    monkeypatch.setattr(stw, "data_budget", lambda cfg: None)
    with pytest.raises(RuntimeError, match="needs its data entry's attack_trace_budget"):
        tr.run_external_test(project="P", models=None, env="dev", load_module=lambda s: external["module"], kind="budget")


def test_a_failed_external_test_is_recorded_and_reported(external):
    external["module"].test = staticmethod(lambda cfg, **kw: (_ for _ in ()).throw(RuntimeError("no GPU")))
    with pytest.raises(RuntimeError, match="External test failed for: model_a"):
        _run(external)
    entry = json.loads((external["root"] / "artifacts/test_registry.json").read_text())["h1"]
    (record,) = entry["external_tests"].values()
    assert record["status"] == "FAILED"
    assert record["status_reason"] == "system failure: GPU: RuntimeError: no GPU"
    assert entry["status"] == "SKIPPED"


def test_a_recorded_retest_keeps_the_batches_and_the_external_tests(project, monkeypatch):
    import utils.sweep_test_runner as stw

    kept = {"h1": {**RECORDED, "external_tests": {"abcd1234": {"index": 1, "status": "DONE"}}}}
    saved = {}
    monkeypatch.setattr(stw, "sync_run_map_before_planning", lambda root: None)
    monkeypatch.setattr(stw, "load_train_registry", lambda root: {"h1": {"model_name": "model_a", "status": "DONE", "env": "dev", "path": "p"}})
    monkeypatch.setattr(stw, "load_test_registry", lambda root: json.loads(json.dumps(kept)))
    monkeypatch.setattr(stw, "load_run_map", lambda root: {})
    # every write goes through update_test_registry (the lock): the change is applied to the "file" here
    monkeypatch.setattr(stw, "update_test_registry", lambda root, change, **kw: (change(saved), saved)[1])
    monkeypatch.setattr(stw, "_build_test_cfg", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop here")))
    with pytest.raises(RuntimeError, match="Test failed"):
        stw.run_test_sweep(lambda *a, **k: None, {"env": "dev"}, "test", project, ["model_a"], False, False, force=True)
    entry = saved["h1"]
    assert entry["batches"] == ["fig24"] and entry["batch_ids"] == RECORDED["batch_ids"]
    assert entry["external_tests"] == {"abcd1234": {"index": 1, "status": "DONE"}}
    assert entry["status"] == "FAILED"


def test_a_test_that_cannot_start_is_recorded_with_its_reason(external, monkeypatch):
    """A checkpoint that fails its check: filed under the requested settings, with why."""
    import utils.checkpoint_policy as cp

    def broken(cfg, model_name):
        raise cp.CheckpointPolicyError("final-epoch checkpoint SHA-256 mismatch")

    monkeypatch.setattr(cp, "resolve_test_checkpoint", broken)
    with pytest.raises(RuntimeError, match="checkpoint missing or corrupt"):
        _run(external)
    (key, record), = json.loads((external["root"] / "artifacts/test_registry.json").read_text())["h1"]["external_tests"].items()
    assert key == "h1_et_1" and record["status"] == "FAILED" and record["settings"]["checkpoint_sha256"] is None
    assert record["status_reason"] == (
        "checkpoint missing or corrupt: CheckpointPolicyError: final-epoch checkpoint SHA-256 mismatch")
    assert "report" not in record and external["tested"] == []


@pytest.mark.parametrize("exc, category", [
    (tr.TestRequestError("Training snapshot not found: /x"), "snapshot missing"),
    (tr.TestRequestError("m (h) is not in artifacts/run_registry.json"), "model not found"),
    (tr.TestRequestError("the external test wrote no key-recovery report (test.json)"), "report incomplete"),
    (ValueError("params_out snapshot /x is truncated or damaged: missing ['data']"), "snapshot corrupt"),
    (ValueError("Snapshot of m hashes to 1234abcd, not to its registered hash"), "snapshot does not match its model hash"),
    (FileNotFoundError("test.h5 missing for model m"), "data missing"),
    (OSError("Unable to open file (truncated file)"), "data corrupt"),
    (OSError("No space left on device"), "system failure: disk or file system"),
    (MemoryError(), "system failure: out of memory"),
    (RuntimeError("CUDA error: device-side assert triggered"), "system failure: GPU"),
    (ValueError("perform_attack.test.attack_times must divide the population"), "settings incomplete"),
    (ZeroDivisionError("float division by zero"), "test failure"),
])
def test_failure_reasons_say_what_failed(exc, category):
    reason = tr.failure_reason(exc)
    assert reason.startswith(category + ": ") and type(exc).__name__ in reason


def test_done_reasons():
    assert tr.done_reason({"key_recovered": True, "recovered_key_rank": 1}) == "completed: the key is recovered (true-key rank 1)"
    assert tr.done_reason({"key_recovered": False, "recovered_key_rank": 37}) == "completed: the key is not recovered (true-key rank 37)"


def test_records_of_the_earlier_format_are_rekeyed_never_merged(external):
    """puzzle_bog, 2026-10-14: a record keyed by its settings hash (no settings_id) beside one of the same settings
    under <hash>_et_2: the next test re-keys the old one to its index, keeps both, and is appended as et_3."""
    first = _run(external)[0]
    registry_path = external["root"] / "artifacts/test_registry.json"
    registry = strip_seal(json.loads(registry_path.read_text()))   # an edit by hand: written back unsealed
    old = dict(registry["h1"]["external_tests"].pop("h1_et_1"), runs=1, finished_at="2026-10-14T21:28:45+08:00")
    old.pop("settings_id")
    dup = dict(old, index=2, settings_id=first["settings_id"], finished_at="2026-10-14T21:46:49+08:00")
    registry["h1"]["external_tests"] = {first["settings_id"]: old, "h1_et_2": dup}
    registry_path.write_text(json.dumps(registry))
    again = _run(external)[0]
    tests = json.loads(registry_path.read_text())["h1"]["external_tests"]
    assert sorted(tests) == ["h1_et_1", "h1_et_2", "h1_et_3"] and again["key"] == "h1_et_3"
    assert tests["h1_et_1"]["finished_at"] == "2026-10-14T21:28:45+08:00" and tests["h1_et_1"]["settings_id"]

