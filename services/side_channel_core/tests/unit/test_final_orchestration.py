# services/side_channel_core/tests/unit/test_final_orchestration.py
"""Orchestration, CLI entry points and registries (audit 2026-09-29, lane O).

SYS-11: a signal interrupts only this process's hashes; status updates never
        overwrite a registry entry's model_id.
SYS-27: train-cmd -n and the no-flag orchestrate fallback run registered
        hashes from their snapshot through train_and_test, keyed by
        model_hash, under the retry budget; failures exit non-zero.
SYS-30: registry.jsonl keeps long arrays by reference (.npy sidecar), and
        its readers cope with the reference.

Verifier follow-ups: a DONE run that cannot be re-run stays DONE; the no-flag
pass and the shell's host-side reconciliation leave other orchestrators' live
runs alone; snapshot and name-map readers refuse damaged files.

Run from the repository root (CPU only):
    PYTHONPATH="$PWD:$PWD/services/side_channel_core:$PWD/services/side_channel_core/src" \
        python -m pytest services/side_channel_core/tests/unit/test_final_orchestration.py
"""
import json
import logging
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from utils.seal import strip as strip_seal  # rows and registries carry their "_tfseal" (utils/seal.py)



# --------------------------------------------------------------------- helpers
def _write_registries(root, runs, tests=None, run_map=None):
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "run_registry.json").write_text(json.dumps(runs))
    (artifacts / "test_registry.json").write_text(json.dumps(tests or {}))
    if run_map:
        (artifacts / "run_map.jsonl").write_text("".join(json.dumps(r) + "\n" for r in run_map))


def _read(root, name):
    return strip_seal(json.loads((root / "artifacts" / name).read_text()))


def _entry(name, status, model_id="MLP1", env="dev", **extra):
    return {"model_name": name, "status": status, "env": env, "model_id": model_id,
            "path": f"artifacts/{env}/generated_models/{name}", **extra}


@pytest.fixture
def orch(monkeypatch):
    """core.orchestrator with fresh globals; its import-time signal handlers are undone."""
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    from core import orchestrator
    for sig, handler in saved.items():
        signal.signal(sig, handler)
    monkeypatch.setattr(orchestrator, "SELECTED_HASHES", set())
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", None)
    monkeypatch.setattr(orchestrator, "RUN_ENV", None)
    return orchestrator


# ---------------------------------------------------------------------- SYS-11
def test_update_status_never_overwrites_model_id(tmp_path, monkeypatch):
    from utils.sweep_engine import update_status

    monkeypatch.setenv("MODEL_ID", "CNN1")
    _write_registries(
        tmp_path,
        {"aaaa": _entry("m-a", "in_progress", model_id="MLP1"), "bbbb": _entry("m-b", "in_progress", model_id=None)},
        run_map=[{"model_hash": "bbbb", "model_name": "m-b", "model_id": "CNN9", "status": "IN_PROGRESS"}],
    )
    update_status(tmp_path, model_id="aaaa", new_status="INTERRUPTED")
    update_status(tmp_path, model_id="bbbb", new_status="INTERRUPTED")

    runs = _read(tmp_path, "run_registry.json")
    assert runs["aaaa"]["status"] == "INTERRUPTED" and runs["aaaa"]["model_id"] == "MLP1"
    # Missing: backfilled from the hash's own run_map record, not the caller's MODEL_ID.
    assert runs["bbbb"]["model_id"] == "CNN9"


def test_update_test_status_backfills_model_id_only_when_missing(tmp_path, monkeypatch):
    from utils.test_registry import update_test_status

    monkeypatch.setenv("MODEL_ID", "CNN1")
    _write_registries(
        tmp_path,
        {h: _entry(f"m-{h}", "DONE") for h in ("aaaa", "bbbb", "cccc")},
        tests={"aaaa": _entry("m-a", "queued", model_id="MLP1"), "bbbb": _entry("m-b", "queued", model_id="")},
        run_map=[{"model_hash": "bbbb", "model_name": "m-b", "model_id": "CNN9"}],
    )
    for h in ("aaaa", "bbbb", "cccc"):
        update_test_status(tmp_path, model_id=h, new_status="FAILED")

    tests = _read(tmp_path, "test_registry.json")
    assert tests["aaaa"]["model_id"] == "MLP1"
    assert tests["bbbb"]["model_id"] == "CNN9"
    assert tests["cccc"]["model_id"] == "CNN1"  # no entry, no run_map record: the caller's MODEL_ID


def test_update_test_status_rejects_a_model_name_key(tmp_path, caplog):
    from utils.test_registry import update_test_status

    _write_registries(tmp_path, {"aaaa": _entry("m-a", "DONE")}, tests={"aaaa": _entry("m-a", "DONE")})
    with pytest.raises(KeyError, match="keyed by model_hash"):
        update_test_status(tmp_path, model_id="m-a", new_status="DONE")
    assert list(_read(tmp_path, "test_registry.json")) == ["aaaa"]


def test_signal_interrupts_only_this_process_hashes(tmp_path, monkeypatch, orch):
    monkeypatch.setenv("MODEL_ID", "MLP1")
    _write_registries(
        tmp_path,
        {
            "aaaa": _entry("m-a", "in_progress", model_id="MLP1"),   # ours
            "bbbb": _entry("m-b", "in_progress", model_id="CNN1"),   # another orchestrator
            "cccc": _entry("m-c", "in_progress", env="staging"),     # ours by hash, other env
        },
        tests={h: _entry(f"m-{h}", "queued") for h in ("aaaa", "bbbb")},
    )
    interrupted = []
    monkeypatch.setattr(orch, "interrupt_run_map_entry", lambda **kw: interrupted.append(kw["model_hash"]))
    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(orch, "RUN_ENV", "dev")
    orch.SELECTED_HASHES.update({"aaaa", "cccc"})

    with pytest.raises(SystemExit) as exc:
        orch.handle_signal(signal.SIGTERM, None)

    assert exc.value.code == 130
    runs, tests = _read(tmp_path, "run_registry.json"), _read(tmp_path, "test_registry.json")
    assert runs["aaaa"]["status"] == "INTERRUPTED" and tests["aaaa"]["status"] == "FAILED"
    assert runs["bbbb"]["status"] == "in_progress" and runs["bbbb"]["model_id"] == "CNN1"
    assert tests["bbbb"]["status"] == "queued"
    assert runs["cccc"]["status"] == "in_progress"
    assert interrupted == ["aaaa"]


def test_a_terminated_run_closes_its_heartbeat(tmp_path, monkeypatch, orch):
    """Terminate Run stops the container: SIGINT reaches this process, whose handler ends it with sys.exit. The
    run's heartbeat is closed first (interrupted), not left beating "running" for a run that is gone."""
    from utils import heartbeat

    monkeypatch.setattr(heartbeat, "_active", None)
    beat = heartbeat.begin({"model_name": "m-a", "model_hash": "aaaa"}, tmp_path, "dev")
    assert json.loads(heartbeat.heartbeat_path(tmp_path, "dev").read_text())["status"] == "running"
    with pytest.raises(SystemExit) as exc:
        orch.handle_signal(signal.SIGINT, None)
    assert exc.value.code == 130
    written = json.loads(heartbeat.heartbeat_path(tmp_path, "dev").read_text())
    assert written["status"] == "interrupted" and written["model_hash"] == "aaaa"
    assert not beat._thread.is_alive()                                    # it beats no more


def test_planned_hashes_are_claimed_for_the_signal_handler(monkeypatch, orch):
    monkeypatch.setattr(orch, "_plan_scc_models_from_sweep_engine", lambda **kw: [("dddd", {"model_name": "m-d"})])
    assert orch._plan_next(cfg={}, project_root=None, limit=1) == [("dddd", {"model_name": "m-d"})]
    assert orch.SELECTED_HASHES == {"dddd"}


# ---------------------------------------------------------------------- SYS-27
@pytest.fixture
def main_env(tmp_path, monkeypatch, orch):
    """orchestrator.main() on a tmp project: every heavy step is recorded instead of run."""
    cfg = {
        "project": {"root": str(tmp_path), "name": "P", "env_configs": {"dev": "config/env/dev.yaml"}},
        "data": {"ascad": {"name": "ascad"}},
        "model": [{"model_id": "MLP1"}],
        "paths": {"artifacts_root": "artifacts/{{env}}/generated_models/{{model_name}}",
                  "params_out": "artifacts/{{env}}/generated_models/{{model_name}}/config/{{model_name}}_params.yaml"},
    }
    calls = {"resolve": [], "train_and_test": [], "plan": []}

    def _resolve(**kw):
        calls["resolve"].append(kw["model_hash"])
        name = kw["info"]["model_name"]
        return {"model_hash": kw["model_hash"], "model_name": name, "runtime": {}, "env": "dev",
                "paths": {"artifacts_root": f"artifacts/dev/generated_models/{name}"}}, "snapshot"

    def _plan(**kw):
        calls["plan"].append(kw)
        return []

    monkeypatch.delenv("ORCHESTRATOR_SESSION_ID", raising=False)
    monkeypatch.delenv("AUTO_SEARCH_SESSION_ID", raising=False)
    monkeypatch.setenv("MODEL_ID", "MLP1")
    monkeypatch.setattr(orch, "get_project_config", lambda *a, **k: json.loads(json.dumps(cfg)))
    monkeypatch.setattr(orch, "_derive_attempt_window", lambda *a, **k: (1, 1, 1))
    monkeypatch.setattr(orch, "resolve_registered_run_cfg", _resolve)
    monkeypatch.setattr(orch, "_plan_scc_models_from_sweep_engine", _plan)
    monkeypatch.setattr(orch, "prepare_trial_runtime", lambda **kw: None)
    monkeypatch.setattr(orch, "ensure_runtime_identifiers", lambda *a, **k: None)
    monkeypatch.setattr(orch, "resolve_path", lambda value, *a, **k: value)
    monkeypatch.setattr(orch, "complete_run_map_entry", lambda **kw: None)
    monkeypatch.setattr(orch, "fail_run_map_entry", lambda **kw: None)
    monkeypatch.setattr(orch, "train_and_test", lambda run_cfg, env, **kw: calls["train_and_test"].append(run_cfg))

    def run(**kw):
        return orch.main(project="P", dataset="ascad", env="dev", model_id="MLP1", **kw)

    return tmp_path, calls, run


def _fail(*a, **kw):
    raise RuntimeError("boom")


def test_no_flag_pass_runs_a_failed_hash_from_its_snapshot(main_env):
    root, calls, run = main_env
    _write_registries(root, {"aaaa": _entry("m-a", "FAILED", failed_attempts=1)},
                      tests={"aaaa": _entry("m-a", "FAILED")})

    with pytest.raises(SystemExit) as exc:
        run()

    assert exc.value.code == 0
    assert calls["resolve"] == ["aaaa"]
    assert [(c["model_hash"], c["model_name"]) for c in calls["train_and_test"]] == [("aaaa", "m-a")]
    runs, tests = _read(root, "run_registry.json"), _read(root, "test_registry.json")
    assert runs["aaaa"]["status"] == "DONE" and "failed_attempts" not in runs["aaaa"]
    assert tests == {"aaaa": tests["aaaa"]} and tests["aaaa"]["status"] == "DONE"


def test_no_flag_pass_failure_is_recorded_and_raised(main_env, monkeypatch, orch):
    root, calls, run = main_env
    _write_registries(root, {"aaaa": _entry("m-a", "INTERRUPTED")}, tests={"aaaa": _entry("m-a", "FAILED")})
    monkeypatch.setattr(orch, "train_and_test", _fail)

    with pytest.raises(RuntimeError, match="boom"):
        run()

    runs, tests = _read(root, "run_registry.json"), _read(root, "test_registry.json")
    assert runs["aaaa"]["status"] == "FAILED" and runs["aaaa"]["failed_attempts"] == 1
    assert list(tests) == ["aaaa"] and tests["aaaa"]["status"] == "FAILED"


def test_no_flag_pass_respects_the_retry_budget(main_env, monkeypatch, orch):
    root, calls, run = main_env
    _write_registries(root, {"aaaa": _entry("m-a", "FAILED", failed_attempts=orch.MAX_FAILED_ATTEMPTS)})

    with pytest.raises(SystemExit) as exc:
        run()

    assert exc.value.code == orch.EXIT_RETRY_EXHAUSTED
    assert calls["train_and_test"] == []


def test_no_flag_pass_leaves_an_untested_hash_to_make_test(main_env):
    root, calls, run = main_env
    _write_registries(root, {"aaaa": _entry("m-a", "DONE")}, tests={"aaaa": _entry("m-a", "FAILED")})

    with pytest.raises(RuntimeError, match=r"make test ENV=dev PROJECT=P MODEL=m-a"):
        run()
    assert calls["train_and_test"] == []
    assert _read(root, "test_registry.json")["aaaa"]["status"] == "FAILED"


def test_registered_run_is_keyed_by_hash_and_never_planned(main_env):
    root, calls, run = main_env
    _write_registries(root, {"aaaa": _entry("m-a", "DONE")}, tests={"aaaa": _entry("m-a", "DONE")})

    with pytest.raises(SystemExit) as exc:
        run(include_models=["m-a"], registered_only=True)

    assert exc.value.code == 0
    assert calls["plan"] == [] and calls["resolve"] == ["aaaa"]
    assert len(calls["train_and_test"]) == 1
    assert sorted(_read(root, "run_registry.json")) == ["aaaa"]
    assert sorted(_read(root, "test_registry.json")) == ["aaaa"]


def test_registered_run_failure_is_recorded_and_raised(main_env, monkeypatch, orch):
    root, calls, run = main_env
    _write_registries(root, {"aaaa": _entry("m-a", "DONE")}, tests={"aaaa": _entry("m-a", "DONE")})
    monkeypatch.setattr(orch, "train_and_test", _fail)

    with pytest.raises(RuntimeError, match="boom"):
        run(include_models=["aaaa"], registered_only=True)

    runs = _read(root, "run_registry.json")
    assert runs["aaaa"]["status"] == "FAILED" and runs["aaaa"]["failed_attempts"] == 1
    assert _read(root, "test_registry.json")["aaaa"]["status"] == "FAILED"


@pytest.mark.parametrize(
    "entry, selector, error, message",
    [
        (_entry("m-a", "DONE"), "m-zzz", ValueError, "not registered runs"),
        (_entry("m-a", "DONE", model_id="CNN1"), "m-a", ValueError, "not registered runs"),
        (_entry("m-a", "FAILED", failed_attempts=3), "m-a", RuntimeError, "retry budget exhausted"),
    ],
)
def test_registered_run_outside_scope_or_budget_raises(main_env, entry, selector, error, message):
    root, calls, run = main_env
    _write_registries(root, {"aaaa": entry})

    with pytest.raises(error, match=message):
        run(include_models=[selector], registered_only=True)
    assert calls["plan"] == [] and calls["train_and_test"] == []


@pytest.fixture
def cli(tmp_path, monkeypatch, orch):
    """core.cli.train_cmd on a tmp project; orchestrator.main is recorded."""
    monkeypatch.setenv("ALLOW_HOST_RUN", "1")
    monkeypatch.delenv("MODEL_ID", raising=False)
    from core import cli as cli_mod

    cfg = {"project": {"root": str(tmp_path), "name": "P", "scripts": {"train": "train.py"}},
           "data": {"ascad": {"name": "ascad"}}}
    monkeypatch.setattr(cli_mod, "get_project_config", lambda *a, **k: json.loads(json.dumps(cfg)))
    monkeypatch.setattr(cli_mod, "_load_module", lambda path: object())
    monkeypatch.setattr(cli_mod, "run_sweep", lambda **kw: pytest.fail("train-cmd -n must not run a sweep"))
    calls = []
    monkeypatch.setattr(orch, "main", lambda **kw: calls.append(kw))

    def train_cmd(models, **overrides):
        args = dict(project="P", dataset="ascad", hypothesis=None, env="dev", include_models=models,
                    model_id=None, retry_failed=False, resume_interrupted=False, force=False, alias=None)
        args.update(overrides)
        return cli_mod.train_cmd(**args)

    return calls, train_cmd


def test_train_cmd_n_runs_registered_models_through_the_orchestrator(tmp_path, cli):
    calls, train_cmd = cli
    _write_registries(tmp_path, {"aaaa1111": _entry("m-a", "DONE", model_id="CNN1"),
                                 "bbbb2222": _entry("m-b", "FAILED", model_id="CNN1")})

    train_cmd(["m-a", "x-bbbb2222"], force=True)

    assert calls == [dict(project="P", dataset="ascad", env="dev", include_models=["m-a", "bbbb2222"],
                          model_id="CNN1", force=True, registered_only=True)]


def test_train_cmd_n_says_that_retry_flags_do_not_apply(tmp_path, cli, caplog):
    calls, train_cmd = cli
    _write_registries(tmp_path, {"aaaa1111": _entry("m-a", "DONE")})

    with caplog.at_level(logging.WARNING):
        train_cmd(["m-a"], retry_failed=True, model_id="MLP1")

    assert "-r/-c have no effect" in caplog.text
    assert [c["include_models"] for c in calls] == [["m-a"]]


@pytest.mark.parametrize("models, alias, message", [
    (["m-unknown"], None, "make orchestrate"),
    (["m-a"], "new-name", "identity is frozen"),
])
def test_train_cmd_n_rejects_new_runs(tmp_path, cli, models, alias, message):
    calls, train_cmd = cli
    _write_registries(tmp_path, {"aaaa1111": _entry("m-a", "DONE")})

    with pytest.raises(ValueError, match=message):
        train_cmd(models, alias=alias)
    assert calls == []


def test_unexpanded_learning_rate_is_rejected():
    from utils.cli_helpers import _ensure_train_safe_cfg

    optimizer = {"type": "RMSprop", "learning_rate": [1e-5, 2e-5]}
    with pytest.raises(RuntimeError, match=r"trainer\.optimizer\.learning_rate=\[1e-05, 2e-05\]"):
        _ensure_train_safe_cfg({"trainer": {"optimizer": optimizer}}, model_id="MLP1")


# ---------------------------------------------------------------------- SYS-30
def test_graph_loader_does_not_take_the_sidecar_for_an_epoch(tmp_path):
    from services.graph_manager.metric_loader import epoch_array_candidates

    folder = tmp_path / "metrics/arrays/test/final_ranks"
    folder.mkdir(parents=True)
    np.save(folder / "ep0001.npy", np.arange(3))
    np.save(folder / "registry.npy", np.arange(100))
    assert [p.name for p in epoch_array_candidates(tmp_path / "metrics", "test", "final_ranks")] == ["ep0001.npy"]


# ------------------------------------------------- verifier follow-ups: SYS-27
def _legacy_snapshot(root, name, model_hash):
    """A params_out snapshot written before 2026-09-29: every section, no snapshot_sha256."""
    config = root / "artifacts/dev/generated_models" / name / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / f"{name}_params.yaml").write_text(yaml.safe_dump({
        "project": {"name": "P"}, "data": {"name": "ascad"}, "model": {"model_id": "MLP1"},
        "trainer": {"epochs": 1}, "seed": 1, "sca": {"leakage_model": "ID"},
        "paths": {"params_out": "artifacts/{{env}}/generated_models/{{model_name}}/config/{{model_name}}_params.yaml"},
        "model_hash": model_hash, "model_name": name, "env": "dev", "model_id": "MLP1",
    }))


def test_done_run_that_cannot_be_rerun_stays_done(main_env, monkeypatch, orch):
    from utils.cli_helpers import resolve_registered_run_cfg

    root, calls, run = main_env
    monkeypatch.setattr(orch, "resolve_registered_run_cfg", resolve_registered_run_cfg)
    _write_registries(root, {"aaaa": _entry("m-a", "DONE", finished_at="2026-09-01T10:00:00")},
                      tests={"aaaa": _entry("m-a", "DONE")})
    _legacy_snapshot(root, "m-a", "aaaa")

    with pytest.raises(RuntimeError, match=r"\['aaaa'\] cannot run"):
        run(include_models=["m-a"], registered_only=True)

    runs, tests = _read(root, "run_registry.json"), _read(root, "test_registry.json")
    assert runs["aaaa"]["status"] == "DONE" and runs["aaaa"]["finished_at"] == "2026-09-01T10:00:00"
    assert not {"retry_exhausted", "failed_attempts", "reason"} & set(runs["aaaa"])
    assert tests["aaaa"]["status"] == "DONE"
    assert calls["train_and_test"] == []


def _unresolvable(**kw):
    raise RuntimeError("snapshot does not match its snapshot_sha256")


@pytest.mark.parametrize("before, expected_status, terminal", [
    ("DONE", "DONE", False),        # -f claimed it (in_progress) before resolving: DONE is put back
    ("FAILED", "FAILED", True),     # not DONE: unrecoverable is terminal, as before
])
def test_unresolvable_hash_keeps_done_and_fails_the_rest(tmp_path, monkeypatch, orch, before, expected_status, terminal):
    monkeypatch.setattr(orch, "resolve_registered_run_cfg", _unresolvable)
    registry_before = {"aaaa": _entry("m-a", before, finished_at="2026-09-01T10:00:00")}
    _write_registries(tmp_path, {"aaaa": _entry("m-a", "in_progress")})   # the planner's claim

    runnable, blocked = orch._resolve_selected_models(
        [("aaaa", {"model_name": "m-a", "run_cfg": {}})], cfg={}, project_root=tmp_path,
        registry_before=registry_before, force=True, run_session_id=None,
    )

    entry = _read(tmp_path, "run_registry.json")["aaaa"]
    assert runnable == [] and blocked == ["aaaa"]
    assert entry["status"] == expected_status
    assert bool(entry.get("retry_exhausted")) is terminal
    if before == "DONE":
        assert entry["finished_at"] == "2026-09-01T10:00:00"


def test_no_flag_pass_leaves_live_runs_of_another_orchestrator_alone(main_env, orch):
    root, calls, run = main_env
    live = {"run_session_id": "20260929T100000Z-1"}
    _write_registries(
        root,
        {"aaaa": _entry("m-a", "in_progress", **live), "bbbb": _entry("m-b", "DONE", **live)},
        tests={"aaaa": _entry("m-a", "queued", **live), "bbbb": _entry("m-b", "in_progress", **live)},
    )

    with pytest.raises(SystemExit) as exc:
        run()

    assert exc.value.code == orch.EXIT_NOTHING_TO_RUN
    assert calls["resolve"] == [] and calls["train_and_test"] == []
    runs, tests = _read(root, "run_registry.json"), _read(root, "test_registry.json")
    assert runs["aaaa"]["status"] == "in_progress" and tests["bbbb"]["status"] == "in_progress"


def test_resume_flag_still_selects_an_in_progress_hash(main_env):
    root, calls, run = main_env
    _write_registries(root, {"aaaa": _entry("m-a", "in_progress")}, tests={"aaaa": _entry("m-a", "queued")})

    with pytest.raises(SystemExit) as exc:
        run(resume_interrupted=True)

    assert exc.value.code == 0 and calls["resolve"] == ["aaaa"]


# ------------------------------------------ verifier follow-ups: SYS-11 shell
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "orchestrator.sh"


def _host_exit_heredoc() -> str:
    text = _SCRIPT.read_text()
    body = re.search(r"_mark_interrupted_on_host_exit\(\) \{\n\s*\S+ - <<'PY'\n(.*?)\nPY\n", text, re.S)
    assert body, "orchestrator.sh: _mark_interrupted_on_host_exit heredoc not found"
    return body.group(1)


def _host_exit(root, **env):
    return subprocess.run(
        [sys.executable, "-c", _host_exit_heredoc()],
        env={**os.environ, "PROJECT_ROOT": str(root), **env},
        capture_output=True, text=True,
    )


def test_host_side_reconciliation_marks_only_this_session(tmp_path):
    ours = {"run_session_id": "S1"}
    runs = {
        "aaaa": _entry("m-a", "in_progress", **ours),                              # ours
        "bbbb": _entry("m-b", "in_progress", run_session_id="S2"),                 # another session
        "cccc": _entry("m-c", "in_progress", model_id="CNN1", **ours),             # another MODEL_ID
        "dddd": _entry("m-d", "in_progress", env="staging", **ours),               # another ENV
        "eeee": _entry("m-e", "DONE", **ours),
    }
    tests = {h: _entry(e["model_name"], "queued") for h, e in runs.items()}
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    # The registries' real layout: indented JSON (save_registry) and JSONL.
    (artifacts / "run_registry.json").write_text(json.dumps(runs, indent=2))
    (artifacts / "test_registry.json").write_text(json.dumps(tests, indent=2))
    (artifacts / "run_map.jsonl").write_text(
        json.dumps({"model_hash": "ffff", "model_name": "m-f", "status": "IN_PROGRESS", "env": "dev",
                    "model_id": "MLP1", "run_session_id": "S1"}) + "\n"
    )

    done = _host_exit(tmp_path, ENV="dev", MODEL_ID="MLP1", ORCHESTRATOR_SESSION_ID="S1")

    assert done.returncode == 0, done.stderr
    after_runs, after_tests = _read(tmp_path, "run_registry.json"), _read(tmp_path, "test_registry.json")
    assert {h: e["status"] for h, e in after_runs.items()} == {
        "aaaa": "INTERRUPTED", "bbbb": "in_progress", "cccc": "in_progress", "dddd": "in_progress", "eeee": "DONE",
    }
    assert after_tests["aaaa"]["status"] == "FAILED" and after_tests["bbbb"]["status"] == "queued"
    run_map = [strip_seal(json.loads(line)) for line in (artifacts / "run_map.jsonl").read_text().splitlines()]
    assert [(r["model_hash"], r["status"]) for r in run_map] == [("ffff", "INTERRUPTED")]


def test_host_side_reconciliation_without_a_session_marks_nothing(tmp_path):
    _write_registries(tmp_path, {"aaaa": _entry("m-a", "in_progress")})
    before = (tmp_path / "artifacts/run_registry.json").read_text()

    done = _host_exit(tmp_path, ENV="dev", MODEL_ID="", ORCHESTRATOR_SESSION_ID="")

    assert done.returncode == 0, done.stderr
    assert (tmp_path / "artifacts/run_registry.json").read_text() == before


# --------------------------------- verifier follow-ups: snapshot and name map
def _filter_project(root, snapshot_text):
    from services.side_channel_core.utils.data_reproducibility import SNAPSHOT_DIGEST_KEY, snapshot_digest

    run = {"model_hash": "h1", "model_name": "cnn_a", "env": "dev", "status": "DONE", "model_id": "M",
           "path": "artifacts/dev/generated_models/cnn_a", "sampled": {}}
    (root / "artifacts").mkdir(parents=True)
    (root / "artifacts/run_map.jsonl").write_text(json.dumps(run) + "\n")
    body = {"project": {"name": "P"}, "data": {"name": "ascad"}, "model": [{"model_id": "M"}],
            "trainer": {"epochs": 10}, "seed": 1, "sca": {"leakage_model": "ID"}, "paths": {},
            "cross_validation": {"n_folds": 10}, "model_hash": "h1", "model_name": "cnn_a"}
    config = root / "artifacts/dev/generated_models/cnn_a/config"
    config.mkdir(parents=True)
    text = yaml.safe_dump({SNAPSHOT_DIGEST_KEY: snapshot_digest(body), **body}, sort_keys=False)
    (config / "cnn_a_params.yaml").write_text(snapshot_text(text))


def _filter_file(tmp_path, where):
    path = tmp_path / "model_filter.yaml"
    path.write_text(yaml.safe_dump({"model_filter": {"where": where}}))
    return path


def test_model_filter_reads_a_verified_snapshot(tmp_path):
    from services.model_filter.model_filter import run_model_filter

    _filter_project(tmp_path / "P", lambda text: text)
    out = run_model_filter(_filter_file(tmp_path, [{"path": "cross_validation.n_folds", "equals": 10}]),
                           project="P", env="dev", projects_root=tmp_path)
    assert out["models"] == ["cnn_a"]


@pytest.mark.parametrize("damage", [
    lambda text: text.replace("n_folds: 10", "n_folds: 5"),     # edited after training
    lambda text: text[: len(text) // 2],                        # truncated
])
def test_model_filter_refuses_a_damaged_snapshot(tmp_path, damage):
    from services.model_filter.model_filter import ModelFilterError, run_model_filter

    _filter_project(tmp_path / "P", damage)
    with pytest.raises(ModelFilterError, match=r"cnn_a \(h1\)"):
        run_model_filter(_filter_file(tmp_path, [{"path": "cross_validation.n_folds", "equals": 5}]),
                         project="P", env="dev", projects_root=tmp_path)
    assert not (tmp_path / "P/artifacts/filtered_models.yaml").exists()


def test_model_filter_without_a_snapshot_leaves_its_keys_unmatched(tmp_path):
    from services.model_filter.model_filter import run_model_filter

    _filter_project(tmp_path / "P", lambda text: text)
    (tmp_path / "P/artifacts/dev/generated_models/cnn_a/config/cnn_a_params.yaml").unlink()
    out = run_model_filter(_filter_file(tmp_path, [{"path": "cross_validation.n_folds", "exists": False}]),
                           project="P", env="dev", projects_root=tmp_path)
    assert out["models"] == ["cnn_a"]


def test_planner_keeps_the_registered_name_when_the_map_lacks_the_hash(tmp_path, monkeypatch):
    from utils import sweep_engine

    monkeypatch.setattr(sweep_engine, "make_experiment_hash", lambda cfg: "abcd1234")
    _write_registries(tmp_path, {"abcd1234": _entry("mesh_chicken", "FAILED")})   # no model_name_map.json

    assert sweep_engine.determine_run_identity({"project": {"root": str(tmp_path)}}) == ("mesh_chicken", "abcd1234")
    assert _read(tmp_path, "model_name_map.json") == {"abcd1234": "mesh_chicken"}


@pytest.mark.parametrize("file, text, message", [
    ("model_name_map.json", '["abcd1234"]', "must hold a JSON object"),
    ("run_registry.json", "[]", "not a JSON object keyed by model_hash"),
])
def test_delete_database_refuses_a_damaged_registry_file(tmp_path, file, text, message):
    from services.side_channel_core.utils.delete_database import main

    root = tmp_path / "P"
    (root / "config").mkdir(parents=True)
    _write_registries(root, {"abcd1234": _entry("m-a", "DONE")}, tests={"abcd1234": _entry("m-a", "DONE")})
    (root / "artifacts" / file).write_text(text)
    snapshot = {p.name: p.read_text() for p in (root / "artifacts").iterdir()}

    with pytest.raises((ValueError, SystemExit), match=message):
        main(["--project-root", str(root), "--env", "dev", "--yes"])
    assert {p.name: p.read_text() for p in (root / "artifacts").iterdir()} == snapshot


# ------------------------------------------------------ FORCE=1 queue (2026-09-30)
@pytest.fixture
def force_env(main_env, monkeypatch, orch):
    """Two registered runs of the scope plus one of another MODEL_ID, session S."""
    root, calls, run = main_env
    monkeypatch.setenv("ORCHESTRATOR_SESSION_ID", "S")
    _write_registries(
        root,
        {
            "garl": _entry("garlic_chart", "INTERRUPTED", trial_index=1, started_at="2026-09-30T13:27:03+08:00",
                           run_session_id="OLD2", status_reason="signal 2 (SIGINT): ..."),
            "tuli": _entry("tulip_fish", "DONE", trial_index=1, started_at="2026-09-30T11:36:45+08:00",
                           run_session_id="OLD1"),
            "othr": _entry("other_model", "DONE", model_id="CNN1", trial_index=1),
        },
        tests={"garl": _entry("garlic_chart", "FAILED"), "tuli": _entry("tulip_fish", "DONE")},
    )
    planned = {"garl", "tuli"}                       # the hashes the current sweep plans (sweep_engine.planned_hashes)
    monkeypatch.setattr(orch, "planned_hashes", lambda cfg, project_root: set(planned))
    return root, calls, run


def _statuses(root, name):
    return {h: (e["status"], e.get("trial_index")) for h, e in _read(root, name).items()}


def test_force_queues_every_registered_run_before_training_the_first(force_env, monkeypatch, orch):
    root, calls, run = force_env
    seen = []
    monkeypatch.setattr(orch, "train_and_test", lambda run_cfg, env, **kw: seen.append(
        (run_cfg["model_hash"], _statuses(root, "run_registry.json"), _statuses(root, "test_registry.json"))))

    with pytest.raises(SystemExit) as exc:
        run(force=True)

    assert exc.value.code == 0
    assert calls["plan"] == []                       # no new hash while registered runs wait
    trained, runs, tests = seen[0]
    assert trained == "tuli"                         # tie on trial_index 1: earlier started_at first
    assert runs["tuli"] == ("in_progress", 1) and runs["garl"] == ("queued", 2)
    assert tests["tuli"][0] == "queued" and tests["garl"][0] == "queued"
    assert runs["othr"] == ("DONE", 1)               # another MODEL_ID is not ours
    garl = _read(root, "run_registry.json")["garl"]
    assert garl["run_session_id"] == "S" and "status_reason" not in garl


def test_force_trains_the_queue_in_order_then_plans_new_hashes(force_env, orch):
    root, calls, run = force_env
    for expected in (["tuli"], ["tuli", "garl"]):
        with pytest.raises(SystemExit) as exc:
            run(force=True)
        assert exc.value.code == 0
        assert [c["model_hash"] for c in calls["train_and_test"]] == expected
        assert calls["plan"] == []

    runs = _read(root, "run_registry.json")
    assert (runs["tuli"]["status"], runs["tuli"]["trial_index"]) == ("DONE", 1)
    assert (runs["garl"]["status"], runs["garl"]["trial_index"]) == ("DONE", 2)
    assert "force_previous_status" not in runs["tuli"]
    assert {e["status"] for e in _read(root, "test_registry.json").values()} == {"DONE"}

    with pytest.raises(SystemExit) as exc:               # queue empty: the planner takes over
        run(force=True)
    assert exc.value.code == orch.EXIT_NOTHING_TO_RUN
    assert calls["plan"] and len(calls["train_and_test"]) == 2


def test_signal_during_force_interrupts_the_started_run_and_gives_the_waiting_ones_their_status_back(force_env, monkeypatch, orch):
    root, calls, run = force_env
    monkeypatch.setattr(orch, "interrupt_run_map_entry", lambda **kw: None)
    monkeypatch.setattr(orch, "train_and_test", lambda run_cfg, env, **kw: orch.handle_signal(signal.SIGINT, None))

    with pytest.raises(SystemExit) as exc:
        run(force=True)

    assert exc.value.code == 130
    runs, tests = _read(root, "run_registry.json"), _read(root, "test_registry.json")
    assert runs["tuli"]["status"] == "INTERRUPTED" and "in_progress" in runs["tuli"]["status_reason"]
    # garlic_chart waited in the queue, never started: INTERRUPTED as before FORCE, its test FAILED as before
    assert runs["garl"]["status"] == "INTERRUPTED" and "before it started" in runs["garl"]["status_reason"]
    assert tests["garl"]["status"] == "FAILED" and "force_previous_status" not in runs["garl"]
    assert runs["othr"]["status"] == "DONE"


def test_signal_during_force_leaves_a_waiting_done_run_done_with_its_own_trial_numbers(force_env, monkeypatch, orch):
    """fig20's FORCE restart, stopped on 2026-10-20, left four DONE runs it never started INTERRUPTED, their tests FAILED."""
    root, calls, run = force_env
    registry = _read(root, "run_registry.json")
    registry["garl"].update(status="DONE", status_reason=None)
    registry["garl"].pop("status_reason")
    _write_registries(root, registry, tests={"garl": _entry("garlic_chart", "DONE"), "tuli": _entry("tulip_fish", "DONE")})
    snapshot = root / "artifacts/dev/generated_models/garlic_chart/config/garlic_chart_params.yaml"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("x: 1\n")
    monkeypatch.setattr(orch, "read_params_snapshot", lambda path: {"runtime": {
        "trial_index": 4, "total_trials": 5, "total_remaining_trials": 1, "run_session_id": "OLD2"}})
    monkeypatch.setattr(orch, "interrupt_run_map_entry", lambda **kw: None)
    monkeypatch.setattr(orch, "train_and_test", lambda run_cfg, env, **kw: orch.handle_signal(signal.SIGINT, None))

    with pytest.raises(SystemExit):
        run(force=True)

    runs, tests = _read(root, "run_registry.json"), _read(root, "test_registry.json")
    garl = runs["garl"]
    assert (garl["status"], tests["garl"]["status"]) == ("DONE", "DONE")
    assert "status_reason" not in garl and "force_previous_status" not in garl
    assert (garl["trial_index"], garl["total_trials"], garl["total_remaining_trials"], garl["run_session_id"]) == (4, 5, 1, "OLD2")


def test_force_retrains_only_what_the_sweep_plans_and_keeps_another_batchs_done_run(force_env, monkeypatch, orch):
    """fig20's FORCE (BATCH=fig20) retrained five runs of an earlier fig20 sweep and fig24's haberdashery_flock."""
    root, calls, run = force_env
    registry = _read(root, "run_registry.json")
    registry["stal"] = _entry("stale_run", "DONE", trial_index=3)                      # planned under earlier settings
    registry["shar"] = _entry("shared_run", "DONE", trial_index=2, batches=["fig24"])  # another figure's curve
    _write_registries(root, registry, tests=_read(root, "test_registry.json"))
    monkeypatch.setattr(orch, "planned_hashes", lambda cfg, project_root: {"garl", "tuli", "shar"})
    monkeypatch.setenv("BATCH", "fig20")
    monkeypatch.setenv("BATCH_ID", "fig20-20261020T085403Z")
    seen = []
    monkeypatch.setattr(orch, "train_and_test", lambda run_cfg, env, **kw: seen.append(
        (run_cfg["model_hash"], _statuses(root, "run_registry.json"))))

    with pytest.raises(SystemExit):
        run(force=True)

    trained, runs = seen[0]
    assert trained == "tuli" and runs["garl"][0] == "queued"
    assert runs["stal"] == ("DONE", 3) and runs["shar"] == ("DONE", 2)               # neither queued


def test_force_keeps_a_done_run_that_cannot_be_rerun(force_env, monkeypatch, orch):
    root, calls, run = force_env

    def _unrecoverable(**kw):
        raise ValueError("snapshot written before 2026-09-29")

    monkeypatch.setattr(orch, "resolve_registered_run_cfg", _unrecoverable)
    with pytest.raises(SystemExit):
        run(force=True)

    runs, tests = _read(root, "run_registry.json"), _read(root, "test_registry.json")
    assert runs["tuli"]["status"] == "DONE" and tests["tuli"]["status"] == "DONE"
    assert "retry_exhausted" not in runs["tuli"]


def test_force_queue_numbers_remaining_trials_with_the_trial_index(force_env, monkeypatch, orch):
    root, calls, run = force_env
    monkeypatch.setattr(orch, "_derive_attempt_window", lambda *a, **k: (1, 12, 12))
    seen = []
    monkeypatch.setattr(orch, "train_and_test", lambda run_cfg, env, **kw: seen.append(
        {h: (e["trial_index"], e["total_remaining_trials"], e["total_trials"])
         for h, e in _read(root, "run_registry.json").items() if h != "othr"}))

    with pytest.raises(SystemExit):
        run(force=True)

    # trial i of 12 leaves 12 - i; the waiting run is renumbered too, not only its trial_index
    assert seen[0] == {"tuli": (1, 11, 12), "garl": (2, 10, 12)}


# ------------------------------- an exhausted FAILED hash must not hide INTERRUPTED
def test_retry_and_resume_together_resume_an_interrupted_hash_beside_an_exhausted_failure(main_env, orch):
    root, calls, run = main_env
    _write_registries(root, {
        "aaaa": _entry("m-a", "FAILED", failed_attempts=orch.MAX_FAILED_ATTEMPTS, retry_exhausted=True),
        "bbbb": _entry("m-b", "INTERRUPTED"),
    })

    with pytest.raises(SystemExit) as exc:
        run(retry_failed=True, resume_interrupted=True)

    assert exc.value.code == 0
    assert [c["model_hash"] for c in calls["train_and_test"]] == ["bbbb"]


def _run_map_flags_heredoc() -> str:
    text = _SCRIPT.read_text()
    body = re.search(r"_py_run_map_flags\(\) \{\n\s*\S+ - <<'PY'\n(.*?)\nPY\n", text, re.S)
    assert body, "orchestrator.sh: _py_run_map_flags heredoc not found"
    return body.group(1)


@pytest.mark.parametrize("statuses, expected", [
    (["FAILED", "INTERRUPTED", "DONE"], "--retry-failed --resume-interrupted"),
    (["FAILED", "DONE"], "--retry-failed"),
    (["INTERRUPTED"], "--resume-interrupted"),
    (["DONE"], ""),
])
def test_run_map_recovery_passes_every_flag_it_needs(tmp_path, statuses, expected):
    records = [{"model_hash": f"h{i}", "model_name": f"m{i}", "status": st, "env": "dev", "model_id": "MLP1"}
               for i, st in enumerate(statuses)]
    run_map = tmp_path / "run_map.jsonl"
    run_map.write_text("".join(json.dumps(r) + "\n" for r in records))
    out = subprocess.run(
        [sys.executable, "-c", _run_map_flags_heredoc()],
        env={**os.environ, "RUN_MAP": str(run_map), "ENV": "dev", "MODEL_ID": "MLP1"},
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == expected


@pytest.mark.parametrize("entry_env, logged", [("dev", True), ("prod", False)])
def test_empty_plan_names_the_existing_experiment_and_points_to_force(tmp_path, monkeypatch, caplog, entry_env, logged):
    from utils import sweep_engine

    base = {"trainer": {"optimizer": {"learning_rate": 0.001}}, "paths": {"artifacts_root": str(tmp_path / "run")}}
    same = lambda cfg, *a, **kw: cfg
    for name, fake in {
        "sync_run_map_before_planning": lambda root: None,
        "get_project_config": lambda name: dict(base),
        "generate_param_combinations": lambda cfg: iter([dict(base)]),
        "count_all_param_combinations": lambda cfg: 1,
        "coerce_types": same,
        "deep_template_resolve": same,
        "resolve_model_contract": lambda cfg, **kw: cfg,
        "assert_concrete_sca_runtime": lambda sca: None,
        "_assert_trainer_preserved": lambda before, after: None,
        "_scc_sampled_payload": lambda cfg: {},
        "sample_key": lambda payload: "k",
        "determine_run_identity": lambda cfg, **kw: ("mesh_chicken", "abcd1234"),
        "_canonical_env": lambda cfg, run_dir: "dev",
    }.items():
        monkeypatch.setattr(sweep_engine, name, fake)
    monkeypatch.delenv("MODELS", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("SCC_SINGLE_RUN", raising=False)
    _write_registries(tmp_path, {"abcd1234": _entry("mesh_chicken", "DONE", env=entry_env)})

    cfg = {"project": {"name": "P", "root": str(tmp_path)}, "data": {"name": "ascad"}, "env": "dev", "model_id": "MLP1"}
    with caplog.at_level(logging.INFO):
        planned = list(sweep_engine.get_pending_runs(cfg, tmp_path, retry_failed=True, resume_interrupted=True))

    assert planned == []
    message = ("the requested experiment already exists model_hash=abcd1234; model_name=mesh_chicken. "
               "run scope with FORCE=1 to regenerate the fresh results.")
    assert (message in caplog.messages) is logged


def test_runs_selected_in_a_batch_join_it(main_env, monkeypatch):
    root, calls, run = main_env
    monkeypatch.setenv("BATCH", "fig6")
    monkeypatch.setenv("BATCH_ID", "fig6-20261006T083000Z")
    _write_registries(root, {"bbbb": _entry("m-b", "INTERRUPTED", batches=["fig5"], batch_ids=["fig5-1"])})

    with pytest.raises(SystemExit) as exc:
        run(resume_interrupted=True)

    assert exc.value.code == 0 and [c["model_hash"] for c in calls["train_and_test"]] == ["bbbb"]
    entry = _read(root, "run_registry.json")["bbbb"]
    assert entry["batches"] == ["fig5", "fig6"] and entry["batch_ids"] == ["fig5-1", "fig6-20261006T083000Z"]


def test_a_planned_run_is_numbered_by_its_place_in_the_sweep_grid(tmp_path, monkeypatch):
    """Trial k of N is the run's place among the N combinations, whatever this pass skipped as DONE."""
    from utils import sweep_engine
    from utils.run_map import load_run_map

    combos = [{"trainer": {"optimizer": {"learning_rate": lr}}, "paths": {"artifacts_root": str(tmp_path / f"run{lr}")}}
              for lr in (1, 2, 3)]
    names = {1: ("one_run", "aaaa0001"), 2: ("two_run", "aaaa0002"), 3: ("three_run", "aaaa0003")}
    same = lambda cfg, *a, **kw: cfg
    for name, fake in {
        "sync_run_map_before_planning": lambda root: None,
        "get_project_config": lambda name: dict(combos[0]),
        "generate_param_combinations": lambda cfg: iter([dict(c) for c in combos]),
        "count_all_param_combinations": lambda cfg: 3,
        "coerce_types": same,
        "deep_template_resolve": same,
        "resolve_model_contract": lambda cfg, **kw: cfg,
        "assert_concrete_sca_runtime": lambda sca: None,
        "_assert_trainer_preserved": lambda before, after: None,
        "_scc_sampled_payload": lambda cfg: {},
        "sample_key": lambda payload: "k",
        "determine_run_identity": lambda cfg, **kw: names[cfg["trainer"]["optimizer"]["learning_rate"]],
        "_canonical_env": lambda cfg, run_dir: "dev",
    }.items():
        monkeypatch.setattr(sweep_engine, name, fake)
    for var in ("MODELS", "MODEL_NAME", "SCC_SINGLE_RUN", "BATCH", "BATCH_ID"):
        monkeypatch.delenv(var, raising=False)
    _write_registries(tmp_path, {"aaaa0001": _entry("one_run", "DONE")})

    cfg = {"project": {"name": "P", "root": str(tmp_path)}, "data": {"name": "ascad"}, "env": "dev", "model_id": "MLP1"}
    first = next(iter(sweep_engine.get_pending_runs(cfg, tmp_path)))
    model_hash, run_cfg = first[0], first[1]
    assert model_hash == "aaaa0002"                                     # the DONE first combination is skipped
    runtime = run_cfg["runtime"]
    assert (runtime["trial_index"], runtime["sweep_position"], runtime["trial_total"]) == (2, 2, 3)
    assert load_run_map(tmp_path)["aaaa0002"]["trial_index"] == 2


def test_the_orchestrator_numbers_a_planned_run_by_its_sweep_position(main_env, monkeypatch, orch):
    """The session counter (_derive_attempt_window: 1 here) must not override the grid position."""
    root, calls, run = main_env
    _write_registries(root, {"cccc": _entry("m-c", "in_progress")})
    run_cfg = {"model_hash": "cccc", "model_name": "m-c", "env": "dev",
               "runtime": {"sweep_position": 7, "trial_index": 7, "trial_total": 14},
               "paths": {"artifacts_root": "artifacts/dev/generated_models/m-c"}}
    planned = [("cccc", {"model_name": "m-c", "model_id": "MLP1", "env": "dev", "run_cfg": run_cfg,
                         "run_cfg_source": "planner"})]
    monkeypatch.setattr(orch, "_plan_scc_models_from_sweep_engine", lambda **kw: planned if not calls["plan"].append(kw) else [])

    with pytest.raises(SystemExit):
        run()

    assert calls["train_and_test"][0]["runtime"]["trial_index"] == 7
    entry = _read(root, "run_registry.json")["cccc"]
    assert (entry["trial_index"], entry["total_trials"], entry["total_remaining_trials"]) == (7, 14, 7)
