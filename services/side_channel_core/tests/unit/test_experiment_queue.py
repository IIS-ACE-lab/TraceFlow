# services/side_channel_core/tests/unit/test_experiment_queue.py
import subprocess
from types import SimpleNamespace

import pytest

from services.experiment_queue import experiment_queue as eq
from services.experiment_queue.experiment_queue import (
    ExperimentQueueError, batch_experiments, build_argv, load_queue, planned_experiments, run_queue,
)

A = "make orchestrate ENV=dev PROJECT=P DATASET=ascad MODEL_ID=A"
B = "make orchestrate ENV=dev PROJECT=P DATASET=ascad MODEL_ID=B"
C = "make orchestrate ENV=dev PROJECT=P DATASET=ascad MODEL_ID=C"


def _queue_file(tmp_path, text):
    path = tmp_path / "experiment_queue.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def calls(monkeypatch, tmp_path):
    """Record the commands the queue executes; the exit codes come from `codes`. The project's
    root (its config folder, frozen at queue start) is `root`; `during(index)` runs inside command index."""
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "config" / "params.yaml").write_text("seed: 1\n", encoding="utf-8")
    seen = SimpleNamespace(argv=[], env=[], codes={}, root=root, during=lambda index: None)

    def fake_run(argv, cwd=None, env=None, check=False):
        seen.argv.append(argv)
        seen.env.append(env)
        seen.during(len(seen.argv))
        return subprocess.CompletedProcess(argv, seen.codes.get(len(seen.argv), 0))

    monkeypatch.setattr(eq.subprocess, "run", fake_run)
    monkeypatch.setattr(eq, "project_root_of", lambda project: root)
    return seen


def test_queue_runs_its_commands_in_the_listed_order(tmp_path, calls):
    config = _queue_file(tmp_path, f"fig6:\n  - {B}\n  - {A}\nfig7:\n  - {C}\n")

    assert run_queue(config, project="P", queue="fig6", env="dev") == 0
    assert [a[-3] for a in calls.argv] == ["MODEL_ID=B", "MODEL_ID=A"]


def test_force_adds_force_to_every_command_and_resume_adds_nothing(tmp_path, calls):
    config = _queue_file(tmp_path, f"fig6:\n  - {A}\n  - {B}\n")

    run_queue(config, project="P", queue="fig6", env="dev", force=True)
    run_queue(config, project="P", queue="fig6", env="dev")

    assert [a[-3] for a in calls.argv] == ["FORCE=1", "FORCE=1", "MODEL_ID=A", "MODEL_ID=B"]


def test_queue_stops_at_the_first_failed_command(tmp_path, calls):
    config = _queue_file(tmp_path, f"fig6:\n  - {A}\n  - {B}\n  - {C}\n")
    calls.codes[2] = 3

    assert run_queue(config, project="P", queue="fig6", env="dev") == 3
    assert len(calls.argv) == 2


def test_commands_do_not_inherit_the_variables_of_make_experiment(tmp_path, calls, monkeypatch):
    monkeypatch.setenv("MAKEFLAGS", " -- QUEUE=fig6 FORCE=1")
    config = _queue_file(tmp_path, f"fig6:\n  - {A}\n")

    run_queue(config, project="P", queue="fig6", env="dev")

    assert "MAKEFLAGS" not in calls.env[0]


def test_run_selectors_of_make_experiment_never_reach_a_queued_line(tmp_path, calls, monkeypatch):
    """make exports MODELS, REPEAT, ...; a child make reads them with ?=, so they narrowed every line."""
    for name, value in {"MODELS": "x", "REPEAT": "3", "MODEL_ID": "Z", "ENV": "prod", "PROJECT": "Q"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CONTAINER_NAME", "sidechannel")                    # a machine setting stays
    config = _queue_file(tmp_path, "fig6:\n  - make orchestrate DATASET=ascad MODEL_ID=A\n")

    run_queue(config, project="P", queue="fig6", env="dev")

    env, argv = calls.env[0], calls.argv[0]
    assert not {"MODELS", "REPEAT", "MODEL_ID", "ENV", "PROJECT"} & set(env)
    assert env["CONTAINER_NAME"] == "sidechannel"
    assert argv[:6] == ["make", "orchestrate", "DATASET=ascad", "MODEL_ID=A", "ENV=dev", "PROJECT=P"]


@pytest.mark.parametrize("command, message", [
    ("rm -rf artifacts", "only make commands"),
    (A + " FORCE=1", "FORCE is set by make experiment"),
    (A.replace("PROJECT=P", "PROJECT=Q"), "PROJECT=Q, not P"),
    (A.replace("ENV=dev", "ENV=prod"), "ENV=prod, not dev"),
])
def test_a_command_outside_the_scope_is_refused_before_anything_runs(tmp_path, calls, command, message):
    config = _queue_file(tmp_path, f"fig6:\n  - {A}\n  - {command}\n")

    with pytest.raises(ExperimentQueueError, match=message):
        run_queue(config, project="P", queue="fig6", env="dev")
    assert calls.argv == []


@pytest.mark.parametrize("text, queue, message", [
    (f"fig6:\n  - {A}\n", "fig9", "available: fig6"),
    ("fig6: []\n", "fig6", "non-empty list"),
    ("- a\n", "fig6", "must be a mapping"),
])
def test_a_damaged_queue_file_is_reported(tmp_path, text, queue, message):
    with pytest.raises(ExperimentQueueError, match=message):
        load_queue(_queue_file(tmp_path, text), queue)


def test_missing_queue_file_is_reported(tmp_path):
    with pytest.raises(ExperimentQueueError, match="missing queue file"):
        load_queue(tmp_path / "experiment_queue.yaml", "fig6")


def test_the_project_queue_file_holds_only_runnable_commands():
    import yaml
    from pathlib import Path

    path = Path(__file__).resolve().parents[4] / "projects/DLSCA_ASCAD/config/experiment_queue.yaml"
    queues = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert "fig6" in queues
    for name in queues:
        for command in load_queue(path, name):
            build_argv(command, project="DLSCA_ASCAD", env="dev", force=False)


def test_every_command_of_one_execution_gets_the_same_batch(tmp_path, calls):
    config = _queue_file(tmp_path, f"fig6:\n  - {A}\n  - {B}\n")

    run_queue(config, project="P", queue="fig6", env="dev")
    run_queue(config, project="P", queue="fig6", env="dev")

    assert [a[-2] for a in calls.argv] == ["BATCH=fig6"] * 4
    ids = [a[-1] for a in calls.argv]
    assert all(i.startswith("BATCH_ID=fig6-") for i in ids)
    assert ids[0] == ids[1]                    # one id per execution of the queue


@pytest.mark.parametrize("owned", ["BATCH=x", "BATCH_ID=x-1"])
def test_a_queued_line_cannot_set_its_own_batch(owned):
    with pytest.raises(ExperimentQueueError, match="set by make experiment"):
        build_argv(f"{A} {owned}", project="P", env="dev", force=False)


def _registry(root, batch_id, statuses):
    """A run_registry.json whose entries belong to batch_id, one per status."""
    import json
    (root / "projects/P/artifacts").mkdir(parents=True)
    entries = {f"h{i}": {"model_name": f"m{i}", "status": st, "batch_ids": [batch_id]} for i, st in enumerate(statuses)}
    entries["other"] = {"model_name": "x", "status": "DONE", "batch_ids": ["fig9-20260101T000000Z"]}
    (root / "projects/P/artifacts/run_registry.json").write_text(json.dumps(entries), encoding="utf-8")


def test_experiments_of_a_batch_are_counted_by_status_from_the_registry(tmp_path):
    _registry(tmp_path, "fig3-20261006T061136Z", ["DONE", "DONE", "FAILED"])

    assert batch_experiments(tmp_path / "projects/P", "fig3-20261006T061136Z") == {"DONE": 2, "FAILED": 1}
    assert batch_experiments(tmp_path / "projects/P", "fig3-20270101T000000Z") == {}
    assert batch_experiments(tmp_path / "nowhere", "fig3-20261006T061136Z") == {}


def test_the_queue_reports_the_experiments_of_its_batch_not_only_its_commands(tmp_path, calls, caplog, monkeypatch):
    import logging
    config = _queue_file(tmp_path, f"fig3:\n  - {A}\n")
    monkeypatch.setattr(eq, "batch_id_for", lambda queue: "fig3-20261006T061136Z")
    _registry(tmp_path, "fig3-20261006T061136Z", ["DONE", "DONE", "DONE"])     # one command, three sweep points

    with caplog.at_level(logging.INFO):
        run_queue(config, project="P", queue="fig3", env="dev", cwd=tmp_path)

    assert any("3 experiment(s): 3 DONE" in m for m in caplog.messages)


def test_the_queue_numbers_the_planned_experiments_across_its_commands(tmp_path, calls, caplog, monkeypatch):
    import logging
    config = _queue_file(tmp_path, f"fig3:\n  - {A}\n  - {B}\n")
    monkeypatch.setattr(eq, "planned_experiments", lambda project: 12)

    with caplog.at_level(logging.INFO):
        run_queue(config, project="P", queue="fig3", env="dev", cwd=tmp_path)

    assert any("24 experiment(s) planned by params.yaml, 12 per command" in m for m in caplog.messages)
    assert any("command 1/2 (experiments 1-12 of 24)" in m for m in caplog.messages)
    assert any("command 2/2 (experiments 13-24 of 24)" in m for m in caplog.messages)


def test_planned_experiments_follow_the_sweep_lists_of_params_yaml(monkeypatch):
    import types, sys
    cfg = {"trainer": {"epochs": [50, 75, 100], "batch_size": [50, 100, 200, 500]}}
    loader = types.ModuleType("utils.config_loader"); loader.get_project_config = lambda name: cfg
    engine = types.ModuleType("utils.sweep_engine"); engine.count_all_param_combinations = lambda c: 12
    monkeypatch.setitem(sys.modules, "utils.config_loader", loader)
    monkeypatch.setitem(sys.modules, "utils.sweep_engine", engine)

    assert planned_experiments("P") == 12


def test_the_config_is_frozen_at_queue_start_and_every_command_reads_that_copy(tmp_path, calls, monkeypatch):
    monkeypatch.delenv(eq.ENV_DIR, raising=False)
    config = _queue_file(tmp_path, f"fig6:\n  - {A}\n  - {B}\n")
    live = calls.root / "config" / "params.yaml"
    frozen_seen = []

    def during(index):
        rel = calls.env[-1][eq.ENV_DIR]
        frozen_seen.append((calls.root / rel / "params.yaml").read_text(encoding="utf-8"))
        live.write_text("seed: 2\n", encoding="utf-8")         # an edit while the queue runs

    calls.during = during
    assert run_queue(config, project="P", queue="fig6", env="dev") == 0
    rels = {e[eq.ENV_DIR] for e in calls.env}
    assert len(rels) == 1 and next(iter(rels)).startswith("config/.frozen/fig6-")
    assert {e[eq.ENV_PROJECT] for e in calls.env} == {"P"}
    assert frozen_seen == ["seed: 1\n", "seed: 1\n"]           # the second command still reads the queue's start
    assert live.read_text(encoding="utf-8") == "seed: 2\n"       # the edit stays, for the next queue
    assert not (calls.root / next(iter(rels))).exists()            # the copy is removed when the queue ends
    assert eq.ENV_DIR not in __import__("os").environ


def test_a_failed_queue_removes_its_frozen_copy_too(tmp_path, calls):
    config = _queue_file(tmp_path, f"fig6:\n  - {A}\n  - {B}\n")
    calls.codes = {1: 2}
    assert run_queue(config, project="P", queue="fig6", env="dev") == 2
    assert list((calls.root / "config" / ".frozen").iterdir()) == []
