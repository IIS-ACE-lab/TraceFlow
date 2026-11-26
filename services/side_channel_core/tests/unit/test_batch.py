# services/side_channel_core/tests/unit/test_batch.py
"""Batches: a common label for the runs made together (utils/batch.py)."""

import json

import pytest
import yaml

from utils import batch as batch_mod
from utils.batch import add_batch, current_batch, label_models


def _write_registry(root, entries):
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "run_registry.json").write_text(json.dumps(entries))


def _registry(root):
    return json.loads((root / "artifacts" / "run_registry.json").read_text())


def _entry(name, status="DONE", env="dev", model_id="MLP1", **extra):
    return {"model_name": name, "status": status, "env": env, "model_id": model_id,
            "path": f"artifacts/{env}/generated_models/{name}", **extra}


def test_current_batch_reads_the_environment():
    assert current_batch({}) is None
    assert current_batch({"BATCH": "fig6", "BATCH_ID": "fig6-20261006T083000Z"}) == ("fig6", "fig6-20261006T083000Z")
    # by hand (make orchestrate BATCH=fig6): one id per orchestrator session
    assert current_batch({"BATCH": "fig6", "ORCHESTRATOR_SESSION_ID": "20261006T083000Z-1"}) == (
        "fig6", "fig6-20261006T083000Z-1")
    for bad in ({"BATCH": "fig 6"}, {"BATCH": "-fig6"}, {"BATCH_ID": "x-1"}):
        with pytest.raises(ValueError):
            current_batch(bad)


def test_add_batch_keeps_every_batch_once():
    entry = {}
    assert add_batch(entry, "fig5", "fig5-1") is True
    assert add_batch(entry, "fig6", "fig6-1") is True
    assert add_batch(entry, "fig6", "fig6-1") is False
    assert entry == {"batches": ["fig5", "fig6"], "batch_ids": ["fig5-1", "fig6-1"]}


@pytest.fixture
def planner(tmp_path, monkeypatch):
    """sweep_engine on a one-combination sweep planning mesh_chicken (abcd1234), DONE in batch fig5, under BATCH=fig6."""
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
    for var in ("MODELS", "MODEL_NAME", "SCC_SINGLE_RUN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("BATCH", "fig6")
    monkeypatch.setenv("BATCH_ID", "fig6-20261006T083000Z")
    _write_registry(tmp_path, {
        "abcd1234": _entry("mesh_chicken", batches=["fig5"], batch_ids=["fig5-20261001T080000Z"]),
        "ffff0000": _entry("other_scope", model_id="CNN9"),
    })

    cfg = {"project": {"name": "P", "root": str(tmp_path)}, "data": {"name": "ascad"}, "env": "dev", "model_id": "MLP1"}
    return sweep_engine, cfg


def test_a_skipped_done_run_in_scope_joins_the_batch(tmp_path, planner):
    sweep_engine, cfg = planner
    assert list(sweep_engine.get_pending_runs(cfg, tmp_path)) == []        # DONE: not trained again

    reg = _registry(tmp_path)
    assert reg["abcd1234"]["batches"] == ["fig5", "fig6"]
    assert reg["abcd1234"]["batch_ids"] == ["fig5-20261001T080000Z", "fig6-20261006T083000Z"]
    assert "batches" not in reg["ffff0000"]                                # not in this scope


def test_force_keeps_a_done_run_of_another_batch_unless_named(tmp_path, planner, monkeypatch):
    """fig20's FORCE (2026-10-20) retrained fig24's haberdashery_flock: another figure's finished curve."""
    sweep_engine, cfg = planner
    assert list(sweep_engine.get_pending_runs(cfg, tmp_path, force=True)) == []
    assert _registry(tmp_path)["abcd1234"]["status"] == "DONE"
    assert _registry(tmp_path)["abcd1234"]["batches"] == ["fig5", "fig6"]   # joins fig6 as a skipped DONE run

    monkeypatch.setenv("MODELS", "mesh_chicken")                            # named: retrained
    assert [h for h, *_ in sweep_engine.get_pending_runs(cfg, tmp_path, force=True)] == ["abcd1234"]


def test_planned_hashes_are_the_hashes_the_planner_makes_and_write_nothing(tmp_path, planner):
    sweep_engine, cfg = planner
    before = (tmp_path / "artifacts" / "run_registry.json").read_text()
    assert sweep_engine.planned_hashes(cfg, tmp_path) == {"abcd1234"}
    assert (tmp_path / "artifacts" / "run_registry.json").read_text() == before
    assert not (tmp_path / "run").exists()                                  # no run folder made
    assert not (tmp_path / "artifacts" / "model_name_map.json").exists()    # no phantom name either


def test_a_name_is_written_into_the_map_only_when_its_run_is_decided(tmp_path, planner, monkeypatch):
    """Planning names every combination it looks at; model_name_map.json gets the name of the run
    that starts, not of a skipped one (five phantom names, audit 2026-10-23)."""
    sweep_engine, cfg = planner
    name_map = tmp_path / "artifacts" / "model_name_map.json"
    _write_registry(tmp_path, {"abcd1234": _entry("mesh_chicken", status="DONE", batches=["fig6"])})
    assert list(sweep_engine.get_pending_runs(cfg, tmp_path)) == []        # DONE: skipped, not decided
    assert not name_map.exists()
    _write_registry(tmp_path, {})
    assert [h for h, *_ in sweep_engine.get_pending_runs(cfg, tmp_path)] == ["abcd1234"]
    assert {k: v for k, v in json.loads(name_map.read_text()).items() if k != "_tfseal"} == {"abcd1234": "mesh_chicken"}


def test_held_by_another_batch():
    assert batch_mod.held_by_another_batch({"batches": ["fig24", "fig20"]}, ("fig20", "fig20-1"))
    assert not batch_mod.held_by_another_batch({"batches": ["fig20"]}, ("fig20", "fig20-1"))
    assert batch_mod.held_by_another_batch({"batches": ["fig20"]}, None)     # outside any batch: every label is another
    assert not batch_mod.held_by_another_batch({}, None)


def test_find_model_selects_a_batch_from_the_registry(tmp_path):
    from services.model_filter.model_filter import run_model_filter

    root = tmp_path / "P"
    runs = {"h1": "ref", "h2": "conv0", "h3": "unrelated"}
    (root / "artifacts").mkdir(parents=True)
    (root / "artifacts" / "run_map.jsonl").write_text("".join(
        json.dumps({"model_hash": h, "model_name": n, "env": "dev", "status": "DONE", "model_id": "M",
                    "path": f"artifacts/dev/generated_models/{n}", "sampled": {}}) + "\n" for h, n in runs.items()))
    _write_registry(root, {
        "h1": _entry("ref", batches=["fig5", "fig6"]),
        "h2": _entry("conv0", batches=["fig6"]),
        "h3": _entry("unrelated"),
    })
    flt = tmp_path / "f.yaml"
    flt.write_text(yaml.safe_dump({"model_filter": {"where": [{"path": "batches", "contains": "fig6"}]}}))

    assert run_model_filter(flt, project="P", env="dev", projects_root=tmp_path)["models"] == ["ref", "conv0"]


def test_batch_label_labels_the_models_of_a_find_model_output(tmp_path, monkeypatch):
    _write_registry(tmp_path, {"h1": _entry("a"), "h2": _entry("b"), "h3": _entry("c")})
    out = tmp_path / "artifacts" / "filtered_models.yaml"
    out.write_text(yaml.safe_dump({"models": ["a", "b"], "details": [{"model_name": "a", "model_hash": "h1"}]}))
    monkeypatch.setattr(batch_mod, "utc_stamp", lambda: "20261006T090000Z")

    assert label_models(tmp_path, batch="fig6", env="dev") == [("h1", "a"), ("h2", "b")]
    reg = _registry(tmp_path)
    assert reg["h1"]["batches"] == ["fig6"] and reg["h1"]["batch_ids"] == ["fig6-label-20261006T090000Z"]
    assert reg["h2"]["batches"] == ["fig6"] and "batches" not in reg["h3"]

    out.write_text(yaml.safe_dump({"models": ["a", "nobody"]}))
    with pytest.raises(ValueError, match="nobody"):
        label_models(tmp_path, batch="fig7", env="dev")
    assert "fig7" not in _registry(tmp_path)["h1"]["batches"]             # nothing written


def _run_map(root, records):
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "run_map.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def _run_map_latest(root):
    out = {}
    for line in (root / "artifacts" / "run_map.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["model_hash"]] = rec
    return out


def test_stamp_labels_the_test_entry_and_the_run_map_record_too(tmp_path):
    from utils.batch import stamp_registry

    _write_registry(tmp_path, {"h1": _entry("a"), "h2": _entry("b"), "h3": _entry("c")})
    (tmp_path / "artifacts" / "test_registry.json").write_text(json.dumps({"h1": _entry("a"), "h2": _entry("b")}))
    _run_map(tmp_path, [{"model_hash": h, "model_name": n, "env": "dev", "status": "DONE", "model_id": "MLP1",
                         "path": f"artifacts/dev/generated_models/{n}", "sampled": {}}
                        for h, n in (("h1", "a"), ("h2", "b"))])

    assert stamp_registry(tmp_path, ["h1", "h2", "h3"], batch=("fig6", "fig6-1")) == ["h1", "h2", "h3"]

    tests = json.loads((tmp_path / "artifacts" / "test_registry.json").read_text())
    for h in ("h1", "h2"):                                     # every entry, not only the first
        assert tests[h]["batches"] == ["fig6"] and tests[h]["batch_ids"] == ["fig6-1"]
    assert "h3" not in tests                                   # no test entry is invented
    records = _run_map_latest(tmp_path)
    for h in ("h1", "h2"):
        assert records[h]["batches"] == ["fig6"] and records[h]["batch_ids"] == ["fig6-1"]


def test_entries_written_later_copy_the_batches_of_run_registry(tmp_path):
    from utils.run_map import upsert_run_map_entry
    from utils.test_registry import update_test_status

    _write_registry(tmp_path, {"h1": _entry("a", status="in_progress", batches=["fig5", "fig6"],
                                            batch_ids=["fig5-1", "fig6-1"])})

    update_test_status(tmp_path, model_id="h1", new_status="queued", env="dev")
    upsert_run_map_entry(tmp_path, model_hash="h1", model_name="a", model_id="MLP1", sampled={},
                         status="IN_PROGRESS", env="dev", run_dir="artifacts/dev/generated_models/a")

    tests = json.loads((tmp_path / "artifacts" / "test_registry.json").read_text())
    assert tests["h1"]["batches"] == ["fig5", "fig6"] and tests["h1"]["batch_ids"] == ["fig5-1", "fig6-1"]
    rec = _run_map_latest(tmp_path)["h1"]
    assert rec["batches"] == ["fig5", "fig6"] and rec["batch_ids"] == ["fig5-1", "fig6-1"]


@pytest.mark.parametrize("asked", ["fig7|fig6|pig7", ["fig7", "fig6", "pig7"]])
def test_contains_takes_alternatives_in_the_written_order(tmp_path, caplog, asked):
    import logging

    from services.model_filter.model_filter import run_model_filter

    root = tmp_path / "P"
    runs = {"h1": "ref", "h2": "conv0", "h3": "filt16", "h4": "other"}
    (root / "artifacts").mkdir(parents=True)
    (root / "artifacts" / "run_map.jsonl").write_text("".join(
        json.dumps({"model_hash": h, "model_name": n, "env": "dev", "status": "DONE", "model_id": "M",
                    "path": f"artifacts/dev/generated_models/{n}", "sampled": {}}) + "\n" for h, n in runs.items()))
    _write_registry(root, {
        "h1": _entry("ref", batches=["fig5", "fig6"]),
        "h2": _entry("conv0", batches=["fig6"]),
        "h3": _entry("filt16", batches=["fig7"]),
        "h4": _entry("other", batches=["fig9"]),
    })
    flt = tmp_path / "f.yaml"
    flt.write_text(yaml.safe_dump({"model_filter": {"where": [{"path": "batches", "contains": asked}]}}))

    with caplog.at_level(logging.WARNING, logger="ModelFilter"):
        out = run_model_filter(flt, project="P", env="dev", projects_root=tmp_path)

    assert out["models"] == ["filt16", "ref", "conv0"]          # fig7 first, as written
    assert any("pig7" in m and "matched no run" in m for m in caplog.messages)


def test_the_run_folder_is_made_under_the_project_root_only_for_the_run_that_starts(tmp_path, planner, monkeypatch):
    """paths.artifacts_root is project-relative: the planner resolves it against the project root, never
    the working directory (as root that wrote junk into the container's /app; as the user it failed with
    PermissionError, 2026-10-23), and makes the folder only when the run is decided, so a skipped DONE
    combination leaves no empty folder behind."""
    sweep_engine, cfg = planner
    base = sweep_engine.get_project_config("P")
    base["paths"] = {"artifacts_root": "artifacts/dev/generated_models/mesh_chicken"}      # as the blueprint writes it
    monkeypatch.setattr(sweep_engine, "get_project_config", lambda name: dict(base))
    monkeypatch.setattr(sweep_engine, "generate_param_combinations", lambda c: iter([dict(base)]))
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir(); monkeypatch.chdir(elsewhere)
    _write_registry(tmp_path, {"abcd1234": _entry("mesh_chicken", status="DONE", batches=["fig6"])})
    assert list(sweep_engine.get_pending_runs(cfg, tmp_path)) == []                        # skipped: no folder
    assert not (tmp_path / "artifacts" / "dev").exists() and not (elsewhere / "artifacts").exists()
    _write_registry(tmp_path, {})
    runs = list(sweep_engine.get_pending_runs(cfg, tmp_path))
    assert [h for h, *_ in runs] == ["abcd1234"]
    run_dir = runs[0][2]
    assert run_dir == tmp_path / "artifacts" / "dev" / "generated_models" / "mesh_chicken" and run_dir.is_absolute()
    assert (run_dir / "config").is_dir() and not (elsewhere / "artifacts").exists()
    assert _registry(tmp_path)["abcd1234"]["path"] == "artifacts/dev/generated_models/mesh_chicken"   # project-relative in the registry
