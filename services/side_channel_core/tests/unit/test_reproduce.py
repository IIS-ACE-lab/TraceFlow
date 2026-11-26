"""Reproducing runs from their params_out snapshots alone (utils/reproduce.py)."""

import json
from pathlib import Path

import pytest
import yaml

from utils import data_reproducibility as dr
from utils import reproduce
from utils.seal import strip as strip_seal  # registries carry their "_tfseal" (utils/seal.py)


def _cfg(name, h, batch=None):
    return {
        "project": {"name": "P", "root": "."},
        "data": {"name": "ascad"},
        "model": [{"model_id": "ASCAD_MLPbest"}],
        "model_id": "ASCAD_MLPbest",
        "trainer": {"epochs": 20, "batch_size": 100},
        "seed": 42,
        "sca": {"leakage_model": "HW"},
        "paths": {"params_out": reproduce.DEFAULT_PARAMS_OUT},
        "env": "dev",
        "model_hash": h,
        "model_name": name,
    }


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A receiver's project: config/ and the shipped snapshots only."""
    root = tmp_path / "P"
    (root / "config").mkdir(parents=True)
    (root / "config" / "params.yaml").write_text(yaml.safe_dump({"paths": {"params_out": reproduce.DEFAULT_PARAMS_OUT}}))
    monkeypatch.setattr(reproduce, "_live_project_block", lambda r, name=None: {"name": "P", "root": str(r)})
    # make_experiment_hash of a snapshot: its recorded hash unless the test marks it edited
    import utils.model_naming as mn
    monkeypatch.setattr(mn, "make_experiment_hash", lambda cfg: cfg.get("trainer", {}).get("tampered") or cfg["model_hash"])
    monkeypatch.delenv("BATCH", raising=False)
    monkeypatch.delenv("BATCH_ID", raising=False)
    return root


def ship(root, name, h, batch="fig26", monkeypatch=None):
    if monkeypatch is not None and batch:
        monkeypatch.setenv("BATCH", batch)
        monkeypatch.setenv("BATCH_ID", f"{batch}-20261012T000000Z")
    path = reproduce.snapshot_path(root, "dev", name)
    dr.write_params_snapshot(_cfg(name, h), path)
    if monkeypatch is not None:
        monkeypatch.delenv("BATCH", raising=False)
        monkeypatch.delenv("BATCH_ID", raising=False)
    return path


def test_snapshot_records_its_batch_outside_the_seal(project, monkeypatch):
    path = ship(project, "cake_bacon", "816fd431", "fig26", monkeypatch)
    snap = dr.read_params_snapshot(path)
    assert snap["batches"] == ["fig26"]
    assert list(snap)[:2] == ["snapshot_sha256", "batches"]
    # a later label keeps the seal (batches are outside the digest)
    assert dr.add_snapshot_batch(path, "fig14") is True
    assert dr.read_params_snapshot(path)["batches"] == ["fig26", "fig14"]
    assert dr.add_snapshot_batch(path, "fig14") is False
    # anything else edited still breaks it
    text = path.read_text().replace("batch_size: 100", "batch_size: 200")
    path.write_text(text)
    with pytest.raises(ValueError, match="fails its seal"):           # its seal (utils/seal.py)
        dr.read_params_snapshot(path)
    path.write_text(text.split("\n", 1)[1])                           # and, without one, its digest
    with pytest.raises(ValueError, match="does not match its snapshot_sha256"):
        dr.read_params_snapshot(path)


def test_register_enters_the_batch_under_its_own_name_and_hash(project, monkeypatch):
    ship(project, "cake_bacon", "816fd431", "fig26", monkeypatch)
    ship(project, "celery_plush", "7b8a271d", "fig26", monkeypatch)
    ship(project, "other_run", "aaaaaaaa", "fig6", monkeypatch)

    rows = reproduce.discover(project, "dev")
    assert {r["model_name"]: (r["verified"], r["registered"]) for r in rows} == {
        "cake_bacon": (True, False), "celery_plush": (True, False), "other_run": (True, False)}

    names = reproduce.register(project, env="dev", batch="fig26")
    assert sorted(names) == ["cake_bacon", "celery_plush"]
    registry = strip_seal(json.loads((project / "artifacts" / "run_registry.json").read_text()))
    entry = registry["816fd431"]
    assert entry["status"] == "NEW" and entry["imported_snapshot"] is True and entry["model_name"] == "cake_bacon"
    assert "fig26" in entry["batches"] and entry["env"] == "dev" and entry["model_id"] == "ASCAD_MLPbest"
    assert "aaaaaaaa" not in registry
    assert json.loads((project / "artifacts" / "model_name_map.json").read_text())["7b8a271d"] == "celery_plush"

    # again (e.g. after an interruption): the unfinished ones come back, nothing duplicated; DONE ones are left
    registry["7b8a271d"]["status"] = "DONE"
    (project / "artifacts" / "run_registry.json").write_text(json.dumps(registry))
    assert reproduce.register(project, env="dev", batch="fig26") == ["cake_bacon"]


def test_register_refuses_an_edited_snapshot_and_a_taken_name(project, monkeypatch):
    path = ship(project, "cake_bacon", "816fd431", "fig26", monkeypatch)
    snap = yaml.safe_load(path.read_text())
    snap["trainer"]["tampered"] = "deadbeef"                      # hashes to another identity
    snap["snapshot_sha256"] = dr.snapshot_digest(snap)             # resealed, but the hash gives it away
    path.write_text(yaml.safe_dump(snap, sort_keys=False))
    with pytest.raises(ValueError, match="hashes to deadbeef here"):
        reproduce.register(project, env="dev", batch="fig26")
    assert not (project / "artifacts" / "run_registry.json").exists()

    ship(project, "cake_bacon", "816fd431", "fig26", monkeypatch)  # restored
    (project / "artifacts" / "model_name_map.json").write_text(json.dumps({"12345678": "cake_bacon"}))
    with pytest.raises(ValueError, match="already belongs to hash 12345678"):
        reproduce.register(project, env="dev", batch="fig26")


def test_register_needs_a_scope_and_matches_names(project, monkeypatch):
    ship(project, "cake_bacon", "816fd431", "", monkeypatch)       # no batch label (written before labels)
    with pytest.raises(ValueError, match="give BATCH"):
        reproduce.register(project, env="dev")
    with pytest.raises(ValueError, match="no snapshot of env=dev matches"):
        reproduce.register(project, env="dev", batch="fig26")
    assert reproduce.register(project, env="dev", models=["816fd431"]) == ["cake_bacon"]


def test_an_imported_entry_is_trained_from_its_snapshot(monkeypatch):
    from utils import cli_helpers

    loaded = {}
    monkeypatch.setattr(cli_helpers, "load_run_snapshot",
                        lambda cfg, info, model_name: loaded.setdefault("cfg", {"model_hash": "816fd431", "snapshot_sha256": "x"}))
    monkeypatch.setattr(cli_helpers, "verify_snapshot_identity", lambda run_cfg, info, model_hash, model_name: run_cfg)
    prior = {"model_name": "cake_bacon", "status": "NEW", "imported_snapshot": True}
    run_cfg, source = cli_helpers.resolve_registered_run_cfg(
        cfg={"project": {"root": "."}}, project_root=".", model_hash="816fd431",
        info=dict(prior), registry_before={"816fd431": prior})
    assert source == "snapshot" and run_cfg is loaded["cfg"]


def test_reproduce_waits_for_the_current_run(tmp_path):
    """One experiment at a time: a running make/trainer or a live heartbeat stops make reproduce."""
    hb = tmp_path / "projects" / "P" / "artifacts" / "dev" / "heartbeat.json"
    hb.parent.mkdir(parents=True)
    now = 1_800_000_000 * 10**9
    procs = [(10, "make experiment ENV=dev PROJECT=P QUEUE=fig26"), (11, "make reproduce ENV=dev PROJECT=P BATCH=fig26"),
             (12, "/venv/bin/python -m core.cli train-cmd --project P --env dev -n a"), (13, "vim params.yaml")]
    # our own make reproduce (pid 11) is not "another run"
    found = reproduce.running_experiments(tmp_path, processes=procs, own={11}, now_ns=now)
    assert found == ["make experiment ENV=dev PROJECT=P QUEUE=fig26", "/venv/bin/python -m core.cli train-cmd --project P --env dev -n a"]
    hb.write_text(json.dumps({"status": "running", "written_ns": now - 5 * 10**9, "model_name": "cake_bacon", "phase": "fold"}))
    assert reproduce.running_experiments(tmp_path, processes=[], own=set(), now_ns=now) == [
        "P ENV=dev: cake_bacon training (fold, last beat 5 s ago)"]
    hb.write_text(json.dumps({"status": "running", "written_ns": now - 600 * 10**9, "model_name": "cake_bacon"}))
    assert reproduce.running_experiments(tmp_path, processes=[], own=set(), now_ns=now) == []      # a dead run's leftover
    hb.write_text(json.dumps({"status": "done", "written_ns": now, "model_name": "cake_bacon"}))
    assert reproduce.running_experiments(tmp_path, processes=[], own=set(), now_ns=now) == []


def test_make_reproduce_refuses_while_a_run_trains(tmp_path, monkeypatch, capsys):
    (tmp_path / "projects" / "P" / "config").mkdir(parents=True)
    monkeypatch.setattr(reproduce, "running_experiments", lambda root, **kw: ["P ENV=dev: cake_bacon training (fold, last beat 2 s ago)"])
    with pytest.raises(SystemExit, match="Let the current run finish"):
        reproduce.main(["register", "--project-root", str(tmp_path / "projects" / "P"), "--env", "dev", "--batch", "fig26"])
    assert not (tmp_path / "projects" / "P" / "artifacts").exists()                                  # nothing registered


def test_the_scan_pattern_is_the_project_root_joined_with_params_out(project):
    assert reproduce.scan_pattern(project, project.parent) == \
        "P/artifacts/<ENV>/generated_models/<model_name>/config/<model_name>_params.yaml"


def test_custom_folder_is_the_root_of_the_reproduced_runs(project, tmp_path, monkeypatch):
    """CUSTOM: the chosen folder is the root (every paths: template inside it); the project gives configs."""
    import shutil

    monkeypatch.setattr(reproduce, "resolve_project", lambda name: project if name == "P" else (_ for _ in ()).throw(KeyError(name)))
    src = ship(project, "cake_bacon", "816fd431", "fig26", monkeypatch)
    pikachu = tmp_path / "1" / "2" / "3" / "pikachu"
    (pikachu / "shipped").mkdir(parents=True)
    custom = pikachu / "shipped" / "cake_bacon_params.yaml"
    shutil.copy2(src, custom)
    shutil.rmtree(project / "artifacts")                          # the project itself has nothing

    rows = reproduce.discover_folder(pikachu)
    assert len(rows) == 1
    r = rows[0]
    assert r["problem"] == "", r["problem"]
    assert (r["model_name"], r["env"], r["project"], r["origin"], r["verified"], r["placed"], r["registered"]) == \
        ("cake_bacon", "dev", "P", "pikachu", True, False, False)
    target = pikachu / "artifacts/dev/generated_models/cake_bacon/config/cake_bacon_params.yaml"
    assert r["target"] == str(target)

    assert reproduce.place(custom, pikachu)["placed"] is True
    assert target.read_bytes() == custom.read_bytes()
    assert not (project / "artifacts").exists()                   # nothing goes to the project
    # registered in the folder's own registry, under its own name and hash; configs from the project
    assert reproduce.register(pikachu, env="dev", models=["cake_bacon"], project_name="P", config_root=project) == ["cake_bacon"]
    registry = json.loads((pikachu / "artifacts" / "run_registry.json").read_text())
    assert registry["816fd431"]["model_name"] == "cake_bacon" and registry["816fd431"]["imported_snapshot"] is True
    row = [x for x in reproduce.discover_folder(pikachu) if x["path"] == str(target)][0]
    assert row["registered"] and row["status"] == "NEW" and row["placed"]

    # a different snapshot already there is never overwritten
    other = pikachu / "other" / "cake_bacon_params.yaml"
    other.parent.mkdir()
    other.write_text(custom.read_text().replace("batch_size: 100", "batch_size: 200"))
    with pytest.raises(ValueError, match="refused"):
        reproduce.place(other, pikachu)

    # a file named after another model, and a snapshot of an unknown project
    bad = tmp_path / "bad"
    bad.mkdir()
    shutil.copy2(custom, bad / "wrong_params.yaml")
    snap = yaml.safe_load(custom.read_text())
    snap["project"]["name"] = "ELSEWHERE"
    snap["snapshot_sha256"] = dr.snapshot_digest(snap)
    (bad / "cake_bacon_params.yaml").write_text(yaml.safe_dump(snap, sort_keys=False))
    problems = {Path(x["path"]).name: x["problem"] for x in reproduce.discover_folder(bad)}
    assert "does not match its file name" in problems["wrong_params.yaml"]
    assert "not registered in master.yaml" in problems["cake_bacon_params.yaml"]


def test_a_folder_holding_the_repository_is_refused():
    repo = Path(reproduce.__file__).resolve().parents[3]
    with pytest.raises(ValueError, match="holds the repository"):
        reproduce.check_output_root(repo.parent)
    assert reproduce.check_output_root(repo / "projects" / "X") == (repo / "projects" / "X").resolve()


def test_force_includes_done_runs(project, monkeypatch):
    ship(project, "cake_bacon", "816fd431", "fig26", monkeypatch)
    assert reproduce.register(project, env="dev", batch="fig26") == ["cake_bacon"]
    registry = strip_seal(json.loads((project / "artifacts" / "run_registry.json").read_text()))
    registry["816fd431"]["status"] = "DONE"
    (project / "artifacts" / "run_registry.json").write_text(json.dumps(registry))
    assert reproduce.register(project, env="dev", models=["cake_bacon"]) == []
    assert reproduce.register(project, env="dev", models=["cake_bacon"], force=True) == ["cake_bacon"]


def test_a_written_snapshot_holds_no_test_request(tmp_path):
    """test_config.yaml merged by master.yaml extra_configs is not sealed into the run's snapshot."""
    cfg = _cfg("wallet_banana", "h1")
    cfg["test"] = {"test_request": {"model": ["mesh_chicken"]}}
    cfg["project"]["extra_configs"] = {"test": "config/test_config.yaml"}
    out = tmp_path / "wallet_banana_params.yaml"
    dr.write_params_snapshot(cfg, out)
    snap = yaml.safe_load(out.read_text())
    assert "test" not in snap
    assert snap["project"]["extra_configs"] == {"test": "config/test_config.yaml"}     # the hashed path stays


def test_place_into_another_destination_leaves_the_scanned_folder_alone(project, tmp_path, monkeypatch):
    """Reproduce into a chosen folder: checked where it was scanned, copied to its params_out path in the
    destination, with the destination's registry state; never over a different snapshot there."""
    monkeypatch.setattr(reproduce, "resolve_project", lambda name: project if name == "P" else (_ for _ in ()).throw(KeyError(name)))
    src = ship(project, "cake_bacon", "816fd431", "fig26", monkeypatch)
    dest = tmp_path / "elsewhere"
    dest.mkdir()
    row = reproduce.place(src, dest, source=project)
    target = dest / "artifacts/dev/generated_models/cake_bacon/config/cake_bacon_params.yaml"
    assert (row["target"], row["placed"], row["registered"], row["status"]) == (str(target), True, False, "")
    assert target.read_bytes() == src.read_bytes()
    assert reproduce.place(src, dest, source=project)["placed"] is True       # again: the same snapshot, nothing changes
    target.write_text(target.read_text().replace("batch_size: 100", "batch_size: 200"))
    with pytest.raises(ValueError, match="a different snapshot already sits"):
        reproduce.place(src, dest, source=project)
