"""Experiment identity, params_out snapshots and run data (2026-09-29: SYS-05, SYS-10, SYS-28, SYS-29)."""

import json
import logging
import shutil
from pathlib import Path

import pytest
import yaml

from services.side_channel_core.utils import model_naming
from services.side_channel_core.utils.model_naming import _load_mapping, choose_run_name, make_experiment_hash
from utils import data_reproducibility as dr
from utils.data_reproducibility import (
    SNAPSHOT_DIGEST_KEY,
    prepare_artifacts_data,
    read_params_snapshot,
    write_params_snapshot,
)
from utils.path_utils import _get_workspace_root, resolve_path, to_portable_path
from utils.sweep_test_runner import load_run_snapshot, verify_snapshot_identity

REPO = Path(__file__).resolve().parents[4]
DATABASE = "services/side_channel_core/data/ASCAD.h5"


def _cfg(root: Path) -> dict:
    """A resolved run config as config_loader builds it for a checkout at `root`."""
    scc = root / "services" / "side_channel_core"
    return {
        "env": "dev",
        "seed": 42,
        "device": "cpu",
        "project": {
            "name": "P",
            "root": str(root / "projects" / "P"),
            "scripts": {"train": str(scc / "src/sca/train.py"), "test": str(scc / "src/sca/test.py")},
            "core_src_dir": str(scc / "src"),
        },
        "data": {"name": "ascad", "database": str(root / DATABASE), "profiling_range": [0, 100]},
        "distributed": {"enabled": False, "ddp_script": str(scc / "src/distributed/ddp_training.py")},
        "model": {"model_id": "m", "layers": [{"type": "Dense", "units": 10}]},
        "trainer": {"epochs": 1, "learning_rate": 1e-3},
        "sca": {"variant": "fixed_key", "attack_byte": 2, "leakage_model": "ID"},
        "metrics_to_log": ["guessing_entropy"],
        "perform_attack": {"enabled": True, "attack_step": 10},
        "paths": {
            "params_out": "artifacts/{{env}}/generated_models/{{model_name}}/config/{{model_name}}_params.yaml",
            "data": "artifacts/{{env}}/generated_models/{{model_name}}/data",
            "data_out": "artifacts/{{env}}/generated_models/{{model_name}}/data/{{dataset_alias}}.h5",
        },
    }


def _register(root: Path, monkeypatch, name: str = "alpha") -> tuple[dict, str, Path]:
    """Hash a run at `root` and write its params_out snapshot, as training does."""
    monkeypatch.setenv("WORKSPACE_ROOT", str(root))
    cfg = _cfg(root)
    model_hash = make_experiment_hash(cfg)
    cfg.update(model_hash=model_hash, model_name=name)
    path = resolve_path(cfg["paths"]["params_out"], cfg)
    write_params_snapshot(cfg, path)
    return cfg, model_hash, path


# =============================================================================
# SYS-05: layout-independent identity
# =============================================================================
def test_hash_ignores_where_the_checkout_lives(tmp_path, monkeypatch):
    hashes = set()
    for root in (tmp_path / "home" / "clone_a", tmp_path / "srv" / "clone_b", Path("/app")):
        monkeypatch.setenv("WORKSPACE_ROOT", str(root))
        hashes.add(make_experiment_hash(_cfg(root)))
    assert len(hashes) == 1

    # Content still counts: another dataset file is another experiment.
    cfg = _cfg(Path("/app"))
    cfg["data"]["database"] = "/app/services/side_channel_core/data/ASCAD_desync50.h5"
    assert make_experiment_hash(cfg) not in hashes


def test_every_logged_metric_joins_the_hash_fit_time_included(monkeypatch):
    # Owner ruling 2026-10-14: a run without its training time is another experiment.
    monkeypatch.setenv("WORKSPACE_ROOT", "/app")
    cfg = _cfg(Path("/app"))
    base = make_experiment_hash(cfg)
    hashes = {base}
    for extra in ("mean_accuracy", "mean_fit_time_{s}", "fit_time_{per_epoch}"):
        timed = _cfg(Path("/app"))
        timed["metrics_to_log"] = ["guessing_entropy", extra]
        hashes.add(make_experiment_hash(timed))
    assert len(hashes) == 4
    # The order of the list does not matter.
    reordered = _cfg(Path("/app"))
    reordered["metrics_to_log"] = ["mean_fit_time_{s}", "guessing_entropy"]
    timed = _cfg(Path("/app"))
    timed["metrics_to_log"] = ["guessing_entropy", "mean_fit_time_{s}"]
    assert make_experiment_hash(reordered) == make_experiment_hash(timed)


def test_snapshot_stores_relative_paths_and_reproduces_the_hash_after_relocation(tmp_path, monkeypatch):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _cfg_a, model_hash, path = _register(root_a, monkeypatch)
    text = path.read_text()
    assert str(root_a) not in text
    assert yaml.safe_load(text)["data"]["database"] == DATABASE

    # Move the checkout; the same run is reloaded from its snapshot there.
    shutil.copytree(root_a / "projects", root_b / "projects")
    monkeypatch.setenv("WORKSPACE_ROOT", str(root_b))
    live = _cfg(root_b)
    loaded = load_run_snapshot(live, {"env": "dev"}, model_name="alpha")
    verify_snapshot_identity(loaded, {"env": "dev"}, model_hash=model_hash, model_name="alpha")
    assert SNAPSHOT_DIGEST_KEY not in loaded
    assert loaded["project"]["root"] == str(root_b / "projects" / "P")
    assert resolve_path(loaded["data"]["database"], loaded) == root_b / DATABASE


def test_relative_database_resolves_from_a_relocated_root(tmp_path, monkeypatch):
    from services.side_channel_core.utils.config_loader import _resolve_dataset_database_path

    root = tmp_path / "elsewhere"
    (root / DATABASE).parent.mkdir(parents=True)
    (root / DATABASE).write_bytes(b"")
    monkeypatch.setenv("WORKSPACE_ROOT", str(root))
    resolved = _resolve_dataset_database_path(
        DATABASE, cfg={}, project_name="P", project_root=root / "projects" / "P",
        projects_root=root / "projects", scc_root=tmp_path / "no_scc", ds_name="ascad",
    )
    assert Path(resolved) == (root / DATABASE).resolve()


# Every location config_loader accepts for data.database (raw value -> file),
# with the form model_hash and the snapshot store (None: kept absolute).
_DATABASE_LAYOUTS = [
    ("services/side_channel_core/data/ASCAD/A.h5", "services/side_channel_core/data/ASCAD/A.h5",
     "services/side_channel_core/data/ASCAD/A.h5"),                                   # repo-relative
    ("data/ASCAD/A.h5", "services/side_channel_core/data/ASCAD/A.h5",
     "services/side_channel_core/data/ASCAD/A.h5"),                                   # SCC-relative
    ("ASCAD/A.h5", "services/side_channel_core/data/ASCAD/A.h5",
     "services/side_channel_core/data/ASCAD/A.h5"),                                   # SCC data/ shorthand
    ("data/D.h5", "projects/P/data/D.h5", "projects/P/data/D.h5"),                    # project data/
    ("raw/E.h5", "projects/P/raw/E.h5", "raw/E.h5"),                                  # project, other folder
    ("config/G.h5", "projects/P/config/G.h5", "config/G.h5"),                         # project config/
    ("shared/C.h5", "projects/shared/C.h5", "projects/shared/C.h5"),                  # projects root
    ("datasets/B.h5", "datasets/B.h5", None),                                         # workspace top level
    ("data/H.h5", "data/H.h5", None),                                                 # workspace data/
]


@pytest.mark.parametrize("raw, located, portable", _DATABASE_LAYOUTS)
def test_every_accepted_database_location_round_trips(tmp_path, monkeypatch, caplog, raw, located, portable):
    from services.side_channel_core.utils.config_loader import _resolve_dataset_database_path

    monkeypatch.delenv("SCC_ROOT", raising=False)

    def _checkout(root: Path) -> dict:
        (root / located).parent.mkdir(parents=True, exist_ok=True)
        (root / located).write_bytes(b"")
        monkeypatch.setenv("WORKSPACE_ROOT", str(root))
        return {"project": {"name": "P", "root": str(root / "projects" / "P")}}

    root_a, root_b = tmp_path / "a", tmp_path / "b"
    cfg_a = _checkout(root_a)
    project_root = Path(cfg_a["project"]["root"])
    with caplog.at_level(logging.WARNING):
        resolved = _resolve_dataset_database_path(
            raw, cfg={}, project_name="P", project_root=project_root,
            projects_root=root_a / "projects", scc_root=root_a / "services" / "side_channel_core",
            ds_name="ascad",
        )
    assert Path(resolved) == root_a / located

    stored = to_portable_path(resolved, project_root=project_root)
    assert resolve_path(stored, cfg_a) == Path(resolved)
    warned = any("no location-independent form" in r.getMessage() for r in caplog.records)
    if portable is None:
        assert stored == resolved and warned
        return
    assert stored == portable and not warned

    # The stored form finds the same file in a checkout elsewhere.
    cfg_b = _checkout(root_b)
    assert resolve_path(stored, cfg_b) == root_b / located


def test_project_root_is_stored_workspace_relative_and_resolves_back(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    project_root = tmp_path / "projects" / "P"
    cfg = {"project": {"name": "P", "root": str(project_root)}}
    assert to_portable_path(project_root, project_root=project_root) == "projects/P"
    assert resolve_path("projects/P", cfg) == project_root
    # The workspace root has no relative form once a project root is known.
    assert to_portable_path(tmp_path, project_root=project_root) == str(tmp_path)


def test_project_data_database_survives_relocation_through_the_snapshot(tmp_path, monkeypatch):
    """The verifier's case: <project>/data/X.h5 was stored as 'data/X.h5' and resolved into SCC."""
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    monkeypatch.delenv("SCC_ROOT", raising=False)
    monkeypatch.setenv("WORKSPACE_ROOT", str(root_a))
    cfg = _cfg(root_a)
    cfg["data"]["database"] = str(root_a / "projects/P/data/X.h5")
    model_hash = make_experiment_hash(cfg)
    cfg.update(model_hash=model_hash, model_name="alpha")
    path = resolve_path(cfg["paths"]["params_out"], cfg)
    write_params_snapshot(cfg, path)
    assert yaml.safe_load(path.read_text())["data"]["database"] == "projects/P/data/X.h5"

    shutil.copytree(root_a / "projects", root_b / "projects")
    monkeypatch.setenv("WORKSPACE_ROOT", str(root_b))
    live = _cfg(root_b)
    loaded = load_run_snapshot(live, {"env": "dev"}, model_name="alpha")
    verify_snapshot_identity(loaded, {"env": "dev"}, model_hash=model_hash, model_name="alpha")
    assert resolve_path(loaded["data"]["database"], loaded) == root_b / "projects/P/data/X.h5"

    # The live config of the relocated checkout plans the same hash.
    live["data"]["database"] = str(root_b / "projects/P/data/X.h5")
    assert make_experiment_hash(live) == model_hash


def test_workspace_root_is_the_checkout_not_the_working_directory(tmp_path, monkeypatch):
    for var in ("WORKSPACE_ROOT", "REPO_ROOT", "SCC_ROOT", "PROJECTS_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    assert _get_workspace_root() == REPO


def test_snapshot_that_no_longer_hashes_to_its_model_hash_is_rejected(tmp_path, monkeypatch, caplog):
    _cfg_a, model_hash, _path = _register(tmp_path, monkeypatch)
    live = _cfg(tmp_path)
    live["project"]["scripts"]["train"] = str(tmp_path / "services/side_channel_core/src/sca/other_train.py")
    loaded = load_run_snapshot(live, {"env": "dev"}, model_name="alpha")
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match=f"not to its registered model_hash {model_hash}"):
        verify_snapshot_identity(loaded, {"env": "dev"}, model_hash=model_hash, model_name="alpha")
    assert any("hashes to" in r.getMessage() for r in caplog.records if r.levelno == logging.ERROR)


# =============================================================================
# SYS-10: model_name_map.json is never read as empty when damaged
# =============================================================================
def _write_map(root: Path, text: str) -> Path:
    path = root / "artifacts" / "model_name_map.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_missing_name_map_is_empty(tmp_path):
    assert _load_mapping(tmp_path) == {}


@pytest.mark.parametrize(
    "text, message",
    [
        ('{"h1": "alpha", "h2": ', "not valid JSON"),
        ('["alpha"]', "must hold a JSON object"),
        ('{"h1": ""}', "invalid entry"),
        ('{"h1": 7}', "invalid entry"),
        ('{"h1": "alpha", "h2": "alpha"}', "same model_name"),
    ],
)
def test_damaged_name_map_raises_and_is_not_rewritten(tmp_path, text, message, caplog):
    path = _write_map(tmp_path, text)
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match=message):
        _load_mapping(tmp_path)
    assert any("Restore" in r.getMessage() for r in caplog.records if r.levelno == logging.ERROR)

    # The writers that load it first stop too; nothing rewrites the file.
    from services.side_channel_core.utils.sweep_engine import determine_run_identity

    cfg = _cfg(tmp_path)
    cfg["project"]["root"] = str(tmp_path)
    for write in (lambda: determine_run_identity(dict(cfg)), lambda: choose_run_name(dict(cfg))):
        with pytest.raises(ValueError, match=message):
            write()
    assert path.read_text() == text


def test_make_test_rejects_a_damaged_name_map(tmp_path, monkeypatch):
    from services.side_channel_core.src.sca import test_request as tr

    _write_map(tmp_path, "{not json")
    monkeypatch.setattr(tr, "_project_block", lambda project: ({}, tmp_path))
    with pytest.raises(ValueError, match="not valid JSON"):
        tr.resolve_models("P", ["alpha"], env="dev")


def test_alias_taken_by_another_hash_is_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["project"]["root"] = str(tmp_path)
    path = _write_map(tmp_path, json.dumps({"other123": "alpha"}))
    with pytest.raises(ValueError, match="already the model_name of another hash"):
        choose_run_name(cfg, alias="alpha")
    assert json.loads(path.read_text()) == {"other123": "alpha"}


# =============================================================================
# SYS-28: atomic, verifiable params_out snapshot
# =============================================================================
def test_snapshot_is_written_atomically_with_its_digest_first(tmp_path, monkeypatch):
    _cfg_a, _h, path = _register(tmp_path, monkeypatch)
    assert next(iter(yaml.safe_load(path.read_text()))) == SNAPSHOT_DIGEST_KEY
    assert [p.name for p in path.parent.iterdir()] == [path.name]      # no temp file left
    assert read_params_snapshot(path)["model_name"] == "alpha"


def test_failed_write_keeps_the_previous_snapshot(tmp_path, monkeypatch):
    cfg, _h, path = _register(tmp_path, monkeypatch)
    before = path.read_text()

    def _crash(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(dr.yaml, "safe_dump", _crash)
    cfg["trainer"]["epochs"] = 99
    with pytest.raises(OSError, match="disk full"):
        write_params_snapshot(cfg, path)
    assert path.read_text() == before
    assert [p.name for p in path.parent.iterdir()] == [path.name]


@pytest.mark.parametrize("fraction", [0.001, 0.25, 0.5, 0.9, 0.995])
def test_truncated_snapshot_is_rejected(tmp_path, monkeypatch, fraction, caplog):
    _cfg_a, _h, path = _register(tmp_path, monkeypatch)
    text = path.read_text()
    path.write_text(text[: max(1, int(len(text) * fraction))])
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="params_out snapshot"):
        read_params_snapshot(path)
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_edited_snapshot_fails_its_digest(tmp_path, monkeypatch):
    _cfg_a, _h, path = _register(tmp_path, monkeypatch)
    edited = path.read_text().replace("epochs: 1", "epochs: 2")
    path.write_text(edited)
    with pytest.raises(ValueError, match="fails its seal"):           # its seal (utils/seal.py) first
        read_params_snapshot(path)
    path.write_text(edited.split("\n", 1)[1])                         # no seal line: as written before seals
    with pytest.raises(ValueError, match="does not match its snapshot_sha256"):
        read_params_snapshot(path)


def test_legacy_snapshot_without_digest_is_accepted_with_a_warning(tmp_path, monkeypatch, caplog):
    cfg, model_hash, path = _register(tmp_path, monkeypatch)
    legacy = yaml.safe_load(path.read_text())
    legacy.pop(SNAPSHOT_DIGEST_KEY)
    legacy["model_hash"] = "0ld0hash"                 # pre-2026-09-29 hashes cannot be recomputed
    path.write_text(yaml.safe_dump(legacy, sort_keys=False))
    with caplog.at_level(logging.WARNING):
        loaded = load_run_snapshot(_cfg(tmp_path), {"env": "dev"}, model_name="alpha")
        verify_snapshot_identity(loaded, {"env": "dev"}, model_hash="0ld0hash", model_name="alpha")
    assert loaded["model_hash"] == "0ld0hash"
    assert sum("2026-09-29" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING) == 2

    # Without model_hash (its last key, lost to truncation) it is rejected.
    legacy.pop("model_hash")
    path.write_text(yaml.safe_dump(legacy, sort_keys=False))
    with pytest.raises(ValueError, match="missing \\['model_hash'\\]"):
        read_params_snapshot(path)


def test_make_test_rejects_a_truncated_snapshot(tmp_path, monkeypatch):
    from services.side_channel_core.src.sca import test_request as tr

    _cfg_a, model_hash, path = _register(tmp_path, monkeypatch)
    path.write_text(path.read_text()[:-40])
    project_root = tmp_path / "projects" / "P"
    monkeypatch.setattr(tr, "_project_block", lambda project: ({}, project_root))
    run = {"model_name": "alpha", "model_hash": model_hash, "env": "dev",
           "path": "artifacts/dev/generated_models/alpha"}
    with pytest.raises(ValueError, match="fails its seal"):           # the seal line stays: it is the first
        tr.load_snapshot("P", run)
    path.write_text(path.read_text().split("\n", 1)[1])               # and without it, the missing keys
    with pytest.raises(ValueError, match="truncated or damaged: missing \\['model_hash', 'model_name'\\]"):
        tr.load_snapshot("P", run)


def _retry(cli_helpers, root, model_hash):
    return cli_helpers.resolve_registered_run_cfg(
        cfg=_cfg(root), project_root=root / "projects" / "P", model_hash=model_hash,
        info={"model_name": "alpha"},
        registry_before={model_hash: {"model_name": "alpha", "status": "FAILED", "env": "dev"}},
    )


def test_retry_runs_from_a_verified_snapshot(tmp_path, monkeypatch):
    from utils import cli_helpers

    _cfg_a, model_hash, _path = _register(tmp_path, monkeypatch)
    run_cfg, source = _retry(cli_helpers, tmp_path, model_hash)
    assert source == "snapshot" and run_cfg["model_hash"] == model_hash
    assert SNAPSHOT_DIGEST_KEY not in run_cfg


@pytest.mark.parametrize("damage", ["truncate", "legacy"])
def test_retry_does_not_rerun_a_damaged_or_legacy_snapshot(tmp_path, monkeypatch, damage):
    from utils import cli_helpers

    _cfg_a, model_hash, path = _register(tmp_path, monkeypatch)
    if damage == "truncate":
        path.write_text(path.read_text()[: len(path.read_text()) // 2])
        expected = (ValueError, "params_out snapshot")
    else:
        legacy = yaml.safe_load(path.read_text())
        legacy.pop(SNAPSHOT_DIGEST_KEY)
        path.write_text(yaml.safe_dump(legacy, sort_keys=False))
        expected = (RuntimeError, "registered before 2026-09-29.*make delete_database")
    monkeypatch.setattr(cli_helpers, "_replan_single_hash",
                        lambda **_: pytest.fail("the snapshot must not be replaced by a re-planned config"))
    with pytest.raises(expected[0], match=expected[1]):
        _retry(cli_helpers, tmp_path, model_hash)


# =============================================================================
# SYS-29: the live params.yaml is not copied into the run's data/ folder
# =============================================================================
def test_prepare_artifacts_data_writes_the_snapshot_not_a_params_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    cfg = _cfg(tmp_path)
    project_root = Path(cfg["project"]["root"])
    (project_root / "config").mkdir(parents=True)
    (project_root / "config" / "params.yaml").write_text("trainer: {learning_rate: [0.1, 0.01]}\n")
    cfg.update(model_hash=make_experiment_hash(cfg), model_name="alpha")
    monkeypatch.setattr(dr, "prepare_dataset_artifacts", lambda cfg: None)

    prepare_artifacts_data(cfg)
    run_dir = project_root / "artifacts" / "dev" / "generated_models" / "alpha"
    assert not (run_dir / "data" / "params.yaml").exists()
    assert read_params_snapshot(run_dir / "config" / "alpha_params.yaml")["model_hash"] == cfg["model_hash"]


def test_the_panels_hashed_keys_are_the_ones_the_run_hash_reads(monkeypatch):
    """choices.HASHED (Load Blueprint says which staged keys join the run hash): each changes the
    hash when set, and keys outside it do not."""
    import copy

    from services.visualization.vscode_monitor.choices import HASHED

    monkeypatch.setenv("WORKSPACE_ROOT", "/app")
    base_cfg = _cfg(Path("/app"))
    base_cfg["cross_validation"] = {"enabled": True, "n_folds": 10, "strategy": "k_fold", "shuffle": True}
    base = make_experiment_hash(base_cfg)

    def put(cfg, dotted, value):
        keys = [k for k in dotted.split(".") if k != "*"]     # data.*: a run's data is its one entry
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    probes = {"seed": 7, "device": "cuda", "sca.attack_byte": 3, "sca.leakage_model": "HW", "sca.variant": "variable_key",
              "metrics_to_log": ["accuracy"], "deep_learning": {"framework": "keras"}, "cross_validation.enabled": False,
              "cross_validation.n_folds": 5, "cross_validation.strategy": "stratified_k_fold", "cross_validation.shuffle": False,
              "validation.rank_step": 3, "validation.rank_eval_interval": 2, "key_recovery_reporting.recovery_rank": 2,
              "key_recovery_reporting.require_stable_recovery": False, "dataloader.drop_last_train": False}
    for key in HASHED:
        cfg = copy.deepcopy(base_cfg)
        put(cfg, key, probes[key]) if key in probes else put(cfg, f"{key}.probe", 1)   # a block: a key added in it
        assert make_experiment_hash(cfg) != base, key
    for key in ("num_workers.train", "paths.logs", "monitor_interval", "dataloader.pin_memory", "key_recovery_reporting.log_test",
                "cross_validation.report_aggregate"):
        cfg = copy.deepcopy(base_cfg)
        put(cfg, key, 3)
        assert make_experiment_hash(cfg) == base, key
    from services.visualization.vscode_monitor.choices import UNHASHED

    for key in UNHASHED:                       # inside a hashed block, left out of the hash
        for value in (True, False):
            cfg = copy.deepcopy(base_cfg)
            put(cfg, key, value)
            assert make_experiment_hash(cfg) == base, key


def test_drop_last_train_joins_the_hash_only_when_false(monkeypatch):
    """dataloader.drop_last_train false trains each epoch's last incomplete batch: other traces and another
    OneCycleLR length, so another run. True is the default, absent too: every hash made with it stays."""
    import copy

    monkeypatch.setenv("WORKSPACE_ROOT", "/app")
    base_cfg = _cfg(Path("/app"))
    base = make_experiment_hash(base_cfg)
    hashes = {}
    for value in (True, None, False):
        cfg = copy.deepcopy(base_cfg)
        if value is not None:
            cfg.setdefault("dataloader", {})["drop_last_train"] = value
        hashes[value] = make_experiment_hash(cfg)
    assert hashes[True] == hashes[None] == base
    assert hashes[False] != base
