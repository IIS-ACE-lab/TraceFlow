# services/side_channel_core/tests/unit/test_config_freeze.py
"""A run reads its frozen copy of config/ (utils/config_freeze.py), and the copy plans exactly what the files do."""
import os
from pathlib import Path

import pytest
import yaml

from services.side_channel_core.utils import config_freeze as cf
from services.side_channel_core.utils.config_loader import get_project_config
from services.side_channel_core.utils.model_naming import make_experiment_hash

PROJECT = "DLSCA_ASCAD"


@pytest.fixture
def frozen(monkeypatch):
    root = cf.project_root_of(PROJECT)
    rel = cf.freeze(root, f"pytest-{os.getpid()}")
    monkeypatch.setenv(cf.ENV_DIR, rel)
    monkeypatch.setenv(cf.ENV_PROJECT, PROJECT)
    yield root, rel
    cf.release(root, rel)


def _without_config_abs(node):
    if isinstance(node, dict):
        return {k: _without_config_abs(v) for k, v in node.items() if k != "config_abs"}
    if isinstance(node, list):
        return [_without_config_abs(v) for v in node]
    return node


def test_the_copy_gives_the_same_config_as_the_files(frozen, monkeypatch):
    copy_cfg = get_project_config(PROJECT, validate_data=False)
    monkeypatch.delenv(cf.ENV_DIR)
    live_cfg = get_project_config(PROJECT, validate_data=False)
    # the same config, so the same sweep and the same model hashes; only config_abs names the copy
    assert _without_config_abs(copy_cfg) == _without_config_abs(live_cfg)
    root, rel = frozen
    graph = ((copy_cfg.get("visualization") or {}).get("graph_manager") or {}).get("config_abs")
    if graph:
        assert Path(graph).is_relative_to((root / rel).resolve())


def test_a_run_reads_the_copy_not_params_yaml(frozen):
    root, rel = frozen
    params = root / rel / "params.yaml"
    data = yaml.safe_load(params.read_text(encoding="utf-8"))
    data["seed"] = 987654
    params.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    assert get_project_config(PROJECT, validate_data=False)["seed"] == 987654


def test_another_project_and_a_missing_copy(frozen, monkeypatch):
    root, rel = frozen
    monkeypatch.setenv(cf.ENV_PROJECT, "OTHER")
    assert cf.frozen_dir(PROJECT, root) is None
    monkeypatch.setenv(cf.ENV_PROJECT, PROJECT)
    monkeypatch.setenv(cf.ENV_DIR, "config/.frozen/gone")
    with pytest.raises(FileNotFoundError, match="frozen configuration of this run is missing"):
        get_project_config(PROJECT, validate_data=False)


def test_the_copy_leaves_out_older_copies_and_release_removes_only_a_copy(tmp_path):
    (tmp_path / "config" / ".frozen" / "old").mkdir(parents=True)
    (tmp_path / "config" / "params.yaml").write_text("seed: 1\n", encoding="utf-8")
    rel = cf.freeze(tmp_path, "q-1")
    assert sorted(p.name for p in (tmp_path / rel).iterdir()) == ["params.yaml"]
    with pytest.raises(FileExistsError):
        cf.freeze(tmp_path, "q-1")
    cf.release(tmp_path, "config")                         # not a copy: kept
    assert (tmp_path / "config" / "params.yaml").exists()
    cf.release(tmp_path, rel)
    assert not (tmp_path / rel).exists()


def test_registered_hashes_are_unchanged_under_a_copy(frozen):
    """Each DONE run's snapshot still hashes to its registry hash with the copy in place."""
    from services.side_channel_core.utils.seal import load_json
    from services.side_channel_core.utils.data_reproducibility import read_params_snapshot
    root, _ = frozen
    cfg = get_project_config(PROJECT, validate_data=False)
    registry = load_json(root / "artifacts" / "run_registry.json")
    checked = 0
    for model_hash, entry in registry.items():
        if not isinstance(entry, dict) or str(entry.get("status")).upper() != "DONE":
            continue
        name, env = entry.get("model_name"), entry.get("env") or "dev"
        snap = root / "artifacts" / env / "generated_models" / name / "config" / f"{name}_params.yaml"
        if not snap.is_file():
            continue
        saved = read_params_snapshot(snap)
        saved["project"] = {**cfg["project"], **{k: v for k, v in (saved.get("project") or {}).items()
                                                  if k in ("extra_configs", "distributed_override")}}
        assert make_experiment_hash(saved) == model_hash, name
        checked += 1
    if not checked:
        pytest.skip("no DONE run with a snapshot in this checkout")


def test_make_test_reads_test_config_from_the_copy(frozen):
    from services.side_channel_core.src.sca.test_request import test_config_path
    root, rel = frozen
    assert test_config_path(PROJECT) == (root / rel / "test_config.yaml").resolve()
    assert test_config_path(PROJECT, "config/test_config.yaml") == (root / rel / "test_config.yaml").resolve()
