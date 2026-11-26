# services/side_channel_core/tests/unit/test_clearml_env_path.py
"""master.yaml names the ClearML credential file relative to the repository (secrets/clearml_global.env):
the tracker resolves it against WORKSPACE_ROOT (the container's /app) or this checkout, so a run's
snapshot records the relative path, never a container path (audit 2026-10-23)."""
from pathlib import Path

import yaml

from services.visualization.clearml import clearml_tracker as ct

ROOT = Path(__file__).resolve().parents[4]


def test_workspace_root_is_the_env_or_this_checkout(monkeypatch):
    monkeypatch.delenv("WORKSPACE_ROOT", raising=False)
    assert ct.workspace_root() == ROOT
    monkeypatch.setenv("WORKSPACE_ROOT", "/app")
    assert ct.workspace_root() == Path("/app")


def test_master_and_blueprint_name_the_env_file_relative_to_the_repository():
    for rel in ("services/side_channel_core/config/master.yaml",
                "services/side_channel_core/config/blueprint/master_project.yaml"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        files = [line.split(":", 1)[1].strip().strip('"') for line in text.splitlines() if "clearml_env_file:" in line]
        assert files, rel
        assert all(f.startswith("secrets/") for f in files), (rel, files)
        assert not any(f.startswith("/app") for f in files), (rel, files)
