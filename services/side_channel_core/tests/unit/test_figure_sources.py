# services/side_channel_core/tests/unit/test_figure_sources.py
"""A figure's sources.json names the runs it was drawn from and is signed (2026-10-15)."""

import json

import numpy as np
import pytest

from services.graph_manager import sources as src
from utils import seal as sl


@pytest.fixture(autouse=True)
def key_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACEFLOW_SEAL_DIR", str(tmp_path / "keys"))
    sl._signer_cache.clear()
    sl._verified.clear()
    yield
    sl._signer_cache.clear()
    sl._verified.clear()


def _project(tmp_path):
    project = tmp_path / "P"
    run = project / "artifacts" / "dev" / "generated_models" / "cake_bacon"
    curve = run / "metrics" / "arrays" / "test" / "rank" / "ep0001.npy"
    curve.parent.mkdir(parents=True)
    np.save(curve, np.arange(5.0))
    sl.seal_npy_file(curve)
    legacy = run / "cross_validation" / "summary.yaml"           # written before seals existed
    legacy.parent.mkdir(parents=True)
    legacy.write_text("headline: 1\n")
    registry = project / "artifacts" / "run_registry.json"
    registry.write_text(sl.seal_text(registry, json.dumps({"abcd1234": {"model_name": "cake_bacon"}})))
    return project, run, curve, legacy, registry


def _draw(project, run, curve, legacy, registry):
    with sl.recording() as reads:
        sl.load_json(registry)
        sl.load_npy(curve)
        sl.load_yaml(legacy)
    job_dir = project / "graphs" / "jobs" / "job_001_rank"
    job_dir.mkdir(parents=True)
    figure = job_dir / "solo" / "cake_bacon" / "test" / "rank.pdf"      # jobs write into subfolders
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"%PDF-1.4 figure")
    sources = src.job_sources(
        job={"id": 1, "name": "rank", "models": ["cake_bacon"]},
        bound=[{"model_name": "cake_bacon", "model_hash": "abcd1234", "status": "DONE"}],
        reads=reads, outputs=[figure], project_root=project, env="dev", folder=job_dir)
    return src.write_sources(job_dir, sources), figure


def test_a_graph_job_records_its_runs_their_files_and_their_seal_state(tmp_path):
    path, figure = _draw(*_project(tmp_path))
    data = sl.strip(json.loads(path.read_text()))
    (run,) = data["runs"]
    assert (run["model_name"], run["model_hash"], run["env"], run["status"]) == ("cake_bacon", "abcd1234", "dev", "DONE")
    assert run["folder"] == "artifacts/dev/generated_models/cake_bacon"
    files = {f["path"]: f for f in run["files"]}
    assert files["metrics/arrays/test/rank/ep0001.npy"]["seal"] == "ok"
    assert "sha256" not in files["metrics/arrays/test/rank/ep0001.npy"]       # the signature pins it
    assert files["cross_validation/summary.yaml"]["seal"] == "unsealed"
    assert len(files["cross_validation/summary.yaml"]["sha256"]) == 64      # an unsealed file is pinned by its digest
    assert run["seal"] == {"ok": 1, "unsealed": 1}
    assert [f["path"] for f in data["other_files"]] == ["artifacts/run_registry.json"]
    assert data["outputs"] == [{"file": "solo/cake_bacon/test/rank.pdf", "sha256": src.file_sha256(figure)}]
    assert data["code"]["commit"]


def test_sources_json_is_signed_for_its_place_and_make_verify_flags_a_changed_figure(tmp_path, capsys):
    project, *rest = _project(tmp_path)
    path, figure = _draw(project, *rest)
    assert sl.verify(path).state == "ok" and sl.scope_of(path) == "graphs/jobs/job_001_rank/sources.json"
    assert sl.report([project / "graphs"]) == 0
    figure.write_bytes(b"%PDF-1.4 edited")
    assert sl.report([project / "graphs"]) == 1
    assert "solo/cake_bacon/test/rank.pdf no longer matches its SHA-256" in capsys.readouterr().out
    other = project / "graphs" / "jobs" / "job_002_other"
    other.mkdir()
    (other / "sources.json").write_bytes(path.read_bytes())                 # moved to another job: refused
    assert sl.verify(other / "sources.json").state == "tampered"


def test_a_facet_names_its_panels_and_the_runs_behind_them(tmp_path):
    project, *rest = _project(tmp_path)
    job_sources, figure = _draw(project, *rest)
    out = project / "graphs" / "facets" / "facet_001_x" / "x.pdf"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"%PDF facet")
    data = src.facet_sources(facet={"id": 1, "name": "x"}, panels=[figure], outputs=[out],
                             graph_root=project / "graphs", folder=out.parent)
    (panel,) = data["panels"]
    assert panel["file"] == "jobs/job_001_rank/solo/cake_bacon/test/rank.pdf"
    assert panel["job_sources"] == {"file": "jobs/job_001_rank/sources.json",
                                    "sha256": src.file_sha256(job_sources), "seal": "ok"}
    assert panel["runs"] == [{"model_name": "cake_bacon", "model_hash": "abcd1234"}]


def test_the_device_record_names_the_training_image(monkeypatch):
    from utils.device_record import image_version

    monkeypatch.delenv("TRACEFLOW_IMAGE", raising=False)
    assert image_version()["tag"].startswith("not recorded")
    monkeypatch.setenv("TRACEFLOW_IMAGE", "sidechannel:dev-f142de3")
    monkeypatch.setenv("TRACEFLOW_IMAGE_ID", "sha256:" + "a" * 64)
    monkeypatch.setenv("TRACEFLOW_IMAGE_DIGESTS", '["sidechannel@sha256:' + "b" * 64 + '"]')
    assert image_version() == {"tag": "sidechannel:dev-f142de3", "id": "sha256:" + "a" * 64,
                               "repo_digests": ["sidechannel@sha256:" + "b" * 64]}
