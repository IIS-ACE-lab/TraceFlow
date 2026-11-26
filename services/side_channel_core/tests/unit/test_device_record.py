"""<run>/device.yaml: the device a run trained on and its timing probe (2026-10-15)."""

import json

import pytest
import torch
import yaml

from utils import device_record as dr

CFG = {"device": "cpu", "deep_learning": {"framework": "pytorch"}, "trainer": {"precision": "default"}}


@pytest.fixture
def small_probe(monkeypatch):
    # The real workload takes seconds on a CPU; the tests time a smaller one.
    for key, value in (("batch", 4), ("warmup_steps", 1), ("steps", 1), ("repeats", 2)):
        monkeypatch.setitem(dr.PROBE_SPEC, key, value)


def test_record_round_trip_and_fingerprint(tmp_path, small_probe):
    path = dr.write_device_record(CFG, tmp_path)
    assert path == tmp_path / "device.yaml"
    record = dr.read_device_record(tmp_path)          # default: the run folder
    assert record["kind"] == "cpu"
    assert record["software"]["torch"] == str(torch.__version__)
    assert record["precision"]["setting"] == "default"
    assert record["probe"]["version"] == dr.PROBE_SPEC["version"]
    assert len(record["probe"]["repeats_seconds_per_step"]) == 2
    assert dr.probe_seconds(record) == record["probe"]["seconds_per_step"] > 0
    assert dr.device_fingerprint(record) == ("cpu", record["name"], str(torch.__version__), record["software"]["cuda"],
                                             record["software"]["cudnn"], record["precision"]["cudnn_allow_tf32"],
                                             record["precision"]["matmul_allow_tf32"])
    assert list(yaml.safe_load(path.read_text())) == ["run_device"]
    assert not list(tmp_path.glob(".device.*"))           # no temporary file left
    assert dr.read_device_record(tmp_path / "none") is None and dr.device_fingerprint(None) is None


def test_probe_keeps_the_random_state_of_the_run(tmp_path, small_probe):
    torch.manual_seed(7)
    expected = torch.rand(4)
    torch.manual_seed(7)
    dr.write_device_record(CFG, tmp_path)
    assert torch.equal(torch.rand(4), expected)


def test_a_failed_probe_is_recorded_and_never_stops_the_run(tmp_path, monkeypatch):
    def broken(cfg):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(dr, "run_device_probe", broken)
    dr.write_device_record(CFG, tmp_path)
    record = dr.read_device_record(tmp_path)
    assert record["probe"] == {"version": dr.PROBE_SPEC["version"], "failed": "RuntimeError: out of memory"}
    assert dr.probe_seconds(record) is None


def test_a_probe_of_another_version_is_not_used():
    assert dr.probe_seconds({"probe": {"version": dr.PROBE_SPEC["version"] + 1, "seconds_per_step": 0.1}}) is None
    assert dr.probe_seconds({"probe": {"version": dr.PROBE_SPEC["version"], "seconds_per_step": 0.1}}) == 0.1


def test_cuda_requested_without_a_gpu_fails_instead_of_recording_the_cpu(monkeypatch, tmp_path, small_probe):
    from utils.device import GPUFailedError

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(GPUFailedError, match="^GPU FAILED"):
        dr.write_device_record({**CFG, "device": "cuda"}, tmp_path)
    assert dr.read_device_record(tmp_path) is None


def test_model_filter_selects_runs_by_their_device(tmp_path):
    from services.model_filter.model_filter import find_runs

    root = tmp_path / "P"
    (root / "artifacts").mkdir(parents=True)
    runs = []
    for h, name, gpu in (("h1", "a", "RTX 5090"), ("h2", "b", "GTX 1080 Ti"), ("h3", "c", None)):
        runs.append({"model_hash": h, "model_name": name, "env": "dev", "status": "DONE",
                     "path": f"artifacts/dev/generated_models/{name}"})
        if gpu:
            folder = root / "artifacts/dev/generated_models" / name
            folder.mkdir(parents=True)
            (folder / "device.yaml").write_text(yaml.safe_dump({"run_device": {"kind": "cuda", "name": gpu}}))
    (root / "artifacts" / "run_map.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs))
    found = find_runs(root, [{"path": "run_device.name", "equals": "GTX 1080 Ti"}])
    assert [r.model_name for r in found] == ["b"]
    assert [r.model_name for r in find_runs(root, [{"path": "run_device.name", "exists": False}])] == ["c"]


def test_the_record_goes_where_paths_device_says_and_readers_find_it(tmp_path, small_probe, monkeypatch):
    """paths.device is rendered like every path; readers follow the run's own snapshot."""
    project = tmp_path / "P"
    run = project / "artifacts/dev/generated_models/funny_hamster"
    (run / "config").mkdir(parents=True)
    template = "artifacts/{{env}}/devices/{{model_name}}"
    (run / "config" / "funny_hamster_params.yaml").write_text(
        yaml.safe_dump({"env": "dev", "model_name": "funny_hamster", "paths": {"device": template}}))
    assert dr.locate_device_record(run) == project / "artifacts/dev/devices/funny_hamster/device.yaml"

    monkeypatch.chdir(project)
    cfg = {**CFG, "env": "dev", "model_name": "funny_hamster", "project": {"root": str(project)},
           "paths": {"device": template}}
    written = dr.write_device_record(cfg)
    assert written.resolve() == (project / "artifacts/dev/devices/funny_hamster/device.yaml").resolve()
    assert dr.read_device_record(run)["kind"] == "cpu"

    # Without paths.device: the run folder, never config/ (shipped by a reproduction).
    other = project / "artifacts/dev/generated_models/sprinkle_tofu"
    assert dr.locate_device_record(other) == other / "device.yaml"


def test_code_version_comes_from_the_makefile_then_git_then_says_why_not(monkeypatch):
    """The container has no .git: the Makefile passes the commit; a bare run asks git."""
    monkeypatch.setenv("TRACEFLOW_COMMIT", "a" * 40)
    monkeypatch.setenv("TRACEFLOW_CODE_MODIFIED", "1")
    monkeypatch.setenv("TRACEFLOW_CODE_DIFF", "0123456789abcdef")
    assert dr.code_version() == {"commit": "a" * 40, "services_modified": True,
                                 "services_diff_sha256": "0123456789abcdef",
                                 "source": "the host checkout, read by the Makefile"}
    monkeypatch.setenv("TRACEFLOW_CODE_MODIFIED", "0")
    assert dr.code_version()["services_diff_sha256"] == "none"

    monkeypatch.delenv("TRACEFLOW_COMMIT")
    from_git = dr.code_version()          # this checkout is a git repository
    assert len(from_git["commit"]) == 40 and from_git["source"] == "git in this process"

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(dr.subprocess, "run", no_git)
    assert dr.code_version()["commit"].startswith("not recorded: TRACEFLOW_COMMIT unset")


def test_the_device_record_carries_the_code_version(tmp_path, small_probe, monkeypatch):
    monkeypatch.setenv("TRACEFLOW_COMMIT", "b" * 40)
    monkeypatch.setenv("TRACEFLOW_CODE_MODIFIED", "0")
    dr.write_device_record(CFG, tmp_path)
    assert dr.read_device_record(tmp_path)["code"]["commit"] == "b" * 40


def test_the_device_record_is_read_only_and_readable_by_everyone(tmp_path, small_probe):
    """mkstemp writes 0600: a record written by the container's root was unreadable on the host."""
    import stat
    path = dr.write_device_record(CFG, tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    path = dr.write_device_record(CFG, tmp_path)           # a retry replaces it
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
