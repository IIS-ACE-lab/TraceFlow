# services/side_channel_core/tests/unit/test_external_test_device.py
"""External tests on another device and another checkout root (2026-10-27): test_config.yaml test_request.device
runs an external test on cpu | cuda instead of the snapshot's, and a checkpoint manifest written in the container
(/app/projects/...) names the same run file under a host checkout (projects/...), its SHA-256 still checked."""

from pathlib import Path

import pytest
import yaml


def _config(tmp_path, block):
    path = tmp_path / "test_config.yaml"
    path.write_text(yaml.safe_dump({"test_request": {"model": ["m"], **block}}))
    return path


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_an_external_test_runs_on_the_device_the_request_names(tmp_path, device):
    from src.sca.test_request import external_device

    assert external_device(_config(tmp_path, {"device": device})) == device
    assert external_device(_config(tmp_path, {})) is None                 # empty: the snapshot's
    assert external_device(_config(tmp_path, {"device": ""})) is None


def test_a_device_the_system_does_not_run_on_is_refused(tmp_path):
    from src.sca.test_request import TestRequestError, external_device

    with pytest.raises(TestRequestError, match="device must be one of"):
        external_device(_config(tmp_path, {"device": "gpu"}))


def test_a_manifest_written_in_the_container_names_the_same_run_file_on_the_host(tmp_path):
    from utils.checkpoint_policy import _same_run_file

    host = tmp_path / "checkout" / "projects" / "ZvW" / "artifacts" / "dev" / "generated_models" / "a" / "model" / "a.pt"
    assert _same_run_file(Path("/app/projects/ZvW/artifacts/dev/generated_models/a/model/a.pt"), host)
    assert not _same_run_file(Path("/app/projects/ZvW/artifacts/dev/generated_models/b/model/a.pt"), host)   # another run
    assert not _same_run_file(Path("/app/projects/P/artifacts/dev/generated_models/a/model/a.pt"), host)     # another project
    assert not _same_run_file(Path("/app/x/a.pt"), tmp_path / "x" / "a.pt")                               # no projects folder


def test_the_host_test_still_checks_the_checkpoint_sha256(tmp_path):
    from utils.checkpoint_policy import CheckpointPolicyError, _verify_manifest_checkpoint

    run = tmp_path / "projects" / "ZvW" / "artifacts" / "dev" / "generated_models" / "a" / "model"
    run.mkdir(parents=True)
    ckpt = run / "a.pt"
    ckpt.write_bytes(b"weights")
    named = Path("/app/projects/ZvW/artifacts/dev/generated_models/a/model/a.pt")
    with pytest.raises(CheckpointPolicyError):
        _verify_manifest_checkpoint(path=named, expected_path=ckpt, expected_sha256="0" * 64, field="final checkpoint")
