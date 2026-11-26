"""Artifact seals (utils/seal.py): every format carries its own Ed25519 signature (2026-10-15)."""

import json
import os
import zipfile

import h5py
import numpy as np
import pytest
import torch
import yaml

from utils import seal as sl


@pytest.fixture(autouse=True)
def key_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACEFLOW_SEAL_DIR", str(tmp_path / "keys"))
    sl._signer_cache.clear()
    sl._verified.clear()
    sl._warned.clear()
    yield tmp_path / "keys"
    sl._signer_cache.clear()
    sl._verified.clear()


def run_file(tmp_path, rel):
    path = tmp_path / "P" / "artifacts" / "dev" / "generated_models" / "funny_hamster" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_the_key_is_created_once_private_and_reused(key_dir, tmp_path):
    path = run_file(tmp_path, "a.yaml")
    path.write_text(sl.seal_text(path, "a: 1\n"))
    key_file = key_dir / "signing.key"
    assert key_file.is_file() and (key_file.stat().st_mode & 0o777) == 0o600
    first = sl.own_key_id()
    sl._signer_cache.clear()
    assert sl.own_key_id() == first and len(first) == 16


@pytest.mark.parametrize("rel,text", [
    ("cross_validation/summary.yaml", "headline:\n  traces: 120\n"),
    ("cross_validation/aggregate/scalar_statistics.json", json.dumps({"a": 1.5, "b": [1, 2, None], "c": float("nan")}, indent=2)),
    ("metrics/scalars/x_history.jsonl", "\n".join(json.dumps({"epoch": e, "acc": e / 10}) for e in (1, 2, 3)) + "\n"),
])
def test_text_formats_round_trip_and_detect_edits(tmp_path, rel, text):
    path = run_file(tmp_path, rel)
    path.write_text(sl.seal_text(path, text))
    assert sl.verify(path).state == "ok"
    if rel.endswith(".yaml"):
        assert sl.load_yaml(path) == yaml.safe_load(text)
        path.write_text(path.read_text().replace("120", "121"))
    elif rel.endswith(".jsonl"):
        assert [r["epoch"] for r in sl.read_jsonl(path)] == [1, 2, 3]
        assert all(sl.SEAL_KEY not in r for r in sl.read_jsonl(path))
        path.write_text(path.read_text().replace('"acc":0.2', '"acc":0.9'))
    else:
        loaded = sl.load_json(path)
        assert sl.SEAL_KEY not in loaded and loaded["a"] == 1.5
        path.write_text(path.read_text().replace("1.5", "2.5"))
    sl._verified.clear()
    with pytest.raises(sl.TamperedArtifactError, match="no longer matches"):
        sl.check(path)


def test_jsonl_chain_catches_a_deleted_or_reordered_row(tmp_path):
    path = run_file(tmp_path, "metrics/h.jsonl")
    path.write_text(sl.seal_text(path, "\n".join(json.dumps({"epoch": e}) for e in (1, 2, 3)) + "\n"))
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert sl.verify(path).state == "tampered"
    path.write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n")
    assert sl.verify(path).state == "tampered"


def test_npy_npz_pt_and_h5_load_unchanged_and_detect_edits(tmp_path):
    curve = np.arange(10, dtype=np.float64)
    npy = run_file(tmp_path, "metrics/arrays/validation/c/ep0001.npy")
    np.save(npy, curve)
    sl.seal_npy_file(npy)
    assert np.array_equal(sl.load_npy(npy), curve) and np.array_equal(np.load(npy, mmap_mode="r"), curve)

    npz = run_file(tmp_path, "cross_validation/fold_indices.npz")
    np.savez_compressed(npz, a=curve)
    sl.seal_zip_file(npz)
    assert sl.verify(npz).state == "ok" and np.array_equal(np.load(npz)["a"], curve)

    pt = run_file(tmp_path, "model/m_trained.pt")
    torch.save({"w": torch.ones(3)}, pt)
    sl.seal_zip_file(pt)
    assert sl.verify(pt).state == "ok" and torch.load(pt, weights_only=True)["w"].sum() == 3

    h5 = run_file(tmp_path, "data/train.h5")
    with h5py.File(h5, "w") as handle:
        handle["traces"] = np.arange(20, dtype=np.int8).reshape(4, 5)
        handle.attrs["source"] = "ASCAD.h5"
    sl.seal_h5_file(h5)
    assert sl.verify(h5).state == "ok"

    raw = bytearray(npy.read_bytes())
    raw[200] ^= 1
    npy.write_bytes(bytes(raw))
    assert sl.verify(npy).state == "tampered"
    with h5py.File(h5, "r+") as handle:
        handle["traces"][0, 0] = 99
    assert sl.verify(h5).state == "tampered"
    torch.save({"w": torch.zeros(3)}, pt.with_name("other.pt"))
    with zipfile.ZipFile(pt) as old:
        comment = old.comment
    torch.save({"w": torch.zeros(3)}, pt)
    with zipfile.ZipFile(pt, "a") as archive:
        archive.comment = comment
    assert sl.verify(pt).state == "tampered"


def test_a_log_takes_a_seal_line_per_writer_and_detects_edits_above_it(tmp_path):
    log = run_file(tmp_path, "logs/Training.log")
    log.write_text("epoch 1\nepoch 2\n")
    sl.seal_log_file(log)
    with open(log, "a") as handle:
        handle.write("test done\n")
    sl.seal_log_file(log)
    assert sl.verify(log).state == "ok"
    log.write_text(log.read_text().replace("epoch 2", "epoch 9"))
    assert sl.verify(log).state == "tampered"


def test_a_file_moved_inside_the_run_fails_but_a_renamed_run_folder_does_not(tmp_path):
    a = run_file(tmp_path, "metrics/arrays/v/c/ep0001.npy")
    np.save(a, np.ones(3))
    sl.seal_npy_file(a)
    b = a.with_name("ep0002.npy")
    b.write_bytes(a.read_bytes())
    assert sl.verify(b).state == "tampered"
    renamed = tmp_path / "Q" / "artifacts" / "dev" / "generated_models" / "other_name" / "metrics/arrays/v/c/ep0001.npy"
    renamed.parent.mkdir(parents=True)
    renamed.write_bytes(a.read_bytes())
    assert sl.verify(renamed).state == "ok"


def test_a_sealed_run_refuses_a_removed_seal_and_a_file_signed_by_another_key(tmp_path, key_dir, monkeypatch):
    device = run_file(tmp_path, "device.yaml")
    device.write_text(sl.seal_text(device, yaml.safe_dump({"run_device": {"artifact_seal": sl.seal_declaration()}})))
    summary = run_file(tmp_path, "cross_validation/summary.yaml")
    summary.write_text("headline: 1\n")
    with pytest.raises(sl.TamperedArtifactError, match="seal was removed"):
        sl.check(summary)

    monkeypatch.setenv("TRACEFLOW_SEAL_DIR", str(tmp_path / "intruder"))
    summary.write_text(sl.seal_text(summary, "headline: 2\n"))
    sl._verified.clear()
    with pytest.raises(sl.TamperedArtifactError, match="re-signed"):
        sl.check(summary)


def test_a_removed_seal_is_caught_whatever_mtime_the_file_is_given(tmp_path, key_dir):
    """Backdating the file's mtime below the run's `since` hid a removed seal: the judge reads the
    change time, which the kernel sets and no editor can set back (audit 2026-10-23)."""
    device = run_file(tmp_path, "device.yaml")
    declaration = sl.seal_declaration()
    device.write_text(sl.seal_text(device, yaml.safe_dump({"run_device": {"artifact_seal": declaration}})))
    summary = run_file(tmp_path, "cross_validation/summary.yaml")
    summary.write_text(sl.seal_text(summary, "headline: 1\n"))
    assert sl.check(summary).state == "ok"
    sl._verified.clear()
    summary.write_text("headline: 2\n")                                 # the seal stripped, the value changed
    old = int(declaration["since"]) - 3600
    os.utime(summary, (old, old))                                        # and the mtime set back an hour
    assert summary.stat().st_mtime < declaration["since"] <= summary.stat().st_ctime
    with pytest.raises(sl.TamperedArtifactError, match="seal was removed"):
        sl.check(summary)


def test_files_outside_sealed_runs_without_a_seal_only_warn(tmp_path, caplog):
    legacy = tmp_path / "P" / "artifacts" / "run_registry.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"6c176540": {"status": "DONE"}}))
    assert sl.load_json(legacy) == {"6c176540": {"status": "DONE"}}
    assert "no seal" in caplog.text


def test_no_key_writes_unsealed_and_never_raises(tmp_path, monkeypatch):
    blocked = tmp_path / "ro"
    blocked.mkdir()
    blocked.chmod(0o500)
    monkeypatch.setenv("TRACEFLOW_SEAL_DIR", str(blocked / "seal"))
    sl._signer_cache.clear()
    path = run_file(tmp_path, "x.yaml")
    assert sl.seal_text(path, "a: 1\n") == "a: 1\n"
    blocked.chmod(0o700)


def test_a_killed_writer_leaves_an_unsealed_log_which_is_not_called_tampered(tmp_path):
    device = run_file(tmp_path, "device.yaml")
    device.write_text(sl.seal_text(device, yaml.safe_dump({"run_device": {"artifact_seal": sl.seal_declaration()}})))
    log = run_file(tmp_path, "logs/Training.log")
    log.write_text("epoch 1\n")
    assert sl.check(log).state == "unsealed"


def test_without_the_cryptography_package_files_are_unsealed_and_reads_go_on(tmp_path, monkeypatch):
    path = run_file(tmp_path, "a.json")
    path.write_text(sl.seal_text(path, json.dumps({"a": 1})))
    monkeypatch.setitem(__import__("sys").modules, "cryptography.hazmat.primitives.asymmetric.ed25519", None)
    sl._signer_cache.clear()
    assert sl.seal_text(path.with_suffix(".yaml"), "a: 1\n") == "a: 1\n"
    assert sl.check(path).state == "unverifiable" and sl.load_json(path) == {"a": 1}


def test_make_verify_reports_and_make_seal_signs_a_run_written_before_seals(tmp_path, capsys):
    run = run_file(tmp_path, "device.yaml").parent
    (run / "device.yaml").write_text(yaml.safe_dump({"run_device": {"gpu": "x"}}))
    summary = run_file(tmp_path, "cross_validation/summary.yaml")
    summary.write_text("headline: 1\n")
    rows = run_file(tmp_path, "metrics/scalars/h.jsonl")
    rows.write_text('{"epoch":1}\n{"epoch":2}\n')
    curve = run_file(tmp_path, "metrics/arrays/v/c/ep0001.npy")
    np.save(curve, np.ones(3))
    checkpoint = run_file(tmp_path, "model/m.pt")
    torch.save({"w": torch.ones(1)}, checkpoint)
    before = checkpoint.read_bytes()

    assert sl.report([run]) == 0 and "unsealed" in capsys.readouterr().out
    assert sl.seal_existing([run]) == 0
    assert sl.verify(summary).state == sl.verify(rows).state == sl.verify(curve).state == "ok"
    assert checkpoint.read_bytes() == before                  # its SHA-256 is in the checkpoint manifest
    declared = yaml.safe_load((run / "device.yaml").read_text())["run_device"]
    assert declared["gpu"] == "x" and declared["artifact_seal"]["sealed_after_run"] is True
    assert sl.verify(run / "device.yaml").state == "ok"

    summary.write_text(summary.read_text().replace("headline: 1", "headline: 2"))
    sl._verified.clear()
    assert sl.report([run]) == 1 and "TAMPERED" in capsys.readouterr().out
    assert sl.seal_existing([run]) == 1                       # a changed file is never re-signed


def test_a_reproducer_trains_from_the_authors_snapshot_under_their_own_key(tmp_path, monkeypatch):
    snapshot = run_file(tmp_path, "config/funny_hamster_params.yaml")
    snapshot.write_text(sl.seal_text(snapshot, "snapshot_sha256: abc\nmodel_name: funny_hamster\n"))  # the author
    author = sl.verify(snapshot).key
    monkeypatch.setenv("TRACEFLOW_SEAL_DIR", str(tmp_path / "reproducer"))
    sl._signer_cache.clear()
    device = run_file(tmp_path, "device.yaml")
    device.write_text(sl.seal_text(device, yaml.safe_dump({"run_device": {"artifact_seal": sl.seal_declaration()}})))
    assert sl.own_key_id() != author
    assert sl.check(snapshot).key == author                   # any valid key: its seal names who wrote it
    snapshot.write_text(snapshot.read_text().replace("funny_hamster\n", "other\n"))
    sl._verified.clear()
    with pytest.raises(sl.TamperedArtifactError):
        sl.check(snapshot)
