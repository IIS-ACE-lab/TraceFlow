# services/side_channel_core/tests/unit/test_dataset_converters.py
"""Converted databases (src/sca/converters/, utils/dataset_fetch.ensure): the ASCAD layout from the official files."""

import hashlib
import json

import h5py
import numpy as np
import pytest

from services.side_channel_core.src.sca import dataset_types as dt
from services.side_channel_core.utils import dataset_fetch as df
from sca.converters import aes_rd, ches_ctf_2018


def _ches_source(path, n_prof=6, n_att=4, samples=5):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for name, n, key in (("profiling", n_prof, 7), ("attacking", n_att, 9)):
            data = np.zeros((n, 48), np.uint8)
            data[:, 0:16] = rng.integers(0, 256, (n, 16))
            data[:, 16:32] = rng.integers(0, 256, (n, 16))
            data[:, 32:48] = key
            f[f"{name}_traces"] = rng.standard_normal((n, samples)).astype(np.float32)
            f[f"{name}_data"] = data


def test_ches_ctf_2018_rows_become_plaintext_ciphertext_and_key(tmp_path):
    _ches_source(tmp_path / "ches_ctf.h5")
    ches_ctf_2018.convert({"ches_ctf.h5": tmp_path / "ches_ctf.h5"}, tmp_path / "out.h5")
    with h5py.File(tmp_path / "out.h5") as out, h5py.File(tmp_path / "ches_ctf.h5") as src:
        for group, prefix in (("Profiling_traces", "profiling"), ("Attack_traces", "attacking")):
            meta, data = out[group]["metadata"][()], src[f"{prefix}_data"][()]
            assert np.array_equal(meta["plaintext"], data[:, :16]) and np.array_equal(meta["ciphertext"], data[:, 16:32])
            assert np.array_equal(meta["key"], data[:, 32:])
            assert np.array_equal(out[group]["traces"][()], src[f"{prefix}_traces"][()])   # traces kept as they are
        assert out["Profiling_traces"]["metadata"]["key"][0, 0] == 7 and out["Attack_traces"]["metadata"]["key"][0, 0] == 9


def test_aes_rd_is_transposed_split_25k_25k_and_given_the_readme_key(tmp_path, monkeypatch):
    from scipy.io import savemat

    monkeypatch.setattr(aes_rd, "PROFILING", 3)                     # 3 + 2 traces instead of 25,000 + 25,000
    traces = np.arange(4 * 5, dtype=np.uint8).reshape(4, 5)          # 4 samples x 5 traces, as in the .mat
    plaintext = np.arange(5 * 16, dtype=np.uint8).reshape(5, 16)
    savemat(tmp_path / "ctraces_fm16x4_2.mat", {"CompressedTraces": traces, "plaintext": plaintext})
    aes_rd.convert({"ctraces_fm16x4_2.mat": tmp_path / "ctraces_fm16x4_2.mat"}, tmp_path / "out.h5")
    with h5py.File(tmp_path / "out.h5") as out:
        assert np.array_equal(out["Profiling_traces"]["traces"][()], traces.T[:3])
        assert np.array_equal(out["Attack_traces"]["traces"][()], traces.T[3:])
        assert np.array_equal(out["Attack_traces"]["metadata"]["plaintext"], plaintext[3:])
        assert bytes(out["Attack_traces"]["metadata"]["key"][0]).hex() == "2b7e151628aed2a6abf7158809cf4f3c"


@pytest.fixture
def converted_type(tmp_path, monkeypatch):
    web = tmp_path / "web"
    web.mkdir()
    _ches_source(web / "ches_ctf.h5")
    sha = hashlib.sha256((web / "ches_ctf.h5").read_bytes()).hexdigest()
    source = dt.SourceFile("FAKE/FAKE_data/source/ches_ctf.h5", sha, (web / "ches_ctf.h5").as_uri())
    fake = dt.DatasetType(name="FAKE", variant="fixed_key", required=frozenset(), optional=frozenset(),
                          description="test", conversions=(dt.Conversion("FAKE/FAKE_data/FAKE.h5", (source,), "ches_ctf_2018"),))
    monkeypatch.setitem(dt.TYPES, "FAKE", fake)
    return tmp_path / "services" / "side_channel_core" / "data", source


def test_ensure_downloads_converts_and_records_its_sources(converted_type):
    data, source = converted_type
    out = df.ensure(data / "FAKE/FAKE_data/FAKE.h5")
    with h5py.File(out) as f:
        assert f.attrs["traceflow_converter"] == "ches_ctf_2018" and int(f.attrs["traceflow_converter_version"]) == 1
        assert json.loads(f.attrs["traceflow_source_sha256"]) == {source.path: source.sha256}
        assert f["Attack_traces"]["traces"].shape == (4, 5)
    assert (data / source.path).is_file()                                       # the download is kept beside it


def test_a_changed_source_or_a_failing_converter_is_data_failed(converted_type, monkeypatch, tmp_path):
    data, source = converted_type
    master = tmp_path / "master.yaml"
    master.write_text("data_fetch:\n  on_mismatch: refuse\n")
    monkeypatch.setenv("SCC_MASTER_YAML", str(master))
    (data / source.path).parent.mkdir(parents=True)
    (data / source.path).write_bytes(b"not the published file")
    with pytest.raises(df.DataFetchError, match="^DATA FAILED: .* has SHA-256"):
        df.ensure(data / "FAKE/FAKE_data/FAKE.h5")
    (data / source.path).unlink()
    monkeypatch.setattr(ches_ctf_2018, "convert", lambda inputs, output: (_ for _ in ()).throw(ValueError("bad rows")))
    with pytest.raises(df.DataFetchError, match="^DATA FAILED: converting ches_ctf.h5 into FAKE.h5 failed"):
        df.ensure(data / "FAKE/FAKE_data/FAKE.h5")
    assert not (data / "FAKE/FAKE_data/FAKE.h5").exists() and not list((data / "FAKE/FAKE_data").glob(".*part"))


def test_a_changed_source_is_moved_aside_and_fetched_again_under_refetch(converted_type, monkeypatch, tmp_path):
    data, source = converted_type
    master = tmp_path / "master.yaml"
    master.write_text("data_fetch:\n  on_mismatch: refetch\n")
    monkeypatch.setenv("SCC_MASTER_YAML", str(master))
    (data / source.path).parent.mkdir(parents=True)
    (data / source.path).write_bytes(b"not the published file")
    out = df.ensure(data / "FAKE/FAKE_data/FAKE.h5")
    assert df.verify(out)["converter"] == "ches_ctf_2018"
    kept = list((data / source.path).parent.glob("*.mismatch-*"))
    assert len(kept) == 1 and kept[0].read_bytes() == b"not the published file"


def test_the_registered_converted_types():
    assert dt.TYPES["CHES_CTF_2018"].shared_key is False and dt.TYPES["AES_RD"].shared_key is True
    assert dt.registered("/app/services/side_channel_core/data/AES_RD/AES_RD_data/AES_RD.h5")
    assert dt.conversion_for("services/side_channel_core/data/CHES_CTF_2018/CHES_CTF_2018_data/CHES_CTF_2018.h5").converter == "ches_ctf_2018"


def test_a_converted_file_from_another_converter_version_is_refused(converted_type, monkeypatch):
    data, _ = converted_type
    out = df.ensure(data / "FAKE/FAKE_data/FAKE.h5")
    assert df.verify(out)["converter_version"] == 1
    fake = dt.TYPES["FAKE"]
    newer = dt.Conversion(fake.conversions[0].output, fake.conversions[0].inputs, "ches_ctf_2018", version=2)
    monkeypatch.setitem(dt.TYPES, "FAKE", dt.DatasetType(**{**fake.__dict__, "conversions": (newer,)}))
    with pytest.raises(df.DataFetchError, match=r"not made by sca/converters/ches_ctf_2018.py v2 .* delete it"):
        df.verify(out)


def test_the_run_records_its_checked_database_in_device_yaml(converted_type, tmp_path, monkeypatch):
    import yaml
    from utils import device_record as dr
    from utils.seal import check

    data, _ = converted_type
    out = df.ensure(data / "FAKE/FAKE_data/FAKE.h5")
    monkeypatch.setattr(dr, "device_record_dir", lambda cfg: tmp_path / "run")
    dr._write_record(tmp_path / "run" / dr.DEVICE_RECORD_FILE, {"kind": "cpu"})
    dr.record_database({}, {**df.verify(out), "registered": True})
    text = (tmp_path / "run" / dr.DEVICE_RECORD_FILE).read_text()
    database = yaml.safe_load(text)[dr.RECORD_KEY]["database"]
    assert database["converter"] == "ches_ctf_2018" and database["sha256"] == df.file_sha256(out)
    assert check(tmp_path / "run" / dr.DEVICE_RECORD_FILE, cache=False).state != "tampered"   # resealed whole
