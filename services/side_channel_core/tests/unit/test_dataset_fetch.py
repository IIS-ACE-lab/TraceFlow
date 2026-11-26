# services/side_channel_core/tests/unit/test_dataset_fetch.py
"""A missing registered database is fetched once, checked against its published SHA-256 (utils/dataset_fetch.py)."""

import hashlib
import os
import subprocess
import time
import zipfile

import pytest

from services.side_channel_core.src.sca import dataset_types as dt
from services.side_channel_core.utils import dataset_fetch as df


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A fake type whose files are served from file:// URLs: one plain file, two in one zip."""
    web = tmp_path / "web"
    web.mkdir()
    plain, first, second = b"plain database", b"first member", b"second member"
    (web / "plain.h5").write_bytes(plain)
    with zipfile.ZipFile(web / "pack.zip", "w") as z:
        z.writestr("pack/a.h5", first)
        z.writestr("pack/b.h5", second)
    zip_sha = _sha((web / "pack.zip").read_bytes())
    fake = dt.DatasetType(
        name="FAKE", variant="fixed_key", required=frozenset(), optional=frozenset(), description="test",
        sources=(
            dt.SourceFile("FAKE/plain.h5", _sha(plain), (web / "plain.h5").as_uri()),
            dt.SourceFile("FAKE/a.h5", _sha(first), (web / "pack.zip").as_uri(), member="pack/a.h5", archive_sha256=zip_sha),
            dt.SourceFile("FAKE/b.h5", _sha(second), (web / "pack.zip").as_uri(), member="pack/b.h5", archive_sha256=zip_sha),
        ),
    )
    monkeypatch.setitem(dt.TYPES, "FAKE", fake)
    data = tmp_path / "checkout" / "services" / "side_channel_core" / "data"
    return data, fake


def test_a_missing_file_is_downloaded_and_checked(served):
    data, _ = served
    assert df.fetch(data / "FAKE/plain.h5").read_bytes() == b"plain database"


def test_a_zip_is_downloaded_once_for_all_its_registered_files(served):
    data, _ = served
    df.fetch(data / "FAKE/a.h5")
    assert (data / "FAKE/b.h5").read_bytes() == b"second member"                 # taken from the same archive
    assert not list((data / "FAKE").glob(".*part"))                             # no partial file left


def test_a_file_that_does_not_match_its_published_sha256_is_refused(served, monkeypatch):
    data, fake = served
    bad = dt.SourceFile("FAKE/plain.h5", "0" * 64, fake.sources[0].url)
    monkeypatch.setitem(dt.TYPES, "FAKE", dt.DatasetType(**{**fake.__dict__, "sources": (bad,)}))
    with pytest.raises(df.DataFetchError, match="^DATA FAILED: the downloaded plain.h5 has SHA-256"):
        df.fetch(data / "FAKE/plain.h5")
    assert not (data / "FAKE/plain.h5").exists() and not list((data / "FAKE").glob(".*"))


def test_an_unregistered_database_is_not_fetched(tmp_path):
    with pytest.raises(df.DataFetchError, match="not a registered database"):
        df.fetch(tmp_path / "services/side_channel_core/data/NOWHERE/x.h5")


def test_the_owners_submodule_is_used_when_its_server_is_reachable(served, tmp_path, monkeypatch):
    data, fake = served
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "protocol.file.allow=always"]
    store = tmp_path / "store"                                   # the dataset's own repository
    store.mkdir()
    subprocess.run(git + ["init", "-q", str(store)], check=True)
    (store / "plain.h5").write_bytes(b"plain database")
    subprocess.run(git + ["-C", str(store), "add", "."], check=True)
    subprocess.run(git + ["-C", str(store), "commit", "-qm", "data"], check=True)
    checkout = tmp_path / "checkout"
    subprocess.run(git + ["init", "-q", str(checkout)], check=True)
    subprocess.run(git + ["-C", str(checkout), "submodule", "add", "-q", str(store), "services/side_channel_core/data/FAKE"],
                   check=True)
    subprocess.run(git + ["-C", str(checkout), "commit", "-qm", "sub"], check=True)
    subprocess.run(["git", "-C", str(checkout), "submodule", "deinit", "-f", "services/side_channel_core/data/FAKE"],
                   check=True, capture_output=True)
    assert not (data / "FAKE/plain.h5").exists()
    (tmp_path / "web" / "plain.h5").unlink()                     # the official download would fail now
    assert df.fetch(data / "FAKE/plain.h5").read_bytes() == b"plain database"


def test_a_submodule_file_in_a_subfolder_comes_from_the_submodule(served, tmp_path, monkeypatch):
    """Every real type keeps its files below the submodule's top (source/, ASCAD_databases/): fetch must
    not create that folder before the clone, or git refuses the non-empty submodule folder."""
    data, fake = served
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "protocol.file.allow=always"]
    store = tmp_path / "store"
    (store / "source").mkdir(parents=True)
    subprocess.run(git + ["init", "-q", str(store)], check=True)
    (store / "source" / "plain.h5").write_bytes(b"plain database")
    subprocess.run(git + ["-C", str(store), "add", "."], check=True)
    subprocess.run(git + ["-C", str(store), "commit", "-qm", "data"], check=True)
    origin = tmp_path / "origin"                                 # a fresh clone: the submodule was never fetched
    subprocess.run(git + ["init", "-q", str(origin)], check=True)
    subprocess.run(git + ["-C", str(origin), "submodule", "add", "-q", str(store), "services/side_channel_core/data/FAKE"],
                   check=True)
    subprocess.run(git + ["-C", str(origin), "commit", "-qm", "sub"], check=True)
    subprocess.run(git + ["clone", "-q", str(origin), str(tmp_path / "checkout")], check=True)
    assert not any((data / "FAKE").iterdir())
    nested = dt.SourceFile("FAKE/source/plain.h5", fake.sources[0].sha256, fake.sources[0].url)
    monkeypatch.setitem(dt.TYPES, "FAKE", dt.DatasetType(**{**fake.__dict__, "sources": (nested,)}))
    (tmp_path / "web" / "plain.h5").unlink()                     # only the submodule can deliver it
    assert df.fetch(data / "FAKE/source/plain.h5").read_bytes() == b"plain database"


def test_a_failed_fetch_is_the_runs_status_reason():
    from utils.device import failure_reason

    try:
        try:
            raise df.DataFetchError("DATA FAILED: x.h5 could not be downloaded")
        except df.DataFetchError as inner:
            raise RuntimeError("data generation failed") from inner
    except RuntimeError as outer:
        assert failure_reason(outer) == "DATA FAILED: x.h5 could not be downloaded"


def test_a_database_in_place_is_checked_each_time_and_hashed_once(served, monkeypatch):
    data, _ = served
    target = df.fetch(data / "FAKE/plain.h5")
    calls = []
    real = df.file_sha256
    monkeypatch.setattr(df, "file_sha256", lambda p: calls.append(p) or real(p))
    assert df.verify(target)["sha256"] == _sha(b"plain database")
    assert df.verify(target)["published_sha256"] == _sha(b"plain database")
    assert len(calls) == 1 and (data / df.DIGEST_CACHE).is_file()                    # the second check used the cache
    stamp = target.stat().st_mtime_ns
    target.write_bytes(b"plain databasf")                                            # same size, new content
    os.utime(target, ns=(stamp + 10**9, stamp + 10**9))                             # and a new mtime
    with pytest.raises(df.DataFetchError, match="^DATA FAILED: .* the file was changed or replaced"):
        df.verify(target)
    assert df.database_record(target)["mismatch"].startswith("DATA FAILED")         # the record never raises
    assert df.database_record(data / "FAKE/absent.h5")["not_recorded"]


def test_a_database_changed_and_given_back_its_old_mtime_is_hashed_again(served, monkeypatch):
    """The digest cache is keyed by size, mtime, change time and inode: a file edited in place and
    then given its old mtime back (same size) has a new change time, so it is hashed, not trusted."""
    data, _ = served
    target = df.fetch(data / "FAKE/plain.h5")
    assert df.verify(target)["sha256"] == _sha(b"plain database")
    stamp = target.stat().st_mtime_ns
    time.sleep(0.05)                                                                 # past the kernel's timestamp tick
    target.write_bytes(b"plain databasf")                                            # same size, new content
    os.utime(target, ns=(stamp, stamp))                                              # the old mtime restored
    assert target.stat().st_mtime_ns == stamp
    with pytest.raises(df.DataFetchError, match="^DATA FAILED: .* the file was changed or replaced"):
        df.verify(target)


def test_an_unregistered_database_is_recorded_with_its_digest(tmp_path):
    custom = tmp_path / "mine.h5"
    custom.write_bytes(b"my traces")
    assert df.database_record(custom) == {"file": "mine.h5", "sha256": _sha(b"my traces"), "registered": False}


def _master(tmp_path, monkeypatch, value):
    master = tmp_path / "master.yaml"
    master.write_text(f"data_fetch:\n  on_mismatch: {value}\n" if value else "timezone: {}\n")
    monkeypatch.setenv("SCC_MASTER_YAML", str(master))


def test_a_changed_database_is_moved_aside_and_fetched_again_by_default(served, tmp_path, monkeypatch):
    data, _ = served
    _master(tmp_path, monkeypatch, None)                                             # no data_fetch block: refetch
    target = df.fetch(data / "FAKE/plain.h5")
    target.write_bytes(b"a corrupt copy")
    record = df.verify_or_refetch(target)
    assert target.read_bytes() == b"plain database" and record["sha256"] == _sha(b"plain database")
    kept = list((data / "FAKE").glob("plain.h5.mismatch-*"))
    assert len(kept) == 1 and kept[0].read_bytes() == b"a corrupt copy"             # never deleted


def test_on_mismatch_refuse_stops_the_run_and_leaves_the_file(served, tmp_path, monkeypatch):
    data, _ = served
    _master(tmp_path, monkeypatch, "refuse")
    target = df.fetch(data / "FAKE/plain.h5")
    target.write_bytes(b"a corrupt copy")
    with pytest.raises(df.DataFetchError, match="the file was changed or replaced"):
        df.verify_or_refetch(target)
    assert target.read_bytes() == b"a corrupt copy" and not list((data / "FAKE").glob("*.mismatch-*"))


def test_on_mismatch_takes_only_its_two_values(tmp_path, monkeypatch):
    _master(tmp_path, monkeypatch, "delete")
    with pytest.raises(df.DataFetchError, match="refetch, refuse"):
        df.on_mismatch()
