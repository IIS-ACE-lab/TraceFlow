# services/side_channel_core/utils/dataset_fetch.py
"""
Fetch a database the configuration names but this checkout does not hold yet.

A registered database (src/sca/dataset_types.py SourceFile) that is missing is
fetched once, on first use, into its place under services/side_channel_core/data/:

  1. its git submodule, when the checkout has one for it and its server is
     reachable (the owner's machines: ../<NAME>-data.git next to origin);
  2. otherwise its official download (a zip member or the file itself), the
     only route for a clone from GitHub, whose relative submodule URLs do not
     resolve.

Either way the file must match the SHA-256 its authors publish before it is
put in place (atomically, so a cut download leaves nothing behind). Nothing
falls back silently: a database that cannot be fetched or does not match
raises DataFetchError, whose message starts with "DATA FAILED".

A database already in place is checked again each time a run opens it
(verify): a downloaded file against its published SHA-256, a converted file
against the sources and converter version the registry names. One that no longer
matches is moved aside as <name>.mismatch-<SHA-256 prefix> and put in place
again (master.yaml data_fetch.on_mismatch: refetch, the default), or the run
stops with DATA FAILED (refuse). Its digest is
what the run records in device.yaml (database_record). Digests are cached in
services/side_channel_core/data/.traceflow_digests.json by file size and
modification time, so a multi-GB file is hashed once.

Author: Shraddha Mishra
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Optional

from utils.logger import get_logger
from services.side_channel_core.src.sca.dataset_types import TYPES, SourceFile, conversion_for, source_file

logger = get_logger("dataset_fetch")

DATA_FAILED = "DATA FAILED"
_CHUNK = 1 << 20


class DataFetchError(RuntimeError):
    """A registered database could not be fetched, or does not match its published SHA-256."""

    states_its_reason = True   # its "DATA FAILED: ..." message is the run's status_reason (utils/device.py)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root(start: Path) -> Optional[Path]:
    """The checkout holding `start` (a folder with .gitmodules and .git), if any."""
    for folder in (start, *start.parents):
        if (folder / ".gitmodules").is_file() and (folder / ".git").exists():
            return folder
    return None


def _submodule_for(target: Path) -> Optional[tuple[Path, str]]:
    """(checkout, submodule path) of the submodule whose folder holds target, if the checkout has one."""
    root = _repo_root(target.parent)
    if root is None:
        return None
    try:
        listing = subprocess.run(
            ["git", "-C", str(root), "config", "-f", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    relative = target.resolve().relative_to(root.resolve()).as_posix()
    for line in listing.splitlines():
        sub = line.split(" ", 1)[1].strip() if " " in line else ""
        if sub and (relative == sub or relative.startswith(sub.rstrip("/") + "/")):
            return root, sub
    return None


def _from_submodule(target: Path) -> bool:
    found = _submodule_for(target)
    if found is None:
        return False
    root, sub = found
    logger.info("Fetching %s: git submodule update --init %s", target.name, sub)
    try:
        # The owner's data repositories are local paths (/srv/git/<NAME>-data.git): git refuses the
        # file transport for submodules by default (2.38+); these URLs come from our own .gitmodules.
        done = subprocess.run(["git", "-c", "protocol.file.allow=always", "-C", str(root), "submodule", "update", "--init", sub],
                              capture_output=True, text=True, timeout=6 * 3600, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Submodule %s not fetched (%s); trying the official download", sub, exc)
        return False
    if done.returncode != 0:
        logger.warning("Submodule %s not fetched (%s); trying the official download",
                       sub, (done.stderr or done.stdout).strip().splitlines()[-1:] or "no reason given")
        return False
    return target.is_file()


def _download(url: str, destination: Path) -> str:
    """Stream url to destination; return its SHA-256."""
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "TraceFlow dataset fetch"})
    with urllib.request.urlopen(request, timeout=120) as response, open(destination, "wb") as out:
        for chunk in iter(lambda: response.read(_CHUNK), b""):
            out.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def _from_download(source: SourceFile, target: Path) -> None:
    partial = target.with_name(f".{target.name}.part")
    archive = target.with_name(f".{target.name}.zip.part")
    try:
        if source.member is None:
            logger.info("Downloading %s from %s", target.name, source.url)
            _download(source.url, partial)
        else:
            logger.info("Downloading %s (for %s) from %s", Path(source.member).parts[0], target.name, source.url)
            got = _download(source.url, archive)
            if source.archive_sha256 and got != source.archive_sha256:
                raise DataFetchError(f"{DATA_FAILED}: {source.url} has SHA-256 {got}, "
                                     f"its authors publish {source.archive_sha256}")
            with zipfile.ZipFile(archive) as zipped:
                with zipped.open(source.member) as member, open(partial, "wb") as out:
                    shutil.copyfileobj(member, out, _CHUNK)
                _extract_siblings(zipped, source, target)
        got = file_sha256(partial)
        if got != source.sha256:
            raise DataFetchError(f"{DATA_FAILED}: the downloaded {target.name} has SHA-256 {got}, "
                                 f"its authors publish {source.sha256}")
        os.chmod(partial, 0o644)
        os.replace(partial, target)
    except DataFetchError:
        raise
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise DataFetchError(f"{DATA_FAILED}: {target.name} could not be downloaded from {source.url} "
                             f"({type(exc).__name__}: {exc})") from exc
    finally:
        partial.unlink(missing_ok=True)
        archive.unlink(missing_ok=True)


def _extract_siblings(zipped: zipfile.ZipFile, source: SourceFile, target: Path) -> None:
    """The other registered files of the same archive that are missing: taken now, checked, so the
    archive is downloaded once (ASCADf's four databases share one 4.2 GB zip)."""
    data_root = Path(str(target)[: len(str(target)) - len(source.path)])
    for dtype in TYPES.values():
        for other in dtype.sources:
            if other is source or other.url != source.url or other.member is None:
                continue
            path = data_root / other.path
            if path.is_file() or other.member not in zipped.namelist():
                continue
            partial = path.with_name(f".{path.name}.part")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(other.member) as member, open(partial, "wb") as out:
                    shutil.copyfileobj(member, out, _CHUNK)
                if file_sha256(partial) == other.sha256:
                    os.chmod(partial, 0o644)
                    os.replace(partial, path)
                    logger.info("Also in place from the same archive: %s", path)
                else:
                    logger.warning("%s from the archive does not match its published SHA-256; not kept", path.name)
            finally:
                partial.unlink(missing_ok=True)


def fetch(target: Path) -> Path:
    """Put the registered database `target` in place (submodule, else official download), checked."""
    target = Path(target)
    source = source_file(target)
    if source is None:
        raise DataFetchError(f"{DATA_FAILED}: {target} is not a registered database (src/sca/dataset_types.py)")
    # The folder is made only for a download: made before the submodule's first clone, it would sit
    # inside the empty submodule folder (source/, ASCAD_databases/) and git refuses to clone into it.
    if not _from_submodule(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        _from_download(source, target)
    got = file_sha256(target)
    if got != source.sha256:
        raise DataFetchError(f"{DATA_FAILED}: {target} has SHA-256 {got}, its authors publish {source.sha256}")
    logger.info("Database %s in place, SHA-256 checked", target)
    return target


def ensure(target: Path) -> Path:
    """Put a registered database in place: fetch a download (fetch), or make a converted one from
    its downloads (each fetched and checked), recording the inputs' SHA-256 and the converter
    version in the output. The output is published atomically."""
    target = Path(target)
    conversion = conversion_for(target)
    if conversion is None:
        return fetch(target)
    import importlib
    import json

    import h5py

    data_root = Path(str(target.resolve())[: len(str(target.resolve())) - len(conversion.output)])
    inputs = {}
    for source in conversion.inputs:
        path = data_root / source.path
        if path.is_file() and file_sha256(path) != source.sha256:
            problem = f"{DATA_FAILED}: {path} has SHA-256 {file_sha256(path)}, its authors publish {source.sha256}"
            if on_mismatch() == "refuse":
                raise DataFetchError(problem)
            logger.warning("%s. Moved it to %s and fetching it again (master.yaml data_fetch.on_mismatch: "
                           "refetch)", problem, _move_aside(path))
        if not path.is_file():
            fetch(path)
        inputs[Path(source.path).name] = path
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.part")
    try:
        module = importlib.import_module(f"sca.converters.{conversion.converter}")
        logger.info("Converting %s into %s (sca/converters/%s.py)", ", ".join(inputs), target, conversion.converter)
        module.convert(inputs, partial)
        with h5py.File(partial, "r+") as handle:
            handle.attrs["traceflow_converter"] = conversion.converter
            handle.attrs["traceflow_converter_version"] = int(conversion.version)
            handle.attrs["traceflow_source_sha256"] = json.dumps({s.path: s.sha256 for s in conversion.inputs})
        os.chmod(partial, 0o644)
        os.replace(partial, target)
    except DataFetchError:
        raise
    except Exception as exc:
        raise DataFetchError(f"{DATA_FAILED}: converting {', '.join(inputs)} into {target.name} failed "
                             f"({type(exc).__name__}: {exc})") from exc
    finally:
        partial.unlink(missing_ok=True)
    logger.info("Database %s made from its checked downloads", target)
    return target


DIGEST_CACHE = ".traceflow_digests.json"
_DATA_ROOT = "services/side_channel_core/data"


def _cache_dir(path: Path) -> Path:
    """The data root holding path (host and container share it), else the file's own folder."""
    text = path.as_posix()
    marker = "/" + _DATA_ROOT + "/"
    if marker in text:
        return Path(text[: text.index(marker) + len(marker) - 1])
    return path.parent


def database_digest(path: Path) -> str:
    """SHA-256 of a database file, cached by (size, mtime_ns, ctime_ns, inode); a cache that cannot be written
    is skipped. The change time and inode are the kernel's: a file edited and given back its old mtime is
    hashed again (audit 2026-10-23)."""
    import json

    path = Path(path).resolve()
    stat = path.stat()
    folder = _cache_dir(path)
    cache_file = folder / DIGEST_CACHE
    key = path.relative_to(folder).as_posix()
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.is_file() else {}
    except (OSError, ValueError):
        cache = {}
    hit = cache.get(key) if isinstance(cache, dict) else None
    if (isinstance(hit, dict) and hit.get("size") == stat.st_size and hit.get("mtime_ns") == stat.st_mtime_ns
            and hit.get("ctime_ns") == stat.st_ctime_ns and hit.get("ino") == stat.st_ino):
        return str(hit["sha256"])
    logger.info("Hashing %s (%.1f GB, once; then cached)", path.name, stat.st_size / 1e9)
    digest = file_sha256(path)
    cache = cache if isinstance(cache, dict) else {}
    cache[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
                  "ino": stat.st_ino, "sha256": digest}
    try:
        partial = cache_file.with_name(f".{cache_file.name}.{os.getpid()}.part")
        partial.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
        os.chmod(partial, 0o666)          # host user and container root both update it
        os.replace(partial, cache_file)
    except OSError as exc:
        logger.warning("Digest cache %s not written (%s); %s is hashed again next time", cache_file, exc, path.name)
    return digest


_MASTER_YAML = Path(__file__).resolve().parents[1] / "config" / "master.yaml"
ON_MISMATCH = ("refetch", "refuse")


def on_mismatch() -> str:
    """master.yaml data_fetch.on_mismatch: what a run does with a registered database in place that
    no longer matches (refetch when the block is absent)."""
    import yaml

    path = Path(os.environ.get("SCC_MASTER_YAML") or _MASTER_YAML)
    block = {}
    if path.is_file():
        with open(path, "r", encoding="utf-8") as handle:
            block = (yaml.safe_load(handle) or {}).get("data_fetch") or {}
    value = str(block.get("on_mismatch", "refetch")).strip().lower()
    if value not in ON_MISMATCH:
        raise DataFetchError(f"{DATA_FAILED}: master.yaml data_fetch.on_mismatch is {value!r}; "
                             f"set one of {', '.join(ON_MISMATCH)}")
    return value


def _move_aside(path: Path) -> Path:
    """Keep a mismatching file next to its place as <name>.mismatch-<SHA-256 prefix>; never deleted."""
    aside = path.with_name(f"{path.name}.mismatch-{file_sha256(path)[:8]}")
    counter = 1
    while aside.exists():
        aside = path.with_name(f"{path.name}.mismatch-{file_sha256(path)[:8]}-{counter}")
        counter += 1
    os.replace(path, aside)
    return aside


def verify_or_refetch(target: Path) -> dict[str, Any]:
    """verify(target); on a mismatch under data_fetch.on_mismatch: refetch, move the file aside, put
    the database in place again from its official sources (ensure) and verify that."""
    target = Path(target)
    try:
        return verify(target)
    except DataFetchError as exc:
        if on_mismatch() == "refuse":
            raise
        aside = _move_aside(target)
        logger.warning("%s. Moved it to %s and fetching it again (master.yaml data_fetch.on_mismatch: "
                       "refetch)", exc, aside)
    ensure(target)
    return verify(target)


def verify(target: Path) -> dict[str, Any]:
    """Check a registered database in place and return what the run records about it:
    a download against its published SHA-256, a converted file against the sources and
    converter version of the registry. A mismatch raises DataFetchError."""
    import json

    target = Path(target)
    record: dict[str, Any] = {"file": target.name, "sha256": database_digest(target)}
    source = source_file(target)
    if source is not None:
        record["published_sha256"] = source.sha256
        if record["sha256"] != source.sha256:
            raise DataFetchError(
                f"{DATA_FAILED}: {target} has SHA-256 {record['sha256']}, its authors publish "
                f"{source.sha256}: the file was changed or replaced; delete it and the run fetches it again"
            )
        return record
    conversion = conversion_for(target)
    if conversion is None:
        return record
    import h5py

    expected = {s.path: s.sha256 for s in conversion.inputs}
    with h5py.File(target, "r") as handle:
        converter = str(handle.attrs.get("traceflow_converter", ""))
        version = int(handle.attrs.get("traceflow_converter_version", -1))
        recorded = json.loads(handle.attrs.get("traceflow_source_sha256", "{}"))
    record.update(converter=converter, converter_version=version, source_sha256=recorded)
    if (converter, version, recorded) != (conversion.converter, int(conversion.version), expected):
        raise DataFetchError(
            f"{DATA_FAILED}: {target} was not made by sca/converters/{conversion.converter}.py "
            f"v{conversion.version} from the registered sources (it records {converter or 'no converter'} "
            f"v{version}); delete it and the run makes it again"
        )
    return record


def database_record(path: Path) -> dict[str, Any]:
    """What device.yaml records about the run's database; never raises (a mismatch is recorded)."""
    path = Path(path)
    if not path.is_file():
        return {"file": path.name, "not_recorded": "not here when the record was written"}
    try:
        if source_file(path) is not None or conversion_for(path) is not None:
            return {**verify(path), "registered": True}
        return {"file": path.name, "sha256": database_digest(path), "registered": False}
    except DataFetchError as exc:
        return {"file": path.name, "registered": True, "mismatch": str(exc)}
    except Exception as exc:
        return {"file": path.name, "not_recorded": f"{type(exc).__name__}: {exc}"}


def needed_files(database: Path) -> list[Path]:
    """The registered files a run of this database reads first: the file, or a conversion's inputs."""
    database = Path(database)
    conversion = conversion_for(database)
    if conversion is None:
        return [database] if source_file(database) is not None else []
    if database.is_file():
        return []
    root = Path(str(database)[: len(str(database)) - len(conversion.output)])
    return [root / s.path for s in conversion.inputs]


def prefetch_submodules(project: str, dataset: str = "") -> int:
    """Host side, before the container starts (Makefile dataset-submodule): put the submodule of each
    missing registered database the selected entries need in place. The container mounts the data
    folder but not the checkout's .git or the owner's server, so it can only download. Returns the
    number of files that are still missing (the run then downloads them, checked)."""
    from utils.config_loader import get_project_config

    cfg = get_project_config(project_name=project, validate_data=False)
    entries = cfg.get("data") or {}
    selected = [entries[dataset]] if dataset and dataset in entries else list(entries.values())
    missing = 0
    for entry in selected:
        database = Path(str(entry.get("database", "")))
        for path in needed_files(database.resolve()):
            if path.is_file():
                continue
            if _from_submodule(path):
                logger.info("Database file in place from its submodule: %s", path)
            else:
                logger.info("%s: no reachable submodule; the run downloads it from its official source", path.name)
                missing += 1
    return missing


__all__ = ["DATA_FAILED", "DataFetchError", "database_digest", "database_record", "ensure", "fetch",
           "file_sha256", "needed_files", "prefetch_submodules", "verify", "verify_or_refetch", "on_mismatch"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Put the submodules of a project's missing databases in place (host).")
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", default="")
    arguments = parser.parse_args()
    prefetch_submodules(arguments.project, arguments.dataset)
