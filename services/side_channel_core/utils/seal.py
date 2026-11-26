# services/side_channel_core/utils/seal.py
"""
Artifact seals: every file TraceFlow writes carries its own Ed25519 signature,
so a change made outside TraceFlow is detected when the file is next read.

Where each format keeps its seal (``tfseal1:<public key>:<signature>``, both
base64; the public key travels with the seal, so any machine checks any file
without exchanging keys):

    YAML     a first line ``# tfseal1:...`` over every byte below it (a comment:
             YAML readers ignore it; a truncated file keeps its seal and fails it)
    JSON     the top-level key ``"_tfseal"``, over the canonical JSON of the
             object without it
    JSONL    ``"_tfseal"`` in every row, chained: a row's signature covers the
             previous row's seal, so a changed, deleted or reordered row breaks
             the chain from that row on
    .npy     a trailer after the array data (np.load ignores it)
    .npz/.pt the zip archive comment, over every member's name and bytes
    .h5      the root attribute ``tfseal``, over every dataset and attribute
    .log     a ``# tfseal1:...`` line, appended when a writer closes, over every
             byte above it; a log grows, so each seal line covers what precedes it

A signature also covers the file's path inside its run folder (or its name,
for a project file), so files cannot be swapped between places. Signing uses
a private key created on first use in ``~/.config/traceflow/seal`` (or
``$TRACEFLOW_SEAL_DIR``), readable only by its owner and mounted read-only into
the training container. Nothing is asked of the user and no key is ever
shared: a run's device.yaml records the key it was signed with, every file of
the run must carry that key, and the key id of published results can be
printed with them for readers to compare.

Reading verifies (``load_json``, ``load_yaml``, ``read_jsonl``, ``load_npy``,
``check``):
    signature matches                 read
    signature does not match          TamperedArtifactError, naming the file
    signed by another key than its
    run's device.yaml                 TamperedArtifactError (a file re-signed)
    no seal                           warning once per folder (written before
                                      seals existed), or TamperedArtifactError
                                      inside a run whose device.yaml declares
                                      seals (the seal was removed)

Author: Shraddha Mishra
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import sys
import threading
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

TAG = "tfseal1"
SEAL_KEY = "_tfseal"
H5_ATTR = "tfseal"
SEAL_RE = re.compile(r"tfseal1:([A-Za-z0-9+/=]{44}):([A-Za-z0-9+/=]{88})")
_SEAL_BYTES = rb"tfseal1:[A-Za-z0-9+/=]{44}:[A-Za-z0-9+/=]{88}"
YAML_HEADER = re.compile(rb"\A# (" + _SEAL_BYTES + rb")\n")
LOG_LINE = re.compile(rb"^# (" + _SEAL_BYTES + rb")$", re.M)
NPY_MARK = b"\n" + TAG.encode() + b":"

logger = logging.getLogger("Seal")


class TamperedArtifactError(RuntimeError):
    """A sealed artifact no longer matches its seal."""


# =============================================================================
# Keys
# =============================================================================
def seal_dir() -> Path:
    return Path(os.environ.get("TRACEFLOW_SEAL_DIR") or Path.home() / ".config" / "traceflow" / "seal")


_lock = threading.Lock()
_signer_cache: dict[str, Any] = {}
_warned: set[str] = set()


def _key_id(public_raw: bytes) -> str:
    return hashlib.sha256(public_raw).hexdigest()[:16]


def _public_raw(private) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _signer():
    """(private key, key id) of this machine, created on first use; None if none can be had."""
    directory = seal_dir()
    with _lock:
        if str(directory) in _signer_cache:
            return _signer_cache[str(directory)]
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        except ImportError:
            _warn_once("nocrypto", "The cryptography package is missing (infra/environment/requirements.txt): "
                                   "files are written without seals")
            _signer_cache[str(directory)] = None
            return None

        key_file = directory / "signing.key"
        result = None
        try:
            if key_file.is_file():
                private = serialization.load_pem_private_key(key_file.read_bytes(), password=None)
            else:
                directory.mkdir(parents=True, exist_ok=True)
                os.chmod(directory, 0o700)
                private = Ed25519PrivateKey.generate()
                pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                            serialization.NoEncryption())
                fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(pem)
                logger.info("Created the artifact signing key %s", key_file)
            result = (private, _key_id(_public_raw(private)))
        except OSError as exc:
            _warn_once("nokey", "No artifact signing key (%s): files are written without seals", exc)
        _signer_cache[str(directory)] = result
        return result


def key_id(public_raw: bytes) -> str:
    """A short, printable name of a public key (the first 16 hex digits of its SHA-256)."""
    return _key_id(public_raw)


def public_key_of(seal: str) -> Optional[bytes]:
    match = SEAL_RE.fullmatch(seal or "")
    return base64.b64decode(match.group(1)) if match else None


def own_key_id() -> Optional[str]:
    """This machine's key id, to print with published results."""
    signer = _signer()
    return signer[1] if signer else None


# =============================================================================
# Signing and checking a digest
# =============================================================================
def _message(kind: str, scope: str, digest: bytes) -> bytes:
    return f"{TAG}|{kind}|{scope}|".encode() + digest


def sign(kind: str, scope: str, digest: bytes) -> Optional[str]:
    signer = _signer()
    if signer is None:
        return None
    private = signer[0]
    signature = private.sign(_message(kind, scope, digest))
    public = base64.b64encode(_public_raw(private)).decode()
    return f"{TAG}:{public}:{base64.b64encode(signature).decode()}"


@dataclass(frozen=True)
class SealStatus:
    state: str          # "ok" | "tampered" | "unsealed" | "unverifiable" | "skipped"
    path: str
    detail: str = ""
    key: str = ""       # key id of the signer


def _check_signature(seal: str, kind: str, scope: str, digest: bytes, path: Path) -> SealStatus:
    match = SEAL_RE.fullmatch(seal or "")
    if not match:
        return SealStatus("tampered", str(path), f"malformed seal {seal!r}")
    public, signature = base64.b64decode(match.group(1)), base64.b64decode(match.group(2))
    signer = _key_id(public)
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return SealStatus("unverifiable", str(path), "the cryptography package is missing", signer)

    try:
        Ed25519PublicKey.from_public_bytes(public).verify(signature, _message(kind, scope, digest))
    except (InvalidSignature, ValueError):
        return SealStatus("tampered", str(path), "the content no longer matches its seal", signer)
    return SealStatus("ok", str(path), "", signer)


# =============================================================================
# Scope: the path a signature is bound to
# =============================================================================
def scope_of(path: str | Path) -> str:
    """The file's path inside its run folder (generated_models/<name>/...), or inside the
    project's graphs/ folder (a figure's sources.json), else its name."""
    parts = Path(path).parts
    if "generated_models" in parts:
        i = len(parts) - 1 - parts[::-1].index("generated_models")
        if len(parts) > i + 2:
            return "/".join(parts[i + 2:])
    if "graphs" in parts:
        i = len(parts) - 1 - parts[::-1].index("graphs")
        return "/".join(parts[i:])
    return parts[-1]


def run_folder_of(path: str | Path) -> Optional[Path]:
    parts = Path(path).parts
    if "generated_models" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("generated_models")
    return Path(*parts[: i + 2]) if len(parts) > i + 2 else None


# =============================================================================
# Digests per format
# =============================================================================
def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True).encode()


def _without_seal(obj: Any) -> Any:
    if isinstance(obj, dict) and SEAL_KEY in obj:
        return {k: v for k, v in obj.items() if k != SEAL_KEY}
    return obj


def kind_of(path: str | Path) -> Optional[str]:
    name = Path(path).name.lower()
    for suffix, kind in ((".yaml", "yaml"), (".yml", "yaml"), (".jsonl", "jsonl"), (".json", "json"),
                         (".npy", "npy"), (".npz", "zip"), (".pt", "zip"), (".pth", "zip"), (".keras", "zip"),
                         (".h5", "h5"), (".hdf5", "h5"), (".log", "log")):
        if name.endswith(suffix):
            return kind
    return None


# ---- JSON -------------------------------------------------------------------
def seal_json_obj(obj: Any, path: str | Path) -> Any:
    """The object with "_tfseal" set (a dict; anything else is returned as is)."""
    if not isinstance(obj, dict):
        return obj
    plain = json.loads(json.dumps(_without_seal(obj), ensure_ascii=False, allow_nan=True))
    seal = sign("json", scope_of(path), hashlib.sha256(canonical_json(plain)).digest())
    out = dict(_without_seal(obj))
    if seal:
        out[SEAL_KEY] = seal
    return out


def _check_json_obj(obj: Any, path: Path) -> SealStatus:
    if not isinstance(obj, dict) or SEAL_KEY not in obj:
        return SealStatus("unsealed", str(path))
    digest = hashlib.sha256(canonical_json(_without_seal(obj))).digest()
    return _check_signature(str(obj[SEAL_KEY]), "json", scope_of(path), digest, path)


# ---- JSONL ------------------------------------------------------------------
def _row_digest(previous_seal: str, row: Any) -> bytes:
    return hashlib.sha256(previous_seal.encode() + b"\n" + canonical_json(_without_seal(row))).digest()


def seal_jsonl_rows(rows: Iterable[Any], path: str | Path) -> list[Any]:
    """Rows (dicts) each with a chained "_tfseal"."""
    scope, previous, out = scope_of(path), "", []
    for row in rows:
        if isinstance(row, dict):
            plain = json.loads(json.dumps(_without_seal(row), ensure_ascii=False, allow_nan=True))
            seal = sign("jsonl", scope, _row_digest(previous, plain))
            row = dict(_without_seal(row))
            if seal:
                row[SEAL_KEY] = seal
                previous = seal
        out.append(row)
    return out


def _check_jsonl_rows(rows: list[Any], path: Path) -> SealStatus:
    scope, previous, sealed, signer = scope_of(path), "", 0, ""
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or SEAL_KEY not in row:
            if sealed:
                return SealStatus("tampered", str(path), f"row {number} lost its seal")
            continue
        status = _check_signature(str(row[SEAL_KEY]), "jsonl", scope, _row_digest(previous, row), path)
        if status.state != "ok":
            return SealStatus(status.state, str(path), f"row {number}: {status.detail}", status.key)
        if sealed and status.key != signer:
            return SealStatus("tampered", str(path), f"row {number} is signed by another key than the rows before it", status.key)
        previous, sealed, signer = str(row[SEAL_KEY]), sealed + 1, status.key
    if sealed and sealed != sum(isinstance(r, dict) for r in rows):
        return SealStatus("tampered", str(path), "rows before the first sealed row lost their seals")
    return SealStatus("ok" if sealed else "unsealed", str(path), "", signer if sealed else "")


# ---- YAML and logs (text with seal lines) -----------------------------------
def seal_yaml_text(text: str, path: str | Path) -> str:
    body = YAML_HEADER.sub(b"", text.encode()).decode()
    if not body.endswith("\n"):
        body += "\n"
    seal = sign("yaml", scope_of(path), hashlib.sha256(body.encode()).digest())
    return f"# {seal}\n" + body if seal else body


def _check_yaml_bytes(data: bytes, path: Path) -> SealStatus:
    match = YAML_HEADER.match(data)
    if not match:
        return SealStatus("unsealed", str(path))
    body = data[match.end():]
    return _check_signature(match.group(1).decode(), "yaml", scope_of(path), hashlib.sha256(body).digest(), path)


def seal_log_file(path: str | Path) -> None:
    """Append a seal line covering every byte of the log written so far."""
    import fcntl

    path = Path(path)
    if not path.is_file():
        return
    with open(path, "rb+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            data = handle.read()
            last = data.rstrip(b"\n").rsplit(b"\n", 1)[-1]
            if not data.strip() or LOG_LINE.fullmatch(last):
                return  # nothing written since the last seal
            if not data.endswith(b"\n"):
                data += b"\n"
                handle.write(b"\n")
            seal = sign("log", scope_of(path), hashlib.sha256(data).digest())
            if seal:
                handle.write(f"# {seal}\n".encode())
                handle.flush()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _check_log_bytes(data: bytes, path: Path) -> SealStatus:
    matches = list(LOG_LINE.finditer(data))
    if not matches:
        return SealStatus("unsealed", str(path))
    signers = set()
    for match in matches:
        status = _check_signature(match.group(1).decode(), "log", scope_of(path),
                                  hashlib.sha256(data[: match.start()]).digest(), path)
        if status.state != "ok":
            return status
        signers.add(status.key)
    if len(signers) > 1:
        return SealStatus("tampered", str(path), f"seal lines by several keys: {sorted(signers)}")
    tail = data[matches[-1].end():].strip()
    return SealStatus("ok", str(path), "unsealed tail: a writer is still open or ended without closing" if tail else "",
                      signers.pop())


# ---- .npy -------------------------------------------------------------------
def seal_npy_file(path: str | Path) -> None:
    path = Path(path)
    data = path.read_bytes()
    cut = data.rfind(NPY_MARK)
    if cut >= 0 and SEAL_RE.fullmatch(data[cut + 1:].decode("ascii", "replace").strip()):
        data = data[:cut]
    seal = sign("npy", scope_of(path), hashlib.sha256(data).digest())
    if seal:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.seal")
        with open(temporary, "wb") as handle:
            handle.write(data + b"\n" + seal.encode())
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)


def _check_npy_bytes(data: bytes, path: Path) -> SealStatus:
    cut = data.rfind(NPY_MARK)
    if cut < 0 or not SEAL_RE.fullmatch(data[cut + 1:].decode("ascii", "replace")):
        return SealStatus("unsealed", str(path))
    return _check_signature(data[cut + 1:].decode(), "npy", scope_of(path), hashlib.sha256(data[:cut]).digest(), path)


# ---- zip (.npz, .pt) ----------------------------------------------------------
def _zip_digest(path: Path) -> bytes:
    total = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            member = hashlib.sha256()
            with archive.open(name) as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    member.update(chunk)
            total.update(name.encode() + b"\0" + member.digest())
    return total.digest()


def archive_digest(path: str | Path) -> str:
    """SHA-256 over a zip archive's members (names and bytes), not its comment: two copies of
    one checkpoint sealed for different paths have the same archive digest."""
    return _zip_digest(Path(path)).hex()


def seal_zip_file(path: str | Path, scope_path: str | Path | None = None) -> None:
    """Seal a zip archive (an .npz, a torch .pt) in its comment; scope_path: the final path."""
    path = Path(path)
    seal = sign("zip", scope_of(scope_path or path), _zip_digest(path))
    if seal:
        with zipfile.ZipFile(path, "a") as archive:
            archive.comment = seal.encode()


def _check_zip(path: Path) -> SealStatus:
    try:
        with zipfile.ZipFile(path) as archive:
            comment = archive.comment.decode("ascii", "replace")
    except zipfile.BadZipFile:
        return SealStatus("skipped", str(path), "not a zip archive")
    if not comment:
        return SealStatus("unsealed", str(path))
    return _check_signature(comment, "zip", scope_of(path), _zip_digest(path), path)


# ---- HDF5 -------------------------------------------------------------------
def _h5_digest(handle) -> bytes:
    import h5py
    import numpy as np

    total = hashlib.sha256()

    def attrs(name, obj):
        for key in sorted(obj.attrs.keys()):
            if name == "" and key == H5_ATTR:
                continue
            value = np.asarray(obj.attrs[key])
            payload = value.tobytes() if value.dtype.kind not in "OUS" else repr(value.tolist()).encode()
            total.update(f"A|{name}|{key}|{value.dtype.str}|{value.shape}|".encode() + payload)

    attrs("", handle)
    names: list[str] = []
    handle.visit(names.append)
    for name in sorted(names):
        obj = handle[name]
        attrs(name, obj)
        if isinstance(obj, h5py.Dataset):
            total.update(f"D|{name}|{obj.dtype.str}|{obj.shape}|".encode())
            if obj.shape == ():
                total.update(np.asarray(obj[()]).tobytes())
            else:
                step = max(1, (64 << 20) // max(1, int(np.prod(obj.shape[1:], dtype=np.int64)) * obj.dtype.itemsize))
                for start in range(0, obj.shape[0], step):
                    total.update(np.ascontiguousarray(obj[start:start + step]).tobytes())
    return total.digest()


def seal_h5_file(path: str | Path, scope_path: str | Path | None = None) -> None:
    import h5py

    with h5py.File(path, "r+") as handle:
        if H5_ATTR in handle.attrs:
            del handle.attrs[H5_ATTR]
        seal = sign("h5", scope_of(scope_path or path), _h5_digest(handle))
        if seal:
            handle.attrs[H5_ATTR] = seal


def _check_h5(path: Path) -> SealStatus:
    import h5py

    with h5py.File(path, "r") as handle:
        seal = handle.attrs.get(H5_ATTR)
        if seal is None:
            return SealStatus("unsealed", str(path))
        seal = seal.decode() if isinstance(seal, bytes) else str(seal)
        return _check_signature(seal, "h5", scope_of(path), _h5_digest(handle), path)


# =============================================================================
# Verification
# =============================================================================
def is_snapshot(path: str | Path, head: bytes) -> bool:
    """A run's params_out snapshot (utils/data_reproducibility.py). It travels: whoever
    reproduces a run trains from the author's snapshot, so any valid key is accepted on it
    and its seal names who wrote it; its snapshot_sha256 still guards its content."""
    return Path(path).name.endswith("_params.yaml") and b"snapshot_sha256" in head[:400]


def verify(path: str | Path) -> SealStatus:
    """The seal state of one file, by its format."""
    path = Path(path)
    kind = kind_of(path)
    if kind is None or not path.is_file():
        return SealStatus("skipped", str(path), "not a sealed format" if kind is None else "missing")
    if kind == "zip":
        return _check_zip(path)
    if kind == "h5":
        return _check_h5(path)
    data = path.read_bytes()
    if kind == "npy":
        return _check_npy_bytes(data, path)
    if kind == "log":
        return _check_log_bytes(data, path)
    if kind == "yaml":
        if is_snapshot(path, data) and not YAML_HEADER.match(data):
            return SealStatus("ok", str(path), "checked by its snapshot_sha256 (written before seals)")
        return _check_yaml_bytes(data, path)
    text = data.decode("utf-8")
    if kind == "json":
        return _check_json_obj(json.loads(text) if text.strip() else None, path)
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return _check_jsonl_rows(rows, path)


def run_declaration(path: str | Path) -> Optional[dict[str, Any]]:
    """{key, since} the run holding path declared in its device.yaml at its start, if any."""
    run = run_folder_of(path)
    if run is None:
        return None
    try:
        import yaml

        record = yaml.safe_load((run / "device.yaml").read_text(encoding="utf-8")) or {}
        seal = (record.get("run_device") or {}).get("artifact_seal") or {}
        if seal.get("version") and seal.get("key"):
            return {"key": str(seal["key"]), "since": float(seal.get("since") or 0)}
    except (OSError, ValueError, AttributeError, KeyError, TypeError):
        pass
    return None


def run_key(path: str | Path) -> Optional[str]:
    declared = run_declaration(path)
    return declared["key"] if declared else None


def seal_declaration() -> Optional[dict[str, Any]]:
    """What device.yaml records about the run's seals: {version, key, since}; None without a key.
    since: the Unix time the run started sealing; an unsealed file older than it is a
    leftover of an earlier attempt of the run, not a removed seal."""
    import time

    key = own_key_id()
    return {"version": 1, "key": key, "since": int(time.time())} if key else None  # floor: never after a file of the run


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


_verified: dict[tuple, SealStatus] = {}
_recorders: list[dict[str, SealStatus]] = []


@contextmanager
def recording():
    """Collect every artifact check() verifies inside the block: {resolved path: SealStatus}.
    A graph job wraps itself in it to record which files its figures were drawn from."""
    reads: dict[str, SealStatus] = {}
    _recorders.append(reads)
    try:
        yield reads
    finally:
        _recorders.remove(reads)


def _record(path: Path, status: SealStatus) -> SealStatus:
    for reads in _recorders:
        reads[str(path.resolve())] = status
    return status


def judge(path: str | Path) -> SealStatus:
    """verify() under the rules of the file's run (its device.yaml declaration); never raises."""
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return SealStatus("skipped", str(path), "missing")
    status = verify(path)  # a file that does not parse raises, as its reader would
    declared = run_declaration(path) if status.state in ("unsealed", "ok") else None
    # A log is sealed when its writer closes; a run killed outright leaves its log unsealed.
    # The file's change time (ctime), not its modification time: mtime is set by whoever edits the
    # file (a backdated mtime hid a removed seal, audit 2026-10-23), ctime by the kernel alone.
    if (status.state == "unsealed" and declared and stat.st_ctime >= declared["since"]
            and path.name != "device.yaml" and kind_of(path) != "log"):
        status = SealStatus("tampered", str(path), "the seal was removed from a file of a sealed run")
    elif (status.state == "ok" and declared and status.key and status.key != declared["key"]
          and not is_snapshot(path, path.read_bytes()[:400])):
        status = SealStatus("tampered", str(path),
                            f"signed by key {status.key}, but its run was sealed by key {declared['key']} (re-signed)", status.key)
    return status


def check(path: str | Path, *, cache: bool = True) -> SealStatus:
    """Verify before use: raise on a broken seal, warn once on a missing or unknown one."""
    path = Path(path)
    if "artifacts" not in path.resolve().parts:
        return SealStatus("skipped", str(path), "not a TraceFlow artifact (an input or a config file)")
    try:
        stat = path.stat()
        token = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError:
        return SealStatus("skipped", str(path), "missing")
    if cache and token in _verified:
        return _record(path, _verified[token])
    status = judge(path)
    if status.state == "tampered":
        raise TamperedArtifactError(
            f"{path}: {status.detail}. The file was changed outside TraceFlow after it was written; "
            "restore it, or delete the run and retrain it."
        )
    if status.state == "unverifiable":
        _warn_once("unverifiable", "%s: seals cannot be checked (%s)", path, status.detail)
    if status.state == "unsealed":
        _warn_once(f"unsealed:{path.parent}", "%s: no seal (written before seals existed); later edits to such files go undetected", path.parent)
    if cache:
        _verified[token] = status
    return _record(path, status)


# =============================================================================
# Readers that verify first
# =============================================================================
def strip(obj: Any) -> Any:
    return _without_seal(obj)


def load_json(path: str | Path) -> Any:
    path = Path(path)
    check(path)
    return _without_seal(json.loads(path.read_text(encoding="utf-8")))


def load_yaml(path: str | Path) -> Any:
    import yaml

    path = Path(path)
    check(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[Any]:
    path = Path(path)
    check(path)
    return [_without_seal(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_npy(path: str | Path, **kwargs: Any):
    import numpy as np

    check(path)
    return np.load(path, **kwargs)


# =============================================================================
# Text writers' entry point
# =============================================================================
def seal_text(path: str | Path, text: str) -> str:
    """The text of a YAML, JSON or JSONL file with its seal, by the target's suffix."""
    kind = kind_of(path)
    if kind == "yaml":
        return seal_yaml_text(text, path)
    if kind == "json":
        if not text.strip():
            return text
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return text
        indent = 2 if "\n" in text.strip() else None
        sealed = seal_json_obj(obj, path)
        return json.dumps(sealed, indent=indent, ensure_ascii=False, allow_nan=True) + ("\n" if text.endswith("\n") else "")
    if kind == "jsonl":
        lines = [line for line in text.splitlines() if line.strip()]
        rows = seal_jsonl_rows([json.loads(line) for line in lines], path)
        out = "\n".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False, allow_nan=True) for row in rows)
        return out + ("\n" if text.endswith("\n") and out else "")
    return text


# =============================================================================
# make verify / make seal
# =============================================================================
def _artifact_files(roots: Iterable[str | Path]) -> Iterator[Path]:
    for root in roots:
        root = Path(root)
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in files:
            if not path.name.startswith(".") and kind_of(path) is not None:
                yield path


def report(roots: Iterable[str | Path]) -> int:
    """make verify: the seal state of every artifact, per run folder; 1 when a file is tampered."""
    groups: dict[str, dict[str, Any]] = {}
    tampered: list[SealStatus] = []
    for path in _artifact_files(roots):
        try:
            status = judge(path)
        except Exception as exc:  # unparsable: changed, or cut short
            status = SealStatus("tampered", str(path), f"unreadable ({type(exc).__name__}: {exc})")
        if status.state == "ok" and path.name == "sources.json":
            changed = _changed_figures(path)   # a figure's provenance lists its files' SHA-256
            if changed:
                status = SealStatus("tampered", str(path), "figure " + "; ".join(changed), status.key)
        group = groups.setdefault(str(run_folder_of(path) or path.parent), {"keys": set()})
        group[status.state] = group.get(status.state, 0) + 1
        if status.key:
            group["keys"].add(status.key)
        if status.state == "tampered":
            tampered.append(status)
    for name, group in groups.items():
        counts = "  ".join(f"{state} {group[state]}" for state in ("ok", "unsealed", "unverifiable", "tampered", "skipped")
                           if group.get(state))
        declared = run_key(Path(name) / "device.yaml")
        keys = ", ".join(sorted(group["keys"])) or "none"
        print(f"{name}\n    {counts}\n    signed by {keys}" + (f" (device.yaml declares {declared})" if declared else ""))
    for status in tampered:
        print(f"TAMPERED {status.path}: {status.detail}")
    print(f"{len(tampered)} tampered file(s)" if tampered else "no tampered file")
    return 1 if tampered else 0


def _changed_figures(sources: Path) -> list[str]:
    """Figures a signed sources.json (services/graph_manager/sources.py) lists whose bytes changed."""
    try:
        listed = json.loads(sources.read_text(encoding="utf-8")).get("outputs") or []
    except (OSError, ValueError, AttributeError):
        return []
    changed = []
    for entry in listed:
        figure = sources.parent / str(entry.get("file"))
        if not figure.is_file():
            changed.append(f"{entry.get('file')} is missing")
            continue
        digest = hashlib.sha256()
        with open(figure, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry.get("sha256"):
            changed.append(f"{entry.get('file')} no longer matches its SHA-256")
    return changed


def _rewrite(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.seal")
    temporary.write_bytes(data)
    os.chmod(temporary, path.stat().st_mode & 0o777)
    os.replace(temporary, path)


def seal_existing(roots: Iterable[str | Path]) -> int:
    """make seal: sign the unsealed files of runs written before seals existed.

    Each run's device.yaml then declares the key, with sealed_after_run: the
    seals vouch for the files as they were found now, not as they were
    written. Checkpoints (.pt, .keras), .npz and .h5 files are left as they
    are: their SHA-256 sits in the checkpoint manifest and the split contracts,
    which a seal in the file would no longer match; sealing those records
    covers them. A file whose seal is broken is reported, never re-signed.
    """
    import yaml

    if _signer() is None:
        print("no signing key: nothing sealed")
        return 2
    sealed, refused, runs = 0, 0, set()
    for path in _artifact_files(roots):
        try:
            status = judge(path)
        except Exception as exc:
            status = SealStatus("tampered", str(path), f"unreadable ({type(exc).__name__}: {exc})")
        if status.state == "tampered":
            print(f"REFUSED {path}: {status.detail}")
            refused += 1
            continue
        kind = kind_of(path)
        if status.state != "unsealed" or path.name == "device.yaml" or kind in ("zip", "h5"):
            continue
        if kind == "npy":
            seal_npy_file(path)
        elif kind == "log":
            seal_log_file(path)
        else:
            _rewrite(path, seal_text(path, path.read_text(encoding="utf-8")).encode("utf-8"))
        sealed += 1
        if run_folder_of(path):
            runs.add(run_folder_of(path))
    for run in sorted(runs):
        device = run / "device.yaml"
        record = (yaml.safe_load(device.read_text(encoding="utf-8")) if device.is_file() else None) or {}
        run_device = record.setdefault("run_device", {})
        if (run_device.get("artifact_seal") or {}).get("key"):
            continue
        declaration = seal_declaration()
        declaration["sealed_after_run"] = True
        run_device["artifact_seal"] = declaration
        text = seal_text(device, yaml.safe_dump(record, sort_keys=False)).encode("utf-8")
        if device.is_file():
            _rewrite(device, text)
        else:
            device.write_bytes(text)
            os.chmod(device, 0o444)
    print(f"sealed {sealed} file(s) in {len(runs)} run(s); key {own_key_id()}" + (f"; {refused} refused" if refused else ""))
    return 1 if refused else 0


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m utils.seal", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("verify", "seal", "key"))
    parser.add_argument("paths", nargs="*", help="files or folders (an artifacts/ folder, a run folder)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if args.command == "key":
        key = own_key_id()
        print(key or f"none: no signing key in {seal_dir()}")
        return 0 if key else 1
    if not args.paths:
        parser.error("name at least one file or folder")
    missing = [p for p in args.paths if not Path(p).exists()]
    if missing:
        print("not found: " + ", ".join(missing))
        return 2
    return report(args.paths) if args.command == "verify" else seal_existing(args.paths)


__all__ = [
    "SEAL_KEY", "TamperedArtifactError", "SealStatus", "check", "verify", "load_json", "load_yaml",
    "read_jsonl", "load_npy", "seal_text", "seal_json_obj", "seal_jsonl_rows", "seal_yaml_text",
    "seal_npy_file", "seal_zip_file", "seal_h5_file", "seal_log_file", "strip", "own_key_id",
    "seal_declaration", "run_key", "scope_of", "seal_dir", "judge", "archive_digest", "recording", "report", "seal_existing",
]



def _one_module() -> None:
    """One module under both import paths the code uses (utils.seal and
    services.side_channel_core.utils.seal): one TamperedArtifactError, one key cache, one
    read recorder. Two copies would let an `except` against one miss the other's error."""
    import importlib

    me = sys.modules[__name__]
    for name in ("utils.seal", "services.side_channel_core.utils.seal"):
        if name == __name__ or name in sys.modules:
            continue
        parent = name.rpartition(".")[0]
        sys.modules[name] = me          # before the parent import: a module it imports gets this one
        try:
            setattr(importlib.import_module(parent), "seal", me)
        except ImportError:             # that path is not importable here
            del sys.modules[name]


if __name__ != "__main__":
    _one_module()

if __name__ == "__main__":
    raise SystemExit(main())
