# services/side_channel_core/utils/device_record.py
"""
The device a run trained on, and a timing probe of that device.

Absolute training time depends on the hardware; ratios between runs of one
device do not (graph job `relative:`). Every run therefore writes
device.yaml at the start of training into the folder paths.device names
(default: the run folder itself, beside config/, which is what a reproduction
ships, and not inside it), under one key `run_device`:

    run_device:
      kind, name, capability, memory_gb   the GPU (or "cpu")
      software                            python, torch, CUDA, cuDNN, Keras, framework
      code                                the commit of the code that ran and whether
                                          services/ differed from it (code_version)
      precision                           trainer.precision and the TF32 switches in force
      probe                               seconds per step of a fixed workload
      recorded_at

The probe trains a fixed network (PROBE_SPEC: two Conv1d blocks and two dense
layers, RMSprop, cross-entropy, synthetic traces of the ASCAD shape) for a few
steps on the run's device under the run's precision and records the median
seconds per step. It depends on no setting of the run, so dividing a run's
time by its own probe expresses it in probe steps, comparable across devices
(graph job relative: {rule: probe}). The probe runs inside
torch.random.fork_rng, so the run's random state is unchanged, and a failed
probe is recorded as failed without stopping the training.

Author: Shraddha Mishra
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import yaml

from utils.timestamps import now_iso
from utils.seal import check as check_seal, seal_declaration, seal_text

DEVICE_RECORD_FILE = "device.yaml"
RECORD_KEY = "run_device"
# paths.device when a configuration does not set it.
DEFAULT_DEVICE_DIR = "artifacts/{{env}}/generated_models/{{model_name}}"
# device.yaml is read-only for everyone (r--r--r--), like a sealed fact of the run.
DEVICE_RECORD_MODE = 0o444

# Change the version whenever the workload changes: probes of two versions
# are not comparable.
PROBE_SPEC = {
    "version": 1,
    "workload": (
        "Conv1d(1, 64, 11, same) ReLU AvgPool1d(2), Conv1d(64, 128, 11, same) ReLU "
        "AvgPool1d(2), Flatten, Linear(22400, 512) ReLU, Linear(512, 256); "
        "RMSprop(lr 1e-5), CrossEntropyLoss; synthetic batch of 200 traces x 700 samples"
    ),
    "batch": 200,
    "samples": 700,
    "warmup_steps": 3,
    "steps": 10,
    "repeats": 3,
}

# Fields that must agree for two runs' times to be compared directly.
SAME_DEVICE_FIELDS = (
    ("kind",),
    ("name",),
    ("software", "torch"),
    ("software", "cuda"),
    ("software", "cudnn"),
    ("precision", "cudnn_allow_tf32"),
    ("precision", "matmul_allow_tf32"),
)


def _resolve_device(cfg: Mapping[str, Any]) -> torch.device:
    """cfg['device'] as training uses it: a CUDA device that is not available raises
    GPUFailedError (utils/device.py), never a record of the CPU."""
    from utils.device import require_device

    return require_device(cfg)


def image_version() -> dict[str, Any]:
    """
    The training image the run ran in: its tag, its content id (sha256 of the image
    configuration, which names the image exactly even when it was never pushed) and its
    registry digests when it was pushed. The Makefile reads them on the host (nerdctl image
    inspect) and passes TRACEFLOW_IMAGE, TRACEFLOW_IMAGE_ID and TRACEFLOW_IMAGE_DIGESTS.
    """
    import json

    tag = (os.environ.get("TRACEFLOW_IMAGE") or "").strip()
    if not tag:
        return {"tag": "not recorded: run outside the training container (TRACEFLOW_IMAGE unset)"}
    image: dict[str, Any] = {"tag": tag, "id": (os.environ.get("TRACEFLOW_IMAGE_ID") or "").strip() or "not recorded"}
    try:
        digests = json.loads(os.environ.get("TRACEFLOW_IMAGE_DIGESTS") or "[]") or []
    except ValueError:
        digests = []
    if digests:
        image["repo_digests"] = [str(d) for d in digests]
    return image


def code_version() -> dict[str, Any]:
    """
    The code that ran: the full commit of HEAD, whether services/ differed from
    it (tracked edits or new files) and a short SHA-256 of that difference, so
    two runs from the same uncommitted state can be told apart from others.

    The training container mounts services/side_channel_core without .git, so
    the Makefile reads these on the host and passes TRACEFLOW_COMMIT,
    TRACEFLOW_CODE_MODIFIED and TRACEFLOW_CODE_DIFF into the container. A run
    started without them asks git directly, and otherwise says why there is
    no commit.
    """
    commit = os.environ.get("TRACEFLOW_COMMIT", "").strip()
    if commit:
        modified = os.environ.get("TRACEFLOW_CODE_MODIFIED", "").strip() not in ("", "0")
        return {
            "commit": commit,
            "services_modified": modified,
            "services_diff_sha256": os.environ.get("TRACEFLOW_CODE_DIFF", "").strip() if modified else "none",
            "source": "the host checkout, read by the Makefile",
        }
    here = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(here), *args], capture_output=True, text=True,
                              timeout=20, check=True).stdout
    try:
        top = git("rev-parse", "--show-toplevel").strip()
        commit = git("rev-parse", "HEAD").strip()
        status = subprocess.run(["git", "-C", top, "status", "--porcelain", "--", "services"],
                                capture_output=True, text=True, timeout=20, check=True).stdout
        modified = bool(status.strip())
        diff = ""
        if modified:
            import hashlib
            raw = subprocess.run(["git", "-C", top, "diff", "HEAD", "--no-ext-diff", "--", "services"],
                                 capture_output=True, timeout=60, check=True).stdout
            diff = hashlib.sha256(raw + status.encode()).hexdigest()[:16]
        return {"commit": commit, "services_modified": modified,
                "services_diff_sha256": diff if modified else "none", "source": "git in this process"}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "not recorded: TRACEFLOW_COMMIT unset (run started without make) and no git repository here"}


def collect_device_info(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The hardware, library versions and arithmetic settings of this process."""
    device = _resolve_device(cfg)
    info: dict[str, Any] = {"kind": device.type}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info.update(
            name=torch.cuda.get_device_name(index),
            capability=f"{props.major}.{props.minor}",
            memory_gb=round(props.total_memory / 1024**3, 1),
            visible_devices=torch.cuda.device_count(),
        )
    else:
        info["name"] = platform.processor() or platform.machine() or "cpu"
    framework = str(((cfg.get("deep_learning") or {}).get("framework")) or "pytorch")
    software: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda or "not built with CUDA"),
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "not available",
        "framework": framework,
    }
    if framework == "keras":
        import keras

        software["keras"] = str(keras.__version__)
    info["software"] = software
    info["code"] = code_version()
    info["image"] = image_version()
    info["database"] = database_of(cfg)
    info["precision"] = {
        "setting": str(((cfg.get("trainer") or {}).get("precision")) or "default"),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    return info


def database_of(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The run's database file and its SHA-256 (utils/dataset_fetch.database_record); a database
    fetched later in the run is recorded then (record_database)."""
    data = cfg.get("data") if isinstance(cfg, Mapping) else None
    database = data.get("database") if isinstance(data, Mapping) else None
    if not database:
        return {"not_recorded": "no data entry selected"}
    from utils.dataset_fetch import database_record
    from utils.path_utils import resolve_path

    path = Path(str(database))
    if not path.is_absolute():
        path = Path(resolve_path(str(database), dict(cfg)))
    return database_record(path)


def run_device_probe(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Median seconds per training step of PROBE_SPEC on the run's device."""
    device = _resolve_device(cfg)
    spec = PROBE_SPEC
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def timed() -> list[float]:
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(spec["batch"], 1, spec["samples"], generator=generator).to(device)
        y = torch.randint(0, 256, (spec["batch"],), generator=generator).to(device)
        net = torch.nn.Sequential(
            torch.nn.Conv1d(1, 64, 11, padding="same"), torch.nn.ReLU(), torch.nn.AvgPool1d(2),
            torch.nn.Conv1d(64, 128, 11, padding="same"), torch.nn.ReLU(), torch.nn.AvgPool1d(2),
            torch.nn.Flatten(), torch.nn.Linear(128 * (spec["samples"] // 4), 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
        ).to(device)
        optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-5)
        loss_fn = torch.nn.CrossEntropyLoss()

        def step() -> None:
            optimizer.zero_grad(set_to_none=True)
            loss_fn(net(x), y).backward()
            optimizer.step()

        for _ in range(spec["warmup_steps"]):
            step()
        sync()
        seconds = []
        for _ in range(spec["repeats"]):
            start = time.perf_counter()
            for _ in range(spec["steps"]):
                step()
            sync()
            seconds.append((time.perf_counter() - start) / spec["steps"])
        return seconds

    # The workload's tensors are freed when timed() returns.
    with torch.random.fork_rng(devices=devices):
        per_step = timed()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        **spec,
        "seconds_per_step": round(statistics.median(per_step), 6),
        "repeats_seconds_per_step": [round(s, 6) for s in per_step],
    }


def device_record_dir(cfg: Mapping[str, Any]) -> Path:
    """The folder of device.yaml: paths.device of the run, rendered like every path."""
    from utils.path_utils import resolve_path

    template = ((cfg.get("paths") or {}).get("device")) or DEFAULT_DEVICE_DIR
    return Path(resolve_path(template, dict(cfg), inject=True))


def write_device_record(cfg: Mapping[str, Any], run_dir: str | Path | None = None, logger=None) -> Path:
    """
    Record the device and its probe in device.yaml (atomic write), in run_dir
    when given, otherwise in the folder paths.device names.
    """
    if run_dir is None:
        run_dir = device_record_dir(cfg)
    record = collect_device_info(cfg)
    try:
        record["probe"] = run_device_probe(cfg)
    except Exception as exc:  # the probe never stops a training
        record["probe"] = {"version": PROBE_SPEC["version"], "failed": f"{type(exc).__name__}: {exc}"}
        if logger is not None:
            logger.warning("Device probe failed (%s); the run trains without one", exc)
    record["recorded_at"] = now_iso()
    # The key every file of this run is signed with (utils/seal.py): a file of
    # the run without a seal, or signed by another key, is then a tampered one.
    declaration = seal_declaration()
    if declaration:
        record["artifact_seal"] = declaration
    path = Path(run_dir) / DEVICE_RECORD_FILE
    _write_record(path, record)
    if logger is not None:
        probe = record["probe"].get("seconds_per_step", "failed")
        logger.info("Device %s (%s); probe v%s: %s s/step -> %s",
                    record.get("name"), record.get("kind"), PROBE_SPEC["version"], probe, path)
    return path


def record_database(cfg: Mapping[str, Any], database: Mapping[str, Any]) -> None:
    """Put the database the run opened (checked, sca/ascad_data_loader.check_file_exists) into
    the run's device.yaml, when the run has one; a command without a run folder records nothing."""
    try:
        path = device_record_dir(cfg) / DEVICE_RECORD_FILE
    except Exception:
        return
    if not path.is_file():
        return
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        record = dict(loaded.get(RECORD_KEY) or {})
        if record.get("database") == dict(database):
            return
        record["database"] = dict(database)
        _write_record(path, record)
    except Exception as exc:   # as write_device_record: the record never stops a training
        from utils.logger import get_logger

        get_logger("device_record").warning("Database not recorded in %s: %s", path, exc)


def _write_record(path: Path, record: Mapping[str, Any]) -> None:
    """device.yaml written atomically and sealed (utils/seal.py)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".device.", suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(seal_text(path, yaml.safe_dump({RECORD_KEY: record}, sort_keys=False)))
        # mkstemp creates the file 0600: written by the container's root, the
        # host user could not read it. The record is a fact of the run: read-only
        # for everyone (a retry still replaces it, which needs only the folder).
        os.chmod(tmp, DEVICE_RECORD_MODE)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def locate_device_record(run_dir: str | Path) -> Path:
    """
    Where a run's device.yaml is: paths.device of the run's own snapshot
    (<run>/config/<name>_params.yaml) rendered for that run, relative to its
    project (<project>/artifacts/<env>/generated_models/<name>), or the
    default, the run folder itself.
    """
    run_dir = Path(run_dir)
    name = run_dir.name
    default = run_dir / DEVICE_RECORD_FILE
    snapshot = run_dir / "config" / f"{name}_params.yaml"
    try:
        data = yaml.safe_load(snapshot.read_text(encoding="utf-8")) if snapshot.is_file() else None
    except (OSError, yaml.YAMLError):
        data = None
    template = ((data or {}).get("paths") or {}).get("device") if isinstance(data, Mapping) else None
    if not template or len(run_dir.parents) < 4:
        return default
    env = str(data.get("env") or run_dir.parent.parent.name)
    rendered = str(template)
    for key, value in (("env", env), ("model_name", name)):
        rendered = rendered.replace("{{" + key + "}}", value).replace("{" + key + "}", value)
    folder = Path(rendered)
    if not folder.is_absolute():
        folder = run_dir.parents[3] / folder          # <project>/<template>
    return folder / DEVICE_RECORD_FILE


def read_device_record(run_dir: str | Path) -> Optional[dict[str, Any]]:
    """The run_device mapping of a run's device.yaml (locate_device_record), or None."""
    path = locate_device_record(run_dir)
    if not path.is_file():
        return None
    check_seal(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    record = data.get(RECORD_KEY) if isinstance(data, Mapping) else None
    return dict(record) if isinstance(record, Mapping) else None


def device_fingerprint(record: Optional[Mapping[str, Any]]) -> Optional[tuple]:
    """The SAME_DEVICE_FIELDS of a record, or None when there is no record."""
    if not record:
        return None
    values = []
    for path in SAME_DEVICE_FIELDS:
        node: Any = record
        for key in path:
            node = node.get(key) if isinstance(node, Mapping) else None
        values.append(node)
    return tuple(values)


def probe_seconds(record: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Seconds per probe step of a record (current PROBE_SPEC version), or None."""
    probe = (record or {}).get("probe") or {}
    if probe.get("version") != PROBE_SPEC["version"] or "seconds_per_step" not in probe:
        return None
    value = float(probe["seconds_per_step"])
    return value if value > 0 else None
