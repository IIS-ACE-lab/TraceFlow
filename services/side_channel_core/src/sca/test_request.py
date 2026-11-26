# services/side_channel_core/src/sca/test_request.py
"""
test_request.py – re-test trained models by name (``make test``).

Called by ``core/cli.py test-cmd --test-request``. The project's
``config/test_config.yaml`` lists only the model names to test again:

    test_request:
      model:
        - model_a
        - model_b

Each name is looked up in ``artifacts/model_name_map.json`` (hash -> name)
to get its model hash, and the hash in ``artifacts/run_map.jsonl`` to get
the trained run (env, path, status). The model is then tested through the
orchestrator's own test path (``utils/sweep_test_runner.run_test_sweep``)
from its resolved training snapshot
``generated_models/<model>/config/<model>_params.yaml``, so it is tested on
exactly the parameters it was trained with. ``params.yaml`` is not read.

Both the test and its key recovery are fresh: the model is run on the test
traces again, test.json is rewritten, and key enumeration (if the model
uses it) is applied to that new ranking with the strategy fixed at
training (``key_recovery_helper.run_test_key_recovery_summary``). The
result replaces the model's recorded test: its test metric row and
``artifacts/test_registry.json``. Training/validation key-recovery
results are neither rewritten nor reported.

External tests (``make test EXTERNAL=1``, the panel's TESTS sub-panel)
attack the same trained model with the attack settings of
``test_config.yaml`` instead of the snapshot's:

    test_request:
      model: [model_a]
      perform_attack:                    # every model's external test
        test: {attack_times: 100, shuffle: true, attack_distribution: full_population,
               attack_trace_budget: 1500}  # empty: every trace
        attack_step: 1
        ge_thresholds: 1
      models:                            # one model's own settings, over the ones above
        model_a:
          perform_attack:
            test: {attack_times: 10}

Only perform_attack.test.{attack_times, shuffle, attack_distribution,
attack_trace_budget}, perform_attack.attack_step and perform_attack.ge_thresholds may change; the
snapshot is checked against its model hash first, and everything else is the
snapshot's. An external test never touches the recorded test: its reports
go to ``paths.test`` (``<run>/test/``) under ``<model_hash>_et_<index>/``, and
its outcome is appended to the model's ``test_registry.json`` entry under
``external_tests``, keyed ``<model_hash>_et_<index>``. Its settings_id, a hash
of the effective attack settings, the tested checkpoint policy and the
checkpoint's SHA-256, finds it again: the same settings on the same weights
update their own record (results, status_reason, finished_at, runs);
different settings append a new one, with the next index.

Author: Shraddha Mishra
"""

from __future__ import annotations

from services.side_channel_core.utils.config_errors import ConfigError

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import yaml

from services.side_channel_core.utils.model_naming import _load_mapping
from services.side_channel_core.utils.seal import seal_text
from services.side_channel_core.utils.config_freeze import config_path, frozen_dir
from utils.config_loader import PROJECTS_ROOT, SCC_CONFIG_DIR, SCC_ROOT, _resolve_scripts_dict
from utils.data_reproducibility import SNAPSHOT_DIGEST_KEY, read_params_snapshot
from utils.logger import get_logger
from utils.run_map import load_run_map, project_relative_path
from utils.timestamps import now_iso

logger = get_logger("TestRequest")

CONFIG_KEY = "test_request"
MODEL_NAME_MAP = "model_name_map.json"


class TestRequestError(ConfigError):     # a configuration error (utils/config_errors.py), a ValueError as before
    """A test request cannot be carried out as configured."""

    __test__ = False  # not a pytest test class


def _load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# =============================================================================
# Request file
# =============================================================================
def _project_block(project: str) -> tuple[dict, Path]:
    master = _load_yaml(SCC_CONFIG_DIR / "master.yaml") or {}
    block = next(
        (p for p in master.get("project") or [] if (p or {}).get("name") == project),
        None,
    )
    if block is None:
        raise TestRequestError(f"No project named '{project}' in master.yaml")
    return block, (PROJECTS_ROOT / block["root"]).resolve()


def test_config_path(project: str, override: Optional[str] = None) -> Path:
    """The project's test config: master.yaml extra_configs.test, or ``override``; in the run's
    frozen copy of config/ when there is one (utils/config_freeze.py: make test freezes at its start)."""
    block, project_root = _project_block(project)
    frozen = frozen_dir(project, project_root)
    if override:
        path = Path(override)
        return config_path(path if path.is_absolute() else (project_root / path).resolve(), project_root, frozen)
    rel = (block.get("extra_configs") or {}).get("test")
    if not rel:
        raise TestRequestError(
            f"master.yaml project '{project}' has no extra_configs.test entry; "
            "add  test: \"config/test_config.yaml\""
        )
    return config_path((project_root / rel).resolve(), project_root, frozen)


def _as_names(value: Any, *, where: str) -> list[str]:
    """A model name, a comma-separated string, or a list of names -> list."""
    if value is None:
        return []
    if isinstance(value, str):
        items: Iterable[Any] = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        raise TestRequestError(f"{where} must be a model name or a list of names, got {value!r}")
    names = [str(item).strip() for item in items if item is not None and str(item).strip()]
    return list(dict.fromkeys(names))  # keep order, drop repeats


def load_test_request(path: Path) -> list[str]:
    """Model names listed in the test config (may be empty); its external settings are checked too."""
    block = _request_block(path)
    _check_attack_block(block.get("perform_attack"), where=f"{CONFIG_KEY}.perform_attack")
    for name, own in (_models_block(block)).items():
        _model_attack(own, name)
    return _as_names(block.get("model"), where=f"{CONFIG_KEY}.model")


def _request_block(path: Path) -> Mapping[str, Any]:
    """test_request: of the test config, its keys checked."""
    if not path.is_file():
        raise TestRequestError(
            f"Test config not found: {path}. Copy "
            "services/side_channel_core/config/blueprint/test_config.yaml"
        )
    raw = _load_yaml(path) or {}
    if not isinstance(raw, Mapping) or CONFIG_KEY not in raw:
        raise TestRequestError(f"{path} must contain a top-level '{CONFIG_KEY}:' block")
    block = raw[CONFIG_KEY] or {}
    if not isinstance(block, Mapping):
        raise TestRequestError(f"'{CONFIG_KEY}' must be a mapping")
    unknown = sorted(set(block) - REQUEST_KEYS)
    if unknown:
        raise TestRequestError(
            f"Unknown key(s) in {CONFIG_KEY}: {unknown}. A test request has `model:`, and for an "
            "external test `perform_attack:` and `models: {<name>: {perform_attack: …}}`; every "
            "other setting comes from the model's training snapshot."
        )
    return block


# =============================================================================
# External test settings (test_config.yaml)
# =============================================================================
REQUEST_KEYS = frozenset({"model", "perform_attack", "models", "device"})
# test_config.yaml test_request.device (external tests only): cpu | cuda, the device the test runs on instead of the
# snapshot's (e.g. cpu beside a training on the GPU). Empty: the snapshot's. A set device joins the test's settings
# (its settings_id), so a CPU test is a record of its own; the model hash, checked on the snapshot first, is not changed.
TEST_DEVICES = ("cpu", "cuda")
ATTACK_KEYS = frozenset({"test", "attack_step", "ge_thresholds"})
TEST_ATTACK_KEYS = frozenset({"attack_times", "shuffle", "attack_distribution"})
EXTERNAL_KEY = "external_tests"
# A test's kind (owner 2026-10-27): all attacks every test trace; budget attacks the data entry's
# data[].attack_trace_budget traces per attack (utils/sweep_test_runner.data_budget). The panel's
# Retest All / the Done Budget box's Test (the snapshot's attack) and Test All / the TESTS Budget box's Test (test_config.yaml's).
KINDS = ("all", "budget")


def _check_attack_block(block: Any, *, where: str) -> dict:
    """The perform_attack settings an external test may change, checked; {} when absent."""
    if block is None:
        return {}
    if not isinstance(block, Mapping):
        raise TestRequestError(f"{where} must be a mapping")
    unknown = sorted(set(block) - ATTACK_KEYS)
    if unknown:
        raise TestRequestError(
            f"{where}: unknown key(s) {unknown}. An external test changes only "
            "perform_attack.test.{attack_times, shuffle, attack_distribution, attack_trace_budget}, "
            "perform_attack.attack_step and perform_attack.ge_thresholds."
        )
    out: dict = {}
    test = block.get("test")
    if test is not None:
        if not isinstance(test, Mapping):
            raise TestRequestError(f"{where}.test must be a mapping")
        bad = sorted(set(test) - TEST_ATTACK_KEYS - {"attack_trace_budget"})
        if bad:
            raise TestRequestError(f"{where}.test: unknown key(s) {bad}; allowed: {sorted(TEST_ATTACK_KEYS)}")
        if test.get("attack_trace_budget") is not None:
            raise TestRequestError(
                f"{where}.test.attack_trace_budget: the budget is the data entry's (params.yaml "
                "data[].attack_trace_budget); a test attacks with it as a Budget test (make test BUDGET=1)"
            )
        out["test"] = {k: v for k, v in test.items() if v is not None and k != "attack_trace_budget"}
    if block.get("attack_step") is not None:            # null (a blueprint) = not set
        step = block["attack_step"]
        if isinstance(step, (list, tuple)):
            if len(step) != 1:
                raise TestRequestError(f"{where}.attack_step: one value, not a sweep list {list(step)}")
            step = step[0]
        if type(step) is not int or step < 1:
            raise TestRequestError(f"{where}.attack_step must be an integer >= 1, got {block['attack_step']!r}")
        out["attack_step"] = step
    if block.get("ge_thresholds") is not None:
        out["ge_thresholds"] = block["ge_thresholds"]
    return out


def _models_block(block: Mapping[str, Any]) -> Mapping[str, Any]:
    models = block.get("models") or {}
    if not isinstance(models, Mapping):
        raise TestRequestError(f"{CONFIG_KEY}.models must map model names to their settings")
    return models


def _model_attack(own: Any, name: str) -> dict:
    where = f"{CONFIG_KEY}.models.{name}"
    if own is None:
        return {}
    if not isinstance(own, Mapping):
        raise TestRequestError(f"{where} must be a mapping")
    unknown = sorted(set(own) - {"perform_attack"})
    if unknown:
        raise TestRequestError(f"{where}: unknown key(s) {unknown}; a model sets only perform_attack")
    return _check_attack_block(own.get("perform_attack"), where=f"{where}.perform_attack")


def external_settings(path: Path, model_name: str) -> dict:
    """
    The attack settings of one model's external test: test_request.perform_attack, then
    test_request.models.<model_name>.perform_attack over it. An external test attacks:
    perform_attack.test.attack_times must be >= 1.
    """
    block = _request_block(path)
    base = _check_attack_block(block.get("perform_attack"), where=f"{CONFIG_KEY}.perform_attack")
    own = _model_attack(_models_block(block).get(model_name), model_name)
    merged = {**{k: v for k, v in base.items() if k != "test"}, **{k: v for k, v in own.items() if k != "test"}}
    merged["test"] = {**base.get("test", {}), **own.get("test", {})}
    times = merged["test"].get("attack_times")
    if type(times) is not int or times < 1:
        raise TestRequestError(
            f"{model_name}: an external test attacks the test traces: set "
            f"{CONFIG_KEY}.perform_attack.test.attack_times (or {CONFIG_KEY}.models.{model_name}"
            f".perform_attack.test.attack_times) to an integer >= 1, got {times!r}"
        )
    return merged


# =============================================================================
# Name -> hash -> trained run
# =============================================================================
def resolve_models(project: str, names: list[str], *, env: Optional[str] = None) -> list[dict]:
    """
    Resolve model names (or hashes) to trained runs.

    model_name_map.json gives the hash, run_map.jsonl the run. Every name is
    checked before any test starts; unknown or unfinished models fail together.
    """
    if not names:
        raise TestRequestError(
            "No model to test: pass MODEL=<name>[,<name>…] or list names under test_request.model"
        )
    _block, project_root = _project_block(project)
    map_path = project_root / "artifacts" / MODEL_NAME_MAP
    # Raises on a damaged map (never read as empty).
    name_map = _load_mapping(project_root)
    hash_by_name = {name: model_hash for model_hash, name in name_map.items()}
    run_map = load_run_map(project_root)

    resolved, problems = [], []
    for name in names:
        model_hash = hash_by_name.get(name) or (name if name in name_map else None)
        if model_hash is None:
            problems.append(f"{name}: not in {map_path}")
            continue
        run = run_map.get(model_hash)
        if not isinstance(run, Mapping):
            problems.append(f"{name} ({model_hash}): no trained run in artifacts/run_map.jsonl")
            continue
        status = str(run.get("status") or "").upper()
        if status != "DONE":
            problems.append(f"{name} ({model_hash}): training status {status or 'unknown'}, not DONE")
            continue
        run_env = str(run.get("env") or "dev")
        if env and env != run_env:
            problems.append(f"{name} ({model_hash}): trained with ENV={run_env}; run the test with ENV={run_env}")
            continue
        resolved.append({**dict(run), "model_name": str(name_map[model_hash]), "model_hash": model_hash})

    if problems:
        available = sorted(str(n) for n in name_map.values())
        raise TestRequestError(
            "Cannot test:\n  - " + "\n  - ".join(problems)
            + f"\nTrained models of {project}: {available or 'none'}"
        )
    return resolved


def _live_project(project: str, block: Mapping[str, Any], project_root: Path) -> dict:
    """The project block as config_loader builds it, rooted on this machine."""
    return {
        "name": project,
        "root": str(project_root),
        "params": block.get("params"),
        "code_dir": block.get("code_dir", "src"),
        "logs": block.get("logs", "artifacts/logs/"),
        "models": block.get("models", "artifacts/models/"),
        "metrics": block.get("metrics", "artifacts/metrics/"),
        "use_core_src": block.get("use_core_src", True),
        "env_configs": block.get("env_configs", {}),
        "print_config": block.get("print_config", True),
        "scripts": _resolve_scripts_dict(dict(block), project_root),
        "core_src_dir": str((SCC_ROOT / "src").resolve()),
        "project_src_dir": str((project_root / block.get("code_dir", "src")).resolve()),
    }


def load_snapshot(project: str, run: Mapping[str, Any]) -> dict:
    """The model's resolved training config, rooted on the live project."""
    block, project_root = _project_block(project)
    model_name, model_hash = run["model_name"], run["model_hash"]
    rel = run.get("path") or f"artifacts/{run.get('env', 'dev')}/generated_models/{model_name}"
    # A path in run_registry.json's form (<project>/artifacts/...) is read as artifacts/...
    snapshot_path = project_root / project_relative_path(rel, project_root) / "config" / f"{model_name}_params.yaml"
    if not snapshot_path.is_file():
        raise TestRequestError(
            f"Training snapshot not found: {snapshot_path}. Testing needs the exact "
            "configuration the model was trained with."
        )
    # Rejects a truncated/damaged snapshot (digest, required sections, model_hash);
    # the test sweep re-reads it and recomputes its model_hash.
    cfg = read_params_snapshot(snapshot_path)
    cfg.pop(SNAPSHOT_DIGEST_KEY, None)
    for field, expected in (("model_name", model_name), ("model_hash", model_hash)):
        recorded = cfg.get(field)
        if recorded and str(recorded) != expected:
            raise TestRequestError(f"Snapshot {field}={recorded!r} conflicts with run_map.jsonl ({expected!r})")
        cfg[field] = expected
    cfg["env"] = str(run.get("env") or "dev")
    cfg["project"] = _live_project(project, block, project_root)
    cfg.pop("paths_abs", None)
    return cfg


# =============================================================================
# Execution
# =============================================================================
def run_test_request(
    *,
    project: str,
    models: Optional[list[str]],
    env: Optional[str],
    load_module: Callable[[str], Any],
    test_config: Optional[str] = None,
    external: bool = False,
    kind: str = "all",
) -> list[dict]:
    """
    Test the requested models again (``cli.py test-cmd --test-request``), each test appended (owner 2026-10-27).

    ``models`` (from MODEL=) wins over the names in test_config.yaml. A Retest attacks with the model's own
    snapshot settings; ``external`` (Test) with test_config.yaml's. ``kind`` all attacks every test trace,
    budget the data entry's attack_trace_budget. Every test is a record of its own, under
    test_registry.json[<hash>].external_tests[<hash>_et_<index>] and <run>/test/<hash>_et_<index>/, never
    over another (the same settings again append a new index): the panel and the reports read the latest
    of each kind, so a later better result cannot be chosen over the newest. The test the training ran
    (the recorded test) is never rewritten.
    """
    return run_external_test(project=project, models=models, env=env, load_module=load_module,
                             test_config=test_config, retest=not external, kind=kind)


# =============================================================================
# External tests
# =============================================================================
def external_test_id(record: Mapping[str, Any]) -> str:
    """The settings_id of an external test: its effective attack settings, checkpoint policy and checkpoint."""
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()[:8]


def external_device(path: Path) -> Optional[str]:
    """test_request.device of the test config: cpu | cuda, or None (the snapshot's device)."""
    device = _request_block(path).get("device")
    if device in (None, ""):
        return None
    if device not in TEST_DEVICES:
        raise TestRequestError(f"{CONFIG_KEY}.device must be one of {list(TEST_DEVICES)} (empty: the snapshot's), got {device!r}")
    return str(device)


def _apply_attack(run_cfg: dict, settings: Mapping[str, Any]) -> dict:
    """Put the external settings into the run's perform_attack; return the effective ones."""
    attack = run_cfg.setdefault("perform_attack", {})
    attack["test"] = {**(attack.get("test") or {}), **settings.get("test", {})}
    attack["test"].pop("attack_trace_budget", None)        # the kind decides it (_apply_kind), never a snapshot
    for key in ("attack_step", "ge_thresholds"):
        if key in settings:
            attack[key] = settings[key]
    return {"test": dict(attack["test"]), "attack_step": attack.get("attack_step"),
            "ge_thresholds": attack.get("ge_thresholds")}


def _apply_kind(run_cfg: dict, attack: dict, kind: str) -> Optional[int]:
    """A test of kind all attacks every test trace; of kind budget, its data entry's attack_trace_budget traces per
    attack (refused when the entry has none). Puts it into the run's and the effective attack; returns the budget."""
    from utils.sweep_test_runner import data_budget

    if kind not in KINDS:
        raise TestRequestError(f"a test's kind is one of {list(KINDS)}, got {kind!r}")
    if kind == "all":
        return None
    budget = data_budget(run_cfg)
    if budget is None:
        raise TestRequestError(
            f"a Budget test needs its data entry's attack_trace_budget: data entry {((run_cfg.get('data') or {}).get('name'))!r} "
            "has none (params.yaml data[].attack_trace_budget, the traces of one test attack)"
        )
    run_cfg["perform_attack"]["test"]["attack_trace_budget"] = budget
    attack["test"]["attack_trace_budget"] = budget
    return budget


def external_key(model_hash: str, index: int) -> str:
    """The key of a model's external test in test_registry.json and its folder under <run>/test/."""
    return f"{model_hash}_et_{index}"


def _consolidate(tests: dict, model_hash: str) -> bool:
    """
    Records of the earlier format (keyed by their settings hash, no settings_id) get their settings_id and the
    key of the next index, <model_hash>_et_<index>. Records are never merged (owner 2026-10-27): every test is
    appended, the same settings too, and the latest of each kind is the one read. Returns True when it changed
    something.
    """
    changed = False
    top = max((int(r.get("index") or 0) for r in tests.values() if isinstance(r, dict)), default=0)
    for key in [k for k, r in tests.items() if isinstance(r, dict) and not r.get("settings_id")]:
        record = tests.pop(key)
        top += 0 if record.get("index") else 1
        index = int(record.get("index") or top)
        record.update(index=index, settings_id=external_test_id(record.get("settings") or {}))
        tests[external_key(model_hash, index)] = record
        changed = True
    return changed


def _find_external(project_root: Path, run: Mapping[str, Any], identity: Mapping[str, Any]) -> str:
    """The key the model's next test gets: the next index (every test is appended). The model's records of the
    earlier format are re-keyed first (_consolidate)."""
    from utils.test_registry import load_test_registry, test_registry_lock, save_test_registry

    with test_registry_lock(project_root):
        registry = load_test_registry(project_root, heal=False)
        entry = registry.get(run["model_hash"])
        tests = (entry or {}).get(EXTERNAL_KEY) or {}
        if isinstance(entry, dict) and tests and _consolidate(tests, run["model_hash"]):
            save_test_registry(registry, project_root)
    # every test is appended (owner 2026-10-27): the next index, also for settings already tested (settings_id
    # stays a record's identity of what it attacked with)
    index = 1 + max((int(r.get("index") or 0) for r in tests.values() if isinstance(r, dict)), default=0)
    return external_key(run["model_hash"], index)


def _record_external(project_root: Path, run: Mapping[str, Any], key: str, update: Mapping[str, Any]) -> dict:
    """
    Write one external test's record into test_registry.json[<hash>].external_tests[<hash>_et_<index>]:
    the recorded test's own fields are left as they are.
    """
    from utils.test_registry import update_test_registry

    written: dict = {}

    def change(registry: dict) -> None:
        entry = registry.get(run["model_hash"])
        if not isinstance(entry, dict):
            entry = registry[run["model_hash"]] = {
                "model_name": run["model_name"], "path": run.get("path"), "env": run.get("env"),
            }
        tests = entry.setdefault(EXTERNAL_KEY, {})
        record = tests.get(key)
        if not isinstance(record, dict):
            record = tests[key] = {"index": int(key.rsplit("_", 1)[1])}
        record.update(update)
        written.update(record)

    update_test_registry(project_root, change, heal=False)   # under the registry's lock
    return written


# =============================================================================
# status_reason: why an external test ended as it did
# =============================================================================
def done_reason(outcome: Mapping[str, Any]) -> str:
    """status_reason of a completed external test, from its key-recovery outcome."""
    recovered, rank = outcome.get("key_recovered"), outcome.get("recovered_key_rank")
    if recovered is True:
        return f"completed: the key is recovered (true-key rank {rank})"
    if recovered is False:
        return f"completed: the key is not recovered (true-key rank {rank})"
    return f"completed: {recovered}"


def failure_reason(exc: BaseException) -> str:
    """
    status_reason of a failed external test: what failed, then the error. Categories: model not
    found, snapshot missing, snapshot corrupt, snapshot does not match its model hash, settings
    incomplete, checkpoint missing or corrupt, data missing, data corrupt, report incomplete,
    system failure (GPU, memory, disk), test failure (anything else).
    """
    from src.sca.key_recovery import KeyRecoveryError
    from utils.checkpoint_policy import CheckpointPolicyError
    from utils.key_recovery_helper import KeyRecoveryHelperError

    from utils.device import GPU_FAILED, failure_reason as device_failure

    detail = f"{type(exc).__name__}: {exc}".strip()
    low = str(exc).lower()
    if device_failure(exc).startswith(GPU_FAILED):
        return device_failure(exc)
    if isinstance(exc, TestRequestError):
        if "snapshot not found" in low:
            category = "snapshot missing"
        elif "not in" in low and ("model_name_map" in low or "run_registry" in low or "run_map" in low):
            category = "model not found"
        elif "test.json" in low or "report" in low:
            category = "report incomplete"
        else:
            category = "settings incomplete"
    elif isinstance(exc, CheckpointPolicyError):
        category = "checkpoint missing or corrupt"
    elif "params_out snapshot" in low:
        category = "snapshot corrupt"
    elif "hashes to" in low or "conflicts with registry" in low:
        category = "snapshot does not match its model hash"
    elif isinstance(exc, FileNotFoundError):
        category = "data missing" if (".h5" in low or "data" in low) else "file missing"
    elif isinstance(exc, MemoryError) or "out of memory" in low:
        category = "system failure: out of memory"
    elif any(word in low for word in ("cuda", "cudnn", "nccl", "no gpu", "device-side")):
        category = "system failure: GPU"
    elif isinstance(exc, OSError) and any(word in low for word in ("hdf5", ".h5", "truncated", "unable to open", "signature")):
        category = "data corrupt"
    elif isinstance(exc, OSError):
        category = "system failure: disk or file system"
    elif isinstance(exc, (KeyRecoveryError, KeyRecoveryHelperError)):
        category = "report incomplete"
    elif isinstance(exc, (KeyError, TypeError, ValueError)) and ("perform_attack" in low or "attack_" in low):
        category = "settings incomplete"
    else:
        category = "test failure"
    return f"{category}: {detail}"[:500]


def _external_outcome(run_cfg: dict, model_name: str, log: Any) -> dict:
    """The key-recovery outcome of the external test's own test.json, as the registries write it."""
    from src.sca.key_recovery import (
        key_recovery_outcome, key_recovery_paths, log_combined_key_recovery_summary,
        validate_key_recovery_contract,
    )
    from utils.key_recovery_helper import _load_optional_report, _registry_outcome, _true_key
    from utils.lifecycle_outcome import LIFECYCLE_REGISTRY_FIELDS, _jsonable

    if not validate_key_recovery_contract(run_cfg)["enabled"]:
        note = "not available: key_recovery_reporting.enabled is false"
        return {key: note for key in ("key_recovered", "true_key", "recovered_key", "recovered_key_rank")}
    report = _load_optional_report(
        path=key_recovery_paths(run_cfg, model_name=model_name)["test"], expected_process="test", logger=log,
    )
    if report is None:
        raise TestRequestError("the external test wrote no key-recovery report (test.json)")
    cv = (run_cfg.get("cross_validation") or {}).get("enabled") is True
    fields = _registry_outcome(key_recovery_outcome(report), _true_key(run_cfg, "test", cv_enabled=cv, logger=log, reports=[report]))
    log_combined_key_recovery_summary(log, cfg=run_cfg, validation_report=None, test_report=report, include_validation=False)
    return {key: _jsonable(value) for key, value in fields.items() if key in LIFECYCLE_REGISTRY_FIELDS}


def run_external_test(
    *,
    project: str,
    models: Optional[list[str]],
    env: Optional[str],
    load_module: Callable[[str], Any],
    test_config: Optional[str] = None,
    retest: bool = False,
    kind: str = "all",
) -> list[dict]:
    """
    Test the requested models with test_config.yaml's attack settings (a Retest: each snapshot's own), beside
    their recorded test; kind all or budget (run_test_request).

    For each model: its snapshot is checked against its model hash, the external settings are put
    into perform_attack, the checkpoint is resolved (its SHA-256 joins the settings_id), and the test
    runs with its metrics, key-recovery report and logs under <run>/test/<hash>_et_<index>/. Its
    outcome goes to test_registry.json[<hash>].external_tests[<hash>_et_<index>], with a status_reason
    (done_reason, failure_reason). Every attempt is recorded, also one that fails before the test
    starts (filed under the requested settings). Every model's settings are checked before any test
    starts; a failed test does not stop the others.
    """
    from utils.checkpoint_policy import resolve_test_checkpoint
    from utils.cross_validation import require_refit_model
    from utils.logger import configure_global_logging
    from utils.sweep_engine import load_registry as load_train_registry
    from utils.sweep_test_runner import _build_test_cfg
    from src.sca.key_enumeration import validate_key_enumeration_contract
    from src.sca.perform_attack import attack_plan

    config_path = test_config_path(project, test_config)
    names = _as_names(models, where="MODEL") or load_test_request(config_path)
    runs = resolve_models(project, names, env=env)
    # a Retest: the snapshot's attack settings (nothing over them); a Test: test_config.yaml's
    settings = {run["model_name"]: ({} if retest else external_settings(config_path, run["model_name"])) for run in runs}
    device = external_device(config_path)
    _block, project_root = _project_block(project)
    train_registry = load_train_registry(project_root, heal=False)
    config_rel = _relative(config_path, project_root)
    logger.info("External test: project=%s models=%s config=%s", project, [r["model_name"] for r in runs], config_path)

    outcomes, failed = [], []
    for run in runs:
        model_name, model_hash = run["model_name"], run["model_hash"]
        # Until the checkpoint is known, the attempt is filed under the requested settings.
        identity: dict = {"perform_attack": settings[model_name], "load_model": None, "checkpoint_sha256": None}
        identity_kind = {"kind": kind, "source": "retest" if retest else "test"}     # what the panel and the reports group by
        key, out, update = None, None, {}
        try:
            cfg = load_snapshot(project, run)
            os.environ["ENV"] = cfg["env"]
            train_info = train_registry.get(model_hash)
            if not isinstance(train_info, Mapping):
                raise TestRequestError(f"{model_name} ({model_hash}) is not in artifacts/run_registry.json")
            run_cfg = _build_test_cfg(dict(train_info), model_hash=model_hash, model_name=model_name, cfg=cfg)
            attack = _apply_attack(run_cfg, settings[model_name])
            _apply_kind(run_cfg, attack, kind)
            if device:                                                 # after the snapshot's hash check (load_snapshot)
                run_cfg["device"] = device
            attack_plan(run_cfg, "test")                               # the combination is valid
            require_refit_model(run_cfg, model_name)                   # a CV run without a refit has no model
            checkpoint = resolve_test_checkpoint(run_cfg, model_name=model_name)
            identity = {"perform_attack": attack, "load_model": checkpoint.effective_policy,
                        "checkpoint_sha256": checkpoint.checkpoint_sha256, **({"device": device} if device else {})}
            key = _find_external(project_root, run, identity)
            paths = run_cfg["paths"]
            base = Path(paths["test"]) if paths.get("test") else Path(paths["artifacts_root"]) / "test"
            out = base / key
            paths["metrics"] = out / "metrics" / "attack_metrics"
            paths["key_recovery"] = out / "metrics" / "key_recovery"
            paths["logs"] = out / "logs"
            run_cfg.setdefault("runtime", {}).update(model_hash=model_hash, model_name=model_name)
            enumeration = validate_key_enumeration_contract(run_cfg)["enabled"]
            out.mkdir(parents=True, exist_ok=True)
            (out / "test_settings.json").write_text(seal_text(out / "test_settings.json", json.dumps(
                {"model_name": model_name, "model_hash": model_hash, "key": key,
                 "settings_id": external_test_id(identity), **identity, "test_config": config_rel},
                indent=2, default=str) + "\n"))
            started = now_iso()
            _record_external(project_root, run, key, {
                "settings": identity, "settings_id": external_test_id(identity), **identity_kind, "status": "in_progress",
                "status_reason": f"running: started {started}", "started_at": started, "finished_at": None,
                "report": _relative(out, project_root), "test_config": config_rel,
                **({"key_enumeration": "not applied to external tests"} if enumeration else {}),
            })
            log_dir = configure_global_logging(cfg=run_cfg, model_hash=model_hash)
            test_log = get_logger("Testing", log_dir=str(log_dir))
            test_log.info("External test %s of %s (%s): %s", key, model_name, model_hash, identity)
            script = run_cfg["project"]["scripts"].get("test")
            if not script:
                raise TestRequestError("master.yaml project scripts.test is not set")
            load_module(str(script)).test(run_cfg, exec_process="test", override_model_name=model_name,
                                          model_id=run_cfg.get("model_id"))
            outcome = _external_outcome(run_cfg, model_name, test_log)
            update = {"status": "DONE", "status_reason": done_reason(outcome), **outcome}
        except KeyboardInterrupt:
            update = {"status": "INTERRUPTED", "status_reason": "interrupted: stopped by the user during the test"}
            raise
        except Exception as exc:
            update = {"status": "FAILED", "status_reason": failure_reason(exc)}
            failed.append(f"{model_name}: {update['status_reason']}")
            logger.exception("External test FAILED %s (%s): %s", model_name, model_hash, exc)
        finally:
            key = key or _find_external(project_root, run, identity)
            previous = ((((_load_entry(project_root, model_hash) or {}).get(EXTERNAL_KEY) or {}).get(key)) or {})
            now = now_iso()
            record = _record_external(project_root, run, key, {
                "settings": identity, "settings_id": external_test_id(identity), **identity_kind, "test_config": config_rel,
                **({"report": _relative(out, project_root)} if out is not None else {}),
                **update, "finished_at": now, "runs": int(previous.get("runs") or 0) + 1,
                "first_finished_at": previous.get("first_finished_at") or now,
            })
        outcomes.append({"model_name": model_name, "model_hash": model_hash, "key": key, **record})
    if failed:
        raise RuntimeError("External test failed for: " + "; ".join(failed))
    return outcomes


def _relative(path: Path, root: Path) -> str:
    """A path as the registries write it: relative to the project root, when inside it."""
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def _load_entry(project_root: Path, model_hash: str) -> Optional[dict]:
    from utils.test_registry import load_test_registry

    entry = load_test_registry(project_root, heal=False).get(model_hash)
    return entry if isinstance(entry, dict) else None


__all__ = [
    "CONFIG_KEY",
    "EXTERNAL_KEY",
    "done_reason",
    "external_key",
    "external_settings",
    "failure_reason",
    "external_test_id",
    "run_external_test",
    "TestRequestError",
    "load_snapshot",
    "load_test_request",
    "resolve_models",
    "run_test_request",
    "test_config_path",
]
