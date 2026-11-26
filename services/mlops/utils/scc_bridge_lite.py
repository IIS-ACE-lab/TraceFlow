# services/mlops/utils/scc_bridge_lite.py
"""
SCC Bridge Lite
===============

Lightweight bridge for executing SCC training/testing runs from OptunaLite.

Responsibilities
----------------
- Resolve the concrete SCC runtime contract for one sampled auto-search attempt.
- Resolve stable SCC run identity: model_name + model_hash.
- Ensure the normal SCC train/test registries contain model_hash entries.
- Update only the authoritative SCC registries:
    - artifacts/run_registry.json
    - artifacts/test_registry.json
- Run SCC train, and optionally SCC test when mode="orchestrate".
- Return runtime metadata to OptunaLite for objective extraction.

Important Registry Contract
---------------------------
This module does NOT create or maintain a separate trial registry.

trial_id is runtime progress/debug metadata only.
model_hash is the only registry identity.

Author: Shraddha Mishra
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
import copy
from services.mlops.utils.zombie_reaper import cleanup_trial_runtime
from services.side_channel_core.src.core.model_contract_resolver import resolve_model_contract
from services.side_channel_core.src.sca.test import test as scc_test
from services.side_channel_core.src.sca.train import train as scc_train
from services.side_channel_core.utils.logger import (
    get_logger,
    pretty_print,
)
from services.side_channel_core.utils.key_recovery_helper import (
    _log_completed_cycle_key_recovery_summary,
)
from services.side_channel_core.utils.sweep_engine import load_registry,save_registry,update_status
from services.side_channel_core.utils.test_registry import load_test_registry,save_test_registry,update_test_status
from utils.logger import configure_global_logging

try:
    from services.side_channel_core.utils.path_utils import resolve_path as scc_resolve_path
except Exception:
    scc_resolve_path = None

logger = get_logger("SCCBridgeLite")

class _ScopedEnv:
    """
    Temporarily set environment variables for the active SCC trial.

    Used to expose AUTO_SEARCH metadata to downstream SCC code without
    permanently mutating the parent process environment.
    """

    def __init__(self, updates: Dict[str, str]):
        self.updates = updates
        self.prev: Dict[str, Optional[str]] = {}

    def __enter__(self):
        for key, value in self.updates.items():
            self.prev[key] = os.environ.get(key)
            os.environ[key] = str(value)
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, old_value in self.prev.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
        return False


def _resolve_metrics_dir_from_scc_cfg(run_cfg: dict) -> Optional[str]:
    """
    Resolve metrics directory from SCC config.

    Preferred:
        paths.metrics through SCC path resolver.

    Fallback:
        paths.artifacts_root / metrics
    """
    if scc_resolve_path is not None:
        try:
            return str(scc_resolve_path(run_cfg["paths"]["metrics"], run_cfg))
        except Exception:
            pass

    try:
        artifacts_root = run_cfg.get("paths", {}).get("artifacts_root")
        if artifacts_root:
            return str(Path(artifacts_root) / "metrics")
    except Exception:
        return None

    return None


def _resolve_run_dir_from_scc_cfg(
    run_cfg: dict,
    model_name: str,
    model_hash: str,
) -> str:
    """
    Resolve the concrete per-run directory for logging/artifacts before SCC execution.

    Priority:
      1. paths.log_dir if present
      2. paths.artifacts_root / <model_name>
      3. project.root / artifacts / <model_name>
    """
    paths = run_cfg.get("paths") or {}
    if not isinstance(paths, dict):
        paths = {}

    log_dir = paths.get("log_dir")
    if isinstance(log_dir, str) and log_dir.strip():
        if scc_resolve_path is not None:
            try:
                resolved = scc_resolve_path(log_dir, run_cfg)
                path = Path(resolved)
                path.mkdir(parents=True, exist_ok=True)
                return str(path)
            except Exception:
                pass

        path = Path(log_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    artifacts_root = paths.get("artifacts_root")
    if isinstance(artifacts_root, str) and artifacts_root.strip():
        if scc_resolve_path is not None:
            try:
                resolved_root = Path(scc_resolve_path(artifacts_root, run_cfg))

                # artifacts_root already resolves to:
                # artifacts/{{env}}/generated_models/{{model_name}}
                # Do NOT append model_name again.
                path = resolved_root

                path.mkdir(parents=True, exist_ok=True)
                return str(path)
            except Exception:
                pass

        path = Path(artifacts_root).expanduser().resolve()

        # Safety guard against accidental:
        # butter_mozzarella/butter_mozzarella
        parts = path.parts
        if len(parts) >= 2 and parts[-1] == str(model_name) and parts[-2] == str(model_name):
            path = Path(*parts[:-1])

        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    project_root = Path(run_cfg["project"]["root"]).expanduser().resolve()
    path = project_root / "artifacts" / str(model_name)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _runtime_env(run_cfg: dict) -> str:
    return str(run_cfg.get("env") or os.environ.get("ENV") or "dev").strip() or "dev"

def _ensure_train_registry_entry(
    *,
    run_cfg: dict,
    model_hash: str,
    model_name: str,
    run_dir: str,
) -> None:
    """
    Ensure run_registry.json contains model_hash before update_status() is called.

    Guarantees:
    - model_id is written immediately at NEW/in_progress setup time
    - env is stable
    - existing entries are repaired without wiping timestamps/status
    """
    project_root = Path(run_cfg["project"]["root"]).resolve()
    registry = load_registry(project_root)

    env_val = _runtime_env(run_cfg)
    model_id_val = str(
        run_cfg.get("model_id")
        or os.environ.get("MODEL_ID")
        or ""
    ).strip()

    if model_hash not in registry or not isinstance(registry.get(model_hash), dict):
        entry = {
            "model_name": str(model_name),
            "status": "NEW",
            "path": str(run_dir),
            "env": env_val,
        }
        registry[model_hash] = entry
    else:
        entry = registry[model_hash]
        entry["model_name"] = str(model_name)
        entry["path"] = str(run_dir)

        if not str(entry.get("env") or "").strip():
            entry["env"] = env_val

    if model_id_val:
        entry["model_id"] = model_id_val

    save_registry(registry, project_root)


def _ensure_test_registry_entry(
    *,
    run_cfg: dict,
    model_hash: str,
    model_name: str,
    run_dir: str,
) -> None:
    """
    Ensure test_registry.json contains model_hash before update_test_status() is called.

    Guarantees:
    - model_id is written immediately at NEW/in_progress setup time
    - env is stable
    - existing entries are repaired without wiping timestamps/status
    """
    project_root = Path(run_cfg["project"]["root"]).resolve()
    registry = load_test_registry(project_root)

    env_val = _runtime_env(run_cfg)
    model_id_val = str(
        run_cfg.get("model_id")
        or os.environ.get("MODEL_ID")
        or ""
    ).strip()

    if model_hash not in registry or not isinstance(registry.get(model_hash), dict):
        entry = {
            "model_name": str(model_name),
            "status": "NEW",
            "path": str(run_dir),
            "env": env_val,
        }
        registry[model_hash] = entry
    else:
        entry = registry[model_hash]
        entry["model_name"] = str(model_name)
        entry["path"] = str(run_dir)

        if not str(entry.get("env") or "").strip():
            entry["env"] = env_val

    if model_id_val:
        entry["model_id"] = model_id_val

    save_test_registry(registry, project_root)
def _inject_dataset_selector(run_cfg: dict) -> None:
    """
    Inject cfg['dataset'] and cfg['dataset_name'] when cfg['data'] is a dict-of-datasets.

    Accepted downstream selectors:
      - cfg["dataset"]
      - cfg["dataset_name"]

    Resolution order:
      1. existing dataset / dataset_name
      2. dataset_alias if it matches cfg["data"]
      3. sole data key if only one dataset exists
      4. DATASET environment variable if it matches cfg["data"]
    """
    if not isinstance(run_cfg, dict):
        return

    data = run_cfg.get("data")
    if not isinstance(data, dict) or not data:
        return

    dataset = run_cfg.get("dataset")
    if isinstance(dataset, str) and dataset.strip():
        run_cfg["dataset"] = dataset.strip()
        run_cfg.setdefault("dataset_name", dataset.strip())
        return

    dataset_name = run_cfg.get("dataset_name")
    if isinstance(dataset_name, str) and dataset_name.strip():
        run_cfg["dataset_name"] = dataset_name.strip()
        run_cfg.setdefault("dataset", dataset_name.strip())
        return

    alias = run_cfg.get("dataset_alias")
    if isinstance(alias, str) and alias.strip() and alias.strip() in data:
        run_cfg["dataset"] = alias.strip()
        run_cfg["dataset_name"] = alias.strip()
        return

    names = [key for key in data.keys() if isinstance(key, str) and key.strip()]
    if len(names) == 1:
        run_cfg["dataset"] = names[0]
        run_cfg["dataset_name"] = names[0]
        return

    env_dataset = os.environ.get("DATASET", "").strip()
    if env_dataset and env_dataset in data:
        run_cfg["dataset"] = env_dataset
        run_cfg["dataset_name"] = env_dataset


def _inject_test_model_name(run_cfg: dict, model_name: str) -> None:
    """
    Ensure SCC test config points to the exact model produced by training.
    """
    if not isinstance(run_cfg, dict):
        return

    run_cfg.setdefault("test", {})
    if not isinstance(run_cfg["test"], dict):
        run_cfg["test"] = {}

    run_cfg["test"]["model_name"] = str(model_name)

def _emit_completed_key_recovery_summary(
    *,
    run_cfg: dict,
    model_name: str,
    model_hash: str,
) -> bool:
    """
    Print one combined validation/test key-recovery summary after a successful
    AutoSearch train -> test lifecycle.

    Reporting failures are non-fatal and must not invalidate a completed trial.
    """
    summary_cfg = copy.deepcopy(
        run_cfg
    )

    summary_cfg["model_name"] = str(model_name)
    summary_cfg["model_hash"] = str(model_hash)

    summary_cfg.setdefault("runtime", {})

    if not isinstance(summary_cfg["runtime"], dict):
        summary_cfg["runtime"] = {}

    summary_cfg["runtime"]["model_name"] = str(model_name)
    summary_cfg["runtime"]["model_hash"] = str(model_hash)

    try:
        complete = _log_completed_cycle_key_recovery_summary(
            cfg=summary_cfg,
            logger=logger,
        )

        if not complete:
            logger.warning(
                "AutoSearch train → test completed, but the final "
                "key-recovery summary was incomplete: hash=%s name=%s",
                model_hash,
                model_name,
            )

        return bool(complete)

    except Exception as exc:
        logger.warning(
            "AutoSearch train → test completed, but final key-recovery "
            "reporting failed: hash=%s name=%s error=%s",
            model_hash,
            model_name,
            exc,
            exc_info=True,
        )

        return False
def _get_frozen_identity(run_cfg: dict) -> tuple[str, str]:
    """
    Read the already-resolved identity from run_cfg.

    OptunaLite is responsible for:
      sampled params -> resolved run_cfg -> model_name/model_hash -> frozen_identity

    SCC Bridge Lite only consumes this identity.
    """
    frozen = run_cfg.get("frozen_identity") or {}

    if isinstance(frozen, dict):
        model_name = str(frozen.get("model_name") or "").strip()
        model_hash = str(frozen.get("model_hash") or "").strip()

        if model_name and model_hash:
            return model_name, model_hash

    model_name = str(run_cfg.get("model_name") or "").strip()
    model_hash = str(run_cfg.get("model_hash") or "").strip()

    if model_name and model_hash:
        return model_name, model_hash

    raise ValueError(
        "scc_bridge_lite requires pre-resolved identity. "
        "Missing run_cfg['frozen_identity']['model_name'] / ['model_hash']. "
        "OptunaLite must call build_run_identity_label() before bridge execution."
    )

def run_trial_via_scc_lite(
    *,
    run_cfg: dict,
    mode: str,
    trial_id: str,
) -> Dict[str, Any]:
    """
    Execute one OptunaLite sampled config through SCC.

    Parameters
    ----------
    run_cfg:
        SCC-compatible config for one sampled attempt.

    mode:
        "train" or "orchestrate".
        - train: run SCC train only.
        - orchestrate: run SCC train followed by SCC test.

    trial_id:
        Runtime progress/debug identifier only.
        It is not used for registry identity.

    Returns
    -------
    dict
        Runtime metadata for objective extraction by optuna_lite.py.
    """
    if not isinstance(run_cfg, dict):
        raise TypeError("run_cfg must be a dict")

    if "project" not in run_cfg or "paths" not in run_cfg:
        raise ValueError("run_cfg must contain at least ['project', 'paths']")

    if "root" not in run_cfg["project"]:
        raise ValueError("run_cfg['project'] must include 'root'")

    _inject_dataset_selector(run_cfg)

    run_cfg.setdefault("runtime", {})
    run_cfg["runtime"]["execution_source"] = "auto_search"
    run_cfg["runtime"]["driver"] = "optuna_lite"
    run_cfg["runtime"]["trial_id"] = str(trial_id)
    run_cfg["runtime"]["dataset"] = run_cfg.get("dataset") or run_cfg.get("dataset_name")

    trial_index = run_cfg.get("runtime", {}).get("trial_index")
    trial_total = run_cfg.get("runtime", {}).get("trial_total")

    pretty_print(
        f"[scc_bridge_lite] Optuna trial progress: "
        f"{trial_index}/{trial_total} trial_id={trial_id}",
        color="cyan",
        bold=True,
    )

    model_id_val = str(
    run_cfg.get("model_id")
    or os.environ.get("MODEL_ID")
    or ""
    ).strip()

    env_updates = {
        "AUTO_SEARCH": "1",
        "AUTO_SEARCH_TRIAL_ID": str(trial_id),
    }

    if model_id_val:
        env_updates["MODEL_ID"] = model_id_val

    started = time.time()

    with _ScopedEnv(env_updates):
        runtime_contract = run_cfg.get("runtime_contract") or {}

        if runtime_contract.get("resolved") is True:
            # Already resolved by OptunaLite. Do not resolve again because
            # search_space has already been removed.
            pass
        else:
            run_cfg = resolve_model_contract(
                cfg=run_cfg,
                mode="auto_search",
                model_id=run_cfg.get("model_id"),
            )
        # -------------------------------------------------------------
        # Freeze the scientific AutoSearch contract before SCC execution.
        #
        # train/test may legitimately mutate their working configuration,
        # but final lifecycle reporting must use the concrete scientific
        # contract selected by AutoSearch.
        #
        # In particular:
        #
        #   key_enumeration.mode = comparison
        #
        # must survive even though comparison later freezes an ordinary
        # strategy such as local_hamming for independent test evaluation.
        # -------------------------------------------------------------
        reporting_cfg = copy.deepcopy(
            run_cfg
        )
        model_name, model_hash = _get_frozen_identity(run_cfg)

        run_cfg["frozen_identity"] = {
            "model_name": str(model_name),
            "model_hash": str(model_hash),
        }
        run_cfg["model_name"] = str(model_name)
        run_cfg["model_hash"] = str(model_hash)
        run_cfg.setdefault("run", {})
        run_cfg["run"]["model_name"] = str(model_name)

        run_dir = _resolve_run_dir_from_scc_cfg(
            run_cfg,
            model_name=str(model_name),
            model_hash=str(model_hash),
        )

        project_root = Path(run_cfg["project"]["root"]).resolve()

        _ensure_train_registry_entry(
            run_cfg=run_cfg,
            model_hash=str(model_hash),
            model_name=str(model_name),
            run_dir=str(run_dir),
        )

        run_cfg.setdefault("runtime", {})
        run_cfg["runtime"]["model_name"] = str(model_name)
        run_cfg["runtime"]["model_hash"] = str(model_hash)
        run_cfg["runtime"]["run_dir"] = str(run_dir)

        run_cfg.setdefault("logging", {})
        if isinstance(run_cfg["logging"], dict):
            run_cfg["logging"]["log_dir"] = str(run_dir)

        update_status(
            project_root=project_root,
            model_id=str(model_hash),
            new_status="in_progress",
        )

        configure_global_logging(
            cfg=run_cfg,
            log_dir=str(run_dir),
            model_hash=str(model_hash),
        )

        try:
            print(
                "[scc_bridge_lite] dataset selector:",
                {
                    "dataset": run_cfg.get("dataset"),
                    "dataset_name": run_cfg.get("dataset_name"),
                    "dataset_alias": run_cfg.get("dataset_alias"),
                    "data_keys": (
                        list((run_cfg.get("data") or {}).keys())
                        if isinstance(run_cfg.get("data"), dict)
                        else None
                    ),
                },
                flush=True,
            )

            train_success = False

            scc_train(
                cfg=run_cfg,
                exec_process="train",
                override_model_name=None,
                model_id=run_cfg.get("model_id"),
            )

            train_success = True

            update_status(
                project_root=project_root,
                model_id=str(model_hash),
                new_status="DONE",
            )

            if str(mode).strip().lower() == "orchestrate":
                if not train_success:
                    raise RuntimeError(
                        f"Refusing to test model_hash={model_hash}: "
                        "training did not complete."
                    )

                _inject_test_model_name(run_cfg, str(model_name))

                _ensure_test_registry_entry(
                    run_cfg=run_cfg,
                    model_hash=str(model_hash),
                    model_name=str(model_name),
                    run_dir=str(run_dir),
                )

                update_test_status(
                    project_root=project_root,
                    model_id=str(model_hash),
                    new_status="in_progress",
                )

                scc_test(
                    cfg=run_cfg,
                    exec_process="test",
                    override_model_name=None,
                    model_id=run_cfg.get("model_id"),
                )

                update_test_status(
                    project_root=project_root,
                    model_id=str(model_hash),
                    new_status="DONE",
                )

                # Both persisted reports now exist:
                #   validation_best.json
                #   test.json
                #
                # Print exactly one combined terminal summary for this completed model.
                _emit_completed_key_recovery_summary(
                    run_cfg=reporting_cfg,
                    model_name=str(model_name),
                    model_hash=str(model_hash),
                )

            return {
                "trial_id": str(trial_id),  # runtime progress only
                "model_name": str(model_name),
                "model_hash": str(model_hash),
                "run_dir": str(run_dir),
                "status": "DONE",
                "elapsed_s": round(time.time() - started, 3),
                "metrics_dir": _resolve_metrics_dir_from_scc_cfg(run_cfg),
            }


        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"[:500]
            try:
                update_status(
                    project_root=project_root,
                    model_id=str(model_hash),
                    new_status="FAILED",
                    reason=failure,
                )
            except Exception:
                pass

            if str(mode).strip().lower() == "orchestrate":
                try:
                    update_test_status(
                        project_root=project_root,
                        model_id=str(model_hash),
                        new_status="FAILED",
                        reason=f"train/test failed: {failure}",
                    )
                except Exception:
                    pass

            raise

        finally:
            cleanup_trial_runtime()
