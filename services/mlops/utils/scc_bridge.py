# services/mlops/utils/scc_bridge.py
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# --- MLOps config resolver (your plugin config loader) ---
from services.mlops.utils.auto_config import MLOpsConfig

# --- SCC integration points (NO SCC refactors required) ---
from services.side_channel_core.utils.sweep_engine import determine_run_identity
from services.side_channel_core.src.sca.train import train as scc_train
from services.side_channel_core.utils.timestamps import now_iso
from services.side_channel_core.utils.seal import load_json, seal_text

# Optional: best-effort path resolution (don’t hard-fail if SCC utils move)
try:
    from services.side_channel_core.utils.path_utils import resolve_path as scc_resolve_path
except Exception:  # pragma: no cover
    scc_resolve_path = None

# Optional: pruned detection (Optuna)
try:
    import optuna
    _OPTUNA_PRUNED_EXC = optuna.TrialPruned
except Exception:  # pragma: no cover
    _OPTUNA_PRUNED_EXC = type("TrialPruned", (Exception,), {})


# =========================
# Trial Registry (MLOps-owned)
# =========================

@dataclass(frozen=True)
class TrialRegistryKeys:
    TRIAL_ID: str = "trial_id"
    HYPOTHESIS_ID: str = "hypothesis_id"
    SOTA_TRIAL: str = "sota_trial"
    MODEL_HASH: str = "model_hash"
    MODEL_NAME: str = "model_name"
    STATUS: str = "status"
    CREATED_AT: str = "created_at"
    UPDATED_AT: str = "updated_at"
    LAST_ERROR: str = "last_error"
    LAST_TRACEBACK: str = "last_traceback"


class TrialStatus:
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    PRUNED = "PRUNED"


def _now_iso() -> str:
    return now_iso()


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """
    Atomic write (same filesystem): write to temp, fsync, replace.
    """
    _ensure_parent(path)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{int(time.time()*1000)}")
    data = seal_text(path, json.dumps(payload, indent=2, sort_keys=True))  # utils/seal.py
    with tmp.open("w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return load_json(path)  # utils/seal.py: verified, seal key dropped
    except json.JSONDecodeError:
        # salvage: keep a backup and start fresh (never crash the orchestration)
        broken = path.with_suffix(path.suffix + f".broken.{int(time.time())}")
        try:
            os.replace(path, broken)
        except Exception:
            pass
        return {}


def _with_file_lock(lock_path: Path, fn, *args, **kwargs):
    """
    Simple cross-process lock on Linux via fcntl.
    If fcntl isn't available, we still run (best-effort).
    """
    _ensure_parent(lock_path)
    try:
        import fcntl  # Linux-only
        with lock_path.open("w", encoding="utf-8") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            return fn(*args, **kwargs)
    except Exception:
        return fn(*args, **kwargs)


def _registry_update(
    registry_path: Path,
    *,
    trial_id: str,
    hypothesis_id: str,
    sota_trial: Optional[str],
    model_hash: str,
    model_name: str,
    status: str,
    last_error: Optional[str] = None,
    last_traceback: Optional[str] = None,
) -> dict:
    """
    Update the MLOps trial registry keyed by trial_id.
    Collision guards:
      - trial_id must map to only ONE model_hash forever.
      - multiple trial_ids may map to the same model_hash (allowed).
    """
    keys = TrialRegistryKeys()
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")

    def _do():
        reg = _load_json(registry_path)
        now = _now_iso()

        entry = reg.get(str(trial_id))
        if entry is None:
            entry = {
                keys.TRIAL_ID: str(trial_id),
                keys.HYPOTHESIS_ID: str(hypothesis_id),
                keys.SOTA_TRIAL: str(sota_trial) if sota_trial is not None else None,
                keys.MODEL_HASH: str(model_hash),
                keys.MODEL_NAME: str(model_name),
                keys.STATUS: status,
                keys.CREATED_AT: now,
                keys.UPDATED_AT: now,
                keys.LAST_ERROR: last_error,
                keys.LAST_TRACEBACK: last_traceback,
            }
            reg[str(trial_id)] = entry
        else:
            # HARD GUARD: trial_id cannot mutate to a different model_hash
            prev_hash = entry.get(keys.MODEL_HASH)
            if prev_hash and str(prev_hash) != str(model_hash):
                raise RuntimeError(
                    f"[auto_mlops] Trial registry collision: trial_id={trial_id} "
                    f"already mapped to model_hash={prev_hash}, got model_hash={model_hash}"
                )
            # Update fields
            entry[keys.HYPOTHESIS_ID] = str(hypothesis_id)
            entry[keys.SOTA_TRIAL] = str(sota_trial) if sota_trial is not None else entry.get(keys.SOTA_TRIAL)
            entry[keys.MODEL_HASH] = str(model_hash)
            entry[keys.MODEL_NAME] = str(model_name)
            entry[keys.STATUS] = status
            entry[keys.UPDATED_AT] = now
            if last_error is not None:
                entry[keys.LAST_ERROR] = last_error
            if last_traceback is not None:
                entry[keys.LAST_TRACEBACK] = last_traceback

        _atomic_write_json(registry_path, reg)
        return reg[str(trial_id)]

    return _with_file_lock(lock_path, _do)


# =========================
# Env scoping (clean + reversible)
# =========================

class _ScopedEnv:
    def __init__(self, updates: Dict[str, str]):
        self.updates = updates
        self.prev: Dict[str, Optional[str]] = {}

    def __enter__(self):
        for k, v in self.updates.items():
            self.prev[k] = os.environ.get(k)
            os.environ[k] = str(v)
        return self

    def __exit__(self, exc_type, exc, tb):
        for k, old in self.prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        return False


# =========================
# Bridge: production-grade trial runner
# =========================

def _resolve_metrics_dir_from_scc_cfg(run_cfg: dict) -> Optional[str]:
    """
    Best-effort: return the resolved metrics directory.
    This is helpful for Optuna to read final objective from SCC outputs.
    """
    # If SCC resolve_path exists, use it (handles {{env}}, {{model_name}}, etc.)
    if scc_resolve_path is not None:
        try:
            return str(scc_resolve_path(run_cfg["paths"]["metrics"], run_cfg))
        except Exception:
            pass

    # Fallback: try to construct from artifacts_root
    try:
        art_root = run_cfg.get("paths", {}).get("artifacts_root")
        if art_root:
            # If still templated, this might not resolve perfectly—but it’s still a pointer.
            return str(Path(art_root) / "metrics")
    except Exception:
        return None

    return None


def run_trial_via_scc(
    *,
    run_cfg: dict,
    mode: str,
    trial_id: str,
    hypothesis_id: str,
    sota_trial: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Production-grade bridge:
      - Sets AUTO_MLOPS env markers (scoped, restored after call)
      - Uses SCC's identity (model_hash/name) via determine_run_identity()
      - Creates/updates MLOps trial registry (trial_id -> model_hash/name + lifecycle)
      - Calls SCC train() without override_model_name (SCC remains owner of naming)
      - Returns identifiers + metrics pointer for Optuna integration

    NO SCC fundamental changes required.
    """

    # 0) Validate minimum contract
    if not isinstance(run_cfg, dict):
        raise TypeError("run_cfg must be a dict")
    if "project" not in run_cfg or "paths" not in run_cfg:
        raise ValueError("run_cfg must contain at least ['project', 'paths']")
    if "root" not in run_cfg["project"]:
        raise ValueError("run_cfg['project'] must include 'root'")

    # 1) Resolve MLOps registry path via plugin config
    #    You said: use MLOpsConfig.get_map(...) for any key access.
    cfg = MLOpsConfig()
    registry_path = cfg.get_map(
        "auto_mlops",
        key_map="paths.auto_trial_registry",
        project_name=run_cfg["project"].get("name"),
        resolve_as_path=True,
        mkdir=True,
        strict=True,
    )
    registry_path = Path(registry_path)

    # 2) Scoped env markers (visible to SCC logs/traceability)
    env_updates = {
        "AUTO_MLOPS": "1",
        "AUTO_MLOPS_TRIAL_ID": str(trial_id),
        "AUTO_MLOPS_HYPOTHESIS_ID": str(hypothesis_id),
    }
    if sota_trial is not None:
        env_updates["AUTO_MLOPS_SOTA_TRIAL"] = str(sota_trial)

    started = time.time()
    with _ScopedEnv(env_updates):
        # 3) Let SCC compute identity (hash/name) deterministically from run_cfg
        model_name, model_hash = determine_run_identity(run_cfg, override_model_name=None)

        # 4) Registry: CREATED -> RUNNING (CREATED is useful if you ever pre-materialize trials)
        _registry_update(
            registry_path,
            trial_id=str(trial_id),
            hypothesis_id=str(hypothesis_id),
            sota_trial=sota_trial,
            model_hash=model_hash,
            model_name=model_name,
            status=TrialStatus.RUNNING,
        )

        # 5) Execute SCC (no override_model_name: SCC is sovereign)
        try:
            scc_train(
                cfg=run_cfg,
                mode="cli",
                exec_process=mode,
                override_model_name=None,
                model_id=run_cfg.get("model_id"),
            )

            _registry_update(
                registry_path,
                trial_id=str(trial_id),
                hypothesis_id=str(hypothesis_id),
                sota_trial=sota_trial,
                model_hash=model_hash,
                model_name=model_name,
                status=TrialStatus.DONE,
                last_error=None,
                last_traceback=None,
            )

            return {
                "trial_id": str(trial_id),
                "hypothesis_id": str(hypothesis_id),
                "sota_trial": str(sota_trial) if sota_trial is not None else None,
                "model_name": model_name,
                "model_hash": model_hash,
                "status": TrialStatus.DONE,
                "elapsed_s": round(time.time() - started, 3),
                "metrics_dir": _resolve_metrics_dir_from_scc_cfg(run_cfg),
                "trial_registry_path": str(registry_path),
            }

        except _OPTUNA_PRUNED_EXC as e:
            # If your pruning layer raises TrialPruned, we must persist PRUNED
            _registry_update(
                registry_path,
                trial_id=str(trial_id),
                hypothesis_id=str(hypothesis_id),
                sota_trial=sota_trial,
                model_hash=model_hash,
                model_name=model_name,
                status=TrialStatus.PRUNED,
                last_error=str(e),
                last_traceback=None,
            )
            raise

        except Exception as e:
            tb = traceback.format_exc(limit=200)
            _registry_update(
                registry_path,
                trial_id=str(trial_id),
                hypothesis_id=str(hypothesis_id),
                sota_trial=sota_trial,
                model_hash=model_hash,
                model_name=model_name,
                status=TrialStatus.FAILED,
                last_error=str(e),
                last_traceback=tb,
            )
            raise
