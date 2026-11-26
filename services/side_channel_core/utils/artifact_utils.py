#services/side_channel_core/utils/artifact_utils.py
""" 
Generate and manage run directories for SCA models.
This module provides functions to prepare run directories for SCA models,
ensuring that the correct structure is created and maintained.
It handles the creation of subdirectories, renaming of directories when model names change,
and the management of a registry that tracks model hashes and their associated run directories.
It also provides functionality to list saved parameter configurations for models.

functions: prepare_run_dir, get_run_dirs


Author: Shraddha Mishra

"""


from pathlib import Path
from utils.logger import get_logger
from utils.path_utils import resolve_path
from .sweep_engine import load_registry, save_registry
import shutil, os

logger = get_logger(__name__)
def _canonical_registry_path(*, project_root: Path, run_dir: Path) -> str:
    """
    Store registry paths in a single canonical form:
    'projects/<project_name>/artifacts/<env>/generated_models/<model_name>'
    """
    # project_root is .../projects/DLSCA_ASCAD
    # so parent is .../projects
    projects_root = project_root.parent

    try:
        rel = run_dir.resolve().relative_to(projects_root.resolve())
        return str(rel)
    except Exception:
        # fallback: relative to project_root
        try:
            rel = run_dir.resolve().relative_to(project_root.resolve())
            # prefix with project name for stability
            return str(Path(project_root.name) / rel)
        except Exception:
            return str(run_dir)


def _checkpointing(cfg: dict) -> bool:
    """evaluation.validation.checkpointing; True when unreadable (the training loop reports that)."""
    try:
        from utils.checkpoint_policy import get_validation_checkpointing

        return get_validation_checkpointing(cfg)
    except Exception:
        return True


def prepare_run_dir(
    cfg : dict,
    project_root: Path,
    model_hash:    str,
    model_name:    str,
    subdirs:      tuple[str,...] = ("logs","graphs","model","checkpoints","data","metrics","config"),
) -> Path:
    """
        Ensures:
            - Correct env + model_name resolved
            - Registry handles rename
            - Old folder cleanup if needed
            - Creates subdirs if missing
            - Cleans stale metric CSVs

    Returns: Path to the active run directory.
            Make sure there’s exactly one directory under
                <project_root>/artifacts/{env}/generated_models/{*old* or new name}
            for this model_hash.

            If this hash was already in the registry under a *different* name,
            we only update the registry to point at the *new* model_name,
            but we continue to write into the *old* folder on disk.

            Returns the Path to the run_dir (always the on-disk folder
            where metrics/logs/etc live).
    """
        # 1. Finalize env + model_name in config

    env = cfg.get("env", os.environ.get("ENV", "dev"))
    cfg["model_name"] = model_name
    cfg["env"] = env

    artifacts_root = Path(resolve_path(cfg["paths"]["artifacts_root"], cfg, model_name=model_name)).resolve()

    # RERUN / isolated replay path: do not touch main registry-driven run root
    if cfg.get("bypass_registry", False):
        artifacts_root.mkdir(parents=True, exist_ok=True)
        for sd in subdirs:
            (artifacts_root / sd).mkdir(parents=True, exist_ok=True)

        for process in ("train", "validation", "test"):
            hist = artifacts_root / "metrics" / f"{model_name}_metrics_{process}_history.csv"
            if hist.exists():
                hist.unlink()

        return artifacts_root

    registry = load_registry(project_root)
    entry = registry.get(model_hash)

    if entry is not None:
        old_name = entry.get("model_name")
        old_path = entry.get("path")

        old_abs = None
        if isinstance(old_path, str) and old_path:
            projects_root = project_root.parent
            cand = (projects_root / old_path).resolve()
            if cand.exists():
                old_abs = cand

        if old_name and old_name != model_name and old_abs and old_abs.exists():
            logger.info("Model name changed for same hash: %s → %s. Removing old dir: %s",
                        old_name, model_name, old_abs)
            shutil.rmtree(old_abs, ignore_errors=True)

    # Update in place: the planner stores run_cfg, model_id, env and SCA
    # fields here, and test-cmd rebuilds the test config from run_cfg.
    current = registry.get(model_hash)
    if not isinstance(current, dict):
        current = {"status": "NEW"}
    current["model_name"] = model_name
    current["path"] = _canonical_registry_path(project_root=project_root, run_dir=artifacts_root)
    registry[model_hash] = current
    save_registry(registry, project_root)

    artifacts_root.mkdir(parents=True, exist_ok=True)
    for sd in subdirs:
        if sd == "checkpoints" and not _checkpointing(cfg):
            continue                    # nothing is written there (checkpoint_policy.clear_checkpoint_artifacts)
        (artifacts_root / sd).mkdir(parents=True, exist_ok=True)

    for process in ("train", "validation", "test"):
        hist = artifacts_root / "metrics" / f"{model_name}_metrics_{process}_history.csv"
        if hist.exists():
            hist.unlink()

    return artifacts_root

def get_run_dirs(cfg: dict) -> dict:
    run_dir = resolve_path(cfg["paths"]["artifacts_root"], cfg)
    return {
        "root": run_dir,
        "logs": run_dir / "logs",
        "checkpoints": run_dir / "checkpoints",
        "model": run_dir / "model",
        "metrics": run_dir / "metrics",
        "model_graphs": run_dir / "graphs",
        "data": run_dir / "data",
    }
