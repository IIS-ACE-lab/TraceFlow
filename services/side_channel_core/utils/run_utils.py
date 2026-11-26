# utils/run_utils.py
'''
This script infers the most recent model directory under `artifacts/generated_models/{model_name}`.
It skips invalid entries like files or folders named 'metrics', 'common', etc.


Author: Shraddha Mishra

'''

from utils.path_utils import resolve_path


def infer_latest_model(cfg: dict, project_root: str) -> str:
    """
    Infer the most recently modified model directory under:
      artifacts/{{env}}/generated_models/{{model_name}}

    Skips invalid entries like files or folders named 'metrics', 'common', etc.

    Args:
        project_root (str): Root directory of the project.

    Returns:
        str: Name of the latest model directory, or None if not found.
    """
    model_root = resolve_path(cfg["paths"]["generated_models"], cfg)
    if not model_root.exists():
        return None

    ignore = {"metrics", "common", "logs", "checkpoints", "graphs", "models", "data"}
    candidates = [
        d for d in model_root.iterdir()
        if d.is_dir() and d.name not in ignore
    ]

    if not candidates:
        return None

    # Sort by most recently modified
    latest = sorted(candidates, key=lambda d: d.stat().st_mtime, reverse=True)[0]
    return latest.name
def _norm(value) -> str:
    return str(value or "").strip()


def _status(info: dict) -> str:
    if not isinstance(info, dict):
        return ""
    return _norm(info.get("status")).lower()


def _entry_matches_env(info: dict, env: str | None) -> bool:
    """
    Match registry entry to requested env.

    Prefer explicit info['env']; fallback to path inspection for older entries.
    """
    if not isinstance(info, dict):
        return False

    env = _norm(env)
    if not env:
        return True

    entry_env = _norm(info.get("env"))
    if entry_env:
        return entry_env == env

    path = _norm(info.get("path"))
    return (
        f"/{env}/" in path
        or path.startswith(f"{env}/")
        or path.startswith(f"artifacts/{env}/")
    )


def _entry_matches_model_id(info: dict, requested_model_id: str | None) -> bool:
    """
    If MODEL_ID is explicitly requested, registry entries must carry the same model_id.

    Important:
    - Unknown/legacy model_id must NOT match when MODEL_ID is supplied.
    - This prevents MODEL_ID=CNN1 from selecting old MLP1 entries.
    """
    requested = _norm(requested_model_id)
    if not requested:
        return True

    if not isinstance(info, dict):
        return False

    actual = _norm(info.get("model_id"))
    return bool(actual) and actual == requested

def _entry_matches_name_or_hash(
    model_hash: str,
    info: dict,
    include_models: list[str] | None,
) -> bool:
    if not include_models:
        return True

    include_set = {_norm(x) for x in include_models if _norm(x)}
    if not include_set:
        return True

    model_name = _norm(info.get("model_name")) if isinstance(info, dict) else ""
    return _norm(model_hash) in include_set or model_name in include_set


def _entry_matches_scope(
    model_hash: str,
    info: dict,
    *,
    env: str | None,
    model_id: str | None,
    include_models: list[str] | None = None,
) -> bool:
    return (
        isinstance(info, dict)
        and _entry_matches_env(info, env)
        and _entry_matches_model_id(info, model_id)
        and _entry_matches_name_or_hash(model_hash, info, include_models)
    )


def _select_registry_models(
    registry: dict,
    *,
    env: str | None,
    model_id: str | None,
    include_models: list[str] | None,
    statuses: set[str] | None,
) -> list[tuple[str, dict]]:
    """
    Select registry entries within the current orchestration scope.

    Scope includes:
    - env
    - model_id
    - optional model_name/model_hash include filter
    - optional status filter
    """
    selected: list[tuple[str, dict]] = []

    for h, info in registry.items():
        if not _entry_matches_scope(
            h,
            info,
            env=env,
            model_id=model_id,
            include_models=include_models,
        ):
            continue

        if statuses is not None and _status(info) not in statuses:
            continue

        selected.append((h, info))

    return selected
