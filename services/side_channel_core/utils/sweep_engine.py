# services/side_channel_core/utils/sweep_engine.py
"""
Run a sweep of parameter combinations for training or testing SCA models.

This module provides functions to:
- generate parameter combinations from a base YAML config (sweep)
- check run statuses
- manage the run registry (run_registry.json)
- filter runs based on status / include filters
- update registry status and names

Trainer grid syntax:
- epochs: [75, 100] and optimizer.learning_rate: [1e-5, 2e-5]
- optimizer.args.betas: [0.9, 0.999] is one literal pair
- optimizer.args.betas: [[0.9, 0.999], [0.8, 0.99]] sweeps two pairs
- optimizer/loss/scheduler may be lists of complete component mappings;
  each alternative expands only its own arguments
- disabled scheduler arguments are excluded from the grid

CRITICAL REGISTRY CONTRACT (FIXED):
- load_registry() ALWAYS returns the FULL registry.
- env scoping is READ-ONLY via scope_registry_by_env().
- save_registry() enforces stable, permanent "env" for every entry.
- registry entries include per-entry timestamps:
    started_at, finished_at (local timezone, ISO8601 with offset)
- registry JSON order:
    RUNNING first, QUEUED second, then everything else; latest-first inside each group

Author: Shraddha Mishra
"""
from __future__ import annotations

import copy
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from services.side_channel_core.utils.timestamps import now_iso
from services.side_channel_core.utils.atomic_io import atomic_write_text
from services.side_channel_core.utils.run_lifecycle_logger import (
    start_run_map_entry,
    sync_run_map_before_planning,
)
from services.side_channel_core.src.core.model_contract_resolver import resolve_model_contract
from services.side_channel_core.src.core.framework_switch import optimizer_args_table, wire_optimizer_args
from services.side_channel_core.utils.sca_sweep import (
    SCASweepError,
    assert_concrete_sca_runtime,
    count_sca_combinations,
    iter_sca_combinations,
)
from utils.config_loader import get_project_config
from utils.logger import get_logger, pretty_print
from services.side_channel_core.utils.model_naming import _load_mapping, _save_mapping, generate_unique_model_name, make_experiment_hash
from utils.type_coerce import coerce_types
from utils.batch import add_batch, current_batch, held_by_another_batch, stamp_registry
from services.side_channel_core.utils.run_map import load_run_map, sample_key
from services.side_channel_core.utils.seal import load_json
from utils.lifecycle_outcome import begin_attempt
REGISTRY_FILENAME = "run_registry.json"
logger = get_logger("SweepEngine")


# =============================================================================
# Time + status helpers
# =============================================================================

def _now_local_iso() -> str:
    """Current time, ISO 8601 in the configured timezone (utils/timestamps.py)."""
    return now_iso()


def _status_norm(s: str) -> str:
    return (s or "").strip().lower()
def _scc_sampled_payload(run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    from sca.rank_curve import PUBLIC_RANK_BASE

    sca = copy.deepcopy(run_cfg.get("sca") or {})
    # sca.rank_base is no longer a setting (2026-10-19: TraceFlow ranks from 1, rank_curve refuses any
    # other base); the sweep point keeps it, so a run's sampled_key stays the one runs made with it had.
    sca.setdefault("rank_base", PUBLIC_RANK_BASE)
    return {
        "mode": "scc",
        "model_id": run_cfg.get("model_id"),
        "trainer": copy.deepcopy(run_cfg.get("trainer") or {}),
        "sca": sca,
        "model": copy.deepcopy(run_cfg.get("model") or []),
    }

def _is_running_status(s: str) -> bool:
    return _status_norm(s) in {"in_progress", "running"}


def _is_queued_status(s: str) -> bool:
    return _status_norm(s) in {"new", "queued", "pending"}


def _is_terminal_status(s: str) -> bool:
    return _status_norm(s) in {"done", "failed", "interrupted", "cancelled", "canceled"}

# The one list under a data entry that sweeps (2026-10-24): data.profiling_split.random_state, a run per split (Wouters
# et al. TCHES 2020(3) Sect. 3.2: every configuration trained on ten splits). Its prefixes stay open so the walk reaches it.
_DATA_SWEEP_PATHS = frozenset({("data",), ("data", "profiling_split"), ("data", "profiling_split", "random_state")})
SPLIT_SWEEP_PATH = ("data", "profiling_split", "random_state")

_NON_SWEEP_DOMAINS = frozenset({
    "sca", "trainer", "graph", "test", "visualization", "paths", "project",
    "cross_validation", "clearml", "logging", "distributed",
    "import", "model_import",       # import_config.yaml, and an imported run's record (utils/model_import.py)
})


def _is_ignored_sweep_path(
    path: Tuple[Any, ...],
) -> bool:
    """
    Return True when a path must not be handled by the generic SCC Cartesian
    sweep engine.

    Dedicated or structural domains excluded from generic list expansion:

    1. ``sca``
       SCA contains conditional dimensions. For example, iterative enumeration
       does not use hybrid/support profile parameters. The entire SCA subtree
       is therefore compiled by utils.sca_sweep instead of generic list
       expansion.

    2. ``trainer`` is expanded by the trainer-aware compiler below.

    3. ``model[*].architecture_args.search_space``
       Architecture search space belongs to AutoSearch/OptunaLite rather than
       deterministic SCC grid expansion.

    4. Reporting/tooling blocks merged from extra_configs (``graph``,
       ``test``, ``import``), an imported run's ``model_import`` record
       (its released file's facts) and runtime plumbing (``visualization``, ``paths``,
       ``project``, ``cross_validation``). Their lists are plot/model
       selections, not experiment choices; expanding them multiplied
       ``trial_total`` and re-ran the same model hash.

    5. Graph models (``model[*].architecture_args.nodes`` / ``outputs``): a
       node's ``inputs``, the ``outputs`` list and every list inside a node
       (e.g. ZeroPad1d's ``padding: [left, right]``) are the wiring of the
       network, not experiment choices. A ZeroPad1d ``padding`` pair in a
       layer list is likewise one value.

    Dataset ranges, a dataset's trace_normalization (its fitted scalers'
    ``steps`` chain and ``feature_range`` pair), metric lists and
    perform_attack.ge_thresholds are structural values, not sweep choices.
    """
    # cross_validation.partition_seed alone is an experiment choice: a list of them is a sweep over
    # fold partitions (the crossed folds x seeds design); every other cross_validation list is not.
    if path[:1] == ("cross_validation",):
        return path not in {("cross_validation",), ("cross_validation", "partition_seed")}
    if path and path[0] in _NON_SWEEP_DOMAINS:
        return True
    # Nothing else under data sweeps: a run takes one entry, chosen by DATASET, and the planner keeps that entry
    # as written (_scc_identity), so a list in it is a value (ranges, a normalization chain, target points), never
    # a grid; counted as a grid it planned runs that could not be made (audit 2026-10-23). The exception is the
    # selected entry's profiling_split.random_state (_DATA_SWEEP_PATHS): a list of splits, one run each.
    if path and path[0] == "data":
        return path not in _DATA_SWEEP_PATHS
    if path and path[0] in {"metrics", "metrics_to_log"}:
        return True
    # perform_attack reads ge_thresholds as the set of thresholds to report,
    # so the list is a value, not a grid of one-threshold runs.
    if path and path[0] == "perform_attack" and path[-1] == "ge_thresholds":
        return True

    if "architecture_args" in path and ("nodes" in path or "outputs" in path):
        return True
    # deep_learning.<framework>.optimizer_args are trainer arguments: the trainer
    # compiler expands them once wired into trainer.optimizer.args (pairs kept).
    if path[:1] == ("deep_learning",) and "optimizer_args" in path:
        return True
    if "ZeroPad1d" in path and path[-1] == "padding":
        return True

    return (
        "architecture_args" in path
        and "search_space" in path
    )
# =============================================================================
# Env helpers (permanent env in registry)
# =============================================================================
def _env_from_path(p: str) -> str:
    """
    Infer env from paths like:
      - "DLSCA_ASCAD/artifacts/dev/generated_models/..."
      - "/app/projects/DLSCA_ASCAD/artifacts/dev/generated_models/..."
      - "artifacts/dev/generated_models/..."
    """
    if not isinstance(p, str):
        return ""
    marker = "/artifacts/"
    if marker in p:
        rest = p.split(marker, 1)[1]          # "dev/generated_models/..."
        return rest.split("/", 1)[0].strip()  # "dev"

    marker2 = "artifacts/"
    if marker2 in p:
        rest = p.split(marker2, 1)[1]
        return rest.split("/", 1)[0].strip()

    return ""


def _canonical_env(cfg: Dict[str, Any], run_dir: Optional[Path] = None) -> str:
    """
    Deterministic env resolution (never None):
      1) cfg["env"]
      2) os.getenv("ENV")
      3) derive from run_dir or cfg["paths"]["artifacts_root"]
      4) "dev"
    """
    v = str(cfg.get("env") or "").strip()
    if v:
        return v

    v = str(os.getenv("ENV") or "").strip()
    if v:
        return v

    if run_dir is not None:
        v = _env_from_path(str(run_dir))
        if v:
            return v

    paths = cfg.get("paths") or {}
    ar = paths.get("artifacts_root")
    if isinstance(ar, (str, Path)):
        v = _env_from_path(str(ar))
        if v:
            return v

    return "dev"


def scope_registry_by_env(registry: Dict[str, Any], env: str) -> Dict[str, Any]:
    """
    READ-ONLY view. NEVER save() the output of this function.
    Keeps entries whose env matches:
      - entry["env"] == env
      - OR inferred env_from_path(entry["path"]) == env
    """
    env = str(env).strip()
    if not env:
        return dict(registry)

    scoped: Dict[str, Any] = {}
    for k, v in registry.items():
        if not isinstance(v, dict):
            continue
        entry_env = str(v.get("env") or "").strip()
        if entry_env == env:
            scoped[k] = v
            continue
        p = v.get("path", "")
        if _env_from_path(str(p)) == env:
            scoped[k] = v
    return scoped


# =============================================================================
# Registry I/O (atomic + env stabilization + ordering)
# =============================================================================

def _registry_path(project_root: Path) -> Path:
    return Path(project_root) / "artifacts" / REGISTRY_FILENAME


def _normalize_registry_entry(
    model_hash: str,
    info: Dict[str, Any],
    project_root: Path,
) -> Dict[str, Any]:
    """
    In-place normalization:
    - ensure 'path' is project-root-relative if possible
    - ensure 'env' exists and is stable
    - ensure timestamps make sense:
        * running => started_at exists
        * terminal => finished_at exists
    """
    # The registry holds lifecycle metadata only; the resolved config lives in
    # the run's params_out snapshot. Drop copies written by older versions.
    info.pop("run_cfg", None)

    # normalize path
    p = info.get("path")
    if isinstance(p, (str, Path)):
        pth = Path(str(p))
        if pth.is_absolute():
            try:
                info["path"] = str(pth.relative_to(project_root))
            except ValueError:
                logger.warning(f"Registry path is absolute and outside project_root: {pth}")

    # enforce env permanence (staging/prod bug fix: fall back to ENV too)
    env_val = str(info.get("env") or "").strip()
    if not env_val:
        env_val = (
            _env_from_path(str(info.get("path", "")))
            or str(os.getenv("ENV") or "").strip()
            or "dev"
        )
        info["env"] = env_val

        # enforce model_id permanence
    model_id_val = _resolve_model_id_for_registry_entry(
        model_hash=str(model_hash),
        info=info,
        project_root=project_root,
    )
    if model_id_val:
        info["model_id"] = model_id_val
    # normalize status casing (we keep original values but timestamps depend on it)
    status = str(info.get("status") or "")

    # running => started_at
    if _is_running_status(status) and not str(info.get("started_at") or "").strip():
        info["started_at"] = _now_local_iso()

    # terminal => finished_at
    if _is_terminal_status(status) and not str(info.get("finished_at") or "").strip():
        info["finished_at"] = _now_local_iso()

    return info

def _resolve_model_id_for_registry_entry(
    *,
    model_hash: str,
    info: Dict[str, Any],
    project_root: Path,
) -> str:
    """
    Resolve architecture model_id for a registry entry.

    Priority:
    1. existing entry["model_id"] (never overwritten)
    2. run_map[model_hash]["model_id"] (recorded for this hash)
    3. environment MODEL_ID (the calling process; last resort)

    This makes model_id a registry invariant, not a caller-dependent field.
    """
    existing = str(info.get("model_id") or "").strip()
    if existing:
        return existing

    rec = load_run_map(project_root).get(str(model_hash)) or {}
    mapped = str(rec.get("model_id") or "").strip()
    if mapped:
        return mapped

    return str(os.getenv("MODEL_ID") or "").strip()

def _entry_latest_ts(entry: Dict[str, Any]) -> str:
    # ISO8601 sorts lexicographically for our use-case
    return str(entry.get("finished_at") or entry.get("started_at") or "").strip()


def _group_rank(status: str) -> int:
    # lower is earlier in JSON
    if _is_running_status(status):
        return 0
    if _is_queued_status(status):
        return 1
    return 2


def _reorder_registry_for_write(registry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a NEW dict with insertion order:
      - running first (latest first)
      - queued second (latest first)
      - then everything else (latest first)
    """
    items: List[Tuple[str, Dict[str, Any]]] = [
        (k, v) for k, v in registry.items() if isinstance(v, dict)
    ]

    # stable multi-pass sort:
    # 1) latest timestamp desc
    items.sort(key=lambda kv: _entry_latest_ts(kv[1]), reverse=True)
    # 2) group rank asc (running, queued, rest)
    items.sort(key=lambda kv: _group_rank(str(kv[1].get("status") or "")))

    out: Dict[str, Any] = {}
    for k, v in items:
        out[k] = v
    return out


def load_registry(project_root: Path, *, heal: bool = True) -> Dict[str, Any]:
    """
    Load FULL registry from <project_root>/artifacts/run_registry.json.

    If heal=True (default), auto-repair missing env / absolute paths / timestamps / junk entries
    by re-saving through save_registry(). This defeats "env comes and goes" caused
    by other writers that bypass save_registry().
    """
    reg_path = _registry_path(project_root)
    if not reg_path.exists():
        return {}

    # A damaged registry raises: returning {} would let the next save
    # rewrite it with only the current run, deleting every other entry.
    try:
        data = load_json(reg_path)  # utils/seal.py: verified, seal key dropped
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{reg_path} is not valid JSON ({e}); fix it before running. The run "
            "registry is not rewritten while it is damaged."
        ) from e

    if not isinstance(data, dict):
        raise ValueError(f"{reg_path} must hold a JSON object keyed by model_hash")

    if not heal:
        return data

    before = json.dumps(data, sort_keys=True, default=str)

    cleaned: Dict[str, Any] = {}
    for model_hash, info in data.items():
        if not isinstance(info, dict):
            continue
        cleaned[model_hash] = _normalize_registry_entry(
            model_hash=str(model_hash),
            info=info,
            project_root=Path(project_root),
        )

    # also enforce ordering on heal
    cleaned = _reorder_registry_for_write(cleaned)

    after = json.dumps(cleaned, sort_keys=True, default=str)

    if after != before:
        save_registry(cleaned, project_root)
        return cleaned

    return data


def save_registry(registry: Dict[str, Any], project_root: Path) -> None:
    """
    Save registry atomically to <project_root>/artifacts/run_registry.json.

    HARD GUARANTEE:
    - every entry has stable "env"
    - paths are made project-relative when possible
    - timestamps are present where applicable
    - JSON insertion order: running first, queued next, latest-first inside each
    """
    project_root = Path(project_root)
    art_dir = project_root / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    # normalize entries
    cleaned: Dict[str, Any] = {}
    for model_hash, info in registry.items():
        if not isinstance(info, dict):
            continue
        cleaned[model_hash] = _normalize_registry_entry(
            model_hash=str(model_hash),
            info=info,
            project_root=project_root,
        )

    # reorder
    cleaned = _reorder_registry_for_write(cleaned)

    reg_path = _registry_path(project_root)

    try:
        atomic_write_text(reg_path, json.dumps(cleaned, indent=2))  # DO NOT sort_keys; we want order
    except Exception as e:
        logger.error(f"Failed to save registry: {reg_path} ({e})")
        raise


def is_done(model_id: str, project_root: Path) -> bool:
    """
    Return True if this model_id is already marked DONE in the registry.
    """
    registry = load_registry(project_root)
    return str(registry.get(model_id, {}).get("status", "")).strip().upper() == "DONE"


# =============================================================================
# Sweep config discovery
# =============================================================================

def _is_sweep_list(lst: Any) -> bool:
    """
    True if lst is a non-empty list of primitives (not dicts/lists),
    i.e. something you want to sweep over.
    """
    return (
        isinstance(lst, list)
        and bool(lst)
        and all(not isinstance(el, (list, dict)) for el in lst)
    )


def _find_sweep_paths(cfg: Any, prefix: Tuple[Any, ...] = ()) -> List[Tuple[Tuple[Any, ...], List[Any]]]:
    """
    Walk dicts & lists to collect every (path, sweep_list) where sweep_list
    is a list of primitives.

    - dict: recurse into values
    - list:
        * if primitive-list => record
        * else recurse into each element by index
    """
    paths: List[Tuple[Tuple[Any, ...], List[Any]]] = []

    if _is_ignored_sweep_path(prefix):
        return paths

    if isinstance(cfg, dict):
        for key, val in cfg.items():
            path = prefix + (key,)

            if _is_ignored_sweep_path(path):
                continue

            if _is_sweep_list(val):
                paths.append((path, val))
            else:
                paths.extend(_find_sweep_paths(val, path))

    elif isinstance(cfg, list):
        if not _is_sweep_list(cfg):
            for idx, element in enumerate(cfg):
                paths.extend(_find_sweep_paths(element, prefix + (idx,)))

    return paths


# Sweep lists that move together: the i-th entries of the group form one sweep
# point (zipped), never their Cartesian product. attack_step and rank_step are
# the trace grids of the repeated-attack and the single-run rank curves; a
# point keeps one grid pair, so [5, 10] with [5, 10] gives the pairs (5, 5) and
# (10, 10), two runs, not four. A scalar in the group goes with every entry.
_LINKED_SWEEP_GROUPS: Tuple[Tuple[Tuple[str, ...], ...], ...] = (
    (("perform_attack", "attack_step"), ("validation", "rank_step")),
)


class _LinkedPaths(tuple):
    """A sweep dimension over several config paths whose values move together."""


_MISSING = object()


def _get_path(cfg: Any, path: Tuple[Any, ...]) -> Any:
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _sweep_dimensions(cfg: Any) -> List[Tuple[Tuple[Any, ...], List[Any]]]:
    """_find_sweep_paths with each linked group as one dimension of value tuples (zipped)."""
    dims = _find_sweep_paths(cfg)
    for group in _LINKED_SWEEP_GROUPS:
        values = {path: _get_path(cfg, path) for path in group}
        values = {path: value for path, value in values.items() if value is not _MISSING}
        lists = {path: value for path, value in values.items() if isinstance(value, list)}
        if not lists:
            continue                                     # one scalar each: no dimension
        dotted = {path: ".".join(map(str, path)) for path in values}
        lengths = sorted({len(value) for value in lists.values()})
        if len(lengths) > 1 or lengths[0] == 0:
            got = ", ".join(f"{dotted[path]}={len(value)}" for path, value in lists.items())
            raise ValueError(
                f"{' and '.join(dotted.values())} are one linked sweep (their i-th entries pair "
                f"up, never every combination): give lists of the same, non-zero length, or one "
                f"scalar; got lengths {got}"
            )
        present = _LinkedPaths(values)
        dims = [(path, pool) for path, pool in dims if tuple(path) not in present]
        dims.append((present, [
            tuple(value[i] if isinstance(value, list) else value for value in values.values())
            for i in range(lengths[0])
        ]))
    return dims


def _expand_trainer_value(value, path=(), optimizer_type=None, optimizer_args=None):
    """Expand trainer choices; keep optimizer pairs as complete values.

    A flat list is a choice list except betas/etas/step_sizes and Adafactor
    eps, which are pairs. Sweep pairs with [[a, b], [c, d]]. Component lists
    of mappings are complete alternatives, each with its own argument grid.
    Tuples are literal values, not search dimensions.

    optimizer_args: the selected framework's deep_learning.<framework>.optimizer_args
    (framework_switch.optimizer_args_table); each selected optimizer type gets
    its entry as trainer.optimizer.args before the arguments expand.
    """
    pair = path in {
        ("optimizer", "args", "betas"),
        ("optimizer", "args", "etas"),
        ("optimizer", "args", "step_sizes"),
    } or (path == ("optimizer", "args", "eps") and optimizer_type == "adafactor")
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Empty trainer sweep at {'.'.join(path)}")
        if pair and all(not isinstance(item, (list, tuple, dict)) for item in value):
            if len(value) != 2:
                raise ValueError(f"trainer.{'.'.join(path)} requires a pair")
            yield copy.deepcopy(value)
            return
        if pair:
            for choice in value:
                if not isinstance(choice, (list, tuple)) or len(choice) != 2:
                    raise ValueError(f"trainer.{'.'.join(path)} choices must be pairs")
                yield copy.deepcopy(choice)
            return
        for choice in value:
            yield from _expand_trainer_value(choice, path, optimizer_type, optimizer_args)
        return
    if not isinstance(value, dict):
        yield copy.deepcopy(value)
        return
    # Resolve component type before its arguments, regardless of YAML key order.
    if "type" in value:
        for selected_type in _expand_trainer_value(value["type"], path + ("type",), optimizer_type):
            if not isinstance(selected_type, str):
                raise TypeError(f"trainer.{'.'.join(path)}.type must be a string")
            remainder = {k: v for k, v in value.items() if k != "type"}
            current_optimizer = selected_type.strip().lower() if path == ("optimizer",) else optimizer_type
            if path in {("scheduler",), ("optimizer", "scheduler")} and selected_type.strip().lower() in {"none", "off", "null", ""}:
                remainder.pop("args", None)
            if path == ("optimizer",):
                remainder = wire_optimizer_args(remainder, selected_type, optimizer_args)
            for resolved in _expand_trainer_value(remainder, path, current_optimizer, optimizer_args):
                yield {"type": selected_type, **resolved}
        return
    keys = list(value)
    pools = [list(_expand_trainer_value(value[k], path + (k,), optimizer_type, optimizer_args)) for k in keys]
    for selected in itertools.product(*pools):
        yield dict(zip(keys, copy.deepcopy(selected)))


def _assert_trainer_preserved(selected, resolved, path="trainer"):
    """Reject resolver changes to already selected trainer settings."""
    if isinstance(selected, dict):
        if not isinstance(resolved, dict):
            raise ValueError(f"Resolver replaced {path} with a non-mapping")
        for key, value in selected.items():
            if key not in resolved:
                raise ValueError(f"Resolver removed {path}.{key}")
            _assert_trainer_preserved(value, resolved[key], f"{path}.{key}")
    elif selected != resolved:
        raise ValueError(f"Resolver changed selected {path}: {selected!r} -> {resolved!r}")


def generate_param_combinations(base_cfg: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield the Cartesian product of generic, trainer and conditional SCA choices.

    Component alternatives may be lists of complete mappings. Use shared args
    with a type list only when those args are valid for every selected type.
    No unsupported optimizer argument is silently discarded.
    """
    if not isinstance(base_cfg, dict):
        raise TypeError("base_cfg must be a dict")
    trainer = base_cfg.get("trainer")
    if trainer is not None and not isinstance(trainer, dict):
        raise TypeError("trainer must be a mapping")
    for generic_cfg in _generate_nontrainer_combinations(base_cfg):
        if trainer is None:
            yield generic_cfg
            continue
        for selected in _expand_trainer_value(trainer, optimizer_args=optimizer_args_table(generic_cfg)):
            concrete = copy.deepcopy(generic_cfg)
            concrete["trainer"] = selected
            yield coerce_types(concrete)


def _generate_nontrainer_combinations(
    base_cfg: Dict[str, Any],
) -> Iterator[Dict[str, Any]]:
    """
    Yield concrete SCC experiment configurations.

    Expansion occurs in two independent layers:

        generic non-SCA dimensions
            ×
        conditional SCA dimensions

    Generic dimensions
    ------------------
    Primitive list-valued parameters outside ``sca`` are expanded using the
    normal Cartesian SCC grid.

    SCA dimensions
    --------------
    The complete ``sca`` subtree is delegated to ``utils.sca_sweep`` because
    SCA contains conditional dimensions:

    - candidate_ranking.top_k is active only for mode=top_k;
    - hybrid parameters are active only for enumeration mode=hybrid;
    - iterative parameters are active only for mode=iterative;
    - support parameters are active only for mode=support.

    Every yielded configuration is therefore one scientifically concrete SCC
    experiment rather than a Cartesian combination of inactive parameters.
    """
    if not isinstance(base_cfg, dict):
        raise TypeError(
            "base_cfg must be a dict"
        )

    generic_sweep_paths = (
        _sweep_dimensions(
            base_cfg
        )
    )

    #
    # First expand ordinary non-SCA SCC parameters.
    #
    if generic_sweep_paths:
        paths, pools = zip(
            *generic_sweep_paths
        )

        generic_choices = (
            itertools.product(
                *pools
            )
        )

    else:
        paths = ()
        generic_choices = [
            (),
        ]

    for choice in generic_choices:
        generic_cfg = copy.deepcopy(
            base_cfg
        )

        for path, value in zip(
            paths,
            choice,
        ):
            if isinstance(path, _LinkedPaths):           # a linked group: one value per path
                for linked_path, linked_value in zip(path, value):
                    target = _get_path(generic_cfg, linked_path[:-1])
                    target[linked_path[-1]] = copy.deepcopy(linked_value)
                continue
            target: Any = (
                generic_cfg
            )

            for key in path[:-1]:
                target = target[key]

            target[
                path[-1]
            ] = copy.deepcopy(
                value
            )

        #
        # Then expand SCA conditionally.
        #
        sca_cfg = generic_cfg.get(
            "sca"
        )

        if sca_cfg is None:
            yield generic_cfg
            continue

        if not isinstance(
            sca_cfg,
            dict,
        ):
            raise SCASweepError(
                "cfg.sca must be a mapping"
            )

        for concrete_sca in (
            iter_sca_combinations(
                sca_cfg
            )
        ):
            concrete_cfg = (
                copy.deepcopy(
                    generic_cfg
                )
            )

            concrete_cfg[
                "sca"
            ] = concrete_sca

            #
            # Hard boundary:
            #
            # generate_param_combinations() is the last search-space layer
            # before SCC runtime contract resolution.
            #
            assert_concrete_sca_runtime(
                concrete_cfg[
                    "sca"
                ]
            )

            yield concrete_cfg
# =============================================================================
# Template resolution
# =============================================================================

def deep_template_resolve(config: Any, template_values: Dict[str, Any]) -> Any:
    """
    Recursively replace {{var}} templates in a config dict using template_values.
    Works on nested dicts/lists/Paths.
    """
    if isinstance(config, dict):
        return {k: deep_template_resolve(v, template_values) for k, v in config.items()}
    if isinstance(config, list):
        return [deep_template_resolve(i, template_values) for i in config]
    if isinstance(config, Path):
        config = str(config)

    if isinstance(config, str):
        out = config
        for key, val in template_values.items():
            out = out.replace(f"{{{{{key}}}}}", str(val))
        return out

    return config


# =============================================================================
# Pending runs + identity
# =============================================================================

def _scc_identity(
    sweep_cfg: Dict[str, Any], cfg: Dict[str, Any], project_root: Path, selected_model_id: str, requested_env: str,
    reserved: Optional[set] = None,
) -> Tuple[Dict[str, Any], str, str]:
    """
    One sweep combination resolved to its concrete SCC run config and its
    identity (run_cfg, model_name, model_hash). No side effect: get_pending_runs
    plans from it, and planned_hashes lists the hashes a pass would plan with it.
    """
    run_cfg = copy.deepcopy(sweep_cfg)
    run_cfg = coerce_types(run_cfg)

    if selected_model_id:
        run_cfg["model_id"] = selected_model_id

    run_cfg["project"] = cfg["project"]
    run_cfg["data"] = cfg["data"]
    # the one data list that sweeps: this run's split (profiling_split.random_state), the entry otherwise as written
    chosen = _get_path(sweep_cfg, SPLIT_SWEEP_PATH)
    written = _get_path(cfg, SPLIT_SWEEP_PATH)
    if chosen is not _MISSING and isinstance(written, list):
        run_cfg["data"] = copy.deepcopy(cfg["data"])
        run_cfg["data"]["profiling_split"]["random_state"] = chosen

    if "root" not in run_cfg["project"]:
        run_cfg["project"]["root"] = str(project_root)
    run_cfg["env"] = requested_env

    run_cfg = coerce_types(run_cfg)
    #
    # SCA sweep compilation must already be complete here.
    #
    # No active SCA search list may cross into model-contract resolution.
    #
    sca_runtime = (
        run_cfg.get("sca")
        or {}
    )

    assert_concrete_sca_runtime(
        sca_runtime
    )
    _lr = run_cfg["trainer"]["optimizer"]["learning_rate"]
    if isinstance(_lr, list):
        raise RuntimeError(
            "BUG: trainer.optimizer.learning_rate is still a list after "
            "generate_param_combinations(). SCC expansion failed before resolver. "
            f"learning_rate={_lr!r}"
        )

    #
    # Resolve the concrete SCC contract.
    #
    # Search-space compilation has already occurred. The model resolver
    # may normalize types/contracts, but it must never recreate an
    # unresolved SCA search dimension.
    #
    selected_trainer = copy.deepcopy(run_cfg["trainer"])
    run_cfg = resolve_model_contract(
        cfg=run_cfg,
        mode="scc",
        model_id=run_cfg.get(
            "model_id"
        ),
    )

    assert_concrete_sca_runtime(
        run_cfg.get(
            "sca"
        )
        or {}
    )

    run_cfg = coerce_types(run_cfg)
    _assert_trainer_preserved(selected_trainer, run_cfg["trainer"])

    # Final identity exactly once. No temporary model_name_map writes.
    run_cfg.pop("frozen_identity", None)
    run_cfg.pop("model_hash", None)
    run_cfg.pop("model_name", None)
    if isinstance(run_cfg.get("run"), dict):
        run_cfg["run"].pop("model_name", None)

    sampled_payload = _scc_sampled_payload(run_cfg)
    run_cfg.setdefault("runtime", {})
    if not isinstance(run_cfg["runtime"], dict):
        run_cfg["runtime"] = {}

    run_cfg["runtime"]["sampled_key"] = sample_key(sampled_payload)
    run_cfg["sampled"] = sampled_payload

    model_name, model_hash = determine_run_identity(run_cfg, persist=False, reserved=reserved)
    return run_cfg, str(model_name), str(model_hash)


def _base_cfg_for(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The project's file as the planner sweeps it, with the run's resolved attack byte: sca.attack_byte left
    empty follows the data entry (dataset_types.resolve_attack_byte wrote it into cfg at the run's start), and
    the file's empty list would be refused by the sweep (owner 2026-10-23; the planner re-reads the file)."""
    base_cfg = get_project_config(cfg["project"]["name"])
    base_sca = base_cfg.get("sca") if isinstance(base_cfg.get("sca"), dict) else None
    run_sca = cfg.get("sca") if isinstance(cfg.get("sca"), dict) else None
    if base_sca is not None and base_sca.get("attack_byte") in (None, []) and run_sca is not None and run_sca.get("attack_byte") not in (None, []):
        base_sca["attack_byte"] = copy.deepcopy(run_sca["attack_byte"])
    # the run's selected data entry (DATASET), not the file's every entry: its profiling_split.random_state list is
    # the one data list that sweeps (a run per split); the rest of data never sweeps (_is_ignored_sweep_path)
    if isinstance(cfg.get("data"), dict) and ("database" in cfg["data"] or "type" in cfg["data"]):
        base_cfg["data"] = copy.deepcopy(cfg["data"])
    return base_cfg


def planned_hashes(cfg: Dict[str, Any], project_root: Path) -> set[str]:
    """
    The model hashes one pass of this config plans (ENV, MODEL_ID and the
    project's sweep lists, as get_pending_runs resolves them), without writing
    anything. FORCE retrains only these: the registered runs of the same
    ENV + MODEL_ID that another queue trained under other settings are not
    this pass's (fig20's FORCE retrained fig24's haberdashery_flock, 2026-10-20).
    """
    base_cfg = _base_cfg_for(cfg)
    selected_model_id = str(cfg.get("model_id") or os.environ.get("MODEL_ID") or "").strip()
    if selected_model_id:
        base_cfg["model_id"] = selected_model_id
    requested_env = str(cfg.get("env") or os.environ.get("ENV") or "dev").strip() or "dev"
    return {
        _scc_identity(sweep_cfg, cfg, Path(project_root).resolve(), selected_model_id, requested_env)[2]
        for sweep_cfg in generate_param_combinations(base_cfg)
    }


def get_pending_runs(
    cfg: Dict[str, Any],
    project_root: Path,
    *,
    retry_failed: bool = False,
    resume_interrupted: bool = False,
    force: bool = False,
) -> Iterator[Tuple[str, Dict[str, Any], Path, str]]:
    """
    Yield executable SCC runs only.

    HARD RULES:
    - Do NOT pre-create NEW registry entries for every SCC combination.
    - Only create/update the single model_hash currently being executed.
    - Resume/retry only inside current orchestration scope:
        ENV + MODEL_ID + optional MODELS/MODEL_NAME.
    - Write run_map IN_PROGRESS before training starts.
    """
    project_root = Path(project_root).resolve()

    try:
        registry = load_registry(project_root)
        sync_run_map_before_planning(project_root)
        force_session_id = str(
            os.environ.get("ORCHESTRATOR_SESSION_ID")
            or os.environ.get("AUTO_SEARCH_SESSION_ID")
            or ""
        ).strip()
        session_run_map = load_run_map(project_root) if force else {}
        base_cfg = _base_cfg_for(cfg)

        selected_model_id = (
            str(cfg.get("model_id") or os.environ.get("MODEL_ID") or "").strip()
        )
        if selected_model_id:
            base_cfg["model_id"] = selected_model_id

    except Exception as e:
        # A damaged registry or config must stop the run, not plan zero runs.
        logger.error(f"Failed to initialize config or registry: {e}")
        raise

    requested_env = str(cfg.get("env") or os.environ.get("ENV") or "dev").strip() or "dev"

    requested_names_raw = (
        os.environ.get("MODELS")
        or os.environ.get("MODEL_NAME")
        or ""
    ).strip()

    requested_names = {
        x.strip()
        for x in requested_names_raw.split(",")
        if x.strip()
    }
    single_run = str(os.environ.get("SCC_SINGLE_RUN") or "").strip() == "1"
    single_hash = str(os.environ.get("SCC_SINGLE_HASH") or "").strip()

    if single_run and single_hash:
        requested_names.add(single_hash)
    if not (retry_failed or resume_interrupted or force):
        allowed = {"NEW"}
    else:
        allowed = {"NEW"}
        if resume_interrupted:
            # QUEUED: selected by an orchestrator pass that died before starting it.
            allowed |= {"IN_PROGRESS", "INTERRUPTED", "RUNNING", "IN_PROCESS", "QUEUED"}
        if retry_failed:
            allowed |= {"FAILED"}
        if force:
            allowed |= {"DONE", "IN_PROGRESS", "INTERRUPTED", "FAILED", "RUNNING", "IN_PROCESS", "QUEUED"}

    # Coerce after expansion: component alternatives are lists of mappings.
    iter_cfgs = generate_param_combinations(base_cfg)

    try:
        expanded_count = count_all_param_combinations(base_cfg)
    except Exception:
        expanded_count = -1
    trial_index = 0                  # runs planned in this pass (an empty pass explains itself below)
    grid_position = 0                # place of the current combination in the sweep grid, 1..total_trials:
                                     # counts every unique hash, planned or skipped (DONE), so a run's
                                     # trial number is the same in every pass and on resume
    existing_skipped: list[tuple[str, str]] = []
    active_batch = current_batch()   # (BATCH, BATCH_ID) of make experiment / make orchestrate BATCH=
    yielded_hashes: set[str] = set()
    trainer_by_hash: Dict[str, Any] = {}
    total_trials = int(expanded_count) if isinstance(expanded_count, int) and expanded_count > 0 else 0
    logger.info(
        "[sweep_engine] SCC scoped expansion active: count_estimate=%s "
        "force=%s retry_failed=%s resume_interrupted=%s env=%s model_id=%s names=%s",
        expanded_count,
        force,
        retry_failed,
        resume_interrupted,
        requested_env,
        selected_model_id or "<unset>",
        sorted(requested_names),
    )

    reserved_names: set = set()           # names this pass gave hashes it has not started (not in the map)
    for sweep_cfg in iter_cfgs:
        run_cfg, model_name, model_hash = _scc_identity(
            sweep_cfg, cfg, project_root, selected_model_id, requested_env, reserved=reserved_names,
        )
        reserved_names.add(str(model_name))
        sampled_payload = run_cfg["sampled"]

        prior_trainer = trainer_by_hash.get(str(model_hash))
        if prior_trainer is not None and prior_trainer != run_cfg["trainer"]:
            raise RuntimeError(
                "Different trainer configurations produced the same model_hash. "
                "Update make_experiment_hash to include the nested trainer settings."
            )
        trainer_by_hash[str(model_hash)] = copy.deepcopy(run_cfg["trainer"])

        if str(model_hash) in yielded_hashes:
            logger.debug(
                "[sweep_engine] duplicate model_hash skipped in same stream: "
                "hash=%s name=%s env=%s model_id=%s",
                model_hash,
                model_name,
                requested_env,
                selected_model_id or run_cfg.get("model_id"),
            )
            continue

        yielded_hashes.add(str(model_hash))
        grid_position += 1

        run_cfg["model_name"] = str(model_name)
        run_cfg["model_hash"] = str(model_hash)
        run_cfg["frozen_identity"] = {
            "model_name": str(model_name),
            "model_hash": str(model_hash),
        }
        run_cfg.setdefault("run", {})
        run_cfg["run"]["model_name"] = str(model_name)

        if requested_names:
            if model_name not in requested_names and model_hash not in requested_names:
                continue

        template_values = {
            "env": run_cfg["env"],
            "model_name": str(model_name),
            "dataset_alias": run_cfg.get("data", {}).get("name", "dataset"),
        }
        run_cfg = deep_template_resolve(run_cfg, template_values)

        assert model_name and model_name != "None", f"model_name not resolved: {model_name}"
        assert "paths" in run_cfg and "artifacts_root" in run_cfg["paths"], "paths.artifacts_root missing"
        assert "None" not in str(run_cfg["paths"]["artifacts_root"]), (
            f"model_name/env not resolved in path: {run_cfg['paths']['artifacts_root']}"
        )

        run_cfg = coerce_types(run_cfg)
        # paths.artifacts_root is a project-relative template ("inside the project folder", the
        # blueprint): resolved against the project root, as utils/path_utils.resolve_path does, never
        # against the process's working directory. Relative, it made the run folder wherever the
        # process stood: junk under the container's /app as root, a PermissionError as the user
        # (2026-10-23). The folder itself is made below, once the run is decided.
        run_dir = Path(str(run_cfg["paths"]["artifacts_root"]))
        if not run_dir.is_absolute():
            run_dir = project_root / run_dir

        env_val = _canonical_env(run_cfg, run_dir)
        run_cfg["env"] = env_val

        entry = registry.get(model_hash)

        # Existing entry must match current scope.
        if isinstance(entry, dict):
            entry_env = str(entry.get("env") or "").strip()
            entry_model_id = str(entry.get("model_id") or "").strip()

            if entry_env and entry_env != env_val:
                continue

            if selected_model_id and entry_model_id and entry_model_id != selected_model_id:
                continue

            # IMPORTANT:
            # If MODEL_ID is supplied, legacy entries without model_id are unsafe to resume.
            if selected_model_id and not entry_model_id:
                logger.warning(
                    "[sweep_engine] skipping legacy registry entry without model_id: "
                    "hash=%s name=%s requested_model_id=%s",
                    model_hash,
                    entry.get("model_name"),
                    selected_model_id,
                )
                continue

            prev_status = str(entry.get("status", "NEW")).upper()
        else:
            prev_status = "NEW"

        # FORCE never retrains a finished run another batch holds (another figure's curve: fig20's FORCE rewrote
        # fig24's haberdashery_flock); it joins this batch as a skipped DONE run. MODELS=<name> retrains it.
        if (force and prev_status == "DONE" and held_by_another_batch(entry, active_batch)
                and not {str(model_name), str(model_hash)} & requested_names):
            logger.info(
                "[sweep_engine] force: hash=%s name=%s is DONE in batch(es) %s; kept (MODELS=%s retrains it)",
                model_hash, model_name, entry.get("batches"), model_name,
            )
            prev_status = "KEPT"
        if prev_status not in allowed:
            existing_skipped.append((str(model_hash), str(model_name)))
            # A DONE run in this batch's scope is part of the batch although
            # it is not trained again (e.g. a reference run of another figure).
            if active_batch is not None and isinstance(entry, dict):
                stamp_registry(project_root, [str(model_hash)], batch=active_batch)
            continue

        # FORCE re-runs every scoped hash once per orchestrator session: a hash
        # already attempted under this session id is not planned again, so the
        # sweep advances instead of repeating the first grid hash forever.
        if force and force_session_id and _attempted_in_session(
            (entry if isinstance(entry, dict) else {}, session_run_map.get(str(model_hash)) or {}),
            force_session_id,
        ):
            logger.info(
                "[sweep_engine] force: hash=%s name=%s already trained (DONE/FAILED) in session %s; skipping",
                model_hash, model_name, force_session_id,
            )
            continue
        # The run is decided: its name is written into model_name_map.json now, not when it was
        # planned (a skipped combination leaves no name behind).
        remembered = remember_run_name(project_root, model_hash, model_name)
        if remembered != str(model_name):
            raise RuntimeError(
                f"[sweep_engine] hash={model_hash} is named {remembered!r} in model_name_map.json, but this "
                f"pass planned it as {model_name!r}: another planner named it meanwhile; start again"
            )
        (run_dir / "config").mkdir(parents=True, exist_ok=True)   # the run's folder, for the run that starts
        trial_index += 1

        run_cfg.setdefault("runtime", {})
        if not isinstance(run_cfg["runtime"], dict):
            run_cfg["runtime"] = {}

        run_cfg["runtime"]["trial_index"] = int(grid_position)
        run_cfg["runtime"]["sweep_position"] = int(grid_position)   # the orchestrator numbers the run by it
        run_cfg["runtime"]["trial_total"] = int(total_trials)
        run_cfg["runtime"]["selection_source"] = "scc_sweep"

        pretty_print(
            f"SCC trial={grid_position}/{total_trials} | "
            f"hash={model_hash} | "
            f"name={model_name} | "
            f"env={env_val} | "
            f"model_id={selected_model_id or run_cfg.get('model_id')}",
            color="pink",
            style="bold",
        )
        # Create/update ONLY the current executable hash.
        registry = load_registry(project_root)

        current = registry.get(model_hash)
        if not isinstance(current, dict):
            current = {}

        if not _is_running_status(str(current.get("status") or "")) or not str(current.get("started_at") or "").strip():
            begin_attempt(current, _now_local_iso())
        current["model_name"] = model_name
        current["model_id"] = selected_model_id or run_cfg.get("model_id")
        current["status"] = "in_progress"
        current["path"] = str(run_dir)
        current["env"] = env_val
        current["leakage_model"] = (run_cfg.get("sca") or {}).get("leakage_model")
        concrete_sca = (
            run_cfg.get("sca")
            or {}
        )

        current[
            "attack_byte"
        ] = concrete_sca.get(
            "attack_byte"
        )

        # The resolved SCA modes live in the run's params_out snapshot; the
        # registry carries the lifecycle outcome instead.
        current.pop("candidate_ranking_mode", None)
        current.pop("key_enumeration_mode", None)
        # The resolved run config is persisted as the run's params_out snapshot
        # (<run>/config/<model_name>_params.yaml), not in the registry.
        current.pop("run_cfg", None)

        if not str(current.get("started_at") or "").strip():
            current["started_at"] = _now_local_iso()

        current.pop("finished_at", None)
        if active_batch is not None:
            add_batch(current, *active_batch)

        registry[model_hash] = current
        save_registry(registry, project_root)

        start_run_map_entry(
            cfg=run_cfg,
            model_hash=model_hash,
            model_name=model_name,
            model_id=selected_model_id or run_cfg.get("model_id"),
            sampled=sampled_payload,
            mode="scc",
            run_dir=run_dir,
            trial_index=int(grid_position),
            total_trials=int(total_trials),
            selection_source="scc_sweep",
            verify=True,
        )

        logger.info(
            "[sweep_engine] yielding scoped SCC run: hash=%s name=%s env=%s model_id=%s status_before=%s",
            model_hash,
            model_name,
            env_val,
            selected_model_id or run_cfg.get("model_id"),
            prev_status,
        )

        yield model_hash, run_cfg, run_dir, model_name

        # ---------------------------------------------------------
        # SINGLE HASH LOCK
        #
        # train_and_test() launches child train/test with:
        #   SCC_SINGLE_RUN=1
        #   SCC_SINGLE_HASH=<selected hash>
        #
        # Once this exact hash has been yielded once,
        # stop further SCC expansion inside this child process.
        # ---------------------------------------------------------
        if single_run and single_hash and str(model_hash) == str(single_hash):
            logger.info(
                "[sweep_engine] SCC_SINGLE_HASH satisfied. "
                "Stopping further SCC expansion for hash=%s",
                model_hash,
            )
            return

    # Nothing was planned and FORCE is off: name every scoped experiment that
    # already exists, so an empty pass explains itself.
    if trial_index == 0 and not force:
        for skipped_hash, skipped_name in existing_skipped:
            logger.info(
                "the requested experiment already exists model_hash=%s; model_name=%s. "
                "run scope with FORCE=1 to regenerate the fresh results.",
                skipped_hash,
                skipped_name,
            )


def _attempted_in_session(records: Iterable[Any], session_id: str) -> bool:
    """
    True when a registry / run_map record shows the hash DONE or FAILED under
    session_id. FORCE trains each hash once per session: DONE ones are not
    repeated, and FAILED ones are not retried in a loop (FORCE ignores the
    retry budget); a later session trains them again.
    """
    return any(
        isinstance(rec, dict)
        and str(rec.get("run_session_id") or "").strip() == session_id
        and str(rec.get("status") or "").strip().upper() in {"DONE", "FAILED"}
        for rec in records
    )


def update_status(project_root: Path, model_id: str, new_status: str, reason: str | None = None) -> None:
    """
    Set run_registry[model_hash]["status"]. FAILED / INTERRUPTED record
    `reason` as status_reason ("unspecified", with a warning, when none is
    given); any other status drops it.
    """
    registry = load_registry(project_root)  # heal=True by default

    if model_id not in registry:
        logger.error(f"Couldn't find run {model_id} in registry to update")
        return

    entry = registry[model_id]
    if not isinstance(entry, dict):
        logger.error(f"Registry entry for {model_id} is not a dict")
        return

    previous = str(entry.get("status") or "")
    ns = str(new_status)
    entry["status"] = ns
    if _status_norm(ns) in {"failed", "interrupted", "cancelled", "canceled"}:
        if not str(reason or "").strip():
            logger.warning(
                "update_status: %s set to %s without a reason; status_reason=unspecified",
                model_id, ns,
            )
        entry["status_reason"] = str(reason or "").strip() or "unspecified"
    else:
        entry.pop("status_reason", None)
    if _status_norm(ns) in {"done", "failed"}:
        # FORCE queued it (orchestrator._force_queue); it has now been retrained.
        entry.pop("force_previous_status", None)

    # Here `model_id` argument is actually the model_hash key. The entry's
    # architecture model_id is not touched: save_registry() backfills it only
    # when missing (_resolve_model_id_for_registry_entry). Overwriting it with
    # the MODEL_ID of whichever process updates the status would orphan the
    # hash, because the planner skips a hash whose model_id is not its own.

    # The legacy failure field; the registry's reason is status_reason.
    entry.pop("reason", None)

    # Timestamps belong to the attempt: entering in_progress starts a new one
    # (begin_attempt: its start time, no finish time, no earlier outcome), and
    # entering a terminal status ends it. A repeated status keeps its time.
    if _is_running_status(ns) and (
        not _is_running_status(previous) or not str(entry.get("started_at") or "").strip()
    ):
        begin_attempt(entry, _now_local_iso())

    if _is_terminal_status(ns) and (
        not _is_terminal_status(previous) or not str(entry.get("finished_at") or "").strip()
    ):
        entry["finished_at"] = _now_local_iso()

    save_registry(registry, project_root)


def update_name(project_root: Path, model_id: str, new_name: str) -> None:
    """
    Set registry[model_id]["model_name"] = new_name, then save.
    """
    registry = load_registry(project_root)

    if model_id not in registry:
        logger.error(f"Couldn't find run {model_id} in registry to update")
        return

    entry = registry[model_id]
    if not isinstance(entry, dict):
        logger.error(f"Registry entry for {model_id} is not a dict")
        return

    entry["model_name"] = str(new_name)
    save_registry(registry, project_root)

def build_run_identity_label(
    cfg: Dict[str, Any],
    *,
    sampled: Optional[Dict[str, Any]] = None,
    model_id: Optional[str] = None,
    override_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a durable identity label for one resolved run config.

    This does not write run_map.jsonl.
    It only returns a structured identity object that AutoSearch can persist.
    """
    model_name, model_hash = determine_run_identity(
        cfg,
        override_model_name=override_model_name,
    )

    cfg["model_name"] = str(model_name)
    cfg["model_hash"] = str(model_hash)
    cfg.setdefault("run", {})
    if isinstance(cfg["run"], dict):
        cfg["run"]["model_name"] = str(model_name)

    return {
        "model_hash": str(model_hash),
        "model_name": str(model_name),
        "model_id": str(model_id) if model_id is not None else cfg.get("model_id"),
        "sampled": sampled or {},
    }


def determine_run_identity(
    cfg: Dict[str, Any],
    override_model_name: Optional[str] = None,
    *,
    persist: bool = True,
    reserved: Optional[set] = None,
) -> Tuple[str, str]:
    """
    Sets cfg['model_name'] and cfg['model_hash'].
    Returns (model_name, model_hash).

    persist=False: the name is resolved (the map's, the registry's, else a fresh one) but
    model_name_map.json is not written. The planner names every combination it looks at;
    only a run it decides to start is written into the map (remember_run_name), so a
    planning pass (planned_hashes, a skipped DONE run) leaves no phantom name behind.

    Guarantees:
    - model_hash is deterministic from config unless rerun is pinned.
    - one hash maps to one stable model_name.
    - generated model_name is unique across different hashes.
    - rerun with pinned hash never recomputes identity.
    """
    frozen = cfg.get("frozen_identity") or {}
    if isinstance(frozen, dict):
        frozen_name = frozen.get("model_name")
        frozen_hash = frozen.get("model_hash")

        if frozen_name and frozen_hash:
            cfg["model_name"] = str(frozen_name)
            cfg["model_hash"] = str(frozen_hash)
            cfg.setdefault("run", {})
            if isinstance(cfg["run"], dict):
                cfg["run"]["model_name"] = str(frozen_name)

            return str(frozen_name), str(frozen_hash)
    name = override_model_name or cfg.get("model_name")

    pinned = cfg.get("pinned_model_hash")
    if cfg.get("rerun") and pinned:
        h = str(pinned)
    elif cfg.get("rerun") and cfg.get("model_hash"):
        h = str(cfg["model_hash"])
    else:
        h = make_experiment_hash(cfg)

    project_root = Path(cfg["project"]["root"])
    mapping = _load_mapping(project_root)

    # Existing hash always wins unless explicit override/rerun-pinned name is supplied.
    if not name and h in mapping:
        name = str(mapping[h])

    # A hash already in run_registry.json keeps its registered model_name when
    # model_name_map.json lacks it (e.g. the map was deleted): the name is its
    # run directory and params_out snapshot. The map is repaired below.
    if not name:
        registered = load_registry(project_root, heal=False).get(h)
        if isinstance(registered, dict) and str(registered.get("model_name") or "").strip():
            name = str(registered["model_name"]).strip()

    # Prevent two different hashes from sharing the same generated/display name.
    used_names_by_other_hash = {
        str(v)
        for k, v in mapping.items()
        if str(k) != str(h)
    } | {str(n) for n in (reserved or ())}

    if not name:
        name = generate_unique_model_name(used_names_by_other_hash)

    # An override name that another hash already uses: generate a fresh one.
    if name in used_names_by_other_hash:
        name = generate_unique_model_name(used_names_by_other_hash)

    # Persist mapping safely.
    if override_model_name:
        mapping[h] = str(name)
    else:
        if h not in mapping:
            mapping[h] = str(name)
        else:
            if cfg.get("rerun") and cfg.get("pinned_model_hash"):
                mapping[h] = str(name)
            else:
                name = str(mapping[h])

    if persist:
        _save_mapping(project_root, mapping)

    cfg["model_name"] = str(name)
    cfg["model_hash"] = str(h)

    return str(name), str(h)


def remember_run_name(project_root: Path, model_hash: str, model_name: str) -> str:
    """
    The run is decided: its name goes into model_name_map.json now. The map's own name for the
    hash wins (another pass wrote it first); a name another hash took meanwhile is replaced by a
    fresh one. Returns the name to run under.
    """
    mapping = _load_mapping(Path(project_root))
    h = str(model_hash)
    if h in mapping and str(mapping[h]).strip():
        return str(mapping[h])
    used_by_others = {str(v) for k, v in mapping.items() if str(k) != h}
    name = str(model_name)
    if not name.strip() or name in used_by_others:
        name = generate_unique_model_name(used_by_others)
    mapping[h] = name
    _save_mapping(Path(project_root), mapping)
    return name

# =============================================================================
# Counting
# =============================================================================
def count_all_param_combinations(
    base_cfg: Dict[str, Any],
) -> int:
    """
    Count the exact deterministic SCC grid size.

    Count:

        non-SCA Cartesian combinations
        ×
        valid conditional SCA combinations

    The SCA count uses the same compiler as actual execution, so inactive
    candidate-ranking/enumeration fields never inflate ``trial_total``.

    Architecture ``search_space`` fields remain excluded because OptunaLite
    owns architecture search.
    """
    if not isinstance(
        base_cfg,
        dict,
    ):
        raise TypeError(
            "base_cfg must be a dict"
        )

    # sca.attack_byte left empty is one value per run: the data entry's own byte, which
    # dataset_types.resolve_attack_byte writes in before a run's sweep (the count is read
    # from the file before any entry is selected, so it counts the one value here).
    sca_raw = base_cfg.get("sca")
    if isinstance(sca_raw, dict) and sca_raw.get("attack_byte") in (None, []):
        base_cfg = {**base_cfg, "sca": {**sca_raw, "attack_byte": [0]}}

    generic_paths = (
        _sweep_dimensions(
            base_cfg
        )
    )

    generic_count = 1

    for _path, values in generic_paths:
        generic_count *= len(
            values
        )

    sca_cfg = base_cfg.get(
        "sca"
    )

    if sca_cfg is None:
        sca_count = 1

    else:
        if not isinstance(
            sca_cfg,
            dict,
        ):
            raise SCASweepError(
                "cfg.sca must be a mapping"
            )

        sca_count = (
            count_sca_combinations(
                sca_cfg
            )
        )

    trainer = base_cfg.get("trainer")
    if trainer is not None and not isinstance(trainer, dict):
        raise TypeError("trainer must be a mapping")
    if trainer is None:
        return int(generic_count * sca_count)
    # The trainer grid depends on the framework (its optimizer_args are wired in),
    # so a swept deep_learning.framework counts each framework's own grid.
    framework_path = ("deep_learning", "framework")
    frameworks = next((values for path, values in generic_paths if tuple(path) == framework_path), None)
    if frameworks is None:
        trainer_count = sum(1 for _ in _expand_trainer_value(trainer, optimizer_args=optimizer_args_table(base_cfg)))
        return int(generic_count * sca_count * trainer_count)
    per_framework = generic_count // len(frameworks)
    total = 0
    for framework in frameworks:
        point = copy.deepcopy(base_cfg)
        point["deep_learning"]["framework"] = framework
        total += sum(1 for _ in _expand_trainer_value(trainer, optimizer_args=optimizer_args_table(point)))
    return int(per_framework * sca_count * total)
