# services/mlops/core/optuna_lite.py
"""
optuna_lite.py
==============

Lean Optuna-based search runner for project-scoped auto_search.yaml.

Purpose
-------
This module implements a minimal, fast, modular HPO/search layer for
paper reproduction and basic experiment sweeps, without pulling in the
full hypothesis-driven AutoMLOps stack.

Design Contract
---------------
- SCC owns training/testing/data/model execution.
- config_loader.get_project_config(project_name) returns the final merged cfg.
- cfg["search"] contains search engine settings.
- cfg["model"] may contain a list of model entries with "model_id".
- cfg["trainer"] may contain list-valued fields used as search choices.
- model_hash is the only registry identity.
- trial_id is runtime/debug/progress metadata only.
- AutoMLOps signal handling is delegated to services.mlops.utils.brakes.

Registry Contract
-----------------
This module does not maintain any separate Optuna/trial registry.

Author: Shraddha Mishra
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from services.mlops.core.optuna_engine import OptunaEngine
from services.side_channel_core.utils.sweep_engine import (
    build_run_identity_label,
    generate_param_combinations,
    _assert_trainer_preserved,
)
from services.side_channel_core.utils.type_coerce import coerce_types
from services.mlops.utils.scc_bridge_lite import run_trial_via_scc_lite
from services.mlops.utils.zombie_reaper import cleanup_trial_runtime
from services.side_channel_core.src.core.model_contract_resolver import resolve_model_contract
from services.side_channel_core.utils.config_loader import get_project_config
from services.side_channel_core.utils.logger import pretty_print
from services.side_channel_core.utils.resolver_mode import _derive_attempt_window
from services.side_channel_core.utils.sca_sweep import (
    assert_concrete_sca_runtime,
    iter_sca_combinations,
)
from services.side_channel_core.utils.run_map import (
    load_run_map,
    prioritize_sample_points,
    sample_key as run_map_sample_key,
)

from services.side_channel_core.utils.run_lifecycle_logger import (
    complete_run_map_entry,
    fail_run_map_entry,
    start_run_map_entry,
    sync_run_map_before_planning,
)
from services.side_channel_core.utils.timestamps import now_str
from services.side_channel_core.utils.seal import check as check_seal, strip as strip_seal
Json = Dict[str, Any]





# search.method: bo (Bayesian, TPE) | rs (random search); also the config panel's menu.
SEARCH_METHODS = frozenset({"bo", "rs"})

def _log(msg: str) -> None:
    ts = now_str()
    print(f"[mlops/optuna_lite {ts}] {msg}", flush=True)


def _read_last_jsonl_record(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None

    last_obj = None

    # utils/seal.py: verified first, seal keys dropped (the objective is read from here)
    check_seal(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(obj, dict):
                last_obj = strip_seal(obj)

    return last_obj

def _build_sca_variants(
    sca_cfg: Json,
) -> List[Json]:
    """
    Compile the project SCA search specification into concrete conditional
    runtime variants.

    Each returned item is one complete SCA experiment contract. Conditional
    dimensions are already resolved by utils.sca_sweep, so inactive
    candidate-ranking/key-enumeration parameters cannot multiply the space.

    These variants are treated as atomic categorical choices by OptunaLite.
    """
    if not isinstance(
        sca_cfg,
        dict,
    ):
        return []

    variants = [
        copy.deepcopy(
            concrete
        )
        for concrete in iter_sca_combinations(
            sca_cfg
        )
    ]

    if not variants:
        raise ValueError(
            "SCA search compilation produced no concrete variants"
        )

    for index, variant in enumerate(
        variants
    ):
        try:
            assert_concrete_sca_runtime(
                variant
            )
        except Exception as exc:
            raise ValueError(
                "SCA compiler produced a non-concrete variant: "
                f"index={index}"
            ) from exc

    return variants

def _materialize_sca_variant(
    sampled: Mapping[str, Any],
    *,
    sca_variants: List[Json],
) -> Json:
    """
    Replace internal ``sca_variant_index`` with the corresponding concrete SCA
    configuration.

    Planning/search uses a small integer categorical dimension so Optuna never
    recursively interprets SCA internals.

    Registry identity and SCC execution use the materialized scientific
    configuration instead:

        sampled["sca"] = {... concrete SCA contract ...}
    """
    out = copy.deepcopy(
        dict(sampled)
    )

    index_raw = out.pop(
        "sca_variant_index",
        None,
    )

    if index_raw is None:
        if sca_variants:
            if len(sca_variants) != 1:
                raise ValueError(
                    "sample is missing sca_variant_index while multiple "
                    "SCA variants exist"
                )

            concrete_sca = copy.deepcopy(
                sca_variants[0]
            )

        else:
            return out

    else:
        if isinstance(
            index_raw,
            bool,
        ):
            raise ValueError(
                "sca_variant_index must be an integer"
            )

        try:
            index = int(
                index_raw
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Invalid sca_variant_index={index_raw!r}"
            ) from exc

        if not 0 <= index < len(
            sca_variants
        ):
            raise ValueError(
                "sca_variant_index is outside compiled SCA variant range: "
                f"index={index}, variants={len(sca_variants)}"
            )

        concrete_sca = copy.deepcopy(
            sca_variants[
                index
            ]
        )

    assert_concrete_sca_runtime(
        concrete_sca
    )

    out["sca"] = concrete_sca

    return out
def _materialize_sample(
    sampled: Mapping[str, Any],
    *,
    sca_variants: List[Json],
    trainer_variants: List[Json],
) -> Json:
    """Replace categorical indices with concrete contracts before identity."""
    out = _materialize_sca_variant(sampled, sca_variants=sca_variants)
    index = out.pop("trainer_variant_index", None)
    if index is None:
        if len(trainer_variants) != 1:
            raise ValueError("Sample must select a trainer_variant_index")
        index = 0
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("trainer_variant_index must be an integer")
    if not 0 <= index < len(trainer_variants):
        raise ValueError("trainer_variant_index is outside the compiled trainer range")
    if "trainer" in out:
        raise ValueError("Sample contains both a trainer payload and a trainer variant selection")
    out["trainer"] = copy.deepcopy(trainer_variants[index])
    return out


def _prepare_resolver_inputs(
    *,
    cfg: Json,
    sampled: Mapping[str, Any],
) -> tuple[
    Json,
    Json,
]:
    """
    Prepare one AutoSearch sample for model-contract resolution.

    SCA is injected directly into the base config because SCA has already been
    fully resolved by the dedicated conditional sweep compiler.

    The concrete trainer is also injected directly. The remaining sampled
    payload is passed to model_contract_resolver for model/architecture handling.
    """
    resolver_cfg = copy.deepcopy(
        cfg
    )

    resolver_sampled = copy.deepcopy(
        dict(sampled)
    )

    concrete_sca = resolver_sampled.pop(
        "sca",
        None,
    )

    if concrete_sca is not None:
        if not isinstance(
            concrete_sca,
            dict,
        ):
            raise ValueError(
                "Materialized sampled.sca must be a mapping"
            )

        assert_concrete_sca_runtime(
            concrete_sca
        )

        resolver_cfg[
            "sca"
        ] = copy.deepcopy(
            concrete_sca
        )

    concrete_trainer = resolver_sampled.pop("trainer", None)
    if not isinstance(concrete_trainer, dict):
        raise ValueError("Materialized sampled.trainer must be a mapping")
    resolver_cfg["trainer"] = copy.deepcopy(concrete_trainer)

    return (
        resolver_cfg,
        resolver_sampled,
    )

def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out

    try:
        text = str(value).strip()
        if not text:
            return None

        out = float(text)
        if math.isnan(out) or math.isinf(out):
            return None

        return out
    except Exception:
        return None


def _extract_metric_from_record(
    record: Dict[str, Any],
    metric_name: str,
) -> Optional[float]:
    if not isinstance(record, dict):
        return None

    candidates = [
        metric_name,
        metric_name.replace("@", "_at_"),
        metric_name.replace("@", ""),
    ]

    for key in candidates:
        value = record.get(key)

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

        value = _coerce_float(value)
        if value is not None:
            return float(value)

    # History rows keep what a requested metric depends on under "dependencies".
    for nested in (record.get("metrics"), record.get("dependencies")):
        if not isinstance(nested, dict):
            continue
        for key in candidates:
            value = nested.get(key)

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)

            value = _coerce_float(value)
            if value is not None:
                return float(value)

    return None


def _read_objective_values(
    *,
    metrics_dir: Optional[str],
    objective_metrics: List[str],
) -> Dict[str, float]:
    if not metrics_dir:
        raise RuntimeError("metrics_dir missing from SCC result")

    metrics_path = Path(metrics_dir)
    if not metrics_path.exists():
        raise RuntimeError(f"metrics dir not found: {metrics_path}")

    scalars_dir = metrics_path / "scalars"
    if not scalars_dir.is_dir():
        raise RuntimeError(f"no scalars dir found in metrics dir: {metrics_path}")

    validation_candidates = sorted(
        scalars_dir.glob("*_metrics_validation_history.jsonl")
    )
    train_candidates = sorted(
        scalars_dir.glob("*_metrics_train_history.jsonl")
    )

    out: Dict[str, float] = {}
    missing: List[str] = []

    for metric in objective_metrics:
        found = None

        for candidate in [*validation_candidates, *train_candidates]:
            record = _read_last_jsonl_record(candidate)
            if record is None:
                continue

            value = _extract_metric_from_record(record, metric)
            if value is not None:
                found = float(value)
                break

        if found is None:
            missing.append(metric)
        else:
            out[metric] = found

    if missing:
        raise RuntimeError(
            f"could not read objective metrics {missing} "
            f"from JSONL histories under {scalars_dir}"
        )

    return out


@dataclass(frozen=True)
class LiteTrialResult:
    """
    In-memory result summary for one sampled attempt.

    This is not registry state.
    Registry state is controlled only by run_registry.json/test_registry.json
    through update_status() and update_test_status().
    """

    trial_id: int
    status: str  # DONE | FAIL
    model_id: Optional[str] = None
    model_name: str = ""
    model_hash: str = ""
    run_dir: Optional[str] = None
    objectives: Optional[Dict[str, float]] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class LiteRunSummary:
    attempted: int
    done: int
    failed: int
    study_name: str


def _extract_model_ids(cfg: Json) -> List[str]:
    model_block = cfg.get("model")

    if not isinstance(model_block, list):
        return []

    out: List[str] = []

    for entry in model_block:
        if not isinstance(entry, dict):
            continue

        model_id = entry.get("model_id")
        if isinstance(model_id, str) and model_id.strip():
            out.append(model_id.strip())

    return out


def _build_trainer_variants(trainer_cfg: Json, deep_learning: Json | None = None) -> List[Json]:
    """Use the same trainer choice and literal-pair rules as deterministic SCC.

    deep_learning supplies the framework's optimizer_args, wired into
    trainer.optimizer.args as in the deterministic sweep.
    """
    if not isinstance(trainer_cfg, dict):
        raise TypeError("trainer must be a mapping")
    variants = []
    seen = set()
    base = {"trainer": trainer_cfg}
    if deep_learning is not None:
        base["deep_learning"] = deep_learning
    for point in generate_param_combinations(base):
        trainer = point["trainer"]
        key = json.dumps(trainer, sort_keys=True)
        if key not in seen:
            seen.add(key)
            variants.append(trainer)
    return variants


def _build_model_search_space(cfg: Json) -> Json:
    """
    Build architecture search space from model[*].architecture_args.search_space.

    Output shape:
        {
            "CNN1": {...},
            "MLP1": {...},
        }

    The selected model_id decides which model-specific search space is used.
    """
    model_block = cfg.get("model")

    if not isinstance(model_block, list):
        return {}

    out: Json = {}

    for entry in model_block:
        if not isinstance(entry, dict):
            continue

        model_id = str(entry.get("model_id", "")).strip()
        if not model_id:
            continue

        arch_args = entry.get("architecture_args") or {}
        if not isinstance(arch_args, dict):
            continue

        search_space = arch_args.get("search_space") or {}
        if not isinstance(search_space, dict):
            continue

        if not bool(search_space.get("enabled", False)):
            continue

        clean_space = {
            key: copy.deepcopy(value)
            for key, value in search_space.items()
            if key != "enabled"
        }

        if clean_space:
            out[model_id] = clean_space

    return out


def _build_space_from_cfg(
    cfg: Json,
    *,
    sca_variants: List[Json],
    trainer_variants: List[Json],
) -> Json:
    space: Json = {}

    selected_model_id = cfg.get("model_id") or os.environ.get("MODEL_ID")

    if selected_model_id:
        selected_model_id = str(selected_model_id).strip()
        space["model_id"] = [selected_model_id]
    else:
        model_ids = _extract_model_ids(cfg)
        if model_ids:
            space["model_id"] = copy.deepcopy(model_ids)

    if trainer_variants:
        space["trainer_variant_index"] = list(range(len(trainer_variants)))

    #
    # SCA is represented as one atomic categorical dimension.
    #
    # The actual conditional SCA dictionaries live in sca_variants and are
    # materialized only after one search point is selected.
    #
    if sca_variants:
        space[
            "sca_variant_index"
        ] = list(
            range(
                len(
                    sca_variants
                )
            )
        )
    model_search_space = _build_model_search_space(cfg)
    if model_search_space:
        if selected_model_id:
            if selected_model_id in model_search_space:
                space["model_search"] = {
                    selected_model_id: model_search_space[selected_model_id]
                }
            else:
                raise KeyError(
                    f"MODEL_ID={selected_model_id!r} selected, "
                    "but no enabled architecture_args.search_space found for it"
                )
        else:
            space["model_search"] = model_search_space

    return space


def _count_space_combinations(space: Any) -> int:
    """
    Count total concrete combinations in a finite lite search space.

    Rules:
    - scalar -> 1
    - list   -> len(list)
    - dict   -> product of child counts
    """
    if space is None:
        return 1

    if isinstance(space, (str, int, float, bool)):
        return 1

    if isinstance(space, list):
        return max(1, len(space))

    if isinstance(space, dict):
        total = 1
        for value in space.values():
            total *= max(1, _count_space_combinations(value))
        return max(1, total)

    return 1


def _enumerate_space(space: Any) -> List[Any]:
    """
    Enumerate all concrete realizations of a finite search-space object.
    """
    if space is None:
        return [None]

    if isinstance(space, (str, int, float, bool)):
        return [copy.deepcopy(space)]

    if isinstance(space, list):
        out: List[Any] = []
        for item in space:
            out.extend(_enumerate_space(item))
        return out

    if isinstance(space, dict):
        partials: List[Dict[str, Any]] = [{}]

        for key, child in space.items():
            child_values = _enumerate_space(child)
            next_partials: List[Dict[str, Any]] = []

            for base in partials:
                for value in child_values:
                    merged = copy.deepcopy(base)
                    merged[str(key)] = copy.deepcopy(value)
                    next_partials.append(merged)

            partials = next_partials

        return partials

    return [copy.deepcopy(space)]


def _enumerate_sample_points(
    space: Json,
    *,
    sca_variants: List[Json],
    trainer_variants: List[Json],
) -> List[Json]:
    """
    Enumerate finite AutoSearch points and materialize SCA and trainer contracts.

    Returned sample points never expose internal SCA/trainer variant indices.
    Therefore run_map/sample identity always describes the actual scientific
    configuration rather than an implementation-specific categorical index.
    """
    raw_points = _enumerate_space(
        space
    )

    final: List[
        Json
    ] = []

    for item in raw_points:
        if not isinstance(
            item,
            dict,
        ):
            continue

        final.append(
            _materialize_sample(
                item,
                trainer_variants=trainer_variants,
                sca_variants=(
                    sca_variants
                ),
            )
        )

    return final

def _sample_key(sampled: Mapping[str, Any]) -> str:
    return run_map_sample_key(sampled)
def _count_run_map_statuses(run_map: Mapping[str, Any]) -> Dict[str, int]:
    """
    Count run_map statuses for resume-planner visibility.

    This is diagnostic only. It does not affect scheduling.
    """
    counts: Dict[str, int] = {}

    for rec in run_map.values():
        if not isinstance(rec, dict):
            continue

        status = str(rec.get("status") or "UNKNOWN").strip().upper()
        if not status:
            status = "UNKNOWN"

        counts[status] = counts.get(status, 0) + 1

    return counts
def _remaining_points(
    all_points: List[Json],
    seen_keys: set[str],
) -> List[Json]:
    return [
        point
        for point in all_points
        if _sample_key(point) not in seen_keys
    ]


def _resolve_effective_max_trials(
    raw_budget: Any,
    inferred_max_trials: int,
) -> int:
    """
    Final budget policy for finite lite spaces:
    - None / auto / infer / <=0 -> infer exact finite size
    - explicit positive int      -> cap to finite size
    """
    auto_budget = (
        raw_budget is None
        or (
            isinstance(raw_budget, str)
            and raw_budget.strip().lower() in {"auto", "infer", ""}
        )
        or (isinstance(raw_budget, int) and raw_budget <= 0)
    )

    if auto_budget:
        return int(inferred_max_trials)

    try:
        explicit = int(raw_budget)
    except Exception:
        return int(inferred_max_trials)

    if explicit <= 0:
        return int(inferred_max_trials)

    return min(int(explicit), int(inferred_max_trials))


def _normalize_search_cfg(
    cfg: Json,
    project_name: str,
    env: str,
) -> Tuple[Json, Json]:
    """
    Returns:
        engine_cfg:
            Config shape expected by OptunaEngine.

        runtime_meta:
            Normalized search metadata for logging/runtime decisions.
    """
    search = cfg.get("search") or {}
    if not isinstance(search, dict):
        raise TypeError("cfg['search'] must be a dict")

    engine_name = str(search.get("engine", "optuna")).strip().lower()
    method = str(search.get("method", "bo")).strip().lower()

    if engine_name != "optuna":
        raise ValueError(
            f"unsupported search.engine={engine_name!r}; "
            "only 'optuna' is supported in optuna_lite"
        )

    if method not in SEARCH_METHODS:
        raise ValueError(
            f"unsupported search.method={method!r}; expected one of {sorted(SEARCH_METHODS)}"
        )

    budget = search.get("budget") or {}
    if not isinstance(budget, dict):
        raise TypeError("cfg['search']['budget'] must be a dict")

    raw_max_trials = budget.get("max_trials", None)
    parallel_trials = int(budget.get("parallel_trials", 1))

    objective = search.get("objective") or {}
    if not isinstance(objective, dict):
        raise TypeError("cfg['search']['objective'] must be a dict when provided")

    objective_metric = str(
        objective.get("metric", "guessing_entropy")
    ).strip()
    objective_direction = str(
        objective.get("direction", "minimize")
    ).strip().lower()

    if objective_direction not in {"minimize", "maximize"}:
        raise ValueError(
            "search.objective.direction must be 'minimize' or 'maximize'"
        )

    sampler_name = "tpe" if method == "bo" else "random"

    engine_cfg: Json = {
        "env": env,
        "project": {"name": project_name},
        "search": {
            "budget": {
                "max_trials": raw_max_trials,
                "parallel_trials": parallel_trials,
            },
            "study": {
                "name": f"{project_name}_{env}_{method}_lite",
            },
            "sampler": {
                "name": sampler_name,
            },
            "pruner": {
                "name": "none",
            },
        },
        "objectives": {
            "directions": {
                objective_direction: [objective_metric],
            }
        },
    }

    runtime_meta: Json = {
        "engine": engine_name,
        "method": method,
        "max_trials": raw_max_trials,
        "parallel_trials": parallel_trials,
        "objective_metric": objective_metric,
        "objective_direction": objective_direction,
    }

    return engine_cfg, runtime_meta

def _pick_next_sample(
    *,
    method: str,
    engine: OptunaEngine,
    space: Json,
    all_points: List[Json],
    seen_keys: set[str],
    project_name: str,
    env: str,
    sca_variants: List[Json],
    trainer_variants: List[Json],
) -> Tuple[
    Optional[Any],
    Optional[Json],
    Optional[str],
    str,
]:
    """
    Pick the next unique sampled config.

    Finite enumerated mode:
        Does not call Optuna ask(), so max_trials=N means exactly N executed
        unique configs.

    Huge unenumerated mode:
        Uses OptunaEngine.ask().
    """
    if not all_points:
        candidate = engine.ask(
            space=space,
            trial_user_attrs={
                "project_name": (
                    project_name
                ),
                "env": env,
                "search_method": (
                    method
                ),
            },
        )

        candidate_params = (
            _materialize_sample(
                candidate.params,
                trainer_variants=trainer_variants,
                sca_variants=(
                    sca_variants
                ),
            )
        )

        candidate_key = (
            _sample_key(
                candidate_params
            )
        )

        return (
            candidate,
            candidate_params,
            candidate_key,
            "optuna_no_enum",
        )
    remaining = _remaining_points(all_points, seen_keys)
    if not remaining:
        return None, None, None, "none"

    if method == "rs":
        sampled = copy.deepcopy(random.choice(remaining))
        sampled_key = _sample_key(sampled)
        return None, sampled, sampled_key, "finite_rs"

    if method == "bo":
        sampled = copy.deepcopy(remaining[0])
        sampled_key = _sample_key(sampled)
        return None, sampled, sampled_key, "finite_bo"

    raise ValueError(f"unsupported search method={method!r}; expected 'bo' or 'rs'")



class OptunaLiteRunner:
    """
    Lean project-scoped Optuna runner for auto_search.yaml.
    """

    def __init__(self, *, env: Optional[str] = None) -> None:
        self.env = str(env or os.environ.get("ENV", "dev")).strip() or "dev"

    def run(
        self,
        *,
        project_name: str,
        mode: str = "train",
        max_trials_override: Optional[int] = None,
    ) -> Tuple[List[LiteTrialResult], LiteRunSummary]:
        cfg = get_project_config(project_name)
        # sca.attack_byte left empty follows the data entry (dataset_types.resolve_attack_byte at a run's start);
        # the search space is compiled before any entry is selected, so it takes the byte every entry defines
        sca_block = cfg.get("sca") if isinstance(cfg.get("sca"), dict) else None
        if sca_block is not None and sca_block.get("attack_byte") in (None, []):
            from services.side_channel_core.src.sca.dataset_types import project_attack_byte

            shared = project_attack_byte(cfg)
            if shared is None:
                raise RuntimeError(
                    f"project={project_name}: sca.attack_byte is empty and its data entries do not all define one "
                    "byte (one allows several, or they differ): set sca.attack_byte for AutoSearch"
                )
            sca_block["attack_byte"] = [shared]

        project_root = Path(cfg["project"]["root"]).resolve()


        # Sync run_map.jsonl with run_registry.json before choosing search points.
        # This lets startup repair mark old in_progress runs as INTERRUPTED,
        # then AutoSearch sees those points as retryable.
        sync_run_map_before_planning(project_root)
        run_map = load_run_map(project_root)

        force_all = (
            str(os.environ.get("FORCE", "0")).strip() == "1"
            or str(os.environ.get("AUTO_SEARCH_FORCE_RUN_ALL", "0")).strip() == "1"
        )
        selected_model_id = os.environ.get("MODEL_ID") or cfg.get("model_id")
        if selected_model_id:
            cfg["model_id"] = str(selected_model_id).strip()

        auto_search_meta = cfg.get("auto_search") or {}
        if not isinstance(auto_search_meta, dict):
            raise TypeError("cfg['auto_search'] must be a dict when present")

        if not bool(auto_search_meta.get("enabled", False)):
            raise RuntimeError(f"auto_search is not enabled for project={project_name}")

        engine_cfg, runtime_meta = (
            _normalize_search_cfg(
                cfg,
                project_name,
                self.env,
            )
        )

        sca_variants = (
            _build_sca_variants(
                cfg.get(
                    "sca"
                )
                or {}
            )
        )

        trainer_variants = _build_trainer_variants(cfg.get("trainer") or {}, cfg.get("deep_learning"))
        space = _build_space_from_cfg(
            cfg,
            trainer_variants=trainer_variants,
            sca_variants=(
                sca_variants
            ),
        )
        _log(
            "DEBUG search space="
            f"{json.dumps(space, default=str)[:5000]}"
        )

        _log(
            "DEBUG conditional SCA variants="
            f"{len(sca_variants)}"
        )

        if not space:
            raise ValueError(
                "No search space detected. Expected at least one model_id "
                "in cfg['model'] list or list-valued trainer parameters."
            )

        inferred_max_trials = _count_space_combinations(space)
        raw_budget = ((cfg.get("search") or {}).get("budget") or {}).get(
            "max_trials",
            None,
        )

        effective_max_trials = _resolve_effective_max_trials(
            raw_budget,
            inferred_max_trials,
        )

        if max_trials_override is not None:
            try:
                effective_max_trials = min(
                    int(max_trials_override),
                    int(inferred_max_trials),
                )
            except Exception:
                effective_max_trials = int(inferred_max_trials)

        runtime_meta["max_trials"] = int(effective_max_trials)
        engine_cfg["search"]["budget"]["max_trials"] = int(effective_max_trials)

        pretty_print(
            (
                f"[optuna_lite] DEBUG "
                f"raw_max_trials={raw_budget!r} "
                f"inferred_max_trials={inferred_max_trials} "
                f"effective_max_trials={effective_max_trials}"
            ),
            color="yellow",
            bold=True,
        )

        engine = OptunaEngine(compiled=engine_cfg)
        objective_metrics = list(engine.objective_metrics)

        attempted = 0
        results: List[LiteTrialResult] = []
        seen_samples: set[str] = set()
        trainer_by_hash: Dict[str, Json] = {}
        next_trial_index, total_remaining_trials, total_trials = _derive_attempt_window(
                project_root,
                env=self.env,
                model_id=selected_model_id,
            )

        max_enumerate_points = 100_000

        if inferred_max_trials <= max_enumerate_points:
            all_points = (
                _enumerate_sample_points(
                    space,
                    trainer_variants=trainer_variants,
                    sca_variants=(
                        sca_variants
                    ),
                )
            )
            all_points = prioritize_sample_points(
                all_points,
                run_map,
                force=force_all,
            )

            status_counts = _count_run_map_statuses(run_map)

            _log(
                "resume planner: "
                f"force_all={force_all} "
                f"status_counts={json.dumps(status_counts, sort_keys=True)} "
                f"pending_points={len(all_points)} "
                f"unique_space={inferred_max_trials}"
            )
        else:
            all_points = []
            _log(
                f"finite space too large to enumerate: {inferred_max_trials}. "
                "Using Optuna sampling without full enumeration."
            )
        if all_points:
            effective_max_trials = min(int(effective_max_trials), len(all_points))
            runtime_meta["max_trials"] = int(effective_max_trials)
            engine_cfg["search"]["budget"]["max_trials"] = int(effective_max_trials)
        _log(
            f"start: project={project_name} env={self.env} "
            f"engine={runtime_meta['engine']} method={runtime_meta['method']} "
            f"study={engine.study_name} max_trials={effective_max_trials} "
            f"unique_space={inferred_max_trials} pending_points={len(all_points)}"
        )

        while True:
            if engine.should_stop():
                break

            if attempted >= int(effective_max_trials):
                break

            if all_points and len(seen_samples) >= len(all_points):
                _log("stop gate: all pending/resumable search points have been explored")
                break

            ask, sampled, sampled_key, selection_source = _pick_next_sample(
                trainer_variants=trainer_variants,
                method=runtime_meta["method"],
                engine=engine,
                space=space,
                all_points=all_points,
                seen_keys=seen_samples,
                project_name=project_name,
                env=self.env,
                sca_variants=sca_variants,
            )

            if sampled is None or sampled_key is None:
                _log("stop gate: no remaining unique finite point")
                break

            # Reserve this concrete search point immediately.
            # This prevents the same point from being selected again during this
            # AutoSearch invocation, even if compilation, training, testing, or
            # metric collection later fails.
            sampled_key = str(sampled_key)
            seen_samples.add(sampled_key)

            # Count this point as attempted as soon as it is accepted for execution.
            attempted += 1
            _log(
                f"selected unique point: "
                f"attempt={attempted}/{effective_max_trials} "
                f"sample_key={sampled_key}"
            )
            global_trial_index = int(
                next_trial_index + attempted - 1
            )

            global_remaining_trials = max(
                int(total_remaining_trials - attempted),
                0,
            )

            trial_id_i = (
                int(ask.trial_id)
                if ask is not None
                else int(global_trial_index)
            )

            trial_for_resolver = (
                None
                if str(selection_source).startswith("finite_")
                else ask
            )

            #
            # Materialized SCA is already a concrete scientific contract.
            #
            # Inject it into resolver_cfg rather than passing it through the
            # generic sampled overlay. This prevents the original list-valued
            # cfg["sca"] search specification from replacing the selected
            # concrete SCA variant.
            #
            (
                resolver_cfg,
                resolver_sampled,
            ) = _prepare_resolver_inputs(
                cfg=cfg,
                sampled=sampled,
            )

            run_cfg = resolve_model_contract(
                cfg=resolver_cfg,
                mode="auto_search",
                sampled=resolver_sampled,
                trial=trial_for_resolver,
            )
            run_cfg = coerce_types(run_cfg)
            _assert_trainer_preserved(sampled["trainer"], run_cfg["trainer"])

            #
            # Hard runtime boundary:
            #
            # AutoSearch must hand SCC one concrete SCA configuration.
            # No active SCA sweep list is allowed beyond this point.
            #
            assert_concrete_sca_runtime(
                run_cfg.get("sca") or {}
            )
            sampled_sca = sampled.get("sca")

            if sampled_sca is not None:
                resolved_sca = run_cfg.get("sca")

                if resolved_sca != sampled_sca:
                    raise RuntimeError(
                        "AutoSearch SCA contract changed during "
                        "model_contract_resolver resolution. "
                        "The concrete sampled SCA configuration must be "
                        "preserved exactly."
                    )
            # Downstream SCC train.py must consume this resolved contract as-is and must
            # not re-run resolve_model_contract() in SCC mode.
            run_cfg.setdefault("runtime_contract", {})
            if not isinstance(run_cfg["runtime_contract"], dict):
                run_cfg["runtime_contract"] = {}

            run_cfg["runtime_contract"]["resolved"] = True
            run_cfg["runtime_contract"]["mode"] = "auto_search"
            run_cfg["runtime_contract"]["resolver"] = "optuna_lite"
            run_cfg["runtime_contract"]["sampled_key"] = str(sampled_key)
            run_cfg["runtime_contract"]["selection_source"] = str(selection_source)
            
            sampled_model_id = run_cfg.get("model_id")
            #set up mapping between model_hash, model_name and parameters
            identity = build_run_identity_label(
                run_cfg,
                sampled=sampled,
                model_id=sampled_model_id,
            )

            model_name = str(identity["model_name"])
            model_hash = str(identity["model_hash"])
            if model_hash in trainer_by_hash and trainer_by_hash[model_hash] != run_cfg["trainer"]:
                raise RuntimeError("Different trainer choices share a model_hash; update nested trainer hashing")
            trainer_by_hash[model_hash] = copy.deepcopy(run_cfg["trainer"])

            run_cfg["model_name"] = str(model_name)
            run_cfg["model_hash"] = str(model_hash)
            run_cfg["frozen_identity"] = {
                "model_name": str(model_name),
                "model_hash": str(model_hash),
            }
            run_cfg.setdefault("run", {})
            run_cfg["run"]["model_name"] = str(model_name)

            project_root = Path(run_cfg["project"]["root"]).resolve()


            run_cfg.setdefault("runtime", {})
            if not isinstance(run_cfg["runtime"], dict):
                run_cfg["runtime"] = {}

            run_cfg["runtime"]["execution_source"] = "auto_search"
            run_cfg["runtime"]["driver"] = "optuna_lite"
            run_cfg["runtime"]["trial_index"] = int(global_trial_index)

            run_cfg["runtime"]["trial_total"] = int(total_trials)
            run_cfg["runtime"]["total_trials"] = int(total_trials)

            run_cfg["runtime"]["total_remaining_trials"] = int(
                global_remaining_trials
            )

            run_cfg["runtime"]["trial_id"] = str(trial_id_i)

            try:
                _log(
                    f"▶  SCC trial={global_trial_index}/{total_trials} "
                    f"remaining={global_remaining_trials} "
                    f"trial_id={trial_id_i} "
                    f"exec_mode={mode} "
                    f"source={selection_source} "
                    f"model_id={sampled_model_id} "
                    f"sampled={json.dumps(sampled, default=str)}"
                )
                start_run_map_entry(
                    cfg=run_cfg,
                    model_hash=model_hash,
                    model_name=model_name,
                    model_id=identity.get("model_id"),
                    sampled=identity.get("sampled") or sampled,
                    mode="auto_search",
                    trial_index=int(global_trial_index),
                    total_remaining_trials=int(global_remaining_trials),
                    total_trials=int(effective_max_trials),
                    selection_source=str(selection_source),
                    verify=True,
                )

                _log(
                    f"run_map verify-after-write: model_hash={model_hash} "
                    f"file_status=IN_PROGRESS "
                    f"path={project_root / 'artifacts' / 'run_map.jsonl'}"
                )
                out = run_trial_via_scc_lite(
                    run_cfg=run_cfg,
                    mode=mode,
                    trial_id=str(trial_id_i),
                )

            except (KeyboardInterrupt, SystemExit):
                # Process lifecycle/interruption is handled by orchestrator_auto_mlops.sh
                # and startup registry repair.
                raise

            except Exception as exc:
                _log(f"Trial {trial_id_i} FAIL during SCC execution: {exc}")

                if ask is not None:
                    try:
                        engine.tell(
                            trial_id=trial_id_i,
                            values={metric: 0.0 for metric in objective_metrics},
                            state="FAIL",
                        )
                    except Exception:
                        pass

                model_hash = str(run_cfg.get("model_hash") or "").strip()

                results.append(
                    LiteTrialResult(
                        trial_id=trial_id_i,
                        status="FAIL",
                        model_id=(
                            str(sampled_model_id)
                            if sampled_model_id is not None
                            else None
                        ),
                        model_hash=model_hash,
                        reason=f"scc_exec_error:{exc}",
                    )
                )


                try:
                    fail_run_map_entry(
                        cfg=run_cfg,
                        model_hash=model_hash,
                        model_name=model_name,
                        model_id=identity.get("model_id"),
                        sampled=identity.get("sampled") or sampled,
                        mode="auto_search",
                        trial_index=int(global_trial_index),
                        total_remaining_trials=int(global_remaining_trials),
                        total_trials=int(effective_max_trials),
                        selection_source=str(selection_source),
                        reason=f"scc_exec_error:{exc}",
                    )
                except Exception:
                    pass

                continue

            finally:
                cleanup_trial_runtime()

            model_name = str(out.get("model_name") or "")
            model_hash = str(out.get("model_hash") or "")
            run_dir = out.get("run_dir") or None
            metrics_dir = out.get("metrics_dir") or None



            try:
                objectives = _read_objective_values(
                    metrics_dir=metrics_dir,
                    objective_metrics=objective_metrics,
                )

            except (KeyboardInterrupt, SystemExit):
                raise

            except Exception as exc:
                _log(f"Trial {trial_id_i} FAIL reading objectives: {exc}")

                if ask is not None:
                    try:
                        engine.tell(
                            trial_id=trial_id_i,
                            values={metric: 0.0 for metric in objective_metrics},
                            state="FAIL",
                        )
                    except Exception:
                        pass

                results.append(
                    LiteTrialResult(
                        trial_id=trial_id_i,
                        status="FAIL",
                        model_id=(
                            str(sampled_model_id)
                            if sampled_model_id is not None
                            else None
                        ),
                        model_name=model_name,
                        model_hash=model_hash,
                        run_dir=str(run_dir) if run_dir else None,
                        reason=f"metrics_read_error:{exc}",
                    )
                )
                try:
                    fail_run_map_entry(
                        cfg=run_cfg,
                        model_hash=model_hash,
                        model_name=model_name,
                        model_id=identity.get("model_id"),
                        sampled=identity.get("sampled") or sampled,
                        mode="auto_search",
                        run_dir=str(run_dir) if run_dir else None,
                        trial_index=int(global_trial_index),
                        total_remaining_trials=int(global_remaining_trials),
                        total_trials=int(effective_max_trials),
                        selection_source=str(selection_source),
                        reason=f"metrics_read_error:{exc}",
                    )
                except Exception:
                    pass

                continue

            _log(
                f"Trial {trial_id_i} DONE: "
                f"model_id={sampled_model_id} model_hash={model_hash} "
                f"model_name={model_name} objectives={json.dumps(objectives)}"
            )



            if ask is not None:
                try:
                    engine.tell(
                        trial_id=trial_id_i,
                        values=objectives,
                        state="DONE",
                    )
                except Exception:
                    pass

            results.append(
                LiteTrialResult(
                    trial_id=trial_id_i,
                    status="DONE",
                    model_id=(
                        str(sampled_model_id)
                        if sampled_model_id is not None
                        else None
                    ),
                    model_name=model_name,
                    model_hash=model_hash,
                    run_dir=str(run_dir) if run_dir else None,
                    objectives=objectives,
                )
            )
            complete_run_map_entry(
                    cfg=run_cfg,
                    model_hash=model_hash,
                    model_name=model_name,
                    model_id=identity.get("model_id"),
                    sampled=identity.get("sampled") or sampled,
                    mode="auto_search",
                    run_dir=str(run_dir) if run_dir else None,
                    total_remaining_trials=int(global_remaining_trials),
                    trial_index=int(global_trial_index),
                    total_trials=int(effective_max_trials),
                    selection_source=str(selection_source),
                )

        cleanup_trial_runtime()

        summary = LiteRunSummary(
            attempted=attempted,
            done=sum(1 for result in results if result.status == "DONE"),
            failed=sum(1 for result in results if result.status == "FAIL"),
            study_name=engine.study_name,
        )

        _log(
            f"done: attempted={summary.attempted} "
            f"done={summary.done} fail={summary.failed}"
        )

        return results, summary
