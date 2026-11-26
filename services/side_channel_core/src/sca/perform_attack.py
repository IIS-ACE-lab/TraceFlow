# services/side_channel_core/src/sca/perform_attack.py
"""
Repeated-attack evaluation for Side-Channel Analysis (SCA).

This module evaluates model probabilities on the active validation or test
split. Without ``attack_trace_budget`` every attack accumulates the whole
population and repetitions vary only its order. With a budget B, every attack
accumulates B traces of it (below). The budget is never a config key of
perform_attack: a Budget test puts its data entry's data[].attack_trace_budget
into the test's in-memory config (src/sca/test_request._apply_kind). Traces are never replaced and a budget is
never clipped: a budget larger than the population is an error.

Scientific contract
-------------------
For repetition ``r`` and evaluated trace count ``t``::

    rank_r(t) = one-based rank of the true AES-byte hypothesis
    GE(t)     = mean_r rank_r(t)
    SR@k(t)   = mean_r [rank_r(t) <= k]

With ``attack_distribution: full_population`` and ``shuffle`` true, every
repetition is a distinct permutation of
all ``N`` traces in the active split. Consequently, intermediate rank, GE, and
SR curves may vary with trace order, while the final accumulated key scores and
final true-key rank must be invariant. At the full-population endpoint, GE is
therefore the deterministic final rank and SR@k is exactly zero or one.

Attack trace budget
-------------------
``attack_trace_budget: B`` of the test's attack (set by a Budget test, from the
data entry) sets the traces of one attack:

- ``full_population``: attack r draws its own seeded permutation of all N
  traces and accumulates its first B traces (the protocol of Zaid et al. 2020
  and Wouters et al. 2020). The attacks' traces overlap, so their endpoints
  are not independent trials, but they differ: GE at B is the mean of the
  attacks' final ranks and SR@k counts the attacks at rank <= k. Attack r's
  curve is exactly the first B points of the attack r drawn without a budget
  (same permutation, same tie-break stream). Needs B <= N.
- ``disjoint``: the N traces are cut into attack_times blocks of B traces,
  ``attack_times * B <= N``; the remaining traces are not attacked.

Evaluation populations
----------------------
Population size is defined exclusively by the active dataset split::

    validation -> data.validation_range
    test       -> data.test_range

During CV, validation instead uses cross_validation_runtime.validation_indices,
the ordered held-out row indices into the generated profiling pool. The CV
pipeline must also supply profiling_count, fold_index, and n_folds. These are
runtime provenance, not additional YAML settings. A CV population has no
contiguous configured_range; its ordered indices are fingerprinted separately.

The number of received predictions must equal the selected population size.
The legacy top-level ``perform_attack.attack_trace_budget``, subset-sampling
controls and budget-clipping policies are rejected; the budget is a key of
``perform_attack.validation`` / ``perform_attack.test``.

Canonical configuration
-----------------------
::

    perform_attack:
      enabled: true
      validation: {attack_times: 1, shuffle: false, attack_distribution: full_population}
      test: {attack_times: 100, shuffle: true, attack_distribution: full_population}
    data: [{name: ..., attack_trace_budget: 1500}]  # a Budget test's traces per attack
      attack_step: 1
      ge_thresholds: [1, 5]

``attack_times: 0`` means the process is not attacked. ``attack_distribution``
is ``full_population`` (every attack uses all N traces, as described above) or
``disjoint`` (the N traces are split into attack_times independent attacks, so
the endpoint GE and SR are means over independent attacks).

Endpoint trials (``sr_trials``): a full-population endpoint without a budget
is one trial however many orderings are drawn, so SR@k there is 0 or 1 out of
1; with a budget below N, and for ``disjoint``, the endpoint has
``attack_times`` trials.

Key-recovery evidence (final rank, top hypothesis, recovery traces) always
describes ONE attack on all N traces: the full-population attack (with a
budget below N, attack 1's permutation of all N traces), or for ``disjoint``
the combined attack (the partition in sequence: the blocks, then the traces no
block took). GE and SR stay the per-attack statistics, at the budget.

Ties are broken uniformly at random from seeded streams (see rank_curve.py).

``shuffle`` controls only ordering:

- ``true``: distinct reproducible full-population permutations
- ``false``: one deterministic full-population ordering; when repetitions are
  greater than one this configuration is rejected because it would duplicate
  the same curve

Public rank contract
--------------------
TraceFlow exposes one-based AES-byte ranks only:

- best rank: 1
- worst rank: 256
- top-k success: rank <= k
- GE threshold k: mean rank <= k

``sca.rank_base`` must therefore be exactly ``1``.

The returned protocol and recovery evidence record the configured split,
population size, ordering fingerprints, final-rank invariance, candidate
scores, and recovery thresholds for reproducible downstream reporting.

Author: Shraddha Mishra
"""


from __future__ import annotations

import hashlib
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sca.ascad_data_loader import resolve_active_data_config
from sca.dataset_types import label_target, require_attack_byte, resolved_variant
from sca.leakage_models import resolve_runtime_leakage_model
from sca.rank_curve import (
    KEYSPACE_SIZE,
    PUBLIC_RANK_BASE,
    TIE_BREAK_POLICY,
    fixed_key_attack_evidence,
    full_ranks_curve,
    order_with_true_key_at,
    score_rule,
    tie_break_rng,
    true_key_ranks,
)
from utils import attack_progress

KEY_RECOVERY_EVIDENCE_KEY = "__key_recovery_evidence__"

_SUPPORTED_PROCESSES = frozenset({"validation", "test"})
_PROCESS_STREAM_ID = {
    "validation": 0x56414C,
    "test": 0x54455354,
}
# Tie-break draws use their own streams, so they never shift trace orderings.
_TIE_STREAM_ID = 0x544945
_ENDPOINT_TIE_ATTACK = -1


def endpoint_tie_seed(cfg: Mapping[str, Any], process: Any) -> Any:
    """The tie-break seed of a process's full-population endpoint (tie_seed(_ENDPOINT_TIE_ATTACK) in
    perform_attacks): the single-run rank curve of the same process draws its ties from it too, so a tied
    endpoint gives the row one key_rank (owner 2026-10-23). A process the attack does not know (train):
    the run's seed, as before."""
    try:
        resolved = _normalize_process(process)
    except Exception:  # noqa: BLE001 - not an attacked process
        return _get_randomization_state(cfg)
    if resolved not in _PROCESS_STREAM_ID:
        return _get_randomization_state(cfg)
    return np.random.SeedSequence([_get_randomization_state(cfg), _PROCESS_STREAM_ID[resolved], _TIE_STREAM_ID,
                                   _ENDPOINT_TIE_ATTACK + 2])
_COMBINED_TIE_ATTACK = -2

# full_population: every attack uses all N traces, reordered when shuffled.
# disjoint: the N traces are split into attack_times independent attacks.
_ATTACK_DISTRIBUTIONS = frozenset({"full_population", "disjoint"})
# The keys of perform_attack.validation / perform_attack.test.
PROCESS_ATTACK_KEYS = ("attack_times", "shuffle", "attack_distribution", "attack_trace_budget")
_MOVED_ATTACK_FIELDS = ("attack_repetitions", "attack_shuffle")

_OBSOLETE_ATTACK_FIELDS = frozenset(          # top level; the budget is a per-process key
    {
        "attack_trace_budget",
        "sampling_mode",
        "budget_exceed_policy",
        "attack_budget_overflow_policy",
        "allow_budget_clipping",
        "require_independent_final_subsets",
        "require_distinct_final_subsets",
    }
)


# =============================================================================
# Strict configuration helpers
# =============================================================================
def _strict_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be true or false, got {value!r}")
    return value


def _strict_int(
    value: Any,
    *,
    field: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, got boolean {value!r}")

    if isinstance(value, (int, np.integer)):
        result = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise TypeError(f"{field} must be an integer, got empty string")
        try:
            result = int(text, 10)
        except ValueError as exc:
            raise TypeError(f"{field} must be an integer, got {value!r}") from exc
    else:
        raise TypeError(f"{field} must be an integer, got {type(value).__name__}")

    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be <= {maximum}, got {result}")
    return result


def _strict_nonnegative_float(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be numeric, got boolean {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric, got {value!r}") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and >= 0, got {result!r}")
    return result


def _normalize_process(process: Any) -> str:
    normalized = str(process).strip().lower()
    aliases = {
        "val": "validation",
        "validate": "validation",
        "valid": "validation",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _SUPPORTED_PROCESSES:
        raise ValueError(
            "Repeated attacks require process='validation' or process='test', "
            f"got {process!r}"
        )
    return normalized


def _get_perform_attack_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    section = cfg.get("perform_attack") or {}
    if not isinstance(section, Mapping):
        raise TypeError("cfg['perform_attack'] must be a mapping")
    return section


def _validate_attack_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    section = _get_perform_attack_cfg(cfg)
    moved = [field for field in _MOVED_ATTACK_FIELDS if field in section]
    if moved:
        raise ValueError(
            f"perform_attack.{moved} moved into perform_attack.validation / "
            "perform_attack.test as attack_times / shuffle"
        )
    stale = sorted(field for field in _OBSOLETE_ATTACK_FIELDS if field in section)
    if stale:
        raise ValueError(
            "Obsolete perform_attack configuration detected: "
            f"{stale}. Remove these fields. Validation/test population size is "
            "defined only by data.validation_range and data.test_range; the traces "
            "of one test attack by the data entry's data[].attack_trace_budget (a Budget test)."
        )
    return section


def _get_perform_attack_enabled(cfg: Mapping[str, Any]) -> bool:
    section = _validate_attack_config(cfg)
    return _strict_bool(
        section.get("enabled", True),
        field="perform_attack.enabled",
    )


def attack_plan(cfg: Mapping[str, Any], process: str) -> Dict[str, Any]:
    """
    ``perform_attack.<process>``: attack_times (0 = no attack on this process),
    shuffle, attack_distribution and attack_trace_budget (the traces of one
    attack; empty or absent: every trace, returned as None).
    """
    field = f"perform_attack.{process}"
    if process not in _SUPPORTED_PROCESSES:
        raise ValueError(f"{field}: process must be one of {sorted(_SUPPORTED_PROCESSES)}")
    block = _validate_attack_config(cfg).get(process)
    if not isinstance(block, Mapping):
        raise KeyError(
            f"{field} is required: {{attack_times, shuffle, attack_distribution}}"
        )
    unknown = sorted(set(block) - set(PROCESS_ATTACK_KEYS))
    if unknown:
        raise ValueError(f"{field} has unknown fields {unknown}")
    times = _strict_int(block.get("attack_times"), field=f"{field}.attack_times", minimum=0)
    shuffle = _strict_bool(block.get("shuffle"), field=f"{field}.shuffle")
    distribution = block.get("attack_distribution")
    if distribution not in _ATTACK_DISTRIBUTIONS:
        raise ValueError(
            f"{field}.attack_distribution must be one of "
            f"{sorted(_ATTACK_DISTRIBUTIONS)}, got {distribution!r}"
        )
    if distribution == "full_population" and times > 1 and not shuffle:
        raise ValueError(
            f"{field}: attack_times > 1 over the full population with "
            "shuffle=false repeats the identical attack. Set shuffle: true."
        )
    budget = block.get("attack_trace_budget")
    if budget is not None:
        budget = _strict_int(budget, field=f"{field}.attack_trace_budget", minimum=1)
    return {"attack_times": times, "shuffle": shuffle, "attack_distribution": distribution,
            "attack_trace_budget": budget}


def traces_per_attack(plan: Mapping[str, Any], population_size: int, *, field: str) -> int:
    """The traces one attack of ``plan`` (attack_plan) accumulates out of ``population_size``.

    Raises ValueError when the population cannot hold the attacks: a budget above N, or for
    disjoint attacks attack_times blocks that do not fit (no budget: attack_times must divide N)."""
    times = int(plan["attack_times"])
    budget = plan.get("attack_trace_budget")
    disjoint = plan["attack_distribution"] == "disjoint"
    population_size = int(population_size)
    if budget is None:
        if disjoint and times and population_size % times:
            raise ValueError(
                f"{field}: {population_size} traces do not split into {times} equal "
                "disjoint attacks (set attack_trace_budget to cut blocks of that size)"
            )
        return population_size // times if (disjoint and times) else population_size
    budget = int(budget)
    if disjoint:
        if times * budget > population_size:
            raise ValueError(
                f"{field}: {times} disjoint attacks of attack_trace_budget={budget} traces need "
                f"{times * budget} traces; the population has {population_size}"
            )
    elif budget > population_size:
        raise ValueError(
            f"{field}.attack_trace_budget={budget} exceeds the {population_size} traces of the "
            "population (a budget is never clipped)"
        )
    return budget


def _get_attack_step(cfg: Mapping[str, Any]) -> int:
    return _strict_int(
        _get_perform_attack_cfg(cfg).get("attack_step", 1),      # default 1, every trace (owner 2026-10-23)
        field="perform_attack.attack_step",
        minimum=1,
    )


def _get_ge_thresholds(cfg: Mapping[str, Any]) -> List[int]:
    values = _get_perform_attack_cfg(cfg).get("ge_thresholds", [1])     # default [1] (owner 2026-10-23)

    if isinstance(values, (int, np.integer)) and not isinstance(
        values, (bool, np.bool_)
    ):
        values = [int(values)]
    elif isinstance(values, str):
        text = (
            values.strip()
            .replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
        )
        if not text:
            raise ValueError("perform_attack.ge_thresholds must not be empty")
        values = [part for part in text.replace(",", " ").split() if part]
    elif isinstance(values, tuple):
        values = list(values)
    elif isinstance(values, np.ndarray):
        values = values.tolist()

    if not isinstance(values, list):
        raise TypeError(
            "perform_attack.ge_thresholds must be an integer or list-like, "
            f"got {type(values).__name__}"
        )

    normalized = {
        _strict_int(
            raw,
            field=f"perform_attack.ge_thresholds[{index}]",
            minimum=1,
            maximum=KEYSPACE_SIZE,
        )
        for index, raw in enumerate(values)
    }
    if not normalized:
        raise ValueError("perform_attack.ge_thresholds must not be empty")

    # Rank-1 recovery is always part of TraceFlow's recovery contract.
    normalized.add(1)
    return sorted(normalized)


def _get_rank_base(cfg: Mapping[str, Any]) -> int:
    sca = cfg.get("sca") or {}
    if not isinstance(sca, Mapping):
        raise TypeError("cfg['sca'] must be a mapping")
    rank_base = _strict_int(
        sca.get("rank_base", PUBLIC_RANK_BASE),
        field="sca.rank_base",
    )
    if rank_base != PUBLIC_RANK_BASE:
        raise ValueError(
            f"Unsupported sca.rank_base={rank_base}. TraceFlow requires "
            "one-based public ranks."
        )
    return rank_base


def _get_attack_byte(cfg: Mapping[str, Any]) -> int:
    sca = cfg.get("sca") or {}
    if not isinstance(sca, Mapping):
        raise TypeError("cfg['sca'] must be a mapping")
    if "attack_byte" not in sca:
        raise KeyError("perform_attack requires cfg['sca']['attack_byte']")
    return _strict_int(
        sca["attack_byte"],
        field="sca.attack_byte",
        minimum=0,
        maximum=15,
    )


def _get_leakage_model(cfg: Mapping[str, Any]) -> str:
    if not isinstance(cfg, MutableMapping):
        mutable_cfg: MutableMapping[str, Any] = dict(cfg)
    else:
        mutable_cfg = cfg
    return str(
        resolve_runtime_leakage_model(
            mutable_cfg,
            default=None,
            mutate=False,
        )
    )


def _get_rank_log_eps(cfg: Mapping[str, Any]) -> float:
    metrics_cfg = cfg.get("metrics_cfg") or {}
    if not isinstance(metrics_cfg, Mapping):
        raise TypeError("cfg['metrics_cfg'] must be a mapping")
    # 0 disables the per-trace floor (exact float64 log-likelihood; needs
    # log-probability input). See rank_curve.py "Numeric contract".
    value = _strict_nonnegative_float(
        metrics_cfg.get("rank_log_eps", 1e-40),
        field="metrics_cfg.rank_log_eps",
    )
    if value > 1.0:
        raise ValueError("metrics_cfg.rank_log_eps must be <= 1")
    return value


def _get_strict_prob_check(cfg: Mapping[str, Any]) -> bool:
    metrics_cfg = cfg.get("metrics_cfg") or {}
    if not isinstance(metrics_cfg, Mapping):
        raise TypeError("cfg['metrics_cfg'] must be a mapping")
    return _strict_bool(
        metrics_cfg.get("strict_prob_check", True),
        field="metrics_cfg.strict_prob_check",
    )


def _get_prob_atol(cfg: Mapping[str, Any]) -> float:
    metrics_cfg = cfg.get("metrics_cfg") or {}
    if not isinstance(metrics_cfg, Mapping):
        raise TypeError("cfg['metrics_cfg'] must be a mapping")
    return _strict_nonnegative_float(
        metrics_cfg.get("prob_atol", 1e-3),
        field="metrics_cfg.prob_atol",
    )


def _get_prob_rtol(cfg: Mapping[str, Any]) -> float:
    metrics_cfg = cfg.get("metrics_cfg") or {}
    if not isinstance(metrics_cfg, Mapping):
        raise TypeError("cfg['metrics_cfg'] must be a mapping")
    return _strict_nonnegative_float(
        metrics_cfg.get("prob_rtol", 1e-3),
        field="metrics_cfg.prob_rtol",
    )


def _get_randomization_state(cfg: Mapping[str, Any]) -> int:
    from utils.seed import run_seed     # seed: null: the seed this run drew (its params_out snapshot holds it)

    return _strict_int(
        run_seed(cfg),
        field="seed",
        minimum=0,
        maximum=2**63 - 1,
    )


def _get_variable_key(cfg: Mapping[str, Any]) -> bool:
    """Resolve key mode from sca.variant and reject conflicting aliases."""
    sca = cfg.get("sca") or {}
    if not isinstance(sca, Mapping):
        raise TypeError("cfg['sca'] must be a mapping")

    variant_raw = resolved_variant(cfg)   # the database type gives it (sca/dataset_types.py)
    alias_present = "variable_key" in sca
    alias_value = (
        _strict_bool(sca["variable_key"], field="sca.variable_key")
        if alias_present
        else None
    )

    if variant_raw is None:
        if alias_value is None:
            raise KeyError(
                "perform_attack requires cfg['sca']['variant']; do not infer "
                "fixed-key semantics during evaluation"
            )
        return bool(alias_value)
    if not isinstance(variant_raw, str):
        raise TypeError(f"sca.variant must be a string, got {variant_raw!r}")

    variant = variant_raw.strip().lower()
    if variant not in {"fixed_key", "random_key", "variable_key"}:
        raise ValueError(
            "sca.variant must be 'fixed_key', 'random_key', or 'variable_key', "
            f"got {variant_raw!r}"
        )
    resolved = variant in {"random_key", "variable_key"}
    if alias_value is not None and alias_value != resolved:
        raise ValueError(
            "Conflicting key-mode configuration: "
            f"sca.variant={variant!r}, sca.variable_key={alias_value!r}"
        )
    return resolved


def _key_recovery_reporting_enabled(cfg: Mapping[str, Any]) -> bool:
    reporting = cfg.get("key_recovery_reporting") or {}
    if not isinstance(reporting, Mapping):
        raise TypeError("cfg['key_recovery_reporting'] must be a mapping")
    return _strict_bool(
        reporting.get("enabled", False),
        field="key_recovery_reporting.enabled",
    )


# =============================================================================
# Evaluation-population resolution
# =============================================================================
def _active_data_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    # Generation, splitting, loading, and attack evaluation must select the
    # same active dataset from params.yaml/runtime context.
    return resolve_active_data_config(cfg)


def _strict_range(value: Any, *, field: str) -> Tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must be [start, end], got {value!r}")

    start = _strict_int(value[0], field=f"{field}[0]", minimum=0)
    end = _strict_int(value[1], field=f"{field}[1]", minimum=1)
    if end <= start:
        raise ValueError(f"{field} end must be > start, got {value!r}")
    return start, end


def _resolve_evaluation_population(
    cfg: Mapping[str, Any],
    *,
    process: Any,
    n_available: int,
) -> Dict[str, Any]:
    resolved_process = _normalize_process(process)
    cv = cfg.get("cross_validation", {})
    runtime = cfg.get("cross_validation_runtime", {})
    if not isinstance(cv, Mapping) or not isinstance(runtime, Mapping):
        raise TypeError("cross_validation and cross_validation_runtime must be mappings")
    cv_enabled = _strict_bool(cv.get("enabled", False), field="cross_validation.enabled")
    is_fold = _strict_bool(runtime.get("is_fold", False), field="cross_validation_runtime.is_fold")
    is_refit = _strict_bool(runtime.get("is_refit", False), field="cross_validation_runtime.is_refit")
    if resolved_process == "validation" and (cv_enabled or is_fold or is_refit):
        # Folds attack their held-out part and never read Attack_traces. The
        # final training is never attacked: its only traces are its training
        # traces.
        if is_refit:
            raise ValueError(
                "The CV final training is never attacked: its only traces are its "
                "training traces. Rank metrics come from held-out data (the CV "
                "folds, process validation, and the test)"
            )
        if not cv_enabled or not is_fold or runtime.get("active") is not True:
            raise ValueError("CV validation requires active fold runtime provenance")
        profiling_count = _strict_int(
            runtime.get("profiling_count"), field="cross_validation_runtime.profiling_count", minimum=1,
        )
        n_folds = _strict_int(
            runtime.get("n_folds"), field="cross_validation_runtime.n_folds", minimum=2,
            maximum=profiling_count,
        )
        fold_index = (
            _strict_int(
                runtime.get("fold_index"), field="cross_validation_runtime.fold_index",
                minimum=0, maximum=n_folds - 1,
            )
            if is_fold
            else None
        )
        raw_indices = runtime.get("validation_indices")
        if not isinstance(raw_indices, (list, tuple, np.ndarray)):
            raise TypeError("CV validation requires ordered validation_indices from the fold loader")
        indices = np.asarray(raw_indices)
        if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu":
            raise ValueError("CV validation_indices must be a non-empty one-dimensional integer sequence")
        if np.any(indices < 0) or np.any(indices >= profiling_count):
            raise ValueError("CV validation_indices are outside the profiling population")
        if np.unique(indices).size != indices.size:
            raise ValueError("CV validation_indices contain duplicate rows")
        if n_available != indices.size:
            raise ValueError(
                f"CV validation has {n_available} predictions but {indices.size} held-out indices"
            )
        return {
            "process": resolved_process,
            "population_source": "cross_validation_runtime.validation_indices",
            "configured_range": None,
            "evaluation_population_size": int(indices.size),
            "cv_provenance": {
                "source_group": "Profiling_traces",
                "index_coordinate_system": "configuration_specific_profiling_rows",
                "profiling_count": profiling_count,
                "fold_index": fold_index,
                "n_folds": n_folds,
                "validation_indices_sha256": _indices_fingerprint(indices, order_sensitive=True),
            },
        }
    data_cfg = _active_data_cfg(cfg)

    field = (
        "validation_range"
        if resolved_process == "validation"
        else "test_range"
    )
    if field not in data_cfg:
        raise KeyError(
            f"data.{field} is required for {resolved_process} attack evaluation"
        )

    start, end = _strict_range(data_cfg[field], field=f"data.{field}")
    expected = end - start

    if int(n_available) != expected:
        raise ValueError(
            f"Active {resolved_process} evaluation contains {n_available} traces, "
            f"but data.{field}=[{start}, {end}] defines {expected}. "
            "The DataLoader/split and runtime configuration are inconsistent."
        )

    return {
        "process": resolved_process,
        "population_source": f"data.{field}",
        "configured_range": [start, end],
        "evaluation_population_size": expected,
    }


# =============================================================================
# Sampling helpers
# =============================================================================
def _subset_metadata(metadata: Any, indices: np.ndarray) -> Any:
    try:
        return metadata[indices]
    except Exception:
        return [metadata[int(index)] for index in indices.tolist()]


def _indices_fingerprint(indices: np.ndarray, *, order_sensitive: bool) -> str:
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"indices must be one-dimensional, got {values.shape}")
    if not order_sensitive:
        values = np.sort(values)

    canonical = values.astype("<i8", copy=False)
    digest = hashlib.sha256()
    digest.update(b"TraceFlow repeated attack indices v2\0")
    digest.update(b"ordered\0" if order_sensitive else b"population\0")
    digest.update(np.asarray([canonical.size], dtype="<u8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _draw_full_population_order(
    *,
    rng: np.random.Generator,
    n_total: int,
    shuffle: bool,
    seen_order_fingerprints: set[str],
    repetition: int,
) -> Tuple[np.ndarray, str]:
    if not shuffle:
        indices = np.arange(n_total, dtype=np.int64)
        return indices, _indices_fingerprint(indices, order_sensitive=True)

    for _attempt in range(64):
        indices = rng.permutation(n_total).astype(np.int64, copy=False)
        order_hash = _indices_fingerprint(indices, order_sensitive=True)
        if order_hash not in seen_order_fingerprints:
            return indices, order_hash

    raise RuntimeError(
        "Could not generate a distinct full-population trace ordering for "
        f"repetition {repetition + 1}"
    )


# =============================================================================
# Metric helpers
# =============================================================================
def _validate_curve(
    curve: Any,
    *,
    expected_population: int,
    repetition: int,
) -> np.ndarray:
    array = np.asarray(curve, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        raise ValueError(
            f"Repeated attack {repetition + 1} returned invalid rank curve shape "
            f"{array.shape}; expected [M,2]"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"Repeated attack {repetition + 1} rank curve contains NaN/Inf"
        )

    x = array[:, 0]
    ranks = array[:, 1]
    if np.any(x <= 0.0) or np.any(np.diff(x) <= 0.0):
        raise ValueError(
            f"Repeated attack {repetition + 1} trace-count axis must be "
            "strictly increasing"
        )
    if not np.all(np.equal(x, np.floor(x))):
        raise ValueError(
            f"Repeated attack {repetition + 1} trace-count axis must be "
            "integer-valued"
        )
    if int(x[-1]) != int(expected_population):
        raise ValueError(
            f"Repeated attack {repetition + 1} ended at {int(x[-1])} traces; "
            f"expected {expected_population} traces per attack"
        )
    if np.any(ranks < PUBLIC_RANK_BASE) or np.any(ranks > KEYSPACE_SIZE):
        raise ValueError(
            f"Repeated attack {repetition + 1} ranks must lie in [1,256]"
        )
    if not np.all(np.equal(ranks, np.floor(ranks))):
        raise ValueError(
            f"Repeated attack {repetition + 1} ranks must be integer-valued"
        )
    return array


def _threshold_trace_points(
    curve: np.ndarray,
    *,
    threshold: int,
) -> Tuple[Optional[int], Optional[int]]:
    reached = curve[:, 1].astype(np.float64) <= float(threshold)
    if not np.any(reached):
        return None, None

    first_index = int(np.flatnonzero(reached)[0])
    first_trace = int(curve[first_index, 0])

    suffix_all = np.logical_and.accumulate(reached[::-1])[::-1]
    stable_indices = np.flatnonzero(suffix_all)
    stable_trace = (
        int(curve[int(stable_indices[0]), 0])
        if stable_indices.size
        else None
    )
    return first_trace, stable_trace


def _optional_trace_metric(value: Optional[int]) -> Optional[int]:
    """Preserve an unreached threshold as None; never publish NaN."""
    return int(value) if value is not None else None


def _success_rate_from_final_ranks(
    final_ranks: np.ndarray,
    *,
    threshold: int,
) -> Tuple[int, float]:
    if final_ranks.ndim != 1 or final_ranks.size == 0:
        raise ValueError("Cannot compute success rate from zero repeated attacks")
    count = int(
        np.count_nonzero(final_ranks.astype(np.float64) <= float(threshold))
    )
    return count, float(count / final_ranks.size)


def _stable_trace_where_curve_reaches(
    curve: np.ndarray,
    *,
    threshold: float,
) -> Optional[int]:
    """Earliest trace count from which the curve stays >= threshold to the end."""
    mask = curve[:, 1].astype(np.float64) >= float(threshold)
    suffix_all = np.logical_and.accumulate(mask[::-1])[::-1]
    stable = np.flatnonzero(suffix_all)
    return int(curve[int(stable[0]), 0]) if stable.size else None


def _candidate_rows(
    scores: np.ndarray,
    order: np.ndarray,
) -> List[Dict[str, float | int]]:
    scores = np.asarray(scores, dtype=np.float64)
    order = np.asarray(order, dtype=np.int64)

    if scores.shape != (KEYSPACE_SIZE,):
        raise ValueError(
            f"Candidate scores must have shape ({KEYSPACE_SIZE},), got {scores.shape}"
        )
    if order.shape != (KEYSPACE_SIZE,):
        raise ValueError(
            f"Candidate order must have shape ({KEYSPACE_SIZE},), got {order.shape}"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("Candidate scores contain non-finite values")
    if (
        np.any(order < 0)
        or np.any(order >= KEYSPACE_SIZE)
        or np.unique(order).size != KEYSPACE_SIZE
    ):
        raise ValueError(
            "Candidate order must contain every AES-byte hypothesis exactly once"
        )

    rows: List[Dict[str, float | int]] = []
    for rank_index, key_guess in enumerate(order, start=PUBLIC_RANK_BASE):
        guess = int(key_guess)
        rows.append(
            {
                "rank": rank_index,
                "key_guess": guess,
                "score": float(scores[guess]),
            }
        )
    return rows


# =============================================================================
# Public API
# =============================================================================
def perform_attacks(
    preds: torch.Tensor,
    metadata: Sequence[Any],
    cfg: Mapping[str, Any],
    *,
    process: Optional[str] = None,
    preds_are_log_probs: bool = False,
) -> Dict[str, Any]:
    """
    Run repeated full-population validation/test attacks.

    ``perform_attack.<process>`` sets the attacks: full_population repetitions
    consume all traces of the split, disjoint ones an equal share each.
    ``shuffle`` controls only the trace ordering. ``preds`` are float64
    log-probabilities when ``preds_are_log_probs`` is true (preferred),
    probabilities otherwise.
    """
    preds_are_log_probs = _strict_bool(
        preds_are_log_probs, field="preds_are_log_probs"
    )
    if not _get_perform_attack_enabled(cfg):
        raise ValueError(
            "perform_attack.enabled is false but perform_attacks() was called"
        )
    if process is None:
        raise ValueError(
            "perform_attacks requires process='validation' or process='test'"
        )
    if metadata is None:
        raise ValueError("perform_attacks requires metadata")
    if not isinstance(preds, torch.Tensor):
        raise TypeError(f"preds must be torch.Tensor, got {type(preds).__name__}")
    if preds.dim() != 2:
        raise ValueError(f"preds must be [N,C], got {tuple(preds.shape)}")
    if int(preds.shape[0]) == 0:
        raise ValueError("perform_attacks received empty predictions")
    if preds_are_log_probs:
        # log(0) = -inf is a legal log-probability; NaN and +inf are not.
        if torch.isnan(preds).any() or torch.isposinf(preds).any():
            raise ValueError("perform_attacks received NaN/+Inf log-probabilities")
    elif not torch.isfinite(preds).all():
        raise ValueError("perform_attacks received NaN/Inf predictions")
    if not hasattr(metadata, "__len__"):
        raise TypeError("metadata must be sized")

    n_total = int(preds.shape[0])
    if len(metadata) != n_total:
        raise ValueError(
            f"metadata length {len(metadata)} does not match N={n_total}"
        )

    _get_rank_base(cfg)
    population = _resolve_evaluation_population(
        cfg,
        process=process,
        n_available=n_total,
    )
    resolved_process = str(population["process"])
    population_size = int(population["evaluation_population_size"])

    plan = attack_plan(cfg, resolved_process)
    repetitions = plan["attack_times"]
    shuffle = plan["shuffle"]
    disjoint = plan["attack_distribution"] == "disjoint"
    budget = plan["attack_trace_budget"]
    if repetitions == 0:
        raise ValueError(
            f"perform_attack.{resolved_process}.attack_times is 0: this process "
            "is not attacked, so its rank metrics cannot be computed"
        )
    n_per_attack = traces_per_attack(plan, population_size, field=f"perform_attack.{resolved_process}")
    # Every attack accumulates the whole population (in its own order): its endpoint is the
    # order-invariant full-population endpoint. False for disjoint attacks and a budget below N.
    whole_population = not disjoint and n_per_attack == population_size

    step = _get_attack_step(cfg)
    thresholds = _get_ge_thresholds(cfg)
    attack_byte = _get_attack_byte(cfg)
    leakage_model = _get_leakage_model(cfg)
    target = label_target(cfg)                 # the database type's label intermediate (sca/dataset_types.py)
    require_attack_byte(cfg)
    rank_eps = _get_rank_log_eps(cfg)
    randomization_state = _get_randomization_state(cfg)
    strict_prob_check = _get_strict_prob_check(cfg)
    prob_atol = _get_prob_atol(cfg)
    prob_rtol = _get_prob_rtol(cfg)

    sca = cfg.get("sca") or {}
    if not isinstance(sca, Mapping):
        raise TypeError("cfg['sca'] must be a mapping")

    variable_key = _get_variable_key(cfg)
    if variable_key:
        raise ValueError(
            "Repeated full-population key recovery is defined only for "
            "sca.variant='fixed_key'. random_key/variable_key data can use "
            "the diagnostic grouped curve in rank_curve.py, whose axis is "
            "traces per key group rather than the validation/test population."
        )
    variable_key_aggregation = str(
        sca.get("variable_key_aggregation", "weighted")
    ).strip().lower()
    if variable_key_aggregation not in {"weighted", "equal"}:
        raise ValueError(
            f"Invalid variable_key_aggregation={variable_key_aggregation!r}. "
            "Expected 'weighted' or 'equal'."
        )
    if variable_key and _key_recovery_reporting_enabled(cfg):
        raise ValueError(
            "Explicit key_recovery_reporting is not defined for a mixed "
            "variable-key population. Disable reporting or evaluate fixed-key "
            "groups separately."
        )

    root_sequence = np.random.SeedSequence(
        [randomization_state, _PROCESS_STREAM_ID[resolved_process]]
    )
    child_sequences = root_sequence.spawn(repetitions)

    def tie_seed(attack: int) -> np.random.SeedSequence:
        # Non-negative entropy words; attack -1/-2 are the endpoint/combined draws.
        return np.random.SeedSequence(
            [randomization_state, _PROCESS_STREAM_ID[resolved_process],
             _TIE_STREAM_ID, attack + 2]
        )

    per_attack_curves: List[np.ndarray] = []
    fixed_attack_evidence: List[Dict[str, Any]] = []
    order_fingerprints: List[str] = []
    seen_order_fingerprints: set[str] = set()

    attacked = np.zeros(population_size, dtype=bool)
    full_population_indices = np.arange(population_size, dtype=np.int64)
    full_population_fingerprint = _indices_fingerprint(
        full_population_indices,
        order_sensitive=False,
    )

    if disjoint:
        # One partition of the population; attack r takes block r of it.
        partition = (
            np.random.default_rng(root_sequence).permutation(population_size).astype(np.int64)
            if shuffle
            else full_population_indices
        )

    for repetition, child_sequence in enumerate(child_sequences):
        if disjoint:
            indices = partition[
                repetition * n_per_attack:(repetition + 1) * n_per_attack
            ]
            order_hash = _indices_fingerprint(indices, order_sensitive=True)
            seen_order_fingerprints.add(order_hash)
        else:
            rng = np.random.default_rng(child_sequence)
            full_order, full_hash = _draw_full_population_order(
                rng=rng,
                n_total=population_size,
                shuffle=shuffle,
                seen_order_fingerprints=seen_order_fingerprints,
                repetition=repetition,
            )
            seen_order_fingerprints.add(full_hash)       # distinct permutations of all N
            if repetition == 0:
                first_full_order = full_order            # the key evidence's attack on all N (budget < N)
            # With a budget, the attack is the permutation's first B traces: exactly the first B
            # points of the attack drawn without one.
            indices = full_order[:n_per_attack]
            order_hash = (full_hash if whole_population
                          else _indices_fingerprint(indices, order_sensitive=True))
        order_fingerprints.append(order_hash)
        attacked[indices] = True

        if indices.size != n_per_attack:
            raise RuntimeError(
                "Internal error: repeated attack did not select "
                f"{n_per_attack} traces"
            )
        if np.unique(indices).size != n_per_attack:
            raise RuntimeError(
                "Internal error: repeated attack trace ordering contains duplicates"
            )

        index_tensor = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=preds.device,
        )
        preds_ordered = preds.index_select(0, index_tensor)
        metadata_ordered = _subset_metadata(metadata, indices)

        if variable_key:
            curve = full_ranks_curve(
                preds=preds_ordered,
                metadata=metadata_ordered,
                target_byte=attack_byte,
                step=step,
                eps=rank_eps,
                strict_prob_check=strict_prob_check,
                prob_atol=prob_atol,
                prob_rtol=prob_rtol,
                check_fixed_key=False,
                variable_key=True,
                variable_key_aggregation=variable_key_aggregation,
                rank_base=PUBLIC_RANK_BASE,
                leakage_model=leakage_model,
                target=target,
                preds_are_log_probs=preds_are_log_probs,
                tie_break_seed=tie_seed(repetition),
            )
        else:
            evidence = fixed_key_attack_evidence(
                preds=preds_ordered,
                metadata=metadata_ordered,
                target_byte=attack_byte,
                step=step,
                eps=rank_eps,
                strict_prob_check=strict_prob_check,
                prob_atol=prob_atol,
                prob_rtol=prob_rtol,
                check_fixed_key=True,
                rank_base=PUBLIC_RANK_BASE,
                leakage_model=leakage_model,
                target=target,
                preds_are_log_probs=preds_are_log_probs,
                tie_break_seed=tie_seed(repetition),
            )
            if not isinstance(evidence, Mapping):
                raise TypeError("fixed_key_attack_evidence must return a mapping")
            fixed_attack_evidence.append(dict(evidence))
            curve = evidence.get("curve")

        normalized_curve = _validate_curve(
            curve,
            expected_population=n_per_attack,
            repetition=repetition,
        )

        if not variable_key:
            reported_rank = _strict_int(
                fixed_attack_evidence[-1].get("final_true_key_rank"),
                field=f"attack[{repetition + 1}].final_true_key_rank",
                minimum=1,
                maximum=KEYSPACE_SIZE,
            )
            curve_rank = int(normalized_curve[-1, 1])
            if reported_rank != curve_rank:
                raise ValueError(
                    f"Repeated attack {repetition + 1} evidence rank "
                    f"{reported_rank} disagrees with curve final rank {curve_rank}"
                )

        per_attack_curves.append(normalized_curve)
        # the TraceFlow Monitor's test progress (utils/attack_progress.py): a no-op but in the test's own block
        attack_progress.report(resolved_process, repetition + 1, len(child_sequences), normalized_curve)

    reference_shape = per_attack_curves[0].shape
    reference_axis = per_attack_curves[0][:, 0]
    for repetition, curve in enumerate(per_attack_curves[1:], start=1):
        if curve.shape != reference_shape:
            raise ValueError(
                "Repeated attack curves have inconsistent shapes: "
                f"attack 1={reference_shape}, attack "
                f"{repetition + 1}={curve.shape}"
            )
        if not np.array_equal(curve[:, 0], reference_axis):
            raise ValueError(
                "Repeated attack curves have inconsistent trace-count axes"
            )

    x = reference_axis.astype(np.float64, copy=False)
    rank_stack = np.stack(
        [
            curve[:, 1].astype(np.float64, copy=False)
            for curve in per_attack_curves
        ],
        axis=0,
    )
    # Full-population attacks consume the same traces, so the endpoint is one
    # population: rank it once, from the mean scores (float64 summation order
    # can flip near-tied keys), with one tie-break draw shared by every
    # ordering. A material score difference still raises in the evidence check.
    endpoint: Optional[Dict[str, Any]] = None
    if whole_population and not variable_key:
        endpoint_scores = np.mean(
            [item["final_key_scores"] for item in fixed_attack_evidence], axis=0
        )
        endpoint_key = int(fixed_attack_evidence[0]["true_key_byte"])
        endpoint_ranks, endpoint_ties = true_key_ranks(
            endpoint_scores, endpoint_key, tie_break_rng(tie_seed(_ENDPOINT_TIE_ATTACK)),
            population_size,
        )
        endpoint = {
            "scores": endpoint_scores,
            "rank": int(endpoint_ranks[0]),
            "tie_count": int(endpoint_ties[0]),
        }
        rank_stack[:, -1] = endpoint["rank"]
        for curve in per_attack_curves:
            curve[-1, 1] = endpoint["rank"]

    mean_ranks = np.mean(rank_stack, axis=0)
    std_ranks = np.std(rank_stack, axis=0, ddof=0)
    mean_attack_rank_curve = np.stack([x, mean_ranks], axis=1)
    std_attack_rank_curve = np.stack([x, std_ranks], axis=1)

    final_ranks = rank_stack[:, -1]
    # Endpoint trials: one per attack, except a whole-population endpoint, which
    # is one trial however many orderings were drawn.
    trial_ranks = final_ranks[:1] if whole_population else final_ranks
    sr_trials = int(trial_ranks.size)

    guessing_entropy = float(np.mean(final_ranks))

    success_count_k: Dict[int, int] = {}
    success_rate_k: Dict[int, float] = {}
    success_rate_percent_k: Dict[int, float] = {}
    success_rate_curves: Dict[int, np.ndarray] = {}
    traces_to_sr_k: Dict[int, Optional[int]] = {}
    ge_threshold_status: Dict[int, Dict[str, Any]] = {}
    # Headline traces_to_ge_k is the STABLE crossing: the earliest trace count
    # from which the mean rank stays <= k to the end of the population. The
    # first (possibly transient) crossing is kept as first_traces_to_ge_k.
    traces_to_ge_k: Dict[int, Optional[int]] = {}
    first_traces_to_ge_k: Dict[int, Optional[int]] = {}
    stable_traces_to_ge_k: Dict[int, Optional[int]] = {}

    for threshold in thresholds:
        success_count, success_rate = _success_rate_from_final_ranks(
            trial_ranks,
            threshold=threshold,
        )
        success_count_k[threshold] = success_count
        success_rate_k[threshold] = success_rate
        success_rate_percent_k[threshold] = 100.0 * success_rate
        sr_values = np.mean(
            rank_stack <= float(threshold),
            axis=0,
            dtype=np.float64,
        )
        sr_curve = np.stack([x, sr_values], axis=1)
        success_rate_curves[threshold] = sr_curve
        traces_to_sr_k[threshold] = _optional_trace_metric(
            _stable_trace_where_curve_reaches(sr_curve, threshold=1.0)
        )

        first_trace, stable_trace = _threshold_trace_points(
            mean_attack_rank_curve,
            threshold=threshold,
        )
        trace_population = int(x[-1])
        if first_trace is None:
            message = (
                f"GE never reached <= {threshold} within "
                f"{trace_population} traces"
            )
        elif stable_trace is None:
            message = (
                f"GE touched <= {threshold} at trace {first_trace} but did not "
                f"stay there through {trace_population} traces (not stable)"
            )
        else:
            message = (
                f"GE stably <= {threshold} from trace {stable_trace} "
                f"(first crossing at {first_trace})"
            )

        ge_threshold_status[threshold] = {
            "threshold": threshold,
            "reached": first_trace is not None,
            "first_trace": first_trace,
            "stable": stable_trace is not None,
            "stable_trace": stable_trace,
            "message": message,
        }
        traces_to_ge_k[threshold] = _optional_trace_metric(stable_trace)
        first_traces_to_ge_k[threshold] = _optional_trace_metric(first_trace)
        stable_traces_to_ge_k[threshold] = _optional_trace_metric(stable_trace)

    success_count = success_count_k[1]
    failed_attack_count = sr_trials - success_count
    success_rate = success_rate_k[1]
    success_rate_percent = success_rate_percent_k[1]

    protocol: Dict[str, Any] = {
        "process": resolved_process,
        "population_source": population["population_source"],
        "configured_range": population["configured_range"],
        "evaluation_population_size": population_size,
        "traces_available": population_size,
        "traces_used": population_size,                 # the key evidence's attack: every trace
        "full_population_used": True,
        "attack_distribution": plan["attack_distribution"],
        "attack_trace_budget": budget,                  # None: every trace
        "traces_per_attack": n_per_attack,
        "traces_attacked": int(np.count_nonzero(attacked)),   # distinct traces some attack used
        "sampling_mode": (
            ("disjoint_random_partition" if shuffle else "disjoint_ordered_blocks")
            if disjoint
            else (
                ("random_permutation_without_replacement" if whole_population
                 else "random_permutation_first_budget_traces")
                if shuffle
                else ("ordered_full_population" if whole_population else "ordered_first_budget_traces")
            )
        ),
        "attack_shuffle": shuffle,
        "repetition_unit": (
            "disjoint_trace_subset" if disjoint
            else ("trace_ordering" if whole_population else "overlapping_trace_subset")
        ),
        "curve_semantics": (
            "mean_over_independent_attacks" if disjoint
            else ("mean_over_trace_orderings" if whole_population else "mean_over_budget_attacks")
        ),
        "endpoint_semantics": (
            "mean_final_rank_over_independent_attacks"
            if disjoint
            else ("deterministic_full_evaluation_population" if whole_population
                  else "mean_final_rank_over_overlapping_budget_attacks")
        ),
        "endpoint_is_independent_attack_estimate": disjoint,
        "endpoint_rank_order_invariant": whole_population,
        "attack_repetitions": repetitions,
        "sr_trials": sr_trials,
        "tie_break_policy": TIE_BREAK_POLICY,
        "score_rule": score_rule(leakage_model),
        "unique_trace_orders": len(set(order_fingerprints)),
        "unique_final_subsets": 1 if whole_population else repetitions,
        "final_subset_variability": not whole_population and repetitions > 1,
        "randomization_state": randomization_state,
        "final_subset_fingerprint_sha256": full_population_fingerprint,
        "order_fingerprints_sha256": list(order_fingerprints),
    }
    if "cv_provenance" in population:
        protocol["cv_provenance"] = population["cv_provenance"]

    result: Dict[str, Any] = {
        "mean_attack_rank_curve": mean_attack_rank_curve,
        "std_attack_rank_curve": std_attack_rank_curve,
        "guessing_entropy": guessing_entropy,
        "success_rate": success_rate,
        "success_rate_percent": success_rate_percent,
        "success_count": success_count,
        "successful_attacks": success_count,
        "failed_attack_count": failed_attack_count,
        "attack_repetitions": repetitions,
        "sr_trials": sr_trials,
        "success_rate_k": success_rate_k,
        "success_rate_percent_k": success_rate_percent_k,
        "success_count_k": success_count_k,
        "mean_attack_success_rate_curves": success_rate_curves,
        "traces_to_sr_1": traces_to_sr_k[1],
        "traces_to_sr_k": traces_to_sr_k,
        "traces_to_ge_1": traces_to_ge_k[1],
        "first_traces_to_ge_1": first_traces_to_ge_k[1],
        "stable_traces_to_ge_1": stable_traces_to_ge_k[1],
        "traces_to_ge_k": traces_to_ge_k,
        "first_traces_to_ge_k": first_traces_to_ge_k,
        "stable_traces_to_ge_k": stable_traces_to_ge_k,
        "traces_to_ge_1_message": ge_threshold_status[1]["message"],
        "ge_threshold_status": ge_threshold_status,
        "per_attack_curves": per_attack_curves,
        "final_ranks": final_ranks.astype(np.int64).tolist(),
        "final_rank_min": float(np.min(final_ranks)),
        "final_rank_mean": guessing_entropy,
        "final_rank_std": float(np.std(final_ranks, ddof=0)),
        "final_rank_max": float(np.max(final_ranks)),
        "sampling": protocol,
        "attack_protocol": protocol,
    }

    if not variable_key:
        if len(fixed_attack_evidence) != repetitions:
            raise ValueError(
                "Fixed-key repeated attack evidence count does not match repetitions"
            )

        true_keys = {
            _strict_int(
                item.get("true_key_byte"),
                field="fixed_attack_evidence.true_key_byte",
                minimum=0,
                maximum=255,
            )
            for item in fixed_attack_evidence
        }
        if len(true_keys) != 1:
            raise ValueError(
                "Repeated fixed-key attacks produced inconsistent true key bytes: "
                f"{sorted(true_keys)}"
            )
        true_key = next(iter(true_keys))

        score_vectors: List[np.ndarray] = []
        for repetition, item in enumerate(fixed_attack_evidence, start=1):
            scores = np.asarray(item.get("final_key_scores"), dtype=np.float64)
            if scores.shape != (KEYSPACE_SIZE,) or not np.all(np.isfinite(scores)):
                raise ValueError(
                    f"Repeated attack {repetition} returned invalid final_key_scores"
                )
            score_vectors.append(scores)

        score_stack = np.stack(score_vectors, axis=0)

        # Addition order may introduce tiny floating-point differences, but all
        # complete-population score vectors must remain numerically equivalent.
        if whole_population and repetitions > 1:
            reference_scores = score_stack[0]
            if not np.allclose(
                score_stack,
                reference_scores[None, :],
                rtol=1e-10,
                atol=1e-8,
            ):
                max_delta = float(
                    np.max(np.abs(score_stack - reference_scores[None, :]))
                )
                raise FloatingPointError(
                    "Full-population final key scores changed materially across "
                    f"trace permutations (max_abs_delta={max_delta:.3e})"
                )

        # The key evidence describes one attack on all N traces: the
        # full-population attack (its endpoint above); with a budget below N,
        # attack 1's permutation of all N traces; for disjoint the combined
        # attack, the partition in sequence (the blocks, then the traces no
        # block took). Rank, top hypothesis and recovery traces then come from
        # the same curve and scores.
        if not whole_population:
            combined_indices = partition if disjoint else first_full_order
            combined = fixed_key_attack_evidence(
                preds=preds.index_select(
                    0, torch.as_tensor(combined_indices, dtype=torch.long, device=preds.device)
                ),
                metadata=_subset_metadata(metadata, combined_indices),
                target_byte=attack_byte,
                step=step,
                eps=rank_eps,
                strict_prob_check=strict_prob_check,
                prob_atol=prob_atol,
                prob_rtol=prob_rtol,
                check_fixed_key=True,
                rank_base=PUBLIC_RANK_BASE,
                leakage_model=leakage_model,
                preds_are_log_probs=preds_are_log_probs,
                tie_break_seed=tie_seed(_COMBINED_TIE_ATTACK),
                target=target,
            )
            mean_scores = np.asarray(combined["final_key_scores"], dtype=np.float64)
            consensus_rank = int(combined["final_true_key_rank"])
            tie_count = int(combined["true_key_tie_count"])
            first_rank1 = combined["first_rank1_trace"]
            stable_rank1 = combined["stable_rank1_trace"]
        else:
            mean_scores = endpoint["scores"]
            consensus_rank = endpoint["rank"]
            tie_count = endpoint["tie_count"]
            first_rank1 = ge_threshold_status[1]["first_trace"]
            stable_rank1 = ge_threshold_status[1]["stable_trace"]
        consensus_order = order_with_true_key_at(
            mean_scores, true_key, consensus_rank, population_size
        )

        if not np.all(np.equal(final_ranks, np.floor(final_ranks))):
            raise ValueError("Fixed-key final ranks must be integer-valued")

        margin = None
        if consensus_order.size >= 2:
            top1 = int(consensus_order[0])
            top2 = int(consensus_order[1])
            margin = float(mean_scores[top1] - mean_scores[top2])

        evidence: Dict[str, Any] = {
            "cipher": "AES",
            "evaluation_process": resolved_process,
            "attack_byte": attack_byte,
            "key_mode": "fixed_key",
            "leakage_model": leakage_model,
            "true_key_byte": true_key,
            "top_key_hypothesis": int(consensus_order[0]),
            "final_true_key_rank": int(consensus_rank),
            "true_key_tie_count": tie_count,
            "tie_break_policy": TIE_BREAK_POLICY,
            "score_rule": score_rule(leakage_model),
            "ranking_basis": (
                "combined_attack_on_all_disjoint_traces"
                if disjoint
                else ("full_population_log_scores_order_invariant" if whole_population
                      else "first_attack_order_on_all_traces")
            ),
            "final_ranks": final_ranks.astype(np.int64).tolist(),
            "first_rank1_trace": first_rank1,
            "stable_rank1_trace": stable_rank1,

            # Canonical population provenance.
            "trace_population_source": population["population_source"],
            "trace_population_range": population["configured_range"],
            **({"cv_provenance": population["cv_provenance"]} if "cv_provenance" in population else {}),
            "traces_available": population_size,
            "traces_used": population_size,
            "full_population_used": True,

            "attack_repetitions": repetitions,
            "sr_trials": sr_trials,
            "recovered_attacks": success_count,
            "failed_attacks": failed_attack_count,
            "success_rate": success_rate,
            "success_rate_percent": success_rate_percent,
            "final_sr_at_1": success_rate,
            "final_ge": guessing_entropy,
            "final_rank_standard_deviation": float(
                np.std(final_ranks, ddof=0)
            ),
            "ge_threshold_status": ge_threshold_status,
            "sampling": protocol,
            "attack_protocol": protocol,
            "top_key_candidates": _candidate_rows(mean_scores, consensus_order),
            "top1_top2_score_margin": margin,
        }
        result[KEY_RECOVERY_EVIDENCE_KEY] = evidence

    return result
