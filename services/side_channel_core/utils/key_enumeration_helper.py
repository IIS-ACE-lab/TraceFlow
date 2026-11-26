# services/side_channel_core/utils/key_enumeration_helper.py
"""
Completed-cycle key-enumeration integration for Side-Channel Analysis (SCA).

This helper owns lifecycle integration only. Pure enumeration algorithms live
in ``src.sca.key_enumeration``.

Completed lifecycle:

construct validation
        ↓
construct test
        ↓
validate comparison validation report
        ↓
validate selected-mode test report
        ↓
ONLY NOW stage/write files

    training
    -> validation key ranking
    -> testing key ranking
    -> key enumeration
    -> strict report validation
    -> atomic report replacement
    -> combined summary

Supported enumeration modes:

    local_hamming
    neighbour_sweep
    support_weighted
    comparison

Policy rules
------------
local_hamming
    top_k, hamming_radius, and local_budget come directly from the validated
    configuration. LOCAL_HAMMING executes that policy exactly and performs no
    validation-time AI fallback or parameter resolution.

neighbour_sweep
    The policy comes directly from configuration. No validation or test true
    key participates in policy selection.

support_weighted
    ``top_k``, ``hamming_radius``, and ``local_budget`` are frozen from the
    validated configuration before split enumeration. Validation and test
    independently recompute weighted Hamming support from their own AI scores.

comparison
    Raw AI, local_hamming, neighbour_sweep, and support_weighted are benchmarked on validation
    evidence.

    Raw AI participates in final selection as the baseline, and an
    enumeration algorithm is selected only when it strictly improves upon
    the AI true-key rank.

    When AI is selected, no enumeration algorithm executes on test.
    Otherwise, the exact validation-selected enumeration policy is frozen
    and evaluated once on independent test evidence.

Scientific isolation
--------------------
- Test evidence never influences policy or strategy selection.
- Validation candidate values/scores are never reused as test candidate
  values/scores.
- Comparison evaluates all strategies only on validation evidence.
- Test evaluates only the validation-selected strategy.
- Existing top-level enumeration evidence is discarded before regeneration.
- Base key-recovery evidence is validated independently and is never silently
  repaired.

Artifact safety
---------------
Both enriched reports are fully constructed and validated before either
existing artifact is replaced. Replacement uses staged files and rollback
backups so an ordinary write/replace failure restores the original pair.

Author: Shraddha Mishra
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.sca.key_enumeration import (
    KEYSPACE_SIZE,
    PUBLIC_RANK_BASE,
    KeyEnumerationError,
    run_key_enumeration,
    validate_key_enumeration_contract,
)
from src.sca.key_recovery import (
    key_recovery_paths,
    validate_key_enumeration_report_section,
)
from services.side_channel_core.utils.seal import strip as strip_seal


class KeyEnumerationHelperError(RuntimeError):
    """Raised when completed key-recovery artifacts cannot be enumerated."""


def _resolve_model_name(
    cfg: Mapping[str, Any],
) -> str:
    """Resolve the concrete model identity used by key-recovery paths."""
    runtime = cfg.get("runtime") or {}

    if not isinstance(runtime, Mapping):
        raise KeyEnumerationHelperError(
            "cfg.runtime must be a mapping"
        )

    model_name = (
        cfg.get("model_name")
        or runtime.get("model_name")
    )

    if (
        not model_name
        or not str(model_name).strip()
    ):
        raise KeyEnumerationHelperError(
            "model_name is required to resolve key-recovery reports"
        )

    return str(model_name).strip()


def _load_json_object(
    path: Path,
) -> Dict[str, Any]:
    """Load one JSON object without trusting stale enumeration evidence."""
    if not path.is_file():
        raise KeyEnumerationHelperError(
            f"Required key-recovery report is missing: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = strip_seal(json.load(handle))

    except (OSError, json.JSONDecodeError) as exc:
        raise KeyEnumerationHelperError(
            f"Failed to read key-recovery report: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise KeyEnumerationHelperError(
            f"Key-recovery report must contain a JSON object: {path}"
        )

    return payload


def _strict_int(
    value: Any,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    """Validate one strict persisted integer."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise KeyEnumerationHelperError(
            f"{field} must be an integer in "
            f"[{minimum}, {maximum}]"
        )

    if not minimum <= value <= maximum:
        raise KeyEnumerationHelperError(
            f"{field} must be in "
            f"[{minimum}, {maximum}], got {value}"
        )

    return value


def _finite_float(
    value: Any,
    *,
    field: str,
) -> float:
    """Validate one finite persisted score."""
    if isinstance(value, bool):
        raise KeyEnumerationHelperError(
            f"{field} must be a finite number"
        )

    try:
        built = float(value)
    except (TypeError, ValueError) as exc:
        raise KeyEnumerationHelperError(
            f"{field} must be a finite number"
        ) from exc

    if not math.isfinite(built):
        raise KeyEnumerationHelperError(
            f"{field} must be finite"
        )

    return built


def _validate_ranked_candidates(
    candidates: Any,
    *,
    field: str,
) -> list[Dict[str, Any]]:
    """
    Validate the complete persisted 256-candidate AI ranking.

    This validation is deliberately independent of enumeration evidence.
    """
    if (
        isinstance(candidates, (str, bytes))
        or not isinstance(candidates, Sequence)
    ):
        raise KeyEnumerationHelperError(
            f"{field} must be a sequence"
        )

    if len(candidates) != KEYSPACE_SIZE:
        raise KeyEnumerationHelperError(
            f"{field} must contain exactly "
            f"{KEYSPACE_SIZE} candidates; "
            f"received {len(candidates)}"
        )

    normalized: list[Dict[str, Any]] = []

    seen_guesses: set[int] = set()
    seen_ranks: set[int] = set()

    for expected_rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        if not isinstance(candidate, Mapping):
            raise KeyEnumerationHelperError(
                f"{field}[{expected_rank}] must be a mapping"
            )

        rank = _strict_int(
            candidate.get("rank"),
            field=f"{field}[{expected_rank}].rank",
            minimum=1,
            maximum=KEYSPACE_SIZE,
        )

        if rank != expected_rank:
            raise KeyEnumerationHelperError(
                f"{field}[{expected_rank}].rank must be "
                f"{expected_rank}, got {rank}"
            )

        if rank in seen_ranks:
            raise KeyEnumerationHelperError(
                f"{field} contains duplicate rank {rank}"
            )

        seen_ranks.add(rank)

        key_guess = _strict_int(
            candidate.get(
                "key_guess",
                candidate.get("key"),
            ),
            field=f"{field}[{expected_rank}].key_guess",
            minimum=0,
            maximum=KEYSPACE_SIZE - 1,
        )

        if key_guess in seen_guesses:
            raise KeyEnumerationHelperError(
                f"{field} contains duplicate key guess "
                f"0x{key_guess:02X}"
            )

        seen_guesses.add(key_guess)

        score = _finite_float(
            candidate.get("score"),
            field=f"{field}[{expected_rank}].score",
        )

        expected_hex = f"0x{key_guess:02X}"
        key_guess_hex = candidate.get(
            "key_guess_hex"
        )

        if (
            key_guess_hex is not None
            and key_guess_hex != expected_hex
        ):
            raise KeyEnumerationHelperError(
                f"{field}[{expected_rank}].key_guess_hex "
                "does not match key_guess"
            )

        normalized.append(
            {
                "rank": rank,
                "key_guess": key_guess,
                "key_guess_hex": expected_hex,
                "score": score,
            }
        )

    if seen_guesses != set(range(KEYSPACE_SIZE)):
        raise KeyEnumerationHelperError(
            f"{field} must contain every AES-byte hypothesis exactly once"
        )

    if seen_ranks != set(range(1, KEYSPACE_SIZE + 1)):
        raise KeyEnumerationHelperError(
            f"{field} must contain every public rank 1..256 exactly once"
        )

    return normalized


def _validate_base_report(
    report: Mapping[str, Any],
    *,
    expected_process: str,
    path: Path,
) -> Dict[str, Any]:
    """
    Validate non-enumeration evidence and return a clean report copy.

    Existing enumeration evidence is intentionally discarded before
    regeneration. Invalid base report evidence is never discarded or hidden.
    """
    if not isinstance(report, Mapping):
        raise KeyEnumerationHelperError(
            f"Key-recovery report must be a mapping: {path}"
        )

    clean = deepcopy(dict(report))

    # Enumeration is derived evidence. Old schemas must not influence the
    # regenerated result.
    clean.pop("enumeration", None)

    process = str(
        clean.get("process", "")
    ).strip().lower()

    if process != expected_process:
        raise KeyEnumerationHelperError(
            f"Expected process={expected_process!r} in {path}, "
            f"got {process!r}"
        )

    key = clean.get("key")

    if not isinstance(key, Mapping):
        raise KeyEnumerationHelperError(
            f"{path}: report.key must be a mapping"
        )

    true_key = _strict_int(
        key.get("true_key_byte"),
        field=f"{path}: report.key.true_key_byte",
        minimum=0,
        maximum=KEYSPACE_SIZE - 1,
    )

    final_rank = _strict_int(
        key.get("final_true_key_rank"),
        field=f"{path}: report.key.final_true_key_rank",
        minimum=1,
        maximum=KEYSPACE_SIZE,
    )

    true_key_hex = key.get("true_key_hex")

    if (
        true_key_hex is not None
        and true_key_hex != f"0x{true_key:02X}"
    ):
        raise KeyEnumerationHelperError(
            f"{path}: report.key.true_key_hex does not "
            "match true_key_byte"
        )

    diagnostics = clean.get("diagnostics")

    if not isinstance(diagnostics, Mapping):
        raise KeyEnumerationHelperError(
            f"{path}: report.diagnostics must be a mapping"
        )

    ranked = _validate_ranked_candidates(
        diagnostics.get("top_key_candidates"),
        field=(
            f"{path}: "
            "report.diagnostics.top_key_candidates"
        ),
    )

    ranked_true_key = next(
        candidate
        for candidate in ranked
        if candidate["key_guess"] == true_key
    )

    if ranked_true_key["rank"] != final_rank:
        raise KeyEnumerationHelperError(
            f"{path}: report.key.final_true_key_rank={final_rank} "
            "does not match the complete AI ranking "
            f"rank={ranked_true_key['rank']}"
        )

    # Replace candidate evidence with the normalized copy so downstream
    # enumeration receives one stable representation.
    clean["diagnostics"] = dict(diagnostics)
    clean["diagnostics"][
        "top_key_candidates"
    ] = ranked

    return clean


def _load_required_base_report(
    *,
    path: Path,
    expected_process: str,
) -> Dict[str, Any]:
    """Load and validate one base validation/test report."""
    raw = _load_json_object(path)

    return _validate_base_report(
        raw,
        expected_process=expected_process,
        path=path,
    )


def _validate_matching_lifecycle(
    validation_report: Mapping[str, Any],
    test_report: Mapping[str, Any],
) -> None:
    """Require validation and test artifacts from the same model lifecycle."""
    validation_model = (
        validation_report.get("model")
        or {}
    )
    test_model = (
        test_report.get("model")
        or {}
    )

    if not isinstance(
        validation_model,
        Mapping,
    ) or not isinstance(
        test_model,
        Mapping,
    ):
        raise KeyEnumerationHelperError(
            "report.model must be a mapping in validation and test reports"
        )

    for field in (
        "name",
        "hash",
    ):
        validation_value = (
            validation_model.get(field)
        )
        test_value = (
            test_model.get(field)
        )

        if validation_value != test_value:
            raise KeyEnumerationHelperError(
                "Validation and test reports belong to "
                "different model lifecycles: "
                f"model.{field} differs "
                f"({validation_value!r} != {test_value!r})"
            )

    validation_target = (
        validation_report.get("target")
        or {}
    )
    test_target = (
        test_report.get("target")
        or {}
    )

    if not isinstance(
        validation_target,
        Mapping,
    ) or not isinstance(
        test_target,
        Mapping,
    ):
        raise KeyEnumerationHelperError(
            "report.target must be a mapping in validation and test reports"
        )

    for field in (
        "cipher",
        "attack_byte",
        "key_mode",
        "leakage_model",
        "keyspace_size",
    ):
        if (
            validation_target.get(field)
            != test_target.get(field)
        ):
            raise KeyEnumerationHelperError(
                "Validation and test reports target "
                "different attack contracts: "
                f"target.{field} differs"
            )


def _extract_enumeration_inputs(
    report: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    list[Dict[str, Any]],
]:
    """Extract validated true-key and complete AI-ranking evidence."""
    key = report.get("key")

    if not isinstance(key, Mapping):
        raise KeyEnumerationHelperError(
            "report.key must be a mapping"
        )

    diagnostics = report.get("diagnostics")

    if not isinstance(
        diagnostics,
        Mapping,
    ):
        raise KeyEnumerationHelperError(
            "report.diagnostics must be a mapping"
        )

    if "true_key_byte" not in key:
        raise KeyEnumerationHelperError(
            "report.key.true_key_byte is required"
        )

    if "final_true_key_rank" not in key:
        raise KeyEnumerationHelperError(
            "report.key.final_true_key_rank is required"
        )

    ranked = _validate_ranked_candidates(
        diagnostics.get("top_key_candidates"),
        field="report.diagnostics.top_key_candidates",
    )

    return key, ranked


def _policy_fingerprint(
    policy: Mapping[str, Any],
) -> str:
    """Return the SHA-256 fingerprint of one frozen enumeration policy."""
    canonical = json.dumps(
        dict(policy),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(
        canonical
    ).hexdigest()

def _comparison_profile_contract(
    *,
    comparison_contract: Mapping[str, Any],
    mode: str,
) -> Dict[str, Any]:
    """
    Materialize one ordinary algorithm contract from comparison mode.

    Comparison profiles have already been validated by
    validate_key_enumeration_contract(). This helper converts one profile
    into the same concrete contract shape consumed by the existing policy
    freezing machinery.

    No ranking, true-key, validation, or test evidence is consulted here.
    """
    profiles = comparison_contract.get(
        "comparison_profiles"
    )

    if not isinstance(
        profiles,
        Mapping,
    ):
        raise KeyEnumerationHelperError(
            "Validated comparison contract is missing "
            "comparison_profiles"
        )

    profile = profiles.get(
        mode
    )

    if not isinstance(
        profile,
        Mapping,
    ):
        raise KeyEnumerationHelperError(
            "Validated comparison contract is missing "
            f"profile for mode={mode!r}"
        )

    if mode not in {
        "local_hamming",
        "neighbour_sweep",
        "support_weighted",
    }:
        raise KeyEnumerationHelperError(
            f"Unsupported comparison algorithm: {mode!r}"
        )

    return {
        "enabled": True,
        "mode": mode,
        "top_k": profile.get(
            "top_k"
        ),
        "hamming_radius": profile.get(
            "hamming_radius"
        ),
        "local_budget": profile.get(
            "local_budget"
        ),
        "stop_on_success": (
            comparison_contract[
                "stop_on_success"
            ]
        ),
    }
def _cfg_for_enumeration_mode(
    *,
    cfg: Mapping[str, Any],
    mode: str,
    profile: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Return a concrete cfg for exactly one enumeration algorithm.

    This is used internally by comparison mode so the ordinary enumeration
    engine receives local_hamming/neighbour_sweep/support_weighted rather than the meta-mode
    ``comparison``.

    The caller's configuration is never mutated.
    """
    if mode not in {
        "local_hamming",
        "neighbour_sweep",
        "support_weighted",
    }:
        raise KeyEnumerationHelperError(
            f"Unsupported enumeration algorithm: {mode!r}"
        )

    concrete_cfg = deepcopy(
        dict(cfg)
    )

    sca = concrete_cfg.get(
        "sca"
    )

    if not isinstance(
        sca,
        dict,
    ):
        raise KeyEnumerationHelperError(
            "cfg.sca must be a mapping"
        )

    key_enumeration = sca.get(
        "key_enumeration"
    )

    if not isinstance(
        key_enumeration,
        dict,
    ):
        raise KeyEnumerationHelperError(
            "cfg.sca.key_enumeration must be a mapping"
        )

    key_enumeration["mode"] = mode
    key_enumeration["modes"] = {
        mode: {
            key: value
            for key, value in profile.items()
            if value is not None
        }
    }

    return concrete_cfg


def _freeze_configuration_policy(
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Freeze one enumeration algorithm directly from validated configuration.

    No true-key evidence, candidate ranking, validation result, or test result
    participates in policy construction.

    LOCAL_HAMMING, NEIGHBOUR_SWEEP, and SUPPORT_WEIGHTED therefore execute exactly the concrete
    parameters supplied by the runtime contract.
    """
    mode = contract.get("mode")

    if mode == "local_hamming":
        for field in (
            "top_k",
            "hamming_radius",
            "local_budget",
        ):
            if contract.get(field) is None:
                raise KeyEnumerationHelperError(
                    f"Validated local_hamming contract is missing {field}"
                )

        return {
            "mode": "local_hamming",
            "policy_source": "configuration",
            "top_k": contract["top_k"],
            "hamming_radius": contract["hamming_radius"],
            "local_budget": contract["local_budget"],
            "fallback_order": "ai",
            "stop_on_success": contract["stop_on_success"],
        }

    if mode == "neighbour_sweep":
        hamming_radius = contract.get(
            "hamming_radius"
        )

        if hamming_radius is None:
            raise KeyEnumerationHelperError(
                "Validated neighbour_sweep contract is missing hamming_radius"
            )

        return {
            "mode": "neighbour_sweep",
            "policy_source": "configuration",
            "hamming_radius": hamming_radius,
            "stop_on_success": contract["stop_on_success"],
        }

    if mode == "support_weighted":
        for field in (
            "top_k",
            "hamming_radius",
            "local_budget",
        ):
            if contract.get(field) is None:
                raise KeyEnumerationHelperError(
                    f"Validated support_weighted contract is missing {field}"
                )

        return {
            "mode": "support_weighted",
            "policy_source": "configuration",
            "top_k": contract["top_k"],
            "hamming_radius": contract["hamming_radius"],
            "local_budget": contract["local_budget"],
            "fallback_order": "ai",
            "stop_on_success": contract["stop_on_success"],
        }

    raise KeyEnumerationHelperError(
        "Configuration policy freezing is valid only for "
        "local_hamming, neighbour_sweep, or support_weighted mode"
    )
def _build_frozen_policy(
    *,
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build one immutable enumeration policy directly from configuration.

    Validation and test evidence never participate in policy construction.
    """
    mode = contract.get("mode")

    if mode not in {
        "local_hamming",
        "neighbour_sweep",
        "support_weighted",
    }:
        raise KeyEnumerationHelperError(
            f"Unsupported enumeration mode: {mode!r}"
        )

    policy = _freeze_configuration_policy(
        contract
    )

    frozen = dict(policy)

    frozen["policy_fingerprint"] = (
        _policy_fingerprint(
            policy
        )
    )

    return frozen

def _build_ai_comparison_result(
    report: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build the raw-AI baseline used only by comparison-mode selection.

    This does not execute enumeration. The AI enumeration-equivalent position
    is its original true-key rank.
    """
    key, _ = _extract_enumeration_inputs(
        report
    )

    ai_rank = _strict_int(
        key.get("final_true_key_rank"),
        field="report.key.final_true_key_rank",
        minimum=1,
        maximum=KEYSPACE_SIZE,
    )

    return {
        "enabled": False,
        "mode": "ai",
        "executed": False,
        "ai_rank_at_success": ai_rank,
        "enumeration_position": ai_rank,
        "candidates_evaluated": ai_rank,
        "position_delta": 0,
        "success_phase": "ai",
    }


def _run_validation_comparison(
    *,
    cfg: Mapping[str, Any],
    contract: Mapping[str, Any],
    validation_report: Mapping[str, Any],
) -> tuple[
    Dict[str, Any],
    Optional[Dict[str, Any]],
]:
    """
    Compare raw AI and all enumeration algorithms on validation evidence.

    LOCAL_HAMMING, NEIGHBOUR_SWEEP and SUPPORT_WEIGHTED execute their
    configured algorithms exactly.

    AI participates in final selection as the baseline: an enumeration
    algorithm is accepted only when it produces a strictly better validation
    position than raw AI, so ties stay with the model.

    Test evidence never participates in selection.
    """
    process = str(
        validation_report.get(
            "process",
            "",
        )
    ).strip().lower()

    if process != "validation":
        raise KeyEnumerationHelperError(
            "Enumeration comparison may be selected "
            "only from validation evidence"
        )

    validation_results: Dict[
        str,
        Dict[str, Any],
    ] = {
        "ai": _build_ai_comparison_result(
            validation_report
        )
    }

    frozen_policies: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for mode in (
        "local_hamming",
        "neighbour_sweep",
        "support_weighted",
    ):
        ordinary_contract = (
            _comparison_profile_contract(
                comparison_contract=contract,
                mode=mode,
            )
        )

        frozen_policy = (
            _build_frozen_policy(
                contract=ordinary_contract,
            )
        )

        profile = {
            key: ordinary_contract.get(key)
            for key in (
                "top_k",
                "hamming_radius",
                "local_budget",
            )
            if ordinary_contract.get(key) is not None
        }

        concrete_cfg = (
            _cfg_for_enumeration_mode(
                cfg=cfg,
                mode=mode,
                profile=profile,
            )
        )

        enriched = _enrich_report(
            cfg=concrete_cfg,
            report=validation_report,
            frozen_policy=frozen_policy,
        )

        enumeration = enriched.get(
            "enumeration"
        )

        if not isinstance(
            enumeration,
            Mapping,
        ):
            raise KeyEnumerationHelperError(
                f"{mode} validation comparison "
                "did not produce enumeration evidence"
            )

        validation_results[mode] = deepcopy(
            dict(enumeration)
        )

        frozen_policies[mode] = deepcopy(
            frozen_policy
        )

    mode_priority = {
        "local_hamming": 0,
        "neighbour_sweep": 1,
        "support_weighted": 2,
    }

    best_enumeration_mode = min(
        (
            "local_hamming",
            "neighbour_sweep",
            "support_weighted",
        ),
        key=lambda candidate_mode: (
            validation_results[
                candidate_mode
            ]["enumeration_position"],
            validation_results[
                candidate_mode
            ]["candidates_evaluated"],
            validation_results[
                candidate_mode
            ]["position_delta"],
            mode_priority[
                candidate_mode
            ],
        ),
    )

    ai_position = validation_results[
        "ai"
    ]["enumeration_position"]

    best_enumeration_position = (
        validation_results[
            best_enumeration_mode
        ]["enumeration_position"]
    )

    # The model wins ties: enumeration must strictly improve to be selected.
    if (
        ai_position
        <= best_enumeration_position
    ):
        selected_mode = "ai"
        winning_policy = None

    else:
        selected_mode = (
            best_enumeration_mode
        )

        winning_policy = deepcopy(
            frozen_policies[
                selected_mode
            ]
        )

    comparison_evidence: Dict[str, Any] = {
        "enabled": True,
        "mode": "comparison",
        "executed": True,
        "success": True,
        "selection_split": "validation",
        "selection_basis": (
            "model_guarded_minimum"
        ),
        "selected_mode": selected_mode,
        "results": validation_results,
    }

    if winning_policy is None:
        comparison_evidence[
            "selected_policy"
        ] = None

        comparison_evidence[
            "selected_policy_fingerprint"
        ] = None

    else:
        comparison_evidence[
            "selected_policy"
        ] = deepcopy(
            winning_policy
        )

        comparison_evidence[
            "selected_policy_fingerprint"
        ] = winning_policy[
            "policy_fingerprint"
        ]

    return (
        comparison_evidence,
        winning_policy,
    )

MODEL_SOURCE = "MODEL"
ENUMERATION_SOURCE = "MODEL+KEY_ENUMERATION"


def _apply_reported_rank(
    enumeration: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Record the headline rank of one split next to both raw ranks.

    The model rank and the enumeration position are both kept as measured.
    The headline is the better of the two and the model wins ties, so the
    reported rank can never be worse than what the model achieved on its own,
    and a reader can always see which one produced it.
    """
    ai_rank = _strict_int(
        enumeration.get("ai_rank_at_success"),
        field="enumeration.ai_rank_at_success",
        minimum=1,
        maximum=KEYSPACE_SIZE,
    )

    position = enumeration.get(
        "enumeration_position"
    )

    if position is None:
        reported, source = ai_rank, MODEL_SOURCE

    else:
        position = _strict_int(
            position,
            field="enumeration.enumeration_position",
            minimum=1,
            maximum=KEYSPACE_SIZE,
        )

        if ai_rank <= position:
            reported, source = ai_rank, MODEL_SOURCE
        else:
            reported, source = position, ENUMERATION_SOURCE

    enumeration["key_rank_reported"] = reported
    enumeration["key_rank_source"] = source
    enumeration["key_recovery"] = (
        "DONE"
        if reported == PUBLIC_RANK_BASE
        else "FAILED"
    )

    return enumeration


def _build_ai_selected_test_report(
    report: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Preserve raw test AI evidence when validation comparison selected AI.

    No key-enumeration algorithm is executed on test.
    """
    key, _ = _extract_enumeration_inputs(
        report
    )

    ai_rank = _strict_int(
        key.get("final_true_key_rank"),
        field="report.key.final_true_key_rank",
        minimum=1,
        maximum=KEYSPACE_SIZE,
    )

    enriched = deepcopy(
        dict(report)
    )

    enriched.pop(
        "enumeration",
        None,
    )

    enriched["enumeration"] = _apply_reported_rank({
        "enabled": True,
        "mode": "comparison",
        "executed": False,
        "selected_by_comparison": True,
        "selection_split": "validation",
        "selected_mode": "ai",
        "ai_rank_at_success": ai_rank,
        "enumeration_position": ai_rank,
        "position_delta": 0,
    })

    return enriched

def _build_comparison_reports(
    *,
    cfg: Mapping[str, Any],
    contract: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    test_report: Mapping[str, Any],
) -> tuple[
    Dict[str, Any],
    Dict[str, Any],
    Optional[Dict[str, Any]],
]:
    """
    Select an enumeration strategy on validation and evaluate only that
    frozen strategy on test evidence.

    Test evidence never participates in strategy selection.
    """
    (
        comparison_evidence,
        winning_policy,
    ) = _run_validation_comparison(
        cfg=cfg,
        contract=contract,
        validation_report=validation_report,
    )

    selected_mode = (
        comparison_evidence[
            "selected_mode"
        ]
    )
    if selected_mode == "ai":
        if winning_policy is not None:
            raise KeyEnumerationHelperError(
                "AI comparison selection must not have "
                "an enumeration policy"
            )

        enriched_test = (
            _build_ai_selected_test_report(
                test_report
            )
        )

    else:
        if winning_policy is None:
            raise KeyEnumerationHelperError(
                "Enumeration comparison winner is missing "
                "its frozen policy"
            )

        profiles = contract.get(
            "comparison_profiles"
        )

        if not isinstance(
            profiles,
            Mapping,
        ):
            raise KeyEnumerationHelperError(
                "Comparison contract is missing comparison_profiles"
            )

        selected_profile = profiles.get(
            selected_mode
        )

        if not isinstance(
            selected_profile,
            Mapping,
        ):
            raise KeyEnumerationHelperError(
                "Comparison winner has no validated profile"
            )

        concrete_cfg = (
            _cfg_for_enumeration_mode(
                cfg=cfg,
                mode=selected_mode,
                profile=selected_profile,
            )
        )

        enriched_test = _enrich_report(
            cfg=concrete_cfg,
            report=test_report,
            frozen_policy=winning_policy,
        )

        test_enumeration = (
            enriched_test[
                "enumeration"
            ]
        )

        if (
            test_enumeration.get("mode")
            != selected_mode
        ):
            raise KeyEnumerationHelperError(
                "Test enumeration did not use the "
                "validation-selected mode"
            )

        if (
            test_enumeration.get(
                "policy_fingerprint"
            )
            != winning_policy[
                "policy_fingerprint"
            ]
        ):
            raise KeyEnumerationHelperError(
                "Test enumeration did not use the "
                "validation-frozen winning policy"
            )

        test_enumeration[
            "selected_by_comparison"
        ] = True

        test_enumeration[
            "selection_split"
        ] = "validation"

        test_enumeration[
            "selected_mode"
        ] = selected_mode

    enriched_validation = deepcopy(
        dict(validation_report)
    )

    enriched_validation.pop(
        "enumeration",
        None,
    )

    selected_results = (
        comparison_evidence["results"][
            comparison_evidence["selected_mode"]
        ]
    )

    comparison_evidence["ai_rank_at_success"] = (
        comparison_evidence["results"]["ai"]["enumeration_position"]
    )

    comparison_evidence["enumeration_position"] = (
        selected_results["enumeration_position"]
    )

    enriched_validation[
        "enumeration"
    ] = _apply_reported_rank(
        comparison_evidence
    )

    return (
        enriched_validation,
        enriched_test,
        winning_policy,
    )

def _verify_enumeration_policy(
    *,
    enumeration: Mapping[str, Any],
    frozen_policy: Mapping[str, Any],
) -> None:
    """Require one split result to reflect the frozen policy exactly."""
    mode = frozen_policy.get("mode")

    if enumeration.get("mode") != mode:
        raise KeyEnumerationHelperError(
            "Enumeration result mode differs from frozen policy"
        )

    if (
        enumeration.get("stop_on_success")
        != frozen_policy.get(
            "stop_on_success"
        )
    ):
        raise KeyEnumerationHelperError(
            "Enumeration result changed stop_on_success"
        )

    if mode == "local_hamming":
        fields = (
            "mode",
            "policy_source",
            "top_k",
            "hamming_radius",
            "local_budget",
            "fallback_order",
            "stop_on_success",
        )

    elif mode == "neighbour_sweep":
        fields = (
            "mode",
            "policy_source",
            "hamming_radius",
            "stop_on_success",
        )

    elif mode == "support_weighted":
        fields = (
            "mode",
            "policy_source",
            "top_k",
            "hamming_radius",
            "local_budget",
            "fallback_order",
            "stop_on_success",
        )

    else:
        raise KeyEnumerationHelperError(
            f"Unsupported enumeration mode: {mode!r}"
        )

    mismatches = {
        field: {
            "expected": frozen_policy.get(
                field
            ),
            "observed": enumeration.get(
                field
            ),
        }
        for field in fields
        if (
            enumeration.get(field)
            != frozen_policy.get(field)
        )
    }

    if mismatches:
        raise KeyEnumerationHelperError(
            "Enumeration did not apply the frozen policy: "
            f"{mismatches}"
        )


def _enrich_report(
    *,
    cfg: Mapping[str, Any],
    report: Mapping[str, Any],
    frozen_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Enumerate one split using its own complete AI ranking.

    Test ranking/scores therefore remain completely independent from
    validation candidate values and validation candidate scores.
    """
    key, ranked_candidates = (
        _extract_enumeration_inputs(
            report
        )
    )

    mode = frozen_policy.get("mode")

    enumeration = run_key_enumeration(
        cfg=cfg,
        true_key_byte=key["true_key_byte"],
        ranked_candidates=ranked_candidates,
    )


    if not enumeration.get(
        "executed"
    ):
        raise KeyEnumerationHelperError(
            f"{mode} enumeration did not execute"
        )

    if not enumeration.get(
        "success"
    ):
        raise KeyEnumerationHelperError(
            f"{mode} enumeration failed to reach the true key "
            "despite receiving the complete AES-byte keyspace"
        )

    report_rank = _strict_int(
        key.get("final_true_key_rank"),
        field="report.key.final_true_key_rank",
        minimum=1,
        maximum=KEYSPACE_SIZE,
    )

    enumeration_rank = (
        enumeration.get(
            "ai_rank_at_success"
        )
    )

    if enumeration_rank != report_rank:
        raise KeyEnumerationHelperError(
            "Inconsistent key-recovery report: "
            f"enumeration AI rank={enumeration_rank!r}, "
            f"report.key.final_true_key_rank={report_rank}"
        )

    _verify_enumeration_policy(
        enumeration=enumeration,
        frozen_policy=frozen_policy,
    )

    evaluated = enumeration.get(
        "evaluated_candidates"
    )

    if not isinstance(evaluated, list):
        raise KeyEnumerationHelperError(
            "Enumeration result must contain evaluated_candidates"
        )

    guesses = [
        row.get("key_guess")
        for row in evaluated
        if isinstance(row, Mapping)
    ]

    if len(guesses) != len(set(guesses)):
        raise KeyEnumerationHelperError(
            "Enumeration evaluated a candidate more than once"
        )

    stop_on_success = (
        frozen_policy[
            "stop_on_success"
        ]
    )

    if not stop_on_success:
        if (
            enumeration.get(
                "candidates_evaluated"
            )
            != KEYSPACE_SIZE
        ):
            raise KeyEnumerationHelperError(
                f"{mode} must evaluate exactly 256 candidates "
                "when stop_on_success=false"
            )

        if len(evaluated) != KEYSPACE_SIZE:
            raise KeyEnumerationHelperError(
                f"{mode} evaluated_candidates must contain "
                "exactly 256 rows when stop_on_success=false"
            )

        if set(guesses) != set(
            range(KEYSPACE_SIZE)
        ):
            raise KeyEnumerationHelperError(
                f"{mode} must evaluate every AES byte exactly once "
                "when stop_on_success=false"
            )

    enumeration[
        "policy_frozen"
    ] = True

    enumeration[
        "policy_fingerprint"
    ] = frozen_policy[
        "policy_fingerprint"
    ]

    enriched = deepcopy(
        dict(report)
    )

    # ``report`` has already had stale enumeration removed, but make the
    # replacement explicit.
    enriched.pop(
        "enumeration",
        None,
    )

    enriched[
        "enumeration"
    ] = _apply_reported_rank(
        enumeration
    )

    return enriched


def _validate_enriched_report(
    report: Mapping[str, Any],
    *,
    expected_mode: str,
    expected_fingerprint: Optional[str],
) -> None:
    """
    Strictly validate one regenerated enumeration section.

    key_recovery.py independently reconstructs candidate ordering and,
    for comparison, independently reconstructs the validation winner.
    """
    validated = validate_key_enumeration_report_section(
        report
    )

    if validated is None:
        raise KeyEnumerationHelperError(
            "Regenerated report is missing enumeration evidence"
        )

    observed_mode = str(
        validated.get(
            "mode",
            "",
        )
    ).strip().lower()

    if observed_mode != expected_mode:
        raise KeyEnumerationHelperError(
            "Validated enumeration mode differs from expected mode: "
            f"expected={expected_mode!r}, observed={observed_mode!r}"
        )

    if expected_mode == "comparison":
        observed_fingerprint = validated.get(
            "selected_policy_fingerprint"
        )
    else:
        observed_fingerprint = validated.get(
            "policy_fingerprint"
        )

    if observed_fingerprint != expected_fingerprint:
        raise KeyEnumerationHelperError(
            "Validated enumeration policy fingerprint differs "
            "from the frozen policy: "
            f"expected={expected_fingerprint!r}, "
            f"observed={observed_fingerprint!r}"
        )

def _stage_json(
    path: Path,
    payload: Mapping[str, Any],
) -> Path:
    """
    Serialize and fsync one staged JSON replacement.

    Filesystem ownership and permission policy is intentionally delegated to
    the operating/deployment environment. Application code does not chmod,
    chown, or inherit file metadata.
    """
    try:
        serialized = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise KeyEnumerationHelperError(
            f"Failed to serialize enriched report for {path}"
        ) from exc

    # Plain exclusive creation: mkstemp() would force mode 0600 onto the
    # published report through os.replace().
    staged_path = path.parent / f".{path.name}.{os.urandom(8).hex()}.staged"

    try:
        with staged_path.open(
            "x",
            encoding="utf-8",
        ) as handle:
            handle.write(
                serialized
            )
            handle.write(
                "\n"
            )
            handle.flush()

            os.fsync(
                handle.fileno()
            )

    except Exception:
        try:
            staged_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        raise

    return staged_path


def _make_rollback_copy(
    path: Path,
) -> Path:
    """
    Create a same-directory rollback reference without changing filesystem
    metadata.

    A hard link preserves the existing inode naturally. The application does
    not copy, chmod, chown, or otherwise define file metadata policy.
    """
    fd, backup_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".rollback",
        dir=str(
            path.parent
        ),
    )

    os.close(
        fd
    )

    backup_path = Path(
        backup_name
    )

    # os.link requires the destination not to exist.
    backup_path.unlink()

    try:
        os.link(
            path,
            backup_path,
        )

    except Exception:
        backup_path.unlink(
            missing_ok=True
        )
        raise

    return backup_path
def _fsync_directory(
    directory: Path,
) -> None:
    """Best-effort fsync of directory metadata after atomic replacements."""
    flags = (
        os.O_RDONLY
        | getattr(
            os,
            "O_DIRECTORY",
            0,
        )
    )

    try:
        fd = os.open(
            directory,
            flags,
        )
    except OSError:
        return

    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _replace_report_pair(
    *,
    validation_path: Path,
    validation_staged: Path,
    test_path: Path,
    test_staged: Path,
) -> None:
    """
    Replace validation/test reports as one guarded operation.

    Both staged artifacts already exist and have already been validated before
    entering this function. If either replacement raises, rollback copies are
    restored before the error is propagated.
    """
    validation_backup: Optional[
        Path
    ] = None

    test_backup: Optional[
        Path
    ] = None

    validation_replaced = False
    test_replaced = False

    try:
        validation_backup = (
            _make_rollback_copy(
                validation_path
            )
        )

        test_backup = (
            _make_rollback_copy(
                test_path
            )
        )

        os.replace(
            validation_staged,
            validation_path,
        )
        validation_replaced = True

        os.replace(
            test_staged,
            test_path,
        )
        test_replaced = True

        _fsync_directory(
            validation_path.parent
        )

        if (
            test_path.parent
            != validation_path.parent
        ):
            _fsync_directory(
                test_path.parent
            )

    except Exception as exc:
        rollback_errors: list[
            str
        ] = []

        if (
            validation_replaced
            and validation_backup is not None
            and validation_backup.exists()
        ):
            try:
                os.replace(
                    validation_backup,
                    validation_path,
                )
                validation_backup = None
            except Exception as rollback_exc:
                rollback_errors.append(
                    "validation rollback failed: "
                    f"{rollback_exc}"
                )

        if (
            test_replaced
            and test_backup is not None
            and test_backup.exists()
        ):
            try:
                os.replace(
                    test_backup,
                    test_path,
                )
                test_backup = None
            except Exception as rollback_exc:
                rollback_errors.append(
                    "test rollback failed: "
                    f"{rollback_exc}"
                )

        details = (
            "; ".join(
                rollback_errors
            )
            if rollback_errors
            else "rollback completed"
        )

        raise KeyEnumerationHelperError(
            "Failed to replace the validation/test "
            f"enumeration report pair; {details}"
        ) from exc

    finally:
        for temporary in (
            validation_staged,
            test_staged,
            validation_backup,
            test_backup,
        ):
            if (
                temporary is not None
                and temporary.exists()
            ):
                try:
                    temporary.unlink()
                except OSError:
                    pass

def _run_completed_cycle_key_enumeration(
    *,
    cfg: Mapping[str, Any],
    logger: Any,
    test_only: bool = False,
) -> bool:
    """
    Enrich one completed validation/test lifecycle.

    test_only=True (make test): the strategy is still selected on the
    stored validation evidence, exactly as at training, but only the fresh
    test report is enriched and written; validation_best.json is untouched.

    Ordinary modes
    --------------
    local_hamming:
        Freeze the configured local_hamming policy and apply that exact algorithm
        independently to validation and test rankings.

    neighbour_sweep:
        Freeze the configured neighbour_sweep policy, then apply it independently
        to validation and test rankings.

    support_weighted:
        Freeze the configured support_weighted policy, while each split recomputes
        weighted support from its own AI ranking/scores.

    comparison:
        Benchmark local_hamming, neighbour_sweep, and support_weighted on validation evidence only.
        Select the algorithm with the smallest validation enumeration position,
        freeze the exact winning mode/policy, and execute only that frozen
        winner on test evidence.

        Test evidence never participates in strategy selection.

    Enumeration/reporting failures remain non-fatal to an already-successful
    training/testing lifecycle and are represented by ``False``.
    """
    try:
        # -------------------------------------------------------------
        # 1. Validate configured key-enumeration runtime contract
        # -------------------------------------------------------------
        contract = (
            validate_key_enumeration_contract(
                cfg
            )
        )

        if not contract[
            "enabled"
        ]:
            logger.info(
                "Key enumeration skipped: "
                "sca.key_enumeration.enabled=false"
            )
            return True

        mode = contract[
            "mode"
        ]

        if mode not in {
            "local_hamming",
            "neighbour_sweep",
            "support_weighted",
            "comparison",
        }:
            raise KeyEnumerationHelperError(
                f"Unsupported key-enumeration mode: {mode!r}"
            )

        # -------------------------------------------------------------
        # 2. Resolve completed model artifacts
        # -------------------------------------------------------------
        model_name = (
            _resolve_model_name(
                cfg
            )
        )

        paths = key_recovery_paths(
            cfg,
            model_name=model_name,
        )

        # -------------------------------------------------------------
        # 3. Load clean base validation/test reports
        #
        # Existing enumeration evidence is discarded by
        # _load_required_base_report().
        # -------------------------------------------------------------
        validation_report = (
            _load_required_base_report(
                path=paths[
                    "validation_best"
                ],
                expected_process="validation",
            )
        )

        test_report = (
            _load_required_base_report(
                path=paths[
                    "test"
                ],
                expected_process="test",
            )
        )

        # -------------------------------------------------------------
        # 4. Ensure validation/test belong to the same lifecycle
        # -------------------------------------------------------------
        _validate_matching_lifecycle(
            validation_report,
            test_report,
        )

        # =============================================================
        # 5A. COMPARISON META-MODE
        # =============================================================
        if mode == "comparison":
            #
            # VALIDATION:
            #
            #   local_hamming
            #   neighbour_sweep
            #   support_weighted
            #
            # are benchmarked against the same validation AI ranking.
            #
            # The best validation enumeration position wins.
            #
            # TEST:
            #
            # only the validation-selected mode and frozen policy execute.
            #
            (
                enriched_validation,
                enriched_test,
                frozen_policy,
            ) = _build_comparison_reports(
                cfg=cfg,
                contract=contract,
                validation_report=validation_report,
                test_report=test_report,
            )

            validation_enumeration = (
                enriched_validation[
                    "enumeration"
                ]
            )

            test_enumeration = (
                enriched_test[
                    "enumeration"
                ]
            )

            if not isinstance(
                validation_enumeration,
                Mapping,
            ):
                raise KeyEnumerationHelperError(
                    "Comparison validation report is missing "
                    "enumeration evidence"
                )

            if not isinstance(
                test_enumeration,
                Mapping,
            ):
                raise KeyEnumerationHelperError(
                    "Comparison test report is missing "
                    "enumeration evidence"
                )

            selected_mode = (
                validation_enumeration.get(
                    "selected_mode"
                )
            )

            if selected_mode not in {
                "ai",
                "local_hamming",
                "neighbour_sweep",
                "support_weighted",
            }:
                raise KeyEnumerationHelperError(
                    "Comparison did not select a valid "
                    f"enumeration algorithm: {selected_mode!r}"
                )

            if selected_mode == "ai":
                if frozen_policy is not None:
                    raise KeyEnumerationHelperError(
                        "AI comparison winner unexpectedly has "
                        "a frozen enumeration policy"
                    )

                fingerprint = None

            else:
                if not isinstance(
                    frozen_policy,
                    Mapping,
                ):
                    raise KeyEnumerationHelperError(
                        "Enumeration comparison winner is missing "
                        "its frozen policy"
                    )

                fingerprint = frozen_policy.get(
                    "policy_fingerprint"
                )

                if (
                    not isinstance(
                        fingerprint,
                        str,
                    )
                    or not fingerprint
                ):
                    raise KeyEnumerationHelperError(
                        "Comparison winner has no frozen "
                        "policy fingerprint"
                    )


            validation_fingerprint = (
                validation_enumeration.get(
                    "selected_policy_fingerprint"
                )
            )

            test_fingerprint = (
                test_enumeration.get(
                    "policy_fingerprint"
                )
            )

            if (
                validation_fingerprint
                != fingerprint
            ):
                raise KeyEnumerationHelperError(
                    "Validation comparison published a policy "
                    "fingerprint different from the frozen winner"
                )

            if (
                test_fingerprint
                != fingerprint
            ):
                raise KeyEnumerationHelperError(
                    "Test enumeration did not use the "
                    "validation-frozen winning policy"
                )

            expected_test_mode = (
                "comparison"
                if selected_mode == "ai"
                else selected_mode
            )

            if (
                test_enumeration.get(
                    "mode"
                )
                != expected_test_mode
            ):
                raise KeyEnumerationHelperError(
                    "Test enumeration mode differs from the "
                    "validation-selected comparison execution contract: "
                    f"selected_mode={selected_mode!r}, "
                    f"expected_test_mode={expected_test_mode!r}, "
                    f"observed={test_enumeration.get('mode')!r}"
                )

            if (
                test_enumeration.get(
                    "selected_by_comparison"
                )
                is not True
            ):
                raise KeyEnumerationHelperError(
                    "Test enumeration is missing "
                    "selected_by_comparison=true"
                )

            if (
                test_enumeration.get(
                    "selection_split"
                )
                != "validation"
            ):
                raise KeyEnumerationHelperError(
                    "Comparison test result must declare "
                    "selection_split='validation'"
                )

            # Validate the complete validation comparison envelope.
            #
            # key_recovery.py independently validates all three contained
            # algorithms and independently recomputes the comparison winner.
            _validate_enriched_report(
                enriched_validation,
                expected_mode="comparison",
                expected_fingerprint=fingerprint,
            )

            # Validate test independently using only the policy selected and
            # frozen from validation. Test must never perform comparison again.
            _validate_enriched_report(
                enriched_test,
                expected_mode=(
                    "comparison"
                    if selected_mode == "ai"
                    else selected_mode
                ),
                expected_fingerprint=fingerprint,
            )

        # =============================================================
        # 5B. ORDINARY SINGLE-MODE ENUMERATION
        # =============================================================
        else:
            #
            # Construct exactly one frozen policy before either split runs.
            #
            # Every algorithm mode freezes its configuration directly.
            # No ordinary algorithm performs validation-time policy selection.
            #
            frozen_policy = (
                _build_frozen_policy(
                    contract=contract,
                )
            )

            fingerprint = (
                frozen_policy[
                    "policy_fingerprint"
                ]
            )

            selected_mode = mode

            #
            # Each split uses its own complete AI ranking/scores.
            #
            enriched_validation = (
                _enrich_report(
                    cfg=cfg,
                    report=(
                        validation_report
                    ),
                    frozen_policy=(
                        frozen_policy
                    ),
                )
            )

            enriched_test = (
                _enrich_report(
                    cfg=cfg,
                    report=test_report,
                    frozen_policy=(
                        frozen_policy
                    ),
                )
            )

            validation_enumeration = (
                enriched_validation[
                    "enumeration"
                ]
            )

            test_enumeration = (
                enriched_test[
                    "enumeration"
                ]
            )

            # ---------------------------------------------------------
            # Require exact frozen-policy identity across splits
            # ---------------------------------------------------------
            validation_fingerprint = (
                validation_enumeration.get(
                    "policy_fingerprint"
                )
            )

            test_fingerprint = (
                test_enumeration.get(
                    "policy_fingerprint"
                )
            )

            if (
                validation_fingerprint
                != test_fingerprint
            ):
                raise KeyEnumerationHelperError(
                    "Validation and test policy fingerprints differ"
                )

            if (
                validation_fingerprint
                != fingerprint
            ):
                raise KeyEnumerationHelperError(
                    "Published enumeration policy differs "
                    "from the frozen policy"
                )

            # ---------------------------------------------------------
            # Independently reconstruct/validate both ordinary reports
            # before either artifact is touched.
            # ---------------------------------------------------------
            _validate_enriched_report(
                enriched_validation,
                expected_mode=mode,
                expected_fingerprint=(
                    fingerprint
                ),
            )

            _validate_enriched_report(
                enriched_test,
                expected_mode=mode,
                expected_fingerprint=(
                    fingerprint
                ),
            )

        if test_only:
            test_staged = _stage_json(paths["test"], enriched_test)
            os.replace(test_staged, paths["test"])
            test_enum = enriched_test.get("enumeration") or {}
            logger.info(
                "Test key enumeration completed: model=%s mode=%s "
                "selected_mode=%s test_position=%s",
                model_name,
                mode,
                str(test_enum.get("selected_mode") or test_enum.get("mode") or mode).upper(),
                test_enum.get("enumeration_position"),
            )
            return True

        # =============================================================
        # 6. Stage BOTH artifacts before replacing either live report
        # =============================================================
        validation_staged = (
            _stage_json(
                paths[
                    "validation_best"
                ],
                enriched_validation,
            )
        )

        try:
            test_staged = (
                _stage_json(
                    paths[
                        "test"
                    ],
                    enriched_test,
                )
            )

        except Exception:
            validation_staged.unlink(
                missing_ok=True
            )
            raise

        # =============================================================
        # 7. Guarded pair replacement
        # =============================================================
        _replace_report_pair(
            validation_path=paths[
                "validation_best"
            ],
            validation_staged=(
                validation_staged
            ),
            test_path=paths[
                "test"
            ],
            test_staged=(
                test_staged
            ),
        )
        # -------------------------------------------------------------
        # Persistence-boundary diagnostic
        # -------------------------------------------------------------
        persisted_validation = (
            _load_json_object(
                paths["validation_best"]
            )
        )

        persisted_test = (
            _load_json_object(
                paths["test"]
            )
        )

        persisted_validation_enum = (
            persisted_validation.get(
                "enumeration"
            )
            or {}
        )

        persisted_test_enum = (
            persisted_test.get(
                "enumeration"
            )
            or {}
        )

        logger.info(
            "Enumeration persistence check: "
            "configured_mode=%s "
            "validation_mode=%s "
            "validation_selected_mode=%s "
            "validation_has_results=%s "
            "test_mode=%s",
            mode,
            persisted_validation_enum.get(
                "mode"
            ),
            persisted_validation_enum.get(
                "selected_mode"
            ),
            isinstance(
                persisted_validation_enum.get(
                    "results"
                ),
                Mapping,
            ),
            persisted_test_enum.get(
                "mode"
            ),
        )
        # =============================================================
        # 8. Completion logging
        # =============================================================
        if mode == "comparison":
            comparison_results = (
                validation_enumeration.get(
                    "results"
                )
            )

            if not isinstance(
                comparison_results,
                Mapping,
            ):
                raise KeyEnumerationHelperError(
                    "Comparison validation evidence "
                    "is missing results"
                )

            selected_validation_result = (
                comparison_results.get(
                    selected_mode
                )
            )

            if not isinstance(
                selected_validation_result,
                Mapping,
            ):
                raise KeyEnumerationHelperError(
                    "Comparison selected mode has no "
                    "validation result"
                )

            logger.info(
                "KEY ENUMERATION COMPARISON completed: "
                "model=%s selected_mode=%s "
                "validation_position=%s "
                "test_position=%s "
                "policy_fingerprint=%s",
                model_name,
                str(
                    selected_mode
                ).upper(),
                selected_validation_result[
                    "enumeration_position"
                ],
                test_enumeration[
                    "enumeration_position"
                ],
                (
                    fingerprint[:12]
                    if isinstance(fingerprint, str)
                    else "N/A (AI baseline)"
                ),
            )

        else:
            logger.info(
                "%s key enumeration completed: "
                "model=%s validation_position=%s "
                "test_position=%s "
                "policy_fingerprint=%s",
                str(
                    mode
                ).upper(),
                model_name,
                validation_enumeration[
                    "enumeration_position"
                ],
                test_enumeration[
                    "enumeration_position"
                ],
                fingerprint[:12],
            )

        logger.info(
            "Enriched key-recovery reports: "
            "validation=%s test=%s",
            paths[
                "validation_best"
            ],
            paths[
                "test"
            ],
        )

        return True

    except (
        KeyEnumerationError,
        KeyEnumerationHelperError,
    ):
        logger.error(
            "Key enumeration failed; the run is not reported as recovered "
            "with enumeration evidence.",
        )

        raise

    except Exception as exc:
        logger.error(
            "Unexpected key-enumeration reporting failure: %s",
            exc,
        )

        raise KeyEnumerationHelperError(
            f"Key enumeration failed: {exc}"
        ) from exc
__all__ = [
    "KeyEnumerationHelperError",
    "_run_completed_cycle_key_enumeration",
]
