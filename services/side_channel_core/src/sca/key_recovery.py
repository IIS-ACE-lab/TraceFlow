#services/side_channel_core/src/sca/key_recovery.py
"""
Key-recovery reporting for Side-Channel Analysis (SCA).

This module converts structured attack evidence into stable, serializable,
human-readable key-recovery reports.

It deliberately does not calculate AES key scores itself. Rank/attack
computation remains owned by rank_curve.py and metrics.py.

Enumeration validation
----------------------
Completed validation/test reports may contain one top-level ``enumeration``
section.

Ordinary executable modes:

    local_hamming
    neighbour_sweep
    support_weighted

Validation reports may additionally contain the meta-mode:

    comparison

``comparison`` contains the raw AI ranking baseline together with independently
validated local_hamming, neighbour_sweep, and support_weighted results produced from the same
validation ranking.

Raw AI participates in final validation selection and is retained unless an
enumeration algorithm achieves a strictly smaller enumeration position, so the
reported rank is never worse than the one the model achieved on its own and
ties are attributed to the model.

If an enumeration algorithm wins validation, only that frozen algorithm is
evaluated on test. If raw AI wins validation, no enumeration algorithm is
executed on test.

The persisted enumeration result is not trusted blindly. This module
independently reconstructs the expected candidate ordering from the persisted
complete 256-candidate AI ranking and verifies:

- complete candidate/rank evidence;
- finite AI scores;
- candidate uniqueness;
- exact deterministic enumeration order;
- true-key enumeration position;
- AI rank at success;
- position delta;
- success phase;
- mode-specific evidence;
- frozen policy fingerprint;
- stop-on-success semantics.

No true-key value is used while reconstructing candidate order. The true key
is consulted only after the expected order is constructed.

Rank contract
-------------
All public ranks are one-based:

    best rank  = 1
    worst rank = 256

Full-population reporting contract
----------------------------------
Validation and test reports describe the complete configured HDF5 split.
Repeated evaluations vary trace order only; every repetition consumes the
same full population. The final rank must therefore be identical across all
trace orders. At that endpoint, ``final_ge`` equals the final true-key rank
and ``final_sr_at_1`` is exactly 1.0 when that rank is 1, otherwise 0.0.

Schema version 2.1 stores the population source, half-open configured range,
available and used trace counts, full-population assertion, and trace-order
repetition count, the complete final-rank vector, evaluator-supplied final GE
and SR@1, and SHA-256 checkpoint provenance for test reports. It does not
represent repetitions as independent attacks and does not expose
requested/clipped trace-budget or recovered-trial fields.

Schema version 2.2 adds ``key.true_key_tie_count`` and ``key.tie_break_policy``:
ties are broken uniformly at random (rank_curve.py), and ``key.recovered``
requires rank 1 with no tie. For disjoint attacks the key section describes the
combined attack on all their traces; GE and SR stay per-attack means.

Data Flow:
-------------
enumeration
    ↓
comparison envelope

selected_enumeration
    ↓
selected local_hamming / neighbour_sweep / support_weighted result

effective_mode
    ↓
algorithm whose detailed fields should be exported
-----------------------------------------------------------------
Author: Shraddha Mishra
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, TextIO, Union

from utils.path_utils import resolve_path
from utils.seal import check as check_seal, seal_text, strip as strip_seal
from sca.dataset_types import resolved_variant


KEYSPACE_SIZE = 256
KEY_WIDTH_BITS = (KEYSPACE_SIZE - 1).bit_length()
PUBLIC_RANK_BASE = 1

ENUMERATION_ALGORITHM_MODES = frozenset(
    {
        "local_hamming",
        "neighbour_sweep",
        "support_weighted",
    }
)

SUPPORTED_ENUMERATION_MODES = frozenset(
    {
        *ENUMERATION_ALGORITHM_MODES,
        "comparison",
    }
)

HYBRID_ENUMERATION_PHASES = frozenset(
    {
        "anchor",
        "hamming_local",
        "ai_fallback",
    }
)

ITERATIVE_ENUMERATION_PHASES = frozenset(
    {
        "anchor",
        "hamming_local",
    }
)

SUPPORT_ENUMERATION_PHASES = frozenset(
    {
        "support_promoted",
        "ai_fallback",
    }
)

DEFAULT_RECOVERY_RANK = 1
# key_recovery_reporting.key_display: how the recovered key is printed (also the panel's menu).
KEY_DISPLAYS = ("full", "masked", "none")

DEFAULT_CANDIDATE_RANKING_MODE = "top_k"
DEFAULT_CANDIDATE_RANKING_TOP_K = 5

SUPPORTED_CANDIDATE_RANKING_MODES = frozenset(
    {
        "all",
        "top_k",
        "none",
    }
)

SCHEMA_VERSION = "2.2"
RANK_CONTRACT_VERSION = "1.0"
# rank_curve.score_rule(): ID uses the log-likelihood, HW/MSB subtract the
# log class prior.
SCORE_RULES = frozenset({"log_likelihood", "log_likelihood_minus_log_class_prior"})

PathLike = Union[str, Path]


class KeyRecoveryError(ValueError):
    """Raised when attack evidence violates the key-recovery contract."""


def _to_builtin(value: Any) -> Any:
    """Recursively convert NumPy/PyTorch-like containers to JSON values."""
    if value is None or isinstance(
        value,
        (str, bool, int, float),
    ):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _to_builtin(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            _to_builtin(item)
            for item in value
        ]

    item = getattr(
        value,
        "item",
        None,
    )

    if callable(item):
        try:
            return _to_builtin(
                item()
            )
        except Exception:
            pass

    tolist = getattr(
        value,
        "tolist",
        None,
    )

    if callable(tolist):
        try:
            return _to_builtin(
                tolist()
            )
        except Exception:
            pass

    return value


def _finite_float(
    value: Any,
    *,
    field: str,
) -> float:
    """Return one finite numeric value."""
    if isinstance(value, bool):
        raise KeyRecoveryError(
            f"{field} must be numeric, got bool"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise KeyRecoveryError(
            f"{field} must be numeric, got {value!r}"
        ) from exc

    if not math.isfinite(result):
        raise KeyRecoveryError(
            f"{field} must be finite, got {result!r}"
        )

    return result


def _optional_positive_int(
    value: Any,
    *,
    field: str,
) -> Optional[int]:
    """Return an optional integer >= 1."""
    if value is None:
        return None

    if isinstance(value, bool):
        raise KeyRecoveryError(
            f"{field} must be an integer or null"
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise KeyRecoveryError(
            f"{field} must be an integer or null"
        ) from exc

    if result < 1:
        raise KeyRecoveryError(
            f"{field} must be >= 1 when present"
        )

    return result


def _byte(
    value: Any,
    *,
    field: str,
) -> int:
    """Return one strict AES byte."""
    if isinstance(value, bool):
        raise KeyRecoveryError(
            f"{field} must be an AES byte, got bool"
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise KeyRecoveryError(
            f"{field} must be an AES byte"
        ) from exc

    if not 0 <= result < KEYSPACE_SIZE:
        raise KeyRecoveryError(
            f"{field} must be in [0, 255], got {result}"
        )

    return result


def _rank(
    value: Any,
    *,
    field: str,
) -> int:
    """Return one public rank in [1, 256]."""
    if isinstance(value, bool):
        raise KeyRecoveryError(
            f"{field} must be a one-based rank"
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise KeyRecoveryError(
            f"{field} must be a one-based rank"
        ) from exc

    if not PUBLIC_RANK_BASE <= result <= KEYSPACE_SIZE:
        raise KeyRecoveryError(
            f"{field} must use public rank range [1, 256], "
            f"got {result}"
        )

    return result


def _as_rank_list(
    values: Any,
) -> list[int]:
    """Normalize repeated attack ranks."""
    built = _to_builtin(
        values
    )

    if (
        not isinstance(built, list)
        or not built
    ):
        raise KeyRecoveryError(
            "final_ranks must be a non-empty sequence"
        )

    ranks: list[int] = []

    for index, value in enumerate(
        built
    ):
        if isinstance(value, list):
            raise KeyRecoveryError(
                "final_ranks must be one-dimensional"
            )

        ranks.append(
            _rank(
                value,
                field=f"final_ranks[{index}]",
            )
        )

    return ranks


def _report_int(
    value: Any,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    """Return one strict integer from persisted JSON."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise KeyRecoveryError(
            f"{field} must be an integer in "
            f"[{minimum}, {maximum}]"
        )

    if not minimum <= value <= maximum:
        raise KeyRecoveryError(
            f"{field} must be in "
            f"[{minimum}, {maximum}], got {value}"
        )

    return value


def _report_bool(
    value: Any,
    *,
    field: str,
) -> bool:
    """Return one strict persisted boolean."""
    if not isinstance(value, bool):
        raise KeyRecoveryError(
            f"{field} must be boolean"
        )

    return value


def _optional_report_int(
    value: Any,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> Optional[int]:
    """Return one optional strict persisted integer."""
    if value is None:
        return None

    return _report_int(
        value,
        field=field,
        minimum=minimum,
        maximum=maximum,
    )


def _resolved_text_scalar(
    value: Any,
    *,
    field: str,
) -> str:
    """Return one non-empty resolved scalar string.

    Lists, mappings, and other containers indicate that run expansion has not
    completed and must never leak into a per-run recovery report.
    """
    if (
        value is None
        or isinstance(value, (Mapping, list, tuple, set))
        or isinstance(value, bool)
    ):
        raise KeyRecoveryError(f"{field} must be one resolved scalar value")

    result = str(value).strip()
    if not result:
        raise KeyRecoveryError(f"{field} must be a non-empty scalar value")
    return result


def _policy_fingerprint(
    policy: Mapping[str, Any],
) -> str:
    """Calculate deterministic SHA-256 for one frozen policy."""
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


def _reporting_cfg(
    cfg: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return reporting configuration."""
    section = (
        cfg.get("key_recovery_reporting")
        or {}
    )

    if not isinstance(
        section,
        Mapping,
    ):
        raise KeyRecoveryError(
            "key_recovery_reporting must be a mapping"
        )

    return section


def validate_key_recovery_contract(
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate and normalize key-recovery reporting configuration."""
    reporting = _reporting_cfg(
        cfg
    )

    enabled = bool(
        reporting.get(
            "enabled",
            False,
        )
    )

    sca_cfg = (
        cfg.get("sca")
        or {}
    )

    if not isinstance(
        sca_cfg,
        Mapping,
    ):
        raise KeyRecoveryError(
            "sca must be a mapping"
        )

    # sca.rank_base is no longer a setting (TraceFlow ranks from 1): a config without it is base 1;
    # a snapshot that still names it must name 1 (a bool is not a base).
    rank_base = sca_cfg.get(
        "rank_base",
        PUBLIC_RANK_BASE,
    )

    if isinstance(rank_base, bool) or rank_base != PUBLIC_RANK_BASE:
        raise KeyRecoveryError(
            "Key-recovery reporting requires "
            "sca.rank_base: 1"
        )

    recovery_rank = reporting.get(
        "recovery_rank",
        DEFAULT_RECOVERY_RANK,
    )

    if isinstance(
        recovery_rank,
        bool,
    ):
        raise KeyRecoveryError(
            "key_recovery_reporting.recovery_rank "
            "must be integer 1"
        )

    try:
        recovery_rank = int(
            recovery_rank
        )
    except (TypeError, ValueError) as exc:
        raise KeyRecoveryError(
            "key_recovery_reporting.recovery_rank "
            "must be integer 1"
        ) from exc

    if recovery_rank != DEFAULT_RECOVERY_RANK:
        raise KeyRecoveryError(
            "Target-byte recovery is defined only at rank 1"
        )

    key_display = str(
        reporting.get(
            "key_display",
            "full",
        )
    ).strip().lower()

    if key_display not in KEY_DISPLAYS:
        raise KeyRecoveryError(
            "key_recovery_reporting.key_display must "
            f"be one of: {', '.join(KEY_DISPLAYS)}"
        )

    candidate_ranking = (
        sca_cfg.get(
            "candidate_ranking"
        )
        or {}
    )

    if not isinstance(
        candidate_ranking,
        Mapping,
    ):
        raise KeyRecoveryError(
            "sca.candidate_ranking must be a mapping"
        )

    candidate_ranking_mode = str(
        candidate_ranking.get(
            "mode",
            DEFAULT_CANDIDATE_RANKING_MODE,
        )
    ).strip().lower()

    if (
        candidate_ranking_mode
        not in SUPPORTED_CANDIDATE_RANKING_MODES
    ):
        raise KeyRecoveryError(
            "sca.candidate_ranking.mode must be one of: "
            "all, none, top_k"
        )

    top_k_raw = candidate_ranking.get(
        "top_k"
    )

    if (
        candidate_ranking_mode == "top_k"
        and top_k_raw is None
    ):
        if "candidate_ranking" not in sca_cfg:
            top_k_raw = (
                DEFAULT_CANDIDATE_RANKING_TOP_K
            )
        else:
            raise KeyRecoveryError(
                "sca.candidate_ranking.top_k is required "
                "when mode='top_k'"
            )

    candidate_ranking_top_k: Optional[
        int
    ] = None

    if top_k_raw is not None:
        if isinstance(
            top_k_raw,
            bool,
        ):
            raise KeyRecoveryError(
                "sca.candidate_ranking.top_k "
                "must be an integer"
            )

        try:
            candidate_ranking_top_k = int(
                top_k_raw
            )
        except (TypeError, ValueError) as exc:
            raise KeyRecoveryError(
                "sca.candidate_ranking.top_k "
                "must be an integer"
            ) from exc

        if not 1 <= candidate_ranking_top_k <= KEYSPACE_SIZE:
            raise KeyRecoveryError(
                "sca.candidate_ranking.top_k "
                "must be in [1, 256]"
            )

    return {
        "enabled": enabled,
        "log_validation": bool(
            reporting.get(
                "log_validation",
                True,
            )
        ),
        "log_test": bool(
            reporting.get(
                "log_test",
                True,
            )
        ),
        "save_validation_reports": bool(
            reporting.get(
                "save_validation_reports",
                True,
            )
        ),
        "save_test_report": bool(
            reporting.get(
                "save_test_report",
                True,
            )
        ),
        "recovery_rank": recovery_rank,
        "require_stable_recovery": bool(
            reporting.get(
                "require_stable_recovery",
                True,
            )
        ),
        "candidate_ranking_mode": (
            candidate_ranking_mode
        ),
        "candidate_ranking_top_k": (
            candidate_ranking_top_k
        ),
        "include_score_margin": bool(
            reporting.get(
                "include_score_margin",
                True,
            )
        ),
        "key_display": key_display,
    }


def _candidate_rows(
    candidates: Any,
    *,
    mode: str,
    top_k: Optional[int],
) -> list[Dict[str, Any]]:
    """Normalize candidate-ranking evidence."""
    if mode == "none":
        return []

    if candidates is None:
        raise KeyRecoveryError(
            f"sca.candidate_ranking.mode={mode!r} "
            "requires candidate evidence"
        )

    built = _to_builtin(
        candidates
    )

    if not isinstance(
        built,
        list,
    ):
        raise KeyRecoveryError(
            "top_key_candidates must be a sequence"
        )

    if mode == "all":
        if len(built) != KEYSPACE_SIZE:
            raise KeyRecoveryError(
                "candidate_ranking.mode='all' requires "
                "exactly 256 candidates"
            )

        selected = built

    else:
        if top_k is None:
            raise KeyRecoveryError(
                "candidate_ranking top_k is missing"
            )

        if len(built) < top_k:
            raise KeyRecoveryError(
                f"Requested top_k={top_k}, but candidate "
                f"evidence contains only {len(built)}"
            )

        selected = built[
            :top_k
        ]

    normalized: list[
        Dict[str, Any]
    ] = []

    seen_guesses: set[
        int
    ] = set()

    for index, candidate in enumerate(
        selected,
        start=1,
    ):
        if isinstance(
            candidate,
            Mapping,
        ):
            guess_raw = candidate.get(
                "key_guess",
                candidate.get("key"),
            )

            score_raw = candidate.get(
                "score"
            )

            rank_raw = candidate.get(
                "rank",
                index,
            )

        elif (
            isinstance(
                candidate,
                (list, tuple),
            )
            and len(candidate) >= 2
        ):
            guess_raw = candidate[0]
            score_raw = candidate[1]
            rank_raw = index

        else:
            raise KeyRecoveryError(
                "Each top_key_candidates entry must be "
                "a mapping or [key_guess, score]"
            )

        key_guess = _byte(
            guess_raw,
            field=(
                f"top_key_candidates[{index}].key_guess"
            ),
        )

        rank = _rank(
            rank_raw,
            field=(
                f"top_key_candidates[{index}].rank"
            ),
        )

        if rank != index:
            raise KeyRecoveryError(
                f"top_key_candidates[{index}].rank "
                f"must be {index}, got {rank}"
            )

        if key_guess in seen_guesses:
            raise KeyRecoveryError(
                "top_key_candidates contains duplicate "
                f"0x{key_guess:02X}"
            )

        seen_guesses.add(
            key_guess
        )

        row: Dict[
            str,
            Any,
        ] = {
            "rank": rank,
            "key_guess": key_guess,
            "key_guess_hex": (
                f"0x{key_guess:02X}"
            ),
        }

        if score_raw is not None:
            row["score"] = _finite_float(
                score_raw,
                field=(
                    f"top_key_candidates[{index}].score"
                ),
            )

        normalized.append(
            row
        )

    return normalized


def _complete_ranked_candidates(
    report: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """Return validated complete 256-entry ranking from one report."""
    diagnostics = (
        report.get(
            "diagnostics"
        )
        or {}
    )

    if not isinstance(
        diagnostics,
        Mapping,
    ):
        raise KeyRecoveryError(
            "report.diagnostics must be a mapping"
        )

    ranked = _candidate_rows(
        diagnostics.get(
            "top_key_candidates"
        ),
        mode="all",
        top_k=None,
    )

    if len(ranked) != KEYSPACE_SIZE:
        raise KeyRecoveryError(
            "Enumeration requires exactly 256 candidates"
        )

    seen = {
        item["key_guess"]
        for item in ranked
    }

    if seen != set(
        range(KEYSPACE_SIZE)
    ):
        raise KeyRecoveryError(
            "Enumeration ranking must contain every "
            "AES byte exactly once"
        )

    for index, candidate in enumerate(
        ranked,
        start=1,
    ):
        if "score" not in candidate:
            raise KeyRecoveryError(
                "Enumeration requires finite AI scores "
                f"for candidate rank {index}"
            )

    return ranked


def _hamming_distance(
    left: int,
    right: int,
) -> int:
    """Return byte Hamming distance."""
    return (
        left ^ right
    ).bit_count()


def _hamming_neighbours(
    value: int,
    *,
    radius: int,
) -> list[tuple[int, int]]:
    """
    Reconstruct deterministic neighbour_sweep neighbourhood ordering.

    Distance 1 precedes distance 2, etc. Within one distance, bit-index
    combinations are lexicographic.
    """
    neighbours: list[
        tuple[int, int]
    ] = []

    for distance in range(
        1,
        radius + 1,
    ):
        for bit_indexes in combinations(
            range(KEY_WIDTH_BITS),
            distance,
        ):
            mask = 0

            for bit_index in bit_indexes:
                mask |= (
                    1 << bit_index
                )

            neighbours.append(
                (
                    value ^ mask,
                    distance,
                )
            )

    return neighbours


def _reconstruct_local_hamming_order(
    ranked: Sequence[
        Mapping[str, Any]
    ],
    *,
    top_k: int,
    radius: int,
    budget: int,
) -> tuple[
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    int,
    int,
]:
    """Independently reconstruct local_hamming candidate ordering."""
    anchors = [
        dict(item)
        for item in ranked[:top_k]
    ]

    anchor_values = tuple(
        item["key_guess"]
        for item in anchors
    )

    anchor_keys = set(
        anchor_values
    )

    decorated: list[
        Dict[str, Any]
    ] = []

    for candidate in ranked:
        distance = min(
            _hamming_distance(
                candidate["key_guess"],
                anchor,
            )
            for anchor in anchor_values
        )

        decorated.append(
            {
                **candidate,
                "hamming_distance": distance,
                "is_anchor": (
                    candidate["key_guess"]
                    in anchor_keys
                ),
            }
        )

    anchor_rows = [
        candidate
        for candidate in decorated
        if candidate["key_guess"]
        in anchor_keys
    ]

    local_pool = [
        candidate
        for candidate in decorated
        if candidate["key_guess"]
        not in anchor_keys
        and candidate[
            "hamming_distance"
        ] <= radius
    ]

    local_pool.sort(
        key=lambda candidate: (
            candidate["rank"]
        )
    )

    promoted = local_pool[
        :budget
    ]

    promoted_keys = {
        candidate["key_guess"]
        for candidate in promoted
    }

    fallback = [
        candidate
        for candidate in decorated
        if candidate["key_guess"]
        not in anchor_keys
        and candidate["key_guess"]
        not in promoted_keys
    ]

    ordered = [
        *[
            {
                **candidate,
                "enumeration_phase": "anchor",
                "is_local_promoted": False,
            }
            for candidate in anchor_rows
        ],
        *[
            {
                **candidate,
                "enumeration_phase": "hamming_local",
                "is_local_promoted": True,
            }
            for candidate in promoted
        ],
        *[
            {
                **candidate,
                "enumeration_phase": "ai_fallback",
                "is_local_promoted": False,
            }
            for candidate in fallback
        ],
    ]

    return (
        ordered,
        anchors,
        len(local_pool),
        len(promoted),
    )


def _reconstruct_neighbour_sweep_order(
    ranked: Sequence[
        Mapping[str, Any]
    ],
    *,
    radius: int,
) -> list[Dict[str, Any]]:
    """Independently reconstruct neighbour_sweep remaining-set traversal."""
    by_guess = {
        item["key_guess"]: item
        for item in ranked
    }

    remaining = set(
        by_guess
    )

    ordered: list[
        Dict[str, Any]
    ] = []

    for ranked_item in ranked:
        anchor = ranked_item[
            "key_guess"
        ]

        if anchor not in remaining:
            continue

        remaining.remove(
            anchor
        )

        ordered.append(
            {
                **ranked_item,
                "enumeration_phase": "anchor",
                "is_anchor": True,
                "anchor_key_guess": anchor,
                "anchor_key_guess_hex": (
                    f"0x{anchor:02X}"
                ),
                "anchor_ai_rank": (
                    ranked_item["rank"]
                ),
                "hamming_distance": 0,
            }
        )

        for (
            neighbour,
            distance,
        ) in _hamming_neighbours(
            anchor,
            radius=radius,
        ):
            if neighbour not in remaining:
                continue

            remaining.remove(
                neighbour
            )

            candidate = by_guess[
                neighbour
            ]

            ordered.append(
                {
                    **candidate,
                    "enumeration_phase": "hamming_local",
                    "is_anchor": False,
                    "anchor_key_guess": anchor,
                    "anchor_key_guess_hex": (
                        f"0x{anchor:02X}"
                    ),
                    "anchor_ai_rank": (
                        ranked_item["rank"]
                    ),
                    "hamming_distance": (
                        distance
                    ),
                }
            )

    if remaining:
        raise KeyRecoveryError(
            "Internal neighbour_sweep reconstruction "
            "left candidates unevaluated"
        )

    return ordered


def _reconstruct_support_weighted_order(
    ranked: Sequence[
        Mapping[str, Any]
    ],
    *,
    top_k: int,
    radius: int,
    budget: int,
) -> tuple[
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    int,
    int,
]:
    """Independently reconstruct Weighted Hamming-Support ordering."""
    anchors = [
        dict(item)
        for item in ranked[:top_k]
    ]

    score_max = max(
        anchor["score"]
        for anchor in anchors
    )

    weighted_anchors = [
        {
            **anchor,
            "relative_weight": math.exp(
                anchor["score"]
                - score_max
            ),
        }
        for anchor in anchors
    ]

    supported: list[
        Dict[str, Any]
    ] = []

    for candidate in ranked:
        support = 0.0
        supporting_count = 0

        minimum_distance: Optional[
            int
        ] = None

        for anchor in weighted_anchors:
            distance = _hamming_distance(
                candidate[
                    "key_guess"
                ],
                anchor[
                    "key_guess"
                ],
            )

            if (
                distance == 0
                or distance > radius
            ):
                continue

            support += (
                anchor[
                    "relative_weight"
                ]
                / float(distance)
            )

            supporting_count += 1

            if (
                minimum_distance is None
                or distance
                < minimum_distance
            ):
                minimum_distance = distance

        if supporting_count == 0:
            continue

        supported.append(
            {
                **candidate,
                "weighted_hamming_support": support,
                "supporting_anchor_count": (
                    supporting_count
                ),
                "minimum_anchor_distance": (
                    minimum_distance
                ),
            }
        )

    supported.sort(
        key=lambda candidate: (
            -candidate[
                "weighted_hamming_support"
            ],
            -candidate[
                "supporting_anchor_count"
            ],
            candidate[
                "minimum_anchor_distance"
            ],
            candidate["rank"],
            candidate["key_guess"],
        )
    )

    promoted = supported[
        :budget
    ]

    promoted_keys = {
        candidate["key_guess"]
        for candidate in promoted
    }

    fallback = [
        candidate
        for candidate in ranked
        if candidate["key_guess"]
        not in promoted_keys
    ]

    ordered = [
        *[
            {
                **candidate,
                "enumeration_phase": "support_promoted",
                "is_support_promoted": True,
            }
            for candidate in promoted
        ],
        *[
            {
                **candidate,
                "enumeration_phase": "ai_fallback",
                "is_support_promoted": False,
            }
            for candidate in fallback
        ],
    ]

    return (
        ordered,
        weighted_anchors,
        len(supported),
        len(promoted),
    )


def _validate_expected_order(
    ordered: Sequence[
        Mapping[str, Any]
    ],
    *,
    mode: str,
) -> None:
    """Validate one internally reconstructed full keyspace ordering."""
    if len(ordered) != KEYSPACE_SIZE:
        raise KeyRecoveryError(
            f"Internal {mode} ordering must contain "
            "exactly 256 candidates"
        )

    guesses = [
        candidate["key_guess"]
        for candidate in ordered
    ]

    if len(set(guesses)) != KEYSPACE_SIZE:
        raise KeyRecoveryError(
            f"Internal {mode} ordering contains duplicates"
        )

    if set(guesses) != set(
        range(KEYSPACE_SIZE)
    ):
        raise KeyRecoveryError(
            f"Internal {mode} ordering does not contain "
            "every AES byte"
        )


def _validate_policy_fingerprint(
    raw: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> str:
    """Validate persisted frozen policy fingerprint."""
    prefix = "report.enumeration"

    policy_frozen = _report_bool(
        raw.get(
            "policy_frozen"
        ),
        field=(
            f"{prefix}.policy_frozen"
        ),
    )

    if not policy_frozen:
        raise KeyRecoveryError(
            f"{prefix}.policy_frozen must be true"
        )

    fingerprint = raw.get(
        "policy_fingerprint"
    )

    if (
        not isinstance(
            fingerprint,
            str,
        )
        or len(fingerprint) != 64
        or any(
            character
            not in "0123456789abcdef"
            for character in fingerprint
        )
    ):
        raise KeyRecoveryError(
            f"{prefix}.policy_fingerprint must be a "
            "lowercase SHA-256 hexadecimal string"
        )

    expected = _policy_fingerprint(
        policy
    )

    if fingerprint != expected:
        raise KeyRecoveryError(
            f"{prefix}.policy_fingerprint does not "
            "match the persisted frozen policy"
        )

    return fingerprint


def _validate_common_enumeration_fields(
    raw: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate fields shared by all enumeration modes."""
    prefix = "report.enumeration"

    enabled = _report_bool(
        raw.get("enabled"),
        field=f"{prefix}.enabled",
    )

    executed = _report_bool(
        raw.get("executed"),
        field=f"{prefix}.executed",
    )

    success = _report_bool(
        raw.get("success"),
        field=f"{prefix}.success",
    )

    if not enabled:
        raise KeyRecoveryError(
            f"{prefix}.enabled must be true in "
            "persisted enumeration evidence"
        )

    if not executed:
        raise KeyRecoveryError(
            f"{prefix}.executed must be true"
        )

    if not success:
        raise KeyRecoveryError(
            "Complete AES-byte enumeration evidence "
            "must reach the true key"
        )

    mode = str(
        raw.get(
            "mode",
            "",
        )
    ).strip().lower()

    if mode not in ENUMERATION_ALGORITHM_MODES:
        raise KeyRecoveryError(
            "Ordinary report.enumeration.mode must be one of: "
            "local_hamming, neighbour_sweep, support_weighted"
        )

    position = _report_int(
        raw.get(
            "enumeration_position"
        ),
        field=(
            f"{prefix}.enumeration_position"
        ),
        minimum=1,
        maximum=KEYSPACE_SIZE,
    )

    candidates_evaluated = _report_int(
        raw.get(
            "candidates_evaluated"
        ),
        field=(
            f"{prefix}.candidates_evaluated"
        ),
        minimum=1,
        maximum=KEYSPACE_SIZE,
    )

    ai_rank = _report_int(
        raw.get(
            "ai_rank_at_success"
        ),
        field=(
            f"{prefix}.ai_rank_at_success"
        ),
        minimum=1,
        maximum=KEYSPACE_SIZE,
    )

    position_delta = _report_int(
        raw.get(
            "position_delta"
        ),
        field=(
            f"{prefix}.position_delta"
        ),
        minimum=-(KEYSPACE_SIZE - 1),
        maximum=KEYSPACE_SIZE - 1,
    )

    if position_delta != (
        position - ai_rank
    ):
        raise KeyRecoveryError(
            "report.enumeration.position_delta must equal "
            "enumeration_position - ai_rank_at_success"
        )

    stop_on_success = _report_bool(
        raw.get(
            "stop_on_success"
        ),
        field=(
            f"{prefix}.stop_on_success"
        ),
    )

    stopped_on_success = _report_bool(
        raw.get(
            "stopped_on_success"
        ),
        field=(
            f"{prefix}.stopped_on_success"
        ),
    )

    if stop_on_success:
        if not stopped_on_success:
            raise KeyRecoveryError(
                "stopped_on_success must be true "
                "when stop_on_success=true"
            )

        if (
            candidates_evaluated
            != position
        ):
            raise KeyRecoveryError(
                "candidates_evaluated must equal "
                "enumeration_position when stopping on success"
            )

    else:
        if stopped_on_success:
            raise KeyRecoveryError(
                "stopped_on_success must be false "
                "when stop_on_success=false"
            )

        if (
            candidates_evaluated
            != KEYSPACE_SIZE
        ):
            raise KeyRecoveryError(
                "Every enumeration mode must evaluate "
                "all 256 candidates when stop_on_success=false"
            )

    return {
        "mode": mode,
        "position": position,
        "candidates_evaluated": (
            candidates_evaluated
        ),
        "ai_rank_at_success": ai_rank,
        "position_delta": (
            position_delta
        ),
        "stop_on_success": (
            stop_on_success
        ),
        "stopped_on_success": (
            stopped_on_success
        ),
    }


def _validate_evaluated_prefix(
    raw: Mapping[str, Any],
    *,
    expected_order: Sequence[
        Mapping[str, Any]
    ],
    mode: str,
) -> list[Dict[str, Any]]:
    """Validate persisted evaluated rows against reconstructed ordering."""
    prefix = "report.enumeration"

    evaluated_raw = raw.get(
        "evaluated_candidates"
    )

    if not isinstance(
        evaluated_raw,
        list,
    ):
        raise KeyRecoveryError(
            f"{prefix}.evaluated_candidates "
            "must be a sequence"
        )

    expected_count = _report_int(
        raw.get(
            "candidates_evaluated"
        ),
        field=(
            f"{prefix}.candidates_evaluated"
        ),
        minimum=1,
        maximum=KEYSPACE_SIZE,
    )

    if len(evaluated_raw) != expected_count:
        raise KeyRecoveryError(
            "evaluated_candidates length must equal "
            "candidates_evaluated"
        )

    seen_guesses: set[
        int
    ] = set()

    observed: list[
        Dict[str, Any]
    ] = []

    for index, item in enumerate(
        evaluated_raw,
        start=1,
    ):
        if not isinstance(
            item,
            Mapping,
        ):
            raise KeyRecoveryError(
                f"{prefix}.evaluated_candidates[{index}] "
                "must be a mapping"
            )

        item_position = _report_int(
            item.get(
                "enumeration_position"
            ),
            field=(
                f"{prefix}.evaluated_candidates"
                f"[{index}].enumeration_position"
            ),
            minimum=1,
            maximum=KEYSPACE_SIZE,
        )

        if item_position != index:
            raise KeyRecoveryError(
                "enumeration_position must match "
                "evaluated row index"
            )

        ai_rank = _report_int(
            item.get(
                "ai_rank"
            ),
            field=(
                f"{prefix}.evaluated_candidates"
                f"[{index}].ai_rank"
            ),
            minimum=1,
            maximum=KEYSPACE_SIZE,
        )

        key_guess = _byte(
            item.get(
                "key_guess"
            ),
            field=(
                f"{prefix}.evaluated_candidates"
                f"[{index}].key_guess"
            ),
        )

        if key_guess in seen_guesses:
            raise KeyRecoveryError(
                "enumeration evaluated a candidate twice: "
                f"0x{key_guess:02X}"
            )

        seen_guesses.add(
            key_guess
        )

        if (
            item.get(
                "key_guess_hex"
            )
            != f"0x{key_guess:02X}"
        ):
            raise KeyRecoveryError(
                "evaluated key_guess_hex does not match key_guess"
            )

        score = _finite_float(
            item.get(
                "score"
            ),
            field=(
                f"{prefix}.evaluated_candidates"
                f"[{index}].score"
            ),
        )

        phase = str(
            item.get(
                "enumeration_phase",
                "",
            )
        ).strip().lower()

        expected = expected_order[
            index - 1
        ]

        if (
            ai_rank
            != expected["rank"]
            or key_guess
            != expected["key_guess"]
            or score
            != expected["score"]
            or phase
            != expected["enumeration_phase"]
        ):
            raise KeyRecoveryError(
                f"{mode} evaluated_candidates does not "
                "match reconstructed candidate ordering"
            )

        if mode == "local_hamming":
            distance = _report_int(
                item.get(
                    "hamming_distance"
                ),
                field=(
                    f"{prefix}.evaluated_candidates"
                    f"[{index}].hamming_distance"
                ),
                minimum=0,
                maximum=KEY_WIDTH_BITS,
            )

            is_anchor = _report_bool(
                item.get(
                    "is_anchor"
                ),
                field=(
                    f"{prefix}.evaluated_candidates"
                    f"[{index}].is_anchor"
                ),
            )

            promoted = _report_bool(
                item.get(
                    "is_local_promoted"
                ),
                field=(
                    f"{prefix}.evaluated_candidates"
                    f"[{index}].is_local_promoted"
                ),
            )

            if (
                distance
                != expected[
                    "hamming_distance"
                ]
                or is_anchor
                != expected[
                    "is_anchor"
                ]
                or promoted
                != expected[
                    "is_local_promoted"
                ]
            ):
                raise KeyRecoveryError(
                    "local_hamming mode-specific evidence does not "
                    "match reconstructed ordering"
                )

        elif mode == "neighbour_sweep":
            distance = _report_int(
                item.get(
                    "hamming_distance"
                ),
                field=(
                    f"{prefix}.evaluated_candidates"
                    f"[{index}].hamming_distance"
                ),
                minimum=0,
                maximum=KEY_WIDTH_BITS,
            )

            is_anchor = _report_bool(
                item.get(
                    "is_anchor"
                ),
                field=(
                    f"{prefix}.evaluated_candidates"
                    f"[{index}].is_anchor"
                ),
            )

            anchor_guess = _byte(
                item.get(
                    "anchor_key_guess"
                ),
                field=(
                    f"{prefix}.evaluated_candidates"
                    f"[{index}].anchor_key_guess"
                ),
            )

            anchor_rank = _report_int(
                item.get(
                    "anchor_ai_rank"
                ),
                field=(
                    f"{prefix}.evaluated_candidates"
                    f"[{index}].anchor_ai_rank"
                ),
                minimum=1,
                maximum=KEYSPACE_SIZE,
            )

            if (
                distance
                != expected[
                    "hamming_distance"
                ]
                or is_anchor
                != expected[
                    "is_anchor"
                ]
                or anchor_guess
                != expected[
                    "anchor_key_guess"
                ]
                or anchor_rank
                != expected[
                    "anchor_ai_rank"
                ]
            ):
                raise KeyRecoveryError(
                    "neighbour_sweep traversal evidence does not "
                    "match reconstructed ordering"
                )

            if (
                item.get(
                    "anchor_key_guess_hex"
                )
                != f"0x{anchor_guess:02X}"
            ):
                raise KeyRecoveryError(
                    "anchor_key_guess_hex does not match "
                    "anchor_key_guess"
                )

        elif mode == "support_weighted":
            promoted = _report_bool(
                item.get(
                    "is_support_promoted"
                ),
                field=(
                    f"{prefix}.evaluated_candidates"
                    f"[{index}].is_support_promoted"
                ),
            )

            expected_promoted = expected[
                "is_support_promoted"
            ]

            if promoted != expected_promoted:
                raise KeyRecoveryError(
                    "Support promotion evidence does not "
                    "match reconstructed ordering"
                )

            if expected_promoted:
                support = _finite_float(
                    item.get(
                        "weighted_hamming_support"
                    ),
                    field=(
                        f"{prefix}.evaluated_candidates"
                        f"[{index}].weighted_hamming_support"
                    ),
                )

                supporting_count = _report_int(
                    item.get(
                        "supporting_anchor_count"
                    ),
                    field=(
                        f"{prefix}.evaluated_candidates"
                        f"[{index}].supporting_anchor_count"
                    ),
                    minimum=1,
                    maximum=KEYSPACE_SIZE,
                )

                minimum_distance = _report_int(
                    item.get(
                        "minimum_anchor_distance"
                    ),
                    field=(
                        f"{prefix}.evaluated_candidates"
                        f"[{index}].minimum_anchor_distance"
                    ),
                    minimum=1,
                    maximum=KEY_WIDTH_BITS,
                )

                if not math.isclose(
                    support,
                    expected[
                        "weighted_hamming_support"
                    ],
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    raise KeyRecoveryError(
                        "weighted_hamming_support does not "
                        "match independent reconstruction"
                    )

                if (
                    supporting_count
                    != expected[
                        "supporting_anchor_count"
                    ]
                    or minimum_distance
                    != expected[
                        "minimum_anchor_distance"
                    ]
                ):
                    raise KeyRecoveryError(
                        "Support corroboration evidence does not "
                        "match independent reconstruction"
                    )

        observed.append(
            {
                "enumeration_position": (
                    item_position
                ),
                "ai_rank": ai_rank,
                "key_guess": key_guess,
                "score": score,
                "enumeration_phase": phase,
                **dict(item),
            }
        )

    return observed

def _validate_sha256_hex(
    value: Any,
    *,
    field: str,
) -> str:
    """Return one strict lowercase SHA-256 hexadecimal fingerprint."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise KeyRecoveryError(
            f"{field} must be a lowercase "
            "64-character SHA-256 hexadecimal string"
        )

    return value

def _validate_ai_comparison_result(
    report: Mapping[str, Any],
    raw: Any,
) -> Dict[str, Any]:
    """
    Validate the raw-AI baseline embedded in validation comparison evidence.

    ``ai`` is not an executable key-enumeration mode. It exists only as the
    immutable baseline against which LOCAL_HAMMING/NEIGHBOUR_SWEEP/SUPPORT_WEIGHTED are compared.
    """
    if not isinstance(
        raw,
        Mapping,
    ):
        raise KeyRecoveryError(
            "report.enumeration.results.ai must be a mapping"
        )

    mode = str(
        raw.get(
            "mode",
            "",
        )
    ).strip().lower()

    if mode != "ai":
        raise KeyRecoveryError(
            "Comparison AI baseline must have mode='ai'"
        )

    enabled = _report_bool(
        raw.get(
            "enabled"
        ),
        field=(
            "report.enumeration.results.ai.enabled"
        ),
    )

    executed = _report_bool(
        raw.get(
            "executed"
        ),
        field=(
            "report.enumeration.results.ai.executed"
        ),
    )



    if enabled:
        raise KeyRecoveryError(
            "Comparison AI baseline enabled must be false"
        )

    if executed:
        raise KeyRecoveryError(
            "Comparison AI baseline executed must be false"
        )



    key = (
        report.get(
            "key"
        )
        or {}
    )

    if not isinstance(
        key,
        Mapping,
    ):
        raise KeyRecoveryError(
            "report.key must be a mapping"
        )

    final_rank = _rank(
        key.get(
            "final_true_key_rank"
        ),
        field=(
            "report.key.final_true_key_rank"
        ),
    )

    ai_rank = _report_int(
        raw.get(
            "ai_rank_at_success"
        ),
        field=(
            "report.enumeration.results.ai."
            "ai_rank_at_success"
        ),
        minimum=1,
        maximum=KEYSPACE_SIZE,
    )

    position = _report_int(
        raw.get(
            "enumeration_position"
        ),
        field=(
            "report.enumeration.results.ai."
            "enumeration_position"
        ),
        minimum=1,
        maximum=KEYSPACE_SIZE,
    )

    candidates_evaluated = _report_int(
        raw.get(
            "candidates_evaluated"
        ),
        field=(
            "report.enumeration.results.ai."
            "candidates_evaluated"
        ),
        minimum=1,
        maximum=KEYSPACE_SIZE,
    )

    position_delta = _report_int(
        raw.get(
            "position_delta"
        ),
        field=(
            "report.enumeration.results.ai."
            "position_delta"
        ),
        minimum=0,
        maximum=0,
    )

    if (
        ai_rank != final_rank
        or position != final_rank
        or candidates_evaluated != final_rank
        or position_delta != 0
    ):
        raise KeyRecoveryError(
            "Comparison AI baseline must reproduce the "
            "original final_true_key_rank exactly"
        )

    return _to_builtin(
        dict(raw)
    )

def _validate_comparison_policy(
    *,
    policy: Any,
    selected_mode: str,
    expected_fingerprint: str,
) -> Dict[str, Any]:
    """
    Validate the frozen policy published by comparison selection.

    The policy fingerprint is independently recomputed after removing the
    embedded ``policy_fingerprint`` field because the lifecycle helper hashes
    the policy before adding that field.
    """
    if not isinstance(
        policy,
        Mapping,
    ):
        raise KeyRecoveryError(
            "report.enumeration.selected_policy must be a mapping"
        )

    built = dict(
        policy
    )

    policy_mode = str(
        built.get(
            "mode",
            "",
        )
    ).strip().lower()

    if policy_mode != selected_mode:
        raise KeyRecoveryError(
            "Comparison selected_policy.mode does not "
            "match selected_mode"
        )

    embedded_fingerprint = _validate_sha256_hex(
        built.get(
            "policy_fingerprint"
        ),
        field=(
            "report.enumeration."
            "selected_policy.policy_fingerprint"
        ),
    )

    if embedded_fingerprint != expected_fingerprint:
        raise KeyRecoveryError(
            "Comparison selected_policy fingerprint does "
            "not match selected_policy_fingerprint"
        )

    policy_without_fingerprint = dict(
        built
    )

    policy_without_fingerprint.pop(
        "policy_fingerprint",
        None,
    )

    independently_computed = (
        _policy_fingerprint(
            policy_without_fingerprint
        )
    )

    if independently_computed != expected_fingerprint:
        raise KeyRecoveryError(
            "Comparison selected policy fingerprint does "
            "not match independently reconstructed policy hash"
        )

    return built


def _validate_comparison_report_section(
    report: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Strictly validate validation-time comparison evidence.

    Comparison contains:

        ai
        local_hamming
        neighbour_sweep
        support_weighted

    Raw AI participates in final selection and wins ties; an enumeration
    algorithm is selected only when it strictly improves the true-key rank.
    """
    process = str(
        report.get(
            "process",
            "",
        )
    ).strip().lower()

    if process != "validation":
        raise KeyRecoveryError(
            "Full enumeration comparison evidence is valid "
            "only in a validation report"
        )

    enabled = _report_bool(
        raw.get(
            "enabled"
        ),
        field="report.enumeration.enabled",
    )

    executed = _report_bool(
        raw.get(
            "executed"
        ),
        field="report.enumeration.executed",
    )

    success = _report_bool(
        raw.get(
            "success"
        ),
        field="report.enumeration.success",
    )

    if not enabled:
        raise KeyRecoveryError(
            "Comparison enumeration.enabled must be true"
        )

    if not executed:
        raise KeyRecoveryError(
            "Comparison enumeration.executed must be true"
        )

    if not success:
        raise KeyRecoveryError(
            "Comparison enumeration.success must be true"
        )

    selection_split = str(
        raw.get(
            "selection_split",
            "",
        )
    ).strip().lower()

    if selection_split != "validation":
        raise KeyRecoveryError(
            "Comparison selection_split must be 'validation'"
        )

    selection_basis = str(
        raw.get(
            "selection_basis",
            "",
        )
    ).strip().lower()

    if selection_basis != "model_guarded_minimum":
        raise KeyRecoveryError(
            "Comparison selection_basis must be 'model_guarded_minimum': "
            "the reported rank is the better of the model rank and the "
            "enumeration position, and the model wins ties"
        )

    selected_mode = str(
        raw.get(
            "selected_mode",
            "",
        )
    ).strip().lower()

    if selected_mode not in {
        "ai",
        "local_hamming",
        "neighbour_sweep",
        "support_weighted",
    }:
        raise KeyRecoveryError(
            "Comparison selected_mode must be one of: "
            "ai, local_hamming, neighbour_sweep, support_weighted"
        )

    results = raw.get(
        "results"
    )

    if not isinstance(
        results,
        Mapping,
    ):
        raise KeyRecoveryError(
            "report.enumeration.results must be a mapping"
        )

    expected_modes = {
        "ai",
        "local_hamming",
        "neighbour_sweep",
        "support_weighted",
    }

    if set(results) != expected_modes:
        raise KeyRecoveryError(
            "Comparison results must contain exactly: "
            "ai, local_hamming, neighbour_sweep, support_weighted"
        )

    validated_results: Dict[
        str,
        Dict[str, Any],
    ] = {}

    validated_results[
        "ai"
    ] = _validate_ai_comparison_result(
        report,
        results["ai"],
    )

    for algorithm_mode in (
        "local_hamming",
        "neighbour_sweep",
        "support_weighted",
    ):
        nested = results.get(
            algorithm_mode
        )

        if not isinstance(
            nested,
            Mapping,
        ):
            raise KeyRecoveryError(
                "Comparison result must be a mapping: "
                f"{algorithm_mode}"
            )

        if (
            str(
                nested.get(
                    "mode",
                    "",
                )
            ).strip().lower()
            != algorithm_mode
        ):
            raise KeyRecoveryError(
                "Comparison result mode does not match "
                f"its key: {algorithm_mode}"
            )

        nested_report = dict(
            report
        )

        nested_report[
            "enumeration"
        ] = dict(
            nested
        )

        validated = (
            validate_key_enumeration_report_section(
                nested_report
            )
        )

        if validated is None:
            raise KeyRecoveryError(
                "Comparison nested enumeration unexpectedly "
                "validated as unavailable"
            )

        validated_results[
            algorithm_mode
        ] = validated

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
            validated_results[
                candidate_mode
            ][
                "enumeration_position"
            ],
            validated_results[
                candidate_mode
            ][
                "candidates_evaluated"
            ],
            validated_results[
                candidate_mode
            ][
                "position_delta"
            ],
            mode_priority[
                candidate_mode
            ],
        ),
    )

    ai_position = validated_results[
        "ai"
    ][
        "enumeration_position"
    ]

    best_enumeration_position = (
        validated_results[
            best_enumeration_mode
        ][
            "enumeration_position"
        ]
    )

    # The model wins ties; enumeration must strictly improve to be selected.
    if (
        ai_position
        <= best_enumeration_position
    ):
        expected_selected_mode = "ai"
    else:
        expected_selected_mode = (
            best_enumeration_mode
        )

    if selected_mode != expected_selected_mode:
        raise KeyRecoveryError(
            "Comparison selected_mode is not the independently "
            "reconstructed validation winner: "
            f"reported={selected_mode!r}, "
            f"expected={expected_selected_mode!r}"
        )

    #
    # AI selection owns no enumeration policy.
    #
    if selected_mode == "ai":
        if raw.get(
            "selected_policy"
        ) is not None:
            raise KeyRecoveryError(
                "AI comparison selection must have "
                "selected_policy=null"
            )

        if raw.get(
            "selected_policy_fingerprint"
        ) is not None:
            raise KeyRecoveryError(
                "AI comparison selection must have "
                "selected_policy_fingerprint=null"
            )

        selected_policy = None
        selected_fingerprint = None

    else:
        selected_fingerprint = (
            _validate_sha256_hex(
                raw.get(
                    "selected_policy_fingerprint"
                ),
                field=(
                    "report.enumeration."
                    "selected_policy_fingerprint"
                ),
            )
        )

        selected_result = (
            validated_results[
                selected_mode
            ]
        )

        result_fingerprint = (
            _validate_sha256_hex(
                selected_result.get(
                    "policy_fingerprint"
                ),
                field=(
                    "report.enumeration.results."
                    f"{selected_mode}."
                    "policy_fingerprint"
                ),
            )
        )

        if (
            result_fingerprint
            != selected_fingerprint
        ):
            raise KeyRecoveryError(
                "Comparison selected policy fingerprint "
                "does not match selected algorithm result"
            )

        selected_policy = (
            _validate_comparison_policy(
                policy=raw.get(
                    "selected_policy"
                ),
                selected_mode=(
                    selected_mode
                ),
                expected_fingerprint=(
                    selected_fingerprint
                ),
            )
        )

    return _to_builtin(
        {
            **dict(raw),
            "selected_mode": selected_mode,
            "selected_policy": selected_policy,
            "selected_policy_fingerprint": (
                selected_fingerprint
            ),
            "results": validated_results,
        }
    )
    
def _validate_ai_selected_test_section(
    report: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Validate test evidence when validation comparison selected raw AI.

    No enumeration algorithm executes on test.
    """
    process = str(
        report.get(
            "process",
            "",
        )
    ).strip().lower()

    if process != "test":
        raise KeyRecoveryError(
            "AI-selected comparison result is valid only for test"
        )

    enabled = _report_bool(
        raw.get(
            "enabled"
        ),
        field="report.enumeration.enabled",
    )

    executed = _report_bool(
        raw.get(
            "executed"
        ),
        field="report.enumeration.executed",
    )



    if not enabled:
        raise KeyRecoveryError(
            "AI-selected comparison evidence must remain enabled"
        )

    if executed:
        raise KeyRecoveryError(
            "No enumeration algorithm may execute when "
            "validation selected raw AI"
        )



    if (
        raw.get(
            "selected_by_comparison"
        )
        is not True
    ):
        raise KeyRecoveryError(
            "AI-selected test evidence must declare "
            "selected_by_comparison=true"
        )

    if (
        str(
            raw.get(
                "selection_split",
                "",
            )
        ).strip().lower()
        != "validation"
    ):
        raise KeyRecoveryError(
            "AI-selected test evidence must use "
            "selection_split='validation'"
        )

    if (
        str(
            raw.get(
                "selected_mode",
                "",
            )
        ).strip().lower()
        != "ai"
    ):
        raise KeyRecoveryError(
            "AI-selected test evidence must have selected_mode='ai'"
        )

    key = (
        report.get(
            "key"
        )
        or {}
    )

    if not isinstance(
        key,
        Mapping,
    ):
        raise KeyRecoveryError(
            "report.key must be a mapping"
        )

    final_rank = _rank(
        key.get(
            "final_true_key_rank"
        ),
        field="report.key.final_true_key_rank",
    )

    ai_rank = _report_int(
        raw.get(
            "ai_rank_at_success"
        ),
        field=(
            "report.enumeration.ai_rank_at_success"
        ),
        minimum=1,
        maximum=KEYSPACE_SIZE,
    )

    position = _report_int(
        raw.get(
            "enumeration_position"
        ),
        field=(
            "report.enumeration.enumeration_position"
        ),
        minimum=1,
        maximum=KEYSPACE_SIZE,
    )

    delta = _report_int(
        raw.get(
            "position_delta"
        ),
        field=(
            "report.enumeration.position_delta"
        ),
        minimum=0,
        maximum=0,
    )

    if (
        ai_rank != final_rank
        or position != final_rank
        or delta != 0
    ):
        raise KeyRecoveryError(
            "AI-selected test evidence must preserve "
            "the independent test AI rank exactly"
        )

    return _to_builtin(
        dict(raw)
    )

def validate_key_enumeration_report_section(
    report: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Strictly validate optional persisted enumeration evidence.

    Candidate ordering is reconstructed independently from the complete
    AI-ranking evidence before the true key is inspected.
    """
    if not isinstance(
        report,
        Mapping,
    ):
        raise KeyRecoveryError(
            "key-recovery report must be a mapping"
        )

    raw = report.get(
        "enumeration"
    )

    if raw is None:
        return None

    if not isinstance(
        raw,
        Mapping,
    ):
        raise KeyRecoveryError(
            "report.enumeration must be a mapping"
        )

    process = str(
        report.get(
            "process",
            "",
        )
    ).strip().lower()

    if process not in {
        "validation",
        "test",
    }:
        raise KeyRecoveryError(
            "Enumeration evidence is valid only for "
            "validation or test reports"
        )
    raw_mode = str(
        raw.get(
            "mode",
            "",
        )
    ).strip().lower()

    if (
        raw_mode
        not in SUPPORTED_ENUMERATION_MODES
    ):
        raise KeyRecoveryError(
            "report.enumeration.mode must be one of: "
            "local_hamming, neighbour_sweep, support_weighted, comparison"
        )

    #
    # Comparison is a validation-time envelope around three ordinary
    # enumeration results. It must be handled before ordinary common-field
    # validation because the envelope itself has no single enumeration
    # position.
    #
    if raw_mode == "comparison":
        if process == "validation":
            return (
                _validate_comparison_report_section(
                    report,
                    raw,
                )
            )

        return (
            _validate_ai_selected_test_section(
                report,
                raw,
            )
        )

    common = (
        _validate_common_enumeration_fields(
            raw
        )
    )

    mode = common[
        "mode"
    ]

    ranked = (
        _complete_ranked_candidates(
            report
        )
    )

    if mode == "local_hamming":
        policy_source = str(
            raw.get(
                "policy_source",
                "",
            )
        ).strip().lower()

        if policy_source != "configuration":
            raise KeyRecoveryError(
                "local_hamming policy_source must be 'configuration'"
            )

        top_k = _report_int(
            raw.get(
                "top_k"
            ),
            field="report.enumeration.top_k",
            minimum=1,
            maximum=KEYSPACE_SIZE - 1,
        )

        radius = _report_int(
            raw.get(
                "hamming_radius"
            ),
            field=(
                "report.enumeration.hamming_radius"
            ),
            minimum=1,
            maximum=KEY_WIDTH_BITS,
        )

        budget = _report_int(
            raw.get(
                "local_budget"
            ),
            field=(
                "report.enumeration.local_budget"
            ),
            minimum=1,
            maximum=KEYSPACE_SIZE - top_k,
        )

        fallback_order = str(
            raw.get(
                "fallback_order",
                "",
            )
        ).strip().lower()

        if fallback_order != "ai":
            raise KeyRecoveryError(
                "local_hamming fallback_order must be 'ai'"
            )

        policy = {
            "mode": "local_hamming",
            "policy_source": "configuration",
            "top_k": top_k,
            "hamming_radius": radius,
            "local_budget": budget,
            "fallback_order": fallback_order,
            "stop_on_success": (
                common[
                    "stop_on_success"
                ]
            ),
        }

        _validate_policy_fingerprint(
            raw,
            policy=policy,
        )

        (
            expected_order,
            expected_anchors,
            local_available,
            local_promoted,
        ) = _reconstruct_local_hamming_order(
            ranked,
            top_k=top_k,
            radius=radius,
            budget=budget,
        )

        _validate_expected_order(
            expected_order,
            mode=mode,
        )

        reported_available = _report_int(
            raw.get(
                "local_candidates_available"
            ),
            field=(
                "report.enumeration."
                "local_candidates_available"
            ),
            minimum=0,
            maximum=KEYSPACE_SIZE,
        )

        reported_promoted = _report_int(
            raw.get(
                "local_candidates_promoted"
            ),
            field=(
                "report.enumeration."
                "local_candidates_promoted"
            ),
            minimum=0,
            maximum=KEYSPACE_SIZE,
        )

        if (
            reported_available
            != local_available
            or reported_promoted
            != local_promoted
        ):
            raise KeyRecoveryError(
                "local_hamming local candidate counts do not "
                "match reconstruction"
            )

        anchor_candidates = raw.get(
            "anchor_candidates"
        )

        if not isinstance(
            anchor_candidates,
            list,
        ):
            raise KeyRecoveryError(
                "report.enumeration.anchor_candidates "
                "must be a sequence"
            )

        if (
            anchor_candidates
            != expected_anchors
        ):
            raise KeyRecoveryError(
                "local_hamming anchor_candidates must equal "
                "the top-K AI candidates"
            )

        phases = (
            HYBRID_ENUMERATION_PHASES
        )


    elif mode == "neighbour_sweep":
        policy_source = str(
            raw.get(
                "policy_source",
                "",
            )
        ).strip().lower()

        if policy_source != "configuration":
            raise KeyRecoveryError(
                "neighbour_sweep policy_source must be "
                "'configuration'"
            )

        radius = _report_int(
            raw.get(
                "hamming_radius"
            ),
            field=(
                "report.enumeration.hamming_radius"
            ),
            minimum=1,
            maximum=KEY_WIDTH_BITS,
        )

        policy = {
            "mode": "neighbour_sweep",
            "policy_source": (
                "configuration"
            ),
            "hamming_radius": radius,
            "stop_on_success": (
                common[
                    "stop_on_success"
                ]
            ),
        }

        _validate_policy_fingerprint(
            raw,
            policy=policy,
        )

        expected_order = (
            _reconstruct_neighbour_sweep_order(
                ranked,
                radius=radius,
            )
        )

        _validate_expected_order(
            expected_order,
            mode=mode,
        )

        phases = (
            ITERATIVE_ENUMERATION_PHASES
        )

    else:
        policy_source = str(
            raw.get(
                "policy_source",
                "",
            )
        ).strip().lower()

        if policy_source != "configuration":
            raise KeyRecoveryError(
                "Support policy_source must be "
                "'configuration'"
            )

        top_k = _report_int(
            raw.get(
                "top_k"
            ),
            field=(
                "report.enumeration.top_k"
            ),
            minimum=1,
            maximum=KEYSPACE_SIZE - 1,
        )

        radius = _report_int(
            raw.get(
                "hamming_radius"
            ),
            field=(
                "report.enumeration.hamming_radius"
            ),
            minimum=1,
            maximum=KEY_WIDTH_BITS,
        )

        budget = _report_int(
            raw.get(
                "local_budget"
            ),
            field=(
                "report.enumeration.local_budget"
            ),
            minimum=1,
            maximum=KEYSPACE_SIZE - 1,
        )

        fallback_order = str(
            raw.get(
                "fallback_order",
                "",
            )
        ).strip().lower()

        if fallback_order != "ai":
            raise KeyRecoveryError(
                "Support fallback_order must be 'ai'"
            )

        policy = {
            "mode": "support_weighted",
            "policy_source": (
                "configuration"
            ),
            "top_k": top_k,
            "hamming_radius": radius,
            "local_budget": budget,
            "fallback_order": (
                fallback_order
            ),
            "stop_on_success": (
                common[
                    "stop_on_success"
                ]
            ),
        }

        _validate_policy_fingerprint(
            raw,
            policy=policy,
        )

        (
            expected_order,
            expected_anchors,
            supported_available,
            supported_promoted,
        ) = _reconstruct_support_weighted_order(
            ranked,
            top_k=top_k,
            radius=radius,
            budget=budget,
        )

        _validate_expected_order(
            expected_order,
            mode=mode,
        )

        reported_available = _report_int(
            raw.get(
                "supported_candidates_available"
            ),
            field=(
                "report.enumeration."
                "supported_candidates_available"
            ),
            minimum=0,
            maximum=KEYSPACE_SIZE,
        )

        reported_promoted = _report_int(
            raw.get(
                "supported_candidates_promoted"
            ),
            field=(
                "report.enumeration."
                "supported_candidates_promoted"
            ),
            minimum=0,
            maximum=KEYSPACE_SIZE,
        )

        if (
            reported_available
            != supported_available
            or reported_promoted
            != supported_promoted
        ):
            raise KeyRecoveryError(
                "Support candidate counts do not "
                "match independent reconstruction"
            )

        anchor_candidates = raw.get(
            "anchor_candidates"
        )

        if not isinstance(
            anchor_candidates,
            list,
        ):
            raise KeyRecoveryError(
                "report.enumeration.anchor_candidates "
                "must be a sequence"
            )

        if len(anchor_candidates) != top_k:
            raise KeyRecoveryError(
                "Support anchor_candidates length "
                "must equal top_k"
            )

        for index, (
            observed_anchor,
            expected_anchor,
        ) in enumerate(
            zip(
                anchor_candidates,
                expected_anchors,
            ),
            start=1,
        ):
            if not isinstance(
                observed_anchor,
                Mapping,
            ):
                raise KeyRecoveryError(
                    "Support anchor candidate must be a mapping"
                )

            if (
                observed_anchor.get(
                    "rank"
                )
                != expected_anchor[
                    "rank"
                ]
                or observed_anchor.get(
                    "key_guess"
                )
                != expected_anchor[
                    "key_guess"
                ]
                or observed_anchor.get(
                    "score"
                )
                != expected_anchor[
                    "score"
                ]
            ):
                raise KeyRecoveryError(
                    f"Support anchor_candidates[{index}] "
                    "does not match AI ranking"
                )

            observed_weight = _finite_float(
                observed_anchor.get(
                    "relative_weight"
                ),
                field=(
                    "report.enumeration."
                    f"anchor_candidates[{index}]."
                    "relative_weight"
                ),
            )

            if not math.isclose(
                observed_weight,
                expected_anchor[
                    "relative_weight"
                ],
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise KeyRecoveryError(
                    "Support anchor relative_weight does "
                    "not match independent reconstruction"
                )

        phases = (
            SUPPORT_ENUMERATION_PHASES
        )

    observed_rows = (
        _validate_evaluated_prefix(
            raw,
            expected_order=expected_order,
            mode=mode,
        )
    )

    success_phase = str(
        raw.get(
            "success_phase",
            "",
        )
    ).strip().lower()

    if success_phase not in phases:
        raise KeyRecoveryError(
            f"Invalid {mode} success_phase: "
            f"{success_phase!r}"
        )

    key = (
        report.get(
            "key"
        )
        or {}
    )

    if not isinstance(
        key,
        Mapping,
    ):
        raise KeyRecoveryError(
            "report.key must be a mapping"
        )

    true_key = _byte(
        key.get(
            "true_key_byte"
        ),
        field=(
            "report.key.true_key_byte"
        ),
    )

    final_rank = _rank(
        key.get(
            "final_true_key_rank"
        ),
        field=(
            "report.key.final_true_key_rank"
        ),
    )

    success_row = observed_rows[
        common["position"] - 1
    ]
    recovered_key = _byte(
        raw.get(
            "recovered_key_byte"
        ),
        field=(
            "report.enumeration."
            "recovered_key_byte"
        ),
    )

    recovered_key_hex = raw.get(
        "recovered_key_hex"
    )

    if (
        recovered_key_hex
        != f"0x{recovered_key:02X}"
    ):
        raise KeyRecoveryError(
            "report.enumeration.recovered_key_hex "
            "does not match recovered_key_byte"
        )

    if (
        recovered_key
        != success_row[
            "key_guess"
        ]
    ):
        raise KeyRecoveryError(
            "report.enumeration.recovered_key_byte "
            "does not match the successful enumeration candidate"
        )

    if recovered_key != true_key:
        raise KeyRecoveryError(
            "Enumeration reported a recovered key "
            "different from the benchmark true key"
        )
    if (
        success_row[
            "key_guess"
        ]
        != true_key
    ):
        raise KeyRecoveryError(
            "enumeration_position does not identify "
            "the report true key"
        )

    if (
        success_row[
            "ai_rank"
        ]
        != common[
            "ai_rank_at_success"
        ]
    ):
        raise KeyRecoveryError(
            "ai_rank_at_success does not match "
            "the success row"
        )

    if (
        common[
            "ai_rank_at_success"
        ]
        != final_rank
    ):
        raise KeyRecoveryError(
            "ai_rank_at_success must equal "
            "report.key.final_true_key_rank"
        )

    if (
        success_row[
            "enumeration_phase"
        ]
        != success_phase
    ):
        raise KeyRecoveryError(
            "success_phase does not match "
            "the success row"
        )

    if mode == "local_hamming":
        expected_fallback_used = any(
            candidate[
                "enumeration_phase"
            ] == "ai_fallback"
            for candidate in expected_order[
                :common[
                    "candidates_evaluated"
                ]
            ]
        )

        fallback_used = _report_bool(
            raw.get(
                "fallback_used"
            ),
            field=(
                "report.enumeration.fallback_used"
            ),
        )

        if (
            fallback_used
            != expected_fallback_used
        ):
            raise KeyRecoveryError(
                "local_hamming fallback_used does not "
                "match evaluated phases"
            )

        distance_at_success = (
            _report_int(
                raw.get(
                    "hamming_distance_at_success"
                ),
                field=(
                    "report.enumeration."
                    "hamming_distance_at_success"
                ),
                minimum=0,
                maximum=KEY_WIDTH_BITS,
            )
        )

        if (
            distance_at_success
            != success_row[
                "hamming_distance"
            ]
        ):
            raise KeyRecoveryError(
                "local_hamming hamming_distance_at_success "
                "does not match success row"
            )

    elif mode == "neighbour_sweep":
        distance_at_success = (
            _report_int(
                raw.get(
                    "hamming_distance_at_success"
                ),
                field=(
                    "report.enumeration."
                    "hamming_distance_at_success"
                ),
                minimum=0,
                maximum=KEY_WIDTH_BITS,
            )
        )

        if (
            distance_at_success
            != success_row[
                "hamming_distance"
            ]
        ):
            raise KeyRecoveryError(
                "neighbour_sweep hamming_distance_at_success "
                "does not match success row"
            )

        anchor_guess = _byte(
            raw.get(
                "anchor_key_guess_at_success"
            ),
            field=(
                "report.enumeration."
                "anchor_key_guess_at_success"
            ),
        )

        if (
            anchor_guess
            != success_row[
                "anchor_key_guess"
            ]
        ):
            raise KeyRecoveryError(
                "neighbour_sweep anchor_key_guess_at_success "
                "does not match success row"
            )

        anchor_rank = _report_int(
            raw.get(
                "anchor_ai_rank_at_success"
            ),
            field=(
                "report.enumeration."
                "anchor_ai_rank_at_success"
            ),
            minimum=1,
            maximum=KEYSPACE_SIZE,
        )

        if (
            anchor_rank
            != success_row[
                "anchor_ai_rank"
            ]
        ):
            raise KeyRecoveryError(
                "neighbour_sweep anchor_ai_rank_at_success "
                "does not match success row"
            )

    else:
        expected_fallback_used = any(
            candidate[
                "enumeration_phase"
            ] == "ai_fallback"
            for candidate in expected_order[
                :common[
                    "candidates_evaluated"
                ]
            ]
        )

        fallback_used = _report_bool(
            raw.get(
                "fallback_used"
            ),
            field=(
                "report.enumeration.fallback_used"
            ),
        )

        if (
            fallback_used
            != expected_fallback_used
        ):
            raise KeyRecoveryError(
                "Support fallback_used does not "
                "match evaluated phases"
            )

        if success_phase == "support_promoted":
            support_at_success = _finite_float(
                raw.get(
                    "weighted_hamming_support_at_success"
                ),
                field=(
                    "report.enumeration."
                    "weighted_hamming_support_at_success"
                ),
            )

            if not math.isclose(
                support_at_success,
                success_row[
                    "weighted_hamming_support"
                ],
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise KeyRecoveryError(
                    "weighted_hamming_support_at_success "
                    "does not match success row"
                )

            supporting_count = (
                _report_int(
                    raw.get(
                        "supporting_anchor_count_at_success"
                    ),
                    field=(
                        "report.enumeration."
                        "supporting_anchor_count_at_success"
                    ),
                    minimum=1,
                    maximum=KEYSPACE_SIZE,
                )
            )

            if (
                supporting_count
                != success_row[
                    "supporting_anchor_count"
                ]
            ):
                raise KeyRecoveryError(
                    "supporting_anchor_count_at_success "
                    "does not match success row"
                )

            minimum_distance = (
                _report_int(
                    raw.get(
                        "minimum_anchor_distance_at_success"
                    ),
                    field=(
                        "report.enumeration."
                        "minimum_anchor_distance_at_success"
                    ),
                    minimum=1,
                    maximum=KEY_WIDTH_BITS,
                )
            )

            if (
                minimum_distance
                != success_row[
                    "minimum_anchor_distance"
                ]
            ):
                raise KeyRecoveryError(
                    "minimum_anchor_distance_at_success "
                    "does not match success row"
                )

        else:
            for field in (
                "weighted_hamming_support_at_success",
                "supporting_anchor_count_at_success",
                "minimum_anchor_distance_at_success",
            ):
                if raw.get(field) is not None:
                    raise KeyRecoveryError(
                        f"report.enumeration.{field} must be null "
                        "when success occurs in AI fallback"
                    )

    return _to_builtin(
        dict(raw)
    )


def _validate_trace_population(population: Mapping[str, Any]) -> None:
    """Validate a contiguous attack split or an explicitly indexed CV fold."""
    source = population.get("source")
    available = _report_int(population.get("available"), field="trace_population.available",
                            minimum=1, maximum=2**63 - 1)
    if source == "cross_validation_runtime.validation_indices":
        if population.get("range") is not None:
            raise KeyRecoveryError("Indexed CV population must have range=null")
        provenance = population.get("cv_provenance")
        if not isinstance(provenance, Mapping):
            raise KeyRecoveryError("Indexed CV population requires cv_provenance")
        if (provenance.get("source_group") != "Profiling_traces" or
                provenance.get("index_coordinate_system") != "configuration_specific_profiling_rows"):
            raise KeyRecoveryError("CV population must identify generated profiling rows")
        count = _report_int(provenance.get("profiling_count"), field="cv.profiling_count",
                            minimum=2, maximum=2**63 - 1)
        folds = _report_int(provenance.get("n_folds"), field="cv.n_folds",
                            minimum=2, maximum=count)
        _report_int(provenance.get("fold_index"), field="cv.fold_index",
                    minimum=0, maximum=folds - 1)
        indices = population.get("validation_indices")
        if not isinstance(indices, (list, tuple)) or len(indices) != available:
            raise KeyRecoveryError("CV validation indices must match the evaluated trace count")
        rows = [_report_int(row, field="cv.validation_indices", minimum=0, maximum=count - 1)
                for row in indices]
        if len(set(rows)) != len(rows):
            raise KeyRecoveryError("CV validation indices contain duplicate rows")
        # Same ordered-index encoding as perform_attack._indices_fingerprint.
        digest = hashlib.sha256()
        digest.update(b"TraceFlow repeated attack indices v2\0")
        digest.update(b"ordered\0")
        digest.update(len(rows).to_bytes(8, "little", signed=False))
        for row in rows:
            digest.update(row.to_bytes(8, "little", signed=True))
        if provenance.get("validation_indices_sha256") != digest.hexdigest():
            raise KeyRecoveryError("CV validation index fingerprint does not match the report")
        return
    if not isinstance(source, str) or not source.strip():
        raise KeyRecoveryError("trace_population.source is required")
    span = population.get("range")
    if not isinstance(span, Sequence) or isinstance(span, (str, bytes, bytearray)) or len(span) != 2:
        raise KeyRecoveryError("trace_population.range must be [start, end] for a non-CV split")
    start = _report_int(span[0], field="trace_population.range[0]", minimum=0, maximum=2**63 - 1)
    end = _report_int(span[1], field="trace_population.range[1]", minimum=1, maximum=2**63 - 1)
    if end - start != available:
        raise KeyRecoveryError("trace_population.range length must equal available")


def trained_settings(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Key-recovery-relevant settings of the run that produced the model.

    Built from the run's own resolved config (the training snapshot for every
    test path: make test, orchestrated test, rerun), so later edits to
    params.yaml never change what a model's report says it was trained with.
    """
    cfg = cfg or {}
    sca = cfg.get("sca") or {}
    enumeration = sca.get("key_enumeration") or {}
    enumeration_on = isinstance(enumeration, Mapping) and enumeration.get("enabled") is True
    cv = cfg.get("cross_validation") or {}
    cv_on = isinstance(cv, Mapping) and cv.get("enabled") is True
    attack = cfg.get("perform_attack") or {}
    reporting = cfg.get("key_recovery_reporting") or {}
    data = cfg.get("data") or {}
    if not isinstance(data, Mapping) or "name" not in data:
        data = {}
    paths = cfg.get("paths") or {}
    return {
        "source": str(cfg.get("replay_snapshot_path") or paths.get("params_out") or "run config"),
        "key_enumeration": (
            {
                "enabled": True,
                "mode": enumeration.get("mode"),
                "stop_on_success": enumeration.get("stop_on_success"),
                "modes": enumeration.get("modes"),
            }
            if enumeration_on
            else {"enabled": False}
        ),
        "candidate_ranking": sca.get("candidate_ranking"),
        "perform_attack": {
            k: attack.get(k)
            for k in ("validation", "test", "attack_step", "ge_thresholds")
        },
        "key_recovery": {
            k: reporting.get(k) for k in ("recovery_rank", "require_stable_recovery")
        },
        "cross_validation": (
            {"enabled": True, "n_folds": cv.get("n_folds"), "strategy": cv.get("strategy")}
            if cv_on
            else {"enabled": False}
        ),
        "checkpoint_policy": ((cfg.get("evaluation") or {}).get("test") or {}).get("load_model"),
        "data": {
            k: data.get(k)
            for k in ("name", "database", "test_range", "validation_range", "attack_desync")
        },
    }


def _trained_enumeration_enabled(*reports: Optional[Mapping[str, Any]]) -> Optional[bool]:
    """Enumeration switch recorded in a report's trained_settings, if any."""
    for report in reports:
        settings = (report or {}).get("trained_settings") if isinstance(report, Mapping) else None
        if isinstance(settings, Mapping):
            return bool((settings.get("key_enumeration") or {}).get("enabled"))
    return None


def _trained_settings_lines(report: Optional[Mapping[str, Any]]) -> list[str]:
    """Compact 'trained with' block for the terminal summary."""
    settings = (report or {}).get("trained_settings") if isinstance(report, Mapping) else None
    if not isinstance(settings, Mapping):
        return []
    enumeration = settings.get("key_enumeration") or {}
    cv = settings.get("cross_validation") or {}
    attack = settings.get("perform_attack") or {}
    recovery = settings.get("key_recovery") or {}
    ranking = settings.get("candidate_ranking") or {}
    data = settings.get("data") or {}
    return [
        "TRAINED SETTINGS (from the model's training snapshot)",
        "-" * 80,
        _line("Key enumeration", str(enumeration.get("mode")).upper() if enumeration.get("enabled") else "OFF"),
        _line("Candidate ranking", ranking.get("mode", "n/a") if isinstance(ranking, Mapping) else ranking),
        _line("Cross-validation", f"ON ({cv.get('n_folds')} folds)" if cv.get("enabled") else "OFF"),
        _line(
            "Attack",
            "; ".join(
                f"{process} {(attack.get(process) or {}).get('attack_times')}x "
                f"{(attack.get(process) or {}).get('attack_distribution')} "
                f"shuffle {(attack.get(process) or {}).get('shuffle')} "
                f"budget {(attack.get(process) or {}).get('attack_trace_budget') or 'all traces'}"
                for process in ("validation", "test")
            )
            + f"; step {attack.get('attack_step')}",
        ),
        _line(
            "Recovery criterion",
            f"rank <= {recovery.get('recovery_rank')}, stable={recovery.get('require_stable_recovery')}",
        ),
        _line("Checkpoint policy", settings.get("checkpoint_policy")),
        _line("Dataset", f"{data.get('name')} test_range={data.get('test_range')}"),
        "",
    ]


def build_key_recovery_report(
    *,
    cfg: Mapping[str, Any],
    process: str,
    evidence: Mapping[str, Any],
    epoch: Optional[int] = None,
    model_name: Optional[str] = None,
    model_hash: Optional[str] = None,
    source_checkpoint: Optional[PathLike] = None,
    source_checkpoint_epoch: Optional[int] = None,
    source_checkpoint_sha256: Optional[str] = None,
    selection_metric: Optional[str] = None,
    selection_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Build one validated validation/test key-recovery report."""
    contract = validate_key_recovery_contract(
        cfg
    )

    if not contract[
        "enabled"
    ]:
        raise KeyRecoveryError(
            "Cannot build report while "
            "key_recovery_reporting.enabled=false"
        )

    normalized_process = str(
        process
    ).strip().lower()

    if normalized_process not in {
        "validation",
        "test",
    }:
        raise KeyRecoveryError(
            "process must be 'validation' or 'test'"
        )

    if normalized_process == "validation":
        if (
            epoch is None
            or isinstance(
                epoch,
                bool,
            )
            or int(epoch) < 1
        ):
            raise KeyRecoveryError(
                "validation reports require epoch >= 1"
            )

        epoch = int(
            epoch
        )

    elif epoch is not None:
        epoch = int(
            epoch
        )

    true_key = _byte(
        evidence.get(
            "true_key_byte"
        ),
        field="true_key_byte",
    )

    top_guess = _byte(
        evidence.get(
            "top_key_hypothesis"
        ),
        field="top_key_hypothesis",
    )

    final_rank = _rank(
        evidence.get(
            "final_true_key_rank"
        ),
        field="final_true_key_rank",
    )

    final_ranks = _as_rank_list(
        evidence.get(
            "final_ranks"
        )
    )

    repetitions_raw = evidence.get(
        "trace_order_repetitions",
        len(final_ranks),
    )

    if isinstance(
        repetitions_raw,
        bool,
    ):
        raise KeyRecoveryError(
            "trace_order_repetitions must be integer"
        )

    repetitions = int(
        repetitions_raw
    )

    if repetitions < 1:
        raise KeyRecoveryError(
            "trace_order_repetitions must be >= 1"
        )

    if repetitions != len(
        final_ranks
    ):
        raise KeyRecoveryError(
            "trace_order_repetitions must equal "
            "len(final_ranks)"
        )

    # Disjoint attacks end on independent subsets, and attacks with a budget below
    # N on overlapping ones, so their ranks differ. Only whole-population attacks
    # share one order-invariant endpoint (a record made before attack_trace_budget
    # has the field too, or is whole-population exactly when not disjoint).
    attack_protocol = evidence.get("attack_protocol") or {}
    disjoint = attack_protocol.get("attack_distribution") == "disjoint"
    invariant = bool(attack_protocol.get("endpoint_rank_order_invariant", not disjoint))
    if invariant and any(rank != final_rank for rank in final_ranks):
        raise KeyRecoveryError(
            "Full-population final ranks must be order invariant and equal "
            "final_true_key_rank"
        )

    traces_available = (
        _optional_positive_int(
            evidence.get(
                "traces_available"
            ),
            field="traces_available",
        )
    )

    traces_used = (
        _optional_positive_int(
            evidence.get(
                "traces_used"
            ),
            field="traces_used",
        )
    )

    if (
        traces_available is None
        or traces_used is None
    ):
        raise KeyRecoveryError(
            "traces_available and traces_used are required"
        )

    if traces_used != traces_available:
        raise KeyRecoveryError(
            "Full-population evaluation requires "
            "traces_used == traces_available"
        )

    population_source = str(
        evidence.get("trace_population_source", "")
    ).strip()
    if not population_source:
        raise KeyRecoveryError("trace_population_source is required")

    trace_population = {
        "source": population_source,
        "range": evidence.get("trace_population_range"),
        "available": traces_available,
        "used": traces_used,
        "full_population_used": True,
    }
    if population_source == "cross_validation_runtime.validation_indices":
        runtime = cfg.get("cross_validation_runtime")
        if (normalized_process != "validation" or not isinstance(runtime, Mapping)
                or runtime.get("active") is not True or runtime.get("is_fold") is not True
                or runtime.get("is_refit", False) is not False):
            raise KeyRecoveryError("Indexed CV recovery reporting requires an active validation fold")
        provenance = evidence.get("cv_provenance")
        if not isinstance(provenance, Mapping):
            raise KeyRecoveryError("CV attack evidence is missing cv_provenance")
        for field in ("fold_index", "n_folds", "profiling_count"):
            if provenance.get(field) != runtime.get(field):
                raise KeyRecoveryError(f"CV evidence disagrees with runtime {field}")
        trace_population["cv_provenance"] = dict(provenance)
        indices = runtime.get("validation_indices")
        if not isinstance(indices, (list, tuple)):
            raise KeyRecoveryError("CV runtime validation_indices must be a resolved sequence")
        trace_population["validation_indices"] = list(indices)
    _validate_trace_population(trace_population)
    if evidence.get("full_population_used") is not True:
        raise KeyRecoveryError("full_population_used must be true")

    first_rank1 = _optional_positive_int(
        evidence.get(
            "first_rank1_trace"
        ),
        field="first_rank1_trace",
    )

    stable_rank1 = _optional_positive_int(
        evidence.get(
            "stable_rank1_trace"
        ),
        field="stable_rank1_trace",
    )

    for field, value in (
        (
            "first_rank1_trace",
            first_rank1,
        ),
        (
            "stable_rank1_trace",
            stable_rank1,
        ),
    ):
        if (
            value is not None
            and value > traces_used
        ):
            raise KeyRecoveryError(
                f"{field} cannot exceed traces_used"
            )

    if (
        first_rank1 is not None
        and stable_rank1 is not None
        and stable_rank1 < first_rank1
    ):
        raise KeyRecoveryError(
            "stable_rank1_trace cannot occur "
            "before first_rank1_trace"
        )

    recovery_rank = contract[
        "recovery_rank"
    ]

    # These are attack outputs, not reporting defaults. Requiring and checking
    # them prevents this layer from concealing a missing GE/SR computation.
    if "final_ge" not in evidence:
        raise KeyRecoveryError("final_ge is required from the attack evaluator")
    if "final_sr_at_1" not in evidence:
        raise KeyRecoveryError(
            "final_sr_at_1 is required from the attack evaluator"
        )

    final_ge = _finite_float(evidence["final_ge"], field="final_ge")
    final_sr = _finite_float(
        evidence["final_sr_at_1"], field="final_sr_at_1"
    )
    expected_final_ge = sum(final_ranks) / repetitions
    expected_final_sr = (
        sum(rank <= recovery_rank for rank in final_ranks) / repetitions
    )
    if not math.isclose(
        final_ge, expected_final_ge, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise KeyRecoveryError(
            "final_ge does not equal the mean of final_ranks supplied by "
            "the attack evaluator"
        )
    if not math.isclose(
        final_sr, expected_final_sr, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise KeyRecoveryError(
            "final_sr_at_1 does not equal the rank-1 success fraction "
            "derived from final_ranks"
        )
    if invariant and not math.isclose(final_ge, float(final_rank), rel_tol=0.0, abs_tol=0.0):
        raise KeyRecoveryError(
            "Full-population final GE must equal final_true_key_rank"
        )
    expected_endpoint_sr = 1.0 if final_rank <= recovery_rank else 0.0
    if invariant and not math.isclose(
        final_sr, expected_endpoint_sr, rel_tol=0.0, abs_tol=0.0
    ):
        raise KeyRecoveryError(
            "Full-population final SR@1 must be exactly 0 or 1"
        )

    final_rank_recovered = (
        final_rank
        <= recovery_rank
    )

    hypothesis_matches = (
        top_guess
        == true_key
    )

    if (
        final_rank_recovered
        != hypothesis_matches
    ):
        raise KeyRecoveryError(
            "Inconsistent attack evidence between "
            "final rank and top key hypothesis"
        )

    # A true key tied with other keys holds rank 1 only by the tie-break draw;
    # the attack has not singled it out, so it is never recovered.
    tie_count = _optional_positive_int(
        evidence.get("true_key_tie_count"), field="true_key_tie_count"
    )
    if tie_count is None:
        raise KeyRecoveryError("true_key_tie_count is required from the attack evaluator")
    final_rank_unique = tie_count == 1
    recovered = final_rank_recovered and final_rank_unique
    if contract[
        "require_stable_recovery"
    ]:
        recovered = (
            recovered
            and stable_rank1
            is not None
        )

    sca_cfg = (
        cfg.get("sca")
        or {}
    )

    attack_byte = _byte(
        sca_cfg.get("attack_byte", evidence.get("attack_byte")),
        field="sca.attack_byte",
    )
    if "attack_byte" in evidence and _byte(
        evidence["attack_byte"], field="evidence.attack_byte"
    ) != attack_byte:
        raise KeyRecoveryError(
            "evidence.attack_byte does not match resolved sca.attack_byte"
        )
    leakage_model = _resolved_text_scalar(
        sca_cfg.get("leakage_model", evidence.get("leakage_model")),
        field="sca.leakage_model",
    ).upper()
    if leakage_model not in {"ID", "HW", "MSB"}:
        raise KeyRecoveryError("sca.leakage_model must be ID, HW, or MSB")
    if "leakage_model" in evidence and _resolved_text_scalar(
        evidence["leakage_model"], field="evidence.leakage_model"
    ).upper() != leakage_model:
        raise KeyRecoveryError(
            "evidence.leakage_model does not match resolved sca.leakage_model"
        )
    key_mode = _resolved_text_scalar(
        resolved_variant(cfg) or evidence.get("key_mode"),   # the database type's key mode
        field="sca.variant",
    ).lower()
    if key_mode not in {"fixed_key", "variable_key"}:
        raise KeyRecoveryError(
            "sca.variant must be 'fixed_key' or 'variable_key'"
        )
    if "key_mode" in evidence and _resolved_text_scalar(
        evidence["key_mode"], field="evidence.key_mode"
    ).lower() != key_mode:
        raise KeyRecoveryError(
            "evidence.key_mode does not match resolved sca.variant"
        )
    cipher = _resolved_text_scalar(
        evidence.get("cipher", "AES"), field="cipher"
    ).upper()
    if cipher != "AES":
        raise KeyRecoveryError("key-recovery reporting currently requires AES")

    checkpoint_path: Optional[str] = None
    checkpoint_epoch: Optional[int] = None
    checkpoint_sha256: Optional[str] = None
    if source_checkpoint is not None:
        checkpoint_path = str(source_checkpoint).strip()
        if not checkpoint_path:
            raise KeyRecoveryError("source_checkpoint must not be empty")
    if source_checkpoint_epoch is not None:
        checkpoint_epoch = _optional_positive_int(
            source_checkpoint_epoch, field="source_checkpoint_epoch"
        )
    if source_checkpoint_sha256 is not None:
        checkpoint_sha256 = _validate_sha256_hex(
            source_checkpoint_sha256, field="source_checkpoint_sha256"
        )
    if normalized_process == "test" and (
        checkpoint_path is None
        or checkpoint_epoch is None
        or checkpoint_sha256 is None
    ):
        raise KeyRecoveryError(
            "test reports require source_checkpoint, "
            "source_checkpoint_epoch, and source_checkpoint_sha256 provenance"
        )

    score_rule = evidence.get("score_rule")
    if score_rule not in SCORE_RULES:
        raise KeyRecoveryError(f"score_rule must be one of {sorted(SCORE_RULES)}, got {score_rule!r}")
    target = {
        "cipher": cipher,
        "attack_byte": attack_byte,
        "key_mode": key_mode,
        "leakage_model": leakage_model,
        # HW/MSB scores subtract the log class prior (rank_curve.py).
        "score_rule": score_rule,
        "keyspace_size": (
            KEYSPACE_SIZE
        ),
    }

    top_candidates = _candidate_rows(
        evidence.get(
            "top_key_candidates"
        ),
        mode=contract[
            "candidate_ranking_mode"
        ],
        top_k=contract[
            "candidate_ranking_top_k"
        ],
    )

    score_margin: Optional[
        float
    ] = None

    if contract[
        "include_score_margin"
    ]:
        raw_margin = evidence.get(
            "top1_top2_score_margin"
        )

        if raw_margin is not None:
            score_margin = _finite_float(
                raw_margin,
                field=(
                    "top1_top2_score_margin"
                ),
            )

    report: Dict[
        str,
        Any,
    ] = {
        "schema_version": SCHEMA_VERSION,
        "rank_contract_version": (
            RANK_CONTRACT_VERSION
        ),
        "rank_base": PUBLIC_RANK_BASE,
        "process": normalized_process,
        "epoch": epoch,
        "model": {
            "name": (
                model_name
                or cfg.get(
                    "model_name"
                )
            ),
            "hash": (
                model_hash
                or cfg.get(
                    "model_hash"
                )
            ),
            "source_checkpoint": checkpoint_path,
            "source_checkpoint_epoch": checkpoint_epoch,
            "source_checkpoint_sha256": checkpoint_sha256,
        },
        "selection": {
            "metric": (
                selection_metric
            ),
            "score": (
                _finite_float(
                    selection_score,
                    field=(
                        "selection_score"
                    ),
                )
                if selection_score
                is not None
                else None
            ),
        },
        "target": target,
        "key": {
            "true_key_byte": (
                true_key
            ),
            "true_key_hex": (
                f"0x{true_key:02X}"
            ),
            "top_key_hypothesis": (
                top_guess
            ),
            "top_key_hypothesis_hex": (
                f"0x{top_guess:02X}"
            ),
            "final_true_key_rank": (
                final_rank
            ),
            "recovered": bool(
                recovered
            ),
            "final_rank_recovered": bool(
                final_rank_recovered
            ),
            "true_key_tie_count": tie_count,
            "tie_break_policy": evidence.get("tie_break_policy"),
            "stable_recovery_required": bool(
                contract[
                    "require_stable_recovery"
                ]
            ),
        },
        "trace_population": trace_population,
        "recovery": {
            "recovery_rank": (
                recovery_rank
            ),
            "first_rank1_trace": (
                first_rank1
            ),
            "stable_rank1_trace": (
                stable_rank1
            ),
            "trace_order_repetitions": repetitions,
            "attack_distribution": "disjoint" if disjoint else "full_population",
            "attack_trace_budget": attack_protocol.get("attack_trace_budget"),   # None: every trace
            "endpoint_rank_order_invariant": invariant,
            "final_ranks": final_ranks,
            "final_sr_at_1": (
                final_sr
            ),
            "final_ge": (
                final_ge
            ),
            "final_rank_min": min(
                final_ranks
            ),
            "final_rank_mean": (
                final_ge
            ),
            "final_rank_max": max(
                final_ranks
            ),
        },
        "diagnostics": {
            "top_key_candidates": (
                top_candidates
            ),
            "top1_top2_score_margin": (
                score_margin
            ),
        },
        "trained_settings": trained_settings(cfg),
    }

    built_report = _to_builtin(report)
    _validate_base_key_recovery_report(built_report)
    validate_key_enumeration_report_section(built_report)
    return built_report


def resolve_key_recovery_dir(
    cfg: Mapping[str, Any],
    *,
    model_name: Optional[str] = None,
) -> Path:
    """Resolve/create key-recovery artifact directory."""
    paths = (
        cfg.get(
            "paths"
        )
        or {}
    )

    if not isinstance(
        paths,
        Mapping,
    ):
        raise KeyRecoveryError(
            "cfg.paths must be a mapping"
        )

    resolved_model_name = (
        model_name
        or cfg.get(
            "model_name"
        )
    )

    if not resolved_model_name:
        raise KeyRecoveryError(
            "model_name is required to resolve "
            "key-recovery paths"
        )

    if paths.get(
        "key_recovery"
    ):
        directory = Path(
            resolve_path(
                paths[
                    "key_recovery"
                ],
                cfg,
                model_name=(
                    resolved_model_name
                ),
            )
        )

    elif paths.get(
        "metrics"
    ):
        directory = Path(
            resolve_path(
                paths[
                    "metrics"
                ],
                cfg,
                model_name=(
                    resolved_model_name
                ),
            )
        ) / "key_recovery"

    else:
        raise KeyRecoveryError(
            "cfg.paths must define key_recovery or metrics"
        )

    _ensure_key_recovery_directory(
        directory
    )

    return directory


def key_recovery_paths(
    cfg: Mapping[str, Any],
    *,
    model_name: Optional[str] = None,
) -> Dict[str, Path]:
    """Return canonical recovery artifact paths."""
    directory = (
        resolve_key_recovery_dir(
            cfg,
            model_name=model_name,
        )
    )

    return {
        "directory": directory,
        "validation_history": (
            directory
            / "validation_history.jsonl"
        ),
        "validation_best": (
            directory
            / "validation_best.json"
        ),
        "test": (
            directory
            / "test.json"
        ),
    }


def _ensure_key_recovery_directory(directory: Path) -> None:
    """Create the key-recovery artifact directory when it is absent."""
    directory.mkdir(parents=True, exist_ok=True)


def _open_atomic_temp(path: Path) -> tuple[TextIO, Path]:
    """Create an exclusive same-directory temporary text file."""
    for _ in range(100):
        temp_path = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
        try:
            handle = temp_path.open(
                "x",
                encoding="utf-8",
            )
        except FileExistsError:
            continue
        return handle, temp_path

    raise FileExistsError(
        f"Unable to allocate temporary key-recovery artifact beside {path}"
    )




def _atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> Path:
    """
    Atomically write one JSON artifact.
    """
    _ensure_key_recovery_directory(
        path.parent
    )

    serialized = json.dumps(
        _to_builtin(
            payload
        ),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    serialized = seal_text(path, serialized)  # utils/seal.py

    handle, temp_path = _open_atomic_temp(
        path
    )

    try:
        with handle:
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

        os.replace(
            temp_path,
            path,
        )

    except Exception:
        temp_path.unlink(
            missing_ok=True
        )
        raise

    return path

def _atomic_write_jsonl(
    path: Path,
    line: str,
) -> Path:
    """Create one JSONL artifact atomically; every row is sealed (utils/seal.py)."""
    _ensure_key_recovery_directory(
        path.parent
    )
    line = seal_text(path, line)

    handle, temp_path = _open_atomic_temp(
        path
    )

    try:
        with handle:
            handle.write(
                line
            )
            handle.write(
                "\n"
            )
            handle.flush()

            os.fsync(
                handle.fileno()
            )

        os.replace(
            temp_path,
            path,
        )

    except Exception:
        temp_path.unlink(
            missing_ok=True
        )
        raise

    return path

def _append_jsonl(
    path: Path,
    payload: Mapping[str, Any],
) -> Path:
    """
    Write one validation-history entry, replacing instead of appending.

    Entries at or after this entry's epoch come from an earlier run of the
    same model and are dropped first (one entry per epoch of the latest
    run). The file is replaced atomically.
    """
    _ensure_key_recovery_directory(
        path.parent
    )

    line = json.dumps(
        _to_builtin(
            payload
        ),
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )

    epoch = payload.get("epoch")
    kept = []
    if path.exists():
        check_seal(path, cache=False)  # carried rows are re-signed: a changed row stops here
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                row_epoch = json.loads(raw).get("epoch")
            except (ValueError, AttributeError):
                continue
            if isinstance(epoch, int) and isinstance(row_epoch, int) and row_epoch >= epoch:
                continue
            kept.append(raw)
    kept.append(line)

    return _atomic_write_jsonl(path, "\n".join(kept))
def save_validation_report(
    *,
    cfg: Mapping[str, Any],
    report: Mapping[str, Any],
    is_best: bool,
    model_name: Optional[str] = None,
) -> Dict[str, Optional[Path]]:
    """Append validation history and optionally replace best report."""
    contract = (
        validate_key_recovery_contract(
            cfg
        )
    )

    if (
        report.get(
            "process"
        )
        != "validation"
    ):
        raise KeyRecoveryError(
            "save_validation_report requires "
            "a validation report"
        )

    _validate_base_key_recovery_report(report)
    validate_key_enumeration_report_section(report)

    result: Dict[
        str,
        Optional[Path],
    ] = {
        "validation_history": None,
        "validation_best": None,
    }

    if not contract[
        "save_validation_reports"
    ]:
        return result

    paths = key_recovery_paths(
        cfg,
        model_name=model_name,
    )

    result[
        "validation_history"
    ] = _append_jsonl(
        paths[
            "validation_history"
        ],
        report,
    )

    if is_best:
        result[
            "validation_best"
        ] = _atomic_write_json(
            paths[
                "validation_best"
            ],
            report,
        )

    return result


def save_test_report(
    *,
    cfg: Mapping[str, Any],
    report: Mapping[str, Any],
    model_name: Optional[str] = None,
) -> Optional[Path]:
    """Atomically replace one-shot test report."""
    contract = (
        validate_key_recovery_contract(
            cfg
        )
    )

    if (
        report.get(
            "process"
        )
        != "test"
    ):
        raise KeyRecoveryError(
            "save_test_report requires a test report"
        )

    _validate_base_key_recovery_report(report)
    validate_key_enumeration_report_section(report)

    if not contract[
        "save_test_report"
    ]:
        return None

    paths = key_recovery_paths(
        cfg,
        model_name=model_name,
    )

    test_report_path = _atomic_write_json(
        paths["test"],
        report,
    )

    return test_report_path


def _validate_base_key_recovery_report(report: Mapping[str, Any]) -> None:
    """Validate the canonical non-enumeration report contract."""
    if report.get("schema_version") != SCHEMA_VERSION:
        raise KeyRecoveryError(
            f"Unsupported key-recovery schema_version={report.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if report.get("rank_base") != PUBLIC_RANK_BASE:
        raise KeyRecoveryError("Persisted key-recovery rank_base must be 1")
    if report.get("rank_contract_version") != RANK_CONTRACT_VERSION:
        raise KeyRecoveryError(
            "Persisted rank_contract_version does not match this implementation"
        )
    if report.get("process") not in {"validation", "test"}:
        raise KeyRecoveryError("Persisted process must be validation or test")

    target = report.get("target")
    key = report.get("key")
    model = report.get("model")
    population = report.get("trace_population")
    recovery = report.get("recovery")
    diagnostics = report.get("diagnostics")
    for name, section in (
        ("target", target),
        ("key", key),
        ("model", model),
        ("trace_population", population),
        ("recovery", recovery),
        ("diagnostics", diagnostics),
    ):
        if not isinstance(section, Mapping):
            raise KeyRecoveryError(f"Persisted report.{name} must be a mapping")

    process = report["process"]
    if process == "validation":
        _report_int(
            report.get("epoch"), field="report.epoch",
            minimum=1, maximum=2**31 - 1,
        )
    else:
        checkpoint = _resolved_text_scalar(
            model.get("source_checkpoint"),
            field="report.model.source_checkpoint",
        )
        if not checkpoint:
            raise KeyRecoveryError("Test checkpoint path is required")
        _report_int(
            model.get("source_checkpoint_epoch"),
            field="report.model.source_checkpoint_epoch",
            minimum=1, maximum=2**31 - 1,
        )
        _validate_sha256_hex(
            model.get("source_checkpoint_sha256"),
            field="report.model.source_checkpoint_sha256",
        )

    cipher = _resolved_text_scalar(
        target.get("cipher"), field="report.target.cipher"
    ).upper()
    if cipher != "AES":
        raise KeyRecoveryError("Persisted report.target.cipher must be AES")
    _byte(target.get("attack_byte"), field="report.target.attack_byte")
    key_mode = _resolved_text_scalar(
        target.get("key_mode"), field="report.target.key_mode"
    ).lower()
    if key_mode not in {"fixed_key", "variable_key"}:
        raise KeyRecoveryError(
            "Persisted report.target.key_mode must be fixed_key or variable_key"
        )
    leakage_model = _resolved_text_scalar(
        target.get("leakage_model"), field="report.target.leakage_model"
    ).upper()
    if target.get("score_rule") not in SCORE_RULES:
        raise KeyRecoveryError(
            f"Persisted report.target.score_rule must be one of {sorted(SCORE_RULES)}"
        )
    if leakage_model not in {"ID", "HW", "MSB"}:
        raise KeyRecoveryError(
            "Persisted report.target.leakage_model must be ID, HW, or MSB"
        )

    keyspace_size = _report_int(
        target.get("keyspace_size"),
        field="report.target.keyspace_size",
        minimum=KEYSPACE_SIZE,
        maximum=KEYSPACE_SIZE,
    )
    final_rank = _rank(
        key.get("final_true_key_rank"),
        field="report.key.final_true_key_rank",
    )
    true_key = _byte(key.get("true_key_byte"), field="report.key.true_key_byte")
    top_guess = _byte(
        key.get("top_key_hypothesis"),
        field="report.key.top_key_hypothesis",
    )
    if key.get("true_key_hex") != f"0x{true_key:02X}":
        raise KeyRecoveryError("report.key.true_key_hex is inconsistent")
    if key.get("top_key_hypothesis_hex") != f"0x{top_guess:02X}":
        raise KeyRecoveryError(
            "report.key.top_key_hypothesis_hex is inconsistent"
        )
    if keyspace_size != KEYSPACE_SIZE:
        raise KeyRecoveryError("Persisted keyspace_size must be 256")

    available = _report_int(
        population.get("available"),
        field="report.trace_population.available",
        minimum=1,
        maximum=2**63 - 1,
    )
    used = _report_int(
        population.get("used"),
        field="report.trace_population.used",
        minimum=1,
        maximum=2**63 - 1,
    )
    if used != available or population.get("full_population_used") is not True:
        raise KeyRecoveryError(
            "Persisted report must use the complete evaluation population"
        )
    _validate_trace_population(population)
    if (population.get("source") == "cross_validation_runtime.validation_indices"
            and report.get("process") != "validation"):
        raise KeyRecoveryError("Indexed CV population is valid only for validation reports")

    repetitions = _report_int(
        recovery.get("trace_order_repetitions"),
        field="report.recovery.trace_order_repetitions",
        minimum=1,
        maximum=2**31 - 1,
    )
    if repetitions < 1:
        raise KeyRecoveryError("trace_order_repetitions must be positive")
    raw_final_ranks = recovery.get("final_ranks")
    if not isinstance(raw_final_ranks, list) or not raw_final_ranks:
        raise KeyRecoveryError(
            "report.recovery.final_ranks must be a non-empty persisted list"
        )
    final_ranks = [
        _report_int(
            value, field=f"report.recovery.final_ranks[{index}]",
            minimum=PUBLIC_RANK_BASE, maximum=KEYSPACE_SIZE,
        )
        for index, value in enumerate(raw_final_ranks)
    ]
    if len(final_ranks) != repetitions:
        raise KeyRecoveryError(
            "trace_order_repetitions must equal len(recovery.final_ranks)"
        )
    disjoint = recovery.get("attack_distribution") == "disjoint"
    invariant = bool(recovery.get("endpoint_rank_order_invariant", not disjoint))
    if invariant and any(rank != final_rank for rank in final_ranks):
        raise KeyRecoveryError(
            "Persisted full-population final ranks must be invariant and "
            "equal final_true_key_rank"
        )
    final_ge = _finite_float(
        recovery.get("final_ge"), field="report.recovery.final_ge"
    )
    final_sr = _finite_float(
        recovery.get("final_sr_at_1"), field="report.recovery.final_sr_at_1"
    )
    if not invariant:
        # Disjoint or budget attacks: GE and SR@1 are means over their final ranks.
        expected_ge = sum(final_ranks) / len(final_ranks)
        expected_sr = sum(rank == DEFAULT_RECOVERY_RANK for rank in final_ranks) / len(final_ranks)
        tolerance = 1e-12
    else:
        expected_ge = float(final_rank)
        expected_sr = 1.0 if final_rank == DEFAULT_RECOVERY_RANK else 0.0
        tolerance = 0.0
    if not math.isclose(final_ge, expected_ge, rel_tol=tolerance, abs_tol=tolerance):
        raise KeyRecoveryError("Final GE is inconsistent with the final ranks")
    if not math.isclose(final_sr, expected_sr, rel_tol=tolerance, abs_tol=tolerance):
        raise KeyRecoveryError("Final SR@1 is inconsistent with the final ranks")

    for name, expected in (
        ("final_rank_min", min(final_ranks)),
        ("final_rank_mean", sum(final_ranks) / len(final_ranks)),
        ("final_rank_max", max(final_ranks)),
    ):
        observed = _finite_float(
            recovery.get(name), field=f"report.recovery.{name}"
        )
        if not math.isclose(
            observed, float(expected), rel_tol=0.0, abs_tol=0.0
        ):
            raise KeyRecoveryError(
                f"report.recovery.{name} is inconsistent with final_ranks"
            )

    recovery_rank = _report_int(
        recovery.get("recovery_rank"),
        field="report.recovery.recovery_rank",
        minimum=DEFAULT_RECOVERY_RANK, maximum=DEFAULT_RECOVERY_RANK,
    )
    first_rank1 = _optional_report_int(
        recovery.get("first_rank1_trace"),
        field="report.recovery.first_rank1_trace",
        minimum=1, maximum=used,
    )
    stable_rank1 = _optional_report_int(
        recovery.get("stable_rank1_trace"),
        field="report.recovery.stable_rank1_trace",
        minimum=1, maximum=used,
    )
    if (
        first_rank1 is not None
        and stable_rank1 is not None
        and stable_rank1 < first_rank1
    ):
        raise KeyRecoveryError(
            "report.recovery.stable_rank1_trace cannot precede first_rank1_trace"
        )

    final_rank_recovered = final_rank <= recovery_rank
    if final_rank_recovered != (top_guess == true_key):
        raise KeyRecoveryError(
            "Top hypothesis and final true-key rank are inconsistent"
        )
    if _report_bool(
        key.get("final_rank_recovered"),
        field="report.key.final_rank_recovered",
    ) != final_rank_recovered:
        raise KeyRecoveryError("report.key.final_rank_recovered is inconsistent")
    stable_required = _report_bool(
        key.get("stable_recovery_required"),
        field="report.key.stable_recovery_required",
    )
    tie_count = _report_int(
        key.get("true_key_tie_count"),
        field="report.key.true_key_tie_count",
        minimum=1, maximum=KEYSPACE_SIZE,
    )
    expected_recovered = final_rank_recovered and tie_count == 1 and (
        stable_rank1 is not None or not stable_required
    )
    if _report_bool(
        key.get("recovered"), field="report.key.recovered"
    ) != expected_recovered:
        raise KeyRecoveryError("report.key.recovered is inconsistent")

    candidates = diagnostics.get("top_key_candidates")
    if not isinstance(candidates, list):
        raise KeyRecoveryError(
            "report.diagnostics.top_key_candidates must be a list"
        )
    if candidates:
        normalized_candidates = _candidate_rows(
            candidates, mode="top_k", top_k=len(candidates)
        )
        if normalized_candidates[0]["key_guess"] != top_guess:
            raise KeyRecoveryError(
                "First diagnostic candidate must equal top_key_hypothesis"
            )
        scored = [candidate for candidate in normalized_candidates if "score" in candidate]
        if scored and len(scored) != len(normalized_candidates):
            raise KeyRecoveryError(
                "Candidate scores must be present for every candidate or none"
            )
        # Keys tied with the true key (rank_curve.tie_tolerance) may be out
        # of order by float64 rounding; the tolerance bounds 2**20 traces.
        scale = max([1.0] + [abs(item["score"]) for item in scored])
        rounding = 64 * 2**20 * 2.220446049250313e-16 * scale
        if scored and any(
            left["score"] < right["score"] - rounding
            for left, right in zip(scored, scored[1:])
        ):
            raise KeyRecoveryError("Diagnostic candidate scores must be descending")
        margin = diagnostics.get("top1_top2_score_margin")
        if margin is not None:
            if len(scored) < 2:
                raise KeyRecoveryError(
                    "top1_top2_score_margin requires two scored candidates"
                )
            observed_margin = _finite_float(
                margin, field="report.diagnostics.top1_top2_score_margin"
            )
            expected_margin = scored[0]["score"] - scored[1]["score"]
            if not math.isclose(
                observed_margin, expected_margin, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise KeyRecoveryError(
                    "top1_top2_score_margin is inconsistent with candidate scores"
                )


def load_key_recovery_report(
    path: PathLike,
) -> Dict[str, Any]:
    """Load and validate one persisted key-recovery report."""
    check_seal(path)  # utils/seal.py
    with Path(
        path
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = strip_seal(json.load(
            handle
        ))

    if not isinstance(
        payload,
        dict,
    ):
        raise KeyRecoveryError(
            f"Expected JSON object in {path}"
        )

    _validate_base_key_recovery_report(payload)
    validate_key_enumeration_report_section(
        payload
    )

    return payload


def load_best_validation_report(
    cfg: Mapping[str, Any],
    *,
    model_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load selected validation report when available."""
    path = key_recovery_paths(
        cfg,
        model_name=model_name,
    )[
        "validation_best"
    ]

    if not path.exists():
        return None

    return load_key_recovery_report(
        path
    )


def _cv_fold_line(
    fold_number: int,
    seed: Any,
    report: Optional[Mapping[str, Any]],
    *,
    key_display: str,
    problem: Optional[str] = None,
) -> str:
    """One-line final-epoch key-recovery result of one CV fold."""
    prefix = f"  fold {fold_number:02d} | seed {seed}"
    if report is None:
        return f"{prefix} | {problem or 'report unavailable'}"
    key = report["key"]
    recovery = report["recovery"]
    stable = recovery["stable_rank1_trace"]
    return (
        f"{prefix} | top {_display_key(key['top_key_hypothesis_hex'], key_display)}"
        f" | rank {key['final_true_key_rank']}/{report['target']['keyspace_size']}"
        f" | GE {recovery['final_ge']:.2f}"
        f" | SR@1 {recovery['final_sr_at_1']:.2f}"
        f" | stable rank-1 {'@' + str(stable) if stable is not None else 'not reached'}"
        f" /{report['trace_population']['used']}"
        f" | {'RECOVERED' if key['recovered'] else 'NOT RECOVERED'}"
    )


def _cv_headline_section(
    headline: Mapping[str, Any],
    *,
    key_display: str = "full",
) -> list[str]:
    """Render the CV aggregate headline plus one key-recovery line per fold."""
    def number(value: Any, fmt: str) -> str:
        return format(value, fmt) if isinstance(value, (int, float)) else "n/a"

    stable_trace = headline.get("traces_to_ge_1_mean_curve")
    # No mean curve means nothing was computed, which is not "not reached".
    curve_computed = headline.get("mean_curve_final_ge") is not None
    return [
        f"CROSS-VALIDATION (profiling only, {headline.get('n_folds')} folds, final epoch)",
        "-" * 80,
        _line("Epoch", headline.get("epoch")),
        _line(
            "Final GE (mean curve)",
            f"{number(headline.get('mean_curve_final_ge'), '.4f')} "
            f"at {headline.get('mean_curve_traces')} traces",
        ),
        _line(
            "Traces to GE<=1 (stable)",
            f"{stable_trace if stable_trace is not None else 'not reached' if curve_computed else 'not computed'} "
            "(mean curve, held to the end)",
        ),
        _line("Folds reaching GE<=1", headline.get("folds_reaching_ge_1")),
        _line(
            "Accuracy",
            f"{number(headline.get('accuracy_mean'), '.6f')} "
            f"+/- {number(headline.get('accuracy_std'), '.6f')}",
        ),
        _line(
            "Training time per fold",
            f"{number(headline.get('training_time_mean_s'), '.1f')}s "
            f"+/- {number(headline.get('training_time_std_s'), '.1f')}s",
        ),
        _line("Aggregate", headline.get("aggregate_path")),
    ] + (
        ["", "Per-fold key recovery (held-out part, final epoch):"]
        + [
            _cv_fold_line(
                item["fold_number"],
                item["seed"],
                item["report"],
                key_display=key_display,
                problem=item.get("problem"),
            )
            for item in headline.get("fold_reports") or []
        ]
        if headline.get("fold_reports")
        else []
    )


def _display_key(
    hex_value: str,
    mode: str,
) -> str:
    if mode == "full":
        return hex_value

    if mode == "masked":
        return "0x**"

    return "hidden"


def _traces_per_attack(population: Mapping[str, Any], recovery: Mapping[str, Any]) -> int:
    """Length of one attack: disjoint attacks share the population equally. With an
    attack_trace_budget on full_population attacks the crossings printed are the key
    evidence's attack on all the traces, so this is the population."""
    if recovery.get("attack_distribution") == "disjoint":
        if recovery.get("attack_trace_budget") is not None:
            return int(recovery["attack_trace_budget"])
        return population["used"] // recovery["trace_order_repetitions"]
    return population["used"]


def _fmt_trace(
    value: Optional[int],
    *,
    traces_used: int,
) -> str:
    if value is None:
        return (
            f"not reached within "
            f"{traces_used} traces"
        )

    return f"trace {value}"


def _line(
    label: str,
    value: Any,
) -> str:
    return (
        f"{label:<24}: {value}"
    )


def _enumeration_lines(
    report: Mapping[str, Any],
    *,
    expected_mode: Optional[str] = None,
    key_display: str = "full",
) -> list[str]:
    """Return mode-aware terminal enumeration summary."""
    enumeration = (
        validate_key_enumeration_report_section(
            report
        )
    )

    if enumeration is None:
        if expected_mode is None:
            return []

        return [
            _line(
                "Enumeration mode",
                expected_mode.upper(),
            ),
            _line(
                "Enumeration Recovery status",
                "UNAVAILABLE",
            ),
            _line(
                "Enumeration position",
                f"unavailable / {KEYSPACE_SIZE}",
            ),
        ]

    mode = str(
        enumeration[
            "mode"
        ]
    )
    if mode == "comparison":
        #
        # TEST may contain a lightweight comparison envelope only when
        # validation selected raw AI.
        #
        if (
            str(
                report.get(
                    "process",
                    "",
                )
            ).strip().lower()
            == "test"
        ):
            return [
                _line(
                    "Enumeration mode",
                    "COMPARISON",
                ),
                _line(
                    "Best Rank Selected",
                    "true",
                ),
                _line(
                    "Selected mode",
                    "AI",
                ),
                _line(
                    "Enumeration executed",
                    "no",
                ),
                _line(
                    "AI true-key rank",
                    (
                        f"{enumeration['ai_rank_at_success']} "
                        f"/ {KEYSPACE_SIZE}"
                    ),
                ),
                _line(
                    "Selected position",
                    (
                        f"{enumeration['enumeration_position']} "
                        f"/ {KEYSPACE_SIZE}"
                    ),
                ),
            ]

        results = enumeration[
            "results"
        ]

        selected_mode = enumeration[
            "selected_mode"
        ]

        def _comparison_row(
            mode_name: str,
            result: Mapping[str, Any],
        ) -> str:
            selected_marker = (
                "  <-- SELECTED"
                if mode_name == selected_mode
                else ""
            )

            return (
                f"{mode_name.upper():<12}"
                f"{result['ai_rank_at_success']:>12}"
                f"{result['enumeration_position']:>23}"
                f"{selected_marker}"
            )

        lines = [
            "",
            "KEY ENUMERATION COMPARISON",
            "",
            (
                f"{'MODE':<12}"
                f"{'AI RANK':>12}"
                f"{'ENUMERATION POSITION':>23}"
            ),
            "-" * 47,
            _comparison_row(
                "ai",
                results["ai"],
            ),
            _comparison_row(
                "local_hamming",
                results["local_hamming"],
            ),
            _comparison_row(
                "neighbour_sweep",
                results["neighbour_sweep"],
            ),
            _comparison_row(
                "support_weighted",
                results["support_weighted"],
            ),
            "",
            _line(
                "Enumeration mode",
                "COMPARISON",
            ),
            _line(
                "Selection split",
                "VALIDATION",
            ),
            _line(
                "Selection rule",
                "better rank, model wins ties",
            ),
            _line(
                "Selected mode",
                selected_mode.upper(),
            ),
            _line(
                "Selected position",
                (
                    f"{results[selected_mode]['enumeration_position']} "
                    f"/ {KEYSPACE_SIZE}"
                ),
            ),
        ]

        if selected_mode == "ai":
            lines.extend(
                [
                    _line(
                        "Selected policy",
                        "none",
                    ),
                ]
            )

        else:
            lines.extend(
                [
                    _line(
                        "Recovered key byte",
                        _display_key(
                            results[
                                selected_mode
                            ][
                                "recovered_key_hex"
                            ],
                            key_display,
                        ),
                    ),
                    _line(
                        "Selected policy",
                        enumeration[
                            "selected_policy_fingerprint"
                        ][:12],
                    ),
                ]
            )

        return lines

    position_delta = int(
        enumeration[
            "position_delta"
        ]
    )

    lines = [
        _line(
            "Enumeration mode",
            mode.upper(),
        ),
        _line(
            "Enumeration Recovery status",
            (
                "SUCCESS"
                if enumeration[
                    "success"
                ]
                else "FAILED"
            ),
        ),
        _line(
            "Recovered key byte",
            _display_key(
                enumeration[
                    "recovered_key_hex"
                ],
                key_display,
            ),
        ),
        _line(
            "AI true-key rank",
            (
                f"{enumeration['ai_rank_at_success']} "
                f"/ {KEYSPACE_SIZE}"
            ),
        ),
        _line(
            "Enumeration position",
            (
                f"{enumeration['enumeration_position']} "
                f"/ {KEYSPACE_SIZE}"
            ),
        ),
        _line(
            "Position delta",
            f"{position_delta:+d}",
        ),
        _line(
            "Success phase",
            str(
                enumeration[
                    "success_phase"
                ]
            ).replace(
                "_",
                " ",
            ).upper(),
        ),
        _line(
            "Candidates evaluated",
            (
                f"{enumeration['candidates_evaluated']} "
                f"/ {KEYSPACE_SIZE}"
            ),
        ),
        _line(
            "Policy source",
            str(
                enumeration[
                    "policy_source"
                ]
            ).upper(),
        ),
        _line(
            "Policy fingerprint",
            enumeration[
                "policy_fingerprint"
            ][:12],
        ),
    ]

    if mode == "local_hamming":
        lines.extend(
            [
                _line(
                    "Anchor count",
                    enumeration[
                        "top_k"
                    ],
                ),
                _line(
                    "Hamming radius",
                    enumeration[
                        "hamming_radius"
                    ],
                ),
                _line(
                    "Local budget",
                    enumeration[
                        "local_budget"
                    ],
                ),
                _line(
                    "Local candidates",
                    (
                        "available="
                        f"{enumeration['local_candidates_available']} | "
                        "promoted="
                        f"{enumeration['local_candidates_promoted']}"
                    ),
                ),
                _line(
                    "Hamming distance",
                    enumeration[
                        "hamming_distance_at_success"
                    ],
                ),
                _line(
                    "AI fallback used",
                    (
                        "yes"
                        if enumeration[
                            "fallback_used"
                        ]
                        else "no"
                    ),
                ),
            ]
        )

    elif mode == "neighbour_sweep":
        lines.extend(
            [
                _line(
                    "Hamming radius",
                    enumeration[
                        "hamming_radius"
                    ],
                ),
                _line(
                    "Success anchor",
                    enumeration.get(
                        "anchor_key_guess_hex_at_success"
                    ),
                ),
                _line(
                    "Anchor AI rank",
                    enumeration.get(
                        "anchor_ai_rank_at_success"
                    ),
                ),
                _line(
                    "Hamming distance",
                    enumeration[
                        "hamming_distance_at_success"
                    ],
                ),
            ]
        )

    else:
        lines.extend(
            [
                _line(
                    "Anchor count",
                    enumeration[
                        "top_k"
                    ],
                ),
                _line(
                    "Hamming radius",
                    enumeration[
                        "hamming_radius"
                    ],
                ),
                _line(
                    "Local budget",
                    enumeration[
                        "local_budget"
                    ],
                ),
                _line(
                    "Supported candidates",
                    (
                        "available="
                        f"{enumeration['supported_candidates_available']} | "
                        "promoted="
                        f"{enumeration['supported_candidates_promoted']}"
                    ),
                ),
                _line(
                    "AI fallback used",
                    (
                        "yes"
                        if enumeration[
                            "fallback_used"
                        ]
                        else "no"
                    ),
                ),
            ]
        )

        if (
            enumeration[
                "success_phase"
            ]
            == "support_promoted"
        ):
            lines.extend(
                [
                    _line(
                        "Weighted support",
                        enumeration[
                            "weighted_hamming_support_at_success"
                        ],
                    ),
                    _line(
                        "Supporting anchors",
                        enumeration[
                            "supporting_anchor_count_at_success"
                        ],
                    ),
                    _line(
                        "Minimum distance",
                        enumeration[
                            "minimum_anchor_distance_at_success"
                        ],
                    ),
                ]
            )

    lines.append(
        _line(
            "Stopped on success",
            (
                "yes"
                if enumeration[
                    "stopped_on_success"
                ]
                else "no"
            ),
        )
    )

    return lines


def format_key_recovery_block(
    report: Mapping[str, Any],
    *,
    key_display: str = "full",
) -> str:
    """Format one validation/test report for terminal output."""
    process = str(
        report[
            "process"
        ]
    ).upper()

    epoch = report.get(
        "epoch"
    )

    title = (
        f"KEY RECOVERY | {process}"
    )

    if (
        process == "VALIDATION"
        and epoch is not None
    ):
        title += (
            f" | EPOCH {epoch}"
        )

    target = report[
        "target"
    ]

    key = report[
        "key"
    ]

    population = report[
        "trace_population"
    ]

    recovery = report[
        "recovery"
    ]

    true_key = _display_key(
        str(
            key[
                "true_key_hex"
            ]
        ),
        key_display,
    )

    top_guess = _display_key(
        str(
            key[
                "top_key_hypothesis_hex"
            ]
        ),
        key_display,
    )

    lines = [
        "=" * 80,
        title,
        "=" * 80,
        _line(
            "Target",
            (
                f"{target['cipher']} key byte "
                f"{target['attack_byte']} | "
                f"{target['key_mode']} | "
                f"leakage={target['leakage_model']}"
            ),
        ),
        _line(
            "True key byte",
            true_key,
        ),
        _line(
            "Top key hypothesis",
            top_guess,
        ),
        _line(
            "AI Recovery status",
            (
                "RECOVERED"
                if key[
                    "recovered"
                ]
                else "NOT RECOVERED"
            ),
        ),
        _line(
            "Final true-key rank",
            (
                f"{key['final_true_key_rank']} "
                f"/ {target['keyspace_size']}"
            ),
        ),
        "",
        _line(
            "Trace population",
            (
                f"source={population['source']} | "
                f"range={population['range']} | "
                f"used={population['used']}"
            ),
        ),
        _line(
            "First rank-1",
            _fmt_trace(
                recovery[
                    "first_rank1_trace"
                ],
                traces_used=_traces_per_attack(population, recovery),
            ),
        ),
        _line(
            "Stable rank-1",
            _fmt_trace(
                recovery[
                    "stable_rank1_trace"
                ],
                traces_used=_traces_per_attack(population, recovery),
            ),
        ),
        "",
        _line(
            "Trace-order repetitions",
            recovery[
                "trace_order_repetitions"
            ],
        ),
        _line(
            "Final SR@1",
            f"{recovery['final_sr_at_1']:.4f}",
        ),
        _line(
            "Final GE",
            f"{recovery['final_ge']:.4f}",
        ),
    ]

    enum_lines = (
        _enumeration_lines(
            report,
            key_display=key_display,
        )
    )

    if enum_lines:
        lines.append(
            ""
        )
        lines.extend(
            enum_lines
        )

    lines.append(
        "=" * 80
    )

    return "\n".join(
        lines
    )


def log_key_recovery_block(
    logger: Any,
    report: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
) -> None:
    """Log one formatted report."""
    contract = (
        validate_key_recovery_contract(
            cfg
        )
    )

    process = report.get(
        "process"
    )

    should_log = (
        process == "validation"
        and contract[
            "log_validation"
        ]
    ) or (
        process == "test"
        and contract[
            "log_test"
        ]
    )

    if not should_log:
        return

    logger.info(
        "\n%s",
        format_key_recovery_block(
            report,
            key_display=contract[
                "key_display"
            ],
        ),
    )


def _summary_section(
    heading: str,
    report: Optional[
        Mapping[str, Any]
    ],
    *,
    key_display: str,
    expected_enumeration_mode: Optional[
        str
    ] = None,
    show_enumeration: bool = True,
    unavailable_reason: Optional[str] = None,
) -> list[str]:
    """
    Build one half of combined summary (no enumeration lines when disabled).

    ``unavailable_reason`` says why there is no report, when that is known
    (the process is not attacked: perform_attack.<process>.attack_times 0).
    """
    lines = [
        heading,
        "-" * 80,
    ]

    if report is None:
        lines.append(
            "Report                  : unavailable"
        )
        if unavailable_reason:
            lines.append(_line("Reason", unavailable_reason))

        if (
            show_enumeration
            and expected_enumeration_mode
            is not None
        ):
            lines.extend(
                [
                    "",
                    _line(
                        "Enumeration mode",
                        expected_enumeration_mode.upper(),
                    ),
                    _line(
                        "Enumeration Recovery status",
                        "UNAVAILABLE",
                    ),
                ]
            )

        return lines

    key = report[
        "key"
    ]

    recovery = report[
        "recovery"
    ]

    population = report[
        "trace_population"
    ]

    if report.get(
        "epoch"
    ) is not None:
        lines.append(
            _line(
                "Epoch",
                report[
                    "epoch"
                ],
            )
        )

    lines.extend(
        [
            _line(
                "True key byte",
                _display_key(
                    key[
                        "true_key_hex"
                    ],
                    key_display,
                ),
            ),
            _line(
                "Top key hypothesis",
                _display_key(
                    key[
                        "top_key_hypothesis_hex"
                    ],
                    key_display,
                ),
            ),
            _line(
                "AI Recovery status",
                (
                    "RECOVERED"
                    if key[
                        "recovered"
                    ]
                    else "NOT RECOVERED"
                ),
            ),
            _line(
                "Final true-key rank",
                (
                    f"{key['final_true_key_rank']} "
                    f"/ {report['target']['keyspace_size']}"
                ),
            ),
            _line(
                "Stable rank-1",
                _fmt_trace(
                    recovery[
                        "stable_rank1_trace"
                    ],
                    traces_used=_traces_per_attack(population, recovery),
                ),
            ),
            _line(
                "Final SR@1",
                f"{recovery['final_sr_at_1']:.4f}",
            ),
            _line(
                "Final GE",
                f"{recovery['final_ge']:.4f}",
            ),
        ]
    )

    enum_lines = _enumeration_lines(
        report,
        expected_mode=expected_enumeration_mode,
        key_display=key_display,
    ) if show_enumeration else []

    if enum_lines:
        lines.append(
            ""
        )
        lines.extend(
            enum_lines
        )

    return lines


def format_combined_key_recovery_summary(
    *,
    validation_report: Optional[
        Mapping[str, Any]
    ],
    test_report: Optional[
        Mapping[str, Any]
    ],
    key_display: str = "full",
    expected_enumeration_mode: Optional[
        str
    ] = None,
    cv_headline: Optional[
        Mapping[str, Any]
    ] = None,
    include_validation: bool = True,
    show_enumeration: bool = True,
    unavailable_reasons: Optional[
        Mapping[str, str]
    ] = None,
) -> str:
    """
    Format the combined selected-validation/final-test key-recovery summary.

    ``unavailable_reasons`` maps "validation" / "test" to the reason its
    report is unavailable (unattacked_process_reasons); it is printed under
    "Report : unavailable".

    With cross-validation, ``cv_headline`` (the CV aggregate) replaces the
    single selected validation snapshot. ``include_validation=False`` (make
    test) prints the test section only; ``validation_report`` is then used
    only to resolve which enumeration mode the test ran.

    In comparison mode, validation owns all strategy selection and publishes
    the comparison envelope containing the raw AI baseline together with the
    local_hamming, neighbour_sweep, and support_weighted results.

    Test behavior is determined exclusively by the validation selection:

    - if validation selects local_hamming, neighbour_sweep, or support_weighted, test contains only
    that frozen validation-selected enumeration algorithm;

    - if validation selects the raw AI baseline because no algorithm improved and
    no enumeration algorithm strictly improves the validation AI rank, no
    enumeration algorithm executes on test and the test report preserves its
    independent raw AI ranking inside the lightweight comparison envelope.

    in short:
    selected_enumeration
    ↓
    selected AI baseline / local_hamming / neighbour_sweep / support_weighted result
    Test evidence never participates in strategy selection.
    """
    validation_expected_mode = (
        expected_enumeration_mode
    )

    test_expected_mode = (
        expected_enumeration_mode
    )

    if (
        expected_enumeration_mode
        == "comparison"
        and validation_report
        is not None
    ):
        validation_enumeration = (
            validation_report.get(
                "enumeration"
            )
            or {}
        )

        if isinstance(
            validation_enumeration,
            Mapping,
        ):
            selected_mode = str(
                validation_enumeration.get(
                    "selected_mode",
                    "",
                )
            ).strip().lower()

            if selected_mode in {
                "local_hamming",
                "neighbour_sweep",
                "support_weighted",
            }:
                test_expected_mode = (
                    selected_mode
                )

            elif selected_mode == "ai":
                test_expected_mode = (
                    "comparison"
                )

    lines = [
        "=" * 80,
        "KEY RECOVERY SUMMARY" if include_validation else "KEY RECOVERY SUMMARY - TEST",
        "=" * 80,
        "",
    ]
    lines.extend(_trained_settings_lines(test_report or validation_report))

    if not include_validation:
        lines.pop()  # no validation section: drop its spacer
    elif cv_headline is not None:
        lines.extend(
            _cv_headline_section(cv_headline, key_display=key_display)
        )
    else:
        lines.extend(
            _summary_section(
                "VALIDATION - SELECTED SNAPSHOT",
                validation_report,
                key_display=key_display,
                expected_enumeration_mode=(
                    validation_expected_mode
                ),
                show_enumeration=show_enumeration,
                unavailable_reason=(unavailable_reasons or {}).get("validation"),
            )
        )

    lines.append(
        ""
    )

    lines.extend(
        _summary_section(
            "TEST - FINAL EVALUATION",
            test_report,
            key_display=key_display,
            expected_enumeration_mode=(
                test_expected_mode
            ),
            show_enumeration=show_enumeration,
            unavailable_reason=(unavailable_reasons or {}).get("test"),
        )
    )

    lines.append(
        "=" * 80
    )

    return "\n".join(
        lines
    )


def _enumeration_enabled(cfg: Mapping[str, Any]) -> bool:
    """sca.key_enumeration.enabled of the run; enumeration fields show only then."""
    enumeration = ((cfg or {}).get("sca") or {}).get("key_enumeration") or {}
    return isinstance(enumeration, Mapping) and enumeration.get("enabled") is True


def _show_enumeration(cfg, validation_report, test_report) -> bool:
    """What the model was trained with decides; the run config only for older reports."""
    recorded = _trained_enumeration_enabled(test_report, validation_report)
    return _enumeration_enabled(cfg) if recorded is None else recorded


def unattacked_process_reasons(cfg: Mapping[str, Any]) -> Dict[str, str]:
    """
    Why a process has no key-recovery report: perform_attack.<process>.attack_times
    is 0, so its traces were not attacked. Only such processes are listed.
    """
    plan = (cfg or {}).get("perform_attack") or {}
    reasons: Dict[str, str] = {}
    for process in ("validation", "test"):
        block = plan.get(process) if isinstance(plan, Mapping) else None
        if isinstance(block, Mapping) and block.get("attack_times") == 0:
            reasons[process] = (
                f"Because perform_attack.{process}.attack_times is 0, "
                f"the {process} traces were not attacked."
            )
    return reasons


def log_combined_key_recovery_summary(
    logger: Any,
    *,
    cfg: Mapping[str, Any],
    validation_report: Optional[
        Mapping[str, Any]
    ],
    test_report: Optional[
        Mapping[str, Any]
    ],
    expected_enumeration_mode: Optional[
        str
    ] = None,
    cv_headline: Optional[
        Mapping[str, Any]
    ] = None,
    include_validation: bool = True,
) -> None:
    """Log combined key-recovery summary."""
    contract = (
        validate_key_recovery_contract(
            cfg
        )
    )

    logger.info(
        "\n%s",
        format_combined_key_recovery_summary(
            validation_report=(
                validation_report
            ),
            test_report=(
                test_report
            ),
            key_display=contract[
                "key_display"
            ],
            expected_enumeration_mode=(
                expected_enumeration_mode
            ),
            cv_headline=cv_headline,
            include_validation=include_validation,
            show_enumeration=_show_enumeration(cfg, validation_report, test_report),
            unavailable_reasons=unattacked_process_reasons(cfg),
        ),
    )


def key_recovery_outcome(
    report: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Outcome of one key-recovery report, for the registries.

    Without enumeration the model rank is the whole story; with enumeration
    the headline is the better of the two ranks, attributed to whichever
    produced it (``_apply_reported_rank`` writes those fields).

    key_recovered / true_key / recovered_key / recovered_key_rank are the
    lifecycle-registry view of the same outcome: recovered_key is the key
    byte enumeration recovered, else the model's top hypothesis.
    """
    key = report.get("key") or {}
    target = report.get("target") or {}
    enumeration = report.get("enumeration") or {}

    model_rank = key.get(
        "final_true_key_rank"
    )

    reported_rank = (
        enumeration.get(
            "key_rank_reported"
        )
        or model_rank
    )

    # Model-only verdict = report.key.recovered (rank 1, no tie, stable
    # when required), the same flag as target_key_byte_recovered.
    key_recovery = (
        enumeration.get(
            "key_recovery"
        )
        or (
            "DONE"
            if key.get("recovered") is True
            else "FAILED"
        )
    )
    recovered_key = enumeration.get("recovered_key_byte")
    if recovered_key is None:
        recovered_key = key.get("top_key_hypothesis")

    return {
        "key_recovery": key_recovery,
        "key_recovered_through": (
            enumeration.get(
                "key_rank_source"
            )
            or "MODEL"
        ),
        "final_true_key_rank": model_rank,
        "key_enumeration_position": (
            enumeration.get(
                "enumeration_position"
            )
        ),
        "key_rank_reported": reported_rank,
        "attack_byte": target.get(
            "attack_byte"
        ),
        "key_recovered": key_recovery == "DONE",
        "true_key": key.get("true_key_byte"),
        "recovered_key": recovered_key,
        "recovered_key_rank": reported_rank,
    }


__all__ = [
    "KEYSPACE_SIZE",
    "PUBLIC_RANK_BASE",
    "ENUMERATION_ALGORITHM_MODES",
    "KeyRecoveryError",
    "build_key_recovery_report",
    "format_combined_key_recovery_summary",
    "format_key_recovery_block",
    "key_recovery_paths",
    "load_best_validation_report",
    "load_key_recovery_report",
    "log_combined_key_recovery_summary",
    "log_key_recovery_block",
    "resolve_key_recovery_dir",
    "save_test_report",
    "save_validation_report",
    "validate_key_enumeration_report_section",
    "validate_key_recovery_contract",
]
