#services/side_channel_core/src/sca/key_enumeration.py

"""
Deterministic AES-byte key enumeration for Side-Channel Analysis (SCA).

This module owns only pure post-ranking enumeration algorithms. It does not
train models, calculate the original AI key ranking, read/write artifacts, or
perform lifecycle orchestration.

Supported modes
---------------
local_hamming
    Evaluate the top-K AI anchors first, promote a bounded number of
    Hamming-local candidates in original AI-rank order, then append all
    remaining candidates in original AI order. Validation may resolve a
    smaller radius/budget policy within configured bounds; that policy is
    frozen before test enumeration.

neighbour_sweep
    Repeatedly select the highest-ranked remaining AI candidate as the current
    anchor, evaluate it, then evaluate still-remaining Hamming neighbours in a
    deterministic bit-combination order. Any local candidate is permanently
    removed and can never later become an anchor.

support_weighted
    Weighted Hamming-Support Enumeration. Use the top-K AI candidates as
    anchors, compute weighted corroboration for every byte candidate from
    nearby anchors, promote at most local_budget supported candidates using a
    deterministic multi-key ordering, then append all remaining candidates in
    original AI order.

The known true byte is never used to construct an enumeration order. It is
used only after ordering to measure success and to implement stop_on_success.
No enumeration algorithm performs validation-time fallback to the original
AI ranking. AI-versus-enumeration selection is owned exclusively by
comparison-mode orchestration.

Author: Shraddha Mishra
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, Dict, List, Optional


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
MODE_PROFILE_FIELDS = {
    "local_hamming": frozenset(
        {
            "top_k",
            "hamming_radius",
            "local_budget",
        }
    ),
    "neighbour_sweep": frozenset(
        {
            "hamming_radius",
        }
    ),
    "support_weighted": frozenset(
        {
            "top_k",
            "hamming_radius",
            "local_budget",
        }
    ),
}


class KeyEnumerationError(ValueError):
    """Raised when enumeration configuration or candidate evidence is invalid."""


CROSS_VALIDATION_CONFLICT_MESSAGE = (
    "sca.key_enumeration.enabled=true is not supported while "
    "cross_validation.enabled=true: cross-validation keeps its key recovery "
    "per fold (metrics/cross_validation/fold_NN/key_recovery), so there is no "
    "single validation ranking to enumerate or to select a strategy from. "
    "Hint: if you want to utilise the TraceFlow Key Enumeration feature, "
    "disable cross_validation."
)


def assert_enumeration_lifecycle_supported(
    cfg: Mapping[str, Any],
) -> None:
    """
    Reject the one lifecycle combination enumeration cannot serve.

    Called early (model-contract resolution) so a run fails before training
    rather than after, and again when the runtime contract is validated.
    """
    if not isinstance(cfg, Mapping):
        raise KeyEnumerationError(
            "cfg must be a mapping"
        )

    enumeration = (
        (cfg.get("sca") or {})
        .get("key_enumeration")
        or {}
    )

    cross_validation = (
        cfg.get("cross_validation")
        or {}
    )

    if (
        isinstance(enumeration, Mapping)
        and enumeration.get("enabled") is True
        and isinstance(cross_validation, Mapping)
        and cross_validation.get("enabled") is True
    ):
        raise KeyEnumerationError(
            CROSS_VALIDATION_CONFLICT_MESSAGE
        )


def _byte(value: Any, *, field: str) -> int:
    """Return one validated AES-byte value."""
    if isinstance(value, bool):
        raise KeyEnumerationError(
            f"{field} must be an integer in [0, 255]"
        )

    try:
        built = int(value)
    except (TypeError, ValueError) as exc:
        raise KeyEnumerationError(
            f"{field} must be an integer in [0, 255]"
        ) from exc

    if not 0 <= built < KEYSPACE_SIZE:
        raise KeyEnumerationError(
            f"{field} must be in [0, 255], got {built}"
        )

    return built


def _positive_int(
    value: Any,
    *,
    field: str,
    maximum: int,
) -> int:
    """Return a validated positive integer with an inclusive upper bound."""
    if isinstance(value, bool):
        raise KeyEnumerationError(
            f"{field} must be an integer in [1, {maximum}]"
        )

    try:
        built = int(value)
    except (TypeError, ValueError) as exc:
        raise KeyEnumerationError(
            f"{field} must be an integer in [1, {maximum}]"
        ) from exc

    if not 1 <= built <= maximum:
        raise KeyEnumerationError(
            f"{field} must be in [1, {maximum}], got {built}"
        )

    return built


def _non_negative_int(
    value: Any,
    *,
    field: str,
    maximum: int,
) -> int:
    """Return a validated non-negative integer with an inclusive upper bound."""
    if isinstance(value, bool):
        raise KeyEnumerationError(
            f"{field} must be an integer in [0, {maximum}]"
        )

    try:
        built = int(value)
    except (TypeError, ValueError) as exc:
        raise KeyEnumerationError(
            f"{field} must be an integer in [0, {maximum}]"
        ) from exc

    if not 0 <= built <= maximum:
        raise KeyEnumerationError(
            f"{field} must be in [0, {maximum}], got {built}"
        )

    return built


def _finite_float(
    value: Any,
    *,
    field: str,
) -> float:
    """Return one finite candidate score."""
    if isinstance(value, bool):
        raise KeyEnumerationError(
            f"{field} must be a finite number"
        )

    try:
        built = float(value)
    except (TypeError, ValueError) as exc:
        raise KeyEnumerationError(
            f"{field} must be a finite number"
        ) from exc

    if not math.isfinite(built):
        raise KeyEnumerationError(
            f"{field} must be finite, got {built!r}"
        )

    return built

def _reject_unresolved_sweep_value(
    value: Any,
    *,
    field: str,
) -> None:
    """
    Reject unresolved search-space values at runtime.

    ``key_enumeration.py`` consumes only one concrete experiment contract.
    List-valued sweep/search dimensions must be resolved before this module is
    called by SCC or MLOps.
    """
    if isinstance(value, list):
        raise KeyEnumerationError(
            f"{field} is still an unresolved sweep list: {value!r}. "
            "SCC/MLOps must resolve SCA sweep dimensions before "
            "key-enumeration runtime validation."
        )

def _normalize_mode_profile(
    *,
    mode: str,
    profile: Mapping[str, Any],
) -> Dict[str, Optional[int]]:
    """
    Validate one concrete runtime key-enumeration mode profile.

    Important
    ---------
    This function does NOT expand sweep/search lists.

    Search configuration may contain values such as:

        hamming_radius: [1, 2]

    but before runtime reaches this module, the sweep layer must reduce that
    to exactly one concrete value:

        hamming_radius: 2

    Only the selected runtime mode profile should normally remain in the
    concrete run configuration.
    """
    if mode not in ENUMERATION_ALGORITHM_MODES:
        raise KeyEnumerationError(
            f"Unsupported key-enumeration profile: {mode!r}"
        )

    if not isinstance(profile, Mapping):
        raise KeyEnumerationError(
            f"sca.key_enumeration.modes.{mode} must be a mapping"
        )

    allowed_fields = MODE_PROFILE_FIELDS[mode]

    unknown_fields = sorted(
        set(profile) - allowed_fields
    )

    if unknown_fields:
        rendered = ", ".join(
            f"sca.key_enumeration.modes.{mode}.{field}"
            for field in unknown_fields
        )

        raise KeyEnumerationError(
            f"Unsupported field(s) for mode={mode!r}: {rendered}"
        )

    missing_fields = sorted(
        allowed_fields - set(profile)
    )

    if missing_fields:
        rendered = ", ".join(
            f"sca.key_enumeration.modes.{mode}.{field}"
            for field in missing_fields
        )

        raise KeyEnumerationError(
            f"Missing required field(s) for mode={mode!r}: {rendered}"
        )

    for field in allowed_fields:
        _reject_unresolved_sweep_value(
            profile[field],
            field=(
                f"sca.key_enumeration.modes.{mode}.{field}"
            ),
        )

    top_k: Optional[int] = None
    hamming_radius: Optional[int] = None
    local_budget: Optional[int] = None

    if "top_k" in allowed_fields:
        top_k = _positive_int(
            profile["top_k"],
            field=(
                f"sca.key_enumeration.modes.{mode}.top_k"
            ),
            maximum=KEYSPACE_SIZE - 1,
        )

    hamming_radius = _positive_int(
        profile["hamming_radius"],
        field=(
            f"sca.key_enumeration.modes.{mode}.hamming_radius"
        ),
        maximum=KEY_WIDTH_BITS,
    )

    if "local_budget" in allowed_fields:
        if mode == "local_hamming":
            if top_k is None:
                raise KeyEnumerationError(
                    "Internal local_hamming profile validation error: "
                    "top_k is unavailable"
                )

            maximum_budget = (
                KEYSPACE_SIZE - top_k
            )

        else:
            maximum_budget = (
                KEYSPACE_SIZE - 1
            )

        local_budget = _positive_int(
            profile["local_budget"],
            field=(
                f"sca.key_enumeration.modes.{mode}.local_budget"
            ),
            maximum=maximum_budget,
        )

    return {
        "top_k": top_k,
        "hamming_radius": hamming_radius,
        "local_budget": local_budget,
    }
def validate_key_enumeration_contract(
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Validate one concrete runtime ``sca.key_enumeration`` contract.

    Search-space configuration is allowed to contain lists, but those lists
    must already have been resolved by SCC/MLOps before this function runs.

    Expected concrete runtime shape
    -------------------------------

    local_hamming:

        key_enumeration:
          enabled: true
          mode: local_hamming
          stop_on_success: true
          modes:
            local_hamming:
              top_k: 5
              hamming_radius: 2
              local_budget: 16

    neighbour_sweep:

        key_enumeration:
          enabled: true
          mode: neighbour_sweep
          stop_on_success: true
          modes:
            neighbour_sweep:
              hamming_radius: 1

    support_weighted:

        key_enumeration:
          enabled: true
          mode: support_weighted
          stop_on_success: true
          modes:
            support_weighted:
              top_k: 10
              hamming_radius: 2
              local_budget: 8

    Runtime invariants
    ------------------
    - mode must be one scalar string;
    - candidate_ranking.mode must be the scalar string ``all``;
    - unresolved sweep lists are rejected explicitly;
    - local_hamming/neighbour_sweep/support_weighted require one concrete selected profile;
    - comparison requires concrete local_hamming, neighbour_sweep, and support_weighted profiles;
    - comparison owns no profile of its own;
    - comparison orchestration is performed by key_enumeration_helper.py.
    """
    if not isinstance(cfg, Mapping):
        raise KeyEnumerationError(
            "cfg must be a mapping"
        )

    assert_enumeration_lifecycle_supported(
        cfg
    )

    sca_cfg = (
        cfg.get("sca")
        or {}
    )

    if not isinstance(
        sca_cfg,
        Mapping,
    ):
        raise KeyEnumerationError(
            "sca must be a mapping"
        )

    raw = (
        sca_cfg.get(
            "key_enumeration"
        )
        or {}
    )

    if not isinstance(
        raw,
        Mapping,
    ):
        raise KeyEnumerationError(
            "sca.key_enumeration must be a mapping"
        )

    allowed_top_level_fields = {
        "enabled",
        "mode",
        "stop_on_success",
        "modes",
    }

    if "best_rank" in raw:
        raise KeyEnumerationError(
            "sca.key_enumeration.best_rank was retired. The reported key rank "
            "is now always the better of the model rank and the enumeration "
            "position, with ties attributed to the model, in every mode. "
            "Remove the key; the raw per-mode positions stay in the report."
        )

    unknown_top_level_fields = sorted(
        set(raw)
        - allowed_top_level_fields
    )

    if unknown_top_level_fields:
        rendered = ", ".join(
            f"sca.key_enumeration.{field}"
            for field in unknown_top_level_fields
        )

        raise KeyEnumerationError(
            "Mode-specific enumeration parameters must "
            "be stored under sca.key_enumeration.modes.<mode>. "
            f"Unexpected top-level field(s): {rendered}"
        )

    enabled = raw.get(
        "enabled",
        False,
    )

    if not isinstance(
        enabled,
        bool,
    ):
        raise KeyEnumerationError(
            "sca.key_enumeration.enabled must be boolean"
        )

    mode_raw = raw.get(
        "mode"
    )

    if isinstance(mode_raw, list):
        raise KeyEnumerationError(
            "sca.key_enumeration.mode is still an unresolved "
            f"sweep list: {mode_raw!r}. "
            "Resolve SCA sweep dimensions before runtime."
        )

    mode: Optional[str] = None

    if mode_raw is not None:
        if (
            not isinstance(
                mode_raw,
                str,
            )
            or not mode_raw.strip()
        ):
            raise KeyEnumerationError(
                "sca.key_enumeration.mode must be "
                "a non-empty string"
            )

        mode = (
            mode_raw
            .strip()
            .lower()
        )

        if (
            mode
            not in SUPPORTED_ENUMERATION_MODES
        ):
            supported = ", ".join(
                sorted(
                    SUPPORTED_ENUMERATION_MODES
                )
            )

            raise KeyEnumerationError(
                "sca.key_enumeration.mode must be one of: "
                f"{supported}; got {mode!r}"
            )

    if (
        enabled
        and mode is None
    ):
        raise KeyEnumerationError(
            "sca.key_enumeration.mode is required "
            "when enabled=true"
        )

    stop_raw = raw.get(
        "stop_on_success"
    )

    if isinstance(
        stop_raw,
        list,
    ):
        raise KeyEnumerationError(
            "sca.key_enumeration.stop_on_success must "
            "already be resolved to one boolean value"
        )

    if (
        enabled
        and stop_raw is None
    ):
        raise KeyEnumerationError(
            "sca.key_enumeration.stop_on_success "
            "is required when enabled=true"
        )

    stop_on_success = (
        False                         # owner 2026-10-23: the whole budget is enumerated unless asked to stop
        if stop_raw is None
        else stop_raw
    )

    if not isinstance(
        stop_on_success,
        bool,
    ):
        raise KeyEnumerationError(
            "sca.key_enumeration.stop_on_success "
            "must be boolean"
        )
    modes_raw = (
        raw.get(
            "modes"
        )
        or {}
    )

    if not isinstance(
        modes_raw,
        Mapping,
    ):
        raise KeyEnumerationError(
            "sca.key_enumeration.modes must be a mapping"
        )

    unknown_modes = sorted(
        set(modes_raw)
        - ENUMERATION_ALGORITHM_MODES
    )

    if unknown_modes:
        rendered = ", ".join(
            str(item)
            for item in unknown_modes
        )

        raise KeyEnumerationError(
            "Unsupported profile(s) under "
            "sca.key_enumeration.modes: "
            f"{rendered}"
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
        raise KeyEnumerationError(
            "sca.candidate_ranking must be a mapping"
        )

    ranking_mode_raw = (
        candidate_ranking.get(
            "mode"
        )
    )

    if isinstance(
        ranking_mode_raw,
        list,
    ):
        raise KeyEnumerationError(
            "sca.candidate_ranking.mode is still an unresolved "
            f"sweep list: {ranking_mode_raw!r}. "
            "Resolve SCA sweep dimensions before runtime."
        )

    ranking_mode = str(
        ranking_mode_raw
        if ranking_mode_raw is not None
        else ""
    ).strip().lower()

    if enabled and ranking_mode != "all":
        raise KeyEnumerationError(
            "Key enumeration requires "
            "sca.candidate_ranking.mode='all'"
        )

    if not enabled:
        return {
            "enabled": False,
            "mode": mode,
            "top_k": None,
            "hamming_radius": None,
            "local_budget": None,
            "stop_on_success": (
                stop_on_success
            ),
        }

    if mode is None:
        raise KeyEnumerationError(
            "Enabled key enumeration has no runtime mode"
        )
    #
    # Comparison is a post-ranking meta-mode.
    #
    # It does not own an enumeration profile of its own. Instead, it requires
    # one concrete profile for every real enumeration algorithm so that all
    # three algorithms can be evaluated against the same validation ranking.
    #
    if mode == "comparison":
        missing_profiles = sorted(
            ENUMERATION_ALGORITHM_MODES
            - set(modes_raw)
        )

        if missing_profiles:
            rendered = ", ".join(
                (
                    "sca.key_enumeration."
                    f"modes.{item}"
                )
                for item in missing_profiles
            )

            raise KeyEnumerationError(
                "comparison mode requires one runtime "
                "profile for every enumeration algorithm. "
                f"Missing: {rendered}"
            )

        comparison_profiles: Dict[
            str,
            Dict[str, Optional[int]],
        ] = {}

        for algorithm_mode in sorted(
            ENUMERATION_ALGORITHM_MODES
        ):
            comparison_profiles[
                algorithm_mode
            ] = _normalize_mode_profile(
                mode=algorithm_mode,
                profile=modes_raw[
                    algorithm_mode
                ],
            )

        return {
            "enabled": True,
            "mode": "comparison",
            "top_k": None,
            "hamming_radius": None,
            "local_budget": None,
            "stop_on_success": (
                stop_on_success
            ),
            "comparison_profiles": (
                comparison_profiles
            ),
        }

    if mode not in modes_raw:
        raise KeyEnumerationError(
            "Selected key-enumeration mode has no "
            "runtime profile: "
            f"sca.key_enumeration.modes.{mode}"
        )

    #
    # Ordinary algorithm mode:
    # validate only the selected runtime profile.
    #
    # Inactive profiles may remain unresolved sweep spaces and therefore
    # must not influence or invalidate this concrete run.
    #
    selected_profile = (
        _normalize_mode_profile(
            mode=mode,
            profile=modes_raw[
                mode
            ],
        )
    )

    return {
        "enabled": True,
        "mode": mode,
        "top_k": (
            selected_profile.get(
                "top_k"
            )
        ),
        "hamming_radius": (
            selected_profile.get(
                "hamming_radius"
            )
        ),
        "local_budget": (
            selected_profile.get(
                "local_budget"
            )
        ),
        "stop_on_success": (
            stop_on_success
        ),
        "comparison_profiles": None,
    }

def _normalize_ranked_candidates(
    ranked_candidates: Any,
) -> List[Dict[str, Any]]:
    """
    Validate and copy the complete one-based AI candidate ranking.

    Requirements:
    - exactly 256 entries;
    - public ranks exactly 1..256;
    - every AES byte exactly once;
    - finite score for every candidate.
    """
    if (
        isinstance(ranked_candidates, (str, bytes))
        or not isinstance(
            ranked_candidates,
            Sequence,
        )
    ):
        raise KeyEnumerationError(
            "ranked_candidates must be a sequence"
        )

    if len(ranked_candidates) != KEYSPACE_SIZE:
        raise KeyEnumerationError(
            "Key enumeration requires exactly 256 ranked candidates; "
            f"received {len(ranked_candidates)}"
        )

    normalized: List[Dict[str, Any]] = []

    seen_guesses: set[int] = set()
    seen_ranks: set[int] = set()

    for expected_rank, item in enumerate(
        ranked_candidates,
        start=PUBLIC_RANK_BASE,
    ):
        if not isinstance(item, Mapping):
            raise KeyEnumerationError(
                f"ranked_candidates[{expected_rank}] "
                "must be a mapping"
            )

        rank = _positive_int(
            item.get("rank"),
            field=(
                f"ranked_candidates[{expected_rank}].rank"
            ),
            maximum=KEYSPACE_SIZE,
        )

        if rank in seen_ranks:
            raise KeyEnumerationError(
                "ranked_candidates contains duplicate "
                f"rank {rank}"
            )

        seen_ranks.add(rank)

        if rank != expected_rank:
            raise KeyEnumerationError(
                f"ranked_candidates[{expected_rank}].rank "
                f"must be {expected_rank}, got {rank}"
            )

        key_guess = _byte(
            item.get(
                "key_guess",
                item.get("key"),
            ),
            field=(
                f"ranked_candidates[{expected_rank}].key_guess"
            ),
        )

        if key_guess in seen_guesses:
            raise KeyEnumerationError(
                "ranked_candidates contains duplicate "
                f"key guess 0x{key_guess:02X}"
            )

        seen_guesses.add(key_guess)

        score = _finite_float(
            item.get("score"),
            field=(
                f"ranked_candidates[{expected_rank}].score"
            ),
        )

        normalized.append(
            {
                "rank": rank,
                "key_guess": key_guess,
                "key_guess_hex": (
                    f"0x{key_guess:02X}"
                ),
                "score": score,
            }
        )

    expected_guesses = set(
        range(KEYSPACE_SIZE)
    )

    if seen_guesses != expected_guesses:
        missing = sorted(
            expected_guesses - seen_guesses
        )
        raise KeyEnumerationError(
            "ranked_candidates must contain every "
            "AES-byte hypothesis exactly once; "
            f"missing={missing}"
        )

    expected_ranks = set(
        range(
            PUBLIC_RANK_BASE,
            KEYSPACE_SIZE + 1,
        )
    )

    if seen_ranks != expected_ranks:
        missing_ranks = sorted(
            expected_ranks - seen_ranks
        )
        raise KeyEnumerationError(
            "ranked_candidates must contain every "
            "public rank exactly once; "
            f"missing={missing_ranks}"
        )

    return normalized


def hamming_distance(
    left: int,
    right: int,
) -> int:
    """Return bitwise Hamming distance between two AES bytes."""
    lhs = _byte(
        left,
        field="left",
    )

    rhs = _byte(
        right,
        field="right",
    )

    return (
        lhs ^ rhs
    ).bit_count()


def _validate_complete_order(
    ordered_candidates: Sequence[
        Mapping[str, Any]
    ],
    *,
    mode: str,
) -> None:
    """
    Require exactly one deterministic permutation of the AES-byte space.

    This invariant guarantees that stop_on_success=false evaluates exactly
    256 unique candidates for every enumeration mode.
    """
    if len(
        ordered_candidates
    ) != KEYSPACE_SIZE:
        raise KeyEnumerationError(
            f"{mode} ordering must contain exactly "
            f"{KEYSPACE_SIZE} candidates"
        )

    guesses = [
        _byte(
            item.get("key_guess"),
            field=(
                f"{mode}.ordered.key_guess"
            ),
        )
        for item in ordered_candidates
    ]

    if len(
        set(guesses)
    ) != KEYSPACE_SIZE:
        raise KeyEnumerationError(
            f"{mode} ordering contains duplicate candidates"
        )

    if set(
        guesses
    ) != set(
        range(KEYSPACE_SIZE)
    ):
        raise KeyEnumerationError(
            f"{mode} ordering must contain every "
            "AES-byte candidate exactly once"
        )




def _local_hamming_candidate_order(
    ranked_candidates: Any,
    *,
    top_k: int,
    hamming_radius: int,
    local_budget: int,
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, int],
]:
    """
    Build bounded local Hamming promotion followed by AI fallback.

    Order:
        AI anchors
        -> bounded local candidates in AI-rank order
        -> every remaining candidate in original AI order
    """
    ranked = _normalize_ranked_candidates(
        ranked_candidates
    )

    anchor_count = _positive_int(
        top_k,
        field="top_k",
        maximum=KEYSPACE_SIZE - 1,
    )

    radius = _non_negative_int(
        hamming_radius,
        field="hamming_radius",
        maximum=KEY_WIDTH_BITS,
    )

    budget = _non_negative_int(
        local_budget,
        field="local_budget",
        maximum=(
            KEYSPACE_SIZE
            - anchor_count
        ),
    )

    if (
        radius == 0
    ) != (
        budget == 0
    ):
        raise KeyEnumerationError(
            "hamming_radius and local_budget "
            "must either both be zero or "
            "both be positive"
        )

    anchors = ranked[
        :anchor_count
    ]

    anchor_values = tuple(
        item["key_guess"]
        for item in anchors
    )

    anchor_keys = set(
        anchor_values
    )

    decorated: List[
        Dict[str, Any]
    ] = []

    for item in ranked:
        distance = min(
            hamming_distance(
                item["key_guess"],
                anchor,
            )
            for anchor
            in anchor_values
        )

        decorated.append(
            {
                **item,
                "hamming_distance": (
                    distance
                ),
                "is_anchor": (
                    item["key_guess"]
                    in anchor_keys
                ),
            }
        )

    anchor_rows = [
        item
        for item in decorated
        if item["key_guess"]
        in anchor_keys
    ]

    local_pool = [
        item
        for item in decorated
        if (
            item["key_guess"]
            not in anchor_keys
        )
        and (
            item[
                "hamming_distance"
            ] <= radius
        )
    ]

    local_pool.sort(
        key=lambda item: (
            item["rank"]
        )
    )

    promoted = local_pool[
        :budget
    ]

    promoted_keys = {
        item["key_guess"]
        for item in promoted
    }

    fallback = [
        item
        for item in decorated
        if (
            item["key_guess"]
            not in anchor_keys
        )
        and (
            item["key_guess"]
            not in promoted_keys
        )
    ]

    ordered = [
        *[
            {
                **item,
                "enumeration_phase": (
                    "anchor"
                ),
                "is_local_promoted": (
                    False
                ),
            }
            for item in anchor_rows
        ],
        *[
            {
                **item,
                "enumeration_phase": (
                    "hamming_local"
                ),
                "is_local_promoted": (
                    True
                ),
            }
            for item in promoted
        ],
        *[
            {
                **item,
                "enumeration_phase": (
                    "ai_fallback"
                ),
                "is_local_promoted": (
                    False
                ),
            }
            for item in fallback
        ],
    ]

    _validate_complete_order(
        ordered,
        mode="local_hamming",
    )

    order_policy = {
        "top_k": anchor_count,
        "hamming_radius": radius,
        "local_budget": budget,
        "local_candidates_available": (
            len(local_pool)
        ),
        "local_candidates_promoted": (
            len(promoted)
        ),
    }

    return (
        ordered,
        [
            dict(item)
            for item in anchors
        ],
        order_policy,
    )



def _hamming_neighbours(
    value: int,
    *,
    radius: int,
) -> List[
    tuple[int, int]
]:
    """
    Generate deterministic Hamming neighbours.

    Ordering:
        Hamming distance 1 first,
        then distance 2,
        ...
        then ``radius``.

    Within one distance, bit-index combinations are lexicographic.

    For radius=1 this gives exactly:

        bit 0, bit 1, ..., bit 7

    Returns:
        ``(candidate_byte, hamming_distance)``
    """
    anchor = _byte(
        value,
        field="value",
    )

    maximum_radius = _positive_int(
        radius,
        field="radius",
        maximum=KEY_WIDTH_BITS,
    )

    neighbours: List[
        tuple[int, int]
    ] = []

    for distance in range(
        1,
        maximum_radius + 1,
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
                    anchor ^ mask,
                    distance,
                )
            )

    return neighbours


def _neighbour_sweep_candidate_order(
    ranked_candidates: Any,
    *,
    hamming_radius: int,
) -> List[
    Dict[str, Any]
]:
    """
    Build the professor's neighbour_sweep remaining-set traversal.

    A candidate consumed as a Hamming-local neighbour is permanently removed
    and can never become a later anchor.
    """
    ranked = _normalize_ranked_candidates(
        ranked_candidates
    )

    radius = _positive_int(
        hamming_radius,
        field="hamming_radius",
        maximum=KEY_WIDTH_BITS,
    )

    by_guess = {
        item["key_guess"]: item
        for item in ranked
    }

    remaining = {
        item["key_guess"]
        for item in ranked
    }

    ordered: List[
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
                "enumeration_phase": (
                    "anchor"
                ),
                "is_anchor": True,
                "anchor_key_guess": (
                    anchor
                ),
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
                    "enumeration_phase": (
                        "hamming_local"
                    ),
                    "is_anchor": False,
                    "anchor_key_guess": (
                        anchor
                    ),
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
        raise KeyEnumerationError(
            "neighbour_sweep traversal ended with "
            "unevaluated candidates"
        )

    _validate_complete_order(
        ordered,
        mode="neighbour_sweep",
    )

    return ordered


def _support_weighted_candidate_order(
    ranked_candidates: Any,
    *,
    top_k: int,
    hamming_radius: int,
    local_budget: int,
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, int],
]:
    """
    Build Weighted Hamming-Support ordering.

    For each candidate c:

        support(c) =
            sum(
                weight(anchor) / HD(c, anchor)
            )

    for anchors satisfying:

        1 <= HD(c, anchor) <= radius

    Anchor weights:

        exp(score_i - score_max)

    Supported candidates are sorted by:

        1. higher weighted support;
        2. more distinct supporting anchors;
        3. smaller minimum anchor distance;
        4. better original AI rank;
        5. smaller numeric byte value.

    At most local_budget candidates are promoted. Every non-promoted candidate
    is appended in original AI order.
    """
    ranked = _normalize_ranked_candidates(
        ranked_candidates
    )

    anchor_count = _positive_int(
        top_k,
        field="top_k",
        maximum=KEYSPACE_SIZE - 1,
    )

    radius = _positive_int(
        hamming_radius,
        field="hamming_radius",
        maximum=KEY_WIDTH_BITS,
    )

    budget = _positive_int(
        local_budget,
        field="local_budget",
        maximum=KEYSPACE_SIZE - 1,
    )

    anchors = ranked[
        :anchor_count
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

    supported: List[
        Dict[str, Any]
    ] = []

    for candidate in ranked:
        support = 0.0
        supporting_anchors = 0

        minimum_distance: Optional[
            int
        ] = None

        for anchor in weighted_anchors:
            distance = hamming_distance(
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

            supporting_anchors += 1

            if (
                minimum_distance
                is None
                or distance
                < minimum_distance
            ):
                minimum_distance = (
                    distance
                )

        if supporting_anchors == 0:
            continue

        supported.append(
            {
                **candidate,
                "weighted_hamming_support": (
                    support
                ),
                "supporting_anchor_count": (
                    supporting_anchors
                ),
                "minimum_anchor_distance": (
                    minimum_distance
                ),
            }
        )

    supported.sort(
        key=lambda item: (
            -item[
                "weighted_hamming_support"
            ],
            -item[
                "supporting_anchor_count"
            ],
            item[
                "minimum_anchor_distance"
            ],
            item["rank"],
            item["key_guess"],
        )
    )

    promoted = supported[
        :budget
    ]

    promoted_keys = {
        item["key_guess"]
        for item in promoted
    }

    fallback = [
        item
        for item in ranked
        if (
            item["key_guess"]
            not in promoted_keys
        )
    ]

    ordered = [
        *[
            {
                **item,
                "enumeration_phase": (
                    "support_promoted"
                ),
                "is_support_promoted": (
                    True
                ),
            }
            for item in promoted
        ],
        *[
            {
                **item,
                "enumeration_phase": (
                    "ai_fallback"
                ),
                "is_support_promoted": (
                    False
                ),
            }
            for item in fallback
        ],
    ]

    _validate_complete_order(
        ordered,
        mode="support_weighted",
    )

    order_policy = {
        "top_k": anchor_count,
        "hamming_radius": radius,
        "local_budget": budget,
        "supported_candidates_available": (
            len(supported)
        ),
        "supported_candidates_promoted": (
            len(promoted)
        ),
    }

    return (
        ordered,
        weighted_anchors,
        order_policy,
    )


def _evaluate_order(
    ordered_candidates: Sequence[
        Mapping[str, Any]
    ],
    *,
    true_key_byte: int,
    mode: str,
    stop_on_success: bool,
) -> Dict[str, Any]:
    """
    Evaluate one precomputed order against the benchmark true byte.

    Candidate ordering is already complete before this function sees the true
    key. Therefore the true key cannot influence ordering.
    """
    if mode not in ENUMERATION_ALGORITHM_MODES:
        raise KeyEnumerationError(
            f"Unsupported enumeration mode: {mode!r}"
        )

    if not isinstance(
        stop_on_success,
        bool,
    ):
        raise KeyEnumerationError(
            "stop_on_success must be boolean"
        )

    _validate_complete_order(
        ordered_candidates,
        mode=mode,
    )

    true_key = _byte(
        true_key_byte,
        field="true_key_byte",
    )

    evaluated: List[
        Dict[str, Any]
    ] = []

    success_position: Optional[
        int
    ] = None

    success_item: Optional[
        Mapping[str, Any]
    ] = None

    passthrough_fields = (
        "rank",
        "score",
        "enumeration_phase",
        "hamming_distance",
        "is_anchor",
        "is_local_promoted",
        "anchor_key_guess",
        "anchor_key_guess_hex",
        "anchor_ai_rank",
        "weighted_hamming_support",
        "supporting_anchor_count",
        "minimum_anchor_distance",
        "is_support_promoted",
    )

    for position, item in enumerate(
        ordered_candidates,
        start=1,
    ):
        key_guess = _byte(
            item.get("key_guess"),
            field="key_guess",
        )

        row: Dict[
            str,
            Any,
        ] = {
            "enumeration_position": (
                position
            ),
            "key_guess": key_guess,
            "key_guess_hex": (
                f"0x{key_guess:02X}"
            ),
        }

        for field in passthrough_fields:
            if field not in item:
                continue

            output_field = (
                "ai_rank"
                if field == "rank"
                else field
            )

            row[
                output_field
            ] = item[
                field
            ]

        evaluated.append(
            row
        )

        if (
            key_guess == true_key
            and success_position
            is None
        ):
            success_position = (
                position
            )

            success_item = item

            if stop_on_success:
                break

    if (
        success_position is None
        or success_item is None
    ):
        raise KeyEnumerationError(
            "Complete AES-byte enumeration order "
            "did not contain the true key"
        )
    recovered_key = _byte(
        success_item.get(
            "key_guess"
        ),
        field="success_item.key_guess",
    )
    ai_rank_at_success = (
        _positive_int(
            success_item.get("rank"),
            field="success_item.rank",
            maximum=KEYSPACE_SIZE,
        )
    )

    return {
        "enabled": True,
        "mode": mode,
        "executed": True,
        "success": True,
        "recovered_key_byte": (
            recovered_key
        ),
        "recovered_key_hex": (
            f"0x{recovered_key:02X}"
        ),
            "enumeration_position": (
            success_position
        ),
        "candidates_evaluated": (
            len(evaluated)
        ),
        "stop_on_success": (
            stop_on_success
        ),
        "stopped_on_success": (
            stop_on_success
        ),
        "evaluated_candidates": (
            evaluated
        ),
        "ai_rank_at_success": (
            ai_rank_at_success
        ),
        "position_delta": (
            success_position
            - ai_rank_at_success
        ),
        "success_phase": (
            success_item.get(
                "enumeration_phase"
            )
        ),
    }
def enumerate_local_hamming(
    *,
    ranked_candidates: Any,
    true_key_byte: int,
    top_k: int,
    hamming_radius: int,
    local_budget: int,
    stop_on_success: bool,
) -> Dict[str, Any]:
    """
    Run the configured LOCAL_HAMMING enumeration algorithm exactly as specified.

    LOCAL_HAMMING owns no validation-time AI fallback and performs no policy
    optimization. The configured top_k, hamming_radius, and local_budget
    values directly define the enumeration order.

    Any decision to retain the original AI ranking instead of an enumeration
    result belongs exclusively to comparison-mode orchestration.
    """
    (
        ordered,
        anchors,
        order_policy,
    ) = _local_hamming_candidate_order(
        ranked_candidates,
        top_k=top_k,
        hamming_radius=hamming_radius,
        local_budget=local_budget,
    )

    result = _evaluate_order(
        ordered,
        true_key_byte=true_key_byte,
        mode="local_hamming",
        stop_on_success=stop_on_success,
    )

    success_row = result[
        "evaluated_candidates"
    ][
        result["enumeration_position"] - 1
    ]

    result.update(
        {
            "policy_source": "configuration",
            "top_k": order_policy["top_k"],
            "hamming_radius": order_policy["hamming_radius"],
            "local_budget": order_policy["local_budget"],
            "local_candidates_available": (order_policy["local_candidates_available"]),
            "local_candidates_promoted": (order_policy["local_candidates_promoted"]),
            "fallback_order": "ai",
            "fallback_used": any(
                row.get("enumeration_phase")
                == "ai_fallback"
                for row in result["evaluated_candidates"]),
            "anchor_candidates": anchors,
            "hamming_distance_at_success": (success_row.get("hamming_distance")),
        }
    )

    return result

def enumerate_neighbour_sweep(
    *,
    ranked_candidates: Any,
    true_key_byte: int,
    hamming_radius: int,
    stop_on_success: bool,
) -> Dict[str, Any]:
    """
    Run neighbour_sweep anchor/Hamming-neighbour traversal.

    No validation truth is used to select or alter this order.
    """
    radius = _positive_int(
        hamming_radius,
        field="hamming_radius",
        maximum=KEY_WIDTH_BITS,
    )

    ordered = (
        _neighbour_sweep_candidate_order(
            ranked_candidates,
            hamming_radius=radius,
        )
    )

    result = _evaluate_order(
        ordered,
        true_key_byte=(
            true_key_byte
        ),
        mode="neighbour_sweep",
        stop_on_success=(
            stop_on_success
        ),
    )

    success_row = result[
        "evaluated_candidates"
    ][
        result[
            "enumeration_position"
        ] - 1
    ]

    result.update(
        {
            "policy_source": (
                "configuration"
            ),
            "hamming_radius": (
                radius
            ),
            "anchor_key_guess_at_success": (
                success_row.get(
                    "anchor_key_guess"
                )
            ),
            "anchor_key_guess_hex_at_success": (
                success_row.get(
                    "anchor_key_guess_hex"
                )
            ),
            "anchor_ai_rank_at_success": (
                success_row.get(
                    "anchor_ai_rank"
                )
            ),
            "hamming_distance_at_success": (
                success_row.get(
                    "hamming_distance"
                )
            ),
        }
    )

    return result


def enumerate_support_weighted(
    *,
    ranked_candidates: Any,
    true_key_byte: int,
    top_k: int,
    hamming_radius: int,
    local_budget: int,
    stop_on_success: bool,
) -> Dict[str, Any]:
    """
    Run Weighted Hamming-Support Enumeration followed by AI fallback.

    Support is recomputed from the supplied split's own AI ranking and scores.
    """
    (
        ordered,
        anchors,
        order_policy,
    ) = _support_weighted_candidate_order(
        ranked_candidates,
        top_k=top_k,
        hamming_radius=(
            hamming_radius
        ),
        local_budget=(
            local_budget
        ),
    )

    result = _evaluate_order(
        ordered,
        true_key_byte=(
            true_key_byte
        ),
        mode="support_weighted",
        stop_on_success=(
            stop_on_success
        ),
    )

    success_row = result[
        "evaluated_candidates"
    ][
        result[
            "enumeration_position"
        ] - 1
    ]

    result.update(
        {
            "policy_source": (
                "configuration"
            ),
            "top_k": (
                order_policy[
                    "top_k"
                ]
            ),
            "hamming_radius": (
                order_policy[
                    "hamming_radius"
                ]
            ),
            "local_budget": (
                order_policy[
                    "local_budget"
                ]
            ),
            "supported_candidates_available": (
                order_policy[
                    "supported_candidates_available"
                ]
            ),
            "supported_candidates_promoted": (
                order_policy[
                    "supported_candidates_promoted"
                ]
            ),
            "fallback_order": "ai",
            "fallback_used": any(
                row.get(
                    "enumeration_phase"
                ) == "ai_fallback"
                for row
                in result[
                    "evaluated_candidates"
                ]
            ),
            "anchor_candidates": (
                anchors
            ),
            "weighted_hamming_support_at_success": (
                success_row.get(
                    "weighted_hamming_support"
                )
            ),
            "supporting_anchor_count_at_success": (
                success_row.get(
                    "supporting_anchor_count"
                )
            ),
            "minimum_anchor_distance_at_success": (
                success_row.get(
                    "minimum_anchor_distance"
                )
            ),
        }
    )

    return result

def run_key_enumeration(
    *,
    cfg: Mapping[str, Any],
    true_key_byte: int,
    ranked_candidates: Any = None,
) -> Dict[str, Any]:
    """
    Execute one concrete enumeration algorithm.

    Executable modes:
        local_hamming
        neighbour_sweep
        support_weighted

    ``comparison`` is a lifecycle meta-mode and must be orchestrated by
    key_enumeration_helper.py using validation evidence.

    ``resolved_local_hamming_policy`` is legal only for local_hamming mode and must already
    have been selected from validation before test execution.
    """
    contract = (
        validate_key_enumeration_contract(
            cfg
        )
    )


    if not contract["enabled"]:
        return {
            "enabled": False,
            "mode": contract["mode"],
            "executed": False,
            "success": None,
            "enumeration_position": None,
            "candidates_evaluated": 0,
            "stop_on_success": (
                contract[
                    "stop_on_success"
                ]
            ),
            "stopped_on_success": False,
        }

    mode = contract[
        "mode"
    ]

    if mode == "comparison":
        raise KeyEnumerationError(
            "comparison is a lifecycle meta-mode and cannot be executed "
            "directly by run_key_enumeration(). "
            "Comparison must be orchestrated by "
            "utils.key_enumeration_helper using validation evidence."
        )

    if mode == "local_hamming":
        if any(
            contract[field] is None
            for field in (
                "top_k",
                "hamming_radius",
                "local_budget",
            )
        ):
            raise KeyEnumerationError(
                "Validated local_hamming contract is "
                "missing required fields"
            )

        return enumerate_local_hamming(
            ranked_candidates=ranked_candidates,
            true_key_byte=true_key_byte,
            top_k=contract[
                "top_k"
            ],
            hamming_radius=contract[
                "hamming_radius"
            ],
            local_budget=contract[
                "local_budget"
            ],
            stop_on_success=contract[
                "stop_on_success"
            ],

        )

    if mode == "neighbour_sweep":
        if (
            contract[
                "hamming_radius"
            ]
            is None
        ):
            raise KeyEnumerationError(
                "Validated neighbour_sweep contract "
                "is missing hamming_radius"
            )

        return enumerate_neighbour_sweep(
            ranked_candidates=ranked_candidates,
            true_key_byte=true_key_byte,
            hamming_radius=contract[
                "hamming_radius"
            ],
            stop_on_success=contract[
                "stop_on_success"
            ],
        )

    if mode == "support_weighted":
        if any(
            contract[field] is None
            for field in (
                "top_k",
                "hamming_radius",
                "local_budget",
            )
        ):
            raise KeyEnumerationError(
                "Validated support_weighted contract is "
                "missing required fields"
            )

        return enumerate_support_weighted(
            ranked_candidates=ranked_candidates,
            true_key_byte=true_key_byte,
            top_k=contract[
                "top_k"
            ],
            hamming_radius=contract[
                "hamming_radius"
            ],
            local_budget=contract[
                "local_budget"
            ],
            stop_on_success=contract[
                "stop_on_success"
            ],
        )

    raise KeyEnumerationError(
        f"Unsupported executable enumeration mode: {mode!r}"
    )

__all__ = [
    "CROSS_VALIDATION_CONFLICT_MESSAGE",
    "KEYSPACE_SIZE",
    "KEY_WIDTH_BITS",
    "PUBLIC_RANK_BASE",
    "ENUMERATION_ALGORITHM_MODES",
    "SUPPORTED_ENUMERATION_MODES",
    "MODE_PROFILE_FIELDS",
    "KeyEnumerationError",
    "assert_enumeration_lifecycle_supported",
    "enumerate_local_hamming",
    "enumerate_neighbour_sweep",
    "enumerate_support_weighted",
    "hamming_distance",
    "run_key_enumeration",
    "validate_key_enumeration_contract",
]
