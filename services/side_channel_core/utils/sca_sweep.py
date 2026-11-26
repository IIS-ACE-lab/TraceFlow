# services/side_channel_core/utils/sca_sweep.py

"""
Conditional SCA sweep/search-space compiler.

This module is the canonical search-space layer for Side-Channel Analysis
configuration shared by:

    - SCC grid sweeps;
    - AutoSearch / OptunaLite;
    - full AutoMLOps trial compilation.

It converts list-valued SCA configuration into concrete runtime SCA contracts
without allowing inactive conditional parameters to multiply the experiment
space.

Example search configuration
----------------------------

sca:
  attack_byte: [0, 3]
  leakage_model: ["ID", "HW"]

  candidate_ranking:
    mode: [all, top_k]
    top_k: [5, 10]

  key_enumeration:
    enabled: true
    mode: [local_hamming, neighbour_sweep, support_weighted]
    stop_on_success: true

    modes:
      local_hamming:
        top_k: [1, 5]
        hamming_radius: [1, 2]
        local_budget: [8, 16]

      neighbour_sweep:
        hamming_radius: [1, 2]

      support_weighted:
        top_k: [5, 10]
        hamming_radius: [1, 2]
        local_budget: [8, 16]

Conditional semantics
---------------------

candidate_ranking.mode=all
    top_k is inactive and must not multiply the search space.

candidate_ranking.mode=none
    top_k is inactive and must not multiply the search space.

candidate_ranking.mode=top_k
    top_k becomes an active search dimension.

key_enumeration.mode=local_hamming
    only modes.local_hamming fields become active.

key_enumeration.mode=neighbour_sweep
    only modes.neighbour_sweep fields become active.

key_enumeration.mode=support_weighted
    only modes.support_weighted fields become active.

The output of this module is always concrete and runtime-safe:
- no active sweep lists remain;
- only the selected key-enumeration profile is retained;
- inactive conditional configuration cannot influence model identity;
- key_enumeration.py remains runtime-only.

key_enumeration.mode=comparison
    comparison is a post-ranking meta-mode.

    One concrete comparison configuration retains one concrete profile for
    each real enumeration algorithm:

        local_hamming
        neighbour_sweep
        support_weighted

    All three algorithms are benchmarked against the same validation ranking.
    The validation winner is frozen and evaluated independently on test
    evidence.

    comparison itself does not own modes.comparison.
Author: Shraddha Mishra
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Dict, List, Optional

from services.side_channel_core.src.sca.key_enumeration import (
    MODE_PROFILE_FIELDS,
    SUPPORTED_ENUMERATION_MODES, ENUMERATION_ALGORITHM_MODES,
)


Json = Dict[str, Any]

SUPPORTED_CANDIDATE_RANKING_MODES = frozenset(
    {
        "all",
        "top_k",
        "none",
    }
)

SCA_DIRECT_SWEEP_FIELDS = frozenset(
    {
        "attack_byte",
        "leakage_model",
    }
)


class SCASweepError(ValueError):
    """Raised when SCA search/sweep configuration is invalid."""


def _is_primitive(
    value: Any,
) -> bool:
    """Return True for scalar values supported as finite sweep choices."""
    return (
        value is None
        or isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        )
    )


def _as_choices(
    value: Any,
    *,
    field: str,
    allow_empty: bool = False,
) -> List[Any]:
    """
    Normalize one scalar-or-list search field to a list of choices.

    A scalar becomes a one-element choice list.

    Nested lists/dicts are rejected because this module owns only finite
    primitive parameter choices.
    """
    if isinstance(
        value,
        list,
    ):
        if (
            not value
            and not allow_empty
        ):
            raise SCASweepError(
                f"{field} sweep list must not be empty"
            )

        for index, item in enumerate(
            value
        ):
            if not _is_primitive(
                item
            ):
                raise SCASweepError(
                    f"{field}[{index}] must be a primitive scalar"
                )

        return copy.deepcopy(
            value
        )

    if not _is_primitive(
        value
    ):
        raise SCASweepError(
            f"{field} must be a primitive scalar or list of primitives"
        )

    return [
        copy.deepcopy(
            value
        )
    ]


def _unique_choices(
    values: Sequence[Any],
) -> List[Any]:
    """
    Preserve declaration order while removing duplicate primitive choices.

    Duplicate values should not create duplicate experiments.
    """
    out: List[Any] = []

    for value in values:
        if value not in out:
            out.append(
                copy.deepcopy(
                    value
                )
            )

    return out


def _string_choices(
    value: Any,
    *,
    field: str,
    allowed: frozenset[str],
) -> List[str]:
    """Normalize one finite string choice field."""
    raw_choices = _as_choices(
        value,
        field=field,
    )

    normalized: List[str] = []

    for index, item in enumerate(
        raw_choices
    ):
        if (
            not isinstance(
                item,
                str,
            )
            or not item.strip()
        ):
            raise SCASweepError(
                f"{field}[{index}] must be a non-empty string"
            )

        built = (
            item.strip()
            .lower()
        )

        if built not in allowed:
            supported = ", ".join(
                sorted(
                    allowed
                )
            )

            raise SCASweepError(
                f"{field}[{index}] must be one of: "
                f"{supported}; got {built!r}"
            )

        normalized.append(
            built
        )

    return _unique_choices(
        normalized
    )


def _positive_int_choices(
    value: Any,
    *,
    field: str,
    minimum: int = 1,
    maximum: Optional[int] = None,
) -> List[int]:
    """Normalize one positive finite integer search dimension."""
    raw_choices = _as_choices(
        value,
        field=field,
    )

    normalized: List[int] = []

    for index, item in enumerate(
        raw_choices
    ):
        if isinstance(
            item,
            bool,
        ):
            raise SCASweepError(
                f"{field}[{index}] must be an integer"
            )

        try:
            built = int(
                item
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise SCASweepError(
                f"{field}[{index}] must be an integer"
            ) from exc

        if built < minimum:
            raise SCASweepError(
                f"{field}[{index}] must be >= {minimum}"
            )

        if (
            maximum is not None
            and built > maximum
        ):
            raise SCASweepError(
                f"{field}[{index}] must be <= {maximum}"
            )

        normalized.append(
            built
        )

    return _unique_choices(
        normalized
    )


def _normalize_attack_byte_choices(
    sca_cfg: Mapping[str, Any],
) -> List[int]:
    """Resolve attack_byte sweep choices."""
    raw = sca_cfg.get(
        "attack_byte"
    )

    if raw is None:
        return []

    return _positive_int_choices(
        raw,
        field="sca.attack_byte",
        minimum=0,
        maximum=255,
    )


def _normalize_leakage_choices(
    sca_cfg: Mapping[str, Any],
) -> List[str]:
    """Resolve leakage-model sweep choices."""
    raw = sca_cfg.get(
        "leakage_model"
    )

    if raw is None:
        return []

    choices = _as_choices(
        raw,
        field="sca.leakage_model",
    )

    normalized: List[str] = []

    for index, value in enumerate(
        choices
    ):
        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            raise SCASweepError(
                "sca.leakage_model"
                f"[{index}] must be a non-empty string"
            )

        normalized.append(
            value.strip()
        )

    return _unique_choices(
        normalized
    )


def _candidate_ranking_variants(
    sca_cfg: Mapping[str, Any],
) -> List[Json]:
    """
    Compile conditional candidate-ranking variants.

    Correct behavior:

        mode=all
            -> {"mode": "all"}

        mode=none
            -> {"mode": "none"}

        mode=top_k
            -> one variant per top_k choice

    Therefore top_k never creates duplicate ``all`` or ``none`` experiments.
    """
    raw = (
        sca_cfg.get(
            "candidate_ranking"
        )
        or {}
    )

    if not isinstance(
        raw,
        Mapping,
    ):
        raise SCASweepError(
            "sca.candidate_ranking must be a mapping"
        )

    mode_raw = raw.get(
        "mode",
        "top_k",
    )

    modes = _string_choices(
        mode_raw,
        field=(
            "sca.candidate_ranking.mode"
        ),
        allowed=(
            SUPPORTED_CANDIDATE_RANKING_MODES
        ),
    )

    top_k_raw = raw.get(
        "top_k"
    )

    variants: List[
        Json
    ] = []

    for mode in modes:
        if mode in {
            "all",
            "none",
        }:
            variants.append(
                {
                    "mode": mode,
                }
            )

            continue

        if top_k_raw is None:
            raise SCASweepError(
                "sca.candidate_ranking.top_k is required "
                "when candidate_ranking.mode includes 'top_k'"
            )

        top_k_choices = (
            _positive_int_choices(
                top_k_raw,
                field=(
                    "sca.candidate_ranking.top_k"
                ),
                minimum=1,
                maximum=256,
            )
        )

        for top_k in top_k_choices:
            variants.append(
                {
                    "mode": "top_k",
                    "top_k": top_k,
                }
            )

    if not variants:
        raise SCASweepError(
            "No candidate-ranking variants were produced"
        )

    return variants


def _normalize_enumeration_profile_choices(
    *,
    mode: str,
    profile: Mapping[str, Any],
) -> Dict[str, List[int]]:
    """
    Normalize one mode-specific key-enumeration search profile.

    Only fields declared in MODE_PROFILE_FIELDS are legal.
    """
    if mode not in ENUMERATION_ALGORITHM_MODES:
        raise SCASweepError(
            f"Unsupported key-enumeration mode: {mode!r}"
        )

    if not isinstance(
        profile,
        Mapping,
    ):
        raise SCASweepError(
            f"sca.key_enumeration.modes.{mode} must be a mapping"
        )

    allowed_fields = (
        MODE_PROFILE_FIELDS[
            mode
        ]
    )

    unknown = sorted(
        set(profile)
        - allowed_fields
    )

    if unknown:
        rendered = ", ".join(
            f"sca.key_enumeration.modes.{mode}.{field}"
            for field in unknown
        )

        raise SCASweepError(
            f"Unsupported key-enumeration field(s): {rendered}"
        )

    missing = sorted(
        allowed_fields
        - set(profile)
    )

    if missing:
        rendered = ", ".join(
            f"sca.key_enumeration.modes.{mode}.{field}"
            for field in missing
        )

        raise SCASweepError(
            f"Missing key-enumeration field(s): {rendered}"
        )

    normalized: Dict[
        str,
        List[int],
    ] = {}

    for field in allowed_fields:
        if field == "hamming_radius":
            maximum = 8
        else:
            maximum = 255

        normalized[field] = (
            _positive_int_choices(
                profile[field],
                field=(
                    "sca.key_enumeration."
                    f"modes.{mode}.{field}"
                ),
                minimum=1,
                maximum=maximum,
            )
        )

    return normalized


def _profile_cartesian_variants(
    *,
    mode: str,
    choices: Mapping[
        str,
        Sequence[int],
    ],
) -> List[Json]:
    """Build concrete parameter dictionaries for one active mode."""
    
    if mode not in ENUMERATION_ALGORITHM_MODES:
        raise SCASweepError(
            "Enumeration profile Cartesian expansion "
            "supports only local_hamming, neighbour_sweep, and support_weighted; "
            f"got {mode!r}"
        )

    fields = sorted(
        MODE_PROFILE_FIELDS[
            mode
        ]
    )

    pools = [
        list(
            choices[field]
        )
        for field in fields
    ]

    variants: List[
        Json
    ] = []

    for values in itertools.product(
        *pools
    ):
        profile = {
            field: int(value)
            for field, value
            in zip(
                fields,
                values,
            )
        }

        #
        # local_hamming-specific structural invariant:
        #
        # local_budget may not exceed the number of non-anchor hypotheses.
        #
        if mode == "local_hamming":
            top_k = int(
                profile[
                    "top_k"
                ]
            )

            local_budget = int(
                profile[
                    "local_budget"
                ]
            )

            if (
                local_budget
                > 256 - top_k
            ):
                continue

        variants.append(
            profile
        )

    if not variants:
        raise SCASweepError(
            f"No valid concrete profile combinations for mode={mode!r}"
        )

    return variants

def _comparison_enumeration_variants(
    *,
    profiles_raw: Mapping[str, Any],
    stop_on_success: bool,
) -> List[Json]:
    """
    Compile concrete comparison-mode enumeration configurations.

    ``comparison`` is a post-ranking meta-mode. It has no profile of its own.

    Instead, one concrete comparison configuration contains one concrete
    profile for each real enumeration algorithm:

        local_hamming
        neighbour_sweep
        support_weighted

    Parameter lists inside those three profiles retain normal sweep
    capability. Their Cartesian product produces alternative comparison
    policies, but comparison itself never expands into three independent
    enumeration modes.

    With singleton lists, exactly one comparison variant is produced.
    """
    required_modes = (
        "local_hamming",
        "neighbour_sweep",
        "support_weighted",
    )

    missing_modes = [
        mode
        for mode in required_modes
        if mode not in profiles_raw
    ]

    if missing_modes:
        rendered = ", ".join(
            (
                "sca.key_enumeration."
                f"modes.{mode}"
            )
            for mode in missing_modes
        )

        raise SCASweepError(
            "key_enumeration.mode='comparison' requires "
            "profiles for local_hamming, neighbour_sweep, and support_weighted. "
            f"Missing: {rendered}"
        )

    algorithm_variants: Dict[
        str,
        List[Json],
    ] = {}

    for mode in required_modes:
        choices = (
            _normalize_enumeration_profile_choices(
                mode=mode,
                profile=profiles_raw[
                    mode
                ],
            )
        )

        algorithm_variants[
            mode
        ] = (
            _profile_cartesian_variants(
                mode=mode,
                choices=choices,
            )
        )

    variants: List[
        Json
    ] = []

    for (
        hybrid_profile,
        iterative_profile,
        support_profile,
    ) in itertools.product(
        algorithm_variants[
            "local_hamming"
        ],
        algorithm_variants[
            "neighbour_sweep"
        ],
        algorithm_variants[
            "support_weighted"
        ],
    ):
        variants.append(
            {
                "enabled": True,
                "mode": "comparison",
                "stop_on_success": (
                    stop_on_success
                ),
                "modes": {
                    "local_hamming": copy.deepcopy(
                        hybrid_profile
                    ),
                    "neighbour_sweep": copy.deepcopy(
                        iterative_profile
                    ),
                    "support_weighted": copy.deepcopy(
                        support_profile
                    ),
                },
            }
        )

    if not variants:
        raise SCASweepError(
            "No valid concrete comparison "
            "enumeration variants were produced"
        )

    return variants

def _key_enumeration_variants(
    sca_cfg: Mapping[str, Any],
) -> List[Json]:
    """
    Compile conditional key-enumeration variants.

    Ordinary modes
    --------------
    local_hamming / neighbour_sweep / support_weighted:

        Only the selected algorithm profile is active and survives into the
        concrete runtime configuration.

    Comparison mode
    ---------------
    comparison is a post-ranking meta-mode.

    It retains one concrete profile for each real algorithm:

        local_hamming
        neighbour_sweep
        support_weighted

    so that one trained model can benchmark all three algorithms on the same
    validation ranking.

    Parameter lists remain valid sweep dimensions. Singleton lists therefore
    produce one concrete policy, while multi-value lists intentionally produce
    multiple concrete variants.
    """
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
        raise SCASweepError(
            "sca.key_enumeration must be a mapping"
        )

    enabled = raw.get(
        "enabled",
        False,
    )

    if isinstance(
        enabled,
        list,
    ):
        raise SCASweepError(
            "sca.key_enumeration.enabled is not a sweep dimension; "
            "use one boolean value"
        )

    if not isinstance(
        enabled,
        bool,
    ):
        raise SCASweepError(
            "sca.key_enumeration.enabled must be boolean"
        )

    stop_on_success = raw.get(
        "stop_on_success",
        True,
    )

    if isinstance(
        stop_on_success,
        list,
    ):
        raise SCASweepError(
            "sca.key_enumeration.stop_on_success is not currently "
            "a sweep dimension; use one boolean value"
        )

    if not isinstance(
        stop_on_success,
        bool,
    ):
        raise SCASweepError(
            "sca.key_enumeration.stop_on_success must be boolean"
        )

    if not enabled:
        return [
            {
                "enabled": False,
                "stop_on_success": (
                    stop_on_success
                ),
            }
        ]

    mode_raw = raw.get(
        "mode"
    )

    if mode_raw is None:
        raise SCASweepError(
            "sca.key_enumeration.mode is required "
            "when enabled=true"
        )

    modes = _string_choices(
        mode_raw,
        field=(
            "sca.key_enumeration.mode"
        ),
        allowed=(
            SUPPORTED_ENUMERATION_MODES
        ),
    )

    profiles_raw = (
        raw.get(
            "modes"
        )
        or {}
    )

    if not isinstance(
        profiles_raw,
        Mapping,
    ):
        raise SCASweepError(
            "sca.key_enumeration.modes must be a mapping"
        )

    #
    # Only real enumeration algorithms own profiles.
    #
    # modes.comparison is intentionally forbidden.
    #
    unknown_profiles = sorted(
        set(
            profiles_raw
        )
        - ENUMERATION_ALGORITHM_MODES
    )

    if unknown_profiles:
        rendered = ", ".join(
            str(mode)
            for mode in unknown_profiles
        )

        raise SCASweepError(
            "Unsupported profile(s) under "
            "sca.key_enumeration.modes: "
            f"{rendered}. "
            "'comparison' is a meta-mode and does not "
            "own modes.comparison."
        )

    variants: List[
        Json
    ] = []

    for mode in modes:
        # =========================================================
        # COMPARISON META-MODE
        # =========================================================
        if mode == "comparison":
            variants.extend(
                _comparison_enumeration_variants(
                    profiles_raw=(
                        profiles_raw
                    ),
                    stop_on_success=(
                        stop_on_success
                    ),
                )
            )

            continue

        # =========================================================
        # ORDINARY ALGORITHM MODE
        # =========================================================
        if (
            mode
            not in ENUMERATION_ALGORITHM_MODES
        ):
            raise SCASweepError(
                "Unsupported ordinary key-enumeration "
                f"algorithm: {mode!r}"
            )

        if mode not in profiles_raw:
            raise SCASweepError(
                "Missing selected key-enumeration profile: "
                f"sca.key_enumeration.modes.{mode}"
            )

        choices = (
            _normalize_enumeration_profile_choices(
                mode=mode,
                profile=profiles_raw[
                    mode
                ],
            )
        )

        profiles = (
            _profile_cartesian_variants(
                mode=mode,
                choices=choices,
            )
        )

        for concrete_profile in profiles:
            variants.append(
                {
                    "enabled": True,
                    "mode": mode,
                    "stop_on_success": (
                        stop_on_success
                    ),
                    "modes": {
                        mode: copy.deepcopy(
                            concrete_profile
                        )
                    },
                }
            )

    if not variants:
        raise SCASweepError(
            "No concrete key-enumeration variants were produced"
        )

    return variants

def build_sca_search_space(
    sca_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Return normalized SCA search metadata.

    This representation is intentionally explicit rather than pretending every
    nested YAML list is an independent Cartesian dimension.
    """
    if not isinstance(
        sca_cfg,
        Mapping,
    ):
        raise SCASweepError(
            "sca must be a mapping"
        )

    attack_bytes = (
        _normalize_attack_byte_choices(
            sca_cfg
        )
    )

    leakage_models = (
        _normalize_leakage_choices(
            sca_cfg
        )
    )

    candidate_variants = (
        _candidate_ranking_variants(
            sca_cfg
        )
    )

    enumeration_variants = (
        _key_enumeration_variants(
            sca_cfg
        )
    )

    return {
        "attack_byte": (
            attack_bytes
        ),
        "leakage_model": (
            leakage_models
        ),
        "candidate_ranking_variants": (
            candidate_variants
        ),
        "key_enumeration_variants": (
            enumeration_variants
        ),
    }


def iter_sca_combinations(
    sca_cfg: Mapping[str, Any],
) -> Iterator[Json]:
    """
    Yield concrete SCA configurations using conditional Cartesian semantics.

    Non-sweep SCA fields are preserved unchanged.
    """
    if not isinstance(
        sca_cfg,
        Mapping,
    ):
        raise SCASweepError(
            "sca must be a mapping"
        )

    search = (
        build_sca_search_space(
            sca_cfg
        )
    )

    base = copy.deepcopy(
        dict(
            sca_cfg
        )
    )

    #
    # These fields are rebuilt from concrete choices below.
    #
    base.pop(
        "attack_byte",
        None,
    )

    base.pop(
        "leakage_model",
        None,
    )

    base.pop(
        "candidate_ranking",
        None,
    )

    base.pop(
        "key_enumeration",
        None,
    )

    attack_choices = (
        search[
            "attack_byte"
        ]
        or [
            None
        ]
    )

    leakage_choices = (
        search[
            "leakage_model"
        ]
        or [
            None
        ]
    )

    candidate_choices = search[
        "candidate_ranking_variants"
    ]

    enumeration_choices = search[
        "key_enumeration_variants"
    ]

    for (
        attack_byte,
        leakage_model,
        candidate_ranking,
        key_enumeration,
    ) in itertools.product(
        attack_choices,
        leakage_choices,
        candidate_choices,
        enumeration_choices,
    ):
        concrete = copy.deepcopy(
            base
        )

        if attack_byte is not None:
            concrete[
                "attack_byte"
            ] = int(
                attack_byte
            )

        if leakage_model is not None:
            concrete[
                "leakage_model"
            ] = str(
                leakage_model
            )

        concrete[
            "candidate_ranking"
        ] = copy.deepcopy(
            candidate_ranking
        )

        concrete[
            "key_enumeration"
        ] = copy.deepcopy(
            key_enumeration
        )

        #
        # Scientific compatibility invariant:
        #
        # enabled key enumeration requires the complete AI candidate ranking.
        #
        if bool(
            key_enumeration.get(
                "enabled",
                False,
            )
        ):
            ranking_mode = str(
                candidate_ranking.get(
                    "mode",
                    "",
                )
            ).strip().lower()

            if ranking_mode != "all":
                continue

        yield concrete


def expand_sca_sweep(
    cfg: Mapping[str, Any],
) -> List[Json]:
    """
    Expand a full project configuration across SCA dimensions only.

    Non-SCA fields are copied unchanged.

    This is suitable for SCC's deterministic grid layer.
    """
    if not isinstance(
        cfg,
        Mapping,
    ):
        raise SCASweepError(
            "cfg must be a mapping"
        )

    sca_cfg = (
        cfg.get(
            "sca"
        )
        or {}
    )

    if not isinstance(
        sca_cfg,
        Mapping,
    ):
        raise SCASweepError(
            "cfg.sca must be a mapping"
        )

    out: List[
        Json
    ] = []

    for concrete_sca in (
        iter_sca_combinations(
            sca_cfg
        )
    ):
        concrete_cfg = copy.deepcopy(
            dict(
                cfg
            )
        )

        concrete_cfg[
            "sca"
        ] = concrete_sca

        out.append(
            concrete_cfg
        )

    if not out:
        raise SCASweepError(
            "SCA sweep produced no valid concrete configurations"
        )

    return out


def count_sca_combinations(
    sca_cfg: Mapping[str, Any],
) -> int:
    """Count valid conditional SCA combinations."""
    return sum(
        1
        for _ in iter_sca_combinations(
            sca_cfg
        )
    )


def extract_sca_sample(
    concrete_sca: Mapping[str, Any],
) -> Json:
    """
    Return the reproducibility/identity payload for one concrete SCA run.

    Only scientifically active SCA choices are included.
    """
    if not isinstance(
        concrete_sca,
        Mapping,
    ):
        raise SCASweepError(
            "concrete_sca must be a mapping"
        )

    out: Json = {}

    for field in (
        "variant",
        "attack_byte",
        "leakage_model",
        "rank_base",
    ):
        if field in concrete_sca:
            out[field] = copy.deepcopy(
                concrete_sca[
                    field
                ]
            )

    ranking = (
        concrete_sca.get(
            "candidate_ranking"
        )
        or {}
    )

    if isinstance(
        ranking,
        Mapping,
    ):
        out[
            "candidate_ranking"
        ] = copy.deepcopy(
            dict(
                ranking
            )
        )

    enumeration = (
        concrete_sca.get(
            "key_enumeration"
        )
        or {}
    )

    if isinstance(
        enumeration,
        Mapping,
    ):
        out[
            "key_enumeration"
        ] = copy.deepcopy(
            dict(
                enumeration
            )
        )

    return out


def assert_concrete_sca_runtime(
    sca_cfg: Mapping[str, Any],
) -> None:
    """
    Fail if an active runtime SCA field still contains an unresolved list.

    This is intentionally stricter than generic recursive list detection:
    inactive/unrelated configuration is not interpreted here.
    """
    if not isinstance(
        sca_cfg,
        Mapping,
    ):
        raise SCASweepError(
            "sca must be a mapping"
        )

    for field in (
        "attack_byte",
        "leakage_model",
    ):
        if isinstance(
            sca_cfg.get(
                field
            ),
            list,
        ):
            raise SCASweepError(
                f"sca.{field} is still an unresolved sweep list"
            )

    candidate_ranking = (
        sca_cfg.get(
            "candidate_ranking"
        )
        or {}
    )

    if isinstance(
        candidate_ranking,
        Mapping,
    ):
        for field in (
            "mode",
            "top_k",
        ):
            if isinstance(
                candidate_ranking.get(
                    field
                ),
                list,
            ):
                raise SCASweepError(
                    "sca.candidate_ranking."
                    f"{field} is still an unresolved sweep list"
                )

    key_enumeration = (
        sca_cfg.get(
            "key_enumeration"
        )
        or {}
    )

    if not isinstance(
        key_enumeration,
        Mapping,
    ):
        raise SCASweepError(
            "sca.key_enumeration must be a mapping"
        )

    mode = key_enumeration.get(
        "mode"
    )

    if isinstance(
        mode,
        list,
    ):
        raise SCASweepError(
            "sca.key_enumeration.mode is still an unresolved sweep list"
        )

    enabled = bool(
        key_enumeration.get(
            "enabled",
            False,
        )
    )

    if not enabled:
        return

    if (
        not isinstance(
            mode,
            str,
        )
        or mode.strip().lower()
        not in SUPPORTED_ENUMERATION_MODES
    ):
        raise SCASweepError(
            "Enabled key enumeration requires one concrete mode"
        )

    normalized_mode = (
        mode.strip().lower()
    )

    modes = (
        key_enumeration.get(
            "modes"
        )
        or {}
    )

    if not isinstance(
        modes,
        Mapping,
    ):
        raise SCASweepError(
            "sca.key_enumeration.modes must be a mapping"
        )

    if normalized_mode == "comparison":
        required_modes = (
            "local_hamming",
            "neighbour_sweep",
            "support_weighted",
        )

        for algorithm_mode in required_modes:
            profile = modes.get(
                algorithm_mode
            )

            if not isinstance(
                profile,
                Mapping,
            ):
                raise SCASweepError(
                    "Concrete comparison runtime config "
                    "is missing "
                    "sca.key_enumeration."
                    f"modes.{algorithm_mode}"
                )

            allowed_fields = (
                MODE_PROFILE_FIELDS[
                    algorithm_mode
                ]
            )

            unknown_fields = sorted(
                set(
                    profile
                )
                - allowed_fields
            )

            if unknown_fields:
                rendered = ", ".join(
                    (
                        "sca.key_enumeration."
                        f"modes.{algorithm_mode}.{field}"
                    )
                    for field
                    in unknown_fields
                )

                raise SCASweepError(
                    "Unsupported concrete comparison "
                    f"profile field(s): {rendered}"
                )

            missing_fields = sorted(
                allowed_fields
                - set(
                    profile
                )
            )

            if missing_fields:
                rendered = ", ".join(
                    (
                        "sca.key_enumeration."
                        f"modes.{algorithm_mode}.{field}"
                    )
                    for field
                    in missing_fields
                )

                raise SCASweepError(
                    "Concrete comparison runtime config "
                    f"is missing field(s): {rendered}"
                )

            for field in allowed_fields:
                if isinstance(
                    profile.get(
                        field
                    ),
                    list,
                ):
                    raise SCASweepError(
                        "sca.key_enumeration."
                        f"modes.{algorithm_mode}.{field} "
                        "is still an unresolved sweep list"
                    )

        return

    if (
        normalized_mode
        not in ENUMERATION_ALGORITHM_MODES
    ):
        raise SCASweepError(
            "Unsupported concrete key-enumeration "
            f"algorithm: {normalized_mode!r}"
        )

    profile = modes.get(
        normalized_mode
    )

    if not isinstance(
        profile,
        Mapping,
    ):
        raise SCASweepError(
            "Concrete key-enumeration runtime config "
            f"is missing modes.{normalized_mode}"
        )

    for field in (
        MODE_PROFILE_FIELDS[
            normalized_mode
        ]
    ):
        if isinstance(
            profile.get(
                field
            ),
            list,
        ):
            raise SCASweepError(
                "sca.key_enumeration."
                f"modes.{normalized_mode}.{field} "
                "is still an unresolved sweep list"
            )


__all__ = [
    "SCASweepError",
    "SUPPORTED_CANDIDATE_RANKING_MODES",
    "SCA_DIRECT_SWEEP_FIELDS",
    "assert_concrete_sca_runtime",
    "build_sca_search_space",
    "count_sca_combinations",
    "expand_sca_sweep",
    "extract_sca_sample",
    "iter_sca_combinations",
]
