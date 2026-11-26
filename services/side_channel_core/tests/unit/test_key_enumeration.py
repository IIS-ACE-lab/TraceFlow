"""Key enumeration: lifecycle gate, retired switch, reported rank, registries."""

import json

import pytest

from services.side_channel_core.src.sca.key_enumeration import (
    ENUMERATION_ALGORITHM_MODES,
    KeyEnumerationError,
    assert_enumeration_lifecycle_supported,
    enumerate_local_hamming,
    enumerate_neighbour_sweep,
    enumerate_support_weighted,
    validate_key_enumeration_contract,
)
from services.side_channel_core.src.sca.key_recovery import key_recovery_outcome
from services.side_channel_core.utils.lifecycle_outcome import update_lifecycle_registry_outcome
from services.side_channel_core.utils.key_enumeration_helper import _apply_reported_rank


PROFILES = {
    "local_hamming": {"top_k": 1, "hamming_radius": 1, "local_budget": 3},
    "neighbour_sweep": {"hamming_radius": 1},
    "support_weighted": {"top_k": 2, "hamming_radius": 1, "local_budget": 8},
}


def _cfg(*, enumeration=True, cross_validation=False, **overrides):
    key_enumeration = {
        "enabled": enumeration,
        "mode": "comparison",
        "stop_on_success": True,
        "modes": PROFILES,
        **overrides,
    }
    return {
        "sca": {"candidate_ranking": {"mode": "all"}, "key_enumeration": key_enumeration},
        "cross_validation": {"enabled": cross_validation},
    }


def test_mode_names_state_their_method():
    assert ENUMERATION_ALGORITHM_MODES == {"local_hamming", "neighbour_sweep", "support_weighted"}


def test_cross_validation_and_enumeration_fail_fast():
    with pytest.raises(KeyEnumerationError, match="disable cross_validation"):
        assert_enumeration_lifecycle_supported(_cfg(cross_validation=True))
    with pytest.raises(KeyEnumerationError, match="disable cross_validation"):
        validate_key_enumeration_contract(_cfg(cross_validation=True))

    # Either one alone is fine.
    assert_enumeration_lifecycle_supported(_cfg(cross_validation=False))
    assert_enumeration_lifecycle_supported(_cfg(enumeration=False, cross_validation=True))


def test_best_rank_is_retired_loudly():
    with pytest.raises(KeyEnumerationError, match="best_rank was retired"):
        validate_key_enumeration_contract(_cfg(best_rank=True))
    contract = validate_key_enumeration_contract(_cfg())
    assert "best_rank" not in contract
    assert sorted(contract["comparison_profiles"]) == sorted(PROFILES)


@pytest.mark.parametrize(
    "ai_rank, position, reported, source",
    [
        (5, 5, 5, "MODEL"),                       # tie -> the model
        (5, 9, 5, "MODEL"),                       # enumeration worse -> never the headline
        (9, 5, 5, "MODEL+KEY_ENUMERATION"),       # strictly better -> enumeration
        (1, 1, 1, "MODEL"),
    ],
)
def test_reported_rank_is_the_better_one_with_ties_to_the_model(ai_rank, position, reported, source):
    enumeration = _apply_reported_rank(
        {"ai_rank_at_success": ai_rank, "enumeration_position": position}
    )
    assert (enumeration["key_rank_reported"], enumeration["key_rank_source"]) == (reported, source)
    assert enumeration["key_recovery"] == ("DONE" if reported == 1 else "FAILED")
    # Both raw ranks survive untouched.
    assert enumeration["ai_rank_at_success"] == ai_rank
    assert enumeration["enumeration_position"] == position


def test_every_mode_orders_all_256_candidates_without_the_true_key():
    ranked = [{"rank": i + 1, "key_guess": (i * 7 + 3) % 256, "score": -float(i)} for i in range(256)]
    true_key = ranked[200]["key_guess"]
    common = dict(ranked_candidates=ranked, true_key_byte=true_key, stop_on_success=False)
    results = {
        "local_hamming": enumerate_local_hamming(top_k=1, hamming_radius=1, local_budget=3, **common),
        "neighbour_sweep": enumerate_neighbour_sweep(hamming_radius=1, **common),
        "support_weighted": enumerate_support_weighted(top_k=2, hamming_radius=1, local_budget=8, **common),
    }
    for mode, result in results.items():
        order = [row["key_guess"] for row in result["evaluated_candidates"]]
        assert sorted(order) == list(range(256)), mode
        assert result["enumeration_position"] == order.index(true_key) + 1, mode


def test_registries_record_the_outcome_after_enumeration(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "test_registry.json").write_text(json.dumps({"abcd1234": {"status": "DONE"}}))
    (artifacts / "run_registry.json").write_text(json.dumps({"abcd1234": {"status": "DONE"}}))

    test_fields = {
        "key_recovery": "FAILED", "key_recovered_through": "MODEL+KEY_ENUMERATION",
        "final_true_key_rank": 245, "key_enumeration_position": 162,
        "key_rank_reported": 162, "attack_byte": 2,
        "accuracy": 0.9,                      # not a key-recovery field: ignored
    }
    update_lifecycle_registry_outcome(
        registry_path=artifacts / "test_registry.json", model_hash="abcd1234", fields=test_fields,
    )
    update_lifecycle_registry_outcome(
        registry_path=artifacts / "run_registry.json", model_hash="abcd1234",
        fields={"key_recovery": "DONE", "key_recovered_through": "MODEL",
                "key_rank_reported": 1, "attack_byte": 2, "key_enumeration_position": 1},
    )

    test_entry = json.loads((artifacts / "test_registry.json").read_text())["abcd1234"]
    assert test_entry["status"] == "DONE" and test_entry["key_recovery"] == "FAILED"
    assert test_entry["key_recovered_through"] == "MODEL+KEY_ENUMERATION"
    assert test_entry["key_rank_reported"] == 162 and test_entry["attack_byte"] == 2
    assert "key_enumeration_position" not in test_entry     # lifecycle registries stay small
    assert "accuracy" not in test_entry

    run_entry = json.loads((artifacts / "run_registry.json").read_text())["abcd1234"]
    assert run_entry["key_recovery"] == "DONE" and run_entry["key_rank_reported"] == 1


def test_outcome_without_enumeration_comes_from_the_model_rank():
    report = {"key": {"final_true_key_rank": 236, "recovered": False}, "target": {"attack_byte": 2}}
    assert key_recovery_outcome(report) == {
        "key_recovery": "FAILED", "key_recovered_through": "MODEL",
        "final_true_key_rank": 236, "key_enumeration_position": None,
        "key_rank_reported": 236, "attack_byte": 2,
        "key_recovered": False, "true_key": None, "recovered_key": None, "recovered_key_rank": 236,
    }
    recovered = {"key": {"final_true_key_rank": 1, "recovered": True}, "target": {"attack_byte": 2}}
    assert key_recovery_outcome(recovered)["key_recovery"] == "DONE"
    # Rank 1 that is tied or not stable is not a recovery (report.key.recovered).
    tied = {"key": {"final_true_key_rank": 1, "recovered": False}, "target": {"attack_byte": 2}}
    assert key_recovery_outcome(tied)["key_recovery"] == "FAILED"

    enumerated = {
        "key": {"final_true_key_rank": 245}, "target": {"attack_byte": 2},
        "enumeration": {"key_rank_reported": 162, "key_rank_source": "MODEL+KEY_ENUMERATION",
                        "key_recovery": "FAILED", "enumeration_position": 162},
    }
    outcome = key_recovery_outcome(enumerated)
    assert outcome["key_rank_reported"] == 162 and outcome["final_true_key_rank"] == 245
    assert outcome["key_recovered_through"] == "MODEL+KEY_ENUMERATION"
