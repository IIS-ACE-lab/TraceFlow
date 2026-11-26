# services/side_channel_core/tests/unit/test_trained_settings.py
"""Key-recovery reports carry the settings the model was trained with."""

from src.sca.key_recovery import _show_enumeration, trained_settings


def _cfg(enumeration_on):
    return {
        "sca": {"key_enumeration": {"enabled": enumeration_on, "mode": "comparison"},
                "candidate_ranking": {"mode": "all"}},
        "cross_validation": {"enabled": False, "n_folds": 10},
        "perform_attack": {"test": {"attack_times": 5, "shuffle": True,
                                    "attack_distribution": "full_population"}, "attack_step": 10},
        "key_recovery_reporting": {"recovery_rank": 1, "require_stable_recovery": True},
        "evaluation": {"test": {"load_model": "final_epoch"}},
        "data": {"name": "ascad", "test_range": [2000, 10000]},
        "paths": {"params_out": "artifacts/dev/generated_models/m/config/m_params.yaml"},
    }


def test_settings_record_the_training_config():
    on = trained_settings(_cfg(True))
    assert on["key_enumeration"]["enabled"] is True and on["key_enumeration"]["mode"] == "comparison"
    assert on["perform_attack"]["test"]["attack_times"] == 5
    assert on["source"].endswith("m_params.yaml")
    assert trained_settings(_cfg(False))["key_enumeration"] == {"enabled": False}


def test_report_settings_win_over_the_current_config():
    trained_on = {"trained_settings": trained_settings(_cfg(True))}
    trained_off = {"trained_settings": trained_settings(_cfg(False))}
    assert _show_enumeration(_cfg(False), None, trained_on) is True    # params.yaml turned it off later
    assert _show_enumeration(_cfg(True), None, trained_off) is False
    assert _show_enumeration(_cfg(True), None, {}) is True              # older report: config decides
