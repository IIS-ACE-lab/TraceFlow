# services/side_channel_core/tests/unit/test_code_defaults.py
"""The code's defaults for the keys a file may leave out equal the blueprint's written values (owner
2026-10-23): num_workers train 4 / val 1, prefetch_factor 4, drop_last_train true, attack_step 1,
ge_thresholds [1], key_enumeration.stop_on_success false."""

from __future__ import annotations


def test_dataloader_defaults_are_the_blueprints_values():
    from utils.dataloader_utils import _get_dataloader_cfg as resolve_dataloader_cfg

    dl_cfg, num_workers = resolve_dataloader_cfg({})
    assert (num_workers["train"], num_workers["val"]) == (4, 1)
    assert dl_cfg["drop_last_train"] is True and dl_cfg["prefetch_factor"] == 4
    _, nw = resolve_dataloader_cfg({"num_workers": {"val": 2}})
    assert (nw["train"], nw["val"]) == (4, 2)
    dl_cfg, _ = resolve_dataloader_cfg({"dataloader": {"drop_last_train": False}})
    assert dl_cfg["drop_last_train"] is False


def test_attack_defaults_are_the_blueprints_values():
    from sca.perform_attack import _get_attack_step, _get_ge_thresholds

    assert _get_attack_step({}) == 1
    assert _get_ge_thresholds({}) == [1]
    assert _get_ge_thresholds({"perform_attack": {"ge_thresholds": [1, 5]}}) == [1, 5]
