# services/side_channel_core/tests/unit/test_timestamps.py
"""Unit tests for configured timestamps and replace-not-append history files."""

import json

import pytest

from utils import metrics_logger
from utils.seal import strip as strip_seal  # rows and registries carry their "_tfseal" (utils/seal.py)
from utils.timestamps import TimezoneConfigError, now_iso, resolve_timezone, timezone_name


@pytest.mark.parametrize(
    "location, country, zone",
    [
        ("Taipei", "Taiwan", "Asia/Taipei"),
        ("Asia/Taipei", None, "Asia/Taipei"),
        ("New York", "United States", "America/New_York"),
        ("Bochum", "Germany", "Europe/Berlin"),   # not a zone name; one clock in Germany
        ("Paris", "FR", "Europe/Paris"),          # ISO country code
    ],
)
def test_location_resolves_to_iana_zone(location, country, zone):
    assert resolve_timezone(location, country) == zone


@pytest.mark.parametrize(
    "location, country, message",
    [
        ("Springfield", "United States", "nearest of"),
        ("Taipei", "Atlantis", "Unknown country"),
    ],
)
def test_ambiguous_or_unknown_locations_fail(location, country, message):
    with pytest.raises(TimezoneConfigError, match=message):
        resolve_timezone(location, country)


def test_master_yaml_zone_is_used_for_timestamps():
    assert timezone_name() == "Asia/Taipei"
    assert now_iso().endswith("+08:00")


def _rows(path):
    return [strip_seal(json.loads(line)) for line in path.read_text().splitlines()]


def test_history_rows_are_replaced_not_appended(tmp_path):
    path = tmp_path / "m_metrics_test_history.jsonl"
    for epoch in (1, 2, 3):
        metrics_logger._upsert_epoch_row(path, {"epoch": epoch, "run": "first"})
    # Rerun of the same model: epoch 1 again drops the earlier run's rows.
    metrics_logger._upsert_epoch_row(path, {"epoch": 1, "run": "second"})
    assert _rows(path) == [{"epoch": 1, "run": "second"}]
    # Resume from epoch 2 keeps epoch 1.
    metrics_logger._upsert_epoch_row(path, {"epoch": 2, "run": "second"})
    assert [r["epoch"] for r in _rows(path)] == [1, 2]
