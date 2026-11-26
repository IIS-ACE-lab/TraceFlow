# services/side_channel_core/tests/unit/test_cli_data_entry.py
"""core.cli selects the data entry DATASET names and assumes none (the old default "ascad" failed
every project without an entry of that name: make reproduce on ZvW; audit 2026-10-23)."""
from pathlib import Path

import pytest

from utils.cli_helpers import select_data_entry

CLI = Path(__file__).resolve().parents[2] / "src" / "core" / "cli.py"


def _cfg():
    return {"data": {"ascad_n0_fs01": {"name": "ascad_n0_fs01"}, "dpav4_fs01": {"name": "dpav4_fs01"}}}


def test_the_named_entry_is_selected():
    cfg = _cfg()
    select_data_entry(cfg, "dpav4_fs01", "train-cmd")
    assert cfg["data"] == {"name": "dpav4_fs01"}


def test_no_dataset_is_refused_with_the_entries_listed():
    with pytest.raises(ValueError, match=r"train-cmd: DATASET is required.*ascad_n0_fs01.*dpav4_fs01"):
        select_data_entry(_cfg(), None, "train-cmd")


def test_an_unknown_dataset_is_refused_with_the_entries_listed():
    with pytest.raises(ValueError, match=r"dataset 'ascad' not defined.*ascad_n0_fs01"):
        select_data_entry(_cfg(), "ascad", "test-cmd")


def test_no_cli_command_defaults_the_dataset():
    text = CLI.read_text(encoding="utf-8")
    assert 'typer.Option("ascad"' not in text
    assert text.count('dataset: Optional[str] = typer.Option(None') == 4, "train, test, gen-data, orchestrate"
    assert "from utils.cli_helpers import select_data_entry" in text
