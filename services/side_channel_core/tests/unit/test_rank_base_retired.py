# services/side_channel_core/tests/unit/test_rank_base_retired.py
"""sca.rank_base is no longer a setting (TraceFlow ranks from 1): a config without it runs every
stage that checks the rank contract; one naming another base is refused."""
import copy
from pathlib import Path

import pytest

from services.side_channel_core.src.sca.key_recovery import KeyRecoveryError, validate_key_recovery_contract
from services.side_channel_core.utils.config_loader import get_project_config


@pytest.fixture(scope="module")
def cfg():
    """A run's config as training gets it (a DONE run's snapshot), without sca.rank_base."""
    from services.side_channel_core.utils.data_reproducibility import read_params_snapshot
    from services.side_channel_core.utils.seal import load_json
    root = Path(get_project_config("DLSCA_ASCAD", validate_data=False)["project"]["config_root"])
    registry = load_json(root / "artifacts" / "run_registry.json")
    for entry in registry.values():
        if isinstance(entry, dict) and str(entry.get("status")).upper() == "DONE":
            name, env = entry["model_name"], entry.get("env") or "dev"
            snap = root / "artifacts" / env / "generated_models" / name / "config" / f"{name}_params.yaml"
            if snap.is_file():
                cfg = copy.deepcopy(read_params_snapshot(snap))
                cfg["sca"].pop("rank_base", None)
                return cfg
    pytest.skip("no DONE run with a snapshot in this checkout")


def test_key_recovery_accepts_a_config_without_rank_base(cfg):
    validate_key_recovery_contract(cfg)                       # fig24: raised "requires sca.rank_base: 1"


@pytest.mark.parametrize("base", [0, 2, True])
def test_key_recovery_refuses_another_rank_base(cfg, base):
    bad = copy.deepcopy(cfg)
    bad["sca"]["rank_base"] = base
    with pytest.raises(KeyRecoveryError, match="rank_base"):
        validate_key_recovery_contract(bad)
