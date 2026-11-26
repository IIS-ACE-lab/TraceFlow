# services/side_channel_core/tests/unit/test_seed_null.py
"""seed: null (2026-10-24): an unseeded run, as Zaid et al.'s and Wouters et al.'s code, which seed nothing. The run
draws one seed from the operating system the first time it asks (utils/seed.run_seed), keeps it in cfg["seed_drawn"]
(written into its params_out snapshot) and reuses it; the hash takes seed: null, never the drawn seed. Only a written
null means it: an empty `seed:` is no setting, as every empty key."""

import copy
from pathlib import Path

import pytest

from utils.seed import SEED_DRAWN_KEY, run_seed, set_seed


def test_an_integer_seed_is_used_as_written_and_no_key_is_42_as_before():
    assert run_seed({"seed": 7}) == 7 and set_seed({"seed": 7}) == 7
    assert run_seed({}) == 42 and run_seed(None) == 42


def test_seed_null_draws_once_per_run_and_a_new_run_draws_again():
    cfg = {"seed": None}
    first = set_seed(cfg)
    assert cfg[SEED_DRAWN_KEY] == first and cfg["seed"] is None
    assert set_seed(cfg) == run_seed(cfg) == first                     # the same run: the same seed
    draws = {run_seed({"seed": None}) for _ in range(5)}
    assert len(draws) > 1                                            # fresh runs: fresh seeds
    assert run_seed({"seed": None, SEED_DRAWN_KEY: 123}) == 123        # a snapshot's recorded seed
    with pytest.raises(TypeError, match="seed_drawn must be an integer"):
        run_seed({"seed": None, SEED_DRAWN_KEY: "1"})


@pytest.mark.parametrize("text, null", [("seed: null\n", True), ("seed: ~\n", True), ("seed: Null  # unseeded\n", True),
                                        ("seed:\n", False), ("seed:   # e.g. 42\n", False), ("seed: 5\n", False),
                                        ("other: null\n", False)])
def test_only_a_written_null_is_an_unseeded_run(tmp_path, text, null):
    from utils.config_loader import explicit_null_seed

    path = tmp_path / "params.yaml"
    path.write_text(text, encoding="utf-8")
    assert explicit_null_seed(path) is null


def test_the_hash_takes_seed_null_never_the_drawn_seed(monkeypatch):
    from test_final_identity import _cfg
    from utils.model_naming import make_experiment_hash

    monkeypatch.setenv("WORKSPACE_ROOT", "/app")
    base = _cfg(Path("/app"))
    seeded = make_experiment_hash(base)
    unseeded_cfg = copy.deepcopy(base)
    unseeded_cfg["seed"] = None
    unseeded = make_experiment_hash(unseeded_cfg)
    set_seed(unseeded_cfg)                                           # draws, records seed_drawn
    assert unseeded_cfg[SEED_DRAWN_KEY] is not None
    assert make_experiment_hash(unseeded_cfg) == unseeded != seeded


def test_the_test_reads_the_drawn_seed_from_the_snapshot(tmp_path):
    import yaml
    from sca.test import _load_seed_from_params

    params = tmp_path / "run_params.yaml"
    params.write_text(yaml.safe_dump({"seed": None, SEED_DRAWN_KEY: 987}), encoding="utf-8")
    assert _load_seed_from_params(params, fallback=None) == (None, 987)
    params.write_text(yaml.safe_dump({"seed": 5}), encoding="utf-8")
    assert _load_seed_from_params(params, fallback=None) == (5, None)
    params.write_text(yaml.safe_dump({"seed": None}), encoding="utf-8")
    with pytest.raises(TypeError, match="no integer seed_drawn"):
        _load_seed_from_params(params, fallback=None)
