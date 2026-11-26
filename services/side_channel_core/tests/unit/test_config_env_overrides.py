# services/side_channel_core/tests/unit/test_config_env_overrides.py
"""ENV override files (dev/staging/prod.yaml) over params.yaml: only keys that exist, only values that are
given (utils/config_loader.apply_env_overrides). Owner ruling 2026-10-18."""

import pytest

from utils.config_loader import ConfigError, apply_env_overrides


def _base():
    return {"seed": 42, "trainer": {"epochs": [100], "batch_size": [200], "early_stopping": True,
                                    "optimizer": [{"type": "Adam", "learning_rate": [1e-3]}]},
            "sca": {"attack_byte": [2]}, "graph": {"jobs": []}, "scheduler": None}


def test_an_empty_value_never_overrides():
    base = apply_env_overrides(_base(), {"seed": None, "trainer": {"epochs": None, "batch_size": [], "early_stopping": ""},
                                         "sca": {}}, source="staging.yaml")
    assert base == _base()


def test_a_given_value_overrides_only_its_key_and_zero_or_false_is_a_value():
    base = apply_env_overrides(_base(), {"seed": 0, "trainer": {"epochs": [5], "early_stopping": False}},
                               source="staging.yaml")
    assert base["seed"] == 0 and base["trainer"]["epochs"] == [5] and base["trainer"]["early_stopping"] is False
    assert base["trainer"]["batch_size"] == [200] and base["trainer"]["optimizer"] == _base()["trainer"]["optimizer"]


def test_one_value_for_a_list_is_a_one_value_sweep_as_in_params():
    base = apply_env_overrides(_base(), {"trainer": {"epochs": 5, "batch_size": [16, 32]}}, source="staging.yaml")
    assert base["trainer"]["epochs"] == 5 and base["trainer"]["batch_size"] == [16, 32]


def test_a_list_or_an_empty_key_of_params_is_replaced_whole():
    new = [{"type": "SGD", "learning_rate": [0.1]}]
    base = apply_env_overrides(_base(), {"trainer": {"optimizer": new}, "scheduler": {"type": "StepLR"}}, source="dev.yaml")
    assert base["trainer"]["optimizer"] == new and base["scheduler"] == {"type": "StepLR"}


def test_a_key_that_does_not_exist_fails_with_its_path_and_the_closest_key():
    with pytest.raises(ConfigError, match=r"staging.yaml: trainer.epoch is not a key of params.yaml or of the "
                                          r"project's extra configs.*did you mean trainer.epochs\?"):
        apply_env_overrides(_base(), {"trainer": {"epoch": [5]}}, source="staging.yaml")
    with pytest.raises(ConfigError, match=r"dev.yaml: debug is not a key"):
        apply_env_overrides(_base(), {"debug": True}, source="dev.yaml")


def test_an_unknown_key_with_an_empty_value_is_still_a_wrong_key():
    with pytest.raises(ConfigError, match=r"prod.yaml: outputs is not a key"):
        apply_env_overrides(_base(), {"outputs": {"root": None}}, source="prod.yaml")


def test_an_override_of_another_shape_fails():
    with pytest.raises(ConfigError, match=r"dev.yaml: trainer is a mapping in params.yaml; its override must be one too"):
        apply_env_overrides(_base(), {"trainer": 5}, source="dev.yaml")
    with pytest.raises(ConfigError, match=r"dev.yaml: trainer.optimizer is a list in params.yaml"):
        apply_env_overrides(_base(), {"trainer": {"optimizer": {"type": "SGD"}}}, source="dev.yaml")
    with pytest.raises(ConfigError, match=r"dev.yaml must hold a mapping"):
        apply_env_overrides(_base(), ["seed"], source="dev.yaml")


def test_an_empty_file_changes_nothing():
    assert apply_env_overrides(_base(), None, source="dev.yaml") == _base()


def test_a_project_entry_without_scripts_runs_the_core_scripts(tmp_path):
    """TODO.md §26: an entry of name, root and params alone trained, then every test failed with KeyError 'test'."""
    from utils.config_loader import DEFAULT_SCRIPTS, SCC_ROOT, _resolve_scripts_dict

    scripts = _resolve_scripts_dict({"name": "P"}, tmp_path)
    assert set(scripts) == set(DEFAULT_SCRIPTS)
    assert scripts["test"] == str((SCC_ROOT / "src" / "sca" / "test.py").resolve())
    own = tmp_path / "my_test.py"
    own.write_text("")
    assert _resolve_scripts_dict({"scripts": {"test": str(own)}}, tmp_path)["test"] == str(own)   # the project's own wins


def test_an_empty_key_of_params_is_the_codes_default_once_the_env_overrides_are_applied():
    """Owner 2026-10-22: the blueprint starts a project with every key present and empty; an empty key (null)
    is dropped by the loader (config_loader.drop_empty_keys, step 6b) so the reader's default applies, as
    when the key is absent, and it never joins the hash as null. The ENV overrides come first: they may
    fill an empty key. A null item of a list stays (a value its reader refuses); [] and {} stay too. A block
    whose keys are all empty is absent (owner 2026-10-23), as the run hash took it before the blueprint."""
    from utils.config_loader import drop_empty_keys

    base = {"seed": None, "trainer": {"epochs": [100], "patience": None, "loss": {"type": "CrossEntropyLoss",
                                                                                   "args": {"reduction": "mean", "label_smoothing": None}}},
            "sca": {"attack_byte": [2, None], "candidate_ranking": {"mode": None}, "key_enumeration": {"enabled": None}},
            "metrics_to_log": [], "paths": {}, "data": [{"name": "d", "profiling_desync": None}]}
    filled = apply_env_overrides(base, {"trainer": {"patience": 5}}, source="dev.yaml")
    out = drop_empty_keys(filled)
    assert out == {"trainer": {"epochs": [100], "patience": 5, "loss": {"type": "CrossEntropyLoss", "args": {"reduction": "mean"}}},
                   "sca": {"attack_byte": [2, None]},
                   "metrics_to_log": [], "paths": {}, "data": [{"name": "d"}]}
    assert base["seed"] is None, "the input is not changed"


def test_the_early_stopping_block_is_folded_into_the_flat_keys_the_code_and_the_hash_read():
    """params.yaml writes trainer.early_stopping as a block (owner 2026-10-22); config_loader.canonical_trainer
    gives the readers the flat keys the runs were hashed with: enabled -> early_stopping, monitor_metric always,
    patience and min_delta only with enabled: true (read only then: a run that never stops early hashes alike
    with or without them); monitor_mode is retired; an unknown key of the block is refused by name."""
    from utils.config_loader import ConfigError, canonical_form, canonical_trainer

    off = canonical_trainer({"early_stopping": {"enabled": False, "monitor_metric": "guessing_entropy", "patience": 7, "min_delta": 0.0},
                             "epochs": [200], "monitor_mode": "minimize"})
    assert off == {"early_stopping": False, "monitor_metric": "guessing_entropy", "epochs": [200]}
    on = canonical_trainer({"early_stopping": {"enabled": True, "monitor_metric": "accuracy", "patience": 7, "min_delta": 0.5}})
    assert on == {"early_stopping": True, "monitor_metric": "accuracy", "patience": 7, "min_delta": 0.5}
    flat = {"early_stopping": False, "monitor_metric": "guessing_entropy", "epochs": [1]}
    assert canonical_trainer(flat) == flat, "a snapshot's or an older file's flat keys as they are"
    assert canonical_trainer({"early_stopping": {}}) == {"early_stopping": False}
    with pytest.raises(ConfigError, match=r"trainer.early_stopping.patienc is not a key TraceFlow reads; keys here: enabled"):
        canonical_trainer({"early_stopping": {"enabled": True, "patienc": 3}})
    # the whole form: empty keys dropped first, so an empty patience in the block is no patience
    cfg = canonical_form({"trainer": {"early_stopping": {"enabled": True, "monitor_metric": "accuracy", "patience": None}}, "seed": None})
    assert cfg == {"trainer": {"early_stopping": True, "monitor_metric": "accuracy"}}


def test_the_frameworks_grad_clip_norm_is_folded_to_the_key_the_code_reads():
    """params.yaml writes gradient clipping under deep_learning.pytorch / deep_learning.keras (owner 2026-10-22);
    the chosen framework's value becomes trainer.grad_clip_norm (core/framework_switch reads it), the keys leave
    the blocks (the framework hash payload never held them), and a flat trainer.grad_clip_norm still works."""
    from utils.config_loader import canonical_form

    cfg = canonical_form({"deep_learning": {"framework": "keras", "pytorch": {"grad_clip_norm": 5.0, "weight_init": "default"},
                                            "keras": {"grad_clip_norm": 2.5}}, "trainer": {"epochs": [1]}})
    assert cfg == {"deep_learning": {"framework": "keras", "pytorch": {"weight_init": "default"}, "keras": {}},
                   "trainer": {"epochs": [1], "grad_clip_norm": 2.5}}
    cfg = canonical_form({"deep_learning": {"pytorch": {"grad_clip_norm": 1.0}, "keras": {"grad_clip_norm": None}}, "trainer": {}})
    # keras's only key is empty: no block (a block of empty keys is absent, 2026-10-23)
    assert cfg["trainer"] == {"grad_clip_norm": 1.0} and cfg["deep_learning"] == {"pytorch": {}}, "pytorch when no framework is named; an empty one is none"
    flat = {"deep_learning": {"framework": "pytorch"}, "trainer": {"grad_clip_norm": 3.0}}
    assert canonical_form(flat) == flat


def test_a_block_of_empty_keys_is_no_block_but_a_written_empty_block_stays():
    """The blueprint's `output_transformation: {type: Softmax, args: {dim: }}` hashes as the runs made before it
    (which had no args); a nested block of empty blocks goes too; `args: {}` written by hand is a value."""
    from utils.config_loader import drop_empty_keys

    blueprint = {"evaluation": {"output_transformation": {"type": "Softmax", "args": {"dim": None}}}}
    assert drop_empty_keys(blueprint) == {"evaluation": {"output_transformation": {"type": "Softmax"}}}
    assert drop_empty_keys({"a": {"b": {"c": None}, "d": {"e": None}}, "f": 1}) == {"f": 1}
    written = {"evaluation": {"output_transformation": {"type": "Softmax", "args": {}}}, "optimizer": {"args": {}}}
    assert drop_empty_keys(written) == written
    assert drop_empty_keys({"args": {"dim": 1, "x": None}}) == {"args": {"dim": 1}}       # a filled key keeps it


def test_one_framework_per_file_and_a_swept_early_stopping_keeps_its_patience():
    """Owner 2026-10-23: deep_learning.framework is one framework (a list is refused: the grad-clip fold and the
    optimizer arguments follow one framework); a swept early_stopping.enabled [true, false] keeps patience and
    min_delta for its true trial (they were dropped before: the trial ran with the defaults)."""
    import pytest

    from services.side_channel_core.utils.config_errors import ConfigError
    from utils.config_loader import canonical_form, canonical_trainer

    with pytest.raises(ConfigError, match="takes one framework"):
        canonical_form({"deep_learning": {"framework": ["pytorch", "keras"], "pytorch": {"grad_clip_norm": 5.0}}, "trainer": {}})
    assert canonical_form({"deep_learning": {"framework": "keras", "keras": {"grad_clip_norm": 1.0}}, "trainer": {}})["trainer"] == {"grad_clip_norm": 1.0}
    swept = canonical_trainer({"early_stopping": {"enabled": [True, False], "monitor_metric": "guessing_entropy", "patience": 7, "min_delta": 0.5}})
    assert swept == {"early_stopping": [True, False], "monitor_metric": "guessing_entropy", "patience": 7, "min_delta": 0.5}
    off = canonical_trainer({"early_stopping": {"enabled": [False], "monitor_metric": "guessing_entropy", "patience": 7}})
    assert off == {"early_stopping": [False], "monitor_metric": "guessing_entropy"}


def test_repeat_seeds_of_a_cross_validated_run_must_not_share_a_training_seed():
    """seed: [42, 53, 64] with 10 folds: each run trains with its seed and the next 10, so repeats must be at
    least n_folds + 1 apart; closer seeds are refused by name with a spacing that works (owner 2026-10-23)."""
    import pytest

    from services.side_channel_core.utils.config_errors import ConfigError
    from utils.config_loader import check_seed_spacing

    check_seed_spacing({"seed": [42, 53, 64], "cross_validation": {"enabled": True, "n_folds": 10}})
    check_seed_spacing({"seed": [42, 50], "cross_validation": {"enabled": False, "n_folds": 10}})    # no folds: any seeds
    check_seed_spacing({"seed": 42, "cross_validation": {"enabled": True, "n_folds": 10}})
    with pytest.raises(ConfigError, match=r"seeds 42 and 50 are 8 apart.*at least 11, e\.g\. 42, 53, 64"):
        check_seed_spacing({"seed": [42, 64, 50], "cross_validation": {"enabled": True, "n_folds": 10}})
    with pytest.raises(ConfigError, match="at least 6"):
        check_seed_spacing({"seed": [1, 2], "cross_validation": {"enabled": True, "n_folds": [3, 5]}})   # a swept n_folds: its largest
