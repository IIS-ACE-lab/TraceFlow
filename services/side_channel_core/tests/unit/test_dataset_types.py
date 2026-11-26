# services/side_channel_core/tests/unit/test_dataset_types.py
"""Database types (src/sca/dataset_types.py): the data-entry switch and ASCADf backward compatibility."""

import copy

import pytest

from sca import dataset_types as dt

ASCAD = "services/side_channel_core/data/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"


def _entry(**extra):
    entry = {"name": "ascad", "database": ASCAD, "profiling_range": [0, 50000], "attack_range": [0, 10000],
             "test_range": [0, 10000], "target_range": [0, 700], "profiling_desync": 0}
    entry.update(extra)
    return entry


def test_a_declared_type_is_used_and_an_unknown_one_is_refused():
    assert dt.resolve_dataset_type(_entry(type="ASCADf")).name == "ASCADf"
    with pytest.raises(dt.DatasetTypeError, match="unknown type 'ASCADx'"):
        dt.resolve_dataset_type(_entry(type="ASCADx"))


def test_an_entry_without_type_is_ascadf_only_in_the_ascad_fixed_key_folder():
    assert dt.resolve_dataset_type(_entry()).name == "ASCADf"
    assert dt.resolve_dataset_type(_entry(database="/app/" + ASCAD)).name == "ASCADf"
    with pytest.raises(dt.DatasetTypeError, match="has no type"):
        dt.resolve_dataset_type(_entry(database="services/side_channel_core/data/AES_HD/aes_hd.h5"))


def test_an_entry_must_take_exactly_the_keys_of_its_type():
    assert dt.validate_entry(_entry(type="ASCADf", alias="ascad_full")).name == "ASCADf"
    with pytest.raises(dt.DatasetTypeError, match=r"does not take: \['data_split'\]"):
        dt.validate_entry(_entry(data_split="0.65:0.25:0.10"))
    incomplete = _entry()
    del incomplete["target_range"]
    with pytest.raises(dt.DatasetTypeError, match=r"required keys missing: \['target_range'\]"):
        dt.validate_entry(incomplete)


def test_the_type_gives_sca_variant_and_an_old_sca_variant_must_agree():
    assert dt.dataset_variant({"data": _entry(type="ASCADf"), "sca": {}}) == "fixed_key"
    assert dt.dataset_variant({"data": _entry(), "sca": {"variant": "fixed_key"}}) == "fixed_key"
    with pytest.raises(dt.DatasetTypeError, match="disagrees"):
        dt.dataset_variant({"data": _entry(type="ASCADf"), "sca": {"variant": "variable_key"}})


def _run_cfg(**overrides):
    cfg = {"data": _entry(), "model": {"model_id": "m"}, "trainer": {"epochs": 1}, "seed": 42,
           "project": {"name": "p", "root": "/r", "scripts": "/s"},
           "sca": {"variant": "fixed_key", "attack_byte": 2, "leakage_model": "ID"}, "device": "cuda"}
    cfg.update(overrides)
    return cfg


def test_an_ascadf_run_keeps_its_hash_with_type_and_without_sca_variant():
    from utils.model_naming import make_experiment_hash

    legacy = _run_cfg()                                   # as every campaign run was made
    typed = copy.deepcopy(legacy)
    typed["data"]["type"] = "ASCADf"
    del typed["sca"]["variant"]                           # params.yaml without sca.variant
    assert make_experiment_hash(typed) == make_experiment_hash(legacy)
    changed = copy.deepcopy(typed)
    changed["data"]["profiling_desync"] = 50              # the data entry still identifies the run
    assert make_experiment_hash(changed) != make_experiment_hash(legacy)


def test_the_readers_take_the_variant_from_the_type_and_hand_built_configs_keep_theirs():
    assert dt.resolved_variant({"data": _entry(type="ASCADf"), "sca": {}}) == "fixed_key"
    assert dt.resolved_variant({"sca": {"variant": "variable_key"}}) == "variable_key"     # no data entry
    assert dt.resolved_variant({"data": {"name": "x"}, "sca": {"variant": "fixed_key"}}) == "fixed_key"
    with pytest.raises(dt.DatasetTypeError, match="disagrees"):
        dt.resolved_variant({"data": _entry(type="ASCADf"), "sca": {"variant": "variable_key"}})


def test_the_attack_reads_its_key_mode_from_the_type():
    from sca.perform_attack import _get_variable_key

    assert _get_variable_key({"data": _entry(type="ASCADf"), "sca": {}}) is False
    with pytest.raises(KeyError, match="requires"):
        _get_variable_key({"sca": {}})                     # neither a type nor sca.variant: refused, as before


def test_a_project_whose_data_entry_disagrees_with_its_type_does_not_load(tmp_path, monkeypatch):
    import yaml
    from utils import config_loader as cl

    project = tmp_path / "projects" / "P"
    (project / "config").mkdir(parents=True)
    params = {"data": [_entry(type="ASCADf", data_split="0.65:0.25:0.10")], "sca": {"attack_byte": 2}}
    (project / "config" / "params.yaml").write_text(yaml.safe_dump(params))
    master = {"default_project": "P", "project": [{"name": "P", "root": "P/", "params": "config/params.yaml"}]}
    (tmp_path / "master.yaml").write_text(yaml.safe_dump(master))
    monkeypatch.setattr(cl, "SCC_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cl, "PROJECTS_ROOT", tmp_path / "projects")
    with pytest.raises(Exception, match=r"does not take: \['data_split'\]"):
        cl.get_project_config("P", validate_data=False)


ASCADR = "services/side_channel_core/data/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_variable_key/ASCAD_data/ASCAD_databases/ASCAD.h5"


def test_ascadr_has_random_profiling_keys_a_fixed_attack_key_and_must_be_declared():
    entry = _entry(type="ASCADr", database=ASCADR, profiling_range=[0, 200000], target_range=[0, 1400])
    assert dt.validate_entry(entry).name == "ASCADr"
    assert dt.dataset_variant({"data": entry, "sca": {}}) == "fixed_key"        # the attack set has one key
    assert dt.random_profiling_keys({"data": entry}) is True and dt.random_profiling_keys({"data": _entry()}) is False
    untyped = dict(entry)
    del untyped["type"]
    with pytest.raises(dt.DatasetTypeError, match="has no type"):              # only the fixed-key folder is implied
        dt.resolve_dataset_type(untyped)


def test_every_database_of_a_type_has_its_published_source():
    assert dt.source_file("/app/" + ASCADR).sha256.startswith("d834da6c")
    assert dt.source_file(ASCAD).member == "ASCAD_data/ASCAD_databases/ASCAD.h5"
    assert dt.source_file("services/side_channel_core/data/ELSEWHERE/x.h5") is None
    for dtype in dt.TYPES.values():
        assert all(len(s.sha256) == 64 and s.url.startswith("https://") for s in dtype.sources), dtype.name


def test_ascadr_cross_validation_needs_the_fold_attack_off_and_takes_its_epoch_from_the_folds():
    from types import SimpleNamespace
    from utils import cross_validation as cv

    entry = _entry(type="ASCADr", database=ASCADR, target_range=[0, 1400])
    cfg = {"data": entry, "sca": {}, "cross_validation": {"enabled": True, "n_folds": 2},
           "perform_attack": {"validation": {"attack_times": 1}}}
    with pytest.raises(ValueError, match="attack_times: 0"):
        cv.validate_cross_validation_config(cfg)
    cfg["perform_attack"]["validation"]["attack_times"] = 0
    try:
        cv.validate_cross_validation_config(cfg)                    # later checks need a full config
    except (TypeError, ValueError, KeyError) as exc:
        assert "attack_times: 0" not in str(exc)                     # but the ASCADr rule is satisfied
    plan = SimpleNamespace(spec=SimpleNamespace(n_folds=2), paths=SimpleNamespace(aggregate="/a"))
    folds = [{"final_validation": {"epoch": 2, "metrics": {}}}] * 2
    aggregate = {"rank_curve": cv._aggregate_final_fold_curves(plan, cfg, folds)}
    assert aggregate["rank_curve"]["status"] == "not_available" and "random profiling keys" in aggregate["rank_curve"]["reason"]
    assert cv.build_cv_headline(plan, cfg=cfg, fold_results=folds, aggregate=aggregate)["epoch"] == 2
    ascadf = {**cfg, "data": _entry(type="ASCADf")}                              # ASCADf: as before, no epoch without a curve
    assert cv.build_cv_headline(plan, cfg=ascadf, fold_results=folds, aggregate={"rank_curve": {}})["epoch"] is None


def test_the_selected_entry_refuses_an_attack_byte_its_type_does_not_label_with_the_reason():
    dpav4 = {"name": "dpav4", "type": "DPAv4", "database": "services/side_channel_core/data/DPAv4/DPAv4_data/DPAv4.h5"}
    dt.check_run_settings({"data": dpav4, "sca": {"attack_byte": [0]}})
    with pytest.raises(dt.DatasetTypeError, match=r"\[2\] is not defined for data entry 'dpav4' \(type DPAv4\): "
                                                  r"its label is Sbox\[p\[0\] \^ k\[0\]\] \^ mask.*Set sca.attack_byte: \[0\]"):
        dt.check_run_settings({"data": dpav4, "sca": {"attack_byte": [2]}})
    aes_hd = {"name": "aes_hd", "type": "AES_HD", "database": "services/side_channel_core/data/AES_HD/AES_HD_data/AES_HD.h5"}
    with pytest.raises(dt.DatasetTypeError, match=r"last-round register of byte 11.*Set sca.attack_byte: \[11\]"):
        dt.check_run_settings({"data": aes_hd, "sca": {"attack_byte": [0, 11]}})
    dt.check_run_settings({"data": _entry(type="CHES_CTF_2018", database="services/side_channel_core/data/"
                                         "CHES_CTF_2018/CHES_CTF_2018_data/CHES_CTF_2018.h5"),
                           "sca": {"attack_byte": [0, 2, 15]}})                      # S-box, full traces: any byte


def test_an_ascad_window_file_refuses_a_byte_its_window_was_not_cut_for_with_the_raw_traces_as_hint():
    dt.check_run_settings({"data": _entry(type="ASCADf"), "sca": {"attack_byte": [2]}})
    with pytest.raises(dt.DatasetTypeError, match=r"sca.attack_byte \[0\] lies outside the trace window of data entry "
                                                  r"'ascad' \(type ASCADf\): ASCAD.h5 holds the 700 samples ANSSI cut "
                                                  r"around the first-round S-box of byte 2.*Set sca.attack_byte: \[2\], "
                                                  r"or attack byte 0 on the raw traces \(ATMega8515_raw_traces.h5\)"):
        dt.check_run_settings({"data": _entry(type="ASCADf"), "sca": {"attack_byte": [0, 2]}})
    desync = _entry(type="ASCADf", database=ASCAD.replace("ASCAD.h5", "ASCAD_desync100.h5"))
    with pytest.raises(dt.DatasetTypeError, match=r"\[15\] lies outside .* ASCAD_desync100.h5 holds the 700 samples"):
        dt.check_run_settings({"data": desync, "sca": {"attack_byte": [15]}})
    ascadr = _entry(type="ASCADr", database=ASCADR, target_range=[0, 1400])
    with pytest.raises(dt.DatasetTypeError, match=r"\(type ASCADr\): ASCAD.h5 holds the 1,400 samples"):
        dt.check_run_settings({"data": ascadr, "sca": {"attack_byte": [5]}})
    ascadr_desync = _entry(type="ASCADr", database=ASCADR.replace("ASCAD.h5", "ASCAD_desync50.h5"))
    with pytest.raises(dt.DatasetTypeError, match=r"ASCAD_desync50.h5 holds the samples ANSSI cut"):   # counts not listed
        dt.check_run_settings({"data": ascadr_desync, "sca": {"attack_byte": [5]}})
    raw = _entry(type="ASCADf", database=ASCAD.replace("ASCAD.h5", "ATMega8515_raw_traces.h5"))
    dt.check_run_settings({"data": raw, "sca": {"attack_byte": [0, 5]}})            # raw: the entry's window decides
    # the data loader and the attack call the same check (a hand-built configuration, no check_run_settings)
    with pytest.raises(dt.DatasetTypeError, match="lies outside the trace window"):
        dt.require_attack_byte({"data": _entry(type="ASCADf"), "sca": {"attack_byte": 3}})


def test_the_selected_ascadr_entry_refuses_attacked_folds_when_the_command_loads():
    ascadr = {"name": "ascad_r", "type": "ASCADr",
              "database": "services/side_channel_core/data/" + dt._ASCADR + "ascad-variable.h5"}
    cv_on = {"data": ascadr, "sca": {"attack_byte": [2]}, "cross_validation": {"enabled": True}}
    with pytest.raises(dt.DatasetTypeError, match=r"'ascad_r' \(type ASCADr: a random key per profiling trace\) "
                                                  r"needs perform_attack.validation.attack_times: 0 \(it is 5\)"):
        dt.check_run_settings({**cv_on, "perform_attack": {"validation": {"attack_times": 5}}})
    dt.check_run_settings({**cv_on, "perform_attack": {"validation": {"attack_times": 0}}})
    dt.check_run_settings({**cv_on, "cross_validation": {"enabled": False}})          # no CV: nothing to refuse
    dt.check_run_settings({**cv_on, "data": _entry(type="ASCADf")})                     # fixed key: folds attacked


def test_ranges_that_do_not_fit_the_registered_file_are_refused_at_load():
    dt.validate_entry(_entry(type="ASCADf"))                                         # 50k / 10k x 700
    with pytest.raises(dt.DatasetTypeError, match=r"profiling_range \[0, 60000\] is outside the 50,000 profiling traces"):
        dt.validate_entry(_entry(profiling_range=[0, 60000]))
    with pytest.raises(dt.DatasetTypeError, match=r"test_range \[0, 10000\] is outside the 5,000 traces attack_range selects"):
        dt.validate_entry(_entry(attack_range=[0, 5000]))
    with pytest.raises(dt.DatasetTypeError, match=r"target_range \[0, 700\] plus the desync shift of up to 50 is outside the 700"):
        dt.validate_entry(_entry(profiling_desync=50))
    raw = _entry(database=ASCAD.replace("ASCAD.h5", "ATMega8515_raw_traces.h5"), attack_range=[50000, 60000],
                 target_range=[45400, 46100], profiling_desync=99)                     # the paper's desync100 raw entry
    dt.validate_entry(raw)
    with pytest.raises(dt.DatasetTypeError, match=r"attack_range \[50000, 70000\] is outside the 60,000 rows"):
        dt.validate_entry({**raw, "attack_range": [50000, 70000]})
    unlisted = _entry(type="ASCADr", database="services/side_channel_core/data/" + dt._ASCADR + "ASCAD_desync50.h5",
                      profiling_range=[0, 10**6])
    dt.validate_entry(unlisted)                                                     # counts not listed: the run checks


def test_the_attack_bytes_an_entry_allows_are_the_ones_require_attack_byte_accepts():
    assert dt.attack_bytes_allowed({"type": "DPAv4", "database": "x.h5"}) == [0]
    assert dt.attack_bytes_allowed({"type": "AES_HD", "database": "x.h5"}) == [11]
    ascad = "services/side_channel_core/data/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
    raw = "services/side_channel_core/data/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ATMega8515_raw_traces.h5"
    assert dt.attack_bytes_allowed({"type": "ASCADf", "database": ascad}) == [2]                 # the file's trace window
    assert dt.attack_bytes_allowed({"type": "ASCADf", "database": raw}) == list(range(16))


def test_an_empty_attack_byte_follows_the_data_entry_and_a_type_with_several_bytes_needs_one():
    """Owner 2026-10-23: the source of sca.attack_byte is the choice of the dataset. Left empty (absent, null,
    []), the run takes the one byte its entry's type defines within its file's window, written into cfg before
    the sweep and the hash; an entry allowing several bytes is refused by name; a value set is kept and checked."""
    dpav4 = {"name": "dpav4", "type": "DPAv4", "database": "services/side_channel_core/data/DPAv4/DPAv4_data/DPAv4.h5"}
    aes_hd = {"name": "aes_hd", "type": "AES_HD", "database": "services/side_channel_core/data/AES_HD/AES_HD_data/AES_HD.h5"}
    aes_rd = {"name": "aes_rd", "type": "AES_RD", "database": "services/side_channel_core/data/AES_RD/AES_RD_data/AES_RD.h5"}
    for entry, byte in ((dpav4, 0), (aes_hd, 11), (aes_rd, 0), (_entry(type="ASCADf"), 2)):
        for empty in ({"data": entry}, {"data": entry, "sca": {}}, {"data": entry, "sca": {"attack_byte": []}}, {"data": entry, "sca": {"attack_byte": None}}):
            dt.check_run_settings(empty)
            assert empty["sca"]["attack_byte"] == [byte], (entry["type"], empty)
    assert dt.derived_attack_byte(aes_rd) == 0, "AES_RD: Zaid's first byte (the paper's target Z = Sbox[P1 ^ k*])"
    ches = _entry(type="CHES_CTF_2018", database="services/side_channel_core/data/CHES_CTF_2018/CHES_CTF_2018_data/CHES_CTF_2018.h5")
    with pytest.raises(dt.DatasetTypeError, match=r"sca.attack_byte is empty and data entry 'ascad' \(type CHES_CTF_2018\) defines a target for bytes \[0, 1, .*set the byte"):
        dt.check_run_settings({"data": ches, "sca": {"attack_byte": []}})
    assert dt.derived_attack_byte(ches) is None
    kept = {"data": dpav4, "sca": {"attack_byte": [0]}}
    dt.check_run_settings(kept)
    assert kept["sca"]["attack_byte"] == [0]
    with pytest.raises(dt.DatasetTypeError, match=r"\[2\] is not defined for data entry 'aes_rd' \(type AES_RD\).*Set sca.attack_byte: \[0\]"):
        dt.check_run_settings({"data": aes_rd, "sca": {"attack_byte": [2]}})


def test_a_window_declared_verified_for_one_byte_makes_it_the_entrys_byte():
    """The raw ASCAD traces allow any byte, but an entry cut to a target_range declares the bytes its window was
    verified for (trace_window_verified_by_attack_byte): those are its bytes, so the panel shows no chooser and an
    empty sca.attack_byte takes it (DLSCA_ASCAD's ascad_desync50_raw: byte 2)."""
    raw = _entry(type="ASCADf", database=ASCAD.replace("ASCAD.h5", "ATMega8515_raw_traces.h5"))
    assert dt.attack_bytes_allowed(raw) == list(range(16))
    assert dt.attack_bytes_allowed({**raw, "trace_window_verified_by_attack_byte": {2: True}}) == [2]
    assert dt.attack_bytes_allowed({**raw, "trace_window_verified_by_attack_byte": {"0x02": True, 5: False}}) == [2]
    assert dt.derived_attack_byte({**raw, "trace_window_verified_by_attack_byte": {2: True}}) == 2
    assert dt.attack_bytes_allowed({**raw, "trace_window_verified_by_attack_byte": {2: True, 3: True}}) == [2, 3]


def test_project_attack_byte_is_the_one_byte_every_entry_defines():
    """A reader that compiles the file before any entry is selected (AutoSearch) takes the byte every entry
    defines; entries that differ, or one allowing several bytes, give none."""
    dpav4 = {"name": "dpav4", "type": "DPAv4", "database": "services/side_channel_core/data/DPAv4/DPAv4_data/DPAv4.h5"}
    aes_hd = {"name": "aes_hd", "type": "AES_HD", "database": "services/side_channel_core/data/AES_HD/AES_HD_data/AES_HD.h5"}
    ches = _entry(type="CHES_CTF_2018", database="services/side_channel_core/data/CHES_CTF_2018/CHES_CTF_2018_data/CHES_CTF_2018.h5")
    assert dt.project_attack_byte({"data": {"a": _entry(type="ASCADf"), "b": _entry(type="ASCADf")}}) == 2
    assert dt.project_attack_byte({"data": [dpav4, {**dpav4, "name": "d2"}]}) == 0
    assert dt.project_attack_byte({"data": {"a": dpav4, "b": aes_hd}}) is None
    assert dt.project_attack_byte({"data": [ches]}) is None
    assert dt.project_attack_byte({"data": []}) is None


def test_an_unresolvable_type_stops_every_reader_that_shapes_the_attack_never_guesses_sbox():
    """audit 2026-10-25: label_target, shared_key, random_profiling_keys and resolved_variant fell back to ASCADf's
    on any error, so an entry whose type failed to resolve would be ranked against Sbox[p ^ k] without a word."""
    untyped = {"name": "x", "database": "services/side_channel_core/data/DPAv4/DPAv4_data/DPAv4.h5"}
    for reader in (dt.label_target, dt.shared_key, dt.random_profiling_keys, dt.resolved_variant):
        with pytest.raises(dt.DatasetTypeError, match="has no type"):
            reader({"data": untyped, "sca": {}})
    typed = {**untyped, "type": "DPAv4"}
    assert dt.label_target({"data": typed, "sca": {}}) == "sbox_masked"
    assert dt.label_target({"data": {**untyped, "type": "AES_HD", "database": "services/side_channel_core/data/AES_HD/AES_HD_data/AES_HD.h5"},
                            "sca": {}}) == "inv_sbox_hd"
    for no_data in ({"sca": {}}, {"data": {"name": "stub", "test_range": [0, 10]}, "sca": {}}):   # a unit's hand-built configuration
        assert dt.label_target(no_data) == "sbox" and dt.shared_key(no_data) is True
        assert dt.random_profiling_keys(no_data) is False


def test_a_registered_file_is_the_database_of_its_own_type_only():
    """A declared type fixes the label target: type DPAv4 on ASCAD.h5 would label ASCAD's traces with
    DPAv4's masked S-box output. A registered file of another type, or a conversion's download, is refused."""
    with pytest.raises(dt.DatasetTypeError, match="ASCAD.h5 is a registered ASCADf file, not DPAv4: declare type: ASCADf"):
        dt.validate_entry(_entry(type="DPAv4", profiling_range=[0, 4500], attack_range=[0, 500], test_range=[0, 500],
                                 target_range=[0, 4000]))
    variable_key = ASCAD.replace("ATM_AES_v1_fixed_key", "ATM_AES_v1_variable_key")
    with pytest.raises(dt.DatasetTypeError, match="registered ASCADr file, not ASCADf"):
        dt.validate_entry(_entry(type="ASCADf", database=variable_key))
    download = "services/side_channel_core/data/DPAv4/DPAv4_data/source/DPAv4_dataset.zip"
    with pytest.raises(dt.DatasetTypeError, match="a download DPAv4 is made from.*DPAv4/DPAv4_data/DPAv4.h5"):
        dt.validate_entry(_entry(type="DPAv4", database=download))
    assert dt.validate_entry(_entry(type="ASCADf")).name == "ASCADf"
    assert dt.validate_entry(_entry(type="DPAv4", database="/app/services/side_channel_core/data/DPAv4/DPAv4_data/DPAv4.h5",
                                    profiling_range=[0, 4500], attack_range=[0, 500], test_range=[0, 500],
                                    target_range=[0, 4000])).name == "DPAv4"
    # a file no type registers (the entry's own traces) is taken as its declared type; device.yaml records it unregistered
    assert dt.validate_entry(_entry(type="DPAv4", database="services/side_channel_core/data/MINE/mine.h5")).name == "DPAv4"
