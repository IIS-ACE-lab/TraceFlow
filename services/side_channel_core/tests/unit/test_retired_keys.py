# services/side_channel_core/tests/unit/test_retired_keys.py
"""sca.rank_base and data.multilabel are no longer settings (2026-10-19): a configuration without them
gives a run the identity a run made with them (rank_base: 1, multilabel: 0) has, so DONE runs stay DONE,
and the snapshots naming them still load."""

import copy

import pytest

from services.side_channel_core.utils import config_schema
from services.side_channel_core.utils.run_map import sample_key
from services.side_channel_core.utils.sweep_engine import _scc_sampled_payload
from sca.dataset_types import hash_entry

ENTRY = {"name": "ascad", "type": "ASCADf", "database": "data/ASCAD.h5", "profiling_range": [0, 50000]}
RUN = {"model_id": "M", "trainer": {"epochs": [100]}, "model": [{"model_id": "M"}],
       "sca": {"attack_byte": [2], "leakage_model": ["ID"]}}


def test_a_data_entry_without_multilabel_hashes_as_multilabel_0():
    assert hash_entry(ENTRY) == hash_entry({**ENTRY, "multilabel": 0})
    assert hash_entry({**ENTRY, "multilabel": 1}) != hash_entry(ENTRY)      # a set value still counts


def test_a_sweep_point_without_rank_base_keys_as_rank_base_1():
    with_key = copy.deepcopy(RUN)
    with_key["sca"]["rank_base"] = 1
    assert sample_key(_scc_sampled_payload(RUN)) == sample_key(_scc_sampled_payload(with_key))
    assert _scc_sampled_payload(RUN)["sca"]["rank_base"] == 1


def test_the_snapshots_naming_them_still_pass_the_strict_key_check():
    config_schema.check_keys({"sca": {"rank_base": 1}}, config_schema.PARAMS, source="snapshot")
    with pytest.raises(Exception):
        config_schema.check_keys({"sca": {"rank_bas": 1}}, config_schema.PARAMS, source="snapshot")
