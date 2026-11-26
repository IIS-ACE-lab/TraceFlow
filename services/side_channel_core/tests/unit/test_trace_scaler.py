# services/side_channel_core/tests/unit/test_trace_scaler.py
"""Trace scaling (src/sca/ascad_data_loader.py modes, utils/trace_scaler.py): per-trace scaling
and scalers fitted on the training traces only."""

import json

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler, minmax_scale, scale

from sca.ascad_data_loader import _normalization_from_data_cfg, _normalize_trace
from utils import trace_scaler as ts

rng = np.random.default_rng(3)
TRACES = rng.normal(5.0, 3.0, (40, 12)).astype(np.float32)


def test_per_trace_modes_scale_each_trace_by_itself():
    standard = _normalize_trace(TRACES.copy(), TRACES.dtype, {"mode": "per_trace_standard"})
    assert np.allclose(standard, scale(TRACES, axis=1)) and np.allclose(standard.mean(axis=1), 0, atol=1e-5)
    policy = _normalization_from_data_cfg({"trace_normalization": {"mode": "per_trace_minmax", "feature_range": [-1, 1]}})
    ranged = _normalize_trace(TRACES.copy(), TRACES.dtype, policy)
    assert np.allclose(ranged, minmax_scale(TRACES, feature_range=(-1, 1), axis=1))
    assert np.allclose(ranged.min(axis=1), -1) and np.allclose(ranged.max(axis=1), 1)


def test_a_fitted_policy_names_its_scalers_and_leaves_the_traces_unscaled_at_load():
    policy = _normalization_from_data_cfg({"trace_normalization": {"mode": "fitted", "steps": ["standard", "minmax"]}})
    assert policy["steps"] == ["standard", "minmax"] and policy["feature_range"] == [0.0, 1.0]
    assert np.array_equal(_normalize_trace(TRACES.copy(), TRACES.dtype, policy), TRACES)
    for bad in ([], ["robust"], ["minmax", "minmax"], None):
        with pytest.raises(ValueError, match="scikit-learn scalers to chain"):
            _normalization_from_data_cfg({"trace_normalization": {"mode": "fitted", "steps": bad}})
    assert "feature_range" not in _normalization_from_data_cfg(
        {"trace_normalization": {"mode": "fitted", "steps": ["standard"]}})


def test_the_stored_scaler_is_the_same_sklearn_fit(tmp_path):
    policy = {"mode": "fitted", "eps": 1e-8, "steps": ["standard", "minmax"], "feature_range": [0.0, 1.0]}
    pipeline = ts.fit(policy, TRACES)
    reference = MinMaxScaler((0, 1)).fit(StandardScaler().fit_transform(TRACES.astype(np.float64)))
    assert np.allclose(pipeline.transform(TRACES), reference.transform(StandardScaler().fit(TRACES.astype(np.float64)).transform(TRACES)))
    path = ts.save(pipeline, tmp_path / "scaler.json", policy=policy, fitted_on={"split": "test", "rows": 40})
    record = json.loads(path.read_text())
    assert record["fitted_on"] == {"split": "test", "rows": 40} and [s["class"] for s in record["steps"]] == ["StandardScaler", "MinMaxScaler"]
    assert np.array_equal(ts.load(path).transform(TRACES), pipeline.transform(TRACES))


class _Split:
    """Stands in for ASCADDataset: its unscaled traces, and the scaler attached to it."""

    def __init__(self, traces):
        self.traces, self.scaler = traces, None

    def unscaled_traces(self):
        return self.traces

    def set_trace_scaler(self, scaler):
        self.scaler = scaler


def test_a_fold_is_fitted_on_its_training_rows_only_and_the_final_fit_is_stored(tmp_path, monkeypatch):
    cfg = {"data": {"name": "d", "trace_normalization": {"mode": "fitted", "steps": ["standard"]}}}
    monkeypatch.setattr(ts, "scaler_path", lambda cfg: tmp_path / "scaler.json")
    train_rows = np.arange(30)
    training, validation = _Split(TRACES), _Split(TRACES)
    pipeline = ts.scale_training(cfg, training, train_rows, [training, validation], final=False, label="fold")
    assert training.scaler is pipeline and validation.scaler is pipeline
    assert np.allclose(pipeline.named_steps["standard"].mean_, TRACES[:30].mean(axis=0))      # not the validation rows
    assert not (tmp_path / "scaler.json").exists()                                           # a fold's fit is not stored
    ts.scale_training(cfg, training, None, [training], final=True, label="refit")
    test = _Split(TRACES)
    ts.scale_test(cfg, test)
    assert np.allclose(test.scaler.named_steps["standard"].mean_, TRACES.mean(axis=0))


def test_the_test_refuses_to_run_without_the_runs_scaler(tmp_path, monkeypatch):
    cfg = {"data": {"name": "d", "trace_normalization": {"mode": "fitted", "steps": ["standard"]}}}
    monkeypatch.setattr(ts, "scaler_path", lambda cfg: tmp_path / "scaler.json")
    with pytest.raises(FileNotFoundError, match="its test needs that scaler"):
        ts.scale_test(cfg, _Split(TRACES))
    assert ts.scale_test({"data": {"name": "d"}}, _Split(TRACES)) is None                   # not fitted: nothing to do


def test_a_fitted_split_refuses_to_be_read_without_its_scaler(tmp_path):
    """A split made with trace_normalization.mode fitted yields its traces only once the run's scaler is attached
    (set_trace_scaler); reading it before is refused (before the file is even opened), so no consumer evaluates
    on raw traces by mistake. unscaled_traces, what a scaler is fitted on, is not affected."""
    from sca.ascad_data_loader import ASCADDataset, SCADataContractError

    ds = ASCADDataset.__new__(ASCADDataset)
    ds.h5_path, ds.group_name, ds._has_meta, ds._traces, ds._labels, ds._meta = tmp_path / "train.h5", "Profiling_traces", False, None, None, None
    ds._norm_cfg, ds._trace_scaler, ds._n, ds._pid = {"mode": "fitted", "eps": 1e-8, "steps": ["standard"]}, None, len(TRACES), None
    with pytest.raises(SCADataContractError, match="no scaler is attached"):
        ds[0]
    ds._norm_cfg = {"mode": "none", "eps": 1e-8}
    with pytest.raises((FileNotFoundError, OSError, Exception)):   # mode none reads the file: it is not there, nothing else refused
        ds[0]
