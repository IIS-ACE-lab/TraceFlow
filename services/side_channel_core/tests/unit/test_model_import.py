# services/side_channel_core/tests/unit/test_model_import.py
"""Model import (utils/model_import.py, utils/model_import_weights.py): a released model tested through
TraceFlow without training.

The decisive check is test_the_converted_model_gives_the_released_models_probabilities: a channels-last
Keras model, saved as an HDF5 file, read back by the import (layer list converted, weights copied, the
first Dense after the Flatten reordered) gives the released model's class probabilities; without the
reorder it does not.

Run from the repository root:
    PYTHONPATH="$PWD:$PWD/services/side_channel_core:$PWD/services/side_channel_core/src" \\
        CUDA_VISIBLE_DEVICES="" conda run -n sca python -m pytest \\
        services/side_channel_core/tests/unit/test_model_import.py -q
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from utils import model_import as mi
from utils import model_import_weights as miw
from utils.model_naming import make_experiment_hash

REPO = Path(__file__).resolve().parents[4]


# ── a released model: written by Keras itself, channels-last, as the studies' files ─────────────────────
def _released(tmp_path: Path, name: str = "released.h5") -> Path:
    from core.framework_switch import keras_module

    keras = keras_module()
    keras.utils.set_random_seed(3)
    model = keras.Sequential([
        keras.Input(shape=(32, 1)),
        keras.layers.Conv1D(3, 4, padding="same", activation="selu", name="block1_conv1"),   # even kernel: Keras' own 'same'
        keras.layers.BatchNormalization(name="bn1"),
        keras.layers.AveragePooling1D(2, strides=2, name="pool1"),
        keras.layers.Conv1D(5, 3, padding="valid", activation="relu", name="block2_conv1"),
        keras.layers.Flatten(name="flatten"),
        keras.layers.Dense(7, activation="selu", name="fc1"),
        keras.layers.Dense(9, activation="softmax", name="predictions"),
    ])
    bn = model.get_layer("bn1")                       # running statistics far from the defaults
    bn.set_weights([np.random.uniform(0.5, 2, 3), np.random.normal(size=3), np.random.normal(size=3),
                    np.random.uniform(0.5, 3, 3)])
    path = tmp_path / name
    model.save(str(path))                             # .h5: Keras' HDF5 format (model_config + weights)
    return path


def _traceflow_model(conversion: dict):
    from core.model import build_model

    cfg = {"deep_learning": {"framework": "keras"}, "data": {"target_range": [0, conversion["input"][1]]},
           "sca": {"leakage_model": "HW"}, "model_id": "imported",
           "model": [mi.model_entry({}, conversion, "imported")]}
    return build_model(cfg)


def _probabilities(released, model, x: torch.Tensor):
    model.eval()
    with torch.no_grad():
        ours = torch.softmax(model(x), dim=1).numpy()
    theirs = np.asarray(released(x.permute(0, 2, 1), training=False).detach().cpu())
    return ours, theirs


def test_the_converted_model_gives_the_released_models_probabilities(tmp_path):
    path = _released(tmp_path)
    info = mi.inspect_keras_h5(path)
    conversion = mi.convert_keras_layers(info, name=path.name)
    assert conversion["input"] == [1, 32] and conversion["outputs"] == 9 and conversion["output_activation"] == "softmax"
    assert mi.leakage_suggestion(conversion["outputs"]) == "HW"
    kinds = [next(iter(layer)) for layer in conversion["layers"]]
    assert kinds == ["Conv1d", "BatchNorm1d", "Pooling", "Conv1d", "Flatten", "Linear", "Linear"]
    assert conversion["layers"][0]["Conv1d"]["padding"] == "same"
    assert conversion["layers"][1]["BatchNorm1d"] == {"num_features": 3, "momentum": 0.01, "eps": 0.001}
    assert "activation" not in conversion["layers"][-1]["Linear"]            # the softmax is the test's transformation
    dense = [w for w in conversion["weighted"] if w["kind"] == "dense"]
    assert dense[0]["flatten"] == [5, 14] and dense[1]["flatten"] is None     # only the Dense after the Flatten

    released = miw.load_released_keras(path)
    x = torch.randn(64, 1, 32)

    model = _traceflow_model(conversion)
    assert miw.copy_keras_weights(released, model, conversion["weighted"]) == 12      # 2 + 4 + 2 + 2 + 2
    ours, theirs = _probabilities(released, model, x)
    assert np.allclose(ours, theirs, atol=miw.GATE_ATOL, rtol=miw.GATE_RTOL)

    plain = copy.deepcopy(conversion["weighted"])                              # the Flatten order left as it is
    for w in plain:
        w["flatten"] = None
    model = _traceflow_model(conversion)
    miw.copy_keras_weights(released, model, plain)
    ours, theirs = _probabilities(released, model, x)
    assert np.abs(ours - theirs).max() > 1e-3


def test_the_gate_compares_every_test_trace(tmp_path):
    path = _released(tmp_path)
    conversion = mi.convert_keras_layers(mi.inspect_keras_h5(path), name=path.name)
    released = miw.load_released_keras(path)
    model = _traceflow_model(conversion)
    miw.copy_keras_weights(released, model, conversion["weighted"])
    loader = [(torch.randn(16, 1, 32), torch.zeros(16)) for _ in range(3)]
    gate = miw.keras_gate(released, model, loader, output_activation="softmax", data_format="channels_last",
                          device=torch.device("cpu"))
    assert gate["passed"] and gate["traces"] == 48 and gate["same_top_class"] == 48
    other = _traceflow_model(conversion)                                       # its own initial weights
    gate = miw.keras_gate(released, other, loader, output_activation="softmax", data_format="channels_last",
                          device=torch.device("cpu"))
    assert not gate["passed"]


def test_an_extensionless_keras_file_is_read_as_keras(tmp_path):
    path = _released(tmp_path)
    bare = tmp_path / "ASCAD_desync0"                                          # as Zaid et al. released theirs
    shutil.copyfile(path, bare)
    assert mi.detect_format(bare, None, where="x") == "keras_h5"
    model = miw.load_released_keras(bare)
    assert [layer.name for layer in model.layers if layer.weights][-1] == "predictions"


# ── refused files: code, unknown layers, other formats, another file than the released one ───────────────
def _config_file(tmp_path: Path, layers: list, name: str = "m.h5") -> Path:
    path = tmp_path / name
    config = {"class_name": "Sequential", "config": {"layers": [
        {"class_name": "InputLayer", "config": {"batch_input_shape": [None, 10, 1], "name": "in"}}, *layers]}}
    with h5py.File(path, "w") as handle:
        handle.attrs["model_config"] = json.dumps(config)
        handle.attrs["keras_version"] = "2.2.4"
    return path


def test_a_lambda_layer_is_refused_before_anything_loads(tmp_path):
    path = _config_file(tmp_path, [{"class_name": "Lambda", "config": {"name": "l", "function": "4wEAAAAAAAAA"}}])
    with pytest.raises(mi.ModelImportError, match="Lambda layer"):
        mi.inspect_keras_h5(path)
    with pytest.raises(mi.ModelImportError, match="Lambda layer"):
        miw.load_released_keras(path)                                         # the load checks it again first


def test_a_layer_without_mapping_and_other_formats_are_refused(tmp_path):
    with pytest.raises(mi.ModelImportError, match=r"\['LSTM'\] have no TraceFlow mapping"):
        mi.inspect_keras_h5(_config_file(tmp_path, [{"class_name": "LSTM", "config": {"name": "r", "units": 4}}]))
    with pytest.raises(mi.ModelImportError, match="no model_config"):
        with h5py.File(tmp_path / "weights.h5", "w") as handle:
            handle.create_dataset("w", data=np.zeros(2))
        mi.inspect_keras_h5(tmp_path / "weights.h5")
    pickled = tmp_path / "model.pt"
    pickled.write_bytes(b"\x80\x04K\x01.")
    with pytest.raises(mi.ModelImportError, match="not an HDF5 file"):
        mi.detect_format(pickled, None, where="x")
    with pytest.raises(mi.ModelImportError, match="ONNX is not supported"):
        mi.detect_format(tmp_path / "model.onnx", None, where="x")
    with pytest.raises(mi.ModelImportError, match="Dense layer on the unflattened trace"):
        mi.convert_keras_layers(mi.inspect_keras_h5(_config_file(
            tmp_path, [{"class_name": "Dense", "config": {"name": "d", "units": 4}}], "dense.h5")))


def test_a_file_that_is_not_the_released_one_is_refused(tmp_path):
    path = tmp_path / "released.h5"
    path.write_bytes(b"weights")
    digest = mi.file_sha256(path)
    assert mi.verified_sha256(path, None, where="x") == digest
    assert mi.verified_sha256(path, digest.upper(), where="x") == digest
    with pytest.raises(mi.ModelImportError, match="not the one released"):
        mi.verified_sha256(path, "0" * 64, where="x")
    cfg = {"project": {"root": str(tmp_path)}, mi.IMPORT_KEY: {"file": "released.h5", "sha256": digest}}
    assert miw.released_file(cfg) == path
    path.write_bytes(b"weights, changed")
    with pytest.raises(mi.ModelImportError, match="changed since make import registered it"):
        miw.released_file(cfg)


def test_a_file_outside_the_project_is_refused(tmp_path):
    (tmp_path / "P").mkdir()
    (tmp_path / "elsewhere.h5").write_bytes(b"x")
    with pytest.raises(mi.ModelImportError, match="outside the project folder"):
        mi.released_path(tmp_path / "P", str(tmp_path / "elsewhere.h5"), where="x")


# ── identity: every file its own run; no trained run's hash moves ──────────────────────────────────────────
def _hash_cfg(**extra):
    return {
        "seed": 0,
        "project": {"name": "P", "root": "P", "scripts": {}, "extra_configs": {"test": "config/test_config.yaml"}},
        "data": {"name": "ascad", "database": "data/ascad.h5"},
        "model": [{"model_id": "imported", "architecture": "MLP", "architecture_args": {"layers": []}}],
        "trainer": {"epochs": 1, "batch_size": 1},
        "sca": {"variant": "fixed_key", "attack_byte": 2, "leakage_model": "ID"},
        "metrics_to_log": ["mean_attack_rank_curve"],
        **extra,
    }


def test_every_released_file_is_its_own_run_and_trained_hashes_stay():
    trained = make_experiment_hash(_hash_cfg())
    a = make_experiment_hash(_hash_cfg(model_import={"sha256": "a" * 64, "format": "keras_h5", "source": "S"}))
    b = make_experiment_hash(_hash_cfg(model_import={"sha256": "b" * 64, "format": "keras_h5", "source": "S"}))
    assert len({trained, a, b}) == 3
    # the source and the file's place are provenance, not the weights: same file, same run
    moved = make_experiment_hash(_hash_cfg(model_import={"sha256": "a" * 64, "format": "keras_h5", "source": "T",
                                                         "file": "elsewhere"}))
    assert moved == a
    # extra_configs.import (the import list) never joins a hash: adding it moves no trained run
    listed = _hash_cfg()
    listed["project"]["extra_configs"]["import"] = "config/import_config.yaml"
    assert make_experiment_hash(listed) == trained


def test_an_imported_run_is_no_sweep_and_its_import_list_is_no_sweep():
    from utils.sweep_engine import _find_sweep_paths

    cfg = {"model_import": {"weighted": [{"flatten": [4, 350]}]},
           "import": {"model_import": {"models": [{"files": [{"file": "a.h5"}, {"file": "b.h5"}]}]}}, "seed": [1, 2]}
    assert [path for path, _ in _find_sweep_paths(cfg)] == [("seed",)]


def test_the_import_list_is_not_part_of_a_run_snapshot():
    from utils.config_view import resolved_view

    view = resolved_view({"import": {"model_import": {}}, "model_import": {"sha256": "a"}, "seed": 1})
    assert "import" not in view and view["model_import"] == {"sha256": "a"}


def test_only_an_imported_run_has_an_imported_checkpoint(tmp_path):
    from utils.checkpoint_policy import (CheckpointPolicyError, checkpoint_paths, load_checkpoint_manifest,
                                         write_checkpoint_manifest)

    cfg = {"project": {"root": str(tmp_path)}, "env": "dev", "model_name": "z", "evaluation": {"validation": {"checkpointing": False}},
           "paths": {"checkpoints": "artifacts/{{env}}/generated_models/{{model_name}}/checkpoints",
                     "model_out": "artifacts/{{env}}/generated_models/{{model_name}}/model"}}
    paths = checkpoint_paths(cfg, model_name="z")
    paths.final_epoch.parent.mkdir(parents=True)
    paths.final_epoch.write_bytes(b"weights")

    def manifest(kind):
        write_checkpoint_manifest(cfg=cfg, model_name="z", model_hash="abcd1234", validation_checkpointing=False,
                                  monitor_metric=None, monitor_mode=None, best_epoch=None, best_score=None,
                                  epoch_checkpoint=None, best_validation_checkpoint=None, final_epoch=1,
                                  final_checkpoint=paths.final_epoch, epochs_configured=1, execution={"type": kind})

    imported = {**cfg, mi.IMPORT_KEY: {"sha256": "a" * 64}}
    manifest("model_import")
    assert load_checkpoint_manifest(imported, model_name="z")["execution"]["type"] == "model_import"
    with pytest.raises(CheckpointPolicyError, match="requires a standard training model"):
        load_checkpoint_manifest(cfg, model_name="z")                                # a trained run never takes it
    manifest("standard")
    with pytest.raises(CheckpointPolicyError, match="must come from its model import"):
        load_checkpoint_manifest(imported, model_name="z")


# ── the import config: expansion, settings, the composed run ────────────────────────────────────────────
def _config(models, defaults=None):
    return {"defaults": defaults or {}, "models": models}


def test_a_list_of_files_is_one_run_each_with_its_own_pin():
    files = [{"file": "a.h5", "sha256": "1" * 64}, {"file": "b.h5", "sha256": "2" * 64}]
    runs = mi.expand_imports(_config([{"name": "w", "files": files, "source": "S", "data": "e", "seed": 3}],
                                     defaults={"seed": 0, "sca": {"leakage_model": "ID"}, "perform_attack": {"test": {"attack_times": 10}}}))
    assert [(r["name"], r["file"], r["sha256"]) for r in runs] == [("w_0", "a.h5", "1" * 64), ("w_1", "b.h5", "2" * 64)]
    assert runs[0]["settings"] == {"seed": 3, "sca": {"leakage_model": "ID"}, "perform_attack": {"test": {"attack_times": 10}},
                                   "data": "e"}
    # each file beside its own SHA-256: separate file and sha256 lists are refused, as is file with files
    for bad, message in [({"file": ["a.h5", "b.h5"], "sha256": ["1" * 64, "2" * 64]}, "each file beside its own SHA-256"),
                         ({"file": "a.h5", "files": files}, "not both"),
                         ({"files": [{"file": "a.h5", "sha": "1" * 64}]}, "unknown key"),
                         ({"files": ["a.h5"]}, r"must be \{file: <path>, sha256: <hex>\}")]:
        with pytest.raises(mi.ModelImportError, match=message):
            mi.expand_imports(_config([{**bad, "source": "S"}]))


def test_an_import_takes_the_defaults_for_every_attack_key_it_leaves_empty():
    defaults = {"seed": 100, "perform_attack": {"test": {"attack_times": 100, "shuffle": True,
                                                         "attack_distribution": "full_population"}, "attack_step": 1}}
    own = {"file": "a.h5", "source": "S", "perform_attack": {"test": {"attack_times": 10}}}
    settings = mi.expand_imports(_config([own], defaults=defaults))[0]["settings"]
    assert settings["perform_attack"] == {"test": {"attack_times": 10, "shuffle": True, "attack_distribution": "full_population"},
                                          "attack_step": 1}
    assert settings["seed"] == 100
    with pytest.raises(mi.ModelImportError, match="source is required"):
        mi.expand_imports(_config([{"file": "a.h5"}]))
    with pytest.raises(mi.ModelImportError, match="file is required"):
        mi.expand_imports(_config([{"source": "S"}]))
    with pytest.raises(mi.ModelImportError, match="given twice"):
        mi.expand_imports(_config([{"name": "x", "file": "a.h5", "source": "S"}, {"name": "x", "file": "b.h5", "source": "S"}]))


def test_the_import_config_refuses_unknown_keys(tmp_path):
    path = tmp_path / "import_config.yaml"
    path.write_text(yaml.safe_dump({"model_import": {"models": [{"file": "a.h5", "sourc": "S"}]}}))
    with pytest.raises(mi.ConfigError, match="sourc is not a key TraceFlow reads; did you mean"):
        mi.load_import_config(path)
    shutil.copyfile(REPO / "services/side_channel_core/config/blueprint/import_config.yaml", path)
    assert mi.load_import_config(path) == {"defaults": {}, "models": []}       # the blueprint: every key empty


def _project_cfg():
    return {
        "project": {"name": "P", "root": "P", "scripts": {}},
        "data": {"e": {"name": "e", "target_range": [0, 32], "profiling_split": {"enabled": True, "test_size": 10},
                       "trace_normalization": {"mode": "none"}}},
        "model": [{"model_id": "M"}], "model_id": "M",
        "trainer": {"epochs": 50, "optimizer": [{"type": "Adam", "learning_rate": [1e-3, 1e-4]}]},
        "sca": {"leakage_model": ["ID", "HW"], "candidate_ranking": {"mode": "top_k"}},
        "perform_attack": {"validation": {"attack_times": 100, "shuffle": True, "attack_distribution": "full_population"},
                           "test": {"attack_times": 100, "shuffle": True, "attack_distribution": "full_population"}},
        "evaluation": {"validation": {"checkpointing": True, "judge_log": True}, "test": {"load_model": "best_validation"}},
        "cross_validation": {"enabled": True, "n_folds": 10}, "seed": 42, "import": {"model_import": {}},
    }


@pytest.fixture
def composed(monkeypatch):
    import utils.config_loader as loader
    import services.side_channel_core.src.sca.dataset_types as types

    monkeypatch.setattr(loader, "get_project_config", lambda project_name: copy.deepcopy(_project_cfg()))
    monkeypatch.setattr(types, "check_run_settings", lambda cfg: None)
    conversion = {"layers": [{"Flatten": {}}, {"Linear": {"in_features": 32, "out_features": 9}}], "input": [1, 32],
                  "outputs": 9}
    record = {"file": "released/a.h5", "sha256": "c" * 64, "format": "keras_h5", "source": "S"}

    def compose(**settings):
        run = {"where": "models[0]", "settings": settings, "model": None}
        return mi.compose_run_cfg("P", run, record=record, conversion=conversion, env="dev")
    return compose


def test_the_composed_run_trains_nothing_and_tests_the_imported_weights(composed):
    cfg = composed(data="e", sca={"leakage_model": "HW"}, seed=0, perform_attack={"test": {"attack_times": 5}})
    assert cfg["trainer"] == mi.IMPORT_TRAINER and cfg["cross_validation"] == {"enabled": False}
    assert cfg["model_id"] == "imported" and cfg["model"][0]["model_id"] == "imported"
    assert cfg["model"][0]["architecture_args"]["layers"][-1] == {"Linear": {"in_features": 32, "out_features": 9}}
    assert cfg["deep_learning"] == {"framework": "keras"} and cfg["seed"] == 0
    assert cfg["perform_attack"]["validation"]["attack_times"] == 0                 # nothing validated
    assert cfg["perform_attack"]["test"] == {"attack_times": 5, "shuffle": True, "attack_distribution": "full_population"}
    assert cfg["evaluation"]["validation"]["checkpointing"] is False and cfg["evaluation"]["test"]["load_model"] == "final_epoch"
    assert cfg["sca"]["leakage_model"] == "HW" and cfg["sca"]["candidate_ranking"] == {"mode": "top_k"}
    assert "profiling_split" not in cfg["data"]                                      # nothing fitted: no split needed
    assert "import" not in cfg and cfg[mi.IMPORT_KEY]["sha256"] == "c" * 64


def test_the_settings_a_released_file_does_not_hold_are_required(composed):
    with pytest.raises(mi.ModelImportError, match="its 9 outputs suggest HW"):
        composed(data="e", seed=0)
    with pytest.raises(mi.ModelImportError, match="seed is required"):
        composed(data="e", sca={"leakage_model": "HW"})
    with pytest.raises(mi.ModelImportError, match="data is required"):
        composed(sca={"leakage_model": "HW"}, seed=0)
    with pytest.raises(mi.ModelImportError, match="fit_on applies to a fitted trace_normalization only"):
        composed(data="e", sca={"leakage_model": "HW"}, seed=0, fit_on="profiling")


def test_a_fitted_scaler_fits_where_the_import_says(composed):
    fitted = {"mode": "fitted", "steps": ["standard"]}
    training = composed(data="e", sca={"leakage_model": "HW"}, seed=0, trace_normalization=fitted)
    profiling = composed(data="e", sca={"leakage_model": "HW"}, seed=0, trace_normalization=fitted, fit_on="profiling")
    assert training["data"]["profiling_split"]["enabled"] is True and training["data"]["trace_normalization"] == fitted
    assert "profiling_split" not in profiling["data"]                                # every profiling trace


def test_the_input_length_must_be_the_data_entrys(composed, monkeypatch):
    import utils.config_loader as loader

    other = _project_cfg()
    other["data"]["e"]["target_range"] = [0, 700]
    monkeypatch.setattr(loader, "get_project_config", lambda project_name: copy.deepcopy(other))
    with pytest.raises(mi.ModelImportError, match="takes traces of 32 samples, the data entry 'e' gives 700"):
        composed(data="e", sca={"leakage_model": "HW"}, seed=0)


# ── registering: the run's files, never params.yaml ─────────────────────────────────────────────────────
def test_register_writes_the_run_and_never_params_yaml(tmp_path, monkeypatch):
    root = tmp_path / "P"
    (root / "config").mkdir(parents=True)
    params = root / "config" / "params.yaml"
    params.write_text("trainer: {epochs: 50}\n")
    before = params.read_bytes()
    monkeypatch.setattr(mi, "project_block", lambda project: ({"name": "P"}, root))
    cfg = {"project": {"name": "P", "root": str(root), "scripts": {}}, "data": {"name": "e"}, "model": [{"model_id": "imported"}],
           "model_id": "imported", "trainer": dict(mi.IMPORT_TRAINER), "seed": 0, "sca": {"leakage_model": "ID"},
           "paths": {"params_out": "artifacts/{{env}}/generated_models/{{model_name}}/config/{{model_name}}_params.yaml"},
           "env": "dev", mi.IMPORT_KEY: {"file": "released/a.h5", "sha256": "d" * 64, "format": "keras_h5", "source": "S"}}
    item = {"run": {"where": "models[0]", "name": "z"}, "cfg": cfg, "model_name": "z", "model_hash": "abcd1234",
            "record": cfg[mi.IMPORT_KEY]}
    monkeypatch.setattr(mi, "prepare", lambda project, env, names=(), config=None: [copy.deepcopy(item)])

    assert mi.register("P", env="dev") == ["z"]
    assert params.read_bytes() == before
    registry = json.loads((root / "artifacts" / "run_registry.json").read_text())
    entry = registry["abcd1234"]
    assert entry["status"] == "NEW" and entry["model_id"] == "imported" and entry["imported_snapshot"] is True
    assert entry["model_import"] == {"file": "released/a.h5", "sha256": "d" * 64, "format": "keras_h5", "source": "S"}
    assert "import" in entry["batches"]
    snapshot = root / "artifacts/dev/generated_models/z/config/z_params.yaml"
    from utils.data_reproducibility import read_params_snapshot
    assert read_params_snapshot(snapshot)[mi.IMPORT_KEY]["sha256"] == "d" * 64

    registry["abcd1234"]["status"] = "DONE"                                         # imported and tested
    from utils.sweep_engine import save_registry
    save_registry(registry, root)
    assert mi.register("P", env="dev") == []                                        # DONE: skipped
    assert mi.register("P", env="dev", force=True) == ["z"]                         # FORCE=1: again


# ── a PyTorch state dict: tensors in order, shapes checked ──────────────────────────────────────────────
def test_a_state_dict_is_copied_in_order_and_checked():
    target = torch.nn.Sequential(torch.nn.Conv1d(1, 2, 3), torch.nn.Flatten(), torch.nn.Linear(16, 4))
    theirs = torch.nn.Sequential(torch.nn.Conv1d(1, 2, 3), torch.nn.Flatten(), torch.nn.Linear(16, 4))
    state = {f"net.{k}": v for k, v in theirs.state_dict().items()}                 # their own key names
    assert miw.copy_state_dict(state, target) == 4
    assert all(torch.equal(a, b) for a, b in zip(target.state_dict().values(), theirs.state_dict().values()))
    with pytest.raises(mi.ModelImportError, match="holds 3 tensors"):
        miw.copy_state_dict(dict(list(state.items())[:3]), target)
    wrong = dict(state)
    wrong["net.2.weight"] = torch.zeros(4, 15)
    with pytest.raises(mi.ModelImportError, match="does not fit"):
        miw.copy_state_dict(wrong, target)


# ── the panel's Import a Model helpers (layout.py) ──────────────────────────────────────────────────────
def test_import_add_copies_the_file_in_and_writes_import_config_from_the_blueprint(tmp_path, monkeypatch):
    from services.visualization.vscode_monitor import layout

    shutil.copytree(REPO / layout.BLUEPRINT, tmp_path / layout.BLUEPRINT)
    (tmp_path / layout.MASTER).write_text('project:\n  - name: "P"\n    root: "P/"\n    params: "config/params.yaml"\n')
    folder = tmp_path / "projects" / "P"
    (folder / "config").mkdir(parents=True)
    monkeypatch.setattr(mi, "import_config_path", lambda project, override=None: folder / "config" / "import_config.yaml")
    outside = tmp_path / "download" / "ASCAD_desync0"
    outside.parent.mkdir()
    outside.write_bytes(b"x")

    out = layout.import_add(tmp_path, "P", str(outside), source="Zaid @ ece9760", data="ascad_n0_fs01",
                            leakage_model="ID", name="zaid_ascad_n0", fit_on="profiling")
    assert out["file"] == "released/ASCAD_desync0" and (folder / "released" / "ASCAD_desync0").read_bytes() == b"x"
    text = (folder / "config" / "import_config.yaml").read_text()
    assert "# Model import: test models other studies released" in text             # the blueprint, comments kept
    assert mi.load_import_config(folder / "config" / "import_config.yaml")["models"] == [{
        "name": "zaid_ascad_n0", "file": "released/ASCAD_desync0", "source": "Zaid @ ece9760", "data": "ascad_n0_fs01",
        "sca": {"leakage_model": "ID"}, "fit_on": "profiling"}]
    with pytest.raises(ValueError, match="exists already"):                          # never overwritten
        layout.import_add(tmp_path, "P", str(outside), source="S", data="d", leakage_model="ID")
    inside = layout.import_add(tmp_path, "P", str(folder / "released" / "ASCAD_desync0"), source="S", data="d",
                               leakage_model="ID")
    assert inside["copied"] is None and len(mi.load_import_config(folder / "config" / "import_config.yaml")["models"]) == 2
