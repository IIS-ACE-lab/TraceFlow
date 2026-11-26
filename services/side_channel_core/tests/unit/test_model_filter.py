# services/side_channel_core/tests/unit/test_model_filter.py
"""Unit tests for make find_model (models by parameters)."""

import json

import pytest
import yaml

from services.model_filter.model_filter import (
    ModelFilterError,
    RunView,
    evaluate,
    load_filter_config,
    run_model_filter,
)
from services.side_channel_core.utils.data_reproducibility import SNAPSHOT_DIGEST_KEY, snapshot_digest


def _snapshot_text(model_hash, name, **sections):
    """A params_out snapshot as write_params_snapshot lays it out: digest first, every required section."""
    body = {"project": {"name": "P"}, "data": {"name": "ascad"}, "model": [{"model_id": "M"}],
            "trainer": {"epochs": 10}, "seed": 1, "sca": {"leakage_model": "ID"}, "paths": {},
            "model_hash": model_hash, "model_name": name}
    body.update(sections)
    return yaml.safe_dump({SNAPSHOT_DIGEST_KEY: snapshot_digest(body), **body}, sort_keys=False)


def _run(model_hash, name, **sampled):
    return {
        "model_hash": model_hash, "model_name": name, "env": "dev", "status": "DONE",
        "model_id": sampled.pop("model_id", "M"), "path": f"artifacts/dev/generated_models/{name}",
        "sampled": {
            "model": [{"model_id": "M", "architecture": sampled.pop("arch", "CNN"),
                       "architecture_args": {"layers": sampled.pop("layers", [{"Conv1d": {"kernel_size": 11}}, {"Flatten": {}}])}}],
            "trainer": {"epochs": sampled.pop("epochs", 10), "optimizer": {"type": sampled.pop("opt", "Adam"), "learning_rate": sampled.pop("lr", 1e-3)}},
            "sca": {"leakage_model": sampled.pop("lm", "ID")},
        },
    }


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "P"
    (root / "artifacts").mkdir(parents=True)
    runs = [
        _run("h1", "cnn_a", opt="RMSprop", lr=1e-5),
        _run("h2", "cnn_b", opt="Adam", lr=1e-2, epochs=100),
        _run("h3", "mlp_c", arch="MLP", layers=[{"Linear": {}}] * 3, lm="HW"),
    ]
    runs[2]["status"] = "FAILED"
    (root / "artifacts" / "run_map.jsonl").write_text("\n".join(json.dumps(r) for r in runs) + "\n")
    snap = root / "artifacts/dev/generated_models/cnn_a/config"
    snap.mkdir(parents=True)
    (snap / "cnn_a_params.yaml").write_text(_snapshot_text(
        "h1", "cnn_a", cross_validation={"enabled": True, "n_folds": 10}, data={"name": "ascad", "test_range": [2000, 10000]}))
    return root


def _filter(tmp_path, where, **extra):
    path = tmp_path / "model_filter.yaml"
    path.write_text(yaml.safe_dump({"model_filter": {"where": where, **extra}}))
    return path


def test_conditions_select_by_parameters(project, tmp_path):
    cfg = _filter(tmp_path, [
        {"path": "model.architecture", "equals": "CNN"},
        {"path": "trainer.optimizer.learning_rate", "lte": 1e-3},
    ])
    out = run_model_filter(cfg, project="P", env="dev", projects_root=project.parent)
    assert out["models"] == ["cnn_a"]
    written = yaml.safe_load((project / "artifacts" / "filtered_models.yaml").read_text())
    assert written["models"] == ["cnn_a"] and written["details"][0]["model_hash"] == "h1"
    assert written["details"][0]["values"]["trainer.optimizer.learning_rate"] == 1e-5


@pytest.mark.parametrize("condition", [
    "equals: 1e-5", "equals: 1.0e-5", "equals: 0.00001", "equals: '1e-5'", "in: [1e-5, 1e-4]",
    "lt: 1e-4", "between: [1e-6, 1e-4]",
])
def test_a_float_written_without_a_dot_matches_the_float_the_run_stores(project, tmp_path, condition):
    # YAML 1.1 reads 1e-5 as the string '1e-5'; training coerces it, and so does the filter.
    path = tmp_path / "model_filter.yaml"
    path.write_text("model_filter:\n  where:\n    - path: trainer.optimizer.learning_rate\n"
                    f"      {condition}\n")
    out = run_model_filter(path, project="P", env="dev", projects_root=project.parent)
    assert out["models"] == ["cnn_a"]


def test_a_numeric_string_meets_only_a_number_and_strings_compare_as_written(project, tmp_path):
    path = tmp_path / "model_filter.yaml"
    path.write_text("model_filter:\n  where:\n    - path: trainer.optimizer.type\n      equals: '1e-5'\n")
    assert run_model_filter(path, project="P", env="dev", projects_root=project.parent)["models"] == []


def test_output_is_overwritten_and_failed_runs_are_skipped(project, tmp_path):
    run_model_filter(_filter(tmp_path, [{"path": "model.architecture", "equals": "CNN"}]),
                     project="P", projects_root=project.parent)
    out = run_model_filter(_filter(tmp_path, [{"path": "model.architecture", "equals": "MLP"}]),
                           project="P", projects_root=project.parent)
    assert out["models"] == []                      # mlp_c is FAILED
    out = run_model_filter(_filter(tmp_path, [{"path": "model.architecture", "equals": "MLP"}], status="any"),
                           project="P", projects_root=project.parent)
    assert out["models"] == ["mlp_c"]
    assert yaml.safe_load((project / "artifacts" / "filtered_models.yaml").read_text())["models"] == ["mlp_c"]


def test_snapshot_wildcard_and_operators(project, tmp_path):
    view = RunView("h1", json.loads((project / "artifacts/run_map.jsonl").read_text().splitlines()[0]), project)
    cases = [
        ({"path": "cross_validation.n_folds", "equals": 10}, True),          # snapshot fallback
        ({"path": "data.test_range.1", "equals": 10000}, True),               # list index
        ({"path": "model.architecture_args.layers.*.Conv1d.kernel_size", "equals": 11}, True),
        ({"path": "model.architecture_args.layers", "count": {"min": 2}}, True),
        ({"path": "model_id", "matches": "M.*"}, True),
        ({"path": "trainer.optimizer.type", "in": ["Adam", "SGD"]}, False),
        ({"path": "nope.nothing", "exists": False}, True),
        ({"path": "trainer.epochs", "between": [5, 20]}, True),
    ]
    for cond, expected in cases:
        cond = {"quantifier": "any", **cond}
        op = next(k for k in cond if k not in ("path", "quantifier"))
        normalized = {"path": cond["path"], "op": op, "value": cond[op], "quantifier": "any"}
        if op == "count" and isinstance(normalized["value"], int):
            normalized["value"] = {"min": normalized["value"], "max": normalized["value"]}
        assert evaluate(view, normalized)[0] is expected, cond


@pytest.mark.parametrize(
    "block, message",
    [
        ({"where": []}, "non-empty"),
        ({"where": [{"path": "x", "equals": 1, "gt": 2}]}, "exactly one operator"),
        ({"where": [{"path": "x", "between": [1]}]}, "low, high"),
        ({"where": [{"path": "x", "matches": "("}]}, "regular expression"),
        ({"where": [{"path": "x", "equals": 1}], "match": "some"}, "match"),
        ({"where": [{"path": "x", "equals": 1}], "foo": 1}, "Unknown key"),
    ],
)
def test_bad_configs_are_rejected(tmp_path, block, message):
    path = tmp_path / "f.yaml"
    path.write_text(yaml.safe_dump({"model_filter": block}))
    with pytest.raises(ModelFilterError, match=message):
        load_filter_config(path)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "f.yaml"
    path.write_text("")
    with pytest.raises(ModelFilterError, match="empty"):
        load_filter_config(path)


def test_models_follow_the_order_of_the_in_lists(tmp_path):
    root = tmp_path / "P"
    (root / "artifacts").mkdir(parents=True)
    runs = [  # run_map (training) order deliberately differs from the asked order
        _run("h1", "e130_b200", epochs=130, opt="RMSprop"),
        _run("h2", "e10_b200", epochs=10, opt="RMSprop"),
        _run("h3", "e50_adam", epochs=50, opt="Adam"),
        _run("h4", "e50_b200", epochs=50, opt="RMSprop"),
        _run("h5", "e25_b200", epochs=25, opt="RMSprop"),
    ]
    (root / "artifacts" / "run_map.jsonl").write_text("\n".join(json.dumps(r) for r in runs) + "\n")

    epochs = [{"path": "trainer.epochs", "in": [10, 25, 50, 130]}]
    out = run_model_filter(_filter(tmp_path, epochs), project="P", env="dev", projects_root=tmp_path)
    # equal epochs keep the training order (h3 before h4)
    assert out["models"] == ["e10_b200", "e25_b200", "e50_adam", "e50_b200", "e130_b200"]
    assert [d["model_name"] for d in out["details"]] == out["models"]

    # the first condition orders first; a later in: list breaks its ties
    where = epochs + [{"path": "trainer.optimizer.type", "in": ["RMSprop", "Adam"]}]
    out = run_model_filter(_filter(tmp_path, where), project="P", env="dev", projects_root=tmp_path)
    assert out["models"] == ["e10_b200", "e25_b200", "e50_b200", "e50_adam", "e130_b200"]

    # only in: lists carry an order; other operators keep the training order
    out = run_model_filter(_filter(tmp_path, [{"path": "trainer.epochs", "gte": 10}]),
                           project="P", env="dev", projects_root=tmp_path)
    assert out["models"] == ["e130_b200", "e10_b200", "e50_adam", "e50_b200", "e25_b200"]


def test_pipe_in_equals_means_any_of_the_names_in_that_order(tmp_path):
    root = tmp_path / "P"
    (root / "artifacts").mkdir(parents=True)
    runs = [_run("h1", "on_d0", lm="ID"), _run("h2", "on_hw", lm="HW"), _run("h3", "on_msb", lm="MSB")]
    (root / "artifacts" / "run_map.jsonl").write_text("\n".join(json.dumps(r) for r in runs) + "\n")

    out = run_model_filter(_filter(tmp_path, [{"path": "sca.leakage_model", "equals": "HW | ID"}]),
                           project="P", env="dev", projects_root=tmp_path)
    assert out["models"] == ["on_hw", "on_d0"]               # HW first, as written
    assert out["where"] == [{"path": "sca.leakage_model", "in": ["HW", "ID"]}]

    out = run_model_filter(_filter(tmp_path, [{"path": "sca.leakage_model", "not_equals": "HW|ID"}]),
                           project="P", env="dev", projects_root=tmp_path)
    assert out["models"] == ["on_msb"]

    with pytest.raises(ModelFilterError, match=r"empty alternative"):
        load_filter_config(_filter(tmp_path, [{"path": "sca.leakage_model", "equals": "HW||ID"}]))


def test_the_fields_a_condition_can_name_are_listed_by_where_the_filter_finds_them(project):
    from services.model_filter.model_filter import filter_fields

    out = filter_fields(project)
    fields = {f["path"]: f for f in out["fields"]}
    assert out["runs"] == 3 and "exists" in out["operators"]
    assert fields["status"]["source"] == "run" and fields["status"]["values"] == ["DONE", "FAILED"] and fields["status"]["varies"]
    assert fields["trainer.optimizer.type"]["source"] == "sweep" and fields["trainer.optimizer.type"]["values"] == ["Adam", "RMSprop"]
    assert fields["trainer.epochs"]["kind"] == "number" and fields["trainer.epochs"]["values"] == [10, 100]
    assert fields["model.architecture_args.layers.*.Conv1d.kernel_size"]["values"] == [11]   # one model block: model.; a list of blocks: `*`
    assert fields["cross_validation.n_folds"]["source"] == "snapshot" and fields["cross_validation.n_folds"]["runs"] == 1
    assert "data.test_range" in fields and fields["data.test_range"]["kind"] == "list"
    assert not any(p.startswith(("sampled.", "paths.")) for p in fields)


def test_ticked_conditions_are_written_as_by_hand_and_the_rest_of_the_file_stays(tmp_path):
    from services.model_filter.model_filter import write_where

    path = tmp_path / "model_filter.yaml"
    path.write_text("# my filter\nmodel_filter:\n  match: all   # every condition\n  where:\n    - path: batches\n      contains: fig6\n"
                    "    # - path: trainer.epochs\n    #   in: [10]\n")
    write_where(path, [{"path": "batches", "contains": "fig26"}, {"path": "trainer.epochs", "in": [100, 200]},
                       {"path": "trainer.optimizer.learning_rate", "between": [1e-6, 1e-3]}])
    assert path.read_text() == ("# my filter\nmodel_filter:\n  match: all   # every condition\n  where:\n"
                                "    - path: batches\n      contains: fig26\n    - path: trainer.epochs\n      in: [100, 200]\n"
                                "    - path: trainer.optimizer.learning_rate\n      between: [1.0e-06, 0.001]\n"
                                "    # - path: trainer.epochs\n    #   in: [10]\n")
    assert load_filter_config(path)["where"][1] == {"path": "trainer.epochs", "op": "in", "value": [100, 200], "quantifier": "any"}
    with pytest.raises(ModelFilterError, match="one condition"):
        write_where(path, [])
    with pytest.raises(ModelFilterError, match="exactly one operator"):
        write_where(path, [{"path": "batches"}])
