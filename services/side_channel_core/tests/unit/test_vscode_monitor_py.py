# services/side_channel_core/tests/unit/test_vscode_monitor_py.py
"""Python helpers of the TraceFlow Monitor (services/visualization/vscode_monitor)."""
from __future__ import annotations

import pytest
import yaml

from services.visualization.vscode_monitor.make_blueprint import empty_text

PARAMS = """\
# header comment
seed: 42   # the seed
trainer:
  epochs: [20, 50]        # sweep
  batch_size:
    - 100
    - 200
  optimizer:
    - type: RMSprop       # first
      learning_rate: [1.0e-5]
      args: {rho: 0.9, epsilon: 1.0e-7}
    # between entries
    - type: Adam
      learning_rate: [1.0e-4]
  # after the list
  loss: "CrossEntropyLoss"
labels:
  square_popsicle: "epochs= 20"
  indeed_mesh: "epochs= 800"
name: "a # not a comment"
"""


def test_empty_text_keeps_keys_and_comments_and_empties_values():
    out = empty_text(PARAMS)
    data = yaml.safe_load(out)
    assert data == {
        "seed": None,
        "trainer": {
            "epochs": [],
            "batch_size": [],
            "optimizer": [{"type": None, "learning_rate": [], "args": {"rho": None, "epsilon": None}}],
            "loss": None,
        },
        "labels": {},
        "name": None,
    }
    for comment in ("# header comment", "# the seed", "# sweep", "# first", "# after the list"):
        assert comment in out
    assert "# between entries" not in out          # it sat between deleted entries
    assert "Adam" not in out and "popsicle" not in out


def test_named_key_file_keeps_one_example_key():
    queue = """\
# queue header

# Fig. 2 settings
fig2:
  - make orchestrate MODEL_ID=A
# Fig. 3 settings
fig3:
  - make orchestrate MODEL_ID=B
"""
    out = empty_text(queue, named_keys=True)
    assert yaml.safe_load(out) == {"example": []}
    assert "# queue header" in out and "Fig. 2" not in out and "Fig. 3" not in out


DEFAULTS = {         # params.yaml settings a new project starts with (owner 2026-10-22, later)
    "num_workers": {"train": 4, "val": 1},
    "monitor_interval": 1,
    "dataloader": {"pin_memory": True, "persistent_workers": True, "prefetch_factor": 4, "drop_last_train": True},
    "perform_attack": {"enabled": True,
                       "validation": {"attack_times": 1, "shuffle": False, "attack_distribution": "full_population"},
                       "test": {"attack_times": 1, "shuffle": False, "attack_distribution": "full_population"},
                       "attack_step": [1], "ge_thresholds": 1},
    "validation": {"rank_eval_interval": 1, "rank_step": [1]},
    "key_recovery_reporting": {"enabled": True, "log_validation": True, "log_test": True,
                               "save_validation_reports": True, "save_test_report": True, "recovery_rank": 1,
                               "require_stable_recovery": True, "top_k_candidates": 5, "include_score_margin": True,
                               "key_display": "full"},
}


def test_blueprint_files_start_a_project_with_every_key_empty_but_the_paths_and_the_defaults():
    """Owner 2026-10-22: a new project starts with every key the system reads present and empty (the panel's
    menus offer the choices), the standard paths layout filled, data entries, models, queues and single runs
    empty (Add a ... in the panel). Owner 2026-10-22, later: the data-loading, monitor, attack, rank-grid and
    key-recovery-reporting settings start with defaults (DEFAULTS)."""
    from pathlib import Path

    from services.visualization.vscode_monitor.layout import BLUEPRINT, SCAFFOLD_FILES

    def leaves(node, path=()):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from leaves(v, (*path, str(k)))
        elif isinstance(node, list) and node and all(isinstance(v, dict) for v in node):
            for v in node:
                yield from leaves(v, (*path, "*"))
        else:
            yield ".".join(path), node

    root = Path(__file__).resolve().parents[4]
    for name in SCAFFOLD_FILES:
        if name == "graph_config.yaml":
            continue
        data = yaml.safe_load((root / BLUEPRINT / name).read_text())
        filled = {p: v for p, v in leaves(data if data is not None else {}) if v not in (None, [], {})}
        if name == "params.yaml":
            # the paths, metrics_cfg's numeric constants (owner 2026-10-22: no study's choice; DLSCA_ASCAD's) and
            # the defaults every new project starts with (owner 2026-10-22, later)
            assert set(filled) == ({f"paths.{k}" for k in data["paths"]} | {f"metrics_cfg.{k}" for k in data["metrics_cfg"]}
                                   | {p for p in filled if p.split(".")[0] in DEFAULTS}), name
            assert all(isinstance(v, str) and v for k, v in filled.items() if k.startswith("paths."))
            assert data["metrics_cfg"] == {"ce_eps": 1e-12, "rank_log_eps": 1e-40, "preds_are_probs": False, "lm_enabled": False,
                                           "strict_prob_check": True, "prob_atol": 1e-3, "prob_rtol": 1e-3}
            assert {key: data[key] for key in DEFAULTS} == DEFAULTS
            assert data["data"] == [] and data["model"] == [] and data["sca"]["attack_byte"] == []
        else:
            assert filled == {}, (name, filled)
    assert yaml.safe_load((root / BLUEPRINT / "experiment_queue.yaml").read_text()) == {}
    assert yaml.safe_load((root / BLUEPRINT / "single_runs.yaml").read_text()) == {"runs": []}
    templates = yaml.safe_load((root / BLUEPRINT / "templates.yaml").read_text())
    # every value empty but a data entry's three print / guard switches (owner 2026-10-23: true) and its
    # profiling_split switch (owner 2026-10-24: always written, false trains on all of profiling_range)
    assert {p: v for p, v in leaves(templates) if v not in (None, [], {})} == {
        "data.debug_h5_inspect": True, "data.show_contract": True, "data.require_contract": True,
        "data.profiling_split.enabled": False}, "templates.yaml"


def test_add_a_dataset_writes_the_blueprint_data_entry():
    """Add a Dataset (params.yaml data): templates.yaml's data entry (a new project's data: starts empty)."""
    from pathlib import Path

    from services.visualization.vscode_monitor.choices import data_template, params_forms
    templates = Path(__file__).resolve().parents[2] / "config" / "blueprint" / "templates.yaml"
    entry = yaml.safe_load(templates.read_text(encoding="utf-8"))["data"]
    assert data_template() == entry == params_forms()["data_template"]
    assert {"name", "type", "database", "profiling_range", "attack_range"} <= set(entry)


def test_the_panel_gets_every_blueprint_file_its_hints_and_the_list_templates():
    """choices.blueprint_forms: what a key a project's file lacks is drawn from (its value, menu and hint)."""
    from services.visualization.vscode_monitor.choices import blueprint_forms
    from services.visualization.vscode_monitor.layout import SCAFFOLD_FILES

    bp = blueprint_forms()
    assert set(bp["files"]) == set(SCAFFOLD_FILES) - {"graph_config.yaml"}
    params = bp["files"]["params.yaml"]
    assert "patience" in params["tree"]["trainer"]["early_stopping"] and params["tree"]["trainer"]["early_stopping"]["patience"] is None
    assert params["hints"]["trainer.early_stopping.patience"].startswith("epochs without improvement")
    assert params["hints"]["sca.leakage_model"].startswith("The label: ID")       # the comment lines above it
    assert bp["lists"] == {"data": "data", "model": "model", "trainer.optimizer": "optimizer",
                           "model_import.models": "import_model"}     # an import's keys: the defaults' (2026-10-27)
    assert set(bp["templates"]["import_model"]) == {"data", "trace_normalization", "fit_on", "sca", "seed", "perform_attack"}
    assert set(bp["templates"]["import_model"]["perform_attack"]["test"]) == {"attack_times", "shuffle", "attack_distribution"}
    assert set(bp["templates"]) == {"data", "model", "optimizer", "import_model"}
    assert bp["template_hints"]["data.type"].startswith("ASCADf")


def test_metric_options_are_the_values_sca_metrics_reads():
    """METRIC_OPTIONS (the panel asks these): each option names the recorded column; a bare name is its default."""
    from services.visualization.vscode_monitor.choices import params_forms
    from sca.metrics import METRIC_OPTIONS, requested_column
    for name, (opt, default) in METRIC_OPTIONS.items():
        assert requested_column(name, {opt: default}) != requested_column(name, {opt: default + 1}), name
    assert requested_column("top_k_accuracy") == requested_column("top_k_accuracy", {"k": 5}) == "top_5_accuracy"
    assert requested_column("rank_k_success") == requested_column("rank_k_success", {"k": 1}) == "rank_k_success@1"
    assert requested_column("traces_to_ge_k", {"k": 3}) == "traces_to_ge_3"
    assert params_forms()["metric_options"]["top_k_accuracy"] == {"option": "k", "default": 5}


def test_master_control_check_and_template():
    """check-master loads every project of master.yaml and resolves the timezone; project-template is
    blueprint/master_project.yaml under the name given, the same keys as a declared project's entry."""
    from pathlib import Path

    from services.visualization.vscode_monitor.layout import MASTER, check_master, project_template
    from utils import config_schema
    root = Path(__file__).resolve().parents[4]
    master = yaml.safe_load((root / MASTER).read_text(encoding="utf-8"))
    out = check_master(root)
    assert out["errors"] == [] and set(out["projects"]) == {p["name"] for p in master["project"]}
    assert all(err == "" for err in out["projects"].values()) and out["timezone"]
    entry = project_template(root, "NEW_P")
    assert (entry["name"], entry["root"], entry["description"]) == ("NEW_P", "NEW_P/", "")
    assert set(master["project"][0]) <= set(entry), "every key a declared entry has (the template may offer more: auto_mlops)"
    config_schema.check_keys(entry, config_schema.MASTER_PROJECT, source="master_project.yaml")


def test_single_runs_are_make_orchestrate_lines_of_their_project():
    """RUN SINGLE EXPERIMENT: each line of single_runs.yaml with its variables, or why it cannot run."""
    from services.visualization.vscode_monitor import layout

    runs = layout._single_runs({"runs": [
        "make orchestrate ENV=dev PROJECT=P DATASET=ascad MODEL_ID=A",
        "make orchestrate ENV=dev DATASET=ascad MODEL_ID=A",                    # PROJECT left out: this one
        "make orchestrate ENV=dev PROJECT=Q DATASET=ascad MODEL_ID=A",
        "make orchestrate ENV=dev PROJECT=P DATASET=ascad MODEL_ID=A BATCH=x",
        "make orchestrate ENV=dev PROJECT=P DATASET=ascad",
        "make experiment QUEUE=fig2",
        None,
    ]}, "P")
    assert [r["error"] == "" for r in runs] == [True, True, True, False, False, False, False]
    assert (runs[0]["ENV"], runs[0]["DATASET"], runs[0]["MODEL_ID"], runs[0]["index"]) == ("dev", "ascad", "A", 0)
    assert runs[1]["PROJECT"] == "P" and runs[2]["PROJECT"] == "Q"           # left out: this project; another named: it
    assert "BATCH=x" in runs[3]["error"]
    assert runs[4]["error"] == "ENV, DATASET and MODEL_ID are needed" and runs[5]["error"] == "not a make orchestrate line"
    assert layout._single_runs({"runs": None}, "P") == [] and layout._single_runs(None, "P") == []


def test_the_blueprint_single_runs_is_scaffolded_and_empty(tmp_path):
    import shutil
    from pathlib import Path

    from services.visualization.vscode_monitor import layout

    root = Path(__file__).resolve().parents[4]
    shutil.copytree(root / layout.BLUEPRINT, tmp_path / layout.BLUEPRINT)
    (tmp_path / layout.MASTER).write_text('project:\n  - name: "P"\n    root: "P/"\n    params: "config/params.yaml"\n')
    created = layout.scaffold(tmp_path)
    assert "projects/P/config/single_runs.yaml" in created and len(created) == 12, "the eleven config files and the README"
    assert "projects/P/config/import_config.yaml" in created           # make import's list (extra_configs.import)
    assert yaml.safe_load((tmp_path / "projects/P/config/single_runs.yaml").read_text()) == {"runs": []}
    # the project README template (owner 2026-10-23): the figure index and the set-before-each-queue table
    readme = (tmp_path / "projects/P/README.md").read_text(encoding="utf-8")
    assert "projects/P/README.md" in created and "## Figure index: model, dataset, leakage model, batch size, epochs, optimizer" in readme
    assert "### CNN figures" in readme and "### MLP figures" in readme and "**Set before each queue.**" in readme
    assert layout.scaffold(tmp_path) == [], "nothing copied twice"


def test_master_timezone_menus_offer_what_the_resolver_reads():
    """timezone.current_country: the countries with zones; current_location: that country's zone cities, the
    principal one first; every pair offered resolves (utils/timestamps.resolve_timezone)."""
    from services.side_channel_core.utils.timestamps import resolve_timezone
    from services.visualization.vscode_monitor.choices import _timezone_rules

    country, city = _timezone_rules()
    assert (country["p"], city["p"], city["by"]["key"], city["by"]["at"]) == (
        "timezone.current_country", "timezone.current_location", "current_country", 1)
    assert "Taiwan" in country["o"] and city["by"]["map"]["Taiwan"] == ["Taipei"] == city["by"]["map"]["TW"]
    assert city["by"]["map"]["Germany"][0] == "Berlin" and city["by"]["map"]["US"][0] == "New York"
    assert resolve_timezone("Taipei", "Taiwan") == "Asia/Taipei"
    for name in country["o"]:
        for place in city["by"]["map"][name]:
            assert resolve_timezone(place, name)


def test_model_forms_names_the_layers_and_types_a_layer_list_cannot_build():
    """Plan a Model: the menus show what core.model cannot build from a layer list, disabled with the reason."""
    from core.model import LAYER_LIST_LIMITS
    from services.visualization.vscode_monitor.choices import model_forms

    out = model_forms()
    assert out["limits"] == LAYER_LIST_LIMITS and set(out["limits"]) <= set(out["layers"])
    assert set(out["disabledKinds"]) == {"TRANSFORMER", "HYBRID"}
    assert "MultiheadAttention" in out["disabledKinds"]["TRANSFORMER"] and "LSTM" in out["disabledKinds"]["HYBRID"]
    assert out["kerasKinds"] == ["CNN", "MLP"]


def _fake_project(monkeypatch, framework, layers, model_id="m"):
    import services.side_channel_core.utils.config_loader as loader

    cfg = {"deep_learning": {"framework": framework}, "sca": {"leakage_model": ["ID"], "attack_byte": [2]},
           "data": {"ascad": {"target_range": [0, 100]}},
           "model": [{"model_id": model_id, "architecture": "CNN", "architecture_args": {"layers": layers}}]}
    monkeypatch.setattr(loader, "get_project_config", lambda *a, **k: cfg)
    return cfg


_SMALL_CNN = [{"Conv1d": {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1}},
              {"BatchNorm1d": {"num_features": 4, "activation": "relu"}},
              {"Pooling": {"dim": 1, "type": "max", "kernel_size": 2, "stride": 2}}, {"Flatten": {}},
              {"Linear": {"in_features": 200, "out_features": "auto"}}]


@pytest.mark.parametrize("model_id", ["ASCAD_CNNbase", "ASCAD_CNNbest", "ASCAD_MLPbest", "ASCAD_VGG16"])
def test_an_unsaved_layer_list_checks_like_the_saved_model(model_id):
    from services.side_channel_core.utils.config_loader import get_project_config
    from services.visualization.vscode_monitor import model_check

    model = next(m for m in get_project_config("DLSCA_ASCAD", validate_data=False)["model"] if m["model_id"] == model_id)
    saved = model_check.check("DLSCA_ASCAD", model_id)
    unsaved = model_check.check_layers("DLSCA_ASCAD", model["architecture_args"]["layers"], model["architecture"])
    assert saved["error"] is None and saved.pop("fits") == ["ascad"] and {**saved, "model_id": None} == unsaved   # fits: the saved model's entry (one per trace length)


def test_check_layers_reports_the_layer_that_breaks(monkeypatch):
    from services.visualization.vscode_monitor import model_check

    _fake_project(monkeypatch, "pytorch", [])
    layers = [dict(l) for l in _SMALL_CNN]
    layers[4] = {"Linear": {"in_features": 7, "out_features": "auto"}}
    out = model_check.check_layers("P", layers)
    assert out["at"] == 4 and "7" in out["error"] and len(out["layers"]) == 4


def test_a_keras_project_is_checked_with_keras(monkeypatch):
    from services.visualization.vscode_monitor import model_check

    layers = [dict(l) for l in _SMALL_CNN]
    layers[2] = {"Pooling": {"dim": 1, "type": "max", "kernel_size": 2, "stride": 2, "padding": 1}}
    layers[4] = {"Linear": {"in_features": 204, "out_features": "auto"}}
    _fake_project(monkeypatch, "pytorch", [])
    assert model_check.check_layers("P", layers)["error"] is None           # PyTorch pads pooling freely
    _fake_project(monkeypatch, "keras", [])
    out = model_check.check_layers("P", layers)
    assert out["framework"] == "keras" and "no Keras equivalent" in out["error"]


def test_a_pytorch_project_is_checked_with_pytorch(monkeypatch):
    from services.visualization.vscode_monitor import model_check

    _fake_project(monkeypatch, "pytorch", [])
    out = model_check.check_layers("P", _SMALL_CNN)
    assert (out["framework"], out["error"]) == ("pytorch", None) and out["layers"][-1]["out"] == [256]


def test_the_unsaved_list_builds_under_both_frameworks(monkeypatch):
    from services.visualization.vscode_monitor import model_check

    _fake_project(monkeypatch, "pytorch", [])
    torch_out = model_check.check_layers("P", _SMALL_CNN)
    _fake_project(monkeypatch, "keras", [])
    keras_out = model_check.check_layers("P", _SMALL_CNN)
    assert torch_out["error"] is None and keras_out["error"] is None
    assert torch_out["layers"] == keras_out["layers"] and torch_out["params"] == keras_out["params"]


def test_the_packed_extension_holds_every_webview_file():
    from services.visualization.vscode_monitor.pack import FILES, HERE

    webview = {f"webview/{p.name}" for p in (HERE / "webview").iterdir() if p.suffix in {".html", ".css", ".js"}}
    assert webview <= set(FILES)


def test_describe_gives_the_planner_the_framework_the_input_and_the_files_models():
    from services.visualization.vscode_monitor import model_check

    from pathlib import Path

    params = Path(__file__).resolve().parents[4] / "projects" / "DLSCA_ASCAD" / "config" / "params.yaml"
    out = model_check.describe("DLSCA_ASCAD", str(params))
    assert out["framework"] in ("pytorch", "keras") and out["input"] == [1, 700] and out["dataset"]
    ids = [m["model_id"] for m in out["models"]]
    assert "ASCAD_CNNbase" in ids and ids == [m["model_id"] for m in yaml.safe_load(params.read_text())["model"]]


def test_a_project_folder_master_yaml_does_not_declare_is_an_orphan_with_what_is_left_of_it(tmp_path):
    """Owner 2026-10-22: a project whose master.yaml entry is gone (or was never written) is still shown, as
    "master.yaml integration missing", with its root, config and artifacts present or deleted."""
    from services.visualization.vscode_monitor import layout

    (tmp_path / layout.MASTER).parent.mkdir(parents=True)
    (tmp_path / layout.MASTER).write_text('project:\n  - name: "A"\n    root: "A/"\n    params: "config/params.yaml"\n')
    for name, parts in {"A": ("config",), "GONE": ("config", "artifacts"), "NO_ART": ("config",), "notes": ()}.items():
        (tmp_path / "projects" / name).mkdir(parents=True)
        for part in parts:
            (tmp_path / "projects" / name / part).mkdir()
    (tmp_path / "projects" / "notes" / "paper.md").write_text("")
    (tmp_path / "projects" / ".hidden").mkdir()
    declared = layout.declared_projects(tmp_path, [])
    orphans = layout.orphan_projects(tmp_path, declared)
    assert [(o["name"], o["config"], o["artifacts"], o["present"]) for o in orphans] == \
        [("GONE", True, True, True), ("NO_ART", True, False, True)]           # A is declared; notes and .hidden are no projects
    assert orphans[0]["root"] == str(tmp_path / "projects" / "GONE")
    assert [o["name"] for o in layout.export(tmp_path)["orphans"]] == ["GONE", "NO_ART"]


def test_complete_project_adds_the_blueprint_keys_a_projects_file_lacks(tmp_path, monkeypatch):
    """Owner 2026-10-22: a project from an older template (or the blueprint grown since) gets the keys the
    blueprint has: empty ones written empty, the lists a project fills left alone; hash-safe, a key with a
    value (a path, a constant, an empty block) is left for the panel's Complete; --all adds those too.
    Comments and every other line stay."""
    from services.visualization.vscode_monitor import make_blueprint as mb

    config = tmp_path / "config"
    config.mkdir()
    old = ('# my notes\nseed: 42   # kept\ntrainer:\n  epochs: [10]\n  loss:\n    type: CrossEntropyLoss\n'
           'cross_validation:\n  enabled: true\ndata:\n  - name: d\nmodel: []\nmetrics_to_log: [accuracy]\n')
    (config / "params.yaml").write_text(old)
    (config / "model_judge.yaml").write_text("model_judge:\n  ranking:\n    primary: {process: test, metric: accuracy}\n")
    report = mb.complete_project(config)
    added = report["params.yaml"]["added"]
    assert "trainer.early_stopping" in added and "cross_validation.refit" in added and "sca" in added and "dataloader.drop_last_train" not in added
    assert "trainer.evaluation" in added, "a block of empty keys only (output_transformation's type and args.dim): added"
    assert "paths" in report["params.yaml"]["left"] and "metrics_cfg" in report["params.yaml"]["left"]
    text = (config / "params.yaml").read_text()
    data = yaml.safe_load(text)
    assert text.startswith("# my notes\nseed: 42   # kept\n") and data["seed"] == 42 and data["trainer"]["epochs"] == [10]
    assert data["data"] == [{"name": "d"}] and data["model"] == [] and data["metrics_to_log"] == ["accuracy"]
    assert data["trainer"]["early_stopping"] == {"enabled": None, "monitor_metric": None, "patience": None, "min_delta": None}
    assert data["cross_validation"]["refit"] == {"enabled": None}
    ke = data["sca"]["key_enumeration"]
    assert ke["enabled"] is None and ke["mode"] is None and set(ke["modes"]) == {"local_hamming", "neighbour_sweep", "support_weighted"}
    assert "paths" not in data
    assert "patience: null" not in text and "enabled: null" not in text, "written empty, not null"
    assert "model_judge.yaml" in report and "model_judge.filters" in report["model_judge.yaml"]["added"], "a missing block, as a whole"
    again = mb.complete_project(config)
    assert again["params.yaml"]["added"] == [] and again["params.yaml"]["left"] == report["params.yaml"]["left"], "nothing added the second time"
    assert yaml.safe_load((config / "params.yaml").read_text())["sca"]["attack_byte"] == [], "a block's empty list stays a list"
    # --all: the rest too; then nothing is left at all
    everything = mb.complete_project(config, hash_safe=False)
    assert "paths" in everything["params.yaml"]["added"] and everything["params.yaml"]["left"] == []
    data = yaml.safe_load((config / "params.yaml").read_text())
    assert data["paths"]["logs"].startswith("artifacts/{{env}}") and data["metrics_cfg"]["ce_eps"] == 1e-12
    assert mb.complete_project(config, hash_safe=False) == {}
    # a retired key is never written, whatever the blueprint holds
    (config / "params.yaml").write_text("seed: 1\n")
    monkeypatch.setattr(mb, "BLUEPRINT", tmp_path / "bp")
    (tmp_path / "bp").mkdir()
    for name in mb.COMPLETED_FILES:
        (tmp_path / "bp" / name).write_text("seed:\nsca:\n  rank_base:\n  attack_byte: []\n" if name == "params.yaml" else "")
    r = mb.complete_project(config, hash_safe=False)
    assert r["params.yaml"]["added"] == ["sca"] and yaml.safe_load((config / "params.yaml").read_text())["sca"] == {"attack_byte": []}


def test_scaffold_completes_every_declared_projects_files(tmp_path):
    """layout scaffold: the files copied, then every declared project's files completed, hash-safe."""
    import shutil
    from pathlib import Path

    from services.visualization.vscode_monitor import layout

    root = Path(__file__).resolve().parents[4]
    shutil.copytree(root / layout.BLUEPRINT, tmp_path / layout.BLUEPRINT)
    (tmp_path / layout.MASTER).write_text('project:\n  - name: "P"\n    root: "P/"\n    params: "config/params.yaml"\n'
                                           '  - name: "OLD"\n    root: "OLD/"\n    params: "config/params.yaml"\n')
    (tmp_path / "projects" / "OLD" / "config").mkdir(parents=True)
    (tmp_path / "projects" / "OLD" / "config" / "params.yaml").write_text("seed: 1\ntrainer:\n  epochs: [1]\n")
    created = layout.scaffold(tmp_path)
    assert len(created) == 23, "P's eleven files and README, OLD's ten and README (its params.yaml is its own)"
    done = layout.complete_projects(tmp_path)
    assert set(done) == {"projects/OLD/config/params.yaml"}
    assert "trainer.early_stopping" in done["projects/OLD/config/params.yaml"]["added"] and "paths" in done["projects/OLD/config/params.yaml"]["left"]
    assert yaml.safe_load((tmp_path / "projects/OLD/config/params.yaml").read_text())["seed"] == 1
    assert all(r["added"] == [] for r in layout.complete_projects(tmp_path).values()), "nothing added the second time"


def test_a_project_without_a_data_entry_is_checked_against_a_typed_input_length(monkeypatch):
    """Plan a Model in a project that has no data entry yet (every project but the first starts so): the page's
    typed trace length stands in for the entry's target_range; without either the check says what to do."""
    from services.visualization.vscode_monitor import model_check

    cfg = _fake_project(monkeypatch, "pytorch", [])
    cfg["data"] = {}
    out = model_check.check_layers("P", _SMALL_CNN)
    assert out["input"] is None and out["error"] == (
        "no data entry in params.yaml yet: type the trace length the first layer sees (Add a Dataset gives it)")
    out = model_check.check_layers("P", _SMALL_CNN, input_length=100)
    assert (out["dataset"], out["input"], out["error"]) == (None, [1, 100], None) and out["layers"][-1]["out"] == [256]
    out = model_check.check_layers("P", _SMALL_CNN, dataset="ascad", input_length=100)
    assert out["error"] == "no data entry 'ascad' in params.yaml", "a named entry is never replaced by a typed length"


def test_describe_lists_every_data_entry_with_its_trace_length_for_the_planners_input_menu(monkeypatch, tmp_path):
    from services.visualization.vscode_monitor import model_check

    cfg = _fake_project(monkeypatch, "pytorch", [])
    cfg["data"] = {"ascad": {"target_range": [0, 100]}, "raw": {"target_range": [45400, 46100]}, "odd": {}}
    params = tmp_path / "params.yaml"
    params.write_text("model: []\n", encoding="utf-8")
    out = model_check.describe("P", str(params))
    assert out["entries"] == [{"name": "ascad", "length": 100}, {"name": "raw", "length": 700}, {"name": "odd", "length": None}]
    assert (out["input"], out["dataset"], out["error"]) == ([1, 100], "ascad", None)
    cfg["data"] = {}
    out = model_check.describe("P", str(params))
    assert out["entries"] == [] and out["input"] is None and out["error"].startswith("no data entry in params.yaml yet")


def test_the_normalization_union_follows_the_loaders_validator():
    """data[].trace_normalization: the panel shows the chosen mode's keys (choices.normalization_forms from
    sca/ascad_data_loader.NORMALIZATION_KEYS); every supported mode has an entry, eps belongs to zscore and
    minmax only, and the template's entry starts with the mode alone."""
    from sca.ascad_data_loader import _SUPPORTED_NORMALIZATION_MODES, NORMALIZATION_KEYS, _normalization_from_data_cfg
    from services.visualization.vscode_monitor.choices import data_template, normalization_forms

    assert set(NORMALIZATION_KEYS) == set(_SUPPORTED_NORMALIZATION_MODES)
    assert {m for m, keys in NORMALIZATION_KEYS.items() if "eps" in keys} == {"zscore", "minmax"}
    forms = normalization_forms()
    assert forms["keys"] == {m: list(k) for m, k in NORMALIZATION_KEYS.items()}
    assert set(forms["hints"]) == {k for keys in NORMALIZATION_KEYS.values() for k in keys}
    assert data_template()["trace_normalization"] == {"mode": None}
    # the validator reads exactly those keys: a mode given its keys resolves, and reads no other
    out = _normalization_from_data_cfg({"trace_normalization": {"mode": "fitted", "steps": ["standard", "minmax"], "feature_range": [-1, 1]}})
    assert out["steps"] == ["standard", "minmax"] and out["feature_range"] == [-1.0, 1.0]
    out = _normalization_from_data_cfg({"trace_normalization": {"mode": "zscore", "mean": 1.0, "std": 2.0}})
    assert (out["mean"], out["std"], out["eps"]) == (1.0, 2.0, 1.0e-8)
    assert "feature_range" not in _normalization_from_data_cfg({"trace_normalization": {"mode": "fitted", "steps": ["standard"]}})


def test_check_the_model_finds_the_data_entry_the_model_fits(monkeypatch):
    """Owner 2026-10-23: no data entry named (the panel's Check the model, Visualise): the entries' distinct trace
    lengths are tried in order and the first whose build passes is the model's; the result names it. A model that
    fits none reports the first entry's error."""
    from services.visualization.vscode_monitor import model_check

    cfg = _fake_project(monkeypatch, "pytorch", [])
    cfg["data"] = {"short": {"target_range": [0, 100]}, "long": {"target_range": [0, 400]}, "long2": {"target_range": [0, 400]}}
    long_model = [dict(l) for l in _SMALL_CNN]
    long_model[4] = {"Linear": {"in_features": 800, "out_features": "auto"}}     # 4 x 200: the 400-sample entries
    cfg["model"] = [{"model_id": "m", "architecture": "CNN", "architecture_args": {"layers": long_model}}]
    out = model_check.check("P", "m")
    assert (out["dataset"], out["input"], out["error"], out["fits"]) == ("long", [1, 400], None, ["long"]), "one entry per trace length that fits"
    assert model_check.check("P", "m", "short")["error"] is not None, "a named entry is never replaced"
    cfg["model"][0]["architecture_args"]["layers"][4] = {"Linear": {"in_features": 7, "out_features": "auto"}}
    out = model_check.check("P", "m")
    assert out["dataset"] == "short" and out["at"] == 4, "fits none: the first entry's error"


def test_describe_opens_visualise_on_the_entry_the_model_fits(monkeypatch, tmp_path):
    from services.visualization.vscode_monitor import model_check

    cfg = _fake_project(monkeypatch, "pytorch", [])
    cfg["data"] = {"short": {"target_range": [0, 100]}, "long": {"target_range": [0, 400]}}
    long_model = [dict(l) for l in _SMALL_CNN]
    long_model[4] = {"Linear": {"in_features": 800, "out_features": "auto"}}
    cfg["model"] = [{"model_id": "m", "architecture": "CNN", "architecture_args": {"layers": long_model}}]
    params = tmp_path / "params.yaml"
    params.write_text("model:\n  - model_id: m\n    architecture: CNN\n    architecture_args: {layers: []}\n", encoding="utf-8")
    out = model_check.describe("P", str(params), model_id="m")
    assert (out["dataset"], out["input"], out["error"]) == ("long", [1, 400], None)
    assert model_check.describe("P", str(params))["dataset"] == "short", "Plan a Model: the first entry"


def test_trials_stand_in_a_byte_when_the_file_leaves_the_attack_byte_empty(monkeypatch):
    """layout trials (the panel's and the status page's batch ETA): sca.attack_byte empty follows the data entry
    at the run's start; the sweep would refuse the file's empty list, so the listing stands in one byte (the
    trials need only epochs and passes). Refused, the panel had no trial counts and no batch ETA (2026-10-23)."""
    from utils import config_loader as loader
    from services.visualization.vscode_monitor import layout

    import copy
    cfg = copy.deepcopy(loader.get_project_config("ZvW", validate_data=False))       # a real file: its sca block complete
    cfg["sca"]["attack_byte"] = []
    cfg["seed"] = [42, 53, 64]
    monkeypatch.setattr(loader, "get_project_config", lambda *a, **k: cfg)
    got = layout.trials("ZvW")["trials"]
    epochs = cfg["trainer"]["epochs"][0]                                  # the file's own: it changes per queue
    assert [t["position"] for t in got] == [1, 2, 3] and all(t["epochs"] == epochs and t["passes"] == 1 for t in got)
    assert cfg["sca"]["attack_byte"] == []                                # the file's own list untouched
