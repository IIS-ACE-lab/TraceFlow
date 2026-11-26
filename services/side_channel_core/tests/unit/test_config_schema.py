# services/side_channel_core/tests/unit/test_config_schema.py
"""A key no code reads fails with ConfigError naming the file, the key path and the closest key
(utils/config_schema.py). Owner ruling 2026-10-18."""

from pathlib import Path

import pytest
import yaml

from services.side_channel_core.utils import config_schema as cs
from services.side_channel_core.utils.config_errors import ConfigError

REPO = Path(__file__).resolve().parents[4]
BLUEPRINT = REPO / "services/side_channel_core/config/blueprint"


@pytest.mark.parametrize("name, schema", [("params.yaml", cs.PARAMS), ("graph_config.yaml", cs.GRAPH),
                                          ("model_judge.yaml", cs.JUDGE), ("auto_search.yaml", cs.AUTO_SEARCH)])
def test_the_blueprint_templates_hold_only_keys_the_code_reads(name, schema):
    cs.check_keys(yaml.safe_load((BLUEPRINT / name).read_text()), schema, source=name)


def test_the_blueprint_list_templates_hold_only_keys_the_code_reads():
    """templates.yaml's items, put in their lists of params.yaml, pass its check (data entries: ANY here)."""
    t = yaml.safe_load((BLUEPRINT / "templates.yaml").read_text())
    params = yaml.safe_load((BLUEPRINT / "params.yaml").read_text())
    params["data"], params["model"] = [t["data"]], [t["model"]]
    params["trainer"]["optimizer"] = [t["optimizer"]]
    cs.check_keys(params, cs.PARAMS, source="params.yaml")


def test_master_yaml_and_every_project_entry_hold_only_keys_the_code_reads():
    master = yaml.safe_load((REPO / "services/side_channel_core/config/master.yaml").read_text())
    cs.check_keys(master, cs.MASTER, source="master.yaml")
    for entry in master["project"]:
        cs.check_keys(entry, cs.MASTER_PROJECT, source="master.yaml")


def test_a_misspelled_key_names_its_path_and_the_closest_key():
    with pytest.raises(ConfigError, match=r"params.yaml: trainer.epoch is not a key TraceFlow reads; "
                                          r"did you mean trainer.epochs\? Keys here: .*epochs"):
        cs.check_keys({"trainer": {"epoch": [5]}}, cs.PARAMS, source="params.yaml")
    with pytest.raises(ConfigError, match=r"params.yaml: debug is not a key TraceFlow reads\. Keys here"):
        cs.check_keys({"debug": True}, cs.PARAMS, source="params.yaml")


def test_list_items_and_named_blocks_are_checked_with_their_index():
    with pytest.raises(ConfigError, match=r"graph_config.yaml: jobs\[1\].style.axes.x_lim is not a key"):
        cs.check_keys({"jobs": [{"id": 0}, {"style": {"axes": {"x_lim": [0, 1]}}}]}, cs.GRAPH, source="graph_config.yaml")
    with pytest.raises(ConfigError, match=r"profiles.global.web_ur is not a key"):
        cs.check_keys({"visualization": {"clearml": {"profiles": {"global": {"web_ur": "x"}}}}}, cs.MASTER_PROJECT,
                      source="master.yaml")
    cs.check_keys({"deep_learning": {"pytorch": {"optimizer_args": {"Adam": {"anything": 1}}}}}, cs.PARAMS,
                  source="params.yaml")                                    # names and their arguments are free


def test_blocks_their_own_reader_checks_and_renamed_keys_pass_to_that_reader():
    cs.check_keys({"data": [{"name": "x", "whatever": 1}], "model": [{"model_id": "M", "architecture_args": {"layers": []}}],
                   "perform_attack": {"test": {"attack_repetitions": 3}}, "trainer": {"lr": 1e-3}},
                  cs.PARAMS, source="params.yaml")
    with pytest.raises(ConfigError, match=r"model\[0\].layers is not a key"):
        cs.check_keys({"model": [{"model_id": "M", "layers": []}]}, cs.PARAMS, source="params.yaml")


def test_a_block_where_a_mapping_belongs_is_refused():
    with pytest.raises(ConfigError, match=r"params.yaml: sca must be a mapping, got list"):
        cs.check_keys({"sca": [2]}, cs.PARAMS, source="params.yaml")


def test_the_readers_errors_are_config_errors():
    from services.graph_manager.faceting import FacetValidationError
    from services.graph_manager.graph_job_validator import GraphJobValidationError
    from services.model_filter.model_filter import ModelFilterError
    from services.side_channel_core.src.sca.dataset_types import DatasetTypeError
    from services.side_channel_core.src.sca.test_request import TestRequestError

    for cls in (DatasetTypeError, TestRequestError, ModelFilterError, GraphJobValidationError, FacetValidationError):
        assert issubclass(cls, ConfigError) and issubclass(cls, ValueError)


def _project(tmp_path, monkeypatch, *, params=None, entry=None, graph=None, judge=None):
    from utils import config_loader as cl

    project = tmp_path / "projects" / "P"
    (project / "config").mkdir(parents=True)
    base = {"seed": 1, "sca": {"attack_byte": [2]}, "trainer": {"epochs": [1]},
            "data": [{"name": "d", "type": "ASCADf", "database": "services/side_channel_core/data/ASCAD/ATMEGA_AES_v1/"
                      "ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5", "profiling_range": [0, 10],
                      "attack_range": [0, 10], "test_range": [0, 10], "target_range": [0, 700]}]}
    (project / "config" / "params.yaml").write_text(yaml.safe_dump({**base, **(params or {})}))
    (project / "config" / "graph_config.yaml").write_text(yaml.safe_dump(graph or {"jobs": []}))
    if judge is not None:
        (project / "config" / "model_judge.yaml").write_text(yaml.safe_dump(judge))
    master = {"default_project": "P", "project": [{"name": "P", "root": "P/", "params": "config/params.yaml",
                                                   "extra_configs": {"graph": "config/graph_config.yaml"}, **(entry or {})}]}
    (tmp_path / "master.yaml").write_text(yaml.safe_dump(master))
    monkeypatch.setattr(cl, "SCC_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cl, "PROJECTS_ROOT", tmp_path / "projects")
    return cl, project


def test_a_project_loads_and_each_of_its_files_refuses_a_wrong_key(tmp_path, monkeypatch):
    cl, _ = _project(tmp_path, monkeypatch)
    cl.get_project_config("P", validate_data=False)
    for kwargs, message in (
        ({"params": {"trainer": {"epochs": [1], "epoch": 2}}}, r"params.yaml: trainer.epoch is not a key"),
        ({"entry": {"scirpts": {}}}, r"master.yaml \(project P\): scirpts is not a key TraceFlow reads; did you mean scripts\?"),
        ({"graph": {"jobs": [{"id": 0, "stlye": {}}]}}, r"graph_config.yaml: jobs\[0\].stlye is not a key"),
    ):
        root = tmp_path / str(abs(hash(message)))
        cl, _ = _project(root, monkeypatch, **kwargs)
        with pytest.raises(ConfigError, match=message):
            cl.get_project_config("P", validate_data=False)


def test_model_judge_yaml_refuses_a_wrong_key(tmp_path):
    from services.model_judge.model_judge import load_model_judge_config

    path = tmp_path / "model_judge.yaml"
    path.write_text(yaml.safe_dump({"model_judge": {"benchmark_dir": "benchmark", "ranking": {"primary": {}}}}))
    with pytest.raises(ConfigError, match=r"model_judge.yaml: model_judge.benchmark_dir is not a key TraceFlow reads"):
        load_model_judge_config(path)


def test_the_blueprint_offers_every_key_of_each_schema_and_no_retired_one():
    """make_blueprint.check: the blueprint holds every key a file's schema lists but RETIRED (a key added to a
    schema fails here until `make_blueprint sync` put it in the blueprint), and no retired key."""
    from services.visualization.vscode_monitor import make_blueprint

    report = make_blueprint.check()
    assert set(report) == set(make_blueprint.FILE_SCHEMAS)
    assert {name: r for name, r in report.items() if r["missing"] or r["not offered"]} == {}


def test_make_blueprint_sync_puts_a_new_schema_key_into_the_blueprint_empty(tmp_path, monkeypatch):
    """A key added to PARAMS (here: a block and a key under a list item): sync adds it to params.yaml /
    templates.yaml, empty, its comments and every other line untouched; check is then clean."""
    import shutil

    from services.visualization.vscode_monitor import make_blueprint

    shutil.copytree(BLUEPRINT, tmp_path / "blueprint")
    monkeypatch.setattr(make_blueprint, "BLUEPRINT", tmp_path / "blueprint")
    params = {**cs.PARAMS, "new_block": {"a": cs.ANY, "deeper": {"b": cs.ANY}}, "trainer": {**cs.PARAMS["trainer"], "new_key": cs.ANY},
              "model": cs.Seq({**cs.PARAMS["model"].item, "new_item_key": cs.ANY})}
    monkeypatch.setitem(make_blueprint.FILE_SCHEMAS, "params.yaml", lambda _cs: params)
    before = (tmp_path / "blueprint" / "params.yaml").read_text()
    assert make_blueprint.check()["params.yaml"]["missing"] == ["model.*.new_item_key", "new_block", "new_block.a", "new_block.deeper",
                                                                   "new_block.deeper.b", "trainer.new_key"]
    added = make_blueprint.sync()
    assert added == {"params.yaml": ["new_block", "new_block.a", "new_block.deeper.b", "trainer.new_key"],
                     "templates.yaml": ["model.new_item_key"]}
    after = (tmp_path / "blueprint" / "params.yaml").read_text()
    assert yaml.safe_load(after)["new_block"] == {"a": None, "deeper": {"b": None}}
    assert yaml.safe_load(after)["trainer"]["new_key"] is None
    assert "new_key:" in after and "new_key: null" not in after and "b: null" not in after, "written empty, not null"
    assert yaml.safe_load((tmp_path / "blueprint" / "templates.yaml").read_text())["model"]["new_item_key"] is None
    kept = [line for line in before.splitlines() if line.strip()]
    assert all(line in after for line in kept), "every other line as it was"
    assert make_blueprint.check()["params.yaml"] == {"missing": [], "not offered": []}
    # a retired key written into the blueprint is reported
    monkeypatch.setitem(make_blueprint.FILE_SCHEMAS, "params.yaml", lambda _cs: _cs.PARAMS)
    (tmp_path / "blueprint" / "params.yaml").write_text(after.replace("seed:", "seed:\nmetrics: []", 1))
    assert make_blueprint.check()["params.yaml"]["not offered"] == ["metrics"]


def test_the_graph_schema_names_every_key_graph_manager_reads_from_a_job_or_facet():
    """Owner 2026-10-23: the graph_config.yaml schema recognises every level graph_manager reads (jobs, facets,
    their style, legend, export, text rules). Mined from the readers' source: each `<config>.get("key")` on a
    job, facet, panel, legend, export, style, line, axes or rule mapping names a key the schema offers."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    readers = ("graph_job_validator.py", "faceting.py", "plot_style.py")
    holders = {"job", "raw", "facet", "panel", "legend", "export", "style", "line", "axes", "rule", "entry", "job_cfg",
               "facet_cfg", "legend_raw", "export_raw", "line_raw", "axes_raw", "arrangement", "text_rule", "grid", "band",
               "font", "figure", "style_raw", "relative"}
    # read but not a file key: `where` is inside `relative` (ANY); `underline` is refused by the validator itself (Matplotlib
    # has none), so a file carrying it is refused either way; legend_text is a text_rules target value
    not_keys = {"where", "underline", "legend_text"}

    def leaves(schema):
        if isinstance(schema, cs.Seq):
            return leaves(schema.item)
        if not isinstance(schema, dict):
            return set()
        out = set()
        for key, sub in schema.items():
            out.add(key)
            out |= leaves(sub)
        return out

    offered = leaves(cs.GRAPH)
    read = {}
    for name in readers:
        text = (root / "services" / "graph_manager" / name).read_text(encoding="utf-8")
        for m in re.finditer(r'(\w+)\.get\(\s*"([a-z_]+)"', text):
            if m.group(1) in holders:
                read.setdefault(m.group(2), set()).add(name)
    missing = {k: sorted(v) for k, v in read.items() if k not in offered and k not in not_keys}
    assert missing == {}, f"graph_manager reads keys the graph_config.yaml schema does not offer: {missing}"
    cs.check_keys({"jobs": [{"id": 1, "style": {"line": {"styles": {"run_a": "--"}}}}]}, cs.GRAPH, source="graph_config.yaml")
