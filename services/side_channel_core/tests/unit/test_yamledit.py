# services/side_channel_core/tests/unit/test_yamledit.py
"""services/visualization/vscode_monitor/yamledit.py: the panel's span-preserving config editor."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from services.visualization.vscode_monitor import yamledit as ye
from services.visualization.vscode_monitor import yamlspan

REPO = Path(__file__).resolve().parents[4]
TEXT = """\
# header
seed: 42   # the seed
name: run
blank:
trainer:
  epochs: [20, 50]        # sweep
  batch_size:
    - 100   # small
    - 200
  lr: 1.0e-5
  tag: "v1"
  optimizer:
    - type: RMSprop       # first
      rho: 0.9
    - type: Adam
      beta: 0.5
# trailing comment
"""


def _edit(op, path, value=None, text=TEXT):
    return ye.edit_text(text, op, path, value)


def _changed_lines(old, new):
    return [line for line in new.splitlines() if line not in old.splitlines()]


def test_tree_has_entries_only_with_paths():
    t = ye.tree(TEXT)["tree"]
    keys = [e["key"] for e in t["entries"]]
    assert keys == ["seed", "name", "blank", "trainer"]
    trainer = t["entries"][3]["node"]
    epochs = trainer["entries"][0]["node"]
    assert (epochs["k"], epochs["flow"], epochs["scalars"]) == ("list", True, True)
    assert epochs["items"][1] == {"k": "scalar", "path": ["trainer", "epochs", 1], "text": "50", "type": "int",
                                  "quoted": False, "multiline": False}
    assert t["entries"][2]["node"]["text"] == "" and t["entries"][2]["node"]["type"] == "null"
    assert "header" not in json.dumps(t) and "sweep" not in json.dumps(t)


def test_set_changes_only_that_value_and_keeps_its_comment():
    new = _edit("set", ["seed"], "7")
    assert new == TEXT.replace("seed: 42   # the seed", "seed: 7   # the seed")


def test_set_fills_an_empty_value():
    assert "blank: x\n" in _edit("set", ["blank"], "x")


def test_values_with_hash_or_colon_are_quoted():
    new = _edit("set", ["name"], "a # b")
    assert yaml.safe_load(new)["name"] == "a # b" and 'name: "a # b"' in new
    new = _edit("set", ["name"], "k: v")
    assert yaml.safe_load(new)["name"] == "k: v"


def test_number_types_follow_yaml_and_quoted_stays_quoted():
    assert yaml.safe_load(_edit("set", ["trainer", "lr"], "1e-4"))["trainer"]["lr"] == 1e-4
    assert yaml.safe_load(_edit("set", ["seed"], "true"))["seed"] is True
    new = _edit("set", ["trainer", "tag"], "123")
    assert yaml.safe_load(new)["trainer"]["tag"] == "123" and 'tag: "123"' in new
    assert yaml.safe_load(_edit("set", ["seed"], ""))["seed"] is None


def test_add_and_remove_in_flow_and_block_lists():
    new = _edit("add", ["trainer", "epochs"], "80")
    assert "epochs: [20, 50, 80]        # sweep" in new
    new = _edit("remove", ["trainer", "epochs", 0])
    assert "epochs: [50]        # sweep" in new
    new = _edit("add", ["trainer", "batch_size"], "300")
    assert "    - 200\n    - 300\n  lr:" in new
    new = _edit("remove", ["trainer", "batch_size", 0])
    assert "small" not in new and yaml.safe_load(new)["trainer"]["batch_size"] == [200]
    once = _edit("remove", ["trainer", "batch_size", 0])
    assert yaml.safe_load(_edit("remove", ["trainer", "batch_size", 0], text=once))["trainer"]["batch_size"] == []
    flow_one = _edit("remove", ["trainer", "epochs", 0], text=_edit("remove", ["trainer", "epochs", 1]))
    assert "epochs: []" in flow_one


def test_duplicate_and_delete_blocks():
    new = _edit("duplicate", ["trainer", "optimizer", 0])
    opts = yaml.safe_load(new)["trainer"]["optimizer"]
    assert [o["type"] for o in opts] == ["RMSprop", "RMSprop", "Adam"]
    assert new.count("# first") == 2 and "# trailing comment" in new
    new = _edit("delete", ["trainer", "optimizer", 1])
    assert [o["type"] for o in yaml.safe_load(new)["trainer"]["optimizer"]] == ["RMSprop"]
    assert "# trailing comment" in new
    new = _edit("delete", ["name"])
    assert "name" not in yaml.safe_load(new) and new.count("\n") == TEXT.count("\n") - 1


def test_duplicate_a_mapping_entry_under_a_new_key():
    queue = "# queues\nfig2:\n  - make a   # one\nfig3:\n  - make b\n"
    new = ye.edit_text(queue, "duplicate", ["fig2"], "fig2b")
    assert yaml.safe_load(new) == {"fig2": ["make a"], "fig2b": ["make a"], "fig3": ["make b"]}
    with pytest.raises(ye.EditError, match="exists"):
        ye.edit_text(queue, "duplicate", ["fig2"], "fig3")


def test_refusals():
    with pytest.raises(ye.EditError, match="not a list of values"):
        _edit("add", ["trainer", "optimizer"], "x")
    with pytest.raises(ye.EditError, match="no entry"):
        _edit("set", ["trainer", "nope"], "1")
    with pytest.raises(ye.EditError, match="at least one entry"):
        ye.edit_text("only: 1\n", "delete", ["only"])
    # a plain `a, b` would split the flow mapping: the read-back check refuses or it is quoted
    flow = "m: {x: 1, y: 2}\n"
    assert yaml.safe_load(ye.edit_text(flow, "set", ["m", "x"], "a, b"))["m"] == {"x": "a, b", "y": 2}


def test_stale_file_is_not_written_and_blueprint_is_refused(tmp_path):
    f = tmp_path / "params.yaml"
    f.write_text(TEXT)
    assert ye.edit_file(f, "0" * 64, "set", ["seed"], "1")["stale"] is True
    assert f.read_text() == TEXT
    out = ye.edit_file(f, ye.sha256(TEXT), "set", ["seed"], "1")
    assert out["ok"] and yaml.safe_load(f.read_text())["seed"] == 1 and out["sha256"] == ye.sha256(f.read_text())
    with pytest.raises(ye.EditError, match="blueprint"):
        ye.edit_file(REPO / "services/side_channel_core/config/blueprint/params.yaml", "x", "set", ["seed"], "1")


def test_cli_prints_json(tmp_path, capsys):
    f = tmp_path / "c.yaml"
    f.write_text(TEXT)
    ye.main(["tree", str(f)])
    sha = json.loads(capsys.readouterr().out)["sha256"]
    ye.main(["edit", str(f), "--sha", sha, "--op", "set", "--path", '["trainer", "epochs", 0]', "--value", "30"])
    assert json.loads(capsys.readouterr().out)["ok"] is True
    ye.main(["edit", str(f), "--sha", sha, "--op", "set", "--path", '["seed"]', "--value", "3"])
    assert json.loads(capsys.readouterr().out)["stale"] is True


def _tree_scalars(node):
    if node is None:
        return
    if node["k"] == "scalar":
        yield node
    for child in node.get("entries", []):
        yield from _tree_scalars(child["node"])
    for child in node.get("items", []):
        yield from _tree_scalars(child)


def _scalar_nodes(node, path=()):
    if isinstance(node, yaml.ScalarNode):
        yield list(path), node
    elif isinstance(node, yaml.MappingNode):
        for k, v in node.value:
            yield from _scalar_nodes(v, path + (str(k.value),))
    elif isinstance(node, yaml.SequenceNode):
        for i, v in enumerate(node.value):
            yield from _scalar_nodes(v, path + (i,))


def _value_at(data, item_path):
    for item in item_path:
        data = data[item] if isinstance(data, list) else data[next(k for k in data if str(k) == item)]
    return data


@pytest.mark.parametrize("name", ["params.yaml", "experiment_queue.yaml", "graph_config.yaml", "model_filter.yaml",
                                  "model_judge.yaml", "test_config.yaml", "dev.yaml"])
def test_every_value_of_the_real_configs_round_trips(name):
    """Each shown value, typed back, means the same value; a spread of them also as whole-file edits."""
    path = REPO / "projects" / "DLSCA_ASCAD" / "config" / name
    if not path.exists():
        pytest.skip(f"{path} absent")
    text = path.read_text()
    data = yamlspan.load(text)
    shown = {tuple(n["path"]): n["text"] for n in _tree_scalars(ye.tree(text)["tree"])}
    nodes = list(_scalar_nodes(yamlspan.compose(text)))
    for item_path, node in nodes:
        _rendered, value = ye.render(shown[tuple(item_path)], node)
        assert ye._same(value, _value_at(data, item_path)), item_path
    for item_path, _node in nodes[:: max(1, len(nodes) // 10)]:
        new = ye.edit_text(text, "set", item_path, shown[tuple(item_path)])     # the shown text, written back
        assert yamlspan.load(new) == data, item_path


def test_crlf_file_stays_crlf_and_values_may_start_with_a_dash(tmp_path, capsys):
    f = tmp_path / "c.yaml"
    f.write_bytes(b"a: 1\r\nlst:\r\n  - x\r\nb: 2\r\n")
    ye.main(["tree", str(f)])
    sha = json.loads(capsys.readouterr().out)["sha256"]
    ye.main(["edit", str(f), "--sha", sha, "--op", "set", "--path", '["a"]', "--value=-1e-5"])
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    ye.main(["edit", str(f), "--sha", out["sha256"], "--op", "add", "--path", '["lst"]', "--value=-y"])
    assert json.loads(capsys.readouterr().out)["ok"] is True
    raw = f.read_bytes()
    assert raw == b"a: -1.0e-05\r\nlst:\r\n  - x\r\n  - -y\r\nb: 2\r\n"
    assert yaml.safe_load(raw) == {"a": -1e-05, "lst": ["x", "-y"], "b": 2}


GRAPH = """\
jobs:
  - id: 3
    models:
      # names from done_model_names.yaml
    legend:
      handle: ["line", 1.5]  # [shape, pt]
      labels:
        # <model_name>: "label"

    style:
      font: {size: 9}
"""


def test_add_to_an_empty_key_starts_a_block_list_and_put_fills_labels():
    new = ye.edit_text(GRAPH, "add", ["jobs", 0, "models"], "mesh_chicken")
    new = ye.edit_text(new, "add", ["jobs", 0, "models"], "tulip_fish")
    assert yamlspan.load(new)["jobs"][0]["models"] == ["mesh_chicken", "tulip_fish"]
    assert "# names from done_model_names.yaml" in new                       # comments stay
    new = ye.edit_text(new, "put", ["jobs", 0, "legend", "labels"], "batch size=50", key="mesh_chicken")
    new = ye.edit_text(new, "put", ["jobs", 0, "legend", "labels"], "batch size=100", key="tulip_fish")
    new = ye.edit_text(new, "put", ["jobs", 0, "legend", "labels"], "bs 50", key="mesh_chicken")   # existing: set
    job = yamlspan.load(new)["jobs"][0]
    assert job["legend"]["labels"] == {"mesh_chicken": "bs 50", "tulip_fish": "batch size=100"}
    assert job["style"] == {"font": {"size": 9}} and job["legend"]["handle"] == ["line", 1.5]
    assert ye.edit_text("m: {}\n", "put", ["m"], "x", key="ep0001") == "m: {ep0001: x}\n"


def test_put_refuses_a_non_mapping():
    with pytest.raises(ye.EditError, match="not a mapping"):
        ye.edit_text(GRAPH, "put", ["jobs", 0, "legend", "handle"], "x", key="k")


def test_set_writes_a_whole_flow_range():
    text = "data:\n  - name: a\n    attack_range: []   # lower, upper\n    test_range:\n    target_range: [1, 2]\n"
    new = ye.edit_text(text, "set", ["data", 0, "attack_range"], "[50000, 60000]")
    new = ye.edit_text(new, "set", ["data", 0, "test_range"], "[0, 10000]")
    assert "attack_range: [50000, 60000]   # lower, upper" in new
    assert yamlspan.load(new)["data"][0] == {"name": "a", "attack_range": [50000, 60000], "test_range": [0, 10000],
                                            "target_range": [1, 2]}
    with pytest.raises(ye.EditError, match="only by a list"):
        ye.edit_text(text, "set", ["data", 0, "target_range"], "5")


def test_what_the_panel_sends_as_json_reads_back_as_the_data_it_meant():
    text = "jobs:\n  - id: 1\n    name: a\n    style:\n      axes:\n        x_limits: [0, 950]\n      text_rules:\n        - target: title\n          text: Rank\n"
    load = lambda t: yaml.safe_load(t)["jobs"][0]
    # a list with 1e-7 (JSON writes no decimal point; YAML 1.1 would read text): numbers
    assert load(ye.edit_text(text, "set", ["jobs", 0, "style", "axes", "x_limits"], "[1e-7,1]"))["style"]["axes"]["x_limits"] == [1e-7, 1]
    assert load(ye.edit_text(text, "put", ["jobs", 0, "style", "axes"], "[0,1e+21]", key="y_limits"))["style"]["axes"]["y_limits"] == [0, 1e21]
    # a drafted text is a JSON string: that text, whatever it looks like
    assert load(ye.edit_text(text, "put", ["jobs", 0], '"2020"', key="note"))["note"] == "2020"
    t = ye.edit_text(text, "put", ["jobs", 0], "[]", key="formats")
    for f in ('"png"', '"pdf"'):
        t = ye.edit_text(t, "add", ["jobs", 0, "formats"], f)
    assert load(t)["formats"] == ["png", "pdf"]
    # a key holding text keeps text when typed a yes / no / null word; a number still becomes one
    for typed in ("yes", "off", "null", "True"):
        assert load(ye.edit_text(text, "set", ["jobs", 0, "style", "text_rules", 0, "text"], typed))["style"]["text_rules"][0]["text"] == typed
    assert load(ye.edit_text("jobs:\n  - id: 1\n    out: auto\n", "set", ["jobs", 0, "out"], "256"))["out"] == 256


# ── arrange: the planner's edits of a visualised model's layers (insert, move, change), comments kept ──
TEXT_LAYERS = """\
model:
  - model_id: "M"   # the model
    architecture_args:
      layers:
        # stem: the first convolution
        - Conv1d: {in_channels: 1, out_channels: 64, kernel_size: 3, padding: 1}
        - Pooling: {dim: 1, type: "max", kernel_size: 2, stride: 2}

        # head
        - Flatten: {}
        - Linear: {in_features: 22400, out_features: "auto"}   # the scores
other: 1
"""
LAYERS = ["model", 0, "architecture_args", "layers"]


def _arrange(entries, text=TEXT_LAYERS):
    return ye.edit_text(text, "arrange", LAYERS, json.dumps(entries))


def _names(text):
    return [next(iter(layer)) for layer in yaml.safe_load(text)["model"][0]["architecture_args"]["layers"]]


def test_arrange_with_the_same_order_changes_nothing():
    assert _arrange([{"from": 0}, {"from": 1}, {"from": 2}, {"from": 3}]) == TEXT_LAYERS


def test_arrange_moves_a_layer_with_its_comment_and_keeps_blank_lines():
    new = _arrange([{"from": 1}, {"from": 0}, {"from": 2}, {"from": 3}])
    assert _names(new) == ["Pooling", "Conv1d", "Flatten", "Linear"]
    assert new.index("# stem: the first convolution") < new.index("- Conv1d") and new.index("- Pooling") < new.index("# stem")
    assert "\n\n        # head\n        - Flatten" in new and "# the scores" in new and new.endswith("other: 1\n")


def test_arrange_inserts_a_layer_in_the_lists_own_style():
    new = _arrange([{"from": 0}, {"value": {"Dropout": {"p": 0.2}}}, {"from": 1}, {"from": 2}, {"from": 3}])
    assert _names(new) == ["Conv1d", "Dropout", "Pooling", "Flatten", "Linear"]
    assert "        - Dropout: {p: 0.2}\n" in new


def test_arrange_changes_a_layers_arguments_and_keeps_its_comment():
    new = _arrange([{"from": 0}, {"from": 1}, {"from": 2},
                    {"from": 3, "value": {"Linear": {"in_features": 11200, "out_features": "auto"}}}])
    layer = yaml.safe_load(new)["model"][0]["architecture_args"]["layers"][3]
    assert layer == {"Linear": {"in_features": 11200, "out_features": "auto"}} and "# the scores" in new
    assert len(_changed_lines(TEXT_LAYERS, new)) == 1


def test_arrange_drops_a_layer_but_keeps_the_comments_between_layers():
    new = _arrange([{"from": 0}, {"from": 2}, {"from": 3}])
    assert _names(new) == ["Conv1d", "Flatten", "Linear"] and "# head" in new and "Pooling" not in new


def test_arrange_refuses_a_from_index_out_of_range():
    with pytest.raises(ye.EditError, match="no layer 7"):
        _arrange([{"from": 7}])


def test_arrange_keeps_a_sweep_list_argument_verbatim():
    text = TEXT_LAYERS.replace("out_channels: 64", "out_channels: [64, 128]")
    assert _arrange([{"from": 0}, {"from": 1}, {"from": 2}, {"from": 3}], text) == text
    moved = _arrange([{"from": 1}, {"from": 0}, {"from": 2}, {"from": 3}], text)
    assert "- Conv1d: {in_channels: 1, out_channels: [64, 128], kernel_size: 3, padding: 1}" in moved


def test_preview_shows_the_lines_add_would_write_and_writes_nothing(tmp_path, capsys):
    path = tmp_path / "params.yaml"
    path.write_text(TEXT_LAYERS)
    model = {"model_id": "N", "architecture": "MLP", "architecture_args": {"strict_graph_args": True, "layers": [{"Flatten": {}}]}}
    assert ye.main(["preview", str(path), "--op", "add", "--path", '["model"]', f"--value={json.dumps(model)}"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert any("model_id: N" in line for line in out["added"]) and out["removed"] == []
    assert yaml.safe_load(out["text"])["model"][1]["model_id"] == "N" and path.read_text() == TEXT_LAYERS


@pytest.mark.parametrize("project", ["DLSCA_ASCAD", "ZvW"])
def test_arrange_in_the_same_order_leaves_every_real_model_byte_identical(project):
    text = (REPO / "projects" / project / "config" / "params.yaml").read_text(encoding="utf-8")
    for i, model in enumerate(yaml.safe_load(text)["model"]):
        layers = (model.get("architecture_args") or {}).get("layers")
        if layers:
            same = [{"from": k} for k in range(len(layers))]
            assert ye.edit_text(text, "arrange", ["model", i, "architecture_args", "layers"], json.dumps(same)) == text, model["model_id"]


# ── final review fixes: gaps follow their layer, comments survive changed arguments, an empty list fills ──
GROUPED = """\
model:
  - model_id: "G"
    architecture_args:
      layers:
        - Conv1d: {in_channels: 1, out_channels: 8, kernel_size: 3}

        - Pooling: {dim: 1, type: "max", kernel_size: 2, stride: 2}

        # head
        - Flatten: {}
        - Linear: {in_features: 2800, out_features: "auto"}   # scores, see #12
"""
G = ["model", 0, "architecture_args", "layers"]


def test_arrange_insert_keeps_each_blank_line_and_comment_before_its_own_layer():
    new = ye.edit_text(GROUPED, "arrange", G, json.dumps([{"from": 0}, {"value": {"Dropout": {"p": 0.1}}}, {"from": 1}, {"from": 2}, {"from": 3}]))
    assert "        - Dropout: {p: 0.1}\n\n        - Pooling" in new
    assert "\n\n        # head\n        - Flatten" in new


def test_arrange_changed_flow_layer_keeps_its_whole_trailing_comment():
    new = ye.edit_text(GROUPED, "arrange", G, json.dumps([{"from": 0}, {"from": 1}, {"from": 2},
                                                          {"from": 3, "value": {"Linear": {"in_features": 1400, "out_features": "auto"}}}]))
    assert "   # scores, see #12" in new and "in_features: 1400" in new


def test_arrange_changed_block_layer_keeps_the_comments_inside_it():
    text = ("model:\n  - model_id: B\n    architecture_args:\n      layers:\n        - Conv1d:\n"
            "            in_channels: 1   # one trace channel\n            out_channels: 64   # Table 1\n            kernel_size: 3\n"
            "        - Flatten: {}\n")
    new = ye.edit_text(text, "arrange", ["model", 0, "architecture_args", "layers"],
                       json.dumps([{"from": 0, "value": {"Conv1d": {"in_channels": 1, "out_channels": 128, "kernel_size": 3}}}, {"from": 1}]))
    assert "# one trace channel" in new and "out_channels: 128   # Table 1" in new


def test_arrange_fills_an_empty_layer_list():
    text = "model:\n  - model_id: E\n    architecture_args:\n      layers: []\nother: 1\n"
    new = ye.edit_text(text, "arrange", ["model", 0, "architecture_args", "layers"], json.dumps([{"value": {"Flatten": {}}}, {"value": {"Linear": {"in_features": 700, "out_features": "auto"}}}]))
    assert yaml.safe_load(new)["model"][0]["architecture_args"]["layers"] == [{"Flatten": {}}, {"Linear": {"in_features": 700, "out_features": "auto"}}]
    assert new.endswith("other: 1\n")


def test_value_at_prints_the_data_at_a_path(tmp_path, capsys):
    path = tmp_path / "params.yaml"
    path.write_text(GROUPED)
    assert ye.main(["value", str(path), "--path", json.dumps(G)]) == 0
    assert json.loads(capsys.readouterr().out)["value"][1] == {"Pooling": {"dim": 1, "type": "max", "kernel_size": 2, "stride": 2}}


def test_comments_gives_each_keys_comment_by_its_dotted_path():
    """yamledit.comments (the blueprint's key explanations): a key's comment on its line and the comment lines right above
    it at its indentation, by dotted path (a list item: *); a section rule line is no comment; a key without
    one is absent (the panel then shows the blueprint's)."""
    from services.visualization.vscode_monitor import yamledit

    text = ("# ── SETTINGS ──────\n# the seed of everything\n# (two lines)\nseed: 42   # inline wins\n"
            "trainer:\n  # how long\n  epochs: [1]\n  batch_size: [2]\ndata:\n  - name: a   # the entry's name\n    type: X\n")
    hints = yamledit.comments(text)
    assert hints["seed"] == "the seed of everything (two lines) inline wins"
    assert hints["trainer.epochs"] == "how long" and "trainer.batch_size" not in hints
    assert hints["data.*.name"] == "the entry's name" and "data.*.type" not in hints
    assert "trainer" not in hints, "a section rule line is no comment"
