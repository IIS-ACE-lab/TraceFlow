# services/side_channel_core/tests/unit/test_model_diagram.py
"""The planner's Download (services/visualization/vscode_monitor/model_diagram.py): the diagram drawn whole, saved
as graph_manager saves figures into paths.model_preview, copied into model_out of the runs with these layers."""
import os
import stat

import pytest

from services.visualization.vscode_monitor import model_diagram as MD

LAYERS = [{"name": "Flatten", "group": "shape", "args": {}, "out": [700]},
          {"name": "Linear", "group": "dense", "args": {"in_features": 700, "out_features": 200, "activation": "relu"}, "out": [200]},
          {"name": "Linear", "group": "dense", "args": {"in_features": 200, "out_features": "auto"}, "out": [256]}]


def request(**extra):
    return {"title": "ASCAD_MLPbase", "subtitle": "DLSCA_ASCAD · MLP · keras", "facts": [["Input", "1 × 700"], ["Weights", ""]],
            "layers": LAYERS, "max": 700, "model_id": "ASCAD_MLPbase", "env": "dev",
            "export": {"formats": ["png", "pdf", "svg"], "png": {"dpi": 100}, "margin": 0.05}, **extra}


def test_the_scale_s_labels_never_touch_as_in_the_planner():
    """planner_core.scaleTicks gives the same ticks (test/planner.test.js)."""
    assert MD.scale_ticks(10752, 170) == [1, 10, 100, 10752]
    assert MD.scale_ticks(700, 170) == [1, 10, 100, 700]
    assert MD.scale_ticks(100000, 170) == [1, 10, 100, 1000, 100000]


def test_every_argument_is_written_the_activation_last():
    assert MD.args_line({"in_channels": 1, "kernel_size": 11, "activation": "relu", "pool": [2, 2]}) == \
        "in_channels 1 · kernel_size 11 · pool [2,2] · relu"


def test_auto_is_alike_what_a_run_made_of_it():
    run = [{"Linear": {"in_features": 200, "out_features": 256}}]
    assert MD._same(run, [{"Linear": {"in_features": 200, "out_features": "auto"}}])
    assert not MD._same(run, [{"Linear": {"in_features": 100, "out_features": "auto"}}])
    assert not MD._same(run, [{"Linear": {"in_features": 200}}])


def test_it_is_saved_as_graph_manager_saves_and_copied_into_the_runs_with_these_layers(tmp_path, monkeypatch):
    preview = tmp_path / "artifacts" / "dev" / "model_preview"
    same, locked = tmp_path / "runs" / "r1" / "model", tmp_path / "runs" / "r2" / "model"
    locked.mkdir(parents=True)
    (same / "r1_diagram.png").parent.mkdir(parents=True)
    (same / "r1_diagram.png").write_bytes(b"old")
    monkeypatch.setattr(MD, "_preview_dir", lambda project, env: preview)
    monkeypatch.setattr(MD, "_runs_drawn", lambda project, model_id, layers: ([("r1", same), ("r2", locked)], ["r3"]))
    os.chmod(locked, stat.S_IRUSR | stat.S_IXUSR)                       # as a folder the container wrote as root
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("running as root: every folder is writable")
        out = MD.download("DLSCA_ASCAD", request())
    finally:
        os.chmod(locked, stat.S_IRWXU)
    assert [os.path.basename(p) for p in out["written"]] == ["ASCAD_MLPbase.png", "ASCAD_MLPbase.pdf", "ASCAD_MLPbase.svg"]
    assert (preview / "ASCAD_MLPbase.png").read_bytes()[:4] == b"\x89PNG"
    assert (preview / "ASCAD_MLPbase.pdf").read_bytes()[:4] == b"%PDF"
    svg = (preview / "ASCAD_MLPbase.svg").read_text()
    assert "<script" not in svg
    assert sorted(os.path.basename(p) for p in out["copied"]) == ["r1_diagram.pdf", "r1_diagram.png", "r1_diagram.svg"]
    assert (same / "r1_diagram.png").read_bytes() == (preview / "ASCAD_MLPbase.png").read_bytes(), "an earlier diagram is replaced"
    assert out["skipped"] == ["r3"] and out["denied"] == ["r2"]


def test_only_the_formats_ticked_and_graph_manager_refuses_what_it_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(MD, "_preview_dir", lambda project, env: tmp_path)
    out = MD.download("DLSCA_ASCAD", request(model_id="", export={"formats": ["svg"]}))
    assert [os.path.basename(p) for p in out["written"]] == ["ASCAD_MLPbase.svg"] and out["copied"] == []
    with pytest.raises(Exception, match="jpg|jpeg|JPG"):
        MD.download("DLSCA_ASCAD", request(model_id="", export={"formats": ["jpg"]}))


def test_the_preview_folder_is_params_yamls_paths_model_preview():
    path = MD._preview_dir("DLSCA_ASCAD", "staging")
    assert path.parts[-3:] == ("artifacts", "staging", "model_preview")


TRACE = {"values": [0.1, 0.5, -0.2, 0.3], "band": [[0.0, 0.4, -0.3, 0.2], [0.2, 0.6, -0.1, 0.4]], "lo": -0.3, "hi": 0.6}


def _texts(fig):
    return [t.get_text() for t in fig.axes[0].texts]


@pytest.mark.parametrize("extra", [
    {"style": "full"}, {"style": "cards", "arrows": True},
    {"style": "sparkline", "sparkline": {"orientation": "portrait", "legend": "top-right"}},
    {"style": "sparkline", "arrows": True, "sparkline": {"orientation": "landscape", "legend": "bottom-right", "ticks": False}},
    {"style": "sparkline", "sparkline": {"arguments": False, "legend": "none"}},
])
def test_every_style_draws_every_layer(extra):
    import matplotlib.pyplot as plt

    fig = MD.figure(request(**extra))
    try:
        texts = _texts(fig)
        assert "ASCAD_MLPbase" in texts
        for i in range(len(LAYERS)):
            assert any(t.startswith(f"#{i}") for t in texts), i
        spark = extra["style"] == "sparkline"
        assert any("out_features 200" in t for t in texts) == (not spark or extra["sparkline"].get("arguments", True))
        assert ("Linear ×2" in texts) == (spark and extra["sparkline"].get("legend") != "none"), "the layer legend: each type and how many"
    finally:
        plt.close(fig)


def test_a_text_rule_restyles_its_text_and_replaces_the_title_s_words():
    import matplotlib.pyplot as plt

    fig = MD.figure(request(text_rules={"title": {"text": "MLP base", "bold": False, "font": {"family": "serif", "size": 16}},
                                        "arguments": {"italic": True, "color": "C3"}}))
    try:
        title = next(t for t in fig.axes[0].texts if t.get_text() == "MLP base")
        assert title.get_fontsize() == 16 and title.get_fontweight() in ("normal", 400)
        args = next(t for t in fig.axes[0].texts if "out_features 200" in t.get_text())
        assert args.get_fontstyle() == "italic" and args.get_color() == "C3"
    finally:
        plt.close(fig)


@pytest.mark.parametrize("extra, why", [
    ({"style": "poster"}, "diagram style"), ({"style": "sparkline", "sparkline": {"orientation": "diagonal"}}, "orientation"),
    ({"style": "sparkline", "sparkline": {"legend": "middle"}}, "legend"), ({"text_rules": {"x_label": {"bold": True}}}, "target"),
])
def test_a_choice_it_does_not_know_is_refused(extra, why):
    with pytest.raises(ValueError, match=why):
        MD.figure(request(**extra))


def test_trace_lines_under_the_cards_with_their_caption():
    import matplotlib.pyplot as plt

    layers = [{**layer, "trace": TRACE} for layer in LAYERS]
    fig = MD.figure(request(layers=layers, traces=True, trace={"caption": "untrained weights · trace 0 of ascad", "input": TRACE}))
    try:
        texts = _texts(fig)
        assert "untrained weights · trace 0 of ascad" in texts
        assert texts.count("-0.3 … 0.6") == len(LAYERS) + 1, "each layer's range, and the input's"
        assert len(fig.axes[0].collections) == len(LAYERS) + 1, "a band under each mean"
    finally:
        plt.close(fig)


def test_the_file_names_the_model_the_run_and_the_trace_and_trace_lines_of_a_run_go_to_that_run_only(tmp_path, monkeypatch):
    preview = tmp_path / "preview"
    r1, r2 = tmp_path / "r1" / "model", tmp_path / "r2" / "model"
    monkeypatch.setattr(MD, "_preview_dir", lambda project, env: preview)
    monkeypatch.setattr(MD, "_runs_drawn", lambda project, model_id, layers: ([("r1", r1), ("r2", r2)], []))
    layers = [{**layer, "trace": TRACE} for layer in LAYERS]
    out = MD.download("DLSCA_ASCAD", request(layers=layers, traces=True, run="r1", tag="trace_ascad_t0-9",
                                             name="ASCAD_MLPbase__r1__trace_ascad_t0-9", export={"formats": ["png"]}))
    assert [os.path.basename(p) for p in out["written"]] == ["ASCAD_MLPbase__r1__trace_ascad_t0-9.png"]
    assert [os.path.relpath(p, tmp_path) for p in out["copied"]] == [os.path.join("r1", "model", "r1_diagram__trace_ascad_t0-9.png")]
    assert out["skipped"] == ["r2"], "r2's folder does not get lines drawn through r1's weights"
    out = MD.download("DLSCA_ASCAD", request(run="r1", name="ASCAD_MLPbase__r1", export={"formats": ["png"]}))
    assert sorted(os.path.relpath(p, tmp_path) for p in out["copied"]) == [os.path.join("r1", "model", "r1_diagram.png"),
                                                                            os.path.join("r2", "model", "r2_diagram.png")]


@pytest.mark.parametrize("theme", __import__("services.visualization.utils.fig_theme", fromlist=["names"]).names())
def test_every_theme_of_fig_theme_draws_on_its_own_page(theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from services.visualization.utils import fig_theme

    fig = MD.figure(request(theme=theme, style="sparkline"))
    try:
        assert to_hex(fig.get_facecolor()) == fig_theme.get(theme)["page"].lower()
        assert MD.C["dense"] == fig_theme.get(theme)["support"], "a layer group takes its palette slot"
    finally:
        plt.close(fig)


def test_traceflow_is_the_default_and_keeps_the_planner_s_colours():
    import matplotlib.pyplot as plt

    plt.close(MD.figure(request()))
    assert (MD.C["convolution"], MD.C["pooling"], MD.C["dense"], MD.C["page"]) == ("#1A85FF", "#D18616", "#652D90", "#FFFFFF")
    with pytest.raises(ValueError, match="diagram theme 'vaporwave'"):
        MD.figure(request(theme="vaporwave"))


def test_the_paper_figures_take_their_themes_from_services_visualization_utils():
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    system = root / "paper" / "draft_4" / "figures" / "system"
    assert not (system / "fig_theme.py").exists()
    for script in sorted(system.glob("*.py")):
        text = script.read_text(encoding="utf-8")
        if "fig_theme" in text:
            assert "from services.visualization.utils import fig_theme" in text, script.name


def test_every_layer_s_arguments_wrap_at_the_one_measure_the_longest_sets():
    ink = MD.Ink("arguments", {})
    texts = ["in_channels 128 · out_channels 256 · kernel_size 3 · stride 1 · padding 1", "dim 1 · type max · kernel_size 2 · stride 2",
             "num_features 64 · relu"]
    room = 220.0
    even = ink.even_width(texts, room)
    longest = ink.wrap(texts[0], even)
    assert len(longest) == len(ink.wrap(texts[0], room)), "no more lines than the room needs"
    assert len(ink.wrap(texts[0], even - 1)) > len(longest), "the narrowest such width: the longest splits evenly"
    assert even <= room
    widths = [ink.width(s) for s in longest]
    assert max(widths) - min(widths) < 0.5 * max(widths), "its lines are about as long as each other"
    for t in texts:
        assert all(ink.width(s) <= even + 0.01 for s in ink.wrap(t, even)), t


def test_the_legend_fills_its_rows_evenly():
    ink = MD.Ink("legend_text", {})
    layers = [{"name": n, "group": g, "args": {}, "out": [8]} for n, g in
              (("Conv1d", "convolution"), ("BatchNorm1d", "normalization"), ("Pooling", "pooling"), ("Flatten", "shape"),
               ("Linear", "dense"), ("Dropout", "dropout"))]
    full = MD._legend({"layers": layers}, ink, 10_000)
    narrow = MD._legend({"layers": layers}, ink, full.w * 0.7)            # too narrow for one row: two
    rows = {}
    for op in narrow.ops:
        if op[0] == "text":
            rows.setdefault(round(op[2], 1), []).append(op[3])
    assert len(rows) == 2
    assert [len(v) for v in rows.values()] == [3, 3], "3 and 3, not 5 and 1"


def test_a_text_rule_switched_off_is_not_applied():
    import matplotlib.pyplot as plt

    fig = MD.figure(request(text_rules={"title": {"enabled": False, "text": "not drawn", "font": {"size": 8}},
                                        "subtitle": {"enabled": True, "text": "drawn"}}))
    try:
        texts = _texts(fig)
        assert "not drawn" not in texts and "ASCAD_MLPbase" in texts
        assert "drawn" in texts
    finally:
        plt.close(fig)
