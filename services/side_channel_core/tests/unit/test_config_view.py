"""utils/config_view.py: the resolved view written to the snapshot and the logged CONFIG."""
import copy

from utils.config_view import EXCLUDED_SECTIONS, drop_unselected_variants, resolved_view


def _dl(framework="keras"):
    return {
        "framework": framework,
        "pytorch": {"weight_init": "xavier_uniform", "weight_init_args": {"gain": 1.0}, "bias_init": "zeros", "bias_init_args": {}},
        "keras": {"weight_init": "glorot_uniform", "weight_init_args": {}, "bias_init": "zeros", "bias_init_args": {}},
    }


def test_only_the_selected_variant_stays():
    assert drop_unselected_variants(_dl("keras")) == {"framework": "keras", "keras": _dl()["keras"]}
    assert drop_unselected_variants(_dl("pytorch")) == {"framework": "pytorch", "pytorch": _dl()["pytorch"]}


def test_any_depth_and_inside_lists():
    cfg = {"a": {"b": [{"x": 1, "dl": _dl("pytorch")}]}}
    assert drop_unselected_variants(cfg)["a"]["b"][0]["dl"] == {"framework": "pytorch", "pytorch": _dl()["pytorch"]}


def test_unrelated_sibling_mappings_stay():
    node = {"framework": "keras", "keras": {"weight_init": "x", "bias_init": "z"},
            "pytorch": {"weight_init": "y", "bias_init": "z"}, "common": {"device": "cuda", "precision": "fp32"}}
    assert set(drop_unselected_variants(node)) == {"framework", "keras", "common"}


def test_no_selector_no_change():
    node = {"loss": {"type": "CrossEntropyLoss", "args": {"reduction": "mean"}}, "optimizer": {"type": "RMSprop"}}
    assert drop_unselected_variants(node) == node


def test_graph_is_left_out_and_the_input_is_not_changed():
    cfg = {"graph": {"jobs": [1, 2]}, "deep_learning": _dl(), "seed": 7}
    before = copy.deepcopy(cfg)
    view = resolved_view(cfg)
    assert "graph" in EXCLUDED_SECTIONS and "graph" not in view
    assert view["deep_learning"] == {"framework": "keras", "keras": _dl()["keras"]} and view["seed"] == 7
    assert cfg == before


def test_the_snapshot_and_the_logged_config_leave_out_test_and_graph():
    # test_config.yaml (make test reads it itself) and graph_config.yaml are not what a run was trained with
    cfg = {"graph": {}, "test": {"test_request": {"model": ["mesh_chicken"]}}, "seed": 1}
    for view in (resolved_view(cfg), resolved_view(cfg, for_log=True)):
        assert "test" not in view and "graph" not in view and view["seed"] == 1
