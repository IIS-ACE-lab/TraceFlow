# services/side_channel_core/tests/unit/test_option_lists.py
"""Option lists the config panel's menus show (services/visualization/vscode_monitor/choices.py)
are the sets the code accepts: each listed name works, an unlisted one is refused."""

import pytest
import torch

from core.model import POOLING_TYPES, POOLING_TYPES_1D, Pooling


def _build(kind):
    layer = Pooling(type=kind, dim=1, kernel_size=2, stride=2, output_size=4, levels=[1, 2])
    return layer(torch.randn(2, 3, 16))


@pytest.mark.parametrize("kind", POOLING_TYPES_1D)
def test_every_pooling_type_the_panel_offers_builds_on_1d_traces(kind):
    assert _build(kind).shape[0] == 2


@pytest.mark.parametrize("kind", sorted(set(POOLING_TYPES) - set(POOLING_TYPES_1D)))
def test_the_pooling_types_left_out_of_the_menu_are_refused_on_1d_traces_with_the_reason(kind):
    # TODO.md §25: refused at construction, naming why and what builds instead.
    with pytest.raises(ValueError, match=r"cannot pool 1-D traces \(dim=1\): .*; use one of max, avg"):
        _build(kind)


def test_an_unlisted_pooling_type_is_refused_with_the_list():
    with pytest.raises(ValueError, match="one of max, avg"):
        Pooling(type="median", dim=1)


def test_the_panel_menus_name_only_what_the_code_accepts():
    from services.visualization.vscode_monitor.choices import static

    out = static()
    assert out["errors"] == []
    rules = {r["p"]: r for r in out["rules"]["params.yaml"]}
    assert set(rules["model.*.architecture_args.layers.*.Pooling.type"]["o"]) == set(POOLING_TYPES_1D)
    assert rules["trainer.loss.args.reduction"]["o"] == ["mean", "sum"]
    assert {"p": "search.method", "o": ["bo", "rs"]} in out["rules"]["auto_search.yaml"]
