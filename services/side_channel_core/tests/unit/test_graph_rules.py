# services/side_channel_core/tests/unit/test_graph_rules.py
"""A graph job's plot_type fits its metric; export keeps each format's settings under it; the
keys graph_manager no longer reads are refused with their replacement."""

import pytest

from services.graph_manager.graph_job_validator import GraphJobValidationError, validate_graph_job
from services.graph_manager.metric_router import metric_kind, plot_types_for
from services.graph_manager.plotters import get_dpi


def _job(**extra):
    job = {"id": 5, "name": "j", "type": "model_overlay", "enabled": True, "metric": "mean_accuracy",
           "source": "cross_validation", "models": ["m"], "processes": ["validation"], "output_file_name": "j"}
    job.update(extra)
    return job


def test_each_metric_is_offered_the_plot_types_its_drawer_reads():
    assert metric_kind("mean_accuracy") == "scalar" and metric_kind("some_new_scalar") == "scalar"
    assert metric_kind("guessing_entropy_curve") == "curve" and metric_kind("sr_curve_3") == "curve"
    assert metric_kind("final_ranks") == metric_kind("fold_rank_curves") == "fixed"
    assert plot_types_for("solo", ["accuracy"]) == ["auto", "line", "line_scatter", "scatter", "step", "bar"]
    assert plot_types_for("model_overlay", "accuracy")[-2:] == ["barh", "heatmap"]   # heatmap: key_recovery folds
    assert plot_types_for("model_overlay", ["mean_attack_rank_curve"]) == ["auto", "line", "line_scatter", "scatter", "step"]
    assert plot_types_for("solo", ["mean_attack_rank_curve"]) == ["auto"]
    assert plot_types_for("solo", ["mean_attack_rank_curve"], "overlay")[-1] == "step"
    assert plot_types_for("solo", ["key_candidate_scores"]) == plot_types_for("model_explanation", ["x"]) == ["auto"]
    assert plot_types_for("model_overlay", ["accuracy", "mean_attack_rank_curve"])[-1] == "step"   # what both take


def test_a_plot_type_that_does_not_change_the_drawing_is_refused():
    assert validate_graph_job(_job(plot_type="barh"))["plot_type"] == "barh"
    for bad in (dict(type="solo", plot_type="barh"), dict(metric="mean_attack_rank_curve", plot_type="bar"),
                dict(plot_type="histogram")):
        with pytest.raises(GraphJobValidationError, match="does not change how"):
            validate_graph_job(_job(**bad))
    with pytest.raises(GraphJobValidationError, match="source key_recovery"):    # a heatmap reads the folds' reports
        validate_graph_job(_job(plot_type="heatmap"))


def test_dpi_belongs_to_png_and_pdf_has_no_settings():
    job = validate_graph_job(_job(export={"formats": ["png", "pdf"], "png": {"dpi": 150}}))
    assert job["export"]["png"] == {"dpi": 150} and "pdf" not in job["export"] and get_dpi(job) == 150
    assert get_dpi(validate_graph_job(_job())) == 300
    with pytest.raises(GraphJobValidationError, match="export.png.dpi"):
        validate_graph_job(_job(export={"formats": ["png"], "dpi": 300}))
    with pytest.raises(GraphJobValidationError, match="export.pdf was removed"):
        validate_graph_job(_job(export={"formats": ["pdf"], "pdf": {"vector": True}}))


def test_text_rules_without_underline_and_the_targets_that_did_nothing():
    rule = {"target": "title", "text": "T", "bold": True, "italic": True, "color": "C1", "font": {"family": "serif", "size": 9}}
    out = validate_graph_job(_job(style={"text_rules": [rule]}))["style"]["text_rules"][0]
    assert out == rule
    for bad, why in (({**rule, "underline": False}, "underline"), ({"target": "legend"}, "legend_text"),
                     ({"target": "plot_labels"}, "nothing")):
        with pytest.raises(GraphJobValidationError, match=why):
            validate_graph_job(_job(style={"text_rules": [bad]}))
