# services/side_channel_core/tests/unit/test_model_explanation.py
"""Model-explanation figures (services/graph_manager/explain.py) on networks whose answer is known."""

import numpy as np
import pytest
import torch

from services.graph_manager import explain

torch.manual_seed(0)
T, FILTERS, NEURONS = 6, 2, 3


def _cnn():
    return torch.nn.Sequential(torch.nn.Conv1d(1, FILTERS, 1), torch.nn.Flatten(), torch.nn.Linear(FILTERS * T, NEURONS))


def test_weight_visualization_is_zaids_mean_over_filters_then_neurons():
    model = _cnn()
    weights = torch.arange(NEURONS * FILTERS * T, dtype=torch.float32).reshape(NEURONS, FILTERS * T) - 10
    with torch.no_grad():
        model[2].weight.copy_(weights)
    got = explain.weight_visualization(model, torch.zeros(1, 1, T))
    # PyTorch flattens (filters, time) channel-first: input index f * T + t.
    want = np.abs(weights.numpy()).reshape(NEURONS, FILTERS, T).mean(axis=1).mean(axis=0)
    assert got.shape == (T,) and np.allclose(got, want)


def test_gradient_and_gradient_x_input_of_a_linear_model_are_its_true_class_row():
    linear = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(T, 4))
    x = torch.randn(5, 1, T)
    labels = torch.tensor([0, 1, 2, 3, 1])
    rows = linear[1].weight.detach().numpy()[labels.numpy()]                        # d score_y / d x = W[y]
    assert np.allclose(explain.attribution(linear, x, labels, "gradient", "mean_abs"), np.abs(rows).mean(axis=0))
    product = x.reshape(5, T).numpy() * rows
    assert np.allclose(explain.attribution(linear, x, labels, "gradient_x_input", "mean"), product.mean(axis=0), atol=1e-6)
    assert np.allclose(explain.attribution(linear, x, labels, "gradient_x_input", "mean_abs"), np.abs(product).mean(axis=0), atol=1e-6)


def test_a_heatmap_is_the_conv_output_averaged_over_filters_and_traces():
    model = torch.nn.Sequential(torch.nn.Conv1d(1, FILTERS, 1), torch.nn.SELU(), torch.nn.Conv1d(FILTERS, 4, 3, padding=1),
                                torch.nn.Flatten(), torch.nn.Linear(4 * T, 2))
    x = torch.randn(7, 1, T)
    first = model[0](x).detach().mean(dim=1).numpy()                                  # Eq. (6), before the activation
    maps = explain.heatmaps(model, x, [1, 2], "mean")
    assert sorted(maps) == [1, 2] and np.allclose(maps[1], first.mean(axis=0), atol=1e-6)
    assert maps[2].shape == (T,)
    with pytest.raises(ValueError, match="outside the model's 2 Conv1d layers"):
        explain.heatmaps(model, x, [3], "mean")


def test_an_explanation_job_needs_a_method_and_no_metric():
    from services.graph_manager.graph_job_validator import GraphJobValidationError, validate_graph_job

    job = {"id": 9, "name": "w", "type": "model_explanation", "enabled": True, "method": "heatmap",
           "layers": [1], "traces": 100, "models": ["m"], "output_file_name": "w"}
    checked = validate_graph_job(job)
    assert checked["aggregate"] == "mean_abs" and checked["models"] == ["m"]
    for bad, match in (({"method": "lrp"}, "use one of"), ({"aggregate": "max"}, "aggregate"),
                       ({"traces": 0}, "positive count"), ({"method": "gradient"}, "heatmap only")):
        with pytest.raises(GraphJobValidationError, match=match):
            validate_graph_job({**job, **bad})
