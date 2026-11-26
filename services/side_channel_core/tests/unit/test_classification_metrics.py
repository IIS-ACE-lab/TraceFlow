# services/side_channel_core/tests/unit/test_classification_metrics.py
"""Classification metrics of src/sca/metrics.py are exact count ratios."""

import torch


def test_top_k_accuracy_is_the_exact_fraction_of_hits():
    from sca.metrics import compute_metrics

    n, hits = 500, 29
    labels = torch.arange(n) % 256
    logits = torch.zeros(n, 256)
    rows = torch.arange(n)
    logits[rows[:hits], labels[:hits]] = 10.0                       # the true label first
    logits[rows[hits:], (labels[hits:] + 1) % 256] = 10.0           # two other labels above it
    logits[rows[hits:], (labels[hits:] + 2) % 256] = 9.0
    cfg = {
        "metrics_to_log": [{"top_k_accuracy": {"k": 2}}],
        "metrics_cfg": {"preds_are_probs": False, "ce_eps": 1e-12},
        "trainer": {"loss": {"type": "CrossEntropyLoss", "args": {"reduction": "mean"}},
                    "evaluation": {"output_transformation": {"type": "Softmax"}}},
        "sca": {"leakage_model": "ID"},
        "deep_learning": {"framework": "pytorch"},
    }
    value = compute_metrics(logits, labels, cfg, process="train")["top_2_accuracy"]
    assert value == hits / n == 0.058                                # float32 gave 0.057999998331069946


def test_the_output_transformation_is_checked_against_what_the_caller_accepts():
    import pytest
    from utils.output_transformation_util import build_output_transformation_from_config

    def evaluation(name):
        return {"output_transformation": {"type": name, "args": {}}}

    for name, produced in (("Identity", "logits"), ("Softmax", "probabilities"), ("LogSoftmax", "log_probabilities")):
        transform = build_output_transformation_from_config(
            evaluation(name), expected_representation=("logits", "probabilities", "log_probabilities"))
        assert transform.representation == produced
    with pytest.raises(ValueError, match="Softmax produces probabilities, but the caller accepts logits or log_probabilities"):
        build_output_transformation_from_config(evaluation("Softmax"), expected_representation=("logits", "log_probabilities"))
