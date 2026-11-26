# services/side_channel_core/utils/parameter_count.py
"""
Trainable parameters of a built model: the metric trainable_parameters
(src/sca/metrics.py) and the count of the panel's model check
(services/visualization/vscode_monitor/model_check.py).

  PyTorch  the parameters with requires_grad; BatchNorm's running statistics are
           buffers, not parameters, and a lazy layer not yet run has no shape, so
           neither is counted.
  Keras    the model's trainable_variables (core/framework_switch.KerasModel), not
           count_params(), which adds BatchNormalization's moving statistics.

Zaid et al. 2020 and Wouters et al. 2020 tabulate count_params(): 16,960 for
Zaid's ASCAD N0 network, whose trainable parameters are 16,952.

Author: Shraddha Mishra
"""

from __future__ import annotations

import math

import torch


def trainable_parameter_count(model: torch.nn.Module) -> int:
    """The number of trainable weights of a PyTorch model or of a wrapped Keras model."""
    keras_variables = getattr(model, "trainable_variables", None)
    if keras_variables is not None:
        return int(sum(math.prod(variable.shape) for variable in keras_variables))
    return int(sum(
        p.numel() for p in model.parameters()
        if p.requires_grad and not isinstance(p, torch.nn.parameter.UninitializedParameter)
    ))


__all__ = ["trainable_parameter_count"]
