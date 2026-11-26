#services/side_channel_core/utils/output_transformation_util.py

"""Transform raw SCA logits [batch, classes] for evaluation.

Supported options (case-insensitive): Identity, Softmax, LogSoftmax.
Identity returns logits; Softmax returns probabilities; LogSoftmax returns
log-probabilities. The class dimension is fixed to 1. These transformations
must not be applied before the logits-based loss built by loss_util.py.
The caller owns eval() and no_grad()/inference_mode(); this module does not
change model state, detach tensors, or move data between devices.

Author: Shraddha Mishra
"""

from collections.abc import Mapping

import torch


OUTPUT_TRANSFORM_REGISTRY = {
    "Identity": (torch.nn.Identity, "logits"),
    "Softmax": (torch.nn.Softmax, "probabilities"),
    "LogSoftmax": (torch.nn.LogSoftmax, "log_probabilities"),
}


def build_output_transformation_from_config(
    evaluation_cfg: Mapping,
    *,
    expected_representation: str | tuple[str, ...],
) -> torch.nn.Module:
    """Build trainer.evaluation.output_transformation.

    Pass cfg['trainer']['evaluation']; the transformation block contains
    {type: Softmax, args: {}}.

    expected_representation is what the consuming function actually accepts:
    one of logits, probabilities, log_probabilities, or a tuple of them when it
    handles several. Do not derive it from this same YAML setting; doing so
    would defeat the compatibility check. The returned module carries the
    representation it produces as ``.representation``, so the consumer
    branches on what the transformation gives, not on a reading of the YAML.
    Only dim=1 is supported for [batch, classes] predictions.
    """
    accepted = (expected_representation,) if isinstance(expected_representation, str) else tuple(expected_representation)
    if not accepted or set(accepted) - {"logits", "probabilities", "log_probabilities"}:
        raise ValueError("Invalid expected output representation")
    if not isinstance(evaluation_cfg, Mapping):
        raise TypeError("evaluation must be a mapping")
    cfg = evaluation_cfg.get("output_transformation")
    if not isinstance(cfg, Mapping):
        raise TypeError(
            "trainer.evaluation.output_transformation must be a resolved "
            f"mapping, got {cfg!r}"
        )
    if "type" not in cfg or set(cfg) - {"type", "args"}:
        raise ValueError(
            "trainer.evaluation.output_transformation requires type "
            f"and optionally args; received keys: {list(cfg)}"
        )
    name = cfg["type"]
    if not isinstance(name, str):
        raise TypeError("output_transformation.type must be a resolved string")
    canonical = next((key for key in OUTPUT_TRANSFORM_REGISTRY if key.casefold() == name.strip().casefold()), None)
    if canonical is None:
        raise ValueError(f"Unsupported transformation {name!r}; supported: {tuple(OUTPUT_TRANSFORM_REGISTRY)}")
    raw_args = cfg.get("args", {})
    if not isinstance(raw_args, Mapping):
        raise TypeError("output_transformation.args must be a mapping")
    args = dict(raw_args)
    transform_class, representation = OUTPUT_TRANSFORM_REGISTRY[canonical]
    if representation not in accepted:
        raise ValueError(
            f"{canonical} produces {representation}, but the caller accepts "
            f"{' or '.join(accepted)}"
        )
    if canonical == "Identity":
        if args:
            raise ValueError("Identity accepts no configured arguments")
        transform = transform_class()
    else:
        if set(args) - {"dim"}:
            raise ValueError("Softmax and LogSoftmax accept only dim")
        dim = args.get("dim", 1)
        if type(dim) is not int or dim != 1:
            raise ValueError("SCA outputs are [batch, classes]; dim must be integer 1")
        transform = transform_class(dim=1)
    transform.representation = representation
    return transform
