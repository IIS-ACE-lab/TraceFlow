# services/side_channel_core/utils/h5_metadata_utils.py
"""
HDF5 metadata and metric-gating utilities for Side-Channel Analysis (SCA).

Responsibilities
----------------
- Unpack ordinary ``(x, y)`` and SCA ``(x, y, metadata)`` batches.
- Normalize supported metadata layouts into an ordered per-sample sequence.
- Collect model logits, labels, and metadata with strict batch alignment.
- Classify metrics that require plaintext/key metadata.
- Remove metadata-dependent metrics from metadata-free passes, especially
  profiling-data training.
- Resolve rank-evaluation scheduling and artifact directories.

Scientific contract
-------------------
Rank, guessing-entropy, success-rate, LDD, Lm, and explicit key-recovery
metrics require per-sample metadata containing exactly one 16-byte plaintext
and one 16-byte key. This module never fabricates metadata and never converts
malformed metadata into ``None``. If a loader supplies metadata, that metadata
must be complete, byte-valid, and aligned with predictions and labels.

Metric configuration contract
-----------------------------
``src.sca.metrics.compute_metrics`` gives ``metrics_to_log`` precedence over
``metrics``. Previous code filtered only ``metrics``; rank evaluation could
therefore still execute through the untouched ``metrics_to_log`` list. This
module treats both keys as mirrored views of one canonical list and rejects
conflicting definitions.

Author: Shraddha Mishra
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch

from src.sca.metrics import (
    VALID_METRIC_PROCESSES,
    canonical_fit_time_name,
    is_metric_valid_for_process,
)
from utils.path_utils import resolve_path



# =============================================================================
# Batch contract
# =============================================================================
def unpack_batch(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
    """Return ``(inputs, labels, metadata_or_none)`` from one DataLoader batch."""
    if not isinstance(batch, (tuple, list)):
        raise TypeError(
            "Expected DataLoader batch as tuple/list containing (x, y) or "
            f"(x, y, metadata), got {type(batch).__name__}"
        )

    if len(batch) == 2:
        inputs, labels = batch
        metadata = None
    elif len(batch) == 3:
        inputs, labels, metadata = batch
    else:
        raise ValueError(
            "Expected DataLoader batch length 2 or 3, "
            f"got length={len(batch)}"
        )

    if not isinstance(inputs, torch.Tensor):
        raise TypeError(
            f"Batch inputs must be torch.Tensor, got {type(inputs).__name__}"
        )

    if not isinstance(labels, torch.Tensor):
        raise TypeError(
            f"Batch labels must be torch.Tensor, got {type(labels).__name__}"
        )

    return inputs, labels, metadata


# =============================================================================
# HDF5 provenance helpers
# =============================================================================
def _strict_int(value: Any, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, got boolean {value!r}")

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or float(value) != float(int(value)):
            raise ValueError(f"{field} must be an integer, got {value!r}")

    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be integer-like, got {value!r}") from exc

    return result


def _decode_h5_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_attack_byte_attr(h5_path: str | Path) -> int:
    """
    Read and cross-check the active attacked byte from HDF5 provenance.

    New artifacts use ``sca_active_attack_byte`` and retain
    ``sca_attack_byte`` as a compatibility attribute. When the canonical JSON
    data contract is present, ``active.attack_byte`` is checked as an
    independent third source. Missing provenance returns ``-1`` for legacy
    compatibility; malformed, tampered, or conflicting provenance raises.
    """
    path = Path(h5_path)
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    sources: dict[str, int] = {}
    with h5py.File(str(path), "r") as handle:
        for attribute in ("sca_active_attack_byte", "sca_attack_byte"):
            raw = handle.attrs.get(attribute)
            if raw is None:
                continue
            value = _strict_int(_decode_h5_attr(raw), field=attribute)
            if not 0 <= value <= 15:
                raise ValueError(
                    f"{attribute} must be in [0, 15], got {value}"
                )
            sources[attribute] = value

        raw_contract = handle.attrs.get("sca_data_contract_json")
        raw_digest = handle.attrs.get("sca_data_contract_sha256")

        if raw_contract is None and raw_digest is not None:
            raise ValueError(
                f"{path} has sca_data_contract_sha256 but no contract payload"
            )
        if raw_contract is not None and raw_digest is None:
            raise ValueError(
                f"{path} has sca_data_contract_json but no SHA-256 fingerprint"
            )

        if raw_contract is not None:
            serialized = str(_decode_h5_attr(raw_contract))
            stored_digest = str(_decode_h5_attr(raw_digest))
            computed_digest = hashlib.sha256(
                serialized.encode("utf-8")
            ).hexdigest()
            if stored_digest != computed_digest:
                raise ValueError(
                    f"Data-contract SHA-256 mismatch in {path}"
                )

            try:
                contract = json.loads(serialized)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid sca_data_contract_json in {path}"
                ) from exc

            if not isinstance(contract, Mapping):
                raise ValueError(
                    f"sca_data_contract_json in {path} must encode an object"
                )
            active_contract = contract.get("active") or {}
            if not isinstance(active_contract, Mapping):
                raise ValueError(
                    f"sca_data_contract_json in {path} has invalid active section"
                )
            raw_byte = active_contract.get("attack_byte")
            if raw_byte is not None:
                contract_byte = _strict_int(
                    raw_byte,
                    field="contract.active.attack_byte",
                )
                if not 0 <= contract_byte <= 15:
                    raise ValueError(
                        "contract.active.attack_byte must be in [0, 15], "
                        f"got {contract_byte}"
                    )
                sources["contract.active.attack_byte"] = contract_byte

    if not sources:
        return -1

    unique_values = set(sources.values())
    if len(unique_values) != 1:
        details = ", ".join(
            f"{name}={value}" for name, value in sorted(sources.items())
        )
        raise ValueError(
            f"Conflicting HDF5 attack-byte provenance in {path}: {details}"
        )

    return next(iter(unique_values))


# =============================================================================
# Strict metadata normalization
# =============================================================================
def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_u8_16(value: Any, *, name: str) -> np.ndarray:
    """Return one validated 16-byte vector without uint8 wraparound."""
    if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.shape == (16,):
        # Already a byte vector: every check below holds by construction.
        return value.copy()
    array = _to_numpy(value)

    if array.shape != (16,):
        raise ValueError(f"{name} must have shape (16,), got {array.shape}")

    if array.dtype.kind == "b":
        raise TypeError(f"{name} must contain byte integers, not booleans")

    if array.dtype.kind in {"f", "c"}:
        if array.dtype.kind == "c":
            raise TypeError(f"{name} must contain real byte integers")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinity")
        if not np.all(np.equal(array, np.floor(array))):
            raise ValueError(f"{name} contains non-integer floating-point values")

    try:
        integer_array = array.astype(np.int64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain integer-like byte values") from exc

    if np.any(integer_array < 0) or np.any(integer_array > 255):
        minimum = int(np.min(integer_array))
        maximum = int(np.max(integer_array))
        raise ValueError(
            f"{name} contains values outside byte range [0, 255]: "
            f"min={minimum}, max={maximum}"
        )

    return integer_array.astype(np.uint8, copy=True)


def _copy_optional_value(value: Any) -> Any:
    """Detach tensors and copy NumPy-backed optional metadata safely."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return array.copy() if array.shape else array.item()

    if isinstance(value, np.ndarray):
        return value.copy()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Mapping):
        return {str(key): _copy_optional_value(item) for key, item in value.items()}

    if isinstance(value, tuple):
        return tuple(_copy_optional_value(item) for item in value)

    if isinstance(value, list):
        return [_copy_optional_value(item) for item in value]

    return value


def _normalize_optional_field(name: str, value: Any) -> Any:
    copied = _copy_optional_value(value)

    if name in {"desync", "source_index", "source_indices"}:
        array = np.asarray(copied)
        if array.shape == ():
            return _strict_int(array.item(), field=name)

    return copied


def _normalize_sample_mapping(
    sample: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    if "plaintext" not in sample or "key" not in sample:
        missing = [name for name in ("plaintext", "key") if name not in sample]
        raise KeyError(f"{context} metadata missing required field(s): {missing}")

    normalized: dict[str, Any] = {
        "plaintext": _as_u8_16(
            sample["plaintext"],
            name=f"{context}.plaintext",
        ),
        "key": _as_u8_16(
            sample["key"],
            name=f"{context}.key",
        ),
    }

    for field, value in sample.items():
        field_name = str(field)
        if field_name in {"plaintext", "key"}:
            continue
        normalized[field_name] = _normalize_optional_field(field_name, value)

    return normalized


def _structured_row_to_dict(row: np.void, *, context: str) -> dict[str, Any]:
    fields = row.dtype.fields or {}
    if "plaintext" not in fields or "key" not in fields:
        missing = [name for name in ("plaintext", "key") if name not in fields]
        raise KeyError(
            f"{context} structured metadata missing required field(s): {missing}"
        )

    mapping = {field: row[field] for field in fields}
    return _normalize_sample_mapping(mapping, context=context)


def _slice_batched_optional_value(value: Any, *, index: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and int(value.shape[0]) == batch_size:
            return value[index]
        return value

    if isinstance(value, np.ndarray):
        if value.ndim > 0 and int(value.shape[0]) == batch_size:
            return value[index]
        return value

    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[index]

    return value


def _mapping_batch_to_samples(
    metadata: Mapping[str, Any],
    *,
    expected_batch_size: Optional[int],
    context: str,
) -> list[dict[str, Any]]:
    if "plaintext" not in metadata or "key" not in metadata:
        missing = [name for name in ("plaintext", "key") if name not in metadata]
        raise KeyError(f"{context} metadata missing required field(s): {missing}")

    plaintext = _to_numpy(metadata["plaintext"])
    key = _to_numpy(metadata["key"])

    if plaintext.shape == (16,) and key.shape == (16,):
        if expected_batch_size not in (None, 1):
            raise ValueError(
                f"{context} contains one metadata sample but batch size is "
                f"{expected_batch_size}"
            )
        return [_normalize_sample_mapping(metadata, context=f"{context}[0]")]

    if plaintext.ndim != 2 or key.ndim != 2:
        raise ValueError(
            f"{context}.plaintext and {context}.key must have shape [B,16] "
            f"or [16], got plaintext={plaintext.shape}, key={key.shape}"
        )

    if plaintext.shape != key.shape or plaintext.shape[1] != 16:
        raise ValueError(
            f"{context} plaintext/key shape mismatch: "
            f"plaintext={plaintext.shape}, key={key.shape}; expected [B,16]"
        )

    batch_size = int(plaintext.shape[0])
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise ValueError(
            f"{context} metadata batch size {batch_size} does not match "
            f"prediction/label batch size {expected_batch_size}"
        )

    samples: list[dict[str, Any]] = []
    for index in range(batch_size):
        sample: dict[str, Any] = {
            "plaintext": plaintext[index],
            "key": key[index],
        }

        for field, value in metadata.items():
            field_name = str(field)
            if field_name in {"plaintext", "key"}:
                continue
            sample[field_name] = _slice_batched_optional_value(
                value,
                index=index,
                batch_size=batch_size,
            )

        samples.append(
            _normalize_sample_mapping(
                sample,
                context=f"{context}[{index}]",
            )
        )

    return samples


def _looks_like_plaintext_key_pair(metadata: Sequence[Any]) -> bool:
    if len(metadata) != 2:
        return False

    if all(isinstance(item, Mapping) for item in metadata):
        return False

    try:
        plaintext = _to_numpy(metadata[0])
        key = _to_numpy(metadata[1])
    except Exception:
        return False

    valid_shapes = {
        (16,),
    }

    if plaintext.shape in valid_shapes and key.shape == plaintext.shape:
        return True

    return (
        plaintext.ndim == 2
        and key.ndim == 2
        and plaintext.shape == key.shape
        and plaintext.shape[1] == 16
    )


def _plaintext_key_pair_to_samples(
    metadata: Sequence[Any],
    *,
    expected_batch_size: Optional[int],
    context: str,
) -> list[dict[str, Any]]:
    plaintext = _to_numpy(metadata[0])
    key = _to_numpy(metadata[1])

    if plaintext.shape == (16,):
        if expected_batch_size not in (None, 1):
            raise ValueError(
                f"{context} contains one plaintext/key pair but batch size is "
                f"{expected_batch_size}"
            )
        return [
            _normalize_sample_mapping(
                {"plaintext": plaintext, "key": key},
                context=f"{context}[0]",
            )
        ]

    batch_size = int(plaintext.shape[0])
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise ValueError(
            f"{context} plaintext/key pair batch size {batch_size} does not "
            f"match prediction/label batch size {expected_batch_size}"
        )

    return [
        _normalize_sample_mapping(
            {"plaintext": plaintext[index], "key": key[index]},
            context=f"{context}[{index}]",
        )
        for index in range(batch_size)
    ]


def _normalize_metadata_unit(
    metadata: Any,
    *,
    expected_batch_size: Optional[int],
    context: str,
) -> list[dict[str, Any]]:
    if metadata is None:
        raise ValueError(f"{context} metadata is None")

    if isinstance(metadata, Mapping):
        return _mapping_batch_to_samples(
            metadata,
            expected_batch_size=expected_batch_size,
            context=context,
        )

    if isinstance(metadata, np.void):
        if expected_batch_size not in (None, 1):
            raise ValueError(
                f"{context} contains one structured row but batch size is "
                f"{expected_batch_size}"
            )
        return [_structured_row_to_dict(metadata, context=f"{context}[0]")]

    if isinstance(metadata, np.ndarray):
        if metadata.dtype.fields is not None:
            rows = metadata.reshape(-1)
            if expected_batch_size is not None and len(rows) != expected_batch_size:
                raise ValueError(
                    f"{context} structured metadata count {len(rows)} does not "
                    f"match batch size {expected_batch_size}"
                )
            return [
                _structured_row_to_dict(row, context=f"{context}[{index}]")
                for index, row in enumerate(rows)
            ]

        if metadata.dtype == object:
            return _normalize_metadata_unit(
                metadata.tolist(),
                expected_batch_size=expected_batch_size,
                context=context,
            )

        raise TypeError(
            f"{context} numeric ndarray is ambiguous metadata; provide a "
            "mapping, structured rows, per-sample mappings, or (plaintext, key)"
        )

    if isinstance(metadata, torch.Tensor):
        raise TypeError(
            f"{context} bare tensor is ambiguous metadata; provide "
            "(plaintext_batch, key_batch) or a metadata mapping"
        )

    if isinstance(metadata, (list, tuple)):
        if not metadata:
            if expected_batch_size in (None, 0):
                return []
            raise ValueError(
                f"{context} metadata is empty for batch size {expected_batch_size}"
            )

        if _looks_like_plaintext_key_pair(metadata):
            return _plaintext_key_pair_to_samples(
                metadata,
                expected_batch_size=expected_batch_size,
                context=context,
            )

        samples: list[dict[str, Any]] = []
        for index, item in enumerate(metadata):
            item_samples = _normalize_metadata_unit(
                item,
                expected_batch_size=1,
                context=f"{context}[{index}]",
            )
            if len(item_samples) != 1:
                raise ValueError(
                    f"{context}[{index}] expanded to {len(item_samples)} samples; "
                    "expected exactly one per-sample metadata item"
                )
            samples.extend(item_samples)

        if expected_batch_size is not None and len(samples) != expected_batch_size:
            raise ValueError(
                f"{context} metadata count {len(samples)} does not match "
                f"batch size {expected_batch_size}"
            )

        return samples

    raise TypeError(
        f"Unsupported {context} metadata type: {type(metadata).__name__}"
    )


def normalize_metadata(
    meta_batches: Any,
    *,
    expected_count: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Normalize metadata batches into ``np.ndarray(dtype=object)`` of dictionaries.

    ``None`` or an empty outer sequence means metadata was not supplied. Any
    non-empty supplied metadata that cannot be normalized raises; it is never
    silently converted into ``None``.
    """
    if meta_batches is None:
        return None

    if isinstance(meta_batches, Mapping) or isinstance(meta_batches, np.void):
        outer_batches = [meta_batches]
    elif isinstance(meta_batches, np.ndarray) and meta_batches.dtype.fields is not None:
        outer_batches = [meta_batches]
    elif isinstance(meta_batches, (list, tuple)):
        if not meta_batches:
            if expected_count not in (None, 0):
                raise ValueError(
                    f"No metadata was collected for {int(expected_count)} samples"
                )
            return None
        if _looks_like_plaintext_key_pair(meta_batches):
            outer_batches = [meta_batches]
        else:
            outer_batches = list(meta_batches)
    else:
        outer_batches = [meta_batches]

    normalized: list[dict[str, Any]] = []
    for batch_index, metadata in enumerate(outer_batches):
        normalized.extend(
            _normalize_metadata_unit(
                metadata,
                expected_batch_size=None,
                context=f"metadata_batch[{batch_index}]",
            )
        )

    if not normalized:
        if expected_count in (None, 0):
            return None
        raise ValueError(
            f"Metadata normalized to zero samples, expected {expected_count}"
        )

    if expected_count is not None and len(normalized) != int(expected_count):
        raise ValueError(
            f"Metadata count {len(normalized)} does not match expected "
            f"sample count {int(expected_count)}"
        )

    output = np.empty(len(normalized), dtype=object)
    output[:] = normalized
    return output


# =============================================================================
# Evaluation collection
# =============================================================================
@torch.inference_mode()
def collect_logits_labels_meta(
    model: torch.nn.Module,
    device: torch.device,
    loader: Iterable,
) -> tuple[torch.Tensor, torch.Tensor, Optional[np.ndarray]]:
    """
    Collect aligned logits, labels, and optional per-sample metadata.

    The loader must be non-empty. Metadata may be absent from every batch, or
    present in every batch; mixed metadata presence is rejected because it
    would make rank evidence incomplete.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model).__name__}")

    was_training = bool(model.training)
    model.eval()

    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    normalized_metadata: list[dict[str, Any]] = []

    saw_metadata = False
    saw_missing_metadata = False
    expected_classes: Optional[int] = None
    total_samples = 0

    try:
        for batch_index, batch in enumerate(loader):
            inputs, labels, metadata = unpack_batch(batch)

            if inputs.dim() < 1:
                raise ValueError(
                    f"Batch {batch_index} inputs must have a batch dimension, "
                    f"got shape={tuple(inputs.shape)}"
                )

            if labels.dim() != 1:
                raise ValueError(
                    f"Batch {batch_index} labels must have shape [B], "
                    f"got {tuple(labels.shape)}"
                )
            if labels.dtype == torch.bool:
                raise TypeError(
                    f"Batch {batch_index} labels must be integer class indices, "
                    "not booleans"
                )
            if labels.is_complex():
                raise TypeError(
                    f"Batch {batch_index} labels must be real integer class indices"
                )
            if labels.is_floating_point():
                if not torch.isfinite(labels).all():
                    raise ValueError(
                        f"Batch {batch_index} labels contain NaN or infinity"
                    )
                if not torch.equal(labels, torch.floor(labels)):
                    raise ValueError(
                        f"Batch {batch_index} labels contain non-integer values"
                    )

            batch_size = int(labels.shape[0])
            if batch_size <= 0:
                raise ValueError(f"Batch {batch_index} is empty")

            if int(inputs.shape[0]) != batch_size:
                raise ValueError(
                    f"Batch {batch_index} input/label size mismatch: "
                    f"inputs={int(inputs.shape[0])}, labels={batch_size}"
                )

            inputs_device = inputs.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True, dtype=torch.long)

            logits = model(inputs_device)
            if not isinstance(logits, torch.Tensor):
                raise TypeError(
                    f"Model output for batch {batch_index} must be torch.Tensor, "
                    f"got {type(logits).__name__}"
                )

            if logits.dim() != 2:
                raise ValueError(
                    f"Model output for batch {batch_index} must have shape [B,C], "
                    f"got {tuple(logits.shape)}"
                )

            if int(logits.shape[0]) != batch_size:
                raise ValueError(
                    f"Batch {batch_index} logits/labels size mismatch: "
                    f"logits={int(logits.shape[0])}, labels={batch_size}"
                )

            class_count = int(logits.shape[1])
            if class_count <= 1:
                raise ValueError(
                    f"Batch {batch_index} model output must contain at least "
                    f"two classes, got shape={tuple(logits.shape)}"
                )

            if expected_classes is None:
                expected_classes = class_count
            elif class_count != expected_classes:
                raise ValueError(
                    f"Model class dimension changed across batches: "
                    f"expected {expected_classes}, got {class_count} at batch "
                    f"{batch_index}"
                )

            # Finite/range checks run once on the concatenated pass below;
            # per-batch .item() calls forced a device synchronisation each.
            logits_batches.append(logits.detach())
            label_batches.append(labels_device.detach())
            total_samples += batch_size

            if metadata is None:
                saw_missing_metadata = True
            else:
                saw_metadata = True
                batch_metadata = _normalize_metadata_unit(
                    metadata,
                    expected_batch_size=batch_size,
                    context=f"batch[{batch_index}].metadata",
                )
                normalized_metadata.extend(batch_metadata)

            if saw_metadata and saw_missing_metadata:
                raise ValueError(
                    "Loader mixes batches with and without metadata. SCA metadata "
                    "must be present for every sample or absent for the entire pass."
                )

        if not logits_batches:
            raise ValueError("Evaluation loader produced zero batches")

        predictions = torch.cat(logits_batches, dim=0)
        labels = torch.cat(label_batches, dim=0).to(dtype=torch.long)
        class_count = int(predictions.shape[1]) if predictions.dim() == 2 else 0
        if not bool(torch.isfinite(predictions).all()):
            raise ValueError("Model output contains NaN or infinity in the evaluation pass")
        if labels.numel():
            minimum_label = int(labels.min().item())
            maximum_label = int(labels.max().item())
            if minimum_label < 0 or maximum_label >= class_count:
                raise ValueError(
                    "Evaluation labels are outside model class range "
                    f"[0, {class_count - 1}]: min={minimum_label}, max={maximum_label}"
                )
        predictions = predictions.cpu()
        labels = labels.cpu()

        if predictions.dim() != 2 or int(predictions.shape[0]) != total_samples:
            raise RuntimeError(
                "Internal prediction concatenation error: "
                f"shape={tuple(predictions.shape)}, expected samples={total_samples}"
            )

        if labels.shape != (total_samples,):
            raise RuntimeError(
                "Internal label concatenation error: "
                f"shape={tuple(labels.shape)}, expected ({total_samples},)"
            )

        metadata_array: Optional[np.ndarray]
        if saw_metadata:
            if len(normalized_metadata) != total_samples:
                raise ValueError(
                    f"Metadata misalignment: metadata={len(normalized_metadata)}, "
                    f"predictions={total_samples}"
                )
            metadata_array = np.empty(total_samples, dtype=object)
            metadata_array[:] = normalized_metadata
        else:
            metadata_array = None

        return predictions, labels, metadata_array

    finally:
        if was_training:
            model.train()


# =============================================================================
# Path helpers used by epoch utilities
# =============================================================================
def _resolved_dir(cfg: dict, key: str, *, model_name: str) -> Path:
    paths = cfg.get("paths") or {}
    if not isinstance(paths, Mapping):
        raise TypeError("cfg['paths'] must be a mapping")

    raw = paths.get(key)
    if not raw:
        raise KeyError(f"cfg['paths']['{key}'] missing")

    return Path(resolve_path(raw, cfg, model_name=model_name))


# =============================================================================
# Metadata-dependent metric contract
# =============================================================================
# Derive the metadata-dependent families from the canonical metric registry.
# This removes the former second, incomplete metric list from this module.
_METADATA_REQUIRED_METRICS = frozenset(
    metric
    for metric, valid_processes in VALID_METRIC_PROCESSES.items()
    if "train" not in valid_processes
    and any(process in valid_processes for process in ("validation", "test"))
)

# Public contract plus backward-compatible private name used by older callers.
METADATA_REQUIRED_METRICS = _METADATA_REQUIRED_METRICS
_RANK_METRICS = _METADATA_REQUIRED_METRICS

_METRIC_ALIASES = {
    "ge": "guessing_entropy",
    "final_ge": "guessing_entropy",
    "auc_guessing_entropy": "auc_ge",
    "success_rate_curve": "sr_curve",
    "auc_success_rate": "auc_sr",
    "rank_success": "rank_k_success",
}

_MATERIALIZED_METADATA_METRIC_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^traces_to_ge_[1-9][0-9]*$"), "traces_to_ge_k"),
    (re.compile(r"^rank@[1-9][0-9]*$"), "rank_at_traces"),
    (re.compile(r"^ge@[1-9][0-9]*$"), "ge_at_traces"),
    (re.compile(r"^rank_k_success@[1-9][0-9]*$"), "rank_k_success"),
    (re.compile(r"^sr_curve@[1-9][0-9]*$"), "sr_curve"),
    (re.compile(r"^sr_curve_[1-9][0-9]*$"), "sr_curve"),
    (re.compile(r"^auc_sr@[1-9][0-9]*$"), "auc_sr"),
    (re.compile(r"^time_to_recover@[1-9][0-9]*$"), "time_to_recover"),
    (re.compile(r"^ge_reduction_rate@[1-9][0-9]*$"), "ge_reduction_rate"),
    (re.compile(r"^rank_[0-9]+_percentile$"), "rank_percentile"),
)


def _canonical_metric_name(name: Any) -> str:
    normalized = str(name).strip()
    if not normalized:
        raise ValueError("Metric name must not be empty")

    fit_time = canonical_fit_time_name(normalized)
    if fit_time is not None:
        return fit_time

    lowered = normalized.lower()
    alias = _METRIC_ALIASES.get(lowered)
    if alias is not None:
        return alias

    if lowered == "ldd":
        return "LDD"
    if lowered == "lm":
        return "Lm"

    for canonical in _METADATA_REQUIRED_METRICS:
        if canonical.lower() == lowered:
            return canonical

    return lowered


def _metric_family(name: Any) -> str:
    canonical = _canonical_metric_name(name)

    for pattern, family in _MATERIALIZED_METADATA_METRIC_PATTERNS:
        if pattern.fullmatch(canonical):
            return family

    return canonical


def is_metadata_required_metric(name: Any) -> bool:
    """Return whether a configured or materialized metric requires SCA metadata."""
    family = _metric_family(name)
    return not is_metric_valid_for_process(family, "train")


# Descriptive public alias retained for predicate-style callers.
metric_requires_metadata = is_metadata_required_metric


def _parse_metric_specs(metrics: Any) -> list[tuple[str, dict[str, Any]]]:
    if metrics is None:
        return []

    if not isinstance(metrics, list):
        raise TypeError(
            "metrics_to_log / metrics must be a list, "
            f"got {type(metrics).__name__}"
        )

    parsed: list[tuple[str, dict[str, Any]]] = []
    for index, metric in enumerate(metrics):
        if isinstance(metric, str):
            name = metric.strip()
            if not name:
                raise ValueError(f"Metric at index {index} must not be empty")
            parsed.append((name, {}))
            continue

        if isinstance(metric, Mapping) and len(metric) == 1:
            name_raw, options_raw = next(iter(metric.items()))
            name = str(name_raw).strip()
            if not name:
                raise ValueError(f"Metric at index {index} has an empty name")

            if options_raw is None:
                options: dict[str, Any] = {}
            elif isinstance(options_raw, Mapping):
                options = copy.deepcopy(dict(options_raw))
            else:
                raise TypeError(
                    f"Metric options for {name!r} must be a mapping or null, "
                    f"got {type(options_raw).__name__}"
                )

            parsed.append((name, options))
            continue

        raise TypeError(
            f"Invalid metric specification at index {index}: {metric!r}. "
            "Expected a metric name string or one-entry mapping."
        )

    return parsed


def _metric_specs_to_config(
    parsed: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[Any]:
    output: list[Any] = []
    for name, options in parsed:
        output.append({name: copy.deepcopy(dict(options))} if options else name)
    return output


def _configured_metric_list(cfg: Mapping[str, Any]) -> list[Any]:
    has_primary = "metrics_to_log" in cfg and cfg.get("metrics_to_log") is not None
    has_legacy = "metrics" in cfg and cfg.get("metrics") is not None

    if not has_primary and not has_legacy:
        return []

    if has_primary:
        primary = cfg.get("metrics_to_log")
        _parse_metric_specs(primary)
    else:
        primary = cfg.get("metrics")
        _parse_metric_specs(primary)

    if has_primary and has_legacy:
        legacy = cfg.get("metrics")
        _parse_metric_specs(legacy)
        if primary != legacy:
            raise ValueError(
                "Conflicting metric configuration: 'metrics_to_log' and "
                "'metrics' differ. metrics.py prioritizes metrics_to_log; define "
                "one canonical list or make both identical."
            )

    return copy.deepcopy(primary)  # type: ignore[arg-type]


def _write_metric_list(cfg: dict, metrics: Sequence[Any]) -> None:
    canonical = copy.deepcopy(list(metrics))
    cfg["metrics_to_log"] = copy.deepcopy(canonical)
    cfg["metrics"] = copy.deepcopy(canonical)


def _maybe_sync_metrics_key(cfg: dict) -> None:
    """Mirror the canonical metric list into both supported configuration keys."""
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg).__name__}")

    if (
        ("metrics_to_log" not in cfg or cfg.get("metrics_to_log") is None)
        and ("metrics" not in cfg or cfg.get("metrics") is None)
    ):
        return

    canonical = _configured_metric_list(cfg)
    _write_metric_list(cfg, canonical)


def _strip_rank_metrics(cfg_in: dict) -> dict:
    """Return a copy with every metadata-dependent metric removed from both keys."""
    if not isinstance(cfg_in, dict):
        raise TypeError(f"cfg_in must be a dict, got {type(cfg_in).__name__}")

    cfg = copy.deepcopy(cfg_in)
    parsed = _parse_metric_specs(_configured_metric_list(cfg))
    kept = [
        (name, options)
        for name, options in parsed
        if not is_metadata_required_metric(name)
    ]
    _write_metric_list(cfg, _metric_specs_to_config(kept))
    return cfg


def filter_metrics_for_metadata(cfg: dict, has_metadata: bool) -> dict:
    """
    Return a config whose metric list is safe for the metadata availability.

    Metadata-present validation/test passes retain every requested metric.
    Metadata-free training passes remove all rank, repeated-attack, and LDD/Lm
    metrics from both ``metrics_to_log`` and ``metrics``.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg).__name__}")
    if not isinstance(has_metadata, bool):
        raise TypeError(
            f"has_metadata must be boolean, got {type(has_metadata).__name__}"
        )

    cfg_copy = copy.deepcopy(cfg)
    canonical = _configured_metric_list(cfg_copy)

    if has_metadata:
        _write_metric_list(cfg_copy, canonical)
        return cfg_copy

    return _strip_rank_metrics(cfg_copy)


def _needs_rank_metrics(cfg: dict) -> bool:
    """Return whether the canonical metric list requests metadata-dependent work."""
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg).__name__}")

    return any(
        is_metadata_required_metric(name)
        for name, _options in _parse_metric_specs(_configured_metric_list(cfg))
    )


# =============================================================================
# Rank-evaluation scheduling
# =============================================================================
def _get_validation_cfg(cfg: dict) -> dict:
    """Return one merged legacy/new validation configuration."""
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg).__name__}")

    legacy_raw = cfg.get("validation")
    if legacy_raw is None:
        legacy: dict[str, Any] = {}
    elif isinstance(legacy_raw, dict):
        legacy = legacy_raw
    else:
        raise TypeError(
            f"cfg['validation'] must be a mapping, got {type(legacy_raw).__name__}"
        )

    evaluation_raw = cfg.get("evaluation")
    if evaluation_raw is None:
        evaluation: dict[str, Any] = {}
    elif isinstance(evaluation_raw, dict):
        evaluation = evaluation_raw
    else:
        raise TypeError(
            f"cfg['evaluation'] must be a mapping, got {type(evaluation_raw).__name__}"
        )

    modern_raw = evaluation.get("validation")
    if modern_raw is None:
        modern: dict[str, Any] = {}
    elif isinstance(modern_raw, dict):
        modern = modern_raw
    else:
        raise TypeError(
            "cfg['evaluation']['validation'] must be a mapping, "
            f"got {type(modern_raw).__name__}"
        )

    for key in set(legacy).intersection(modern):
        legacy_value = legacy[key]
        modern_value = modern[key]
        if key == "rank_eval_interval":
            legacy_value = _strict_int(
                legacy_value,
                field="validation.rank_eval_interval",
            )
            modern_value = _strict_int(
                modern_value,
                field="evaluation.validation.rank_eval_interval",
            )
        if legacy_value != modern_value:
            raise ValueError(
                "Conflicting validation configuration for "
                f"{key!r}: validation={legacy[key]!r}, "
                f"evaluation.validation={modern[key]!r}"
            )

    merged = copy.deepcopy(legacy)
    merged.update(copy.deepcopy(modern))
    return merged


def _rank_eval_interval(cfg: dict) -> int:
    """Return a strict positive validation rank-evaluation interval."""
    validation = _get_validation_cfg(cfg)
    raw = validation.get("rank_eval_interval", 1)

    if isinstance(raw, (bool, np.bool_)):
        raise TypeError("validation.rank_eval_interval must be an integer, not bool")

    interval = _strict_int(raw, field="validation.rank_eval_interval")
    if interval < 1:
        raise ValueError(
            f"validation.rank_eval_interval must be >= 1, got {interval}"
        )

    return interval


def _monitor_is_rank_metric(monitor_metric: Optional[str]) -> bool:
    if monitor_metric is None:
        return False

    name = str(monitor_metric).strip()
    if not name:
        return False

    return is_metadata_required_metric(name)


__all__ = [
    "METADATA_REQUIRED_METRICS",
    "collect_logits_labels_meta",
    "filter_metrics_for_metadata",
    "is_metadata_required_metric",
    "metric_requires_metadata",
    "normalize_metadata",
    "read_attack_byte_attr",
    "unpack_batch",
    # Existing private helpers imported by epoch_utils.
    "_maybe_sync_metrics_key",
    "_monitor_is_rank_metric",
    "_needs_rank_metrics",
    "_rank_eval_interval",
    "_resolved_dir",
    "_strip_rank_metrics",
]
