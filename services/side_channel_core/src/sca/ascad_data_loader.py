# services/side_channel_core/src/sca/ascad_data_loader.py
"""
ASCAD data generation and HDF5-backed PyTorch loading for Side-Channel Analysis.

The module owns two related paths:

1. Configuration-specific dataset generation
   A source ASCAD-like HDF5 is sliced according to the active trace selection,
   relabeled for exactly one resolved attack byte and leakage model, and written
   as one complete profiling/attack HDF5 before downstream train/validation/test
   splitting.

2. Split-file loading
   Prepared train/validation/test HDF5 files are exposed as PyTorch datasets
   while preserving metadata required by key-rank, GE, SR, LDD, and Lm metrics.

Runtime-resolution contract
---------------------------
Search-space values such as::

    sca.attack_byte: [0, 3]
    sca.leakage_model: ["ID", "HW"]

are valid in the project configuration, but they are not valid inside one
executing trial. Before this module generates or loads data, the sweep/runtime
layer must resolve each field to one scalar. An unresolved list is rejected;
there is no fallback to AES byte 2 or to the first leakage-model entry.

Data provenance contract
------------------------
Generated HDF5 files preserve both:

- source provenance: source path/layout/attributes and source-label semantics
  when those semantics can be declared or inferred; and
- active provenance: the runtime attack byte, leakage model, class count,
  trace selection, desynchronization, normalization policy, and output counts.

The complete contract is stored in ``sca_data_contract_json`` with a SHA-256
fingerprint. ``data.<dataset>.show_contract`` controls terminal display only;
provenance storage is always enabled.

Trace normalization
-------------------
The default is ``none``. ``auto``, ``zscore``, and ``minmax`` remain available
as explicit configuration choices (fixed constants). ``per_trace_standard`` and
``per_trace_minmax`` scale each trace by its own statistics
(sklearn.preprocessing.scale / minmax_scale along the samples). ``fitted``
leaves the traces unscaled here: a scikit-learn scaler fitted on the training
traces is attached later (utils/trace_scaler.py, ASCADDataset.set_trace_scaler).

Author: Shraddha Mishra
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from sca.dataset_types import (
    label_target, profiling_split_enabled, profiling_split_problems, profiling_split_reproducible, random_profiling_keys, registered, require_attack_byte,
    resolved_variant, shared_key,
)
from sca.leakage_models import target_labels
from sca.leakage_models import (
    compute_leakage_label,
    get_num_classes,
    normalize_leakage_model,
)
from utils.logger import get_logger
from utils.path_utils import resolve_path
from utils.seed import set_seed
from utils.seal import check as check_seal, seal_h5_file

logger = get_logger("sca.data_loader")

DATA_CONTRACT_VERSION = "2.1"

# Contract validation remains per file/load.  Terminal publication is a
# run-level concern: generation and the train/validation/test dataset loaders
# all receive equivalent configuration and must not print the same contract.
_PUBLISHED_CONTRACT_KEYS: set[tuple[int, str, str, int, str]] = set()
_CONTRACT_PUBLICATION_LOCK = threading.Lock()


class SCADataContractError(ValueError):
    """Raised when source, generated, or runtime SCA data violates its contract."""

_GROUP_BY_SPLIT = {
    "profiling": "Profiling_traces",
    "attack": "Attack_traces",
}
_SUPPORTED_VARIANTS = frozenset({"fixed_key", "random_key", "variable_key"})
_SUPPORTED_NORMALIZATION_MODES = frozenset(
    {"none", "auto", "zscore", "minmax", "per_trace_standard", "per_trace_minmax", "fitted"}
)
_FITTED_STEPS = ("standard", "minmax")
# The keys each trace_normalization.mode reads (_normalization_from_data_cfg): eps guards the fixed std or range of
# zscore and minmax; fitted reads feature_range only when its steps include minmax. The panel shows the chosen
# mode's keys and no other (a data entry starts with the mode alone).
NORMALIZATION_KEYS: dict[str, tuple[str, ...]] = {
    "none": (), "auto": (), "per_trace_standard": (),
    "zscore": ("mean", "std", "eps"), "minmax": ("min", "max", "eps"),
    "per_trace_minmax": ("feature_range",), "fitted": ("steps", "feature_range"),
}
_DIRECT_DATASET_KEYS = frozenset(
    {
        "name",
        "alias",
        "database",
        "profiling_range",
        "attack_range",
        "validation_range",
        "test_range",
        "target_range",
        "target_points",
        "target_range_by_attack_byte",
        "trace_normalization",
        "show_contract",
        "require_contract",
        "trace_window_verified",
        "trace_window_verified_by_attack_byte",
        "contract_verification_samples",
        "contract_check_samples",
    }
)


# =============================================================================
# Strict configuration helpers
# =============================================================================
def _strict_bool(value: Any, *, field: str, default: Optional[bool] = None) -> bool:
    """Return a strict boolean and reject ambiguous truthy strings."""
    if value is None:
        if default is None:
            raise TypeError(f"{field} must be boolean")
        return bool(default)

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    ):
        if int(value) in (0, 1):
            return bool(int(value))

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False

    raise TypeError(f"{field} must be boolean, got {value!r}")


def _strict_int(
    value: Any,
    *,
    field: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Return an integer without silently truncating fractional values."""
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, got boolean {value!r}")

    if isinstance(value, (list, tuple, set, Mapping)):
        raise TypeError(
            f"{field} must be one resolved scalar value, got unresolved "
            f"{type(value).__name__}: {value!r}"
        )

    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise TypeError(
                f"{field} must be one resolved scalar value, got array shape "
                f"{value.shape}"
            )
        value = value.item()

    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise TypeError(f"{field} must be an integer, got {value!r}")
        result = int(numeric)
    else:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{field} must be an integer, got {value!r}") from exc

    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be <= {maximum}, got {result}")
    return result


def _strict_float(
    value: Any,
    *,
    field: str,
    minimum: Optional[float] = None,
    minimum_inclusive: bool = True,
) -> float:
    """Return one finite floating-point value."""
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric, got boolean {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {result!r}")

    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            relation = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{field} must be {relation} {minimum}, got {result}")
    return result


def _dataset_identity(entry: Mapping[str, Any], *, fallback: Optional[str] = None) -> str:
    """Return the dataset alias used for runtime selection and path templating."""
    raw = entry.get("alias") or entry.get("name") or fallback
    if raw is None or not str(raw).strip():
        raise KeyError("Each dataset entry must define 'name' or 'alias'")
    return str(raw).strip()


def _dataset_selection_hints(cfg: Mapping[str, Any]) -> list[str]:
    """Collect explicit active-dataset hints in priority order."""
    hints: list[str] = []
    runtime = cfg.get("runtime") or {}
    sweep_context = cfg.get("sweep_context") or {}

    candidates = [
        cfg.get("dataset_alias"),
        cfg.get("dataset_name"),
        runtime.get("dataset_alias") if isinstance(runtime, Mapping) else None,
        runtime.get("dataset_name") if isinstance(runtime, Mapping) else None,
        sweep_context.get("dataset_alias") if isinstance(sweep_context, Mapping) else None,
        sweep_context.get("data.name") if isinstance(sweep_context, Mapping) else None,
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text and text not in hints:
            hints.append(text)
    return hints


def resolve_active_data_config(cfg: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """
    Resolve the active dataset from list, direct-map, or keyed-map schemas.

    Multiple configured datasets require an explicit ``dataset_alias`` or
    equivalent runtime hint. The first entry is never chosen silently.
    """
    if not isinstance(cfg, Mapping):
        raise TypeError(f"cfg must be a mapping, got {type(cfg).__name__}")

    data = cfg.get("data")
    if data is None:
        raise KeyError("cfg['data'] is required")

    entries: list[tuple[str, MutableMapping[str, Any]]] = []

    if isinstance(data, list):
        if not data:
            raise ValueError("cfg['data'] is an empty list")
        for index, entry in enumerate(data):
            if not isinstance(entry, MutableMapping):
                raise TypeError(
                    f"cfg['data'][{index}] must be a mapping, "
                    f"got {type(entry).__name__}"
                )
            identity = _dataset_identity(entry, fallback=f"dataset_{index}")
            entries.append((identity, entry))

    elif isinstance(data, MutableMapping):
        if _DIRECT_DATASET_KEYS.intersection(data.keys()):
            identity = _dataset_identity(data, fallback="dataset")
            entries.append((identity, data))
        else:
            for key, entry in data.items():
                if not isinstance(entry, MutableMapping):
                    continue
                entry.setdefault("name", str(key))
                identity = _dataset_identity(entry, fallback=str(key))
                entries.append((identity, entry))

            if not entries:
                raise TypeError(
                    "cfg['data'] does not match a supported dataset schema"
                )
    else:
        raise TypeError(
            f"cfg['data'] must be a list or mapping, got {type(data).__name__}"
        )

    hints = _dataset_selection_hints(cfg)
    if hints:
        for hint in hints:
            matching = [
                entry
                for identity, entry in entries
                if hint
                in {
                    identity,
                    str(entry.get("name", "")).strip(),
                    str(entry.get("alias", "")).strip(),
                }
            ]
            if len(matching) == 1:
                return matching[0]
            if len(matching) > 1:
                raise ValueError(
                    f"Dataset selector {hint!r} matches multiple data entries"
                )

        available = [identity for identity, _entry in entries]
        raise KeyError(
            f"Active dataset selector(s) {hints!r} do not match configured "
            f"datasets {available!r}"
        )

    if len(entries) == 1:
        return entries[0][1]

    available = [identity for identity, _entry in entries]
    raise ValueError(
        "Multiple datasets are configured but no active dataset was resolved. "
        f"Set cfg['dataset_alias']; available={available!r}"
    )


# Backward-compatible private name used by older callers.
def _active_data_cfg(cfg: Mapping[str, Any]) -> MutableMapping[str, Any]:
    return resolve_active_data_config(cfg)


def _sca_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    sca = cfg.get("sca") or {}
    if not isinstance(sca, Mapping):
        raise TypeError("cfg['sca'] must be a mapping")
    return sca


def _get_attack_byte(cfg: Mapping[str, Any]) -> int:
    """Return one resolved AES attack byte; unresolved lists are an error."""
    sca = _sca_cfg(cfg)
    if "attack_byte" not in sca:
        raise KeyError("cfg['sca']['attack_byte'] is required")
    return _strict_int(
        sca["attack_byte"],
        field="sca.attack_byte",
        minimum=0,
        maximum=15,
    )


def _optional_runtime_attack_byte(cfg: Optional[Mapping[str, Any]]) -> Optional[int]:
    if not isinstance(cfg, Mapping):
        return None
    sca = _sca_cfg(cfg)
    if "attack_byte" not in sca:
        return None
    return _get_attack_byte(cfg)


def _get_leakage_model(cfg: Mapping[str, Any]) -> str:
    """Return one resolved leakage model; unresolved lists are an error."""
    sca = _sca_cfg(cfg)
    if "leakage_model" not in sca:
        raise KeyError("cfg['sca']['leakage_model'] is required")

    raw = sca["leakage_model"]
    if isinstance(raw, (list, tuple, set, Mapping)):
        raise TypeError(
            "sca.leakage_model must be one resolved scalar value, got "
            f"unresolved {type(raw).__name__}: {raw!r}"
        )
    if isinstance(raw, np.ndarray) and raw.shape != ():
        raise TypeError(
            "sca.leakage_model must be one resolved scalar value, got "
            f"array shape {raw.shape}"
        )
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    return normalize_leakage_model(raw)


def _optional_runtime_leakage_model(
    cfg: Optional[Mapping[str, Any]],
) -> Optional[str]:
    if not isinstance(cfg, Mapping):
        return None
    sca = _sca_cfg(cfg)
    if "leakage_model" not in sca:
        return None
    return _get_leakage_model(cfg)


def _get_variant(cfg: Mapping[str, Any]) -> str:
    """The run's key mode, given by its database type (sca/dataset_types.py), or ``unknown``."""
    raw = resolved_variant(cfg)
    if raw is None or not str(raw).strip():
        return "unknown"
    if isinstance(raw, (list, tuple, set, Mapping)):
        raise TypeError(
            "sca.variant must be one resolved scalar value, got "
            f"{type(raw).__name__}: {raw!r}"
        )
    normalized = str(raw).strip().lower()
    if normalized not in _SUPPORTED_VARIANTS:
        raise ValueError(
            f"sca.variant must be one of {sorted(_SUPPORTED_VARIANTS)}, "
            f"got {raw!r}"
        )
    return normalized


def _get_fixed_key_override(cfg: Mapping[str, Any]) -> Optional[np.ndarray]:
    sca = _sca_cfg(cfg)
    raw = sca.get("fixed_key_hex")
    if raw is None or not str(raw).strip():
        return None
    if isinstance(raw, (list, tuple, set, Mapping)):
        raise TypeError("sca.fixed_key_hex must be one scalar hexadecimal string")
    return _hex_to_key_bytes(str(raw))


def _require_metadata(cfg: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(cfg, Mapping):
        return True
    sca = _sca_cfg(cfg)
    return _strict_bool(
        sca.get("require_metadata", True),
        field="sca.require_metadata",
        default=True,
    )


def _show_contract(cfg: Mapping[str, Any]) -> bool:
    data_cfg = resolve_active_data_config(cfg)
    return _strict_bool(
        data_cfg.get("show_contract", False),
        field=f"data.{_dataset_identity(data_cfg)}.show_contract",
        default=False,
    )


def _require_contract(cfg: Optional[Mapping[str, Any]]) -> bool:
    """Require canonical active provenance on generated/split files by default."""
    if not isinstance(cfg, Mapping):
        return False
    data_cfg = resolve_active_data_config(cfg)
    return _strict_bool(
        data_cfg.get("require_contract", True),
        field=f"data.{_dataset_identity(data_cfg)}.require_contract",
        default=True,
    )


def _target_points_for_attack_byte(
    data_cfg: Mapping[str, Any],
    *,
    attack_byte: int,
) -> Sequence[int]:
    """Resolve a byte-specific selection when configured, else the generic one."""
    per_byte = data_cfg.get("target_range_by_attack_byte")
    if per_byte is not None:
        if not isinstance(per_byte, Mapping):
            raise TypeError("data.target_range_by_attack_byte must be a mapping")
        for key in (attack_byte, str(attack_byte), f"0x{attack_byte:02X}"):
            if key in per_byte:
                selected = per_byte[key]
                if not isinstance(selected, (list, tuple)):
                    raise TypeError(
                        "data.target_range_by_attack_byte entries must be "
                        "list/tuple selections"
                    )
                return selected

    generic = data_cfg.get("target_range", data_cfg.get("target_points"))
    if generic is None:
        raise KeyError(
            "Active dataset must define target_range/target_points or a matching "
            "target_range_by_attack_byte entry"
        )
    if not isinstance(generic, (list, tuple)):
        raise TypeError("data.target_range/target_points must be a list or tuple")
    return generic


def _contract_verification_sample_size(data_cfg: Mapping[str, Any]) -> int:
    """Resolve the canonical deterministic trace/metadata verification sample."""
    canonical = data_cfg.get("contract_verification_samples")
    legacy = data_cfg.get("contract_check_samples")
    if canonical is not None and legacy is not None:
        canonical_value = _strict_int(
            canonical,
            field="data.contract_verification_samples",
            minimum=1,
        )
        legacy_value = _strict_int(
            legacy,
            field="data.contract_check_samples",
            minimum=1,
        )
        if canonical_value != legacy_value:
            raise ValueError(
                "data.contract_verification_samples and deprecated "
                "data.contract_check_samples disagree"
            )
        return canonical_value
    raw = canonical if canonical is not None else legacy
    return _strict_int(
        512 if raw is None else raw,
        field=(
            f"data.{_dataset_identity(data_cfg)}."
            "contract_verification_samples"
        ),
        minimum=1,
    )


# =============================================================================
# General helpers
# =============================================================================
def _bytes_to_hex16(value: np.ndarray) -> str:
    bytes_array = np.asarray(value, dtype=np.uint8).reshape(16)
    return bytes(bytes_array.tolist()).hex()


def _hex_to_key_bytes(hex_str: str) -> np.ndarray:
    normalized = str(hex_str).strip().lower().replace("0x", "").replace(" ", "")
    if len(normalized) != 32:
        raise ValueError(
            "fixed_key_hex must contain exactly 32 hexadecimal characters "
            f"(16 bytes), got length={len(normalized)}"
        )
    try:
        raw = bytes.fromhex(normalized)
    except ValueError as exc:
        raise ValueError("fixed_key_hex contains non-hexadecimal characters") from exc
    return np.frombuffer(raw, dtype=np.uint8).copy()


def _label_dtype_for_leakage(leakage_model: str) -> np.dtype:
    get_num_classes(leakage_model)
    return np.dtype(np.uint8)


def check_file_exists(filename: str, cfg: Optional[Mapping[str, Any]] = None) -> str:
    """Resolve and validate one dataset path without hiding resolution errors."""
    requested = str(filename)
    candidate = requested

    if cfg is not None and not Path(requested).is_absolute():
        candidate = str(resolve_path(requested, cfg))

    path = Path(candidate).resolve()
    if registered(path):
        # A registered database: fetched (or made from its downloads) once when it is not here
        # yet, and checked each time a run opens it: its published SHA-256, or for a converted
        # file the registered sources and converter version. A file that no longer matches is
        # moved aside and fetched again, or refused (master.yaml data_fetch.on_mismatch); any other
        # failure raises DataFetchError ("DATA FAILED: ...") (utils/dataset_fetch.py). The run's
        # device.yaml records it.
        from utils.dataset_fetch import ensure, verify_or_refetch
        from utils.device_record import record_database

        if not path.is_file():
            ensure(path)
        record = verify_or_refetch(path)
        if cfg is not None:
            record_database(cfg, {**record, "registered": True})
        return str(path)
    if not path.is_file():
        project_root = None
        if isinstance(cfg, Mapping):
            project = cfg.get("project") or {}
            if isinstance(project, Mapping):
                project_root = project.get("root")

        raise FileNotFoundError(
            "Dataset file not found.\n"
            f"  requested    : {requested}\n"
            f"  resolved     : {path}\n"
            f"  cwd          : {os.getcwd()}\n"
            f"  project.root : {project_root}\n"
            f"  ENV          : {os.environ.get('ENV')}\n"
            f"  PROJECTS_ROOT: {os.environ.get('PROJECTS_ROOT')}\n"
            f"  SCC_ROOT     : {os.environ.get('SCC_ROOT')}"
        )
    return str(path)


def sca_collate_fn(batch: Sequence[Any]) -> Any:
    """Collate traces while preserving metadata as ``list[dict]``."""
    if not batch:
        return batch

    first = batch[0]
    if not isinstance(first, (tuple, list)):
        raise TypeError(
            f"Expected batch elements as tuple/list, got {type(first).__name__}"
        )

    if len(first) == 2:
        traces, labels = zip(*batch)
        return torch.stack(traces, dim=0), torch.stack(labels, dim=0)

    if len(first) == 3:
        traces, labels, metadata = zip(*batch)
        return (
            torch.stack(traces, dim=0),
            torch.stack(labels, dim=0),
            list(metadata),
        )

    raise ValueError(
        f"Unexpected batch tuple length={len(first)}; expected 2 or 3"
    )


def compute_labels_from_meta(
    plaintexts: np.ndarray,
    keys: np.ndarray,
    *,
    byte_idx: int,
    leakage_model: str,
) -> np.ndarray:
    """Compute one leakage label per trace from plaintext/key metadata."""
    attack_byte = _strict_int(
        byte_idx,
        field="byte_idx",
        minimum=0,
        maximum=15,
    )
    model = normalize_leakage_model(leakage_model)

    plaintext_array = _validated_byte_matrix(plaintexts, field="plaintexts")
    key_array = _validated_byte_matrix(keys, field="keys")
    if plaintext_array.shape[0] != key_array.shape[0]:
        raise ValueError(
            "plaintexts and keys must contain the same number of rows, got "
            f"{plaintext_array.shape[0]} and {key_array.shape[0]}"
        )

    labels = compute_leakage_label(
        plaintext_array[:, attack_byte],
        key_array[:, attack_byte],
        model,
    )
    normalized = np.asarray(labels, dtype=np.uint8)
    expected_shape = (plaintext_array.shape[0],)
    if normalized.shape != expected_shape:
        raise ValueError(
            f"Leakage-label computation returned {normalized.shape}; "
            f"expected {expected_shape}"
        )
    return normalized


def labels_for_target(
    rows: Any,
    *,
    attack_byte: int,
    leakage_model: str,
    target: str = "sbox",
) -> np.ndarray:
    """Labels of metadata rows under the database type's target (sca/dataset_types.py): the
    first-round S-box output as before for "sbox", else sca/leakage_models.target_labels."""
    if target == "sbox":
        return compute_labels_from_meta(rows["plaintext"], rows["key"], byte_idx=attack_byte,
                                        leakage_model=leakage_model)
    return target_labels(rows, target=target, byte=attack_byte, leakage_model=leakage_model)


def _file_label_target(handle: Any) -> str:
    """The label target a generated file was made with; files without the attribute are sbox."""
    raw = handle.file.attrs.get("sca_label_target", "sbox")
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _validated_byte_matrix(value: Any, *, field: str) -> np.ndarray:
    """Return an ``[N,16]`` uint8 matrix without lossy coercion."""
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 16:
        raise ValueError(f"{field} must have shape [N,16], got {array.shape}")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError(f"{field} must contain numeric byte values, got {array.dtype}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError(f"{field} contains non-finite or fractional values")
    if array.size and (np.any(array < 0) or np.any(array > 255)):
        raise ValueError(f"{field} contains values outside the byte range [0,255]")
    return np.asarray(array, dtype=np.uint8)


# =============================================================================
# Source layout, metadata, and provenance helpers
# =============================================================================
def _get_raw_layout(input_file: h5py.File) -> str:
    if "Profiling_traces" in input_file and "Attack_traces" in input_file:
        return "grouped_both"
    if "Profiling_traces" in input_file:
        return "grouped_profiling_only"
    if "traces" in input_file and "metadata" in input_file:
        return "flat"
    return "unknown"


def _read_group_views(
    input_file: h5py.File,
    group_name: str,
) -> tuple[h5py.Dataset, h5py.Dataset, Optional[h5py.Dataset]]:
    group = input_file[group_name]
    if "traces" not in group or "metadata" not in group:
        raise KeyError(
            f"Missing {group_name}/traces or {group_name}/metadata"
        )
    labels = group.get("labels")
    return group["traces"], group["metadata"], labels


def _metadata_fields(metadata: h5py.Dataset) -> Mapping[str, Any]:
    fields = getattr(metadata.dtype, "fields", None)
    if fields is None:
        raise TypeError(
            "metadata must be an HDF5 structured array with named fields"
        )
    return fields


def _read_h5_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read arbitrary rows without relying on h5py's sorted-index restriction."""
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if indices.size == 0:
        return np.empty((0,), dtype=dataset.dtype)

    strictly_increasing = bool(
        indices.size == 1 or np.all(indices[1:] > indices[:-1])
    )
    if strictly_increasing:
        return np.asarray(dataset[indices])

    rows = np.empty((indices.size,), dtype=dataset.dtype)
    for output_index, source_index in enumerate(indices.tolist()):
        rows[output_index] = dataset[int(source_index)]
    return rows


def _infer_variant_from_keys(
    keys_field: Any,
    *,
    sample_n: int = 512,
    seed: int = 12345,
) -> str:
    """Infer fixed versus varying key behavior from a deterministic sample."""
    if keys_field is None:
        return "unknown"
    try:
        count = int(keys_field.shape[0])
    except Exception:
        return "unknown"
    if count <= 0:
        return "unknown"

    rng = np.random.default_rng(int(seed))
    indices = rng.choice(count, size=min(sample_n, count), replace=False)
    indices.sort()

    try:
        first_key = np.asarray(keys_field[0], dtype=np.uint8)
        sampled_keys = np.asarray(keys_field[indices], dtype=np.uint8)
    except Exception:
        return "unknown"

    if first_key.shape != (16,):
        return "unknown"
    if sampled_keys.ndim != 2 or sampled_keys.shape[1] != 16:
        return "unknown"

    return "fixed_key" if np.all(sampled_keys == first_key) else "random_key"


def _validate_selected_fixed_key(
    metadata: h5py.Dataset,
    indices: np.ndarray,
    *,
    expected_key: Optional[np.ndarray],
    where: str,
    chunk_size: int,
) -> np.ndarray:
    """Validate every selected fixed-key row and return the observed key."""
    fields = _metadata_fields(metadata)
    if "key" not in fields:
        if expected_key is None:
            raise KeyError(f"{where}: fixed-key metadata has no 'key' field")
        return np.asarray(expected_key, dtype=np.uint8)

    reference = (
        np.asarray(expected_key, dtype=np.uint8)
        if expected_key is not None
        else None
    )
    for start in range(0, int(indices.size), int(chunk_size)):
        rows = _read_h5_rows(metadata, indices[start : start + int(chunk_size)])
        keys = _validated_byte_matrix(rows["key"], field=f"{where}.metadata.key")
        if reference is None:
            reference = np.array(keys[0], copy=True)
        if not np.all(keys == reference):
            reason = (
                "does not match sca.fixed_key_hex"
                if expected_key is not None
                else "is not constant"
            )
            raise SCADataContractError(
                f"{where}: selected metadata key {reason}"
            )

    if reference is None:
        raise SCADataContractError(f"{where}: no selected fixed-key rows")
    return reference


def _json_safe(value: Any) -> Any:
    """Convert HDF5/NumPy attribute values into JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _snapshot_h5_attrs(attributes: h5py.AttributeManager) -> dict[str, Any]:
    return {
        str(key): _json_safe(attributes[key])
        for key in sorted(attributes.keys())
    }


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _first_attr(
    attributes: h5py.AttributeManager,
    names: Sequence[str],
) -> Any:
    for name in names:
        if name in attributes:
            return _decode_attr(attributes[name])
    return None


def _declared_source_attack_byte(attributes: h5py.AttributeManager) -> Optional[int]:
    raw = _first_attr(
        attributes,
        (
            "sca_active_attack_byte",
            "sca_attack_byte",
            "attack_byte",
        ),
    )
    if raw is None:
        return None
    return _strict_int(
        raw,
        field="source HDF5 attack-byte attribute",
        minimum=0,
        maximum=15,
    )


def _declared_source_leakage_model(
    attributes: h5py.AttributeManager,
) -> Optional[str]:
    raw = _first_attr(
        attributes,
        (
            "sca_active_leakage_model",
            "sca_leakage_model",
            "leakage_model",
        ),
    )
    if raw is None:
        return None
    return normalize_leakage_model(raw)


def _infer_source_label_matches(
    labels: Optional[h5py.Dataset],
    metadata: h5py.Dataset,
    *,
    seed: int,
    sample_n: int = 512,
) -> list[dict[str, Any]]:
    """Infer source label semantics by matching labels against all bytes/models."""
    if labels is None:
        return []

    fields = _metadata_fields(metadata)
    if "plaintext" not in fields or "key" not in fields:
        return []

    count = min(int(labels.shape[0]), int(metadata.shape[0]))
    if count <= 0:
        return []

    rng = np.random.default_rng(int(seed))
    indices = rng.choice(count, size=min(sample_n, count), replace=False)
    indices.sort()
    indices = indices.astype(np.int64)

    sampled_labels = np.asarray(_read_h5_rows(labels, indices), dtype=np.int64).reshape(-1)
    rows = _read_h5_rows(metadata, indices)
    plaintexts = _validated_byte_matrix(
        rows["plaintext"], field="source metadata plaintext"
    )
    keys = _validated_byte_matrix(rows["key"], field="source metadata key")

    if plaintexts.ndim != 2 or plaintexts.shape[1] != 16:
        return []
    if keys.ndim != 2 or keys.shape[1] != 16:
        return []

    matches: list[dict[str, Any]] = []
    for leakage_model in ("ID", "HW", "MSB"):
        class_count = get_num_classes(leakage_model)
        if np.any(sampled_labels < 0) or np.any(sampled_labels >= class_count):
            continue

        for attack_byte in range(16):
            expected = compute_labels_from_meta(
                plaintexts,
                keys,
                byte_idx=attack_byte,
                leakage_model=leakage_model,
            ).astype(np.int64, copy=False)
            if np.array_equal(sampled_labels, expected):
                matches.append(
                    {
                        "attack_byte": attack_byte,
                        "leakage_model": leakage_model,
                        "checked": int(sampled_labels.size),
                    }
                )
    return matches


def _resolve_source_label_contract(
    input_file: h5py.File,
    *,
    profiling_labels: Optional[h5py.Dataset],
    profiling_metadata: h5py.Dataset,
    attack_labels: Optional[h5py.Dataset],
    attack_metadata: h5py.Dataset,
    seed: int,
) -> dict[str, Any]:
    """
    Resolve source-label semantics from both profiling and attack groups.

    Declared source attributes are preserved, but when source labels are
    available they must agree with independently recomputed labels for both
    groups. This prevents a stale source attribute from being treated as truth.
    """
    declared_byte = _declared_source_attack_byte(input_file.attrs)
    declared_model = _declared_source_leakage_model(input_file.attrs)

    profiling_matches = _infer_source_label_matches(
        profiling_labels,
        profiling_metadata,
        seed=seed,
    )
    attack_matches = _infer_source_label_matches(
        attack_labels,
        attack_metadata,
        seed=seed + 1,
    )

    def pair_set(matches: Sequence[Mapping[str, Any]]) -> set[tuple[int, str]]:
        return {
            (int(item["attack_byte"]), str(item["leakage_model"]))
            for item in matches
        }

    profiling_pairs = pair_set(profiling_matches)
    attack_pairs = pair_set(attack_matches)

    if profiling_pairs and attack_pairs:
        compatible_pairs = profiling_pairs.intersection(attack_pairs)
        if not compatible_pairs:
            raise ValueError(
                "Source Profiling_traces and Attack_traces labels imply "
                "different attack-byte/leakage-model semantics"
            )
    else:
        compatible_pairs = profiling_pairs or attack_pairs

    matches = [
        {
            "attack_byte": attack_byte,
            "leakage_model": leakage_model,
        }
        for attack_byte, leakage_model in sorted(compatible_pairs)
    ]

    if declared_byte is not None and declared_model is not None and compatible_pairs:
        if (declared_byte, declared_model) not in compatible_pairs:
            raise ValueError(
                "Source HDF5 label attributes disagree with labels recomputed "
                "from source metadata"
            )

    resolved_byte = declared_byte
    resolved_model = declared_model
    status = "unknown"

    if declared_byte is not None and declared_model is not None:
        status = (
            "attribute_verified"
            if compatible_pairs
            else "attribute_unverified"
        )
    elif len(matches) == 1:
        resolved_byte = int(matches[0]["attack_byte"])
        resolved_model = str(matches[0]["leakage_model"])
        status = "inferred"
    elif matches:
        compatible = [
            item
            for item in matches
            if (declared_byte is None or item["attack_byte"] == declared_byte)
            and (declared_model is None or item["leakage_model"] == declared_model)
        ]
        if len(compatible) == 1:
            resolved_byte = int(compatible[0]["attack_byte"])
            resolved_model = str(compatible[0]["leakage_model"])
            status = "attribute_completed_by_inference"
        else:
            status = "ambiguous"
    elif declared_byte is not None or declared_model is not None:
        status = "attribute_partial_unverified"

    return {
        "attack_byte": resolved_byte,
        "leakage_model": resolved_model,
        "status": status,
        "declared_attack_byte": declared_byte,
        "declared_leakage_model": declared_model,
        "inferred_matches": matches,
        "profiling_inferred_matches": profiling_matches,
        "attack_inferred_matches": attack_matches,
    }


# =============================================================================
# Trace-selection and index helpers
# =============================================================================
@dataclass(frozen=True)
class TraceSelection:
    kind: str
    configured: tuple[int, ...]
    start: int
    end_exclusive: int
    relative_points: Optional[np.ndarray]
    output_length: int


def _resolve_trace_selection(target_points: Sequence[Any]) -> TraceSelection:
    if not isinstance(target_points, (list, tuple)):
        raise TypeError("target_range/target_points must be a list or tuple")
    if len(target_points) < 2:
        raise ValueError("target_range/target_points must contain at least two integers")

    configured = tuple(
        _strict_int(value, field=f"target_points[{index}]", minimum=0)
        for index, value in enumerate(target_points)
    )

    if len(configured) == 2:
        start, end = configured
        if end <= start:
            raise ValueError(
                f"target_range end must be greater than start, got {configured!r}"
            )
        return TraceSelection(
            kind="window",
            configured=configured,
            start=start,
            end_exclusive=end,
            relative_points=None,
            output_length=end - start,
        )

    if len(set(configured)) != len(configured):
        raise ValueError("target_points contains duplicate sample indices")

    start = min(configured)
    end = max(configured) + 1
    relative_points = np.asarray(
        [point - start for point in configured],
        dtype=np.int64,
    )
    return TraceSelection(
        kind="points",
        configured=configured,
        start=start,
        end_exclusive=end,
        relative_points=relative_points,
        output_length=len(configured),
    )


def _normalize_indices(
    values: Sequence[Any],
    *,
    field: str,
    available: int,
) -> np.ndarray:
    try:
        raw_values = list(values)
    except TypeError as exc:
        raise TypeError(f"{field} must be an iterable of indices") from exc
    if not raw_values:
        raise ValueError(f"{field} must not be empty")

    indices = np.asarray(
        [
            _strict_int(
                value,
                field=f"{field}[{index}]",
                minimum=0,
                maximum=available - 1,
            )
            for index, value in enumerate(raw_values)
        ],
        dtype=np.int64,
    )
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{field} contains duplicate trace indices")
    return indices


def _validate_trace_bounds(
    traces: h5py.Dataset,
    selection: TraceSelection,
    *,
    desync: int,
    where: str,
) -> None:
    if traces.ndim != 2:
        raise ValueError(f"{where}/traces must be two-dimensional, got {traces.shape}")
    trace_length = int(traces.shape[1])
    required_end = selection.end_exclusive + desync
    if required_end > trace_length:
        raise ValueError(
            f"{where}: target selection requires samples through {required_end - 1}, "
            f"but source trace length is {trace_length}"
        )


def _read_trace_chunk(
    traces: h5py.Dataset,
    indices: np.ndarray,
    *,
    selection: TraceSelection,
    desync: int,
    rng: Optional[np.random.Generator] = None,
    offsets: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read one trace chunk and return the exact applied desynchronization offsets.

    Passing ``offsets`` is used by the post-write trace-selection verifier so
    the verifier reconstructs the persisted trace from immutable provenance
    rather than drawing new random offsets.
    """
    count = int(indices.size)
    result = np.empty(
        (count, selection.output_length),
        dtype=traces.dtype,
    )

    if offsets is None:
        if desync > 0:
            if rng is None:
                raise ValueError("rng is required when desync > 0")
            applied_offsets = rng.integers(
                0,
                desync + 1,
                size=count,
                dtype=np.int32,
            )
        else:
            applied_offsets = np.zeros(count, dtype=np.int32)
    else:
        applied_offsets = np.asarray(offsets, dtype=np.int32)
        if applied_offsets.shape != (count,):
            raise ValueError(
                f"offsets must have shape {(count,)}, got {applied_offsets.shape}"
            )
        if np.any(applied_offsets < 0) or np.any(applied_offsets > desync):
            raise ValueError(
                f"offsets must be in [0, {desync}], got "
                f"min={int(applied_offsets.min())} max={int(applied_offsets.max())}"
            )

    strictly_increasing = bool(
        count == 1 or np.all(indices[1:] > indices[:-1])
    )

    if np.all(applied_offsets == 0) and strictly_increasing:
        block = np.asarray(
            traces[indices, selection.start : selection.end_exclusive]
        )
        if selection.relative_points is not None:
            block = block[:, selection.relative_points]
        if block.shape != result.shape:
            raise ValueError(
                f"Trace chunk shape mismatch: got {block.shape}, expected {result.shape}"
            )
        result[...] = block
        return result, applied_offsets

    for output_index, source_index in enumerate(indices.tolist()):
        offset = int(applied_offsets[output_index])
        start = selection.start + offset
        end = selection.end_exclusive + offset
        segment = np.asarray(traces[int(source_index), start:end])
        if selection.relative_points is not None:
            segment = segment[selection.relative_points]
        if segment.shape != (selection.output_length,):
            raise ValueError(
                f"Trace {source_index} produced shape {segment.shape}; "
                f"expected {(selection.output_length,)}"
            )
        result[output_index] = segment
    return result, applied_offsets


# =============================================================================
# Metadata output helpers
# =============================================================================
def _output_metadata_dtype(
    source_dtype: np.dtype,
    *,
    add_key: bool,
) -> np.dtype:
    fields = source_dtype.fields
    if fields is None:
        raise TypeError("metadata must use a structured NumPy dtype")
    if not add_key or "key" in fields:
        return source_dtype

    descriptors: list[Any] = []
    for name in source_dtype.names or ():
        field_dtype = source_dtype.fields[name][0]
        descriptors.append((name, field_dtype))
    descriptors.append(("key", np.uint8, (16,)))
    return np.dtype(descriptors)


def _prepare_metadata_rows(
    rows: np.ndarray,
    *,
    output_dtype: np.dtype,
    fixed_key_override: Optional[np.ndarray],
) -> np.ndarray:
    if rows.dtype == output_dtype:
        output = np.array(rows, copy=True)
    else:
        output = np.empty(rows.shape, dtype=output_dtype)
        for name in rows.dtype.names or ():
            output[name] = rows[name]

    source_fields = rows.dtype.fields or {}
    if fixed_key_override is not None and "key" not in source_fields:
        if "key" not in (output.dtype.fields or {}):
            raise KeyError("Output metadata dtype does not contain a key field")
        output["key"] = np.tile(
            fixed_key_override.reshape(1, 16),
            (int(output.shape[0]), 1),
        )
    return output


# =============================================================================
# Data-contract construction, persistence, and display
# =============================================================================
def _normalization_from_data_cfg(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = data_cfg.get("trace_normalization")
    if raw is None:
        return {"mode": "none", "eps": 1.0e-8}
    if not isinstance(raw, Mapping):
        raise TypeError("trace_normalization must be a mapping")

    mode = str(raw.get("mode", "none")).strip().lower()
    if mode not in _SUPPORTED_NORMALIZATION_MODES:
        raise ValueError(
            "trace_normalization.mode must be one of "
            f"{sorted(_SUPPORTED_NORMALIZATION_MODES)}, got {mode!r}"
        )

    output: dict[str, Any] = {
        "mode": mode,
        "eps": _strict_float(
            raw.get("eps", 1.0e-8),
            field="trace_normalization.eps",
            minimum=0.0,
            minimum_inclusive=False,
        ),
    }

    if mode == "zscore":
        if "mean" not in raw or "std" not in raw:
            raise KeyError("zscore normalization requires explicit mean and std")
        output["mean"] = _strict_float(
            raw["mean"], field="trace_normalization.mean"
        )
        output["std"] = _strict_float(
            raw["std"],
            field="trace_normalization.std",
            minimum=0.0,
            minimum_inclusive=False,
        )
        if output["std"] < output["eps"]:
            raise ValueError("trace_normalization.std is too close to zero")

    elif mode in ("per_trace_minmax", "fitted"):
        if mode == "fitted":
            steps = raw.get("steps")
            if (not isinstance(steps, (list, tuple)) or not steps
                    or any(str(step) not in _FITTED_STEPS for step in steps)
                    or len(set(map(str, steps))) != len(steps)):
                raise ValueError(
                    "trace_normalization.steps (mode fitted) must list scikit-learn scalers to chain, "
                    f"each once, from {list(_FITTED_STEPS)}, e.g. [standard, minmax]; got {steps!r}"
                )
            output["steps"] = [str(step) for step in steps]
        if mode == "per_trace_minmax" or "minmax" in output.get("steps", []):
            feature_range = raw.get("feature_range", [0.0, 1.0])
            if not isinstance(feature_range, (list, tuple)) or len(feature_range) != 2:
                raise ValueError(f"trace_normalization.feature_range must be [low, high], got {feature_range!r}")
            low = _strict_float(feature_range[0], field="trace_normalization.feature_range[0]")
            high = _strict_float(feature_range[1], field="trace_normalization.feature_range[1]")
            if high <= low:
                raise ValueError("trace_normalization.feature_range high must be greater than low")
            output["feature_range"] = [low, high]

    elif mode == "minmax":
        if "min" not in raw or "max" not in raw:
            raise KeyError("minmax normalization requires explicit min and max")
        output["min"] = _strict_float(
            raw["min"], field="trace_normalization.min"
        )
        output["max"] = _strict_float(
            raw["max"], field="trace_normalization.max"
        )
        if output["max"] <= output["min"]:
            raise ValueError("trace_normalization.max must be greater than min")
        if output["max"] - output["min"] < output["eps"]:
            raise ValueError("trace_normalization max-min is too close to zero")

    return output


def _get_trace_normalization_cfg(
    cfg: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve normalization from the active dataset; default is ``none``."""
    if not isinstance(cfg, Mapping):
        return {"mode": "none", "eps": 1.0e-8}
    return _normalization_from_data_cfg(resolve_active_data_config(cfg))


def _trace_window_verification(
    data_cfg: Mapping[str, Any],
    *,
    attack_byte: int,
) -> str:
    """Return the configured scientific-support declaration for this byte."""
    raw = None
    field = f"data.{_dataset_identity(data_cfg)}.trace_window_verified"
    per_byte = data_cfg.get("trace_window_verified_by_attack_byte")
    if per_byte is not None:
        if not isinstance(per_byte, Mapping):
            raise TypeError(
                "data.trace_window_verified_by_attack_byte must be a mapping"
            )
        for key in (attack_byte, str(attack_byte), f"0x{attack_byte:02X}"):
            if key in per_byte:
                raw = per_byte[key]
                field = (
                    f"data.{_dataset_identity(data_cfg)}."
                    f"trace_window_verified_by_attack_byte[{key!r}]"
                )
                break

    if raw is None and "trace_window_verified" in data_cfg:
        raise SCADataContractError(
            "data.trace_window_verified is ambiguous for multi-byte runs; use "
            "data.trace_window_verified_by_attack_byte with an explicit entry "
            f"for byte {attack_byte}"
        )
    if raw is None:
        return "not_declared_for_attack_byte"
    return "verified" if _strict_bool(raw, field=field) else "not_verified"


def _require_scientifically_valid_trace_window(
    data_cfg: Mapping[str, Any],
    *,
    attack_byte: int,
    selection: TraceSelection,
    profiling_traces: h5py.Dataset,
    attack_traces: h5py.Dataset,
    profiling_desync: int,
    attack_desync: int,
) -> str:
    """Fail closed unless the selected window is full-source or byte-verified."""
    uses_full_source = bool(
        selection.kind == "window"
        and selection.start == 0
        and selection.end_exclusive == int(profiling_traces.shape[1])
        and selection.end_exclusive == int(attack_traces.shape[1])
        and profiling_desync == 0
        and attack_desync == 0
    )
    if uses_full_source:
        return "full_source"

    status = _trace_window_verification(data_cfg, attack_byte=attack_byte)
    if status != "verified":
        raise SCADataContractError(
            "The selected trace window is not scientifically verified for "
            f"attack byte {attack_byte}. Configure "
            "data.<dataset>.trace_window_verified_by_attack_byte with an "
            f"explicit true entry for byte {attack_byte}, or select the complete "
            "source trace. Do not mark it true until the window is known to "
            "contain leakage for that byte."
        )
    return status


def _build_data_contract(
    *,
    source_path: Path,
    source_layout: str,
    source_attrs: Mapping[str, Any],
    source_label_contract: Mapping[str, Any],
    profiling_traces: h5py.Dataset,
    attack_traces: h5py.Dataset,
    active_data_cfg: Mapping[str, Any],
    variant: str,
    attack_byte: int,
    leakage_model: str,
    selection: TraceSelection,
    profiling_count: int,
    attack_count: int,
    profiling_desync: int,
    attack_desync: int,
    output_path: Path,
    trace_window_verification: str,
) -> dict[str, Any]:
    dataset_name = str(active_data_cfg.get("name") or "unknown").strip()
    dataset_alias = _dataset_identity(active_data_cfg, fallback=dataset_name)
    normalization = _normalization_from_data_cfg(active_data_cfg)

    source_byte = source_label_contract.get("attack_byte")
    source_model = source_label_contract.get("leakage_model")
    source_semantics_differ = bool(
        source_byte is not None
        and source_model is not None
        and (
            int(source_byte) != attack_byte
            or str(source_model).upper() != leakage_model
        )
    )

    full_profiling_window = (
        selection.start == 0
        and selection.end_exclusive == int(profiling_traces.shape[1])
        and selection.relative_points is None
        and profiling_desync == 0
    )
    full_attack_window = (
        selection.start == 0
        and selection.end_exclusive == int(attack_traces.shape[1])
        and selection.relative_points is None
        and attack_desync == 0
    )

    return {
        "version": DATA_CONTRACT_VERSION,
        "source": {
            "path": str(source_path.resolve()),
            "file_name": source_path.name,
            "layout": source_layout,
            "attributes": dict(source_attrs),
            "profiling_trace_count": int(profiling_traces.shape[0]),
            "attack_trace_count": int(attack_traces.shape[0]),
            "profiling_trace_length": int(profiling_traces.shape[1]),
            "attack_trace_length": int(attack_traces.shape[1]),
            "label_contract": dict(source_label_contract),
        },
        "active": {
            "dataset_name": dataset_name,
            "dataset_alias": dataset_alias,
            "variant": variant,
            "attack_byte": attack_byte,
            "leakage_model": leakage_model,
            "num_classes": get_num_classes(leakage_model),
            "labels_recomputed": True,
            "source_label_semantics_differ": source_semantics_differ,
            "profiling_count": profiling_count,
            "attack_count": attack_count,
            "profiling_desync": profiling_desync,
            "attack_desync": attack_desync,
            "trace_selection": {
                "kind": selection.kind,
                "configured": list(selection.configured),
                "start": selection.start,
                "end_exclusive": selection.end_exclusive,
                "output_length": selection.output_length,
                "relative_to_source_traces": True,
                "uses_full_profiling_window": full_profiling_window,
                "uses_full_attack_window": full_attack_window,
                "verification_for_attack_byte": trace_window_verification,
            },
            "trace_normalization": normalization,
        },
        "verification": {
            "status": "pending",
            "performed_before_publish": False,
            "label_integrity_verified": False,
            "trace_selection_verified": False,
            "metadata_mapping_verified": False,
        },
        "output": {
            "path": str(output_path.resolve()),
            "layout": "grouped_both",
            "groups": ["Profiling_traces", "Attack_traces"],
            "provenance_datasets": ["source_indices", "desync_offsets"],
        },
    }


def _write_data_contract_attrs(
    output_file: h5py.File,
    contract: Mapping[str, Any],
    *,
    fixed_key_override: Optional[np.ndarray],
) -> None:
    serialized = json.dumps(
        _json_safe(contract),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    active = contract["active"]
    source = contract["source"]
    trace_selection = active["trace_selection"]
    source_label = source["label_contract"]

    output_file.attrs["sca_contract_version"] = DATA_CONTRACT_VERSION
    output_file.attrs["sca_data_contract_json"] = serialized
    output_file.attrs["sca_data_contract_sha256"] = fingerprint

    output_file.attrs["sca_dataset_name"] = str(active["dataset_name"])
    output_file.attrs["sca_dataset_alias"] = str(active["dataset_alias"])
    output_file.attrs["sca_variant"] = str(active["variant"])

    # Active fields plus compatibility names used by current downstream code.
    output_file.attrs["sca_active_attack_byte"] = int(active["attack_byte"])
    output_file.attrs["sca_attack_byte"] = int(active["attack_byte"])
    output_file.attrs["sca_active_leakage_model"] = str(active["leakage_model"])
    output_file.attrs["sca_leakage_model"] = str(active["leakage_model"])
    output_file.attrs["sca_num_classes"] = int(active["num_classes"])
    output_file.attrs["sca_labels_recomputed"] = True

    output_file.attrs["sca_source_path"] = str(source["path"])
    output_file.attrs["sca_source_layout"] = str(source["layout"])
    output_file.attrs["sca_source_label_contract_status"] = str(
        source_label.get("status", "unknown")
    )
    output_file.attrs["sca_source_attack_byte"] = int(
        source_label["attack_byte"]
        if source_label.get("attack_byte") is not None
        else -1
    )
    output_file.attrs["sca_source_leakage_model"] = str(
        source_label.get("leakage_model") or "unknown"
    )

    output_file.attrs["sca_trace_selection_kind"] = str(trace_selection["kind"])
    output_file.attrs["sca_trace_start"] = int(trace_selection["start"])
    output_file.attrs["sca_trace_end_exclusive"] = int(
        trace_selection["end_exclusive"]
    )
    output_file.attrs["sca_trace_length"] = int(trace_selection["output_length"])
    output_file.attrs["sca_trace_window_verification"] = str(
        trace_selection["verification_for_attack_byte"]
    )
    output_file.attrs["sca_trace_normalization_mode"] = str(
        active["trace_normalization"]["mode"]
    )

    verification = contract.get("verification") or {}
    if not isinstance(verification, Mapping):
        raise SCADataContractError("contract.verification must be a mapping")
    output_file.attrs["sca_verification_json"] = json.dumps(
        _json_safe(verification),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    output_file.attrs["sca_label_integrity_verified"] = bool(
        verification.get("label_integrity_verified", False)
    )
    output_file.attrs["sca_trace_selection_verified"] = bool(
        verification.get("trace_selection_verified", False)
    )
    output_file.attrs["sca_metadata_mapping_verified"] = bool(
        verification.get("metadata_mapping_verified", False)
    )
    output_file.attrs["sca_source_indices_stored"] = True
    output_file.attrs["sca_desync_offsets_stored"] = True

    if fixed_key_override is not None:
        output_file.attrs["sca_fixed_key_hex"] = _bytes_to_hex16(
            fixed_key_override
        )


def _format_contract_block(contract: Mapping[str, Any]) -> str:
    source = contract["source"]
    source_label = source["label_contract"]
    active = contract["active"]
    trace_selection = active["trace_selection"]

    source_byte = source_label.get("attack_byte")
    source_model = source_label.get("leakage_model")
    source_semantics = (
        f"byte={source_byte}, leakage={source_model}"
        if source_byte is not None and source_model is not None
        else "unknown/ambiguous"
    )

    lines = [
        "=" * 80,
        "SCA DATA CONTRACT",
        "=" * 80,
        f"Source file              : {source['path']}",
        f"Source layout            : {source['layout']}",
        f"Source trace lengths      : profiling={source['profiling_trace_length']} | attack={source['attack_trace_length']}",
        f"Source label semantics    : {source_semantics} ({source_label.get('status', 'unknown')})",
        f"Active dataset            : {active['dataset_alias']}",
        f"Active target             : byte={active['attack_byte']} | leakage={active['leakage_model']} | classes={active['num_classes']}",
        "Labels recomputed         : yes",
        f"Source/active labels differ: {'yes' if active['source_label_semantics_differ'] else 'no/unknown'}",
        f"Trace selection           : {trace_selection['kind']} {trace_selection['configured']} -> length={trace_selection['output_length']}",
        f"Trace-window verification : {trace_selection['verification_for_attack_byte']}",
        f"Trace normalization       : {active['trace_normalization']['mode']}",
        f"Output counts             : profiling={active['profiling_count']} | attack={active['attack_count']}",
        f"Label-integrity gate      : {contract.get('verification', {}).get('status', 'pending')} ",
        f"Trace-selection gate      : {contract.get('verification', {}).get('status', 'pending')}",
        f"Metadata-mapping gate     : {contract.get('verification', {}).get('status', 'pending')}",
        f"Output file               : {contract['output']['path']}",
        "=" * 80,
    ]
    return "\n".join(lines)


def _log_data_contract(contract: Mapping[str, Any], *, cfg: Mapping[str, Any]) -> None:
    """Publish one contract block per model execution in this process.

    The contract is still stored in and validated from every generated HDF5
    artifact.  This gate controls terminal presentation only.
    """
    if not _show_contract(cfg):
        return

    active = contract.get("active")
    if not isinstance(active, Mapping):
        raise SCADataContractError("data contract is missing active provenance")

    model_name = str(cfg.get("model_name") or "<unscoped-model>").strip()
    publication_key = (
        os.getpid(),
        model_name,
        str(active.get("dataset_alias") or "<unknown-dataset>"),
        _strict_int(
            active.get("attack_byte"),
            field="contract.active.attack_byte",
            minimum=0,
            maximum=15,
        ),
        normalize_leakage_model(active.get("leakage_model")),
    )

    with _CONTRACT_PUBLICATION_LOCK:
        if publication_key in _PUBLISHED_CONTRACT_KEYS:
            return
        logger.info("\n%s", _format_contract_block(contract))
        _PUBLISHED_CONTRACT_KEYS.add(publication_key)


def _verification_positions(count: int, *, sample_n: int, seed: int) -> np.ndarray:
    """Return deterministic sorted positions and always include both endpoints."""
    total = _strict_int(count, field="verification count", minimum=1)
    requested = _strict_int(sample_n, field="verification sample_n", minimum=1)
    minimum_size = 2 if total > 1 else 1
    size = min(total, max(requested, minimum_size))
    if size == total:
        return np.arange(total, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    required = {0, total - 1}
    remaining_size = max(0, size - len(required))
    candidates = np.arange(1, max(1, total - 1), dtype=np.int64)
    if remaining_size > 0:
        chosen = rng.choice(candidates, size=remaining_size, replace=False)
        required.update(int(value) for value in chosen.tolist())

    positions = np.asarray(sorted(required), dtype=np.int64)
    return positions


def _verify_metadata_mapping(
    *,
    source_rows: np.ndarray,
    generated_rows: np.ndarray,
    fixed_key_override: Optional[np.ndarray],
    where: str,
) -> None:
    """Verify sampled generated metadata against its recorded source rows."""
    source_fields = source_rows.dtype.fields or {}
    generated_fields = generated_rows.dtype.fields or {}

    for field_name in source_fields:
        if field_name not in generated_fields:
            raise SCADataContractError(
                f"{where}: generated metadata lost source field {field_name!r}"
            )

        if field_name == "key" and fixed_key_override is not None:
            expected = np.tile(
                fixed_key_override.reshape(1, 16),
                (int(generated_rows.shape[0]), 1),
            )
        else:
            expected = source_rows[field_name]

        if not np.array_equal(generated_rows[field_name], expected):
            raise SCADataContractError(
                f"{where}: generated metadata field {field_name!r} does not "
                "match its recorded source rows"
            )

    if fixed_key_override is not None:
        if "key" not in generated_fields:
            raise SCADataContractError(
                f"{where}: fixed-key override was configured but generated "
                "metadata has no key field"
            )
        expected_keys = np.tile(
            fixed_key_override.reshape(1, 16),
            (int(generated_rows.shape[0]), 1),
        )
        if not np.array_equal(generated_rows["key"], expected_keys):
            raise SCADataContractError(
                f"{where}: generated metadata does not contain the configured "
                "fixed-key override"
            )


def _verify_generated_group(
    *,
    generated_group: h5py.Group,
    source_traces: h5py.Dataset,
    source_metadata: h5py.Dataset,
    expected_indices: np.ndarray,
    selection: TraceSelection,
    desync: int,
    attack_byte: int,
    leakage_model: str,
    fixed_key_override: Optional[np.ndarray],
    sample_n: int,
    seed: int,
    chunk_size: int,
    where: str,
) -> dict[str, Any]:
    """
    Verify every active label and a deterministic trace/metadata provenance sample.

    The source-index vector is verified in full. Trace and metadata reconstruction
    use the persisted source indices and applied desynchronization offsets, so a
    self-consistent but wrongly indexed output cannot pass this gate.
    """
    required = {
        "traces",
        "labels",
        "metadata",
        "source_indices",
        "desync_offsets",
    }
    missing = sorted(required.difference(generated_group.keys()))
    if missing:
        raise SCADataContractError(
            f"Generated {where} group is missing datasets: {missing}"
        )

    traces = generated_group["traces"]
    labels = generated_group["labels"]
    metadata = generated_group["metadata"]
    source_indices_ds = generated_group["source_indices"]
    offsets_ds = generated_group["desync_offsets"]

    count = int(expected_indices.size)
    if traces.shape != (count, selection.output_length):
        raise SCADataContractError(
            f"Generated {where}/traces shape mismatch: {traces.shape}; "
            f"expected {(count, selection.output_length)}"
        )
    for name, dataset in (
        ("labels", labels),
        ("metadata", metadata),
        ("source_indices", source_indices_ds),
        ("desync_offsets", offsets_ds),
    ):
        if int(dataset.shape[0]) != count:
            raise SCADataContractError(
                f"Generated {where}/{name} row count does not match expected {count}"
            )

    recorded_indices = np.asarray(source_indices_ds[:], dtype=np.int64)
    if not np.array_equal(recorded_indices, expected_indices):
        mismatch = np.flatnonzero(recorded_indices != expected_indices)
        first = int(mismatch[0]) if mismatch.size else -1
        raise SCADataContractError(
            f"Generated {where}/source_indices do not reproduce the configured "
            f"indices; first_mismatch={first}"
        )

    recorded_offsets = np.asarray(offsets_ds[:], dtype=np.int64)
    if np.any(recorded_offsets < 0) or np.any(recorded_offsets > int(desync)):
        raise SCADataContractError(
            f"Generated {where}/desync_offsets contain values outside "
            f"[0, {int(desync)}]"
        )

    labels_checked = 0
    for start in range(0, count, int(chunk_size)):
        end = min(start + int(chunk_size), count)
        metadata_rows = np.asarray(metadata[start:end])
        fields = metadata_rows.dtype.fields or {}
        if "plaintext" not in fields or "key" not in fields:
            raise SCADataContractError(
                f"Generated {where} metadata must contain plaintext and key"
            )
        expected_labels = labels_for_target(
            metadata_rows,
            attack_byte=attack_byte,
            leakage_model=leakage_model,
            target=_file_label_target(generated_group),
        )
        observed_labels = np.asarray(labels[start:end], dtype=np.uint8)
        if not np.array_equal(observed_labels, expected_labels):
            mismatch = np.flatnonzero(observed_labels != expected_labels)
            first = start + int(mismatch[0])
            raise SCADataContractError(
                f"Label-integrity gate failed for generated {where} at row {first}"
            )
        labels_checked += end - start

    positions = _verification_positions(count, sample_n=sample_n, seed=seed)
    sampled_source_indices = recorded_indices[positions]
    sampled_offsets = recorded_offsets[positions].astype(np.int32, copy=False)
    observed_traces = np.asarray(traces[positions])
    expected_traces, verified_offsets = _read_trace_chunk(
        source_traces,
        sampled_source_indices,
        selection=selection,
        desync=int(desync),
        offsets=sampled_offsets,
    )
    if not np.array_equal(sampled_offsets, verified_offsets):
        raise SCADataContractError(
            f"Generated {where} desynchronization provenance changed during verification"
        )
    if not np.array_equal(observed_traces, expected_traces):
        raise SCADataContractError(
            f"Trace-selection integrity gate failed for generated {where}"
        )

    generated_metadata_rows = _read_h5_rows(metadata, positions)
    source_metadata_rows = _read_h5_rows(
        source_metadata,
        sampled_source_indices,
    )
    _verify_metadata_mapping(
        source_rows=source_metadata_rows,
        generated_rows=generated_metadata_rows,
        fixed_key_override=fixed_key_override,
        where=where,
    )

    return {
        "labels_verified": True,
        "labels_checked": labels_checked,
        "label_scope": "all_rows",
        "source_indices_verified": True,
        "source_indices_checked": count,
        "desync_offsets_verified": True,
        "metadata_mapping_verified": True,
        "metadata_rows_checked": int(positions.size),
        "trace_selection_verified": True,
        "trace_rows_checked": int(positions.size),
        "trace_scope": "deterministic_sample_with_endpoints",
        "maximum_desync": int(desync),
    }


def _verify_generated_file(
    *,
    generated_file: h5py.File,
    profiling_source_traces: h5py.Dataset,
    profiling_source_metadata: h5py.Dataset,
    profiling_indices: np.ndarray,
    attack_source_traces: h5py.Dataset,
    attack_source_metadata: h5py.Dataset,
    attack_indices: np.ndarray,
    selection: TraceSelection,
    profiling_desync: int,
    attack_desync: int,
    attack_byte: int,
    leakage_model: str,
    fixed_key_override: Optional[np.ndarray],
    sample_n: int,
    seed: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Run both independent gates before the temporary HDF5 is published."""
    for group_name in ("Profiling_traces", "Attack_traces"):
        if group_name not in generated_file:
            raise SCADataContractError(
                f"Generated HDF5 is missing group {group_name}"
            )

    profiling_result = _verify_generated_group(
        generated_group=generated_file["Profiling_traces"],
        source_traces=profiling_source_traces,
        source_metadata=profiling_source_metadata,
        expected_indices=profiling_indices,
        selection=selection,
        desync=profiling_desync,
        attack_byte=attack_byte,
        leakage_model=leakage_model,
        fixed_key_override=fixed_key_override,
        sample_n=sample_n,
        seed=seed + 701,
        chunk_size=chunk_size,
        where="Profiling_traces",
    )
    attack_result = _verify_generated_group(
        generated_group=generated_file["Attack_traces"],
        source_traces=attack_source_traces,
        source_metadata=attack_source_metadata,
        expected_indices=attack_indices,
        selection=selection,
        desync=attack_desync,
        attack_byte=attack_byte,
        leakage_model=leakage_model,
        fixed_key_override=fixed_key_override,
        sample_n=sample_n,
        seed=seed + 1701,
        chunk_size=chunk_size,
        where="Attack_traces",
    )

    return {
        "status": "verified",
        "performed_before_publish": True,
        "label_integrity_verified": True,
        "trace_selection_verified": True,
        "metadata_mapping_verified": True,
        "sample_count_requested_per_split": int(sample_n),
        "profiling": profiling_result,
        "attack": attack_result,
    }


def _fsync_file(path: Path) -> None:
    """Flush the verified temporary file to stable storage before replacement."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    """Best-effort synchronization of the containing directory entry."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


# =============================================================================
# Generator: source HDF5 -> complete configuration-specific labeled HDF5
# =============================================================================
def extract_traces(
    *,
    traces_file: str,
    labeled_file: str,
    profiling_index: Sequence[int],
    attack_index: Sequence[int],
    target_points: Sequence[int],
    profiling_desync: int = 0,
    attack_desync: int = 0,
    multilabel: int = 0,
    cfg: Mapping[str, Any],
    seed: int = 42,
    chunk_size: int = 4096,
) -> Path:
    """
    Build one complete configuration-specific ASCAD-style HDF5 atomically.

    Labels are always recomputed from metadata for the active scalar attack byte
    and leakage model. Source labels are inspected only for provenance; they are
    never reused as the active labels.
    """
    if not isinstance(cfg, Mapping):
        raise TypeError("extract_traces requires cfg as a mapping")

    source_filename = check_file_exists(traces_file, cfg=cfg)
    source_path = Path(source_filename)
    output_path = Path(labeled_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path.resolve() == output_path.resolve():
        raise ValueError("Source and generated HDF5 paths must be different")

    attack_byte = _get_attack_byte(cfg)
    leakage_model = _get_leakage_model(cfg)
    variant = _get_variant(cfg)
    fixed_key_override = _get_fixed_key_override(cfg)
    target = label_target(cfg)        # sca/dataset_types.py: the type's label intermediate
    require_attack_byte(cfg)
    active_data_cfg = resolve_active_data_config(cfg)
    contract_verification_samples = _contract_verification_sample_size(
        active_data_cfg
    )

    multilabel_value = _strict_int(
        multilabel,
        field="multilabel",
        minimum=0,
    )
    if multilabel_value != 0:
        raise NotImplementedError(
            "multilabel modes are not implemented in this SCA generator. "
            "Set data.multilabel: 0."
        )

    profiling_desync_value = _strict_int(
        profiling_desync,
        field="profiling_desync",
        minimum=0,
    )
    attack_desync_value = _strict_int(
        attack_desync,
        field="attack_desync",
        minimum=0,
    )
    chunk_size_value = _strict_int(
        chunk_size,
        field="chunk_size",
        minimum=1,
    )
    seed_value = _strict_int(seed, field="seed")
    selection = _resolve_trace_selection(target_points)
    label_dtype = _label_dtype_for_leakage(leakage_model)

    # Let HDF5 create the file normally instead of using tempfile.mkstemp(),
    # whose owner-only 0600 mode survives the final atomic replacement.
    # Access policy and ownership remain a container/deployment responsibility.
    temporary_path = output_path.with_name(
        f".{output_path.name}.{uuid4().hex}.tmp"
    )

    contract: Optional[dict[str, Any]] = None

    try:
        with h5py.File(source_path, "r") as input_file:
            layout = _get_raw_layout(input_file)
            if layout == "unknown":
                raise KeyError(
                    f"Unrecognized ASCAD layout in {source_path}; "
                    f"top-level keys={list(input_file.keys())}"
                )

            if layout == "grouped_both":
                profiling_traces, profiling_metadata, profiling_labels = (
                    _read_group_views(input_file, "Profiling_traces")
                )
                attack_traces, attack_metadata, attack_labels = (
                    _read_group_views(input_file, "Attack_traces")
                )
            elif layout == "grouped_profiling_only":
                raise SCADataContractError(
                    "Source contains Profiling_traces but no Attack_traces. "
                    "Implicitly reusing profiling traces as attack traces is not "
                    "permitted because it can contaminate validation/test evidence. "
                    "Use a complete grouped source or an explicit flat source with "
                    "configured profiling and attack row ranges."
                )
            else:
                profiling_traces = input_file["traces"]
                profiling_metadata = input_file["metadata"]
                profiling_labels = input_file.get("labels")
                attack_traces = profiling_traces
                attack_metadata = profiling_metadata
                attack_labels = profiling_labels

            profiling_fields = _metadata_fields(profiling_metadata)
            attack_fields = _metadata_fields(attack_metadata)
            for where, fields in (
                ("profiling", profiling_fields),
                ("attack", attack_fields),
            ):
                if "plaintext" not in fields:
                    raise KeyError(
                        f"{where} metadata is missing required field 'plaintext'"
                    )
                if "key" not in fields and fixed_key_override is None:
                    raise KeyError(
                        f"{where} metadata is missing 'key'; configure "
                        "sca.fixed_key_hex to supply the fixed key"
                    )

            if variant == "unknown":
                profiling_key_field = (
                    profiling_metadata["key"]
                    if "key" in profiling_fields
                    else None
                )
                attack_key_field = (
                    attack_metadata["key"] if "key" in attack_fields else None
                )
                profiling_variant = _infer_variant_from_keys(
                    profiling_key_field,
                    seed=seed_value,
                )
                attack_variant = _infer_variant_from_keys(
                    attack_key_field,
                    seed=seed_value + 1,
                )
                inferred_variants = {
                    value
                    for value in (profiling_variant, attack_variant)
                    if value != "unknown"
                }
                if len(inferred_variants) != 1:
                    raise ValueError(
                        "sca.variant is missing and could not be inferred "
                        f"unambiguously: profiling={profiling_variant}, "
                        f"attack={attack_variant}"
                    )
                variant = inferred_variants.pop()
                logger.warning(
                    "sca.variant was not configured; inferred variant=%s",
                    variant,
                )

            profiling_indices = _normalize_indices(
                profiling_index,
                field="profiling_index",
                available=int(profiling_traces.shape[0]),
            )
            attack_indices = _normalize_indices(
                attack_index,
                field="attack_index",
                available=int(attack_traces.shape[0]),
            )

            if layout == "flat":
                overlap = np.intersect1d(
                    profiling_indices,
                    attack_indices,
                    assume_unique=True,
                )
                if overlap.size:
                    raise SCADataContractError(
                        "Flat-source profiling and attack selections overlap; "
                        f"first shared source row={int(overlap[0])}. Training and "
                        "evaluation evidence must be disjoint."
                    )

            if fixed_key_override is not None and variant != "fixed_key":
                raise SCADataContractError(
                    "sca.fixed_key_hex is only valid when sca.variant=fixed_key"
                )

            if variant == "fixed_key":
                # A database with a random key per profiling trace (ASCADr, sca/dataset_types.py)
                # is checked on its attack set only; its labels take each trace's own key.
                profiling_key = None if random_profiling_keys(cfg) else _validate_selected_fixed_key(
                    profiling_metadata,
                    profiling_indices,
                    expected_key=fixed_key_override,
                    where="Profiling_traces",
                    chunk_size=chunk_size_value,
                )
                attack_key = _validate_selected_fixed_key(
                    attack_metadata,
                    attack_indices,
                    expected_key=fixed_key_override,
                    where="Attack_traces",
                    chunk_size=chunk_size_value,
                )
                if profiling_key is not None and not np.array_equal(profiling_key, attack_key) and shared_key(cfg):
                    raise SCADataContractError(
                        "Fixed-key profiling and attack selections use different keys"
                    )   # unless the database type attacks another key (CHES_CTF_2018, sca/dataset_types.py)

            _validate_trace_bounds(
                profiling_traces,
                selection,
                desync=profiling_desync_value,
                where="Profiling_traces",
            )
            _validate_trace_bounds(
                attack_traces,
                selection,
                desync=attack_desync_value,
                where="Attack_traces",
            )

            trace_window_verification = _require_scientifically_valid_trace_window(
                active_data_cfg,
                attack_byte=attack_byte,
                selection=selection,
                profiling_traces=profiling_traces,
                attack_traces=attack_traces,
                profiling_desync=profiling_desync_value,
                attack_desync=attack_desync_value,
            )

            source_label_contract = _resolve_source_label_contract(
                input_file,
                profiling_labels=profiling_labels,
                profiling_metadata=profiling_metadata,
                attack_labels=attack_labels,
                attack_metadata=attack_metadata,
                seed=seed_value + 211,
            )

            source_attrs = _snapshot_h5_attrs(input_file.attrs)
            profiling_output_dtype = _output_metadata_dtype(
                profiling_metadata.dtype,
                add_key=(
                    fixed_key_override is not None
                    and "key" not in profiling_fields
                ),
            )
            attack_output_dtype = _output_metadata_dtype(
                attack_metadata.dtype,
                add_key=(
                    fixed_key_override is not None
                    and "key" not in attack_fields
                ),
            )

            contract = _build_data_contract(
                source_path=source_path,
                source_layout=layout,
                source_attrs=source_attrs,
                source_label_contract=source_label_contract,
                profiling_traces=profiling_traces,
                attack_traces=attack_traces,
                active_data_cfg=active_data_cfg,
                variant=variant,
                attack_byte=attack_byte,
                leakage_model=leakage_model,
                selection=selection,
                profiling_count=int(profiling_indices.size),
                attack_count=int(attack_indices.size),
                profiling_desync=profiling_desync_value,
                attack_desync=attack_desync_value,
                output_path=output_path,
                trace_window_verification=trace_window_verification,
            )
            rng_profiling = np.random.default_rng(seed_value)
            rng_attack = np.random.default_rng(seed_value + 1337)

            with h5py.File(temporary_path, "x") as output_file:
                if target != "sbox":
                    # Only a non-S-box target is recorded, so ASCADf files stay byte-identical; the
                    # splits copy it (utils/data_split._copy_root_attrs) and the verifiers read it.
                    output_file.attrs["sca_label_target"] = target
                profiling_group = output_file.create_group("Profiling_traces")
                attack_group = output_file.create_group("Attack_traces")
                _write_data_contract_attrs(
                    output_file,
                    contract,
                    fixed_key_override=fixed_key_override,
                )

                profiling_count = int(profiling_indices.size)
                attack_count = int(attack_indices.size)
                trace_length = int(selection.output_length)

                profiling_trace_chunks = (
                    min(chunk_size_value, profiling_count),
                    trace_length,
                )
                attack_trace_chunks = (
                    min(chunk_size_value, attack_count),
                    trace_length,
                )

                profiling_out_traces = profiling_group.create_dataset(
                    "traces",
                    shape=(profiling_count, trace_length),
                    dtype=profiling_traces.dtype,
                    chunks=profiling_trace_chunks,
                )
                profiling_out_labels = profiling_group.create_dataset(
                    "labels",
                    shape=(profiling_count,),
                    dtype=label_dtype,
                    chunks=(min(chunk_size_value, profiling_count),),
                )
                profiling_out_metadata = profiling_group.create_dataset(
                    "metadata",
                    shape=(profiling_count,),
                    dtype=profiling_output_dtype,
                    chunks=(min(chunk_size_value, profiling_count),),
                )
                profiling_out_source_indices = profiling_group.create_dataset(
                    "source_indices",
                    shape=(profiling_count,),
                    dtype=np.int64,
                    chunks=(min(chunk_size_value, profiling_count),),
                )
                profiling_out_desync_offsets = profiling_group.create_dataset(
                    "desync_offsets",
                    shape=(profiling_count,),
                    dtype=np.int32,
                    chunks=(min(chunk_size_value, profiling_count),),
                )

                attack_out_traces = attack_group.create_dataset(
                    "traces",
                    shape=(attack_count, trace_length),
                    dtype=attack_traces.dtype,
                    chunks=attack_trace_chunks,
                )
                attack_out_labels = attack_group.create_dataset(
                    "labels",
                    shape=(attack_count,),
                    dtype=label_dtype,
                    chunks=(min(chunk_size_value, attack_count),),
                )
                attack_out_metadata = attack_group.create_dataset(
                    "metadata",
                    shape=(attack_count,),
                    dtype=attack_output_dtype,
                    chunks=(min(chunk_size_value, attack_count),),
                )
                attack_out_source_indices = attack_group.create_dataset(
                    "source_indices",
                    shape=(attack_count,),
                    dtype=np.int64,
                    chunks=(min(chunk_size_value, attack_count),),
                )
                attack_out_desync_offsets = attack_group.create_dataset(
                    "desync_offsets",
                    shape=(attack_count,),
                    dtype=np.int32,
                    chunks=(min(chunk_size_value, attack_count),),
                )

                def write_split(
                    *,
                    indices: np.ndarray,
                    source_traces: h5py.Dataset,
                    source_metadata: h5py.Dataset,
                    output_traces: h5py.Dataset,
                    output_labels: h5py.Dataset,
                    output_metadata: h5py.Dataset,
                    output_source_indices: h5py.Dataset,
                    output_desync_offsets: h5py.Dataset,
                    output_metadata_dtype: np.dtype,
                    desync: int,
                    rng: np.random.Generator,
                ) -> None:
                    total = int(indices.size)
                    for output_start in range(0, total, chunk_size_value):
                        output_end = min(output_start + chunk_size_value, total)
                        chunk_indices = indices[output_start:output_end]

                        trace_rows, applied_offsets = _read_trace_chunk(
                            source_traces,
                            chunk_indices,
                            selection=selection,
                            desync=desync,
                            rng=rng,
                        )
                        raw_metadata_rows = _read_h5_rows(
                            source_metadata,
                            chunk_indices,
                        )
                        metadata_rows = _prepare_metadata_rows(
                            raw_metadata_rows,
                            output_dtype=output_metadata_dtype,
                            fixed_key_override=fixed_key_override,
                        )

                        fields = metadata_rows.dtype.fields or {}
                        if "plaintext" not in fields or "key" not in fields:
                            raise KeyError(
                                "Generated metadata must contain plaintext and key"
                            )
                        labels = labels_for_target(
                            metadata_rows,
                            attack_byte=attack_byte,
                            leakage_model=leakage_model,
                            target=target,
                        )

                        output_traces[output_start:output_end] = trace_rows
                        output_labels[output_start:output_end] = labels.astype(
                            label_dtype,
                            copy=False,
                        )
                        output_metadata[output_start:output_end] = metadata_rows
                        output_source_indices[output_start:output_end] = chunk_indices
                        output_desync_offsets[output_start:output_end] = applied_offsets

                write_split(
                    indices=profiling_indices,
                    source_traces=profiling_traces,
                    source_metadata=profiling_metadata,
                    output_traces=profiling_out_traces,
                    output_labels=profiling_out_labels,
                    output_metadata=profiling_out_metadata,
                    output_source_indices=profiling_out_source_indices,
                    output_desync_offsets=profiling_out_desync_offsets,
                    output_metadata_dtype=profiling_output_dtype,
                    desync=profiling_desync_value,
                    rng=rng_profiling,
                )
                write_split(
                    indices=attack_indices,
                    source_traces=attack_traces,
                    source_metadata=attack_metadata,
                    output_traces=attack_out_traces,
                    output_labels=attack_out_labels,
                    output_metadata=attack_out_metadata,
                    output_source_indices=attack_out_source_indices,
                    output_desync_offsets=attack_out_desync_offsets,
                    output_metadata_dtype=attack_output_dtype,
                    desync=attack_desync_value,
                    rng=rng_attack,
                )
                output_file.flush()

                verification = _verify_generated_file(
                    generated_file=output_file,
                    profiling_source_traces=profiling_traces,
                    profiling_source_metadata=profiling_metadata,
                    profiling_indices=profiling_indices,
                    attack_source_traces=attack_traces,
                    attack_source_metadata=attack_metadata,
                    attack_indices=attack_indices,
                    selection=selection,
                    profiling_desync=profiling_desync_value,
                    attack_desync=attack_desync_value,
                    attack_byte=attack_byte,
                    leakage_model=leakage_model,
                    fixed_key_override=fixed_key_override,
                    sample_n=contract_verification_samples,
                    seed=seed_value,
                    chunk_size=chunk_size_value,
                )
                contract["verification"] = verification
                _write_data_contract_attrs(
                    output_file,
                    contract,
                    fixed_key_override=fixed_key_override,
                )
                output_file.flush()

            _fsync_file(temporary_path)

        # Signed for its final path before it is published (utils/seal.py).
        seal_h5_file(temporary_path, scope_path=output_path)
        os.replace(temporary_path, output_path)
        _fsync_directory(output_path.parent)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    if contract is None:
        raise RuntimeError("Internal error: data contract was not created")

    logger.info(
        "Wrote configuration-specific ASCAD dataset: %s "
        "(profiling=%d, attack=%d, length=%d, byte=%d, variant=%s, leakage=%s)",
        output_path,
        contract["active"]["profiling_count"],
        contract["active"]["attack_count"],
        contract["active"]["trace_selection"]["output_length"],
        attack_byte,
        variant,
        leakage_model,
    )
    _log_data_contract(contract, cfg=cfg)
    return output_path.resolve()


def _range_pair(data_cfg: Mapping[str, Any], key: str) -> tuple[int, int]:
    raw = data_cfg.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise TypeError(f"data.{key} must be [start, end]")
    start = _strict_int(raw[0], field=f"data.{key}[0]", minimum=0)
    end = _strict_int(raw[1], field=f"data.{key}[1]", minimum=0)
    if end <= start:
        raise ValueError(f"data.{key} end must be greater than start")
    return start, end


def profiling_indices(data_cfg: Mapping[str, Any]) -> np.ndarray:
    """The Profiling_traces rows a run trains on, ascending. Without data.profiling_split, or with
    enabled: false: all of profiling_range. With enabled: true: the training part scikit-learn's
    train_test_split draws from those rows, with test_size, shuffle and random_state as written (Wouters et
    al. TCHES 2020(3): test_size 0.1, random_state 0). The held-out part is not read: a run's validation
    traces come from the attack set."""
    start, end = _range_pair(data_cfg, "profiling_range")
    rows = np.arange(start, end, dtype=np.int64)
    problems = profiling_split_problems(data_cfg)
    if problems:
        raise SCADataContractError("data." + "; data.".join(problems))
    if not profiling_split_enabled(data_cfg):
        return rows
    split = data_cfg["profiling_split"]
    from sklearn.model_selection import train_test_split

    # random_state null: unseeded, a new cut every run (Zaid et al.'s random.shuffle has no seed). A fresh RandomState
    # draws its seed from the operating system; scikit-learn's own None would read NumPy's global generator, which
    # utils/seed.set_seed has seeded with the run's seed.
    random_state = split.get("random_state")
    if isinstance(random_state, list):
        raise SCADataContractError(
            f"data.profiling_split.random_state {random_state} is a list of splits: the sweep makes one run per value "
            "(utils/sweep_engine), and a run reads one")
    trained, _held_out = train_test_split(
        rows,
        test_size=split["test_size"],
        shuffle=split["shuffle"],
        random_state=random_state if random_state is not None else np.random.RandomState(),
    )
    return np.sort(trained)


def generate(cfg: MutableMapping[str, Any]) -> Path:
    """Generate the complete active configuration-specific HDF5 at paths.data_out."""
    if not isinstance(cfg, MutableMapping):
        raise TypeError(f"cfg must be a mutable mapping, got {type(cfg).__name__}")

    data_cfg = resolve_active_data_config(cfg)
    for required_key in ("profiling_range", "attack_range", "database"):
        if required_key not in data_cfg:
            raise KeyError(f"Active dataset is missing required field {required_key!r}")

    # Fail before any filesystem writes if a search space was not resolved.
    attack_byte = _get_attack_byte(cfg)
    _get_leakage_model(cfg)
    target_points = _target_points_for_attack_byte(
        data_cfg,
        attack_byte=attack_byte,
    )

    profiling_rows = profiling_indices(data_cfg)
    attack_start, attack_end = _range_pair(data_cfg, "attack_range")
    seed = int(set_seed(cfg))

    model_name = cfg.get("model_name")
    if model_name is None or not str(model_name).strip():
        raise ValueError("cfg['model_name'] is required to resolve paths.data_out")

    paths = cfg.get("paths") or {}
    if not isinstance(paths, Mapping) or not paths.get("data_out"):
        raise KeyError("cfg['paths']['data_out'] is required")

    dataset_alias = _dataset_identity(data_cfg)
    path_cfg = dict(cfg)
    path_cfg.setdefault("dataset_alias", dataset_alias)
    output_file = Path(
        resolve_path(
            paths["data_out"],
            path_cfg,
            model_name=str(model_name),
        )
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating complete configuration-specific dataset: alias=%s seed=%d output=%s",
        dataset_alias,
        seed,
        output_file,
    )

    return extract_traces(
        traces_file=str(data_cfg["database"]),
        labeled_file=str(output_file),
        profiling_index=profiling_rows,
        attack_index=range(attack_start, attack_end),
        target_points=target_points,
        profiling_desync=_strict_int(
            data_cfg.get("profiling_desync", 0),
            field="data.profiling_desync",
            minimum=0,
        ),
        attack_desync=_strict_int(
            data_cfg.get("attack_desync", 0),
            field="data.attack_desync",
            minimum=0,
        ),
        multilabel=_strict_int(
            data_cfg.get("multilabel", 0),
            field="data.multilabel",
            minimum=0,
        ),
        cfg=cfg,
        seed=seed,
        chunk_size=_strict_int(
            data_cfg.get("chunk_size", 4096),
            field="data.chunk_size",
            minimum=1,
        ),
    )


# =============================================================================
# Trace normalization
# =============================================================================
def _normalize_trace(
    trace_f32: np.ndarray,
    raw_dtype: np.dtype,
    norm_cfg: Mapping[str, Any],
) -> np.ndarray:
    mode = str(norm_cfg.get("mode", "none")).strip().lower()
    eps = float(norm_cfg.get("eps", 1.0e-8))

    if mode == "none":
        result = trace_f32
    elif mode == "auto":
        if np.issubdtype(raw_dtype, np.int8):
            result = trace_f32 / 128.0
        elif np.issubdtype(raw_dtype, np.int16):
            result = trace_f32 / 32768.0
        elif np.issubdtype(raw_dtype, np.uint8):
            result = (trace_f32 - 128.0) / 128.0
        else:
            result = trace_f32
    elif mode == "zscore":
        mean = float(norm_cfg["mean"])
        std = float(norm_cfg["std"])
        if abs(std) < eps:
            raise ValueError("zscore standard deviation is too close to zero")
        result = (trace_f32 - mean) / std
    elif mode == "per_trace_standard":
        # Each trace by its own mean and standard deviation (scikit-learn, along the samples).
        from sklearn.preprocessing import scale

        result = scale(np.atleast_2d(trace_f32), axis=1).reshape(trace_f32.shape)
    elif mode == "per_trace_minmax":
        from sklearn.preprocessing import minmax_scale

        result = minmax_scale(np.atleast_2d(trace_f32), feature_range=tuple(norm_cfg["feature_range"]),
                              axis=1).reshape(trace_f32.shape)
    elif mode == "fitted":
        # Scaled later by a scaler fitted on the training traces (utils/trace_scaler.py).
        result = trace_f32
    elif mode == "minmax":
        minimum = float(norm_cfg["min"])
        maximum = float(norm_cfg["max"])
        denominator = maximum - minimum
        if abs(denominator) < eps:
            raise ValueError("minmax range is too close to zero")
        result = 2.0 * ((trace_f32 - minimum) / denominator) - 1.0
    else:
        raise ValueError(f"Unsupported trace normalization mode {mode!r}")

    return np.asarray(result, dtype=np.float32)


# =============================================================================
# HDF5-backed PyTorch dataset
# =============================================================================
def _meta_void_to_dict(metadata_row: np.void) -> dict[str, Any]:
    fields = metadata_row.dtype.fields or {}
    output: dict[str, Any] = {}

    if "plaintext" in fields:
        output["plaintext"] = np.asarray(
            metadata_row["plaintext"], dtype=np.uint8
        )
    if "key" in fields:
        output["key"] = np.asarray(metadata_row["key"], dtype=np.uint8)
    if "masks" in fields:
        output["masks"] = np.asarray(metadata_row["masks"], dtype=np.uint8)
    # The inputs of the non-S-box label targets (sca/leakage_models.py): AES_HD, DPAv4.
    if "ciphertext" in fields:
        output["ciphertext"] = np.asarray(metadata_row["ciphertext"], dtype=np.uint8)
    if "mask" in fields:
        output["mask"] = np.uint8(metadata_row["mask"])
    if "desync" in fields:
        desync = metadata_row["desync"]
        if hasattr(desync, "__len__") and not isinstance(desync, (bytes, str)):
            desync = desync[0]
        output["desync"] = int(desync)
    return output


def _file_attack_byte(attributes: h5py.AttributeManager) -> Optional[int]:
    raw = _first_attr(
        attributes,
        ("sca_active_attack_byte", "sca_attack_byte"),
    )
    if raw is None:
        return None
    return _strict_int(
        raw,
        field="HDF5 active attack byte",
        minimum=0,
        maximum=15,
    )


def _file_leakage_model(attributes: h5py.AttributeManager) -> Optional[str]:
    raw = _first_attr(
        attributes,
        ("sca_active_leakage_model", "sca_leakage_model"),
    )
    return normalize_leakage_model(raw) if raw is not None else None


def _file_variant(attributes: h5py.AttributeManager) -> str:
    raw = _first_attr(attributes, ("sca_variant",))
    if raw is None:
        return "unknown"
    normalized = str(raw).strip().lower()
    if normalized not in _SUPPORTED_VARIANTS:
        raise ValueError(f"HDF5 sca_variant is invalid: {raw!r}")
    return normalized


def _validate_label_dataset(
    labels: h5py.Dataset,
    *,
    num_classes: int,
    where: str,
    chunk_size: int = 65536,
) -> None:
    """Validate label range without materializing the complete dataset."""
    if labels.ndim != 1:
        raise ValueError(f"{where}/labels must be one-dimensional, got {labels.shape}")
    count = int(labels.shape[0])
    if count <= 0:
        raise ValueError(f"{where}/labels is empty")

    observed_min = num_classes
    observed_max = -1
    for start in range(0, count, int(chunk_size)):
        end = min(start + int(chunk_size), count)
        values = np.asarray(labels[start:end], dtype=np.int64)
        if values.size == 0:
            continue
        observed_min = min(observed_min, int(values.min()))
        observed_max = max(observed_max, int(values.max()))

    if observed_min < 0 or observed_max >= num_classes:
        raise ValueError(
            f"{where}/labels range [{observed_min}, {observed_max}] is invalid "
            f"for num_classes={num_classes}"
        )


def _validate_loaded_labels_against_metadata(
    group: h5py.Group,
    *,
    attack_byte: int,
    leakage_model: str,
    where: str,
    chunk_size: int = 65536,
) -> None:
    """Recompute every loaded label so copied or modified split files fail closed."""
    if "metadata" not in group:
        raise SCADataContractError(
            f"{where}: metadata is required to verify label provenance"
        )
    metadata = group["metadata"]
    labels = group["labels"]
    fields = _metadata_fields(metadata)
    if "plaintext" not in fields or "key" not in fields:
        raise SCADataContractError(
            f"{where}: metadata must contain plaintext and key"
        )

    count = int(labels.shape[0])
    for start in range(0, count, int(chunk_size)):
        end = min(start + int(chunk_size), count)
        rows = np.asarray(metadata[start:end])
        expected = labels_for_target(
            rows,
            attack_byte=attack_byte,
            leakage_model=leakage_model,
            target=_file_label_target(group),
        )
        observed = np.asarray(labels[start:end], dtype=np.int64)
        mismatch = np.flatnonzero(observed != expected.astype(np.int64, copy=False))
        if mismatch.size:
            raise SCADataContractError(
                f"{where}: label/metadata mismatch at row "
                f"{start + int(mismatch[0])}"
            )


def _load_validated_data_contract(
    input_file: h5py.File,
    *,
    path: Path,
    required: bool,
) -> Optional[dict[str, Any]]:
    """Load and independently validate the canonical HDF5 data contract."""
    raw_contract = input_file.attrs.get("sca_data_contract_json")
    raw_digest = input_file.attrs.get("sca_data_contract_sha256")

    if raw_contract is None:
        if required:
            raise SCADataContractError(
                f"{path} is missing required sca_data_contract_json provenance"
            )
        if raw_digest is not None:
            raise SCADataContractError(
                f"{path} has a data-contract digest but no contract payload"
            )
        return None

    if raw_digest is None:
        raise SCADataContractError(
            f"{path} has a data contract but no SHA-256 fingerprint"
        )

    decoded = str(_decode_attr(raw_contract))
    computed_digest = hashlib.sha256(decoded.encode("utf-8")).hexdigest()
    stored_digest = str(_decode_attr(raw_digest))
    if computed_digest != stored_digest:
        raise SCADataContractError(
            f"Data-contract fingerprint mismatch in {path}"
        )

    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise SCADataContractError(
            f"Invalid sca_data_contract_json in {path}"
        ) from exc
    if not isinstance(parsed, dict):
        raise SCADataContractError(
            "sca_data_contract_json must encode an object"
        )

    version = str(parsed.get("version", "")).strip()
    attr_version = str(
        _decode_attr(input_file.attrs.get("sca_contract_version", ""))
    ).strip()
    if version != DATA_CONTRACT_VERSION or attr_version != DATA_CONTRACT_VERSION:
        raise SCADataContractError(
            f"Unsupported data-contract version in {path}: "
            f"payload={version!r}, attribute={attr_version!r}, "
            f"expected={DATA_CONTRACT_VERSION!r}"
        )

    verification = parsed.get("verification")
    if not isinstance(verification, Mapping):
        raise SCADataContractError(
            f"{path} data contract is missing verification evidence"
        )
    required_flags = (
        "label_integrity_verified",
        "trace_selection_verified",
        "metadata_mapping_verified",
    )
    if str(verification.get("status", "")).strip().lower() != "verified":
        raise SCADataContractError(
            f"{path} data contract was not verified before publication"
        )
    if verification.get("performed_before_publish") is not True:
        raise SCADataContractError(
            f"{path} data contract lacks pre-publication verification evidence"
        )
    for field in required_flags:
        if verification.get(field) is not True:
            raise SCADataContractError(
                f"{path} data contract has {field}=false/missing"
            )

    attr_flag_names = {
        "label_integrity_verified": "sca_label_integrity_verified",
        "trace_selection_verified": "sca_trace_selection_verified",
        "metadata_mapping_verified": "sca_metadata_mapping_verified",
    }
    for contract_field, attr_name in attr_flag_names.items():
        raw_flag = input_file.attrs.get(attr_name)
        if raw_flag is None or not _strict_bool(raw_flag, field=f"HDF5 {attr_name}"):
            raise SCADataContractError(
                f"{path} is missing positive {attr_name} verification"
            )
        if verification.get(contract_field) is not True:
            raise SCADataContractError(
                f"{path} contract and HDF5 {attr_name} disagree"
            )

    verification_json = input_file.attrs.get("sca_verification_json")
    if verification_json is None:
        raise SCADataContractError(
            f"{path} is missing sca_verification_json"
        )
    try:
        verification_attr = json.loads(str(_decode_attr(verification_json)))
    except json.JSONDecodeError as exc:
        raise SCADataContractError(
            f"Invalid sca_verification_json in {path}"
        ) from exc
    if verification_attr != verification:
        raise SCADataContractError(
            f"{path} verification attribute and canonical contract disagree"
        )

    return parsed


class ASCADDataset(Dataset):
    """HDF5-backed ASCAD dataset with one lazily opened handle per worker."""

    def __init__(
        self,
        h5_path: str | Path,
        split: str = "attack",
        return_metadata: bool = True,
        require_metadata: bool = True,
        cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        # Initialize process-local state before any validation that may raise.
        self._h5: Optional[h5py.File] = None
        self._group: Optional[h5py.Group] = None
        self._pid: Optional[int] = None
        self._traces: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._meta: Optional[np.ndarray] = None
        self._trace_scaler: Any = None      # a fitted sklearn Pipeline (utils/trace_scaler.py)
        self.data_contract: Optional[dict[str, Any]] = None

        if split not in _GROUP_BY_SPLIT:
            raise ValueError(
                f"split must be one of {sorted(_GROUP_BY_SPLIT)}, got {split!r}"
            )
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        self.split = split
        self.group_name = _GROUP_BY_SPLIT[split]
        self.return_metadata = _strict_bool(
            return_metadata,
            field="return_metadata",
        )
        self.require_metadata = _strict_bool(
            require_metadata,
            field="require_metadata",
        )
        self._norm_cfg = _get_trace_normalization_cfg(cfg)
        self._contract_required = _require_contract(cfg)

        runtime_attack_byte = _optional_runtime_attack_byte(cfg)
        runtime_leakage_model = _optional_runtime_leakage_model(cfg)
        runtime_variant = _get_variant(cfg) if isinstance(cfg, Mapping) else "unknown"
        runtime_dataset_alias = (
            _dataset_identity(resolve_active_data_config(cfg))
            if isinstance(cfg, Mapping)
            else None
        )

        check_seal(self.h5_path)  # utils/seal.py
        with h5py.File(self.h5_path, "r") as input_file:
            self.data_contract = _load_validated_data_contract(
                input_file,
                path=self.h5_path,
                required=self._contract_required,
            )
            contract_active = (
                self.data_contract.get("active")
                if isinstance(self.data_contract, Mapping)
                else {}
            )
            if not isinstance(contract_active, Mapping):
                raise SCADataContractError(
                    f"{self.h5_path} contract.active must be a mapping"
                )
            if self._contract_required:
                if contract_active.get("labels_recomputed") is not True:
                    raise SCADataContractError(
                        f"{self.h5_path} does not prove active labels were recomputed"
                    )
                contract_alias = str(
                    contract_active.get("dataset_alias", "")
                ).strip()
                if not contract_alias:
                    raise SCADataContractError(
                        f"{self.h5_path} contract has no active dataset alias"
                    )
                if (
                    runtime_dataset_alias is not None
                    and runtime_dataset_alias != contract_alias
                ):
                    raise SCADataContractError(
                        "Runtime/file dataset mismatch: "
                        f"runtime={runtime_dataset_alias}, file={contract_alias}, "
                        f"path={self.h5_path}"
                    )

            if self.group_name not in input_file:
                raise KeyError(
                    f"Group {self.group_name!r} is missing from {self.h5_path}; "
                    f"available={list(input_file.keys())}"
                )
            group = input_file[self.group_name]
            for required_name in ("traces", "labels"):
                if required_name not in group:
                    raise KeyError(
                        f"Missing {self.group_name}/{required_name} in {self.h5_path}"
                    )

            traces = group["traces"]
            labels = group["labels"]
            if traces.ndim != 2:
                raise ValueError(
                    f"{self.group_name}/traces must have shape [N,L], got {traces.shape}"
                )
            if labels.ndim != 1:
                raise ValueError(
                    f"{self.group_name}/labels must have shape [N], got {labels.shape}"
                )
            if int(traces.shape[0]) != int(labels.shape[0]):
                raise ValueError(
                    f"Trace/label count mismatch in {self.group_name}: "
                    f"{traces.shape[0]} vs {labels.shape[0]}"
                )
            if int(traces.shape[0]) <= 0 or int(traces.shape[1]) <= 0:
                raise ValueError(f"{self.group_name} contains an empty trace array")

            if self._contract_required:
                count_field = (
                    "profiling_count" if split == "profiling" else "attack_count"
                )
                contract_count = _strict_int(
                    contract_active.get(count_field),
                    field=f"contract.active.{count_field}",
                    minimum=1,
                )
                if contract_count != int(labels.shape[0]):
                    raise SCADataContractError(
                        f"{self.h5_path}:{self.group_name} has {labels.shape[0]} "
                        f"rows but contract.active.{count_field}={contract_count}"
                    )

            self._has_meta = "metadata" in group
            if self._has_meta and int(group["metadata"].shape[0]) != int(labels.shape[0]):
                raise ValueError(
                    f"Metadata/label count mismatch in {self.group_name}: "
                    f"{group['metadata'].shape[0]} vs {labels.shape[0]}"
                )
            if self.return_metadata and self.require_metadata and not self._has_meta:
                raise KeyError(
                    f"Metadata is required but missing: {self.group_name}/metadata"
                )

            if self._contract_required:
                for provenance_name in ("source_indices", "desync_offsets"):
                    if provenance_name not in group:
                        raise SCADataContractError(
                            f"{self.h5_path}:{self.group_name} is missing required "
                            f"provenance dataset {provenance_name!r}"
                        )
                    if int(group[provenance_name].shape[0]) != int(labels.shape[0]):
                        raise SCADataContractError(
                            f"{self.h5_path}:{self.group_name}/{provenance_name} "
                            "row count does not match labels"
                        )

            contract_attack_byte = None
            if "attack_byte" in contract_active:
                contract_attack_byte = _strict_int(
                    contract_active["attack_byte"],
                    field="contract.active.attack_byte",
                    minimum=0,
                    maximum=15,
                )
            contract_leakage_model = None
            if "leakage_model" in contract_active:
                contract_leakage_model = normalize_leakage_model(
                    contract_active["leakage_model"]
                )
            contract_variant = str(
                contract_active.get("variant", "unknown")
            ).strip().lower()
            if contract_variant not in _SUPPORTED_VARIANTS | {"unknown"}:
                raise SCADataContractError(
                    f"Invalid contract.active.variant={contract_variant!r}"
                )

            attr_attack_byte = _file_attack_byte(input_file.attrs)
            attr_leakage_model = _file_leakage_model(input_file.attrs)
            attr_variant = _file_variant(input_file.attrs)

            if (
                attr_attack_byte is not None
                and contract_attack_byte is not None
                and attr_attack_byte != contract_attack_byte
            ):
                raise SCADataContractError(
                    "HDF5 attributes and data-contract attack byte disagree"
                )
            if (
                attr_leakage_model is not None
                and contract_leakage_model is not None
                and attr_leakage_model != contract_leakage_model
            ):
                raise SCADataContractError(
                    "HDF5 attributes and data-contract leakage model disagree"
                )
            if (
                attr_variant != "unknown"
                and contract_variant != "unknown"
                and attr_variant != contract_variant
            ):
                raise SCADataContractError(
                    "HDF5 attributes and data-contract variant disagree"
                )

            file_attack_byte = (
                attr_attack_byte
                if attr_attack_byte is not None
                else contract_attack_byte
            )
            file_leakage_model = (
                attr_leakage_model
                if attr_leakage_model is not None
                else contract_leakage_model
            )
            file_variant = (
                attr_variant if attr_variant != "unknown" else contract_variant
            )

            if (
                runtime_attack_byte is not None
                and file_attack_byte is not None
                and runtime_attack_byte != file_attack_byte
            ):
                raise SCADataContractError(
                    "Runtime/file attack-byte mismatch: "
                    f"runtime={runtime_attack_byte}, file={file_attack_byte}, "
                    f"path={self.h5_path}"
                )
            if (
                runtime_leakage_model is not None
                and file_leakage_model is not None
                and runtime_leakage_model != file_leakage_model
            ):
                raise SCADataContractError(
                    "Runtime/file leakage-model mismatch: "
                    f"runtime={runtime_leakage_model}, file={file_leakage_model}, "
                    f"path={self.h5_path}"
                )
            if (
                runtime_variant != "unknown"
                and file_variant != "unknown"
                and runtime_variant != file_variant
            ):
                raise SCADataContractError(
                    "Runtime/file variant mismatch: "
                    f"runtime={runtime_variant}, file={file_variant}, "
                    f"path={self.h5_path}"
                )

            self._attack_byte = (
                file_attack_byte
                if file_attack_byte is not None
                else runtime_attack_byte
            )
            if self._attack_byte is None:
                raise SCADataContractError(
                    f"{self.h5_path} has no attack-byte provenance and no "
                    "resolved runtime sca.attack_byte"
                )

            self._leakage_model = (
                file_leakage_model
                if file_leakage_model is not None
                else runtime_leakage_model
            )
            if self._leakage_model is None:
                raise SCADataContractError(
                    f"{self.h5_path} has no leakage-model provenance and no "
                    "resolved runtime sca.leakage_model"
                )
            self._variant = (
                file_variant if file_variant != "unknown" else runtime_variant
            )
            attr_num_classes = input_file.attrs.get("sca_num_classes")
            contract_num_classes = contract_active.get("num_classes")
            if attr_num_classes is not None:
                attr_num_classes = _strict_int(
                    attr_num_classes,
                    field="HDF5 sca_num_classes",
                    minimum=1,
                )
            if contract_num_classes is not None:
                contract_num_classes = _strict_int(
                    contract_num_classes,
                    field="contract.active.num_classes",
                    minimum=1,
                )
            if (
                attr_num_classes is not None
                and contract_num_classes is not None
                and attr_num_classes != contract_num_classes
            ):
                raise SCADataContractError(
                    "HDF5 attributes and data-contract num_classes disagree"
                )

            expected_classes = get_num_classes(self._leakage_model)
            self._num_classes = int(
                attr_num_classes
                or contract_num_classes
                or expected_classes
            )
            if self._num_classes != expected_classes:
                raise ValueError(
                    f"HDF5 num_classes={self._num_classes} conflicts with "
                    f"leakage_model={self._leakage_model} "
                    f"(expected {expected_classes})"
                )
            _validate_label_dataset(
                labels,
                num_classes=self._num_classes,
                where=self.group_name,
            )
            if self._contract_required:
                _validate_loaded_labels_against_metadata(
                    group,
                    attack_byte=self._attack_byte,
                    leakage_model=self._leakage_model,
                    where=f"{self.h5_path}:{self.group_name}",
                )

            attr_normalization = _first_attr(
                input_file.attrs,
                ("sca_trace_normalization_mode",),
            )
            contract_normalization = contract_active.get("trace_normalization")
            contract_mode = None
            if contract_normalization is not None:
                if not isinstance(contract_normalization, Mapping):
                    raise SCADataContractError(
                        "contract.active.trace_normalization must be a mapping"
                    )
                contract_mode = str(
                    contract_normalization.get("mode", "")
                ).strip().lower()
                if contract_mode not in _SUPPORTED_NORMALIZATION_MODES:
                    raise SCADataContractError(
                        f"Invalid contract trace-normalization mode {contract_mode!r}"
                    )
                canonical_contract_normalization = _normalization_from_data_cfg(
                    {"trace_normalization": contract_normalization}
                )
                if canonical_contract_normalization != self._norm_cfg:
                    raise SCADataContractError(
                        "Runtime/file trace-normalization parameters disagree: "
                        f"runtime={self._norm_cfg}, "
                        f"file={canonical_contract_normalization}, "
                        f"path={self.h5_path}"
                    )

            attr_mode = (
                str(attr_normalization).strip().lower()
                if attr_normalization is not None
                else None
            )
            if (
                attr_mode is not None
                and contract_mode is not None
                and attr_mode != contract_mode
            ):
                raise SCADataContractError(
                    "HDF5 attributes and data-contract normalization mode disagree"
                )

            stored_mode = attr_mode or contract_mode
            runtime_mode = str(self._norm_cfg["mode"]).strip().lower()
            if stored_mode is None and self._contract_required:
                raise SCADataContractError(
                    f"{self.h5_path} has no trace-normalization provenance"
                )
            if stored_mode is not None and stored_mode != runtime_mode:
                raise SCADataContractError(
                    "Runtime/file trace-normalization mismatch: "
                    f"runtime={runtime_mode}, file={stored_mode}, "
                    f"path={self.h5_path}"
                )

            fixed_key_hex = input_file.attrs.get("sca_fixed_key_hex")
            self._fixed_key_override = (
                _hex_to_key_bytes(str(_decode_attr(fixed_key_hex)))
                if fixed_key_hex is not None and str(_decode_attr(fixed_key_hex)).strip()
                else None
            )

            if self.data_contract is not None:
                trace_contract = contract_active.get("trace_selection") or {}
                if not isinstance(trace_contract, Mapping):
                    raise SCADataContractError(
                        "contract.active.trace_selection must be a mapping"
                    )
                contract_length = trace_contract.get("output_length")
                if contract_length is not None and _strict_int(
                    contract_length,
                    field="contract.active.trace_selection.output_length",
                    minimum=1,
                ) != int(traces.shape[1]):
                    raise SCADataContractError(
                        "HDF5 trace length and data-contract output length disagree"
                    )

                trace_window_status = str(
                    trace_contract.get("verification_for_attack_byte", "")
                ).strip().lower()
                if trace_window_status not in {"verified", "full_source"}:
                    raise SCADataContractError(
                        f"{self.h5_path} trace window is not verified for attack "
                        f"byte {self._attack_byte}: {trace_window_status!r}"
                    )
                attr_trace_window_status = _first_attr(
                    input_file.attrs,
                    ("sca_trace_window_verification",),
                )
                if attr_trace_window_status is None:
                    raise SCADataContractError(
                        f"{self.h5_path} has no trace-window verification attribute"
                    )
                if (
                    str(attr_trace_window_status).strip().lower()
                    != trace_window_status
                ):
                    raise SCADataContractError(
                        "HDF5 trace-window attribute and canonical contract disagree"
                    )

                if contract_attack_byte != self._attack_byte:
                    raise SCADataContractError(
                        "Resolved file attack byte and canonical contract disagree"
                    )
                if contract_leakage_model != self._leakage_model:
                    raise SCADataContractError(
                        "Resolved file leakage model and canonical contract disagree"
                    )



            self._n = int(labels.shape[0])
            self._length = int(traces.shape[1])

        # logger.info(
        #     "ASCADDataset: file=%s split=%s n=%d length=%d metadata=%s "
        #     "attack_byte=%s leakage=%s classes=%d normalization=%s",
        #     self.h5_path.name,
        #     self.split,
        #     self._n,
        #     self._length,
        #     self._has_meta,
        #     self._attack_byte,
        #     self._leakage_model,
        #     self._num_classes,
        #     self._norm_cfg["mode"],
        # )

        if isinstance(cfg, Mapping) and self.data_contract:
            _log_data_contract(self.data_contract, cfg=cfg)

        # Load in the parent process so forked DataLoader workers share the
        # arrays copy-on-write instead of each reading HDF5 item by item.
        self._ensure_open()

    def unscaled_traces(self) -> np.ndarray:
        """The split's traces as normalised by its own policy, before any fitted scaler: what a
        scaler is fitted on (utils/trace_scaler.py)."""
        check_seal(self.h5_path)
        with h5py.File(self.h5_path, "r") as handle:
            raw = handle[self.group_name]["traces"][()]
        return np.ascontiguousarray(
            _normalize_trace(np.asarray(raw, dtype=np.float32), raw.dtype, self._norm_cfg), dtype=np.float32
        )

    def set_trace_scaler(self, scaler: Any) -> None:
        """Scale the traces with a scaler fitted on the training traces (mode fitted); the
        split is reloaded so the scaler is applied once, here and in every DataLoader worker."""
        if self._norm_cfg.get("mode") != "fitted":
            raise ValueError(
                f"{self.h5_path}: a fitted scaler needs trace_normalization.mode fitted, "
                f"the split was made with {self._norm_cfg.get('mode')!r}"
            )
        self._trace_scaler = scaler
        if self._traces is not None:
            self.close()
            self._ensure_open()

    def close(self) -> None:
        """Release the in-memory split; the next access reloads it."""
        self._traces = None
        self._labels = None
        self._meta = None
        self._h5 = None
        self._group = None
        self._pid = None

    def _ensure_open(self) -> None:
        """Load the whole split into memory once per process.

        A split is small (ASCAD: 50,000 x 700 int8 = 35 MB raw, 140 MB as
        normalised float32) and every epoch touches every row, so per-item
        HDF5 reads inside the loader workers dominated the epoch time.
        Normalisation is element-wise, so applying it to the whole matrix
        gives the same float32 values as the former per-trace path.
        """
        if self._traces is not None:
            return
        check_seal(self.h5_path)
        with h5py.File(self.h5_path, "r") as handle:
            group = handle[self.group_name]
            raw = group["traces"][()]
            labels = np.asarray(group["labels"][()])
            meta = group["metadata"][()] if self._has_meta else None
        traces = np.ascontiguousarray(
            _normalize_trace(np.asarray(raw, dtype=np.float32), raw.dtype, self._norm_cfg),
            dtype=np.float32,
        )
        if self._trace_scaler is not None:
            traces = np.ascontiguousarray(self._trace_scaler.transform(traces), dtype=np.float32)
        finite_rows = np.isfinite(traces).all(axis=1)
        if not bool(finite_rows.all()):
            raise SCADataContractError(
                "Non-finite trace values after normalization at index "
                f"{int(np.flatnonzero(~finite_rows)[0])}"
            )
        self._traces = traces
        self._labels = labels
        self._meta = meta
        self._pid = os.getpid()

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> Any:
        if self._norm_cfg.get("mode") == "fitted" and self._trace_scaler is None:
            raise SCADataContractError(
                f"{self.h5_path}: trace_normalization.mode fitted, but no scaler is attached: the traces would be "
                "read unscaled. Attach the run's scaler first (utils/trace_scaler.scale_training for a training "
                "loop, scale_test for the test; ASCADDataset.set_trace_scaler)"
            )
        self._ensure_open()
        item_index = int(index)
        if item_index < 0:
            item_index += self._n
        if item_index < 0 or item_index >= self._n:
            raise IndexError(item_index)
        if self._traces is None or self._labels is None:
            raise RuntimeError("Split arrays are not loaded")

        trace = self._traces[item_index]
        label = int(self._labels[item_index])
        if not 0 <= label < self._num_classes:
            raise SCADataContractError(
                f"Label {label} at index {item_index} is outside [0, "
                f"{self._num_classes - 1}]"
            )

        x = torch.from_numpy(trace).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)

        if not self.return_metadata:
            return x, y
        if not self._has_meta:
            return x, y, {}

        metadata_row = self._meta[item_index]
        metadata = _meta_void_to_dict(metadata_row)
        if self._variant == "fixed_key" and self._fixed_key_override is not None:
            metadata["key"] = np.asarray(
                self._fixed_key_override,
                dtype=np.uint8,
            )

        if self.require_metadata and (
            "plaintext" not in metadata or "key" not in metadata
        ):
            raise KeyError(
                "metadata must include plaintext and key for rank/GE/LDD metrics"
            )
        return x, y, metadata

    def __getstate__(self) -> dict[str, Any]:
        """Drop process-local HDF5 handles before DataLoader worker pickling."""
        self.close()
        state = dict(self.__dict__)
        state["_h5"] = None
        state["_group"] = None
        state["_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def load_split_dataset(
    cfg: Mapping[str, Any],
    filename_or_path: str,
    split: str = "profiling",
    return_metadata: bool = True,
) -> Dataset:
    """Load a generated split file by filename or explicit path."""
    model_name = cfg.get("model_name")
    if model_name is None or not str(model_name).strip():
        raise ValueError("cfg['model_name'] is required")

    path = Path(str(filename_or_path))
    if not path.is_absolute():
        paths = cfg.get("paths") or {}
        if not isinstance(paths, Mapping) or not paths.get("data"):
            raise KeyError("cfg['paths']['data'] is required")
        data_directory = resolve_path(
            paths["data"],
            cfg,
            model_name=str(model_name),
        )
        path = Path(data_directory) / str(filename_or_path)

    return ASCADDataset(
        path,
        split=split,
        return_metadata=return_metadata,
        require_metadata=_require_metadata(cfg),
        cfg=cfg,
    )


__all__ = [
    "ASCADDataset",
    "DATA_CONTRACT_VERSION",
    "SCADataContractError",
    "check_file_exists",
    "compute_labels_from_meta",
    "extract_traces",
    "generate",
    "load_split_dataset",
    "resolve_active_data_config",
    "sca_collate_fn",
]
