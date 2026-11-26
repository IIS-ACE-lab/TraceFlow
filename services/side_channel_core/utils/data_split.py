# services/side_channel_core/utils/data_split.py
"""
Contract-preserving ASCAD train/validation/test splitting and DataLoader helpers.

Scientific lifecycle
--------------------
The complete configuration-specific HDF5 is generated first by
``src.sca.ascad_data_loader.generate``::

    source ASCAD HDF5
        + resolved attack byte
        + resolved leakage model
        + resolved trace selection
        -> paths.data_out

This module then derives row-only split artifacts from that verified file::

    train.h5 <- data_out/Profiling_traces[:]
    val.h5   <- data_out/Attack_traces[validation_range]
    test.h5  <- data_out/Attack_traces[test_range]

Hard contracts
--------------
- Split generation never reads labels from the original source database.
- Split generation never recomputes labels.
- Split generation never applies target_range a second time.
- Traces, labels, metadata, source_indices, and desync_offsets are copied
  exactly from the verified complete configuration-specific HDF5.
- Validation and test ranges are disjoint and relative to the generated
  Attack_traces population.
- Every split receives a derived canonical data contract linked to the parent
  contract fingerprint.
- Split files are written and verified before publication.
- Runtime search spaces must already be scalar. Lists are rejected rather than
  silently selecting their first value.
- Existing split files are reused only when their parent fingerprint, row
  derivation, dataset hashes, active SCA target, and runtime configuration all
  match.

Cross-validation
----------------
When cross_validation.enabled is true, validation_range is ignored and only
train.h5 and test.h5 are prepared. Fold validation comes from profiling indices;
an existing val.h5 is never loaded by this module in CV mode. With CV disabled,
the ordinary disjoint attack-derived validation/test split above is retained.

Author: Shraddha Mishra
"""


from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import h5py
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.sca.ascad_data_loader import (
    ASCADDataset,
    DATA_CONTRACT_VERSION,
    SCADataContractError,
    generate,
    load_split_dataset,
    profiling_indices,
    resolve_active_data_config,
    sca_collate_fn,
)
from src.sca.dataset_types import profiling_split_reproducible, resolved_variant
from src.sca.leakage_models import get_num_classes, normalize_leakage_model
from utils.dataloader_utils import _get_dataloader_cfg
from utils.seal import TamperedArtifactError, check as check_seal, seal_h5_file
from utils.h5_utils import inspect_and_log_h5
from utils.logger import get_logger
from utils.path_utils import resolve_path
from utils.trace_scaler import scale_test, scale_training
from utils.seed import make_torch_generator, make_worker_init_fn, set_seed

logger = get_logger("data_split")

SPLIT_DERIVATION_VERSION = "1.1"
_REQUIRED_DATASETS = (
    "traces",
    "labels",
    "metadata",
    "source_indices",
    "desync_offsets",
)
_DYNAMIC_CONTRACT_ATTRS = frozenset(
    {
        "sca_contract_version",
        "sca_data_contract_json",
        "sca_data_contract_sha256",
        "sca_verification_json",
        "sca_label_integrity_verified",
        "sca_trace_selection_verified",
        "sca_metadata_mapping_verified",
    }
)


class SCASplitContractError(SCADataContractError):
    """Raised when a derived train/validation/test split violates its contract."""


@dataclass(frozen=True)
class SplitSpec:
    """One deterministic row-subset derivation from the complete HDF5."""

    name: str
    group_name: str
    start: int
    end: int
    output_path: Path

    @property
    def row_count(self) -> int:
        return int(self.end - self.start)

    @property
    def loader_split(self) -> str:
        return "profiling" if self.group_name == "Profiling_traces" else "attack"


# =============================================================================
# Strict configuration helpers
# =============================================================================
def _strict_bool(value: Any, *, field: str, default: Optional[bool] = None) -> bool:
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
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, got boolean {value!r}")

    if isinstance(value, (list, tuple, set, Mapping)):
        raise TypeError(
            f"{field} must be one resolved scalar integer, got unresolved "
            f"{type(value).__name__}: {value!r}"
        )

    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise TypeError(
                f"{field} must be one resolved scalar integer, got array "
                f"shape={value.shape}"
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
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be numeric, got boolean {value!r}")
    if isinstance(value, (list, tuple, set, Mapping)):
        raise TypeError(f"{field} must be one scalar number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {result!r}")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{field} must be {operator} {minimum}, got {result}")
    return result


def _get_data_cfg(cfg: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Return the same active dataset selected by ascad_data_loader."""
    return resolve_active_data_config(cfg)


def _dataset_alias(data_cfg: Mapping[str, Any]) -> str:
    raw = data_cfg.get("alias") or data_cfg.get("name")
    if raw is None or not str(raw).strip():
        raise KeyError("Active dataset must define name or alias")
    return str(raw).strip()


def _prepare_runtime_cfg(cfg: MutableMapping[str, Any], *, model_name: str) -> None:
    """Inject non-ambiguous path context without changing scientific fields."""
    if not isinstance(cfg, MutableMapping):
        raise TypeError(f"cfg must be mutable mapping, got {type(cfg).__name__}")

    normalized_model_name = str(model_name).strip()
    if not normalized_model_name:
        raise ValueError("model_name must not be empty")

    existing_model_name = cfg.get("model_name")
    if existing_model_name is not None and str(existing_model_name) != normalized_model_name:
        raise ValueError(
            "model_name argument conflicts with cfg['model_name']: "
            f"argument={normalized_model_name!r}, cfg={existing_model_name!r}"
        )
    cfg["model_name"] = normalized_model_name

    data_cfg = _get_data_cfg(cfg)
    alias = _dataset_alias(data_cfg)
    existing_alias = cfg.get("dataset_alias")
    if existing_alias is not None and str(existing_alias).strip() != alias:
        raise ValueError(
            "Resolved active dataset alias conflicts with cfg['dataset_alias']: "
            f"resolved={alias!r}, cfg={existing_alias!r}"
        )
    cfg["dataset_alias"] = alias


def _get_attack_byte_from_cfg(cfg: Mapping[str, Any]) -> int:
    sca = cfg.get("sca") or {}
    if not isinstance(sca, Mapping):
        raise TypeError("cfg['sca'] must be a mapping")
    if "attack_byte" not in sca:
        raise KeyError("cfg['sca']['attack_byte'] is required")
    return _strict_int(
        sca["attack_byte"],
        field="sca.attack_byte",
        minimum=0,
        maximum=15,
    )


def _get_runtime_leakage_model_from_cfg(cfg: Mapping[str, Any]) -> str:
    sca = cfg.get("sca") or {}
    if not isinstance(sca, Mapping):
        raise TypeError("cfg['sca'] must be a mapping")
    if "leakage_model" not in sca:
        raise KeyError("cfg['sca']['leakage_model'] is required")

    raw = sca["leakage_model"]
    if isinstance(raw, (list, tuple, set, Mapping)):
        raise TypeError(
            "sca.leakage_model must be one resolved scalar value, got "
            f"unresolved {type(raw).__name__}: {raw!r}"
        )
    if isinstance(raw, np.ndarray):
        if raw.shape != ():
            raise TypeError(
                "sca.leakage_model must be scalar, got array "
                f"shape={raw.shape}"
            )
        raw = raw.item()
    return normalize_leakage_model(raw)


def _get_variant(cfg: Mapping[str, Any]) -> str:
    sca = cfg.get("sca") or {}
    if not isinstance(sca, Mapping):
        raise TypeError("cfg['sca'] must be a mapping")
    raw = resolved_variant(cfg)   # the database type gives it (sca/dataset_types.py)
    if raw is None or not str(raw).strip():
        return "unknown"
    if isinstance(raw, (list, tuple, set, Mapping)):
        raise TypeError("sca.variant must be one resolved scalar value")
    normalized = str(raw).strip().lower()
    if normalized not in {"fixed_key", "random_key", "variable_key"}:
        raise ValueError(
            "sca.variant must be fixed_key, random_key, or variable_key, "
            f"got {raw!r}"
        )
    return normalized


def _require_metadata(cfg: Mapping[str, Any]) -> bool:
    sca = cfg.get("sca") or {}
    if not isinstance(sca, Mapping):
        raise TypeError("cfg['sca'] must be a mapping")
    return _strict_bool(
        sca.get("require_metadata", True),
        field="sca.require_metadata",
        default=True,
    )


def _reject_legacy_mutating_loader_options(data_cfg: Mapping[str, Any]) -> None:
    """Reject legacy settings that mutate scientific artifacts during load."""
    if _strict_bool(
        data_cfg.get("autofix_labels_on_load", False),
        field="data.autofix_labels_on_load",
        default=False,
    ):
        raise ValueError(
            "data.autofix_labels_on_load is no longer supported. Persisted "
            "SCA labels are immutable; regenerate paths.data_out and its "
            "derived split files instead of rewriting labels during loading."
        )
    if _strict_bool(
        data_cfg.get("verify_labels_on_load", False),
        field="data.verify_labels_on_load",
        default=False,
    ):
        raise ValueError(
            "data.verify_labels_on_load is obsolete. Full label integrity is "
            "verified before paths.data_out is published, and each derived "
            "split is hash-verified against that parent before use."
        )


def _validate_range(raw: Any, *, name: str) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise TypeError(f"{name} must be [start, end], got {raw!r}")
    start = _strict_int(raw[0], field=f"{name}[0]", minimum=0)
    end = _strict_int(raw[1], field=f"{name}[1]", minimum=0)
    if end <= start:
        raise ValueError(f"{name} end must be greater than start, got {raw!r}")
    return start, end


def _target_selection_for_runtime(
    data_cfg: Mapping[str, Any],
    *,
    attack_byte: int,
) -> list[int]:
    per_byte = data_cfg.get("target_range_by_attack_byte")
    if per_byte is not None:
        if not isinstance(per_byte, Mapping):
            raise TypeError("data.target_range_by_attack_byte must be a mapping")
        for key in (attack_byte, str(attack_byte), f"0x{attack_byte:02X}"):
            if key in per_byte:
                selected = per_byte[key]
                if not isinstance(selected, (list, tuple)):
                    raise TypeError(
                        "target_range_by_attack_byte entries must be list/tuple"
                    )
                return [
                    _strict_int(
                        value,
                        field=(
                            "data.target_range_by_attack_byte"
                            f"[{key!r}][{index}]"
                        ),
                        minimum=0,
                    )
                    for index, value in enumerate(selected)
                ]

    generic = data_cfg.get("target_range", data_cfg.get("target_points"))
    if not isinstance(generic, (list, tuple)):
        raise TypeError(
            "Active dataset must define target_range/target_points as a list or tuple"
        )
    return [
        _strict_int(value, field=f"data.target_selection[{index}]", minimum=0)
        for index, value in enumerate(generic)
    ]


def _canonical_normalization(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The entry's normalisation policy exactly as the parent contract records it: the loader's
    own validator (sca/ascad_data_loader.py), so the two can never disagree."""
    from sca.ascad_data_loader import _normalization_from_data_cfg

    return _normalization_from_data_cfg(data_cfg)


def _resolve_training_batch_size(trainer_cfg: Mapping[str, Any]) -> int:
    raw = (trainer_cfg or {}).get("batch_size", 64)
    return _strict_int(
        raw,
        field="trainer.batch_size",
        minimum=1,
    )


def _resolve_loader_seed(
    cfg: MutableMapping[str, Any],
    seed: Optional[int],
) -> int:
    raw = set_seed(cfg) if seed is None else seed
    return _strict_int(raw, field="dataloader seed")


# =============================================================================
# Paths and expected split specifications
# =============================================================================
def _complete_dataset_path(cfg: Mapping[str, Any], *, model_name: str) -> Path:
    paths = cfg.get("paths") or {}
    if not isinstance(paths, Mapping) or not paths.get("data_out"):
        raise KeyError("cfg['paths']['data_out'] is required")
    return Path(
        resolve_path(
            paths["data_out"],
            cfg,
            model_name=model_name,
        )
    )


def _split_dir(cfg: Mapping[str, Any], *, model_name: str) -> Path:
    paths = cfg.get("paths") or {}
    if not isinstance(paths, Mapping) or not paths.get("data"):
        raise KeyError("cfg['paths']['data'] is required")
    directory = Path(
        resolve_path(
            paths["data"],
            cfg,
            model_name=model_name,
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _split_paths(
    cfg: Mapping[str, Any],
    *,
    model_name: str,
) -> Tuple[Path, Path, Path]:
    directory = _split_dir(cfg, model_name=model_name)
    return directory / "train.h5", directory / "val.h5", directory / "test.h5"


def _cv_enabled(cfg: Mapping[str, Any]) -> bool:
    cv_cfg = cfg.get("cross_validation", {})
    if not isinstance(cv_cfg, Mapping):
        raise TypeError("cfg['cross_validation'] must be a mapping")
    return _strict_bool(
        cv_cfg.get("enabled", False),
        field="cross_validation.enabled",
        default=False,
    )


def _require_cv_enabled(cfg: Mapping[str, Any], *, caller: str) -> None:
    if not _cv_enabled(cfg):
        raise ValueError(f"{caller} requires cross_validation.enabled=true")


def _validate_test_range(
    data_cfg: Mapping[str, Any], *, attack_count: int,
) -> tuple[int, int]:
    start, end = _validate_range(data_cfg.get("test_range"), name="data.test_range")
    if end > attack_count:
        raise ValueError(
            "data.test_range exceeds the generated Attack_traces pool: "
            f"range=[{start},{end}), available={attack_count}"
        )
    return start, end


def _validate_evaluation_ranges(
    data_cfg: Mapping[str, Any],
    *,
    attack_count: int,
) -> tuple[int, int, int, int]:
    """Validate disjoint validation/test coordinates in the generated attack pool."""
    validation_start, validation_end = _validate_range(
        data_cfg.get("validation_range"),
        name="data.validation_range",
    )
    test_start, test_end = _validate_test_range(
        data_cfg, attack_count=attack_count,
    )

    if validation_end > attack_count:
        raise ValueError(
            "data.validation_range exceeds the generated Attack_traces pool: "
            f"range=[{validation_start},{validation_end}), available={attack_count}"
        )
    if not (validation_end <= test_start or test_end <= validation_start):
        overlap_start = max(validation_start, test_start)
        overlap_end = min(validation_end, test_end)
        raise ValueError(
            "Validation and test ranges overlap in generated Attack_traces: "
            f"validation=[{validation_start},{validation_end}), "
            f"test=[{test_start},{test_end}), "
            f"overlap=[{overlap_start},{overlap_end})"
        )
    return validation_start, validation_end, test_start, test_end


def _build_split_specs(
    cfg: Mapping[str, Any],
    *,
    model_name: str,
    profiling_count: int,
    attack_count: int,
) -> tuple[SplitSpec, ...]:
    data_cfg = _get_data_cfg(cfg)
    cv_enabled = _cv_enabled(cfg)
    if cv_enabled:
        test_start, test_end = _validate_test_range(data_cfg, attack_count=attack_count)
    else:
        validation_start, validation_end, test_start, test_end = (
            _validate_evaluation_ranges(data_cfg, attack_count=attack_count)
        )

    train_path, validation_path, test_path = _split_paths(
        cfg,
        model_name=model_name,
    )
    specs = [
        SplitSpec(
            name="train",
            group_name="Profiling_traces",
            start=0,
            end=int(profiling_count),
            output_path=train_path,
        ),
        SplitSpec(
            name="test",
            group_name="Attack_traces",
            start=test_start,
            end=test_end,
            output_path=test_path,
        ),
    ]
    if not cv_enabled:
        specs.insert(1, SplitSpec(
            name="validation",
            group_name="Attack_traces",
            start=validation_start,
            end=validation_end,
            output_path=validation_path,
        ))
    return tuple(specs)


# =============================================================================
# Canonical contract and hashing helpers
# =============================================================================
def _decode_attr(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
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


def _serialize_contract(contract: Mapping[str, Any]) -> tuple[str, str]:
    serialized = json.dumps(
        _json_safe(contract),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return serialized, fingerprint


def _read_contract_fingerprint(h5_file: h5py.File, *, path: Path) -> str:
    raw = h5_file.attrs.get("sca_data_contract_sha256")
    if raw is None:
        raise SCASplitContractError(
            f"{path} is missing sca_data_contract_sha256"
        )
    digest = str(_decode_attr(raw)).strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise SCASplitContractError(
            f"{path} has invalid sca_data_contract_sha256={digest!r}"
        )
    return digest


def _dataset_hash(
    dataset: h5py.Dataset,
    *,
    start: int = 0,
    end: Optional[int] = None,
    chunk_rows: int = 4096,
) -> str:
    """Hash one HDF5 dataset slice with dtype/shape domain separation."""
    if dataset.ndim < 1:
        raise ValueError(f"Dataset {dataset.name} must have a row dimension")
    row_end = int(dataset.shape[0]) if end is None else int(end)
    row_start = int(start)
    if row_start < 0 or row_end < row_start or row_end > int(dataset.shape[0]):
        raise ValueError(
            f"Invalid hash slice [{row_start}:{row_end}] for {dataset.name} "
            f"with rows={dataset.shape[0]}"
        )

    digest = hashlib.sha256()
    logical_shape = (row_end - row_start,) + tuple(dataset.shape[1:])
    digest.update(str(dataset.dtype).encode("utf-8"))
    digest.update(json.dumps(logical_shape).encode("ascii"))

    step = max(1, int(chunk_rows))
    for chunk_start in range(row_start, row_end, step):
        chunk_end = min(chunk_start + step, row_end)
        values = np.asarray(dataset[chunk_start:chunk_end])
        if values.dtype.hasobject:
            raise TypeError(
                f"Object/vlen dataset hashing is unsupported: {dataset.name}"
            )
        digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    return digest.hexdigest()


def _hash_parent_slice(
    parent_file: h5py.File,
    *,
    spec: SplitSpec,
    chunk_rows: int,
) -> dict[str, str]:
    group = parent_file[spec.group_name]
    return {
        name: _dataset_hash(
            group[name],
            start=spec.start,
            end=spec.end,
            chunk_rows=chunk_rows,
        )
        for name in _REQUIRED_DATASETS
    }


def _hash_output_group(
    split_file: h5py.File,
    *,
    spec: SplitSpec,
    chunk_rows: int,
) -> dict[str, str]:
    group = split_file[spec.group_name]
    return {
        name: _dataset_hash(
            group[name],
            chunk_rows=chunk_rows,
        )
        for name in _REQUIRED_DATASETS
    }


def _write_split_contract_attrs(
    output_file: h5py.File,
    *,
    contract: Mapping[str, Any],
    spec: SplitSpec,
    parent_path: Path,
    parent_fingerprint: str,
) -> str:
    serialized, fingerprint = _serialize_contract(contract)
    verification = contract.get("verification") or {}
    if not isinstance(verification, Mapping):
        raise SCASplitContractError("contract.verification must be a mapping")

    output_file.attrs["sca_contract_version"] = DATA_CONTRACT_VERSION
    output_file.attrs["sca_data_contract_json"] = serialized
    output_file.attrs["sca_data_contract_sha256"] = fingerprint
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

    output_file.attrs["sca_split_derivation_version"] = SPLIT_DERIVATION_VERSION
    output_file.attrs["sca_split_name"] = spec.name
    output_file.attrs["sca_split_group"] = spec.group_name
    output_file.attrs["sca_split_start"] = int(spec.start)
    output_file.attrs["sca_split_end_exclusive"] = int(spec.end)
    output_file.attrs["sca_split_row_count"] = int(spec.row_count)
    output_file.attrs["sca_split_parent_path"] = str(parent_path.resolve())
    output_file.attrs["sca_split_parent_contract_sha256"] = parent_fingerprint
    output_file.attrs["sca_split_labels_recomputed"] = False
    output_file.attrs["sca_split_trace_transformed"] = False
    output_file.attrs["sca_split_metadata_transformed"] = False
    output_file.attrs["sca_split_copy_verified"] = bool(
        verification.get("split_copy_verified", False)
    )
    return fingerprint


def _derived_contract(
    *,
    parent_contract: Mapping[str, Any],
    parent_path: Path,
    parent_fingerprint: str,
    spec: SplitSpec,
    dataset_hashes: Mapping[str, str],
) -> dict[str, Any]:
    contract = copy.deepcopy(dict(parent_contract))
    active = contract.get("active")
    if not isinstance(active, MutableMapping):
        raise SCASplitContractError("Parent contract.active must be a mapping")

    if spec.group_name == "Profiling_traces":
        active["profiling_count"] = spec.row_count
        active["attack_count"] = 0
    else:
        active["profiling_count"] = 0
        active["attack_count"] = spec.row_count

    contract["derivation"] = {
        "version": SPLIT_DERIVATION_VERSION,
        "kind": "verified_row_subset",
        "split_name": spec.name,
        "parent_path": str(parent_path.resolve()),
        "parent_contract_sha256": parent_fingerprint,
        "source_group": spec.group_name,
        "source_row_start": int(spec.start),
        "source_row_end_exclusive": int(spec.end),
        "row_count": int(spec.row_count),
        "range_coordinate_system": "configuration_specific_parent_group",
        "labels_recomputed": False,
        "traces_transformed": False,
        "metadata_transformed": False,
        "normalization_applied_during_split": False,
    }
    contract["verification"] = {
        "status": "verified",
        "performed_before_publish": True,
        # These scientific properties are inherited from the independently
        # verified parent and preserved by exact-copy verification.
        "label_integrity_verified": True,
        "trace_selection_verified": True,
        "metadata_mapping_verified": True,
        "parent_contract_verified": True,
        "split_copy_verified": True,
        "dataset_hashes_verified": True,
        "dataset_sha256": dict(dataset_hashes),
    }
    contract["output"] = {
        "path": str(spec.output_path.resolve()),
        "layout": "grouped_single",
        "groups": [spec.group_name],
        "split_name": spec.name,
        "row_count": int(spec.row_count),
        "provenance_datasets": ["source_indices", "desync_offsets"],
    }
    return contract


# =============================================================================
# Complete configuration-specific dataset validation/reuse
# =============================================================================
def _validate_parent_contract_against_runtime(
    *,
    cfg: Mapping[str, Any],
    parent_path: Path,
) -> tuple[dict[str, Any], str, int, int]:
    """Validate the complete data_out file and return its contract and counts."""
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)

    # ASCADDataset independently validates canonical contract, active target,
    # class count, normalization, group structure, and row-level provenance.
    profiling_dataset = ASCADDataset(
        parent_path,
        split="profiling",
        return_metadata=False,
        require_metadata=_require_metadata(cfg),
        cfg=cfg,
    )
    try:
        parent_contract = copy.deepcopy(profiling_dataset.data_contract)
        profiling_count = len(profiling_dataset)
    finally:
        profiling_dataset.close()

    attack_dataset = ASCADDataset(
        parent_path,
        split="attack",
        return_metadata=False,
        require_metadata=_require_metadata(cfg),
        cfg=cfg,
    )
    try:
        attack_contract = attack_dataset.data_contract
        attack_count = len(attack_dataset)
    finally:
        attack_dataset.close()

    if not isinstance(parent_contract, dict) or not isinstance(attack_contract, dict):
        raise SCASplitContractError(
            f"{parent_path} must contain a canonical data contract"
        )
    if parent_contract != attack_contract:
        raise SCASplitContractError(
            "Profiling and attack views returned different parent contracts"
        )

    data_cfg = _get_data_cfg(cfg)
    active = parent_contract.get("active")
    source = parent_contract.get("source")
    verification = parent_contract.get("verification")
    if not isinstance(active, Mapping):
        raise SCASplitContractError("Parent contract.active must be a mapping")
    if not isinstance(source, Mapping):
        raise SCASplitContractError("Parent contract.source must be a mapping")
    if not isinstance(verification, Mapping):
        raise SCASplitContractError("Parent contract.verification must be a mapping")

    attack_byte = _get_attack_byte_from_cfg(cfg)
    leakage_model = _get_runtime_leakage_model_from_cfg(cfg)
    variant = _get_variant(cfg)
    profiling_range = _validate_range(
        data_cfg.get("profiling_range"),
        name="data.profiling_range",
    )
    attack_range = _validate_range(
        data_cfg.get("attack_range"),
        name="data.attack_range",
    )
    target_selection = _target_selection_for_runtime(
        data_cfg,
        attack_byte=attack_byte,
    )
    expected_normalization = _canonical_normalization(data_cfg)

    if int(active.get("attack_byte", -1)) != attack_byte:
        raise SCASplitContractError("Parent contract attack byte is stale")
    if normalize_leakage_model(active.get("leakage_model")) != leakage_model:
        raise SCASplitContractError("Parent contract leakage model is stale")
    if int(active.get("num_classes", -1)) != get_num_classes(leakage_model):
        raise SCASplitContractError("Parent contract class count is stale")
    if variant != "unknown" and str(active.get("variant", "")).lower() != variant:
        raise SCASplitContractError("Parent contract variant is stale")
    if str(active.get("dataset_alias", "")) != _dataset_alias(data_cfg):
        raise SCASplitContractError("Parent contract dataset alias is stale")
    expected_profiling = profiling_indices(data_cfg)     # profiling_range, or the rows its profiling_split trains on
    # An unseeded profiling_split (random_state null) draws new rows on every call: the parent's own record of the
    # rows it drew is checked instead (their count, order and range), never compared with a fresh draw.
    reproducible = profiling_split_reproducible(data_cfg)
    if int(active.get("profiling_count", -1)) != int(expected_profiling.size):
        raise SCASplitContractError("Parent profiling count does not match configuration")
    if int(active.get("attack_count", -1)) != attack_range[1] - attack_range[0]:
        raise SCASplitContractError("Parent attack count does not match configuration")

    trace_selection = active.get("trace_selection")
    if not isinstance(trace_selection, Mapping):
        raise SCASplitContractError("Parent contract trace_selection is missing")
    configured_selection = trace_selection.get("configured")
    if list(configured_selection or []) != target_selection:
        raise SCASplitContractError(
            "Parent trace selection does not match the current runtime configuration"
        )
    if active.get("trace_normalization") != expected_normalization:
        raise SCASplitContractError(
            "Parent trace normalization does not match the runtime configuration"
        )

    expected_source_path = Path(
        resolve_path(
            data_cfg["database"],
            cfg,
            model_name=str(cfg["model_name"]),
        )
    ).resolve()
    contract_source_path = Path(str(source.get("path", ""))).resolve()
    if contract_source_path != expected_source_path:
        raise SCASplitContractError(
            "Parent source path does not match active data.database: "
            f"contract={contract_source_path}, expected={expected_source_path}"
        )

    if str(verification.get("status", "")).lower() != "verified":
        raise SCASplitContractError("Parent contract is not verified")
    for flag in (
        "label_integrity_verified",
        "trace_selection_verified",
        "metadata_mapping_verified",
    ):
        if verification.get(flag) is not True:
            raise SCASplitContractError(f"Parent contract has {flag}=false/missing")

    check_seal(parent_path)  # utils/seal.py
    with h5py.File(parent_path, "r") as parent_file:
        parent_fingerprint = _read_contract_fingerprint(
            parent_file,
            path=parent_path,
        )
        for group_name, expected_range, expected_indices in (
            ("Profiling_traces", profiling_range, expected_profiling),
            ("Attack_traces", attack_range, np.arange(attack_range[0], attack_range[1], dtype=np.int64)),
        ):
            if group_name not in parent_file:
                raise SCASplitContractError(
                    f"Parent is missing {group_name}"
                )
            group = parent_file[group_name]
            for dataset_name in _REQUIRED_DATASETS:
                if dataset_name not in group:
                    raise SCASplitContractError(
                        f"Parent is missing {group_name}/{dataset_name}"
                    )
            observed_indices = np.asarray(group["source_indices"][:], dtype=np.int64)
            if group_name == "Profiling_traces" and not reproducible:
                drawn_ok = (observed_indices.size == expected_indices.size
                            and bool(np.all(observed_indices[1:] > observed_indices[:-1]))
                            and int(observed_indices[0]) >= expected_range[0] and int(observed_indices[-1]) < expected_range[1])
                if not drawn_ok:
                    raise SCASplitContractError(
                        f"Parent {group_name}/source_indices are not {expected_indices.size} distinct ascending rows of "
                        f"[{expected_range[0]}:{expected_range[1]}] (its unseeded profiling_split)"
                    )
            elif not np.array_equal(observed_indices, expected_indices):
                raise SCASplitContractError(
                    f"Parent {group_name}/source_indices does not match configured "
                    f"range [{expected_range[0]}:{expected_range[1]}]"
                    + (" and its profiling_split" if group_name == "Profiling_traces"
                       and (data_cfg.get("profiling_split") or {}).get("enabled") is True else "")
                )
            desync_key = (
                "profiling_desync"
                if group_name == "Profiling_traces"
                else "attack_desync"
            )
            expected_desync = _strict_int(
                data_cfg.get(desync_key, 0),
                field=f"data.{desync_key}",
                minimum=0,
            )
            offsets = np.asarray(group["desync_offsets"][:], dtype=np.int64)
            if offsets.size != observed_indices.size:
                raise SCASplitContractError(
                    f"Parent {group_name}/desync_offsets has wrong row count"
                )
            if np.any(offsets < 0) or np.any(offsets > expected_desync):
                raise SCASplitContractError(
                    f"Parent {group_name}/desync_offsets exceeds configured "
                    f"range [0,{expected_desync}]"
                )

    return parent_contract, parent_fingerprint, profiling_count, attack_count


def _generate_data_out(data_cfg: Mapping[str, Any]) -> bool:
    """data.generate_data_out: build missing/stale artifacts (default true)."""
    return _strict_bool(
        data_cfg.get("generate_data_out", True),
        field="data.generate_data_out",
        default=True,
    )


def _ensure_complete_dataset(
    cfg: MutableMapping[str, Any],
    *,
    model_name: str,
) -> tuple[Path, dict[str, Any], str, int, int]:
    data_cfg = _get_data_cfg(cfg)
    parent_path = _complete_dataset_path(cfg, model_name=model_name)
    generate_allowed = _generate_data_out(data_cfg)
    force_regenerate = _strict_bool(
        data_cfg.get("force_regenerate_dataset", False),
        field="data.force_regenerate_dataset",
        default=False,
    )

    if parent_path.is_file() and not force_regenerate:
        try:
            contract, fingerprint, profiling_count, attack_count = (
                _validate_parent_contract_against_runtime(
                    cfg=cfg,
                    parent_path=parent_path,
                )
            )
            # logger.info(
            #     "Reusing verified configuration-specific dataset: %s",
            #     parent_path,
            # )
            return (
                parent_path,
                contract,
                fingerprint,
                profiling_count,
                attack_count,
            )
        except (PermissionError, TamperedArtifactError):
            raise  # a changed file is evidence, not a stale file to regenerate
        except Exception as exc:
            if not generate_allowed:
                raise SCASplitContractError(
                    f"data.generate_data_out=false but paths.data_out is stale or "
                    f"invalid: {parent_path} | reason={exc}"
                ) from exc
            logger.warning(
                "Existing data_out is stale or invalid and will be regenerated: "
                "%s | reason=%s",
                parent_path,
                exc,
            )

    if not generate_allowed:
        raise SCASplitContractError(
            "data.generate_data_out=false but paths.data_out does not exist"
            + (" or data.force_regenerate_dataset=true" if force_regenerate else "")
            + f": {parent_path}"
        )
    generated_path = generate(cfg)
    if generated_path.resolve() != parent_path.resolve():
        raise RuntimeError(
            "ascad_data_loader.generate returned an unexpected path: "
            f"returned={generated_path}, expected={parent_path}"
        )
    contract, fingerprint, profiling_count, attack_count = (
        _validate_parent_contract_against_runtime(
            cfg=cfg,
            parent_path=parent_path,
        )
    )
    return parent_path, contract, fingerprint, profiling_count, attack_count


# =============================================================================
# Split writing, verification, and publication
# =============================================================================
def _copy_root_attrs(parent_file: h5py.File, output_file: h5py.File) -> None:
    for key, value in parent_file.attrs.items():
        if str(key) in _DYNAMIC_CONTRACT_ATTRS:
            continue
        output_file.attrs[key] = value


def _copy_group_attrs(parent_group: h5py.Group, output_group: h5py.Group) -> None:
    for key, value in parent_group.attrs.items():
        output_group.attrs[key] = value


def _dataset_chunks(source: h5py.Dataset, *, rows: int, chunk_rows: int) -> tuple[int, ...]:
    first = min(max(1, int(chunk_rows)), max(1, int(rows)))
    if source.ndim == 1:
        return (first,)
    return (first,) + tuple(int(dim) for dim in source.shape[1:])


def _copy_dataset_slice(
    source: h5py.Dataset,
    destination_group: h5py.Group,
    *,
    name: str,
    start: int,
    end: int,
    chunk_rows: int,
) -> h5py.Dataset:
    rows = int(end - start)
    shape = (rows,) + tuple(source.shape[1:])
    destination = destination_group.create_dataset(
        name,
        shape=shape,
        dtype=source.dtype,
        chunks=_dataset_chunks(source, rows=rows, chunk_rows=chunk_rows),
    )
    for attr_name, attr_value in source.attrs.items():
        destination.attrs[attr_name] = attr_value

    for output_start in range(0, rows, chunk_rows):
        output_end = min(output_start + chunk_rows, rows)
        source_start = start + output_start
        source_end = start + output_end
        destination[output_start:output_end] = source[source_start:source_end]
    return destination


def _write_split_file(
    *,
    parent_path: Path,
    parent_contract: Mapping[str, Any],
    parent_fingerprint: str,
    spec: SplitSpec,
    temporary_path: Path,
    chunk_rows: int,
) -> None:
    if spec.row_count <= 0:
        raise ValueError(f"Split {spec.name} must contain at least one row")

    with h5py.File(parent_path, "r") as parent_file:
        if spec.group_name not in parent_file:
            raise KeyError(
                f"Parent {parent_path} is missing group {spec.group_name}"
            )
        parent_group = parent_file[spec.group_name]
        available = int(parent_group["labels"].shape[0])
        if spec.start < 0 or spec.end > available or spec.end <= spec.start:
            raise ValueError(
                f"Invalid {spec.name} range [{spec.start}:{spec.end}] for "
                f"{spec.group_name} rows={available}"
            )
        for dataset_name in _REQUIRED_DATASETS:
            if dataset_name not in parent_group:
                raise KeyError(
                    f"Parent is missing {spec.group_name}/{dataset_name}"
                )

        with h5py.File(temporary_path, "x") as output_file:
            _copy_root_attrs(parent_file, output_file)
            output_group = output_file.create_group(spec.group_name)
            _copy_group_attrs(parent_group, output_group)

            output_group.attrs["sca_split_name"] = spec.name
            output_group.attrs["sca_split_parent"] = str(parent_path.resolve())
            output_group.attrs["sca_split_start"] = int(spec.start)
            output_group.attrs["sca_split_end_exclusive"] = int(spec.end)
            output_group.attrs["sca_split_labels_recomputed"] = False

            for dataset_name in _REQUIRED_DATASETS:
                _copy_dataset_slice(
                    parent_group[dataset_name],
                    output_group,
                    name=dataset_name,
                    start=spec.start,
                    end=spec.end,
                    chunk_rows=chunk_rows,
                )
            output_file.flush()

    # Full, chunked equality is checked before the canonical split contract is
    # marked verified and before this file can be published.
    dataset_hashes = _verify_temporary_split_copy(
        parent_path=parent_path,
        split_path=temporary_path,
        spec=spec,
        chunk_rows=chunk_rows,
    )
    contract = _derived_contract(
        parent_contract=parent_contract,
        parent_path=parent_path,
        parent_fingerprint=parent_fingerprint,
        spec=spec,
        dataset_hashes=dataset_hashes,
    )
    with h5py.File(temporary_path, "r+") as output_file:
        _write_split_contract_attrs(
            output_file,
            contract=contract,
            spec=spec,
            parent_path=parent_path,
            parent_fingerprint=parent_fingerprint,
        )
        output_file.flush()

    _fsync_file(temporary_path)


def _verify_temporary_split_copy(
    *,
    parent_path: Path,
    split_path: Path,
    spec: SplitSpec,
    chunk_rows: int,
) -> dict[str, str]:
    with h5py.File(parent_path, "r") as parent_file, h5py.File(
        split_path, "r"
    ) as split_file:
        if set(split_file.keys()) != {spec.group_name}:
            raise SCASplitContractError(
                f"{spec.name} split must contain only {spec.group_name}; "
                f"found={list(split_file.keys())}"
            )
        parent_group = parent_file[spec.group_name]
        split_group = split_file[spec.group_name]

        for dataset_name in _REQUIRED_DATASETS:
            parent_dataset = parent_group[dataset_name]
            split_dataset = split_group[dataset_name]
            expected_shape = (spec.row_count,) + tuple(parent_dataset.shape[1:])
            if tuple(split_dataset.shape) != expected_shape:
                raise SCASplitContractError(
                    f"{spec.name}/{dataset_name} shape mismatch: "
                    f"observed={split_dataset.shape}, expected={expected_shape}"
                )
            if split_dataset.dtype != parent_dataset.dtype:
                raise SCASplitContractError(
                    f"{spec.name}/{dataset_name} dtype mismatch: "
                    f"observed={split_dataset.dtype}, expected={parent_dataset.dtype}"
                )

            for output_start in range(0, spec.row_count, chunk_rows):
                output_end = min(output_start + chunk_rows, spec.row_count)
                source_start = spec.start + output_start
                source_end = spec.start + output_end
                expected = np.asarray(parent_dataset[source_start:source_end])
                observed = np.asarray(split_dataset[output_start:output_end])
                if not np.array_equal(observed, expected):
                    raise SCASplitContractError(
                        f"{spec.name}/{dataset_name} differs from parent slice "
                        f"at output rows [{output_start}:{output_end}]"
                    )

        parent_hashes = _hash_parent_slice(
            parent_file,
            spec=spec,
            chunk_rows=chunk_rows,
        )
        output_hashes = _hash_output_group(
            split_file,
            spec=spec,
            chunk_rows=chunk_rows,
        )
        if output_hashes != parent_hashes:
            raise SCASplitContractError(
                f"{spec.name} dataset hashes differ from parent slice"
            )
        return output_hashes


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_path_for(target: Path) -> Path:
    """Return an uncreated collision-resistant sibling for exclusive HDF5 creation."""
    return target.with_name(
        f".{target.name}.{uuid.uuid4().hex}.tmp"
    )


def _publish_split_set(staged: Mapping[Path, Path]) -> None:
    """Publish all staged files with rollback to the previous complete set."""
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for target in staged:
            if target.exists():
                backup = target.with_name(
                    f".{target.name}.backup.{uuid.uuid4().hex}"
                )
                backup.unlink(missing_ok=True)
                os.replace(target, backup)
                backups[target] = backup

        for target, temporary in staged.items():
            seal_h5_file(temporary, scope_path=target)  # utils/seal.py: signed for its final path
            os.replace(temporary, target)
            published.append(target)
            _fsync_file(target)

        if staged:
            _fsync_directory(next(iter(staged)).parent)

    except Exception:
        for target in published:
            target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                os.replace(backup, target)
        if staged:
            _fsync_directory(next(iter(staged)).parent)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
        if staged:
            _fsync_directory(next(iter(staged)).parent)


# =============================================================================
# Existing split validation/reuse
# =============================================================================
def _validate_existing_split(
    *,
    cfg: Mapping[str, Any],
    parent_path: Path,
    parent_fingerprint: str,
    spec: SplitSpec,
    chunk_rows: int,
    verify_hashes: bool,
) -> None:
    if not spec.output_path.is_file():
        raise FileNotFoundError(spec.output_path)

    dataset = ASCADDataset(
        spec.output_path,
        split=spec.loader_split,
        return_metadata=False,
        require_metadata=_require_metadata(cfg),
        cfg=cfg,
    )
    try:
        if len(dataset) != spec.row_count:
            raise SCASplitContractError(
                f"{spec.name} row count mismatch: {len(dataset)} != {spec.row_count}"
            )
        contract = copy.deepcopy(dataset.data_contract)
    finally:
        dataset.close()

    if not isinstance(contract, Mapping):
        raise SCASplitContractError(f"{spec.output_path} has no split contract")
    derivation = contract.get("derivation")
    if not isinstance(derivation, Mapping):
        raise SCASplitContractError(
            f"{spec.output_path} is missing contract.derivation"
        )

    expected_fields = {
        "version": SPLIT_DERIVATION_VERSION,
        "kind": "verified_row_subset",
        "split_name": spec.name,
        "parent_path": str(parent_path.resolve()),
        "parent_contract_sha256": parent_fingerprint,
        "source_group": spec.group_name,
        "source_row_start": int(spec.start),
        "source_row_end_exclusive": int(spec.end),
        "row_count": int(spec.row_count),
        "labels_recomputed": False,
        "traces_transformed": False,
        "metadata_transformed": False,
        "normalization_applied_during_split": False,
    }
    for field, expected in expected_fields.items():
        if derivation.get(field) != expected:
            raise SCASplitContractError(
                f"{spec.name} derivation field {field!r} is stale: "
                f"observed={derivation.get(field)!r}, expected={expected!r}"
            )

    output = contract.get("output")
    if not isinstance(output, Mapping):
        raise SCASplitContractError(f"{spec.name} contract.output is missing")
    if output.get("groups") != [spec.group_name]:
        raise SCASplitContractError(f"{spec.name} output group contract is stale")
    if str(output.get("path", "")) != str(spec.output_path.resolve()):
        raise SCASplitContractError(f"{spec.name} output path contract is stale")
    if output.get("split_name") != spec.name:
        raise SCASplitContractError(f"{spec.name} output split name is stale")
    if int(output.get("row_count", -1)) != spec.row_count:
        raise SCASplitContractError(f"{spec.name} output row count is stale")

    verification = contract.get("verification")
    if not isinstance(verification, Mapping):
        raise SCASplitContractError(f"{spec.name} verification is missing")
    if verification.get("split_copy_verified") is not True:
        raise SCASplitContractError(f"{spec.name} split_copy_verified is false")
    if verification.get("dataset_hashes_verified") is not True:
        raise SCASplitContractError(f"{spec.name} dataset_hashes_verified is false")
    for flag in (
        "performed_before_publish",
        "label_integrity_verified",
        "trace_selection_verified",
        "metadata_mapping_verified",
        "parent_contract_verified",
    ):
        if verification.get(flag) is not True:
            raise SCASplitContractError(
                f"{spec.name} verification field {flag} is false/missing"
            )

    check_seal(spec.output_path)  # utils/seal.py
    with h5py.File(spec.output_path, "r") as split_file:
        expected_root_attrs = {
            "sca_split_derivation_version": SPLIT_DERIVATION_VERSION,
            "sca_split_name": spec.name,
            "sca_split_group": spec.group_name,
            "sca_split_start": int(spec.start),
            "sca_split_end_exclusive": int(spec.end),
            "sca_split_row_count": int(spec.row_count),
            "sca_split_parent_path": str(parent_path.resolve()),
            "sca_split_parent_contract_sha256": parent_fingerprint,
            "sca_split_labels_recomputed": False,
            "sca_split_trace_transformed": False,
            "sca_split_metadata_transformed": False,
            "sca_split_copy_verified": True,
        }
        for attr_name, expected in expected_root_attrs.items():
            if attr_name not in split_file.attrs:
                raise SCASplitContractError(
                    f"{spec.name} is missing root attribute {attr_name}"
                )
            observed = _decode_attr(split_file.attrs[attr_name])
            if isinstance(expected, bool):
                observed = _strict_bool(
                    observed,
                    field=f"{spec.name} HDF5 {attr_name}",
                )
            elif isinstance(expected, int):
                observed = _strict_int(
                    observed,
                    field=f"{spec.name} HDF5 {attr_name}",
                )
            else:
                observed = str(observed)
            if observed != expected:
                raise SCASplitContractError(
                    f"{spec.name} root attribute {attr_name} is stale: "
                    f"observed={observed!r}, expected={expected!r}"
                )
        if set(split_file.keys()) != {spec.group_name}:
            raise SCASplitContractError(
                f"{spec.name} contains unexpected groups {list(split_file.keys())}"
            )
        for dataset_name in _REQUIRED_DATASETS:
            if dataset_name not in split_file[spec.group_name]:
                raise SCASplitContractError(
                    f"{spec.name} is missing {spec.group_name}/{dataset_name}"
                )

        if verify_hashes:
            recorded_hashes = verification.get("dataset_sha256")
            if not isinstance(recorded_hashes, Mapping):
                raise SCASplitContractError(
                    f"{spec.name} has no recorded dataset hashes"
                )
            output_hashes = _hash_output_group(
                split_file,
                spec=spec,
                chunk_rows=chunk_rows,
            )
            if dict(recorded_hashes) != output_hashes:
                raise SCASplitContractError(
                    f"{spec.name} content hash differs from its contract"
                )

    if verify_hashes:
        with h5py.File(parent_path, "r") as parent_file:
            parent_hashes = _hash_parent_slice(
                parent_file,
                spec=spec,
                chunk_rows=chunk_rows,
            )
        if parent_hashes != dict(verification["dataset_sha256"]):
            raise SCASplitContractError(
                f"{spec.name} no longer matches its parent row slice"
            )



# =============================================================================
# Public split-generation API
# =============================================================================
_VERIFIED_ARTIFACTS: dict[str, str] = {}


def _verification_signature(
    cfg: Mapping[str, Any], *, model_name: str
) -> tuple[str, str]:
    """(artifact directory, digest of data config + artifact file stats)."""
    data_cfg = _get_data_cfg(cfg)
    parent = _complete_dataset_path(cfg, model_name=model_name).resolve()
    files = [parent, *(_split_paths(cfg, model_name=model_name))]
    stats = [
        (str(path), path.stat().st_mtime_ns, path.stat().st_size) if path.is_file() else (str(path), None, None)
        for path in files
    ]
    payload = json.dumps(
        {
            "data": _json_safe(dict(data_cfg)),
            "sca": _json_safe(dict(cfg.get("sca") or {})),
            "cv": _cv_enabled(cfg),
            "files": stats,
        },
        sort_keys=True,
        default=str,
    )
    return str(parent.parent), hashlib.sha256(payload.encode("utf-8")).hexdigest()


def perform_and_save_data_split(cfg: MutableMapping[str, Any], model_name: str) -> None:
    """
    Derive train/test in CV mode, or train/val/test with CV disabled.

    This function is idempotent. Existing artifacts are reused only when their
    complete contracts and dataset hashes match the current runtime.
    """
    _prepare_runtime_cfg(cfg, model_name=model_name)

    # Fail on unresolved scientific search spaces before creating directories or
    # writing artifacts.
    data_cfg = _get_data_cfg(cfg)
    _reject_legacy_mutating_loader_options(data_cfg)
    attack_byte = _get_attack_byte_from_cfg(cfg)
    leakage_model = _get_runtime_leakage_model_from_cfg(cfg)
    generate_allowed = _generate_data_out(data_cfg)
    cv_enabled = _cv_enabled(cfg)

    # Skip a second full verification in the same process when neither the
    # data configuration nor any artifact file changed since the last one.
    cache_key, cache_value = _verification_signature(cfg, model_name=model_name)
    if _VERIFIED_ARTIFACTS.get(cache_key) == cache_value:
        return

    seed = _strict_int(set_seed(cfg), field="reproducibility seed")
    logger.info(
        "Preparing SCA data artifacts: model=%s byte=%d leakage=%s seed=%d "
        "generate_data_out=%s cross_validation=%s splits=%s",
        model_name,
        attack_byte,
        leakage_model,
        seed,
        generate_allowed,
        cv_enabled,
        "train+test" if cv_enabled else "train+val+test",
    )

    # Validate split coordinates and output-path separation before any expensive
    # dataset generation or HDF5 publication.
    attack_range = _validate_range(
        data_cfg.get("attack_range"),
        name="data.attack_range",
    )
    validate_ranges = _validate_test_range if _cv_enabled(cfg) else _validate_evaluation_ranges
    validate_ranges(
        data_cfg,
        attack_count=attack_range[1] - attack_range[0],
    )
    parent_candidate = _complete_dataset_path(
        cfg,
        model_name=model_name,
    ).resolve()
    split_candidates = {
        path.resolve()
        for path in _split_paths(cfg, model_name=model_name)
    }
    if parent_candidate in split_candidates:
        raise ValueError(
            "paths.data_out must be distinct from train.h5, val.h5, and test.h5"
        )
    if len(split_candidates) != 3:
        raise ValueError("train, validation, and test output paths must be distinct")

    parent_path, parent_contract, parent_fingerprint, profiling_count, attack_count = (
        _ensure_complete_dataset(cfg, model_name=model_name)
    )
    specs = _build_split_specs(
        cfg,
        model_name=model_name,
        profiling_count=profiling_count,
        attack_count=attack_count,
    )

    chunk_rows = _strict_int(
        data_cfg.get("split_chunk_size", data_cfg.get("chunk_size", 4096)),
        field="data.split_chunk_size",
        minimum=1,
    )
    verify_hashes = _strict_bool(
        data_cfg.get("split_verify_hashes_on_load", True),
        field="data.split_verify_hashes_on_load",
        default=True,
    )
    if not verify_hashes:
        raise ValueError(
            "data.split_verify_hashes_on_load=false is not permitted. "
            "Production SCA splits must be verified against their recorded "
            "content hashes and parent slices."
        )
    force_regenerate = _strict_bool(
        data_cfg.get("force_regenerate_splits", False),
        field="data.force_regenerate_splits",
        default=False,
    )

    if not force_regenerate:
        try:
            for spec in specs:
                _validate_existing_split(
                    cfg=cfg,
                    parent_path=parent_path,
                    parent_fingerprint=parent_fingerprint,
                    spec=spec,
                    chunk_rows=chunk_rows,
                    verify_hashes=verify_hashes,
                )
            # logger.info(
            #     "Reusing verified active splits derived from %s",
            #     parent_path,
            # )
            if not cv_enabled:
                _assert_disjoint_attack_source_indices(specs[1].output_path, specs[2].output_path)
            _VERIFIED_ARTIFACTS[cache_key] = _verification_signature(cfg, model_name=model_name)[1]
            return
        except (PermissionError, TamperedArtifactError):
            raise  # a changed file is evidence, not a stale file to regenerate
        except Exception as exc:
            if not generate_allowed:
                raise SCASplitContractError(
                    "data.generate_data_out=false but split artifacts are missing, "
                    f"stale, or invalid: {exc}"
                ) from exc
            logger.warning(
                "Split artifacts are missing, stale, or invalid and will be "
                "regenerated: %s",
                exc,
            )

    if not generate_allowed:
        raise SCASplitContractError(
            "data.generate_data_out=false conflicts with data.force_regenerate_splits=true"
        )

    staged: dict[Path, Path] = {}
    try:
        for spec in specs:
            spec.output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = _temporary_path_for(spec.output_path)
            staged[spec.output_path] = temporary_path
            _write_split_file(
                parent_path=parent_path,
                parent_contract=parent_contract,
                parent_fingerprint=parent_fingerprint,
                spec=spec,
                temporary_path=temporary_path,
                chunk_rows=chunk_rows,
            )

        _publish_split_set(staged)
        staged.clear()

    finally:
        for temporary_path in staged.values():
            temporary_path.unlink(missing_ok=True)

    # Reopen through the production dataset class and verify hashes after final
    # publication. This catches path/rename/filesystem surprises before training.
    for spec in specs:
        _validate_existing_split(
            cfg=cfg,
            parent_path=parent_path,
            parent_fingerprint=parent_fingerprint,
            spec=spec,
            chunk_rows=chunk_rows,
            verify_hashes=True,
        )

    if not cv_enabled:
        _assert_disjoint_attack_source_indices(specs[1].output_path, specs[2].output_path)
    _VERIFIED_ARTIFACTS[cache_key] = _verification_signature(cfg, model_name=model_name)[1]

    logger.info(
        "SCA split generation complete | rows=%s | "
        "labels_recomputed_during_split=no | parent=%s",
        {spec.name: spec.row_count for spec in specs},
        parent_path,
    )

    debug_inspect = _strict_bool(
        data_cfg.get("debug_h5_inspect", cfg.get("debug_h5_inspect", False)),
        field="data.debug_h5_inspect",
        default=False,
    )
    if debug_inspect:
        for spec in specs:
            inspect_and_log_h5(spec.output_path, label=f"{spec.name} ({spec.group_name})")


# =============================================================================
# DataLoader construction
# =============================================================================
def _dataset_returns_metadata(dataset: Any) -> bool:
    current = dataset
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if bool(getattr(current, "return_metadata", False)):
            return True
        current = getattr(current, "dataset", None)
    return False


def _make_loader(
    *,
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    drop_last: bool = False,
) -> DataLoader:
    nw = _strict_int(num_workers, field="num_workers", minimum=0)
    resolved_batch_size = _strict_int(
        batch_size,
        field="batch_size",
        minimum=1,
    )
    resolved_seed = _strict_int(seed, field="dataloader seed")
    resolved_shuffle = _strict_bool(shuffle, field="dataloader.shuffle")
    resolved_pin_memory = _strict_bool(
        pin_memory,
        field="dataloader.pin_memory",
    )
    resolved_persistent_workers = _strict_bool(
        persistent_workers,
        field="dataloader.persistent_workers",
    )
    resolved_drop_last = _strict_bool(
        drop_last,
        field="dataloader.drop_last",
    )
    generator = make_torch_generator(resolved_seed, device="cpu")
    worker_init = make_worker_init_fn(resolved_seed) if nw > 0 else None

    kwargs: dict[str, Any] = {
        "batch_size": resolved_batch_size,
        "shuffle": resolved_shuffle,
        "num_workers": nw,
        "pin_memory": resolved_pin_memory,
        "generator": generator,
        "worker_init_fn": worker_init,
        "persistent_workers": resolved_persistent_workers and nw > 0,
        "drop_last": resolved_drop_last,
    }
    if nw > 0:
        kwargs["prefetch_factor"] = _strict_int(
            prefetch_factor,
            field="dataloader.prefetch_factor",
            minimum=1,
        )
    if _dataset_returns_metadata(dataset):
        kwargs["collate_fn"] = sca_collate_fn
    return DataLoader(dataset, **kwargs)


def _ensure_splits_ready(cfg: MutableMapping[str, Any]) -> tuple[Path, Optional[Path], Path]:
    model_name = cfg.get("model_name")
    if model_name is None or not str(model_name).strip():
        raise ValueError("cfg['model_name'] is required")
    perform_and_save_data_split(cfg, model_name=str(model_name))
    train_path, validation_path, test_path = _split_paths(cfg, model_name=str(model_name))
    return train_path, None if _cv_enabled(cfg) else validation_path, test_path


def get_train_val_loaders(
    cfg: MutableMapping[str, Any],
    trainer_cfg: Mapping[str, Any],
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Load profiling train and disjoint attack-derived validation datasets."""
    if _cv_enabled(cfg):
        raise ValueError(
            "get_train_val_loaders() is for cross_validation.enabled=false. "
            "Use get_cross_validation_loaders() with profiling fold indices."
        )
    seed_used = _resolve_loader_seed(cfg, seed)
    batch_size = _resolve_training_batch_size(trainer_cfg)
    train_path, validation_path, _test_path = _ensure_splits_ready(cfg)

    train_dataset = load_split_dataset(
        cfg,
        str(train_path),
        split="profiling",
        return_metadata=False,
    )
    validation_dataset = load_split_dataset(
        cfg,
        str(validation_path),
        split="attack",
        return_metadata=True,
    )
    if _require_metadata(cfg) and not bool(
        getattr(validation_dataset, "return_metadata", False)
    ):
        raise ValueError(
            "Validation dataset must return metadata when sca.require_metadata=true"
        )
    # trace_normalization.mode fitted: fitted on the training split, the run's final scaler.
    scale_training(cfg, train_dataset, None, [train_dataset, validation_dataset], final=True,
                   label="training split")

    dataloader_cfg, num_workers = _get_dataloader_cfg(cfg)
    train_loader = _make_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers["train"],
        seed=seed_used,
        pin_memory=dataloader_cfg["pin_memory"],
        persistent_workers=dataloader_cfg["persistent_workers"],
        prefetch_factor=dataloader_cfg["prefetch_factor"],
        drop_last=dataloader_cfg["drop_last_train"],
    )
    validation_loader = _make_loader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers["val"],
        seed=seed_used,
        pin_memory=dataloader_cfg["pin_memory"],
        persistent_workers=dataloader_cfg["persistent_workers"],
        prefetch_factor=dataloader_cfg["prefetch_factor"],
        drop_last=False,
    )

    cfg.setdefault("repro", {})
    if isinstance(cfg["repro"], MutableMapping):
        cfg["repro"]["dataloader_seeded"] = True

    logger.info(
        "Loaded train=%d validation=%d samples with verified split contracts",
        len(train_dataset),
        len(validation_dataset),
    )
    return train_loader, validation_loader


# =============================================================================
# Cross-validation profiling-data APIs
# =============================================================================
def _ensure_profiling_split_ready(cfg: MutableMapping[str, Any]) -> Path:
    """Verify train.h5 for CV folds/refit without reading Attack_traces.

    Full preparation (deriving train.h5 and test.h5 and checking the parent's
    Attack_traces provenance) runs once at CV start. Folds and the refit only
    re-verify the profiling split against the parent contract fingerprint.
    """
    model_name = cfg.get("model_name")
    if model_name is None or not str(model_name).strip():
        raise ValueError("cfg['model_name'] is required")
    model_name = str(model_name)
    _prepare_runtime_cfg(cfg, model_name=model_name)
    data_cfg = _get_data_cfg(cfg)
    profiling_start, profiling_end = _validate_range(
        data_cfg.get("profiling_range"), name="data.profiling_range"
    )
    train_path, _validation_path, _test_path = _split_paths(cfg, model_name=model_name)
    parent_path = _complete_dataset_path(cfg, model_name=model_name)
    with h5py.File(parent_path, "r") as parent_file:
        parent_fingerprint = _read_contract_fingerprint(parent_file, path=parent_path)
    _validate_existing_split(
        cfg=cfg,
        parent_path=parent_path,
        parent_fingerprint=parent_fingerprint,
        spec=SplitSpec(
            name="train",
            group_name="Profiling_traces",
            start=0,
            end=profiling_end - profiling_start,
            output_path=train_path,
        ),
        chunk_rows=_strict_int(
            data_cfg.get("split_chunk_size", data_cfg.get("chunk_size", 4096)),
            field="data.split_chunk_size",
            minimum=1,
        ),
        verify_hashes=True,
    )
    logger.info(
        "CV profiling split verified: %s rows=%d (Attack_traces/test.h5 not accessed)",
        train_path,
        profiling_end - profiling_start,
    )
    return train_path


def _assert_disjoint_attack_source_indices(validation_path: Path, test_path: Path) -> None:
    """Fail when val.h5 and test.h5 share any source Attack_traces row."""
    indices = []
    for path in (validation_path, test_path):
        check_seal(path)
        with h5py.File(path, "r") as h5_file:
            indices.append(np.asarray(h5_file["Attack_traces"]["source_indices"][:], dtype=np.int64))
    overlap = np.intersect1d(indices[0], indices[1])
    if overlap.size:
        raise SCASplitContractError(
            f"Validation and test splits share {overlap.size} Attack_traces source "
            f"rows (first: {overlap[:5].tolist()}); validation traces would also be "
            "test traces. Choose disjoint data.validation_range and data.test_range."
        )


def _normalize_cv_indices(
    indices: Sequence[int] | np.ndarray,
    *,
    name: str,
    dataset_size: int,
) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {values.shape}")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
        values.dtype, np.integer
    ):
        raise TypeError(f"{name} must contain integer indices, got {values.dtype}")

    normalized = values.astype(np.int64, copy=False)
    if np.unique(normalized).size != normalized.size:
        raise ValueError(f"{name} contains duplicate indices")
    if int(normalized.min()) < 0 or int(normalized.max()) >= int(dataset_size):
        raise IndexError(
            f"{name} contains values outside [0,{dataset_size - 1}]"
        )
    return normalized


def get_cross_validation_dataset(
    cfg: MutableMapping[str, Any],
    *,
    return_metadata: bool = False,
    train_path: Optional[Path] = None,
):
    _require_cv_enabled(cfg, caller="get_cross_validation_dataset()")
    if train_path is None:
        train_path = _ensure_profiling_split_ready(cfg)
    dataset = load_split_dataset(
        cfg,
        str(train_path),
        split="profiling",
        return_metadata=_strict_bool(
            return_metadata,
            field="return_metadata",
        ),
    )
    if return_metadata and _require_metadata(cfg) and not bool(
        getattr(dataset, "return_metadata", False)
    ):
        raise ValueError(
            "Cross-validation validation dataset must return metadata"
        )
    if len(dataset) <= 0:
        raise ValueError(f"Cross-validation profiling dataset is empty: {train_path}")
    return dataset


def get_cross_validation_labels(cfg: MutableMapping[str, Any]) -> np.ndarray:
    _require_cv_enabled(cfg, caller="get_cross_validation_labels()")
    train_path = _ensure_profiling_split_ready(cfg)
    check_seal(train_path)
    with h5py.File(train_path, "r") as h5_file:
        group_name = "Profiling_traces"
        group = h5_file[group_name]
        labels = np.asarray(group["labels"][:], dtype=np.int64)
        trace_count = int(group["traces"].shape[0])
        if labels.ndim != 1 or int(labels.shape[0]) != trace_count:
            raise ValueError(
                f"Profiling labels are misaligned: labels={labels.shape}, "
                f"traces={trace_count}"
            )
        if _require_metadata(cfg):
            if "metadata" not in group or int(group["metadata"].shape[0]) != trace_count:
                raise ValueError("Profiling metadata is missing or misaligned")
    if labels.size == 0:
        raise ValueError("Profiling labels are empty")
    return labels


def get_cross_validation_loaders(
    cfg: MutableMapping[str, Any],
    trainer_cfg: Mapping[str, Any],
    train_indices: Sequence[int] | np.ndarray,
    validation_indices: Sequence[int] | np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    _require_cv_enabled(cfg, caller="get_cross_validation_loaders()")
    seed_used = _resolve_loader_seed(cfg, seed)
    batch_size = _resolve_training_batch_size(trainer_cfg)

    train_path = _ensure_profiling_split_ready(cfg)
    training_base = get_cross_validation_dataset(
        cfg, return_metadata=False, train_path=train_path
    )
    validation_base = get_cross_validation_dataset(
        cfg, return_metadata=True, train_path=train_path
    )
    if len(training_base) != len(validation_base):
        raise ValueError("Cross-validation base datasets are misaligned")

    train_idx = _normalize_cv_indices(
        train_indices,
        name="train_indices",
        dataset_size=len(training_base),
    )
    validation_idx = _normalize_cv_indices(
        validation_indices,
        name="validation_indices",
        dataset_size=len(training_base),
    )
    overlap = np.intersect1d(train_idx, validation_idx, assume_unique=True)
    if overlap.size:
        raise ValueError(
            "Cross-validation train and validation indices overlap: "
            f"{overlap[:10].tolist()}"
        )
    if train_idx.size + validation_idx.size != len(training_base):
        covered = np.union1d(train_idx, validation_idx)
        missing_count = len(training_base) - int(covered.size)
        raise ValueError(
            "Cross-validation train and validation indices must partition the "
            f"complete profiling dataset; missing_rows={missing_count}"
        )

    # trace_normalization.mode fitted: fitted on the fold's training rows only.
    scale_training(cfg, training_base, train_idx, [training_base, validation_base], final=False,
                   label="fold training")
    fold_train = Subset(training_base, train_idx.tolist())
    fold_validation = Subset(validation_base, validation_idx.tolist())
    dataloader_cfg, num_workers = _get_dataloader_cfg(cfg)

    train_loader = _make_loader(
        dataset=fold_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers["train"],
        seed=seed_used,
        pin_memory=dataloader_cfg["pin_memory"],
        persistent_workers=dataloader_cfg["persistent_workers"],
        prefetch_factor=dataloader_cfg["prefetch_factor"],
        drop_last=dataloader_cfg["drop_last_train"],
    )
    validation_loader = _make_loader(
        dataset=fold_validation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers["val"],
        seed=seed_used,
        pin_memory=dataloader_cfg["pin_memory"],
        persistent_workers=dataloader_cfg["persistent_workers"],
        prefetch_factor=dataloader_cfg["prefetch_factor"],
        drop_last=False,
    )

    cfg.setdefault("repro", {})
    if isinstance(cfg["repro"], MutableMapping):
        cfg["repro"]["dataloader_seeded"] = True
    return train_loader, validation_loader


def get_full_profiling_loader(
    cfg: MutableMapping[str, Any],
    trainer_cfg: Mapping[str, Any],
    seed: Optional[int] = None,
) -> DataLoader:
    _require_cv_enabled(cfg, caller="get_full_profiling_loader()")
    seed_used = _resolve_loader_seed(cfg, seed)
    batch_size = _resolve_training_batch_size(trainer_cfg)
    profiling_dataset = get_cross_validation_dataset(cfg, return_metadata=False)
    # trace_normalization.mode fitted: fitted on every profiling trace, the run's final scaler.
    scale_training(cfg, profiling_dataset, None, [profiling_dataset], final=True, label="refit profiling")
    dataloader_cfg, num_workers = _get_dataloader_cfg(cfg)

    loader = _make_loader(
        dataset=profiling_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers["train"],
        seed=seed_used,
        pin_memory=dataloader_cfg["pin_memory"],
        persistent_workers=dataloader_cfg["persistent_workers"],
        prefetch_factor=dataloader_cfg["prefetch_factor"],
        # dataloader.drop_last_train, as the folds (2026-10-23): with it the refit's last incomplete batch is dropped
        # each epoch too, so a step-level schedule (OneCycleLR) counts the same way in every loop
        drop_last=dataloader_cfg["drop_last_train"],
    )
    cfg.setdefault("repro", {})
    if isinstance(cfg["repro"], MutableMapping):
        cfg["repro"]["dataloader_seeded"] = True
    return loader


def get_full_profiling_eval_loader(
    cfg: MutableMapping[str, Any],
    trainer_cfg: Mapping[str, Any],
    seed: Optional[int] = None,
) -> DataLoader:
    """Ordered profiling loader without metadata for final-training evaluation.

    Same rows as get_full_profiling_loader(), in file order, for eval-mode
    accuracy and loss on the training population. It carries no metadata, so
    the training traces are never attacked. Attack_traces are not read.
    """
    _require_cv_enabled(cfg, caller="get_full_profiling_eval_loader()")
    seed_used = _resolve_loader_seed(cfg, seed)
    dataset = get_cross_validation_dataset(cfg, return_metadata=False)
    # The same rows as the refit, so the same fit (deterministic: the stored scaler is identical).
    scale_training(cfg, dataset, None, [dataset], final=True, label="refit profiling")
    dataloader_cfg, num_workers = _get_dataloader_cfg(cfg)
    return _make_loader(
        dataset=dataset,
        batch_size=_resolve_training_batch_size(trainer_cfg),
        shuffle=False,
        num_workers=num_workers["val"],
        seed=seed_used,
        pin_memory=dataloader_cfg["pin_memory"],
        persistent_workers=dataloader_cfg["persistent_workers"],
        prefetch_factor=dataloader_cfg["prefetch_factor"],
        drop_last=False,
    )


def get_test_loader(
    cfg: MutableMapping[str, Any],
    trainer_cfg: Mapping[str, Any],
    seed: Optional[int] = None,
) -> DataLoader:
    seed_used = _resolve_loader_seed(cfg, seed)
    batch_size = _resolve_training_batch_size(trainer_cfg)
    _train_path, _validation_path, test_path = _ensure_splits_ready(cfg)

    test_dataset = load_split_dataset(
        cfg,
        str(test_path),
        split="attack",
        return_metadata=True,
    )
    # trace_normalization.mode fitted: the run's final scaler (fitted on its training traces).
    scale_test(cfg, test_dataset)
    dataloader_cfg, num_workers = _get_dataloader_cfg(cfg)
    loader = _make_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers["val"],
        seed=seed_used,
        pin_memory=dataloader_cfg["pin_memory"],
        persistent_workers=dataloader_cfg["persistent_workers"],
        prefetch_factor=dataloader_cfg["prefetch_factor"],
        drop_last=False,
    )

    cfg.setdefault("repro", {})
    if isinstance(cfg["repro"], MutableMapping):
        cfg["repro"]["dataloader_seeded"] = True
    return loader


__all__ = [
    "SCASplitContractError",
    "get_cross_validation_dataset",
    "get_cross_validation_labels",
    "get_cross_validation_loaders",
    "get_full_profiling_eval_loader",
    "get_full_profiling_loader",
    "get_test_loader",
    "get_train_val_loaders",
    "perform_and_save_data_split",
]
