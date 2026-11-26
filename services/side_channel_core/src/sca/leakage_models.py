# services/side_channel_core/src/sca/leakage_models.py
"""
Leakage model utilities for Side-Channel Analysis (SCA).

    -> compute_aes_intermediate(pt, key)
        -> AES_SBOX[pt ^ key]
    -> apply_leakage_model(intermediate, leakage_model)
        ID  -> intermediate
        HW  -> hamming_weight(intermediate)
        MSB -> msb(intermediate)
Purpose
-------
This module centralizes all leakage-model logic used across the SCA system.

It defines how to:
- compute AES intermediate values
- map intermediates to leakage labels
- determine class-space size for a leakage model
- build full key-hypothesis label tables for rank/GE/LDD style evaluation

Supported leakage models
------------------------
- ID  : Identity leakage model
        label = SBOX(plaintext_byte XOR key_byte)
        num_classes = 256

- HW  : Hamming-weight leakage model
        label = HW(SBOX(plaintext_byte XOR key_byte))
        num_classes = 9

- MSB : Most-significant-bit leakage model
        label = MSB(SBOX(plaintext_byte XOR key_byte))
        num_classes = 2

Design goals
------------
- Production-grade validation
- No silent dtype wraparound bugs
- Vectorized operations for speed
- Single source of truth for leakage semantics
- Future-proof extension point for additional leakage models

Notes
-----
- This module currently targets AES byte-wise attacks.
- All public functions accept scalars or numpy arrays where reasonable.
- Metadata schema handling belongs elsewhere; this module only handles leakage math.

Author: Shraddha Mishra
"""

from __future__ import annotations

from functools import lru_cache
from operator import index as integer_index
from typing import Any, Literal, Mapping, MutableMapping, Optional, Union, cast

import numpy as np

from utils.aes import AES_INV_SBOX, AES_SBOX

LeakageModel = Literal["ID", "HW", "MSB"]
LeakageValue = Union[int, np.ndarray]

_VALID_LEAKAGE_MODELS: tuple[LeakageModel, ...] = ("ID", "HW", "MSB")
_NUM_CLASSES: Mapping[LeakageModel, int] = {
    "ID": 256,
    "HW": 9,
    "MSB": 2,
}
_COLLECTION_TYPES = (list, tuple, set, frozenset)


# =============================================================================
# Immutable lookup tables
# =============================================================================
def _validated_aes_sbox(raw: Any) -> np.ndarray:
    """Return the authoritative AES S-box as an immutable uint8 permutation."""
    array = np.asarray(raw)

    if array.shape != (256,):
        raise ValueError(
            "utils.aes.AES_SBOX must have shape (256,), "
            f"got {array.shape}"
        )

    if np.issubdtype(array.dtype, np.bool_):
        raise TypeError("utils.aes.AES_SBOX must contain byte integers, not bool")

    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError("utils.aes.AES_SBOX contains NaN or infinity")
        if np.any(np.floor(array) != array):
            raise ValueError("utils.aes.AES_SBOX contains non-integer values")
    elif not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            "utils.aes.AES_SBOX must contain integer byte values, "
            f"got dtype={array.dtype}"
        )

    values = array.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values > 255):
        raise ValueError("utils.aes.AES_SBOX contains values outside [0, 255]")
    if np.unique(values).size != 256:
        raise ValueError("utils.aes.AES_SBOX must be a permutation of 0..255")

    result = values.astype(np.uint8, copy=True)
    result.setflags(write=False)
    return result


_AES_SBOX = _validated_aes_sbox(AES_SBOX)
_HAMMING_WEIGHT_TABLE = np.fromiter(
    (value.bit_count() for value in range(256)),
    dtype=np.uint8,
    count=256,
)
_HAMMING_WEIGHT_TABLE.setflags(write=False)


# =============================================================================
# Leakage-model normalization and runtime resolution
# =============================================================================
def normalize_leakage_model(leakage_model: Any) -> LeakageModel:
    """Normalize one scalar leakage-model identifier to ``ID``, ``HW``, or ``MSB``."""
    if isinstance(leakage_model, np.ndarray):
        if leakage_model.shape != ():
            raise TypeError(
                "leakage_model must be one scalar string, got ndarray "
                f"shape={leakage_model.shape}"
            )
        leakage_model = leakage_model.item()

    if isinstance(leakage_model, (Mapping, *_COLLECTION_TYPES)):
        raise TypeError(
            "leakage_model must be one resolved scalar string, got "
            f"{type(leakage_model).__name__}: {leakage_model!r}"
        )

    if not isinstance(leakage_model, (str, np.str_)):
        raise TypeError(
            "leakage_model must be a string, got "
            f"{type(leakage_model).__name__}: {leakage_model!r}"
        )

    normalized = str(leakage_model).strip().upper()
    if not normalized:
        raise ValueError("leakage_model must not be empty")
    if normalized not in _VALID_LEAKAGE_MODELS:
        raise ValueError(
            f"Unsupported leakage_model={leakage_model!r}. "
            f"Supported: {list(_VALID_LEAKAGE_MODELS)}"
        )
    return cast(LeakageModel, normalized)


def _mapping_section(
    cfg: Mapping[str, Any],
    key: str,
    *,
    required: bool = False,
) -> Mapping[str, Any]:
    raw = cfg.get(key)
    if raw is None:
        if required:
            raise KeyError(f"cfg[{key!r}] is required")
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"cfg[{key!r}] must be a mapping, got {type(raw).__name__}"
        )
    return raw


def _explicit_scalar_candidate(
    value: Any,
    *,
    source: str,
) -> LeakageModel:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise TypeError(
                f"{source} must be one resolved scalar leakage model; "
                f"got ndarray shape={value.shape}"
            )
        value = value.item()
    if isinstance(value, (Mapping, *_COLLECTION_TYPES)):
        raise TypeError(
            f"{source} must be one resolved scalar leakage model; "
            f"got {type(value).__name__}: {value!r}"
        )
    return normalize_leakage_model(value)


def _declared_search_space(value: Any, *, source: str) -> Optional[tuple[LeakageModel, ...]]:
    """Return a normalized unresolved search space, or ``None`` for scalar input."""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return None
        value = value.tolist()

    if isinstance(value, Mapping):
        raise TypeError(
            f"{source} must be a scalar or sequence of leakage models, got mapping"
        )

    if not isinstance(value, _COLLECTION_TYPES):
        return None

    items = list(value)
    if not items:
        raise ValueError(f"{source} search space must not be empty")

    normalized: list[LeakageModel] = []
    for index, item in enumerate(items):
        normalized.append(
            _explicit_scalar_candidate(item, source=f"{source}[{index}]")
        )

    # Preserve declaration order while removing duplicates.
    return tuple(dict.fromkeys(normalized))


def _add_candidate(
    candidates: list[tuple[str, LeakageModel]],
    *,
    source: str,
    value: Any,
) -> None:
    candidates.append(
        (source, _explicit_scalar_candidate(value, source=source))
    )


def resolve_runtime_leakage_model(
    cfg: MutableMapping[str, Any],
    *,
    default: Optional[str] = None,
    mutate: bool = True,
) -> LeakageModel:
    """
    Resolve exactly one runtime leakage model without silent search-space fallback.

    Canonical input is ``cfg['sca']['leakage_model']``. During sweep expansion,
    a still-present declared search space may be resolved only by an explicit
    scalar runtime hint from one of these compatibility locations:

    - ``cfg['runtime']['leakage_model']``
    - ``cfg['sweep_context']['sca.leakage_model']``
    - ``cfg['sweep_context']['sca']['leakage_model']``
    - legacy ``cfg['leakage_model']``

    All populated scalar sources must agree. If ``sca.leakage_model`` is a
    search space, the selected scalar must also belong to that declared space.
    Missing configuration raises unless the caller explicitly supplies
    ``default=...``.

    When ``mutate=True``, the selected value is written to the canonical
    ``cfg['sca']['leakage_model']`` and ``cfg['runtime']['leakage_model']``.
    A pre-existing legacy top-level alias is synchronized but is not created.
    """
    if not isinstance(cfg, MutableMapping):
        raise TypeError(
            "resolve_runtime_leakage_model expects a mutable mapping, got "
            f"{type(cfg).__name__}"
        )

    sca_raw = cfg.get("sca")
    if sca_raw is None:
        sca_cfg: Mapping[str, Any] = {}
    elif isinstance(sca_raw, Mapping):
        sca_cfg = sca_raw
    else:
        raise TypeError(
            f"cfg['sca'] must be a mapping, got {type(sca_raw).__name__}"
        )

    runtime_cfg = _mapping_section(cfg, "runtime")
    sweep_context = _mapping_section(cfg, "sweep_context")

    candidates: list[tuple[str, LeakageModel]] = []
    declared_space: Optional[tuple[LeakageModel, ...]] = None

    if "leakage_model" in sca_cfg:
        raw_canonical = sca_cfg["leakage_model"]
        declared_space = _declared_search_space(
            raw_canonical,
            source="sca.leakage_model",
        )
        if declared_space is None:
            _add_candidate(
                candidates,
                source="sca.leakage_model",
                value=raw_canonical,
            )

    if "leakage_model" in runtime_cfg:
        _add_candidate(
            candidates,
            source="runtime.leakage_model",
            value=runtime_cfg["leakage_model"],
        )

    if "sca.leakage_model" in sweep_context:
        _add_candidate(
            candidates,
            source="sweep_context['sca.leakage_model']",
            value=sweep_context["sca.leakage_model"],
        )

    nested_sca = sweep_context.get("sca")
    if nested_sca is not None:
        if not isinstance(nested_sca, Mapping):
            raise TypeError("cfg['sweep_context']['sca'] must be a mapping")
        if "leakage_model" in nested_sca:
            _add_candidate(
                candidates,
                source="sweep_context.sca.leakage_model",
                value=nested_sca["leakage_model"],
            )

    if "leakage_model" in cfg:
        _add_candidate(
            candidates,
            source="cfg.leakage_model (legacy)",
            value=cfg["leakage_model"],
        )

    if not candidates:
        if declared_space is not None:
            raise ValueError(
                "sca.leakage_model reached runtime as an unresolved search space "
                f"{list(declared_space)!r}. The sweep engine must select one scalar "
                "value before dataset/model/metric execution."
            )
        if default is None:
            raise KeyError(
                "No runtime leakage model is configured. Set "
                "cfg['sca']['leakage_model'] to one of "
                f"{list(_VALID_LEAKAGE_MODELS)}."
            )
        selected = normalize_leakage_model(default)
    else:
        distinct = {value for _source, value in candidates}
        if len(distinct) != 1:
            details = ", ".join(
                f"{source}={value}" for source, value in candidates
            )
            raise ValueError(
                "Conflicting runtime leakage-model values: " + details
            )
        selected = next(iter(distinct))

    if declared_space is not None and selected not in declared_space:
        raise ValueError(
            f"Resolved leakage_model={selected!r} is not in declared "
            f"sca.leakage_model search space {list(declared_space)!r}"
        )

    if mutate:
        if sca_raw is None:
            canonical_mutable: MutableMapping[str, Any] = {}
            cfg["sca"] = canonical_mutable
        elif isinstance(sca_raw, MutableMapping):
            canonical_mutable = sca_raw
        else:
            canonical_mutable = dict(sca_raw)
            cfg["sca"] = canonical_mutable
        canonical_mutable["leakage_model"] = selected

        runtime_raw = cfg.get("runtime")
        if runtime_raw is None:
            runtime_mutable: MutableMapping[str, Any] = {}
            cfg["runtime"] = runtime_mutable
        elif isinstance(runtime_raw, MutableMapping):
            runtime_mutable = runtime_raw
        elif isinstance(runtime_raw, Mapping):
            runtime_mutable = dict(runtime_raw)
            cfg["runtime"] = runtime_mutable
        else:  # already rejected above; defensive guard
            raise TypeError("cfg['runtime'] must be a mapping")
        runtime_mutable["leakage_model"] = selected

        if "leakage_model" in cfg:
            cfg["leakage_model"] = selected

    return selected


def get_supported_leakage_models() -> tuple[LeakageModel, ...]:
    """Return the supported canonical leakage-model identifiers."""
    return _VALID_LEAKAGE_MODELS


def get_num_classes(leakage_model: Any) -> int:
    """Return the exact model-head class count for one leakage model."""
    return _NUM_CLASSES[normalize_leakage_model(leakage_model)]


# =============================================================================
# Strict byte and integer helpers
# =============================================================================
def _strict_integer(
    value: Any,
    *,
    field: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, got bool")
    try:
        result = int(integer_index(value))
    except TypeError as exc:
        raise TypeError(
            f"{field} must be an integer, got {type(value).__name__}: {value!r}"
        ) from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be <= {maximum}, got {result}")
    return result


def _ensure_uint8_array(value: Any, *, name: str) -> np.ndarray:
    """Convert numeric scalar/array input to uint8 without truncation or wraparound."""
    array = np.asarray(value)

    if array.size == 0:
        if np.issubdtype(array.dtype, np.bool_):
            raise TypeError(f"{name} must contain byte integers, not bool")
        if array.dtype.kind not in "iufO":
            raise TypeError(
                f"{name} must contain numeric byte values, got dtype={array.dtype}"
            )
        return np.empty(array.shape, dtype=np.uint8)

    if np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must contain byte integers, not bool")

    if np.issubdtype(array.dtype, np.integer):
        if np.issubdtype(array.dtype, np.unsignedinteger):
            if np.any(array > 255):
                raise ValueError(f"{name} contains values outside byte range [0, 255]")
        else:
            if np.any(array < 0) or np.any(array > 255):
                raise ValueError(f"{name} contains values outside byte range [0, 255]")
        return array.astype(np.uint8, copy=False)

    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinity")
        if np.any(np.floor(array) != array):
            raise ValueError(f"{name} contains non-integer floating-point values")
        if np.any(array < 0) or np.any(array > 255):
            raise ValueError(f"{name} contains values outside byte range [0, 255]")
        return array.astype(np.uint8)

    if array.dtype.kind == "O":
        result = np.empty(array.shape, dtype=np.uint8)
        for flat_index, item in enumerate(array.flat):
            if isinstance(item, (bool, np.bool_)):
                raise TypeError(f"{name}[{flat_index}] must not be bool")
            if isinstance(item, (int, np.integer)):
                numeric = int(item)
            elif isinstance(item, (float, np.floating)):
                numeric_float = float(item)
                if not np.isfinite(numeric_float):
                    raise ValueError(f"{name}[{flat_index}] is NaN or infinity")
                if not numeric_float.is_integer():
                    raise ValueError(
                        f"{name}[{flat_index}] is not an integer: {item!r}"
                    )
                numeric = int(numeric_float)
            else:
                raise TypeError(
                    f"{name}[{flat_index}] must be numeric, got "
                    f"{type(item).__name__}: {item!r}"
                )
            if not 0 <= numeric <= 255:
                raise ValueError(
                    f"{name}[{flat_index}] is outside byte range [0, 255]: {numeric}"
                )
            result.flat[flat_index] = numeric
        return result

    raise TypeError(
        f"{name} must contain numeric byte values, got dtype={array.dtype}"
    )


def _is_scalar_like(value: Any) -> bool:
    try:
        return np.asarray(value).shape == ()
    except Exception:
        return False


def _scalar_or_array(array: np.ndarray, *, scalar: bool) -> LeakageValue:
    result = np.asarray(array, dtype=np.uint8)
    return int(result.item()) if scalar else result


# =============================================================================
# Bit-level leakage helpers
# =============================================================================
def hamming_weight(value: Any) -> LeakageValue:
    """Return the Hamming weight of byte scalar/array values."""
    bytes_array = _ensure_uint8_array(value, name="value")
    result = _HAMMING_WEIGHT_TABLE[bytes_array]
    return _scalar_or_array(result, scalar=_is_scalar_like(value))


def msb(value: Any) -> LeakageValue:
    """Return the most-significant bit of byte scalar/array values."""
    bytes_array = _ensure_uint8_array(value, name="value")
    result = np.right_shift(bytes_array, 7).astype(np.uint8, copy=False)
    return _scalar_or_array(result, scalar=_is_scalar_like(value))


# =============================================================================
# AES intermediate and leakage-label computation
# =============================================================================
def compute_aes_intermediate(
    plaintext_byte: Any,
    key_byte: Any,
) -> LeakageValue:
    """Compute ``AES_SBOX[plaintext_byte XOR key_byte]`` with NumPy broadcasting."""
    plaintext = _ensure_uint8_array(plaintext_byte, name="plaintext_byte")
    key = _ensure_uint8_array(key_byte, name="key_byte")

    try:
        plaintext_b, key_b = np.broadcast_arrays(plaintext, key)
    except ValueError as exc:
        raise ValueError(
            "plaintext_byte and key_byte are not broadcast-compatible: "
            f"{plaintext.shape} vs {key.shape}"
        ) from exc

    intermediate = _AES_SBOX[np.bitwise_xor(plaintext_b, key_b)]
    scalar = _is_scalar_like(plaintext_byte) and _is_scalar_like(key_byte)
    return _scalar_or_array(intermediate, scalar=scalar)


def apply_leakage_model(
    intermediate: Any,
    leakage_model: Any,
) -> LeakageValue:
    """Map AES intermediate byte values to ID, HW, or MSB classes."""
    model = normalize_leakage_model(leakage_model)
    values = _ensure_uint8_array(intermediate, name="intermediate")

    if model == "ID":
        labels = values
    elif model == "HW":
        labels = _HAMMING_WEIGHT_TABLE[values]
    else:  # MSB
        labels = np.right_shift(values, 7).astype(np.uint8, copy=False)

    return _scalar_or_array(labels, scalar=_is_scalar_like(intermediate))


def compute_leakage_label(
    plaintext_byte: Any,
    key_byte: Any,
    leakage_model: Any,
) -> LeakageValue:
    """Compute the configured leakage label from plaintext and key byte values."""
    intermediate = compute_aes_intermediate(plaintext_byte, key_byte)
    return apply_leakage_model(intermediate, leakage_model)


# =============================================================================
# Key-hypothesis lookup tables
# =============================================================================
@lru_cache(maxsize=len(_VALID_LEAKAGE_MODELS))
def _cached_label_table(model: LeakageModel) -> np.ndarray:
    plaintext = np.arange(256, dtype=np.uint8)[:, None]
    keys = np.arange(256, dtype=np.uint8)[None, :]
    intermediates = _AES_SBOX[np.bitwise_xor(plaintext, keys)]

    if model == "ID":
        table = intermediates.astype(np.uint8, copy=True)
    elif model == "HW":
        table = _HAMMING_WEIGHT_TABLE[intermediates].astype(np.uint8, copy=True)
    else:  # MSB
        table = np.right_shift(intermediates, 7).astype(np.uint8, copy=True)

    if table.shape != (256, 256):
        raise RuntimeError(
            "Internal leakage label table must have shape (256, 256), "
            f"got {table.shape}"
        )
    expected_classes = _NUM_CLASSES[model]
    if int(table.min()) < 0 or int(table.max()) >= expected_classes:
        raise RuntimeError(
            f"Internal {model} table contains labels outside "
            f"[0, {expected_classes - 1}]"
        )

    table.setflags(write=False)
    return table


def build_label_table(leakage_model: Any) -> np.ndarray:
    """
    Return the immutable table ``table[plaintext, key_guess] -> class_label``.

    The same read-only array is cached per leakage model to avoid rebuilding the
    256x256 hypothesis table for every epoch and repeated attack.
    """
    return _cached_label_table(normalize_leakage_model(leakage_model))


def build_key_hypothesis_labels(
    plaintext_byte: Any,
    leakage_model: Any,
) -> np.ndarray:
    """Return labels for all 256 key hypotheses for one scalar or 1-D plaintext array."""
    plaintext = _ensure_uint8_array(plaintext_byte, name="plaintext_byte")
    if plaintext.ndim > 1:
        raise ValueError(
            "plaintext_byte must be scalar or one-dimensional, "
            f"got shape={plaintext.shape}"
        )
    result = build_label_table(leakage_model)[plaintext]
    return np.asarray(result, dtype=np.uint8)


def validate_num_classes_for_leakage(
    num_classes: Any,
    leakage_model: Any,
) -> None:
    """Require the model output dimension to equal the leakage class count."""
    actual = _strict_integer(num_classes, field="num_classes", minimum=1)
    model = normalize_leakage_model(leakage_model)
    expected = _NUM_CLASSES[model]
    if actual != expected:
        raise ValueError(
            f"leakage_model={model} requires num_classes={expected}, "
            f"got {actual}"
        )


# =============================================================================
# Metadata convenience API
# =============================================================================
def compute_label_from_trace_metadata(
    plaintext: Any,
    key: Any,
    *,
    attack_byte: Any,
    leakage_model: Any,
) -> int:
    """Compute one trace label from full 16-byte plaintext/key metadata."""
    byte_index = _strict_integer(
        attack_byte,
        field="attack_byte",
        minimum=0,
        maximum=15,
    )
    plaintext_array = _ensure_uint8_array(plaintext, name="plaintext")
    key_array = _ensure_uint8_array(key, name="key")

    if plaintext_array.shape != (16,):
        raise ValueError(
            f"plaintext must have shape (16,), got {plaintext_array.shape}"
        )
    if key_array.shape != (16,):
        raise ValueError(f"key must have shape (16,), got {key_array.shape}")

    return int(
        compute_leakage_label(
            plaintext_array[byte_index],
            key_array[byte_index],
            leakage_model,
        )
    )


# =============================================================================
# Label targets: the intermediate a database type's labels come from
# =============================================================================
# sbox         Sbox[p[b] ^ k]                       ASCAD, CHES CTF 2018, AES_RD (first round)
# sbox_masked  Sbox[p[b] ^ k] ^ mask                DPAv4 with its known mask (Zaid et al. 2020,
#                                                   DPA-contest v4/exploit_pred.py:70); `mask` is the
#                                                   per-trace mask byte stored in the metadata
# inv_sbox_hd  InvSbox[c[b] ^ k] ^ c[COMPANION[b]]  AES_HD last round (Zaid et al. 2020,
#                                                   AES_HD/exploit_pred.py:80: bytes 11 and 7, 0-based)
TARGETS = ("sbox", "sbox_masked", "inv_sbox_hd")
HD_COMPANION_BYTE = {11: 7}
_AES_INV_SBOX = _validated_aes_sbox(AES_INV_SBOX)
if not np.array_equal(_AES_INV_SBOX[_AES_SBOX], np.arange(256)):
    raise RuntimeError("utils.aes.AES_INV_SBOX does not invert AES_SBOX")


def _field(rows: Any, name: str) -> np.ndarray:
    """One metadata field of N trace rows (a structured array, a mapping of arrays, or a list of dicts)."""
    if isinstance(rows, np.ndarray) and rows.dtype.names:
        if name not in rows.dtype.names:
            raise KeyError(f"metadata has no {name!r} field")
        return np.asarray(rows[name])
    if isinstance(rows, Mapping):
        return np.asarray(rows[name])
    return np.stack([np.asarray(row[name]) for row in rows])


def _target_intermediate(rows: Any, keys: np.ndarray, *, target: str, byte: int) -> np.ndarray:
    """The target's intermediate for every row and key value; keys is [N] or [1, 256] (broadcast)."""
    byte = _strict_integer(byte, field="attack_byte", minimum=0, maximum=15)
    if target == "sbox":
        plaintext = _ensure_uint8_array(_field(rows, "plaintext")[:, byte], name="plaintext")
        return _AES_SBOX[np.bitwise_xor(plaintext.reshape(-1, 1), keys)]
    if target == "sbox_masked":
        plaintext = _ensure_uint8_array(_field(rows, "plaintext")[:, byte], name="plaintext")
        mask = _ensure_uint8_array(_field(rows, "mask").reshape(-1), name="mask")
        return np.bitwise_xor(_AES_SBOX[np.bitwise_xor(plaintext.reshape(-1, 1), keys)], mask.reshape(-1, 1))
    if target == "inv_sbox_hd":
        if byte not in HD_COMPANION_BYTE:
            raise ValueError(f"target inv_sbox_hd is defined for attack byte(s) {sorted(HD_COMPANION_BYTE)}, got {byte}")
        ciphertext = _ensure_uint8_array(_field(rows, "ciphertext"), name="ciphertext")
        return np.bitwise_xor(_AES_INV_SBOX[np.bitwise_xor(ciphertext[:, byte].reshape(-1, 1), keys)],
                              ciphertext[:, HD_COMPANION_BYTE[byte]].reshape(-1, 1))
    raise ValueError(f"unknown label target {target!r}; known: {list(TARGETS)}")


def target_labels(rows: Any, *, target: str, byte: int, leakage_model: Any) -> np.ndarray:
    """One label per trace under its own key: the leakage model of the target's intermediate."""
    keys = _ensure_uint8_array(_field(rows, "key")[:, int(byte)], name="key").reshape(-1, 1)
    intermediate = _target_intermediate(rows, keys, target=target, byte=byte)[:, 0]
    return np.asarray(apply_leakage_model(intermediate, leakage_model), dtype=np.uint8)


def target_hypotheses(rows: Any, *, target: str, byte: int, leakage_model: Any) -> np.ndarray:
    """``[N, 256]``: the label of every trace under every key guess (the attack's hypotheses)."""
    guesses = np.arange(256, dtype=np.uint8).reshape(1, 256)
    intermediate = _target_intermediate(rows, guesses, target=target, byte=byte)
    return np.asarray(apply_leakage_model(intermediate, leakage_model), dtype=np.uint8)


__all__ = [
    "LeakageModel",
    "TARGETS",
    "target_hypotheses",
    "target_labels",
    "apply_leakage_model",
    "build_key_hypothesis_labels",
    "build_label_table",
    "compute_aes_intermediate",
    "compute_label_from_trace_metadata",
    "compute_leakage_label",
    "get_num_classes",
    "get_supported_leakage_models",
    "hamming_weight",
    "msb",
    "normalize_leakage_model",
    "resolve_runtime_leakage_model",
    "validate_num_classes_for_leakage",
]


# TODO: Add HD only after defining the exact state transition required by the
# target dataset/paper. Add masked or higher-order leakage helpers only with an
# equally explicit label and metadata contract.
