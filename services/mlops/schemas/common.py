# services/mlops/schemas/common.py
# ============================================================
# Common schema primitives for MLSCA AutoMLOps
# ============================================================

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union, Iterable, Mapping


# ============================================================
# Enums (shared vocabulary)
# ============================================================

class Direction(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Decision(str, Enum):
    ACCEPT = "accept"
    PRUNE = "prune"
    REJECT = "reject"
    RETRY = "retry"


# ============================================================
# Validation helpers (NO side effects)
# ============================================================

def require_keys(obj: Dict[str, Any], keys: Sequence[str], *, ctx: str = "") -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        prefix = f"[{ctx}] " if ctx else ""
        raise ValueError(f"{prefix}missing required keys: {missing}")


def ensure_one_of(value: Any, allowed: Sequence[Any], *, ctx: str = "") -> None:
    if value not in allowed:
        prefix = f"[{ctx}] " if ctx else ""
        raise ValueError(f"{prefix}value '{value}' not in allowed set {list(allowed)}")


def ensure_non_empty(seq: Sequence[Any], *, ctx: str = "") -> None:
    if not seq:
        prefix = f"[{ctx}] " if ctx else ""
        raise ValueError(f"{prefix}must be non-empty")


# ============================================================
# Utility normalization helpers
# ============================================================

def normalize_list(val: Optional[Union[Any, Sequence[Any]]]) -> List[Any]:
    """
    Accept scalar or list → return list.
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def normalize_bool(val: Any, *, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("1", "true", "yes", "on")
    return bool(val)

def normalize_dict(
    val: Any,
    *,
    default: Optional[Dict[str, Any]] = None,
    ctx: str = "",
) -> Dict[str, Any]:
    """
    Normalize dict-like inputs.

    Accepts:
      - None            -> default or {}
      - dict            -> shallow copy
      - dataclass obj   -> __dict__ (best-effort)
      - anything else   -> error

    Args:
        val: input value
        default: fallback dict if val is None
        ctx: context string for clearer error messages

    Returns:
        dict (shallow copy)

    Raises:
        ValueError if cannot normalize to dict.
    """
    if val is None:
        return dict(default or {})
    if isinstance(val, dict):
        return dict(val)
    # best-effort: mapping types
    if isinstance(val, Mapping):
        return dict(val)
    # best-effort: dataclass / object with __dict__
    d = getattr(val, "__dict__", None)
    if isinstance(d, dict):
        return dict(d)
    prefix = f"[{ctx}] " if ctx else ""
    raise ValueError(f"{prefix}cannot normalize dict from value={val!r} (type={type(val).__name__})")


def normalize_int(val: Any, *, default: Optional[int] = None) -> Optional[int]:
    if val is None:
        return default
    try:
        return int(val)
    except Exception:
        raise ValueError(f"cannot normalize int from value={val!r}")

def normalize_number(
    val: Any,
    *,
    default: Optional[float] = None,
    allow_int: bool = True,
    allow_float: bool = True,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    ctx: str = "",
) -> Optional[float]:
    """
    Normalize numeric inputs (int/float/str) to float.

    Intended for schema fields where YAML might supply:
      - 3
      - 3.0
      - "3"
      - "3.14"

    Returns:
      float (or default if val is None)

    Raises:
      ValueError if val cannot be parsed or violates bounds.
    """
    if val is None:
        return default

    if isinstance(val, bool):
        # bool is an int subclass; reject explicitly
        prefix = f"[{ctx}] " if ctx else ""
        raise ValueError(f"{prefix}cannot normalize number from bool")

    num: Optional[float] = None

    if isinstance(val, (int, float)):
        if isinstance(val, int) and not allow_int:
            prefix = f"[{ctx}] " if ctx else ""
            raise ValueError(f"{prefix}int not allowed here: {val!r}")
        if isinstance(val, float) and not allow_float:
            prefix = f"[{ctx}] " if ctx else ""
            raise ValueError(f"{prefix}float not allowed here: {val!r}")
        num = float(val)
    else:
        s = str(val).strip()
        if s == "":
            return default
        try:
            # allow "3" or "3.14"
            num = float(s)
            if (num.is_integer() and not allow_int) or (not num.is_integer() and not allow_float):
                prefix = f"[{ctx}] " if ctx else ""
                raise ValueError(f"{prefix}number type not allowed: {val!r}")
        except Exception:
            prefix = f"[{ctx}] " if ctx else ""
            raise ValueError(f"{prefix}cannot normalize number from value={val!r}")

    if min_value is not None and num < float(min_value):
        prefix = f"[{ctx}] " if ctx else ""
        raise ValueError(f"{prefix}value {num} < min_value {min_value}")
    if max_value is not None and num > float(max_value):
        prefix = f"[{ctx}] " if ctx else ""
        raise ValueError(f"{prefix}value {num} > max_value {max_value}")

    return num

def normalize_str(
    val: Any,
    *,
    default: Optional[str] = None,
    strip: bool = True,
    lower: bool = False,
    non_empty: bool = False,
    ctx: str = "",
) -> str:
    """
    Normalize string-like inputs.

    Args:
        val: input value
        default: used when val is None
        strip: whether to strip whitespace
        lower: whether to lowercase
        non_empty: if True, raise if result is empty after normalization
        ctx: context string for error messages

    Returns:
        str

    Raises:
        ValueError if non_empty=True and normalized string is empty.
    """
    if val is None:
        s = "" if default is None else str(default)
    else:
        s = str(val)

    if strip:
        s = s.strip()
    if lower:
        s = s.lower()

    if non_empty and s == "":
        prefix = f"[{ctx}] " if ctx else ""
        raise ValueError(f"{prefix}string must be non-empty")

    return s

def normalize_float(val: Any, *, default: Optional[float] = None) -> Optional[float]:
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        raise ValueError(f"cannot normalize float from value={val!r}")


def is_number(val: Any) -> bool:
    """
    Return True if val is a real number (int or float), excluding bool.

    Used by schema validators to distinguish numeric hyperparameters
    from categorical / symbolic values.
    """
    if isinstance(val, bool):
        return False
    return isinstance(val, (int, float))


def forbid_unknown_keys(
    obj: Dict[str, Any],
    allowed: Iterable[str],          # <-- allow positional 2nd arg
    *,
    where: str = "",
    ctx: Optional[str] = None,       # <-- legacy alias
    strict: bool = True,
) -> None:
    """
    Ensure obj contains no keys outside `allowed`.

    Compatible with BOTH call styles:
      - forbid_unknown_keys(cfg, TOP_LEVEL_KEYS, ctx="auto_mlops")
      - forbid_unknown_keys(cfg, allowed=TOP_LEVEL_KEYS, where="auto_mlops")

    Args:
        obj: dict to validate
        allowed: iterable of allowed key names
        where: context string used in error messages (preferred)
        ctx: alias for `where` (compat with older call sites)
        strict: if True -> raise ValueError; if False -> do nothing
    """
    if not strict:
        return
    if not isinstance(obj, dict):
        return

    loc = where or (ctx or "")
    allowed_set = {str(k) for k in allowed}
    unknown = [k for k in obj.keys() if str(k) not in allowed_set]

    if unknown:
        prefix = f"{loc}: " if loc else ""
        raise ValueError(f"{prefix}unknown keys not allowed: {unknown}")

# ============================================================
# End of common schema
# ============================================================
