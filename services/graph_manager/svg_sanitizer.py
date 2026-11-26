# services/graph_manager/svg_sanitizer.py
"""
SVG sanitizer and validator for TraceFlow GraphManager.

Purpose
-------
Matplotlib-generated SVGs are usually safe and valid, but production graph
export should still protect against:

- malformed XML
- null/control characters
- embedded script tags
- external URL/file references
- empty or unreadable SVG files

This module uses only Python standard library tools.

Policy
------
- PNG/PDF export must not depend on SVG success.
- SVG failures can warn or fail based on:
    export.svg.on_error: "warn" | "fail"

Author: Shraddha Mishra
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping


class SVGValidationError(ValueError):
    """Raised when an SVG fails validation or sanitization."""


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

_SCRIPT_BLOCK_RE = re.compile(
    r"<\s*script\b[^>]*>.*?<\s*/\s*script\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)

_EVENT_HANDLER_RE = re.compile(
    r"""\s+on[a-zA-Z]+\s*=\s*(['"]).*?\1""",
    flags=re.IGNORECASE | re.DOTALL,
)

_EXTERNAL_REF_RE = re.compile(
    r"""(?:href|xlink:href)\s*=\s*(['"])\s*(?:https?://|file://|ftp://).*?\1""",
    flags=re.IGNORECASE | re.DOTALL,
)

_JAVASCRIPT_REF_RE = re.compile(
    r"""(?:href|xlink:href)\s*=\s*(['"])\s*javascript:.*?\1""",
    flags=re.IGNORECASE | re.DOTALL,
)

_DOCTYPE_RE = re.compile(
    r"<!DOCTYPE[^>]*>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _svg_cfg(job: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(job, Mapping):
        return {}

    export = job.get("export", {}) or {}
    if not isinstance(export, Mapping):
        return {}

    svg = export.get("svg", {}) or {}
    if not isinstance(svg, Mapping):
        return {}

    return svg


def svg_on_error(job: Mapping[str, Any] | None) -> str:
    """
    Return SVG error policy.

    Valid values:
      warn
      fail
    """

    mode = str(_svg_cfg(job).get("on_error", "warn")).strip().lower()
    if mode not in {"warn", "fail"}:
        return "warn"
    return mode


def svg_sanitize_enabled(job: Mapping[str, Any] | None) -> bool:
    value = _svg_cfg(job).get("sanitize", True)

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def svg_validate_enabled(job: Mapping[str, Any] | None) -> bool:
    value = _svg_cfg(job).get("validate_xml", True)

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _warn_or_raise(message: str, job: Mapping[str, Any] | None = None) -> bool:
    """
    Return False on warning mode, raise on fail mode.
    """

    if svg_on_error(job) == "fail":
        raise SVGValidationError(message)

    print(f"[GraphManager][WARN] SVG sanitizer: {message}")
    return False


def read_svg_text(path: Path) -> str:
    """
    Read an SVG file as UTF-8 text.
    """

    path = Path(path)

    if not path.exists():
        raise SVGValidationError(f"SVG file does not exist: {path}")

    if not path.is_file():
        raise SVGValidationError(f"SVG path is not a file: {path}")

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise SVGValidationError(f"failed to read SVG file: {path}") from exc

    if not text.strip():
        raise SVGValidationError(f"SVG file is empty: {path}")

    return text


def sanitize_svg_text(text: str) -> str:
    """
    Sanitize SVG XML text.

    This is intentionally conservative:
    - removes XML control characters
    - removes doctype declarations
    - removes script blocks
    - removes inline event handlers such as onclick=
    - removes external URL/file/javascript href references
    """

    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\ufeff", "")
    text = _CONTROL_CHAR_RE.sub("", text)

    # Matplotlib SVGs do not need DOCTYPE. Removing it avoids entity risks.
    text = _DOCTYPE_RE.sub("", text)

    # Remove executable/scriptable content.
    text = _SCRIPT_BLOCK_RE.sub("", text)
    text = _EVENT_HANDLER_RE.sub("", text)
    text = _EXTERNAL_REF_RE.sub("", text)
    text = _JAVASCRIPT_REF_RE.sub("", text)

    return text.strip() + "\n"


def validate_svg_text(text: str, *, require_svg_root: bool = True) -> None:
    """
    Validate SVG text as XML.

    Raises SVGValidationError on failure.
    """

    if not text.strip():
        raise SVGValidationError("SVG text is empty after sanitization")

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise SVGValidationError(f"SVG XML parse failed: {exc}") from exc

    if require_svg_root:
        tag = str(root.tag).lower()
        if not tag.endswith("svg"):
            raise SVGValidationError(f"SVG root element is not <svg>: {root.tag!r}")


def validate_svg_file(path: Path) -> bool:
    """
    Validate an SVG file.

    Returns True if valid.
    Raises SVGValidationError if invalid.
    """

    text = read_svg_text(Path(path))
    validate_svg_text(text)
    return True


def sanitize_svg_file(path: Path) -> bool:
    """
    Sanitize an SVG file in-place.

    Returns True if the file was sanitized and remains valid.
    Raises SVGValidationError if sanitization/validation fails.
    """

    path = Path(path)
    original = read_svg_text(path)
    sanitized = sanitize_svg_text(original)
    validate_svg_text(sanitized)

    try:
        path.write_text(sanitized, encoding="utf-8")
    except Exception as exc:
        raise SVGValidationError(f"failed to write sanitized SVG file: {path}") from exc

    return True


def sanitize_and_validate_svg(
    path: Path,
    job: Mapping[str, Any] | None = None,
) -> bool:
    """
    Apply configured SVG sanitization and validation.

    Returns:
      True  -> SVG is usable
      False -> SVG failed, but export.svg.on_error == "warn"

    Raises:
      SVGValidationError -> SVG failed and export.svg.on_error == "fail"
    """

    path = Path(path)

    try:
        sanitize_enabled = svg_sanitize_enabled(job)
        validate_enabled = svg_validate_enabled(job)

        if sanitize_enabled:
            sanitize_svg_file(path)

        elif validate_enabled:
            validate_svg_file(path)

        else:
            # If both sanitize and validation are disabled, still ensure the
            # output exists and is not empty.
            read_svg_text(path)

        return True

    except SVGValidationError as exc:
        return _warn_or_raise(str(exc), job=job)

    except Exception as exc:
        return _warn_or_raise(
            f"unexpected SVG sanitizer failure for {path}: {exc}",
            job=job,
        )
