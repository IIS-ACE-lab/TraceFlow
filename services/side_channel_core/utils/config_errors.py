# services/side_channel_core/utils/config_errors.py
"""
ConfigError: a configuration file names a key or a value TraceFlow does not take.

The one base class of every configuration error, so a caller can catch them as one: the
key checks of utils/config_schema.py and utils/config_loader.apply_env_overrides raise it,
and the readers' own errors derive from it (DatasetTypeError, TestRequestError,
ModelFilterError, GraphJobValidationError, FacetValidationError). It is a ValueError, as
those were. Import it from here (services.side_channel_core.utils.config_errors), so every
import path shares one class.

Author: Shraddha Mishra
"""

from __future__ import annotations


class ConfigError(ValueError):
    """A configuration file names a key or a value TraceFlow does not take."""


__all__ = ["ConfigError"]
