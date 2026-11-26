#services/side_channel_core/utils/secrets.py
""" Secret Management Utilities 

- Load secrets from .env-style files
- Get secrets with environment variable override
- Raise clear errors for missing secrets

Author: Shraddha Mishra"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


class SecretError(RuntimeError):
    pass


def _parse_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise SecretError(f"Secret file not found: {path}")

    data: Dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def load_secret_file(path: str | Path) -> Dict[str, str]:
    return _parse_env_file(Path(path).expanduser().resolve())


def get_secret(
    key: str,
    *,
    secret_file: Optional[str | Path] = None,
    required: bool = True,
    default: Optional[str] = None,
) -> Optional[str]:
    # 1) Environment wins
    value = os.getenv(key)
    if value:
        return value

    # 2) Secret file fallback
    if secret_file:
        secrets = load_secret_file(secret_file)
        if key in secrets and secrets[key]:
            return secrets[key]

    # 3) Default / fail
    if default is not None:
        return default
    if required:
        raise SecretError(f"Missing required secret: {key}")
    return None
