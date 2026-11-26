# services/visualization/utils/helpers.py

import re
from pathlib import Path
from typing import Any, Mapping, Tuple
import socket
from urllib.parse import urlparse

def _extract_trial_suffix(cfg: Mapping[str, Any]) -> str:
    # explicit flags first
    if not bool(cfg.get("rerun")) and not bool(cfg.get("strict_rerun_only")):
        return ""

    paths = cfg.get("paths") or {}
    artifacts_root = paths.get("artifacts_root") or ""
    if not artifacts_root:
        return ""

    name = Path(str(artifacts_root)).name
    m = re.search(r"_trial_(\d+)$", name)
    if not m:
        return ""

    return f"_trial_{m.group(1)}"

def _probe_tcp(url: str, timeout: float = 3.0) -> Tuple[bool, str]:
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port

        if port is None:
            if parsed.scheme == "http":
                port = 80
            elif parsed.scheme == "https":
                port = 443

        if not host or not port:
            return False, f"invalid url for tcp probe: {url}"

        resolved = socket.gethostbyname(host)
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"tcp ok host={host} resolved={resolved} port={port}"

    except Exception as e:
        return False, f"tcp probe failed for {url}: {e}"


def debug_resolve(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port

        if port is None:
            if parsed.scheme == "http":
                port = 80
            elif parsed.scheme == "https":
                port = 443

        if not host or not port:
            return f"invalid url for resolve: {url}"

        ip = socket.gethostbyname(host)
        return f"host={host} ip={ip} port={port}"
    except Exception as e:
        return f"resolve failed for {url}: {e}"
