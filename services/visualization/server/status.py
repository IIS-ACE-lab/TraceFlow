# services/visualization/server/status.py
"""
The status page: what the TraceFlow Monitor's project headings show, read-only, in a phone's browser (saved to
the home screen it opens like an app) and for the people the owner shares a link with.

    python -m services.visualization.server.status serve [--host ADDRESS] [--port PORT]   (default: master.yaml status_share)
    python -m services.visualization.server.status settings
    python -m services.visualization.server.status share add NAME | list | remove NAME
    python -m services.visualization.server.status qr TEXT
    python -m services.visualization.server.status addresses

Settings: master.yaml `status_share: {enabled, port, host}`, optional: without it, or with enabled false, the page
is off (a server that may not open a port). port: a TCP port the network lets through to this machine. host, optional:
the address the links carry and the server listens on; without it, this machine's own (auto_host: the address its
default route leaves from, the one a phone on its network or VPN reaches). serve takes them from there (--host /
--port override); settings prints them for the panel.

serve: a standard-library HTTP server on one address (host; never every address), and on this machine's loopback at the same port, which VS Code forwards to the owner's desktop. Read-only: GET only, nothing starts, stops or writes a run. The page and its data need a share's token
(?t=...), checked against secrets/status_shares.json on every request, so a withdrawn link stops at once.

    /                      the page (services/visualization/mobile/index.html)    token
    /status.json           the facts below                                         token
    /manifest.webmanifest  the home-screen app; its start_url keeps the token     token
    /static/<file>         the page's script and style, the panel's state.js, the home-screen icons (no data)

/status.json, per project of master.yaml: each env's heartbeat (without its pid), the run registry's rows with
the fields the panel's state.js reads (no paths), the project's queues, the queue keys of the experiment
commands running now, the running run's latest metrics (core/monitor.snapshot); nvidia-smi's GPUs; and this
machine's clock, so the page ages the heartbeats on it. The page turns them into lights, phases, progress, ETA
and counts with the panel's own state.js.

share: one link per person (a name and a random token). qr: an SVG QR code of a link (segno). addresses: this
machine's IPv4 addresses a phone could reach (not loopback, Docker or container bridges).

Author: Shraddha Mishra
"""

from __future__ import annotations

import argparse
import hmac
import io
import json
import mimetypes
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[3]
PAGE = Path(__file__).resolve().parents[1] / "mobile"            # the page: services/visualization/mobile
STATIC = {                                   # what /static/ serves: nothing else is reachable
    "status.js": PAGE / "status.js",
    "status.css": PAGE / "status.css",
    "shim.js": PAGE / "shim.js",
    "manifest.js": PAGE / "manifest.js",
    "state.js": ROOT / "services" / "visualization" / "vscode_monitor" / "state.js",
    # the home-screen icons: the Activity Bar mark (brand/make_brand.py web_icons)
    **{f"icon-{px}.png": ROOT / "services" / "visualization" / "vscode_monitor" / "brand" / f"traceflow-web-{px}.png" for px in (180, 192, 512)},
}
ENVS = ("dev", "staging", "prod")
# the registry fields state.js's registryEntries reads, without its path
REGISTRY_FIELDS = ("model_name", "model_id", "env", "status", "status_reason", "batches", "batch_ids", "started_at",
                   "finished_at", "run_session_id", "trial_index", "total_trials", "total_remaining_trials")
GPU_QUERY = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"]                       # vscode_monitor/gpu.js's query
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,39}$")
CACHE_S = 2.0                                # several viewers share one read of the files


# ── master.yaml status_share ─────────────────────────────────────────────────────

def auto_host() -> str:
    """This machine's address on its network: the one its default route leaves from (a UDP socket's connect only
    routes, it sends nothing), else the first of addresses(), else the loopback (no network: this machine only)."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))                  # TEST-NET-1 (RFC 5737): routed, never reached
            return probe.getsockname()[0]
    except OSError:
        found = addresses()
        return found[0] if found else "127.0.0.1"


def settings(root: Path = ROOT) -> dict[str, Any]:
    """master.yaml status_share: {enabled, host, port, auto, why}: host as given, else auto_host() (auto true); why
    says what is missing when it is off or incomplete."""
    import yaml

    from services.visualization.vscode_monitor.layout import MASTER

    try:
        master = yaml.safe_load((root / MASTER).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {"enabled": False, "host": "", "port": None, "auto": False, "why": f"master.yaml cannot be read: {exc}"}
    block = master.get("status_share") if isinstance(master, dict) else None
    if not isinstance(block, dict):
        return {"enabled": False, "host": "", "port": None, "auto": False,
                "why": "master.yaml has no status_share block: the status page is off (optional)"}
    host, port = str(block.get("host") or "").strip(), block.get("port")
    why = ("master.yaml status_share.enabled is false: the status page is off" if block.get("enabled") is not True
           else "master.yaml status_share.port is not a port number (1024 to 65535)"
           if not (isinstance(port, int) and 1024 <= port <= 65535) else "")
    return {"enabled": not why, "host": host or auto_host(), "port": port if isinstance(port, int) else None,
            "auto": not host, "why": why}


# ── share links ──────────────────────────────────────────────────────────────

def shares_path() -> Path:
    return Path(os.environ.get("TRACEFLOW_STATUS_SHARES") or ROOT / "secrets" / "status_shares.json")


def _load_shares() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(shares_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data.get("shares", {}) if isinstance(data, dict) and isinstance(data.get("shares"), dict) else {}


def _save_shares(shares: dict[str, dict[str, Any]]) -> None:
    path = shares_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"shares": shares}, indent=2) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)                     # the tokens are the links: the owner's only
    os.replace(tmp, path)


def share_list() -> list[dict[str, Any]]:
    return [{"name": n, "token": s.get("token"), "created": s.get("created")} for n, s in sorted(_load_shares().items())]


def share_add(name: str) -> dict[str, Any]:
    name = str(name).strip()
    if not NAME.match(name):
        raise ValueError("a name of 1 to 40 letters, digits, spaces, dots, dashes or underscores, starting with a letter or digit")
    shares = _load_shares()
    if name in shares:
        raise ValueError(f"{name} has a link already: withdraw it first to make a new one")
    shares[name] = {"token": secrets.token_urlsafe(18), "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _save_shares(shares)
    return {"name": name, **shares[name]}


def share_remove(name: str) -> dict[str, Any]:
    shares = _load_shares()
    if name not in shares:
        raise ValueError(f"no link named {name}")
    del shares[name]
    _save_shares(shares)
    return {"removed": name}


def viewer(token: Optional[str]) -> Optional[str]:
    """The name of the share whose token this is (compared in constant time), else None."""
    if not token:
        return None
    for name, share in _load_shares().items():
        if hmac.compare_digest(str(share.get("token", "")), str(token)):
            return name
    return None


# ── what the page shows ──────────────────────────────────────────────────────

def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _gpus() -> list[dict[str, Any]]:
    try:
        out = subprocess.run(GPU_QUERY, capture_output=True, text=True, timeout=4, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 6:
            num = lambda v: float(v) if re.match(r"^-?\d+(\.\d+)?$", v) else None   # noqa: E731
            gpus.append({"index": num(parts[0]), "name": parts[1], "util": num(parts[2]), "memUsed": num(parts[3]),
                         "memTotal": num(parts[4]), "temp": num(parts[5])})
    return gpus


def _queue_commands() -> list[str]:
    """The queue keys of the experiment commands running now, as `QUEUE=<key>` (state.js activeQueues reads them)."""
    try:
        out = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True, timeout=4, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    keys = {m.group(1) for line in out.splitlines() if re.search(r"make\s+experiment|experiment_queue", line)
            for m in [re.search(r"(?:^|\s)(?:QUEUE=|--queue[ =])(\S+)", line)] if m}
    return [f"QUEUE={k}" for k in sorted(keys)]


def _live(proj: Path, registry: dict[str, Any], heartbeats: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The running run's latest metrics (core/monitor.snapshot), found through its registry row's folder."""
    from core.monitor import snapshot
    from utils.run_map import project_relative_path

    for hb in heartbeats.values():
        entry = registry.get(str(hb.get("model_hash"))) if hb.get("status") == "running" else None
        if isinstance(entry, dict) and entry.get("path"):
            run_dir = proj / project_relative_path(entry["path"], proj)
            try:
                return {"hash": hb.get("model_hash"), **snapshot(run_dir)} if run_dir.is_dir() else None
            except Exception:  # noqa: BLE001 - metrics half-written: none this time
                return None
    return None


_trials: dict[str, tuple[float, list]] = {}


def _trials_of(name: str, proj: Path) -> list[dict[str, Any]]:
    """The project's sweep (layout.trials: each trial's epochs and passes), for the batch ETA: read again only when
    params.yaml changes, as the panel does (about a second each time)."""
    try:
        mtime = (proj / "config" / "params.yaml").stat().st_mtime
    except OSError:
        return []
    held = _trials.get(name)
    if held and held[0] == mtime:
        return held[1]
    try:
        from services.visualization.vscode_monitor.layout import trials

        out = trials(name)["trials"]
    except Exception:  # noqa: BLE001 - a sweep that does not load: no batch ETA this time
        out = held[1] if held else []
    _trials[name] = (mtime, out)
    return out


def project_status(name: str, folder: str, root: Path = ROOT) -> dict[str, Any]:
    from services.visualization.vscode_monitor.layout import _load, _queues

    proj = root / "projects" / folder
    art = proj / "artifacts"
    heartbeats = {}
    for env in ENVS:
        hb = _json(art / env / "heartbeat.json")
        if isinstance(hb, dict):
            heartbeats[env] = {k: v for k, v in hb.items() if k not in ("pid", "_tfseal")}
    raw = _json(art / "run_registry.json")
    raw = raw if isinstance(raw, dict) else {}
    registry = {h: {k: e.get(k) for k in REGISTRY_FIELDS} for h, e in raw.items() if isinstance(e, dict)}
    queue_file = proj / "config" / "experiment_queue.yaml"
    queues = _queues(_load(queue_file, [])) if queue_file.is_file() else []
    running = any(hb.get("status") == "running" for hb in heartbeats.values())
    return {"name": name, "heartbeats": heartbeats, "registry": registry, "queues": queues,
            "trials": _trials_of(name, proj) if running else [],   # the batch ETA's sweep, read while a run trains
            "live": _live(proj, raw, heartbeats)}


_cache: dict[str, Any] = {"at": 0.0, "data": None}
_lock = threading.Lock()


def status(root: Path = ROOT) -> dict[str, Any]:
    from services.visualization.vscode_monitor.layout import declared_projects

    with _lock:
        if _cache["data"] is not None and time.monotonic() - _cache["at"] < CACHE_S:
            return _cache["data"]
        projects = [project_status(p["name"], p["folder"], root) for p in declared_projects(root, []) if p.get("complete")]
        from services.visualization.vscode_monitor.layout import _timezone

        data = {"now_ms": time.time() * 1000, "timezone": _timezone(), "projects": projects, "gpus": _gpus(),
                "commands": _queue_commands()}
        _cache.update(at=time.monotonic(), data=data)
        return data


# ── the HTTP server ──────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "TraceFlowStatus"
    sys_version = ""

    def log_message(self, fmt: str, *args: Any) -> None:     # the path without its query: tokens stay out of logs
        sys.stderr.write(f"[status] {self.address_string()} {self.command} {urlsplit(self.path).path} {args[1] if len(args) > 1 else ''}\n")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; script-src 'self'; "
                         "style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        url = urlsplit(self.path)
        token = (parse_qs(url.query).get("t") or [None])[0]
        if url.path.startswith("/static/"):
            file = STATIC.get(url.path[len("/static/"):])
            if file is None or not file.is_file():
                return self._send(404, b"not found", "text/plain; charset=utf-8")
            return self._send(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
        who = viewer(token)
        if url.path in ("/", "/status.json", "/manifest.webmanifest") and who is None:
            why = (b"This page opens with a personal link: <address>:<port>/?t=<token> (the TraceFlow panel's share button)."
                   if not token else b"This link is not valid (withdrawn, or mistyped): ask for a new one.")
            return self._send(403, why, "text/plain; charset=utf-8")
        if url.path == "/":
            return self._send(200, (PAGE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if url.path == "/status.json":
            return self._send(200, json.dumps({**status(), "viewer": who}).encode(), "application/json")
        if url.path == "/manifest.webmanifest":
            manifest = {"name": "TraceFlow status", "short_name": "TraceFlow", "start_url": f"/?t={token}", "display": "standalone",
                        "background_color": "#1e1e1e", "theme_color": "#1e1e1e",
                        "icons": [{"src": f"/static/icon-{px}.png", "sizes": f"{px}x{px}", "type": "image/png"} for px in (192, 512)]}
            return self._send(200, json.dumps(manifest).encode(), "application/manifest+json")
        return self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        self._send(405, b"read-only", "text/plain; charset=utf-8")

    do_PUT = do_DELETE = do_PATCH = do_POST


def serve(host: str, port: int) -> None:
    if str(ROOT / "services" / "side_channel_core" / "src") not in sys.path:   # core.monitor, utils.*
        sys.path[:0] = [str(ROOT), str(ROOT / "services" / "side_channel_core"), str(ROOT / "services" / "side_channel_core" / "src")]
    httpd = ThreadingHTTPServer((host, port), Handler)
    # and this machine's loopback, same port: VS Code forwards it to the owner's desktop (the panel's link opens it there)
    local = None
    if host not in ("127.0.0.1", "localhost"):
        try:
            local = ThreadingHTTPServer(("127.0.0.1", httpd.server_address[1]), Handler)
            threading.Thread(target=local.serve_forever, daemon=True).start()
        except OSError as exc:                  # e.g. an `ssh -L <port>:...` tunnel on this machine holds it: it leads here too
            print(f"[status] not on 127.0.0.1:{httpd.server_address[1]} ({exc.strerror}): the address serves alone", flush=True)
    print(f"[status] http://{host}:{httpd.server_address[1]}/{' and http://127.0.0.1:%d/' % httpd.server_address[1] if local else ''} "
          "(read-only; each link has its own token: the panel's share button)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        if local:
            local.shutdown()
            local.server_close()


# ── helpers for the panel ────────────────────────────────────────────────────

def qr_svg(text: str) -> str:
    """An SVG QR code that scales to its box (a viewBox, no fixed size: a fixed one is cut off by a smaller box),
    with the standard quiet zone of 4 modules that phone cameras need, black on white in any theme."""
    import segno

    out = io.BytesIO()
    segno.make(text, error="m").save(out, kind="svg", xmldecl=False, svgns=True, omitsize=True, scale=1, border=4,
                                     dark="#000", light="#fff")
    return out.getvalue().decode()


def addresses() -> list[str]:
    """This machine's IPv4 addresses a phone could reach: not loopback, Docker or container bridges."""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True, timeout=4, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and not re.match(r"^(lo|docker\d*|nerdctl\d*|br-|veth|cni|flannel|virbr)", parts[1]):
            found.append(parts[3].split("/")[0])
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The read-only status page and its share links.")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default=None, help="the address to listen on; default master.yaml status_share.host, else this machine's own")
    s.add_argument("--port", type=int, default=None, help="default master.yaml status_share.port")
    sub.add_parser("settings")
    sh = sub.add_parser("share")
    sh.add_argument("action", choices=["add", "list", "remove"])
    sh.add_argument("name", nargs="?")
    q = sub.add_parser("qr")
    q.add_argument("text")
    sub.add_parser("addresses")
    args = parser.parse_args(argv)
    if args.command == "serve":
        cfg = settings()
        if not cfg["enabled"] and not (args.host and args.port):
            print(f"[status] not served: {cfg['why']}", file=sys.stderr)
            return 2
        serve(args.host or cfg["host"], args.port or cfg["port"])
        return 0
    try:
        out: Any = (settings() if args.command == "settings"
                    else {"addresses": addresses()} if args.command == "addresses"
                    else {"svg": qr_svg(args.text)} if args.command == "qr"
                    else {"shares": share_list()} if args.action == "list"
                    else share_add(args.name or "") if args.action == "add"
                    else share_remove(args.name or ""))
    except ValueError as exc:
        out = {"error": str(exc)}
    print(json.dumps(out))
    return 0 if not (isinstance(out, dict) and out.get("error")) else 1


if __name__ == "__main__":
    sys.exit(main())
