# services/side_channel_core/tests/unit/test_status_share.py
"""The read-only status page (services/visualization/server/status.py): one link per person, checked on every
request; the facts it serves carry no pid and no path; GET only; tokens never reach the log."""

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from services.visualization.server import status as ST


@pytest.fixture
def shares(tmp_path, monkeypatch):
    path = tmp_path / "secrets" / "status_shares.json"
    monkeypatch.setenv("TRACEFLOW_STATUS_SHARES", str(path))
    return path


def test_a_link_per_person_owner_only_withdrawn_at_once(shares):
    a = ST.share_add("Professor")
    assert len(a["token"]) >= 24 and oct(os.stat(shares).st_mode & 0o777) == "0o600"
    assert ST.viewer(a["token"]) == "Professor" and ST.viewer("wrong") is None and ST.viewer("") is None
    with pytest.raises(ValueError, match="has a link already"):
        ST.share_add("Professor")
    with pytest.raises(ValueError, match="a name of 1 to 40"):
        ST.share_add("../etc")
    assert [s["name"] for s in ST.share_list()] == ["Professor"]
    ST.share_remove("Professor")
    assert ST.viewer(a["token"]) is None
    with pytest.raises(ValueError, match="no link named"):
        ST.share_remove("Professor")


def test_the_facts_carry_no_pid_and_no_path(tmp_path):
    proj = tmp_path / "projects" / "P"
    (proj / "artifacts" / "dev").mkdir(parents=True)
    (proj / "config").mkdir()
    (proj / "artifacts" / "dev" / "heartbeat.json").write_text(json.dumps(
        {"status": "done", "pid": 1234, "written_ns": 1, "model_name": "m", "_tfseal": "x"}))
    (proj / "artifacts" / "run_registry.json").write_text(json.dumps(
        {"h1": {"model_name": "m", "status": "DONE", "path": "P/artifacts/dev/generated_models/m", "fold_key_recovery": []}}))
    (proj / "config" / "experiment_queue.yaml").write_text("fig1:\n  - make orchestrate ENV=dev PROJECT=P DATASET=d MODEL_ID=M\n")
    out = ST.project_status("P", "P", tmp_path)
    assert out["heartbeats"]["dev"] == {"status": "done", "written_ns": 1, "model_name": "m"}
    assert set(out["registry"]["h1"]) == set(ST.REGISTRY_FIELDS) and out["registry"]["h1"]["model_name"] == "m"
    assert "path" not in json.dumps(out["registry"]) and out["queues"][0]["key"] == "fig1" and out["live"] is None


@pytest.fixture
def server(shares, monkeypatch):
    monkeypatch.setattr(ST, "status", lambda root=ST.ROOT: {"now_ms": 1, "projects": [], "gpus": [], "commands": []})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ST.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url, method="GET"):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method=method)) as res:
            return res.status, res.headers, res.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers, err.read()


def test_the_page_and_its_data_need_a_live_link_and_nothing_but_get_answers(server, capsys):
    base = server
    token = ST.share_add("me")["token"]
    code, headers, body = _get(f"{base}/?t={token}")
    assert code == 200 and b"/static/state.js" in body
    assert headers["Cache-Control"] == "no-store" and headers["Referrer-Policy"] == "no-referrer"
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    code, _, body = _get(f"{base}/status.json?t={token}")
    assert code == 200 and json.loads(body)["viewer"] == "me"
    manifest = json.loads(_get(f"{base}/manifest.webmanifest?t={token}")[2])
    assert manifest["start_url"] == f"/?t={token}" and manifest["display"] == "standalone"
    assert [i["src"] for i in manifest["icons"]] == ["/static/icon-192.png", "/static/icon-512.png"]
    for px in (180, 192, 512):                                   # the Activity Bar mark, as the phone's icon
        code, headers, png = _get(f"{base}/static/icon-{px}.png")
        assert code == 200 and headers["Content-Type"] == "image/png" and png[:4] == b"\x89PNG"
    assert _get(f"{base}/?t=wrong")[0] == 403 and _get(f"{base}/status.json")[0] == 403
    assert b"personal link" in _get(f"{base}/")[2] and b"not valid" in _get(f"{base}/?t=wrong")[2]
    assert _get(f"{base}/static/state.js")[0] == 200 and _get(f"{base}/static/../status.py")[0] == 404
    assert _get(f"{base}/?t={token}", "POST")[0] == 405
    ST.share_remove("me")
    assert _get(f"{base}/status.json?t={token}")[0] == 403, "a withdrawn link stops at once"
    log = capsys.readouterr().err                      # the request log (pytest captures stderr)
    assert token not in log and "GET /status.json" in log


def test_a_qr_code_is_an_svg_of_attributes_only_that_scales_whole_with_its_quiet_zone():
    import re

    svg = ST.qr_svg("http://192.168.88.2:8765/?t=" + "A" * 24)       # a real link's length
    assert svg.startswith("<svg") and "style=" not in svg and "<script" not in svg
    assert "width=" not in svg.split(">", 1)[0], "no fixed size: a box smaller than it would crop the code"
    size = int(re.search(r'viewBox="0 0 (\d+) \1"', svg).group(1))
    assert re.search(rf'd="M0 0h{size}v{size}h-{size}z"', svg), "the white square spans the whole viewBox"
    assert re.search(r'stroke="#000" d="M4 4\.5h', svg), "the code starts 4 modules in: the quiet zone phone cameras need"


@pytest.mark.parametrize("block, enabled, why", [
    (None, False, "master.yaml has no status_share block: the status page is off (optional)"),
    ({"enabled": False, "host": "10.0.0.2", "port": 8765}, False, "master.yaml status_share.enabled is false: the status page is off"),
    ({"enabled": True, "port": 8765}, True, ""),                       # no host: this machine's own address
    ({"enabled": True, "host": "10.0.0.2", "port": 80}, False, "master.yaml status_share.port is not a port number (1024 to 65535)"),
    ({"enabled": True, "host": "10.0.0.2", "port": 8765}, True, ""),
])
def test_the_settings_are_master_yamls_status_share_and_optional(tmp_path, monkeypatch, block, enabled, why):
    import yaml

    monkeypatch.setattr(ST, "auto_host", lambda: "10.9.8.7")
    from services.visualization.vscode_monitor.layout import MASTER

    (tmp_path / MASTER).parent.mkdir(parents=True)
    (tmp_path / MASTER).write_text(yaml.safe_dump({"default_project": "P", **({"status_share": block} if block else {})}))
    out = ST.settings(tmp_path)
    assert (out["enabled"], out["why"]) == (enabled, why)
    if enabled:
        given = bool(block.get("host"))
        assert (out["host"], out["port"], out["auto"]) == ("10.0.0.2" if given else "10.9.8.7", 8765, not given)


def test_without_a_host_the_address_is_this_machines_own(monkeypatch):
    """The default route's address; with no route, the first reachable address; with none, the loopback."""
    import ipaddress
    import socket

    ipaddress.IPv4Address(ST.auto_host())                               # an address, found without sending anything

    class NoRoute:
        def __init__(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def connect(self, *a): raise OSError("Network is unreachable")

    monkeypatch.setattr(socket, "socket", NoRoute)
    monkeypatch.setattr(ST, "addresses", lambda: ["192.168.1.20", "10.0.0.5"])
    assert ST.auto_host() == "192.168.1.20"
    monkeypatch.setattr(ST, "addresses", lambda: [])
    assert ST.auto_host() == "127.0.0.1"


def test_it_also_listens_on_the_loopback_for_vs_codes_forwarding(shares, monkeypatch):
    import socket
    import time

    monkeypatch.setattr(ST, "status", lambda root=ST.ROOT: {"now_ms": 1, "projects": [], "gpus": [], "commands": []})
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    threading.Thread(target=ST.serve, args=("127.0.0.2", port), daemon=True).start()     # another loopback address
    token = ST.share_add("me")["token"]
    for _ in range(50):
        try:
            assert _get(f"http://127.0.0.1:{port}/status.json?t={token}")[0] == 200
            assert _get(f"http://127.0.0.2:{port}/status.json?t={token}")[0] == 200
            break
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.1)
    else:
        pytest.fail("the page did not answer on both addresses")


def test_a_taken_loopback_port_leaves_the_address_serving(shares, monkeypatch, capsys):
    import socket
    import time

    monkeypatch.setattr(ST, "status", lambda root=ST.ROOT: {"now_ms": 1, "projects": [], "gpus": [], "commands": []})
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))                # e.g. an ssh -L tunnel holds the loopback at that port
    holder.listen()
    port = holder.getsockname()[1]
    threading.Thread(target=ST.serve, args=("127.0.0.3", port), daemon=True).start()
    token = ST.share_add("me")["token"]
    try:
        for _ in range(50):
            try:
                assert _get(f"http://127.0.0.3:{port}/status.json?t={token}")[0] == 200
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.1)
        else:
            pytest.fail("the address did not serve")
        assert f"not on 127.0.0.1:{port}" in capsys.readouterr().out
    finally:
        holder.close()


def test_the_sweep_for_the_batch_eta_is_read_while_a_run_trains_and_kept_until_params_yaml_changes(tmp_path, monkeypatch):
    import os
    import services.visualization.vscode_monitor.layout as L

    proj = tmp_path / "projects" / "P"
    (proj / "artifacts" / "dev").mkdir(parents=True)
    (proj / "config").mkdir()
    (proj / "config" / "params.yaml").write_text("seed: 42\n")
    calls = []
    monkeypatch.setattr(L, "trials", lambda name: calls.append(name) or {"trials": [{"position": 1, "epochs": 5, "passes": 1}]})
    ST._trials.clear()
    beat = proj / "artifacts" / "dev" / "heartbeat.json"
    beat.write_text(json.dumps({"status": "done", "written_ns": 1}))
    assert ST.project_status("P", "P", tmp_path)["trials"] == [] and calls == [], "idle: not read"
    beat.write_text(json.dumps({"status": "running", "written_ns": 1}))
    assert ST.project_status("P", "P", tmp_path)["trials"] == [{"position": 1, "epochs": 5, "passes": 1}]
    ST.project_status("P", "P", tmp_path)
    assert calls == ["P"], "kept while params.yaml is the same"
    os.utime(proj / "config" / "params.yaml", (1, 1))
    ST.project_status("P", "P", tmp_path)
    assert calls == ["P", "P"], "read again after params.yaml changed"
