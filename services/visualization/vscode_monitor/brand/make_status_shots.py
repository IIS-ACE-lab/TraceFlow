# services/visualization/vscode_monitor/brand/make_status_shots.py
"""Photograph the status page as a phone shows it, for README §1.5: the real page (services/visualization/mobile:
index.html, status.css, status.js, the panel's state.js), served as server/status.py serves it, in headless Chromium,
inside a phone frame on a transparent ground. Writes, next to this file:

    status_phone_now_light.png      the page on this machine's data now (server/status.status()), light theme
    status_phone_example_dark.png   the same page while a run's test attacks, the run of the panel picture: the
                                    orange attack bar, progress, the ETA table, the latest metrics and the
                                    key-recovery line (sample numbers), dark theme
    status_phone_home.png           the home screen after Share → Add to Home Screen: the page's icon

Needs Playwright and its Chromium (not in requirements.txt: a picture, not a run):

    pip install playwright && python -m playwright install chromium-headless-shell
    python services/visualization/vscode_monitor/brand/make_status_shots.py
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

BRAND = Path(__file__).resolve().parent
ROOT = BRAND.parents[3]
for p in (ROOT, ROOT / "services" / "side_channel_core", ROOT / "services" / "side_channel_core" / "src"):
    sys.path.insert(0, str(p))

from services.visualization.server import status as server            # noqa: E402

BASE = "http://traceflow.status"
W, H, BAR = 390, 800, 44            # the screen (CSS px), the status bar above the page
SCALE = 2                           # a 2x phone

FRAME = """<!doctype html><meta charset="utf-8"><style>
  html,body{margin:0;background:transparent}
  .phone{display:inline-block;padding:12px;border-radius:56px;background:#15181d;position:relative;margin:24px;
         box-shadow:0 0 0 1.5px #3a414d}
  .island{position:absolute;top:24px;left:50%%;width:112px;height:30px;margin-left:-56px;border-radius:16px;
          background:#15181d;z-index:2}
  .screen{width:%(w)spx;height:%(h)spx;border-radius:44px;overflow:hidden;background:%(bg)s;
          font:600 14px -apple-system,"SF Pro Text","Segoe UI",Roboto,sans-serif;color:%(fg)s}
  .bar{height:%(bar)spx;display:flex;justify-content:space-between;align-items:center;padding:0 30px 0 34px}
  iframe{display:block;border:0;width:%(w)spx;height:%(page)spx}
  %(extra)s
</style><div class="phone"><div class="island"></div><div class="screen">
  <div class="bar"><span>%(clock)s</span><span>VPN · 5G</span></div>%(body)s</div></div>"""

HOME = """.home{height:%(page)spx;padding:18px 26px;display:grid;grid-template-columns:repeat(4,64px);
          justify-content:space-between;align-content:start;row-gap:24px}
  .app{display:grid;justify-items:center;gap:6px;font:500 12px -apple-system,"SF Pro Text",Roboto,sans-serif;color:#fff}
  .app img{width:64px;height:64px;border-radius:15px}"""


WALLPAPER = "linear-gradient(160deg,#3a5a8c,#1d2b44 60%,#151c2b)"   # the home screen, status bar included


def clock(data: dict) -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.fromtimestamp(data["now_ms"] / 1000, ZoneInfo(data["timezone"])).strftime("%H:%M")


def example(now: dict) -> dict:
    """`now` with ZvW's dev run under test, as the panel picture draws it (make_panel_diagram.py): bugger_fantastic
    after its 50 epochs, 98 %, attack 33 of 100 running (the orange bar), its latest metrics and the key-recovery
    line; sample numbers."""
    data = copy.deepcopy(now)
    proj = next((p for p in data["projects"] if p["name"] == "ZvW"), data["projects"][0])
    data["projects"] = [proj] + [p for p in data["projects"] if p is not proj]
    ns = data["now_ms"] * 1e6
    proj["heartbeats"]["dev"] = {
        "status": "running", "written_ns": ns - 3e9, "started_ns": ns - 20 * 60e9, "progress": 0.98,
        "model_name": "bugger_fantastic", "model_id": "Zaid_ASCAD_N100", "dataset": "ascad_n100_none", "phase": "test",
        "epoch": 50, "refit_epochs": 50, "model_hash": "example1", "batches": ["z_fig12_ascad_n100"],
        "trial": 1, "trials": 1}
    proj["registry"]["example1"] = {
        "model_name": "bugger_fantastic", "model_id": "Zaid_ASCAD_N100", "env": "dev", "status": "IN_PROGRESS",
        "batches": ["z_fig12_ascad_n100"], "batch_ids": [], "started_at": "", "run_session_id": "s", "trial_index": 1,
        "total_trials": 1}
    value = lambda process, label, v, improved: {"process": process, "label": label, "value": v, "improved": improved}
    train = lambda name, v, improved: {"name": name, "values": [value("train", "train", v, improved)]}
    validation = lambda name, v, improved: {"name": name, "values": [value("validation", "validation", v, improved)]}
    proj["live"] = {"hash": "example1", "source": None, "epoch": 50,
                    "attack": {"attack": 33, "attacks": 100, "running": True},
                    "key_recovery": {"source": "validation, epoch 50", "recovered": True, "true_key_rank": 1,
                                     "top_guess": "0xE0", "ge": 1, "sr1": 1.0},
                    "metrics": [
        train("trainable_parameters", 141596, None),
        train("η_t", 0.00001, False),
        {"name": "mean_accuracy", "values": [value("train", "train", 0.01887, False),
                                             value("validation", "validation", 0.006, False)]},
        {"name": "CrossEntropyLoss", "values": [value("train", "train", 5.206, True),
                                                value("validation", "validation", 5.362, True)]},
        validation("mean_attack_rank_curve", 1, None),
        validation("per_attack_curves", 1, None),
        validation("final_ranks", 1, None),
        validation("traces_to_ge_1", 548, None),
        validation("per_attack_traces_to_ge_1", 167.79, True)]}
    gpu = (data["gpus"] or [{"index": 0, "name": "NVIDIA GPU", "memTotal": 32607}])[0]
    data["gpus"] = [{**gpu, "util": 0, "memUsed": 750, "temp": 47}]
    return data


def shoot(browser, scheme: str, data: dict, out: Path, home: bool = False) -> None:
    dark = scheme == "dark"
    body = ('<div class="home"><div class="app"><img src="/static/icon-180.png" alt="">TraceFlow</div></div>' if home
            else "<iframe src=\"/?t=x\"></iframe>")
    frame = FRAME % {"w": W, "h": H, "bar": BAR, "page": H - BAR, "clock": clock(data),
                     "bg": WALLPAPER if home else "#0d1117" if dark else "#f6f7f9",   # status.css's page colours
                     "fg": "#ffffff" if home else "#e6edf3" if dark else "#1f2328",
                     "extra": HOME % {"page": H - BAR} if home else "", "body": body}
    page = browser.new_page(device_scale_factor=SCALE, color_scheme=scheme, timezone_id=data["timezone"],
                            viewport={"width": W + 120, "height": H + 120})   # the phone's clock: master.yaml timezone

    def serve(route):
        path = urlparse(route.request.url).path
        if path == "/frame":
            return route.fulfill(body=frame, content_type="text/html")
        if path == "/":
            return route.fulfill(path=server.PAGE / "index.html")
        if path == "/status.json":
            return route.fulfill(body=json.dumps({**data, "viewer": "you"}), content_type="application/json")
        file = server.STATIC.get(path[len("/static/"):]) if path.startswith("/static/") else None
        return route.fulfill(path=file) if file else route.fulfill(status=404, body="")

    page.route("**/*", serve)
    page.goto(f"{BASE}/frame")
    if not home:
        page.frame_locator("iframe").locator(".card").first.wait_for()
    page.wait_for_timeout(300)
    page.locator(".phone").screenshot(path=str(out), omit_background=True)
    page.close()
    print(out)


def main() -> None:
    from playwright.sync_api import sync_playwright

    now = server.status()
    now["now_ms"] = time.time() * 1000
    with sync_playwright() as p:
        browser = p.chromium.launch()
        shoot(browser, "light", now, BRAND / "status_phone_now_light.png")
        shoot(browser, "dark", example(now), BRAND / "status_phone_example_dark.png")
        shoot(browser, "dark", now, BRAND / "status_phone_home.png", home=True)
        browser.close()


if __name__ == "__main__":
    main()
