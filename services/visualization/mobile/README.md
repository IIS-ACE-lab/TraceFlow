# services/visualization/mobile
The status page: a read-only view of every project's runs for a phone's browser (saved to the home screen, it
opens like an app) and for the people the owner shares a link with. Served by
[`../server/status.py`](../server/status.py); started and shared from the TraceFlow panel's share button.

## Contents
```text
mobile/
├── index.html          # the page (home-screen meta tags; icon: vscode_monitor/brand/traceflow-web-*.png, the Activity Bar mark)
├── status.js           # view(data, S, nowMs): one card per project; fetches /status.json every 10 s
├── status.css          # light or dark as the phone is
├── shim.js             # gives the panel's state.js (CommonJS) a module object in the browser
├── manifest.js         # links the home-screen manifest with the link's token (Android: Install app)
├── api_contract.md     # what /status.json holds
├── polling_strategy.md # when the page reads it
└── offline_cache.md    # what the page shows when the server cannot be reached
```

The page draws with the panel's own `vscode_monitor/state.js` (light, runSummary, eta, tiles, activeQueues),
so a phone and the panel agree. Tests: `vscode_monitor/test/mobile.test.js`.

## Use
On the phone, with the VPN on: scan the person's QR code (panel → share button) or open their link; then
iPhone: Safari or Chrome → Share → Add to Home Screen (Chrome: iOS 16.4 or later); Android: Chrome → ⋮ → Add to Home screen.
Any browser opens the page; the camera opens the QR code's link in the default one.
