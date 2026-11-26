# services/visualization/server
`status.py`: the read-only HTTP server of the status page (`../mobile/`), standard library only. The other
files (`app.py`, `routes/`, `security/`) are empty placeholders.

```bash
make status_share ENV=dev                         # the panel's share button → Start (tmux tf-status); HOST= / PORT= override
python -m services.visualization.server.status settings      # master.yaml status_share: enabled, host, port, why
python -m services.visualization.server.status share add NAME | list | remove NAME
python -m services.visualization.server.status qr <link>                    # SVG (segno)
python -m services.visualization.server.status addresses                    # IPv4 addresses a phone could reach
```

- Also listens on this machine's loopback at the same port, for VS Code's forwarding to the owner's desktop (the
  panel's link); skipped with a note when something holds it (an `ssh -L` tunnel to the page leads there anyway).
- Settings: master.yaml `status_share: {enabled, port, host}` (optional; absent or `enabled: false`: off): a TCP
  port the network lets through to this machine; `host` optional, else this machine's own address (`auto_host`:
  the address its default route leaves from, the one its network or VPN reaches).
- Listens on one address only; GET only (405 otherwise).
- `/`, `/status.json`, `/manifest.webmanifest` need a live link's token (`?t=`), checked on every request
  against `secrets/status_shares.json` (mode 600, not in git): a withdrawn link stops at once.
- `/static/` serves only the page's files, the panel's `state.js` and the icon.
- Headers: `Cache-Control: no-store`, `Referrer-Policy: no-referrer`, a `self`-only CSP; the request log has
  paths, never tokens.
- What `/status.json` holds: [`../mobile/api_contract.md`](../mobile/api_contract.md).

Tests: `services/side_channel_core/tests/unit/test_status_share.py`.
