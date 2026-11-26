# 11 · The status page: runs on a phone, links to share

A read-only page showing what the TraceFlow panel's project headings show: each
project's light, the running run, its phase, progress and the panel's ETA
table (when the run and the batch end), the queued / running / done / failed
counts, the running run's latest metrics and the GPU. It opens in any browser; saved to a phone's home screen it opens like
an app. Each person you share it with has a link (and QR code) of their own.
Nothing can be started or stopped from it.

Server: `services/visualization/server/status.py`. Page:
`services/visualization/mobile/`. Panel: the share button on the *TraceFlow Monitor*
title line (top right, glowing). The page follows the device's light or dark theme.

## 11.0 Who can open the links

`<port>` below is your master.yaml `status_share.port` (8765 in the example of 11.0a); the panel's hints show
the port you set.

**In short:** a device opens the links only on the same network as this
machine, or on its VPN with the port allowed. Nothing else reaches it (no
tunnels).

A link is `http://<address>:<port>/?t=<token>` (the QR code is the same link).
It opens on any device that reaches this machine's address and port; no VPN
is needed unless the machine is reachable only through one. The address is
found by itself (11.0a): the one this machine's default route leaves from,
i.e. its address on its own network or VPN.

| This machine is | Who opens the links | To do once |
|---|---|---|
| your own computer, or a lab or home machine on Wi-Fi / LAN | devices on the same network (a phone on the same Wi-Fi) | let the port in through the firewall: Ubuntu `sudo ufw allow <port>/tcp` (nothing to do when no firewall runs) |
| a server reached through a VPN | devices on the VPN | the VPN must let the port through: a VPN that passes only SSH (port 22) drops it (ours did, checked 2026-10-21), so ask IT, below |
| a server with a public address | anyone on the internet with a link | open the port in its firewall. The page is plain HTTP: a token can be read on the way, so prefer a VPN to the open internet |
| a server reached only by SSH (most cloud and HPC machines) | you alone, through SSH port forwarding (11.4) | nothing |

The request to IT, with this machine's address and the port:

> Please allow TCP port <port> from the VPN to &lt;address&gt; (the TraceFlow
> experiment server). It serves a read-only status page over HTTP; every
> request needs a personal link token; nothing can be changed through it.

## 11.0a The settings: master.yaml `status_share` (optional)

```yaml
status_share:
  enabled: true            # false, or no block at all: the page is off
  port: 8765               # a TCP port the network lets through to this machine
  # host: "10.0.0.5"       # optional: the address to use; without it, this machine's own (found by itself)
```

- **The address** is found by itself: the address this machine's default route
  leaves from (with no route, its first network address; with none, `127.0.0.1`,
  this machine only). Set `host` only to choose another one (a machine on
  several networks). The links and QR codes follow it; when the machine gets a
  new address (e.g. a new DHCP lease), **Stop**, **Start** and send the links
  again.
- The share button shows the port **read-only**, never the address (it is in
  each person's link only); Master Control shows the block read-only. They
  change only here, by hand.
- **Optional:** a server that may not open a port leaves the block out or sets
  `enabled: false`; the share button then says the page is off and why.
- Another port: change `port`, then share button → **Stop**, **Start**; the
  people's links and QR codes follow.

## 11.1 Start it and make a link

1. The **share button** on the *TraceFlow Monitor* title line (top right). It shows the port
   and whether the address is found by itself or master.yaml `status_share.host` (11.0a).
2. **Add a person**: a name (yourself, `Professor`, …) → Enter. Each person gets
   a panel with their QR code and **Click to share or open in external browser** (the link behind the words).
3. **Start**. It serves until **Stop**, also with VS Code closed (tmux session
   `tf-status`, log `logs/status.txt`).

From a terminal:

```bash
make status_share ENV=dev            # on this machine's address (or status_share.host) and port, until Ctrl-C (HOST= / PORT= override)
python -m services.visualization.server.status share add Professor
python -m services.visualization.server.status share list       # the tokens
# the link: http://<address>:<port>/?t=<token>; the address: python -m services.visualization.server.status settings
```

**Open it on your desktop:** click **Click to share or open in external browser** in their panel. VS Code
opens it in your desktop's browser through its own port forwarding (the page
also listens on this machine's loopback), so it needs no open port. **⧉** copies
the link, to send it.

## 11.2 Open it on a phone

On this machine's network (the same Wi-Fi, or with its VPN on), scan the
person's QR code with the camera (or open the link), then save it:

| Phone | Save to the home screen |
|---|---|
| iPhone | Safari or Chrome → **Share** → **Add to Home Screen** (Chrome: iOS 16.4 or later; else Safari) |
| Android | Chrome → ⋮ → **Add to Home screen** |

The page reads the status every 10 s while it is open, and at once when it
comes back on screen.

## 11.3 Withdraw a link

The person's panel → **×** → Withdraw. Their link and QR code stop at once; the
other links keep working. From a terminal:
`python -m services.visualization.server.status share remove Professor`.

Links live in `secrets/status_shares.json` (owner-only, not in git).

## 11.4 The link does not open

| Symptom | Cause | Fix |
|---|---|---|
| The browser waits, then fails; `logs/status.txt` shows no request from that device | the port does not get through: this machine's firewall, or a VPN that passes only SSH (port 22) | **The fix:** the firewall: `sudo ufw allow <port>/tcp`; the VPN: ask IT (11.0). **Until then, you:** click the link in the panel (11.1), or forward the port over your SSH session: VS Code → **Ports** → **Forward a Port** → `<address>:<port>`, then open the link with `localhost:<port>` in place of `<address>:<port>`; or `ssh -L <port>:<address>:<port> <user>@<address>`; on a phone, an SSH app's local port forward (e.g. Termius: local port <port> → `<address>:<port>`, then `localhost:<port>` in the browser; iOS suspends the app in the background, so look for its keep-alive option) |
| Same, and the device is on another network | the address is reachable only from this machine's network (or its VPN) | join that network: the same Wi-Fi, or the VPN on |
| It worked, and stopped after a restart of the machine or the network | the machine has a new address; the links carry the old one | share button → **Stop**, **Start**, and send the links again (they show the new address) |
| Start fails: `Cannot assign requested address` in `logs/status.txt` | master.yaml `status_share.host` is not an address of this machine | remove `host` (found by itself) or set this machine's address |
| `This page opens with a personal link: …/?t=<token>` | the address was opened without its token (e.g. `http://localhost:<port>/`) | open the whole link: the panel's link, or the copied one |
| `This link is not valid (withdrawn, or mistyped)` | the token was withdrawn or is wrong | a new link: share button → Add a person |
| Nothing answers at all, even on this machine | the page is not serving | share button → **Start**; if it stops again, read `logs/status.txt` (`tmux attach -t tf-status`) |
| The share button says *Off* | master.yaml has no `status_share` block, or `enabled: false`, or no valid `port` | fill it in master.yaml (11.0a), if the server may open the port |
| `offline: … (same network or VPN?)` at the bottom of the page | the last read failed; the page shows the last answer | join this machine's network (or VPN); check that it still serves |
| The QR code does not scan | the panel was not reloaded after an update | *Developer: Reload Window*, then scan again |
| The home-screen icon is the old one | the phone keeps the icon it saved | remove the icon, add the page again |

Check that it serves, on this machine:

```bash
curl -s -o /dev/null -w "%{http_code}\n" "http://<address>:<port>/?t=<token>"   # 200: it serves; 403: the token
```

## 11.5 What a link can see

Each project's heartbeat (without its process id), the run registry's status
fields (no file paths), the queues, the queue keys running now, the running
run's latest metrics and the GPU. No configuration, metric file, key or path.
Every read checks the link's token; the server's log records paths, never
tokens. Details: [`mobile/api_contract.md`](../visualization/mobile/api_contract.md).
