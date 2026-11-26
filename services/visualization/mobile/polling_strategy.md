# Polling
The page reads `/status.json` once on load, every 10 s while open, and at once when it comes back on screen.
The server reads its files at most once every 2 s, however many phones ask. There is no push, WebSocket or
background refresh: a closed page reads nothing.
