# /status.json
`GET /status.json?t=<token>` (`../server/status.py`): 403 without a live link's token.

| Key | Holds |
|---|---|
| `now_ms` | the server's clock: the page ages heartbeats on it, not on the phone's |
| `timezone` | master.yaml `timezone` (`layout._timezone`): the ETA table's clock |
| `viewer` | the name of the link used |
| `gpus` | `nvidia-smi`: index, name, util %, memUsed / memTotal (MiB), temp °C |
| `commands` | `QUEUE=<key>` of each experiment command running now (`state.activeQueues`) |
| `projects[]` | per project of master.yaml: `name`; `heartbeats` {env: heartbeat without `pid`}; `registry` {hash: the fields `state.registryEntries` reads, no `path`}; `queues` (experiment_queue.yaml); `trials`: the planned sweep (`layout.trials`, for `state.eta`; only while a run is running, cached until params.yaml changes, else `[]`); `live`: the running run's latest metrics (`core/monitor.snapshot`) |

No configuration, metric file, key or path is sent.
