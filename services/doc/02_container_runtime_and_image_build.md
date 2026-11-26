# 02 · Container runtime and image build

`make orchestrate` and `make rerun` build the training image with
`nerdctl` + BuildKit and start it with `nerdctl compose` on **rootful**
containerd.

Install rootful containerd, nerdctl (compose v2, BuildKit) and the NVIDIA
Container Toolkit with the installation guide in
[`services/side_channel_core/deployment/docker/README.md`](../side_channel_core/deployment/docker/README.md),
then check:

```bash
sudo systemctl is-active containerd     # must print: active
sudo nerdctl --version
```

## 2.1 `containerd service is not active (rootful)`

Printed by `make preflight-rootful`, which `orchestrate` and `rerun` run first.

```bash
sudo systemctl enable --now containerd
sudo systemctl is-active containerd      # active
```

If it stays `failed`, read `sudo journalctl -u containerd -n 50`.

## 2.2 `nerdctl: command not found` / `compose` is unknown

nerdctl, or its compose plugin, is not installed on the host. Follow the
deployment README above and check:

```bash
sudo nerdctl --version
sudo nerdctl compose version
```

## 2.3 Image build fails: `buildkitd` not running

```bash
sudo systemctl enable --now buildkit     # service name may be buildkitd
sudo nerdctl build --help >/dev/null && echo ok
```

## 2.4 DNS resolution errors / `FATA[0000] name-store error`

The build cannot resolve the registry. Restart the resolver and test:

```bash
sudo resolvectl flush-caches
sudo systemctl restart systemd-resolved
sudo resolvectl statistics | sed -n '1,80p'
sudo getent hosts registry-1.docker.io   # must print an address
```

## 2.5 The container does not see the GPU

```bash
nvidia-smi                                            # host sees it
sudo nerdctl run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If the second command fails, the NVIDIA Container Toolkit is missing or not
configured for containerd:

```bash
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd
```

## 2.6 Container name conflicts / zombie containers

Error such as `The container name "/sidechannel" is already in use`.
First try the Makefile:

```bash
make stack-down ENV=dev          # stop, keep volumes
make docker-clean ENV=dev        # stop, delete volumes and the orchestrator lock
```

If a task survives, remove it with `ctr`:

```bash
sudo nerdctl --namespace=default ps -a | grep sidechannel
SHORT=<first characters of the container id>
FULL="$(sudo ctr -n default containers list | awk -v p="$SHORT" '$1 ~ "^"p {print $1; exit}')"

sudo ctr -n default tasks info "$FULL" || true      # may hang; skip if it does
ps aux | grep -E "containerd-shim|runc" | grep "$FULL" || true
sudo kill -9 <PID>                                   # the shim PID from above

sudo ctr -n default tasks rm -f "$FULL" 2>/dev/null || true
sudo ctr -n default containers delete "$FULL" 2>/dev/null || true
sudo nerdctl --namespace=default ps -a | grep sidechannel || echo "no sidechannel containers"
```

## 2.7 `Orchestrator failed; volumes preserved for debugging.`

The run failed and the stack's volumes were kept so you can inspect them.
Read the log above that line for the real error (often a config error, see
[04](04_running_experiments.md)). Then clean up:

```bash
make stack-nuke ENV=dev
```

## 2.8 Disk full during build

```bash
sudo nerdctl system df
sudo nerdctl image prune -af
sudo nerdctl builder prune -af
```

`ENV=prod` runs both prunes automatically before each build.

## 2.9 A code change is not picked up

The image is reused when it already exists. Rebuild it:

```bash
make orchestrate ENV=dev PROJECT=<P> FORCE=1
```

`FORCE=1` also re-runs trials the registry marks DONE (see
[04](04_running_experiments.md#44-nothing-runs-all-trials-are-done)).
To rebuild without running: `make build-image ENV=dev FORCE=1`.

## 2.10 What goes into the image

The build context is the repository root, filtered by `.dockerignore`:
code only. Credentials, datasets (`services/side_channel_core/data/`),
`projects/`, Harbor, `paper/` and `.git/` are excluded; every compose
file bind-mounts `side_channel_core`, `mlops`, `projects` and `secrets`
at run time. If a build step needs a file that `.dockerignore` excludes,
add a `!path` exception there rather than removing a whole rule.

Old images pile up (one per commit and ENV, ~11 GB each). Keep the newest
and remove the rest:

```bash
N="sudo nerdctl --namespace default"
$N images | grep '^sidechannel'
$N rmi sidechannel:<old tag>          # one per old tag
$N image prune -f
sudo nerdctl builder prune -f         # build cache
```
