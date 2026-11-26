# 03 · File permissions

Training runs in a rootful container, so files it writes under
`projects/<P>/` can end up owned by `root` or by the container user. The
symptom is `Permission denied` when you delete or edit artifacts, or when
`make judge` / `make graph` write their output.

Run every command below from the repository root:

```bash
cd "$(git rev-parse --show-toplevel)"
```

## 3.1 Cannot delete or write project artifacts

Take ownership of the project back:

```bash
P=DLSCA_ASCAD
sudo chown -R "$(id -u):$(id -g)" "projects/$P"
sudo chmod -R u+rwX "projects/$P"
```

The whole repository, if several directories are affected:

```bash
sudo chown -R "$(id -u):$(id -g)" .
sudo chmod -R u+rwX .
```

## 3.2 Stop pip from trying `--user` inside conda

When pip finds `site-packages` not writable it installs into `~/.local`
instead of the environment. Make the environment writable once:

```bash
conda activate sca
sudo chown -R "$(id -u):$(id -g)" "$(python -c 'import site; print(site.getsitepackages()[0])')"
```

## 3.3 Bind-mount permission or missing-path errors (nerdctl/containerd)

containerd refuses to start a container whose bind-mount source does not
exist, and the container user cannot write into a root-owned directory.
Prepare the project directory:

```bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

UID_NOW="$(id -u)"; GID_NOW="$(id -g)"
PROJ_ABS="$(pwd)/projects/DLSCA_ASCAD"          # your project

# Create the directories the container mounts
sudo install -d -o "$UID_NOW" -g "$GID_NOW" -m 2775 \
  "$PROJ_ABS" "$PROJ_ABS/config" "$PROJ_ABS/artifacts"

# Ownership, then permissions (setgid keeps the group on new files)
sudo chown -R "$UID_NOW:$GID_NOW" "$PROJ_ABS"
sudo find "$PROJ_ABS" -type d -exec chmod 2775 {} \;
sudo find "$PROJ_ABS" -type f -exec chmod 0664 {} \;

test -w "$PROJ_ABS/artifacts" && echo "artifacts writable" || echo "artifacts NOT writable"
```

The image is built with your `USER_ID`/`GROUP_ID` (`make build-image`
passes them), so files written by a freshly built image are owned by you.
An image built by another user keeps that user's IDs — rebuild with
`FORCE=1`.

## 3.4 `Permission denied` on a `.sh` script

```bash
bash services/side_channel_core/scripts/setup.sh    # makes all repo scripts executable
```

## 3.5 Stale lock files

| File | Written by | Safe to delete when |
|---|---|---|
| `projects/<P>/config/.graph_config.yaml.lock` | `make judge` updating job 0 | no `make judge` is running |
| `services/side_channel_core/deployment/docker/tmp/orchestrator.pid` | the orchestrator | no stack is running (`make docker-clean` removes it) |
| `services/side_channel_core/deployment/docker/tmp/cleanup.lock` | trial cleanup | no stack is running |

Lock files are not errors by themselves; they are left over after a crash.
