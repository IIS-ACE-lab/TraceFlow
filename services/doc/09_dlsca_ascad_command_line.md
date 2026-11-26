# 09 · DLSCA_ASCAD from the command line

The commands behind the TraceFlow panel for the ASCAD reproduction
([projects/DLSCA_ASCAD/README.md](../../projects/DLSCA_ASCAD/README.md), which
does each step from the panel). Use them over a plain SSH terminal, without VS
Code, or when a step has no panel button (`make batch_label`, `make graph
FACE_ID=`, `make rerun`, `make rerun-judged`). Run them from the repository root
with the `sca` environment active. The generic reference of every target and
variable is [08_command_line.md](08_command_line.md).

## Contents

1. [Quick start, clean slate and detached runs](#1-quick-start-clean-slate-and-detached-runs)
2. [The figures](#2-the-figures): the commands of every figure's queue
3. [The experiments](#3-the-experiments): queues, reading the results, Experiments 1–27
4. [Ship and reproduce](#4-ship-and-reproduce)

## 1. Quick start, clean slate and detached runs

### 1.1 Clean slate (optional, before the campaign)

Ensure the directory ownership and permissions are set correctly to avoid any permission issues during execution.

```bash
sudo chown -R "$(id -u):$(id -g)" projects/
```


```bash
# clearning registries, run map, run dirs, judge/filter outputs
cd TraceFlow   # the repository root
make delete_database ENV=dev PROJECT=DLSCA_ASCAD CONFIRM=1
rm -rf  projects/DLSCA_ASCAD/config/.graph_config.yaml.lock
```


```bash
# cleearning the GPU, Memory, and disk space
make ENV=dev docker-clean                                            # compose down -v + lock file
sudo nerdctl --namespace default rm -f sidechannel 2>/dev/null
sudo nerdctl --namespace default rmi -f $(sudo nerdctl --namespace default images --format '{{.Repository}}:{{.Tag}}' | grep '^sidechannel:')
sudo nerdctl --namespace default builder prune -af
sudo nerdctl --namespace default image prune -f
sudo nerdctl --namespace default volume prune -f
sudo rm -rf services/side_channel_core/deployment/docker/tmp/* /tmp/orchestrator.pid

```
Run the experiments as per README.md to reproduce the paper results. The following commands will run the experiments and generate the graphs.


```bash
# 1. shortlist config/params.yaml to one training (see "Before you run")
# 2. the headline result: CNNbest, synchronised traces, 10-fold CV + final training + test
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbest
# 3. rank the trained models, then plot
make judge ENV=dev PROJECT=DLSCA_ASCAD
make graph ENV=dev PROJECT=DLSCA_ASCAD GRAPH_ID=1
```
The first `make orchestrate` afterwards rebuilds the image (~10 min).


### 1.2 Detached runs (tmux)

 A queue runs for hours; started in a plain SSH
terminal it dies with the connection. Run it inside a tmux session on the
Linux host instead, so it keeps running after you detach or disconnect:

```bash
cd TraceFlow                       # the repository root
tmux new -s fig26                             # a new session named after the queue
sudo chown -R "$(id -u):$(id -g)" projects   # the container writes artifacts/ as root
mkdir -p projects/DLSCA_ASCAD/artifacts/dev/tmux_logs && make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig26 2>&1 | tee >(sed -u -e 's/\x1b\[[0-9;?]*[A-Za-z]//g' -e 's/.*\r//' >> projects/DLSCA_ASCAD/artifacts/dev/tmux_logs/fig26.txt)
# detach: Ctrl-b, then d      (the queue keeps running)
```

```bash
tmux ls                                       # running sessions
tmux attach -t fig26                          # back to the run's terminal
tmux attach -d -t fig26                       # the same, detaching other terminals (the window takes your size)
```

Inside the session, Ctrl-b [ scrolls back (q leaves scroll mode). The `tee >(sed …)`
also keeps the output, as plain text without colour codes, in `projects/DLSCA_ASCAD/artifacts/dev/tmux_logs/fig26.txt` (beside the runs, git-ignored),
readable without attaching (`tail -f projects/DLSCA_ASCAD/artifacts/dev/tmux_logs/fig26.txt`). Queues run before
2026-10-11 logged to `logs/<queue>.txt` at the repository root.

*What runs where.* The `make` chain in the tmux pane only starts the
container stack and streams its log (`sudo nerdctl compose up`); the
training runs inside the container, which works through every trial of the
queue's command by itself. If that stream ends, the pane drops back to the
prompt but the container keeps training. Follow it again from the pane:

```bash
sudo nerdctl logs -f --tail 50 sidechannel                       # live, from now on
sudo nerdctl logs -f --since 2026-10-11T09:16:46Z sidechannel 2>&1 | tee >(sed -u -e 's/\x1b\[[0-9;?]*[A-Za-z]//g' -e 's/.*\r//' >> projects/DLSCA_ASCAD/artifacts/dev/tmux_logs/fig26.txt)   # refill the gap, then live
```

Ctrl-C there stops only the viewer. To stop the run itself use
`make stack-down ENV=dev`, not `tmux kill-session` (which only ends the
host side; what the container then does is not defined by TraceFlow).
When the host side has ended, `make orchestrate`'s closing steps do not run;
after the last trial run them by hand: `make healthcheck ENV=dev && make reset ENV=dev`.
Each run's complete log is in
`artifacts/dev/generated_models/<model_name>/logs/` (`all_logs.log`,
`Training.log`), written by the container whatever happens to the stream.

*The TraceFlow pane* (the TraceFlow icon in VS Code; see
[`services/visualization/vscode_monitor/README.md`](../../services/visualization/vscode_monitor/README.md))
follows the run without any terminal: it reads the heartbeat the trainer
writes every 5 s and the run registry, so the light, phase, fold, epoch and
progress stay live with every terminal closed. Its queue **Start** does all
of the above for you: tmux session `tf-DLSCA_ASCAD`, the `sca` env,
`artifacts/<ENV>/tmux_logs/<queue>.txt`, and it refuses while another run is going.
Next to the running run's name, **Attach** opens the tmux session running it
(or, when none holds it any more, the container's live log) and **Kill
session** ends that session after asking.

*One session per queue:* queues read the same `config/params.yaml` when each command starts, so run them one after another,
not side by side, unless they need the same settings. A session does not survive
a reboot of the host; after one, start the queue again: finished runs are
`DONE` and skipped, an interrupted one is resumed or retrained as the
orchestrator reports.


## 2. The figures

The runs of each figure of the [Figure index](../../projects/DLSCA_ASCAD/README.md#figure-index-model-dataset-leakage-model-batch-size-epochs-optimizer).
The `+` line runs the whole figure as one queue (the panel's *Queues → Start*);
the lines below it are the same runs one by one. `ascad_desync100_raw` (the
desync-100 traces rebuilt from the raw database with a new jitter draw) is
commented out: uncomment it to use it instead of the official `ASCAD_desync100.h5`.

### Fig. 2

```diff
+# all at once, labelled batch fig2: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig2
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_VGG16 BATCH=fig2
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_ResNet50 BATCH=fig2
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_InceptionV3 BATCH=fig2
```

### Fig. 3

```diff
+# all at once, labelled batch fig3: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig3
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase BATCH=fig3
```

### Fig. 4

```diff
+# all at once, labelled batch fig4: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig4
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase BATCH=fig4

# the same on the official ASCAD_desync100.h5 (the paper's own jitter draw)
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase BATCH=fig4

##if you want to process raw traces instead of the official ASCAD_desync100.h5 (the paper's own jitter draw), use the following commands:
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase BATCH=fig4
```

### Fig. 5

```diff
+# all at once, labelled batch fig5: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig5
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_blocks2 BATCH=fig5
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_blocks3 BATCH=fig5
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase BATCH=fig5
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_blocks5 BATCH=fig5

# the same on the official ASCAD_desync100.h5 (the paper's own jitter draw)
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_blocks2 BATCH=fig5
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_blocks3 BATCH=fig5
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase BATCH=fig5
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_blocks5 BATCH=fig5

##if you want to process raw traces instead of the official ASCAD_desync100.h5 (the paper's own jitter draw), use the following commands:
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_blocks2 BATCH=fig5
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_blocks3 BATCH=fig5
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase BATCH=fig5
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_blocks5 BATCH=fig5
```

### Fig. 6

```diff
+# all at once, labelled batch fig6: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig6
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_conv0 BATCH=fig6
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase BATCH=fig6
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_conv2 BATCH=fig6
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_conv3 BATCH=fig6

# the same on the official ASCAD_desync100.h5 (the paper's own jitter draw)
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_conv0 BATCH=fig6
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase BATCH=fig6
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_conv2 BATCH=fig6
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_conv3 BATCH=fig6

##if you want to process raw traces instead of the official ASCAD_desync100.h5 (the paper's own jitter draw), use the following commands:
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_conv0 BATCH=fig6
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase BATCH=fig6
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_conv2 BATCH=fig6
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_conv3 BATCH=fig6
```

### Fig. 7

```diff
+# all at once, labelled batch fig7: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig7
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_filters16 BATCH=fig7
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_filters32 BATCH=fig7
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase BATCH=fig7
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_filters128 BATCH=fig7

# the same on the official ASCAD_desync100.h5 (the paper's own jitter draw)
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_filters16 BATCH=fig7
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_filters32 BATCH=fig7
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase BATCH=fig7
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_filters128 BATCH=fig7

##if you want to process raw traces instead of the official ASCAD_desync100.h5 (the paper's own jitter draw), use the following commands:
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_filters16 BATCH=fig7
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_filters32 BATCH=fig7
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase BATCH=fig7
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_filters128 BATCH=fig7
```

### Fig. 8 left

```diff
+# all at once, labelled batch fig8_left: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig8_left
# the official ASCAD_desync100.h5 (the paper's own jitter draw)
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase BATCH=fig8_left
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_kernel6 BATCH=fig8_left
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_kernel11 BATCH=fig8_left

##if you want to process raw traces instead of the official ASCAD_desync100.h5 (the paper's own jitter draw), use the following commands:
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase BATCH=fig8_left
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_kernel6 BATCH=fig8_left
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_kernel11 BATCH=fig8_left
```

### Fig. 8 right

```diff
+# all at once, labelled batch fig8_right: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig8_right
# the official ASCAD_desync100.h5 (the paper's own jitter draw)
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_conv3 BATCH=fig8_right

##if you want to process raw traces instead of the official ASCAD_desync100.h5 (the paper's own jitter draw), use the following commands:
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_conv3 BATCH=fig8_right
```

### Fig. 9

```diff
+# all at once, labelled batch fig9: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig9
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_dense0 BATCH=fig9
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_dense1 BATCH=fig9
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase BATCH=fig9
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_dense3 BATCH=fig9

# the same on the official ASCAD_desync100.h5 (the paper's own jitter draw)
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_dense0 BATCH=fig9
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_dense1 BATCH=fig9
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase BATCH=fig9
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_dense3 BATCH=fig9

##if you want to process raw traces instead of the official ASCAD_desync100.h5 (the paper's own jitter draw), use the following commands:
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_dense0 BATCH=fig9
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_dense1 BATCH=fig9
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase BATCH=fig9
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_dense3 BATCH=fig9
```

### Fig. 10

```diff
+# all at once, labelled batch fig10: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig10
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase BATCH=fig10
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_sigmoid BATCH=fig10
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_tanh BATCH=fig10

# the same on the official ASCAD_desync100.h5 (the paper's own jitter draw)
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase BATCH=fig10
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_sigmoid BATCH=fig10
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_tanh BATCH=fig10

##if you want to process raw traces instead of the official ASCAD_desync100.h5 (the paper's own jitter draw), use the following commands:
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase BATCH=fig10
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_sigmoid BATCH=fig10
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_tanh BATCH=fig10
```

### Fig. 11

```diff
+# all at once, labelled batch fig11: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig11
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_VGG16 BATCH=fig11

# the same on the official ASCAD_desync100.h5 (the paper's own jitter draw)
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_VGG16 BATCH=fig11

##if you want to process raw traces instead of the official ASCAD_desync100.h5 (the paper's own jitter draw), use the following commands:
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_VGG16 BATCH=fig11
```

### Fig. 13

```diff
+# all at once, labelled batch fig13: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig13
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbest BATCH=fig13

# the same on the official ASCAD_desync100.h5 (the paper's own jitter draw)
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbest BATCH=fig13

##if you want to process raw traces instead of the official ASCAD_desync100.h5 (the paper's own jitter draw), use the following commands:
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbest BATCH=fig13
```

### Fig. 14

```bash
# three passes; set params.yaml before each (config/experiment_queue.yaml says what)
# trainer.epochs: [75],  trainer.batch_size: [200]
make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig14_cnnbest
# trainer.epochs: [150], trainer.batch_size: [200]
make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig14_vgg16
# trainer.epochs: [800], trainer.batch_size: [100]
make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig14_mlpbest
```

### Fig. 31

```diff
+# all at once, labelled batch fig31: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig31
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase BATCH=fig31
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_avgpool BATCH=fig31
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_stridepool BATCH=fig31

# the same on the official ASCAD_desync100.h5 (the paper's own jitter draw)
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase BATCH=fig31
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_avgpool BATCH=fig31
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_stridepool BATCH=fig31

##if you want to process raw traces instead of the official ASCAD_desync100.h5 (the paper's own jitter draw), use the following commands:
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase BATCH=fig31
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_avgpool BATCH=fig31
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_stridepool BATCH=fig31
```

### Fig. 32

```diff
+# all at once, labelled batch fig32: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig32
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase BATCH=fig32
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_validpad BATCH=fig32

# the same on the official ASCAD_desync100.h5 (the paper's own jitter draw)
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase BATCH=fig32
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_validpad BATCH=fig32

##if you want to process raw traces instead of the official ASCAD_desync100.h5 (the paper's own jitter draw), use the following commands:
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase BATCH=fig32
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_CNNbase_validpad BATCH=fig32
```

### Fig. 18

```diff
+# all at once, labelled batch fig18: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig18
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase BATCH=fig18
```

### Fig. 19

```diff
+# all at once, labelled batch fig19: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig19
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase BATCH=fig19
```

### Fig. 20

```diff
+# all at once, labelled batch fig20: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig20
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase BATCH=fig20
```

### Fig. 21

```diff
+# all at once, labelled batch fig21: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig21
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase BATCH=fig21
```

### Fig. 22

```diff
+# all at once, labelled batch fig22: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig22
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_layers3 BATCH=fig22
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_layers4 BATCH=fig22
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_layers5 BATCH=fig22
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase BATCH=fig22
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_layers7 BATCH=fig22
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_layers8 BATCH=fig22
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_layers11 BATCH=fig22
```

### Fig. 23

```diff
+# all at once, labelled batch fig23: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig23
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_units20 BATCH=fig23
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_units50 BATCH=fig23
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_units100 BATCH=fig23
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_units150 BATCH=fig23
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase BATCH=fig23
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_units250 BATCH=fig23
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_units300 BATCH=fig23
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_units500 BATCH=fig23
```

### Fig. 24

```diff
+# all at once, labelled batch fig24: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig24
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase BATCH=fig24
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_tanh BATCH=fig24
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_sigmoid BATCH=fig24
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_softsign BATCH=fig24
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_softplus BATCH=fig24
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_softmax BATCH=fig24
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_linear BATCH=fig24
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_hard_sigmoid BATCH=fig24
```

### Fig. 25

```diff
+# all at once, labelled batch fig25: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig25
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_SNN BATCH=fig25
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbest BATCH=fig25
```

### Fig. 26

```diff
+# all at once, labelled batch fig26: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig26
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbest BATCH=fig26
```

### Fig. 29

```diff
+# all at once, labelled batch fig29: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig29
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbest BATCH=fig29

# the same on the official ASCAD_desync100.h5 (the paper's own jitter draw)
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_MLPbest BATCH=fig29

##if you want to process raw traces instead of the official ASCAD_desync100.h5 (the paper's own jitter draw), use the following commands:
# make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100_raw MODEL_ID=ASCAD_MLPbest BATCH=fig29
```


## 3. The experiments

**Queues.** `make experiment` runs the commands of one key of
`config/experiment_queue.yaml` in the order listed:

```bash
make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig6            # run, or resume where it stopped
make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig6 FORCE=1    # train every experiment of the queue again
```

**Reading the results (after every experiment).**

```bash
make judge ENV=dev PROJECT=DLSCA_ASCAD               # ranks the finished models, fills graph job 0
make find_model ENV=dev PROJECT=DLSCA_ASCAD          # find the models of the desired params setting
make batch_label ENV=dev PROJECT=DLSCA_ASCAD BATCH=fig6   # label the runs find_model listed (make experiment QUEUE=fig6 labels its runs itself)
make graph ENV=dev PROJECT=DLSCA_ASCAD GRAPH_ID=1    # Enter the names of shorlisted model_names and plan the graph:job:-id for plotting the graph
make graph ENV=dev PROJECT=DLSCA_ASCAD GRAPH_ID=12   # multiple jobs can be designed at once based on the available model_names
make graph ENV=dev PROJECT=DLSCA_ASCAD FACE_ID=1     # plotted graphs can be arranged as facet and made as one Figure. 
```

Each experiment below: what to set in `params.yaml` first is in the figure's
row of the [Figure index](../../projects/DLSCA_ASCAD/README.md) and in the comment above its
queue in `projects/DLSCA_ASCAD/config/experiment_queue.yaml`.

### Experiment 1 — CNNbest, synchronised traces (the headline)

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbest
```

### Experiment 2 — CNNbest, desynchronisation 100

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbest
```

### Experiment 3 — MLPbest, synchronised traces, 800 epochs (the identity 800-epoch run of Experiment 9)

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbest
```

### Experiment 4 — VGG-16, synchronised traces

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_VGG16
```

### Experiment 5 — VGG-16, desynchronisation 100

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_VGG16
```

### Experiment 6 — CNNbase, synchronised traces, 10 to 130 epochs (one run each)

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase
```

### Experiment 7 — CNNbase, desynchronisation 100, 10 to 130 epochs (one run each)

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase
```

### Experiment 8 — MLPbest, desynchronisation 100

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_MLPbest
```

### Experiment 9 — MLPbest, Hamming-weight and identity labels, 20 to 800 epochs (Fig. 26: 14 runs)

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbest
```

### Experiment 10 — Self-normalising network (SNN)

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_SNN
# optional comparison outside the paper:
```

### Experiment 11 — number of blocks (Fig. 5)

```bash
for d in ascad ascad_desync100; do for m in blocks2 blocks3 blocks5; do
  make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=$d MODEL_ID=ASCAD_CNNbase_$m
done; done
```

### Experiment 12 — CONV layers per block (Fig. 6) and the conv3 epoch study (Fig. 8 right)

```bash
for d in ascad ascad_desync100; do for m in conv0 conv2; do
  make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=$d MODEL_ID=ASCAD_CNNbase_$m
done; done
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase_conv3
```

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_conv3
```

### Experiment 13 — number of filters (Fig. 7)

```bash
for d in ascad ascad_desync100; do for m in filters16 filters32 filters128; do
  make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=$d MODEL_ID=ASCAD_CNNbase_$m
done; done
```

### Experiment 14 — kernel size (Fig. 8 left)

```bash
for m in kernel6 kernel11; do
  make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbase_$m
done
```

### Experiment 15 — number of fully-connected layers (Fig. 9)

```bash
for d in ascad ascad_desync100; do for m in dense0 dense1 dense3; do
  make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=$d MODEL_ID=ASCAD_CNNbase_$m
done; done
```

### Experiment 16 — activation function (Fig. 10)

```bash
for d in ascad ascad_desync100; do for m in sigmoid tanh; do
  make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=$d MODEL_ID=ASCAD_CNNbase_$m
done; done
```

### Experiment 17 — pooling layer (Fig. 31)

```bash
for d in ascad ascad_desync100; do for m in avgpool stridepool; do
  make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=$d MODEL_ID=ASCAD_CNNbase_$m
done; done
```

### Experiment 18 — padding (Fig. 32)

```bash
for d in ascad ascad_desync100; do
  make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=$d MODEL_ID=ASCAD_CNNbase_validpad
done
```

### Experiment 19 — CNNbase batch size (Fig. 3)

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbase
```

### Experiment 20 — ResNet-50 and Inception-v3 (Fig. 2)

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_ResNet50
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_InceptionV3
```

### Experiment 21 — MLPbase batch size (Fig. 19)

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase
```

### Experiment 22 — learning rate (Fig. 21)

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase
```

### Experiment 23 — optimizer (Fig. 20)

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase
```

### Experiment 24 — MLP depth (Fig. 22)

```bash
for n in 3 4 5 7 8 11; do
  make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_layers$n
done
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase   # 6 layers: MLPbase itself
```

### Experiment 25 — MLP width (Fig. 23)

```bash
for u in 20 50 100 150 250 300 500; do
  make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_units$u
done
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase   # 200 units: MLPbase itself
```

### Experiment 26 — MLP activation function (Fig. 24)

```bash
for a in tanh sigmoid softsign softplus softmax linear hard_sigmoid; do
  make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLP_$a
done
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase   # ReLU: MLPbase itself
```

### Experiment 27 — size of the profiling set (Fig. 18)

```bash
# profiling_range: [0, 10000], epochs: [1000]
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase
# profiling_range: [0, 20000], epochs: [500]
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase
# profiling_range: [0, 30000], epochs: [350]
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase
# profiling_range: [0, 50000], epochs: [200]
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbase
```

### Fig. 2: the ResNet-50 and Inception-v3 baselines

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_VGG16
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_ResNet50
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_InceptionV3
```


## 4. Ship and reproduce

What a snapshot is, the checks and the refusals:
[Ship and reproduce](../../projects/DLSCA_ASCAD/README.md#ship-and-reproduce-a-batch-from-its-snapshots).

```bash
make snapshots ENV=dev PROJECT=DLSCA_ASCAD               # every snapshot on disk, checked, and its state
make reproduce ENV=dev PROJECT=DLSCA_ASCAD BATCH=fig26   # register the batch, then train it from the snapshots
make reproduce ENV=dev PROJECT=DLSCA_ASCAD MODEL=cake_bacon,celery_plush   # named runs (snapshots without a batch label)
```

Into a folder of your choice (`REPRODUCE_ROOT`, an absolute path; the folder is
the root of the reproduced runs):

```bash
make reproduce ENV=dev PROJECT=DLSCA_ASCAD MODEL=cake_bacon REPRODUCE_ROOT=/data/1/2/3/pikachu
```

Every artifact carries its own signature
([tamper-evident artifacts](../../projects/DLSCA_ASCAD/README.md#tamper-evident-artifacts)):

```bash
make verify ENV=dev PROJECT=DLSCA_ASCAD                  # every file's seal; exit 1 on a changed one
make seal   ENV=dev PROJECT=DLSCA_ASCAD                  # once: sign the runs written before seals existed
make seal-key                                            # the key id to publish with the results
```
