# TraceFlow Monitor

<img src="brand/traceflow-mark-light.svg" alt="TraceFlow logo" width="48" align="right">

A VS Code side panel (the TraceFlow icon in the Activity Bar: a power trace
running into a forward chevron, also the project's mark in `brand/` beside this file)
that watches every project declared in
`services/side_channel_core/config/master.yaml`. Private; installed from this
repository, never published.

Each project and each of its parts (Queues, Figures, Experiment Settings, every config
file and every block in a form) is a **collapsible section**: a heading (the
*summary*) that opens and closes its content, the HTML `<details>` /
`<summary>` disclosure element. The panel remembers which are open.

## Status page (share button)

The share button on the *TraceFlow Monitor* title line (top right; `package.json` command `traceflow.share`, its
icon `brand/traceflow-share-{dark,light}.png`: the share mark in VS Code's focus blue with the planner's + halo,
made by `brand/make_brand.py share`; VS Code draws that line, so the glow is part of the icon) opens the status page's settings in the panel: a read-only view of every project's runs in any browser
(desktop or phone, light or dark as the device), and
links to share it (`services/visualization/server/status.py` serving `services/visualization/mobile/`).

- **Settings**: master.yaml `status_share: {enabled, port, host}`: a TCP port the network lets through to this
  machine (its firewall, or IT for a VPN); the address is found by itself (`host` chooses another). The port is shown read-only
  here (the address only inside each person's link) and the block in Master Control (edit them in the file). Optional: no block, or `enabled: false` (a server that may not open a port): the panel says *Off* and why.
- **Start** / **Stop**: `make status_share ENV=dev` (on those settings) in tmux session `tf-status`
  (it keeps serving when VS Code closes; log `logs/status.txt`).
- **People**: one panel per person: name and date, **×** at the right end (withdraw, asks first; the link
  stops at once); opened, only their QR code and the words **Click to share or open in external browser** (the link
  behind them, its address and token never shown): click to open it in your browser (VS Code forwards the page's
  loopback port over the session), **⧉** copies it to send. Below the panels,
  **Add a person** (or Enter) and the instructions. Links live in `secrets/status_shares.json`.
- A link that opens on this machine but not on another, and the rest: [guide 11](../../doc/11_phone_status_share.md).
- The page (VPN on: scan the QR code or open the link; iPhone: Share → Add to Home Screen; Android: ⋮ → Add to
  Home screen) shows each project's light, run, phase, progress, ETA, counts and the running run's latest
  metrics, drawn with this panel's `state.js`; nothing can be started or stopped from it.

## Master Control

Above the projects, **Master Control** is the form of `master.yaml` (see *Editing*): every project's
entry, `default_project` (a menu of the projects), `timezone` (`current_country` a menu of the countries with
zones, `current_location` a menu of that country's cities: picking a country loads its cities at once and,
when the city is not one of them, stages the country's principal city with it), `distributed`, `secrets`,
`status_share` (read-only: the status page's port and, when given, its address).

- **Read only while any run is ongoing** on this machine: every run (each orchestrator pass) and make
  read master.yaml, and it is not frozen with a run's config (`utils/config_freeze.py`); the form says
  so, and the extension refuses writes too.
- **Shown, not edited** (`webview/menus.js` `masterTier`): a project's `name`, `root`, `params`,
  `env_configs`, `code_dir`, `use_core_src`, `logs`/`models`/`metrics`, `exp_rerun.path`, the
  `auto_search` and `graph_manager` config paths, and `secrets`; each says why (edit it in the file).
- **Run hash** (marked): `distributed`, a project's `scripts`, `extra_configs`, `distributed_override`
  join every new run's model_hash; OK asks first, since after the change no DONE run is skipped.
- The rest is a plain setting: a project's `description`, `print_config`, `auto_search.enabled`,
  `exp_rerun.repeat`, `graph_manager.enabled`, the ClearML upload (`visualization.clearml`: enabled,
  `scope` the profile used, global or local, the profiles' addresses).
- **After every write** each project is loaded as a run loads it (`layout.py check-master`: the
  schema, the files it names) and the timezone resolved; the form says the result, or what fails.
- **Add a Project** asks a name and writes `blueprint/master_project.yaml` under it (root
  `<name>/`); the master.yaml watcher then copies the blueprint's eleven files to
  `projects/<name>/config/` and its `PROJECT_README.md` to `projects/<name>/README.md` (the study's
  record: sources, claims, protocol, the settings to set before each queue, a figure index per model
  kind; as `make scaffold`), and the project appears. A project's line has
  **×** (not on `default_project`): it asks, then removes the entry only; `projects/<name>/` and its
  runs stay on disk.

## When a run and its batch end

Under the progress bar, one row: `| ETA | run: 07:24 Taipei (in 4.5 h) |
batch: 07:24 Taipei (in 4.5 h) |` (`ETA ≤` when early stopping makes it an
upper bound). Times are in master.yaml's `timezone`
(`current_location`, the zone of the registry's and the logs' timestamps).

- **Run:** elapsed x (1 - progress) / progress, from the heartbeat; shown once
  the run is 2 % and a minute in ("estimating…" before).
- **Batch:** the run's end plus the trials still to come, each its epochs x
  passes (n_folds + 1 with cross-validation), at the running run's own speed
  per epoch-pass. The trials come from the sweep of params.yaml
  (`layout.py trials`, read again when params.yaml changes); a queue's later
  commands each add the whole sweep. Trials of different epochs take very
  different times, so the batch is not counted in trials.
- With `trainer.early_stopping` the ETA is an upper bound ("at the latest").

## Following a run without a terminal

The pane follows experiments from files, not from a terminal. The trainer
inside the container rewrites `projects/<P>/artifacts/<env>/heartbeat.json`
every 5 s and the orchestrator keeps `artifacts/run_registry.json`; the
pane reads them every 2 s. So the light, phase, fold, epoch, progress bar and
tiles stay live while every terminal is closed, the tmux session that
started the run is detached or gone, or its log stream has ended; and after
VS Code is closed and opened again it shows the run's state at once. No
terminal has to stay open, and none is needed to look.

## What it shows

- **GPU strip**: utilisation, memory and temperature of each GPU from
  `nvidia-smi`, with a 5-minute sparkline. Without `nvidia-smi` it says so.
  Beneath the GPUs:
  - **Docker tear down** runs `make docker-teardown ENV=…` in a terminal after
    asking the ENV and once more (while an experiment runs or an image builds it
    offers **Force tear down** instead: `make docker-teardown ENV=… FORCE=1`
    kills the experiment's make and python, the build with `make kill-build`,
    and the containers by force, then tears down): `make docker-clean`, every `sidechannel*`
    container and `sidechannel:*` image, `builder prune -af`, `image prune -f`,
    `volume prune -f`, an empty `deployment/docker/tmp/` and no build log. The
    prunes act on the whole containerd namespace `default` (other projects'
    build cache and unused volumes too). Refused while an experiment runs or
    an image builds.
  - beside it, the **image build %**: the Dockerfile steps done out of those
    seen so far, read from `logs/build_image.log` (`make build-image` writes
    it); `built`, `failed at`, or `stopped at` once it ends;
  - beneath that, the **health of each `sidechannel*` container** (`sudo -n
    nerdctl ps -a`, read while the panel shows): running, exited (code 0),
    failed (another code), restarting or created.
- **One collapsible heading per project**, with a light and the progress of
  the running experiment:

  | Light | Meaning |
  |---|---|
  | green | a heartbeat less than 15 s old |
  | amber | the last beat is 15–60 s old |
  | red | the run says `running` but has not beaten for over 60 s |
  | grey | idle: no heartbeat, or the run is `done` / `failed`; also a registry run with no heartbeat for this ENV |

  - **Current run**: a dot, green while the run is alive (its own heartbeat, `running`, a beat
    under 60 s old) and red when it is terminated or dead (no beat, stopped, ended); name
    (click for its detail tab), phase
    (`fold k/n · epoch e/E`, `refit`, `test`), progress bar with percentage,
    and its queue (`fig26 6/7 done`).
  - **Tiles** queued (○, yellow) / running (◉, blue, pulsing while any) /
    done (✓, green) / failed (✕, red) from `run_registry.json` (a running run's row
    has the same green / red dot): the count in
    the state's colour (the theme's chart colours), a glyph so the state reads
    without colour, the word in the theme's text colour, a zero dimmed. A
    tile lists its runs, each with a sparkline of the validation rank at the
    last trace count per epoch (its last fold); a run opens its detail tab.
  - **Trial numbers**: a queue plans `total_trials` (the sweep size of one
    command, in every registry row) x its commands; a run is trial k of
    those, k by start order within the queue. The Queued tile also counts
    the trials a running queue has not started yet (`fig26 trial 11/14`).
    The trainer numbers new runs the same way (`trial_index` = the run's
    place in the sweep grid), so the registry, the logs
    (`SCC trial=k/N`) and the panel agree.
  - **Run Single Experiment** (beside Queues; `config/single_runs.yaml`, the Orchestrate… button it
    replaces is gone): **Add a single Run** (beside Open in editor) writes a `make orchestrate` line
    without a BATCH label at once (the followed ENV, the project, its first dataset and model); each line
    reads `make orchestrate ENV=[▾] PROJECT=[▾] DATASET=[▾] MODEL_ID=[▾]` (master.yaml's projects, that
    project's datasets and models), with **Start** and × at its right end. A menu's change goes through
    OK / Cancel / Reset (Add is refused while one is staged). Start runs the line as the file holds it, as the project it names (that project's session
    and logs), refused while changes are staged, or when the line is not `make orchestrate` with ENV,
    PROJECT, DATASET, MODEL_ID once each, or names a project master.yaml lacks or a dataset or model that
    project's params.yaml lacks; Start's usual check that nothing else runs applies. Its runs belong to no
    queue; they are listed in Done, Housekeeping and FIND MODEL like any run. The same file under
    Experiment Settings shows the lines without Start: runs start from Run Single Experiment.
  - **Queues**: **Set Params** and **Set Queue** at the top open the params.yaml and the
    experiment_queue.yaml form under Experiment Settings. Each `experiment_queue.yaml` key with its state (not started,
    queued, running, failed, incomplete: stopped before every planned trial
    ran, done: all planned trials done), taken from the runs whose `batches` hold
    the key. Its button follows the state: Start; Terminate Run while it runs;
    Retry Start once a run of it failed or was terminated (the same
    `make experiment`, which skips the DONE runs and resumes the others);
    Rerun once it is done (the same line again).
    The graph column lists the graph jobs drawing its runs. **Overlay Graph**
    (before Rerun and Log) shows once every queue of the queue's study is DONE: the study is
    the key before its first `_` (`fig14_cnnbest`, `fig14_vgg16`, `fig14_mlpbest`:
    `fig14`), the key itself when a queue of that stem exists (`fig10_optimizers`
    beside `fig10`). It lists the study's axes (what differs between its runs);
    you tick those drawn as lines (pre-ticked: the axis of most values), each
    value of the others is a figure, then the metrics (`metrics_to_log`
    pre-ticked; a fit time also as **relative**, which then asks for the reference
    model, one of the lines, drawn as 1.0 in every figure): one graph job per figure and metric (`graph_export.py study`,
    `array_epoch: latest`); Draw all draws them. A study TraceFlow cannot make
    says so once, with graph_config.yaml to open.
  - **Figures**: each `graph_config.yaml` job, drawn (●) when
    its folder `graphs/jobs/job_NNN_<name>` (a job of one run: `<run>/graphs/job_NNN_<name>`; exactly that
    name, so a folder left by a deleted job of the same id is not shown) holds a `.pdf`, `.svg` or `.png`; Open shows the
    figure beside the editor, Folder reveals the job folder.
  - **Experiment Settings**: the project's config files in blueprint order, each a form
    (see *Editing*), with Open in editor. Beside **Config folder** a key search (a faint search mark in the box):
    Enter shows, in every config file, the keys that hold the text (exact or part of a key, any case), each under its
    parent panels opened, with the count per file and in all; a matching block shows whole, and its values edit as
    usual. The search stays until its **×**, then Experiment Settings shows as before (the forms only it opened let go).
    While a run of the project is ongoing (`make orchestrate|experiment|test|reproduce` naming its PROJECT, or a
    trainer naming none), the files a run reads (params.yaml, experiment_queue.yaml, dev/staging/prod.yaml,
    test_config.yaml, auto_search.yaml, model_config.yaml) stay editable, with a note: the run reads the config
    frozen at its start (`utils/config_freeze.py`), so changes apply to the next run.
    experiment_queue.yaml: **Add a Queue** beside Open in editor asks the new key (its runs'
    BATCH label) and writes it at the end of the file with one line to fill; each line of a queue
    shows `make orchestrate` as text and menus for `ENV=` (dev, staging, prod), `PROJECT=`
    (master.yaml's projects), `DATASET=` and `MODEL_ID=` (that project's params.yaml), then × (remove)
    and, under the lines, + (a copy of the last line); a line of another shape stays one text box.
    params.yaml: `deep_learning` shows the block of the framework chosen (`deep_learning.framework`,
    `framework_switch.FRAMEWORK_BLOCKS`) and not the other; a list adds with a **+** button, which opens one
    control for the new value (a menu of the options left, else a box; `sca.candidate_ranking.mode` is one
    choice, a menu without + once it has its value, whose options say what they rank: all (all 256 key
    candidates), none (no candidates), top_k (the k best); `top_k` shows with mode top_k only, and a box asks
    it when the file lacks it; × closes the control again), a value's × removes it
    from the form at once (OK writes it, Cancel brings it back), and a list keeps one value at least
    (its last value has no ×); a true/false key reads `True` too; every address (a data entry's
    `database`, every `paths.*` key, a key named `*_path`, `*_dir`, `*_file`, `*_root`) has a folder
    button at the right end of its box that picks the file or folder in the system (a database opens
    at `services/side_channel_core/data` and is written from the TraceFlow root, any other address from
    the project's folder); `trainer.monitor_metric` (for early stopping) is a menu of `none`, the
    metrics of `metrics_to_log` a validation pass writes as one number and `validation_objective`
    (`utils/monitor_metric.monitorable_metrics`; training refuses any other monitor when early stopping
    or validation checkpointing reads it); `trainer.loss.type` offers only CrossEntropyLoss under keras
    (NLLLoss and FocalLoss are PyTorch's, `framework_switch.KERAS_LOSSES`).
    `sca.rank_base` and `data.*.multilabel` are no longer settings (TraceFlow ranks from 1; single-label
    only): a file that still names them shows them read-only, with why.
    `device` is a menu: cuda, cpu and cuda:N for each GPU nvidia-smi reports, with helper text under it for the
    one chosen (cuda: PyTorch's name for an NVIDIA GPU, it has no device called gpu; fails without a usable GPU,
    never a CPU fallback; cpu: on purpose; cuda:N: that GPU only); it is part of the run hash, so another value
    makes new runs.
    `seed` has a **+**: a second value makes it a sweep list (`seed: [42, 53]`, one run per seed; with
    cross-validation a run uses its seed and the next n_folds seeds, so seeds n_folds + 1 apart keep the runs'
    seeds disjoint).
    `evaluation.test.load_model: best_validation` while `evaluation.validation.checkpointing` is not true is
    marked red ("needs checkpointing: true"; the test would load final_epoch); the mark goes as soon as either is fixed.
    test_config.yaml: `attack_distribution` (shared and per model) is a menu: full_population, disjoint.
    `trainer.optimizer` is a list, one block per optimizer: its line (#i, its type and learning rates, × at the right
    end while another one is left) and, opened, its settings, `type` a menu of the framework's optimizers
    (`optimizer_util.OPTIMIZER_REGISTRY`; keras: `framework_switch.keras_optimizers`, every keras.optimizers
    class), `learning_rate`, `scheduler`, `weight_decay`. **Add an Optimizer** writes a new block at once, opened (a
    copy of the first one: its learning rates, no scheduler, no weight decay), to choose its optimizer and values in. `trainer.evaluation.output_transformation.type` offers Identity, Softmax,
    LogSoftmax (every one works in both frameworks: the evaluation applies it to the logits).
    Each `model` and `data` entry's line has × at its right end while another one is left: it asks first,
    naming the queue lines that still name it (`MODEL_ID=` / `DATASET=`; they fail until changed), and Delete
    writes it at once (refused while the form has staged changes; during a run the dialog says the run keeps its copy).
    `data`: **Add a Dataset** under its line writes the blueprint's data entry at once, under a free name
    (`new_dataset`, `new_dataset_2`, ...), opened: its name, type, database, ranges, desync, normalisation
    and switches to fill (until its type and database are filled, a run of the project stops at the config check).
    `model`: **Add a Model** asks a model_id and its kind, a layer model (CNN, MLP, TRANSFORMER, HYBRID, an
    empty layer list) or a named network (ResNet50_1D, InceptionV3_1D, …; PyTorch only), and adds it last.
    A model's `architecture` is a menu while it has no layers; once it has layers it is shown, with what
    they make (convolutions: CNN, attention: TRANSFORMER, recurrence or a mix: HYBRID, else MLP), and a
    label they contradict is flagged with a Set button. Its `layers`: **Check the model** builds it as a
    run would (`model_check.py`: core.model's builder, one zero trace of the first data entry's length;
    PyTorch's shapes, Keras builds the same layers) and each layer line shows its output shape, or the
    build's error at its layer; each layer line has its main values and ▲ ▼ (written at once) ×, and opened
    every argument of its class (torch.nn's signature, `activation` TraceFlow's; a required one says so);
    **+** adds a layer from a menu of every layer core.model builds, by group (Convolution, Pooling,
    Dense, Normalization, Dropout, Shape, Recurrent, Attention, Merge), under keras the ones
    `framework_switch.KERAS_LAYERS` maps (Pooling: `KERAS_POOLING`); a new layer comes with its required
    arguments to fill (Pooling with dim 1). The layers a layer list cannot build on 1-D traces
    (`core.model.LAYER_LIST_LIMITS`: the 2-D and 3-D layers, Embedding, the recurrent and attention layers)
    are listed in every menu but disabled, each with why; TRANSFORMER and HYBRID, which only those layers
    make, are disabled the same way. A layer leaves the list, everywhere at once, when core.model builds it.
    **Plan a Model** (beside Add a Model) and **Visualise** (each layer model's line; not a named network)
    open the planner: see *Plan a Model and Visualise*. **Import a Model** (beside them) takes a model another
    study released: a Keras HDF5 file picked (read with h5py only, `layout.py import-describe`: its Keras version,
    input, outputs and the leakage model they suggest, or why it is refused), then its source, the data entry its
    traces come from (those of its input length first), the leakage model and a fitted scaler's fit set. The import
    goes into the project's `import_config.yaml` (`layout.py import-add`: created from the blueprint when missing;
    a file outside the project copied into its `released/` folder, never over another), which opens there for the
    preprocessing, seed and attack; **Import now** runs `make import` in tmux after the busy checks
    (`utils/model_import.py`: nothing trains, the equality gate checks the copied weights, the standard test follows).
    `sca.attack_byte` is the data entry's (`dataset_types.attack_bytes_allowed`: the type's target bytes within the
    file's trace window, narrowed to the bytes a `trace_window_verified_by_attack_byte` declares): where every entry
    defines one byte, the line has no menu and no chooser, a value the file holds reads as text with × to clear it,
    and the hint says *from data.type at the time of the run* (`type.attack_byte = {…}` stands beside each entry's
    `type`); only where some entry allows several bytes (CHES_CTF_2018, a raw-trace entry without a verified window)
    is it chosen, the unavailable bytes listed with why; a scheduler's `type` the framework's schedules
    (`torch.optim.lr_scheduler`; keras: `framework_switch.keras_schedules`); a layer's options are the ones torch
    builds and runs (ConvTranspose: numeric padding, padding_mode zeros) and the activations core.model builds
    from a name alone.
    `metrics_to_log`: each entry is a menu of every metric sca.metrics computes (fit times as the file
    spells them, `mean_fit_time_{s}`), and + offers the ones not listed yet. A metric that takes a value
    (`sca.metrics.METRIC_OPTIONS`: k of top_k_accuracy, rank_k_success, traces_to_ge_k, sr_curve, auc_sr,
    time_to_recover; traces of ge_at_traces and rank_at_traces; p of rank_percentile; window of
    ge_reduction_rate) shows it beside its name (a bare name: its default, as placeholder); picked in +, its
    value is asked first (Enter adds it), written as `- name: {option: value}`, the column it records
    named after it (`traces_to_ge_3`, `top_7_accuracy`).
    model_filter.yaml (FIND MODEL): `where` is every field the project's runs hold
    (`model_filter.py --fields`), grouped by where the filter finds it (Run, Registry, Sweep values,
    Snapshot, Device; the last two closed), each a tick box, its path engraved and its values; a
    ticked field is a condition: an operator that fits it (a list: contains, count, exists; a number:
    equals … between; text: equals … matches) and its value from the runs' values (tick boxes for
    in / not_in, two boxes for between). **Find Models (n ticked)** writes the ticked conditions to
    `model_filter.where` (`--write-where`) and runs `make find_model`; Reset takes the file's again.
- **Detail tab** (beside the editor): one run's facts and every curve it has
  written: each `arrays/<process>/<metric>/epNNNN.npy` folder (value at the
  last row, per epoch) and each numeric field of its `scalars/*_history.jsonl`.

The panel owns no process: closing it, or VS Code, stops nothing.

## Plan a Model and Visualise

An editor tab (`webview/planner.js`, logic `webview/planner_core.js`); it survives a window reload.

- **The splitter** between the settings and the cards: drag it to widen or narrow the settings (180 px to 60 % of
  the window; ← → by 16 px when focused; double-click: back to 240 px). The tab keeps the width over a reload.
- **Left: settings.** The Type (CNN, MLP, TRANSFORMER, HYBRID; the last two disabled until core.model builds
  their layers) and its settings, none tied to a dataset. Each setting's name, box or menu says what it means on
  hover, and so does each choice of a menu (`planner_core.knobHint`, `optionHint`: e.g. Filter limit's brackets are
  each block's filters with that choice). Numbers are typed (refused with why), steps are menus:
  blocks (with the trace length each leaves), filters (powers of two), **Filter limit** (none, or a power of
  two above the first filters when doubling, below when halving; each option shows the filters per block).
  Width boxes (Dense width, Units per layer, Feed-forward width) take any whole number and suggest, each with
  its reason: the inputs the first dense layer receives, the classes the output gives, the geometric pyramid
  rule √(inputs × classes); for feed-forward 4 × and 2 × d_model; then the project's widths and powers of two.
  The settings rebuild the whole model; after hand edits a change asks: keep my changes, rebuild, cancel.
  A CNN's settings: **Blocks**; **Block Pattern**, what one block is, as chips (**[Conv1d] → [BatchNorm1d] →
  [Pooling]**; ◀ ▶ move one, × takes it out, **+** adds a kind at the end): any kinds, any number of each, in any
  order, none at all (e.g. `conv pool` VGG-style, `norm conv pool` pre-norm, `conv norm` no pooling,
  `conv pool conv pool` two poolings, `pool` alone as ASCAD_CNNbase_conv0); **Repeats per block**, how many times
  the pattern repeats in a block, a pooling at its very end once per block (one number, or one per block such as
  `2,2,3,3,3`; with one conv: its Conv1d layers);
  the filters (first, per block, limit); **Kernel size**; **Padding** (half the kernel, same, valid);
  **Normalization**, the norm chip's type; the dense layers and their width. A design starts with relu on the last
  layer before each pooling and on each hidden dense layer, and with max pooling of window 2.
  Below the settings, three folds set what differs layer by layer (each row then *edited*, the layer's own, kept
  when a setting changes):
  - **Activation per layer** (its tally in its line): every layer that can carry one (Conv1d, the normalizations,
    Linear) with its own menu, none or any activation. An activation runs right after the layer it is on, so the row
    shows where each runs (on a BatchNorm1d: Conv1d → norm → act; on the Conv1d: Conv1d → act → norm). **all that
    have one** sets every layer that has an activation at once.
  - **Pooling per block** (its tally in its line): each block's pooling, its type (none, max, avg) and window (the
    stride too). none takes it away, a type puts it back at the block's end.
  - **Dropout per layer** (its count in its line): every layer a Dropout can follow (a conv, a normalization, a
    pooling, Flatten, a hidden Linear), none or 0.1 … 0.5: a Dropout layer right after it, or none.
  **Edit N layers** sets one argument on the layers ticked, and a card opens all of its own. A model read from
  params.yaml keeps its settings whenever its layers are theirs one for one: its pattern and repeats are read from
  its blocks, its Dropout layers set aside and put back after their layers, and every argument it sets otherwise (an
  activation, a pooling's type or window, one layer's kernel) is that layer's own; `in_features` and the like still
  follow the layer above. Only a model whose layers differ in kind or number reads as "custom" (none of
  DLSCA_ASCAD's).
- **Middle: layer cards**, coloured by group, each opening its arguments; **+** between two cards adds a layer,
  ▲ ▼ × move and remove. `in_channels`, `num_features`, `in_features` follow the layer above unless set by hand.
- **Selection.** A tick box per card; over the cards **Select all**, the count, **Delete N layers**, **Edit N
  layers** (one form: all arguments of one type, the shared ones of mixed types, `(mixed)` where values differ;
  each value set is a hand edit). A skeleton change clears the selection.
- **Right: one bar per layer** on a log axis (length: output length or features; thickness: channels). The
  window follows the Download **style** (its draft, as the theme): *Full Version* the cards and their bars,
  *Layer cards* the cards alone, *Scaled Sparkline* the bars alone, as the diagram draws them (the index at each
  bar's start, the arguments outside the bars, the tick axis, the layer legend at its corner; landscape: a column
  per layer growing up from one baseline); the layers are edited on the cards.
- **Input.** The data entry the model is built for: a menu of params.yaml's entries with the trace length each
  gives (its `target_range`), the first chosen; a project without a data entry yet (every new project) takes the
  length typed there (1 channel; `model_check.py --input LENGTH` stands it in for the entry) and sends no check
  until it is typed. Add a Dataset in the panel gives the entry.
- **Numbers.** Flatten size, weights, the params.yaml model it equals, and the model check (`model_check.py
  --layers-json -`; under keras the Keras build).
- **Buttons.** **Add the Model** appends the design under `model:` with a new, unique model_id (only when the
  check passes and the params.yaml form has no staged changes); **Preview Code** / **Encapsulate** shows or
  folds the YAML it would write; **Reset**; **Cancel** closes the tab; under them, **Download** (also in a trained
  run's view, and with Observe Trace on) saves the model's diagram (`model_diagram.py`, Matplotlib, light). Its
  form shows a choice's settings only once it is chosen:
  - **theme**: the colours, any theme of `services/visualization/utils/fig_theme.py` (the paper's figures use the
    same): `traceflow` (the planner's own light colours, the default), `cyberpunk`, `okabe_ito` (colour-blind
    safe), `viridis`, `grayscale`, `solarized`, `solarized_dark`, `dracula`, `monokai`, `gruvbox`, `tokyo_night`,
    `nord`, `samurai`, … (24); once chosen, the cards and bars in the window take it too (a preview of the
    diagram's colours); each layer group takes one palette slot (`model_diagram.GROUP_SLOT`);
  - **style**: *Full Version* (the layer cards and their bars on the log axis), *Layer cards* (the cards alone:
    #index, name, every argument, the output shape) or *Scaled Sparkline* (the bars alone; then **arguments**
    beside the bars, every layer's wrapped at one measure: the longest wrapped into as few lines as the room allows,
    as evenly as its arguments split, **tick labels**, **orientation** portrait / landscape (layers across the page, the
    arguments reading upward from the bars' edge) and the **layer legend** (each layer type's colour and how
    many, ×n; in as few rows as the page allows, filled evenly) top-right, bottom-right, top-left or none);
  - **data-flow arrows** between the layers; **trace lines** (with Observe Trace showing them; not in Scaled
    Sparkline): each layer's line under its card, the trace on top, the caption of whose weights and which trace;
  - the formats with their own settings under them (**PNG** dpi, **PDF**, **SVG** sanitize / validate_xml /
    on_error, **EPS**) and the margin: a graph job's `export:`, checked and saved by graph_manager
    (`normalize_export_config`, `save_figure`);
  - **text_rules**: for each text of the diagram (title, subtitle, facts, layer_labels, arguments, shapes,
    tick_labels, legend_text, trace_caption) an **enabled** true / false switch, then graph_manager's fields (text,
    bold, italic, color, font family and size); every rule starts off at font size 8, and only an enabled one is
    drawn (text replaces the words of title and subtitle only); **Clear** puts it back (off, size 8).
  - A change to any of these is recognised: **Download settings · N changes recognised** over the buttons, and
    beside **Download** (which waits while changes are pending) **OK** (keep them: Download draws with the
    settings kept), **Cancel** (undo them, fold the text_rules away) and **Reset** (undo them, the form staying as
    it is).

  The files go to params.yaml `paths.model_preview` (`artifacts/<env>/model_preview/<name>.<fmt>`; env: a
  trained run's, else the one followed), named after the model, the trained run chosen and the Observe Trace
  settings drawn: `ASCAD_CNNbase`, `ASCAD_CNNbase__tulip_hold`, `ASCAD_CNNbase__trace_ascad_untrained_t5`,
  `ASCAD_CNNbase__tulip_hold__trace_ascad_t0-9` (a range). They are copied, replacing earlier ones, into
  `paths.model_out` of every DONE run of the model whose recorded layers are the ones drawn (`auto` alike what it
  became) as `<run>_diagram[__trace_…].<fmt>`; trace lines through a trained run's weights go into that run's
  folder only. The tab names runs left out, and runs whose folder is root's (guide 03 §3.1).

**Visualise** starts from that model (its settings, with any layer's arguments of its own; "custom" only when its layers
differ in kind or number from what settings build). Every layer-list model of DLSCA_ASCAD reads its settings exactly.
Its changes are staged in the params.yaml form as one `yamledit.py arrange` of its layers (comments kept).
Any edit (a setting, a chip, a per-layer menu, a card) brings the **Configuration Change · <model_id> · N
changes** bar to the top of the settings (tooltip: changed, added, removed, moved); back at the saved layers
(by hand too) it goes. Its buttons (each with its hover hint):

| Button | Does |
|---|---|
| **OK** | overrides the model in params.yaml with the changes (refused while the form holds other staged changes or the file's layers changed since) |
| **Cancel** | undoes every change and closes the tab at once |
| **Reset** | undoes every change: the model as it was when the tab opened it; the tab stays (Undo takes the Reset back) |
| **Add the Model** | asks a new model_id first (Enter or **Add**; **×** closes the ask), then adds the changed design under it and shows it; the original model stays unchanged (its staged copy is dropped first, staged again when the write fails) |
| **↶ Undo** / **↷ Redo** | one change back / forward (Ctrl+Z; Ctrl+Shift+Z or Ctrl+Y, outside text boxes; the keys work while the bar is gone, too) |

A setting changed in Visualise applies at once, keeping the layers' own values where they still fit (no
question; Undo takes it back); each layer keeps its own line in params.yaml, so changing Kernel size of a
4-block CNN is 4 changes. **Preview Code** shows the model's own entry in params.yaml under its model_id as OK
would write it (its comments and style kept), the edited lines marked.
Visualise has no model_id box or Add the Model of its own. The per-layer rows and the cards mark **edited** only
what differs from params.yaml (the Configuration Change); a model as the file holds it (e.g.
ASCAD_CNNbase_sigmoid's own BatchNorm1d activations) shows no mark. Plan a Model marks its hand edits **edited**.

**Observe Trace** (**Hide Trace** folds it; nothing is traced while hidden) draws under each card what that
layer makes of an attack trace (`trace_view.py`): channels as their root mean square per sample, vectors as
values, at most 120 points; the trace first, the top classes last; drawn top to bottom on arrival (at once with
reduced motion).

| Control | Does |
|---|---|
| **Data trace** | the project's data entry (name · type); one not on this machine is disabled with why; nothing is downloaded |
| **Observe** | single trace, or range of traces (From, To; at most 1000): each line the mean, with a band from the lowest to the highest value |
| **Trace number** | attack trace n: attack_range start + n, the entry's window, a desync offset drawn from the run's seed, its normalization (a fitted scaler is fitted on the entry's profiling traces) |
| **Weights** | untrained (seeded), or (Visualise) a DONE run of that model_id: its snapshot's layers read-only, its data entry fixed, trace n of its own `test.h5` with its scaler, the checkpoint `make test` loads (manifest, SHA-256 and seal checked; Keras runs too) |

The header names the trace; its tooltip gives the file, group, row(s), samples, desync offset and
normalization. Faint notes, as bold-labelled points, explain the desync offset and the classes (outputs,
leakage model and attack byte with where they were read, top classes; trained: true class with its plaintext
and key bytes, its rank against chance; range: median single-trace rank, not guessing entropy).

In the trained view **Show Key Recovery** (**Hide Key Recovery**) shows what the run recorded, from
`run_registry.json` and `test_registry.json` (seals checked, nothing recomputed): how it was trained (k-fold
cross-validation or a hold-out train / validation / test split), one row per CV fold (`fold_key_recovery`) or
its validation attack, the recorded test (or its status and why, e.g. SKIPPED with attack_times 0) and each
external test with its attack; columns Recovered (✓ green, ✗ red), Top key guess, True key rank, Rank 1 from.
Every number in its notes is the run's: snapshot (folds, ranges, perform_attack) and sealed key-recovery
reports (traces per attack, key space, `recovery_rank`, `require_stable_recovery`).

## Buttons

| Button | Runs | Where |
|---|---|---|
| Run Single Experiment → a line's Start | that line of `single_runs.yaml`: `make orchestrate ENV PROJECT DATASET MODEL_ID` | tmux `tf-<project>`, log `projects/<P>/artifacts/<ENV>/tmux_logs/orchestrate_<MODEL_ID>_<DATASET>.txt` |
| Test | `make test ENV PROJECT` (models of test_config.yaml) | tmux `tf-<project>`, log `projects/<P>/artifacts/<ENV>/tmux_logs/test.txt` |
| Queue Start | `make experiment ENV PROJECT QUEUE=<key>` | tmux `tf-<project>`, log `projects/<P>/artifacts/<ENV>/tmux_logs/<key>.txt` |
| Attach / Detach (beside the running run's name) | the tmux session holding the run: the one running its `make` (its batch's first: `QUEUE=`/`BATCH=` in the make line, found from the running make's parent processes, whoever named the session), else a session following the container's log (`nerdctl logs`, e.g. `fig26` after its own stream ended). **Detach** while a terminal is attached to it (`tmux detach-client -s`; the run keeps going), **Attach** when detached (`tmux attach -t`); no session: a notification | a terminal |
| Container log (beside Attach) | the training container's live output, whoever started it: `sudo nerdctl logs -f --tail 100 sidechannel` (sudo may ask for your password) | a terminal |
| Kill session (beside Attach) | `tmux kill-session` of that session, only after *Are you sure? → Yes, kill*; its make chain and log stream end, the container may keep training (`make stack-down ENV=…` stops it); no session: a notification | — |
| Terminate Run (beside Kill session) | stops the run itself, after one confirmation: kills the tmux session holding its make chain (else `make experiment` would start the queue's next command), then `make stack-down ENV=…`, which stops the training container (sudo may ask for your password). The run stays `in progress` in the registry; the next `make orchestrate` resumes or retrains it. Its dot turns red once the heartbeat stops | a terminal |
| Done tile: **Retest** at the right end of the run's line; under it every metric the run recorded (grouped by source · process) and **Export to Graph Config**, or **Delete from Graph Config** when a graph job already draws that metric for the run | Export writes the run into the graph job of its batch, dataset, source, metric and process (`graph_export.py export`; made when there is none, with `models` commented by what differs in the batch, `legend.labels`, a header); **Draw** then offers `make graph`. Delete asks first, then takes the run out of those jobs (`graph_export.py remove`); a job left with no run is deleted and the later jobs move down one id (folders, facet panels, header lines follow) | `graph_config.yaml` |
| Config forms → a facet's panel `address` → **Find Figure** | a file picker in `graphs/jobs/`; the chosen figure's path, relative to `graphs/` (a run's figure: relative to the project, `artifacts/...`), is staged into the box (or type it); OK writes it | `graph_config.yaml` |
| Config forms → a facet's `arrangement` | rows × columns picked on a grid, as Word's Insert Table; the grid is 4 × 4, **Rows** and **Columns** under it set its size (up to 20 × 20; a smaller grid trims a larger layout); then, left-aligned on its own line, one box per cell, each a menu of the facet's panels (file_id: file); a panel goes in one box. One row is written as `mode: horizontal`, one column as `vertical`, else `grid` with `order` one list per row; OK writes `mode` and `order` | `graph_config.yaml` |
| Config forms: a key with a known set of values | a drop-down of exactly the values the system accepts, read from the code that checks the key (`choices.py`: job `type`, `plot_type`, `source`, `processes` by the job's source, `data[*].type`, `trace_normalization.mode`, `leakage_model`, `framework`, `export.formats`, facet `arrangement.mode`, …); a set the code extends (a torch optimizer or scheduler, an activation, a font, a Matplotlib marker) suggests and still takes a typed name; any true/false value is a true/false menu; run names, dataset names, `model_id`s come from the project's files. A value in the file that is not an option stays shown, marked, until you pick one | the config file, on OK |
| FIGURES → **Graphs**: each row `[tick box] (status dot: green drawn, grey not yet)` then id, name and, at the right end, Open, Folder, Delete, Draw (a figure not drawn yet: Draw alone); above the rows **Select all** and **Draw all (n chosen)**. FIGURES → **Faceting**: every facet, `[tick box] (status dot)` id, name, Open, Folder, Delete, Draw (`make graph FACE_ID=`; not drawn yet: Draw alone; Delete removes its `graphs/facets/facet_NNN_<name>/` folder, asks first); above the rows **Select all**, **Draw all (n chosen)** (`make graph FACE_ID=` for each ticked, one after another) and **Delete all (n chosen)** | Draw all runs `make graph GRAPH_ID=<id>` for each ticked figure, in the order ticked, one after another in the project's terminal (one ENV asked once; a job that fails does not stop the rest); the dots turn green as the figures are written | `graphs/jobs/` |
| FIGURES → tick figures → **Export to Facet**, **Clear** | the ticked figures that are drawn, in the order ticked, become one facet of `graph_config.yaml` (`graph_export.py export-facet`): panels numbered in that order, one row for up to three, a grid beyond, a header naming each panel's job, every key commented with its options; **Draw** then runs `make graph FACE_ID=<id>` | `graph_config.yaml` |
| A graph job (or a facet) removed from `graph_config.yaml` in the editor | the panel asks once (the removed job's panels also leave every facet; a facet left with fewer than two goes): **Shift ids** moves every later job down one id (`graph_export.py close-gap`: the id, its figure folder, the faceting addresses into it, the `JOB N`/`GRAPH_ID=N` lines of its header; the removed job's figures to `graphs/jobs/_deleted/`; mentions of old ids elsewhere are listed, not edited), **Leave the gap** keeps every id | `graph_config.yaml`, `graphs/jobs/` |
| Housekeeping (at the end of each project): filters ENV, MODEL_ID, DATASET, STATUS, BATCH, NAME over the runs of the registry (newest first, each with a tick box; dataset from its snapshot); **Delete all** ticks every run shown; **DELETE** | `make delete_database` for the ticked runs the filters show; the deleted names leave every graph job (`models`, `legend.labels`, job 0's `top_models`). ENV, MODEL_ID, DATASET, STATUS and BATCH are menus of the values the registry's runs hold (MODEL_ID and DATASET also params.yaml's, and a model or dataset params.yaml no longer lists, so its runs can still be found), **any** keeping every run; NAME (a part of the name) and PARAM are typed; a filter kept from before as comma-separated alternatives stays chosen as its own entry; a wrong entry (an ENV, model_id or dataset neither params.yaml nor a run has, a status no run has) is shown instead of the list. The filters are kept per project across reloads; Clear empties them. Refused while an experiment runs on the machine. DELETE runs the target's own dry run (`JSON=1`, `MODEL=` the ticked names), once per ENV among them, and shows one dialog: the experiments it would delete (name · ENV · model · dataset · status · batches), their total and the runs per ENV. **OK** runs the same lines with `CONFIRM=1`; **Cancel** closes. Nothing ticked deletes nothing. Runs go from `artifacts/`, `model_selection/` and `graphs/`; the tmux logs stay | a terminal |
| SNAPSHOTS (after the projects): Scan, Open, Reproduce | one sub-panel per project in master.yaml (its root joined with `paths.params_out`, e.g. `projects/DLSCA_ASCAD/artifacts/<ENV>/generated_models/<model_name>/config/<model_name>_params.yaml`; the pattern is shown) and one **CUSTOM** (**Choose folder…**, remembered per workspace: every `<model_name>_params.yaml` under it; the folder is the root of what it reproduces, every `paths:` entry inside it). **Scan** takes the filters above it (BATCH, MODEL_ID, DATASET, ENV, LEAKAGE, STATUS, NAME; comma-separated values are alternatives): none filled, a wrong entry (not an ENV, leakage model or status; for a project not a `model_id` or dataset of its params.yaml) or no match is an error; otherwise the snapshots matching all of them are listed (state.js `snapFilter`; `layout.py snapshots --project` / `--folder` checks each as a re-run would). Every snapshot: origin, ENV from the snapshot, name, hash, state; **Open** opens the file (only files a scan listed); **Reproduce** starts `make reproduce ENV=<its ENV> PROJECT=<its project> MODEL=<its name> BATCH=<its batch>` (CUSTOM: `REPRODUCE_ROOT=<the folder>`, the snapshot copied to its params_out path inside it after the confirmation by `layout.py place --into`, never over a different file; its tmux log in `<folder>/artifacts/<ENV>/tmux_logs/`) in tmux after the busy checks (*Let the current run finish*); a DONE run asks first and gets `FORCE=1`. The sub-panel then shows what is being reproduced; CUSTOM also the run beating in its folder | the tmux session `tf-<project>` |
| Log | `tail -f projects/<P>/artifacts/<ENV>/tmux_logs/<key>.txt` | a terminal |
| Draw | `make graph ENV PROJECT GRAPH_ID=<id>`; the job's PDF opens beside | the project's terminal |
| Find models (model_filter.yaml) | `make find_model ENV PROJECT`; `artifacts/filtered_models.yaml` opens | the project's terminal |
| Judge (model_judge.yaml) | `make judge ENV PROJECT`; `<output_dir>/selected_model.yaml` opens | the project's terminal |
| Test (test_config.yaml) | `make test ENV PROJECT` on the models listed there; `artifacts/test_registry.json` opens when written | tmux `tf-<project>`, log `projects/<P>/artifacts/<ENV>/tmux_logs/test.txt` |

The config files with a button (graph_config: **×** and **Draw** at the right end of each job's and
each facet's line, its summary: × asks first, then `graph_export.py delete-job` / `delete-facet`
(figures or output to `_deleted/`, the later ids move down; job 0 has Draw only; an entry without an
id, a template's, is deleted as it is); Draw runs `make graph GRAPH_ID=` / `FACE_ID=`; a job's or
facet's `id` is typed (an integer; Add a Job / Add a Facet give the next one); experiment_queue:
every queue, empty or not, is a line (its key, N CMD, **×** at the right end that deletes the queue and
its lines, after asking) with its make lines under it, each a row of menus; an empty queue shows one row
to fill, whose first choice stages the line (drawn as a row of menus too; OK writes it), and
**+** adds one more (a copy of the last); no "+ add" box. Every row ends with **×**: a written line's
removes it, a staged line's drops it, the row to fill's folds it away (the queue stays empty). QUEUES' Start is off for a queue without a line; inside an opened job or facet only OK, once one of its values is
changed; model_filter,
model_judge, test_config) are functional configs: each drives a `make`
target, and its output opens beside the panel.

Every line runs inside the `sca` conda env made by
`services/side_channel_core/infra/environment/setup_sca_env.sh`
(`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate sca`),
from the repository root, in the ENV the commands point at (see *Which ENV*), after
`sudo chown -R "$(id -u):$(id -g)" projects/`: the container writes there as
root, so a host-side writer (find_model, judge, graph, test bookkeeping)
would otherwise fail with `Permission denied`. The chown does not disturb a
running container (root writes regardless of owner); sudo may ask for your
password in that terminal. A form edit on a file the container left
root-owned tries the same with `sudo -n` (never a prompt) first.

A long run (tmux) keeps going when its terminal, the panel or VS Code
closes; its session stays open on a shell after the end so the output can
still be read (Attach), and the next Start replaces it. Before starting, the
panel asks, showing the experiments params.yaml plans (the count
`make experiment` logs), and offers `FORCE=1`. It refuses while:

- any `make orchestrate`, `make experiment` or `make test` runs on this
  machine, whoever started it (`make orchestrate` begins with
  `make stack-down`, which would stop it); the panel's top line names it;
- `tf-<project>` is still running something.

## New projects

On activation and on every save of `master.yaml`, the panel runs
`layout.py scaffold`: for every declared project whose entry is complete
(`name`, `root` as one folder name, `params`; a half-typed entry saved by
autosave is skipped and named in the panel's errors) it creates
`projects/<root>/config/` and copies there each blueprint file the project
lacks, of the eleven project templates (params, experiment_queue, single_runs, graph_config,
model_filter, model_judge, test_config, import_config, dev, staging, prod: every key the system reads,
empty, each with its menu in the panel; `paths` the standard layout; data entries, models,
queues and single runs empty, to add in the panel; no project's own notes:
`services/side_channel_core/config/blueprint/README.md`). The blueprint's older
`auto_search.yaml` and `model_config.yaml` are not copied: the config loader
reads a project's `model_config.yaml` whenever it exists. A file the project
has is never touched, and the blueprint is only read. `make scaffold ENV=dev`
does the same from a terminal.

## The same forms for every project

The config forms are one code for every project; what a new project sees differs from DLSCA_ASCAD's
only by its content, and the template gives it the same structure: every key the system reads, empty,
with its menu. Checked against DLSCA_ASCAD (2026-10-22) form by form: graph_config.yaml (a job's
true / false keys are true / false menus even while empty: `choices.BOOLEAN_KEYS`; `processes` is a
row per process with × and a menu to add one, in block or flow style, also while empty; the
template's text rules start as an empty list, each target set in the panel), model_filter.yaml,
model_judge.yaml and test_config.yaml draw alike; what DLSCA_ASCAD shows beyond that is its content
(its runs in the models' and test's menus, its second tie-breaker, its figures).

## A project master.yaml no longer declares

A folder under `projects/` with a `config/` or `artifacts/` that no `project:` entry of master.yaml
names (its entry removed, or never written) is listed after the projects, red, as
**<name> · master.yaml integration missing**, with what is left of it (project root, config,
artifacts: present or deleted) and **Add to master.yaml**, which writes the entry under that name
(as Master Control → Add a Project does); its config folder keeps what it has and the blueprint
fills the missing files. `layout.py` `orphan_projects`.

## Editing

A config form shows entries only, no comments. A value is an input (Enter or
leaving it saves); a list of values has one input per item, × to remove and
`+ add`; a block (a list item that is a mapping, or a mapping entry holding a
block, e.g. an `experiment_queue.yaml` key) has **OK**, which duplicates it (a
mapping entry asks for its new key). The panel deletes only what has a × that asks first (a model,
a data entry, a queue, a job, a facet); any other block is removed by hand in the file.
params.yaml's `paths` block (the standard run layout) is read only in the panel, with **Open in
editor** under its line: edit it in the file. `trainer.early_stopping` is a block (`enabled`, `monitor_metric`, `patience`, `min_delta`): its
`monitor_metric` menu lists the monitors of the file's `metrics_to_log` (early stopping watches it,
best-validation checkpointing picks by it). `sca.key_enumeration`'s line says *system works; mode
needs development* (its `mode` key carries the TODO in the file); `modes` shows only the block of the
chosen `mode` (`comparison`: every one). A data entry's `trace_normalization` is a union on its `mode` too
(`choices.normalization_forms`, from the loader's `NORMALIZATION_KEYS`): the chosen mode's keys are shown,
drafted when the entry lacks them and written on OK (`zscore`: mean, std, eps; `minmax`: min, max, eps;
`per_trace_minmax`: feature_range; `fitted`: steps, and feature_range once the steps include minmax); the
other modes' keys are not drawn, and Add a Dataset writes the mode alone. A data entry's `profiling_split` shows
`enabled` alone while it is false; switched to true, `test_size`, `shuffle` and `random_state` arrive staged with
their defaults (`choices.profiling_split_forms`: 0.1, true, 0; written on OK, editable before), and switched back
they go. `grad_clip_norm` sits under `deep_learning.pytorch` and
`deep_learning.keras` (the chosen framework's is read). `metrics_to_log` (every project alike) lists each metric
as a menu of every metric the system records, with × beside it, and under them one such menu, always
shown, with **+**: the metric chosen is added; one that takes a value (`top_k_accuracy`'s `k`,
`ge_at_traces`' `traces`, ...) shows its value box beside the menu, with its default, and **+** adds both. An addition not written yet is a row like the written ones (its menu or its name and
value, marked changed, × drops it), and a params.yaml list's count on its line is what the form holds
(additions in, removals out); OK writes them. model_judge.yaml's `process` fields are menus of the judge's
processes and its `metric` fields menus of the columns a run records (typing another name stays
possible). Its lists of blocks (data entries, models,
optimizers, metrics, layers; graph jobs and facets) always show their Add button, also while empty.

A form stages its changes; while any is staged, its top shows **N unsaved change(s)** with
**OK** (write them all to the file), **Cancel** (undo them and close the form) and **Reset**
(undo them; the form stays open). A value changed back to the file's (a key the file lacks:
to the value shown at first) is no change: once none is left, the bar goes.

**The blueprint in every form** (`choices.blueprint_forms`, from
`services/side_channel_core/config/blueprint/`: every key the system reads, empty; its README). A key
of the blueprint that the file lacks is drawn faded where it belongs, empty, with its menu (a block the
file lacks: its keys inside, marked *not in the file*; a data entry's, model's or optimizer's key:
from `blueprint/templates.yaml`; not the other framework's `deep_learning` block). Choosing a value
stages it; OK writes it. Every key's label and every block's and item's line names the key's path on hover
(no comment text: the file's comments are not shown as tooltips). A true / false key (`choices.BOOLEAN_KEYS`) is a true / false menu, empty or not. An empty
key is the code's default (`config_loader` drops it when the run loads). **Load
Blueprint** (beside Open in editor) stages, in one go, every key the file lacks, empty (`paths` with
its values; data entries, models and queues stay as they are: Add a ...); nothing is written until OK,
Cancel drops them. The scaffold (every master.yaml save) already writes the empty keys a project's
params, judge, test and filter files lack (`make_blueprint complete`, hash-safe), and says so once;
Complete is for the keys with a value. In params.yaml its note names the blocks of the staged keys that join the run hash
(`choices.HASHED`, the keys `utils/model_naming.make_experiment_hash` reads): after OK, a new run of
a model trained before gets a new hash.

An **Add** button that needs a name (Add a Project, a Queue, a Job, a Facet, a Model, a Dataset) asks it in
the panel, in a row under the button: the name's box (Add a Model: and the model's kind, a menu), **Add**
and ×. The row checks the name as one types (`webview/menus.js` `ASKS` / `askError`: its form, and that the
file has no such project, queue key, job, facet, model_id or data name yet), says why under the box, and Add
stays off until it can be used; Enter adds, Escape closes. The extension checks the name again before it
writes, and a refusal shows in the form.

Each change is one `yamledit.py edit`:

- it carries the SHA-256 of the text the form was drawn from; if the file
  changed since (by hand or elsewhere) nothing is written and the form is
  drawn again from the file: **the file wins**;
- only that entry's text is replaced; every other byte, comment and blank
  line stays;
- the new text is parsed again and written (atomically) only when its data
  equals the old data with exactly that change;
- a value is written plain when it reads back as typed, else quoted; `1e-5`
  where a number stood is written as the float `1.0e-05`; empty is null; a
  quoted value stays a quoted string;
- a value added to a single value makes a flow list (`seed: 42` + 43 → `seed: [42, 43]`, its comment kept);
- multi-line text blocks and flow mappings (`{a: 1}`) are edited in the file;
- the blueprint is never edited.

A hand edit in the editor redraws the open form at once (file watcher).

## Where it reads

| File | Written by | Read |
|---|---|---|
| `projects/<P>/artifacts/<env>/heartbeat.json` | `services/side_channel_core/utils/heartbeat.py`, every 5 s | every poll (2 s) |
| `projects/<P>/artifacts/run_registry.json` | the orchestrator | every poll, if its mtime changed |
| `projects/<P>/artifacts/<env>/generated_models/<name>/metrics/` | the trainer | tiles every poll; the detail tab while open |
| `projects/<P>/graphs/jobs/job_NNN_*/`, a one-run job's `artifacts/<env>/generated_models/<run>/graphs/job_NNN_*/` | `make graph` | every 10 s |
| config YAML, `master.yaml` | you | on change, through `layout.py export` |

`services/side_channel_core` is bind-mounted into the container, so every
trial process started after `heartbeat.py` is in the repository beats,
whatever the image. A run that started before shows grey with "no
heartbeat" (its registry row is still shown).

## Install

From a VS Code terminal in the repository:

```bash
make panel ENV=dev        # or: bash services/visualization/vscode_monitor/install.sh
```

`services/side_channel_core/infra/environment/setup_sca_env.sh` does this as
its last step only when the editor is VS Code (VS Code's `code` CLI is
reachable); the Monitor is for VS Code only, and other editors (Cursor,
VSCodium, …) are skipped.

then run **Developer: Reload Window**. Re-run after changing the extension.

### params.yaml forms

- `leakage_model` takes only what the system defines (ID, HW, MSB, any case;
  `services/side_channel_core/src/sca/leakage_models.py`): another value is
  refused before it is written, with a red "<value> not defined in system";
  one already in the file (typed by hand) is outlined red with the same line.
- A `data` entry's `*_range` is one `[lower, upper]` pair: two boxes, no add or
  remove. An empty or half range keeps both boxes; it is written once both are
  filled.

### graph_config.yaml forms

- `top_models` (job 0) is written by `make judge`: shown, not edited.
- The blocks graph_manager reads are drawn from its own keys (`choices.py graph_forms`), not
  only from what the file holds: a key the file lacks is shown with graph_manager's default
  (in italics) and typing there stages it; OK writes it (a new key or block in block style, a
  new text rule as a list item). × beside a key the file has removes it (the default again). A
  key graph_manager does not read, or one that moved, is shown struck through with where it
  went and × to delete it (`export.dpi` → `png.dpi`, `export.pdf`, a facet's `style`).
- **Add a Job** heads `jobs`: it asks the job's name, then `graph_export.py new-job` writes it at
  the next id with no models yet, `enabled: false` (make graph skips it) and `array_epoch: latest`;
  list its models, pick its source, metric and processes, then set `enabled: true`.
- `id` (a job's, a facet's) is shown, read-only: TraceFlow gives the ids (Add a Job, Export, Delete
  and the id shift keep them in order). The `metric` of a job Export or Overlay Graph made (its header
  says "Exported from the finished runs") is read-only too: it is the metric those runs recorded.
- `plot_type` offers only the types that change how the job's metric is drawn
  (`metric_router.plot_types_for`, by the job's `type`, `metric` and `array_epoch.mode`; the panel's
  copy agrees with it for every metric name, job type and mode, `test/graphforms.test.js`).
- `array_epoch`: a job without it, or without its `mode`, reads `mode: latest` (graph_job_validator's
  default): latest is the option selected;
  `epoch` is faded and unavailable unless `mode: selected`, `epochs` unless
  `overlay`, `summary` unless `summary`; the key the mode needs is offered when missing.
- `export`: `formats` is a row of tick boxes (one format at least); only a ticked format shows its own
  settings: png `dpi`, svg `sanitize`, `validate_xml`, `on_error`; pdf and eps "vector: no settings".
  A facet takes png, pdf and svg (its svg embeds the panels' PNGs). `margin`, `save_raw_plot_data` (jobs).
- `legend.handle`: a menu of the shapes (`line`, the Matplotlib markers; a facet also
  `patch`) and **Character…**, which asks for the character; beside it the thickness in pt.
  `legend.mode: auto` fades the `labels` (a facet: its `entries`), present but unavailable;
  `custom` makes them available.
  A facet's custom `entries`: a row per colour, its colour and the model its first panel's job draws
  in it engraved (`C0 · Adadelta`), then the entry's label; rows the file lacks are filled in order
  (an entry's place is its colour); **+** adds a row.
- `style.axes`: every key (`x_scale`, `y_scale`, `x_limits`, `y_limits`, `x_step`, `y_step`,
  `x_grid`, `x_grid_first`, `x_margin`, `y_margin`); limits, steps and `x_grid` are **Auto**
  (graph_manager's) or **Custom**, which shows the range (lower limit, upper limit) or the number.
  No × in `style`: a key is changed, not removed. `style.font.family` and a text rule's `font.family`
  are menus of the fonts a figure can use (serif, sans-serif, monospace, the Latin Modern fonts
  graph_manager registers, every font Matplotlib finds installed).
- `text_rules` (a job's `style.text_rules`, a facet's own `text_rules`): one panel per target,
  always there (set, or "not set: graph_manager's default"), with `text` (**Clear** empties it), `bold`
  and `italic` (True / False, False by default), `color` (MS Word's palette: Automatic = black, the Office
  theme colours with their lighter and darker shades, the standard colours, and More Colors…: Office's
  Colors honeycomb drawn in the panel, white to black below it, no system colour dialog; written
  `"#RRGGBB"`), `font.family`, `font.size` under it, none with ×, nor a target's line. `legend_text`'s
  text is `{model_name}`, shown: each legend entry is its model's label (`legend.labels`), else its name.
- `legend` (`enabled`, `position`, `mode`) and `export` (its common keys and each format's settings)
  have no × either: a key is changed, not removed (a stray key, one graph_manager no longer reads,
  keeps its × to delete it).
- A job's `processes`: **+** adds one (the processes of the job's source). No underline
  (Matplotlib cannot draw one). A second rule of a target, or a removed target, is flagged.
- **Add a Facet** heads `faceting` (also when the file has none): it asks the facet's name, then
  `graph_export.py new-facet` writes it at the next id with no panels yet (`enabled: false`), drawn in a
  row as listed until a layout is picked. **+ add a panel** under a facet's `panels` adds one at the next
  `file_id`; after OK, Find Figure fills its address. Then set `enabled: true`.
- A facet panel: `label` and `label_position` (bottom, bottom-left, bottom-right, top,
  top-left, top-right) are always offered.
- `legend.labels` has one row per model of the job's `models:` (job 0:
  `top_models`; an epoch overlay: `ep0001`, … from `array_epoch.epochs`), in
  that order, each with a box for the name the legend shows. Typing there
  writes `labels: {<model>: <label>}`; a model added to `models:` (in the panel
  or by hand) gets its row as soon as the file is saved. Labels of models no
  longer listed are shown struck through.
- An empty `models:` takes `+ add` (it becomes a list); a single name where a
  list belongs is flagged.
- **OK** (duplicate) sits on blocks at a file's top level or right under a list
  item (a job's `style`, `legend`, …), not deeper.

## Which ENV

The panel follows the ENV of what runs or has run, never a guess. The
`ENV=` lines of `experiment_queue.yaml` only say what a queue *would* run in,
so they do not count until that queue is started:

1. the running run's ENV: the freshest `artifacts/<env>/heartbeat.json` that
   says `running` (a beat older than 10 minutes is a crashed run's leftover);
2. else the running command's: the `ENV=` of a `make` (or the `--env` of the
   trainer) of this project running on the machine now;
3. else a registry row in progress;
4. else the latest snapshot's `env:` key (the newest
   `artifacts/<env>/generated_models/<name>/config/<name>_params.yaml`);
5. else the latest registry run's;
6. else none: the header shows no ENV until a run starts.

It reads that ENV's heartbeat and shows it next to the project, with where it
came from ("ENV dev, followed from the running command: make experiment
ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig26"). Buttons:

- a queue's **Start** runs in the ENV its own lines name (`make experiment`
  refuses a line naming another); a queue whose lines name different ENVs is
  refused until they agree;
- a single run's **Start** runs in the ENV its line names;
- Test, Draw, Find models, Judge and Terminate Run use the followed ENV; with
  no ENV followed they ask for one first;
- a snapshot's **Reproduce** runs in the ENV its snapshot names.

`traceflow.env`, when set, overrides all of this.

## Settings

| Setting | Default | |
|---|---|---|
| `traceflow.root` | workspace folder holding `master.yaml` | the repository |
| `traceflow.env` | empty | empty: follow the commands (see *Which ENV*); set only to force one ENV |
| `traceflow.python` | the `sca` conda env's python, else `python3` | runs `layout.py` |
| `traceflow.pollSeconds` | 2 | heartbeat and registry poll |

## Files

| File | Does |
|---|---|
| `layout.py` | Python: projects, paths, queues, graph jobs and configs as JSON; `scaffold` copies missing blueprint files |
| `state.js` | heartbeat and registry → light, progress, tiles, queue and figure states |
| `history.js` | `.npy` rank curves, JSONL histories, sparkline paths |
| `gpu.js` | `nvidia-smi` CSV → GPUs, rolling history |
| `extension.js`, `webview/` | VS Code glue and the drawing |
| `pack.py`, `install.sh` | `.vsix` with the standard library; `code --install-extension` |
| `yamledit.py` | Python: a config as a form tree; one span-preserving, hash-guarded, verified edit |
| `model_check.py` | Python: a model built as a run builds it, each layer's output shape (`walk`: layer by layer, PyTorch or Keras) |
| `trace_view.py` | Python: data entries, a model's DONE runs, one trace or a range through a design or a trained run, a run's recorded key recovery |
| `webview/planner*.js`, `planner.css` | the planner tab (Plan a Model, Visualise) |
| `make_blueprint.py`, `yamlspan.py` | builds an emptied copy of DLSCA_ASCAD's config (`--write` replaces the hand-written blueprint); spans shared with `yamledit.py` |

## Tests

```bash
NODE=$(ls -dt ~/.vscode-server/cli/servers/Stable-*/server/node | head -1)
$NODE --test services/visualization/vscode_monitor/test/*.test.js
PYTHONPATH="$PWD:$PWD/services/side_channel_core:$PWD/services/side_channel_core/src" \
  conda run -n sca python -m pytest services/side_channel_core/tests/unit/test_vscode_monitor_py.py \
  services/side_channel_core/tests/unit/test_yamledit.py services/side_channel_core/tests/unit/test_option_lists.py \
  services/side_channel_core/tests/unit/test_trace_view.py services/side_channel_core/tests/unit/test_status_share.py -q
```
