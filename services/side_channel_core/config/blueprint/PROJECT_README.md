# <PROJECT> — <the study: what is tested, under whose settings, with TraceFlow's protocol>

<One paragraph: what this project reproduces, critiques or establishes; what is the object under test
(the settings) and what is TraceFlow's (the protocol). Link the ledger of deviations if the project keeps
one: `Deep_Learning_Paper<N>.md`.>

**Nothing below is assumed.** Every setting is read from a paper, a script, a released model or a
dataset file, and the source is written next to it.

## Sources

| Source | Where | Used for |
|---|---|---|
| <authors, title, venue> | `paper/.../<file>.pdf` | the claims (sections, tables, figures) |
| <their code and released models> | <repository, commit> | architectures, settings, dataset files |

## The claims under test

| ID | Claim, as the paper states it | Tested how |
|---|---|---|
| **S1** | <quote with section and page> | <the runs that test it: queue · model · data entry> |

## The protocol (the same for every run)

| Setting | Value | Why |
|---|---|---|
| framework | `deep_learning.framework: pytorch` (or `keras`), with the authors' definitions set explicitly (initialisers, Adam ε, BatchNorm momentum / ε, precision) | <their framework and version> |
| cross-validation | `n_folds`, `strategy`, `shuffle`, `partition_seed`, refit on all profiling traces | <their split> |
| repeats | `seed: [a, b, c]`, at least `n_folds + 1` apart (the panel says so beside the list; the config check refuses closer seeds) | training randomness measured |
| attacks | `perform_attack.validation` and `.test`: `attack_times`, `shuffle`, `attack_distribution`; a Budget test's traces per attack: `data[].attack_trace_budget` | <their attack> |
| model tested | `evaluation.validation.checkpointing`, `evaluation.test.load_model` | <which epoch they attack> |
| metrics | `metrics_to_log`: <the entries every figure of this project needs; a metric not logged cannot be plotted afterwards> | <the papers' numbers> |
| preprocessing | `data[].trace_normalization` per entry (fitted scalers are fitted on the training rows of each fold and of the refit) | <theirs> |

**Deviations from their code** (each quantified in the ledger): D1 …

## Datasets (`params.yaml` `data`)

One entry per (dataset, preprocessing). `validation_range` is not read with cross-validation; `test_range`
is the whole attack set.

| Dataset | File | Profiling / attack × samples | Byte | Entries (preprocessing) |
|---|---|---|---|---|
| <name> | `<database>` | <counts> | <the byte the type defines a target for> | `<entry names>` |

## Models (`params.yaml` `model`)

| `model_id` | Source | Layers (filters × length, pooling; dense) | Published | Trainable |
|---|---|---|---|---|
| `<model_id>` | <paper section / table / code line> | <layers> | <their count> | <`trainable_parameters`> |

## Training settings: set before each queue

> **Set before each queue.** The queue runner reads `params.yaml` as it is when a queue starts, and the
> epochs, the batch size, the learning rate and the schedule are one value per file. Set the row's values
> (panel: Configs → params.yaml), then Start. `sca.attack_byte` left empty follows the data entry a run is
> started on (the panel writes `type.attack_byte = {…}` beside each entry's type); set it only for a type
> that allows several bytes, or to sweep. **A byte set by hand is checked; wrong epochs, batch size or
> learning rate are not refused: those run as a different experiment under a different hash.** The figure index below repeats each row's values
> beside its figure, so a figure is reproduced from its own row alone.

| Queue | byte (automatic, from the entry) | `trainer.epochs` | `trainer.batch_size` | `learning_rate` | `scheduler` | Evidence |
|---|---|---|---|---|---|---|
| `<queue>` | <byte> (automatic) | `[<n>]` | `[<n>]` | `[<lr>]` | <none, or the schedule and its arguments> | <the script line or released model that gives it> |

## Figure index: model, dataset, leakage model, batch size, epochs, optimizer

Which runs produce which figure, one row per (figure, model, dataset), so that a figure is reproduced from
its row alone: set the row's values in `params.yaml`, then start the Queue. **Bold** marks a value that
differs from the file's default setting. The Graph column names the `graph_config.yaml` job that draws
the figure once it is written.

**The same for every figure** (set once in `params.yaml`, not repeated per row): <framework and its
definitions, the loss, cross-validation, `sca.leakage_model`, `perform_attack`, the rank grid,
`metrics_to_log`>.

| Axis in the paper | `metrics_to_log` | Graph job `metric` | `source` / `processes` | What is plotted |
|---|---|---|---|---|
| mean rank against traces | `mean_attack_rank_curve` | `mean_attack_rank_curve` | `cross_validation` / `validation` | the rank curve of every fold model on its held-out traces, mean over the folds |

### CNN figures

<table>
<tr><th>Fig.</th><th>What varies</th><th><code>MODEL_ID</code></th><th>Dataset (<code>DATASET</code>)</th><th>Leakage model<br><code>sca.leakage_model</code></th><th>Byte<br>(the entry's own: <code>sca.attack_byte</code> empty)</th><th>Batch<br><code>trainer.batch_size</code></th><th>Epochs<br><code>trainer.epochs</code></th><th>Optimizer (learning rate)<br><code>trainer.optimizer</code></th><th>Traces drawn</th><th>Rank grid<br><code>validation.rank_step</code> = <code>perform_attack.attack_step</code></th><th>Metrics (<code>metrics_to_log</code>)</th><th>Plot from<br><code>source</code> / <code>processes</code></th><th>Graph:Job ID #</th><th>Study</th><th>Queue<br>(panel: Queues → Start)</th></tr>
<tr><td>F1</td><td>&lt;what the figure varies&gt;</td><td><code>&lt;model_id&gt;</code></td><td><code>&lt;entry&gt;</code></td><td><code>ID</code></td><td>2 (automatic)</td><td>50</td><td>50</td><td>Adam (5.0e-3), OneCycleLR max 5.0e-3</td><td>0–5,000 (the fold's held-out traces)</td><td>1</td><td><code>mean_attack_rank_curve</code></td><td><code>cross_validation / validation</code></td><td>to write</td><td>S1</td><td><code>&lt;queue&gt;</code><br><code>#configure params.yaml correctly before starting Queue run</code></td></tr>
</table>

### MLP figures

<table>
<tr><th>Fig.</th><th>What varies</th><th><code>MODEL_ID</code></th><th>Dataset (<code>DATASET</code>)</th><th>Leakage model<br><code>sca.leakage_model</code></th><th>Byte<br>(the entry's own: <code>sca.attack_byte</code> empty)</th><th>Batch<br><code>trainer.batch_size</code></th><th>Epochs<br><code>trainer.epochs</code></th><th>Optimizer (learning rate)<br><code>trainer.optimizer</code></th><th>Traces drawn</th><th>Rank grid<br><code>validation.rank_step</code> = <code>perform_attack.attack_step</code></th><th>Metrics (<code>metrics_to_log</code>)</th><th>Plot from<br><code>source</code> / <code>processes</code></th><th>Graph:Job ID #</th><th>Study</th><th>Queue<br>(panel: Queues → Start)</th></tr>
<tr><td>F1</td><td>&lt;what the figure varies&gt;</td><td><code>&lt;model_id&gt;</code></td><td><code>&lt;entry&gt;</code></td><td><code>ID</code></td><td>2 (automatic)</td><td>100</td><td>200</td><td>RMSprop (1e-5)</td><td>0–5,000</td><td>1</td><td><code>mean_attack_rank_curve</code></td><td><code>cross_validation / validation</code></td><td>to write</td><td>S1</td><td><code>&lt;queue&gt;</code><br><code>#configure params.yaml correctly before starting Queue run</code></td></tr>
</table>

## Queues and counts

`config/experiment_queue.yaml`; counts per seed. Every run is `n_folds` folds plus a refit and one test.
A run two queues share (the same hash) is trained once.

| Queue | Study | Configurations | Note |
|---|---|---|---|
| `<queue>` | S1 | <n> | |

Run order: <the smallest first, as the pilot>. `sca.attack_byte` changes between dataset families: never
mix them in one queue.

## Open decisions (owner)

1. <what is still to be decided before the runs start>
