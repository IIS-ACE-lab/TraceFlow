# Tutorial · Training every dataset type

TraceFlow trains and attacks six database types: `ASCADf`, `ASCADr`, `CHES_CTF_2018`, `AES_RD`,
`DPAv4` and `AES_HD`. This tutorial shows, for each one, how its database gets onto the machine,
how to write its `data:` entry and the settings around it, which model trains on it, and how to
read the run it produces. Every setting below was run through the pipeline on 2026-10-18 (§8).

Each key is defined in the blueprint
[`params.yaml`](../side_channel_core/config/blueprint/params.yaml); the types themselves are in
[`src/sca/dataset_types.py`](../side_channel_core/src/sca/dataset_types.py) and the files in
[`data/README.md`](../side_channel_core/data/README.md).

## Contents

1. [The six dataset types at a glance](#1-the-six-dataset-types-at-a-glance)
2. [How a database reaches your machine](#2-how-a-database-reaches-your-machine)
3. [Writing a data entry](#3-writing-a-data-entry)
   - 3.1 [Keys](#31-keys)
   - 3.2 [Trace ranges](#32-trace-ranges)
   - 3.3 [Sample window and desynchronisation](#33-sample-window-and-desynchronisation)
   - 3.4 [Trace normalisation](#34-trace-normalisation)
4. [Settings outside the data entry](#4-settings-outside-the-data-entry)
   - 4.1 [`sca.attack_byte` is one setting for the whole file](#41-scaattack_byte-is-one-setting-for-the-whole-file)
   - 4.2 [Leakage model and how the attack scores a key](#42-leakage-model-and-how-the-attack-scores-a-key)
   - 4.3 [Cross-validation](#43-cross-validation)
   - 4.4 [The model must fit the trace length](#44-the-model-must-fit-the-trace-length)
   - 4.5 [Metrics a type refuses](#45-metrics-a-type-refuses)
5. [Running one dataset](#5-running-one-dataset)
6. [The datasets one by one](#6-the-datasets-one-by-one)
   - 6.1 [ASCADf](#61-ascadf)
   - 6.2 [ASCADr](#62-ascadr)
   - 6.3 [CHES_CTF_2018](#63-ches_ctf_2018)
   - 6.4 [AES_RD](#64-aes_rd)
   - 6.5 [DPAv4](#65-dpav4)
   - 6.6 [AES_HD](#66-aes_hd)
7. [Understanding a per-dataset experiment](#7-understanding-a-per-dataset-experiment)
   - 7.1 [The terminal summary](#71-the-terminal-summary)
   - 7.2 [The run folder](#72-the-run-folder)
   - 7.3 [Reading the numbers](#73-reading-the-numbers)
   - 7.4 [Which key the attack ranks](#74-which-key-the-attack-ranks)
   - 7.5 [Comparing datasets](#75-comparing-datasets)
8. [Verification record, 2026-10-18](#8-verification-record-2026-10-18)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. The six dataset types at a glance

| `type:` | Device, protection | Profiling / attack × samples | Label (`target`) | `sca.attack_byte` | Key of the attack set (attacked byte) |
|---|---|---|---|---|---|
| `ASCADf` | ATMega8515, Boolean masking | 50,000 / 10,000 × 700 | `Sbox[p ^ k]` (`sbox`) | 2 on `ASCAD.h5` and its desync files; any verified byte on the raw traces | `4dfbe0f27221fe10a78d4adc8e490469` (0xE0) |
| `ASCADr` | ATMega8515, Boolean masking | 200,000 / 100,000 × 1,400 | `Sbox[p ^ k]`, each profiling trace under its own key (`sbox`) | 2 on `ASCAD.h5` and its desync files; any verified byte on the raw traces | `00112233445566778899aabbccddeeff` (0x22) |
| `CHES_CTF_2018` | STM32F4, Boolean masking | 45,000 / 5,000 × 2,200 | `Sbox[p ^ k]` (`sbox`) | 0..15 | profiling `175cf299…`, attack `2eee5e79…` (byte 0: 0x17 / 0x2E) |
| `AES_RD` | ATmega16, random delays | 25,000 / 25,000 × 3,500 | `Sbox[p ^ k]` (`sbox`) | 0 only (Zaid et al. 2020's first byte) | `2b7e151628aed2a6abf7158809cf4f3c` (byte 0: 0x2B) |
| `DPAv4` | ATmega163, AES-256 RSM, mask known | 4,500 / 500 × 4,000 | `Sbox[p0 ^ k0] ^ mask` (`sbox_masked`) | 0 only | `8249ceb658c71d41…` (byte 0: 0x82) |
| `AES_HD` | FPGA, unprotected, last round | 50,000 / 25,000 × 1,250 | `InvSbox[c11 ^ k] ^ c7` (`inv_sbox_hd`) | 11 only | recorded as `00…00` (byte 11: 0x00, Zaid et al.'s convention) |

What the type decides for you, so there is nothing to set:

- `sca.variant`: every type is attacked as `fixed_key`. ASCADr has random profiling keys but one
  fixed attack key, and its labels take each trace's own key.
- Whether profiling and attack share the key. CHES_CTF_2018 does not, and the check that refuses
  different keys is lifted for it.
- The label: the `target` column. The attack scores every key guess with the same target.
- The bytes the label is defined for: DPAv4 only 0, AES_HD only 11.
- The bytes a stored trace window holds: ANSSI cut every ASCADf and ASCADr `ASCAD.h5` (and its
  desync files) around byte 2, so only byte 2 is accepted there; the raw traces take any byte
  whose window the entry verifies (§3.3).

The type joins the run hash only when it is not `ASCADf`, so ASCADf runs made before types
existed keep their hash.

## 2. How a database reaches your machine

You download nothing by hand. The first run that selects an entry fetches its database:

1. **Its submodule** on the owner's server (`/srv/git/<NAME>-data.git`), when the checkout has
   one and the server is reachable. `make orchestrate` does this on the host before the container
   starts (`make dataset-submodule`).
2. **Otherwise the official download** (data.gouv.fr, GitHub, Dropbox), by the run itself. This is
   the only route for a clone from GitHub.

Every file must match the SHA-256 its authors publish. `CHES_CTF_2018`, `AES_RD`, `DPAv4` and
`AES_HD` are then converted once into the ASCAD layout the runtime reads
(`src/sca/converters/<type>.py`); the converted file records its sources' SHA-256 and the
converter version. Each later run checks the file again. Any failure stops the run with
`status_reason: "DATA FAILED: ..."`.

| Type | Folder under `services/side_channel_core/data/` | What is fetched | Converted file |
|---|---|---|---|
| `ASCADf` | `ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/` | `ASCAD_databases/ASCAD.h5` (+ desync and raw files when used) | none |
| `ASCADr` | `ASCAD/ATMEGA_AES_v1/ATM_AES_v1_variable_key/ASCAD_data/` | `ASCAD_databases/ASCAD.h5` (438 MB) | none |
| `CHES_CTF_2018` | `CHES_CTF_2018/CHES_CTF_2018_data/` | `source/ches_ctf.h5` (442 MB) | `CHES_CTF_2018.h5` |
| `AES_RD` | `AES_RD/AES_RD_data/` | `source/ctraces_fm16x4_2.mat` (131 MB) | `AES_RD.h5` |
| `DPAv4` | `DPAv4/DPAv4_data/` | `source/DPAv4_dataset.zip` (28 MB) | `DPAv4.h5` |
| `AES_HD` | `AES_HD/AES_HD_data/` | `source/traces_{1..5}.csv`, `labels.csv`, `AES_HD_dataset.zip` (583 MB) | `AES_HD.h5` |

To fetch a database ahead of a run: `git submodule update --init <folder>`. Licences: AES_RD
is CC BY-NC 4.0 (non-commercial only). The original CHES CTF 2018 data is CC-BY-4.0; the
licence of AISyLab's copy is not stated. DPAv4 and AES_HD state no licence
([`data/README.md`](../side_channel_core/data/README.md)).

## 3. Writing a data entry

### 3.1 Keys

```yaml
data:
  - name: dpav4                 # free; DATASET=dpav4 selects it
    type: DPAv4                 # one of the six types
    database: "services/side_channel_core/data/DPAv4/DPAv4_data/DPAv4.h5"
    profiling_range: [0, 4500]
    attack_range:    [0, 500]
    validation_range: [0, 100]  # read when cross_validation is off
    test_range:      [100, 500]
    target_range:    [0, 4000]
    trace_normalization: {mode: fitted, steps: [standard]}
    profiling_desync: 0
    attack_desync: 0
    generate_data_out: true
    show_contract: true
    require_contract: true
    debug_h5_inspect: false
```

Required for every type: `database`, `profiling_range`, `attack_range`, `test_range` and
`target_range`. Optional: `validation_range`, `trace_normalization`, `profiling_desync`,
`attack_desync`, `trace_window_verified_by_attack_byte`, `target_range_by_attack_byte` (a window per
byte of a raw trace file), `generate_data_out`, `debug_h5_inspect`, `show_contract`, `require_contract`, the
operator switches `force_regenerate_dataset`, `force_regenerate_splits`, `split_chunk_size` and
`contract_verification_samples` (none of them joins the run hash), plus `name`, `type` and `alias`. An unknown key, a missing required key or an unknown type is refused when
the configuration loads, with the list of keys the type takes. An entry without `type` is
accepted only if its database lies in the ASCADf folder (old runs and snapshots); it is then ASCADf.

`profiling_split` (written in every entry: `enabled: true | false`) decides the profiling rows a run trains
on. `enabled: false` trains on every row of `profiling_range`, as an entry without the block does, and hashes
like one. `enabled: true` trains on the part scikit-learn's `train_test_split` draws from those rows, with
`test_size` (a fraction in (0, 1), rounded up, or a row count), `shuffle` and `random_state` all written, and
joins the run hash. The held-out part is not read (validation comes from the attack traces), and
`enabled: true` is refused with cross-validation, which partitions `profiling_range` itself. In the panel,
switching `enabled` to true writes the three with defaults you can change before OK: `test_size: 0.1` (a
90/10 split), `shuffle: true` (scikit-learn's own) and `random_state: 0`. `random_state: null` (or empty) shuffles
unseeded: a new cut on every run, which no rerun reproduces (Zaid et al.'s scripts cut this way; the run's own
`seed` does not seed it). Its rows are recorded in the run's data files and checked from there.

`profiling_split.random_state` may also be a list, e.g. `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`: the one list under a data
entry that sweeps, so one queue line makes a run per value (split *i* = `train_test_split(…, random_state=i)`), the
same splits for every model of the entry (Wouters et al.'s ten trainings, TCHES 2020(3) Sect. 3.2). A queue's lines
should share one count: the panel sizes a queue as its lines times one line's runs.

`seed: null` (written as `null`; an empty `seed:` is no setting) makes a run unseeded, as code that seeds nothing:
the run draws one seed from the operating system (weights, batch order, attack orders), records it as
`seed_drawn` in its `params_out` snapshot, and its test and `make reproduce` read that one back. The hash takes
`seed: null`, never the drawn seed, so a second run of the same configuration counts as DONE. Wouters et al.'s
90/10 split (TCHES 2020(3), Sect. 2.1; their code's `random_state=0`):

```yaml
    profiling_range: [0, 50000]
    profiling_split: {enabled: true, test_size: 0.1, shuffle: true, random_state: 0}   # 45,000 rows trained on
```

`database` is relative to the repository root. A registered file (§2) is the database of its own type
only, since the type fixes the label target: TraceFlow fetches it and checks its counts, and refuses it
under another type, as it refuses a download a type is converted from:

```text
data entry 'ascad' (DPAv4): database ASCAD.h5 is a registered ASCADf file, not DPAv4: declare type: ASCADf, or point database at a DPAv4 file (data/DPAv4/DPAv4_data/DPAv4.h5)
```

A file no type registers (your own traces) is taken as the type you declare, on your word: its
counts are not checked, and the run's `device.yaml` records it as `registered: false` with its SHA-256.

### 3.2 Trace ranges

| Key | Indexes | DPAv4 example |
|---|---|---|
| `profiling_range` | rows of `Profiling_traces` | `[0, 4500]`: all profiling traces |
| `attack_range` | rows of `Attack_traces` | `[0, 500]`: all attack traces |
| `validation_range` | the traces `attack_range` selects | `[0, 100]`: the validation split |
| `test_range` | the traces `attack_range` selects | `[100, 500]`: the test split |

With `cross_validation.enabled: false`, the validation and test splits are both cut from the
attack traces, so keep them disjoint. With cross-validation on, `validation_range` is ignored
and the folds come from the profiling traces (§4.3). When the configuration loads, the ranges
are checked against the counts of the registered file (`TRACE_COUNTS` in `dataset_types.py`),
for example:

```text
data entry 'dpav4' (DPAv4): profiling_range [0, 5000] is outside the 4,500 profiling traces of DPAv4.h5 (use 0 <= start < end <= 4500)
```

Maximum ranges per type: ASCADf `[0, 50000]` / `[0, 10000]`, ASCADr `[0, 200000]` / `[0, 100000]`,
CHES_CTF_2018 `[0, 45000]` / `[0, 5000]`, AES_RD `[0, 25000]` / `[0, 25000]`, DPAv4 `[0, 4500]` /
`[0, 500]`, AES_HD `[0, 50000]` / `[0, 25000]`.

### 3.3 Sample window and desynchronisation

`target_range` is the slice of samples the model sees. The full trace (`[0, 700]`, `[0, 1400]`,
`[0, 2200]`, `[0, 3500]`, `[0, 4000]`, `[0, 1250]`) is always accepted. A narrower window, or any
`profiling_desync` / `attack_desync` above 0, is refused until you vouch for it per byte:

```yaml
    trace_window_verified_by_attack_byte:
      2: true     # only once the window is known to hold leakage of byte 2
```

A desync shifts the window by up to N samples, so the window plus N must fit in the trace; a
full-trace window leaves no room. For the converted types, keep both desyncs at 0. AES_RD
already carries its countermeasure (random delays) in the traces.

### 3.4 Trace normalisation

| `mode` | What it does | Used here for |
|---|---|---|
| `none` | traces as stored | ASCADf (the paper feeds int8 traces as they are) |
| `fitted` + `steps: [standard]` | scikit-learn `StandardScaler` per sample point, fitted on the training traces only, stored as `data/scaler.json` and reused by the test | ASCADr, CHES_CTF_2018, AES_RD, DPAv4, AES_HD (§8) |
| `fitted` + `steps: [standard, minmax]`, `feature_range: [0, 1]` | standardise, then scale to the range | — |
| `zscore` / `minmax` | fixed `mean`/`std` or `min`/`max` you give | — |
| `per_trace_standard` / `per_trace_minmax` | each trace by its own statistics | — |
| `auto` | divides by the dtype's range (int8 /128, uint8 (x-128)/128) | — |

The stored dtypes differ: ASCADf/ASCADr int8, AES_RD/AES_HD uint8, CHES_CTF_2018/DPAv4 float32.
The CHES_CTF_2018 copy is already z-scored by its redistributor; `fitted` standardisation leaves
it almost unchanged.

## 4. Settings outside the data entry

### 4.1 `sca.attack_byte` follows the data entry

`sca.attack_byte` sits in the `sca:` block, not in the data entry. Left empty (the blueprint's `[]`),
a run takes the one byte the entry `DATASET` selects defines a target for, its type within its file's
window, written into the run's configuration before the sweep and the hash exactly as if you had set
it (owner ruling 2026-10-23: the source of the byte is the choice of the dataset). The panel writes
`type.attack_byte = {…}` beside each entry's type. So a file with ASCAD, DPAv4 and AES_HD entries
needs no byte set between queues:

| Dataset | byte a run takes | set `sca.attack_byte`? |
|---|---|---|
| ASCADf, ASCADr | 2 (`ASCAD.h5` and its desync files: their window was cut for it) | no |
| DPAv4 | 0 | no |
| AES_HD | 11 | no |
| AES_RD | 0 (Zaid et al. 2020's first byte) | no |
| CHES_CTF_2018, the raw ASCAD traces without a verified window | any of 0..15 | yes: the entry allows several, so an empty byte is refused by name |
| the raw ASCAD traces with `trace_window_verified_by_attack_byte: {2: true}` | 2 (the byte the window is declared verified for) | no |

A value set is kept and checked. A byte the type's label is not defined for is refused before any training:

```text
sca.attack_byte [2] is not defined for data entry 'dpav4' (type DPAv4): its label is Sbox[p[0] ^ k[0]] ^ mask: the stored mask is byte 0's. Set sca.attack_byte: [0]
```

So is a byte outside the window an ASCAD file was cut for:

```text
sca.attack_byte [0] lies outside the trace window of data entry 'ascad' (type ASCADf): ASCAD.h5 holds the 700 samples ANSSI cut around the first-round S-box of byte 2, so another byte's leakage is not in its traces. Set sca.attack_byte: [2], or attack byte 0 on the raw traces (ATMega8515_raw_traces.h5) with a target_range verified for it (trace_window_verified_by_attack_byte)
```

A list (`[0, 2]`) sweeps, and every value must be allowed for the type. The same holds for
`sca.leakage_model` and every `trainer:` list: they are global too, so a run's whole setting is
whatever `params.yaml` holds when its command starts.

### 4.2 Leakage model and how the attack scores a key

`sca.leakage_model` turns the target's intermediate into the class the model predicts: `ID`
(256 classes), `HW` (9) or `MSB` (2). Each target works with all three. The attack ranks the 256
guesses of the attacked key byte by the sum, over the attack traces, of each trace's score under
that guess's label:

- `ID` and `MSB`: the log-probability of the label (`score_rule: log_likelihood`); their classes
  are equally likely;
- `HW`: the log-probability minus the log prior of the class
  (`log_likelihood_minus_log_class_prior`). The HW classes are not equally likely
  (C(8, h)/256), and Bayes' rule gives log p(trace | class) = log p(class | trace) − log P(class) + c.

`metrics/key_recovery/test.json` records the rule as `target.score_rule`. An attack script that
sums the plain log-probabilities under HW gives slightly different ranks (§8 checked both).

### 4.3 Cross-validation

| Type | Folds | What the folds report |
|---|---|---|
| ASCADf, AES_RD, DPAv4, AES_HD | attacked | rank metrics and GE per fold, under the profiling key |
| CHES_CTF_2018 | attacked | rank metrics per fold, under the **profiling** key (0x17 for byte 0); the test attacks the attack key (0x2E) |
| ASCADr | **not attacked** | accuracy and loss only: each profiling trace has its own key, so a fold has no single key to rank |

ASCADr with cross-validation therefore needs `perform_attack.validation.attack_times: 0`, or the
run is refused:

```text
cross_validation with data entry 'ascad_r' (type ASCADr: a random key per profiling trace) needs perform_attack.validation.attack_times: 0 (it is 1): its folds have no single key to rank, so they report accuracy and loss only, and rank metrics and GE come from the test on the fixed-key attack set
```

The run then logs `Fold N has no usable final rank curve` for each fold. That is expected: the
rank and GE of an ASCADr run come from the refit model's test.

`OneCycleLR` (used by the Zaid et al. models below) takes its length from each training loop
when its `args` give none (no `total_steps`, `epochs` or `steps_per_epoch`): every fold, the
refit and a hold-out training run `trainer.epochs x` their own batches per epoch and finish the
cycle, so it works with and without cross-validation. A length written in `args` is one number
and is refused under cross-validation (folds and refit take different numbers of steps per
epoch); without cross-validation it must equal epochs × batches. `CyclicLR` counts its step sizes
in optimizer steps and is refused under cross-validation. Early stopping is refused with
`OneCycleLR` (it would cut the cycle). Keras has no one-cycle schedule: PyTorch only.

### 4.4 The model must fit the trace length

Trace lengths differ by type (700 to 4,000 samples), so a model's first `Linear` `in_features`
(MLP) or the flattened size (CNN) is dataset-specific. An ASCADf model does not run on ASCADr.
Give each dataset its own `model_id`, and write its `in_features` for its `target_range` length
and pooling (§6).

### 4.5 Metrics a type refuses

`LDD` and `Lm` are defined on `Sbox[p ^ k]` only. DPAv4 and AES_HD refuse them:
`LDD/Lm are defined for the first-round S-box target only, not 'sbox_masked'`. Every other
metric of `metrics_to_log` works on all six types.

## 5. Running one dataset

Panel: the project's **Run Single Experiment** → **Add a single Run**, pick the dataset (`data[].name`) and the model
(`model_id`) in its line, OK, then **Start**. The same from a terminal:

```bash
make orchestrate ENV=dev PROJECT=<project> DATASET=dpav4 MODEL_ID=Zaid_DPAv4
```

Several datasets in one queue (`config/experiment_queue.yaml`) only work when they share
`sca.attack_byte`, `sca.leakage_model` and the `trainer:` lists; otherwise make one queue per
dataset and set `params.yaml` before each (§4.1):

```yaml
dpav4_aes_rd:      # sca.attack_byte [0], sca.leakage_model [ID]
  - make orchestrate ENV=dev PROJECT=<project> DATASET=dpav4  MODEL_ID=Zaid_DPAv4
  - make orchestrate ENV=dev PROJECT=<project> DATASET=aes_rd MODEL_ID=Zaid_AES_RD
```

The first run of a dataset fetches and converts its database (seconds to two minutes, §2); later
runs reuse it.

## 6. The datasets one by one

Each section gives the data entry, the settings around it, a model that recovered the key in §8,
and what to expect. The models are written for `deep_learning.framework: pytorch`
(`weight_init: kaiming_uniform` with `nonlinearity: relu`, i.e. He uniform, and bias `zeros`).
Use PyTorch for them: the AES_HD and AES_RD trainings need `OneCycleLR`, which is a
`torch.optim.lr_scheduler` class; Keras has no built-in one-cycle schedule. Each was checked on
the CPU and in the container on the GPU (§8).

### 6.1 ASCADf

The reference: Benadjila et al.'s fixed-key ASCAD. Its entries (`ascad`, `ascad_desync50`,
`ascad_desync100`, `ascad_desync50_raw`, `ascad_desync100_raw`), models and queues are in
[`projects/DLSCA_ASCAD/config/params.yaml`](../../projects/DLSCA_ASCAD/config/params.yaml), and
its commands are in [09_dlsca_ascad_command_line.md](09_dlsca_ascad_command_line.md). Settings:
`sca.attack_byte: [2]`, `trace_normalization: none`, 10-fold cross-validation on the 50,000
profiling traces.

### 6.2 ASCADr

ASCAD v1 with a random key per profiling trace (200,000 traces of 1,400 samples) and one fixed
key in the 100,000 attack traces.

```yaml
  - name: ascad_r
    type: ASCADr
    database: "services/side_channel_core/data/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_variable_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
    profiling_range: [0, 200000]
    attack_range:    [0, 100000]
    validation_range: [0, 5000]
    test_range:      [5000, 25000]
    target_range:    [0, 1400]
    trace_normalization: {mode: fitted, steps: [standard]}
    profiling_desync: 0
    attack_desync: 0
```

Settings: `sca.attack_byte: [2]`, `sca.leakage_model: [HW]`; with cross-validation also
`perform_attack.validation.attack_times: 0` (§4.3).

Model and training that recovered the key (MLP, 4 × 200 ReLU, Adam 1e-4, batch 400, 30 epochs):

```yaml
  - model_id: "MLP_ASCADr"
    architecture: "MLP"
    architecture_args:
      strict_graph_args: true
      layers:
        - Flatten: {}
        - Linear: {in_features: 1400, out_features: 200, activation: "relu"}
        - Linear: {in_features: 200, out_features: 200, activation: "relu"}
        - Linear: {in_features: 200, out_features: 200, activation: "relu"}
        - Linear: {in_features: 200, out_features: 200, activation: "relu"}
        - Linear: {in_features: 200, out_features: "auto"}
```

Expect: key byte 0x22 at rank 1, GE 1 from about 2,700 traces (§8). With `ID` the same MLP
overfitted within 30 epochs (validation loss rising from epoch 1) and ended at rank 44: under ID,
ASCADr needs a stronger or regularised model.

### 6.3 CHES_CTF_2018

The CHES 2018 capture-the-flag traces (STM32F4, masked AES) as AISyLab redistribute them:
45,000 profiling traces under one key, 5,000 attack traces under another, 2,200 float32 samples.

```yaml
  - name: ches_ctf_2018
    type: CHES_CTF_2018
    database: "services/side_channel_core/data/CHES_CTF_2018/CHES_CTF_2018_data/CHES_CTF_2018.h5"
    profiling_range: [0, 45000]
    attack_range:    [0, 5000]
    validation_range: [0, 1000]
    test_range:      [1000, 5000]
    target_range:    [0, 2200]
    trace_normalization: {mode: fitted, steps: [standard]}
    profiling_desync: 0
    attack_desync: 0
```

Settings: `sca.attack_byte: [0]`, `sca.leakage_model: [HW]`. The model, `MLP_CHES`, is the ASCADr
MLP with `in_features: 2200`, trained with Adam 1e-4, batch 400, 30 epochs.

Expect: the hardest of the six. That MLP reached rank 1 for the attack key byte 0x2E only at the
full 4,000 test traces. It overfits from the first epochs (training accuracy 72%, validation loss
rising), and shorter or gentler training did worse (§8). The pipeline is verified on this
dataset; a model that recovers the key in fewer traces still has to be found.
Only 5,000 attack traces exist, so every trace given to `validation_range` is one less for the test.

### 6.4 AES_RD

Coron and Kizhvatov's AES with random delays (ATmega16): 50,000 traces of 3,500 uint8 samples,
split as Zaid et al. 2020 do (first 25,000 profiling, last 25,000 attack).

```yaml
  - name: aes_rd
    type: AES_RD
    database: "services/side_channel_core/data/AES_RD/AES_RD_data/AES_RD.h5"
    profiling_range: [0, 25000]
    attack_range:    [0, 25000]
    validation_range: [0, 5000]
    test_range:      [5000, 25000]
    target_range:    [0, 3500]
    trace_normalization: {mode: fitted, steps: [standard]}
    profiling_desync: 0
    attack_desync: 0
```

Settings: `sca.attack_byte: [0]`, `sca.leakage_model: [ID]`. Model: Zaid et al. 2020's AES_RD
CNN, trained with Adam, batch 50, 50 epochs and the one-cycle schedule:

```yaml
  - model_id: "Zaid_AES_RD"
    architecture: "CNN"
    architecture_args:
      strict_graph_args: true
      layers:
        - Conv1d: {in_channels: 1, out_channels: 8, kernel_size: 1, stride: 1, padding: 0, activation: "selu"}
        - BatchNorm1d: {num_features: 8}
        - Pooling: {dim: 1, type: "avg", kernel_size: 2, stride: 2}
        - Conv1d: {in_channels: 8, out_channels: 16, kernel_size: 50, stride: 1, padding: "same", activation: "selu"}
        - BatchNorm1d: {num_features: 16}
        - Pooling: {dim: 1, type: "avg", kernel_size: 50, stride: 50}
        - Conv1d: {in_channels: 16, out_channels: 32, kernel_size: 3, stride: 1, padding: 1, activation: "selu"}
        - BatchNorm1d: {num_features: 32}
        - Pooling: {dim: 1, type: "avg", kernel_size: 7, stride: 7}
        - Flatten: {}
        - Linear: {in_features: 160, out_features: 10, activation: "selu"}
        - Linear: {in_features: 10, out_features: 10, activation: "selu"}
        - Linear: {in_features: 10, out_features: "auto"}

trainer:
  epochs: [50]
  batch_size: [50]
  optimizer:
    - type: Adam
      learning_rate: [1.0e-3]
      weight_decay: 0.0
      scheduler:
        type: OneCycleLR       # torch.optim.lr_scheduler.OneCycleLR, stepped per batch
        args: {max_lr: 0.01, cycle_momentum: false}   # length: 50 epochs x 500 batches, from the loop
```

The length comes from the training loop (50 epochs × 500 batches of 50 = 25,000 steps, logged as
`OneCycleLR length of this training loop`). A `total_steps` written in `args` must equal it; a
wrong value is refused with the number it expects. Expect: key byte 0x2B at rank 1 from about 5 traces, matching Zaid
et al. Training takes about 8 minutes on the CPU.

### 6.5 DPAv4

The DPA contest v4 traces (ATmega163, AES-256 with rotating S-box masking) as Zaid et al. 2020
distribute them: 4,500 profiling and 500 attack traces of 4,000 samples, with each trace's mask
known. The label is the masked S-box output of byte 0, `Sbox[p0 ^ k0] ^ mask`. The converter
recovers the profiling masks from Zaid's labels and checks that each is one of the 16 RSM masks.

```yaml
  - name: dpav4
    type: DPAv4
    database: "services/side_channel_core/data/DPAv4/DPAv4_data/DPAv4.h5"
    profiling_range: [0, 4500]
    attack_range:    [0, 500]
    validation_range: [0, 100]
    test_range:      [100, 500]
    target_range:    [0, 4000]
    trace_normalization: {mode: fitted, steps: [standard]}
    profiling_desync: 0
    attack_desync: 0
```

Settings: `sca.attack_byte: [0]` (the only byte), `sca.leakage_model: [ID]`; no `LDD`/`Lm`.
Model: Zaid et al.'s DPAv4 CNN, Adam 1e-3, batch 50, 50 epochs (no schedule needed):

```yaml
  - model_id: "Zaid_DPAv4"
    architecture: "CNN"
    architecture_args:
      strict_graph_args: true
      layers:
        - Conv1d: {in_channels: 1, out_channels: 2, kernel_size: 1, stride: 1, padding: 0, activation: "selu"}
        - BatchNorm1d: {num_features: 2}
        - Pooling: {dim: 1, type: "avg", kernel_size: 2, stride: 2}
        - Flatten: {}
        - Linear: {in_features: 4000, out_features: 2, activation: "selu"}
        - Linear: {in_features: 2, out_features: "auto"}
```

Expect: key byte 0x82 at rank 1 within 4 traces in validation and GE 1 by 19 traces in the test.
The easiest of the six: the mask is given, so the leakage is first-order.

### 6.6 AES_HD

AES-128 on a Xilinx Virtex-5 FPGA (SASEBO-GII), unprotected, attacked in the last round:
the official 1,250-sample traces and labels, plus the two ciphertext bytes the label needs (c7, c11)
from Zaid et al.'s copy. The label is the register transition `InvSbox[c11 ^ k] ^ c7`. The
real last-round key is not published: Zaid et al.'s ciphertext arrays make key byte 0x00 the
correct guess, and TraceFlow records the key as zero bytes. Plaintexts are not published either;
the field is zero.

The two ciphertext bytes are not measured either: Zaid et al. re-encode the label as
`c11 = Sbox[i // 256]` and `c7 = label ^ (i // 256)` for row `i` (verified on every row of both
groups). `c11` therefore changes once every 256 traces (98 values in the 25,000 attack traces),
and inside each block every wrong key's hypothesis is the true label XOR one constant. ID and HW
still separate the keys, so Zaid et al.'s results reproduce, but the attack is not one on random
ciphertexts. TraceFlow refuses `sca.leakage_model: MSB` (128 keys stay tied inside a block) and
an attacked process with `shuffle: false` (the first traces of the curve would see one or two
`c11` values) on this type (`sca/dataset_types.py` `require_target_settings`).

```yaml
  - name: aes_hd
    type: AES_HD
    database: "services/side_channel_core/data/AES_HD/AES_HD_data/AES_HD.h5"
    profiling_range: [0, 50000]
    attack_range:    [0, 25000]
    validation_range: [0, 5000]
    test_range:      [5000, 25000]
    target_range:    [0, 1250]
    trace_normalization: {mode: fitted, steps: [standard]}
    profiling_desync: 0
    attack_desync: 0
```

Settings: `sca.attack_byte: [11]` (the only byte), `sca.leakage_model: [ID]`; no `LDD`/`Lm`.
Model: Zaid et al.'s AES_HD CNN (the DPAv4 one with `Linear: {in_features: 1250, ...}`), Adam,
batch 256, 20 epochs, and the one-cycle schedule:

```yaml
      scheduler:
        type: OneCycleLR
        args: {max_lr: 0.01, cycle_momentum: false}   # length from the loop: 20 epochs x 196 batches (50,000 / 256, rounded up)
```

Expect: key byte 0x00 at rank 1, stable from about 900 traces in validation and GE 1 by about
2,000 in the test (Zaid et al.: about 1,050). The schedule matters: with a constant learning rate
of 1e-3, the same model stayed at the random-guess loss (5.557 against ln 256 = 5.545) and ended
at rank 31 after 20,000 traces.

## 7. Understanding a per-dataset experiment

### 7.1 The terminal summary

Each run ends with a key-recovery block for the validation and test processes:

```text
TEST - FINAL EVALUATION
True key byte           : 0x22        # the attacked byte of the key of the traces attacked (§7.4)
Top key hypothesis      : 0x22        # the best-scoring guess after all test traces
AI Recovery status      : RECOVERED
Final true-key rank     : 1 / 256     # TraceFlow ranks from 1: rank 1 = recovered (paper rank 0)
Stable rank-1           : trace 2743  # the mean rank stays at 1 from here on
Final SR@1              : 1.0000      # share of the repeated attacks ending at rank 1
Final GE                : 1.0000      # mean final rank over the repeated attacks
```

`perform_attack.test.attack_times: 20` with `shuffle: true` repeats the test attack over 20 trace
orders; the curve is their mean. The final rank uses all test traces. With
`attack_trace_budget: 300` on the data entry, a Budget test (`make test ... BUDGET=1`) gives each of
the 20 attacks the first 300 traces of its own order: GE and SR are then read at 300 traces, and the
key-recovery result is still one attack on all test traces.

### 7.2 The run folder

`projects/<project>/artifacts/<env>/generated_models/<model_name>/`:

| Path | What it holds |
|---|---|
| `config/<model_name>_params.yaml` | the run's full configuration (snapshot), type included |
| `device.yaml` | machine, code commit and `run_device.database`: file, SHA-256, published SHA-256 or converter and source SHA-256 |
| `data/<dataset>.h5` | the selected traces with their labels; attribute `sca_label_target` for `sbox_masked`/`inv_sbox_hd` |
| `data/train.h5`, `val.h5`, `test.h5` | the splits, each row with `source_indices` back into the database |
| `data/scaler.json` | the fitted trace scaler (mode `fitted`), reused by the test |
| `metrics/attack_metrics/scalars/*_{train,validation,test}_history.jsonl` | one row per epoch: `metrics_to_log` and their `dependencies` |
| `metrics/attack_metrics/arrays/<process>/mean_attack_rank_curve/epNNNN.npy` | the rank curves, `[traces, mean rank]` |
| `metrics/key_recovery/test.json`, `validation_best.json`, `validation_history.jsonl` | the key-recovery reports of §7.1 |
| `cross_validation/` | folds, fold indices, the aggregate (when cross-validation is on) |
| `model/` | trained weights and the resolved model |
| `logs/` | `Training.log`, `Testing.log`, `all_logs.log` |

The registries (`artifacts/run_registry.json`, `test_registry.json`) list every run with its
hash, as for ASCADf. `make judge`, `make find_model` and `make graph` read these folders the same
way for every type.

### 7.3 Reading the numbers

| Quantity | Random guessing | Read it as |
|---|---|---|
| validation `cross_entropy`, ID | ln 256 = 5.545 | above it, the model learned nothing that generalises |
| validation `cross_entropy`, HW | 1.764 (entropy of the HW classes) | the same, for 9 classes |
| `mean_accuracy`, ID | 1/256 = 0.0039 | AES_RD reached 0.20, DPAv4 0.03; masked datasets stay near chance and still recover the key |
| `mean_accuracy`, HW | 0.273 (always predicting HW 4) | accuracy below 0.273 can still recover the key (ASCADr: 0.23) |
| `guessing_entropy` | about 128 | 1 = recovered with this many traces |
| `traces_to_ge_1` | empty | the trace count from which GE stays at 1; `dependencies.first_traces_to_ge_1` is the first crossing |

Rank and GE are the measures; accuracy and loss tell whether training generalises. A validation
loss that rises while the training loss falls is overfitting (ASCADr with ID, CHES_CTF_2018 in §8).

### 7.4 Which key the attack ranks

The attack takes the true key from the metadata of the traces it scores:

- the test: the attack set's key (CHES_CTF_2018: the attack key, not the profiling key);
- cross-validation folds: the profiling set's key (ASCADr's folds are not attacked);
- AES_HD: byte 11 of the recorded zero key, so "True key byte: 0x00".

`True key byte` in §7.1 and `key.true_key_hex` in `test.json` show which key was ranked.

### 7.5 Comparing datasets

Rank and GE curves of different types are directly comparable as "traces to recover one key
byte". Keep in mind what differs:

- the byte (2, 0 or 11) and the label (first-round S-box, masked S-box, last-round transition);
- the attack-set size, which caps the curve (DPAv4 500, CHES_CTF_2018 5,000);
- the leakage model: HW runs use the prior-corrected score (§4.2).

`test.json` records all three under `target` and `trace_population`. Graph jobs can overlay runs
of different types, but put the type and the byte in each label.

## 8. Verification record, 2026-10-18

Done in a separate git worktree (branch `worktree-dataset-check`, from dev
03d2aaf), in two passes:

1. **CPU** (`device: cpu`, 8 threads), through `core.cli orchestrate-cmd`: the orchestrator the
   container runs, started on the host with `ALLOW_HOST_RUN=1`. The databases came through
   `utils/dataset_fetch.ensure()` (official downloads, SHA-256 checked) and its converters.
2. **Container on the GPU** (RTX 5090, `device: cuda`, image `sidechannel:dev-778647a`), through
   `make orchestrate ENV=dev PROJECT=<test project> DATASET=<entry> MODEL_ID=<model>`, with the
   databases removed beforehand. DPAv4 and ASCADr came from their submodules through the fixed
   `fetch()`; AES_HD, AES_RD and CHES_CTF_2018 came from their submodules through
   `make dataset-submodule` on the host and were converted inside the container. No download
   fallback occurred in this pass.

**The data, checked with code independent of TraceFlow** (its own S-box built from GF(2⁸),
the `cryptography` library's AES):

| Type | Check | Result |
|---|---|---|
| ASCADr | stored labels = `Sbox[p2 ^ k2]` under each trace's key | 100%; 200,000 distinct profiling keys, one attack key |
| CHES_CTF_2018 | AES-128(key, plaintext) = ciphertext | 200/200 sampled traces in both sets; profiling and attack keys differ |
| AES_RD | CPA on `HW(Sbox[p0 ^ k])`, 50,000 traces | README key byte 0x2B ranks 1 |
| DPAv4 | CPA on `HW(Sbox[p0 ^ k] ^ mask)` | key ranks 1 in both sets (correlation 0.78 vs 0.08); 16 masks |
| AES_HD | CPA on `HW(InvSbox[c11 ^ k] ^ c7)`, 75,000 traces | key 0x00 ranks 1 (0.107 vs 0.017); only c7 and c11 non-zero |

**The runs.** For every run below, each split file (`<dataset>.h5`, `train.h5`, `val.h5`,
`test.h5`) was compared with the source database row by row through `source_indices`: traces,
metadata, and labels recomputed independently all match. The attack was also recomputed from the
saved weights with a network rebuilt in `torch.nn` and the stored scaler. Every rank and top guess
matched TraceFlow's; for HW, the match was with the prior-corrected score (§4.2).

| Dataset | Model, training | Validation | Test (20 orders) | Time |
|---|---|---|---|---|
| DPAv4 | Zaid_DPAv4, ID, Adam 1e-3, 50 ep, batch 50 | rank 1, stable from 4 traces | GE 1 by 19 of 400 traces | 40 s |
| AES_HD | Zaid_AES_HD, ID, constant Adam 1e-3, 20 ep | rank 88 | GE 31 at 20,000 | 47 s |
| AES_HD | same + OneCycleLR max 0.01 | rank 1, stable from 935 | GE 1 by 1,975 of 20,000 | 47 s |
| AES_RD | Zaid_AES_RD, ID, OneCycleLR max 0.01, 50 ep, batch 50 | rank 1 from 1 trace | GE 1 by 5 of 20,000 | 8 min |
| ASCADr | MLP_ASCADr, ID, Adam 1e-4, 30 ep, batch 400 | rank 22 | GE 44 at 10,000 | 1 min |
| ASCADr | same, HW | rank 1, stable from 1,486 | GE 1 by 2,743 of 20,000 | 1 min |
| CHES_CTF_2018 | MLP_CHES, HW, Adam 1e-4, 30 ep | rank 200 (1,000 traces) | GE 1 only at 4,000 of 4,000 | 30 s |
| CHES_CTF_2018 | 10 ep, Adam 5e-5, L2 1e-4 / 15 ep / 50 ep at 1e-5 (the last two: validation 500, test 4,500 traces) | rank 94 / 184 / 128 | GE 66 / 3 / 91 | 30 s each |
| ASCADr, 2-fold CV | HW, 3 ep, `validation.attack_times: 1` | refused at load (§4.3) | — | — |
| ASCADr, 2-fold CV | same, `attack_times: 0` | folds: accuracy and loss | refit test: rank 1 | 44 s |
| CHES_CTF_2018, 2-fold CV | HW, 3 ep | folds attacked (profiling key) | test ranks the attack key 0x2E | 29 s |

The GPU pass in the container, with the settings that recovered the key on the CPU (§6).
The same two independent checks were run on every run, and every check passed:

| Dataset | Validation | Test (20 orders) |
|---|---|---|
| DPAv4 | rank 1, stable from 4 traces | GE 1 by 19 of 400 |
| AES_HD (OneCycleLR) | rank 1, stable from 935 | GE 1 by 1,973 of 20,000 |
| AES_RD (OneCycleLR) | rank 1 from 1 trace | GE 1 by 5 of 20,000 |
| ASCADr (HW) | rank 1, stable from 273 | GE 1 by 1,274 of 10,000 |
| CHES_CTF_2018 (HW) | rank 166 (1,000 traces) | GE 1 only at 3,997 of 4,000 |

Not covered: `deep_learning.framework: keras` on the five new types (the models of §6 are PyTorch's);
ASCADr's desync and raw files (not registered, so their ranges are checked only when read); a
CHES_CTF_2018 model that recovers the key in fewer than 4,000 traces.

The check found one bug, fixed on the same branch: `utils/dataset_fetch.fetch()` created the
file's folder (`source/`, `ASCAD_databases/`) before trying the submodule. git then refused to clone
into the non-empty submodule folder, and every run fell back to the official download. The
folder is now made only for a download. Regression test:
`test_a_submodule_file_in_a_subfolder_comes_from_the_submodule`.

## 9. Troubleshooting

| Message (start) | Cause | Fix |
|---|---|---|
| `sca.attack_byte [..] is not defined for data entry` | the byte is not one the type's label exists for | DPAv4 `[0]`, AES_HD `[11]` (§4.1) |
| `sca.attack_byte [..] lies outside the trace window of data entry` | ASCADf/ASCADr `ASCAD.h5` (or a desync file) with a byte other than 2 | `[2]`, or that byte on the raw traces with a verified `target_range` (§3.3, §4.1) |
| `cross_validation with data entry .. (type ASCADr ..) needs perform_attack.validation.attack_times: 0` | ASCADr folds have no single key | set it to 0 (§4.3) |
| `data entry .. has no type; declare type:` | an entry outside the ASCADf folder without `type` | add `type:` |
| `unknown type ..; known types: [..]` | a misspelled type (names are case-sensitive) | one of the six names of §1 |
| `keys a <type> entry does not take: [..]` / `required keys missing: [..]` | a key outside §3.1 | the message lists the keys the type takes |
| `.. is outside the N profiling traces of ..` / `.. attack_range selects` / `.. samples per trace` | a range beyond the file | §3.2, §3.3 |
| `The selected trace window is not scientifically verified for attack byte N` | a narrower `target_range` or a desync | full window, or `trace_window_verified_by_attack_byte` (§3.3) |
| `sca.variant=.. disagrees with data entry ..` | an old `sca.variant` left in `params.yaml` | remove `sca.variant`: the type gives it |
| `DATA FAILED: ..` | download, SHA-256 or conversion failed, or the file was changed | read the reason; delete the file and run again to fetch or convert it anew |
| `Submodule .. not fetched (..); trying the official download` | the server is unreachable, or the submodule folder holds stray files from an earlier download | the download is checked the same way; `git submodule update --init <folder>` after emptying the folder restores the submodule route |
| `LDD/Lm are defined for the first-round S-box target only` | `LDD`/`Lm` on DPAv4 or AES_HD | remove them from `metrics_to_log` |
| `OneCycleLR.total_steps=.. but the loop performs ..` | `total_steps` ≠ epochs × batches | remove `total_steps` (the loop gives the length), or use the number in the message |
| a shape error in the first `Linear` | the model was written for another trace length | §4.4 |
| `Fold N has no usable final rank curve` (ASCADr) | expected: folds are not attacked | none |
