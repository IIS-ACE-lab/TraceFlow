# services/graph_manager

Job-driven plotting service. Reads a model's recorded metrics and turns the
jobs declared in `projects/<P>/config/graph_config.yaml` into paper-ready
figures (PNG, SVG, PDF, EPS).

## Usage

```bash
make graph ENV=dev PROJECT=DLSCA_ASCAD GRAPH_ID=1          # one job
```

Runs on the host in the `sca` conda environment. Output (`job_paths.py`): a job of several runs
draws into `projects/<P>/graphs/jobs/job_<id>_<name>/`; a job of one run (its `models`, job 0's
`top_models`) into that run's `paths.model_graphs` folder,
`artifacts/<env>/generated_models/<run>/graphs/job_<id>_<name>/`, unless the job says `folder: project`
(then `graphs/jobs/` like the others, where git keeps it: the P2 jobs of `report_jobs.py`, whose graphs the paper
links one by one; `folder: run` is the default). A facet panel's address is
relative to `graphs/`, or starts with `artifacts/` for a run's figure. `graph_export.py` moves a
job's folder when its run count crosses one, and every folder it moves or renames (the id shift,
`_deleted/`) has its `sources.json` re-signed at the new path if its seal held at the old one.
A folder that already exists is never written over: a new job (Export), or a job moving to a new id
or place, whose `job_<id>_<name>/` is there (left by a deleted job) takes the next name, the way a
file manager names a copy (`acc`, `acc_1`, `acc_2`: `name` and `output_file_name` in
`graph_config.yaml`), and a folder set aside twice on one day becomes `_deleted/<folder>_<date>_1`.
A job added by hand with a deleted job's id and name draws into that job's folder.

Studies (`graph_export.py study-plan` / `study`, the panel's QUEUES → Overlay Graph): the runs of a
finished study (its queues all DONE; `fig14_*` is one study) as graph jobs. Its axes are the
settings that differ between its runs, less those that follow from another axis (the output
layer from `sca.leakage_model`) and those some runs never set (`sca.variant`); the axes ticked as
the legend are a figure's lines, each value of the others is a figure (fig26: one figure per
leakage model, seven epoch lines each), one job per figure and metric, every job reading each
run's last epoch (`array_epoch: latest`, also for Export). A relative fit time divides each figure's runs
by the reference line you choose (`relative: {rule: run, where: [{path: <legend axis>, equals: <value>}]}`,
bars, styled like the latest relative job of that metric). A new job is styled like the latest job of
its metric (a `metric:` list counts). A job already drawing exactly those runs
for that metric is kept. Runs that differ only by their split (`profiling_split.random_state` as a list, a run per split) are repeats of one line,
never lines of their own: a curve job pools them (`groups`: the runs thin, their mean bold, and the band you are asked
for, `min_max`, `std` or `none`, since none is assumed), and a per-epoch scalar draws a bar per run labelled with its split.
When TraceFlow cannot make the jobs (two runs a figure cannot tell apart,
a metric some run lacks), it fails with one message: do it yourself in `graph_config.yaml`.
A run's folder written by the training container is root's: the panel's buttons take `projects/`
back first; from a terminal, `sudo chown -R "$(id -u):$(id -g)" projects/<P>/artifacts`. Every run also prints a colored
"What you can choose, from your current data" table of the metrics each
model recorded, and a red/green table when a job names a model with no
artifacts.

## How a job is resolved

```text
graph_config.yaml job ──> graph_job_validator ──> metric_loader ──> plotters / attack_plotters ──> figures
   (type, source,          (schema, sources,       (source-aware     (line, band, bars,
    metric, models,          processes, model        loading, fold     fold curves, key
    processes, style)        lists)                  aggregation)      scores) + svg_sanitizer
```

Beside its figures, every job writes a signed `sources.json` (`sources.py`): the
runs the figures were drawn from (name, hash, env, status, folder, the key
their `device.yaml` declares), every file of a run the job read with the seal
state it had when read (`utils/seal.py recording()`; an unsealed file adds its
SHA-256), the registries read to bind the models, the figures written with
their SHA-256, the job as it ran and the code commit. A facet's `sources.json`
lists its panel figures with their SHA-256 and, for each, its graph job's
`sources.json` (SHA-256, seal state) and that job's runs. `make verify` checks
the signature and that every listed figure still matches its SHA-256.

| Job field | Values |
|---|---|
| `type` | `solo` (one figure per model), `model_overlay` (models in one figure), `process_overlay` (train/validation/test in one figure), `model_explanation` (what a trained model looks at, below) |
| `source` | `attack_metrics` (train, validation, test), `cross_validation` (fold mean ± std; train, validation), `key_recovery` (validation, test). `cross_validation` is a source, not a process: the paper's curves are `source: cross_validation` with `processes: [validation]` |
| `metric` | any recorded history column; `mean_accuracy` and `mean_fit_time_s`, `_h`, `_per_epoch` (`attack_metrics`: the value of that run; `cross_validation`: the mean over the folds); curves `mean_attack_rank_curve`, `per_attack_curves`, `final_ranks`; solo-only `fold_rank_curves`, `key_candidate_scores` |
| `array_epoch.mode` | `latest`, `selected`, `overlay`, `summary`; `selected` on a per-epoch scalar keeps that epoch, so a model overlay becomes one bar per model; with `plot_type: barh`, `latest` keeps the last epoch of each model. The bar axis has no tick labels; with `legend.enabled` the bars are named in a legend at `legend.position`, one entry per bar |
| `relative` | `model_overlay` only. Empty or absent: values as recorded. `{rule, where, label}`: every scalar series divided by the same metric of one reference run, read as the job reads its models, so the reference is 1.0 (dashed line). `rule: run` takes the reference among the job's models, `rule: anchor` among the finished runs of the ENV (listed or not); `where` is a list of model_filter conditions that must match exactly one run; `label` (default built from `where`) fills `{relative_to}` in `text_rules`; the log names the run and hash. `rule: probe` (no `where`) divides each run's time by its own device probe from `<run>/device.yaml (paths.device)`, in probe steps, comparable across devices; time metrics only. With `run` or `anchor`, a time metric is divided only between runs whose device records agree (GPU, torch, CUDA, cuDNN, TF32 switches); runs without a record are named in a warning. `where` can also select by device: `{path: run_device.name, equals: "NVIDIA GeForce RTX 5090"}` |
| `legend.position` | `top-right`, `top-left`, `bottom-right`, `bottom-left` (also `upper-`/`lower-`), `center`, `best`, `top` / `bottom` (inside the box at its top / bottom, centred), or `below`: outside the axes just under the x label (0.1 in), without a frame, where it covers no curve, bar or value; a legend inside the box that is wider than the box is made again with shorter marks and narrower gaps, then, if still wider, at the size that fits (text, marks and spacing together, not below 6 pt); `legend.columns` lays the entries out in that many per row (default 1); `legend.anchor: [x, y]` puts the legend's `position` point at that place of the plot box (fractions, 0 0 lower left), a legend placed inside exactly; a heat map's grey-cell legend follows `legend.enabled` (a row's heat maps can share one) |
| `style.bar_values` | a number format (`".4f"`, `".1%"`, `",.0f"`): each bar's value printed just past its end (Fig. 20 b's accuracies); empty: none |
| `style.series`, `legend.entries` | `style.series: {<model_name>: {color, hatch, width, dashes}}` (`<model_name>:<metric>` in a job of several metrics): a run's own colour (any Matplotlib colour, e.g. `C0`) instead of its slot's, the hatch of its bars (`""` solid, `////`, `....`, ...), its line's width in pt and its dash pattern `[on, off]` in pt (Fig. 26: HW `[6, 3]` at 1.3 pt, ID solid at 0.7 pt). `legend.entries: [{label, color, hatch, linestyle, width, dashes}]`: the legend's own entries (patches, colour default grey; with `linestyle` or `dashes` a line of that style and width) in place of one per series. `style.series` colours curves too, and `style.line.styles` reaches the curves of a model overlay; a heatmap's row labels take `style.series` colours. Together they put two settings of each model in one bar panel, e.g. Fig. 10's accuracy at N_max 0 (solid) and 100 (hatched) in the activation's colour, the legend naming the hatches |
| `metric_process` | `model_overlay` of several per-epoch scalars: `{metric: process}`, the process a metric is read from instead of the job's (`processes` then lists one, the process of the metrics the map does not name), so one figure holds e.g. the validation accuracy beside the training fit time (`{mean_fit_time_{s}: train}`). Read the same way for the `relative` reference. The figure is filed under the processes it reads, joined (`model_overlay/validation_train/`). With several metrics a model keeps its colour (the one it has in the job's other figures) and a bar its metric's hatch (solid, `////`, `....`, `xxxx` in the job's metric order) |
| `groups` | `model_overlay` of a curve (e.g. `mean_attack_rank_curve`, `source: attack_metrics`), of per-epoch scalars (e.g. `[training_objective, validation_objective]`: one series per group and metric, in its first run's `style.series` colour of that metric, the runs sharing their epochs), or of the key-recovery margin with `plot_type: strip` (a row of dots per group). A list of `{label, models, band, members, color, band_color}`: the runs of one configuration pooled into one series (Wouters et al., TCHES 2020(3), Figs. 1–2: ten trainings of each model and preprocessing). A group draws in `color` (any Matplotlib colour; default its slot of the colour cycle) and its band in `band_color` (default `color`). Each member run's curve thin (not with `members: false`: the mean and band only, as the Model Performance Report draws the loss), their mean bold, and the band the group names shaded at `style.std_band.alpha`: `min_max` (min to max of the runs at each number of traces), `std` (mean ± the sample std, kept within the runs' min to max: a GE band never dips below rank 1), or `none`. There is no default: Wouters never says which band he drew (ledger P9), so the job states one, and the legend says it ("<label> (mean of 10 runs; band: min to max)"). `models` are runs of the job's `models`, each in one group; they must share their x values (the same test traces). A model of the job in no group is drawn as usual, after the groups. |
| `marks` | `process_overlay` of `mean_attack_rank_curve` with `processes: [validation, test]` (`source: attack_metrics`). Any of `validation_final` (the validation curve's last point), `test_at_validation` (the test curve at the validation's trace count: the like-for-like value; a rank falls as traces are added, so only these two compare) and `test_final` (the test curve's last point): the three ranks of the TraceFlow Monitor's Done list, in its colours (● yellow-orange, ◆ light orange, ■ orange-red), each in the legend with its rank and trace count ("test at the same traces: 107.8 @ 2,500"), and a dotted line at the validation's trace count. Read from the stored curves; a mark whose curve is missing is left out. A cross-validated run keeps its validation under `cross_validation`, which a job reading `attack_metrics` does not see. |
| `overlays` | `model_overlay` of `mean_attack_rank_curve` (`source: attack_metrics`, `processes` validation and/or test, `array_epoch.mode` latest or selected): marks drawn over the GE curves, each read from the same record as its run's curve (the job's process, its `test` and `trace_budget`, its epoch). `ge_averages: [geometric, median]` (either or both): from the record's `per_attack_curves` (attacks x traces x [trace count, rank]), the geometric mean over the attacks, exp(mean(log(max(rank, 1)))), dashed, and the median over the attacks, dash-dotted (Wu, Perin, Picek, COSADE 2022 recommend the median), per trace count, in the run's colour; with `groups`, the mean over the group's runs in the group's colour. `ntge: true`: each run's N̄tGE (the last row's `per_attack_traces_to_ge_1` of the process's history) as a dot at GE 1 in its run's (group's) colour, a run never stably at rank 1 (no value) as `>` at the x axis' right end, and the median of the plotted N̄tGE as a red diamond (`#E53935`), one per group (and one for the runs in no group) with `groups`, else one for all the job's runs; not with `trace_budget` (the recorded N̄tGE is of the whole attack: draw `test: budget`). `trace_lines: [n_val, budget]` (either or both): a vertical line at the run's number of validation traces (its snapshot's `data.validation_range` length, else the last trace count of its validation curve), dotted `#9a4a00`, and at its data entry's `attack_trace_budget` (none: no line), dashed `#c00000`, the TraceFlow Monitor's rank-key colours; each distinct count drawn once. `colors: {geometric, median, ntge}` (any of them): that mark's own colour instead of its run's or group's (the Model Performance Report's: `#00C853`, `#FF00FF`, `#222222`); the N̄tGE median stays red. One legend entry per kind of mark, after the runs' ("geometric mean over the attacks", "median over the attacks", "each training's N̄tGE", "their median", "never stably at rank 1", "N_val: validation traces", "N_budget: traces per Budget-test attack"). A run without the record an overlay reads is left out of that overlay with a warning. Empty or absent: none |
| `plot_type` | only a type that changes how the job's metric is drawn (`metric_router.plot_types_for`; another is refused, the panel offers only these): per-epoch scalars `auto`, `line`, `line_scatter`, `scatter`, `step`, `bar` (overlays also `barh`; `strip` for `top1_top2_score_margin` of `source: key_recovery`, process `test` (each run's `metrics/key_recovery/test.json` of the test the job's `test` chooses): a dot per run at x = log10(1 + the best key's score margin), teal `#00897B` when its best key hypothesis (`key.top_key_hypothesis`) is the true key (`key.true_key_byte`), red `#E53935` when not, grey when the record names neither; with `groups` one row per group named by its label (then a row "other runs" of the runs in no group), else one row of all the job's runs; x label "log(1 + score margin)" unless a text rule names one; the Model Performance Report's key confidence; `heatmap` for a `key_recovery` scalar of process `validation`: a model per row, a cross-validation fold per column, each cell that fold's report at the job's epoch on the `style.colormap` scale (any Matplotlib colormap, default `viridis`; Fig. 20 c uses `turbo`), a fold without a value light grey with its final key rank alone, named by a legend under the axes; every cell's text at one size that fits its cell; a colour-bar label longer than its bar on up to four lines; no colour bar when no cell has a value; `style.heatmap: {limits: [low, high], colorbar, cell_values, row_groups}`: one colour scale for several heat maps (default 0 to the largest value), `colorbar: false` none of its own (a facet's `legend.colormap` draws one for all), the cells' number format (default `,.0f`), `row_groups: true` row names `group, item` drawn as the items with a bracket and the group's name beside them; the row names take the `y_label` rule's weight); attack curves in an overlay (or a solo job with `array_epoch.mode: overlay`) `auto`, `line`, `line_scatter`, `scatter`, `step`; a solo attack curve, `final_ranks`, `LDD`, `fold_rank_curves`, `key_candidate_scores` and `model_explanation` only `auto` (their drawers do not read it). `histogram` and `distribution` reach no drawer. Bars of a model overlay take the colour cycle in model order, like its curves |
| `style.figure` | `orientation` `landscape` or `portrait`, optional `width` and `height` in inches, and `margins: {top, bottom, left, right}` in inches: the plot box at those margins and the page kept at its exact size (not trimmed to the drawing), a colour bar in the right margin; panels of one height with the same top and bottom margins align in a facet row |
| `legend.handle` | `[shape, thickness]` of the mark before each legend label, in the series' colour: `"line"` (width in pt), a Matplotlib marker such as `o`, `s`, `^`, `D` or any single character such as `■` (size in pt). Left out: Matplotlib's default |
| `style.std_band` | `enabled` (default `true`): with `source: cross_validation`, load the sample std over the folds and shade mean ± std in the line's colour (error bars on bars); `false` loads the mean only, as the paper's figures plot it. `alpha` (default 0.15) is the band's opacity |
| `export` | `formats` (`png`, `svg`, `pdf`, `eps`; default `svg` only, its text as outlined Latin Modern glyphs), each format's own settings under its name: `png: {dpi}` (default 300; no other format uses a dpi), `svg: {sanitize, validate_xml, on_error}`; pdf and eps have none (vector). `margin`: white border in inches (default 0.05), the saved page of every format is the drawn figure plus it. `export.dpi` (now `export.png.dpi`) and `export.pdf` (its `embed_fonts` / `vector` were never read) are refused |
| `style.text_rules` | one rule per target (`title`, `x_label`, `y_label`, `legend_text`, `colorbar_label`: a heat map's colour-bar label, `""` none; the first rule of a target is used): `text`, `bold`, `italic`, `color`, `font: {family, size}`. No `underline`: Matplotlib cannot underline a text (refused). The targets `legend` (= `legend_text`) and `plot_labels` (styled nothing) are refused |
| `trace_budget`, `test` | `source: attack_metrics` with `test` in `processes`. `trace_budget: B` cuts the attack curves at B test traces and reads the per-attack ranks there (empty: as recorded). `test` names the test the `test` process is drawn from: `latest` (also when empty) the latest All test (the training's own or a later Retest All / Test All), `budget` the latest Budget test, `recorded` the training's own, an integer `<index>` the run's test `<model_hash>_et_<index>` ([`trace_budget.py`](trace_budget.py), [`utils/latest_test.py`](../side_channel_core/utils/latest_test.py)). A test that did not finish is never read; a later test of a kind replaces an earlier one in every figure |
| `processes` | `train`, `validation`, `test` as the source allows, and `- combine: {processes: [...], mode: sum}` items: the metric of several processes added or averaged into one series, inside every fold first for `cross_validation`. Every item is drawn |
| `faceting` (top level, after `jobs`) | finished graph files placed side by side: `panels` (`file_id`, `address` relative to `<project>/graphs/`, `orientation`), `arrangement` (`mode` `horizontal`, `vertical` or `grid`; `fit` `height` (default: every row `panel_height` tall) `width` (every row the widest row's width: stacked figures meet both margins) or `natural` (every panel at its own size in inches, a column the widest panel's width, a row its tallest panel's height, top-aligned: the text of every panel prints at one size; SVG or PDF panels); `order` of file_ids, in a grid one inner list per row such as `[[1, 2], [3, 4, 5]]`), a panel's own `label` (`{letter}`, `{index}`, `{roman}` I, II, …, `{file_id}`, `{file}`) and `label_position` (`bottom`, `bottom-left`, `bottom-right`, `top`, `top-left`, `top-right`, or `left`: beside the panel's left edge, centred on it, e.g. stacked figures indexed I, II, …; font from the `panel_label` rule), `legend` (`position` `top`, `bottom` or `right`, `columns`, `title`, `caption` (a line under the legend, e.g. `"Fig. a"`; `**bold**` and `*italic*` pieces, `\n` a new line), `colormap: {name, limits, label}` (top or bottom: a horizontal spectrum above the entries, the scale the facet's heat maps share), `frame`, `handle` as a job's plus `patch`, `mode`: `auto` (default) the entries of the graph job that drew the first panel, its models as its legend labels them, or `custom` with `entries` of `label` and optional `color`), `panel_height`, `spacing`, `export` (`formats` `png`, `pdf`, `svg`; `margin`; `png: {dpi}`, also the dpi of a pdf or svg of png panels; `svg: {sanitize, validate_xml, on_error}` as a job's; the default `svg` nests each panel's own SVG, its ids prefixed, all vector; a panel without an SVG makes it embed the PNGs; a png or pdf facet needs the panels' PNGs or PDFs), `text_rules` (the facet's own: `title`, `x_label`, `y_label`, `panel_label`, `legend_text`); run with `make FACE_ID=<id> graph`; vector PDF panels need `pypdf`. Moved keys are refused with their new place: `style` → `text_rules`, `export.dpi` → `export.png.dpi`, `legend.style` → `legend.handle` |
| `model_explanation` | no `metric`; `method`: `weight_visualization` (Zaid et al. 2020 Eqs. 4–5: \|W\| of the dense layer after the flatten, mean over the filters at each time point, then over its neurons), `gradient` (Masure et al. 2019, `captum.attr.Saliency` of the true class), `gradient_x_input` (`captum.attr.InputXGradient` of the true class, as Wouters et al. 2020), `heatmap` (Zaid Eq. 6: a Conv1d layer's output averaged over its filters; `layers: [1, …]` counted over the Conv1d layers, all when absent; one figure per layer). `traces`: test traces used (all when absent), `aggregate`: `mean_abs` (default) or `mean` over them. Reads the run's final checkpoint (seal-checked, as the test) and `test.h5` with its stored `scaler.json`; one curve per model on the input's sample axis. PyTorch runs only (`deep_learning.framework: pytorch`) |
| Model lists | job 0 plots `top_models` (rewritten by `make judge`); every other job lists `models` by hand. An entry is a model_name (`projects/<P>/artifacts/done_model_names.yaml`, or `artifacts/filtered_models.yaml` from `make find_model`) or a model_hash (`artifacts/run_registry.json`). Each is bound to its run of the ENV through `run_registry.json` and `model_name_map.json` (both must agree) and logged as `series model_name=... model_hash=... status=...`; a status other than DONE is warned about. Unknown, mismatched or other-ENV entries fail with close matches. Series and legends use the model_name; `legend.labels` may be keyed by either |

## Contents

```text
services/graph_manager/
├── graph_manager.py        # CLI entry, job dispatch, output paths, choice tables
├── graph_job_validator.py  # job schema, sources, processes, job-0 model lists, model_name/model_hash binding
├── metric_router.py        # metric -> storage and plotter route; the plot types each metric takes
├── metric_loader.py        # load scalars/arrays; fold aggregation; key-recovery records
├── metric_catalog.py       # "What you can choose" and model-name tables
├── plotters.py             # line, overlay, curve, distribution, bars, std bands, export
├── faceting.py             # facets: finished graph files composed into one figure (FACE_ID)
├── attack_plotters.py      # SCA curves, fold rank curves, key hypothesis scores
├── plot_style.py           # fonts, lines, grid, std band, axes, text rules
├── svg_sanitizer.py        # SVG cleanup and XML validation
├── missing_plotter.py      # missing-metric warnings (no placeholder images)
├── sources.py              # sources.json: the runs, files and figures of a job or facet, signed
├── explain.py              # model_explanation: weight visualisation, gradient, gradient x input, heatmaps
├── graph_export.py         # finished runs into graph jobs, drawn figures into facets (panel Export to …); prune; shift ids
├── report_jobs.py          # a project's per-configuration jobs (loss, GE + overlays, key confidence) and facets, from its run registry (P2 Fig. 8 at full size)
├── raw_exporter.py         # raw plot-data export; not wired in (see TODOs)
└── __init__.py
```

## Depends on

`services/side_channel_core/utils`: `config_loader` (project config),
`path_utils` (metrics directory), `logger` (terminal output),
`sweep_engine` (`load_registry`, read-only) and `model_naming`
(`_load_mapping`) to bind a job's models to their runs.

## Used by

- Repo-root `Makefile`: `make graph`.
- `services/model_judge/model_judge.py` writes job 0's `top_models` through
  `side_channel_core/utils/graph_config_sync.py`.
- `services/side_channel_core/tests/unit/test_graph_manager.py`,
  `test_final_graph.py` (model binding by name or hash).

## TODOs

No `TODO`/`FIXME` comments in this service. Open issues:

- `raw_exporter.py` is not imported anywhere, so the `export.save_raw_plot_data`
  switch in graph jobs has no effect.
- `source: cross_validation` averages every `metrics/cross_validation/fold_*`
  directory it finds (`metric_loader.fold_metric_dirs`): it does not check the
  CV manifest's status, the fold count or a common epoch, so a stale, failed or
  partial fold enters the figure (repo-root `TODO.md` §19).
