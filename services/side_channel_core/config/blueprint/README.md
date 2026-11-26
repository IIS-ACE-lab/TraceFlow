# services/side_channel_core/config/blueprint
The starting config of a new project, and the system's list of every key a project's config may hold. The TraceFlow Monitor (`services/visualization/vscode_monitor/`) copies every file here that is missing from `projects/<P>/config/` as soon as `<P>` is declared in `services/side_channel_core/config/master.yaml` (`make scaffold` does the same); a file the project already has is never touched.

The files are generic templates (written by hand on 2026-10-18; completed 2026-10-22): every key TraceFlow reads, empty, with a comment per key that says what it does (the panel shows it as the key's hint), no project's own notes. Two blocks hold values: `paths`, the standard run layout every project uses (edit it in the file, not in the panel), and `metrics_cfg`, the metrics' numeric constants (no study's choice: DLSCA_ASCAD's). What belongs to a project's data or study starts empty, added in the panel: `data: []` (Add a Dataset), `model: []` (Add a Model, Plan a Model), `sca.attack_byte: []` (the bytes the data entries allow), `experiment_queue.yaml` `{}` (Add a Queue), `single_runs.yaml` `runs: []` (Add a single Run). `templates.yaml` holds one item of each list a project fills (a data entry, a model, an optimizer): what Add a Dataset writes, and what an item's missing key is drawn from; its data entry starts with `debug_h5_inspect`, `show_contract` and `require_contract` true (the loader prints the file's layout and the contract, and refuses a generated file without one), every other value empty. An empty key is harmless: `config_loader` drops every empty key of params.yaml (the code's default applies); a required key left empty is refused by its reader. Two blocks are written forms the loader folds into the form the code and the run hash read (`config_loader.canonical_form`, so no run made before changes its hash): `trainer.early_stopping: {enabled, monitor_metric, patience, min_delta}` (the flat `early_stopping`, `monitor_metric`, `patience`, `min_delta` are retired in files; `patience` and `min_delta` count only with `enabled: true`, as the code reads them; `monitor_mode` is retired, the direction follows the metric), and `sca.key_enumeration` with every mode's block under `modes` (the code reads the chosen `mode`'s; the panel shows that one). The loop setting `trainer.precision` sits under `trainer` after `epochs` and `batch_size` (a failing epoch always ends the run); `grad_clip_norm` (every optimizer step clips the gradient norm) sits under each framework's block, `deep_learning.pytorch` (`clip_grad_norm_`) and `deep_learning.keras` (`global_clipnorm`), the chosen framework's read (folded to the key the code reads by the loader; the flat `trainer.grad_clip_norm` is retired); `trainer.evaluation.output_transformation.args.dim` is the class axis (1). The ENV files' empty keys override nothing.

**The template follows the code.** Each file's keys are its schema in `utils/config_schema.py` (`make_blueprint.py FILE_SCHEMAS`) minus `config_schema.RETIRED` (renamed or retired keys the readers still accept). `python -m services.visualization.vscode_monitor.make_blueprint check` lists the keys a file lacks or should not offer; `sync` adds every missing key, empty, in place (the comment is then written by hand). A test runs `check`, so a key added to a schema reaches the blueprint, and from there every project's panel form, before the tests pass (guide 10 §7). **The projects follow the blueprint:** on every scaffold (a master.yaml save in the panel, `make scaffold`), each declared project's params.yaml, model_judge.yaml, test_config.yaml and model_filter.yaml get the blueprint keys they lack, written empty (`make_blueprint complete`; the config loader reads an empty key as absent, so no run made before changes its hash); a key with a value (a path, a constant, an empty block) is left to the panel's **Load Blueprint**, which names what joins the run hash. `python -m services.visualization.vscode_monitor.make_blueprint complete --project <P> [--all]` does the same from a terminal (`--all`: every key, for a project without runs).

The panel reads these files (`choices.blueprint_forms`): in every config form, a key the file lacks is drawn faded, empty, with its menu (the system's choices: `choices.py` rules, true / false for `choices.BOOLEAN_KEYS`) and its hint; choosing a value stages it. **Load Blueprint** (a form's button) stages every key the file lacks, empty, and `paths` with its values; nothing is written until OK. In params.yaml it says which staged blocks join the run hash (`choices.HASHED`, checked against `utils/model_naming.make_experiment_hash`): after OK, a new run of a model trained before gets a new hash. Fill a project's own copy, in the panel or by hand.

## Starting a project

1. Declare it in `services/side_channel_core/config/master.yaml`: one item of the `project:` list, the whole block below (copy it, then change `name`, `description` and `root`). The panel's Master Control → **Add a Project** does this for you: it writes `master_project.yaml` of this folder (the same block) under the name you give, then step 2. The panel copies the blueprint as soon as `name`, `root` (one folder name) and `params` are given, but runs read the rest of the block too: without `scripts.test`, for example, the test of every run fails (`KeyError: 'test'`).

   ```yaml
   project:
     - name: "MY_PROJECT"
       description: "What the project studies"
       root: "MY_PROJECT/"            # relative to TraceFlow/projects/
       auto_search:
         enabled: false
         config: "config/auto_search.yaml"
         hard_replace_params: true    # true: auto_search.yaml replaces the params.yaml fields it sets; false: fills missing ones only
       params: "config/params.yaml"
       exp_rerun:
         path:   "artifacts/{{env}}/generated_models/{{model_name}}/rerun/{{timestamp}}_trial_{{trial}}"
         repeat: 1
       print_config: false
       use_core_src: True             # import services/side_channel_core/src/*.py
       code_dir: src                  # with use_core_src: false, look under <root>/src/
       logs:   artifacts/logs/
       models: artifacts/models/
       metrics: artifacts/metrics/
       scripts:                       # join the run hash: keep them as they are
         train:        "src/sca/train.py"
         model:        "src/core/model.py"
         test:         "src/sca/test.py"
         data_loader:  "src/sca/ascad_data_loader.py"
       env_configs:
         dev:     "config/dev.yaml"
         staging: "config/staging.yaml"
         prod:    "config/prod.yaml"
       distributed_override:
         use: False
         num_gpus: 1
       extra_configs:
         graph:  "config/graph_config.yaml"
         test: "config/test_config.yaml"
       visualization:
         clearml:
           enabled: false
           scope: "global"            # global (app.clear.ml) | local (self-hosted)
           fail_fast_on_unreachable_api: true
           profiles:
             global:
               web_url: "https://app.clear.ml"
               api_url: "https://api.clear.ml"
               files_url: "https://files.clear.ml"
               verify_certificate: true
               access_key_name: "CLEARML_GLOBAL_API_ACCESS_KEY"
               secret_key_name: "CLEARML_GLOBAL_API_SECRET_KEY"
               clearml_env_file: "/app/secrets/clearml_global.env"
               healthcheck_enabled: true
               healthcheck_path: "/debug.ping"
               healthcheck_timeout_sec: 5.0
             local:
               web_url: "http://127.0.0.1:8080"
               api_url: "http://127.0.0.1:8008"
               files_url: "http://127.0.0.1:8081"
               verify_certificate: true
               access_key_name: "CLEARML_LOCAL_API_ACCESS_KEY"
               secret_key_name: "CLEARML_LOCAL_API_SECRET_KEY"
               clearml_env_file: "/app/secrets/clearml_local.env"
               healthcheck_enabled: true
               healthcheck_path: "/debug.ping"
               healthcheck_timeout_sec: 5.0
         graph_manager:
           enabled: true
           config: "config/graph_config.yaml"
   ```

2. Save `master.yaml`: the panel (or `make scaffold ENV=dev`) creates `projects/MY_PROJECT/config/` with the eleven files and `projects/MY_PROJECT/README.md` from `PROJECT_README.md` (the study's sources, claims, protocol, the settings to set before each queue and a figure index with one row per figure, model and dataset: a manual, per-queue workflow needs that organised record; `projects/DLSCA_ASCAD/README.md` and `projects/ZvW/README.md` are filled examples), and the project appears in the panel with every setting the system has, empty, each with its menu.
3. In the panel: Add a Dataset (`services/doc/TUTORIAL.md` §6 gives every type's values) and Add a Model (or Plan a Model), then the settings a run needs (each refused, by name, when left empty): `seed`, `device`, `sca.attack_byte`, `sca.leakage_model`, `sca.candidate_ranking.mode`, `perform_attack.validation` and `.test` (`attack_times`, `shuffle`, `attack_distribution`), `perform_attack.attack_step` with `validation.rank_step`, `trainer.epochs`, `trainer.batch_size`, `trainer.loss.type`, the optimizer's `type`, `learning_rate` and `weight_decay`, `trainer.evaluation.output_transformation.type`, `metrics_to_log`; with cross-validation on, `evaluation.validation.checkpointing: false`. Every other setting left empty is the code's default. Then run `make orchestrate ENV=dev PROJECT=MY_PROJECT DATASET=<data name> MODEL_ID=<model_id>`.

Checked 2026-10-18: a project declared this way, scaffolded from these files and filled with the DPAv4 settings of the tutorial (only keys the template holds), trained and tested to DONE (key 0x82 at rank 1). Checked 2026-10-22 with the completed templates: a project scaffolded from them, given a data entry from `templates.yaml` (ASCAD.h5, small ranges), one MLP and the settings of step 3 through the panel's editor (`yamledit`), every other key left empty (dev.yaml: 1 epoch, 2 folds, cpu), loaded without a ConfigError and trained and tested to DONE on the CPU (`core.cli orchestrate-cmd`, `ALLOW_HOST_RUN=1`); each setting of step 3 left empty was refused with a message naming it.

## Contents
```text
blueprint/
├── params.yaml            # every params.yaml key, empty (paths: the standard layout); data and model empty (datasets: services/doc/TUTORIAL.md)
├── experiment_queue.yaml  # make experiment QUEUE=<key>: none yet ({})
├── single_runs.yaml       # the panel's Run Single Experiment: make orchestrate lines, none yet
├── graph_config.yaml      # make graph GRAPH_ID=<id>: job 0 (the judge's, id 0) and an example facet (id 1, off), keys empty
├── model_filter.yaml      # make find_model
├── model_judge.yaml       # make judge
├── test_config.yaml       # make test
├── dev.yaml               # env override for ENV=dev
├── staging.yaml           # env override for ENV=staging
├── prod.yaml              # env override for ENV=prod
├── auto_search.yaml       # OptunaLite search: engine, method, budget, tracking (not from DLSCA_ASCAD)
├── model_config.yaml      # optional generic model_config template (not from DLSCA_ASCAD)
├── master_project.yaml    # a master.yaml project entry (Master Control → Add a Project; not copied to projects)
├── templates.yaml         # one data entry, model and optimizer: Add a Dataset's entry, an item's missing keys (not copied)
├── PROJECT_README.md      # the project's README template: shipped as projects/<P>/README.md (sources, claims, protocol, the set-before-each-queue table, the figure index per model kind)
└── README.md              # this file
```

Where each copy is read once placed in the project:
- `params.yaml`: `master.yaml` `project[].params` -> `utils/config_loader.py`.
- `dev/staging/prod.yaml`: `master.yaml` `env_configs` -> applied key by key by `config_loader.apply_env_overrides`.
- `auto_search.yaml`: `master.yaml` `auto_search.config` -> `config_loader.py`, `services/mlops/core/optuna_lite*.py`.
- `model_judge.yaml`: `services/model_judge/model_judge.py` (`make judge`).
- `single_runs.yaml`: the TraceFlow Monitor only (Run Single Experiment); a line's Start runs it as `make orchestrate`.
- `graph_config.yaml`: `master.yaml` `extra_configs.graph` -> `cfg["graph"]`, read by `services/graph_manager` (`make graph`).
- `model_config.yaml`: `config_loader.py` loads `projects/<P>/config/model_config.yaml` into `cfg["extras"]["model_config"]`; no code reads that key.

## Used by
- The TraceFlow Monitor and `make scaffold`, to start a project's `config/` (missing files only).
- The TraceFlow Monitor's config forms (`choices.blueprint_forms`): a key a file lacks, its value, menu and hint; Load Blueprint.

## TODOs
None.
