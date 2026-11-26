# services/side_channel_core/Makefile

# ========================================
# SCA Side Channel Core — Makefile (HY-enabled, fast builds)
# Purpose:
# - Universal fast build: BuildKit -> containerd image store (NO tarball/unpack)
# - Dev:     compose up --no-build --pull=never
# - Staging: same host => pull=never; remote staging => pull=missing (+ optional push)
# - Prod:    pull=missing + push required (immutable tag)
# Usage:

#   ENV={dev|staging|prod} HY={H1|H2|H3|H4} [FORCE=1] make orchestrate
#   
# Examples:
#	#Basic environment selection

#   make ENV=dev orchestrate
#   make ENV=staging orchestrate
#   make ENV=prod orchestrate

#   #With HY (Hypothesis)selection (required if auto_mlops.enabled=true in master.yaml)

#   make ENV=dev HY=H1 orchestrate
#   make ENV=staging HY=H2 orchestrate
#   make ENV=prod HY=H3 orchestrate
#   make ENV=prod HY=H4 FORCE=1 orchestrate

#   # Project-scoped runs (manual / non-auto-mlops)

#   make ENV=dev PROJECT=DLSCA_ASCAD orchestrate
#   make ENV=dev PROJECT=DLSCA_ASCAD orchestrate
#   make ENV=staging PROJECT=DLSCA_ASCAD orchestrate
#   make ENV=prod PROJECT=DLSCA_ASCAD orchestrate

#   # Rerun specific experiment (manual / non-auto-mlops)
#   make ENV=dev PROJECT=<Project_Name> rerun RERUN_EXPERIMENT=<Model_Name> REPEAT=1
#   
#   #Project + Dataset pinning
#   make ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad orchestrate
#   make ENV=staging PROJECT=DLSCA_ASCAD DATASET=ascad orchestrate
#	# Specific model selection
#   make ENV=dev orchestrate MODEL_ID=CNN1
#   make ENV=dev orchestrate MODEL_ID=MLP2 FORCE=1
#   make ENV=dev orchestrate MODEL_ID=MLP2 BATCH=fig6   # label the runs (make find_model: batches contains fig6)
# 	Graph generation:
#   make ENV=dev GRAPH_ID=1 graph
#   make ENV=dev PROJECT=DLSCA_ASCAD GRAPH_ID=2 graph
#   #Full manual contract (advanced)
#   make ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad  ALIAS=run_001 orchestrate
#   make ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad  ALIAS=eval_001 orchestrate 
#   make ENV=dev HY=H1 orchestrate
#   make ENV=staging HY=H2 orchestrate
#   make ENV=staging HY=H2 STAGING_REMOTE=1 PUSH_STAGING=1 orchestrate
#   make ENV=prod HY=H3 FORCE=1 orchestrate
#   make ENV=dev PROJECT=DLSCA_ASCAD orchestrate
#	make ENV=dev PROJECT=DLSCA_ASCAD orchestrate
#   #DELETE database (destructive)
#	make delete_database ENV=dev PROJECT=DLSCA_ASCAD # shows the plan, deletes nothing
#   make delete_database ENV=dev PROJECT=DLSCA_ASCAD CONFIRM=1  # every dev run of the project
#	make delete_database ENV=dev PROJECT=DLSCA_ASCAD MODEL=mesh_chicken CONFIRM=1   # one run (name or hash, comma-separated)
#
# Notes:
# - Requires rootful containerd + nerdctl + compose v2
# - Assumes single-host deployment unless ENV=staging + STAGING_REMOTE=1
# - Image pushed only for ENV=prod or staging(remote)+PUSH_STAGING=1
# - FORCE=1 to force rebuild even if image exists
# - Cleans up on successful exit; preserves volumes on failure for debugging

#  Author: Shraddha Mishra
# ========================================
# services/side_channel_core/Makefile

# ========================================
# SCA Side Channel Core — Makefile (HY-enabled, fast builds)
# Purpose:
# - Universal fast build: BuildKit -> containerd image store (NO tarball/unpack)
# - Dev:     compose up --no-build --pull=never
# - Staging: same host => pull=never; remote staging => pull=missing (+ optional push)
# - Prod:    pull=missing + push required (immutable tag)
# ========================================

SHELL := /bin/bash
.ONESHELL:
MAKEFLAGS += --no-print-directory

.PHONY: kill-build kill-build-steps own-projects status_share graph_export graph_delete graph_renumber graph_facet graph_facet_delete dataset-submodule orchestrate experiment rerun test import judge rerun-judged find_model batch_label delete_database graph preflight-rootful stack-down stack-nuke build-image maybe-push run-stack \
        docker-clean docker-teardown lint healthcheck reset panel scaffold verify seal seal-key

# -------------------------
# Required knobs
# -------------------------
ifndef ENV
$(error ENV is required! Use: ENV={dev|staging|prod} make orchestrate)
endif

VALID_ENVS := dev staging prod
ifeq (,$(filter $(ENV),$(VALID_ENVS)))
$(error Invalid ENV "$(ENV)". Allowed values: dev, staging, prod)
endif

# Rebuild control (0=skip if image exists, 1=force rebuild)
FORCE ?= 0
MLOPS ?= 0

# Staging mode: 0=same host (pull=never), 1=remote staging behavior (pull=missing)
STAGING_REMOTE ?= 0

# Optional push in staging (only meaningful if STAGING_REMOTE=1)
PUSH_STAGING ?= 0

# -------------------------
# Paths / compose
# -------------------------
DOCKER_COMPOSE_BASE := services/side_channel_core/deployment/docker/docker-compose.yaml
DOCKER_COMPOSE_ENV  := services/side_channel_core/deployment/docker/docker-compose.$(ENV).yaml
DOCKER_ENV_FILE     := services/side_channel_core/deployment/docker/.env

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BUILD_CONTEXT := $(ROOT)
DOCKERFILE    := $(ROOT)/services/side_channel_core/deployment/docker/Dockerfile
# build-image's output, read by the panel for the build progress (gitignored; make docker-teardown removes it)
BUILD_LOG     := $(ROOT)logs/build_image.log

# -------------------------
# Runtime exports
# -------------------------
export USER_ID  := $(shell id -u)
export GROUP_ID := $(shell id -g)

export ENV
export HY
export FORCE
export MLOPS
export CONTAINER_NAME

# -------------------------
# Orchestrator knobs
# -------------------------
PROJECT ?=
DATASET ?=
ALIAS ?=
MODELS ?=
MODEL_ID ?=
# Batch label of the runs (make experiment sets it to the queue name; see utils/batch.py)
BATCH ?=
BATCH_ID ?=
GRAPH_ID ?=
FACE_ID ?=
# -------------------------
# Rerun knobs
# Supports:
#   make ENV=dev PROJECT=DLSCA_ASCAD rerun fungus_elastic
#   make ENV=dev PROJECT=DLSCA_ASCAD rerun fungus_elastic REPEAT=3
#   make ENV=dev PROJECT=DLSCA_ASCAD rerun RERUN_EXPERIMENT=fungus_elastic REPEAT=3
# -------------------------
RERUN_EXPERIMENT ?=
REPEAT ?=
# make reproduce: comma-separated registered runs the container trains from their snapshots (set by make reproduce)
REPRODUCE_MODELS ?=
# make reproduce: a chosen folder as the root of the runs' outputs (every paths: template inside it); absolute
REPRODUCE_ROOT ?=
# -------------------------
# Test request knobs (make test)
#   MODEL       : model name(s) or hash(es), comma-separated (else test_request.model)
#   TEST_CONFIG : optional other request file, relative to the project root
#   EXTERNAL=1  : an external test: test_config.yaml's perform_attack settings instead of the
#                 snapshot's; reports in <run>/test/<test id>/, the outcome in test_registry.json's
#                 external_tests, beside the recorded test (src/sca/test_request.py)
# -------------------------
MODEL ?=
TEST_CONFIG ?=
TEST_MODEL ?=
EXTERNAL ?=
TEST_EXTERNAL ?=
# make test BUDGET=1: a Budget test (each attack the data entry's data[].attack_trace_budget traces), else All
BUDGET ?=
TEST_BUDGET ?=
# Model filter knob (make find_model): another filter file, relative to the project root
FILTER ?=

ifneq ($(filter rerun,$(MAKECMDGOALS)),)
RERUN_GOALS := $(filter-out rerun,$(MAKECMDGOALS))
ifneq ($(strip $(RERUN_GOALS)),)
RERUN_EXPERIMENT ?= $(firstword $(RERUN_GOALS))
$(foreach goal,$(RERUN_GOALS),$(eval $(goal):;@:))
endif
endif

export PROJECT DATASET MODE ALIAS MODELS MODEL_ID BATCH BATCH_ID GRAPH_ID FACE_ID RERUN_EXPERIMENT REPEAT REPRODUCE_MODELS REPRODUCE_ROOT

# -------------------------
# nerdctl / compose
# -------------------------
NAMESPACE ?= default
NERDCTL := sudo -E nerdctl --namespace $(NAMESPACE)
COMPOSE := $(NERDCTL) compose --env-file $(DOCKER_ENV_FILE) -f $(DOCKER_COMPOSE_BASE) -f $(DOCKER_COMPOSE_ENV)

# -------------------------
# Image identity
# -------------------------
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)
# Code version of the runs this call starts, recorded in each run's device.yaml
# (services/side_channel_core/utils/device_record.py): the full commit, whether
# services/ differs from it (tracked edits or new files), and a short SHA-256 of
# that difference. The container mounts no .git, so they are read here.
TRACEFLOW_COMMIT := $(shell git rev-parse HEAD 2>/dev/null)
TRACEFLOW_CODE_MODIFIED := $(if $(shell git status --porcelain -- services 2>/dev/null),1,0)
TRACEFLOW_CODE_DIFF := $(if $(filter 1,$(TRACEFLOW_CODE_MODIFIED)),$(shell { git diff HEAD --no-ext-diff -- services; git status --porcelain -- services; } 2>/dev/null | sha256sum | cut -c1-16),)
# Artifact seals (services/side_channel_core/utils/seal.py): the signing key,
# created on first use and mounted read-only into the container, so the host
# tools and the container sign with one key. A seal carries its public key:
# nothing is shared with whoever reads, clones or reproduces the runs.
TRACEFLOW_SEAL_DIR ?= $(HOME)/.config/traceflow/seal
export TRACEFLOW_SEAL_DIR
# -------------------------
# Timezone (master.yaml timezone: current_location / current_country)
# Resolved once to an IANA zone and exported as TZ, so host commands, the
# container (docker-compose passes TZ) and shell `date` share one clock.
# -------------------------
TRACEFLOW_TZ := $(shell PYTHONPATH="$(ROOT)/services/side_channel_core" python -c 'from utils.timestamps import timezone_name; print(timezone_name())' 2>/dev/null)
ifneq ($(strip $(TRACEFLOW_TZ)),)
export TZ := $(TRACEFLOW_TZ)
endif

IMAGE_REPO ?= sidechannel
# The dev image holds no code (compose binds the checkout over /app): its tag is the hash of what does go
# into it (the Dockerfile, requirements.txt, the user's ids, the timezone), so a commit never rebuilds it
# and an image is built once per change of those. staging and prod copy the code: their tag is the commit.
IMAGE_INPUTS := $(shell { cat "$(DOCKERFILE)" "$(ROOT)/services/side_channel_core/infra/environment/requirements.txt" 2>/dev/null; echo "$(USER_ID):$(GROUP_ID):$(TRACEFLOW_TZ)"; } | sha256sum | cut -c1-12)
ifeq ($(ENV),dev)
IMAGE_TAG  ?= dev-$(IMAGE_INPUTS)
else
IMAGE_TAG  ?= $(ENV)-$(GIT_SHA)
endif
# One tag for the whole make call: build-image and run-stack are sub-makes, and
# without this each would read HEAD again (a commit made during the build
# changed the tag between them: "image not available").
export IMAGE_TAG

IMAGE_LOCAL := $(IMAGE_REPO):$(IMAGE_TAG)

# Set this to your real registry when you go multi-host:
# e.g. IMAGE_REGISTRY=registry.local:5000
IMAGE_REGISTRY ?= docker.io/library
IMAGE_REMOTE   := $(IMAGE_REGISTRY)/$(IMAGE_REPO):$(IMAGE_TAG)

# Compose should run local tag
export IMAGE_REF := $(IMAGE_LOCAL)

define image_exists
  $(NERDCTL) images | awk 'BEGIN{f=0} $$1=="$(IMAGE_REPO)" && $$2=="$(IMAGE_TAG)" {f=1} END{exit !f}'
endef

# -------------------------
# ENV-specific policies
# -------------------------
ifeq ($(ENV),dev)
PULL_POLICY := never
ALLOW_PRUNE := 0
REQUIRE_PUSH := 0
else ifeq ($(ENV),staging)
ifeq ($(STAGING_REMOTE),1)
PULL_POLICY := missing
else
PULL_POLICY := never
endif
ALLOW_PRUNE := 0
REQUIRE_PUSH := 0
else ifeq ($(ENV),prod)
PULL_POLICY := missing
ALLOW_PRUNE := 1
REQUIRE_PUSH := 1
endif

# -------------------------
# Container Name
# -------------------------
MASTER_YAML := $(ROOT)/services/side_channel_core/config/master.yaml

AUTO_MLOPS_ENABLED := $(shell python -c 'import yaml; from pathlib import Path; p=Path("$(MASTER_YAML)"); d=yaml.safe_load(p.read_text()); enabled=any(bool((((item or {}).get("auto_mlops") or {}).get("enabled"))) for item in (d.get("project") or [])); print("1" if enabled else "0")' 2>/dev/null || echo 0)
export AUTO_MLOPS_ENABLED

# -------------------------
# Project selection
# -------------------------
DEFAULT_PROJECT := $(shell python -c 'import yaml; from pathlib import Path; p=Path("$(MASTER_YAML)"); d=yaml.safe_load(p.read_text()) or {}; print(d.get("default_project","ICSP"))' 2>/dev/null || echo ICSP)

RESOLVED_PROJECT := $(if $(strip $(PROJECT)),$(PROJECT),$(DEFAULT_PROJECT))
export RESOLVED_PROJECT

# -------------------------
# HY rules (required only when auto_mlops is enabled)
# -------------------------
VALID_HY := H1 H2 H3 H4

ifeq ($(AUTO_MLOPS_ENABLED),1)
  ifndef HY
    $(error HY is required because auto_mlops.enabled=true in master.yaml. Use: HY={H1|H2|H3|H4})
  endif
  ifeq (,$(filter $(HY),$(VALID_HY)))
    $(error Invalid HY "$(HY)". Allowed values: H1, H2, H3, H4)
  endif
else
  HY ?= none
endif

export HY

define SET_CONTAINER_NAME_FROM_MASTER
	@if [ "$(AUTO_MLOPS_ENABLED)" = "1" ]; then \
		export CONTAINER_NAME="sidechannel-$(ENV)-$(HY)"; \
	else \
		export CONTAINER_NAME="sidechannel"; \
	fi; \
	echo "auto_mlops.enabled=$(AUTO_MLOPS_ENABLED) → CONTAINER_NAME=$$CONTAINER_NAME";
endef

# The run reads its frozen copy of the project's config folder (utils/config_freeze.py): make experiment
# froze it at queue start (TRACEFLOW_FROZEN_CONFIG set); make orchestrate (make reproduce) and make test,
# run alone, freeze it at their start, and RELEASE_CONFIG (their EXIT trap) removes the copy they made.
# Edits made meanwhile reach the next run, not this one. $(1): the copy's label prefix.
define FREEZE_CONFIG
FROZEN_OWN=""; \
if [ -z "$${TRACEFLOW_FROZEN_CONFIG:-}" ]; then \
	TRACEFLOW_FROZEN_CONFIG="$$(PYTHONPATH="$(ROOT)" python -m services.side_channel_core.utils.config_freeze freeze \
		--project "$(RESOLVED_PROJECT)" --label "$(1)-$$(date -u +%Y%m%dT%H%M%SZ)-$$$$")"; \
	TRACEFLOW_FROZEN_PROJECT="$(RESOLVED_PROJECT)"; \
	FROZEN_OWN=1; \
fi; \
export TRACEFLOW_FROZEN_CONFIG TRACEFLOW_FROZEN_PROJECT; \
echo "config frozen: $$TRACEFLOW_FROZEN_CONFIG (project $$TRACEFLOW_FROZEN_PROJECT)"
endef
RELEASE_CONFIG = if [ -n "$$FROZEN_OWN" ]; then PYTHONPATH="$(ROOT)" python -m services.side_channel_core.utils.config_freeze release --project "$(RESOLVED_PROJECT)" --dir "$$TRACEFLOW_FROZEN_CONFIG" || true; fi

# -------------------------
# Preflight
# -------------------------
preflight-rootful:
	@systemctl is-active --quiet containerd || \
	(echo "containerd service is not active (rootful). Fix with: sudo systemctl enable --now containerd"; exit 1)

# -------------------------
# Stack lifecycle
# -------------------------
stack-down:
	@echo "Stopping stack (keeping volumes)..."
	-$(COMPOSE) down --remove-orphans || true
	@# a container compose could not stop (a killed build left it, a stuck task): removed by force
	@c="$$($(NERDCTL) ps -a --format '{{.Names}}' 2>/dev/null | grep '^sidechannel' || true)"; \
	[ -z "$$c" ] || { echo "removing by force: $$c"; $(NERDCTL) rm -f $$c; }

stack-nuke:
	@echo "Stopping stack + removing volumes (destructive)..."
	-$(COMPOSE) down -v --remove-orphans || true

# -------------------------
# Universal fast build
# The layers are compressed with zstd at level 3 for the local containerd store: gzip took 113 s of the
# export for a 7 GB image (BuildKit's default), zstd a fraction of it; containerd unpacks either.
# The Dockerfile stage is the ENV (dev: no COPY, the checkout is bound in by compose; staging and
# prod: the code copied into the image). Without --target the last stage (prod) was built for
# every ENV, so a dev run used the code as copied at build time (audit 2026-10-23).
# -------------------------
build-image:
	@if [ ! -f "$(DOCKERFILE)" ]; then \
		echo "Dockerfile not found: $(DOCKERFILE)"; \
		exit 1; \
	fi

	@set -euo pipefail
	@# A build interrupted here (Ctrl-C, kill, the panel) must not leave BuildKit running its steps:
	@# the trap cancels them (kill-build-steps) before the recipe dies with the interrupt's status.
	@trap 'echo "build interrupted: cancelling the BuildKit steps"; $(MAKE) --no-print-directory kill-build-steps ENV=$(ENV); echo "Build killed: $$(date -Is)" >> "$(BUILD_LOG)"; exit 130' INT TERM; \
	if [ "$(FORCE)" != "1" ] && $(call image_exists); then \
		echo "Image exists: $(IMAGE_LOCAL) (skip build; use FORCE=1)"; \
	else \
		mkdir -p "$(dir $(BUILD_LOG))"; \
		echo "Fast build (type=image -> containerd) => $(IMAGE_LOCAL)" | tee "$(BUILD_LOG)"; \
		$(NERDCTL) build \
			-f $(DOCKERFILE) \
			--target $(ENV) \
			--progress=plain \
			--build-arg USER_ID=$(USER_ID) \
			--build-arg GROUP_ID=$(GROUP_ID) \
			--build-arg TZ_NAME=$(or $(TRACEFLOW_TZ),UTC) \
			--output type=image,name=$(IMAGE_LOCAL),push=false,unpack=true,compression=zstd,compression-level=3 \
			$(BUILD_CONTEXT) 2>&1 | tee -a "$(BUILD_LOG)"; \
		echo "Build complete: $(IMAGE_LOCAL)" >> "$(BUILD_LOG)"; \
	fi

	@$(NERDCTL) image ls --quiet $(IMAGE_LOCAL) >/dev/null 2>&1 || \
		(echo "Image missing after build: $(IMAGE_LOCAL) (namespace=$(NAMESPACE))"; \
		echo "   Debug: $(NERDCTL) images"; \
		$(NERDCTL) images | head -n 120; \
		exit 1)
	@echo "Build complete: $(IMAGE_LOCAL)"

# -------------------------
# Push policy
# -------------------------
maybe-push:
	@set -euo pipefail
	@if [ "$(REQUIRE_PUSH)" = "1" ]; then \
		echo "Tag remote: $(IMAGE_LOCAL) -> $(IMAGE_REMOTE)"; \
		$(NERDCTL) tag $(IMAGE_LOCAL) $(IMAGE_REMOTE); \
		echo "ENV=prod → pushing (required) => $(IMAGE_REMOTE)"; \
		$(NERDCTL) push $(IMAGE_REMOTE); \
	elif [ "$(ENV)" = "staging" ] && [ "$(STAGING_REMOTE)" = "1" ] && [ "$(PUSH_STAGING)" = "1" ]; then \
		echo "Tag remote: $(IMAGE_LOCAL) -> $(IMAGE_REMOTE)"; \
		$(NERDCTL) tag $(IMAGE_LOCAL) $(IMAGE_REMOTE); \
		echo "staging(remote) → pushing => $(IMAGE_REMOTE)"; \
		$(NERDCTL) push $(IMAGE_REMOTE); \
	else \
		echo "No push for ENV=$(ENV)"; \
	fi

# -------------------------
# Run stack
# -------------------------
# The container name compose prefixes to each line ("sidechannel |") is drawn in
# bold cyan on the terminal (tmux pane, make's output). The run folders' logs
# (artifacts/<env>/generated_models/<name>/logs) are written by the container
# itself and never carry it. pipefail: the recipe keeps compose's exit code.
# Give projects/ back to the user before anything writes there. Runs made while the container ran as
# root (before 2026-10-23) left root-owned run folders, registries and graphs, which the user-run
# container, the judge, the graphs and delete_database cannot write or remove. Only when a file in
# projects/ is not the user's (no sudo prompt otherwise); every target that writes there depends on it.
own-projects:
	@if [ -d projects ] && [ -n "$$(find projects -not -user "$$(id -u)" -print -quit 2>/dev/null)" ]; then \
		echo "▶  projects/ holds files not owned by $$(id -un): taking ownership (sudo chown)"; \
		sudo chown -R "$$(id -u):$$(id -g)" projects; \
	fi

run-stack: own-projects seal-key dataset-submodule
	@echo "▶  compose up (no-build, pull=$(PULL_POLICY)) ENV=$(ENV) HY=$(HY)"
	set -o pipefail
	ENV=$(ENV) \
	HY=$(HY) \
	FORCE=$(FORCE) \
	MLOPS=$(MLOPS) \
	USER_ID=$(USER_ID) \
	GROUP_ID=$(GROUP_ID) \
	IMAGE_REF=$(IMAGE_REF) \
	PROJECT="$(PROJECT)" \
	DATASET="$(DATASET)" \
	ALIAS="$(ALIAS)" \
	MODELS="$(MODELS)" \
	MODEL_ID="$(MODEL_ID)" \
	BATCH="$(BATCH)" \
	BATCH_ID="$(BATCH_ID)" \
	TRACEFLOW_FROZEN_CONFIG="$(TRACEFLOW_FROZEN_CONFIG)" \
	TRACEFLOW_FROZEN_PROJECT="$(TRACEFLOW_FROZEN_PROJECT)" \
	RERUN_EXPERIMENT="$(RERUN_EXPERIMENT)" \
	REPEAT="$(REPEAT)" \
	REPRODUCE_MODELS="$(REPRODUCE_MODELS)" \
	REPRODUCE_ROOT="$(REPRODUCE_ROOT)" \
	TEST_MODEL="$(TEST_MODEL)" \
	TEST_CONFIG="$(TEST_CONFIG)" \
	TEST_EXTERNAL="$(TEST_EXTERNAL)" \
	TEST_BUDGET="$(TEST_BUDGET)" \
	TRACEFLOW_COMMIT="$(TRACEFLOW_COMMIT)" \
	TRACEFLOW_CODE_MODIFIED="$(TRACEFLOW_CODE_MODIFIED)" \
	TRACEFLOW_CODE_DIFF="$(TRACEFLOW_CODE_DIFF)" \
	TRACEFLOW_IMAGE="$(IMAGE_REF)" \
	TRACEFLOW_IMAGE_ID="$$($(NERDCTL) image inspect --format '{{.Id}}' $(IMAGE_REF) 2>/dev/null || true)" \
	TRACEFLOW_IMAGE_DIGESTS="$$($(NERDCTL) image inspect --format '{{json .RepoDigests}}' $(IMAGE_REF) 2>/dev/null || true)" \
	TRACEFLOW_SEAL_HOST_DIR="$(TRACEFLOW_SEAL_DIR)" \
	TZ="$(TZ)" \
	$(COMPOSE) up --no-build --pull $(PULL_POLICY) --force-recreate 2>&1 \
		| sed -u -E 's/^(sidechannel[^ |]*) *\|/\x1b[1;36m&\x1b[0m/'
	RC=$$?
	# compose up returns 0 whatever the container returned (nerdctl has no --exit-code-from): the stack's
	# code is the container's, so a failed orchestration or test fails make, and make experiment stops there.
	CODE="$$($(NERDCTL) inspect --format '{{.State.ExitCode}}' "$${CONTAINER_NAME:-sidechannel}" 2>/dev/null || true)"
	if [ "$$RC" -eq 0 ] && [ -n "$$CODE" ] && [ "$$CODE" != "0" ]; then
		echo "container $${CONTAINER_NAME:-sidechannel} exit code=$$CODE"
		RC=$$CODE
	fi
	exit $$RC

# -------------------------
# Rerun stack
# -------------------------
rerun: preflight-rootful
	@set -euo pipefail
	@echo "DEBUG MAKECMDGOALS=$(MAKECMDGOALS)"
	@echo "DEBUG RERUN_EXPERIMENT=$(RERUN_EXPERIMENT)"
	@echo "DEBUG REPEAT=$(REPEAT)"
	@if [ -z "$(RERUN_EXPERIMENT)" ]; then \
		echo "Missing experiment name."; \
		echo "   Use: make ENV=dev PROJECT=DLSCA_ASCAD rerun <experiment_name> REPEAT=3"; \
		echo "   Or:  make ENV=dev PROJECT=DLSCA_ASCAD rerun RERUN_EXPERIMENT=<experiment_name> REPEAT=3"; \
		exit 2; \
	fi

	$(call SET_CONTAINER_NAME_FROM_MASTER)
	@echo "RERUN_EXPERIMENT=$(RERUN_EXPERIMENT) REPEAT=$(REPEAT) ENV=$(ENV) PROJECT=$(PROJECT)"

	@echo "IMAGE_REF=$(IMAGE_REF)"
	trap 'rm -f /tmp/orchestrator.pid' EXIT

	$(MAKE) stack-down

	@if [ "$(ALLOW_PRUNE)" = "1" ]; then \
		echo "ENV=prod → prune images + builder cache"; \
		$(NERDCTL) image prune -af || true; \
		$(NERDCTL) builder prune -af || true; \
	else \
		echo "No prune for ENV=$(ENV)"; \
	fi

	$(MAKE) build-image
	$(MAKE) maybe-push

	set +e
	$(MAKE) run-stack
	RC=$$?
	set -e
	echo "compose exit code=$$RC"

	if [ "$$RC" -eq 0 ]; then \
		$(MAKE) healthcheck || true; \
		$(MAKE) reset || true; \
	else \
		echo "Rerun failed; volumes preserved for debugging."; \
		echo "   (Run: make ENV=$(ENV) HY=$(HY) stack-nuke)"; \
	fi

	exit $$RC

# -------------------------
# Re-test trained models by name (core/cli.py test-cmd --test-request)
# Names come from MODEL= or projects/<P>/config/test_config.yaml
# (test_request.model); each is tested from its own training snapshot,
# params.yaml is not used (but for a Budget test's data[].attack_trace_budget).
# Every test is appended, under <run>/test/<hash>_et_<index>/ and
# test_registry.json external_tests; the training's recorded test is never
# rewritten, and the panel and the reports read the latest of each kind.
# Usage:
#   make test ENV=dev PROJECT=DLSCA_ASCAD                      # models in test_config.yaml
#   make test ENV=dev PROJECT=DLSCA_ASCAD MODEL=model_a,model_b
#   make test ENV=dev PROJECT=DLSCA_ASCAD MODEL=model_a BUDGET=1   # a Budget test (Retest Budget)
#   make test ENV=dev PROJECT=DLSCA_ASCAD MODEL=model_a EXTERNAL=1   # test_config.yaml's attack settings
# -------------------------
test: preflight-rootful
	@set -euo pipefail
	$(call FREEZE_CONFIG,test)
	trap '$(RELEASE_CONFIG)' EXIT
	@PYTHONPATH="$(ROOT):$(ROOT)/services/side_channel_core:$(ROOT)/services/side_channel_core/src" \
	PROJECTS_ROOT="$(ROOT)/projects" \
	python -c 'import sys; sys.tracebacklimit = 0; \
	from services.side_channel_core.src.sca.test_request import test_config_path, load_test_request, resolve_models, external_settings, TestRequestError; \
	path = test_config_path("$(RESOLVED_PROJECT)", "$(TEST_CONFIG)" or None); \
	names = [n.strip() for n in "$(MODEL)".split(",") if n.strip()] or load_test_request(path); \
	print("[test] config=" + str(path)); \
	runs = resolve_models("$(RESOLVED_PROJECT)", names, env="$(ENV)"); \
	[print("[test] external " + r["model_name"] + ": " + str(external_settings(path, r["model_name"]))) for r in runs if "$(EXTERNAL)" == "1"]; \
	print("[test] models=" + ",".join(r["model_name"] for r in runs))' \
	> /tmp/traceflow_test_request.txt 2>&1 || { cat /tmp/traceflow_test_request.txt; exit 2; }
	@grep "^\[test\]" /tmp/traceflow_test_request.txt
	TEST_MODEL="$$(sed -n 's/^\[test\] models=//p' /tmp/traceflow_test_request.txt)"
	rm -f /tmp/traceflow_test_request.txt
	$(call SET_CONTAINER_NAME_FROM_MASTER)
	$(MAKE) stack-down
	$(MAKE) build-image

	set +e
	$(MAKE) run-stack PROJECT="$(RESOLVED_PROJECT)" TEST_MODEL="$$TEST_MODEL" TEST_CONFIG="$(TEST_CONFIG)" TEST_EXTERNAL="$(if $(filter 1,$(EXTERNAL)),1,)" TEST_BUDGET="$(if $(filter 1,$(BUDGET)),1,)"
	RC=$$?
	set -e
	echo "compose exit code=$$RC"

	if [ "$$RC" -eq 0 ]; then \
		$(MAKE) reset || true; \
	else \
		echo "Test request failed; volumes preserved for debugging."; \
		echo "   (Run: make ENV=$(ENV) stack-nuke)"; \
	fi

	exit $$RC

# -------------------------
# Best model judge
# Usage:
#   make ENV=dev PROJECT=DLSCA_ASCAD judge
# -------------------------
judge: own-projects
	@set -euo pipefail; \
	echo "Model Judge target invoked"; \
	echo "   ENV=$(ENV)"; \
	echo "   PROJECT(raw)=$(PROJECT)"; \
	echo "   PROJECT(resolved)=$(RESOLVED_PROJECT)"; \
	if [ -z "$(RESOLVED_PROJECT)" ]; then \
		echo "PROJECT is required."; \
		echo "   Use: make ENV=dev PROJECT=DLSCA_ASCAD judge"; \
		exit 2; \
	fi; \
	if [ ! -f "projects/$(RESOLVED_PROJECT)/config/model_judge.yaml" ]; then \
		echo "Missing model judge config:"; \
		echo "   projects/$(RESOLVED_PROJECT)/config/model_judge.yaml"; \
		exit 2; \
	fi; \
	PYTHONPATH="$(ROOT):$(ROOT)/services/side_channel_core:$(ROOT)/services/side_channel_core/src" \
	python -m services.model_judge.model_judge --project "$(RESOLVED_PROJECT)" --env "$(ENV)"


# -------------------------
# Model filter: models by parameters, not metrics
# Reads projects/<P>/config/model_filter.yaml (or FILTER=) and artifacts/run_map.jsonl;
# writes artifacts/filtered_models.yaml (overwritten every run).
# Usage:
#   make find_model ENV=dev PROJECT=DLSCA_ASCAD
#   make find_model ENV=dev PROJECT=DLSCA_ASCAD FILTER=config/deep_cnns.yaml
# -------------------------
find_model: own-projects
	@set -euo pipefail; \
	if [ -z "$(RESOLVED_PROJECT)" ]; then \
		echo "PROJECT is required."; \
		echo "   Use: make ENV=dev PROJECT=DLSCA_ASCAD find_model"; \
		exit 2; \
	fi; \
	FILTER_FILE="projects/$(RESOLVED_PROJECT)/$(if $(strip $(FILTER)),$(FILTER),config/model_filter.yaml)"; \
	if [ ! -f "$$FILTER_FILE" ]; then \
		echo "Missing model filter config: $$FILTER_FILE"; \
		echo "   Copy services/side_channel_core/config/blueprint/model_filter.yaml"; \
		exit 2; \
	fi; \
	PYTHONPATH="$(ROOT):$(ROOT)/services/side_channel_core:$(ROOT)/services/side_channel_core/src" \
	python -m services.model_filter.model_filter --project "$(RESOLVED_PROJECT)" --env "$(ENV)" --config "$$FILTER_FILE"

# -------------------------
# Delete the run database of one environment (or of named models):
# artifacts/{run,test}_registry.json, run_map.jsonl, model_name_map.json,
# done_model_names.yaml, artifacts/<ENV>/generated_models/<model>, plus the now
# stale artifacts/filtered_models.yaml (make find_model) and model_selection/*
# (make judge) of that environment. Dry run unless CONFIRM=1. graphs/ is kept.
# Usage:
#   make delete_database ENV=dev PROJECT=DLSCA_ASCAD                       # show the plan
#   make delete_database ENV=dev PROJECT=DLSCA_ASCAD CONFIRM=1             # delete every dev run
#   make delete_database ENV=dev PROJECT=DLSCA_ASCAD MODEL=mesh_chicken CONFIRM=1
#   make delete_database ENV=dev PROJECT=DLSCA_ASCAD BATCH=fig26 CONFIRM=1      # the runs of a batch
#   make delete_database ENV=dev PROJECT=DLSCA_ASCAD BATCH=fig26 JSON=1         # dry run as JSON (the panel's list)
# -------------------------
CONFIRM ?= 0
delete_database: own-projects
	@set -euo pipefail; \
	if [ -z "$(RESOLVED_PROJECT)" ]; then \
		echo "PROJECT is required."; \
		echo "   Use: make delete_database ENV=dev PROJECT=DLSCA_ASCAD [DATASET=name] [MODEL=name,hash] [BATCH=name] [CONFIRM=1]"; \
		exit 2; \
	fi; \
	PYTHONPATH="$(ROOT):$(ROOT)/services/side_channel_core:$(ROOT)/services/side_channel_core/src" \
	python -m utils.delete_database --project-root "projects/$(RESOLVED_PROJECT)" --env "$(ENV)" \
		$(if $(strip $(DATASET)),--dataset "$(DATASET)",) \
		$(if $(strip $(MODEL)),--model "$(MODEL)",) \
		$(if $(strip $(BATCH)),--batch "$(BATCH)",) $(if $(filter 1,$(JSON)),--json,) $(if $(filter 1,$(CONFIRM)),--yes,)

# -------------------------
# Label existing runs with a batch (utils/batch.py). Takes the models of a
# make find_model output (artifacts/filtered_models.yaml, or FILE=...).
# Usage:
#   make find_model ENV=dev PROJECT=DLSCA_ASCAD          # select the runs
#   make batch_label ENV=dev PROJECT=DLSCA_ASCAD BATCH=fig6
# -------------------------
batch_label: own-projects
	@set -euo pipefail; \
	if [ -z "$(RESOLVED_PROJECT)" ] || [ -z "$(strip $(BATCH))" ]; then \
		echo "PROJECT and BATCH are required."; \
		echo "   Use: make batch_label ENV=dev PROJECT=DLSCA_ASCAD BATCH=fig6 [FILE=artifacts/filtered_models.yaml]"; \
		exit 2; \
	fi; \
	PYTHONPATH="$(ROOT):$(ROOT)/services/side_channel_core:$(ROOT)/services/side_channel_core/src" \
	python -m utils.batch --project-root "projects/$(RESOLVED_PROJECT)" --env "$(ENV)" \
		--batch "$(strip $(BATCH))" $(if $(strip $(FILE)),--file "$(FILE)",)

# -------------------------
# Rerun judged candidates from Model Judge leaderboard
# Usage:
#   make ENV=dev PROJECT=DLSCA_ASCAD rerun-judged
#   make ENV=dev PROJECT=DLSCA_ASCAD rerun-judged N_JUDGED=3
#   make ENV=dev PROJECT=DLSCA_ASCAD rerun-judged N_JUDGED=3 REPEAT=2
# -------------------------
rerun-judged: own-projects
	@set -euo pipefail; \
	INT_FLAG="/tmp/scc_rerun_judged_interrupted.flag"; \
	CAND_FILE="/tmp/scc_rerun_judged_candidates.txt"; \
	rm -f "$$INT_FLAG" "$$CAND_FILE"; \
	trap 'echo ""; echo "Interrupted. Stopping judged rerun queue."; touch "$$INT_FLAG"; echo "Cleaning GPU/runtime state..."; $(MAKE) -s ENV=$(ENV) PROJECT=$(RESOLVED_PROJECT) healthcheck || true; $(MAKE) -s ENV=$(ENV) PROJECT=$(RESOLVED_PROJECT) reset || true; rm -f "$$INT_FLAG" "$$CAND_FILE"; exit 130' INT TERM; \
	SEL="projects/$(RESOLVED_PROJECT)/model_selection/latest_selection.json"; \
	N="$(or $(N_JUDGED),10)"; \
	R="$(or $(REPEAT),1)"; \
	echo "Rerun judged candidates"; \
	echo "   ENV=$(ENV)"; \
	echo "   PROJECT=$(RESOLVED_PROJECT)"; \
	echo "   selection=$$SEL"; \
	echo "   N_JUDGED=$$N"; \
	echo "   REPEAT=$$R"; \
	if [ -z "$(RESOLVED_PROJECT)" ]; then \
		echo "PROJECT is required."; \
		echo "   Use: make ENV=dev PROJECT=DLSCA_ASCAD rerun-judged"; \
		exit 2; \
	fi; \
	if [ ! -f "$$SEL" ]; then \
		echo "Missing selection file: $$SEL"; \
		echo "   Run first: make ENV=$(ENV) PROJECT=$(RESOLVED_PROJECT) judge"; \
		exit 2; \
	fi; \
	python -c 'import json, sys; p=sys.argv[1]; n=int(sys.argv[2]); d=json.load(open(p)); [print(x["model_name"]) for x in d.get("leaderboard", [])[:n] if x.get("model_name")]' "$$SEL" "$$N" > "$$CAND_FILE"; \
	if [ ! -s "$$CAND_FILE" ]; then \
		echo "No candidates found in $$SEL"; \
		rm -f "$$CAND_FILE"; \
		exit 2; \
	fi; \
	echo "   candidates:"; \
	cat "$$CAND_FILE" | sed "s/^/     - /"; \
	while IFS= read -r MODEL; do \
		if [ -f "$$INT_FLAG" ]; then \
			echo "Interrupt flag found. Stopping before next candidate."; \
			rm -f "$$INT_FLAG" "$$CAND_FILE"; \
			exit 130; \
		fi; \
		echo ""; \
		echo "Rerunning judged candidate: $$MODEL"; \
		set +e; \
		$(MAKE) ENV=$(ENV) PROJECT=$(RESOLVED_PROJECT) rerun RERUN_EXPERIMENT="$$MODEL" REPEAT="$$R"; \
		RC=$$?; \
		set -e; \
		if [ "$$RC" -ne 0 ]; then \
			echo "Rerun stopped/failed. Stopping judged queue. Exit code=$$RC"; \
			rm -f "$$INT_FLAG" "$$CAND_FILE"; \
			exit "$$RC"; \
		fi; \
	done < "$$CAND_FILE"; \
	rm -f "$$INT_FLAG" "$$CAND_FILE"; \
	echo "Judged rerun queue completed."


# -------------------------
# Graph generation
# Usage:
#   make ENV=dev GRAPH_ID=1 graph
#   make ENV=dev PROJECT=DLSCA_ASCAD GRAPH_ID=2 graph
# -------------------------
# Finished runs into graph jobs of graph_config.yaml (services/graph_manager/graph_export.py);
# the panel's Done list does the same with [Export to Graph Config]:
#   make graph_export ENV=dev PROJECT=DLSCA_ASCAD MODELS=a,b,c SOURCE=cross_validation PROCESS=validation METRIC=mean_attack_rank_curve
#   make graph_delete PROJECT=DLSCA_ASCAD GRAPH_ID=35   # later jobs move down one id, folders and facets follow
#   make graph_renumber PROJECT=DLSCA_ASCAD             # close every gap in the job ids
GRAPH_EXPORT = PYTHONPATH="$(ROOT):$(ROOT)/services/side_channel_core:$(ROOT)/services/side_channel_core/src" \
	python -m services.graph_manager.graph_export
graph_export: own-projects
	@$(GRAPH_EXPORT) export --project "$(RESOLVED_PROJECT)" --env "$(ENV)" --models "$(MODELS)" \
		--source "$(or $(SOURCE),cross_validation)" $(if $(PROCESS),--process "$(PROCESS)",) \
		--metric "$(or $(METRIC),mean_attack_rank_curve)"
graph_delete:
	@test -n "$(GRAPH_ID)" || { echo "GRAPH_ID is required: make graph_delete PROJECT=<P> GRAPH_ID=<id>"; exit 2; }
	@$(GRAPH_EXPORT) delete-job --project "$(RESOLVED_PROJECT)" --id "$(GRAPH_ID)"
graph_renumber:
	@$(GRAPH_EXPORT) renumber --project "$(RESOLVED_PROJECT)"
# Drawn figures composed as a facet, or a facet removed with the later ones moving down:
#   make graph_facet PROJECT=DLSCA_ASCAD FILES=jobs/job_001_x/a.pdf,jobs/job_002_y/b.pdf
#   make graph_facet_delete PROJECT=DLSCA_ASCAD FACE_ID=1
graph_facet:
	@$(GRAPH_EXPORT) export-facet --project "$(RESOLVED_PROJECT)" --files "$(FILES)" $(if $(TITLE),--title "$(TITLE)",)
graph_facet_delete:
	@test -n "$(FACE_ID)" || { echo "FACE_ID is required: make graph_facet_delete PROJECT=<P> FACE_ID=<id>"; exit 2; }
	@$(GRAPH_EXPORT) delete-facet --project "$(RESOLVED_PROJECT)" --id "$(FACE_ID)"

graph: own-projects
	@set -euo pipefail
	@if [ -z "$(GRAPH_ID)" ] && [ -z "$(FACE_ID)" ]; then \
		echo "GRAPH_ID or FACE_ID is required."; \
		echo "   One graph job:   make ENV=dev GRAPH_ID=1 graph"; \
		echo "   One facet:       make ENV=dev FACE_ID=1 graph"; \
		echo "   Another project: make ENV=dev PROJECT=DLSCA_ASCAD GRAPH_ID=1 graph"; \
		exit 2; \
	fi
	@if [ -n "$(GRAPH_ID)" ] && [ -n "$(FACE_ID)" ]; then \
		echo "Give GRAPH_ID or FACE_ID, not both."; \
		exit 2; \
	fi

	@echo "Graph target invoked"
	@echo "   ENV=$(ENV)"
	@echo "   PROJECT(raw)=$(PROJECT)"
	@echo "   PROJECT(resolved)=$(RESOLVED_PROJECT)"
	@echo "   GRAPH_ID=$(GRAPH_ID)"
	@echo "   FACE_ID=$(FACE_ID)"

	@python -c 'import yaml; \
	from pathlib import Path; \
	p=Path("$(MASTER_YAML)"); \
	d=yaml.safe_load(p.read_text()) or {}; \
	projects=d.get("project") or []; \
	name="$(RESOLVED_PROJECT)"; \
	match=[x for x in projects if (x or {}).get("name")==name]; \
	assert match, "Project not found in master.yaml: " + name; \
	gm=((match[0].get("visualization") or {}).get("graph_manager") or {}); \
	assert gm, "visualization.graph_manager missing for project: " + name; \
	print("[graph] graph_manager.enabled=" + str(gm.get("enabled", False))); \
	print("[graph] graph_manager.config=" + str(gm.get("config", "<missing>")));'

	MPLCONFIGDIR="$(ROOT)/.cache/matplotlib" \
	PYTHONPATH="$(ROOT)/services/side_channel_core:$(ROOT)/services/side_channel_core/src:$(ROOT)" \
	python services/graph_manager/graph_manager.py \
		--project "$(RESOLVED_PROJECT)" \
		--env "$(ENV)" \
		$(if $(FACE_ID),--face-id "$(FACE_ID)",--graph-id "$(GRAPH_ID)")

# -------------------------
# Main orchestrator
# -------------------------
orchestrate: preflight-rootful
	@set -euo pipefail
	$(call SET_CONTAINER_NAME_FROM_MASTER)
	@echo "ENV=$(ENV) HY=$(HY) FORCE=$(FORCE) MLOPS=$(MLOPS) PULL=$(PULL_POLICY)"

	@echo "IMAGE_REF=$(IMAGE_REF)"
	$(call FREEZE_CONFIG,orchestrate)
	trap 'rm -f /tmp/orchestrator.pid; $(RELEASE_CONFIG)' EXIT

	$(MAKE) stack-down

	@if [ "$(ALLOW_PRUNE)" = "1" ]; then \
		echo "ENV=prod → prune images + builder cache"; \
		$(NERDCTL) image prune -af || true; \
		$(NERDCTL) builder prune -af || true; \
	else \
		echo "No prune for ENV=$(ENV)"; \
	fi

	$(MAKE) build-image
	$(MAKE) maybe-push

	RUN_SUMMARY=services/side_channel_core/deployment/docker/tmp/run_summary.txt
	RUN_STARTED=$$(date +%s)
	set +e
	$(MAKE) run-stack
	RC=$$?
	set -e
	echo "compose exit code=$$RC"

	if [ "$$RC" -eq 0 ]; then \
		$(MAKE) healthcheck || true; \
		$(MAKE) reset || true; \
	else \
		echo "Orchestrator failed; volumes preserved for debugging."; \
		echo "   (Run: make ENV=$(ENV) HY=$(HY) stack-nuke)"; \
	fi

	if [ -f "$$RUN_SUMMARY" ] && [ "$$(stat -c %Y "$$RUN_SUMMARY")" -ge "$$RUN_STARTED" ]; then \
		cat "$$RUN_SUMMARY"; \
	fi
	exit $$RC

# -------------------------
# Reproduce runs from their params_out snapshots alone (utils/reproduce.py).
# A shipped run is its snapshot at the project's root (master.yaml project: root) + paths.params_out:
#   projects/<root>/artifacts/<ENV>/generated_models/<model_name>/config/<model_name>_params.yaml
# make snapshots lists the snapshots on disk, checked as a re-run checks them
# (digest, folder, hash) and their registry state. make reproduce registers the
# snapshots of BATCH (their `batches`) and/or MODEL (names or hashes) under their
# own name and hash, then trains them from the snapshots (make orchestrate with
# REPRODUCE_MODELS: train-cmd -n, never params.yaml). Again: unfinished ones resume.
# Refused while an experiment trains on this machine (let the current run finish:
# one training stack, and make orchestrate starts with make stack-down).
# Usage:
#   make snapshots ENV=dev PROJECT=DLSCA_ASCAD [JSON=1]
#   make reproduce ENV=dev PROJECT=DLSCA_ASCAD BATCH=fig26
#   make reproduce ENV=dev PROJECT=DLSCA_ASCAD MODEL=cake_bacon,celery_plush
#   make reproduce ENV=dev PROJECT=DLSCA_ASCAD MODEL=cake_bacon FORCE=1   # a DONE run: train it again from its snapshot
#   make reproduce ENV=dev PROJECT=DLSCA_ASCAD MODEL=cake_bacon REPRODUCE_ROOT=/abs/1/2/3/pikachu
#       the chosen folder is the runs' root: every paths: template applies inside it
#       (/abs/1/2/3/pikachu/artifacts/<ENV>/generated_models/<model_name>/..., the registries under its artifacts/);
#       the snapshot must sit at its params_out path there; configs and scripts come from PROJECT
# -------------------------
# -------------------------
# Model import: test models other studies released, through TraceFlow's own test (utils/model_import.py).
# Nothing trains. projects/<P>/config/import_config.yaml (master.yaml extra_configs.import; the panel's
# Import a Model) lists the released files; params.yaml is not read for them. make import reads each file
# on the host (h5py only: SHA-256, layer list; a Lambda layer refused), converts its layers into a
# TraceFlow model entry and registers one run per file (snapshot, run_registry.json, model_id imported),
# then runs them as make reproduce does (make orchestrate REPRODUCE_MODELS=, MODEL_ID=imported): the
# released weights are copied into TraceFlow's model, checked against the released model on every test
# trace (the equality gate), sealed, and tested by the standard test. DONE ones are skipped (FORCE=1:
# imported again). Refused while an experiment trains on this machine (make orchestrate stops the stack).
# Usage:
#   make import ENV=dev PROJECT=ZvW                         # every import of import_config.yaml
#   make import ENV=dev PROJECT=ZvW MODEL=zaid_ascad_n0     # these only (run names or hashes)
#   make import ENV=dev PROJECT=ZvW IMPORT_CONFIG=config/other_imports.yaml
# -------------------------
import:
	@set -euo pipefail; \
	if [ -z "$(RESOLVED_PROJECT)" ]; then \
		echo "PROJECT is required: make import ENV=dev PROJECT=ZvW [MODEL=name,...] [FORCE=1]"; \
		exit 2; \
	fi; \
	OUT="$$(PYTHONPATH="$(ROOT):$(ROOT)/services/side_channel_core:$(ROOT)/services/side_channel_core/src" \
		python -m utils.model_import register --project "$(RESOLVED_PROJECT)" --env "$(ENV)" \
		$(if $(strip $(MODEL)),--model "$(MODEL)",) $(if $(strip $(IMPORT_CONFIG)),--config "$(strip $(IMPORT_CONFIG))",) \
		$(if $(filter 1,$(FORCE)),--force,))"; \
	echo "$$OUT" | sed '$$d'; \
	NAMES="$$(echo "$$OUT" | tail -n 1)"; \
	if [ -z "$$NAMES" ]; then echo "nothing to import (every listed model is DONE; FORCE=1 imports again)"; exit 0; fi; \
	$(MAKE) --no-print-directory orchestrate ENV="$(ENV)" PROJECT="$(RESOLVED_PROJECT)" BATCH="$(strip $(BATCH))" \
		MODEL_ID=imported REPRODUCE_MODELS="$$NAMES"

# -------------------------
# Artifact seals (services/side_channel_core/utils/seal.py)
# Every file a run writes carries an Ed25519 signature, checked whenever TraceFlow
# reads the file: a file changed outside TraceFlow stops the read, naming the file.
#   make verify ENV=dev PROJECT=DLSCA_ASCAD [MODEL=name,...]
#       every artifact's seal state per run folder (the whole artifacts/ folder
#       and the figures' sources.json under graphs/, or the named runs of ENV);
#       exit 1 when a file or a figure was changed
#   make seal ENV=dev PROJECT=DLSCA_ASCAD [MODEL=name,...]
#       sign the files of runs written before seals existed (once, after their
#       training); their device.yaml records sealed_after_run
#   make seal-key
#       creates the key if missing and prints its id, to publish with results;
#       a folder the user cannot create or write is created with sudo and
#       handed to the user (the folders it creates, never an existing parent)
# -------------------------
SEAL_PY = PYTHONPATH="$(ROOT):$(ROOT)/services/side_channel_core:$(ROOT)/services/side_channel_core/src" python

# The selected dataset's database from its submodule on the owner's server, on the host: the
# container mounts the data folder but not the checkout's .git or /srv/git, so inside it a missing
# database can only be downloaded (utils/dataset_fetch.py). Never fails the run: a database still
# missing is downloaded and checked by the run.
dataset-submodule:
	@$(SEAL_PY) -m utils.dataset_fetch --project "$(RESOLVED_PROJECT)" --dataset "$(DATASET)" 2>&1 \
		| grep -E "dataset_fetch" || true

seal-key:
	@set -euo pipefail; \
	D="$(TRACEFLOW_SEAL_DIR)"; \
	if [ ! -f "$$D/signing.key" ]; then \
		TOP="$$D"; while [ ! -e "$$(dirname "$$TOP")" ]; do TOP="$$(dirname "$$TOP")"; done; \
		[ -e "$$D" ] && TOP="$$D"; \
		if ! mkdir -p "$$D" 2>/dev/null || [ ! -w "$$D" ]; then \
			echo "$$D cannot be created or written by $$(id -un): creating it with sudo and taking ownership of $$TOP"; \
			sudo mkdir -p "$$D" && sudo chown -R "$$(id -u):$$(id -g)" "$$TOP" || true; \
		fi; \
		chmod 700 "$$D" 2>/dev/null || true; \
		$(SEAL_PY) -m utils.seal key >/dev/null 2>&1 || \
			( umask 077 && openssl genpkey -algorithm ed25519 -out "$$D/signing.key" ) >/dev/null 2>&1 || true; \
		if [ ! -f "$$D/signing.key" ]; then \
			echo "No artifact signing key: $$D could not be created, even with sudo."; \
			echo "   Set TRACEFLOW_SEAL_DIR=<a folder of yours> and run make seal-key."; \
			echo "   Nothing was started: a run without the key would write unsealed artifacts."; \
			exit 1; \
		fi; \
		echo "Created the artifact signing key $$D/signing.key"; \
	fi; \
	if [ "$(MAKECMDGOALS)" = "seal-key" ]; then $(SEAL_PY) -m utils.seal key; fi

verify seal:
	@set -euo pipefail; \
	if [ -z "$(RESOLVED_PROJECT)" ]; then echo "PROJECT is required: make $@ ENV=dev PROJECT=DLSCA_ASCAD [MODEL=name,...]"; exit 2; fi; \
	ROOT_DIR="$$($(SEAL_PY) -c 'from utils.reproduce import resolve_project; print(resolve_project("$(RESOLVED_PROJECT)"))' 2>/dev/null | tail -n 1)"; \
	if [ -n "$(strip $(MODEL))" ]; then \
		TARGETS=""; for m in $$(echo "$(MODEL)" | tr ',' ' '); do TARGETS="$$TARGETS $$ROOT_DIR/artifacts/$(ENV)/generated_models/$$m"; done; \
	else \
		TARGETS="$$ROOT_DIR/artifacts"; [ "$@" = verify ] && [ -d "$$ROOT_DIR/graphs" ] && TARGETS="$$TARGETS $$ROOT_DIR/graphs"; \
	fi; \
	$(SEAL_PY) -m utils.seal $@ $$TARGETS

snapshots:
	@set -euo pipefail; \
	if [ -z "$(RESOLVED_PROJECT)" ]; then echo "PROJECT is required: make snapshots ENV=dev PROJECT=DLSCA_ASCAD"; exit 2; fi; \
	PYTHONPATH="$(ROOT):$(ROOT)/services/side_channel_core:$(ROOT)/services/side_channel_core/src" \
	python -m utils.reproduce list --project "$(RESOLVED_PROJECT)" --env "$(ENV)" $(if $(filter 1,$(JSON)),--json,)

reproduce:
	@set -euo pipefail; \
	if [ -z "$(RESOLVED_PROJECT)" ] || [ -z "$(strip $(BATCH)$(MODEL))" ]; then \
		echo "PROJECT and BATCH (or MODEL) are required."; \
		echo "   Use: make reproduce ENV=dev PROJECT=DLSCA_ASCAD BATCH=fig26 [MODEL=name,hash]"; \
		exit 2; \
	fi; \
	if [ -n "$(strip $(REPRODUCE_ROOT))" ]; then \
		case "$(strip $(REPRODUCE_ROOT))" in /*) ;; *) echo "REPRODUCE_ROOT must be an absolute path"; exit 2;; esac; \
		[ -d "$(strip $(REPRODUCE_ROOT))" ] || { echo "REPRODUCE_ROOT $(REPRODUCE_ROOT) is not a folder"; exit 2; }; \
		sudo chown -R "$$(id -u):$$(id -g)" "$(strip $(REPRODUCE_ROOT))"; \
	fi; \
	OUT="$$(PYTHONPATH="$(ROOT):$(ROOT)/services/side_channel_core:$(ROOT)/services/side_channel_core/src" \
		python -m utils.reproduce register --project "$(RESOLVED_PROJECT)" --env "$(ENV)" \
		$(if $(strip $(BATCH)),--batch "$(strip $(BATCH))",) $(if $(strip $(MODEL)),--model "$(MODEL)",) \
		$(if $(filter 1,$(FORCE)),--force,) $(if $(strip $(REPRODUCE_ROOT)),--output-root "$(strip $(REPRODUCE_ROOT))",))"; \
	echo "$$OUT" | sed '$$d'; \
	NAMES="$$(echo "$$OUT" | tail -n 1)"; \
	if [ -z "$$NAMES" ]; then exit 0; fi; \
	$(MAKE) --no-print-directory orchestrate ENV="$(ENV)" PROJECT="$(RESOLVED_PROJECT)" BATCH="$(strip $(BATCH))" \
		REPRODUCE_MODELS="$$NAMES" REPRODUCE_ROOT="$(strip $(REPRODUCE_ROOT))"

# -------------------------
# Run the experiments lined up under one key of
# projects/<PROJECT>/config/experiment_queue.yaml, in the order listed.
# Without FORCE the queue resumes (DONE experiments are skipped); FORCE=1
# trains every experiment of the queue again. Stops at the first failure.
# Usage:
#   make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig6
#   make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig6 FORCE=1
# -------------------------
experiment:
	@set -euo pipefail; \
	if [ -z "$(RESOLVED_PROJECT)" ] || [ -z "$(strip $(QUEUE))" ]; then \
		echo "PROJECT and QUEUE are required."; \
		echo "   Use: make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig6 [FORCE=1]"; \
		exit 2; \
	fi; \
	PYTHONPATH="$(ROOT):$(ROOT)/services/side_channel_core:$(ROOT)/services/side_channel_core/src" \
	python -m services.experiment_queue.experiment_queue --project "$(RESOLVED_PROJECT)" --env "$(ENV)" \
		--queue "$(strip $(QUEUE))" $(if $(filter 1,$(FORCE)),--force,)

# -------------------------
# TraceFlow Monitor (VS Code side panel, services/visualization/vscode_monitor)
# Usage:
#   make panel ENV=dev       build the .vsix and install it into this machine's VS Code
#                            (run from a VS Code terminal, then 'Developer: Reload Window')
#   make scaffold ENV=dev    copy the blueprint config files every project of master.yaml lacks
# -------------------------
panel:
	@PYTHON="$$(command -v python)" bash "$(ROOT)services/visualization/vscode_monitor/install.sh"

scaffold:
	@PYTHONPATH="$(ROOT)" python -m services.visualization.vscode_monitor.layout scaffold --root "$(ROOT)"

# The read-only status page for phones and shared links (services/visualization/server/status.py), on this machine's
# address (found by itself, or master.yaml status_share.host) and status_share.port (HOST= / PORT= override). Links: the panel's share icon, or `python -m services.visualization.server.status share add NAME`.
status_share:
	@PYTHONPATH="$(ROOT)" python -m services.visualization.server.status serve $(if $(HOST),--host "$(HOST)") $(if $(PORT),--port "$(PORT)")

# -------------------------
# Utilities
# -------------------------
docker-clean:
	@echo "Destructive cleanup: containers + volumes + locks"
	-$(COMPOSE) down -v --remove-orphans || true
	rm -f /tmp/orchestrator.pid

# What an image build and a running experiment look like in the process list (the panel's
# commands.js keeps the same two patterns).
BUILD_PATTERN := ^(\S*/)?nerdctl( \S+)* build( |$$)
BUSY_PATTERN  := ^((\S*/)?make( \S+)* (orchestrate|experiment|test|reproduce)( |$$)|(\S*/)?python[0-9.]* -m core\.cli (orchestrate|test|train)-cmd( |$$))

# make kill-build ENV=dev: cancels an image build for good. Killing the nerdctl client alone is not
# enough: BuildKit keeps running the build's steps as containerd tasks in its own namespace, which is
# "the container that keeps running" after a killed build, and the next tear down waits on it. So: the
# client (INT, then TERM, then KILL), every task of the buildkit namespace, and, if one survives, the
# BuildKit daemon itself (systemd restarts it clean). The build log records the kill.
kill-build:
	@set +e; \
	if ! pgrep -f '$(BUILD_PATTERN)' >/dev/null 2>&1 && [ -z "$$(sudo ctr --namespace buildkit tasks ls 2>/dev/null | awk 'NR>1')" ]; then \
		echo "no image build runs: nothing to kill"; exit 0; \
	fi; \
	sudo pkill -INT -f '$(BUILD_PATTERN)'; sleep 3; sudo pkill -TERM -f '$(BUILD_PATTERN)'; sleep 2; sudo pkill -KILL -f '$(BUILD_PATTERN)'; \
	$(MAKE) --no-print-directory kill-build-steps ENV=$(ENV); \
	for i in 1 2 3 4 5 6 7 8 9 10; do \
		pgrep -f '$(BUILD_PATTERN)' >/dev/null 2>&1 || [ -n "$$(sudo ctr --namespace buildkit tasks ls 2>/dev/null | awk 'NR>1')" ] || break; sleep 2; \
	done; \
	mkdir -p "$$(dirname "$(BUILD_LOG)")"; \
	if pgrep -f '$(BUILD_PATTERN)' >/dev/null 2>&1 || [ -n "$$(sudo ctr --namespace buildkit tasks ls 2>/dev/null | awk 'NR>1')" ]; then \
		echo "Build NOT killed: $$(date -Is)" >> "$(BUILD_LOG)"; echo "build still alive after 20 s: $$(pgrep -af '$(BUILD_PATTERN)')"; sudo ctr --namespace buildkit tasks ls; exit 1; \
	fi; \
	echo "Build killed: $$(date -Is)" >> "$(BUILD_LOG)"; echo "build killed: no client, no BuildKit task left"

# The BuildKit side of a killed build: every build step still running as a containerd task in the
# buildkit namespace, then the daemon itself if one survives (systemd restarts it clean). Called by
# kill-build, and by build-image's own trap, so a Ctrl-C in the terminal cancels the steps too.
kill-build-steps:
	@set +e; \
	for t in $$(sudo ctr --namespace buildkit tasks ls 2>/dev/null | awk 'NR>1 {print $$1}'); do sudo ctr --namespace buildkit tasks kill -s KILL "$$t" 2>/dev/null; done; sleep 1; \
	if [ -n "$$(sudo ctr --namespace buildkit tasks ls 2>/dev/null | awk 'NR>1')" ]; then echo "BuildKit still runs build steps: restarting the daemon"; sudo systemctl restart buildkit; fi; \
	true

# make docker-teardown ENV=dev: frees the GPU, memory and disk the stack holds (the panel's Docker tear down).
# docker-clean, then every sidechannel* container and sidechannel:* image, the build cache, dangling
# images, unused volumes, the compose tmp/ folder and the build log. The three prunes act on the
# whole namespace ($(NAMESPACE)), other projects' containers' build cache and unused volumes included.
# FORCE=1 (the panel's Force tear down): first kill whatever would block it, the running experiment
# (its make and python chain), the image build (make kill-build) and the containers themselves.
docker-teardown:
	@set -uo pipefail
	if [ "$(FORCE)" = "1" ]; then \
		echo "FORCE=1: killing the running experiment, the image build and the containers first"; \
		sudo pkill -TERM -f '$(BUSY_PATTERN)' 2>/dev/null; sleep 2; sudo pkill -KILL -f '$(BUSY_PATTERN)' 2>/dev/null; \
		$(MAKE) --no-print-directory kill-build ENV=$(ENV); \
		c="$$($(NERDCTL) ps -a --format '{{.Names}}' 2>/dev/null | grep '^sidechannel' || true)"; \
		[ -z "$$c" ] || $(NERDCTL) rm -f $$c; \
	fi
	$(MAKE) docker-clean ENV=$(ENV)
	c="$$($(NERDCTL) ps -a --format '{{.Names}}' 2>/dev/null | grep '^sidechannel' || true)"
	[ -z "$$c" ] || $(NERDCTL) rm -f $$c
	i="$$($(NERDCTL) images --format '{{.Repository}}:{{.Tag}}' | grep '^sidechannel:' || true)"
	[ -z "$$i" ] || $(NERDCTL) rmi -f $$i
	$(NERDCTL) builder prune -af
	$(NERDCTL) image prune -f
	$(NERDCTL) volume prune -f
	sudo rm -rf services/side_channel_core/deployment/docker/tmp/* /tmp/orchestrator.pid "$(BUILD_LOG)"
	echo "Docker torn down: GPU, memory and disk freed"

lint:
	@echo "Running Pylint..."
	pylint services.side_channel_core/src      --output-format=colorized --reports=y ; true
	pylint services.side_channel_core/utils    --output-format=colorized --reports=y ; true
	pylint services.side_channel_core/scripts  --output-format=colorized --reports=y ; true
	pylint services.side_channel_core/tests    --output-format=colorized --reports=y ; true

healthcheck:
	bash services/side_channel_core/scripts/healthcheck.sh

reset:
	bash services/side_channel_core/scripts/gpu_reset.sh
