#!/usr/bin/env bash
#services/side_channel_core/scripts/run_cicd.sh
set -e

echo "Running pipeline in ENV=${ENV:-dev}"

# fallback to dev if ENV not set:
ENV=${ENV:-dev}

python ci_cd/__init__.py --config config/pipeline_config.yaml
bash ci_cd/test.sh
bash ci_cd/build.sh
bash ci_cd/deploy.sh
