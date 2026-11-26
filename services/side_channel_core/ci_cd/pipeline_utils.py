#  services/side_channel_core/ci_cd/pipeline_utils.py
'''
Pipeline utility script to run CICD stages defined in a YAML configuration file.
This script reads the pipeline configuration, sets up environment variables,
and executes each stage in sequence, logging success or failure.
Pipeline is loyal to the process —not the project.

Usage:
    python pipeline_utils.py | by default it will run in dev environment
    ENV={dev,staging,prod} python pipeline_utils.py | to run in development,staging,production environment

Author: Shraddha Mishra, 
'''

import subprocess
import os
import yaml
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
#now you can run from anywhere without exporting PYTHONPATH
from side_channel_core.utils.status_store import pipeline_status
from utils.timestamps import now_iso


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PIPELINE_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'pipeline_config.yaml')
PARAMS_PATH = os.path.join(PROJECT_ROOT, 'config', 'params.yaml')

print("Looking for config here:", PIPELINE_CONFIG_PATH)

with open(PIPELINE_CONFIG_PATH, 'r') as f:
    pipeline_config = yaml.safe_load(f)

with open(PARAMS_PATH, 'r') as f:
    global_config = yaml.safe_load(f)

CURRENT_ENV = os.getenv("ENV", "dev")
print(f"Pipeline will run with ENV={CURRENT_ENV}")

def run_pipeline():
    """Run the pipeline stages defined in the configuration file."""
    
    log_dir = global_config['ci_cd']['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    # Prepare final env vars
    env_vars = os.environ.copy()
    for env_entry in pipeline_config['pipeline'].get('environment', []):
        key, value = env_entry.split("=")
        if value.strip() == "{{ENV}}":
            value = CURRENT_ENV
        env_vars[key] = value

    print(f"Final env passed to stages: {env_vars}")

    for stage in pipeline_config['pipeline']['stages']:
        name = stage['name']
        pipeline_status['stage'] = name
        pipeline_status['status'] = f"Running: {name}"
        pipeline_status['timestamp'] = now_iso()
        script = os.path.join(PROJECT_ROOT, stage['script'])
        timeout = stage.get('timeout', 60)

        if not os.path.exists(script):
            raise FileNotFoundError(f"Script not found: {script}")

        print(f"Running stage: {name} with script: {script} (timeout={timeout}s)")
        try:
            subprocess.run(
                ["bash", script],
                check=True,
                timeout=timeout,
                env=env_vars
            )
            log_success(name, log_dir)
        except subprocess.TimeoutExpired:
            log_failure(name, log_dir)
            raise Exception(f"Timeout at stage: {name}")
        except subprocess.CalledProcessError as e:
            log_failure(name, log_dir)
            raise Exception(f"Subprocess error: {e}")

def log_success(stage, log_dir):
    with open(f"{log_dir}/{stage}_success.log", "w") as f:
        f.write(f"{stage} completed successfully.\n")
    print(f"[{stage.upper()}] Success")

def log_failure(stage, log_dir):
    with open(f"{log_dir}/{stage}_failure.log", "w") as f:
        f.write(f"{stage} failed.\n")
    print(f"[{stage.upper()}] Failure")
    raise Exception(f"Pipeline failed at stage: {stage}")
