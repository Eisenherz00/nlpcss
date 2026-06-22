#!/bin/bash
# =====================================================================
# SLURM batch script — runs agent.py on an LRZ AI Systems GPU node.
# Submitted by run_on_lrz.sh via:  sbatch --qos=mcml --parsable ~/seminar/run_agent.sh
#
# NOTE: the lines below starting with "#SBATCH" are SLURM directives,
# not comments. Adjust partition / account to match YOUR LRZ project.
# Find your options on a login node with:  sinfo   and   sacctmgr show assoc user=$USER
# =====================================================================
#SBATCH --job-name=survey-agent
#SBATCH --partition=mcml-hgx-a100-80x4   # <-- CONFIRM with `sinfo`; pick a partition you may use
#SBATCH --qos=mcml                       # quality-of-service (also passed on the sbatch line)
#SBATCH --gres=gpu:1                     # 1 GPU is plenty for a 7B model (~15 GB in fp16)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00                  # walltime; 30 assertions takes a few minutes
#SBATCH --output=%x-%j.out               # stdout -> survey-agent-<jobid>.out
#SBATCH --error=%x-%j.err                # stderr -> survey-agent-<jobid>.err

set -euo pipefail

# Move to the synced project directory (run_on_lrz.sh rsyncs into ~/seminar/)
cd "$HOME/seminar"

# --- Environment ---------------------------------------------------
# Create this venv ONCE on an LRZ login node (the macOS .venv is NOT synced):
#     module load python/3.12   # or whatever Python module LRZ provides
#     python -m venv .venv
#     source .venv/bin/activate
#     pip install -r requirements.txt
source .venv/bin/activate

# Point agent.py at the LARGE model on the dss filesystem (a LOCAL path, so no
# HuggingFace download happens on the compute node). agent.py reads MODEL_NAME.
# Confirm the available model with:
#     ls /dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/
export MODEL_NAME="/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/Qwen2.5-7B-Instruct"

# Force offline so the job never hangs on a network call if a node has no internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Running on node: $(hostname)"
echo "Model: $MODEL_NAME"
nvidia-smi

# --- Run -----------------------------------------------------------
python agent.py

echo "agent.py finished. Results are in ~/seminar/outputs/items.json"
