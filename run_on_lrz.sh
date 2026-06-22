#!/bin/bash
# =====================================================================
# run_on_lrz.sh — ONE command to run the agent on an LRZ GPU node.
#
# Run this FROM YOUR LAPTOP:   ./run_on_lrz.sh
# It does everything end-to-end:
#   1. sync the code to LRZ
#   2. submit a SLURM GPU job that runs agent.py on a compute node
#   3. wait for it to finish
#   4. download the results back to ./outputs/
#
# NOTE: local development does NOT use this script — just run `python agent.py`.
# This file exists purely for the LRZ deployment.
# =====================================================================

LOCAL_DIR="/Users/m245172/Desktop/uni/Seminar/nlpcss/"

# 1. Sync code to LRZ -------------------------------------------------
echo "[1/4] Syncing local files to LRZ..."
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.DS_Store' --exclude 'outputs' \
  "$LOCAL_DIR" lrz:~/seminar/ > /dev/null

# 2. Submit the GPU job ----------------------------------------------
# The SLURM batch script is fed to `sbatch` on stdin via the heredoc below,
# so there is no separate .sh file to maintain. The lines starting with
# "#SBATCH" are SLURM directives (resource requests), NOT comments.
# 'EOF' is quoted, so $VARS and $(...) inside are evaluated on the COMPUTE
# NODE at run time, not on your laptop now.
echo "[2/4] Submitting GPU job..."
JOB_ID=$(ssh lrz "sbatch --qos=mcml --parsable" <<'EOF'
#!/bin/bash
#SBATCH --job-name=survey-agent
#SBATCH --partition=mcml-hgx-a100-80x4   # <-- CONFIRM with `sinfo`; pick a partition you may use
#SBATCH --qos=mcml                       # quality-of-service
#SBATCH --gres=gpu:1                     # 1 GPU is plenty for a 7-9B model (~15-20 GB in fp16)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00                  # walltime; the assertions take a few minutes
#SBATCH --output=survey-agent-%j.out     # stdout -> ~/survey-agent-<jobid>.out
#SBATCH --error=survey-agent-%j.err      # stderr -> ~/survey-agent-<jobid>.err

set -euo pipefail

# Move to the synced project directory (step 1 rsyncs into ~/seminar/).
cd "$HOME/seminar"

# Activate the venv. Create it ONCE on a login node (the macOS .venv is NOT synced):
#   module load python/3.12          # or whatever Python module LRZ provides
#   python -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
source .venv/bin/activate

# Point agent.py at the LARGE model on the dss filesystem (a LOCAL path, so no
# HuggingFace download happens on the compute node). This is the deliverable
# model knob — change it here. Confirm what is available with:
#   ls /dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/
export MODEL_NAME="/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/Qwen2.5-7B-Instruct"

# Force offline so the job never hangs on a network call if a node has no internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Running on node: $(hostname)"
echo "Model: $MODEL_NAME"
nvidia-smi

python agent.py --label lrz
echo "agent.py finished. Results are in ~/seminar/outputs/items_lrz.json"
EOF
)
echo "Job submitted, Job ID: $JOB_ID"

# 3. Poll until the job leaves the queue -----------------------------
echo "[3/4] Waiting for job to finish (queuing + running may take a few minutes)..."
while ssh lrz "squeue --me -h -j $JOB_ID" | grep -q "$JOB_ID"; do
    sleep 10
done
echo "Job finished!"

# 4. Sync results back to local --------------------------------------
echo "[4/4] Downloading results to local machine..."
mkdir -p "${LOCAL_DIR}outputs"
rsync -avz lrz:~/seminar/outputs/ "${LOCAL_DIR}outputs/" > /dev/null

echo "All done! Check the local outputs/ directory for results (items.json)."