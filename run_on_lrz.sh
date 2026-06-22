#!/bin/bash

# 1. Sync code to LRZ
echo "[1/4] Syncing local files to LRZ..."
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.DS_Store' --exclude 'outputs' \
  /Users/m245172/Desktop/uni/Seminar/nlpcss/ lrz:~/seminar/ > /dev/null

# 2. Submit job and capture Job ID
echo "[2/4] Submitting GPU job..."
JOB_ID=$(ssh lrz "sbatch --qos=mcml --parsable ~/seminar/run_agent.sh")
echo "Job submitted, Job ID: $JOB_ID"

# 3. Poll until job completes
echo "[3/4] Waiting for job to finish (queuing + running may take a few minutes)..."
while ssh lrz "squeue --me -h -j $JOB_ID" | grep -q "$JOB_ID"; do
    # Check status every 10 seconds
    sleep 10
done
echo "Job finished!"

# 4. Sync results back to local
echo "[4/4] Downloading results to local machine..."
mkdir -p /Users/m245172/Desktop/uni/Seminar/nlpcss/outputs
rsync -avz lrz:~/seminar/outputs/ /Users/m245172/Desktop/uni/Seminar/nlpcss/outputs/ > /dev/null

echo "All done! Check the local outputs/ directory for results."
