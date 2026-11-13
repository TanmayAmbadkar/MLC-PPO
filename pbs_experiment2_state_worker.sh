#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -N mlc_dapo_states
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
: "${WORKER_RANK:?Set worker rank}"
python experiment2/collect.py --rank "$WORKER_RANK" --world-size 4 --max-new-tokens 4096 \
 --difficulty-config runs/experiment2/difficulty/config.json
