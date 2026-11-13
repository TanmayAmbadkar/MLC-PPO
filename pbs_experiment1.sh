#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -N mlc_value_exp1
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
python experiment1/collect.py --output-dir "${DATA_DIR:-runs/experiment1/data}" --batch-size "${BATCH_SIZE:-8}" \
  --train-problems "${TRAIN_PROBLEMS:-200}" --val-problems "${VAL_PROBLEMS:-50}" --test-problems "${TEST_PROBLEMS:-50}" \
  --rollouts "${ROLLOUTS:-8,16,32}" --max-new-tokens "${MAX_NEW_TOKENS:-512}"
python experiment1/fit.py --data-dir "${DATA_DIR:-runs/experiment1/data}" --output-dir "${RESULTS_DIR:-runs/experiment1/results}"
