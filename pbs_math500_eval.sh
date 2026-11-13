#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q workq
#PBS -N mlc_math500_eval
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
: "${EVAL_MODEL:?Set EVAL_MODEL to a model or full policy checkpoint}"
: "${EVAL_OUTPUT:?Set EVAL_OUTPUT to a new output directory}"
python evaluation/math500.py --model "$EVAL_MODEL" --output-dir "$EVAL_OUTPUT" --max-problems "${MAX_PROBLEMS:-500}"
