#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q workq
#PBS -N mlc_dapo_eval
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
: "${EVAL_MODEL_NAME:?Set EVAL_MODEL_NAME to base, mean, min, or gate}"
case "$EVAL_MODEL_NAME" in
  base) model=Qwen/Qwen2.5-Math-1.5B-Instruct ;;
  mean|min|gate) model="runs/mlc_ppo_dapo_${EVAL_MODEL_NAME}_seed42/checkpoint-100" ;;
  *) echo 'Invalid model' >&2; exit 2 ;;
esac
python evaluation/evaluate.py --model "$model" --name "$EVAL_MODEL_NAME" --output-dir "runs/dapo_eval/$EVAL_MODEL_NAME"
