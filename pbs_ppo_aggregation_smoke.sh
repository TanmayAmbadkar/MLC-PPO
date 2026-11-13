#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -N mlc_agg_smoke
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
for mode in mean min gate; do
  python train_mlc_ppo.py --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --train-file runs/experiment2/prepared/train.jsonl --prompt-key problem --answer-key gold \
    --reward-type trl_math --output-dir "runs/ppo_aggregation_smoke/$mode" --critic-aggregation "$mode" \
    --max-steps 1 --batch-size 1 --max-prompt-length 2048 --max-completion-length 16 \
    --ppo-epochs 2 --critic-epochs 1 --no-save
 done
