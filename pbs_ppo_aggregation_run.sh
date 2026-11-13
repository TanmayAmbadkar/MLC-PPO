#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -N mlc_ppo_compare
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
: "${AGGREGATION:?Set AGGREGATION to mean, min, or gate}"
case "$AGGREGATION" in mean|min|gate) ;; *) echo 'Invalid aggregation' >&2; exit 2;; esac
python train_mlc_ppo.py --model Qwen/Qwen2.5-Math-1.5B-Instruct \
  --train-file runs/experiment2/prepared/train.jsonl --prompt-key problem --answer-key gold \
  --reward-type trl_math --output-dir "runs/mlc_ppo_dapo_${AGGREGATION}_seed42" \
  --critic-aggregation "$AGGREGATION" --critic-layers 0.5,0.75,1.0 \
  --max-steps 100 --batch-size 1 --ppo-epochs 2 --critic-epochs 2 \
  --learning-rate 5e-7 --critic-learning-rate 1e-4 --seed 42 \
  --max-prompt-length 2048 --max-completion-length 2048 --save-steps 50
