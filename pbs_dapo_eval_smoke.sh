#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -N mlc_eval_smoke
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
for name in base mean min gate; do
 if [[ "$name" == base ]]; then model=Qwen/Qwen2.5-Math-1.5B-Instruct; else model="runs/mlc_ppo_dapo_${name}_seed42/checkpoint-100"; fi
 python evaluation/evaluate.py --model "$model" --name "$name" --max-problems 2 --max-new-tokens 128 --output-dir "runs/dapo_eval_smoke/$name"
done
