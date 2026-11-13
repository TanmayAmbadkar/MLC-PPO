#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=2
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -N mlc_full_critic_smoke
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
args=(--model Qwen/Qwen2.5-Math-1.5B-Instruct --algorithm full_ppo --critic-architecture final_single --train-file runs/experiment2/prepared/train.jsonl --prompt-key problem --answer-key gold --reward-type trl_math --output-dir runs/viability_smoke/full_critic --max-steps 2 --batch-size 4 --micro-batch-size 1 --group-size 4 --max-prompt-length 2048 --max-completion-length 16 --ppo-epochs 2 --critic-epochs 1 --learning-rate 5e-7 --critic-learning-rate 1e-4 --generation-temperature 1 --generation-top-p 1 --seed 42 --gradient-checkpointing --save-steps 1 --token-budget 100000 --verify-reward-parity)
python train_full_critic.py "${args[@]}" --stop-after-step 1
python train_full_critic.py "${args[@]}" --stop-after-step 2 --resume-training
