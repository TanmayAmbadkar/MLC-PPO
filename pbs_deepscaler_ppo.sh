#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -N mlc_deepscaler_ppo
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
: "${ARCH:?}" "${RUN_LABEL:?}"
case "$ARCH" in final_ensemble|multilayer_mean) ;; *) exit 2;; esac
python train_mlc_ppo.py --model Qwen/Qwen2.5-Math-1.5B-Instruct \
 --train-file runs/deepscaler/prepared/train.jsonl --prompt-key problem --answer-key gold --reward-type trl_math \
 --output-dir "runs/deepscaler/$RUN_LABEL" --critic-architecture "$ARCH" \
 --critic-warmup-steps 5 --max-steps "${STEPS:-400}" --batch-size 4 --micro-batch-size 1 \
 --max-prompt-length 2048 --max-completion-length 2048 --ppo-epochs 2 --critic-epochs 1 \
 --learning-rate 5e-7 --critic-learning-rate 1e-4 --kl-coefficient 0 \
 --generation-temperature 1.0 --generation-top-p 1.0 --seed 42 --gradient-checkpointing \
 --save-steps 100 --log-rollouts --verify-reward-parity
