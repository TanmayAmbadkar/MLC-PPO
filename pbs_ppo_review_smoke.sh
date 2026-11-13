#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -q workq
#PBS -N mlc_review_smoke
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
: "${ARCH:?Set ARCH}"
case "$ARCH" in final_single|final_ensemble|multilayer_mean|multilayer_min|multilayer_gate) ;; *) exit 2;; esac
python train_mlc_ppo.py --model Qwen/Qwen2.5-Math-1.5B-Instruct \
 --train-file runs/experiment2/prepared/train.jsonl --prompt-key problem --answer-key gold --reward-type trl_math \
 --output-dir "runs/ppo_review_smoke/$ARCH" --critic-architecture "$ARCH" \
 --critic-warmup-steps "${WARMUP:-2}" --max-steps 10 --batch-size 2 --micro-batch-size 1 \
 --max-prompt-length 2048 --max-completion-length 16 --ppo-epochs 2 --critic-epochs 1 \
 --kl-coefficient "${KL:-0.01}" --gradient-checkpointing --save-steps 10 --log-rollouts --verify-reward-parity
python tests/check_checkpoint_gpu.py "runs/ppo_review_smoke/$ARCH/checkpoint-10"
