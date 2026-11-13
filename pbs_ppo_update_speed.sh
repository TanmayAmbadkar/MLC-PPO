#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=100gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -N mlc_update_speed
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_ALLOC_CONF=expandable_segments:True
: "${PILOT_VARIANT:?}"
export PILOT_OUT="runs/vllm_training_speed/$PILOT_VARIANT"
mkdir -p "$PILOT_OUT"
ln -sf "$PWD/runs/vllm_full_update/rollouts.json" "$PILOT_OUT/rollouts.json"
CHECKPOINT=(--gradient-checkpointing)
case "$PILOT_VARIANT" in amp) export PILOT_AMP=1;; amp_sorted) export PILOT_AMP=1 PILOT_SORT=1;; sorted) export PILOT_SORT=1;; no_checkpoint) CHECKPOINT=();; *) exit 2;; esac
/scratch/tsa5252/.conda/envs/grpo_env/bin/python experiments/vllm_pilot/full_update_train.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --algorithm ppo --critic-architecture multilayer_mean --all-layers --train-file runs/viability_prepared/train.jsonl --prompt-key problem --answer-key gold --reward-type trl_math --output-dir "$PILOT_OUT/train" --max-steps 1 --stop-after-step 1 --batch-size 64 --micro-batch-size 4 --group-size 4 --max-prompt-length 2048 --max-completion-length 2048 --ppo-epochs 2 --critic-epochs 1 --learning-rate 5e-7 --critic-learning-rate 1e-4 --generation-temperature 1 --generation-top-p 1 --seed 42 "${CHECKPOINT[@]}" --token-budget 1000000 --verify-reward-parity --no-save
