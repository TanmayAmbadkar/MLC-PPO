#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -N mlc_fast_smoke
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_ALLOC_CONF=expandable_segments:True FAST_VLLM_GPU_FRACTION=0.35
: "${METHOD:?}"
case "$METHOD" in grpo) ALGO=grpo; ARCH=multilayer_mean; SCRIPT=train_fast_viability.py; LAYERS=();; final_single) ALGO=ppo; ARCH=final_single; SCRIPT=train_fast_viability.py; LAYERS=();; multilayer_all) ALGO=ppo; ARCH=multilayer_mean; SCRIPT=train_fast_viability.py; LAYERS=(--all-layers);; full_critic) ALGO=full_ppo; ARCH=final_single; SCRIPT=train_fast_full_critic.py; LAYERS=();; *) exit 2;; esac
OUT="runs/fast_smoke_${PBS_JOBID%%.*}/$METHOD"
COMMON=(--model Qwen/Qwen2.5-Math-1.5B-Instruct --algorithm "$ALGO" --critic-architecture "$ARCH" "${LAYERS[@]}" --train-file runs/viability_prepared/train.jsonl --prompt-key problem --answer-key gold --reward-type trl_math --output-dir "$OUT" --max-steps 2 --batch-size 8 --micro-batch-size 4 --group-size 4 --max-prompt-length 2048 --max-completion-length 128 --ppo-epochs 2 --critic-epochs 1 --learning-rate 5e-7 --critic-learning-rate 1e-4 --generation-temperature 1 --generation-top-p 1 --seed 42 --gradient-checkpointing --save-steps 1 --token-budget 100000 --verify-reward-parity)
/scratch/tsa5252/.conda/envs/grpo_env/bin/python "$SCRIPT" "${COMMON[@]}" --stop-after-step 1
/scratch/tsa5252/.conda/envs/grpo_env/bin/python "$SCRIPT" "${COMMON[@]}" --stop-after-step 2 --resume-training
