#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -N mlc_fast_load
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_ALLOC_CONF=expandable_segments:True FAST_VLLM_GPU_FRACTION=0.35
/scratch/tsa5252/.conda/envs/grpo_env/bin/python train_fast_viability.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --algorithm ppo --critic-architecture multilayer_mean --all-layers --train-file runs/viability_prepared/train.jsonl --prompt-key problem --answer-key gold --reward-type trl_math --output-dir "runs/fast_load_${PBS_JOBID%%.*}" --max-steps 1 --stop-after-step 1 --batch-size 64 --micro-batch-size 4 --group-size 4 --max-prompt-length 2048 --max-completion-length 2048 --ppo-epochs 2 --critic-epochs 1 --learning-rate 5e-7 --critic-learning-rate 1e-4 --generation-temperature 1 --generation-top-p 1 --seed 42 --gradient-checkpointing --token-budget 100000 --verify-reward-parity --no-save
