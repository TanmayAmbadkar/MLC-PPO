#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -N mlc_chunk_bench
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
: "${METHOD:?}"
LAYERS=(); SCRIPT=throughput_experiments/train_chunked.py
case "$METHOD" in grpo) ALGORITHM=grpo; ARCH=multilayer_mean;; final_single) ALGORITHM=ppo; ARCH=final_single;; multilayer_all) ALGORITHM=ppo; ARCH=multilayer_mean; LAYERS=(--all-layers);; full_critic) ALGORITHM=full_ppo; ARCH=final_single; SCRIPT=throughput_experiments/train_full_chunked.py;; *) exit 2;; esac
python "$SCRIPT" --model Qwen/Qwen2.5-Math-1.5B-Instruct --algorithm "$ALGORITHM" --critic-architecture "$ARCH" "${LAYERS[@]}" --train-file runs/viability_prepared/train.jsonl --prompt-key problem --answer-key gold --reward-type trl_math --output-dir "runs/throughput_benchmark_chunked/$METHOD" --max-steps 1 --stop-after-step 1 --batch-size 8 --micro-batch-size 4 --group-size 4 --max-prompt-length 2048 --max-completion-length 2048 --ppo-epochs 2 --critic-epochs 1 --learning-rate 5e-7 --critic-learning-rate 1e-4 --generation-temperature 1 --generation-top-p 1 --seed 99 --gradient-checkpointing --token-budget 100000 --verify-reward-parity --no-save --benchmark-synthetic-rewards
