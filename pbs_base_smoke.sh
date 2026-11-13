#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -N mlc_base_smoke
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True FAST_VLLM_GPU_FRACTION=0.35
: "${METHOD:?}"
SCRIPT=train_base_viability.py; LAYERS=()
case "$METHOD" in full_critic) ALGORITHM=full_ppo; ARCH=final_single; SCRIPT=train_base_full_critic.py;; grpo) ALGORITHM=grpo; ARCH=multilayer_mean;; final_single) ALGORITHM=ppo; ARCH=final_single;; multilayer_all) ALGORITHM=ppo; ARCH=multilayer_mean; LAYERS=(--all-layers);; *) exit 2;; esac
python "$SCRIPT" --model Qwen/Qwen2.5-Math-1.5B --algorithm "$ALGORITHM" --critic-architecture "$ARCH" "${LAYERS[@]}" --train-file runs/base_rlvr/prepared/train.jsonl --prompt-key problem --answer-key gold --reward-type trl_math --output-dir "runs/base_rlvr/smoke/$METHOD" --max-steps 1 --stop-after-step 1 --batch-size 8 --micro-batch-size 4 --group-size 4 --max-prompt-length 2048 --max-completion-length 128 --ppo-epochs 1 --critic-epochs 1 --learning-rate 5e-7 --critic-learning-rate 1e-4 --generation-temperature 1.0 --generation-top-p 1.0 --seed 42 --gradient-checkpointing --token-budget 10000000 --verify-reward-parity --no-save
