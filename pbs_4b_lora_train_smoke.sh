#!/usr/bin/env bash
#PBS -N mlc_4b_lora_smoke
#PBS -q workq
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=04:00:00
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True FAST_VLLM_GPU_FRACTION=0.62 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
RESUME=(); [[ "${STAGE:-1}" = 2 ]] && RESUME=(--resume-training)
python harder_4b/train.py --model Qwen/Qwen3-4B-Base --algorithm ppo --critic-architecture multilayer_mean --all-layers --train-file runs/harder_4b/screen/train_screened.jsonl --prompt-key problem --answer-key gold --reward-type trl_math --output-dir runs/harder_4b/lora_smoke --max-steps 2 --stop-after-step "${STAGE:-1}" --batch-size 2 --micro-batch-size 1 --group-size 4 --max-prompt-length 2048 --max-completion-length 8192 --ppo-epochs 1 --critic-epochs 1 --learning-rate 5e-6 --critic-learning-rate 1e-4 --generation-temperature 1 --generation-top-p 1 --seed 42 --gradient-checkpointing --save-steps 1 --token-budget 10000000 --verify-reward-parity "${RESUME[@]}"
