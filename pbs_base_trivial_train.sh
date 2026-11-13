#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -N mlc_base_loo
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True FAST_VLLM_GPU_FRACTION=0.35
: "${MILESTONE:?}"
OUT=runs/base_rlvr/trivial_loo; mkdir -p "$OUT"
export BASE_OPTIMIZER_STATE_DIR=/tmp/tsa5252-base-rlvr-optimizer/trivial_loo
RESUME=(); [[ -f "$OUT/resume_meta.json" ]] && RESUME=(--resume-training)
python train_base_trivial_loo.py --model Qwen/Qwen2.5-Math-1.5B --algorithm ppo --critic-architecture final_single --train-file runs/base_rlvr/prepared/train.jsonl --prompt-key problem --answer-key gold --reward-type trl_math --output-dir "$OUT" --max-steps 1000 --stop-after-step 1000 --stage-token-limit "$MILESTONE" --batch-size 64 --micro-batch-size 4 --group-size 4 --max-prompt-length 2048 --max-completion-length 2048 --ppo-epochs 2 --critic-epochs 1 --learning-rate 5e-7 --critic-learning-rate 1e-4 --generation-temperature 1.0 --generation-top-p 1.0 --seed 42 --gradient-checkpointing --save-steps 100 --token-budget 10000000 --verify-reward-parity "${RESUME[@]}"
