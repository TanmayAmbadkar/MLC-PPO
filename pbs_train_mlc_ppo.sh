#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q workq
#PBS -N gsm8k_mlc_ppo
#PBS -M tsa5252@psu.edu
#PBS -m bea

set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"

source ~/.bashrc2
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env

export HF_HOME=/scratch/tsa5252/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-data/gsm8k/train.jsonl}"
OUTDIR="${OUTDIR:-runs/gsm8k_qwen25_15b_mlc_ppo_pilot}"

PREP_ARGS=(--output-file "$TRAIN_FILE" --split train)
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
  PREP_ARGS+=(--max-examples "$MAX_EXAMPLES")
fi
python prepare_gsm8k.py "${PREP_ARGS[@]}"

python train_mlc_ppo.py \
  --model "$MODEL" \
  --train-file "$TRAIN_FILE" \
  --output-dir "$OUTDIR" \
  --max-steps "${MAX_STEPS:-100}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --learning-rate "${LEARNING_RATE:-5e-7}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH:-512}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH:-512}" \
  --critic-layers "${CRITIC_LAYERS:-0.5,0.75,1.0}" \
  --gradient-checkpointing
