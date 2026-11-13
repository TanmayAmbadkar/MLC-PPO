#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=100gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -q workq
#PBS -N mlc_vllm_update
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
export TRITON_CACHE_DIR=/scratch/tsa5252/.cache/triton FLASHINFER_WORKSPACE_BASE=/scratch/tsa5252 XDG_CACHE_HOME=/scratch/tsa5252/.cache XDG_CONFIG_HOME=/scratch/tsa5252/.config HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_ALLOC_CONF=expandable_segments:True
export PILOT_BATCH_SIZE="${PILOT_BATCH_SIZE:-64}" PILOT_MAX_TOKENS="${PILOT_MAX_TOKENS:-2048}" PILOT_OUT="${PILOT_OUT:-runs/vllm_full_update}"
mkdir -p "$PILOT_OUT"
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
 GPU_NUMBER=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F, -v target="$CUDA_VISIBLE_DEVICES" '{gsub(/^[ \t]+|[ \t]+$/, "", $2); if ($2 == target) print $1}')
 test -n "$GPU_NUMBER"
 export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_NUMBER"
fi
START_NS=$(date +%s%N)
LD_LIBRARY_PATH="/scratch/tsa5252/.conda/envs/grpo_env/lib:${LD_LIBRARY_PATH:-}" /scratch/tsa5252/.venvs/vllm-pilot/bin/python experiments/vllm_pilot/full_update_generate.py
GENERATED_NS=$(date +%s%N)
/scratch/tsa5252/.conda/envs/grpo_env/bin/python experiments/vllm_pilot/full_update_train.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --algorithm ppo --critic-architecture multilayer_mean --all-layers --train-file runs/viability_prepared/train.jsonl --prompt-key problem --answer-key gold --reward-type trl_math --output-dir "$PILOT_OUT/train" --max-steps 1 --stop-after-step 1 --batch-size "$PILOT_BATCH_SIZE" --micro-batch-size 4 --group-size 4 --max-prompt-length 2048 --max-completion-length "$PILOT_MAX_TOKENS" --ppo-epochs 2 --critic-epochs 1 --learning-rate 5e-7 --critic-learning-rate 1e-4 --generation-temperature 1 --generation-top-p 1 --seed 42 --gradient-checkpointing --token-budget 1000000 --verify-reward-parity --no-save
FINISHED_NS=$(date +%s%N)
/scratch/tsa5252/.conda/envs/grpo_env/bin/python - "$PILOT_OUT" "$START_NS" "$GENERATED_NS" "$FINISHED_NS" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); a,b,c=map(int,sys.argv[2:]);(out/'wall_times.json').write_text(json.dumps({'generation_process_seconds':(b-a)/1e9,'training_process_seconds':(c-b)/1e9,'total_seconds':(c-a)/1e9},indent=2))
PY
