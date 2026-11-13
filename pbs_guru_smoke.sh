#!/usr/bin/env bash
#PBS -N mlc_guru_smoke
#PBS -q workq
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -l walltime=02:00:00
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True FAST_VLLM_GPU_FRACTION=0.35 STEM_LLM_JUDGE_URL=http://127.0.0.1:18419 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
for i in $(seq 1 180); do if curl -fsS http://127.0.0.1:18419/health >/dev/null 2>&1; then break; fi;sleep 2;done
curl -fsS http://127.0.0.1:18419/health >/dev/null
python guru_experiment/train.py --model "${GURU_MODEL:-Qwen/Qwen2.5-Math-1.5B}" --algorithm ppo --critic-architecture multilayer_mean --all-layers --train-file "${GURU_DATA:-runs/guru_1p5b/prepared/smoke.jsonl}" --prompt-key prompt --answer-key ground_truth --reward-type guru --output-dir "${GURU_OUT:-runs/guru_1p5b/smoke}" --max-steps 1 --stop-after-step 1 --batch-size "${GURU_BATCH:-6}" --micro-batch-size 1 --group-size 4 --max-prompt-length 2048 --max-completion-length 2048 --ppo-epochs 1 --critic-epochs 1 --learning-rate 5e-7 --critic-learning-rate 1e-4 --generation-temperature 1 --generation-top-p 1 --seed 42 --gradient-checkpointing --token-budget 10000000 --no-save
