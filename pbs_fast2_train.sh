#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -N mlc_fast2_train
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True FAST_VLLM_GPU_FRACTION=0.35
: "${METHOD:?}" "${STAGE:?}"
TRAIN_SCRIPT=train_fast_viability.py
GPU_COUNT=1
case "$METHOD" in full_critic) ALGORITHM=full_ppo; ARCH=final_single; TRAIN_SCRIPT=train_fast_full_critic.py; GPU_COUNT=2;; grpo) ALGORITHM=grpo; ARCH=multilayer_mean;; final_single|multilayer_mean) ALGORITHM=ppo; ARCH=$METHOD;; multilayer_all) ALGORITHM=ppo; ARCH=multilayer_mean;; *) exit 2;; esac
LAYERS=()
if [ "$METHOD" = multilayer_all ]; then LAYERS=(--all-layers); fi
OUT="runs/viability_fast2/$METHOD"; mkdir -p "$OUT"
START_SECONDS=$SECONDS
record_resources() {
 code=$?
 printf '{"job_id":"%s","method":"%s","stage":%s,"wall_seconds":%s,"gpu_count":%s,"exit_status":%s}\n' "${PBS_JOBID:-local}" "$METHOD" "$STAGE" "$((SECONDS-START_SECONDS))" "$GPU_COUNT" "$code" > "$OUT/job-resource-${PBS_JOBID:-local}.json"
}
trap record_resources EXIT
RESUME=()
if [ "$STAGE" -gt 1 ]; then
 test -f "$OUT/resume_meta.json"
 if python -c 'import json,sys; d=json.load(open(sys.argv[1]));sys.exit(0 if d["tokens"]>=40000000 or d["step"]>=1000 else 1)' "$OUT/resume_meta.json"; then printf 'Budget or update limit already reached.\n'; exit 0; fi
 RESUME=(--resume-training)
fi
python "$TRAIN_SCRIPT" --model Qwen/Qwen2.5-Math-1.5B-Instruct --algorithm "$ALGORITHM" --critic-architecture "$ARCH" "${LAYERS[@]}" \
 --train-file runs/viability_prepared/train.jsonl --prompt-key problem --answer-key gold --reward-type trl_math \
 --output-dir "$OUT" --max-steps 1000 --stop-after-step "$((STAGE*100))" --batch-size 64 --micro-batch-size 4 --group-size 4 \
 --max-prompt-length 2048 --max-completion-length 2048 --ppo-epochs 2 --critic-epochs 1 \
 --learning-rate 5e-7 --critic-learning-rate 1e-4 --generation-temperature 1.0 --generation-top-p 1.0 \
 --seed 42 --gradient-checkpointing --save-steps 100 --token-budget 40000000 --verify-reward-parity "${RESUME[@]}"
