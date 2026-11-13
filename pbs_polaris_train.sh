#!/usr/bin/env bash
#PBS -N mlc_pol_train
#PBS -q workq
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=48:00:00
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True FAST_VLLM_GPU_FRACTION=0.62 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
: "${METHOD:?}" "${SEED:?}" "${MILESTONE:?}"
SCRIPT=harder_4b/train.py; LAYERS=()
case "$METHOD" in
 trivial) SCRIPT=harder_4b/train_trivial.py; ALGO=ppo;ARCH=final_single;;
 final) ALGO=ppo;ARCH=final_single;;
 multilayer) ALGO=ppo;ARCH=multilayer_mean;LAYERS=(--all-layers);;
 grpo) ALGO=grpo;ARCH=multilayer_mean;;
 full_critic) SCRIPT=harder_4b/train_full_critic.py;ALGO=full_ppo;ARCH=final_single;;
 *) exit 2;;
esac
OUT="runs/polaris_4b/$METHOD/seed-$SEED";mkdir -p "$OUT";
START_SECONDS=$SECONDS
GPU_COUNT=1; [[ "$METHOD" = full_critic ]] && GPU_COUNT=2
record_resources() { code=$?; printf '{ "job_id":"%s", "method":"%s", "seed":%s, "milestone":%s, "wall_seconds":%s, "gpu_count":%s, "exit_status":%s }\n' "${PBS_JOBID:-local}" "$METHOD" "$SEED" "$MILESTONE" "$((SECONDS-START_SECONDS))" "$GPU_COUNT" "$code" > "$OUT/resource-${PBS_JOBID:-local}.json"; }
trap record_resources EXIT
export BASE_OPTIMIZER_STATE_DIR="/tmp/tsa5252-polaris-optim/$METHOD/seed-$SEED"
RESUME=(); [[ -f "$OUT/resume_meta.json" ]] && RESUME=(--resume-training)
python "$SCRIPT" --model Qwen/Qwen3-4B-Base --algorithm "$ALGO" --critic-architecture "$ARCH" "${LAYERS[@]}" --train-file runs/polaris_4b/screen/train_screened.jsonl --prompt-key problem --answer-key gold --reward-type trl_math --output-dir "$OUT" --max-steps 10000 --stop-after-step 10000 --stage-token-limit "$MILESTONE" --batch-size 64 --micro-batch-size 1 --group-size 4 --max-prompt-length 2048 --max-completion-length 8192 --ppo-epochs 2 --critic-epochs 1 --learning-rate 1e-5 --critic-learning-rate 1e-4 --generation-temperature 1.0 --generation-top-p 1.0 --seed "$SEED" --gradient-checkpointing --save-steps 100 --token-budget 10000000 --verify-reward-parity "${RESUME[@]}"
