#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=100gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -N mlc_resident_train
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
export FLASHINFER_WORKSPACE_BASE=/scratch/tsa5252 TRITON_CACHE_DIR=/scratch/tsa5252/.cache/triton XDG_CACHE_HOME=/scratch/tsa5252/.cache XDG_CONFIG_HOME=/scratch/tsa5252/.config HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" VLLM_NO_USAGE_STATS=1 PYTORCH_ALLOC_CONF=expandable_segments:True PILOT_AMP=1 PILOT_SORT=1
export PILOT_OUT="runs/persistent_vllm_training_${PBS_JOBID%%.*}"
mkdir -p "$PILOT_OUT"
ln -s "$PWD/runs/vllm_full_update/rollouts.json" "$PILOT_OUT/rollouts.json"
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
 GPU_NUMBER=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F, -v target="$CUDA_VISIBLE_DEVICES" '{gsub(/^[ \t]+|[ \t]+$/, "", $2); if ($2 == target) print $1}')
 test -n "$GPU_NUMBER"; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_NUMBER"
fi
LD_LIBRARY_PATH="/scratch/tsa5252/.conda/envs/grpo_env/lib:${LD_LIBRARY_PATH:-}" /scratch/tsa5252/.venvs/vllm-pilot/bin/python experiments/persistent_vllm/hold.py &
WORKER_PID=$!
cleanup(){ touch "$PILOT_OUT/stop"; kill "$WORKER_PID" 2>/dev/null || true; wait "$WORKER_PID" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 300); do if [ -f "$PILOT_OUT/ready" ]; then break; fi; if ! kill -0 "$WORKER_PID" 2>/dev/null; then wait "$WORKER_PID"; exit 1; fi; sleep 1; done
test -f "$PILOT_OUT/ready"
/scratch/tsa5252/.conda/envs/grpo_env/bin/python experiments/vllm_pilot/full_update_train.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --algorithm ppo --critic-architecture multilayer_mean --all-layers --train-file runs/viability_prepared/train.jsonl --prompt-key problem --answer-key gold --reward-type trl_math --output-dir "$PILOT_OUT/train" --max-steps 1 --stop-after-step 1 --batch-size 64 --micro-batch-size 2 --group-size 4 --max-prompt-length 2048 --max-completion-length 2048 --ppo-epochs 2 --critic-epochs 1 --learning-rate 5e-7 --critic-learning-rate 1e-4 --generation-temperature 1 --generation-top-p 1 --seed 42 --gradient-checkpointing --token-budget 1000000 --verify-reward-parity --no-save
