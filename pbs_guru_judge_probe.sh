#!/usr/bin/env bash
#PBS -N mlc_guru_judge
#PBS -q workq
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -l walltime=02:00:00
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
export FLASHINFER_WORKSPACE_BASE=/scratch/tsa5252 HF_HOME=/scratch/tsa5252/.cache/huggingface XDG_CACHE_HOME=/scratch/tsa5252/.cache TRITON_CACHE_DIR=/scratch/tsa5252/.cache/triton VLLM_HOST_IP=127.0.0.1 VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
 GPU_NUMBER=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F, -v target="$CUDA_VISIBLE_DEVICES" '{gsub(/^[ \t]+|[ \t]+$/, "", $2); if ($2 == target) print $1}');test -n "$GPU_NUMBER";export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_NUMBER"
fi
export LD_LIBRARY_PATH="/scratch/tsa5252/.conda/envs/grpo_env/lib:${LD_LIBRARY_PATH:-}"
/scratch/tsa5252/.venvs/vllm-pilot/bin/vllm serve TIGER-Lab/general-verifier --revision dc59c0c82360170cab0be91c91ae545ff32ea896 --host 127.0.0.1 --port 18419 --max-model-len 4096 --gpu-memory-utilization 0.8 > runs/guru_1p5b/preparation/judge_server.log 2>&1 &
pid=$!;trap 'kill "$pid" 2>/dev/null || true' EXIT
for i in $(seq 1 180); do if curl -fsS http://127.0.0.1:18419/health >/dev/null; then break; fi; if ! kill -0 "$pid" 2>/dev/null; then echo 'Judge server failed'; exit 1; fi; sleep 2; done
curl -fsS http://127.0.0.1:18419/health >/dev/null
/scratch/tsa5252/.conda/envs/grpo_env/bin/python guru_experiment/judge_probe.py
