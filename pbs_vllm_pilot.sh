#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -N mlc_vllm_pilot
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
export FLASHINFER_WORKSPACE_BASE=/scratch/tsa5252 XDG_CACHE_HOME=/scratch/tsa5252/.cache HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# vLLM 0.17.1 expects numeric CUDA_VISIBLE_DEVICES, while PBS gives a UUID.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
 GPU_NUMBER=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F, -v target="$CUDA_VISIBLE_DEVICES" '{gsub(/^[ \t]+|[ \t]+$/, "", $2); if ($2 == target) print $1}')
 test -n "$GPU_NUMBER"
 export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_NUMBER"
fi
LD_LIBRARY_PATH="/scratch/tsa5252/.conda/envs/grpo_env/lib:${LD_LIBRARY_PATH:-}" /scratch/tsa5252/.venvs/vllm-pilot/bin/python experiments/vllm_pilot/generate.py
/scratch/tsa5252/.conda/envs/grpo_env/bin/python experiments/vllm_pilot/compare.py
