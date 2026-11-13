#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=01:00:00
#PBS -q workq
#PBS -N mlc_vllm_eval_test
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" TRITON_CACHE_DIR=/scratch/tsa5252/.cache/triton
python evaluation/viability_vllm.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --output-dir runs/vllm_eval_validation --limit 16 --batch-size 16
