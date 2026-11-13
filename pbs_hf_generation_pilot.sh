#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=01:00:00
#PBS -q workq
#PBS -N mlc_hf_gen_pilot
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
/scratch/tsa5252/.conda/envs/grpo_env/bin/python experiments/vllm_pilot/hf_baseline.py
