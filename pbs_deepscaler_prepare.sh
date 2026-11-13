#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=48gb
#PBS -l walltime=04:00:00
#PBS -q workq
#PBS -N mlc_deepscaler_prepare
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1
python deepscaler/prepare.py
