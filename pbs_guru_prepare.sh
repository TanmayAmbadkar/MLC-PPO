#!/usr/bin/env bash
#PBS -N mlc_guru_prepare
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=24gb
#PBS -l walltime=02:00:00
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1
python guru_experiment/prepare.py
