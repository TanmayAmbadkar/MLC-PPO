#!/usr/bin/env bash
#PBS -N mlc_polaris_prepare
#PBS -q workq
#PBS -l select=1:ncpus=6:mem=48gb
#PBS -l walltime=03:00:00
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
exec python polaris_experiment/prepare.py
