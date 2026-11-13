#!/usr/bin/env bash
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=01:00:00
#PBS -q workq
#PBS -N mlc_state_checks
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1
python experiment2/check_collected_states.py
