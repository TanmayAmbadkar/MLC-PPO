#!/usr/bin/env bash
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=01:00:00
#PBS -q workq
#PBS -N mlc_dapo_report
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
python experiment2/consolidated_report.py
