#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=06:00:00
#PBS -q workq
#PBS -N mlc_dapo_critics
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
python experiment2/analyze.py --difficulty-file runs/experiment2/difficulty/difficulty_map.json --require-counts
