#!/usr/bin/env bash
#PBS -N mlc_4b_lr_eval
#PBS -q workq
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=06:00:00
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
: "${RUN:?}"
exec python harder_4b/lr_evaluate.py
