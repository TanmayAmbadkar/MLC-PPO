#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -q workq
#PBS -N mlc_gradient_smoke
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
python experiment2/gradient_data.py --manifest-dir runs/experiment2/critic_smoke/manifest \
 --output-dir runs/experiment2/critic_smoke/gradient_data --max-problems 3 --max-new-tokens 128
python experiment2/gradient_variance.py --difficulty-dir runs/experiment2/critic_smoke/gradient_data \
 --results-dir runs/experiment2/critic_smoke/results --max-problems 3 --batch-size 1
