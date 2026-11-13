#!/usr/bin/env bash
#PBS -N mlc_pol_eval_smoke
#PBS -q workq
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=03:00:00
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python polaris_experiment/evaluate.py --adapter runs/harder_4b/lr_pilot/lr_1e-5/checkpoint-20 --output-dir runs/polaris_4b/eval_smoke --limit 2
