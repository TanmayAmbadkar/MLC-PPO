#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -N mlc_calibration_gradient
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [ "${SMOKE:-0}" = 1 ]; then
 python calibration_experiment/gradient_variance.py --test-dir runs/deepscaler/calibration_shared/smoke --calibration calibration_experiment/smoke_parameters.json --max-problems 1 --samples-per-problem 2 --output runs/deepscaler/calibration_shared/gradient_smoke.json
else
 python calibration_experiment/gradient_variance.py
fi
