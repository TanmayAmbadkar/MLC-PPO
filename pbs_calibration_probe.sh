#!/usr/bin/env bash
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -N mlc_calibration_probe
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python calibration_experiment/logistic_probe.py
