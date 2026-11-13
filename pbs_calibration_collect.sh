#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -N mlc_calibration_states
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
: "${SPLIT:?}" "${RANK:?}" "${OUTDIR:?}"
python calibration_experiment/collect.py --checkpoint runs/deepscaler/restart_final_from300_seed43/checkpoint-100 --peer-critic runs/deepscaler/restart_multilayer_from200_seed43/checkpoint-200/critic.pt --input "runs/deepscaler/prepared/$SPLIT.jsonl" --output-dir "$OUTDIR" --rank "$RANK" --world-size "${WORLD_SIZE:-2}" --limit "${LIMIT:-50}" --rollouts "${ROLLOUTS:-32}" --batch-size 2
