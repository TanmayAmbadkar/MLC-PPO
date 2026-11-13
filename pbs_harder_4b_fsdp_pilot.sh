#!/usr/bin/env bash
#PBS -N mlc_4b_fsdp
#PBS -q workq
#PBS -l select=1:ncpus=16:mem=140gb:ngpus=4
#PBS -l walltime=06:00:00
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" MODEL_REVISION=906bfd4b4dc7f14ee4320094d8b41684abff8539
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + ${PBS_JOBID%%.*} % 30000))
exec torchrun --nnodes=1 --nproc_per_node=4 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" harder_4b/fsdp_lora_pilot.py
