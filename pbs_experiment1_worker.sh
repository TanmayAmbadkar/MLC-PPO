#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -N mlc_value_worker
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
: "${WORKER_RANK:?Set WORKER_RANK to 0, 1, 2, or 3}"
: "${MARKER_DIR:?Set MARKER_DIR to the marker directory for this submission}"
python experiment1/collect.py --output-dir "${DATA_DIR:-runs/experiment1/data}" \
  --batch-size "${BATCH_SIZE:-8}" --rank "$WORKER_RANK" --world-size 4
mkdir -p "$MARKER_DIR"
touch "$MARKER_DIR/done-$WORKER_RANK"
if [[ "$WORKER_RANK" == 0 ]]; then
  # Leave time for fitting before PBS walltime expires.
  deadline=$((SECONDS + 165600))
  until [[ -f "$MARKER_DIR/done-0" && -f "$MARKER_DIR/done-1" && -f "$MARKER_DIR/done-2" && -f "$MARKER_DIR/done-3" ]]; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for other workers; completed data is preserved." >&2
      exit 1
    fi
    sleep 30
  done
  python experiment1/fit.py --data-dir "${DATA_DIR:-runs/experiment1/data}" \
    --output-dir "${RESULTS_DIR:-runs/experiment1/results}"
fi
