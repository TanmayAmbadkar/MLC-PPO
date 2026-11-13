#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q workq
#PBS -N mlc_base_eval
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
: "${METHOD:?}"
if [[ "$METHOD" = base ]]; then MODEL=Qwen/Qwen2.5-Math-1.5B; OUT=runs/base_rlvr/evaluation/base; else
 META="runs/base_rlvr/$METHOD/resume_meta.json"; STEP=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["step"])' "$META")
 MODEL="runs/base_rlvr/$METHOD/checkpoint-$STEP"; OUT="runs/base_rlvr/evaluation/$METHOD/step-$STEP"; mkdir -p "$OUT"; cp "$META" "$OUT/training_progress.json"
fi
python base_experiment/evaluate.py --model "$MODEL" --output-dir "$OUT" --limit "${EVAL_LIMIT:-0}" --batch-size 32
if [[ "$METHOD" != base ]]; then
 python - "$METHOD" "$STEP" <<'PY'
import re,shutil,sys
from pathlib import Path
method,current=sys.argv[1],int(sys.argv[2]);root=Path('runs/base_rlvr')/method
for p in root.glob('checkpoint-*'):
 m=re.fullmatch(r'checkpoint-(\d+)',p.name)
 if m and int(m.group(1))<current and (Path('runs/base_rlvr/evaluation')/method/f'step-{m.group(1)}'/'summary.json').is_file():shutil.rmtree(p)
PY
fi
