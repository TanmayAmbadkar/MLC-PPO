#!/usr/bin/env bash
#PBS -N mlc_pol_eval
#PBS -q workq
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=24:00:00
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
: "${METHOD:?}" "${SEED:?}"
if [[ "$METHOD" == base ]]; then ADAPTER=none;OUT=runs/polaris_4b/evaluation/base
else
 ROOT="runs/polaris_4b/$METHOD/seed-$SEED";META="$ROOT/resume_meta.json";STEP=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["step"])' "$META")
 ADAPTER="$ROOT/checkpoint-$STEP";OUT="runs/polaris_4b/evaluation/$METHOD/seed-$SEED/step-$STEP";mkdir -p "$OUT";cp "$META" "$OUT/training_progress.json"
fi
python polaris_experiment/evaluate.py --adapter "$ADAPTER" --output-dir "$OUT"
if [[ "$METHOD" != base ]]; then
 python - "$METHOD" "$SEED" "$STEP" <<'PY'
import re,shutil,sys
from pathlib import Path
method,seed,current=sys.argv[1],sys.argv[2],int(sys.argv[3]);root=Path('runs/polaris_4b')/method/f'seed-{seed}'
for p in root.glob('checkpoint-*'):
 m=re.fullmatch(r'checkpoint-(\d+)',p.name)
 if m and int(m.group(1))<current and (Path('runs/polaris_4b/evaluation')/method/f'seed-{seed}'/f'step-{m.group(1)}'/'summary.json').is_file():shutil.rmtree(p)
PY
fi
