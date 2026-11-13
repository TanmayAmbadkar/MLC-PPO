#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q workq
#PBS -N mlc_fast2_eval
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
: "${METHOD:?}" "${STAGE:?}"
if [ "$METHOD" = base ]; then
 MODEL=Qwen/Qwen2.5-Math-1.5B-Instruct; OUT=runs/viability_fast2/evaluation/base
else
 META="runs/viability_fast2/$METHOD/resume_meta.json"
 STEP=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$META")
 if [ "$STEP" -le "$(((STAGE-1)*100))" ]; then printf 'No new checkpoint; budget reached earlier.\n'; exit 0; fi
 MODEL="runs/viability_fast2/$METHOD/checkpoint-$STEP"; OUT="runs/viability_fast2/evaluation/$METHOD/step-$STEP"
 mkdir -p "$OUT"; cp "$META" "$OUT/training_progress.json"
fi
python evaluation/viability_vllm.py --model "$MODEL" --output-dir "$OUT" --limit "${EVAL_LIMIT:-0}" --batch-size 32
# Keep the two latest evaluated checkpoints for recovery; older policy weights
# are redundant once their benchmark records and metrics are on disk.
if [ "$METHOD" != base ]; then
 python - "$METHOD" "$STEP" <<'PY'
import re,shutil,sys
from pathlib import Path
root=Path('runs/viability_fast2')/sys.argv[1]; current=int(sys.argv[2]); checkpoints=[]
for path in root.glob('checkpoint-*'):
 m=re.fullmatch(r'checkpoint-(\d+)',path.name)
 if m and path.is_dir() and int(m.group(1))<=current:checkpoints.append((int(m.group(1)),path))
for step,path in sorted(checkpoints,reverse=True)[2:]:
 if not (Path('runs/viability_fast2/evaluation')/sys.argv[1]/f'step-{step}'/'summary.json').is_file():
  raise RuntimeError(f'Refusing to delete unevaluated checkpoint {path}')
 shutil.rmtree(path)
PY
fi
