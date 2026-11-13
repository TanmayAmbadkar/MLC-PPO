#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -q workq
#PBS -N mlc_critic_smoke
set -euo pipefail
cd "${PBS_O_WORKDIR:-/scratch/tsa5252/RLVR-MLCPPO}"
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env
export HF_HOME=/scratch/tsa5252/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
python - <<'PY'
import json
from pathlib import Path
root=Path('runs/experiment2/critic_smoke_v2/manifest');root.mkdir(parents=True,exist_ok=True)
rows=[json.loads(x) for x in Path('runs/experiment2/smoke_v2/responses.jsonl').read_text().splitlines()][:9]
if len(rows)!=9 or len({r['problem_id'] for r in rows})!=9:raise ValueError('Need nine distinct smoke problems')
for split,subset in [('train',rows[:3]),('validation',rows[3:6]),('test',rows[6:9])]:
 with (root/(split+'.jsonl')).open('w') as f:
  for r in subset:f.write(json.dumps({'id':r['problem_id'],'problem':r['problem'],'gold':r['gold']})+'\n')
PY
python experiment2/collect.py --manifest-dir runs/experiment2/critic_smoke_v2/manifest \
  --output-dir runs/experiment2/critic_smoke_v2/states --trajectories 1 \
  --train-rollouts 2 --validation-rollouts 2 --test-rollouts 2 \
  --batch-size 2 --max-new-tokens 128
python experiment2/analyze.py --data-dir runs/experiment2/critic_smoke_v2/states \
  --output-dir runs/experiment2/critic_smoke_v2/results --seeds 42 --epochs 2 --bootstrap 50
python experiment2/gradient_data.py --manifest-dir runs/experiment2/critic_smoke_v2/manifest \
  --output-dir runs/experiment2/critic_smoke_v2/gradient_data --max-problems 3 --max-new-tokens 128
python experiment2/gradient_variance.py --difficulty-dir runs/experiment2/critic_smoke_v2/gradient_data \
  --results-dir runs/experiment2/critic_smoke_v2/results --max-problems 3 --batch-size 1
