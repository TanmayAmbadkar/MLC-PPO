#!/usr/bin/env python3
"""Add Math-Verify parser outputs to the saved reward-consistency audit."""
import json
from pathlib import Path
from math_verify import parse
from reward import PRED_CONFIG
root=Path('runs/experiment2/smoke_v2')
rows=[json.loads(line) for line in (root/'responses.jsonl').read_text().splitlines()]
with (root/'responses_parsed.jsonl').open('w') as f:
 for row in rows:
  row['parsed_gold']=[str(x) for x in parse(row['gold'],extraction_mode='first_match')]
  row['parsed_answer']=[str(x) for x in parse(row['completion'],extraction_config=PRED_CONFIG,extraction_mode='first_match')]
  f.write(json.dumps(row,ensure_ascii=False)+'\n')
print('saved parser output for',len(rows),'responses',flush=True)
