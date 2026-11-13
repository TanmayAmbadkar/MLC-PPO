#!/usr/bin/env python3
"""Summarize evaluated training checkpoints and problem-paired differences."""
import json,re
from pathlib import Path
from itertools import combinations
import numpy as np
root=Path('runs/viability_fast');names=['base','grpo','final_single','multilayer_all','full_critic'];records={};report={'methods':{},'paired_final':{},'note':'Single-seed viability study. Training stops at the 40M generated-response-token budget (with batch-boundary overshoot) or 1000 updates, whichever comes first. Different prompt group structures and token lengths remain. Evals use greedy decoding; AIME has 30 problems. No paraphrase/pretraining decontamination guarantee.'}
for name in names:
 base=root/'evaluation'/name
 if name=='base':
  dirs=[base]
 else:
  dirs=sorted((p for p in base.glob('step-*') if (p/'summary.json').exists()),key=lambda p:int(p.name.split('-')[-1]))
 if not dirs:raise ValueError(f'No finished evaluations for {name}')
 series=[]
 for path in dirs:
  ev=json.loads((path/'summary.json').read_text());meta=json.loads((path/'training_progress.json').read_text()) if name!='base' else {'step':0,'tokens':0}
  series.append({'step':meta['step'],'generated_training_tokens':meta['tokens'],'benchmarks':ev['benchmarks']})
  for benchmark in ('math500','aime2024'):
   rows=[json.loads(f.read_text()) for f in sorted((path/benchmark).glob('[0-9][0-9][0-9][0-9].json'))]; records[(name,benchmark)]={r['id']:r for r in rows}
 info={'evaluations':series}
 if name!='base':
  mpath=root/name/'metrics.jsonl';metrics=[json.loads(line) for line in mpath.read_text().splitlines()];resources=[json.loads(p.read_text()) for p in (root/name).glob('job-resource-*.json')]
  info['training']={'updates':series[-1]['step'],'generated_tokens':series[-1]['generated_training_tokens'],'peak_gpu_allocated_sum_of_device_peaks_bytes':max(m['peak_gpu_bytes'] for m in metrics),'training_gpu_hours':sum(r['wall_seconds']*r['gpu_count'] for r in resources)/3600,'training_job_count':len(resources),'gpu_hours_scope':'training jobs only; excludes benchmark evaluation','zero_group_fraction_mean':float(np.mean([m['zero_group_fraction'] for m in metrics if m['zero_group_fraction'] is not None])) if any(m['zero_group_fraction'] is not None for m in metrics) else None}
 report['methods'][name]=info
rng=np.random.default_rng(42)
for benchmark in ('math500','aime2024'):
 for first,second in combinations(names,2):
  left,right=records[first,benchmark],records[second,benchmark]
  if set(left)!=set(right):raise ValueError('Mismatched benchmark problem ids')
  ids=sorted(i for i in left if left[i]['status']==right[i]['status']=='scored')
  if any(left[i]['verifier_gold']!=right[i]['verifier_gold'] for i in ids):raise ValueError('Gold mismatch')
  dif=np.array([left[i]['reward']-right[i]['reward'] for i in ids]);draw=rng.integers(0,len(ids),size=(5000,len(ids)))
  report['paired_final'][f'{benchmark}:{first}_minus_{second}']={'problems':len(ids),'accuracy_difference':float(dif.mean()),'problem_bootstrap_95ci':np.quantile(dif[draw].mean(1),[.025,.975]).tolist()}
(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
