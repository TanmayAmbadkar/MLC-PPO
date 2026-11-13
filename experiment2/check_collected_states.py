#!/usr/bin/env python3
"""Validate stored prefix indexing, token reconstruction, labels, and split isolation."""
import json, math
from pathlib import Path
import torch
from transformers import AutoTokenizer
from reward import reward
root=Path('runs/experiment2/critic_smoke_v2/states');config=json.loads((root/'config.json').read_text())
tok=AutoTokenizer.from_pretrained(config['model']);seen={};counts={'problems':0,'states':0,'continuations':0}
for split in ('train','val','test'):
 for path in sorted((root/split).glob('*.json')):
  meta=json.loads(path.read_text());pid=meta['id']
  if pid in seen:raise AssertionError(f'Problem appears in both {seen[pid]} and {split}')
  seen[pid]=split;states=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False)['states']
  initial={r['trajectory']:r['token_ids'] for r in meta['initial_trajectories']}
  if len(states)!=len(meta['states']):raise AssertionError('Feature and audit state counts differ')
  for tensor_state,audit in zip(states,meta['states']):
   original=initial[audit['trajectory']];eos=meta['generation']['eos_token_id']
   if eos in original:original=original[:original.index(eos)]
   expected=max(1,int(len(original)*audit['progress']))
   if audit['prefix_token_ids']!=original[:expected] or tensor_state['prefix_length']!=expected:raise AssertionError('Incorrect prefix token index')
   if tensor_state['progress']!=audit['progress'] or tensor_state['trajectory']!=audit['trajectory']:raise AssertionError('State metadata mismatch')
   if tensor_state['features'].shape[0]!=3:raise AssertionError('Expected three layer activations')
   for j,c in enumerate(audit['continuations']):
    full=tok.decode(audit['prefix_token_ids']+c['token_ids'],skip_special_tokens=True)
    if full!=c['full_response']:raise AssertionError('Continuation token reconstruction differs')
    if reward(full,meta['gold'])!=c['reward'] or c['reward']!=tensor_state['rewards'][j].item():raise AssertionError('Stored reward differs from shared verifier')
    counts['continuations']+=1
   counts['states']+=1
  counts['problems']+=1
report={'passed':True,'counts':counts,'checks':['disjoint problems','prefix index','feature and audit alignment','token reconstruction','reward consistency']}
(root.parent/'state_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
