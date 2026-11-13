#!/usr/bin/env python3
"""Build reproducible, conservatively scoreable hard-math evaluation subsets."""
import json, re, sys, unicodedata
from pathlib import Path
from collections import Counter
from datasets import load_dataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment2.reward import valid_gold, require_validated_versions

ROOT=Path('runs/hard_math_eval');ROOT.mkdir(parents=True,exist_ok=True)
SPECS={'olympiad':('Hothan/OlympiadBench','OE_TO_maths_en_COMP','train','91184b52131e7fc9455fef848035173aea8cc01a'), 'minerva':('svc-huggingface/minerva-math','default','test','aec85355cf57a4337f73d1bb03233288cb0344df')}
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).strip()).casefold()
def boxed(solution):
 matches=list(re.finditer(r'\\boxed\s*\{',solution))
 if not matches:return None
 m=matches[-1];depth=1;start=m.end()
 for i in range(start,len(solution)):
  if solution[i]=='{':depth+=1
  elif solution[i]=='}':
   depth-=1
   if depth==0:return solution[start:i]
 return None
train=[json.loads(x) for x in Path('runs/experiment2/prepared/train.jsonl').read_text().splitlines() if x.strip()]
keys={norm(x['problem']) for x in train}; require_validated_versions()
for name,(repo,config,split,revision) in SPECS.items():
 ds=load_dataset(repo,config,split=split,revision=revision);rows=[];excluded=[]
 for index,x in enumerate(ds):
  problem=x['question'] if name=='olympiad' else x['problem'];reason=None
  if name=='olympiad':
   if x.get('is_multiple_answer') or len(x.get('final_answer') or [])!=1:reason='multiple_or_missing_answers'
   answer=(x.get('final_answer') or [None])[0]
  else:
   answer=boxed(x['solution'])
   if not answer:reason='missing_boxed_gold'
  if norm(problem) in keys:reason='exact_or_normalized_training_overlap'
  if answer:
   gold=str(answer).strip()
   if not gold.startswith(('$',r'\[',r'\(')) and r'\boxed' not in gold:gold='$'+gold+'$'
   if not valid_gold(gold):reason=reason or 'unparseable_gold'
  else:gold=''
  row={'id':str(x.get('id',x.get('idx',index))), 'problem':problem,'gold':gold,'original_answer':answer,'source_index':index}
  if reason:row['reason']=reason;excluded.append(row)
  else:rows.append(row)
 (ROOT/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 (ROOT/f'{name}_excluded.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in excluded))
 report={'dataset':repo,'configuration':config,'split':split,'revision':revision,'total':len(ds),'included':len(rows),'excluded':len(excluded),'exclusion_reasons':dict(Counter(r['reason'] for r in excluded)),'note':'Conservative subset; normalized exact comparison only against actual 1000-problem PPO training split, no paraphrase or pretraining contamination check. Multi-answer Olympiad items excluded. Math-Verify parsing does not certify semantic correctness.'}
 (ROOT/f'{name}_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
