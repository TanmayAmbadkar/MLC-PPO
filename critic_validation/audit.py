#!/usr/bin/env python3
"""Audit saved critic outcomes and sensitivity to manually identified question defects."""
import json, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment2.reward import reward,require_validated_versions
from trl.rewards import accuracy_reward
root=Path('runs/deepscaler/critic_validation');out=root/'audit';out.mkdir(exist_ok=True);require_validated_versions();rng=random.Random(2718)
# Exclusions document concrete, manually inspected source questions; this is a diagnostic, not a curated benchmark.
issues={'776c2464827e':'Problem asks for polygon ABCDEF area but supplies no figure, dimensions or coordinates.', '587fa916c223':'No strictly increasing positive integer initial pair satisfies b9=544: b9=13b1+21b2; the positive solutions are (16,16) and (37,3), neither strictly increasing.', '0d2ba990a80f':'Three-part question; single gold answer covers at most one part, so verifier semantics do not represent full-task correctness.'}
results={};all_samples=[]
for name in ('final_ensemble','multilayer_mean'):
 rows=[json.loads(p.read_text()) for p in sorted((root/name).glob('*.json'))];selected=[]
 for group,fn in [('0_of_8',lambda n:n==0),('1_to_7_of_8',lambda n:0<n<8),('8_of_8',lambda n:n==8)]:
  group_rows=[r for r in rows if fn(sum(c['reward'] for c in r['difficulty_rollouts']))]
  for r in rng.sample(group_rows,min(4,len(group_rows))):
   outcomes=[(s,c) for s in r['states'] for c in s['continuations']];yes=[x for x in outcomes if x[1]['reward']==1];no=[x for x in outcomes if x[1]['reward']==0]
   selected.extend((group,r,s,c) for s,c in rng.sample(yes,min(2,len(yes)))+rng.sample(no,min(2,len(no))))
 for group,r,s,c in selected:
  score=reward(c['full_response'],r['gold']);trl=accuracy_reward([[{'role':'assistant','content':c['full_response']}]],solution=[r['gold']])[0]
  all_samples.append({'model':name,'group':group,'id':r['id'],'problem':r['problem'],'gold':r['gold'],'progress':s['progress'],'prediction':s['value'],'recorded_reward':c['reward'],'recomputed_reward':score,'trl_reward':trl,'parsed_answer':c['parsed_answer'],'ended_with_eos':c['ended_with_eos'],'full_response':c['full_response']})
 def metrics(items):
  s=[v for r in items for v in r['states']]
  if not s:return None
  pred=np.array([v['value'] for v in s]);obs=np.array([np.mean([c['reward'] for c in v['continuations']]) for v in s]);return {'problems':len(items),'states':len(s),'prediction':float(pred.mean()),'observed':float(obs.mean()),'bias':float((pred-obs).mean()),'mse':float(np.mean((pred-obs)**2)),'brier':float(np.mean([np.mean([(v['value']-c['reward'])**2 for c in v['continuations']]) for v in s]))}
 groups={}
 for label,fn in [('all',lambda n:True),('0_of_8',lambda n:n==0),('1_to_7_of_8',lambda n:0<n<8),('8_of_8',lambda n:n==8)]:
  subset=[r for r in rows if fn(sum(c['reward'] for c in r['difficulty_rollouts']))];clean=[r for r in subset if not any(r['id'].startswith(k) for k in issues)]
  groups[label]={'original':metrics(subset),'excluding_identified_defects':metrics(clean),'excluded_ids':[r['id'] for r in subset if r not in clean]}
 results[name]={'groups':groups}
summary={'sampling':'seed 2718; 4 problems per observed difficulty group per model, up to two correct and two incorrect continuations per selected problem; only available classes sampled','reviewed_label_policy':issues,'total_sampled_continuations':len(all_samples),'saved_vs_recomputed_disagreements':sum(s['recorded_reward']!=s['recomputed_reward'] for s in all_samples),'saved_vs_trl_disagreements':sum(s['recorded_reward']!=s['trl_reward'] for s in all_samples),'without_eos':sum(not s['ended_with_eos'] for s in all_samples),'models':results,'caveat':'Targeted post-hoc exclusions and unverified labels; does not establish dataset error rate, population calibration, or full correctness of verifier.'}
(out/'sampled_continuations.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in all_samples));(out/'audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
