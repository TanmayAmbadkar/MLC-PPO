import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from experiment2.reward import reward,require_validated_versions
root=Path('runs/polaris_4b/screen');tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Base',revision='906bfd4b4dc7f14ee4320094d8b41684abff8539');require_validated_versions();groups={}
with (root/'scored.jsonl').open('w') as out:
 for line in (root/'raw.jsonl').read_text().splitlines():
  r=json.loads(line);r['text']=tok.decode(r.pop('token_ids'),skip_special_tokens=True);r['reward']=reward(r['text'],r['gold']);out.write(json.dumps(r,ensure_ascii=False)+'\n');groups.setdefault(r['problem_id'],[]).append(r)
counts=Counter();selected=[]
for pid,rs in groups.items():
 if len(rs)!=8:raise RuntimeError(f'Incomplete group {pid}')
 n=int(sum(r['reward'] for r in rs));counts[n]+=1
 if 2<=n<=6:selected.append({'id':pid,'problem':rs[0]['problem'],'gold':rs[0]['gold'],'screen_successes':n,'screen_rollouts':8})
(root/'train_screened.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected));(root/'summary.json').write_text(json.dumps({'candidate_problems':len(groups),'screen_counts':dict(sorted(counts.items())),'selected':len(selected),'rule':'2–6 successes in 8 independent screening rollouts'},indent=2));print((root/'summary.json').read_text())

if len(selected) < 128: raise SystemExit('Fewer than 128 partially successful problems; comparison not released')
