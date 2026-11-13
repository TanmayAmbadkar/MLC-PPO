import json,statistics
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from experiment2.reward import reward,require_validated_versions
p=Path('runs/base_rlvr/pilot');rows=[json.loads(x) for x in (p/'raw.jsonl').read_text().splitlines()];tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B');require_validated_versions();groups={}
with (p/'scored.jsonl').open('w') as f:
 for r in rows:
  text=tok.decode(r.pop('token_ids'),skip_special_tokens=True);score=reward(text,r['gold']);r.update(text=text,reward=score,boxed='\\boxed' in text);f.write(json.dumps(r,ensure_ascii=False)+'\n');groups.setdefault(r['problem_id'],[]).append(score)
counts=Counter(int(sum(v)) for v in groups.values());report={'problems':len(groups),'responses':len(rows),'correct':sum(sum(v) for v in groups.values()),'accuracy':sum(sum(v) for v in groups.values())/len(rows),'groups_0_4':{str(k):counts[k] for k in range(5)},'mixed_groups':sum(counts[k] for k in (1,2,3)),'boxed_fraction':sum(r['boxed'] for r in rows)/len(rows),'truncated_fraction':sum(r['finish_reason']=='length' for r in rows)/len(rows)}
(p/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if report['correct']==0 or report['correct']==len(rows) or report['mixed_groups']==0:raise SystemExit('Insufficient mixed binary reward signal; training is not released')
