#!/usr/bin/env python3
"""Prepare disjoint DAPO problem splits, excluding exact/normalized benchmark overlaps."""
import hashlib, json, random, re, unicodedata
from pathlib import Path
from datasets import load_dataset
from experiment2.reward import valid_gold
from math_verify import parse, verify

ROOT=Path('runs/experiment2'); OUT=Path('runs/viability_prepared');OUT.mkdir(parents=True,exist_ok=True)
def norm(s): return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).strip()).casefold()
def question(s):
    prefix='Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'
    text=s[len(prefix):].strip() if s.startswith(prefix) else s.strip()
    return re.sub(r'\s*Remember to put your answer on its own line after "Answer:"\.\s*$', '', text).strip()
def field(r):
    for k in ('problem','question','Question','Problem'):
        if isinstance(r.get(k),str): return r[k]
    raise ValueError(f'Unknown benchmark fields: {list(r)}')
bench=[('HuggingFaceH4/MATH-500','test'),('HuggingFaceH4/aime_2024','train'),('math-ai/aime25','test')]
indices={name:{norm(field(row)) for row in load_dataset(name,split=split)} for name,split in bench}
raw_indices={name:{field(row) for row in load_dataset(name,split=split)} for name,split in bench}
source=load_dataset('BytedTsinghua-SIA/DAPO-Math-17k',split='train',streaming=True,revision='65877096c24ffa7abc4e4fa5edb95cf3413a5674')
seen=set(); gold_by_problem={}; conflicts=set(); candidates=[]; nrows=duplicates=invalid=0
matches={name:{'exact':0,'normalized':0} for name,_ in bench}
for row in source:
    nrows+=1
    prompt=row.get('prompt')
    if not isinstance(prompt,list) or len(prompt)!=1 or prompt[0].get('role')!='user':raise ValueError(f'Unexpected prompt at source row {nrows}')
    original=prompt[0]['content']; problem=question(original); key=norm(problem)
    if not key: invalid+=1;continue
    if key in seen:
        duplicates+=1
        gold=(row.get('reward_model') or {}).get('ground_truth')
        previous=gold_by_problem.get(key)
        if previous is not None and gold != previous:
            try:
                if not valid_gold(gold) or not verify(parse(previous),parse(gold)):
                    conflicts.add(key)
            except Exception:
                conflicts.add(key)
        continue
    seen.add(key); overlap=False
    gold_by_problem[key]=(row.get('reward_model') or {}).get('ground_truth')
    for name,_ in bench:
        if problem in raw_indices[name]:matches[name]['exact']+=1
        if key in indices[name]:matches[name]['normalized']+=1;overlap=True
    gold=(row.get('reward_model') or {}).get('ground_truth')
    if overlap:continue
    if not valid_gold(gold):invalid+=1;continue
    candidates.append({'id':hashlib.sha256(key.encode()).hexdigest(),'problem':problem,'original_prompt':original,
                       'gold':gold,'source_index':str((row.get('extra_info') or {}).get('index',''))})
    if nrows%50000==0:print('scanned',nrows,'unique',len(seen),flush=True)
candidates=[row for row in candidates if norm(row['problem']) not in conflicts]
held={name:[json.loads(line) for line in (ROOT/'prepared'/f'{name}.jsonl').read_text().splitlines()] for name in ('validation','test')}
held_keys={norm(r['problem']) for rows in held.values() for r in rows}
candidates=[r for r in candidates if norm(r['problem']) not in held_keys]
if len(candidates)<1000:raise ValueError(f'Only {len(candidates)} valid non-overlapping questions')
random.Random(42).shuffle(candidates)
for name,rows in [('train',candidates),('validation',held['validation']),('test',held['test'])]:
    with (OUT/(name+'.jsonl')).open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
(OUT/'manifest.json').write_text(json.dumps({'dataset':'BytedTsinghua-SIA/DAPO-Math-17k','source_rows':nrows,'unique_problems':len(seen),'duplicate_rows':duplicates,'conflicting_duplicate_problems':len(conflicts),'invalid':invalid,'overlap':matches,'candidates':len(candidates),'split_seed':42,'counts':{'train':len(candidates),'validation':len(held['validation']),'test':len(held['test'])},'warning':'Exact and Unicode/whitespace normalized overlap only; paraphrases are not excluded. No gold correctness certification.'},indent=2))
print('prepared',len(candidates),'candidates',flush=True)
