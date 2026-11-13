#!/usr/bin/env python3
"""Prepare blinded GSM8K candidates and flag issues for independent review."""
import ast
import json
import operator
import random
import re
import sys
from decimal import Decimal
from pathlib import Path
from datasets import load_dataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_mlc_ppo import answer_reward, normalize_answer

OUT=Path('runs/experiment1c/validation'); OUT.mkdir(parents=True,exist_ok=True)
data=load_dataset('openai/gsm8k','main',split='train')
previous=set()
for split in ('train','val','test'):
    previous.update(int(p.stem) for p in (Path('runs/experiment1/data')/split).glob('*.pt'))
audit={191:('exclude','Question permits separate component answers; gold uses their sum.',None),
       2770:('corrected','50 + 20 + 18 + 20 = 108; source final answer says 106.','108'),
       3489:('exclude','Malformed wording does not establish that two contracts are 132 pages each.',None),
       4921:('verified','2 + 5 + 13*0.25 + 20*0.10 + 8*0.05 + 35*0.01 = 13.','13'),
       5181:('corrected','30-minute head start: 15*0.5/(45-15) hours = 15 minutes; source final answer says 10.','15')}
ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
def calc(node):
    if isinstance(node,ast.Expression): return calc(node.body)
    if isinstance(node,ast.Constant) and type(node.value) in (int,float): return Decimal(str(node.value))
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
        return -calc(node.operand) if isinstance(node.op,ast.USub) else calc(node.operand)
    if isinstance(node,ast.BinOp) and type(node.op) in ops: return ops[type(node.op)](calc(node.left),calc(node.right))
    raise ValueError('unsupported expression')
def flags(answer):
    found=[]
    for expression, result in re.findall(r'<<([^<>=]+)=([^<>]+)>>',answer):
        try:
            actual=calc(ast.parse(expression.replace(',',''),mode='eval'))
            expected=Decimal(result.strip().replace(',',''))
            if abs(actual-expected)>Decimal('0.000001'):
                found.append({'expression':expression,'annotated':result,'computed':str(actual)})
        except Exception: found.append({'expression':expression,'annotated':result,'status':'not checked'})
    matches=re.findall(r'<<([^<>=]+)=([^<>]+)>>',answer)
    final=normalize_answer(answer)
    if matches and not answer_reward(matches[-1][1], '#### '+final):
        found.append({'status':'last annotated result differs from final answer','last_result':matches[-1][1],'final':final})
    return found

# Preserve previously studied problems outside the prospective primary sample.
previous_questions={' '.join(data[i]['question'].split()) for i in previous}
seen={}; candidates=[]
for i,row in enumerate(data):
    key=' '.join(row['question'].split())
    if key in seen: continue
    seen[key]=i
    if i not in previous and key not in previous_questions: candidates.append(i)
random.Random(1042).shuffle(candidates)
if len(candidates)<300: raise RuntimeError('Not enough candidate problems')
with (OUT/'vetted_problems.jsonl').open('w',encoding='utf-8') as f:
    for j,i in enumerate(candidates[:300]):
        row=data[i]; split='train' if j<200 else ('validation' if j<250 else 'test')
        record={'problem_id':i,'question':row['question'],'original_gold':normalize_answer(row['answer']),
                'original_solution':row['answer'],'split':split,'review_status':'pending_independent_review',
                'acceptance_decision':'pending','accepted':False,'arithmetic_flags':flags(row['answer']),
                'review_justification':None}
        f.write(json.dumps(record,ensure_ascii=False)+'\n')
with (OUT/'blind_review.jsonl').open('w',encoding='utf-8') as f:
    for j,i in enumerate(candidates[:300]):
        record={'problem_id':i,'question':data[i]['question'],
                'independent_answer':None,'question_unambiguous':None,
                'derivation':None,'reviewer':None}
        f.write(json.dumps(record,ensure_ascii=False)+'\n')
with (OUT/'excluded_problems.jsonl').open('w',encoding='utf-8') as f:
    for i,(status,reason,corrected) in audit.items():
        row=data[i]
        record={'problem_id':i,'question':row['question'],'original_gold':normalize_answer(row['answer']),
                'acceptance_decision':'exclude_from_primary_sample','reason':'Previously inspected Experiment 1 test problem; '+reason,
                'review_status':'audited','audit_classification':status,'corrected_gold':corrected,
                'correction_justification':reason if corrected else None}
        f.write(json.dumps(record,ensure_ascii=False)+'\n')
# Keep every correction auditable; no candidate gold label is silently changed.
corrections=[{'problem_id':i,'original_answer':normalize_answer(data[i]['answer']),'corrected_answer':v[2],
              'justification':v[1],'review_status':'assistant_reviewed_not_independently_human_verified'}
             for i,v in audit.items() if v[0]=='corrected']
(OUT/'corrections.json').write_text(json.dumps(corrections,indent=2))
checks=[('decimal_equivalence',r'\boxed{13.00}','#### 13',1),
        ('leading_decimal',r'\boxed{0.50}','#### .5',1),
        ('wrong_numeric',r'\boxed{12}','#### 13',0),
        ('separate_values_not_scalar',r'\boxed{64} and \boxed{32}','#### 96',0),
        ('bad_gold_not_silently_corrected',r'\boxed{108}','#### 106',0),
        ('documented_correction',r'\boxed{108}','#### 108',1)]
results=[]
for name,response,gold,expected in checks:
    actual=answer_reward(response,gold)
    results.append({'name':name,'response':response,'gold':gold,'expected':expected,'actual':actual,'passed':actual==expected})
results.append({'name':'nested_box_extraction','expected':r'3+\frac{1}{2}',
                'actual':normalize_answer(r'\boxed{3+\frac{1}{2}}'),
                'passed':normalize_answer(r'\boxed{3+\frac{1}{2}}')==r'3+\frac{1}{2}'})
results.append({'name':'multi_answer_support','status':'unsupported: require review and exclusion from primary sample', 'passed':None})
(OUT/'verifier_tests.json').write_text(json.dumps(results,indent=2))
failed=[r for r in results if r['passed'] is False]
print(f'Prepared 300 pending candidates, 0 accepted, {len(audit)} audit exclusions, {len(corrections)} documented corrections; failed verifier checks={len(failed)}',flush=True)
if failed: raise SystemExit(1)
