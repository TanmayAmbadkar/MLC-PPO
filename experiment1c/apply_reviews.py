#!/usr/bin/env python3
"""Apply explicit review records; leave every unreviewed candidate unaccepted."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_mlc_ppo import answer_reward
root=Path('runs/experiment1c/validation')
reviews=json.loads(Path('experiment1c/reviews.json').read_text())
by_id={r['problem_id']:r for r in reviews}
if len(by_id)!=len(reviews): raise ValueError('Duplicate review IDs')
rows=[json.loads(s) for s in (root/'vetted_problems.jsonl').read_text().splitlines()]
ids={r['problem_id'] for r in rows}
if not set(by_id).issubset(ids): raise ValueError('Review ID not in candidate sample')
excluded=[]
for row in rows:
    review=by_id.get(row['problem_id'])
    if not review: continue
    if review['decision']=='accept':
        if not review.get('derivation') or not review.get('independent_answer'):
            raise ValueError('Accepted records require a derivation and independent answer')
        if not answer_reward(review['independent_answer'],row['original_gold']):
            raise ValueError(f"Answer discrepancy requires explicit correction review: {row['problem_id']}")
        row.update(accepted=True, acceptance_decision='accept', review_status='assistant_independently_derived',
                   independent_answer=review['independent_answer'],review_justification=review['derivation'])
    elif review['decision']=='exclude':
        if not review.get('derivation'): raise ValueError('Exclusions need a reason')
        row.update(accepted=False,acceptance_decision='exclude',review_status='assistant_reviewed',review_justification=review['derivation'])
        excluded.append({'problem_id':row['problem_id'],'question':row['question'],'original_gold':row['original_gold'],
                         'acceptance_decision':'exclude','review_status':'assistant_reviewed','reason':review['derivation']})
    else: raise ValueError('Invalid review decision')
(root/'vetted_problems.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
base=[json.loads(s) for s in (root/'excluded_problems.jsonl').read_text().splitlines() if json.loads(s).get('review_status')!='assistant_reviewed']
(root/'excluded_problems.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in base+excluded))
print(f'{sum(r["accepted"] for r in rows)} accepted; {len(excluded)} excluded; {sum(r["acceptance_decision"] == "pending" for r in rows)} pending')
