#!/usr/bin/env python3
"""Summarize recorded scores and a separately labeled counterfactual audit."""
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_mlc_ppo import answer_reward
root=Path('runs/experiment1/verifier_audit')
rows=[json.loads(s) for s in (root/'responses.jsonl').read_text().splitlines()]
annotations={
 191:{'issue':'Ambiguous request: gold sums 64 weasels and 32 rabbits to 96; responses often report separate counts.', 'counterfactual_gold':None},
 2770:{'issue':'Gold final answer is 106, but its own arithmetic and independent calculation give 108.', 'counterfactual_gold':'108'},
 3489:{'issue':'Malformed question says 2 132 contracts; gold assumes two 132-page contracts, responses interpret 2,132 contracts.', 'counterfactual_gold':None},
 4921:{'issue':'Gold 13; original verifier treats numerically equal 13.00 as incorrect.', 'counterfactual_gold':None},
 5181:{'issue':'Question specifies a 30-minute head start; gold assumes 20 minutes and says 10, while 30 minutes yields 15.', 'counterfactual_gold':'15'},
}
report={'note':'Fresh full responses, not the original saved prefixes. Counterfactual corrected gold answers are manual audit annotations; original data and scores are unchanged.', 'problems':[]}
for k in sorted(annotations):
 subset=[r for r in rows if r['problem_id']==k]
 a=annotations[k]
 item={'problem_id':k,'n':len(subset),'gold':subset[0]['extracted_gold'], 'issue':a['issue'],
       'original_verifier_successes':sum(r['reward'] for r in subset),
       'fixed_verifier_successes_same_gold':sum(answer_reward(r['response'],r['gold_answer']) for r in subset),
       'counterfactual_gold':a['counterfactual_gold'],
       'fixed_verifier_successes_counterfactual_gold':sum(answer_reward(r['response'],'#### '+a['counterfactual_gold']) for r in subset) if a['counterfactual_gold'] else None,
       'truncated':sum(not r['ended_with_eos'] for r in subset),
       'original_extracted_answer_counts':dict(Counter(r['extracted_answer'] if len(r['extracted_answer'])<100 else '[long/unparsed response]' for r in subset))}
 report['problems'].append(item)
(root/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
