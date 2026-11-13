"""Batched vLLM greedy evaluation, with the same prompts, scoring and output schema."""
import argparse,json,subprocess,sys,tempfile,time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment2.reward import reward,valid_gold,require_validated_versions
from train_mlc_ppo import format_prompt
from evaluation.math500 import format_gold
from fast_vllm_rollout import _worker_environment
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--batch-size',type=int,default=32);a=p.parse_args()
root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True);versions=require_validated_versions();tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B-Instruct')
specs=[('math500','HuggingFaceH4/MATH-500','test','6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'),('aime2024','HuggingFaceH4/aime_2024','train','2fe88a2f1091d5048c0f36abc874fb997b3dd99a')];records={};items=[]
for name,repo,split,revision in specs:
 rows=load_dataset(repo,split=split,revision=revision);rows=rows.select(range(min(a.limit,len(rows)))) if a.limit else rows;out=root/name;out.mkdir(exist_ok=True);records[name]=[]
 for i,row in enumerate(rows):
  target=out/f'{i:04d}.json'
  if target.exists():records[name].append(json.loads(target.read_text()));continue
  gold=format_gold(row['answer']);problem=row['problem'];pid=str(row.get('unique_id',row.get('id',i)));record={'id':pid,'problem':problem,'original_gold':row['answer'],'verifier_gold':gold}
  if not valid_gold(gold):record.update(status='excluded_unparseable_gold',reward=None);target.write_text(json.dumps(record));records[name].append(record);continue
  prompt=problem+'\n\nSolve step by step and put your final answer in \\boxed{...}.';formatted=format_prompt(tok,prompt,a.model);ids=tok(formatted,add_special_tokens=False).input_ids
  items.append({'name':name,'i':i,'ids':ids,'record':record})
if items:
 with tempfile.TemporaryDirectory(prefix='.eval_',dir=root) as tmp:
  tmp=Path(tmp);raw=tmp/'results.jsonl';timing=tmp/'timing.json';req={'model':a.model,'items':[{'name':x['name'],'i':x['i'],'ids':x['ids']} for x in items],'max_model_len':max(len(x['ids']) for x in items)+2048,'batch_size':a.batch_size,'eos':tok.eos_token_id,'results':str(raw.resolve()),'timing':str(timing.resolve())};(tmp/'request.json').write_text(json.dumps(req))
  subprocess.run(['/scratch/tsa5252/.venvs/vllm-pilot/bin/python',str(Path(__file__).with_name('vllm_eval_worker.py')),str((tmp/'request.json').resolve())],env=_worker_environment(),check=True)
  output=[json.loads(line) for line in raw.read_text().splitlines()];assert len(output)==len(items)
  for item,r in zip(items,output):
   assert (item['name'],item['i'])==(r['name'],r['i']);tokens=r['token_ids'];text=tok.decode(tokens,skip_special_tokens=True);record=item['record'];record.update(status='scored',completion=text,reward=reward(text,record['verifier_gold']),ended_with_eos=bool(tokens and tokens[-1]==tok.eos_token_id),completion_tokens=len(tokens),seconds=r['batch_seconds']/r['batch_count'])
   target=root/item['name']/f"{item['i']:04d}.json";partial=target.with_suffix('.tmp');partial.write_text(json.dumps(record,ensure_ascii=False));partial.replace(target);records[item['name']].append(record)
  (root/'vllm_timing.json').write_text(timing.read_text())
summaries={}
for name,repo,split,revision in specs:
 rr=records[name];scored=[r for r in rr if r['status']=='scored'];summaries[name]={'dataset':repo,'revision':revision,'total':len(rr),'scored':len(scored),'correct':sum(r['reward'] for r in scored),'accuracy':sum(r['reward'] for r in scored)/len(scored) if scored else None,'truncated':sum(not r['ended_with_eos'] for r in scored)}
(root/'summary.json').write_text(json.dumps({'model':a.model,'decoding':'greedy','engine':'vllm 0.17.1','dtype':'float32','batch_size':a.batch_size,'max_new_tokens':2048,'reward_versions':versions,'benchmarks':summaries,'note':'Exact and normalized decontamination only; benchmark contamination and gold correctness are not guaranteed.'},indent=2));print(json.dumps(summaries,indent=2))
