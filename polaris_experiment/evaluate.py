"""Evaluate frozen 4B base or trained LoRA policy on held-out benchmarks."""
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from base_experiment.prompt import base_prompt
from experiment2.reward import reward,valid_gold,require_validated_versions
from evaluation.answers import format_gold
from base_fast_vllm_rollout import _worker_environment
p=argparse.ArgumentParser();p.add_argument('--adapter',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args();model='Qwen/Qwen3-4B-Base';rev='906bfd4b4dc7f14ee4320094d8b41684abff8539';tok=AutoTokenizer.from_pretrained(model,revision=rev);versions=require_validated_versions();root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
specs=[('math500','HuggingFaceH4/MATH-500','test','6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'),('aime2024','HuggingFaceH4/aime_2024','train','2fe88a2f1091d5048c0f36abc874fb997b3dd99a'),('aime2025','math-ai/aime25','test','563bb8404243c5f09de6ec262f2db674fe5bce9b'),('amc2023','math-ai/amc23','test','80815d37005feb82cd7f8fbc6901d5d3eff43057'),('minerva','svc-huggingface/minerva-math','test','aec85355cf57a4337f73d1bb03233288cb0344df')];items=[];scored={n:[] for n,*_ in specs}
for name,repo,split,r in specs:
 rows=[json.loads(x) for x in Path('runs/hard_math_eval/minerva.jsonl').read_text().splitlines()] if name=='minerva' else load_dataset(repo,split=split,revision=r)
 if a.limit:rows=list(rows)[:a.limit]
 for i,row in enumerate(rows):
  row=dict(row);ans=row.get('original_answer') if name=='minerva' else row.get('answer');gold=row['gold'] if name=='minerva' else format_gold(ans);problem=row.get('problem') or row.get('question');record={'name':name,'index':i,'id':str(row.get('unique_id',row.get('id',i))),'problem':problem,'gold':gold}
  if not valid_gold(gold):record.update(status='excluded_unparseable_gold');scored[name].append(record);continue
  record['ids']=tok(base_prompt(problem),add_special_tokens=False).input_ids;items.append(record)
with tempfile.TemporaryDirectory(prefix='polaris-eval-',dir='/tmp') as t:
 t=Path(t);raw=t/'outputs.jsonl';req={'model':model,'revision':rev,'adapter':None if a.adapter=='none' else str(Path(a.adapter).resolve()),'items':[{'name':x['name'],'index':x['index'],'ids':x['ids']} for x in items],'max_model_len':max(map(lambda x:len(x['ids']),items))+8192,'eos':tok.eos_token_id,'output':str(raw)};(t/'request.json').write_text(json.dumps(req));subprocess.run(['/scratch/tsa5252/.venvs/vllm-pilot/bin/python','polaris_experiment/eval_worker.py',str(t/'request.json')],env=_worker_environment(),check=True)
 outputs=[json.loads(x) for x in raw.read_text().splitlines()];assert len(outputs)==len(items)
 for x,o in zip(items,outputs):
  assert (x['name'],x['index'])==(o['name'],o['index']);x.pop('ids');text=tok.decode(o['tokens'],skip_special_tokens=True);x.update(completion=text,reward=reward(text,x['gold']),finish_reason=o['finish_reason'],completion_tokens=len(o['tokens']),status='scored');scored[x['name']].append(x)
for name,rs in scored.items():(root/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
summary={}
for name,repo,split,r in specs:
 rr=[x for x in scored[name] if x['status']=='scored'];summary[name]={'source':repo,'revision':r,'scored':len(rr),'correct':sum(x['reward'] for x in rr),'accuracy':sum(x['reward'] for x in rr)/len(rr) if rr else None,'truncated':sum(x['finish_reason']=='length' for x in rr)}
(root/'summary.json').write_text(json.dumps({'model':model,'revision':rev,'adapter':a.adapter,'decoding':'greedy','max_completion_tokens':8192,'benchmarks':summary,'verifier_versions':versions},indent=2));print(json.dumps(summary,indent=2))
