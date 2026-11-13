"""Fixed held-out sampled validation for choosing one LoRA learning rate."""
import json,os,random,subprocess,tempfile
from pathlib import Path
from transformers import AutoTokenizer
from base_experiment.prompt import base_prompt
from base_fast_vllm_rollout import _worker_environment
from experiment2.reward import reward,require_validated_versions
name=os.environ['RUN']; root=Path('runs/harder_4b/lr_pilot')/name;meta=json.loads((root/'resume_meta.json').read_text());adapter=(root/f"checkpoint-{meta['step']}").resolve()
model='Qwen/Qwen3-4B-Base';revision='906bfd4b4dc7f14ee4320094d8b41684abff8539';tok=AutoTokenizer.from_pretrained(model,revision=revision);require_validated_versions()
rows=[json.loads(x) for x in Path('runs/base_rlvr/prepared/validation.jsonl').read_text().splitlines()];rows=random.Random(314).sample(rows,50);prompts=[tok(base_prompt(r['problem']),add_special_tokens=False).input_ids for r in rows]
with tempfile.TemporaryDirectory(prefix='4b-eval-',dir='/tmp') as t:
 t=Path(t);req={'model':model,'revision':revision,'snapshot':str(adapter),'prompt_token_ids':prompts,'max_tokens':8192,'max_model_len':max(map(len,prompts))+8192,'seed':2026,'eos_token_id':tok.eos_token_id,'output':str(t/'out.json')};(t/'request.json').write_text(json.dumps(req));env=_worker_environment();env['FAST_VLLM_GPU_FRACTION']='0.80';subprocess.run(['/scratch/tsa5252/.venvs/vllm-pilot/bin/python','harder_4b/train_rollout_worker.py',str(t/'request.json')],env=env,check=True);outs=json.loads((t/'out.json').read_text())['records']
 assert len(outs)==len(rows)
 scored=[]
 for row,o in zip(rows,outs):
  text=tok.decode(o['tokens'],skip_special_tokens=True);scored.append({'id':row['id'],'problem':row['problem'],'gold':row['gold'],'completion':text,'reward':reward(text,row['gold']),'finish_reason':o['finish_reason']})
 (root/'validation.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in scored));(root/'validation_summary.json').write_text(json.dumps({'samples':len(scored),'correct':sum(x['reward'] for x in scored),'accuracy':sum(x['reward'] for x in scored)/len(scored),'training_tokens':meta['tokens'],'decoding':'temperature=1 top_p=1, one sample per held-out prompt','max_completion_tokens':8192},indent=2));print((root/'validation_summary.json').read_text())
