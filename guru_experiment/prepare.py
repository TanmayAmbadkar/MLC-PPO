"""Build a pinned balanced pilot with disjoint prompts and recorded metadata."""
import json,random,re,hashlib
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
import pyarrow.parquet as pq
repo='IFM/guru-RL-92k';rev='2e2790a962a3c099bfb5ea61389cbf98a5ea439b';out=Path('runs/guru_1p5b/prepared');out.mkdir(parents=True,exist_ok=True)
files={'math':['train/math__combined_54.4k.parquet'],'code':['train/codegen__leetcode2k_1.3k.parquet'],'logic':['train/logic__graph_logical_1.2k.parquet','train/logic__ordering_puzzle_1.9k.parquet','train/logic__zebra_puzzle_1.3k.parquet'],'simulation':['train/simulation__codeio_3.7k.parquet'],'science':['train/stem__web_3.6k.parquet'],'table':['train/table__hitab_4.3k.parquet','train/table__multihier_1.5k.parquet']}
def load(file,cols=None):
 path=hf_hub_download(repo,filename=file,repo_type='dataset',revision=rev);p=pq.ParquetFile(path);return p.read(columns=[x for x in cols if x in p.schema_arrow.names] if cols else None).to_pylist()
def prompt(row):
 p=row.get('prompt')
 if isinstance(p,list):return '\n'.join(str(x.get('content','')) for x in p if isinstance(x,dict))
 return str(p or '')
def norm(s):return re.sub(r'\s+',' ',s).strip().lower()
info=HfApi().dataset_info(repo,revision=rev,files_metadata=True);eval_seen=set();skipped=[]
for f in info.siblings:
 if not f.rfilename.startswith(('online_eval/','offline_eval/')) or not f.rfilename.endswith('.parquet'):continue
 if f.size and f.size>100_000_000:skipped.append(f.rfilename);continue
 for row in load(f.rfilename,['prompt']):
  q=norm(prompt(row))
  if q:eval_seen.add(q)
print('Evaluation prompt keys',len(eval_seen),'skipped large files',skipped,flush=True)
rows_out={'train':[],'validation':[],'test':[]};rng=random.Random(2501);global_seen=set();stats={}
for domain,paths in files.items():
 candidates=[];excluded=0
 for file in paths:
  for i,row in enumerate(load(file,['prompt','data_source','reward_model','extra_info'])):
   text=prompt(row);key=norm(text);gt=(row.get('reward_model') or {}).get('ground_truth')
   if not key or gt is None or key in eval_seen or key in global_seen:excluded+=1;continue
   global_seen.add(key);ident=hashlib.sha256((file+'\0'+key).encode()).hexdigest()[:24]
   candidates.append({'id':ident,'domain':domain,'data_source':row.get('data_source'),'prompt':text,'ground_truth':gt,'extra_info':row.get('extra_info') or {},'source_file':file,'source_row':i})
 rng.shuffle(candidates)
 if len(candidates)<380:raise ValueError(f'Insufficient data for {domain}: {len(candidates)}')
 for split,part in [('test',candidates[:40]),('validation',candidates[40:80]),('train',candidates[80:380])]:rows_out[split].extend(part)
 stats[domain]={'available':len(candidates),'excluded':excluded,'train':300,'validation':40,'test':40}
 print(domain,stats[domain],flush=True)
for split,rows in rows_out.items():
 rng.shuffle(rows)
 with (out/f'{split}.jsonl').open('w') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
(out/'manifest.json').write_text(json.dumps({'dataset':repo,'revision':rev,'selection_seed':2501,'splits':{x:len(y) for x,y in rows_out.items()},'domains':stats,'excluded_exact_normalized_evaluation_prompts':len(eval_seen),'skipped_evaluation_files':skipped,'note':'Exact normalized full-prompt matching only; semantic or differently formatted duplicates can remain. No scoring has been performed.'},indent=2))
