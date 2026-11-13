"""Inspect pinned GURU training parquet schemas without executing dataset code."""
import json,collections
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
import pyarrow.parquet as pq
repo='IFM/guru-RL-92k';revision='2e2790a962a3c099bfb5ea61389cbf98a5ea439b';out=Path('runs/guru_1p5b/preparation');out.mkdir(parents=True,exist_ok=True)
api=HfApi();info=api.dataset_info(repo,revision=revision,files_metadata=True);files=[x for x in info.siblings if x.rfilename.startswith('train/') and x.rfilename.endswith('.parquet')];records=[]
for item in files:
 rec={'file':item.rfilename,'bytes':item.size}
 if item.size and item.size>100_000_000:
  rec['inspection']='skipped_large_file';records.append(rec);continue
 path=hf_hub_download(repo,filename=item.rfilename,repo_type='dataset',revision=revision);p=pq.ParquetFile(path);rec.update(rows=p.metadata.num_rows,columns=p.schema_arrow.names)
 columns=[x for x in ('data_source','ability','reward_model','extra_info') if x in p.schema_arrow.names];table=p.read_row_group(0,columns=columns).slice(0,32).to_pylist()
 rec['data_sources']=dict(collections.Counter(str(x.get('data_source')) for x in table));rec['ability']=dict(collections.Counter(str(x.get('ability')) for x in table));rec['reward_model_sample']=[str(x.get('reward_model'))[:300] for x in table[:1]];rec['extra_info_sample']=[str(x.get('extra_info'))[:300] for x in table[:1]];records.append(rec);print(json.dumps(rec),flush=True)
(out/'manifest.json').write_text(json.dumps({'dataset':repo,'revision':revision,'reference_repository':'https://github.com/LLM360/Reasoning360','reference_commit':'13158341d2a0dfe5f3bb80e7126ff21de0d16676','files':records},indent=2))
