"""Add AMC23 decontamination to the existing pinned DeepScaleR subset."""
import hashlib,json,re,unicodedata
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT=Path('runs/base_rlvr/prepared');ROOT.mkdir(parents=True,exist_ok=True)
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).strip()).casefold()
revision=HfApi().dataset_info('math-ai/amc23').sha
amc=load_dataset('math-ai/amc23',split='test',revision=revision)
refs=[norm(row.get('problem') or row.get('question')) for row in amc]
splits={name:[json.loads(s) for s in Path(f'runs/deepscaler/prepared/{name}.jsonl').read_text().splitlines()] for name in ('train','validation','test')}
allrows=[r for rows in splits.values() for r in rows];X=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,dtype='float32',sublinear_tf=True).fit_transform(refs+[norm(x['problem']) for x in allrows]); ref=X[:len(refs)]; scores=(X[len(refs):]@ref.T).max(axis=1).toarray().ravel().tolist()
lookup={r['id']:float(s) for r,s in zip(allrows,scores)};excluded=[]
for name,rows in splits.items():
 keep=[]
 for row in rows:
  score=lookup[row['id']]
  if norm(row['problem']) in refs or score>=.82:excluded.append({'id':row['id'],'split':name,'amc_similarity':score})
  else:keep.append(row)
 (ROOT/f'{name}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in keep));splits[name]=keep
manifest={'source_manifest':json.loads(Path('runs/deepscaler/prepared/manifest.json').read_text()),'amc_dataset':'math-ai/amc23','amc_revision':revision,'amc_count':len(refs),'similarity':'char 3–5 gram TF-IDF cosine >= 0.82','amc_exclusions':len(excluded),'split_counts':{k:len(v) for k,v in splits.items()},'note':'Surface overlap filters cannot rule out semantic paraphrases or pretraining contamination.'}
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));(ROOT/'amc_exclusions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in excluded));print(json.dumps(manifest,indent=2))
