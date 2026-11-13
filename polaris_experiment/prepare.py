"""Build a pinned, decontaminated POLARIS subset for the 4B comparison."""
import hashlib,json,random,re,sys,unicodedata
from collections import Counter
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment2.reward import require_validated_versions,valid_gold
require_validated_versions();root=Path('runs/polaris_4b/prepared');root.mkdir(parents=True,exist_ok=True)
repo='POLARIS-Project/Polaris-Dataset-53K';revision='296f8e34132e63f4a1d70e0dcc8bddebb43f03e4';file=hf_hub_download(repo,filename='polaris-data-53K.jsonl',repo_type='dataset',revision=revision)
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).strip()).casefold()
def prompt(row):
 for k in ('problem','question','Question','Problem'):
  if isinstance(row.get(k),str):return row[k]
 raise ValueError('Missing benchmark prompt')
bench=[]
for name,split,rev in [('HuggingFaceH4/MATH-500','test','6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'),('HuggingFaceH4/aime_2024','train','2fe88a2f1091d5048c0f36abc874fb997b3dd99a'),('math-ai/aime25','test','563bb8404243c5f09de6ec262f2db674fe5bce9b'),('math-ai/amc23','test','80815d37005feb82cd7f8fbc6901d5d3eff43057')]:
 for r in load_dataset(name,split=split,revision=rev):bench.append(norm(prompt(r)))
for name in ('olympiad','minerva'):
 for suffix in ('','_excluded'):
  p=Path(f'runs/hard_math_eval/{name}{suffix}.jsonl')
  for l in p.read_text().splitlines():bench.append(norm(json.loads(l)['problem']))
source=[json.loads(x) for x in Path(file).read_text().splitlines() if x.strip()]; order=list(range(len(source)));random.Random(42).shuffle(order);keys={};duplicates=set();counts=Counter()
for i in order:
 r=source[i];p=r.get('problem');a=str(r.get('answer') or '').strip()
 if not isinstance(p,str) or not p.strip() or not a:counts['empty']+=1;continue
 k=norm(p)
 if k in keys:
  counts['duplicate']+=1
  if keys[k][1]!=a:duplicates.add(k)
 else:keys[k]=(i,a)
candidates=list(keys.items())[:10000];texts=[k for k,_ in candidates];refs=bench;v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,dtype='float32',sublinear_tf=True);X=v.fit_transform(refs+texts);mat=X[len(refs):];ref=X[:len(refs)];scores=[]
for i in range(0,len(texts),128):scores.extend((mat[i:i+128]@ref.T).max(axis=1).toarray().ravel().tolist())
seenrefs=set(refs);selected=[];exclusions=[]
for (key,(idx,answer)),sim in zip(candidates,scores):
 r=source[idx];why=None
 if key in duplicates:why='conflicting_duplicate_answers'
 elif key in seenrefs:why='exact_overlap'
 elif sim>=.82:why='near_overlap'
 elif re.search(r'\[asy\]|<image|\\includegraphics|\[img\]',r['problem'],re.I):why='image_markup'
 gold=answer if answer.startswith(('$',r'\[',r'\(')) or r'\boxed' in answer else '$'+answer+'$'
 if not why and not valid_gold(gold):why='unparseable_gold'
 if why:counts[why]+=1;exclusions.append({'source_index':idx,'reason':why,'similarity':sim});continue
 selected.append({'id':hashlib.sha256(key.encode()).hexdigest(),'problem':r['problem'],'gold':gold,'source_index':idx,'near_benchmark_similarity':sim,'source_difficulty':r.get('difficulty')})
 if len(selected)>=5200:break
if len(selected)<5200:raise RuntimeError('Insufficient validated source records')
for name,rows in [('train',selected[:5000]),('validation',selected[5000:5100]),('test',selected[5100:5200])]:
 (root/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(root/'exclusions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in exclusions));(root/'manifest.json').write_text(json.dumps({'dataset':repo,'revision':revision,'source_records':len(source),'candidate_count':len(candidates),'benchmark_prompts':len(refs),'splits':{'train':5000,'validation':100,'test':100},'seed':42,'exclusions':dict(counts),'similarity':'character 3–5 gram TF-IDF cosine >= 0.82','warning':'No guarantee of semantic overlap removal or correctness of source gold labels'},indent=2));print((root/'manifest.json').read_text())

# Prepare the input expected by screen.py; never truncate a training prompt.
from transformers import AutoTokenizer
from base_experiment.prompt import base_prompt
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Base', revision='906bfd4b4dc7f14ee4320094d8b41684abff8539')
train_rows = selected[:5000]
compatible, too_long = [], []
for row in train_rows:
    n = len(tokenizer(base_prompt(row['problem']), add_special_tokens=False)['input_ids'])
    if n <= 2048:
        compatible.append(row)
    else:
        too_long.append({'id': row['id'], 'prompt_tokens': n})
(root/'train_context_valid.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in compatible))
(root/'context_exclusions.json').write_text(json.dumps(too_long, indent=2))
