#!/usr/bin/env python3
"""Prepare DeepScaleR training data with conservative benchmark decontamination."""
import hashlib,json,random,re,sys,unicodedata
from collections import Counter
from pathlib import Path
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment2.reward import require_validated_versions,valid_gold
ROOT=Path('runs/deepscaler/prepared');ROOT.mkdir(parents=True,exist_ok=True)
require_validated_versions()
REV='b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29'
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).strip()).casefold()
def field(row):
 for key in ('problem','question','Question','Problem'):
  if isinstance(row.get(key),str):return row[key]
 raise ValueError(list(row))
bench=[]
for name,split in [('HuggingFaceH4/MATH-500','test'),('HuggingFaceH4/aime_2024','train'),('math-ai/aime25','test')]:
 for row in load_dataset(name,split=split):bench.append((name,field(row)))
for name in ('olympiad','minerva'):
 for suffix in ('', '_excluded'):
  for line in Path(f'runs/hard_math_eval/{name}{suffix}.jsonl').read_text().splitlines():bench.append((name,json.loads(line)['problem']))
source=load_dataset('agentica-org/DeepScaleR-Preview-Dataset',split='train',revision=REV)
texts=[norm(r['problem']) for r in source]; refs=[norm(s) for _,s in bench]
# Character TF-IDF catches surface-level near duplicates; it does not detect semantic paraphrases.
vector=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,dtype='float32',sublinear_tf=True)
X=vector.fit_transform(refs+texts);ref=X[:len(refs)];allx=X[len(refs):]
maximum=[]
for i in range(0,len(texts),256):
 scores=allx[i:i+256]@ref.T
 maximum.extend(scores.max(axis=1).toarray().ravel().tolist())
benchkeys=set(refs);counter=Counter();keys={};conflicts=set()
for i,row in enumerate(source):
 k=texts[i]; answer=str(row['answer'] or '').strip()
 if not k or not answer:counter['empty']+=1;continue
 if k in keys:
  counter['duplicates']+=1
  if keys[k][1]!=answer:conflicts.add(k)
 else:keys[k]=(i,answer)
indices=list(keys.items());random.Random(42).shuffle(indices);selected=[];exclusions=[]
for k,(i,answer) in indices:
 r=source[i];reason=None
 if k in conflicts:reason='conflicting_duplicate_answers'
 elif k in benchkeys:reason='exact_or_normalized_benchmark_overlap'
 elif maximum[i]>=0.82:reason='near_benchmark_tfidf_0.82'
 elif re.search(r'\[asy\]|<image|\\includegraphics|\[img\]',r['problem'],re.I):reason='image_or_diagram_markup'
 if reason:counter[reason]+=1;exclusions.append({'source_index':i,'reason':reason,'similarity':maximum[i]});continue
 gold=answer if answer.startswith(('$',r'\[',r'\(')) or r'\boxed' in answer else '$'+answer+'$'
 if not valid_gold(gold):counter['unparseable_gold']+=1;continue
 selected.append({'id':hashlib.sha256(k.encode()).hexdigest(),'problem':r['problem'],'gold':gold,'source_index':i,'near_benchmark_similarity':maximum[i]})
 if len(selected)>=5200:break
if len(selected)<5200:raise ValueError(f'Insufficient scoreable problems: {len(selected)}')
for name,rows in [('train',selected[:5000]),('validation',selected[5000:5100]),('test',selected[5100:5200])]:
 (ROOT/f'{name}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
(ROOT/'exclusions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in exclusions))
manifest={'source':'agentica-org/DeepScaleR-Preview-Dataset','revision':REV,'source_size':len(source),'benchmark_items':len(refs),'selection_seed':42,'splits':{'train':5000,'validation':100,'test':100},'near_overlap_method':'character 3–5 gram TF-IDF cosine >= 0.82','filter_counts_scanned':dict(counter),'note':'No guarantee of semantic decontamination, image-free questions, or correct gold answers. Filter counts reflect processed candidates plus source duplicates.'}
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)
