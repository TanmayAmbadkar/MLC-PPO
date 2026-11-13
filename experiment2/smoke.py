#!/usr/bin/env python3
"""Check dataset, TRL reward parity, saved completions and frozen feature indexing."""
import json, random, re, unicodedata
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.rewards import accuracy_reward
from reward import reward, valid_gold

OUT=Path('runs/experiment2/smoke_v2'); OUT.mkdir(parents=True,exist_ok=True)
MODEL='Qwen/Qwen2.5-Math-1.5B-Instruct'
DATA='BytedTsinghua-SIA/DAPO-Math-17k'
BENCH=[('HuggingFaceH4/MATH-500','test'),('HuggingFaceH4/aime_2024','train'),('math-ai/aime25','test')]
def normalized(text):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text).strip()).casefold()
def problem_text(text):
    # Strip only the known DAPO wrapper; keep multi-paragraph problem text.
    prefix = 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'
    text = text[len(prefix):].strip() if text.startswith(prefix) else text.strip()
    return re.sub(r'\s*Remember to put your answer on its own line after "Answer:"\.\s*$', '', text).strip()
def field(row):
    for key in ('problem','question','Question','Problem'):
        if isinstance(row.get(key),str): return row[key]
    raise ValueError(f'Unknown benchmark schema: {list(row)}')

bench={}; bench_raw={}
for name,split in BENCH:
    dataset=load_dataset(name,split=split)
    values=[field(row) for row in dataset]
    bench[name]={normalized(v) for v in values};bench_raw[name]=set(values)
    print('benchmark',name,len(values),flush=True)
source=load_dataset(DATA,split='train',streaming=True).shuffle(seed=42,buffer_size=10000)
rows=[]; seen=set(); checked=0; overlap={name:{'exact':0,'normalized':0} for name,_ in BENCH}; excluded=0
for row in source:
    checked+=1
    prompt=row.get('prompt')
    if not isinstance(prompt,list) or len(prompt)!=1 or prompt[0].get('role')!='user': raise ValueError('Unexpected DAPO prompt schema')
    text=prompt[0]['content']; problem=problem_text(text); key=normalized(problem)
    gold=(row.get('reward_model') or {}).get('ground_truth')
    if not key or key in seen: continue
    seen.add(key)
    matches=[]
    for name,_ in BENCH:
        if problem in bench_raw[name]: overlap[name]['exact']+=1
        if key in bench[name]: overlap[name]['normalized']+=1;matches.append(name)
    if matches or not valid_gold(gold): excluded+=1; continue
    rows.append({'problem_id':str((row.get('extra_info') or {}).get('index',checked)), 'problem':problem, 'original_prompt':text, 'gold':gold})
    if len(rows)>=100: break
print('source rows examined',checked,'valid unique',len(rows),'excluded',excluded,flush=True)
if len(rows)!=100: raise RuntimeError('Could not obtain 100 valid uncontaminated unique prompts')
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
for p in model.parameters():p.requires_grad_(False)
tok=AutoTokenizer.from_pretrained(MODEL);tok.padding_side='left'
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
records=[]; mismatches=[]; tested_features=False
response_file = (OUT/'responses.jsonl').open('w')
for start in range(0,len(rows),4):
    batch=rows[start:start+4]
    texts=[tok.apply_chat_template([{'role':'user','content':r['problem']+'\n\nSolve step by step and put your final answer in \\boxed{...}.'}],tokenize=False,add_generation_prompt=True) for r in batch]
    if not all(t.endswith('<|im_start|>assistant\n') for t in texts):
        raise ValueError('Unexpected Qwen generation suffix')
    encoded=tok(texts,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
    with torch.inference_mode():
        generated=model.generate(**encoded,do_sample=True,temperature=0.8,top_p=0.95,top_k=50,max_new_tokens=1024,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    completions=tok.batch_decode(generated[:,encoded.input_ids.shape[1]:],skip_special_tokens=True)
    framework=accuracy_reward([[{'role':'assistant','content':c}] for c in completions],solution=[r['gold'] for r in batch])
    for j,(row,formatted,completion,expected) in enumerate(zip(batch,texts,completions,framework)):
        actual=reward(completion,row['gold']); repeat=reward(completion,row['gold'])
        if expected is None or actual!=expected or actual!=repeat: mismatches.append({'id':row['problem_id'],'standalone':actual,'trl':expected,'repeat':repeat})
        continuation=generated[j,encoded.input_ids.shape[1]:]
        record={**row,'formatted_prompt':formatted,'completion':completion,'standalone_reward':actual,'trl_reward':expected,
                'ended_with_eos':bool((continuation==tok.eos_token_id).any().item()),'max_new_tokens':1024}
        records.append(record)
        response_file.write(json.dumps(record,ensure_ascii=False)+'\n')
        response_file.flush()
    if not tested_features:
        ids=generated[0]; startpos=encoded.input_ids.shape[1]; index=startpos+min(10,generated.shape[1]-startpos-2)
        if index < startpos: raise RuntimeError('Insufficient generated tokens for prefix check')
        # Remove left padding so the two forwards have identical positions and context.
        unpadded_prompt=encoded.input_ids[0][encoded.attention_mask[0].bool()]
        full=torch.cat([unpadded_prompt,generated[0,startpos:]])[None,:]
        prefix_len=len(unpadded_prompt)+(index-startpos)+1
        with torch.inference_mode():
            f=model(full,output_hidden_states=True,use_cache=False)
            p=model(full[:,:prefix_len],output_hidden_states=True,use_cache=False)
            changed=full.clone()
            changed[:,prefix_len:]=tok.eos_token_id
            altered=model(changed,output_hidden_states=True,use_cache=False)
        layers=[round(model.config.num_hidden_layers*z) for z in (.5,.75,1)]
        causal_diffs=[float((f.hidden_states[k][0,prefix_len-1].float()-altered.hidden_states[k][0,prefix_len-1].float()).abs().max()) for k in layers]
        relative_diffs=[float((f.hidden_states[k][0,prefix_len-1].float()-p.hidden_states[k][0,-1].float()).norm() / f.hidden_states[k][0,prefix_len-1].float().norm().clamp_min(1e-8)) for k in layers]
        if max(causal_diffs)>1e-5: raise AssertionError(f'Prefix hidden states depend on future tokens: {causal_diffs}')
        if max(relative_diffs)>0.05: raise AssertionError(f'Prefix-only features differ substantially from full forward: {relative_diffs}')
        features=[p.hidden_states[k][0,-1].detach().clone() for k in layers]
        head=torch.nn.Linear(features[0].numel(),1,device='cuda',dtype=torch.float32)
        loss=sum(head(h.float()).square().mean() for h in features);loss.backward()
        if any(x.grad is not None for x in model.parameters()):raise AssertionError('Backbone received critic gradient')
        tested_features=True
    print('generated',len(records),'of',len(rows),'mismatches',len(mismatches),flush=True)
response_file.close()
report={'model':MODEL,'source':DATA,'source_rows_examined':checked,'unique_candidates_examined':len(seen),'overlap':overlap,'invalid_or_overlapping_exclusions':excluded,'samples':len(records),'reward_disagreements':mismatches,'future_token_perturbation_max_differences':causal_diffs,'prefix_only_relative_differences':relative_diffs,'backbone_gradients_absent':tested_features,'prompt_template':bool(tok.chat_template),'max_new_tokens':1024,'ended_with_eos':sum(r['ended_with_eos'] for r in records),'boxed_answers':sum('\\boxed{' in r['completion'] for r in records),'successes':sum(r['standalone_reward'] for r in records),'note':'Overlap counts cover only the scanned smoke sample; exact and whitespace/Unicode normalized matching do not detect paraphrases.'}
(OUT/'report.json').write_text(json.dumps(report,indent=2))
if mismatches:raise RuntimeError(f'{len(mismatches)} reward disagreements')
print(json.dumps(report,indent=2),flush=True)
