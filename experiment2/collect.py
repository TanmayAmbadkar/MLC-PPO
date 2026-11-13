#!/usr/bin/env python3
"""Collect detached layer features and auditable independent continuations."""
import argparse, hashlib, json, time
from pathlib import Path
import torch
from math_verify import parse
from transformers import AutoModelForCausalLM, AutoTokenizer
from reward import PRED_CONFIG, reward

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest-dir',default='runs/experiment2/prepared')
p.add_argument('--output-dir',default='runs/experiment2/states')
p.add_argument('--model',default='Qwen/Qwen2.5-Math-1.5B-Instruct')
p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
p.add_argument('--train-rollouts',type=int,default=8);p.add_argument('--validation-rollouts',type=int,default=16);p.add_argument('--test-rollouts',type=int,default=32)
p.add_argument('--trajectories',type=int,default=2);p.add_argument('--batch-size',type=int,default=8)
p.add_argument('--max-new-tokens',type=int,default=1024);p.add_argument('--temperature',type=float,default=.8);p.add_argument('--top-p',type=float,default=.95)
p.add_argument('--seed',type=int,default=42);p.add_argument('--limit-per-split',type=int)
p.add_argument('--difficulty-config',help='Independent difficulty sampling configuration to match')
a=p.parse_args()
if a.world_size<1 or not 0<=a.rank<a.world_size:raise ValueError('Invalid worker rank')
if min(a.train_rollouts,a.validation_rollouts,a.test_rollouts,a.trajectories,a.max_new_tokens,a.batch_size)<1:raise ValueError('Counts must be positive')
if a.difficulty_config:
    difficulty_config=json.loads(Path(a.difficulty_config).read_text())
    for key in ('model','temperature','top_p','max_new_tokens'):
        if difficulty_config[key]!=getattr(a,key):raise ValueError(f'Sampling mismatch for {key}')
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
config={k:v for k,v in vars(a).items() if k not in ('rank','world_size')}
config_path=out/'config.json'
if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Collection config mismatch; use a fresh output directory')
if not config_path.exists():
    tmp=out/f'config-{a.rank}.tmp';tmp.write_text(json.dumps(config,indent=2));tmp.replace(config_path)
tok=AutoTokenizer.from_pretrained(a.model);tok.padding_side='left'
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
for parameter in model.parameters():parameter.requires_grad_(False)
layers=[round(model.config.num_hidden_layers*f) for f in (.5,.75,1.)]
generation={'do_sample':True,'temperature':a.temperature,'top_p':a.top_p,'top_k':50,'pad_token_id':tok.pad_token_id,'eos_token_id':tok.eos_token_id,'use_cache':True}
for split,nroll in [('train',a.train_rollouts),('validation',a.validation_rollouts),('test',a.test_rollouts)]:
    rows=[json.loads(x) for x in (Path(a.manifest_dir)/(split+'.jsonl')).read_text().splitlines()]
    if a.limit_per_split is not None:rows=rows[:a.limit_per_split]
    dest=out/('val' if split=='validation' else split);dest.mkdir(exist_ok=True)
    for index,row in enumerate(rows):
        if index%a.world_size!=a.rank:continue
        target=dest/(row['id']+'.pt');audit=dest/(row['id']+'.json')
        if target.exists() and audit.exists():continue
        seed=(int(hashlib.sha256((row['id']+':'+str(a.seed)).encode()).hexdigest()[:8],16))
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        prompt=row['problem']+'\n\nSolve step by step and put your final answer in \\boxed{...}.'
        formatted=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
        prompt_ids=tok(formatted,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
        states=[]; audit_states=[]; initial_trajectories=[]; generated_tokens=0; started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for trajectory in range(a.trajectories):
                initial=model.generate(prompt_ids,max_new_tokens=a.max_new_tokens,**generation)[0,prompt_ids.shape[1]:]
                generated_tokens+=len(initial)
                eos=(initial==tok.eos_token_id).nonzero(as_tuple=True)[0]
                ended=bool(len(eos))
                initial_trajectories.append({'trajectory':trajectory,'token_ids':initial.tolist(),
                    'text':tok.decode(initial,skip_special_tokens=True),'ended_with_eos':ended})
                if len(eos):initial=initial[:int(eos[0])]
                if len(initial)<4:
                    raise ValueError(f'Trajectory too short for three prefixes: {row["id"]}, trajectory {trajectory}, tokens {len(initial)}')
                for fraction in (.25,.5,.75):
                    length=max(1,int(len(initial)*fraction))
                    prefix_tokens=initial[:length];prefix=torch.cat([prompt_ids[0],prefix_tokens])[None,:]
                    f=model(prefix,output_hidden_states=True,use_cache=False)
                    features=torch.stack([f.hidden_states[k][0,-1].float().cpu() for k in layers])
                    del f
                    rewards=[];continuations=[]
                    for start in range(0,nroll,a.batch_size):
                        n=min(a.batch_size,nroll-start)
                        generated=model.generate(prefix.repeat(n,1),max_new_tokens=a.max_new_tokens-length,**generation)
                        for tokens in generated[:,prefix.shape[1]:]:
                            eos=(tokens==tok.eos_token_id).nonzero(as_tuple=True)[0]; ended=bool(len(eos))
                            action_tokens=tokens[:int(eos[0])+1] if ended else tokens
                            if ended:tokens=tokens[:int(eos[0])]
                            generated_tokens+=len(tokens)+int(ended)
                            combined=torch.cat([prefix_tokens,tokens])
                            full=tok.decode(combined,skip_special_tokens=True)
                            score=reward(full,row['gold']);rewards.append(score)
                            parsed=parse(full,extraction_config=PRED_CONFIG,extraction_mode='first_match')
                            continuations.append({'token_ids':action_tokens.tolist(),'text':tok.decode(tokens,skip_special_tokens=True),'full_response':full,
                                'parsed_answer':[str(x) for x in parsed],'reward':score,'ended_with_eos':ended})
                    states.append({'features':features,'rewards':torch.tensor(rewards),'progress':fraction,'trajectory':trajectory,'prefix_length':length})
                    audit_states.append({'trajectory':trajectory,'progress':fraction,'prefix_token_ids':prefix_tokens.tolist(),
                        'prefix_text':tok.decode(prefix_tokens,skip_special_tokens=True),'continuations':continuations})
        payload={'problem_id':row['id'],'split':split,'states':states,'layers':layers}
        tmp=target.with_suffix('.pt.tmp');torch.save(payload,tmp);tmp.replace(target)
        elapsed=time.monotonic()-started
        metadata={'id':row['id'],'question':row['problem'],'gold':row['gold'],'formatted_prompt':formatted,
            'prompt_token_ids':prompt_ids[0].tolist(),'seed':seed,'generation':generation|{'max_new_tokens':a.max_new_tokens},
            'initial_trajectories':initial_trajectories,'states':audit_states,'elapsed_seconds':elapsed,'generated_tokens':generated_tokens,
            'generated_tokens_per_second_including_scoring':generated_tokens/elapsed,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
        tmp=audit.with_suffix('.json.tmp');tmp.write_text(json.dumps(metadata,ensure_ascii=False));tmp.replace(audit)
        print(split,'rank',a.rank,'problem',index+1,'/',len(rows),'states',len(states),flush=True)
