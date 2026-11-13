#!/usr/bin/env python3
"""Evaluate a saved PPO critic on held-out prefixes and independent continuations."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mlcppo import MultiLayerCritic
from experiment2.reward import reward,require_validated_versions,PRED_CONFIG
from math_verify import parse
from train_mlc_ppo import format_prompt
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--peer-critic',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--input',default='runs/deepscaler/prepared/test.jsonl');p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=2);p.add_argument('--limit',type=int,default=50);p.add_argument('--rollouts',type=int,default=32);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--max-new-tokens',type=int,default=2048);p.add_argument('--seed',type=int,default=314159);a=p.parse_args()
if not 0<=a.rank<a.world_size or min(a.limit,a.rollouts,a.batch_size)<1:raise ValueError('Invalid collection parameters')
require_validated_versions();checkpoint=Path(a.checkpoint);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
rows=[json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()][:a.limit]
if len(rows)!=a.limit:raise ValueError('Not enough held-out problems')
tok=AutoTokenizer.from_pretrained(checkpoint);tok.padding_side='left'
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model=AutoModelForCausalLM.from_pretrained(checkpoint,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
for p_ in model.parameters():p_.requires_grad_(False)
def load_critic(path):
 saved=torch.load(path,map_location='cpu',weights_only=True);layers=saved['layers'];head=MultiLayerCritic(model.config.hidden_size,layers,head_size=saved['head_size'],aggregation=saved['aggregation']).cuda().eval();head.load_state_dict(saved['critic'],strict=True);return head,layers
critic,layers=load_critic(checkpoint/'critic.pt');peer,peer_layers=load_critic(a.peer_critic)
gen=GenerationConfig(do_sample=True,temperature=1.,top_p=1.,top_k=0,num_beams=1,repetition_penalty=1.,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True,max_new_tokens=a.max_new_tokens)
def seed(key):
 value=int(hashlib.sha256(f'{a.seed}:{key}'.encode()).hexdigest()[:8],16);torch.manual_seed(value);torch.cuda.manual_seed_all(value);return value
for index,row in enumerate(rows):
 if index%a.world_size!=a.rank:continue
 dest=out/(row['id']+'.json')
 if dest.exists():continue
 started=time.monotonic();prompt=row['problem']+'\n\nSolve step by step and put your final answer in \\boxed{...}.';formatted=format_prompt(tok,prompt,a.checkpoint);prompt_ids=tok(formatted,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
 if prompt_ids.shape[1]>2048:raise ValueError(f'Prompt too long: {row["id"]}')
 states=[];initial=[];difficulty=[]
 with torch.inference_mode():
  difficulty_seed=seed(f'{row["id"]}:difficulty')
  for start in range(0,8,a.batch_size):
   generated=model.generate(input_ids=prompt_ids.repeat(min(a.batch_size,8-start),1),generation_config=gen)
   for raw in generated[:,prompt_ids.shape[1]:]:
    eos=(raw==tok.eos_token_id).nonzero(as_tuple=True)[0];ended=bool(len(eos));tokens=raw[:int(eos[0])] if ended else raw
    text=tok.decode(tokens,skip_special_tokens=True);difficulty.append({'text':text,'reward':reward(text,row['gold']),'ended_with_eos':ended})
  for trajectory in range(2):
   initial_seed=seed(f'{row["id"]}:initial:{trajectory}');full=model.generate(input_ids=prompt_ids,generation_config=gen)[0,prompt_ids.shape[1]:];eos=(full==tok.eos_token_id).nonzero(as_tuple=True)[0];ended=bool(len(eos));tokens=full[:int(eos[0])] if ended else full
   initial.append({'trajectory':trajectory,'seed':initial_seed,'text':tok.decode(tokens,skip_special_tokens=True),'ended_with_eos':ended,'token_count':len(tokens)})
   if len(tokens)<4:raise ValueError(f'Insufficient tokens: {row["id"]}')
   for fraction in (.25,.5,.75):
    length=max(1,int(len(tokens)*fraction));prefix_tokens=tokens[:length];prefix=torch.cat([prompt_ids[0],prefix_tokens])[None,:]
    outputs=model(input_ids=prefix,output_hidden_states=True,use_cache=False);final_feature=outputs.hidden_states[-1][0,-1,:].float().cpu().tolist();features=[outputs.hidden_states[k][:,-1:,:] for k in layers];baseline,heads,_=critic(features);prediction=float(baseline.item());head_values=heads[0,0].cpu().tolist();peer_value=float(peer([outputs.hidden_states[k][:,-1:,:] for k in peer_layers])[0].item());del outputs,features,baseline,heads
    state_seed=seed(f'{row["id"]}:state:{trajectory}:{fraction}');continuations=[]
    for start in range(0,a.rollouts,a.batch_size):
     batch=min(a.batch_size,a.rollouts-start);cfg=GenerationConfig.from_dict(gen.to_dict());cfg.max_new_tokens=a.max_new_tokens-length
     result=model.generate(input_ids=prefix.repeat(batch,1),generation_config=cfg)
     for response in result[:,prefix.shape[1]:]:
      eos=(response==tok.eos_token_id).nonzero(as_tuple=True)[0];has_eos=bool(len(eos));part=response[:int(eos[0])] if has_eos else response
      fulltext=tok.decode(torch.cat([prefix_tokens,part]),skip_special_tokens=True);score=reward(fulltext,row['gold']);parsed=parse(fulltext,extraction_config=PRED_CONFIG,extraction_mode='first_match')
      continuations.append({'text':tok.decode(part,skip_special_tokens=True),'full_response':fulltext,'reward':score,'parsed_answer':[str(x) for x in parsed],'ended_with_eos':has_eos,'token_ids':response[:int(eos[0])+1].tolist() if has_eos else response.tolist()})
    states.append({'trajectory':trajectory,'progress':fraction,'prefix_length':length,'prefix_text':tok.decode(prefix_tokens,skip_special_tokens=True),'value':prediction,'peer_value':peer_value,'final_feature':final_feature,'prefix_token_ids':prefix[0].tolist(),'head_values':head_values,'seed':state_seed,'continuations':continuations})
 metadata={'id':row['id'],'source_index':row['source_index'],'problem':row['problem'],'gold':row['gold'],'checkpoint':a.checkpoint,'peer_critic':a.peer_critic,'layers':layers,'peer_layers':peer_layers,'sampling':{'temperature':1.,'top_p':1.,'top_k':0,'max_total_response_tokens':a.max_new_tokens,'dtype':'float32','seed':a.seed},'difficulty_seed':difficulty_seed,'difficulty_rollouts':difficulty,'initial_trajectories':initial,'states':states,'seconds':time.monotonic()-started}
 tmp=dest.with_suffix('.json.tmp');tmp.write_text(json.dumps(metadata,ensure_ascii=False));tmp.replace(dest);print(f'rank={a.rank} problem={index+1}/{len(rows)} states={len(states)} seconds={metadata["seconds"]:.1f}',flush=True)
