"""One-update benchmark with offline vLLM BF16 behavior; no checkpoint saved."""
import json, os, time
from pathlib import Path
import torch
import train_viability as base
out=Path(os.environ['PILOT_OUT']); records=json.loads((out/'rollouts.json').read_text()); cursor=0
if os.getenv('PILOT_SORT') == '1':
 lengths={(r['prompt'],r['answer']):len(r['token_ids']) for r in records}
 assert len(lengths)==len(records), 'Duplicate prompt/answer pair; cannot safely reorder'
 records.sort(key=lambda r:len(r['token_ids']))
 original_sample=base.random.sample
 def sample_by_length(population,k):
  chosen=original_sample(population,k)
  assert all((r['prompt'],r['answer']) in lengths for r in chosen)
  return sorted(chosen,key=lambda r:lengths[(r['prompt'],r['answer'])])
 base.random.sample=sample_by_length

def build_rollout(model,tokenizer,critic,batch,device,args,layers):
 global cursor
 started=time.monotonic(); selected=records[cursor:cursor+len(batch)]; cursor+=len(batch)
 assert len(selected)==len(batch)
 for r,b in zip(selected,batch): assert r['prompt']==b['prompt'] and r['answer']==b['answer'], 'rollout problem mismatch'
 prompts=[base.format_prompt(tokenizer,r['prompt']+'\n\nSolve step by step and put your final answer in \\boxed{...}.',args.model_id) for r in selected]
 enc=tokenizer(prompts,return_tensors='pt',padding=True,truncation=False,add_special_tokens=False)
 for r,ids,mask in zip(selected,enc.input_ids,enc.attention_mask): assert ids[mask.bool()].tolist()==r['prompt_token_ids'], 'different tokenization'
 prompt_ids=enc.input_ids.to(device); prompt_mask=enc.attention_mask.to(device); width=prompt_ids.shape[1]; t=max(len(r['token_ids']) for r in selected)
 completions=torch.full((len(batch),t),tokenizer.pad_token_id,dtype=torch.long,device=device)
 mask=torch.zeros((len(batch),t),dtype=torch.bool,device=device); behavior=torch.zeros((len(batch),t),dtype=torch.float32,device=device)
 for i,r in enumerate(selected):
  n=len(r['token_ids']); completions[i,:n]=torch.tensor(r['token_ids'],device=device); mask[i,:n]=True; behavior[i,:n]=torch.tensor(r['behavior_log_probs'],device=device)
  if r['finish_reason']=='stop': assert r['token_ids'][-1]==tokenizer.eos_token_id
 generated=torch.cat([prompt_ids,completions],dim=1); full_mask=torch.cat([prompt_mask,mask.long()],dim=1)
 was_training=model.training; model.eval()
 with torch.no_grad():
  output=model(input_ids=generated,attention_mask=full_mask,position_ids=base.position_ids_from_mask(full_mask),output_hidden_states=True,use_cache=False)
  features=base.extract_features(output,layers,width-1,t); baseline,head_values,_=critic(features)
  old=base.sequence_log_probs(output.logits[:,width-1:width-1+t],completions,args.generation_temperature)
  dev=(old[mask]-behavior[mask]).exp().sub(1).abs().float(); qs=torch.quantile(dev,torch.tensor([.5,.95,.99,1],device=device)).cpu().tolist()
  mismatch={'n':int(dev.numel()),'median':qs[0],'p95':qs[1],'p99':qs[2],'max':qs[3],'fraction_above_clip':(dev>args.clip_epsilon).float().mean().item()}
  with (out/'mismatch.jsonl').open('a') as f:f.write(json.dumps(mismatch)+'\n')
  del output,dev
  texts=tokenizer.batch_decode(completions,skip_special_tokens=True)
  from experiment2.reward import reward
  scores=[reward(text,row['answer']) for text,row in zip(texts,batch)]
  if args.verify_reward_parity:
   from trl.rewards import accuracy_reward
   check=accuracy_reward([[{'role':'assistant','content':text}] for text in texts],solution=[row['answer'] for row in batch])
   if any(x is None or x != y for x,y in zip(check,scores)): raise RuntimeError(f'reward disagreement: {scores} {check}')
  rewards=torch.tensor(scores,dtype=torch.float32,device=device); advantages=(rewards[:,None]-baseline).detach()
 if was_training:model.train()
 with (out/'rollout_processing_times.jsonl').open('a') as f:f.write(json.dumps({'seconds':time.monotonic()-started,'tokens':int(mask.sum())})+'\n')
 return {'input_ids':generated,'attention_mask':full_mask,'completions':completions,'mask':mask,'ended_with_eos':((completions==tokenizer.eos_token_id)&mask).any(dim=1),'old_log_probs':old.detach(),'behavior_log_probs':behavior.detach(),'advantages':advantages,'baseline':baseline.detach(),'critic_disagreement':head_values.float().std(dim=-1,unbiased=False).detach(),'critic_features':[x.cpu() for x in features],'rewards':rewards,'rows':batch,'response_text':texts,'prompt_width':width}
base.build_rollout=build_rollout
if __name__=='__main__':
 if os.getenv('PILOT_AMP') == '1':
  critic_loss=base.critic_loss
  def critic_loss_fp32(baseline,head_values,returns,mask,auxiliary_weight):
   with torch.autocast(device_type='cuda',enabled=False):
    return critic_loss(baseline.float(),head_values.float(),returns.float(),mask,auxiliary_weight)
  base.critic_loss=critic_loss_fp32
  with torch.autocast(device_type='cuda',dtype=torch.bfloat16,cache_enabled=False):base.main()
 else:base.main()
