"""Batched vLLM rollout for a single synchronous PPO update."""
import json, logging, os, subprocess, sys, tempfile, time
from pathlib import Path
import torch
LOG=logging.getLogger('fast_rollout')

def _worker_environment():
 env=os.environ.copy(); cuda=env.get('CUDA_VISIBLE_DEVICES','').split(',')[0]
 if cuda.startswith('GPU-'):
  lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()
  matches=[line.split(',')[0].strip() for line in lines if line.split(',')[1].strip()==cuda]
  if len(matches)!=1:raise RuntimeError(f'Cannot map assigned GPU {cuda}')
  env['CUDA_VISIBLE_DEVICES']=matches[0];env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
 elif cuda:env['CUDA_VISIBLE_DEVICES']=cuda
 env['LD_LIBRARY_PATH']='/scratch/tsa5252/.conda/envs/grpo_env/lib:'+env.get('LD_LIBRARY_PATH','')
 env.update(FLASHINFER_WORKSPACE_BASE='/scratch/tsa5252',TRITON_CACHE_DIR='/scratch/tsa5252/.cache/triton',XDG_CACHE_HOME='/scratch/tsa5252/.cache',XDG_CONFIG_HOME='/scratch/tsa5252/.config',HF_HOME='/scratch/tsa5252/.cache/huggingface',VLLM_NO_USAGE_STATS='1')
 return env

def build_batch(base,model,tokenizer,critic,batch,device,args,layers,step):
 start=time.monotonic(); root=Path('/tmp/tsa5252-guru-rollouts') / Path(args.output_dir).name; root.mkdir(parents=True,exist_ok=True)
 texts=[base.format_prompt(tokenizer,r['prompt'],args.model_id) for r in batch]
 prompt_ids=[tokenizer(t,add_special_tokens=False)['input_ids'] for t in texts]
 if max(map(len,prompt_ids))>args.max_prompt_length:raise ValueError('Prompt exceeds maximum length')
 with tempfile.TemporaryDirectory(prefix='.rollout_',dir=root) as tmp:
  tmp=Path(tmp);snapshot=tmp/'policy'; model.save_pretrained(snapshot)
  request={'snapshot':str(snapshot.resolve()),'prompt_token_ids':prompt_ids,'max_tokens':args.max_completion_length,'max_model_len':max(map(len,prompt_ids))+args.max_completion_length,'seed':args.seed+step,'eos_token_id':tokenizer.eos_token_id,'output':str((tmp/'output.json').resolve())}
  path=tmp/'request.json';path.write_text(json.dumps(request));sync_time=time.monotonic()-start
  subprocess.run(['/scratch/tsa5252/.venvs/vllm-pilot/bin/python',str(Path(__file__).resolve().parents[1] / 'fast_vllm_worker.py'),str(path.resolve())],env=_worker_environment(),check=True)
  result=json.loads((tmp/'output.json').read_text())
 records=result['records']; assert len(records)==len(batch)
 # For GRPO keep groups intact until group-relative advantages are computed.
 indexes=list(range(len(batch)))
 if args.algorithm!='grpo':indexes.sort(key=lambda i:len(records[i]['tokens']))
 rollouts=[]; all_dev=[]
 for offset in range(0,len(indexes),args.micro_batch_size):
  idx=indexes[offset:offset+args.micro_batch_size]; chosen=[records[i] for i in idx]; rows=[batch[i] for i in idx]
  padded=tokenizer.pad([{'input_ids':prompt_ids[i]} for i in idx],return_tensors='pt');prompt=padded.input_ids.to(device);pmask=padded.attention_mask.to(device);width=prompt.shape[1];length=max(len(r['tokens']) for r in chosen)
  comp=torch.full((len(idx),length),tokenizer.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(comp,dtype=torch.bool);behavior=torch.zeros_like(comp,dtype=torch.float32)
  for j,r in enumerate(chosen):
   n=len(r['tokens']); assert n>0 and len(r['log_probs'])==n
   if r['finish_reason']=='stop' and r['tokens'][-1]!=tokenizer.eos_token_id:raise RuntimeError('Missing terminal EOS')
   if tokenizer.eos_token_id in r['tokens'][:-1]:raise RuntimeError('Tokens were generated after EOS')
   comp[j,:n]=torch.tensor(r['tokens'],device=device);mask[j,:n]=True;behavior[j,:n]=torch.tensor(r['log_probs'],device=device)
  inputs=torch.cat([prompt,comp],dim=1);attention=torch.cat([pmask,mask.long()],dim=1);was_training=model.training;model.eval()
  with torch.no_grad():
   need_features=critic is not None and args.algorithm!='full_ppo'
   output=model(input_ids=inputs,attention_mask=attention,position_ids=base.position_ids_from_mask(attention),output_hidden_states=need_features,use_cache=False)
   old=base.sequence_log_probs(output.logits[:,width-1:width-1+length],comp,args.generation_temperature)
   if args.algorithm=='full_ppo':
    features=[]; cdevice=next(critic.parameters()).device;critic.eval(); baseline,heads=critic.values(inputs.to(cdevice),attention.to(cdevice),width-1,length);baseline,heads=baseline.to(device),heads.to(device)
   elif critic is None:
    features=[];baseline=torch.zeros_like(mask,dtype=torch.float32);heads=baseline.unsqueeze(-1)
   else:
    features=base.extract_features(output,layers,width-1,length);baseline,heads,_=critic(features)
   all_dev.append((old[mask]-behavior[mask]).exp().sub(1).abs().float().cpu());del output
   decoded=tokenizer.batch_decode(comp,skip_special_tokens=True)
   from guru_experiment.sandbox_verifier import score as guru_score
   scores=[guru_score(row['data_source'],text,row['ground_truth'],row['extra_info']) for text,row in zip(decoded,rows)]
   with (Path(args.output_dir)/'guru_rewards.jsonl').open('a',encoding='utf-8') as handle:
    for row,text,value in zip(rows,decoded,scores): handle.write(json.dumps({'step':step,'id':row['id'],'domain':row['domain'],'data_source':row['data_source'],'completion':text,'reward':value},ensure_ascii=False)+'\n')
   rewards=torch.tensor(scores,dtype=torch.float32,device=device);advantages=(rewards[:,None]-baseline).detach()
  if was_training:model.train()
  rollouts.append({'input_ids':inputs,'attention_mask':attention,'completions':comp,'mask':mask,'ended_with_eos':((comp==tokenizer.eos_token_id)&mask).any(dim=1),'old_log_probs':old.detach(),'behavior_log_probs':behavior.detach(),'advantages':advantages,'baseline':baseline.detach(),'critic_disagreement':heads.float().std(dim=-1,unbiased=False).detach(),'critic_features':[x.cpu() for x in features],'rewards':rewards,'rows':rows,'response_text':decoded,'prompt_width':width})
 dev=torch.cat(all_dev);q=torch.quantile(dev,torch.tensor([.5,.95,.99,1.])); info={'step':step,'tokens':int(dev.numel()),'median':q[0].item(),'p95':q[1].item(),'p99':q[2].item(),'max':q[3].item(),'fraction_above_clip':(dev>args.clip_epsilon).float().mean().item(),'snapshot_seconds':sync_time,'worker_init_seconds':result['init_seconds'],'generation_seconds':result['generation_seconds'],'total_rollout_seconds':time.monotonic()-start}
 with (Path(args.output_dir)/'rollout_mismatch.jsonl').open('a') as f:f.write(json.dumps(info)+'\n')
 LOG.info('rollout_metrics=%s',json.dumps(info))
 return rollouts
