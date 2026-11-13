"""Four-GPU LoRA actor + fully trainable FSDP critic update feasibility pilot.

One rollout is sampled with the LoRA adapter. If shorter than 8192 tokens it is
repeated to stress the full 8192-token update; this is not a learning experiment.
"""
import functools,json,os,subprocess,sys,tempfile,time,traceback
from datetime import timedelta
from pathlib import Path
import torch,torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoConfig,AutoModel,AutoModelForCausalLM,AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from peft import LoraConfig,get_peft_model
from base_experiment.prompt import base_prompt
from fast_vllm_rollout import _worker_environment

MODEL='Qwen/Qwen3-4B-Base'
REVISION=os.environ['MODEL_REVISION']
CAP=int(os.environ.get('PILOT_TOKENS','8192'))
OUT=Path('runs/harder_4b/fsdp_pilot'); OUT.mkdir(parents=True,exist_ok=True)
rank=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=timedelta(hours=2)); assert dist.get_world_size()==4

def actor_logp(model,ids,start,n):
 out=model(input_ids=ids,use_cache=False).logits[:,start:start+n]
 target=ids[:,start+1:start+1+n]; pieces=[]
 for off in range(0,n,64):
  logits=out[:,off:off+64].float(); y=target[:,off:off+64]
  pieces.append(torch.log_softmax(logits,dim=-1).gather(-1,y.unsqueeze(-1)).squeeze(-1))
 return torch.cat(pieces,dim=1)

class ValueModel(nn.Module):
 def __init__(self):
  super().__init__()
  self.backbone=AutoModel.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.float32,attn_implementation='sdpa')
  self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  self.head=nn.Linear(self.backbone.config.hidden_size,1,dtype=torch.float32)
 def forward(self,ids,start,n):
  h=self.backbone(input_ids=ids,use_cache=False).last_hidden_state[:,start:start+n]
  return self.head(h.float()).squeeze(-1)

def main():
 start=time.monotonic(); torch.cuda.reset_peak_memory_stats(); tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
 cfg=AutoConfig.from_pretrained(MODEL,revision=REVISION)
 rows=[json.loads(x) for x in Path('runs/base_rlvr/prepared/train.jsonl').read_text().splitlines()]
 ids0=tok(base_prompt(rows[0]['problem']),add_special_tokens=False).input_ids
 assert len(ids0)+CAP<=cfg.max_position_embeddings
 payload=[None]; actor=None; tmp=None
 if rank==0:
  actor=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
  actor=get_peft_model(actor,LoraConfig(r=32,lora_alpha=64,lora_dropout=0.0,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
  actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); actor.train()
  tmp=tempfile.TemporaryDirectory(prefix='lora-pilot-',dir='/tmp'); path=Path(tmp.name)
  t=time.monotonic();actor.save_pretrained(path/'adapter');save=time.monotonic()-t
  req={'model':MODEL,'revision':REVISION,'adapter':str(path/'adapter'),'max_model_len':len(ids0)+CAP,
       'prompt_ids':ids0,'max_tokens':CAP,'eos':tok.eos_token_id,'output':str(path/'rollout.json')}
  (path/'request.json').write_text(json.dumps(req)); t=time.monotonic()
  worker_env=_worker_environment()
  for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','TORCHELASTIC_RUN_ID','TORCHELASTIC_RESTART_COUNT','TORCHELASTIC_MAX_RESTARTS'):
   worker_env.pop(key,None)
  worker_env['VLLM_HOST_IP']='127.0.0.1'
  subprocess.run(['/scratch/tsa5252/.venvs/vllm-pilot/bin/python','harder_4b/lora_rollout_worker.py',str(path/'request.json')],env=worker_env,check=True)
  result=json.loads((path/'rollout.json').read_text()); result['adapter_save_seconds']=save;result['rollout_worker_seconds']=time.monotonic()-t
  payload[0]=result
 dist.broadcast_object_list(payload,src=0);roll=payload[0]; tokens=roll['tokens']
 # Repeat real sampled tokens solely to guarantee a full-length update stress test.
 stress=(tokens*((CAP+len(tokens)-1)//len(tokens)))[:CAP];ids=torch.tensor([ids0+stress],device=rank);start_idx=len(ids0)-1
 if rank==0:
  actor=actor.to(rank);actor_opt=torch.optim.AdamW((p for p in actor.parameters() if p.requires_grad),lr=5e-6)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): old=actor_logp(actor,ids,start_idx,CAP).detach()
  torch.cuda.synchronize();t=time.monotonic()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   logp=actor_logp(actor,ids,start_idx,CAP);ratio=(logp-old).exp(); advantage=torch.full_like(ratio,0.5)
   loss=-torch.minimum(ratio*advantage,ratio.clamp(0.8,1.2)*advantage).sum()/CAP
  assert torch.isfinite(loss);loss.backward();actor_opt.step();actor_opt.zero_grad(set_to_none=True)
  torch.cuda.synchronize();actor_seconds=time.monotonic()-t
  actor_trainable=sum(p.numel() for p in actor.parameters() if p.requires_grad)
  del old,logp,loss
  t=time.monotonic(); actor.save_pretrained(path/'updated_adapter'); updated_sync_seconds=time.monotonic()-t
  torch.cuda.empty_cache()
 else:actor_seconds=actor_trainable=updated_sync_seconds=None
 dist.barrier()
 torch.manual_seed(42)
 model=ValueModel(); wrap=functools.partial(transformer_auto_wrap_policy,transformer_layer_cls={Qwen3DecoderLayer})
 critic=FSDP(model,auto_wrap_policy=wrap,sharding_strategy=ShardingStrategy.FULL_SHARD,device_id=rank,use_orig_params=True)
 optimizer=torch.optim.AdamW(critic.parameters(),lr=1e-5)
 torch.cuda.synchronize();dist.barrier();t=time.monotonic()
 with torch.autocast('cuda',dtype=torch.bfloat16):values=critic(ids,start_idx,CAP); loss=(values.float()-0.5).square().mean()
 assert torch.isfinite(loss);loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
 torch.cuda.synchronize();critic_seconds=time.monotonic()-t
 stats=[None]*4; local={'rank':rank,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'critic_seconds':critic_seconds}
 dist.all_gather_object(stats,local)
 if rank==0:
  data={'model':MODEL,'revision':REVISION,'response_cap':CAP,'sampled_tokens':len(tokens),'stress_repeated_tokens':len(tokens)<CAP,'actor':'LoRA rank32 alpha64 attention and MLP','actor_trainable_parameters':actor_trainable,'critic':'separate full transformer, FSDP FULL_SHARD over four GPUs','gradient_checkpointing':True,'rollout':{k:v for k,v in roll.items() if k not in ('tokens','log_probs')},'actor_update_seconds':actor_seconds,'updated_adapter_save_seconds':updated_sync_seconds,'per_gpu':stats,'total_seconds':time.monotonic()-start,'note':'Memory stress test with synthetic advantage and return; not a training result. vLLM engine initialization and generation are reported separately from adapter-save time; live in-engine updated adapter reload latency has not been measured.'}
  (OUT/'summary.json').write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2),flush=True)
  tmp.cleanup()
 dist.barrier();dist.destroy_process_group()

if __name__=='__main__': main()
