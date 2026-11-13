#!/usr/bin/env python3
"""Load a saved policy and matching critic; run finite inference through both."""
import argparse,sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mlcppo import MultiLayerCritic
p=argparse.ArgumentParser();p.add_argument('checkpoint');a=p.parse_args();path=Path(a.checkpoint)
tokenizer=AutoTokenizer.from_pretrained(path)
model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
data=torch.load(path/'critic.pt',map_location='cpu',weights_only=True)
critic=MultiLayerCritic(model.config.hidden_size,data['layers'],head_size=data['head_size'],aggregation=data['aggregation']).cuda().eval()
critic.load_state_dict(data['critic'],strict=True)
ids=tokenizer('Solve: 2 + 2 =',return_tensors='pt').input_ids.cuda()
with torch.inference_mode():
 out=model(ids,output_hidden_states=True,use_cache=False)
 vals,_,_=critic([out.hidden_states[k][:,-1:] for k in data['layers']])
assert torch.isfinite(out.logits).all() and torch.isfinite(vals).all()
print('checkpoint loaded:',path,'critic layers:',data['layers'],'aggregation:',data['aggregation'],'value:',vals.item(),flush=True)
