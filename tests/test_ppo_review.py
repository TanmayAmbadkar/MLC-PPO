"""Regression checks for PPO objective, critic isolation, PEFT persistence and checkpointing."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from peft import LoraConfig,get_peft_model,PeftModel
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_base_viability import (actor_surrogate,conditional_reference_kl,critic_architecture_config,
                           disable_dropout,load_policy_tokenizer,save_checkpoint)
from mlcppo import MultiLayerCritic


def tiny():
 return Qwen2ForCausalLM(Qwen2Config(vocab_size=48,hidden_size=16,intermediate_size=32,num_hidden_layers=4,
                 num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,attention_dropout=0.0))

class TokenizerStub:
 def save_pretrained(self,path): Path(path,'tokenizer_stub').write_text('ok')

class ReviewTests(unittest.TestCase):
 def test_actor_length_and_microbatch_invariance(self):
  r=torch.ones(2,3);a=torch.ones(2,3);m=torch.tensor([[1,0,0],[1,1,1]],dtype=torch.bool)
  self.assertEqual(actor_surrogate(r,a,m,.2).item(),-2.0)
  split=sum(actor_surrogate(r[i:i+1],a[i:i+1],m[i:i+1],.2)/2 for i in range(2))
  self.assertTrue(torch.allclose(actor_surrogate(r,a,m,.2),split))
 def test_reference_kl(self):
  x=torch.tensor([[[1.,0.,-1.],[0.,1.,2.]]],requires_grad=True);m=torch.tensor([[True,False]])
  same=conditional_reference_kl(x,x.detach(),m,1.0);self.assertAlmostEqual(same.item(),0,places=6)
  other=conditional_reference_kl(x,torch.zeros_like(x),m,1.0);self.assertGreater(other.item(),0)
  other.backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertTrue(torch.equal(x.grad[:,1],torch.zeros_like(x.grad[:,1])))
 def test_critic_architectures(self):
  self.assertEqual(critic_architecture_config('final_single',4,'0.5,0.75,1.0'),([4],'mean'))
  self.assertEqual(critic_architecture_config('final_ensemble',4,'0.5,0.75,1.0'),([4,4,4],'mean'))
  for mode in ('mean','min','gate'):
   self.assertEqual(critic_architecture_config('multilayer_'+mode,4,'0.5,0.75,1.0'),([2,3,4],mode))
 def test_checkpointing_is_executed(self):
  model=tiny();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  disable_dropout(model);model.train();layer=model.model.layers[0];old=layer._gradient_checkpointing_func;calls=[]
  def spy(*args,**kwargs):calls.append(1);return old(*args,**kwargs)
  layer._gradient_checkpointing_func=spy
  model(torch.tensor([[1,2,3,4]]),use_cache=False).logits.sum().backward()
  self.assertTrue(calls);self.assertTrue(torch.isfinite(model.model.layers[0].self_attn.q_proj.weight.grad).all())
 def test_adapter_only_tokenizer_fallback(self):
  sentinel=object()
  with patch('train_base_viability.AutoTokenizer.from_pretrained',side_effect=[OSError('no tokenizer'),sentinel]) as factory:
   self.assertIs(load_policy_tokenizer('base-model',Path('/adapter')),sentinel)
   self.assertEqual(factory.call_args_list[1].args[0],'base-model')
 def test_full_adapter_state_and_critic_reload(self):
  torch.manual_seed(3);model=get_peft_model(tiny(),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj'],task_type='CAUSAL_LM'))
  for p in model.parameters():p.requires_grad_(True)
  with torch.no_grad():model.get_base_model().model.embed_tokens.weight.add_(0.25)
  disable_dropout(model);model.eval();ids=torch.tensor([[1,2,3]])
  with torch.no_grad():expected=model(ids,use_cache=False).logits
  critic=MultiLayerCritic(16,[2,3,4],head_size=4,aggregation='mean')
  with tempfile.TemporaryDirectory() as folder:
   save_checkpoint(model,TokenizerStub(),critic,Path(folder),1,save_full_adapter_state=True)
   path=Path(folder)/'checkpoint-1'
   restored=PeftModel.from_pretrained(tiny(),str(path),is_trainable=True)
   restored.load_state_dict(torch.load(path/'full_policy_state.pt',map_location='cpu',weights_only=True),strict=True)
   disable_dropout(restored);restored.eval()
   with torch.no_grad():actual=restored(ids,use_cache=False).logits
   self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
   data=torch.load(path/'critic.pt',map_location='cpu',weights_only=True)
   reloaded=MultiLayerCritic(16,data['layers'],head_size=data['head_size'],aggregation=data['aggregation'])
   reloaded.load_state_dict(data['critic'],strict=True)
   for key,value in critic.state_dict().items():self.assertTrue(torch.equal(value,reloaded.state_dict()[key]))
if __name__=='__main__':unittest.main()
