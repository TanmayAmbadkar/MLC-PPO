import sys,tempfile,unittest
from pathlib import Path
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_full_critic import FullValueCritic
class FullCriticTest(unittest.TestCase):
 def test_separate_trainable_backbone_and_scalar_head(self):
  config=Qwen2Config(vocab_size=48,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,attention_dropout=0)
  with tempfile.TemporaryDirectory() as d:
   actor=Qwen2ForCausalLM(config);actor.save_pretrained(d);critic=FullValueCritic(d);critic.train()
   ids=torch.tensor([[1,2,3,4]]);mask=torch.ones_like(ids);v,h=critic.values(ids,mask,1,2)
   self.assertEqual(tuple(v.shape),(1,2));self.assertTrue(torch.allclose(v,torch.full_like(v,.5)));self.assertIsNot(actor.model.embed_tokens.weight,critic.backbone.embed_tokens.weight)
   with torch.no_grad():critic.value_head.weight.fill_(.1)
   v,_=critic.values(ids,mask,1,2);loss=torch.nn.functional.binary_cross_entropy(v,torch.ones_like(v));loss.backward();self.assertIsNotNone(critic.value_head.weight.grad);self.assertIsNotNone(critic.backbone.layers[-1].self_attn.q_proj.weight.grad);self.assertTrue(all(p.grad is None for p in actor.parameters()))
if __name__=='__main__':unittest.main()
