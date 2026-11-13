import sys, unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_viability import group_advantages,actor_surrogate,viability_layers
class TestViability(unittest.TestCase):
 def test_all_layers_selects_every_layer_including_first_and_last(self):
  layers,aggregation=viability_layers('ppo','multilayer_mean',28,'0.5,0.75,1.0',True)
  self.assertEqual(layers,list(range(1,29)));self.assertEqual(aggregation,'mean')
  with self.assertRaises(ValueError):viability_layers('grpo','multilayer_mean',28,'0.5,0.75,1.0',True)
 def test_grpo_group_normalization_and_zero_signal(self):
  r=torch.tensor([0.,0.,1.,1.,1.,1.,1.,1.]);a=group_advantages(r,4)
  self.assertTrue(torch.allclose(a[:4].mean(),torch.tensor(0.),atol=1e-7));self.assertTrue(torch.equal(a[4:],torch.zeros(4)));self.assertLess(a[0],0);self.assertGreater(a[2],0)
 def test_trajectory_normalization(self):
  mask=torch.tensor([[1,0,0],[1,1,1]],dtype=torch.bool);r=torch.ones(2,3);self.assertEqual(actor_surrogate(r,r,mask,.2).item(),-2.)
if __name__=='__main__':unittest.main()
