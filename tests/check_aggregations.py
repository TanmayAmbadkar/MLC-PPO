#!/usr/bin/env python3
"""Check aggregation values, head gradients, and detached input features."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mlcppo import MultiLayerCritic,critic_loss
for mode in ('mean','min','gate'):
    torch.manual_seed(4)
    model=MultiLayerCritic(8,[1,2,3],head_size=4,aggregation=mode)
    for head,bias in zip(model.heads,(-2.,0.,2.)):
        head[-1].bias.data.fill_(bias)
    features=[torch.randn(2,3,8,requires_grad=True) for _ in range(3)]
    baseline,heads,weights=model(features)
    expected={'mean':heads.mean(-1),'min':heads.min(-1).values,'gate':(heads*weights).sum(-1)}[mode]
    assert torch.allclose(baseline,expected)
    assert torch.allclose(weights.sum(-1),torch.ones_like(baseline))
    loss=critic_loss(baseline,heads,torch.tensor([0.,1.]),torch.ones(2,3,dtype=torch.bool),.25)
    loss.backward()
    assert all(x.grad is None for x in features)
    assert all(h[-1].bias.grad is not None and torch.isfinite(h[-1].bias.grad).all() for h in model.heads)
    if mode=='gate':assert any(g.weight.grad is not None for g in model.gates)
    else:assert all(g.weight.grad is None for g in model.gates)
    print(mode,'ok')
