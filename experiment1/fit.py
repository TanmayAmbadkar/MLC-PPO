#!/usr/bin/env python3
"""Fit frozen-feature value heads and evaluate against independent continuation outcomes."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, size, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, width), nn.SiLU(), nn.Linear(width, 1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x).squeeze(-1)


class Ensemble(nn.Module):
    def __init__(self, size, layers, width=128):
        super().__init__()
        self.layers = layers
        self.heads = nn.ModuleList(Head(size, width) for _ in layers)
    def forward(self, x):
        return torch.stack([head(x[:, layer]) for head, layer in zip(self.heads, self.layers)], dim=-1)


def load(directory):
    records = []
    for path in sorted(directory.glob('*.pt')):
        obj = torch.load(path, map_location='cpu', weights_only=False)
        for s in obj['states']:
            records.append((obj['problem_id'], s))
    if not records:
        raise ValueError(f'No states in {directory}')
    x = torch.stack([s['features'] for _, s in records]).float()
    y = torch.tensor([s['rewards'].float().mean().item() for _, s in records])
    return {'x': x, 'y': y, 'problem': np.array([i for i, _ in records]),
            'progress': np.array([s['progress'] for _, s in records]),
            'rewards': [s['rewards'].numpy() for _, s in records]}


def select(data, mask):
    return data['x'][mask], data['y'][mask]


def train(model, train_x, train_y, val_x, val_y, epochs, lr, device):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = float('inf')
    state = None
    stale = 0
    for epoch in range(epochs):
        model.train()
        for ix in torch.randperm(len(train_x)).split(64):
            x, y = train_x[ix].to(device), train_y[ix].to(device)
            predictions = model(x)
            # Every head receives the same supervised target.
            loss = (predictions - y[:, None]).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(val_x.to(device)).mean(-1).cpu()
            score = (pred - val_y).square().mean().item()
        if score < best - 1e-6:
            best = score
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 15:
                break
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='runs/experiment1/data')
    p.add_argument('--output-dir', default='runs/experiment1/results')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--bootstrap', type=int, default=2000)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(a.data_dir)
    tr, val, test = [load(root / name) for name in ('train', 'val', 'test')]
    # Reserve entire training problems for gate fitting; all heads share the remaining problems.
    problems = np.unique(tr['problem'])
    rng = np.random.default_rng(a.seed)
    rng.shuffle(problems)
    gate_problems = problems[:max(1, len(problems)//5)]
    gate_mask = np.isin(tr['problem'], gate_problems)
    fit_mask = ~gate_mask
    if not fit_mask.any() or not gate_mask.any():
        raise ValueError('Need at least two training problems for head/gate split')
    xfit, yfit = select(tr, fit_mask)
    xgate, ygate = select(tr, gate_mask)
    size = tr['x'].shape[-1]
    configs = {'final_head': ([2], 128), 'final_ensemble': ([2, 2, 2], 128),
               'multilayer_mean': ([0, 1, 2], 128), 'final_large': ([2], 384)}
    models = {}
    output = Path(a.output_dir); output.mkdir(parents=True, exist_ok=True)
    for name, (layers, width) in configs.items():
        torch.manual_seed(a.seed)
        model = train(Ensemble(size, layers, width), xfit, yfit, val['x'], val['y'], a.epochs, a.learning_rate, device)
        models[name] = model
        torch.save(model.state_dict(), output / f'{name}.pt')
    # A softmax gate on head predictions, trained only on separate training problems.
    base = models['multilayer_mean']
    with torch.no_grad():
        gate_x = base(xgate.to(device)).detach()
    gate = nn.Linear(3, 3).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=a.learning_rate)
    best, gate_state, stale = float('inf'), None, 0
    with torch.no_grad():
        gate_val = base(val['x'].to(device)).detach()
    for epoch in range(a.epochs):
        gate.train(); opt.zero_grad()
        weights = torch.softmax(gate(gate_x), dim=-1)
        loss = ((weights * gate_x).sum(-1) - ygate.to(device)).square().mean()
        loss.backward(); opt.step()
        gate.eval()
        with torch.no_grad():
            pred = (torch.softmax(gate(gate_val), dim=-1) * gate_val).sum(-1)
            score = (pred.cpu() - val['y']).square().mean().item()
        if score < best - 1e-6:
            best = score; gate_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= 15: break
    gate.load_state_dict(gate_state); gate.eval()
    torch.save(gate.state_dict(), output / 'gate.pt')
    x = test['x'].to(device)
    predictions = {}
    timings = {}
    with torch.no_grad():
        for name, model in models.items():
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            pred = model(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            timings[name] = (time.perf_counter() - start) / len(x)
            predictions[name] = pred.mean(-1).cpu().numpy()
            if name == 'multilayer_mean':
                predictions['multilayer_min'] = pred.min(-1).values.cpu().numpy()
                predictions['multilayer_gate'] = (torch.softmax(gate(pred), -1) * pred).sum(-1).cpu().numpy()
    ids = test['problem']; unique = np.unique(ids)
    draws = [rng.choice(unique, size=len(unique), replace=True) for _ in range(a.bootstrap)]
    report = {'seed': a.seed, 'counts': {n: {'states': len(d['y']), 'problems': len(np.unique(d['problem']))} for n,d in [('train',tr),('val',val),('test',test)]},
              'gate_training_problems': gate_problems.tolist(), 'parameter_counts': {n: sum(p.numel() for p in m.parameters()) for n,m in models.items()},
              'head_forward_seconds_per_state': timings, 'methods': {}}
    for name, pred in predictions.items():
        y = test['y'].numpy(); errors = (pred-y)**2
        brier = np.array([np.mean((p-r)**2) for p,r in zip(pred, test['rewards'])])
        boot = []
        for sample in draws:
            ix = np.concatenate([np.flatnonzero(ids == problem) for problem in sample])
            boot.append(errors[ix].mean())
        report['methods'][name] = {'mse': float(errors.mean()), 'brier': float(brier.mean()),
            'mse_problem_bootstrap_95ci': np.quantile(boot, [0.025, 0.975]).tolist(),
            'mse_by_progress': {str(f): float(errors[test['progress'] == f].mean()) for f in (0.25,0.5,0.75)}}
    # Descriptive difficulty groups use held-out rollout success rates, not model selection.
    problem_values = {int(problem): float(test['y'].numpy()[ids == problem].mean()) for problem in unique}
    for name, pred in predictions.items():
        errors = (pred - test['y'].numpy())**2
        groups = {}
        for label, lower, upper in [('low', 0.0, 1/3), ('medium', 1/3, 2/3), ('high', 2/3, 1.000001)]:
            mask = np.array([lower <= problem_values[int(problem)] < upper for problem in ids])
            groups[label] = {'states': int(mask.sum()), 'mse': float(errors[mask].mean()) if mask.any() else None}
        report['methods'][name]['mse_by_observed_problem_success'] = groups
    # Paired bootstrap of C - B, preserving the problem as the resampling unit.
    delta = (predictions['multilayer_mean'] - test['y'].numpy())**2 - (predictions['final_ensemble'] - test['y'].numpy())**2
    diffs = [np.concatenate([delta[ids == problem] for problem in sample]).mean() for sample in draws]
    report['paired_multilayer_minus_final_ensemble_mse'] = {'difference': float(delta.mean()), 'problem_bootstrap_95ci': np.quantile(diffs, [0.025, 0.975]).tolist()}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for name in ('final_head', 'final_ensemble', 'multilayer_mean', 'final_large'):
            scores = report['methods'][name]['mse_by_progress']
            ax.plot([25, 50, 75], [scores[str(f)] for f in (0.25, 0.5, 0.75)], marker='o', label=name)
        ax.set(xlabel='Response progress (%)', ylabel='MSE against Monte Carlo reference')
        ax.legend(); fig.tight_layout(); fig.savefig(output / 'mse_by_progress.png', dpi=160); plt.close(fig)
    except ImportError:
        print('matplotlib unavailable; metrics are saved without the plot', flush=True)
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__': main()
