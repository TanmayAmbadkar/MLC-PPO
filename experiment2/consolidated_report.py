#!/usr/bin/env python3
"""Combine finished frozen-value and held-out policy evaluations without causal claims."""
import json
from pathlib import Path
import numpy as np
root=Path('runs/experiment2');value=json.loads((root/'results/metrics.json').read_text());difficulty=json.loads((root/'difficulty/summary.json').read_text())
evaluation=json.loads(Path('runs/dapo_eval/summary.json').read_text())
shared=json.loads((root/'resource_smoke/shared.json').read_text());full=json.loads((root/'resource_smoke/full.json').read_text())
methods=['constant','final_head','final_ensemble','multilayer_mean','multilayer_min','multilayer_gate']
lines=['# DAPO frozen-critic and policy evaluation','',
       'Frozen state-value estimates and policy checkpoint accuracy measure different outcomes. Neither establishes a causal effect from critic architecture.', '',
       '## Difficulty sampling','',f"{difficulty['problems']} problems, {difficulty['responses']} responses; {difficulty['responses_without_eos']} responses ended without EOS. Difficulty groups use independent prompt-level samples.",'',
       '## Held-out critic metrics','',
       '| Critic | Mean MSE across seeds | Mean Brier across seeds | Mean calibration ECE |', '|---|---:|---:|---:|']
for method in methods:
    values=[x['methods'][method] for x in value['seeds'].values()]
    lines.append(f"| {method} | {np.mean([v['mse'] for v in values]):.5f} | {np.mean([v['brier'] for v in values]):.5f} | {np.mean([v['calibration_ece_10bin'] for v in values]):.5f} |")
lines+=['','MSE uses independent finite-rollout state-value references. Brier scores use individual continuation outcomes. Calibration ECE uses 10 bins. Each critic seed is reported separately below.','',
        '| Seed | Mean minus final-ensemble MSE | 95% paired problem bootstrap interval |','|---|---:|---:|']
for seed,entry in value['seeds'].items():
    d=entry['paired_multilayer_minus_final_ensemble'];ci=d['problem_bootstrap_95ci'];lines.append(f"| {seed} | {d['mse_difference']:.5f} | [{ci[0]:.5f}, {ci[1]:.5f}] |")
lines+=['','## Independently measured difficulty groups','', '| Seed | Group | Problems | Mean-head MSE | Final-ensemble MSE | Mean minus ensemble [95% CI] |','|---|---|---:|---:|---:|---:|']
for seed,entry in value['seeds'].items():
    for group,paired in entry['paired_multilayer_minus_final_ensemble_by_difficulty'].items():
        ma=entry['methods']['multilayer_mean']['mse_by_independent_difficulty'][group];fb=entry['methods']['final_ensemble']['mse_by_independent_difficulty'][group]
        if paired['problems']:
            ci=paired['problem_bootstrap_95ci'];lines.append(f"| {seed} | {group} | {paired['problems']} | {ma['mse']:.5f} | {fb['mse']:.5f} | {paired['mse_difference']:.5f} [{ci[0]:.5f}, {ci[1]:.5f}] |")
lines+=['','Per-group calibration bins, bias and Brier scores are in `results/metrics.json`. The observed groups reflect eight independent prompt-level rollouts; they are not true difficulty labels.','',
        '## Greedy held-out checkpoint accuracy','', '| Model | Correct | Accuracy | Truncated | Difference vs base [95% CI] |','|---|---:|---:|---:|---:|']
for name,entry in evaluation['models'].items():
    if name=='base':diff='—'
    else:
        d=evaluation['paired_vs_base'][name];ci=d['problem_bootstrap_95ci'];diff=f"{d['accuracy_difference']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
    lines.append(f"| {name} | {entry['correct']}/{evaluation['problems']} | {entry['accuracy']:.3f} | {entry['truncated']} | {diff} |")
lines+=['','The policy checkpoints each use one training seed. Decoding and the verifier are identical across models; truncations retain their verifier scores.','',
        '## Resource smoke','',f"At batch {shared['batch_size']}, sequence length {shared['sequence_length']}, and {shared['steps']} frozen-actor critic steps, shared heads used {shared['peak_gpu_allocated_bytes']/1e9:.2f} GB peak allocated and {shared['policy_tokens_per_second_including_critic_update']:.0f} policy tokens/s; a separate full critic used {full['peak_gpu_allocated_bytes']/1e9:.2f} GB and {full['policy_tokens_per_second_including_critic_update']:.0f} tokens/s. This short benchmark does not include an actor optimizer step or represent full PPO memory.",'',
        'Cached-feature throughput, per-architecture training time, parameter counts, and peak head-only memory are in `results/metrics.json`. The model and its retained hidden states are excluded from head-only metrics.','',
        '## Limitations','',
        '- Math-Verify checks consistency with its configured reward semantics; it does not certify gold-label correctness.',
        '- DAPO overlap checks cover exact and normalized text for MATH-500 and AIME 2024/2025, not paraphrases or all benchmark years.',
        '- The frozen-study gate takes head predictions as input and was trained on one half of validation problems; early stopping used the other half. The PPO gate instead reads layer features directly.',
        '- Token-budget truncation changes observed reward; record counts before comparing conclusions.',
        '- Multiple architecture metrics were examined; uncertainty intervals are descriptive and not multiplicity-adjusted.','']
(root/'report.md').write_text('\n'.join(lines));print('Wrote',root/'report.md')
