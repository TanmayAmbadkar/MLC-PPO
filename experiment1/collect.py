#!/usr/bin/env python3
"""Collect frozen-policy GSM8K prefix features and independent continuations."""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_mlc_ppo import answer_reward


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', default='runs/experiment1/data')
    p.add_argument('--model', default='Qwen/Qwen2.5-Math-1.5B-Instruct')
    p.add_argument('--train-problems', type=int, default=200)
    p.add_argument('--val-problems', type=int, default=50)
    p.add_argument('--test-problems', type=int, default=50)
    p.add_argument('--rollouts', default='8,16,32')
    p.add_argument('--trajectories', type=int, default=2)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=1)
    a = p.parse_args()
    if a.world_size < 1 or not 0 <= a.rank < a.world_size:
        raise ValueError('rank must be in [0, world-size)')
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / 'config.json'
    config = {k: v for k, v in vars(a).items() if k not in ('rank', 'world_size')}
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing collection configuration differs; use a new output directory')
    if not config_path.exists():
        temporary_config = out / f'config.rank{a.rank}.tmp'
        temporary_config.write_text(json.dumps(config, indent=2))
        temporary_config.replace(config_path)
    data = load_dataset('openai/gsm8k', 'main', split='train')
    # Deduplicate by question before splitting so repeated questions cannot leak.
    seen_questions = set()
    ids = []
    for idx, row in enumerate(data):
        question = ' '.join(row['question'].split())
        if question not in seen_questions:
            seen_questions.add(question)
            ids.append(idx)
    random.Random(a.seed).shuffle(ids)
    ntr, nv, nt = a.train_problems, a.val_problems, a.test_problems
    if ntr + nv + nt > len(ids):
        raise ValueError('Requested more problems than available')
    splits = [('train', ids[:ntr]), ('val', ids[ntr:ntr+nv]), ('test', ids[ntr+nv:ntr+nv+nt])]
    ns = [int(x) for x in a.rollouts.split(',')]
    if len(ns) != 3 or min(ns) < 1:
        raise ValueError('Provide positive train,val,test rollout counts')
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    for param in model.parameters():
        param.requires_grad_(False)
    layers = [round(model.config.num_hidden_layers * f) for f in (0.5, 0.75, 1.0)]
    gen = dict(do_sample=True, temperature=a.temperature, top_p=a.top_p,
               max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id,
               eos_token_id=tok.eos_token_id, use_cache=True)
    for (split, problem_ids), count in zip(splits, ns):
        dest = out / split
        dest.mkdir(exist_ok=True)
        for number, idx in enumerate(problem_ids):
            if number % a.world_size != a.rank:
                continue
            path = dest / f'{idx}.pt'
            if path.exists():
                continue
            # Seed each problem independently, making resumed collections reproducible.
            torch.manual_seed(a.seed + idx)
            torch.cuda.manual_seed_all(a.seed + idx)
            row = data[idx]
            prompt = 'Solve the following math problem. Show your reasoning, then put the final answer in \\boxed{...}.\n\n' + row['question']
            text = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
            prompt_ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
            states = []
            with torch.inference_mode():
                for trajectory in range(a.trajectories):
                    sample = model.generate(prompt_ids, **gen)[0, prompt_ids.shape[1]:]
                    if tok.eos_token_id is not None:
                        eos = (sample == tok.eos_token_id).nonzero(as_tuple=True)[0]
                        if len(eos):
                            sample = sample[:int(eos[0])]
                    if len(sample) < 4:
                        print(f'Skipping short trajectory problem={idx} trajectory={trajectory} tokens={len(sample)}', flush=True)
                        continue
                    for fraction in (0.25, 0.5, 0.75):
                        length = max(1, int(len(sample) * fraction))
                        prefix = torch.cat((prompt_ids[0], sample[:length]))[None, :]
                        output = model(prefix, output_hidden_states=True, use_cache=False)
                        features = torch.stack([output.hidden_states[layer][0, -1].float().cpu() for layer in layers])
                        rewards = []
                        for start in range(0, count, a.batch_size):
                            batch = min(a.batch_size, count - start)
                            continuation_options = dict(gen, max_new_tokens=a.max_new_tokens - length)
                            generated = model.generate(prefix.repeat(batch, 1), **continuation_options)
                            # Reward the full response, including the already sampled prefix.
                            response = generated[:, prompt_ids.shape[1]:]
                            texts = tok.batch_decode(response, skip_special_tokens=True)
                            rewards.extend(answer_reward(t, row['answer']) for t in texts)
                        states.append({'features': features, 'rewards': torch.tensor(rewards),
                                       'progress': fraction, 'trajectory': trajectory,
                                       'prefix_length': length})
            temp = path.with_suffix('.tmp')
            torch.save({'problem_id': idx, 'split': split, 'states': states, 'layers': layers}, temp)
            temp.replace(path)
            print(f'{split} problem {number+1}/{len(problem_ids)} id={idx} states={len(states)}', flush=True)

if __name__ == '__main__':
    main()
