#!/usr/bin/env python3
"""Single-GPU PPO with a detached multi-layer critic and exact-match RLVR."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import re
import string
import time
import os
import hashlib
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

try:
    from peft import PeftConfig, PeftModel
except ImportError:  # PEFT is only needed when --adapter is supplied.
    PeftConfig = PeftModel = None

from mlcppo import MultiLayerCritic, critic_loss, resolve_layers

LOG = logging.getLogger("mlc_ppo")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-synthetic-rewards", action="store_true", help="Throughput benchmark only; replace rewards with alternating binary values")
    p.add_argument("--algorithm", choices=("ppo", "grpo"), default="ppo")
    p.add_argument("--all-layers", action="store_true", help="Attach one critic to every transformer layer")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--stop-after-step", type=int, default=100)
    p.add_argument("--token-budget", type=int, default=20000000)
    p.add_argument("--resume-training", action="store_true")
    p.add_argument("--model", help="HF model id or local model directory")
    p.add_argument("--adapter", help="Optional PEFT adapter; base model is read from its config")
    p.add_argument("--train-file", required=True, help="JSONL containing prompt and answer fields")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prompt-key", default="prompt")
    p.add_argument("--answer-key", default="answer")
    p.add_argument("--reward-type", choices=("exact_match", "trl_math"), default="exact_match",
                   help="trl_math uses the shared Math-Verify reward for DAPO")
    p.add_argument("--critic-layers", default="0.5,0.75,1.0",
                   help="Fractions of transformer depth or explicit 1-based layer numbers")
    p.add_argument("--critic-architecture", choices=("final_single", "final_ensemble", "multilayer_mean", "multilayer_min", "multilayer_gate"), default=None)
    p.add_argument("--critic-aggregation", choices=("mean", "min", "gate"), default=None,
                   help="Legacy multi-layer aggregation switch; prefer --critic-architecture")
    p.add_argument("--critic-checkpoint", help="Load compatible critic.pt before warm-up or PPO")
    p.add_argument("--critic-warmup-steps", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Number of distinct prompts and rollouts per PPO batch")
    p.add_argument("--micro-batch-size", type=int, default=1,
                   help="Rollout and gradient micro-batch size; gradients accumulate over the rollout batch")
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--critic-epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=5e-7)
    p.add_argument("--critic-learning-rate", type=float, default=1e-4)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--critic-aux-weight", type=float, default=0.25)
    p.add_argument("--kl-coefficient", type=float, default=0.0,
                   help="Exact conditional KL against the frozen initial policy; 0 avoids loading a reference")
    p.add_argument("--log-rollouts", action="store_true", help="Save responses, verifier outputs and gold answers as JSONL")
    p.add_argument("--verify-reward-parity", action="store_true", help="Check each DAPO reward against TRL accuracy_reward")
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--generation-temperature", type=float, default=1.0)
    p.add_argument("--generation-top-p", type=float, default=1.0,
                   help="PPO currently requires 1.0 to preserve full policy support")
    p.add_argument("--ratio-tolerance", type=float, default=0.01,
                   help="Maximum allowed absolute error from ratio 1 before the first update")
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--no-save", action="store_true", help="Skip model checkpoints (useful for smoke checks)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-examples", type=int)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--train-full-model", action="store_true",
                   help="Train all policy parameters (default: train PEFT adapter if supplied)")
    return p.parse_args()


def load_rows(path: str, prompt_key: str, answer_key: str, max_examples: int | None):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt, answer = row.get(prompt_key), row.get(answer_key)
            if not isinstance(prompt, str) or answer is None:
                raise ValueError(f"Line {line_no} must contain string {prompt_key!r} and {answer_key!r}")
            rows.append({"prompt": prompt, "answer": str(answer)})
    if max_examples:
        rows = rows[:max_examples]
    if not rows:
        raise ValueError(f"No records loaded from {path}")
    return rows


def normalize_answer(text: str) -> str:
    """Normalize common short-answer formatting, including boxed math answers."""
    text = text.strip()
    boxed = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for position in range(start, len(text)):
            if text[position] == '{':
                depth += 1
            elif text[position] == '}':
                depth -= 1
                if depth == 0:
                    boxed.append(text[start:position])
                    break
    if boxed:
        text = boxed[-1]
    elif "####" in text:
        text = text.rsplit("####", 1)[-1]
    elif "final answer:" in text.lower():
        text = re.split(r"final answer:", text, flags=re.I)[-1]
    text = text.strip().lower().replace(",", "")
    text = re.sub(r"\s+", "", text)
    text = text.strip(string.whitespace + ";:$")
    text = text.rstrip('.')
    return text


def answer_reward(completion: str, answer: str) -> float:
    pred = normalize_answer(completion)
    target = normalize_answer(answer)
    if not target:
        return 0.0
    if pred == target:
        return 1.0
    # Compare plain decimal answers numerically (for example 13 and 13.00).
    # Structured, multi-value, and symbolic answers remain exact matches.
    numeric = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
    if re.fullmatch(numeric, pred) and re.fullmatch(numeric, target):
        try:
            return float(Decimal(pred) == Decimal(target))
        except InvalidOperation:
            pass
    return 0.0


def format_prompt(tokenizer, text: str, model_id: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            LOG.warning("Tokenizer chat template failed (%s); using plain text", exc)
    if "gemma-3" in model_id.lower():
        return f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
    return text.rstrip() + "\n"


def load_policy_tokenizer(model_id, adapter_path):
    if adapter_path:
        try:
            return AutoTokenizer.from_pretrained(str(adapter_path), use_fast=True, trust_remote_code=True, local_files_only=True)
        except (OSError, ValueError):
            LOG.info("Adapter has no usable tokenizer; loading from base model %s", model_id)
    return AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)


def critic_architecture_config(architecture, num_layers, layer_spec):
    if architecture == "final_single":
        return [num_layers], "mean"
    if architecture == "final_ensemble":
        return [num_layers] * 3, "mean"
    if architecture not in {"multilayer_mean", "multilayer_min", "multilayer_gate"}:
        raise ValueError(f"Unknown critic architecture {architecture}")
    return resolve_layers(layer_spec, num_layers), architecture.removeprefix("multilayer_")


def sequence_log_probs(logits, tokens, temperature=1.0):
    return torch.log_softmax(logits.float() / temperature, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def position_ids_from_mask(attention_mask):
    return (attention_mask.long().cumsum(dim=-1) - 1).masked_fill(attention_mask == 0, 0)


def disable_dropout(model):
    """Disable common module/functional dropout paths without disabling train mode."""
    names = ("attention_dropout", "attention_dropout_prob", "attention_probs_dropout_prob",
             "attn_dropout", "attn_pdrop", "resid_pdrop", "embd_pdrop", "dropout_rate", "dropout")
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.p = 0.0
        for name in names:
            value = getattr(module, name, None)
            if isinstance(value, float) and value != 0.0:
                setattr(module, name, 0.0)


def actor_surrogate(ratio, advantages, mask, epsilon):
    """Sum token contributions per trajectory; normalize only by batch size."""
    loss = -torch.minimum(ratio * advantages, ratio.clamp(1-epsilon, 1+epsilon) * advantages)
    return (loss * mask).sum() / mask.shape[0]


def group_advantages(scores, group_size):
    if scores.numel() % group_size or group_size < 2:
        raise ValueError('Invalid GRPO group shape')
    groups = scores.reshape(-1, group_size)
    return ((groups - groups.mean(dim=1, keepdim=True)) /
            (groups.std(dim=1, unbiased=False, keepdim=True) + 1e-4)).flatten()


def resume_invariants(args):
    keys = ('algorithm', 'critic_architecture', 'batch_size', 'micro_batch_size', 'group_size',
            'token_budget', 'seed', 'ppo_epochs', 'critic_epochs', 'learning_rate',
            'critic_learning_rate', 'clip_epsilon', 'critic_aux_weight', 'generation_temperature',
            'generation_top_p', 'max_prompt_length', 'max_completion_length', 'reward_type')
    values = {key: getattr(args, key) for key in keys}
    if args.all_layers:
        values['all_layers'] = True
    values['train_sha256'] = hashlib.sha256(Path(args.train_file).read_bytes()).hexdigest()
    return values


def viability_layers(algorithm, architecture, num_layers, layer_spec, all_layers):
    layers, aggregation = critic_architecture_config(architecture, num_layers, layer_spec)
    if all_layers:
        if algorithm != 'ppo' or architecture != 'multilayer_mean':
            raise ValueError('--all-layers requires PPO and multilayer_mean')
        layers = list(range(1, num_layers + 1))
    return layers, aggregation


def conditional_reference_kl(logits, reference_logits, mask, temperature, chunk_size=16):
    """Exact token-conditional KL at old-policy sampled states, summed per trajectory."""
    total = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, logits.shape[1], chunk_size):
        end = min(start + chunk_size, logits.shape[1])
        policy = torch.log_softmax(logits[:, start:end].float() / temperature, dim=-1)
        reference = torch.log_softmax(reference_logits[:, start:end].float() / temperature, dim=-1)
        per_state = (policy.exp() * (policy - reference)).sum(dim=-1)
        total = total + (per_state * mask[:, start:end]).sum()
    return total / mask.shape[0]


def extract_features(outputs, layers, state_start, time_len):
    hidden_states = outputs.hidden_states
    return [hidden_states[layer][:, state_start:state_start + time_len].detach()
            for layer in layers]


def response_mask(completions: torch.Tensor, eos_id: int, pad_id: int):
    # Include the first EOS as an action; ignore tokens generated after EOS.
    valid = torch.ones_like(completions, dtype=torch.bool)
    if eos_id is not None:
        eos = completions.eq(eos_id)
        seen_before = torch.cat(
            [torch.zeros_like(eos[:, :1]), eos[:, :-1].cumsum(dim=1) > 0], dim=1
        )
        valid &= ~seen_before
        # If no EOS was emitted, the generation length is the full tensor width.
    return valid


def build_rollout(model, tokenizer, critic, batch, device, args, layers):
    if args.generation_temperature <= 0 or args.generation_top_p != 1.0:
        raise ValueError("PPO rollout requires positive temperature and top-p 1.0")
    prompts = [r["prompt"] + ("\n\nSolve step by step and put your final answer in \\boxed{...}."
                              if args.reward_type == "trl_math" else "") for r in batch]
    encoded = tokenizer(
        [format_prompt(tokenizer, text, args.model_id) for text in prompts], return_tensors="pt", padding=True,
        truncation=False, add_special_tokens=False,
    )
    if encoded.input_ids.shape[1] > args.max_prompt_length:
        raise ValueError(f"Prompt exceeds --max-prompt-length ({encoded.input_ids.shape[1]} > {args.max_prompt_length}); refusing to truncate it")
    prompt_ids = encoded.input_ids.to(device)
    prompt_mask = encoded.attention_mask.to(device)
    prompt_width = prompt_ids.shape[1]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            do_sample=True, temperature=args.generation_temperature, top_p=1.0,
            top_k=0, num_beams=1, repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            use_cache=True, max_new_tokens=args.max_completion_length,
            return_dict_in_generate=True, output_scores=True,
        )
        generation = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            generation_config=generation_config,
        )
        generated = generation.sequences
        completions = generated[:, prompt_width:]
        # Use the actual processed sampling scores, including temperature.
        old_log_probs = torch.cat([
            sequence_log_probs(torch.stack(generation.scores[start:start + 8], dim=1),
                               completions[:, start:start + 8])
            for start in range(0, len(generation.scores), 8)
        ], dim=1)
        del generation
        mask = response_mask(completions, tokenizer.eos_token_id, tokenizer.pad_token_id)
        full_mask = torch.cat([prompt_mask, mask.long()], dim=1)
        if critic is None:
            features = []
            baseline = torch.zeros_like(mask, dtype=torch.float32)
            head_values = baseline.unsqueeze(-1)
        else:
            outputs = model(input_ids=generated, attention_mask=full_mask,
                position_ids=position_ids_from_mask(full_mask), output_hidden_states=True, use_cache=False)
            state_start = prompt_width - 1
            features = extract_features(outputs, layers, state_start, completions.shape[1])
            baseline, head_values, _ = critic(features)
        rewards_text = tokenizer.batch_decode(completions, skip_special_tokens=True)
        if args.reward_type == "trl_math":
            from experiment2.reward import reward as score_answer
        else:
            score_answer = answer_reward
        scores = [score_answer(text, row["answer"]) for text, row in zip(rewards_text, batch)]
        if getattr(args, "verify_reward_parity", False):
            if args.reward_type != "trl_math":
                raise ValueError("--verify-reward-parity requires --reward-type trl_math")
            from trl.rewards import accuracy_reward
            framework_scores = accuracy_reward([[{"role": "assistant", "content": text}] for text in rewards_text],
                                               solution=[row["answer"] for row in batch])
            if any(expected is None or actual != expected for actual, expected in zip(scores, framework_scores)):
                raise RuntimeError(f"DAPO/TRL reward disagreement: standalone={scores} TRL={framework_scores}")
        rewards = torch.tensor(
            scores,
            dtype=torch.float32,
            device=device,
        )
        advantages = (rewards[:, None] - baseline).detach()
    if was_training:
        model.train()
    return {
        "input_ids": generated,
        "attention_mask": full_mask,
        "completions": completions,
        "mask": mask,
        "ended_with_eos": ((completions == tokenizer.eos_token_id) & mask).any(dim=1),
        "old_log_probs": old_log_probs.detach(),
        "advantages": advantages,
        "baseline": baseline.detach(),
        "critic_disagreement": head_values.float().std(dim=-1, unbiased=False).detach(),
        "critic_features": [feature.cpu() for feature in features],
        "rewards": rewards,
        "rows": batch,
        "response_text": rewards_text,
        "prompt_width": prompt_width,
    }


def save_checkpoint(model, tokenizer, critic, output_dir: Path, step: int, *, save_full_adapter_state=False):
    destination = output_dir / f"checkpoint-{step}"
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    if save_full_adapter_state:
        # PEFT's save_pretrained only writes adapter weights. Preserve the full
        # policy state as well when the base was trainable; reload below restores it.
        torch.save(model.state_dict(), destination / "full_policy_state.pt")
    tokenizer.save_pretrained(destination)
    torch.save({"critic": critic.state_dict(), "layers": critic.layers,
                "aggregation": critic.aggregation,
                "head_size": critic.heads[0][0].out_features}, destination / "critic.pt")
    LOG.info("Saved checkpoint to %s", destination)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("This training script currently requires a CUDA GPU")
    if not args.model and not args.adapter:
        raise ValueError("Provide either --model or --adapter")
    if args.adapter and PeftConfig is None:
        raise RuntimeError("Install peft to load --adapter")
    if args.generation_temperature <= 0 or args.generation_top_p != 1.0:
        raise ValueError("PPO requires a positive temperature and --generation-top-p 1.0 (full support)")
    if args.ratio_tolerance <= 0:
        raise ValueError("--ratio-tolerance must be positive")
    if args.batch_size < 1 or args.micro_batch_size < 1 or args.critic_warmup_steps < 0 or args.kl_coefficient < 0:
        raise ValueError("Batch sizes must be positive; warm-up steps and KL coefficient must be nonnegative")
    if args.critic_architecture and args.critic_aggregation:
        raise ValueError("Use --critic-architecture or the legacy --critic-aggregation, not both")
    reward_versions = {}
    if args.reward_type == "trl_math":
        from experiment2.reward import require_validated_versions
        reward_versions = require_validated_versions()

    if args.algorithm == "grpo" and (args.group_size < 2 or args.batch_size % args.group_size):
        raise ValueError("GRPO batch size must be divisible by group size >= 2")
    if args.critic_warmup_steps != 0 or args.kl_coefficient != 0:
        raise ValueError("Viability runs use no warm-up and no reference KL")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_meta = output_dir / "resume_meta.json"
    resume_info = json.loads(resume_meta.read_text()) if args.resume_training and resume_meta.exists() else None
    if args.resume_training and resume_info is None:
        raise ValueError("Requested resume but resume metadata is missing")
    if resume_info is not None:
        expected = resume_invariants(args)
        if resume_info["invariants"] != expected: raise ValueError("Resume configuration mismatch")
        args.model = str(output_dir / f"checkpoint-{resume_info['step']}")
        if args.algorithm == 'ppo': args.critic_checkpoint = str(Path(args.model) / 'critic.pt')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # FP32 and disabled TF32 keep teacher-forced scores close to generation scores.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", torch.cuda.current_device())
    rows = load_rows(args.train_file, args.prompt_key, args.answer_key, args.max_examples)
    adapter_path = Path(args.adapter) if args.adapter else None
    model_id = args.model
    if adapter_path:
        model_id = PeftConfig.from_pretrained(adapter_path).base_model_name_or_path
    args.model_id = model_id
    tokenizer = load_policy_tokenizer(model_id, adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        full_state = adapter_path / "full_policy_state.pt"
        if full_state.is_file():
            model.load_state_dict(torch.load(full_state, map_location="cpu", weights_only=True), strict=True)
    model.to(device)
    disable_dropout(model)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if adapter_path and not args.train_full_model:
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        trainable = list(model.parameters())
    if not trainable:
        raise RuntimeError("The policy has no trainable parameters")
    reference_model = None
    if args.kl_coefficient > 0:
        LOG.info("Creating frozen reference policy for exact conditional KL; this requires additional GPU memory")
        reference_model = copy.deepcopy(model).eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)

    config = getattr(model.config, "text_config", model.config)
    hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", None))
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))
    if hidden_size is None or num_layers is None:
        raise ValueError("Could not infer hidden_size and number of layers from model config")
    architecture = args.critic_architecture or f"multilayer_{args.critic_aggregation or 'mean'}"
    args.critic_architecture = architecture
    layers, aggregation = viability_layers(args.algorithm, architecture, num_layers, args.critic_layers, args.all_layers)
    critic = None if args.algorithm == 'grpo' else MultiLayerCritic(hidden_size, layers, aggregation=aggregation).to(device=device, dtype=torch.float32)
    if args.critic_checkpoint and critic is not None:
        state = torch.load(args.critic_checkpoint, map_location="cpu", weights_only=True)
        if state.get("layers") != layers or state.get("aggregation") != aggregation:
            raise ValueError("Critic checkpoint architecture does not match the selected critic")
        critic.load_state_dict(state["critic"], strict=True)
    # A zero-signal GRPO batch must not move the policy through AdamW decay.
    policy_optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_learning_rate) if critic is not None else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"resolved_critic_layers": layers, "reward_versions": reward_versions}, f, indent=2)
    LOG.info("Loaded %d records; %s layers=%s; trainable policy params=%s",
             len(rows), model_id, layers, f"{sum(p.numel() for p in trainable):,}")
    LOG.info("Critic architecture=%s warmup_steps=%d KL coefficient=%g", architecture, args.critic_warmup_steps, args.kl_coefficient)

    # Head count affects initialization RNG consumption. Reset generation RNGs
    # so matched architecture runs start from the same sampling stream.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    start_step = 0
    token_count = 0
    if resume_info:
        state = torch.load(output_dir / resume_info['state_file'], map_location='cpu', weights_only=False)
        start_step = state['step']; token_count = state['tokens']
        policy_optimizer.load_state_dict(state['policy_optimizer'])
        if critic_optimizer is not None: critic_optimizer.load_state_dict(state['critic_optimizer'])
        random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
    total_iterations = min(args.max_steps, args.stop_after_step)
    segment_start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    last_saved = start_step
    for iteration in range(start_step + 1, total_iterations + 1):
        if token_count >= args.token_budget: break
        warmup = iteration <= args.critic_warmup_steps
        step = iteration - args.critic_warmup_steps
        count = args.batch_size if args.algorithm == 'ppo' else args.batch_size // args.group_size
        prompts = random.sample(rows, k=count)
        batch = prompts if args.algorithm == 'ppo' else [row for row in prompts for _ in range(args.group_size)]
        # Every baseline is computed with the same frozen critic before fitting on this batch.
        rollouts = [build_rollout(model, tokenizer, critic, batch[i:i + args.micro_batch_size], device, args, layers)
                    for i in range(0, len(batch), args.micro_batch_size)]
        if args.benchmark_synthetic_rewards:
            if not args.no_save:
                raise ValueError('Synthetic rewards are only allowed in no-save throughput benchmarks')
            offset = 0
            for rollout in rollouts:
                n = rollout['rewards'].shape[0]
                rollout['rewards'] = (torch.arange(offset, offset+n, device=device) % 2).float()
                if args.algorithm != 'grpo':
                    rollout['advantages'] = (rollout['rewards'][:, None] - rollout['baseline']).detach()
                offset += n
        zero_group_fraction = None
        if args.algorithm == 'grpo':
            scores = torch.cat([r['rewards'] for r in rollouts])
            groups = scores.reshape(-1, args.group_size)
            zero_group_fraction = float((groups.max(1).values == groups.min(1).values).float().mean())
            flat = group_advantages(scores, args.group_size); offset = 0
            for rollout in rollouts:
                n = rollout['mask'].shape[0]
                rollout['advantages'] = flat[offset:offset+n, None].expand_as(rollout['baseline']).detach()
                offset += n
        if args.log_rollouts:
            with (output_dir / "rollouts.jsonl").open("a", encoding="utf-8") as handle:
                for rollout in rollouts:
                    for row, text, score, ended in zip(rollout["rows"], rollout["response_text"], rollout["rewards"], rollout["ended_with_eos"]):
                        if args.reward_type == "trl_math":
                            from math_verify import parse
                            from experiment2.reward import PRED_CONFIG
                            extracted = [str(x) for x in parse(text, extraction_config=PRED_CONFIG, extraction_mode="first_match")]
                        else:
                            extracted = [normalize_answer(text)]
                        handle.write(json.dumps({"phase": "warmup" if warmup else "ppo", "iteration": iteration,
                            "prompt": row["prompt"], "gold": row["answer"], "response": text,
                            "extracted": extracted, "reward": score.item(), "ended_with_eos": bool(ended.item())}, ensure_ascii=False) + "\n")
        total_examples = len(batch)
        total_actions = sum(int(r["mask"].sum().item()) for r in rollouts)
        critic_value = 0.0
        # Fit only detached rollout-time features. PPO advantages remain fixed.
        for _ in range(args.critic_epochs if critic is not None else 0):
            critic_optimizer.zero_grad(set_to_none=True)
            critic_value = 0.0
            for rollout in rollouts:
                weight = rollout["mask"].shape[0] / total_examples
                features = [feature.to(device) for feature in rollout["critic_features"]]
                baseline, head_values, _ = critic(features)
                loss_v = critic_loss(baseline, head_values, rollout["rewards"], rollout["mask"], args.critic_aux_weight)
                (weight * loss_v).backward()
                critic_value += weight * loss_v.item()
            clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
            critic_optimizer.step()
        for rollout in rollouts:
            del rollout["critic_features"]
        if warmup:
            LOG.info("warmup=%d/%d critic_loss=%.5f reward=%.3f", iteration, args.critic_warmup_steps,
                     critic_value, sum(r["rewards"].sum().item() for r in rollouts) / total_examples)
            continue

        # Train mode enables Hugging Face gradient checkpointing; dropout has
        # been set to zero, and the initial-ratio guard detects discrepancies.
        model.train()
        has_actor_signal = args.algorithm != 'grpo' or any(bool((r['advantages'] != 0).any()) for r in rollouts)
        actor_value = kl_value = clipped_tokens = 0.0
        for epoch in range(args.ppo_epochs if has_actor_signal else 0):
            policy_optimizer.zero_grad(set_to_none=True)
            actor_value = kl_value = clipped_tokens = 0.0
            for rollout in rollouts:
                mask = rollout["mask"]
                completions = rollout["completions"]
                weight = mask.shape[0] / total_examples
                output = model(input_ids=rollout["input_ids"], attention_mask=rollout["attention_mask"],
                               position_ids=position_ids_from_mask(rollout["attention_mask"]), use_cache=False)
                start = rollout["prompt_width"] - 1
                logits = output.logits[:, start:start + completions.shape[1]]
                new_log_probs = sequence_log_probs(logits, completions, args.generation_temperature)
                ratio = torch.exp(new_log_probs - rollout["old_log_probs"])
                clipped_tokens += ((ratio[mask] - 1).abs() > args.clip_epsilon).sum().item()
                if epoch == 0:
                    max_error = (ratio[mask] - 1).abs().max().item()
                    LOG.info("step=%d initial_ratio_max_error=%.8f", step, max_error)
                    if not torch.isfinite(ratio[mask]).all() or max_error > args.ratio_tolerance:
                        raise RuntimeError(f"Generation and unchanged-policy scores disagree: max ratio error={max_error:.8g}")
                ppo_loss = actor_surrogate(ratio, rollout["advantages"], mask, args.clip_epsilon)
                loss = ppo_loss
                if reference_model is not None:
                    with torch.no_grad():
                        ref_output = reference_model(input_ids=rollout["input_ids"], attention_mask=rollout["attention_mask"],
                            position_ids=position_ids_from_mask(rollout["attention_mask"]), use_cache=False)
                        ref_logits = ref_output.logits[:, start:start + completions.shape[1]]
                    kl = conditional_reference_kl(logits, ref_logits, mask, args.generation_temperature)
                    loss = loss + args.kl_coefficient * kl
                    kl_value += weight * kl.item()
                (weight * loss).backward()
                actor_value += weight * ppo_loss.item()
            clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            policy_optimizer.step()
        reward_mean = sum(r["rewards"].sum().item() for r in rollouts) / total_examples
        baseline_mean = sum(r["baseline"][r["mask"]].sum().item() for r in rollouts) / total_actions
        token_count += total_actions
        disagreement = sum(r["critic_disagreement"][r["mask"]].sum().item() for r in rollouts) / total_actions
        ended = sum(r["ended_with_eos"].sum().item() for r in rollouts) / total_examples
        LOG.info("step=%d reward=%.3f actor_loss=%.5f critic_loss=%.5f reference_kl=%.5f baseline=%.4f disagreement=%.4f clip_fraction=%.4f ended_eos=%.3f actions=%d",
                 step, reward_mean, actor_value, critic_value, kl_value, baseline_mean, disagreement,
                 clipped_tokens / total_actions, ended, total_actions)
        metrics = {'step':step,'generated_tokens':token_count,'batch_generated_tokens':total_actions,
                   'reward':reward_mean,'clip_fraction':clipped_tokens/total_actions,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                   'segment_elapsed_seconds':time.monotonic()-segment_start,'algorithm':args.algorithm,
                   'zero_group_fraction':zero_group_fraction,'actor_update_skipped':not has_actor_signal}
        with (output_dir/'metrics.jsonl').open('a') as fp: fp.write(json.dumps(metrics)+'\n')
        if not args.no_save and (step % args.save_steps == 0 or step == total_iterations or token_count >= args.token_budget):
            dest=output_dir/f'checkpoint-{step}'; dest.mkdir(parents=True,exist_ok=True)
            model.save_pretrained(dest);tokenizer.save_pretrained(dest)
            if critic is not None:
                torch.save({'critic':critic.state_dict(),'layers':critic.layers,'aggregation':critic.aggregation,
                            'head_size':critic.heads[0][0].out_features},dest/'critic.pt')
            state_path=output_dir/f'training-state-{step}.pt'
            torch.save({'step':step,'tokens':token_count,'policy_optimizer':policy_optimizer.state_dict(),
                        'critic_optimizer':critic_optimizer.state_dict() if critic_optimizer is not None else None,
                        'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state_all()},state_path)
            invariants=resume_invariants(args)
            meta={'step':step,'tokens':token_count,'state_file':state_path.name,'invariants':invariants}
            tmp=output_dir/'resume_meta.tmp';tmp.write_text(json.dumps(meta));os.replace(tmp,resume_meta)
            for previous in output_dir.glob('training-state-*.pt'):
                if previous != state_path: previous.unlink()
            last_saved=step
            LOG.info('Saved resumable checkpoint step=%d tokens=%d',step,token_count)
    if not args.no_save and last_saved < min(args.max_steps, args.stop_after_step) and token_count < args.token_budget:
        raise RuntimeError('Stopped without a resumable checkpoint')


if __name__ == "__main__":
    main()
