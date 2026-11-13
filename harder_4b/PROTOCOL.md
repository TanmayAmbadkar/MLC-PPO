# Harder 4B baseline comparison — provisional preregistration

Recorded before the current 1.5B 10M-token results. The feasibility pilot is exploratory, and this document must be frozen or amended with a dated rationale before 4B outcome data are inspected.

## Purpose and controls

Test whether all-layer heads improve on final-layer heads and whether either offers a useful substitute for a separate full critic. Compare constant or leave-one-out batch-mean PPO (prompt-independent), final-layer PPO, middle-layer PPO, mean all-layer PPO, learned-weight all-layer PPO, full-critic PPO, and GRPO as an algorithmic reference. POISE is a related internal-state baseline and should also be included if a faithful implementation is feasible. Record the number of model layers from the actual checkpoint; do not assume the 1.5B model's 28 layers apply to 4B.

Start from the same pinned Qwen/Qwen3-4B-Base checkpoint, use the same filtered DeepScaleR problem pool and verifier, 8192 maximum completion tokens, LoRA actor fine-tuning, and three independent training seeds. Fix and document the plain prompt and decoding parameters. The prompt plus completion must remain within the unscaled native model context. For PPO use one rollout per prompt; GRPO uses four; compare at matched generated tokens and separately GPU-hours. Report actual token counts, prompt counts, learning curves, truncation, throughput and device peak memory. No coding domain in the main comparison.

Screen candidate problems independently of PPO rollouts, using at least eight seeded initial-policy completions and retaining 2–6 successes of 8 as an approximate 20–80% group. Preserve problem IDs, raw completions, sampling settings and scores; exclude the 32-problem feasibility pilot. All methods and seeds use the same screened pool, disjoint from evaluation benchmarks. No adaptive rescreening during the primary experiment; an explicitly logged dynamic curriculum is a separate ablation. Do not claim the measured screening fraction is the true success probability. Benchmark evaluations remain unfiltered.

## Pilot and feasibility gates

The exploratory PBS pilot samples 32 disjoint candidate problems with 4 completions each, an 8192-token limit, and saves full texts and scores. Measure length distribution, EOS/truncation, success frequencies and GPU throughput. This is not the 8-sample screening set and does not determine algorithm rankings. Before a full comparison, run one full-update smoke per method, test pre-token causal indexing, memory, resume and verifier parity, and size checkpoint storage for the existing jobs. A successful generation pilot alone does not establish feasibility of full-model PPO or a 4B full critic on the available 48 GiB GPUs.

## Metrics and decisions

Primary outcome is mean MATH-500 accuracy across the three seeds at 10M generated tokens, with paired seed differences and uncertainty across both seeds and evaluation problems; report individual runs. Supplement with AIME, learning curves, GPU-hours, peak memory, held-out explained variance by head and an appropriately labeled gradient-variance diagnostic. Prespecify a 2 percentage-point practical margin on MATH-500 (10 of 500 questions), subject to power and uncertainty; report paired confidence intervals rather than treating numerical rank as a conclusion.

- A shared head beats the trivial control only if its mean paired improvement is at least +2 points and a prespecified two-sided 95% paired interval excludes zero; otherwise call the result inconclusive or no observed advantage, not equivalence.
- The full critic pulls ahead if its mean paired advantage over all-layer is at least +2 points and the 95% interval excludes zero. Infer that it is better under this setup, not that its architecture is universally better.
- Claim all-layer matches the full critic only if a 95% interval for all-layer minus full critic lies entirely above -2 points; otherwise noninferiority is not established.
- Compare all-layer and final-head with the same paired effect and interval. Three seeds and a 500-problem benchmark may be underpowered for the margin; report this and avoid interpreting a nonsignificant difference as equality.

This is an initial feasibility and analysis plan; decide the precise hierarchical bootstrap and multiple-comparison treatment before any 4B outcome analysis, and preserve this file's version history.

## LoRA implementation requirement

Use the same LoRA configuration and frozen base checkpoint for each actor. The full-critic baseline retains a separately trainable full transformer critic; actor LoRA does not imply a LoRA critic. Record rank, alpha, target modules, dropout, trainable parameter counts and optimizer configuration before training. The current fast vLLM backend explicitly rejects adapters; implement and verify adapter-aware generation, PPO log-probabilities, save/resume and evaluation before launching any 4B training. The queued generation-only pilot does not need an adapter and remains valid.

## Scheduling constraint — November 12

The user permits two GPUs per job only for the separate fully trainable critic; all other methods and pilots must use one GPU per job. The four-GPU FSDP pilot was cancelled. Implement and test a two-GPU actor plus separate fully trainable critic with an appropriate memory strategy; if it is impractical, use a separately labeled LoRA critic as the fallback. Do not characterize the fallback as a fully fine-tuned critic. The 8K cap remains the target and may only be lowered if the pilot requires it.
