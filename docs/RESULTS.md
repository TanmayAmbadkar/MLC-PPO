# Results and experimental status

Snapshot: November 12, 2026. All benchmark accuracies below use greedy decoding and the recorded Math-Verify-based reward implementation. Each method has one training seed; AIME sets have only 30 problems each.

## Qwen2.5-Math-1.5B Base, ~10M generated training tokens

Training used a pinned DeepScaleR subset, full policy fine-tuning, batch size 64, a 2,048-token completion cap, and the same verifier. The critic and policy implementations varied by method. Trivial PPO uses each response's leave-one-out mean batch reward; it does not condition on prompt difficulty.

| Method | MATH-500 | AIME 2024 | AIME 2025 | AMC 2023 | Minerva subset |
|---|---:|---:|---:|---:|---:|
| Initial model | 216/500 (43.2%) | 2/30 | 0/30 | 14/40 | 57/265 |
| Trivial baseline | 345/500 (69.0%) | 4/30 | 0/30 | 20/40 | 67/265 |
| Final-head PPO | 351/500 (70.2%) | 5/30 | 1/30 | 19/40 | 66/265 |
| GRPO | 347/500 (69.4%) | 2/30 | 2/30 | 18/40 | 70/265 |
| All-layer MLC-PPO | 354/500 (70.8%) | 3/30 | 0/30 | 19/40 | 68/265 |
| Separate full-critic PPO | 349/500 (69.8%) | 4/30 | 1/30 | 19/40 | 68/265 |

MLC-PPO versus the trivial baseline: +9/500, or +1.8 percentage points. The paired MATH-500 bootstrap interval, resampling benchmark problems, is approximately [−0.8, +4.4] points. It does not quantify variation between independently trained seeds.

## Qwen2.5-Math-1.5B-Instruct, ~40M generated training tokens

| Method | MATH-500 | AIME 2024 |
|---|---:|---:|
| Initial model | 377/500 (75.4%) | 3/30 |
| Final-head PPO | 378/500 (75.6%) | 6/30 |
| GRPO | 387/500 (77.4%) | 4/30 |
| All-layer MLC-PPO | 387/500 (77.4%) | 5/30 |
| Separate full-critic PPO | 381/500 (76.2%) | 4/30 |

The all-layer run achieved 395/500 at an intermediate checkpoint but 387/500 at the final checkpoint. Use the final checkpoint for the primary comparison; intermediate checkpoint selection changes the interpretation.

## Resource measurements

At the first ~1M-token base-model checkpoint, the all-layer method used one GPU, 43.3 GiB peak PyTorch allocated memory, and 31.3 minutes of training; full-critic PPO used two GPUs, 70.5 GiB combined peak allocated memory, and 36.9 minutes. GRPO used one GPU, 42.3 GiB, and 34.9 minutes. These are single-run measures, include startup, and exclude separate vLLM process peaks. They do not establish a repeatable speed advantage over GRPO.

## 4B feasibility and POLARIS

Qwen3-4B-Base with rank-32 LoRA completed synthetic 8,192-token update memory tests: the actor used one GPU and the separate fully trainable critic used a second GPU with FP32 CPU master weights and Adam states. The two-GPU real-reward full-critic update and checkpoint-resume smoke tests also passed. Synthetic stress tests repeat sampled tokens and are not learning experiments.

The POLARIS reward pilot scored 72/400 responses correct, with mixed outcomes for 42/100 prompts. The untrained Qwen3-4B Base model scored 319/500 (63.8%) on MATH-500 using the project evaluation prompt and an 8,192-token cap. Dataset screening and the 4B comparison are not completed results in this snapshot.

## Interpretation

Results do not establish that multiple heads are necessary or that MLC-PPO outperforms the trivial baseline across seeds. The base-model MATH-500 improvement may partly arise from answer-format learning. Near-duplicate filtering cannot exclude all benchmark leakage, and outcome verifier consistency is not proof that every gold answer is correct.
