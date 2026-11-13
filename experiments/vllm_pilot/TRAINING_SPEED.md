# Matched MLC-PPO training speed tests

These tests replay exactly the same 64 sampled vLLM rollouts (49,988 valid generated tokens) through one 28-head MLC-PPO update on an RTX A6000. All have actor microbatch 4, two PPO epochs, one critic epoch, real binary rewards, the same base weights and data, unchanged-policy full-forward PPO ratio checks, and no checkpoint saving. The rollout behavior is BF16 vLLM; the full-forward PPO baseline is recomputed and rollout/training mismatch is not corrected. These numbers isolate the trainer's update loop (including rollout processing) and exclude vLLM generation and model startup.

| Variant | Trainer time | Peak PyTorch allocation | PBS job |
| --- | ---: | ---: | ---: |
| FP32, original order, checkpointing | 141.29 s | 40.61 GB | 61185 |
| FP32, sorted by response length, checkpointing | 109.48 s | 43.27 GB | 61188 |
| BF16 autocast, FP32 weights and optimizer, checkpointing | 53.05 s | 39.20 GB | 61186 |
| BF16 autocast plus length sorting, checkpointing | 40.99 s | 41.57 GB | 61189 |
| FP32, original order, no checkpointing | OOM | >45.65 GiB allocated | 61187 |

Length sorting reduced response padding from 70,552 to 52,244 token positions in this batch. Sorting was possible because this is PPO with one response per prompt and a single optimizer step per epoch; gradient accumulation reordering can still change floating-point results. GRPO group-relative rewards must be computed before any sorting, preserving group membership.

The AMP paths recompute old-policy probabilities through the same full-sequence path and keep trainable weights in FP32. Critic binary cross-entropy is calculated in FP32. Initial ratio checks passed for each successful run; the same total reward and token count were observed. The full policy update still uses an uncorrected vLLM behavior distribution and BF16 computations alter numerical behavior. One update is insufficient to validate long-run optimizer stability, save/resume, memory safety for other batches, weight synchronization, or statistically compare RL outcomes. Existing production jobs were unchanged.
