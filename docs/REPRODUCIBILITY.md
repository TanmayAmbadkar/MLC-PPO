# Reproducibility and provenance

## Models and data

- 1.5B base: `Qwen/Qwen2.5-Math-1.5B`; Instruct: `Qwen/Qwen2.5-Math-1.5B-Instruct`.
- 1.5B base training source: `agentica-org/DeepScaleR-Preview-Dataset`, revision `b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29`. Prepared subsets were filtered against several benchmark sets. Filter details are in `deepscaler/prepare.py` and `base_experiment/prepare.py`.
- 4B feasibility: `Qwen/Qwen3-4B-Base`, revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Planned POLARIS source: `POLARIS-Project/Polaris-Dataset-53K`, revision `296f8e34132e63f4a1d70e0dcc8bddebb43f03e4`. `polaris_experiment/prepare.py` documents its splits and filtering. Similarity filtering cannot detect all semantic overlap.
- Pinned verifier versions: Math-Verify 0.9.0, latex2sympy2-extended 1.11.0, antlr4-python3-runtime 4.13.2. Evaluation excludes gold answers that the verifier cannot parse.

## Artifacts

`results/benchmarks.csv` contains all exported checkpoint-level scores; `results/training_metrics.csv` contains per-update aggregate metrics. `results/raw/*_math500.csv` contains item index and binary outcome only for the base-model final checkpoints. Full completions, prompt datasets, checkpoint weights, optimizer state, and environment-specific PBS output are intentionally omitted. Original per-problem files align on item index; compare only the same benchmark and evaluation protocol.

Regenerate the paired confidence interval with `python results/paired_bootstrap.py`. This assesses benchmark item uncertainty conditional on fixed trained policies, not training-seed uncertainty.

## Training and inference mismatch

`old_log_probs` are recomputed from a full-sequence forward pass to align the unchanged-policy PPO ratio; separate rollout log probabilities are used to report the behavior/training discrepancy. This does not make the rollout and training distributions identical and is not an importance correction. Do not interpret the initial ratio check as proving exact on-policy sampling.

## GURU experimental components

`guru_experiment/reference/` contains selected reward implementations copied from LLM360/Reasoning360 commit `13158341d2a0dfe5f3bb80e7126ff21de0d16676`, distributed under the included Apache-2.0 license. It retains the upstream executor named `unsafe_local`; this code must only run within the Bubblewrap isolation provided by `guru_experiment/sandbox_verifier.py`. The sandbox implementation was tested with simple examples and is not a formally verified security boundary. It exposes a read-only training Python environment and a temporary filesystem. Validate isolation and resource limits on the target host before running untrusted code.

`python -m pip install --target guru_experiment/vendor pylatexenc==2.10` restores the non-committed dependency expected by that prototype. The science domain additionally requires an independently hosted local verifier model. Two 24-problem GURU signal pilots scored zero across six domains for the evaluated 1.5B initializations; these small pilots do not rule out the dataset or constitute a training result. Large LiveCodeBench evaluation files were not included in the exact prompt-overlap check.

## Cluster-specific details

The original jobs used PBS and RTX A6000 GPUs. Scripts include absolute cluster paths and a separate vLLM Python executable; adjust these before use. CPU-offloaded 4B full-critic smoke checkpoints use large temporary optimizer states. The jobs and output directories included in the scripts are not provisioned by cloning this repository.
