# Reward validation triage

Run `qsub pbs_dataset_validation.sh` from the repository root. The job selects 300 questions without using model predictions and excludes questions previously used in Experiment 1. Its output is under `runs/experiment1c/validation`.

`vetted_problems.jsonl` is a review queue: every candidate starts with `accepted: false` and `review_status: pending_independent_review`. Automated arithmetic flags are triage only. They do not establish question clarity, solution validity, or correctness of the gold answer. `blind_review.jsonl` contains questions without gold solutions or model predictions so reviewers can first derive answers independently. Decisions, derivations, reviewers, and any gold corrections still need to be recorded before these candidates can be used as a vetted dataset.

`excluded_problems.jsonl` documents the previously inspected audit problems and why they are outside the primary sample. `corrections.json` records two assistant-reviewed gold discrepancies; these do not alter source data and still need independent review. `verifier_tests.json` records format regression checks and explicitly marks multi-answer grading as unsupported.

No critic training or continuation generation should use these candidates as vetted until independent review is complete. Rerunning the validation job recreates the triage files; preserve review work separately.
