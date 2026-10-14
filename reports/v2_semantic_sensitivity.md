# LLMGP-v2 semantic sensitivity diagnosis

## Protocol and availability

Inference only. Each saved configured NormalizedGatedPrompt checkpoint and its associated frozen GCN are held fixed across correct, shuffled, random, ZeroPrompt and NoPrompt. No optimization, checkpoint writing or test-class evaluation was performed.

Available V2 seeds: [11]. Missing V2 seeds represented in the saved validation manifest: [23, 37, 53, 71]. The extended-validation runner trained seeds 23/37 in memory without saving their V2 weights; those reported models cannot be reused. V1 prompt weights are not substituted.

All available configured V2 shot settings are diagnosed, with 100 existing manifest episodes per setting. Each condition uses exactly the same support/query IDs, labels, order and classifier (cosine prototypes, temperature 0.1). Only validation IDs [3,4] are encoded and scored. The existing full Cora graph is retained for transductive message passing, including unlabeled nodes from other classes; test-class descriptions and test episodes are not used.

Correct uses the configured Neural Networks / Probabilistic Methods descriptions. Shuffled swaps these two validation descriptions while retaining support-label connectivity. Random replaces their embeddings with deterministic Gaussian directions matched to each original embedding norm. ZeroPrompt retains the same prompt nodes and edges with zero features; NoPrompt removes prompt nodes and edges. All semantic conditions use the same learned projection and gate, avoiding a confound from separately trained checkpoints.

The local semantic_alignment_audit.md verifies the configured class/topic mapping. Here 'correct' means those configured descriptions; this diagnosis assesses dependence on them, not their linguistic quality.

## Condition metrics

Accuracy and Macro-F1 are episode means. Effective norm is measured after normalization and gate multiplication. Gate is the shared sigmoid value, not the raw learned logit; bypassed conditions have no active gate.

| Seed | Shot | Condition | Accuracy | Macro-F1 | Effective prompt norm | Gate |
|---:|---:|---|---:|---:|---:|---:|
| 11 | 1 | correct | 0.614333 | 0.563476 | 1.019959688 | 0.243389875 |
| 11 | 1 | shuffled | 0.607000 | 0.546523 | 1.019959688 | 0.243389875 |
| 11 | 1 | random | 0.594667 | 0.518119 | 1.019959569 | 0.243389875 |
| 11 | 1 | ZeroPrompt | 0.614333 | 0.553858 | 0.000000000 | -- |
| 11 | 1 | NoPrompt | 0.606000 | 0.540619 | 0.000000000 | -- |
| 11 | 3 | correct | 0.712333 | 0.693021 | 0.511210799 | 0.121988691 |
| 11 | 3 | shuffled | 0.716000 | 0.696469 | 0.511210799 | 0.121988691 |
| 11 | 3 | random | 0.716000 | 0.698312 | 0.511210799 | 0.121988691 |
| 11 | 3 | ZeroPrompt | 0.720333 | 0.703237 | 0.000000000 | -- |
| 11 | 3 | NoPrompt | 0.717333 | 0.697877 | 0.000000000 | -- |
| 11 | 5 | correct | 0.768667 | 0.761806 | 0.556964993 | 0.132906869 |
| 11 | 5 | shuffled | 0.775000 | 0.768501 | 0.556964993 | 0.132906869 |
| 11 | 5 | random | 0.771000 | 0.764813 | 0.556964993 | 0.132906869 |
| 11 | 5 | ZeroPrompt | 0.769667 | 0.763411 | 0.000000000 | -- |
| 11 | 5 | NoPrompt | 0.780333 | 0.775080 | 0.000000000 | -- |
| 11 | 10 | correct | 0.828667 | 0.826267 | 0.559735775 | 0.133568048 |
| 11 | 10 | shuffled | 0.830667 | 0.828324 | 0.559735775 | 0.133568048 |
| 11 | 10 | random | 0.828667 | 0.826306 | 0.559735775 | 0.133568048 |
| 11 | 10 | ZeroPrompt | 0.829667 | 0.827341 | 0.000000000 | -- |
| 11 | 10 | NoPrompt | 0.829333 | 0.826991 | 0.000000000 | -- |

## Paired effects relative to correct

Cosine and L2 changes compare corresponding query vectors. Logit change is mean absolute change across the two class logits. Counts refer to query occurrences across episodes, not independent unique nodes. Full precision and maxima are saved in the accompanying JSON.

| Seed | Shot | Condition | Query cosine mean | Query L2 mean | Query L2 maximum | Logit absolute change | Predictions changed |
|---:|---:|---|---:|---:|---:|---:|---:|
| 11 | 1 | shuffled | 0.9999999029 | 0.000352327627 | 0.15513806 | 0.0176664149 | 784/3000 (26.1333%) |
| 11 | 1 | random | 0.9999993122 | 0.00050163895 | 0.226025641 | 0.0343031066 | 441/3000 (14.7000%) |
| 11 | 1 | ZeroPrompt | 0.9999997897 | 0.000600772721 | 0.274996758 | 0.0223899 | 394/3000 (13.1333%) |
| 11 | 1 | NoPrompt | 0.9999997669 | 0.000559951127 | 0.128350168 | 0.0242747033 | 421/3000 (14.0333%) |
| 11 | 3 | shuffled | 0.9999997175 | 0.000339204303 | 0.0680274591 | 0.00284568596 | 151/3000 (5.0333%) |
| 11 | 3 | random | 0.9999992006 | 0.00112405135 | 0.24947542 | 0.00494527467 | 207/3000 (6.9000%) |
| 11 | 3 | ZeroPrompt | 0.9999998018 | 0.000910287108 | 0.184714079 | 0.00338614496 | 92/3000 (3.0667%) |
| 11 | 3 | NoPrompt | 0.9999994626 | 0.00430977337 | 1.19068003 | 0.00670098527 | 163/3000 (5.4333%) |
| 11 | 5 | shuffled | 0.9999999590 | 0.000452952859 | 0.05721974 | 0.000792794863 | 47/3000 (1.5667%) |
| 11 | 5 | random | 0.9999997153 | 0.00110635458 | 0.219336778 | 0.00196085056 | 57/3000 (1.9000%) |
| 11 | 5 | ZeroPrompt | 0.9999995236 | 0.00109552645 | 0.15626806 | 0.00227730656 | 55/3000 (1.8333%) |
| 11 | 5 | NoPrompt | 0.9999990066 | 0.00728701264 | 0.848116338 | 0.00532063993 | 157/3000 (5.2333%) |
| 11 | 10 | shuffled | 0.9999999672 | 0.00106205545 | 0.061690025 | 0.000254164855 | 18/3000 (0.6000%) |
| 11 | 10 | random | 0.9999995683 | 0.00360170092 | 0.234052375 | 0.00118724632 | 28/3000 (0.9333%) |
| 11 | 10 | ZeroPrompt | 0.9999994196 | 0.00274331926 | 0.193319902 | 0.00133980926 | 17/3000 (0.5667%) |
| 11 | 10 | NoPrompt | 0.9999976295 | 0.0177335388 | 1.38780606 | 0.00370659208 | 104/3000 (3.4667%) |

## Prompt injection diagnostics

Effective prompt norm equals the training-node reference norm times the sigmoid gate. The normalized ungated norm equals the reference norm. Prompt-feature changes below establish whether changed descriptions produce distinct injected features before graph propagation.

| Seed | Shot | Reference norm | Gate logit | Correct query norm mean | Shuffled prompt L2 | Random prompt L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 1 | 4.190640926 | -1.134183526 | 5.264050961 | 0.651412785 | 1.64534223 |
| 11 | 3 | 4.190640926 | -1.973731160 | 5.205347538 | 0.460565835 | 0.87195766 |
| 11 | 5 | 4.190640926 | -1.875497699 | 5.188428879 | 0.167203397 | 0.467676133 |
| 11 | 10 | 4.190640926 | -1.869772553 | 5.184727192 | 0.12396957 | 0.536071181 |

## Interpretation

**Diagnosis: description-dependent information reaches a small subset of query representations and influences classification decisions, but these saved validation runs do not show a consistent benefit from correct semantic alignment.**

Correct/shuffled query cosine means range from 0.9999997175 to 0.9999999672. Only 0.37%-5.40% of query occurrences change their vectors when descriptions are swapped. Nevertheless, predictions change by 26.13%, 5.03%, 1.57% and 0.60% at 1, 3, 5 and 10 shots respectively. This is substantial decision sensitivity at 1 shot despite very sparse query-representation changes; support-prototype shifts can affect many queries. At 3 shots, mean query L2 change is 0.000339 versus a mean query norm of 5.20535, while mean absolute logit change is 0.002846. Correct descriptions outperform shuffled descriptions only at 1 shot, by 0.733 percentage points; shuffled descriptions have higher accuracy at 3, 5 and 10 shots. Sensitivity therefore establishes influence, not consistent usefulness.

- Seed 11, 1-shot: swapping descriptions changes 11/3000 query representations and 784 predicted classes; random directions change 11/3000 query representations and 441 predicted classes. Correct-minus-shuffled accuracy is +0.007333.
- Seed 11, 3-shot: swapping descriptions changes 41/3000 query representations and 151 predicted classes; random directions change 41/3000 query representations and 207 predicted classes. Correct-minus-shuffled accuracy is -0.003667.
- Seed 11, 5-shot: swapping descriptions changes 79/3000 query representations and 47 predicted classes; random directions change 79/3000 query representations and 57 predicted classes. Correct-minus-shuffled accuracy is -0.006333.
- Seed 11, 10-shot: swapping descriptions changes 162/3000 query representations and 18 predicted classes; random directions change 162/3000 query representations and 28 predicted classes. Correct-minus-shuffled accuracy is -0.002000.

Nonzero prompt-feature changes alone demonstrate semantic input reaching the prompt projection. Nonzero paired query changes demonstrate propagation into query representations. Logit changes can also arise through support-prototype changes even when a query vector is unchanged. Class flips indicate decision sensitivity; they do not establish a beneficial semantic contribution. ZeroPrompt versus NoPrompt also includes topology and GCN degree-normalization effects, so their differences cannot be attributed solely to description semantics.

With a two-layer GCN and prompt edges only to support nodes, feature-dependent prompt influence reaches original support nodes in one layer and their neighbors in two layers. Queries outside that neighborhood can retain exactly the same representation while their prototype-based logits change.

This is a descriptive diagnosis of saved validation models, with only one available independent V2 seed. Repeated query occurrences and shot settings do not supply independent seed replications. No across-seed significance or test generalization claim is supported.

## Reproducibility and integrity

Runtime: 107.60 seconds on cpu. SHA-256 checks passed for 52 existing checkpoint/manuscript files and the existing V1 artifact snapshot.

Validation manifest SHA-256: `510abeb6ed49032e447ec9d4055944807b6ca1ddc94b6dd5dcfdbfc5085c8050`.

| Seed | Shot | Existing V2 checkpoint | SHA-256 |
|---:|---:|---|---|
| 11 | 1 | `results\v2\checkpoints\normalizedgatedprompt_configured_shot_1_seed_11.pt` | `96dc4ed18aede3ba44acb3fe1a6e3f50e31e48137d5a7cbb4ee7531a678d629e` |
| 11 | 3 | `results\v2\checkpoints\normalizedgatedprompt_configured_shot_3_seed_11.pt` | `263526bcb4f33224c7f3f0b68a3fa7ebfc5fa6f5226691455a4245c815128412` |
| 11 | 5 | `results\v2\checkpoints\normalizedgatedprompt_configured_shot_5_seed_11.pt` | `60497c62d57bf6683bdb608318ec82ef957f58afdefa72e601796c9e1c81dfb5` |
| 11 | 10 | `results\v2\checkpoints\normalizedgatedprompt_configured_shot_10_seed_11.pt` | `2bf6819391099e18571ae8c906d880422e1052ba0626d11259eb07fbe4fdb6a5` |

Command from the implementation directory:

```powershell
& .venv/Scripts/python.exe LLMGP/experiments/run_v2_semantic_sensitivity.py
```

Machine-readable metrics: `reports/v2_semantic_sensitivity.json`. No retraining or test evaluation was run.
