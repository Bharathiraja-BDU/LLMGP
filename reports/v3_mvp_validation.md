# LLMGP-v3 minimal validation: Semantic Prototype Fusion

## Protocol

3-shot, 2-way episodes; seeds 11, 23 and 37; 100 training episodes per trainable condition per seed and 100 shared validation episodes per seed, with 15 queries per class. Train classes [0,1,2], validation [3,4], test [5,6]. Test classes were not evaluated or used to encode descriptions or select models.

The existing self-supervised seed-specific GCN and frozen SciBERT are reused. Original graph embeddings are computed once per seed without virtual prompt nodes. The full graph is retained for transductive message passing; nodes from held-out classes remain unlabeled inputs to the reused backbone. Support class means and projected semantic vectors are individually L2-normalized, then fused as `normalize((1-lambda)*support + lambda*semantic)`. A shared sigmoid gate initializes lambda to 0.1. Only the 768-to-256 linear projection (with bias) and gate are optimized.

All three trainable conditions start from an exact clone of the same per-seed initialization, receive the same ordered training episodes, and take 100 Adam updates (learning rate 0.001, weight decay 0.0001). Final weights after update 100 are evaluated once; no validation selection, early stopping, hyperparameter search or retraining is used. Models remain in memory; existing checkpoints are never written.

Shuffled descriptions are deranged independently within the training and validation splits, including swapping validation classes 3/4. Random vectors are fixed per seed and class, norm-matched to the corresponding frozen SciBERT input embeddings before projection. Neither control introduces test descriptions. The saved V2 validation manifest is loaded read-only and shared by every arm. Training episodes reuse the existing episodic sampler and extended-validation seed rule.

## Seed-wise validation results

Accuracy and Macro-F1 are means of episode metrics. Paired accuracy differences use the same 100 episodes; differences are fractions, so 0.01 equals one percentage point.

| Seed | Model | Accuracy | Macro-F1 | Learned lambda | Delta vs NoPrompt | Delta vs shuffled | Delta vs random |
|---:|---|---:|---:|---:|---:|---:|---:|
| 11 | NoPrompt | 0.717333 | 0.697877 | -- | +0.000000 | +0.134667 | +0.029667 |
| 11 | V3-correct | 0.656333 | 0.631865 | 0.109785 | -0.061000 | +0.073667 | -0.031333 |
| 11 | V3-shuffled | 0.582667 | 0.538974 | 0.109791 | -0.134667 | +0.000000 | -0.105000 |
| 11 | V3-random | 0.687667 | 0.664703 | 0.109354 | -0.029667 | +0.105000 | +0.000000 |
| 23 | NoPrompt | 0.677333 | 0.650864 | -- | +0.000000 | +0.112667 | +0.001333 |
| 23 | V3-correct | 0.625000 | 0.592025 | 0.109579 | -0.052333 | +0.060333 | -0.051000 |
| 23 | V3-shuffled | 0.564667 | 0.520951 | 0.109636 | -0.112667 | +0.000000 | -0.111333 |
| 23 | V3-random | 0.676000 | 0.652975 | 0.109257 | -0.001333 | +0.111333 | +0.000000 |
| 37 | NoPrompt | 0.720667 | 0.702303 | -- | +0.000000 | +0.006667 | +0.049333 |
| 37 | V3-correct | 0.657000 | 0.623777 | 0.109892 | -0.063667 | -0.057000 | -0.014333 |
| 37 | V3-shuffled | 0.714000 | 0.696911 | 0.109767 | -0.006667 | +0.000000 | +0.042667 |
| 37 | V3-random | 0.671333 | 0.651111 | 0.109756 | -0.049333 | -0.042667 | +0.000000 |

## Across-seed summaries

Sample SD uses ddof=1 across the three independent seeds, not across episodes.

| Model | Accuracy mean +/- SD | Macro-F1 mean +/- SD | Lambda mean +/- SD |
|---|---:|---:|---:|
| NoPrompt | 0.705111 +/- 0.024114 | 0.683681 +/- 0.028507 | -- |
| V3-correct | 0.646111 +/- 0.018286 | 0.615889 +/- 0.021059 | 0.109752 +/- 0.000159 |
| V3-shuffled | 0.620444 +/- 0.081520 | 0.585612 +/- 0.096808 | 0.109731 +/- 0.000083 |
| V3-random | 0.678333 +/- 0.008413 | 0.656263 +/- 0.007368 | 0.109456 +/- 0.000264 |

### Paired correct-description accuracy differences

| Comparator | Seed 11 | Seed 23 | Seed 37 | Mean +/- SD |
|---|---:|---:|---:|---:|
| NoPrompt | -0.061000 | -0.052333 | -0.063667 | -0.059000 +/- 0.005925 |
| V3-shuffled | +0.073667 | +0.060333 | -0.057000 | +0.025667 +/- 0.071901 |
| V3-random | -0.031333 | -0.051000 | -0.014333 | -0.032222 +/- 0.018349 |

## Outcome

- Correct descriptions versus NoPrompt: mean accuracy difference -0.059000, mean Macro-F1 difference -0.067792; both metrics improve in 0/3 seeds. Consistent superiority on both metrics: False.
  Correct semantic fusion underperforms NoPrompt on both metrics in every seed.
- Correct descriptions versus V3-shuffled: mean accuracy difference +0.025667, mean Macro-F1 difference +0.030277; both metrics improve in 2/3 seeds. Consistent superiority on both metrics: False.
- Correct descriptions versus V3-random: mean accuracy difference -0.032222, mean Macro-F1 difference -0.040374; both metrics improve in 0/3 seeds. Consistent superiority on both metrics: False.
  Correct semantic fusion underperforms V3-random on both metrics in every seed.

This minimal validation does not establish consistent improvement from correctly aligned semantic fusion over NoPrompt and both semantic controls. Any positive mean differences above are descriptive; with three seeds, no statistical or test-generalization claim is made.

## Verification

- Every projection weight, projection bias and gate gradient was finite and nonzero on every one of the 900 V3 training updates. No update was skipped.
- GCN/SciBERT remain in evaluation mode with requires_grad=False and no accumulated gradients. The optimizer contains only the V3 parameters; detached cached encoder outputs prevent encoder backpropagation. GCN state hashes were also checked after each arm.
- All training/validation episodes passed class-membership, actual-node-label, balanced-count, unique-index and support/query-disjointness checks.
- Correct descriptions agree with the existing semantic alignment audit's ID/topic/count rows and configured description topics; local Cora class counts match that audit.
- All 198 protected existing V1/V2 artifacts and manuscript hashes are unchanged. Full hashes and per-episode validation metrics are saved in `v3_mvp_validation_details.json`.

### Minimum observed gradient norms

| Seed | Condition | Projection weight | Projection bias | Gate logit |
|---:|---|---:|---:|---:|
| 11 | V3-correct | 0.00948054716 | 6.93138281e-05 | 0.000275940343 |
| 11 | V3-shuffled | 0.0105839903 | 8.25278548e-05 | 0.000153257643 |
| 11 | V3-random | 0.00961630605 | 0.000201080766 | 0.00188356463 |
| 23 | V3-correct | 0.00984362699 | 0.000102186248 | 0.000442551333 |
| 23 | V3-shuffled | 0.010664247 | 0.000109887413 | 0.000878861058 |
| 23 | V3-random | 0.00947688613 | 0.000226705321 | 0.00130857469 |
| 37 | V3-correct | 0.0134824775 | 0.000145815138 | 0.000596891157 |
| 37 | V3-shuffled | 0.0111530852 | 0.000147772895 | 0.000628904789 |
| 37 | V3-random | 0.00922586024 | 0.000214833562 | 3.73511975e-05 |

### Shared validation episode identifiers

| Seed | Validation episode SHA-256 | Initial V3 state SHA-256 |
|---:|---|---|
| 11 | `ba0a10e9d5c8c16c0631a75b8b5409f3a1b03fdcb1c5e6894a39833d63545038` | `dcd721794bdcb473c33be629e466a954e2ef820beb150f3e515e07dbb8d39d4e` |
| 23 | `909f0ef5cbd010d3b3320af75079609da6bc8f77951cc3c3a0b0f7059f5fc0e5` | `82eaeeada6691743d2ab949ab0da42472163996ea6fa238a8132a70de0d63ab2` |
| 37 | `dffb1a0927a00aebe92f500481b57edb99f9f4f328b9db575b272e3df4161bdc` | `5caa669dc760a628086018ce4a2b3f0bf607df6b09b6afe17efff24662718e86` |

## Runtime, tests and stop point

Runtime: 11.79 seconds on cpu. Test-class results: **not evaluated**, as requested. No full experiments, extra seeds or shot settings were run.

Unit-test results: 3 passed (tests/v3, pytest; 7.48 seconds). Validation runtime excludes the separate unit-test run.

Targeted unit tests check the fusion formula, normalization, support rescaling invariance, gate endpoints and trainable/frozen gradient behavior. Run with:

```powershell
Set-Location LLMGP
& ../.venv/Scripts/python.exe -m pytest tests/v3 -q
```

Validation command from the implementation directory:

```powershell
& .venv/Scripts/python.exe LLMGP/experiments/run_v3_mvp_validation.py
```
