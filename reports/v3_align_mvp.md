# V3-Align minimal validation

## Protocol and alignment objective

The unchanged SemanticPrototypeFusion architecture, frozen self-supervised seed-specific GCN and frozen SciBERT are reused. 3-shot, 2-way episodes, seeds 11/23/37, 100 training updates and 100 identical validation episodes per seed, with 15 queries per class. Training classes: [0,1,2]; validation: [3,4]; test: [5,6], not evaluated or encoded.

Training uses symmetric contrastive cross entropy over the normalized projected semantic vectors and normalized support means. Cosine similarities divided by alignment temperature form a class-by-class matrix. Its diagonal contains positive pairs; off-diagonal entries contain negatives. The alignment loss averages row-wise semantic-to-support CE and transposed support-to-semantic CE. Each two-way episode supplies one negative per class; the three training classes are covered through the existing sampler. Support targets are detached. No validation prototypes or descriptions enter either training loss.

Total loss = classification CE + **1.0** * alignment loss; alignment temperature **0.1**. These are configurable command-line values; one setting was run, with no sweep or validation tuning. Alignment updates the projection; classification updates both the projection and fusion gate. Lambda initializes at 0.1. Adam learning rate 0.001, weight decay 0.0001, classification temperature 0.1 and final-update evaluation match the original MVP.

V3-Align starts from the exact original per-seed initial weights and uses the exact original ordered training episodes. Original V3-correct is loaded from its existing trained checkpoint without retraining; NoPrompt uses support-only prototypes. All comparisons use the same read-only validation manifest. Original accuracy, Macro-F1 and margins were checked against the archived V3 diagnosis. There is no separate alignment warm-up or extra training budget.

The original full graph is retained for transductive frozen message passing, including unlabeled nodes from other classes. No virtual nodes, architecture changes, test-class evaluation or V4 implementation were introduced.

## Validation results by seed

Accuracy/F1 are episode means. Semantic/support cosine compares matching class rows; alignment advantage is matching minus wrong-class cosine. Margins are correct-class logits minus highest incorrect-class logits, at temperature 0.1. NoPrompt has no semantic prototype or active gate; its alignment fields are not applicable.

| Seed | Model | Accuracy | Macro-F1 | Semantic/support cosine | Matching minus wrong | Margin before | Margin after | Lambda |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | NoPrompt | 0.717333 | 0.697877 | -- | -- | +0.020659395 | +0.020659395 | -- |
| 11 | V3-correct | 0.656333 | 0.631865 | -0.403238940 | -0.006056710 | +0.020659395 | +0.015117205 | 0.109784558 |
| 11 | V3-Align | 0.671000 | 0.647163 | 0.124723632 | -0.003970532 | +0.020659395 | +0.016155824 | 0.110290200 |
| 23 | NoPrompt | 0.677333 | 0.650864 | -- | -- | +0.016252098 | +0.016252098 | -- |
| 23 | V3-correct | 0.625000 | 0.592025 | -0.178697970 | -0.004038370 | +0.016252098 | +0.011898144 | 0.109579250 |
| 23 | V3-Align | 0.576333 | 0.532674 | 0.295997968 | -0.005476276 | +0.016252098 | +0.008928275 | 0.111139208 |
| 37 | NoPrompt | 0.720667 | 0.702303 | -- | -- | +0.020222665 | +0.020222665 | -- |
| 37 | V3-correct | 0.657000 | 0.623777 | -0.329712077 | -0.003128211 | +0.020222665 | +0.017175123 | 0.109892465 |
| 37 | V3-Align | 0.642667 | 0.611582 | 0.089490673 | -0.005174664 | +0.020222665 | +0.013813738 | 0.110203944 |

## Across-seed mean +/- sample SD

SD uses ddof=1 across three seeds, not across repeated query occurrences.

| Model | Accuracy | Macro-F1 | Semantic/support cosine | Matching minus wrong | Margin after | Lambda |
|---|---:|---:|---:|---:|---:|---:|
| NoPrompt | 0.705111 +/- 0.024114 | 0.683681 +/- 0.028507 | -- | -- | 0.019045 +/- 0.002428 | -- |
| V3-correct | 0.646111 +/- 0.018286 | 0.615889 +/- 0.021059 | -0.303883 +/- 0.114477 | -0.004408 +/- 0.001499 | 0.014730 +/- 0.002660 | 0.109752 +/- 0.000159 |
| V3-Align | 0.630000 +/- 0.048588 | 0.597140 +/- 0.058595 | 0.170071 +/- 0.110470 | -0.004874 +/- 0.000797 | 0.012966 +/- 0.003688 | 0.110544 +/- 0.000517 |

## Paired differences and outcome

All differences use paired model metrics on the same episodes. Positive values favor V3-Align.

| Comparator | Seed 11 accuracy delta | Seed 23 accuracy delta | Seed 37 accuracy delta | Mean accuracy delta | Mean Macro-F1 delta |
|---|---:|---:|---:|---:|---:|
| NoPrompt | -0.046333 | -0.101000 | -0.078000 | -0.075111 | -0.086542 |
| V3-correct | +0.014667 | -0.048667 | -0.014333 | -0.016111 | -0.018749 |

- V3-Align versus NoPrompt: both accuracy and Macro-F1 improve in 0/3 seeds.
- V3-Align versus V3-correct: both accuracy and Macro-F1 improve in 1/3 seeds.

**V3-Align fails to outperform NoPrompt on both across-seed mean metrics in this minimal validation.** Alignment changes do not establish classification improvement. The experiment stops here without additional tuning, training or architecture changes.
V3-Align underperforms NoPrompt on both accuracy and Macro-F1 in every seed.

## Alignment and margin interpretation

- Seed 11: matching cosine changes from -0.403238940 to +0.124723632; matching-minus-wrong changes from -0.006056710 to -0.003970532. Relative to NoPrompt, the mean margin changes by -0.004503571; fusion loses 360 correct decisions and gains 221. Final lambda is 0.110290200.
- Seed 23: matching cosine changes from -0.178697970 to +0.295997968; matching-minus-wrong changes from -0.004038370 to -0.005476276. Relative to NoPrompt, the mean margin changes by -0.007323822; fusion loses 547 correct decisions and gains 244. Final lambda is 0.111139208.
- Seed 37: matching cosine changes from -0.329712077 to +0.089490673; matching-minus-wrong changes from -0.003128211 to -0.005174664. Relative to NoPrompt, the mean margin changes by -0.006408928; fusion loses 422 correct decisions and gains 188. Final lambda is 0.110203944.

The contrastive objective did not establish class-discriminative alignment on the unseen validation classes: matching-minus-wrong remains negative in every seed. Greater absolute matching cosine therefore does not translate into correct class separation. The gate also remains above its 0.1 initialization, preserving the harmful fused contribution in these validation runs.

Absolute matching cosine and class-discriminative alignment are distinct: moving toward a common GCN direction can raise matching cosine without improving the matching-minus-wrong gap. Training-only contrastive alignment must transfer to the unseen validation classes to benefit their classification. These observations are limited to three seeds; they do not establish a general causal or test-set result.

## Verification, checkpoints and preservation

- All 300 joint training updates had finite, nonzero projection-weight, projection-bias and fusion-gate gradients. No skipped updates.
- Every training loss explicitly checks that class IDs belong to [0,1,2]. Every episode passed support/query disjointness, balanced counts, actual-node-label and class-membership checks.
- GCN and SciBERT remain frozen in evaluation mode with no gradients. GCN state hashes match before/after every seed; SciBERT state matches before/after the experiment. Cached frozen graph outputs and detached support targets prevent encoder backpropagation.
- New checkpoints passed strict state-dict reload checks before evaluation. Initialization and training/validation episode hashes match the original V3-correct checkpoints.
- All 220 existing V1/V2/V3 artifacts and manuscript hashes remain unchanged. New checkpoints and results are stored only under `results/v3_align/`; this report is separate from existing reports.

| Seed | New V3-Align checkpoint | SHA-256 | Validation episode hash |
|---:|---|---|---|
| 11 | `results/v3_align/checkpoints/semantic_prototype_fusion_align_shot_3_seed_11.pt` | `b45a3db185c5e56497f14a39232c9ef6e174f3387bb2b8c64447e70607c09aa0` | `ba0a10e9d5c8c16c0631a75b8b5409f3a1b03fdcb1c5e6894a39833d63545038` |
| 23 | `results/v3_align/checkpoints/semantic_prototype_fusion_align_shot_3_seed_23.pt` | `3111d6446ac8b5bf04e91f4c7db5a4b028620748e0d1ae106c790005282781a9` | `909f0ef5cbd010d3b3320af75079609da6bc8f77951cc3c3a0b0f7059f5fc0e5` |
| 37 | `results/v3_align/checkpoints/semantic_prototype_fusion_align_shot_3_seed_37.pt` | `1f4f8969b511243217ff535c6bfc7eec46ec65b3eb8ee536037bd26c7b159b66` | `dffb1a0927a00aebe92f500481b57edb99f9f4f328b9db575b272e3df4161bdc` |

Numeric results, per-episode accuracy/F1, gradient summaries, loss/gate histories, checkpoint provenance and protected hashes: `results/v3_align/mvp_results.json`. Scalar seed/model results: `results/v3_align/mvp_results.csv`.

Runtime: 11.08 seconds on cpu (unit tests excluded). Test-class results: **not evaluated**. No full experiments or weight sweep were run.

Unit tests: 3 passed, 1 library deprecation warning; 7.25 seconds.

Command from the implementation directory:

```powershell
& .venv/Scripts/python.exe LLMGP/experiments/run_v3_align_mvp.py --alignment-weight 1.0 --alignment-temperature 0.1
```

Existing output paths cause an error before training; they are never silently overwritten.
