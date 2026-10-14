# LLMGP-v3 alignment diagnosis with saved checkpoints

## Protocol and persistence

The existing V3 SemanticPrototypeFusion architecture and MVP runner were reused without architecture changes. Three seeds (11/23/37), 3-shot, 2-way episodes, 100 training episodes for each semantic arm and 100 identical validation episodes per seed, with 15 queries per class. Training uses only [0,1,2], evaluation only [3,4]. Test classes [5,6] were not evaluated or encoded.

Every arm starts from an exact clone of the same per-seed initialization. Adam, learning rate 0.001, weight decay 0.0001, initial lambda 0.1, temperature 0.1, and final-update evaluation match the prior MVP. No validation model selection, extra training, extra seeds or architecture changes were performed. Shuffled descriptions are deranged within each split; validation descriptions are swapped. Random frozen semantic vectors retain each corresponding SciBERT input norm.

Nine distinct trained checkpoints are saved under `results/v3/checkpoints/`, opened exclusively to prevent overwrites. Each contains the state_dict, seed, shot, semantic arm, full training configuration and statistics, gate history, frozen GCN path/hash, SciBERT state identifier, actual semantic input vectors for classes 0-4, initialization/episode hashes and semantic-assignment provenance. Models were reloaded with strict state-dict matching before their validation measurements. Existing reports, including the original MVP and the unavailable-checkpoint diagnosis, remain unchanged.

Frozen node embeddings are computed from the original full Cora graph without virtual prompt nodes. The reused transductive backbone includes unlabeled nodes from all classes; no test-class labels are used for scoring or optimization.

## Accuracy, Macro-F1 and gate

| Seed | Arm | Accuracy | Macro-F1 | Lambda | Accuracy delta vs NoPrompt |
|---:|---|---:|---:|---:|---:|
| 11 | NoPrompt | 0.717333 | 0.697877 | -- | +0.000000 |
| 11 | V3-correct | 0.656333 | 0.631865 | 0.109784558 | -0.061000 |
| 11 | V3-shuffled | 0.582667 | 0.538974 | 0.109790891 | -0.134667 |
| 11 | V3-random | 0.687667 | 0.664703 | 0.109354496 | -0.029667 |
| 23 | NoPrompt | 0.677333 | 0.650864 | -- | +0.000000 |
| 23 | V3-correct | 0.625000 | 0.592025 | 0.109579250 | -0.052333 |
| 23 | V3-shuffled | 0.564667 | 0.520951 | 0.109635964 | -0.112667 |
| 23 | V3-random | 0.676000 | 0.652975 | 0.109257005 | -0.001333 |
| 37 | NoPrompt | 0.720667 | 0.702303 | -- | +0.000000 |
| 37 | V3-correct | 0.657000 | 0.623777 | 0.109892465 | -0.063667 |
| 37 | V3-shuffled | 0.714000 | 0.696911 | 0.109766662 | -0.006667 |
| 37 | V3-random | 0.671333 | 0.651111 | 0.109755874 | -0.049333 |

Across-seed values use the sample SD (ddof=1), not an episode-level SD.

| Arm | Accuracy mean +/- SD | Macro-F1 mean +/- SD |
|---|---:|---:|
| NoPrompt | 0.705111 +/- 0.024114 | 0.683681 +/- 0.028507 |
| V3-correct | 0.646111 +/- 0.018286 | 0.615889 +/- 0.021059 |
| V3-shuffled | 0.620444 +/- 0.081520 | 0.585612 +/- 0.096808 |
| V3-random | 0.678333 +/- 0.008413 | 0.656263 +/- 0.007368 |

## Semantic/support alignment

Cosines use L2-normalized projected semantic vectors and support-derived GCN class means in the same episode. Matching cosine compares the same class; wrong-class cosine compares the other class. Their difference measures class-specific alignment rather than only a shared direction. Values average 200 class-prototype occurrences per arm; they are not independent samples.

| Seed | Arm | Initial matching cosine | Trained matching cosine | Trained wrong-class cosine | Matching minus wrong |
|---:|---|---:|---:|---:|---:|
| 11 | V3-correct | -0.170804679 | -0.403238940 | -0.397182231 | -0.006056710 |
| 11 | V3-shuffled | -0.170203185 | -0.426353462 | -0.415966873 | -0.010386589 |
| 11 | V3-random | 0.064350438 | 0.031510460 | 0.030213050 | +0.001297410 |
| 23 | V3-correct | -0.046912912 | -0.178697970 | -0.174659600 | -0.004038370 |
| 23 | V3-shuffled | -0.047075904 | -0.363501941 | -0.355062491 | -0.008439450 |
| 23 | V3-random | 0.023356567 | 0.023238099 | 0.022750754 | +0.000487346 |
| 37 | V3-correct | -0.061635012 | -0.329712077 | -0.326583866 | -0.003128211 |
| 37 | V3-shuffled | -0.060789267 | -0.318939637 | -0.322805109 | +0.003865471 |
| 37 | V3-random | 0.010645552 | -0.015381950 | -0.013599471 | -0.001782479 |

## Correct-class margins before and after fusion

Margin = correct-class cosine logit minus the highest incorrect-class logit, with temperature 0.1. A negative margin indicates an incorrect prediction (zero may be an argmax tie). Before fusion uses support-only prototypes on exactly the same queries. Counts describe 3,000 query occurrences per arm. Correct-to-wrong and wrong-to-correct identify whether fusion loses more correct decisions than it gains.

| Seed | Arm | Before mean | After mean | Delta | Margins decreased | Correct to wrong | Wrong to correct |
|---:|---|---:|---:|---:|---:|---:|---:|
| 11 | V3-correct | +0.020659395 | +0.015117205 | -0.005542190 | 56.80% | 424 | 241 |
| 11 | V3-shuffled | +0.020659395 | +0.009003798 | -0.011655597 | 64.07% | 588 | 184 |
| 11 | V3-random | +0.020659395 | +0.020710465 | +0.000051070 | 50.37% | 275 | 186 |
| 23 | V3-correct | +0.016252098 | +0.011898144 | -0.004353953 | 60.13% | 347 | 190 |
| 23 | V3-shuffled | +0.016252098 | +0.006681283 | -0.009570815 | 63.50% | 569 | 231 |
| 23 | V3-random | +0.016252098 | +0.018748766 | +0.002496668 | 44.50% | 224 | 220 |
| 37 | V3-correct | +0.020222665 | +0.017175123 | -0.003047543 | 55.27% | 356 | 165 |
| 37 | V3-shuffled | +0.020222665 | +0.025654508 | +0.005431843 | 41.30% | 274 | 254 |
| 37 | V3-random | +0.020222665 | +0.018016192 | -0.002206473 | 53.90% | 294 | 146 |

### Class-wise correct-description margins

Balanced-query overall means can hide changes in opposite directions for the two classes.

| Seed | Validation class | Before mean | After mean | Delta |
|---:|---|---:|---:|---:|
| 11 | 3: Neural Networks | +0.020234364 | +0.022721416 | +0.002487052 |
| 11 | 4: Probabilistic Methods | +0.021084426 | +0.007512994 | -0.013571431 |
| 23 | 3: Neural Networks | +0.017975311 | +0.019397917 | +0.001422606 |
| 23 | 4: Probabilistic Methods | +0.014528884 | +0.004398371 | -0.010130513 |
| 37 | 3: Neural Networks | +0.021149103 | +0.026545632 | +0.005396530 |
| 37 | 4: Probabilistic Methods | +0.019296228 | +0.007804613 | -0.011491615 |

## Similar semantic directions and possible collapse

There is one distinct validation class pair (IDs 3/4). Its cosine is fixed across episodes for each trained arm because descriptions and projection weights are fixed. Repetition over 100 episodes does not supply 100 independent semantic directions. Near-one cosine indicates similar directions, but alone does not prove complete collapse or identify a cause. Support/fused pairwise cosines provide a geometry reference; train/test class collapse is not evaluated.

| Seed | Arm | Input semantic cosine | Initial projected cosine | Trained projected cosine | Support cosine mean | Fused cosine mean |
|---:|---|---:|---:|---:|---:|---:|
| 11 | V3-correct | 0.962763071 | 0.967318177 | 0.284259498 | 0.990946718 | 0.980607087 |
| 11 | V3-shuffled | 0.962763071 | 0.967318177 | 0.313170969 | 0.990946718 | 0.981750392 |
| 11 | V3-random | 0.034311198 | 0.016220335 | 0.010565007 | 0.990946718 | 0.976427672 |
| 23 | V3-correct | 0.962763071 | 0.953641593 | 0.874869704 | 0.991195723 | 0.990073374 |
| 23 | V3-shuffled | 0.962763071 | 0.953641593 | 0.609378219 | 0.991195723 | 0.986658304 |
| 23 | V3-random | -0.028163355 | -0.035968602 | -0.040153947 | 0.991195723 | 0.975983536 |
| 37 | V3-correct | 0.962763071 | 0.954936266 | 0.616033912 | 0.991181570 | 0.985990632 |
| 37 | V3-shuffled | 0.962763071 | 0.954936266 | 0.627442360 | 0.991181570 | 0.983527515 |
| 37 | V3-random | -0.022787528 | -0.114912570 | -0.148873910 | 0.991181570 | 0.974451364 |

## Gate trajectory

Lambda is a shared sigmoid gate. Its entire 101-point history (initialization plus 100 updates) is recorded in JSON and checkpoints.

| Seed | Arm | Initial | Final | Final minus initial | Minimum | Maximum | Updates decreasing lambda |
|---:|---|---:|---:|---:|---:|---:|---:|
| 11 | V3-correct | 0.100000001 | 0.109784558 | +0.009784557 | 0.099910036 | 0.109784558 | 1/100 |
| 11 | V3-shuffled | 0.100000001 | 0.109790891 | +0.009790890 | 0.099877432 | 0.109790891 | 2/100 |
| 11 | V3-random | 0.100000001 | 0.109354496 | +0.009354495 | 0.099829942 | 0.109354496 | 2/100 |
| 23 | V3-correct | 0.100000001 | 0.109579250 | +0.009579249 | 0.099910036 | 0.109579250 | 1/100 |
| 23 | V3-shuffled | 0.100000001 | 0.109635964 | +0.009635963 | 0.099910036 | 0.109635964 | 1/100 |
| 23 | V3-random | 0.100000001 | 0.109257005 | +0.009257004 | 0.099879190 | 0.109257005 | 2/100 |
| 37 | V3-correct | 0.100000001 | 0.109892465 | +0.009892464 | 0.100000001 | 0.109892465 | 0/100 |
| 37 | V3-shuffled | 0.100000001 | 0.109766662 | +0.009766661 | 0.099910036 | 0.109766662 | 1/100 |
| 37 | V3-random | 0.100000001 | 0.109755874 | +0.009755872 | 0.100000001 | 0.109755874 | 0/100 |

## Diagnostic interpretation

- Seed 11: correct fusion changes accuracy by -0.061000 and Macro-F1 by -0.066012. Its mean margin changes by -0.005542190; it loses 424 correct decisions and gains 241. Matching-over-wrong support alignment is -0.006056710; projected validation-class cosine is 0.284259498.
- Seed 23: correct fusion changes accuracy by -0.052333 and Macro-F1 by -0.058839. Its mean margin changes by -0.004353953; it loses 347 correct decisions and gains 190. Matching-over-wrong support alignment is -0.004038370; projected validation-class cosine is 0.874869704.
- Seed 37: correct fusion changes accuracy by -0.063667 and Macro-F1 by -0.078526. Its mean margin changes by -0.003047543; it loses 356 correct decisions and gains 165. Matching-over-wrong support alignment is -0.003128211; projected validation-class cosine is 0.616033912.
- Correct versus NoPrompt: mean accuracy difference -0.059000, mean Macro-F1 difference -0.067792; both metrics improve in 0/3 seeds.
- Correct versus V3-shuffled: mean accuracy difference +0.025667, mean Macro-F1 difference +0.030277; both metrics improve in 2/3 seeds.
- Correct versus V3-random: mean accuracy difference -0.032222, mean Macro-F1 difference -0.040374; both metrics improve in 0/3 seeds.

All semantic arms finish with lambda above 0.1: the gate does not learn to turn off semantic fusion. Training-class optimization therefore retains semantic influence despite the validation degradation.

**Validation misalignment is observed in every seed.** Correct projected semantic prototypes have negative matching-support cosine (-0.403238940 to -0.178697970) and slightly worse alignment with their own support class than with the other class (matching-minus-wrong cosine -0.006056710 to -0.003128211). Their learned directions therefore do not provide the appropriate validation-class geometry despite correct text-label alignment.

**The projected validation prototypes do not show a training-induced collapse toward a common direction.** Their pairwise cosine decreases in every seed, from the initial approximately 0.95-0.97 to 0.284259498-0.874869704. One seed retains moderately high similarity, but none approaches identical projected directions. The directly measured degradation is more consistent with inappropriate orientation/class discrimination than complete direction collapse. This statement is limited to the two validation classes.

Probabilistic Methods (class 4) loses mean correct-class margin in every seed; Neural Networks (class 3) gains margin in these runs. Fusion therefore introduces an asymmetric class effect rather than a uniform improvement. More correct-to-wrong than wrong-to-correct flips in each correct arm explain its lower accuracy.

These comparisons measure the direct effect of replacing support-only prototypes with fused prototypes at fixed query embeddings. The causes of learned semantic geometry remain observational: separately trained correct/shuffled/random arms do not isolate every training effect. No claim of semantic improvement or test generalization is warranted when the correct arm loses to NoPrompt or the random control.

## Verification and reproducibility

- All 900 updates had finite, nonzero projection-weight, projection-bias and gate gradients. No update was skipped.
- SciBERT and GCN remain frozen, in evaluation mode, with no gradients; optimizer parameters are restricted to V3. GCN state hashes match after every arm and the SciBERT state hash matches before/after the whole experiment.
- Every episode passed balanced class-count, permitted-class, actual-label, unique-index and support/query-disjointness checks. All arms share the saved validation episodes and training order, with equal initialization hashes.
- Each saved model passed strict checkpoint reload and state/semantic-input equality checks before validation.
- 207 existing artifact hashes, including architecture files, V1/V2 checkpoints/reports, original V3 reports and manuscript, remained unchanged. Existing V3 checkpoints are preserved; filename collisions abort before training.
- Full numeric diagnostics, gate histories, episode metrics and artifact hashes are in `v3_alignment_diagnosis_v2.json`; scalar seed/arm metrics are in `v3_alignment_diagnosis_v2.csv`.

Prior MVP reproduction: accuracy, Macro-F1 and final lambda match the archived MVP within 1e-7: **True**.

| Seed | Arm | Saved checkpoint | SHA-256 |
|---:|---|---|---|
| 11 | V3-correct | `results/v3/checkpoints/semantic_prototype_fusion_correct_shot_3_seed_11.pt` | `f9d55988af0b93df4450765f91599321bd4e15344c6d149a9dea5d8fb2b9e4f7` |
| 11 | V3-shuffled | `results/v3/checkpoints/semantic_prototype_fusion_shuffled_shot_3_seed_11.pt` | `e77686f00ae723c4954dcf69149666949882dbe04191cf100996a87456752d98` |
| 11 | V3-random | `results/v3/checkpoints/semantic_prototype_fusion_random_shot_3_seed_11.pt` | `42b79ce721d3191e5f587363f85cae98fdd102a5be0e414f83f71deb3f23eff8` |
| 23 | V3-correct | `results/v3/checkpoints/semantic_prototype_fusion_correct_shot_3_seed_23.pt` | `69c16fa1a39a3e81db4caf575ea9f931788149b0db57b946f74e700f5dcc657b` |
| 23 | V3-shuffled | `results/v3/checkpoints/semantic_prototype_fusion_shuffled_shot_3_seed_23.pt` | `8c4b2b797dde7fc9df180706db634ebd27896881e4a004353dd5c88fbca01669` |
| 23 | V3-random | `results/v3/checkpoints/semantic_prototype_fusion_random_shot_3_seed_23.pt` | `9dd85f5341c816ec8d6a4beccc5d272ad3a5f9153b48ea07f1f0e92d07a50491` |
| 37 | V3-correct | `results/v3/checkpoints/semantic_prototype_fusion_correct_shot_3_seed_37.pt` | `33faa3f88f65dac8f26b9dd48f0650e6abdb654362e27b7da4a61650e579382c` |
| 37 | V3-shuffled | `results/v3/checkpoints/semantic_prototype_fusion_shuffled_shot_3_seed_37.pt` | `b2e8cb850c1110c6cfe5e16626a43edcc00790cae8340bb196b5826d98199045` |
| 37 | V3-random | `results/v3/checkpoints/semantic_prototype_fusion_random_shot_3_seed_37.pt` | `0ae026b7461ec3cd4d290e4ac55056638fddb44be8eb0e2042f95ba2b02f89a7` |

| Seed | Shared validation episode hash | Initialization hash |
|---:|---|---|
| 11 | `ba0a10e9d5c8c16c0631a75b8b5409f3a1b03fdcb1c5e6894a39833d63545038` | `dcd721794bdcb473c33be629e466a954e2ef820beb150f3e515e07dbb8d39d4e` |
| 23 | `909f0ef5cbd010d3b3320af75079609da6bc8f77951cc3c3a0b0f7059f5fc0e5` | `82eaeeada6691743d2ab949ab0da42472163996ea6fa238a8132a70de0d63ab2` |
| 37 | `dffb1a0927a00aebe92f500481b57edb99f9f4f328b9db575b272e3df4161bdc` | `5caa669dc760a628086018ce4a2b3f0bf607df6b09b6afe17efff24662718e86` |

Runtime: 26.00 seconds on cpu (excluding separate unit tests). Test-class results: **not evaluated**. No full experiments or V4 implementation were run.

Targeted V3 unit tests: 6 passed, 1 library deprecation warning; 55.49 seconds. They check the fusion formula, endpoints and gradients, margin signs, harmful misalignment and identical-direction detection without assuming classification failure.

Command from the implementation directory:

```powershell
& .venv/Scripts/python.exe LLMGP/experiments/run_v3_mvp_validation.py
```
