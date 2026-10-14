# LLMGP Performance Diagnosis

## Scope and safeguards
- Inference-only diagnosis of the completed revised 3/2/2 runs. No training was run, no optimizer steps were taken, and the original checkpoints and result CSVs were read-only inputs.
- RandomPrompt is reported only as an inference counterfactual: fixed-seed Gaussian class vectors matched to the mean and standard deviation of SciBERT class embeddings, passed through each saved trained projection. It is not a separately trained RandomPrompt baseline.
- Exact commands: `$py = "c:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/.venv/Scripts/python.exe"`; `Set-Location "C:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/LLMGP"`; `& $py experiments/run_performance_diagnostics.py`.
- Runtime: 1326.6 seconds on cpu.

## Accuracy and Macro-F1 by seed
| Shot | Accuracy by seed 11, 23, 37, 53, 71 | Mean accuracy | Macro-F1 by seed 11, 23, 37, 53, 71 | Mean Macro-F1 |
|---:|---|---:|---|---:|
| 1 | [0.494444, 0.513167, 0.653556, 0.507167, 0.519944] | 0.5377 | [0.392208, 0.379644, 0.600159, 0.358581, 0.373741] | 0.4209 |
| 3 | [0.698778, 0.533833, 0.748833, 0.709556, 0.752944] | 0.6888 | [0.677695, 0.410148, 0.731076, 0.687942, 0.73585] | 0.6485 |
| 5 | [0.523, 0.5285, 0.544667, 0.530389, 0.780444] | 0.5814 | [0.396159, 0.392562, 0.420594, 0.400963, 0.766646] | 0.4754 |
| 10 | [0.539944, 0.548722, 0.601056, 0.541389, 0.592056] | 0.5646 | [0.428602, 0.430297, 0.510165, 0.416933, 0.50474] | 0.4581 |

## Confirmed findings
- Checkpoint associations were verified for all 20 seed/shot pairs: prompt metadata seed/shot, associated GCN seed/checkpoint path, class split, and validation-best epoch all match the saved training logs.
- All 600 test episodes per run were reconstructed exactly from the saved seed rule. Query-node order and true labels matched every raw prediction file, and loaded saved checkpoints reproduced all saved predictions.
- Per-class numeric counts match the fixed mapping's expected count signature; test IDs are 5 and 6. However, the local Planetoid raw files contain no human-readable topic labels, so ID-to-topic-name alignment is not independently verifiable from the local raw source. Support/query node IDs are disjoint in reconstructed episodes.
- Every paired baseline below uses identical support/query indices and labels. These comparisons do not alter the primary metrics.
- The GCN remained frozen with no gradients and unchanged weights during diagnostic gradient probes. Projection gradients were finite; 19/20 seed/shot probes had nonzero mean gradient over five training episodes. Across probes, mean gradient norms ranged from 0.000000 to 0.122194; seed 37 at 1-shot had zero gradient on all five diagnostic episodes.
- GCNConv normalization is enabled, self-loops are added, and cached normalization is disabled. Prompt augmentation adds 4K directed edge entries for K supports per class; each prompt node receives degree K from its class's support nodes.
- Increasing K therefore changes the normalized adjacency, not only the number of labeled support examples. Support-node degrees also increase by one from their prompt edge, affecting normalization of their original graph edges.

## Paired baseline diagnosis
| Shot | Saved LLMGP accuracy / Macro-F1 | NoPrompt accuracy / Macro-F1 | RandomPrompt diagnostic accuracy / Macro-F1 |
|---:|---:|---:|---:|
| 1 | 0.5377 / 0.4209 | 0.6124 / 0.5434 | 0.5166 / 0.3874 |
| 3 | 0.6888 / 0.6485 | 0.7293 / 0.7114 | 0.6566 / 0.6086 |
| 5 | 0.5814 / 0.4754 | 0.7767 / 0.7683 | 0.6448 / 0.5857 |
| 10 | 0.5646 / 0.4581 | 0.8230 / 0.8196 | 0.5816 / 0.5038 |

NoPrompt uses the same seed-specific frozen GCN without prompt nodes. RandomPrompt diagnostic substitutes matched Gaussian class vectors into the saved trained projection. Since the random variant is not separately trained, treat it as a sensitivity check rather than a benchmark result.
- NoPrompt exceeds saved LLMGP mean accuracy by 7.47, 4.05, 19.53, and 25.83 percentage points at 1, 3, 5, and 10 shots, respectively. This is the strongest evidence that prompt conditioning, rather than checkpoint loading or episode construction, is harming these predictions as K grows.

### Test-class error balance
| Shot | Recall class 5 | Recall class 6 | Share of predictions assigned to class 5 |
|---:|---:|---:|---:|
| 1 | 0.7267 | 0.3486 | 0.6891 |
| 3 | 0.7330 | 0.6446 | 0.5442 |
| 5 | 0.7408 | 0.4220 | 0.6594 |
| 10 | 0.4147 | 0.7146 | 0.3501 |
Episodes are query-balanced by construction (15 from each class), despite unequal full-dataset class counts. The changing per-class recall therefore reflects classifier behavior, not different query counts.

## Training and representation diagnostics
| Shot | Mean train-split projection gradient norm | SD | NoPrompt query norm | Prompted query norm | Mean cosine similarity | Projected prompt norm | Prompt/input feature norm ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.037370 | 0.013982 | 5.3846 | 5.4822 | 0.9998 | 90.1231 | 21.54x |
| 3 | 0.013726 | 0.004851 | 5.3841 | 5.3804 | 0.9998 | 63.1255 | 15.09x |
| 5 | 0.056671 | 0.021855 | 5.3843 | 5.5040 | 0.9988 | 82.5168 | 19.72x |
| 10 | 0.063469 | 0.032033 | 5.3868 | 5.5302 | 0.9977 | 86.9201 | 20.78x |

### Existing training-log summary
| Shot | First-epoch train loss (mean) | Last-epoch train loss (mean) | Best validation CE (mean) | Selected-checkpoint validation accuracy (mean) |
|---:|---:|---:|---:|---:|
| 1 | 0.6578 | 0.5989 | 0.6959 | 0.5224 |
| 3 | 0.6687 | 0.6571 | 0.6870 | 0.6624 |
| 5 | 0.6463 | 0.5994 | 0.6959 | 0.5411 |
| 10 | 0.6246 | 0.5583 | 0.6927 | 0.5200 |
Train loss falls at every shot on average, but the selected validation accuracy peaks at 3-shot and is near chance for 5/10-shot. This is consistent with training-to-validation generalization failure, not an evaluation-only checkpoint mix-up.
Checkpoint selection itself matches the minimum validation-loss epoch exactly and did not use test metrics.
- Masked-feature reconstruction loss fell from approximately 1.0 initially to best values between 0.0332 and 0.0352 across the five GCN seeds; final losses were 0.0351–0.0375. This confirms the pretext objective optimized, but does not establish that the resulting frozen embeddings are useful for held-out class discrimination.

## Likely explanations for the post-3-shot decline
- Confirmed design effect: higher shot counts add more prompt-support edges, altering GCN normalization and messages over the whole graph. The 5- and 10-shot tasks are therefore not merely lower-variance versions of the 3-shot graph computation.
- Strong suspected cause: prompt conditioning is detrimental relative to NoPrompt, and its impact grows after 3-shot. Projected prompt feature norms average 15–22x the mean real-node feature norm, while output query embeddings remain directionally close to NoPrompt (cosine 0.9977–0.9998). That combination is consistent with a high-scale prompt perturbation that is weakly represented at the query but still shifts class boundaries; this is a hypothesis, not a causal ablation.
- Gradient stability is a secondary concern: 19/20 run/shot probes have nonzero mean gradients, one probe group is exactly zero, and magnitudes vary substantially across seeds/shots. No NaN/Inf, frozen-weight update, or missing-gradient error was found.
- Error balance also shifts: 5-shot recall for class 6 falls to 0.422 overall, while 10-shot recall for class 5 falls to 0.415. This suggests class-specific prototype/representation shifts rather than query-count imbalance.
- Confirmed comparison limitation: per-shot episode seeds include the shot value, so different shot settings do not share the same sampled query set. Within each shot, all three variants are paired exactly; across shots, some trend variation may come from different episode draws.
- Mapping limitation/suspected confound: if any of the manually authored topic descriptions is assigned to the wrong numeric Cora ID, the semantic prompts are misaligned. Confirm the mapping against the original `cora.content` label column and node IDs before drawing conclusions about SciBERT semantic-prompt quality.
- Test seed variation is substantial for 5-shot (including a strong seed 71) while several other seeds cluster lower; five-seed mean estimates are noisy. Do not select or tune against the held-out test set.
- Self-supervised pretraining losses and settings are available per seed in `results/logs/revised/seed_<seed>_pretraining.json`. Masked-feature MSE is only a pretext objective, not a task-utility guarantee; it can yield representations weak for two-way class separation.

## Recommended corrections
1. Keep the current test results frozen. For a new, explicitly labeled experiment, use common episode seeds/query sets across shot settings where feasible, with nested support sets so only K changes.
2. In a new run, calibrate or constrain prompt-feature scale using train/validation only; log gradient norm, prompt norm, support/query embedding norms, and augmented-edge statistics during training. Never use test metrics for early stopping or model selection.
3. Compare a separately trained RandomPrompt baseline using the same train/validation episodes and compute budget before describing it as a benchmark. The counterfactual here is not that baseline.
4. Run pre-registered ablations on train/validation only to test normalization sensitivity (e.g. edge-weight normalization or prompt-edge scaling) and self-supervised objective quality; evaluate each chosen configuration once on test.
5. Consider a self-supervised objective aligned with node discrimination, such as feature reconstruction combined with graph-view contrastive learning, only as a new experiment and selected by validation performance.

## Diagnostic artifacts
- `results/diagnostics/performance_diagnostics.json` contains the full checkpoint, episode, graph, embedding, and seed diagnostics.
- `results/diagnostics/baseline_comparison.csv` contains per-seed paired baseline results.
- `results/diagnostics/gradient_graph_diagnostics.csv` contains train-split gradient and representation probes.
- Original checkpoints and primary result files were not overwritten.
