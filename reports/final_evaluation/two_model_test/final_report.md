# Final read-only NoPrompt versus V1 test evaluation

## Fixed protocol

Existing validated checkpoints only; five seeds 11/23/37/53/71 and shots 1/3/5/10. Train [0,1,2], validation [3,4], test [5,6] (Case-Based Reasoning / Rule Learning). Each seed/shot uses 600 reproducible 2-way test episodes with 15 queries per class. Both models receive the exact same support/query indices, labels and ordering. No episode has support/query overlap. Node IDs and local labels were checked against actual dataset labels, with unique indices and balanced support/query counts.

The existing test protocol's seed rule is `seed * 10000000 + 9000000 + shot * 100000 + episode_index`. The unchanged `sample_split_episode` then XORs with `0x5DEECE66D` for its internal sampler. This reuses the established test sampling rule rather than choosing episodes based on performance. The full manifest and per-set hashes were fixed before inference.

NoPrompt computes support-only cosine prototypes from the unaugmented frozen GCN graph. V1 uses its trained 768-to-1433 SciBERT projection, appends two prompt nodes and connects them only to matching support nodes, then classifies using GCN support prototypes. Temperature remains 0.1. The same seed-specific frozen GCN is used by both models. SciBERT uses the existing descriptions and original seven-description encoder batch. Query labels are used only for scoring, not graph augmentation or optimization.

## Accuracy and Macro-F1

Each seed value averages 600 episode metrics. Mean and sample SD (ddof=1) are computed across five seed values. Confidence intervals are two-sided 95% Student-t intervals, df=4: `mean +/- t(0.975,4)*sample_SD/sqrt(5)`. Intervals are not clipped to [0,1]. Repeated query nodes and episodes are not treated as independent seed replications.

| Shot | Model | Accuracy mean +/- SD | Accuracy 95% CI | Macro-F1 mean +/- SD | Macro-F1 95% CI |
|---:|---|---:|---|---:|---|
| 1 | NoPrompt | 0.612367 +/- 0.019931 | [0.587620, 0.637114] | 0.543353 +/- 0.024363 | [0.513102, 0.573603] |
| 1 | V1 | 0.537656 +/- 0.065466 | [0.456369, 0.618942] | 0.420867 +/- 0.100954 | [0.295516, 0.546217] |
| 3 | NoPrompt | 0.729267 +/- 0.023501 | [0.700087, 0.758447] | 0.711379 +/- 0.025489 | [0.679730, 0.743027] |
| 3 | V1 | 0.688789 +/- 0.089810 | [0.577275, 0.800303] | 0.648542 +/- 0.135710 | [0.480036, 0.817048] |
| 5 | NoPrompt | 0.776700 +/- 0.024651 | [0.746092, 0.807308] | 0.768281 +/- 0.025735 | [0.736327, 0.800235] |
| 5 | V1 | 0.581400 +/- 0.111556 | [0.442885, 0.719915] | 0.475385 +/- 0.163179 | [0.272771, 0.677999] |
| 10 | NoPrompt | 0.822956 +/- 0.024656 | [0.792342, 0.853569] | 0.819599 +/- 0.025201 | [0.788308, 0.850890] |
| 10 | V1 | 0.564633 +/- 0.029502 | [0.528001, 0.601265] | 0.458148 +/- 0.045343 | [0.401847, 0.514448] |

## Paired model differences

Differences are V1 minus NoPrompt on the same episodes within each seed, then summarized across five paired seed differences. This uses the SD of paired differences, not an unpaired or episode-level confidence interval. Positive differences favor V1. Values are fractions (0.01 is one percentage point). Intervals are per metric/shot and are not adjusted for multiple comparisons.

| Shot | Metric | Paired mean difference +/- sample SD | Paired 95% CI |
|---:|---|---:|---|
| 1 | accuracy | -0.074711 +/- 0.049791 | [-0.136535, -0.012887] |
| 1 | macro_f1 | -0.122486 +/- 0.083890 | [-0.226650, -0.018323] |
| 3 | accuracy | -0.040478 +/- 0.087323 | [-0.148904, +0.067948] |
| 3 | macro_f1 | -0.062837 +/- 0.133991 | [-0.229208, +0.103535] |
| 5 | accuracy | -0.195300 +/- 0.095872 | [-0.314341, -0.076259] |
| 5 | macro_f1 | -0.292896 +/- 0.146893 | [-0.475288, -0.110505] |
| 10 | accuracy | -0.258322 +/- 0.013847 | [-0.275516, -0.241128] |
| 10 | macro_f1 | -0.361451 +/- 0.026597 | [-0.394476, -0.328427] |

## Seed-level results

| Seed | Shot | Model | Accuracy | Macro-F1 |
|---:|---:|---|---:|---:|
| 11 | 1 | NoPrompt | 0.588222 | 0.516764 |
| 11 | 1 | V1 | 0.494444 | 0.392208 |
| 11 | 3 | NoPrompt | 0.702611 | 0.682954 |
| 11 | 3 | V1 | 0.698778 | 0.677695 |
| 11 | 5 | NoPrompt | 0.745944 | 0.736360 |
| 11 | 5 | V1 | 0.523000 | 0.396159 |
| 11 | 10 | NoPrompt | 0.792000 | 0.788322 |
| 11 | 10 | V1 | 0.539944 | 0.428602 |
| 23 | 1 | NoPrompt | 0.605222 | 0.531429 |
| 23 | 1 | V1 | 0.513167 | 0.379644 |
| 23 | 3 | NoPrompt | 0.730444 | 0.712625 |
| 23 | 3 | V1 | 0.533833 | 0.410148 |
| 23 | 5 | NoPrompt | 0.778611 | 0.770354 |
| 23 | 5 | V1 | 0.528500 | 0.392562 |
| 23 | 10 | NoPrompt | 0.821889 | 0.818132 |
| 23 | 10 | V1 | 0.548722 | 0.430297 |
| 37 | 1 | NoPrompt | 0.639611 | 0.578235 |
| 37 | 1 | V1 | 0.653556 | 0.600159 |
| 37 | 3 | NoPrompt | 0.746833 | 0.730503 |
| 37 | 3 | V1 | 0.748833 | 0.731076 |
| 37 | 5 | NoPrompt | 0.794556 | 0.787461 |
| 37 | 5 | V1 | 0.544667 | 0.420594 |
| 37 | 10 | NoPrompt | 0.838722 | 0.835798 |
| 37 | 10 | V1 | 0.601056 | 0.510165 |
| 53 | 1 | NoPrompt | 0.604278 | 0.532885 |
| 53 | 1 | V1 | 0.507167 | 0.358581 |
| 53 | 3 | NoPrompt | 0.709167 | 0.689033 |
| 53 | 3 | V1 | 0.709556 | 0.687942 |
| 53 | 5 | NoPrompt | 0.758722 | 0.749110 |
| 53 | 5 | V1 | 0.530389 | 0.400963 |
| 53 | 10 | NoPrompt | 0.807722 | 0.803782 |
| 53 | 10 | V1 | 0.541389 | 0.416933 |
| 71 | 1 | NoPrompt | 0.624500 | 0.557450 |
| 71 | 1 | V1 | 0.519944 | 0.373741 |
| 71 | 3 | NoPrompt | 0.757278 | 0.741778 |
| 71 | 3 | V1 | 0.752944 | 0.735850 |
| 71 | 5 | NoPrompt | 0.805667 | 0.798121 |
| 71 | 5 | V1 | 0.780444 | 0.766646 |
| 71 | 10 | NoPrompt | 0.854444 | 0.851961 |
| 71 | 10 | V1 | 0.592056 | 0.504740 |

## Training and checkpoint selection differ

Both methods share the previously trained GCN: 100 epochs of masked-feature-reconstruction self-supervision, Adam lr=0.001/wd=0.0001, 15% feature masking. All Cora node features and graph edges, including unlabeled test nodes, were transductively visible. No node labels were used in GCN pretraining or GCN checkpoint selection. The original full graph remains visible during this inference run.

NoPrompt has no prompt-training stage and no separately selected projection. V1 additionally trained its semantic projection on train classes [0,1,2] using episodic classification, with Adam lr=0.001/wd=0.0001, 100 episodes per epoch, maximum 50 epochs, patience 10, and 30 fixed validation episodes on [3,4]. The existing V1 checkpoints were selected by minimum validation cross-entropy. Best epochs and logged training lengths are listed below; they were not changed after test scoring.

**This is not an equal-training-budget comparison.** V1 has additional supervised episodic training and validation-based projection selection; NoPrompt uses only the common pretrained backbone plus episode supports. Test results were not used for checkpoint selection, hyperparameter changes or follow-up experiments.

| Seed | Shot | V1 selected epoch | Logged epochs run | Logged training episodes |
|---:|---:|---:|---:|---:|
| 11 | 1 | 1 | 11 | 1100 |
| 11 | 3 | 6 | 16 | 1600 |
| 11 | 5 | 1 | 11 | 1100 |
| 11 | 10 | 4 | 14 | 1400 |
| 23 | 1 | 1 | 11 | 1100 |
| 23 | 3 | 1 | 11 | 1100 |
| 23 | 5 | 4 | 14 | 1400 |
| 23 | 10 | 18 | 28 | 2800 |
| 37 | 1 | 7 | 17 | 1700 |
| 37 | 3 | 3 | 13 | 1300 |
| 37 | 5 | 10 | 20 | 2000 |
| 37 | 10 | 4 | 14 | 1400 |
| 53 | 1 | 5 | 15 | 1500 |
| 53 | 3 | 2 | 12 | 1200 |
| 53 | 5 | 1 | 11 | 1100 |
| 53 | 10 | 2 | 12 | 1200 |
| 71 | 1 | 3 | 13 | 1300 |
| 71 | 3 | 16 | 26 | 2600 |
| 71 | 5 | 9 | 19 | 1900 |
| 71 | 10 | 9 | 19 | 1900 |

## Exploratory results remain separate

V2, V3 and V3-Align are not part of this test comparison. Their existing findings remain exploratory validation results in their original reports. No exploratory metric was pooled with, renamed as or substituted for a test result. The earlier five-model audit remains an incomplete-comparison audit; this report is the separately authorized NoPrompt/V1 final test only.

The revised 3/2/2 transductive protocol differs from the manuscript's original protocol. This comparison does not claim manuscript reproduction or a new dataset split. Existing revised-protocol V1 test runs used the same test seed rule; this final run adds the explicitly paired NoPrompt comparison without changing checkpoint selection.

## Integrity, artifacts and stop point

- All 40 logical NoPrompt/V1 seed/shot configurations passed fresh checkpoint-state and protocol validation before any test scoring (five GCN files shared across shots and 20 V1 prompt files). Full identifiers, SHA-256 values and metadata are in `checkpoint_audit.json`.
- SciBERT, GCN and V1 projection remain in eval mode, requires_grad=False, with no gradients. Their in-memory state hashes match before/after inference.
- All 235 existing file hashes remain unchanged, including every checkpoint, script, result, report and manuscript included in the snapshot; runtime __pycache__/.pytest_cache files are excluded. No existing artifact was overwritten.
- Test episode manifest SHA-256: `809a00113ab8a9262e7179eda06f8a4c881768537325882c82f31ef0225a7917`.
- `test_episode_manifest.json`: support/query indices, labels, sampler seeds and per-set hashes.
- `episode_metrics.csv`, `seed_metrics.csv`, `aggregate_metrics.csv`: per-episode, per-seed and final metrics.
- `paired_seed_differences.csv`, `paired_differences.csv`: paired seed differences and their confidence intervals.
- `test_predictions.npz`: local class predictions and true labels (0=class 5, 1=class 6), shaped [600,30] per model/seed/shot; corresponding query IDs are in the manifest.
- `results.json`: full numerical results, training provenance, frozen-state and protected-file hashes.

Runtime: 615.84 seconds on cpu. Exactly this one final paired test evaluation was run. No retraining, tuning, architecture change, test-based selection or additional experiment was performed.

Command from the implementation directory:

```powershell
& .venv/Scripts/python.exe LLMGP/reports/final_evaluation/two_model_test/run_evaluation.py
```

The runner refuses existing output filenames before inference to prevent overwriting or accidental repeat evaluation. Work stops after reporting these results.
