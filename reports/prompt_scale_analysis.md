# Prompt Scale and Connectivity Analysis

## Scope and test-status warning
- This is a read-only controlled inference analysis. No models were retrained, no checkpoints were overwritten, and no optimizer steps were taken.
- The test classes/episodes have already informed the previous diagnosis. Therefore, every test result here, including the selected-scale test value, is exploratory and post hoc, not an untouched independent confirmation.
- Scaling was selected using validation classes [3, 4] only, before the test sweep. The single global scale maximizes the unweighted mean validation Macro-F1 across all five seeds and four shot settings; ties select the lower scale because candidates are ordered ascending.
- Runtime includes model loading and both validation and test sweeps.

## Executed command
```powershell
$py = "c:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/.venv/Scripts/python.exe"
Set-Location "C:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/LLMGP"
& $py experiments/run_prompt_scale_analysis.py
```
Runtime: 5611.3 seconds on cpu.

## Control definitions
- **NoPrompt:** original graph only; prompt nodes and prompt edges are absent.
- **ZeroPrompt (scale 0.0):** the same two prompt nodes and support connections remain, but their input features are exactly zero. This is not equivalent to NoPrompt because topology, node degrees, self-loop normalization, and message passing change.
- **OriginalPrompt (scale 1.0):** saved learned projection output and original prompt connectivity.
- **ScaledPrompt:** the saved learned prompt features multiplied by the listed factor; all variants share the same support/query nodes and frozen checkpoints within each episode.
- Validation: 200 reproducible paired episodes per seed/shot on classes [3,4]. Test: the original 600 paired episodes per seed/shot on classes [5,6], reconstructed and checked against saved query IDs/truth labels.
- Accuracy and Macro-F1 confidence intervals are two-sided Student-t intervals over the five seed-level episode means (df=4; sample SD, ddof=1). Validation intervals describe selection data; test intervals are exploratory only.

## Validation-only scale selection
| Scale | Mean validation accuracy | Mean validation Macro-F1 |
|---:|---:|---:|
| 0.00 | 0.7135 | 0.6882 |
| 0.01 | 0.7122 | 0.6868 |
| 0.05 | 0.6889 | 0.6515 |
| 0.10 | 0.6567 | 0.6027 |
| 0.25 | 0.5920 | 0.5022 |
| 0.50 | 0.5709 | 0.4705 |
| 1.00 | 0.5612 | 0.4560 |

Selected global scale: **0.00**, based only on validation mean Macro-F1 0.6882 (mean accuracy 0.7135).

## Test scale sweep (exploratory)
Each entry is mean across five seed means; 95% CI is across seeds, not episodes.
| Shot | Scale | Accuracy (95% CI) | Macro-F1 (95% CI) | Prompt norm / node norm | Query delta L2 | Query cosine vs NoPrompt | Prompt degree | Norm edge-weight mean |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 1 | NoPrompt (--) | 0.6124 [0.5876, 0.6371] | 0.5434 [0.5131, 0.5736] | 0.000 | 0.0000 | 1.00000 | 0.00 | 0.188883 |
| 1 | ZeroPrompt (0.00) | 0.6118 [0.5860, 0.6375] | 0.5454 [0.5137, 0.5771] | 0.000 | 0.0034 | 1.00000 | 1.00 | 0.188939 |
| 1 | ScaledPrompt (0.01) | 0.6129 [0.5933, 0.6325] | 0.5483 [0.5297, 0.5669] | 0.215 | 0.0031 | 1.00000 | 1.00 | 0.188939 |
| 1 | ScaledPrompt (0.05) | 0.5924 [0.5525, 0.6322] | 0.5122 [0.4529, 0.5714] | 1.077 | 0.0057 | 0.99999 | 1.00 | 0.188939 |
| 1 | ScaledPrompt (0.10) | 0.5692 [0.5109, 0.6275] | 0.4701 [0.3781, 0.5622] | 2.154 | 0.0108 | 0.99998 | 1.00 | 0.188939 |
| 1 | ScaledPrompt (0.25) | 0.5528 [0.4820, 0.6237] | 0.4408 [0.3278, 0.5538] | 5.385 | 0.0264 | 0.99993 | 1.00 | 0.188939 |
| 1 | ScaledPrompt (0.50) | 0.5458 [0.4706, 0.6210] | 0.4315 [0.3133, 0.5497] | 10.771 | 0.0530 | 0.99985 | 1.00 | 0.188939 |
| 1 | OriginalPrompt (1.00) | 0.5377 [0.4564, 0.6189] | 0.4209 [0.2955, 0.5462] | 21.542 | 0.1065 | 0.99975 | 1.00 | 0.188939 |
| 3 | NoPrompt (--) | 0.7293 [0.7001, 0.7584] | 0.7114 [0.6797, 0.7430] | 0.000 | 0.0000 | 1.00000 | 0.00 | 0.188883 |
| 3 | ZeroPrompt (0.00) | 0.7263 [0.6959, 0.7567] | 0.7087 [0.6753, 0.7420] | 0.000 | 0.0106 | 1.00000 | 3.00 | 0.188837 |
| 3 | ScaledPrompt (0.01) | 0.7233 [0.6931, 0.7534] | 0.7041 [0.6709, 0.7372] | 0.151 | 0.0134 | 1.00000 | 3.00 | 0.188837 |
| 3 | ScaledPrompt (0.05) | 0.7114 [0.6814, 0.7413] | 0.6857 [0.6518, 0.7196] | 0.754 | 0.0229 | 0.99999 | 3.00 | 0.188837 |
| 3 | ScaledPrompt (0.10) | 0.7121 [0.6676, 0.7566] | 0.6848 [0.6268, 0.7428] | 1.509 | 0.0306 | 0.99998 | 3.00 | 0.188837 |
| 3 | ScaledPrompt (0.25) | 0.7028 [0.6118, 0.7938] | 0.6683 [0.5304, 0.8062] | 3.772 | 0.0404 | 0.99996 | 3.00 | 0.188837 |
| 3 | ScaledPrompt (0.50) | 0.6957 [0.5872, 0.8043] | 0.6573 [0.4909, 0.8237] | 7.544 | 0.0503 | 0.99992 | 3.00 | 0.188837 |
| 3 | OriginalPrompt (1.00) | 0.6888 [0.5773, 0.8003] | 0.6485 [0.4800, 0.8170] | 15.089 | 0.0685 | 0.99984 | 3.00 | 0.188837 |
| 5 | NoPrompt (--) | 0.7767 [0.7461, 0.8073] | 0.7683 [0.7363, 0.8002] | 0.000 | 0.0000 | 1.00000 | 0.00 | 0.188883 |
| 5 | ZeroPrompt (0.00) | 0.7739 [0.7418, 0.8060] | 0.7659 [0.7323, 0.7995] | 0.000 | 0.0177 | 1.00000 | 5.00 | 0.188726 |
| 5 | ScaledPrompt (0.01) | 0.7736 [0.7409, 0.8063] | 0.7653 [0.7311, 0.7996] | 0.197 | 0.0188 | 1.00000 | 5.00 | 0.188726 |
| 5 | ScaledPrompt (0.05) | 0.7513 [0.7073, 0.7953] | 0.7350 [0.6818, 0.7882] | 0.986 | 0.0244 | 0.99997 | 5.00 | 0.188726 |
| 5 | ScaledPrompt (0.10) | 0.7015 [0.6184, 0.7846] | 0.6616 [0.5440, 0.7791] | 1.972 | 0.0318 | 0.99992 | 5.00 | 0.188726 |
| 5 | ScaledPrompt (0.25) | 0.6291 [0.4998, 0.7585] | 0.5462 [0.3599, 0.7325] | 4.931 | 0.0501 | 0.99972 | 5.00 | 0.188726 |
| 5 | ScaledPrompt (0.50) | 0.5980 [0.4516, 0.7444] | 0.4973 [0.2854, 0.7091] | 9.862 | 0.0872 | 0.99937 | 5.00 | 0.188726 |
| 5 | OriginalPrompt (1.00) | 0.5814 [0.4429, 0.7199] | 0.4754 [0.2728, 0.6780] | 19.724 | 0.1745 | 0.99877 | 5.00 | 0.188726 |
| 10 | NoPrompt (--) | 0.8230 [0.7923, 0.8536] | 0.8196 [0.7883, 0.8509] | 0.000 | 0.0000 | 1.00000 | 0.00 | 0.188883 |
| 10 | ZeroPrompt (0.00) | 0.8207 [0.7895, 0.8519] | 0.8174 [0.7855, 0.8493] | 0.000 | 0.0345 | 1.00000 | 10.00 | 0.188419 |
| 10 | ScaledPrompt (0.01) | 0.8195 [0.7874, 0.8517] | 0.8162 [0.7834, 0.8490] | 0.208 | 0.0359 | 1.00000 | 10.00 | 0.188419 |
| 10 | ScaledPrompt (0.05) | 0.8110 [0.7738, 0.8482] | 0.8063 [0.7668, 0.8458] | 1.039 | 0.0417 | 0.99997 | 10.00 | 0.188419 |
| 10 | ScaledPrompt (0.10) | 0.7862 [0.7323, 0.8400] | 0.7758 [0.7132, 0.8385] | 2.078 | 0.0481 | 0.99990 | 10.00 | 0.188419 |
| 10 | ScaledPrompt (0.25) | 0.6825 [0.6025, 0.7626] | 0.6304 [0.5155, 0.7453] | 5.194 | 0.0604 | 0.99959 | 10.00 | 0.188419 |
| 10 | ScaledPrompt (0.50) | 0.6003 [0.5450, 0.6556] | 0.5091 [0.4253, 0.5929] | 10.388 | 0.0970 | 0.99900 | 10.00 | 0.188419 |
| 10 | OriginalPrompt (1.00) | 0.5646 [0.5280, 0.6013] | 0.4581 [0.4018, 0.5144] | 20.776 | 0.2123 | 0.99774 | 10.00 | 0.188419 |

## Paired connectivity comparison
| Shot | NoPrompt accuracy | ZeroPrompt accuracy | OriginalPrompt accuracy | Validation-selected scale test accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.6124 | 0.6118 | 0.5377 | 0.6118 |
| 3 | 0.7293 | 0.7263 | 0.6888 | 0.7263 |
| 5 | 0.7767 | 0.7739 | 0.5814 | 0.7739 |
| 10 | 0.8230 | 0.8207 | 0.5646 | 0.8207 |

### Paired accuracy contrasts (percentage points)
Intervals are paired Student-t intervals across five seeds, using the same episode set within each comparison.
| Shot | Contrast | Mean difference (pp) | 95% CI (pp) |
|---:|---|---:|---:|
| 1 | ZeroPrompt - NoPrompt | -0.06 | [-0.23, 0.11] |
| 1 | Scale 0.01 - ZeroPrompt | 0.11 | [-0.86, 1.08] |
| 1 | OriginalPrompt - ZeroPrompt | -7.41 | [-13.43, -1.39] |
| 3 | ZeroPrompt - NoPrompt | -0.30 | [-0.47, -0.13] |
| 3 | Scale 0.01 - ZeroPrompt | -0.30 | [-0.55, -0.06] |
| 3 | OriginalPrompt - ZeroPrompt | -3.75 | [-14.61, 7.11] |
| 5 | ZeroPrompt - NoPrompt | -0.28 | [-0.50, -0.06] |
| 5 | Scale 0.01 - ZeroPrompt | -0.03 | [-0.27, 0.21] |
| 5 | OriginalPrompt - ZeroPrompt | -19.25 | [-31.14, -7.36] |
| 10 | ZeroPrompt - NoPrompt | -0.23 | [-0.55, 0.10] |
| 10 | Scale 0.01 - ZeroPrompt | -0.11 | [-0.25, 0.03] |
| 10 | OriginalPrompt - ZeroPrompt | -25.61 | [-27.35, -23.86] |

Across-shot paired contrasts average each seed's four shot-specific differences, then form a t interval across the five seed means:
| Contrast | Mean difference (pp) | 95% CI (pp) |
|---|---:|---:|
| ZeroPrompt - NoPrompt | -0.22 | [-0.38, -0.05] |
| Scale 0.01 - ZeroPrompt | -0.08 | [-0.26, 0.09] |
| OriginalPrompt - ZeroPrompt | -14.00 | [-18.90, -9.11] |

## Findings
- A difference between NoPrompt and ZeroPrompt isolates the topology/normalization effect at zero feature magnitude. Scale changes relative to ZeroPrompt then measure the additional effect of nonzero prompt features under the same connectivity.
- Review the table above alongside `diagnostics/prompt_scale_results.csv`, which retains seed-level paired measurements, and `diagnostics/connectivity_comparison.csv`, which labels the connectivity variant explicitly.
- **Hypothesis status: supported for prompt-feature magnitude, not for connectivity as the main driver.** Validation Macro-F1 declines monotonically from scale 0.0 to 1.0, and exploratory test accuracy drops strongly from ZeroPrompt to OriginalPrompt, especially at 5/10-shot. ZeroPrompt remains close to NoPrompt, with a much smaller paired difference. This is a controlled fixed-checkpoint association, not proof that scale alone causes the original cross-shot trend.
- The validation-selected value 0.0 is **ZeroPrompt**, not NoPrompt: prompt nodes and edges remain, but prompt features are zero. The test set was already examined, so its selected-scale score is exploratory and not independent confirmation.
- Feature scaling does not alter edge weights within a fixed connectivity graph; normalization statistics are constant across nonzero scales. They change between NoPrompt and prompt-connected graphs and as support count changes.

## Recommended next experiment
Run a newly preregistered study with a fresh held-out class split or independent dataset, selecting scale using train/validation only. Include trainable or validation-selected prompt-scale controls and separately trained NoPrompt/RandomPrompt baselines. Preserve nested support sets across K and report graph normalization and prompt-feature scale. Independently verify the Cora numeric-ID/topic-description alignment. Do not tune against the currently examined test classes.

## Reproducibility outputs
- `diagnostics/prompt_scale_results.csv`: split-, seed-, shot-, variant-, and scale-level metrics, confidence intervals, embedding shifts, and normalization values.
- `diagnostics/connectivity_comparison.csv`: NoPrompt, ZeroPrompt, OriginalPrompt, and ScaledPrompt rows on the same paired episodes.
- Original result CSVs/checkpoints and `reports/performance_diagnosis.md` were not overwritten.
