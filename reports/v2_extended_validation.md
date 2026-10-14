# LLMGP-v2 Extended Validation

## Scope and safeguards
- Validation only: 3-shot, seeds 11/23/37, 100 V2 training episodes for each semantic arm per seed, and 100 identical validation episodes per seed shared across V1, V2-correct, V2-shuffled, and NoPrompt.
- Test classes [5,6] were not scored, used for model selection, or used to generate class-description embeddings.
- Existing v1 checkpoints/results were hash-compared before/after and remained unchanged.
- Runtime: 132.74 seconds on cpu.

## Reused frozen models and semantic caveat
- Frozen GCN checkpoints are the existing seed-specific v1 self-supervised checkpoints; SHA-256 identifiers are listed below.
- Frozen SciBERT identifier: `allenai/scibert_scivocab_uncased` (pretrained Hugging Face model/cache; this project has no separate SciBERT checkpoint file).
- Only class IDs [0,1,2,3,4] were encoded. Local Cora Planetoid labels are numeric/one-hot and do not include aligned topic names; configured class-description alignment is therefore not independently verified.
- The shuffled arm permutes descriptions only among training class slots [0,1,2]; validation class descriptions remain unchanged.

## Validation results
Values are means over 100 episode-level metrics per seed. Across-seed confidence intervals use a two-sided Student t interval with df=2; with three seeds these intervals are imprecise.
| Model | Accuracy mean +/- SD (95% CI) | Macro-F1 mean +/- SD (95% CI) | Effective prompt norm | Gate |
|---|---|---|---:|---:|
| V1 saved OriginalPrompt | 0.6296 +/- 0.1326 [0.3001, 0.9590] | 0.5692 +/- 0.1910 [0.0947, 1.0437] | 66.1651 | -- |
| V2 configured descriptions | 0.6927 +/- 0.0251 [0.6302, 0.7551] | 0.6721 +/- 0.0253 [0.6093, 0.7350] | 0.4620 | 0.11024673034747441 |
| V2 shuffled train descriptions | 0.6936 +/- 0.0194 [0.6453, 0.7418] | 0.6729 +/- 0.0175 [0.6294, 0.7164] | 0.4626 | 0.1103883683681488 |
| NoPrompt | 0.7049 +/- 0.0195 [0.6563, 0.7534] | 0.6866 +/- 0.0181 [0.6416, 0.7316] | 0.0000 | -- |

### Seed-wise results
| Seed | Model | Accuracy | Macro-F1 | Gate | Train loss (if trained) | Zero-gradient skips |
|---:|---|---:|---:|---:|---:|---:|
| 11 | V1 saved OriginalPrompt | 0.7413 | 0.7220 | -- | -- | -- |
| 11 | V2 configured descriptions | 0.7210 | 0.6996 | 0.11042685806751251 | 0.675224671959877 | 0 |
| 11 | V2 shuffled train descriptions | 0.7160 | 0.6930 | 0.11056237667798996 | 0.6752256172895431 | 0 |
| 11 | NoPrompt | 0.7273 | 0.7074 | -- | -- | -- |
| 23 | V1 saved OriginalPrompt | 0.4830 | 0.3551 | -- | -- | -- |
| 23 | V2 configured descriptions | 0.6730 | 0.6498 | 0.10990200191736221 | 0.6763708788156509 | 0 |
| 23 | V2 shuffled train descriptions | 0.6827 | 0.6609 | 0.11016666144132614 | 0.6764127379655838 | 0 |
| 23 | NoPrompt | 0.6917 | 0.6743 | -- | -- | -- |
| 37 | V1 saved OriginalPrompt | 0.6643 | 0.6305 | -- | -- | -- |
| 37 | V2 configured descriptions | 0.6840 | 0.6670 | 0.11041133105754852 | 0.6693142038583756 | 0 |
| 37 | V2 shuffled train descriptions | 0.6820 | 0.6648 | 0.11043606698513031 | 0.669343758225441 | 0 |
| 37 | NoPrompt | 0.6957 | 0.6781 | -- | -- | -- |

### Checkpoint identifiers
| Seed | Frozen GCN checkpoint | SHA-256 | Saved V1 prompt checkpoint | V1 best epoch |
|---:|---|---|---|---:|
| 11 | results\checkpoints\seed_11_frozen_gcn.pt | 17e5a2011bad5e4a1b0ab713a3b068012ae27e0b2a9ac743072af1d2532da4bd | results\checkpoints\shot_3_seed_11_prompt.pt | 6 |
| 23 | results\checkpoints\seed_23_frozen_gcn.pt | c801d87ce6e32434941db671bc002c7bb23813abb920d7660c44235f8390c350 | results\checkpoints\shot_3_seed_23_prompt.pt | 1 |
| 37 | results\checkpoints\seed_37_frozen_gcn.pt | 082acbf2a173514e34cdc6ad64ca2ef581621fe1df8d5eb9a5faaa9e998668cf | results\checkpoints\shot_3_seed_37_prompt.pt | 3 |

## Consistency decision
V2 is considered consistently better than a comparator only if it exceeds that comparator for both accuracy and Macro-F1 on every seed in this MVP. This is descriptive validation evidence only.
- V2 versus V1 saved OriginalPrompt: 2/3 seeds beat it on both metrics; consistent improvement: **False**.
- V2 versus NoPrompt: 0/3 seeds beat it on both metrics; consistent improvement: **False**.
- Configured versus shuffled descriptions: configured wins on both metrics for 2/3 seeds. Topic-name alignment is unverified, so no claim of correctly aligned semantic benefit is made.

## Command and stop point
```powershell
$py = "c:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/.venv/Scripts/python.exe"
Set-Location "C:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/LLMGP"
& $py experiments/run_v2_extended_validation.py
```
Validation is complete. No test evaluation was run, and no additional seeds or shot settings were started.
