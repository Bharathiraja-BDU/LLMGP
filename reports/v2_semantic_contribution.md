# LLMGP-v2 Semantic Contribution Experiment

## Protocol and safeguards
- 3-shot; seeds 11, 23, 37; 100 training episodes per V2 arm and seed; 100 validation episodes per seed, reused identically by V1, all V2 arms, and NoPrompt.
- V2 correct, shuffled, and random arms use identical episode indices, identical model initialization seeds per run, 100 training episodes, Adam, lr=0.001, and weight decay=0.0001. Validation labels were not used for training, checkpoint selection, or early stopping; final weights after the fixed budget were evaluated.
- NoPrompt is a frozen-GCN reference with no trainable prompt component, so it has no training budget by definition.
- The original V1 prompt checkpoint is loaded for each seed; frozen GCN checkpoints are reused. Existing checkpoint/result hashes remained unchanged.
- Validation classes [3,4] were scored. Test classes [5,6] were not accessed for scoring, training, or selection.
- Class-name mapping is count-verified against the authoritative LINQS `cora.content` source; no alphabetical ID assumption is used.
- Runtime: 164.48 seconds on CPU.

## Accuracy and Macro-F1
Across-seed intervals use Student t with df=2 and are necessarily wide for only three seeds.
| Condition | Accuracy mean +/- SD (95% CI) | Macro-F1 mean +/- SD (95% CI) | Mean gate | Effective prompt norm | Mean training loss |
|---|---|---|---:|---:|---:|
| V1 OriginalPrompt | 0.6296 +/- 0.1326 [0.3001, 0.9590] | 0.5692 +/- 0.1910 [0.0947, 1.0437] | -- | 66.1651 | -- |
| V2 correct | 0.6927 +/- 0.0245 [0.6318, 0.7535] | 0.6721 +/- 0.0248 [0.6106, 0.7337] | 0.11024701098601024 | 0.4620 | 0.67363722940286 |
| V2 shuffled | 0.6918 +/- 0.0264 [0.6262, 0.7573] | 0.6708 +/- 0.0264 [0.6053, 0.7363] | 0.11052973320086797 | 0.4632 | 0.6737344233194987 |
| V2 random | 0.6913 +/- 0.0351 [0.6041, 0.7785] | 0.6724 +/- 0.0359 [0.5833, 0.7615] | 0.10923523704210918 | 0.4578 | 0.6729071120421092 |
| NoPrompt | 0.7049 +/- 0.0195 [0.6563, 0.7534] | 0.6866 +/- 0.0181 [0.6416, 0.7316] | -- | 0.0000 | -- |

## Seed-wise results
| Seed | Condition | Accuracy | Macro-F1 | Gate | Training loss | Zero-gradient episodes |
|---:|---|---:|---:|---:|---:|---:|
| 11 | V1 OriginalPrompt | 0.7413 | 0.7220 | -- | -- | -- |
| 11 | V2 correct | 0.7200 | 0.6986 | 0.11041154712438583 | 0.6752034449577331 | 0 |
| 11 | V2 shuffled | 0.7220 | 0.7005 | 0.11068427562713623 | 0.6753247582912445 | 0 |
| 11 | V2 random | 0.7313 | 0.7126 | 0.10929454118013382 | 0.6744950973987579 | 0 |
| 11 | NoPrompt | 0.7273 | 0.7074 | -- | -- | -- |
| 23 | V1 OriginalPrompt | 0.4830 | 0.3551 | -- | -- | -- |
| 23 | V2 correct | 0.6727 | 0.6495 | 0.10988714545965195 | 0.6763776588439941 | 0 |
| 23 | V2 shuffled | 0.6733 | 0.6502 | 0.11018503457307816 | 0.6764521622657775 | 0 |
| 23 | V2 random | 0.6657 | 0.6437 | 0.10898784548044205 | 0.6755794781446457 | 0 |
| 23 | NoPrompt | 0.6917 | 0.6743 | -- | -- | -- |
| 37 | V1 OriginalPrompt | 0.6643 | 0.6305 | -- | -- | -- |
| 37 | V2 correct | 0.6853 | 0.6683 | 0.11044234037399292 | 0.6693305844068527 | 0 |
| 37 | V2 shuffled | 0.6800 | 0.6617 | 0.11071988940238953 | 0.669426349401474 | 0 |
| 37 | V2 random | 0.6770 | 0.6609 | 0.10942332446575165 | 0.6686467605829239 | 0 |
| 37 | NoPrompt | 0.6957 | 0.6781 | -- | -- | -- |

## Paired differences
Differences are V2-correct minus comparator, paired within seed and identical validation episodes.
| Comparator | Metric | Per-seed differences | Mean difference (95% CI) |
|---|---|---|---|
| V2 correct - V1 OriginalPrompt | accuracy | [-0.0213, 0.1897, 0.021] | 0.0631 [-0.2142, 0.3404] |
| V2 correct - V1 OriginalPrompt | macro_f1 | [-0.0234, 0.2945, 0.0378] | 0.1029 [-0.3160, 0.5219] |
| V2 correct - NoPrompt | accuracy | [-0.0073, -0.019, -0.0103] | -0.0122 [-0.0273, 0.0028] |
| V2 correct - NoPrompt | macro_f1 | [-0.0088, -0.0248, -0.0098] | -0.0145 [-0.0367, 0.0077] |
| V2 correct - V2 shuffled | accuracy | [-0.002, -0.0007, 0.0053] | 0.0009 [-0.0088, 0.0106] |
| V2 correct - V2 shuffled | macro_f1 | [-0.0019, -0.0007, 0.0066] | 0.0013 [-0.0101, 0.0127] |
| V2 correct - V2 random | accuracy | [-0.0113, 0.007, 0.0083] | 0.0013 [-0.0260, 0.0286] |
| V2 correct - V2 random | macro_f1 | [-0.014, 0.0058, 0.0074] | -0.0002 [-0.0298, 0.0293] |

## Interpretation
Configured versus shuffled/random differences are the semantic contribution test. Small or inconsistent differences indicate no demonstrated benefit from correct assignment at this scale; they do not establish equivalence. Numeric-ID/topic-name alignment is source-verified by unique per-class count matching but does not prove that the manually authored descriptions capture every paper-topic nuance.

## Exact command
```powershell
$py = "c:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/.venv/Scripts/python.exe"
Set-Location "C:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/LLMGP"
& $py experiments/run_v2_semantic_contribution.py
```
Validation only; no test evaluation and no full experiment.
