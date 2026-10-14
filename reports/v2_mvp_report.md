# LLMGP-v2 Minimal Validation

## Scope
Minimal validation only: seed 11, 3-shot, 20 training episodes per trainable model, and 20 shared validation episodes from classes [3,4]. Test classes [5,6] were not evaluated. No full v2 matrix or test evaluation was run.

## Checkpoints and leakage safeguards
- Reused frozen GCN: `results\checkpoints\seed_11_frozen_gcn.pt`, SHA-256 `17e5a2011bad5e4a1b0ab713a3b068012ae27e0b2a9ac743072af1d2532da4bd`; v1 objective `masked_feature_reconstruction_mse`.
- SciBERT: `allenai/scibert_scivocab_uncased`, frozen. There is no project-local SciBERT checkpoint; the pretrained model identifier/cache was reused.
- Only permitted description embedding IDs were encoded: [0, 1, 2, 3, 4]. Numeric ID to topic-name alignment remains unverified from local raw Cora files.
- Train classes: [0, 1, 2]; validation classes: [3, 4]; test classes were not scored or used for selection.
- Train-node-only reference feature norm: 4.190641.
- The same fixed validation episodes were used for V1, V2, and NoPrompt. Support/query indices were checked for disjointness.
- Existing v1 artifact SHA-256 manifest was identical before and after the MVP.

## Results
| Model | Validation accuracy | Macro-F1 | Mean loss | Effective prompt norm | Gate | Training updates / skipped zero-signal episodes |
|---|---:|---:|---:|---:|---:|---:|
| V1 OriginalPrompt | 0.6683 | 0.6220 | 0.6839 | 58.8230 | -- | 20 / 0 |
| V2 NormalizedGatedPrompt | 0.7000 | 0.6703 | 0.6815 | 0.4265 | 0.10177845507860184 | 20 / 0 |
| NoPrompt | 0.7167 | 0.6934 | 0.6822 | 0.0000 | -- | not trained |

## Training diagnostics
- V1 training loss over updated episodes: 0.6787; mean projection gradient norm: 0.044356.
- V2 training loss over updated episodes: 0.6774; mean joint projection/gate gradient norm: 0.001903; initial gate 0.100, final gate 0.101778.
- Frozen GCN parameters remained non-trainable and accumulated no gradients. The test suite confirmed finite, nonzero projection and gate gradients on a reachable episode.

## Runtime and command
```powershell
$py = "c:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/.venv/Scripts/python.exe"
Set-Location "C:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/LLMGP"
& $py experiments/run_v2_mvp.py
```
Runtime: 11.36 seconds on cpu.

## Interpretation
These are 20-episode validation estimates for a single seed and shot setting, not a complete statistical comparison. Prompt norm normalization and gating are implemented and gradients were verified. No claim about generalization, correctly aligned topic semantics, or held-out test performance is supported by this MVP.

Stop point: validation-only MVP complete. No test evaluation was run.
