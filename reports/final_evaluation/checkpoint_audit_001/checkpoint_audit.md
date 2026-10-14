# Final LLMGP evaluation: checkpoint preflight

**Stopped before test evaluation: the requested comparison is incomplete.**

Requested models: NoPrompt, V1, V2-correct (NormalizedGatedPrompt/configured), V3-correct and V3-Align. Seeds: 11,23,37,53,71. Shots: 1,3,5,10. Established split: train [0,1,2], validation [3,4], test [5,6]. There are 100 logical model/seed/shot configurations (20 per model). No architecture, hyperparameter, checkpoint or manuscript was changed.

NoPrompt uses the existing trained seed-specific frozen self-supervised GCN; the same backbone legitimately supports all four shot counts. It has no additional prompt/projection/gate checkpoint. This is the existing support-only implementation, not a substitution for missing weights.

## Coverage

| Model | Required | Checkpoint available | Fully validated metadata/state | Missing | Metadata incomplete | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| NoPrompt | 20 | 20 | 20 | 0 | 0 | 0 |
| V1 | 20 | 20 | 20 | 0 | 0 | 0 |
| V2-correct | 20 | 4 | 0 | 16 | 4 | 0 |
| V3-correct | 20 | 3 | 3 | 17 | 0 | 0 |
| V3-Align | 20 | 3 | 3 | 17 | 0 | 0 |

## Missing configurations

| Model | Seed | Missing shots |
|---|---:|---|
| V2-correct | 23 | 1, 3, 5, 10 |
| V2-correct | 37 | 1, 3, 5, 10 |
| V2-correct | 53 | 1, 3, 5, 10 |
| V2-correct | 71 | 1, 3, 5, 10 |
| V3-correct | 11 | 1, 5, 10 |
| V3-correct | 23 | 1, 5, 10 |
| V3-correct | 37 | 1, 5, 10 |
| V3-correct | 53 | 1, 3, 5, 10 |
| V3-correct | 71 | 1, 3, 5, 10 |
| V3-Align | 11 | 1, 5, 10 |
| V3-Align | 23 | 1, 5, 10 |
| V3-Align | 37 | 1, 5, 10 |
| V3-Align | 53 | 1, 3, 5, 10 |
| V3-Align | 71 | 1, 3, 5, 10 |

The full 100-row matrix, expected paths, availability, statuses and checkpoint hashes are in `checkpoint_matrix.csv`. No missing checkpoint was replaced by another seed, shot, semantic arm, architecture or newly initialized model. No common-subset test comparison was run.

## Metadata and state verification

Existing checkpoints were loaded read-only with weights_only=True. State dictionaries were checked for finite tensors, exact architectural keys/shapes via strict loading, expected seed/shot/semantic condition and their seed-specific GCN association. No forward pass, optimizer, dataset loading, SciBERT encoding, episode sampling or test-class scoring was performed.

- GCN checks: matching seed, trained masked-feature-reconstruction provenance, no-label graph visibility, positive epoch count, finite final loss and 1433-to-256-to-256 state compatibility.
- V1 checks: explicit revised protocol and 3/2/2 class split, positive selected epoch, fixed optimizer/classification hyperparameters and matching backbone path.
- V2 checks: configured NormalizedGatedPrompt identity, seed/shot, positive selected epoch, matching backbone path, finite state and gate/state consistency. Its saved metadata lacks explicit protocol, class_split and training_configuration fields; the current runner source alone cannot certify those missing provenance fields. These are flagged as metadata_incomplete, not silently inferred or rewritten.
- V3 checks: explicit split and 100-update/final-update protocol, fixed optimizer, 2-way/15-query/shot configuration, semantic IDs 0-4, frozen GCN hash and unchanged architecture hash. V3-Align additionally verifies fixed alignment weight 1.0/temperature 0.1, original V3 hash and matching initialization, train/validation episode and SciBERT state identifiers.

### Metadata issues

| Model | Seed | Shot | Status | Issue |
|---|---:|---:|---|---|
| V2-correct | 11 | 1 | metadata_incomplete | Checkpoint lacks explicit protocol metadata: protocol, class_split, training_configuration; current runner source is not proof of the checkpoint's training provenance |
| V2-correct | 11 | 3 | metadata_incomplete | Checkpoint lacks explicit protocol metadata: protocol, class_split, training_configuration; current runner source is not proof of the checkpoint's training provenance |
| V2-correct | 11 | 5 | metadata_incomplete | Checkpoint lacks explicit protocol metadata: protocol, class_split, training_configuration; current runner source is not proof of the checkpoint's training provenance |
| V2-correct | 11 | 10 | metadata_incomplete | Checkpoint lacks explicit protocol metadata: protocol, class_split, training_configuration; current runner source is not proof of the checkpoint's training provenance |

### Training/selection protocol distinctions

V1 checkpoint metadata records validation-based epoch selection (maximum 50 epochs, 100 episodes per epoch, patience 10 and 30 selection validation episodes). V3-correct/V3-Align use exactly 100 updates and final-update evaluation without validation selection; V3-Align adds its fixed contrastive objective. Thus these artifacts do not constitute an equal-training-budget comparison. V2's full training configuration is not self-contained in its checkpoint metadata. These distinctions are recorded without changing settings or pooling results from different training runs. Matching class splits/backbones do not remove those training-budget/selection differences.

## Final test metrics

Accuracy: not computed. Macro-F1: not computed. Across-seed mean +/- SD: not computed. Paired differences: not computed. The coverage/provenance gate failed, so no final test episodes were generated or evaluated. Existing validation metrics or earlier result files were not relabeled as final test results.

## Preservation and command

All 231 existing artifact hashes are unchanged, including existing V1/V2/V3/V3-Align checkpoints, reports, architecture source and manuscript. The full SHA-256 manifest and audit details are in `checkpoint_audit.json`. Outputs are written to a newly created numbered audit directory; existing files are never overwritten.

Audit runtime: 2.01 seconds. Training/tuning: not run. Test evaluation: not run.

```powershell
& .venv/Scripts/python.exe LLMGP/experiments/audit_final_evaluation.py
```

Stopped at the availability/provenance audit as requested. No retraining, protocol substitution or manuscript change was performed.
