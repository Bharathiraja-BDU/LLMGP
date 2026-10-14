# LLMGP Smoke Test Report

> Historical pre-revision smoke-test record. Its metrics used the earlier all-label supervised GCN pretraining and are not results from the approved revised protocol.

## Scientific validation
- Cora dataset verified: 2708 nodes, 1433 features, 7 classes.
- Edge-count convention: Planetoid represents the undirected Cora graph with both directions in `edge_index`; this yields 10,556 COO entries, which corresponds to 5,278 unique undirected edges.
- Training/validation/testing splits are disjoint: train = [0, 1, 2, 3], val = [4], test = [5, 6].
- Support and query nodes are non-overlapping within each episode.
- The smoke-test GCN pretraining uses cross-entropy on all labels in `data.y`, including configured validation and test classes; this is not a leakage-safe held-out-class result.
- The GNN backbone is frozen before prompt tuning; its parameters are not trainable.
- Only the projection layer remains trainable: 1,101,977 parameters.
- Gradients reach the trainable projection because prompt embeddings are part of the graph augmentation and downstream classification loss.

## Smoke-test results
- Mean training loss: 0.5083
- Mean query accuracy: 0.7533
- Mean gradient norm: 0.8428
- Runtime: 0.651 seconds
- Projection weights changed: True
- Diagnostic inference accuracy after projection reload: 0.9667; this is from a sampled 5-shot episode on classes [0, 1], which are training classes in split A, and is not validation or held-out test accuracy. The preceding GCN pretraining used labels from all classes.

## Training loss by episode
- Episode 0: loss=0.6197, acc=0.5333, grad_norm=1.8616, weight_change=1.000000e-03
- Episode 1: loss=0.7924, acc=0.5333, grad_norm=3.0713, weight_change=1.001358e-03
- Episode 2: loss=0.4879, acc=0.9667, grad_norm=0.4473, weight_change=1.003604e-03
- Episode 3: loss=0.4959, acc=0.7333, grad_norm=0.5528, weight_change=1.006711e-03
- Episode 4: loss=0.5828, acc=0.5000, grad_norm=0.7892, weight_change=1.009803e-03
- Episode 5: loss=0.5497, acc=0.5667, grad_norm=0.6819, weight_change=1.008321e-03
- Episode 6: loss=0.4384, acc=0.8667, grad_norm=0.2064, weight_change=9.767581e-04
- Episode 7: loss=0.3290, acc=1.0000, grad_norm=0.2648, weight_change=9.638071e-04
- Episode 8: loss=0.3919, acc=0.9000, grad_norm=0.2117, weight_change=9.377152e-04
- Episode 9: loss=0.3956, acc=0.9333, grad_norm=0.3414, weight_change=9.566247e-04

## Checkpoint verification
- The trained projection layer was saved to `results/logs/projection_checkpoint.pt`.
- It was loaded into a fresh `PromptBuilder`, but pre-save and post-load logits were not compared; numerical inference reproducibility was not established.
- The smoke runner cloned logits under `torch.no_grad()` after computing them; the full inference forward pass was not run inside the no-grad context.

## Remaining methodological blockers
- The exact manuscript class descriptions and aligned text-attributed Cora data are still required for a direct reproduction claim.
- Full 1/3/5/10-shot evaluation was not run. The current GCN pretraining and validation protocol must be resolved first.
