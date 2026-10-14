# Revised Protocol Experiment Report

## Status
Full revised-protocol runs completed successfully. These experiments use a methodological change from the manuscript's original 4/1/2 class split: a fixed 3/2/2 split and self-supervised masked-feature GCN pretraining. Results are not claimed to reproduce the manuscript.

## Protocol and class mapping
- Protocol: `revised_self_supervised_3_2_2`.
- Fixed protocol seed: 2026; independent run seeds: 11, 23, 37, 53, 71.
- Train class IDs: [0, 1, 2].
- Validation class IDs: [3, 4].
- Test class IDs: [5, 6].
- Split assignment was fixed before training and was not selected or adjusted using test accuracy.
- Cora class-ID mapping was checked against the canonical per-class node counts in the loaded Planetoid labels.
- Class 0: Theory; prompt description: Research on theoretical foundations and mathematical analysis of machine learning.
- Class 1: Reinforcement Learning; prompt description: Research on reinforcement learning, sequential decisions, and reward-driven agents.
- Class 2: Genetic Algorithms; prompt description: Research on genetic algorithms and evolutionary optimization methods.
- Class 3: Neural Networks; prompt description: Research on neural networks and connectionist machine learning models.
- Class 4: Probabilistic Methods; prompt description: Research on probabilistic models and statistical methods for machine learning.
- Class 5: Case-Based Reasoning; prompt description: Research on case-based reasoning and solving problems by reusing prior cases.
- Class 6: Rule Learning; prompt description: Research on rule learning and methods that infer symbolic decision rules.

## Self-supervised GCN pretraining
- Objective: denoising masked-feature reconstruction. For each epoch, independently mask 15% of observed nonzero bag-of-words entries, run the two-layer GCN on corrupted features and the graph, map the 256-dimensional node states back to 1,433 features, and minimize MSE only on masked entries.
- Graph visibility: all Cora nodes, bag-of-words features, and graph edges are visible to this transductive self-supervised objective. No node labels are read by the pretraining function, used in the objective, or used for checkpoint selection.
- Optimization: Adam, learning rate 0.001, weight decay 0.0001, 100 epochs, hidden size 256, mask probability 0.15.
- One frozen GCN checkpoint was trained and saved for each of the five run seeds; each seed's checkpoint is shared across that seed's four independently trained prompt models.
- SciBERT remains frozen. During prompt training only the 768-to-1,433 projection layer is optimized; the GCN is frozen while gradients propagate through it to the prompt nodes.

## Experimental configuration
- Dataset: Cora Planetoid, 2708 nodes, 1433 bag-of-words features, 7 classes.
- Task: 2-way episodes in train, validation, and test splits; 15 query nodes per class.
- Shots: 1, 3, 5, 10; maximum epochs 50; 100 training episodes per epoch; early stopping patience 10.
- Prompt optimizer: Adam, learning rate 0.001, weight decay 0.0001; cosine temperature 0.1.
- Validation: 30 fixed-seed 2-way episodes per epoch on classes [3, 4]; checkpoint selected by mean validation cross-entropy only.
- Test: 600 episodes per shot and seed on classes [5, 6]; test episodes were evaluated under `torch.no_grad()` with parameter-state equality checks.
- 95% confidence intervals: two-sided Student t intervals over the five independent seed-level mean metrics (df=4, t critical value from SciPy); standard deviation uses the sample estimator (ddof=1).

## Results
| Shot | Revised accuracy (mean +/- SD) | 95% CI | Macro-F1 (mean +/- SD) | Manuscript reference (original 4/1/2) | Absolute difference | Comparison status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.5377 +/- 0.0655 | [0.4564, 0.6189] | 0.4209 +/- 0.1010 | 0.6140 | 0.0763 | Different protocol; not reproduction |
| 3 | 0.6888 +/- 0.0898 | [0.5773, 0.8003] | 0.6485 +/- 0.1357 | 0.7120 | 0.0232 | Different protocol; not reproduction |
| 5 | 0.5814 +/- 0.1116 | [0.4429, 0.7199] | 0.4754 +/- 0.1632 | 0.7590 | 0.1776 | Different protocol; not reproduction |
| 10 | 0.5646 +/- 0.0295 | [0.5280, 0.6013] | 0.4581 +/- 0.0453 | 0.8130 | 0.2484 | Different protocol; not reproduction |

## Artifacts and commands
- GCN checkpoints: `results/checkpoints/seed_<seed>_frozen_gcn.pt`.
- Prompt checkpoints: `results/checkpoints/shot_<shot>_seed_<seed>_prompt.pt`.
- Pretraining logs: `results/logs/revised/seed_<seed>_pretraining.json`.
- Prompt training logs: `results/logs/revised/shot_<shot>_seed_<seed>_training.csv`.
- Raw predictions: `results/metrics/raw_predictions_seed_<seed>_shot_<shot>.csv`.
- Episode metrics: `results/metrics/episode_metrics.csv`.
- Seed-wise metrics: `results/metrics/seed_wise_results.csv`.
- Aggregate metrics and confidence intervals: `results/metrics/final_results.csv`.
- Figures: `results/figures/accuracy_comparison.png` and `results/figures/accuracy_trend.png`.
- Smoke command: `& $py experiments/run_revised_experiments.py --smoke-only`.
- Full command: `& $py experiments/run_revised_experiments.py --full`.
- Total full-run elapsed time, including model load: 3926.22 seconds on cpu.

## Interpretation and limitations
- The 3/2/2 class split resolves the manuscript's 4/1/2 versus 2-way-validation inconsistency for this approved revised study. It is an explicit protocol change.
- Self-supervised pretraining is also a methodological change; all graph features and edges are transductively visible, while labels are excluded from pretraining.
- The manuscript's accuracy values are shown only as reference points from its different original protocol. Absolute differences do not measure reproduction error.
- Class descriptions are explicit manually authored short descriptions of the mapped Cora topic labels; they are not asserted to be the manuscript's exact wording.
- Standard Planetoid Cora provides bag-of-words features but not aligned paper titles/abstracts in this repository; per-node text encoding is not part of these runs.
- The new metrics are verified only for the specified transductive revised protocol and should not be compared as a direct replication of the manuscript.
