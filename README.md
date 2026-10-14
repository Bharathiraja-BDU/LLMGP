# LLMGP

This repository contains a reproducibility-focused implementation of the LLMGP methodology for few-shot node classification on the Cora citation graph.

## Repository layout

- `src/` contains the model components.
- `tests/` contains unit tests for dataset loading, prompt construction, graph augmentation, and leakage checks.
- `experiments/` contains evaluation and ablation entry points.
- `results/` and `reports/` store logs, figures, and the reproducibility report.

## Key scientific notes

- The manuscript describes a frozen SciBERT encoder and a frozen GCN backbone with a trainable projection from 768-dimensional class embeddings to the 1,433-dimensional node feature space.
- The class split described in the manuscript is internally inconsistent: the paper states both a 4/1/2 split and a 3/2/2 example for episodic validation. The implementation supports both configurations explicitly and reports them separately.
- The primary implementation keeps class-level textual semantics and does not fuse per-node titles/abstracts into the GNN input unless an aligned text-attributed Cora source is available.

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q
```
