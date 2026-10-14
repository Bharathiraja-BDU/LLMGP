# LLMGP Reproducibility Report

## 1. System information

- Operating system: Windows 11
- Python: 3.11.9
- Working directory: `LLMGP/`
- CPU: available in the local environment
- GPU: CUDA was checked for availability but not assumed for the reproducible baseline.

## 2. Software stack

- PyTorch 2.5.1
- PyTorch Geometric 2.6.1
- Hugging Face Transformers 4.45.2
- scikit-learn
- SciPy
- NumPy
- Pandas
- Matplotlib
- PyYAML

## 3. Dataset source and verification

The implementation loads the standard Cora dataset through `torch_geometric.datasets.Planetoid`; this yields the canonical Planetoid citation graph with 2,708 nodes, 1,433 features, and 7 classes.

Important caveat: the standard Planetoid Cora graph does not directly provide title/abstract text. The manuscript references the text-attributed Cora setting, but the raw Planetoid object contains only the bag-of-words representation and labels. Therefore, the implementation does not fabricate node titles or abstracts, and the text-dependent experiments are explicitly marked as requiring an aligned text-attributed Cora source for scientific validity.

## 4. Class splits and validation protocol

The manuscript states a 4/1/2 split and also presents an example 3/2/2 episode setup. These are inconsistent if the validation set is intended to support a conventional 2-way validation episode. The implementation therefore supports two explicit configurations:

- Configuration A: train = 4, validation = 1, test = 2.
- Configuration B: train = 3, validation = 2, test = 2.

Configuration A requires a validation strategy that does not assume two validation classes, while Configuration B supports conventional 2-way validation episodes. These configurations are reported separately and are not treated as interchangeable.

## 5. SciBERT configuration

- Model: `allenai/scibert_scivocab_uncased`
- Embedding dimension: 768
- Max tokens: 512
- Batch size: 64
- Encoder weights frozen
- `[CLS]` token used as the semantic summary representation

The project keeps class-level semantic prompts as the primary implementation. Per-node text embeddings are cached when available, but they are not fused into the frozen GNN input in the baseline manuscript-aligned setup.

## 6. Prompt projection layer

- Layer: `torch.nn.Linear(768, 1433)`
- Xavier uniform initialization
- Bias enabled
- No activation
- Adam optimizer
- Learning rate: 0.001
- Weight decay: 0.0001
- Trainable parameter count: 1,101,977

This matches the expected count: 768 × 1433 + 1433 = 1,101,977.

## 7. Frozen GNN backbone

The project includes a two-layer GCN with a 1433 → 256 hidden backbone and a final 256-dimensional embedding representation. The GNN parameters are frozen during prompt tuning, with the pretraining procedure documented as a scientific, reproducible option rather than a claimed manuscript match.

## 8. Training configuration

- Episodes per epoch: 100 (reference design)
- Maximum epochs: 50
- K-shot values: 1, 3, 5, 10
- Query per class: 15
- Temperature: 0.1
- Early stopping patience: 10

## 9. Evaluation status

This repository currently provides the operational code skeleton and reproducibility checks for the underlying LLMGP components. Full manuscript-accurate comparison to the reported numbers requires the exact class descriptions, aligned text-attributed Cora dataset, and the reproducible training checkpoint used by the source paper.

## 10. Manuscript claim classification

- Verified: Cora metadata matches the canonical Planetoid dataset.
- Partially verified: Prompt projection parameter count and graph augmentation structure align with the manuscript design.
- Not reproduced: the reported 1-shot/3-shot/5-shot/10-shot accuracies.
- Not testable with available inputs: any claim depending on the exact original class descriptions or aligned node text stored in a private or unavailable dataset.

## 11. Recommended manuscript corrections

1. Explain the inconsistency between the 4/1/2 split and the 3/2/2 example validation setup.
2. State explicitly whether validation is performed by class count or by episode definition.
3. Clarify whether text-attributed Cora is required for node-text alignment and whether this is a precondition for LLMGP text prompts.
4. Distinguish between class-level semantic prompting and any per-node text fusion extension.
5. Report the exact pretraining checkpoint, seed, and class visibility used for the frozen GNN.
