# Full Experiment Protocol Audit

> Historical pre-revision audit. The approved revised protocol and current experiment status are documented in [revised_protocol_experiment_report.md](revised_protocol_experiment_report.md).

## Status: halted before full training

Phase 1 identified unresolved scientific validity issues. No full few-shot training or 600-episode evaluation was launched. No manuscript-comparison results are available, and no final result CSV or accuracy figures were generated.

## Protocol verification

| Requirement | Finding | Status |
|---|---|---|
| Documented frozen-GNN pretraining procedure | `src/pretrain_gnn.py::pretrain_backbone` performs supervised cross-entropy training using `data.y`. The smoke runner invokes it for two epochs. | Procedure exists, but it is not leakage-safe for the configured held-out-class split. |
| GNN checkpoint | The manuscript says the test classes are unseen during GNN pretraining, but it does not specify the pretraining data/procedure or identify a checkpoint. `results/logs/` contains the prompt projection checkpoint only. | Not reproducible from available evidence |
| No held-out labels in GNN pretraining | The manuscript claims test classes are unseen during pretraining. The available `pretrain_backbone` implementation instead uses the full label vector, including classes 5 and 6. | Manuscript/code conflict; failed in current implementation |
| No held-out labels in prompt training/model selection | The smoke prompt episode uses classes 0 and 1; however, its frozen backbone was already trained using all labels. There is no implemented validation/model-selection loop. | Not established |
| Disjoint class split | Config A specifies train [0, 1, 2, 3], validation [4], and test [5, 6]. These sets are pairwise disjoint. | Passed structurally |
| Validation compatible with 2-way evaluation | The manuscript's data-preparation procedure gives 3/2/2 as an example to support 2-way episodes, but a later protocol passage says the experiments use 4/1/2 and use the single validation class for early stopping. The manuscript does not define a 2-way validation procedure for that single class, and no validation/model-selection code is implemented. | Manuscript ambiguity; blocks training |
| Meaning of 96.67% smoke accuracy | It is a sampled 5-shot diagnostic episode on classes [0, 1], the configured training classes. It is neither validation nor held-out test accuracy. The GNN pretraining used all classes, so this diagnostic is contaminated by held-out-label pretraining. | Clarified; not a test result |
| Support/query separation | `EpisodeSampler.sample_episode` samples disjoint node indices per class, and the smoke run checked this. | Passed for smoke episodes |
| Same protocol for all shot settings | No complete 1/3/5/10-shot runner exists, so consistency across settings has not been tested. | Unverified |

## Manuscript-source findings

The available manuscript is `LLMGP_v0.2.docx` in the implementation parent directory. Its data-preparation text says, “For example,” use three training, two validation, and two test classes to support 2-way episodes. A later experimental-protocol passage says the actual partition is four training, one validation, and two test classes, with the validation class used only for early stopping. These statements do not establish a single compatible validation protocol. The manuscript also says evaluation classes are unseen during GNN pretraining but gives no corresponding pretraining recipe or checkpoint details. Therefore, using split B and inventing a leakage-safe pretraining method would still be a methodological deviation, not a verified reproduction.

## Artifacts and scope

The smoke test completed its ten-episode 5-shot diagnostic. Reported mean loss was 0.5083, mean query accuracy 0.7533, and loop runtime 0.651 seconds. That runtime excludes model/text initialization and GNN pretraining. These figures are pipeline diagnostics only, not manuscript reproduction results.

`results/logs/projection_checkpoint.pt` stores the projection layer weights and bias. It is not a GNN checkpoint. Although the projection state was loaded into a fresh `PromptBuilder`, the smoke runner did not compare pre-save and post-load logits. Its reported 0.9667 accuracy therefore does not verify checkpoint inference reproducibility. The smoke runner also performed the model forward pass outside `torch.no_grad()`.

## Why execution stopped

Starting the requested 50-epoch training and 600-episode held-out evaluation with the current backbone would use test labels in pretraining. Choosing a new safe pretraining mask or inventing a validation strategy would silently change the experimental protocol. The run therefore stopped after protocol verification, before prompt training, evaluation, or manuscript comparison.

## Requested configuration not executed

- Dataset: Cora, Planetoid format (2,708 nodes, 1,433 features, 7 classes)
- SciBERT: `allenai/scibert_scivocab_uncased`, intended frozen
- GCN: intended pretrained and frozen; leakage-safe checkpoint unavailable
- Split A: train [0, 1, 2, 3], validation [4], test [5, 6]
- 2-way shots: 1, 3, 5, 10
- Training: maximum 50 epochs, 100 episodes per epoch, Adam, learning rate 0.001, weight decay 0.0001, temperature 0.1, patience 10
- Evaluation: 600 episodes per shot and seed, 15 query nodes per class, five seeds
- Confidence intervals: not calculated; no final evaluation sample exists

## Commands

Protocol evidence was inspected in `src/pretrain_gnn.py`, `configs/cora.yaml`, `src/episode_sampler.py`, `src/train.py`, and `experiments/run_smoke_test.py`. The smoke pipeline command was:

```powershell
$py = "c:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/.venv/Scripts/python.exe"
Set-Location "C:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/LLMGP"
& $py experiments/run_smoke_test.py
```

## Reproducibility limitations and next gate

The manuscript says class descriptions are manually authored and Cora uses titles/abstracts, but the implementation does not include a documented alignment source for those node texts or an exact verified class-description set. Before running the full experiment, establish and document the GNN pretraining protocol/checkpoint that excludes held-out test labels, resolve whether 3/2/2 or 4/1/2 is the intended split and how validation works, and identify the text inputs. Then verify that model selection does not inspect test labels and that every shot setting uses identical episode-generation and evaluation rules. Until those decisions are grounded in the manuscript or approved as explicit methodological deviations, all requested full-experiment metrics remain unverified.
