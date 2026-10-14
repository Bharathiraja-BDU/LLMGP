# Semantic Alignment Audit

## Audit result
Numeric Cora class IDs were aligned to topic names using the authoritative LINQS `cora.content` labels and their unique per-topic counts. The matching did not assume alphabetical ID order.

## Source verification
- Source: [LINQS Cora dataset archive: https://linqs-data.soe.ucsc.edu/public/datasets/cora/cora.zip, member cora/cora.content](LINQS Cora dataset archive: https://linqs-data.soe.ucsc.edu/public/datasets/cora/cora.zip, member cora/cora.content), archive member `cora/cora.content`.
- Published MD5: `8d12cbc2dcf8d975cc3c0b85686baa8d`; downloaded archive SHA-256: `4c4f5cf1a62fd6e610ad209072fc7dee3c1d0d9e8c88fc903a7bf03c7a88cd20`.
- Source dimensions: 2708 nodes, 1433 bag-of-words features, seven topics.
- LINQS named-topic counts: Case_Based=298, Genetic_Algorithms=418, Neural_Networks=818, Probabilistic_Methods=426, Reinforcement_Learning=217, Rule_Learning=180, Theory=351.
- Local Planetoid numeric-ID counts: ID 0=351, ID 1=217, ID 2=418, ID 3=818, ID 4=426, ID 5=298, ID 6=180.
- Every class count is distinct, producing a unique one-to-one count match. Dimensions and total node count also match. This provides a source-backed alignment, unlike alphabetical-order inference.

## Verified ID-to-topic mapping
| Numeric class ID | Topic name | LINQS named class count | Configured description summary |
|---:|---|---:|---|
| 0 | Theory | 351 | Theoretical foundations and mathematical analysis |
| 1 | Reinforcement Learning | 217 | Sequential decisions and reward-driven agents |
| 2 | Genetic Algorithms | 418 | Evolutionary optimization methods |
| 3 | Neural Networks | 818 | Neural networks and connectionist models |
| 4 | Probabilistic Methods | 426 | Probabilistic models and statistical learning |
| 5 | Case-Based Reasoning | 298 | Reusing prior cases to solve problems |
| 6 | Rule Learning | 180 | Learning symbolic decision rules |

All seven configured descriptions in `src/revised_protocol.py` correspond to the mapped topic names. The shuffled condition permuted descriptions only among training IDs [0,1,2]; validation descriptions at IDs [3,4] remained correctly configured in both arms.

## Small paired validation comparison
- Protocol: seed 11, 3-shot, 20 training episodes per arm, 20 identical validation episodes on classes [3, 4].
- V2 configured descriptions: accuracy 0.7000, Macro-F1 0.6703, mean effective prompt norm 0.4265, final gate 0.101769, train loss 0.6775.
- V2 shuffled training descriptions: accuracy 0.7050, Macro-F1 0.6761, mean effective prompt norm 0.4265, final gate 0.101786, train loss 0.6774.
- Configured minus shuffled: accuracy -0.0050; Macro-F1 -0.0058.
- These are descriptive 20-episode validation measurements from one seed; no inferential confidence interval is meaningful with one independent seed.

## Safeguards and command
- Existing v1 checkpoints/results were hash-compared before/after; they are unchanged. The small V2 models were held in memory and not saved over existing outputs.
- No test classes [5,6] were sampled, scored, or used for selection.
- Exact command: `& $py experiments/run_v2_semantic_alignment_audit.py` from `LLMGP/`.
- Runtime: 14.04 seconds on cpu.
- No full experiments were run.
