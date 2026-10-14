# LLMGP-v3 alignment diagnosis

**Status: stopped because trained V3 checkpoints are unavailable. No retraining or inference evaluation was performed.**

The requested alignment diagnosis cannot establish why V3 underperforms NoPrompt from the saved artifacts. The prior MVP created SemanticPrototypeFusion models in memory and discarded their trained projection weights and gate parameters when the process ended. It saved CSV/JSON metrics, not model state dictionaries. Existing GCN, V1 prompt and V2 prompt checkpoints cannot replace the missing V3 weights.

## Availability and intended scope

The filesystem scan inspected 51 checkpoint-format files (.pt/.pth/.ckpt/.safetensors) under LLMGP, including data caches and V1/V2 artifacts. No V3/fusion checkpoint candidates or V3 checkpoint directory were found.

| Seed | Required V3 arms | Trained V3 checkpoints | Planned validation |
|---:|---|---|---|
| 11 | correct, shuffled, random | unavailable | 3-shot, 100 identical episodes per arm; not run |
| 23 | correct, shuffled, random | unavailable | 3-shot, 100 identical episodes per arm; not run |
| 37 | correct, shuffled, random | unavailable | 3-shot, 100 identical episodes per arm; not run |

The intended split remains training [0,1,2], validation [3,4], test [5,6]. This diagnostic loaded neither the Cora graph nor SciBERT descriptions, sampled no episodes, instantiated no models, and accessed no test classes for evaluation. No training was attempted.

## Requested measurements

| Check | Result |
|---|---|
| Projected semantic versus support-prototype cosine similarity | unavailable: trained 768-to-256 projection weights are missing |
| Correct-class margin before/after fusion | unavailable: saved accuracy/F1 metrics do not contain logits or fused prototypes |
| Learned lambda and movement toward zero | historical final lambda is available below; no new inference or gate trajectory measurement |
| Correct/shuffled/random representation misalignment | unavailable: projected representations were not saved |
| Collapse of projected semantic directions | unavailable: trained projection and projected prototype vectors were not saved |

A checkpoint-backed margin comparison would use the correct-class logit minus the highest incorrect-class logit on the same query, before and after fusion. A collapse check would compare pairwise cosine similarity between projected class directions. Neither quantity can be recovered from episode accuracy, Macro-F1, initialization hashes or gradient norms.

## Recoverable historical gate evidence

The following are archived endpoint values from `v3_mvp_validation_details.json`, not measurements from a new evaluation. The MVP initialized lambda to 0.1. Every reported final lambda is higher than that initialization, so the endpoints moved away from zero. No per-update lambda history was saved; monotonic movement cannot be established.

| Seed | Historical condition | Initial lambda | Final lambda | Final minus initial |
|---:|---|---:|---:|---:|
| 11 | V3-correct | 0.100000 | 0.109784558 | +0.009784558 |
| 11 | V3-shuffled | 0.100000 | 0.109790891 | +0.009790891 |
| 11 | V3-random | 0.100000 | 0.109354496 | +0.009354496 |
| 23 | V3-correct | 0.100000 | 0.109579250 | +0.009579250 |
| 23 | V3-shuffled | 0.100000 | 0.109635964 | +0.009635964 |
| 23 | V3-random | 0.100000 | 0.109257005 | +0.009257005 |
| 37 | V3-correct | 0.100000 | 0.109892465 | +0.009892465 |
| 37 | V3-shuffled | 0.100000 | 0.109766662 | +0.009766662 |
| 37 | V3-random | 0.100000 | 0.109755874 | +0.009755874 |

Those endpoint values show that the gate did not suppress the semantic contribution toward zero during the prior training run. They do not establish representation misalignment, direction collapse, or a causal explanation for the validation loss of accuracy. These remain unmeasured.

## Evidence and preservation

- The existing MVP runner has no model-checkpoint save call and explicitly documents that models remain in memory.
- MVP runner SHA-256: `ff6a0f81836adcc3408154ad9ed4b5133acb03c2ba4ed646909c142284de05a7`.
- Historical metric JSON SHA-256: `8553e6b2ce628ee93057c613a823d28886c9b0b932ec8b9293bb5e31dbd5cdd7`.
- Existing artifacts hash-checked: 206. V1/V2/V3 architectures, checkpoints, results and manuscript remain unchanged.
- The checkpoint inventory is listed below for review. No checkpoint was overwritten.

```text
data/Cora/processed/data.pt
data/Cora/processed/pre_filter.pt
data/Cora/processed/pre_transform.pt
results/checkpoints/seed_11_frozen_gcn.pt
results/checkpoints/seed_23_frozen_gcn.pt
results/checkpoints/seed_37_frozen_gcn.pt
results/checkpoints/seed_53_frozen_gcn.pt
results/checkpoints/seed_71_frozen_gcn.pt
results/checkpoints/shot_10_seed_11_prompt.pt
results/checkpoints/shot_10_seed_23_prompt.pt
results/checkpoints/shot_10_seed_37_prompt.pt
results/checkpoints/shot_10_seed_53_prompt.pt
results/checkpoints/shot_10_seed_71_prompt.pt
results/checkpoints/shot_1_seed_11_prompt.pt
results/checkpoints/shot_1_seed_23_prompt.pt
results/checkpoints/shot_1_seed_37_prompt.pt
results/checkpoints/shot_1_seed_53_prompt.pt
results/checkpoints/shot_1_seed_71_prompt.pt
results/checkpoints/shot_3_seed_11_prompt.pt
results/checkpoints/shot_3_seed_23_prompt.pt
results/checkpoints/shot_3_seed_37_prompt.pt
results/checkpoints/shot_3_seed_53_prompt.pt
results/checkpoints/shot_3_seed_71_prompt.pt
results/checkpoints/shot_5_seed_11_prompt.pt
results/checkpoints/shot_5_seed_23_prompt.pt
results/checkpoints/shot_5_seed_37_prompt.pt
results/checkpoints/shot_5_seed_53_prompt.pt
results/checkpoints/shot_5_seed_71_prompt.pt
results/logs/projection_checkpoint.pt
results/revised_smoke/seed_2026_frozen_gcn.pt
results/revised_smoke/shot_1_seed_11_prompt.pt
results/v2/checkpoints/normalizedgatedprompt_configured_shot_10_seed_11.pt
results/v2/checkpoints/normalizedgatedprompt_configured_shot_1_seed_11.pt
results/v2/checkpoints/normalizedgatedprompt_configured_shot_3_seed_11.pt
results/v2/checkpoints/normalizedgatedprompt_configured_shot_5_seed_11.pt
results/v2/checkpoints/normalizedgatedprompt_random_shot_10_seed_11.pt
results/v2/checkpoints/normalizedgatedprompt_random_shot_1_seed_11.pt
results/v2/checkpoints/normalizedgatedprompt_random_shot_3_seed_11.pt
results/v2/checkpoints/normalizedgatedprompt_random_shot_5_seed_11.pt
results/v2/checkpoints/normalizedgatedprompt_shuffled_shot_10_seed_11.pt
results/v2/checkpoints/normalizedgatedprompt_shuffled_shot_1_seed_11.pt
results/v2/checkpoints/normalizedgatedprompt_shuffled_shot_3_seed_11.pt
results/v2/checkpoints/normalizedgatedprompt_shuffled_shot_5_seed_11.pt
results/v2/checkpoints/normalizedprompt_configured_shot_10_seed_11.pt
results/v2/checkpoints/normalizedprompt_configured_shot_1_seed_11.pt
results/v2/checkpoints/normalizedprompt_configured_shot_3_seed_11.pt
results/v2/checkpoints/normalizedprompt_configured_shot_5_seed_11.pt
results/v2/checkpoints/originalprompt_configured_shot_10_seed_11.pt
results/v2/checkpoints/originalprompt_configured_shot_1_seed_11.pt
results/v2/checkpoints/originalprompt_configured_shot_3_seed_11.pt
results/v2/checkpoints/originalprompt_configured_shot_5_seed_11.pt
```

## Command and stop point

```powershell
& .venv/Scripts/python.exe LLMGP/experiments/diagnose_v3_alignment.py
```

Preflight runtime: 3.11 seconds. New validation metrics and test results: not evaluated. The script checks availability, records the missing-checkpoint stop condition, and exits without training. If checkpoint candidates appear later, it fails closed for provenance inspection rather than assigning them to a seed or semantic arm by filename.
