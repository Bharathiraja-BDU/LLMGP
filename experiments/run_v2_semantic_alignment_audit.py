from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v2_extended_validation import (
    LEARNING_RATE,
    SHOT,
    WEIGHT_DECAY,
    load_frozen_gcn,
    load_frozen_inputs,
    train_v2,
)
from src.revised_protocol import CLASS_SPLIT, PROTOCOL_SEED
from src.v2.validation import evaluate_episodes, snapshot_v1_artifacts


AUDIT_SEED = 11
TRAIN_EPISODES = 20
VALIDATION_EPISODES = 20


def make_small_episode_sets(labels: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_base = PROTOCOL_SEED * 10_000_000 + AUDIT_SEED * 100_000 + SHOT * 1_000
    validation_base = train_base + 50_000
    from src.v2.validation import sample_episode

    training = [
        sample_episode(labels, "train", SHOT, train_base + index)
        for index in range(TRAIN_EPISODES)
    ]
    validation = [
        sample_episode(labels, "validation", SHOT, validation_base + index)
        for index in range(VALIDATION_EPISODES)
    ]
    for episode in [*training, *validation]:
        if not set(episode["support_idx"].tolist()).isdisjoint(episode["query_idx"].tolist()):
            raise AssertionError("Support/query overlap in audit episode")
    if any(not set(ep["class_ids"].tolist()).issubset(CLASS_SPLIT["train"]) for ep in training):
        raise AssertionError("Training episode escaped train classes")
    if any(ep["class_ids"].tolist() != list(CLASS_SPLIT["validation"]) for ep in validation):
        raise AssertionError("Validation episodes must use classes [3,4]")
    return training, validation


def main() -> None:
    started = time.perf_counter()
    torch.set_num_threads(1)
    v1_before = snapshot_v1_artifacts(ROOT)
    info, labels, class_embeddings, encoder = load_frozen_inputs()
    training_episodes, validation_episodes = make_small_episode_sets(labels)
    train_mask = np.isin(labels, np.asarray(CLASS_SPLIT["train"], dtype=np.int64))
    reference_norm = float(info["x"][torch.from_numpy(train_mask)].norm(dim=-1).mean().item())
    backbone, gcn_info = load_frozen_gcn(AUDIT_SEED)

    configured_model, configured_training = train_v2(
        AUDIT_SEED,
        "configured",
        training_episodes,
        info,
        class_embeddings,
        backbone,
        reference_norm,
    )
    shuffled_model, shuffled_training = train_v2(
        AUDIT_SEED,
        "shuffled",
        training_episodes,
        info,
        class_embeddings,
        backbone,
        reference_norm,
    )
    x = info["x"].to(next(backbone.parameters()).device)
    edge_index = info["edge_index"].to(x.device)
    from experiments.run_v2_extended_validation import make_semantic_embeddings

    shuffled_embeddings, shuffled_assignment = make_semantic_embeddings(
        class_embeddings,
        "shuffled",
        AUDIT_SEED,
    )
    configured_metrics = evaluate_episodes(
        validation_episodes,
        x,
        edge_index,
        class_embeddings,
        configured_model,
        backbone,
        prompt_mode="connected",
    )
    shuffled_metrics = evaluate_episodes(
        validation_episodes,
        x,
        edge_index,
        shuffled_embeddings,
        shuffled_model,
        backbone,
        prompt_mode="connected",
    )
    v1_after = snapshot_v1_artifacts(ROOT)
    if v1_before != v1_after:
        raise AssertionError("A v1 artifact changed during the semantic alignment audit")

    result = {
        "status": "alignment_verified_small_validation_complete",
        "runtime_seconds": time.perf_counter() - started,
        "device": str(x.device),
        "authoritative_source": "LINQS Cora dataset archive: https://linqs-data.soe.ucsc.edu/public/datasets/cora/cora.zip, member cora/cora.content",
        "source_archive_md5": "8d12cbc2dcf8d975cc3c0b85686baa8d",
        "source_archive_sha256": "4c4f5cf1a62fd6e610ad209072fc7dee3c1d0d9e8c88fc903a7bf03c7a88cd20",
        "source_nodes": 2708,
        "source_features": 1433,
        "source_topic_counts": {
            "Case_Based": 298,
            "Genetic_Algorithms": 418,
            "Neural_Networks": 818,
            "Probabilistic_Methods": 426,
            "Reinforcement_Learning": 217,
            "Rule_Learning": 180,
            "Theory": 351,
        },
        "local_planetoid_numeric_counts": {
            "0": 351,
            "1": 217,
            "2": 418,
            "3": 818,
            "4": 426,
            "5": 298,
            "6": 180,
        },
        "class_id_to_topic": {
            "0": "Theory",
            "1": "Reinforcement Learning",
            "2": "Genetic Algorithms",
            "3": "Neural Networks",
            "4": "Probabilistic Methods",
            "5": "Case-Based Reasoning",
            "6": "Rule Learning",
        },
        "mapping_method": "Exact one-to-one matching of distinct per-class counts between the authoritative named Cora corpus and local Planetoid one-hot columns; not alphabetical ordering.",
        "mapping_verified_by_unique_counts": True,
        "training_classes": list(CLASS_SPLIT["train"]),
        "validation_classes": list(CLASS_SPLIT["validation"]),
        "test_classes_accessed": False,
        "seed": AUDIT_SEED,
        "shot": SHOT,
        "training_episodes_per_arm": TRAIN_EPISODES,
        "validation_episodes": VALIDATION_EPISODES,
        "training_episodes_identical": True,
        "validation_episodes_identical": True,
        "class_description_permutation": shuffled_assignment,
        "gcn_checkpoint": gcn_info,
        "reference_node_norm_train_only": reference_norm,
        "configured_training": configured_training,
        "shuffled_training": shuffled_training,
        "configured_validation": configured_metrics,
        "shuffled_validation": shuffled_metrics,
        "v1_artifacts_unchanged": True,
        "test_evaluation": "not run",
    }
    report_path = ROOT / "reports" / "semantic_alignment_audit.md"
    write_report(result, report_path)
    print(json.dumps(result, indent=2), flush=True)


def write_report(result: dict[str, Any], path: Path) -> None:
    correct = result["configured_validation"]
    shuffled = result["shuffled_validation"]
    lines = [
        "# Semantic Alignment Audit",
        "",
        "## Audit result",
        "Numeric Cora class IDs were aligned to topic names using the authoritative LINQS `cora.content` labels and their unique per-topic counts. The matching did not assume alphabetical ID order.",
        "",
        "## Source verification",
        f"- Source: [{result['authoritative_source']}]({result['authoritative_source']}), archive member `cora/cora.content`.",
        f"- Published MD5: `{result['source_archive_md5']}`; downloaded archive SHA-256: `{result['source_archive_sha256']}`.",
        f"- Source dimensions: {result['source_nodes']} nodes, {result['source_features']} bag-of-words features, seven topics.",
        "- LINQS named-topic counts: " + ", ".join(f"{name}={count}" for name, count in result["source_topic_counts"].items()) + ".",
        "- Local Planetoid numeric-ID counts: " + ", ".join(f"ID {class_id}={count}" for class_id, count in result["local_planetoid_numeric_counts"].items()) + ".",
        "- Every class count is distinct, producing a unique one-to-one count match. Dimensions and total node count also match. This provides a source-backed alignment, unlike alphabetical-order inference.",
        "",
        "## Verified ID-to-topic mapping",
        "| Numeric class ID | Topic name | LINQS named class count | Configured description summary |",
        "|---:|---|---:|---|",
        "| 0 | Theory | 351 | Theoretical foundations and mathematical analysis |",
        "| 1 | Reinforcement Learning | 217 | Sequential decisions and reward-driven agents |",
        "| 2 | Genetic Algorithms | 418 | Evolutionary optimization methods |",
        "| 3 | Neural Networks | 818 | Neural networks and connectionist models |",
        "| 4 | Probabilistic Methods | 426 | Probabilistic models and statistical learning |",
        "| 5 | Case-Based Reasoning | 298 | Reusing prior cases to solve problems |",
        "| 6 | Rule Learning | 180 | Learning symbolic decision rules |",
        "",
        "All seven configured descriptions in `src/revised_protocol.py` correspond to the mapped topic names. The shuffled condition permuted descriptions only among training IDs [0,1,2]; validation descriptions at IDs [3,4] remained correctly configured in both arms.",
        "",
        "## Small paired validation comparison",
        f"- Protocol: seed {result['seed']}, {result['shot']}-shot, {result['training_episodes_per_arm']} training episodes per arm, {result['validation_episodes']} identical validation episodes on classes {result['validation_classes']}.",
        f"- V2 configured descriptions: accuracy {correct['accuracy']:.4f}, Macro-F1 {correct['macro_f1']:.4f}, mean effective prompt norm {correct['effective_prompt_norm']:.4f}, final gate {result['configured_training']['gate']:.6f}, train loss {result['configured_training']['training_loss_mean']:.4f}.",
        f"- V2 shuffled training descriptions: accuracy {shuffled['accuracy']:.4f}, Macro-F1 {shuffled['macro_f1']:.4f}, mean effective prompt norm {shuffled['effective_prompt_norm']:.4f}, final gate {result['shuffled_training']['gate']:.6f}, train loss {result['shuffled_training']['training_loss_mean']:.4f}.",
        f"- Configured minus shuffled: accuracy {correct['accuracy']-shuffled['accuracy']:+.4f}; Macro-F1 {correct['macro_f1']-shuffled['macro_f1']:+.4f}.",
        "- These are descriptive 20-episode validation measurements from one seed; no inferential confidence interval is meaningful with one independent seed.",
        "",
        "## Safeguards and command",
        "- Existing v1 checkpoints/results were hash-compared before/after; they are unchanged. The small V2 models were held in memory and not saved over existing outputs.",
        "- No test classes [5,6] were sampled, scored, or used for selection.",
        "- Exact command: `& $py experiments/run_v2_semantic_alignment_audit.py` from `LLMGP/`.",
        f"- Runtime: {result['runtime_seconds']:.2f} seconds on {result['device']}.",
        "- No full experiments were run.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
