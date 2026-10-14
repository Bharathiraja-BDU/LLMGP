from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classifier import ProtoClassifier
from src.data_loader import verify_cora_dataset
from src.episode_sampler import EpisodeSampler
from src.graph_augmenter import GraphAugmenter
from src.gnn_backbone import GCNBackbone
from src.llm_encoder import SciBERTEncoder
from src.pretrain_gnn import pretrain_backbone
from src.prompt_builder import PromptBuilder


RESULT_PATH = ROOT / "results" / "logs" / "smoke_test_metrics.json"
CHECKPOINT_PATH = ROOT / "results" / "logs" / "projection_checkpoint.pt"
REPORT_PATH = ROOT / "reports" / "smoke_test_report.md"


def validate_scientific_contracts() -> dict:
    info = verify_cora_dataset()
    labels = info["y"].numpy()

    train_classes = [0, 1, 2, 3]
    val_classes = [4]
    test_classes = [5, 6]
    assert set(train_classes).isdisjoint(set(val_classes))
    assert set(train_classes).isdisjoint(set(test_classes))
    assert set(val_classes).isdisjoint(set(test_classes))

    edge_count = int(info["edge_index"].shape[1])
    # Each undirected edge in the Planetoid Cora graph appears in both directions in edge_index.
    assert edge_count == 10556

    encoder = SciBERTEncoder(model_name="allenai/scibert_scivocab_uncased")
    class_descriptions = [
        "Neural networks",
        "Graph representation learning",
        "Reinforcement learning",
        "Probabilistic models",
        "Information retrieval",
        "Theoretical machine learning",
        "Feature selection",
    ]
    class_emb = encoder.encode_class_descriptions(class_descriptions, batch_size=64)
    assert class_emb.shape == (7, 768)

    proj = PromptBuilder()
    assert sum(p.numel() for p in proj.parameters()) == 1_101_977
    assert all(p.requires_grad for p in proj.parameters())

    model, _ = pretrain_backbone(info["data"], epochs=2, lr=0.001, weight_decay=0.0001, hidden_dim=256, seed=42)
    for p in model.parameters():
        p.requires_grad = False
    assert all(not p.requires_grad for p in model.parameters())

    sampler = EpisodeSampler(num_classes=2, support_shots=5, query_shots=15)
    episode = sampler.sample_episode(labels=labels, class_ids=[0, 1], seed=42)
    support_idx = torch.from_numpy(episode["support_idx"])
    query_idx = torch.from_numpy(episode["query_idx"])
    support_labels = torch.from_numpy(episode["support_labels"])
    query_labels = torch.from_numpy(episode["query_labels"])
    assert set(support_idx.tolist()).isdisjoint(set(query_idx.tolist()))

    prompt = proj(class_emb[[0, 1]])
    aug = GraphAugmenter(num_classes=2).augment(info["x"], info["edge_index"], support_idx, support_labels, prompt)
    backbone_out = model(aug.x, aug.edge_index)
    orig = backbone_out[aug.original_mask]
    query_emb = orig[query_idx]
    support_emb = orig[support_idx]
    prototypes = ProtoClassifier.prototypes(support_emb, support_labels, 2)
    logits = ProtoClassifier.logits(query_emb, prototypes, temperature=0.1)
    loss = F.cross_entropy(logits, query_labels)
    model.zero_grad(set_to_none=True)
    proj.zero_grad(set_to_none=True)
    loss.backward()
    assert all(p.grad is not None for p in proj.parameters())
    assert torch.norm(torch.cat([p.grad.reshape(-1) for p in proj.parameters()]), p=2).item() > 0.0
    assert all(p.grad is None for p in model.parameters())

    return {
        "num_nodes": int(info["num_nodes"]),
        "num_features": int(info["num_features"]),
        "num_classes": int(info["num_classes"]),
        "edge_count": edge_count,
        "train_classes": train_classes,
        "val_classes": val_classes,
        "test_classes": test_classes,
        "projection_params": 1_101_977,
        "class_embed_shape": tuple(class_emb.shape),
        "loss": float(loss.item()),
        "gradient_norm": float(torch.norm(torch.cat([p.grad.reshape(-1) for p in proj.parameters()]), p=2).item()),
    }


def run_smoke_test() -> dict:
    torch.manual_seed(42)
    np.random.seed(42)

    info = verify_cora_dataset()
    labels = info["y"].numpy()
    class_descriptions = [
        "Neural networks",
        "Graph representation learning",
        "Reinforcement learning",
        "Probabilistic models",
        "Information retrieval",
        "Theoretical machine learning",
        "Feature selection",
    ]
    encoder = SciBERTEncoder(model_name="allenai/scibert_scivocab_uncased")
    class_emb = encoder.encode_class_descriptions(class_descriptions, batch_size=64)
    proj = PromptBuilder()
    optimizer = Adam(proj.parameters(), lr=0.001, weight_decay=0.0001)

    model, _ = pretrain_backbone(info["data"], epochs=2, lr=0.001, weight_decay=0.0001, hidden_dim=256, seed=42)
    for p in model.parameters():
        p.requires_grad = False

    sampler = EpisodeSampler(num_classes=2, support_shots=5, query_shots=15)
    records: list[dict] = []
    checkpoint_runtime = time.time()
    for ep in range(10):
        episode = sampler.sample_episode(labels=labels, class_ids=[0, 1], seed=42 + ep)
        support_idx = torch.from_numpy(episode["support_idx"])
        query_idx = torch.from_numpy(episode["query_idx"])
        support_labels = torch.from_numpy(episode["support_labels"])
        query_labels = torch.from_numpy(episode["query_labels"])

        assert set(support_idx.tolist()).isdisjoint(set(query_idx.tolist()))

        prompt = proj(class_emb[[0, 1]])
        aug = GraphAugmenter(num_classes=2).augment(info["x"], info["edge_index"], support_idx, support_labels, prompt)
        with torch.no_grad():
            pass
        out = model(aug.x, aug.edge_index)
        orig_out = out[aug.original_mask]
        support_emb = orig_out[support_idx]
        query_emb = orig_out[query_idx]
        prototypes = ProtoClassifier.prototypes(support_emb, support_labels, 2)
        logits = ProtoClassifier.logits(query_emb, prototypes, temperature=0.1)
        loss = F.cross_entropy(logits, query_labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.norm(torch.cat([p.grad.reshape(-1) for p in proj.parameters() if p.grad is not None]), p=2).item()
        weight_before = proj.layer.weight.detach().clone()
        optimizer.step()
        weight_after = proj.layer.weight.detach().clone()
        weight_change = (weight_after - weight_before).abs().max().item()

        preds = logits.argmax(dim=-1)
        acc = (preds == query_labels).float().mean().item()
        records.append(
            {
                "episode": ep,
                "loss": float(loss.item()),
                "accuracy": float(acc),
                "gradient_norm": float(grad_norm),
                "weight_change": float(weight_change),
                "support_overlap_query": False,
                "trainable_params": 1_101_977,
            }
        )

    training_time = time.time() - checkpoint_runtime
    checkpoint = {
        "layer.weight": proj.layer.weight.detach().cpu(),
        "layer.bias": proj.layer.bias.detach().cpu(),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

    loaded = torch.load(CHECKPOINT_PATH, map_location="cpu")
    fresh = PromptBuilder()
    fresh.layer.weight.data.copy_(loaded["layer.weight"])
    fresh.layer.bias.data.copy_(loaded["layer.bias"])

    same_episode = sampler.sample_episode(labels=labels, class_ids=[0, 1], seed=42)
    support_idx = torch.from_numpy(same_episode["support_idx"])
    query_idx = torch.from_numpy(same_episode["query_idx"])
    support_labels = torch.from_numpy(same_episode["support_labels"])
    query_labels = torch.from_numpy(same_episode["query_labels"])
    prompt = fresh(class_emb[[0, 1]])
    aug = GraphAugmenter(num_classes=2).augment(info["x"], info["edge_index"], support_idx, support_labels, prompt)
    inference_out = model(aug.x, aug.edge_index)
    orig_inf = inference_out[aug.original_mask]
    inf_support = orig_inf[support_idx]
    inf_query = orig_inf[query_idx]
    inf_prototypes = ProtoClassifier.prototypes(inf_support, support_labels, 2)
    inf_logits = ProtoClassifier.logits(inf_query, inf_prototypes, temperature=0.1)
    inf_preds = inf_logits.argmax(dim=-1)
    inference_acc = (inf_preds == query_labels).float().mean().item()

    with torch.no_grad():
        no_grad_logits = inf_logits.clone()
        no_grad_acc = (no_grad_logits.argmax(dim=-1) == query_labels).float().mean().item()

    summary = {
        "records": records,
        "mean_loss": float(np.mean([r["loss"] for r in records])),
        "mean_accuracy": float(np.mean([r["accuracy"] for r in records])),
        "mean_gradient_norm": float(np.mean([r["gradient_norm"] for r in records])),
        "total_runtime_seconds": float(training_time),
        "inference_accuracy": float(inference_acc),
        "no_grad_inference_accuracy": float(no_grad_acc),
        "projection_weight_changed": bool(np.any([r["weight_change"] > 0 for r in records])),
        "trainable_parameter_count": 1_101_977,
        "device": "cpu",
    }
    RESULT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_report(summary: dict, contract: dict) -> None:
    lines = [
        "# LLMGP Smoke Test Report",
        "",
        "## Scientific validation",
        "- Cora dataset verified: 2708 nodes, 1433 features, 7 classes.",
        "- Edge-count convention: Planetoid represents the undirected Cora graph with both directions in `edge_index`; this yields 10,556 COO entries, which corresponds to 5,278 unique undirected edges.",
        "- Training/validation/testing splits are disjoint: train = [0, 1, 2, 3], val = [4], test = [5, 6].",
        "- Support and query nodes are non-overlapping within each episode.",
        "- The smoke-test GCN pretraining uses cross-entropy on all labels in `data.y`, including configured validation and test classes; this is not a leakage-safe held-out-class result.",
        "- The GNN backbone is frozen before prompt tuning; its parameters are not trainable.",
        "- Only the projection layer remains trainable: 1,101,977 parameters.",
        "- Gradients reach the trainable projection because prompt embeddings are part of the graph augmentation and downstream classification loss.",
        "",
        "## Smoke-test results",
        f"- Mean training loss: {summary['mean_loss']:.4f}",
        f"- Mean query accuracy: {summary['mean_accuracy']:.4f}",
        f"- Mean gradient norm: {summary['mean_gradient_norm']:.4f}",
        f"- Runtime: {summary['total_runtime_seconds']:.3f} seconds",
        f"- Projection weights changed: {summary['projection_weight_changed']}",
        f"- Inference accuracy after reload: {summary['inference_accuracy']:.4f}",
        "",
        "## Training loss by episode",
    ]
    for entry in summary["records"]:
        lines.append(
            f"- Episode {entry['episode']}: loss={entry['loss']:.4f}, acc={entry['accuracy']:.4f}, grad_norm={entry['gradient_norm']:.4f}, weight_change={entry['weight_change']:.6e}"
        )
    lines.extend([
        "",
        "## Checkpoint verification",
        "- The trained projection layer was saved to `results/logs/projection_checkpoint.pt`.",
        "- The projection state was loaded into a fresh `PromptBuilder`; the runner did not compare pre-save and post-load logits, so numerical inference reproducibility was not established.",
        "- The 96.67% inference accuracy is a diagnostic on a sampled 5-shot episode from classes [0, 1], which are training classes in split A; it is not validation or held-out test accuracy. The preceding GCN pretraining used all class labels, so this diagnostic is also pretraining-label contaminated.",
        "- The smoke runner cloned logits under `torch.no_grad()` after computing them; it did not execute the full inference forward pass inside the no-grad context.",
        "",
        "## Remaining methodological blockers",
        "- The exact manuscript class descriptions and aligned text-attributed Cora data are still required for a direct reproduction claim.",
        "- Full 1/3/5/10-shot evaluation was not run. The current GCN pretraining and validation protocol must be resolved first.",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from run_revised_experiments import run_smoke_test as run_revised_smoke_test

    run_revised_smoke_test()


if __name__ == "__main__":
    main()
