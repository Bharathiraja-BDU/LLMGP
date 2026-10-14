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

from src.data_loader import verify_cora_dataset
from src.gnn_backbone import GCNBackbone
from src.llm_encoder import SciBERTEncoder
from src.revised_protocol import CLASS_DESCRIPTIONS, CLASS_SPLIT, PROTOCOL_SEED
from src.v2.prompt_models import NormalizedGatedPrompt, OriginalPrompt
from src.v2.validation import (
    episode_forward,
    evaluate_episodes,
    sample_episode,
    snapshot_v1_artifacts,
    validate_trainable_gradients,
)


SEED = 11
SHOT = 3
TRAIN_EPISODES = 20
VALIDATION_EPISODES = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
INITIAL_GATE = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models() -> tuple[dict[str, Any], torch.Tensor, GCNBackbone, dict[str, Any]]:
    info = verify_cora_dataset()
    labels = info["y"].cpu().numpy()
    encoder = SciBERTEncoder(model_name="allenai/scibert_scivocab_uncased", device=str(DEVICE))
    allowed_ids = (*CLASS_SPLIT["train"], *CLASS_SPLIT["validation"])
    descriptions = [CLASS_DESCRIPTIONS[class_id] for class_id in allowed_ids]
    encoded = encoder.encode_class_descriptions(descriptions, batch_size=len(descriptions))
    if any(parameter.requires_grad for parameter in encoder.model.parameters()):
        raise AssertionError("SciBERT is expected to remain frozen")
    class_embeddings = torch.zeros((7, 768), dtype=encoded.dtype)
    for offset, class_id in enumerate(allowed_ids):
        class_embeddings[class_id] = encoded[offset]
    class_embeddings = class_embeddings.to(DEVICE)

    checkpoint_path = ROOT / "results" / "checkpoints" / f"seed_{SEED}_frozen_gcn.pt"
    checkpoint_hash = __import__("hashlib").sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["pretraining"]["seed"]) != SEED:
        raise AssertionError("Unexpected v1 GCN checkpoint seed")
    backbone = GCNBackbone(1433, 256, 256)
    backbone.load_state_dict(checkpoint["model_state_dict"])
    backbone.to(DEVICE).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
        parameter.grad = None
    gcn_details = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "sha256": checkpoint_hash,
        "objective": checkpoint["pretraining"]["objective"],
        "seed": checkpoint["pretraining"]["seed"],
    }
    return {**info, "labels_array": labels}, class_embeddings, backbone, {
        "sci_bert_model": "allenai/scibert_scivocab_uncased",
        "sci_bert_trainable": False,
        "embedded_class_ids": list(allowed_ids),
        "v1_gcn": gcn_details,
        "topic_name_alignment_verified": False,
    }


def train_model(
    model: torch.nn.Module,
    model_name: str,
    reference_norm: float,
    info: dict[str, Any],
    class_embeddings: torch.Tensor,
    backbone: GCNBackbone,
    train_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    x = info["x"].to(DEVICE)
    edge_index = info["edge_index"].to(DEVICE)
    losses: list[float] = []
    zero_signal_episodes = 0
    parameter_grad_norms: list[float] = []

    for episode in train_episodes:
        loss, _, _, _ = episode_forward(
            episode,
            x,
            edge_index,
            class_embeddings,
            model,
            backbone,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is not None for parameter in backbone.parameters()):
            raise AssertionError("Frozen GCN received a gradient")
        if not validate_trainable_gradients(model):
            zero_signal_episodes += 1
            continue
        parameter_grad_norms.append(
            float(torch.sqrt(sum(parameter.grad.detach().pow(2).sum() for parameter in model.parameters())).item())
        )
        optimizer.step()
        losses.append(float(loss.detach().item()))

    if len(losses) == 0:
        raise RuntimeError(f"All training episodes had zero gradients for {model_name}")
    return {
        "model": model_name,
        "reference_node_norm_train_only": reference_norm,
        "training_episodes_requested": len(train_episodes),
        "training_episodes_with_updates": len(losses),
        "zero_signal_episodes_skipped": zero_signal_episodes,
        "training_loss_mean_updated_episodes": float(np.mean(losses)),
        "gradient_norm_mean": float(np.mean(parameter_grad_norms)),
        "gate_after_training": float(model.gate.item()) if isinstance(model, NormalizedGatedPrompt) else None,
    }


def main() -> None:
    started = time.perf_counter()
    torch.set_num_threads(1)
    before = snapshot_v1_artifacts(ROOT)
    info, class_embeddings, backbone, checkpoint_info = load_models()
    labels = info["labels_array"]

    # Reference scale uses training-class nodes only.
    train_node_mask = np.isin(labels, np.asarray(CLASS_SPLIT["train"], dtype=np.int64))
    reference_norm = float(info["x"][torch.from_numpy(train_node_mask)].norm(dim=-1).mean().item())
    if not np.isfinite(reference_norm) or reference_norm <= 0:
        raise AssertionError("Invalid train-only reference norm")

    train_episodes = [
        sample_episode(
            labels,
            "train",
            SHOT,
            PROTOCOL_SEED * 10_000_000 + SEED * 100_000 + SHOT * 1_000 + index,
        )
        for index in range(TRAIN_EPISODES)
    ]
    validation_episodes = [
        sample_episode(
            labels,
            "validation",
            SHOT,
            PROTOCOL_SEED * 10_000_000 + SEED * 100_000 + SHOT * 1_000 + 50_000 + index,
        )
        for index in range(VALIDATION_EPISODES)
    ]
    for episode in [*train_episodes, *validation_episodes]:
        if not set(episode["support_idx"].tolist()).isdisjoint(episode["query_idx"].tolist()):
            raise AssertionError("Support/query overlap in MVP episode set")
    if any(not set(episode["class_ids"].tolist()).issubset(CLASS_SPLIT["train"]) for episode in train_episodes):
        raise AssertionError("Training episode used a class outside the training split")
    if any(episode["class_ids"].tolist() != list(CLASS_SPLIT["validation"]) for episode in validation_episodes):
        raise AssertionError("Validation episode class IDs do not match [3,4]")

    torch.manual_seed(SEED * 1_000_000 + SHOT * 10_000)
    v1 = OriginalPrompt()
    v1_training = train_model(v1, "V1 OriginalPrompt", reference_norm, info, class_embeddings, backbone, train_episodes)

    torch.manual_seed(SEED * 1_000_000 + SHOT * 10_000 + 1)
    v2 = NormalizedGatedPrompt(reference_node_norm=reference_norm, initial_gate=INITIAL_GATE)
    v2_training = train_model(v2, "V2 NormalizedGatedPrompt", reference_norm, info, class_embeddings, backbone, train_episodes)

    x = info["x"].to(DEVICE)
    edges = info["edge_index"].to(DEVICE)
    v1_metrics = evaluate_episodes(
        validation_episodes, x, edges, class_embeddings, v1, backbone, prompt_mode="connected"
    )
    v2_metrics = evaluate_episodes(
        validation_episodes, x, edges, class_embeddings, v2, backbone, prompt_mode="connected"
    )
    no_prompt_metrics = evaluate_episodes(
        validation_episodes, x, edges, class_embeddings, None, backbone, prompt_mode="none"
    )

    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise AssertionError("GCN was unfrozen")
    if any(parameter.grad is not None for parameter in backbone.parameters()):
        raise AssertionError("GCN accumulated gradients")
    after = snapshot_v1_artifacts(ROOT)
    if before != after:
        raise AssertionError("V1 artifact hashes changed during MVP")

    result = {
        "status": "completed_validation_only",
        "runtime_seconds": time.perf_counter() - started,
        "device": str(DEVICE),
        "protocol_seed": PROTOCOL_SEED,
        "run_seed": SEED,
        "shot": SHOT,
        "training_episodes": TRAIN_EPISODES,
        "validation_episodes": VALIDATION_EPISODES,
        "validation_classes": list(CLASS_SPLIT["validation"]),
        "test_classes_accessed": False,
        "sci_bert": checkpoint_info["sci_bert_model"],
        "sci_bert_frozen": not checkpoint_info["sci_bert_trainable"],
        "sci_bert_checkpoint_note": "Used the pretrained Hugging Face model identifier/cache; no project-local SciBERT checkpoint file exists.",
        "embedded_class_ids": checkpoint_info["embedded_class_ids"],
        "class_description_alignment_verified": checkpoint_info["topic_name_alignment_verified"],
        "v1_gcn_checkpoint": checkpoint_info["v1_gcn"],
        "gcn_frozen": True,
        "reference_node_norm_train_only": reference_norm,
        "v1_training": v1_training,
        "v2_training": v2_training,
        "models": [
            {"model": "V1 OriginalPrompt", **v1_metrics, "gate": None, **v1_training},
            {"model": "V2 NormalizedGatedPrompt", **v2_metrics, "gate": float(v2.gate.item()), **v2_training},
            {"model": "NoPrompt", **no_prompt_metrics, "gate": None},
        ],
        "v1_artifacts_unchanged": True,
        "test_evaluation": "not run",
    }
    report_path = ROOT / "reports" / "v2_mvp_report.md"
    write_report(result, report_path)
    print(json.dumps(result, indent=2), flush=True)


def write_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# LLMGP-v2 Minimal Validation",
        "",
        "## Scope",
        "Minimal validation only: seed 11, 3-shot, 20 training episodes per trainable model, and 20 shared validation episodes from classes [3,4]. Test classes [5,6] were not evaluated. No full v2 matrix or test evaluation was run.",
        "",
        "## Checkpoints and leakage safeguards",
        f"- Reused frozen GCN: `{result['v1_gcn_checkpoint']['checkpoint']}`, SHA-256 `{result['v1_gcn_checkpoint']['sha256']}`; v1 objective `{result['v1_gcn_checkpoint']['objective']}`.",
        f"- SciBERT: `{result['sci_bert']}`, frozen. There is no project-local SciBERT checkpoint; the pretrained model identifier/cache was reused.",
        f"- Only permitted description embedding IDs were encoded: {result['embedded_class_ids']}. Numeric ID to topic-name alignment remains unverified from local raw Cora files.",
        f"- Train classes: {list(CLASS_SPLIT['train'])}; validation classes: {result['validation_classes']}; test classes were not scored or used for selection.",
        f"- Train-node-only reference feature norm: {result['reference_node_norm_train_only']:.6f}.",
        "- The same fixed validation episodes were used for V1, V2, and NoPrompt. Support/query indices were checked for disjointness.",
        "- Existing v1 artifact SHA-256 manifest was identical before and after the MVP.",
        "",
        "## Results",
        "| Model | Validation accuracy | Macro-F1 | Mean loss | Effective prompt norm | Gate | Training updates / skipped zero-signal episodes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in result["models"]:
        training = model.get("model")
        if training == "V1 OriginalPrompt":
            count = f"{result['v1_training']['training_episodes_with_updates']} / {result['v1_training']['zero_signal_episodes_skipped']}"
        elif training == "V2 NormalizedGatedPrompt":
            count = f"{result['v2_training']['training_episodes_with_updates']} / {result['v2_training']['zero_signal_episodes_skipped']}"
        else:
            count = "not trained"
        lines.append(
            f"| {model['model']} | {model['accuracy']:.4f} | {model['macro_f1']:.4f} | {model['loss']:.4f} | {model['effective_prompt_norm']:.4f} | {model['gate'] if model['gate'] is not None else '--'} | {count} |"
        )
    lines.extend(
        [
            "",
            "## Training diagnostics",
            f"- V1 training loss over updated episodes: {result['v1_training']['training_loss_mean_updated_episodes']:.4f}; mean projection gradient norm: {result['v1_training']['gradient_norm_mean']:.6f}.",
            f"- V2 training loss over updated episodes: {result['v2_training']['training_loss_mean_updated_episodes']:.4f}; mean joint projection/gate gradient norm: {result['v2_training']['gradient_norm_mean']:.6f}; initial gate {INITIAL_GATE:.3f}, final gate {result['v2_training']['gate_after_training']:.6f}.",
            "- Frozen GCN parameters remained non-trainable and accumulated no gradients. The test suite confirmed finite, nonzero projection and gate gradients on a reachable episode.",
            "",
            "## Runtime and command",
            "```powershell",
            "$py = \"c:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/.venv/Scripts/python.exe\"",
            "Set-Location \"C:/Users/User/Desktop/Papers to be submitted/LLMGPv0.2/implementation/LLMGP\"",
            "& $py experiments/run_v2_mvp.py",
            "```",
            f"Runtime: {result['runtime_seconds']:.2f} seconds on {result['device']}.",
            "",
            "## Interpretation",
            "These are 20-episode validation estimates for a single seed and shot setting, not a complete statistical comparison. Prompt norm normalization and gating are implemented and gradients were verified. No claim about generalization, correctly aligned topic semantics, or held-out test performance is supported by this MVP.",
            "",
            "Stop point: validation-only MVP complete. No test evaluation was run.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
