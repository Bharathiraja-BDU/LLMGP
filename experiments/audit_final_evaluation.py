"""Read-only checkpoint preflight; never train, encode descriptions or score nodes."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gnn_backbone import GCNBackbone
from src.prompt_builder import PromptBuilder
from src.v2.prompt_models import NormalizedGatedPrompt
from src.v3.semantic_prototype_fusion import SemanticPrototypeFusion

SEEDS = (11, 23, 37, 53, 71)
SHOTS = (1, 3, 5, 10)
MODELS = ("NoPrompt", "V1", "V2-correct", "V3-correct", "V3-Align")
SPLIT = {"train": [0, 1, 2], "validation": [3, 4], "test": [5, 6]}
PROTOCOL = "revised_self_supervised_3_2_2"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> dict[str, str]:
    files = {str(p): digest(p) for p in ROOT.rglob("*") if p.is_file()
             and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}
    files.update({str(p): digest(p) for p in ROOT.parent.glob("*.docx")})
    return files


def checkpoint_path(model: str, seed: int, shot: int) -> Path:
    if model == "NoPrompt":
        return ROOT / "results/checkpoints" / f"seed_{seed}_frozen_gcn.pt"
    if model == "V1":
        return ROOT / "results/checkpoints" / f"shot_{shot}_seed_{seed}_prompt.pt"
    if model == "V2-correct":
        return ROOT / "results/v2/checkpoints" / f"normalizedgatedprompt_configured_shot_{shot}_seed_{seed}.pt"
    if model == "V3-correct":
        return ROOT / "results/v3/checkpoints" / f"semantic_prototype_fusion_correct_shot_{shot}_seed_{seed}.pt"
    return ROOT / "results/v3_align/checkpoints" / f"semantic_prototype_fusion_align_shot_{shot}_seed_{seed}.pt"


def reference_path(value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    return (path if path.is_absolute() else ROOT / path).resolve()


def validate_state(model: torch.nn.Module, state: dict) -> None:
    assert state and all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in state.values()), "Nonfinite or missing state tensors"
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def audit_backbone(seed: int) -> dict:
    path = checkpoint_path("NoPrompt", seed, 1)
    result = {"path": path.relative_to(ROOT).as_posix(), "status": "missing", "issues": [], "sha256": ""}
    if not path.is_file():
        result["issues"].append("Frozen seed-specific GCN checkpoint missing")
        return result
    result["sha256"] = digest(path)
    try:
        c = torch.load(path, map_location="cpu", weights_only=True)
        m = c["pretraining"]
        assert m["seed"] == seed, "GCN seed mismatch"
        assert m["objective"] == "masked_feature_reconstruction_mse", "Wrong GCN training protocol"
        assert m["epochs"] > 0 and math.isfinite(m["final_loss"]), "No valid trained GCN provenance"
        assert m["hidden_dim"] == 256, "GCN dimensions mismatch"
        assert "no node labels" in m["graph_visibility"], "GCN supervision protocol not confirmed"
        validate_state(GCNBackbone(1433, 256, 256), c["model_state_dict"])
        result.update({"status": "valid", "metadata": {k: v for k, v in m.items() if k != "loss_history"}})
    except Exception as error:
        result["status"] = "invalid"
        result["issues"].append(f"{type(error).__name__}: {error}")
    return result


def audit_model(model: str, seed: int, shot: int, backbone: dict) -> dict:
    path = checkpoint_path(model, seed, shot)
    row = {"model": model, "seed": seed, "shot": shot, "checkpoint": path.relative_to(ROOT).as_posix(),
           "available": path.is_file(), "status": "missing", "sha256": "", "issues": [],
           "backbone": backbone["path"], "backbone_sha256": backbone["sha256"],
           "state_dict_valid": False, "protocol_metadata_complete": False, "metadata": {}}
    if not path.is_file():
        row["issues"].append("Requested model/seed/shot checkpoint missing")
        return row
    row["sha256"] = digest(path)
    if model == "NoPrompt":
        row.update({"status": backbone["status"], "issues": backbone["issues"],
                    "state_dict_valid": backbone["status"] == "valid",
                    "protocol_metadata_complete": backbone["status"] == "valid", "metadata": backbone.get("metadata", {})})
        return row
    try:
        assert backbone["status"] == "valid", "Associated frozen GCN is unavailable or invalid"
        c = torch.load(path, map_location="cpu", weights_only=True)
        m = c["metadata"] if model in ("V1", "V2-correct") else c
        assert m["seed"] == seed and m["shot"] == shot, "Seed/shot metadata mismatch"
        if model == "V1":
            validate_state(PromptBuilder(), c["prompt_state_dict"])
            row["state_dict_valid"] = True
            assert m["protocol"] == PROTOCOL, "V1 protocol mismatch"
            assert {k: list(v) for k, v in m["class_split"].items()} == SPLIT, "V1 split mismatch"
            assert reference_path(m["backbone_checkpoint"]) == (ROOT / backbone["path"]).resolve(), "V1 backbone association mismatch"
            assert 1 <= m["best_epoch"] <= m["max_epochs"], "Invalid V1 training/selection metadata"
            assert m["temperature"] == 0.1 and m["optimizer"] == "Adam", "V1 hyperparameter identity mismatch"
            assert m["learning_rate"] == 0.001 and m["weight_decay"] == 0.0001
            row["metadata"] = m
            row["protocol_metadata_complete"] = True
        elif model == "V2-correct":
            assert m["model"] == "NormalizedGatedPrompt" and m["semantic_variant"] == "configured", "V2 architecture or semantic condition mismatch"
            assert reference_path(m["v1_backbone_checkpoint"]) == (ROOT / backbone["path"]).resolve(), "V2 backbone association mismatch"
            module = NormalizedGatedPrompt(reference_node_norm=m["reference_node_norm_train_only"], initial_gate=m["initial_gate"])
            validate_state(module, c["state_dict"])
            row["state_dict_valid"] = True
            assert m["best_epoch"] > 0 and math.isfinite(m["best_validation_macro_f1"]), "Invalid V2 training provenance"
            assert math.isclose(float(module.gate.detach()), m["final_gate"], abs_tol=1e-7), "V2 gate metadata/state mismatch"
            row["metadata"] = m
            absent = [k for k in ("protocol", "class_split", "training_configuration") if k not in m]
            if absent:
                row["issues"].append("Checkpoint lacks explicit protocol metadata: " + ", ".join(absent)
                                     + "; current runner source is not proof of the checkpoint's training provenance")
            else:
                assert m["protocol"] == PROTOCOL
                assert {k: list(v) for k, v in m["class_split"].items()} == SPLIT
                row["protocol_metadata_complete"] = True
        else:
            assert m["model"] == "SemanticPrototypeFusion" and m["semantic_condition"] == model, "V3 semantic condition or architecture mismatch"
            assert {k: list(v) for k, v in m["class_split"].items()} == SPLIT, "V3 split mismatch"
            assert reference_path(m["frozen_gcn_checkpoint"]) == (ROOT / backbone["path"]).resolve(), "V3 backbone association mismatch"
            assert m["frozen_gcn_sha256"] == backbone["sha256"], "V3 frozen GCN hash mismatch"
            assert m["architecture_sha256"] == digest(ROOT / "src/v3/semantic_prototype_fusion.py"), "Saved/current V3 architecture mismatch"
            config = m["training_configuration"]
            assert config["support_shots"] == shot and config["episodes"] == 100
            assert config["way"] == 2 and config["query_shots"] == 15 and config["temperature"] == 0.1
            assert config["optimizer"] == "Adam" and config["learning_rate"] == 0.001 and config["weight_decay"] == 0.0001
            assert config["initial_lambda"] == 0.1 and config["selection"] == "final update only; no validation selection"
            assert m["training_statistics"]["updates"] == 100, "V3 training budget metadata mismatch"
            assert m["semantic_class_ids"] == [0, 1, 2, 3, 4], "Unexpected saved semantic class IDs"
            assert m["semantic_embeddings"].shape == (5, 768) and torch.isfinite(m["semantic_embeddings"]).all()
            assert m["scibert_identifier"] == "allenai/scibert_scivocab_uncased"
            validate_state(SemanticPrototypeFusion(), c["state_dict"])
            row["state_dict_valid"] = True
            if model == "V3-Align":
                assert config["alignment_weight"] == 1.0 and config["alignment_temperature"] == 0.1, "Frozen V3-Align loss configuration mismatch"
                original_path = checkpoint_path("V3-correct", seed, shot)
                assert reference_path(m["original_v3_checkpoint"]) == original_path.resolve()
                assert m["original_v3_checkpoint_sha256"] == digest(original_path), "V3-Align original checkpoint hash mismatch"
                original = torch.load(original_path, map_location="cpu", weights_only=True)
                for key in ("initialization_sha256", "training_episode_sha256", "validation_episode_sha256", "scibert_state_sha256"):
                    assert m[key] == original[key], f"V3/V3-Align provenance mismatch: {key}"
            row["metadata"] = {k: m[k] for k in ("model", "seed", "shot", "semantic_condition", "class_split", "training_configuration",
                                 "training_episode_sha256", "validation_episode_sha256", "architecture_sha256", "initialization_sha256")}
            row["protocol_metadata_complete"] = True
        row["status"] = "valid" if row["protocol_metadata_complete"] else "metadata_incomplete"
    except Exception as error:
        row["status"] = "invalid"
        row["issues"].append(f"{type(error).__name__}: {error}")
    return row


def main() -> None:
    started = time.perf_counter()
    torch.set_num_threads(1)
    before = snapshot()
    backbones = {seed: audit_backbone(seed) for seed in SEEDS}
    rows = [audit_model(model, seed, shot, backbones[seed]) for model in MODELS for seed in SEEDS for shot in SHOTS]
    complete = all(row["status"] == "valid" for row in rows)
    # This audit never scores a subset or launches a training/evaluation runner.
    # Complete coverage would require a separate final inference implementation.
    result = {"status": "audit_complete_no_evaluation" if complete else "stopped_incomplete_comparison",
              "comparison_complete": complete, "models": MODELS, "seeds": SEEDS, "shots": SHOTS,
              "class_split": SPLIT, "required_configurations": len(rows), "rows": rows,
              "backbones": backbones, "protected_hashes": before, "existing_artifacts_unchanged": True,
              "test_evaluation_run": False, "training_run": False, "tuning_run": False,
              "test_metrics": None, "runtime_seconds": time.perf_counter() - started}
    assert all(Path(p).is_file() and digest(Path(p)) == sha for p, sha in before.items()), "Existing artifact changed during audit"
    parent = ROOT / "reports/final_evaluation"
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        output = parent / f"checkpoint_audit_{index:03d}"
        try:
            output.mkdir()
            break
        except FileExistsError:
            index += 1
    write_results(result, output)
    assert all(digest(Path(p)) == sha for p, sha in before.items()), "Existing artifact changed while saving audit"
    print(json.dumps({"status": result["status"], "available_configurations": sum(r["available"] for r in rows),
                      "missing_configurations": sum(r["status"] == "missing" for r in rows),
                      "unverified_metadata": sum(r["status"] == "metadata_incomplete" for r in rows),
                      "invalid_configurations": sum(r["status"] == "invalid" for r in rows),
                      "output": output.relative_to(ROOT).as_posix(), "test_evaluation_run": False,
                      "protected_files": len(before)}, indent=2))


def write_results(result: dict, output: Path) -> None:
    rows = result["rows"]
    with (output / "checkpoint_matrix.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = ["model", "seed", "shot", "checkpoint", "available", "status", "sha256", "backbone", "backbone_sha256",
                  "state_dict_valid", "protocol_metadata_complete", "issues"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**{k: row[k] for k in fields}, "issues": "; ".join(row["issues"])} for row in rows)
    lines = ["# Final LLMGP evaluation: checkpoint preflight", "",
             "**Stopped before test evaluation: the requested comparison is incomplete.**", "",
             "Requested models: NoPrompt, V1, V2-correct (NormalizedGatedPrompt/configured), V3-correct and V3-Align. Seeds: 11,23,37,53,71. Shots: 1,3,5,10. Established split: train [0,1,2], validation [3,4], test [5,6]. There are 100 logical model/seed/shot configurations (20 per model). No architecture, hyperparameter, checkpoint or manuscript was changed.", "",
             "NoPrompt uses the existing trained seed-specific frozen self-supervised GCN; the same backbone legitimately supports all four shot counts. It has no additional prompt/projection/gate checkpoint. This is the existing support-only implementation, not a substitution for missing weights.", "",
             "## Coverage", "", "| Model | Required | Checkpoint available | Fully validated metadata/state | Missing | Metadata incomplete | Invalid |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        group = [r for r in rows if r["model"] == model]
        lines.append(f"| {model} | {len(group)} | {sum(r['available'] for r in group)} | {sum(r['status']=='valid' for r in group)} | {sum(r['status']=='missing' for r in group)} | {sum(r['status']=='metadata_incomplete' for r in group)} | {sum(r['status']=='invalid' for r in group)} |")
    lines += ["", "## Missing configurations", "", "| Model | Seed | Missing shots |", "|---|---:|---|"]
    for model in MODELS:
        for seed in SEEDS:
            missing = [str(r["shot"]) for r in rows if r["model"] == model and r["seed"] == seed and r["status"] == "missing"]
            if missing:
                lines.append(f"| {model} | {seed} | {', '.join(missing)} |")
    lines += ["", "The full 100-row matrix, expected paths, availability, statuses and checkpoint hashes are in `checkpoint_matrix.csv`. No missing checkpoint was replaced by another seed, shot, semantic arm, architecture or newly initialized model. No common-subset test comparison was run.", "",
              "## Metadata and state verification", "",
              "Existing checkpoints were loaded read-only with weights_only=True. State dictionaries were checked for finite tensors, exact architectural keys/shapes via strict loading, expected seed/shot/semantic condition and their seed-specific GCN association. No forward pass, optimizer, dataset loading, SciBERT encoding, episode sampling or test-class scoring was performed.", "",
              "- GCN checks: matching seed, trained masked-feature-reconstruction provenance, no-label graph visibility, positive epoch count, finite final loss and 1433-to-256-to-256 state compatibility.",
              "- V1 checks: explicit revised protocol and 3/2/2 class split, positive selected epoch, fixed optimizer/classification hyperparameters and matching backbone path.",
              "- V2 checks: configured NormalizedGatedPrompt identity, seed/shot, positive selected epoch, matching backbone path, finite state and gate/state consistency. Its saved metadata lacks explicit protocol, class_split and training_configuration fields; the current runner source alone cannot certify those missing provenance fields. These are flagged as metadata_incomplete, not silently inferred or rewritten.",
              "- V3 checks: explicit split and 100-update/final-update protocol, fixed optimizer, 2-way/15-query/shot configuration, semantic IDs 0-4, frozen GCN hash and unchanged architecture hash. V3-Align additionally verifies fixed alignment weight 1.0/temperature 0.1, original V3 hash and matching initialization, train/validation episode and SciBERT state identifiers.", "",
              "### Metadata issues", "", "| Model | Seed | Shot | Status | Issue |", "|---|---:|---:|---|---|"]
    for row in rows:
        if row["status"] not in ("valid", "missing"):
            lines.append(f"| {row['model']} | {row['seed']} | {row['shot']} | {row['status']} | {'; '.join(row['issues'])} |")
    lines += ["", "### Training/selection protocol distinctions", "",
              "V1 checkpoint metadata records validation-based epoch selection (maximum 50 epochs, 100 episodes per epoch, patience 10 and 30 selection validation episodes). V3-correct/V3-Align use exactly 100 updates and final-update evaluation without validation selection; V3-Align adds its fixed contrastive objective. Thus these artifacts do not constitute an equal-training-budget comparison. V2's full training configuration is not self-contained in its checkpoint metadata. These distinctions are recorded without changing settings or pooling results from different training runs. Matching class splits/backbones do not remove those training-budget/selection differences.", "",
              "## Final test metrics", "",
              "Accuracy: not computed. Macro-F1: not computed. Across-seed mean +/- SD: not computed. Paired differences: not computed. The coverage/provenance gate failed, so no final test episodes were generated or evaluated. Existing validation metrics or earlier result files were not relabeled as final test results.", "",
              "## Preservation and command", "",
              f"All {len(result['protected_hashes'])} existing artifact hashes are unchanged, including existing V1/V2/V3/V3-Align checkpoints, reports, architecture source and manuscript. The full SHA-256 manifest and audit details are in `checkpoint_audit.json`. Outputs are written to a newly created numbered audit directory; existing files are never overwritten.", "",
              f"Audit runtime: {result['runtime_seconds']:.2f} seconds. Training/tuning: not run. Test evaluation: not run.", "",
              "```powershell", "& .venv/Scripts/python.exe LLMGP/experiments/audit_final_evaluation.py", "```", "",
              "Stopped at the availability/provenance audit as requested. No retraining, protocol substitution or manuscript change was performed."]
    with (output / "checkpoint_audit.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with (output / "checkpoint_audit.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
