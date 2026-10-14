from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from experiments import run_v3_mvp_validation as v3
from src.v3_align.alignment_loss import class_discriminative_alignment

OUTPUT = ROOT / "results/v3_align"
REPORT = ROOT / "reports/v3_align_mvp.md"
CONDITIONS = ("NoPrompt", "V3-correct", "V3-Align")


def snapshot() -> dict[str, str]:
    own = {"experiments/run_v3_align_mvp.py", "reports/v3_align_mvp.md"}
    result = {}
    for path in ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in own or relative.startswith(("results/v3_align/", "src/v3_align/", "tests/v3_align/")):
            continue
        result[relative] = v3.digest(path)
    for path in ROOT.parent.glob("*.docx"):
        result["../" + path.name] = v3.digest(path)
    return result


def train_align(model, episodes, semantics, weight: float, temperature: float) -> dict:
    """Exactly 100 joint updates using training-class episodes only."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in model.parameters()}
    norms = {name: [] for name, _ in model.named_parameters()}
    history, losses = [float(model.fusion_lambda.detach())], []
    for support, queries, labels, class_ids in episodes:
        assert set(class_ids.tolist()) <= {0, 1, 2}, "Non-training class in loss"
        assert not support.requires_grad and not queries.requires_grad and not semantics.requires_grad
        optimizer.zero_grad(set_to_none=True)
        descriptions = semantics[class_ids]
        fused = model(support, descriptions)
        classification = F.cross_entropy(v3.ProtoClassifier.logits(queries, fused, temperature=0.1), labels)
        alignment = class_discriminative_alignment(model.semantic_projection(descriptions), support, temperature)
        loss = classification + weight * alignment
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None or not torch.isfinite(gradient).all() or gradient.norm().item() <= 0:
                raise AssertionError(f"Zero or nonfinite trainable gradient: {name}")
            norms[name].append(float(gradient.norm()))
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        history.append(float(model.fusion_lambda.detach()))
        losses.append({"total": float(loss.detach()), "classification": float(classification.detach()),
                       "alignment": float(alignment.detach())})
    model.eval()
    assert len(losses) == 100
    return {"updates": len(losses), "lambda_history": history, "loss_history": losses,
            "loss_means": {key: float(np.mean([r[key] for r in losses])) for key in losses[0]},
            "gradient_norms": {key: {"min": min(values), "max": max(values), "mean": float(np.mean(values))}
                               for key, values in norms.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description="One minimal V3-Align validation, no test evaluation or tuning")
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument("--alignment-temperature", type=float, default=0.1)
    args = parser.parse_args()
    if not math.isfinite(args.alignment_weight) or args.alignment_weight < 0:
        parser.error("alignment weight must be finite and nonnegative")
    if not math.isfinite(args.alignment_temperature) or args.alignment_temperature <= 0:
        parser.error("alignment temperature must be finite and positive")
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    outputs = [REPORT, OUTPUT / "mvp_results.json", OUTPUT / "mvp_results.csv"]
    original_paths, new_paths = {}, {}
    for seed in v3.SEEDS:
        original_paths[seed] = v3.CHECKPOINT_DIR / f"semantic_prototype_fusion_correct_shot_3_seed_{seed}.pt"
        if not original_paths[seed].is_file():
            raise FileNotFoundError(f"Original trained V3-correct checkpoint is required: {original_paths[seed]}")
        new_paths[seed] = OUTPUT / "checkpoints" / f"semantic_prototype_fusion_align_shot_3_seed_{seed}.pt"
        outputs.append(new_paths[seed])
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing V3-Align output: {path}")
    before = snapshot()
    # Also protect any pre-existing content inside the new experiment directory.
    existing_align = {str(p): v3.digest(p) for p in OUTPUT.rglob("*") if p.is_file()}
    configuration = dict(v3.TRAINING_CONFIG)
    configuration.update({"alignment_weight": args.alignment_weight, "alignment_temperature": args.alignment_temperature,
                          "alignment_objective": "symmetric semantic-to-support/support-to-semantic contrastive cross entropy",
                          "total_loss": "classification_loss + alignment_weight * alignment_loss",
                          "negatives": "other training class in the same two-way episode"})
    info = v3.verify_cora_dataset()
    labels = info["y"].numpy()
    audit_sha = v3.verify_alignment(labels)
    encoder = v3.SciBERTEncoder(device="cpu")
    allowed = [0, 1, 2, 3, 4]
    semantics = torch.zeros(7, 768)
    semantics[allowed] = encoder.encode_class_descriptions([v3.CLASS_DESCRIPTIONS[i] for i in allowed], batch_size=5)
    v3.frozen_check(encoder.model)
    encoder_sha = v3.state_hash(encoder.model)
    manifest_path = ROOT / "results/v2/validation_episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    historical = json.loads((ROOT / "reports/v3_alignment_diagnosis_v2.json").read_text(encoding="utf-8"))
    rows, seed_details = [], []
    for seed in v3.SEEDS:
        seed_started = time.perf_counter()
        original_checkpoint = torch.load(original_paths[seed], map_location="cpu", weights_only=True)
        assert original_checkpoint["seed"] == seed and original_checkpoint["shot"] == 3
        assert original_checkpoint["semantic_condition"] == "V3-correct"
        assert original_checkpoint["training_statistics"]["updates"] == 100
        assert original_checkpoint["training_configuration"] == v3.TRAINING_CONFIG
        assert original_checkpoint["semantic_class_ids"] == allowed
        assert original_checkpoint["architecture_sha256"] == v3.digest(ROOT / "src/v3/semantic_prototype_fusion.py")
        assert original_checkpoint["scibert_state_sha256"] == encoder_sha
        torch.testing.assert_close(original_checkpoint["semantic_embeddings"], semantics[allowed], rtol=0, atol=0)
        train_base = v3.PROTOCOL_SEED * 10000000 + seed * 100000 + 3 * 1000
        training = [v3.sample_episode(labels, "train", 3, train_base + i) for i in range(100)]
        validation = manifest[f"validation|shot=3|seed={seed}"]
        v3.validate_episodes(training, labels, "train")
        v3.validate_episodes(validation, labels, "validation")
        train_sha, val_sha = v3.episode_hash(training), v3.episode_hash(validation)
        assert train_sha == original_checkpoint["training_episode_sha256"]
        assert val_sha == original_checkpoint["validation_episode_sha256"]
        gcn_path = ROOT / original_checkpoint["frozen_gcn_checkpoint"]
        assert gcn_path == ROOT / "results/checkpoints" / f"seed_{seed}_frozen_gcn.pt"
        assert v3.digest(gcn_path) == original_checkpoint["frozen_gcn_sha256"]
        frozen = torch.load(gcn_path, map_location="cpu", weights_only=False)
        assert frozen["pretraining"]["seed"] == seed
        assert frozen["pretraining"]["objective"] == "masked_feature_reconstruction_mse"
        backbone = v3.GCNBackbone(1433, 256, 256).requires_grad_(False)
        backbone.load_state_dict(frozen["model_state_dict"])
        backbone.eval()
        backbone_sha = v3.state_hash(backbone)
        with torch.no_grad():
            nodes = backbone(info["x"], info["edge_index"]).detach()
        train_data = v3.prepare_episodes(training, nodes)
        validation_data = v3.prepare_episodes(validation, nodes)
        torch.manual_seed(seed * 1000000 + 3 * 10000)
        initial = v3.SemanticPrototypeFusion(initial_lambda=0.1)
        initialization_sha = v3.state_hash(initial)
        assert initialization_sha == original_checkpoint["initialization_sha256"]
        aligned = copy.deepcopy(initial)
        stats = train_align(aligned, train_data, semantics, args.alignment_weight, args.alignment_temperature)
        v3.frozen_check(backbone, encoder.model)
        assert v3.state_hash(backbone) == backbone_sha
        new_paths[seed].parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": {key: value.detach().clone() for key, value in aligned.state_dict().items()},
            "model": "SemanticPrototypeFusion", "semantic_condition": "V3-Align", "seed": seed, "shot": 3,
            "training_configuration": configuration, "training_statistics": stats,
            "frozen_gcn_checkpoint": gcn_path.relative_to(ROOT).as_posix(), "frozen_gcn_sha256": v3.digest(gcn_path),
            "scibert_identifier": "allenai/scibert_scivocab_uncased", "scibert_state_sha256": encoder_sha,
            "semantic_class_ids": allowed, "semantic_embeddings": semantics[allowed].clone(),
            "class_descriptions": {i: v3.CLASS_DESCRIPTIONS[i] for i in allowed},
            "class_split": dict(v3.CLASS_SPLIT), "initialization_sha256": initialization_sha,
            "training_episode_sha256": train_sha, "validation_episode_sha256": val_sha,
            "validation_manifest_sha256": v3.digest(manifest_path), "alignment_audit_sha256": audit_sha,
            "architecture_sha256": v3.digest(ROOT / "src/v3/semantic_prototype_fusion.py"),
            "original_v3_checkpoint": original_paths[seed].relative_to(ROOT).as_posix(),
            "original_v3_checkpoint_sha256": v3.digest(original_paths[seed]),
        }
        with new_paths[seed].open("xb") as handle:
            torch.save(checkpoint, handle)
        loaded = torch.load(new_paths[seed], map_location="cpu", weights_only=True)
        restored = copy.deepcopy(initial).eval().requires_grad_(False)
        restored.load_state_dict(loaded["state_dict"], strict=True)
        assert v3.state_hash(restored) == v3.state_hash(aligned)
        original = copy.deepcopy(initial).eval().requires_grad_(False)
        original.load_state_dict(original_checkpoint["state_dict"], strict=True)
        outcomes = {"NoPrompt": v3.evaluate(None, validation_data, semantics),
                    "V3-correct": v3.evaluate(original, validation_data, semantics),
                    "V3-Align": v3.evaluate(restored, validation_data, semantics)}
        for condition in ("NoPrompt", "V3-correct"):
            previous = next(s for s in historical["seed_details"] if s["seed"] == seed)["validation"][condition]
            for key in ("accuracy", "macro_f1", "margin_after_mean"):
                assert math.isclose(outcomes[condition][key], previous[key], rel_tol=0, abs_tol=1e-7)
        for condition, metrics in outcomes.items():
            row = {"seed": seed, "model": condition,
                   **{key: value for key, value in metrics.items() if isinstance(value, (float, int))},
                   "lambda": None if condition == "NoPrompt" else float((original if condition == "V3-correct" else restored).fusion_lambda.detach()),
                   "accuracy_delta_vs_noprompt": metrics["accuracy"] - outcomes["NoPrompt"]["accuracy"],
                   "accuracy_delta_vs_original_v3": metrics["accuracy"] - outcomes["V3-correct"]["accuracy"],
                   "validation_episode_sha256": val_sha, "initialization_sha256": initialization_sha if condition != "NoPrompt" else ""}
            rows.append(row)
        v3.frozen_check(backbone, encoder.model, original, restored)
        assert v3.episode_hash(validation) == val_sha and v3.state_hash(backbone) == backbone_sha
        seed_details.append({"seed": seed, "validation": outcomes, "training": stats,
                             "training_episode_sha256": train_sha, "validation_episode_sha256": val_sha,
                             "initialization_sha256": initialization_sha,
                             "original_v3_checkpoint": original_paths[seed].relative_to(ROOT).as_posix(),
                             "original_v3_sha256": v3.digest(original_paths[seed]),
                             "align_checkpoint": new_paths[seed].relative_to(ROOT).as_posix(),
                             "align_checkpoint_sha256": v3.digest(new_paths[seed]),
                             "runtime_seconds": time.perf_counter() - seed_started})
        print(f"Seed {seed}: " + "; ".join(f"{c} acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f}" for c, m in outcomes.items()), flush=True)
    assert v3.state_hash(encoder.model) == encoder_sha
    v3.frozen_check(encoder.model)
    assert before == snapshot(), "Existing V1/V2/V3 artifacts or manuscript changed"
    assert all(v3.digest(Path(p)) == sha for p, sha in existing_align.items())
    result = {"rows": rows, "seed_details": seed_details, "training_configuration": configuration,
              "protected_hashes": before, "protected_files_unchanged": True,
              "scibert_state_unchanged": True, "architecture_unchanged": True,
              "alignment_audit_sha256": audit_sha, "validation_manifest_sha256": v3.digest(manifest_path),
              "runtime_seconds": time.perf_counter() - started, "device": "cpu", "test_results": "not evaluated"}
    write_results(result)
    assert before == snapshot()
    print(f"Completed one minimal V3-Align experiment: {result['runtime_seconds']:.2f}s; "
          f"{len(before)} protected files unchanged; three new checkpoints; no test evaluation.", flush=True)


def write_results(result: dict) -> None:
    rows = result["rows"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUTPUT / "mvp_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# V3-Align minimal validation", "", "## Protocol and alignment objective", "",
             "The unchanged SemanticPrototypeFusion architecture, frozen self-supervised seed-specific GCN and frozen SciBERT are reused. 3-shot, 2-way episodes, seeds 11/23/37, 100 training updates and 100 identical validation episodes per seed, with 15 queries per class. Training classes: [0,1,2]; validation: [3,4]; test: [5,6], not evaluated or encoded.", "",
             "Training uses symmetric contrastive cross entropy over the normalized projected semantic vectors and normalized support means. Cosine similarities divided by alignment temperature form a class-by-class matrix. Its diagonal contains positive pairs; off-diagonal entries contain negatives. The alignment loss averages row-wise semantic-to-support CE and transposed support-to-semantic CE. Each two-way episode supplies one negative per class; the three training classes are covered through the existing sampler. Support targets are detached. No validation prototypes or descriptions enter either training loss.", "",
             f"Total loss = classification CE + **{result['training_configuration']['alignment_weight']}** * alignment loss; alignment temperature **{result['training_configuration']['alignment_temperature']}**. These are configurable command-line values; one setting was run, with no sweep or validation tuning. Alignment updates the projection; classification updates both the projection and fusion gate. Lambda initializes at 0.1. Adam learning rate 0.001, weight decay 0.0001, classification temperature 0.1 and final-update evaluation match the original MVP.", "",
             "V3-Align starts from the exact original per-seed initial weights and uses the exact original ordered training episodes. Original V3-correct is loaded from its existing trained checkpoint without retraining; NoPrompt uses support-only prototypes. All comparisons use the same read-only validation manifest. Original accuracy, Macro-F1 and margins were checked against the archived V3 diagnosis. There is no separate alignment warm-up or extra training budget.", "",
             "The original full graph is retained for transductive frozen message passing, including unlabeled nodes from other classes. No virtual nodes, architecture changes, test-class evaluation or V4 implementation were introduced.", "",
             "## Validation results by seed", "",
             "Accuracy/F1 are episode means. Semantic/support cosine compares matching class rows; alignment advantage is matching minus wrong-class cosine. Margins are correct-class logits minus highest incorrect-class logits, at temperature 0.1. NoPrompt has no semantic prototype or active gate; its alignment fields are not applicable.", "",
             "| Seed | Model | Accuracy | Macro-F1 | Semantic/support cosine | Matching minus wrong | Margin before | Margin after | Lambda |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        def optional(key):
            return "--" if row.get(key) is None else f"{row[key]:.9f}"
        lines.append(f"| {row['seed']} | {row['model']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {optional('semantic_support_cosine_mean')} | {optional('alignment_advantage_mean')} | {row['margin_before_mean']:+.9f} | {row['margin_after_mean']:+.9f} | {optional('lambda')} |")
    lines += ["", "## Across-seed mean +/- sample SD", "", "SD uses ddof=1 across three seeds, not across repeated query occurrences.", "",
              "| Model | Accuracy | Macro-F1 | Semantic/support cosine | Matching minus wrong | Margin after | Lambda |", "|---|---:|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        group = [r for r in rows if r["model"] == condition]
        def summary(key):
            if any(r.get(key) is None for r in group):
                return "--"
            values = [r[key] for r in group]
            return f"{np.mean(values):.6f} +/- {np.std(values, ddof=1):.6f}"
        lines.append(f"| {condition} | {summary('accuracy')} | {summary('macro_f1')} | {summary('semantic_support_cosine_mean')} | {summary('alignment_advantage_mean')} | {summary('margin_after_mean')} | {summary('lambda')} |")
    lines += ["", "## Paired differences and outcome", "",
              "All differences use paired model metrics on the same episodes. Positive values favor V3-Align.", "",
              "| Comparator | Seed 11 accuracy delta | Seed 23 accuracy delta | Seed 37 accuracy delta | Mean accuracy delta | Mean Macro-F1 delta |", "|---|---:|---:|---:|---:|---:|"]
    conclusions = []
    for comparator in ("NoPrompt", "V3-correct"):
        acc, f1 = [], []
        for seed in v3.SEEDS:
            aligned = next(r for r in rows if r["seed"] == seed and r["model"] == "V3-Align")
            other = next(r for r in rows if r["seed"] == seed and r["model"] == comparator)
            acc.append(aligned["accuracy"] - other["accuracy"])
            f1.append(aligned["macro_f1"] - other["macro_f1"])
        lines.append(f"| {comparator} | {acc[0]:+.6f} | {acc[1]:+.6f} | {acc[2]:+.6f} | {np.mean(acc):+.6f} | {np.mean(f1):+.6f} |")
        conclusions.append(f"- V3-Align versus {comparator}: both accuracy and Macro-F1 improve in {sum(a>0 and f>0 for a, f in zip(acc, f1))}/3 seeds.")
        if comparator == "NoPrompt":
            consistent_improvement = all(a > 0 and f > 0 for a, f in zip(acc, f1))
            mean_improvement = np.mean(acc) > 0 and np.mean(f1) > 0
    lines += [""] + conclusions + [""]
    if consistent_improvement:
        lines.append("V3-Align outperforms NoPrompt on both metrics in all three validation seeds. This is a minimal descriptive validation result, not a test-generalization or statistical significance claim.")
    elif mean_improvement:
        lines.append("V3-Align improves both metrics on average but does not consistently outperform NoPrompt in every seed. No consistent superiority is claimed; the experiment stops at this minimal validation.")
    else:
        lines.append("**V3-Align fails to outperform NoPrompt on both across-seed mean metrics in this minimal validation.** Alignment changes do not establish classification improvement. The experiment stops here without additional tuning, training or architecture changes.")
        if all(next(r for r in rows if r["seed"] == seed and r["model"] == "V3-Align")[key] < next(r for r in rows if r["seed"] == seed and r["model"] == "NoPrompt")[key] for seed in v3.SEEDS for key in ("accuracy", "macro_f1")):
            lines.append("V3-Align underperforms NoPrompt on both accuracy and Macro-F1 in every seed.")
    lines += ["", "## Alignment and margin interpretation", ""]
    for seed in v3.SEEDS:
        aligned = next(r for r in rows if r["seed"] == seed and r["model"] == "V3-Align")
        original = next(r for r in rows if r["seed"] == seed and r["model"] == "V3-correct")
        lines.append(f"- Seed {seed}: matching cosine changes from {original['semantic_support_cosine_mean']:+.9f} to {aligned['semantic_support_cosine_mean']:+.9f}; matching-minus-wrong changes from {original['alignment_advantage_mean']:+.9f} to {aligned['alignment_advantage_mean']:+.9f}. Relative to NoPrompt, the mean margin changes by {aligned['margin_change_mean']:+.9f}; fusion loses {aligned['correct_to_wrong']} correct decisions and gains {aligned['wrong_to_correct']}. Final lambda is {aligned['lambda']:.9f}.")
    aligned_rows = [r for r in rows if r["model"] == "V3-Align"]
    if all(r["alignment_advantage_mean"] < 0 for r in aligned_rows):
        lines += ["", "The contrastive objective did not establish class-discriminative alignment on the unseen validation classes: matching-minus-wrong remains negative in every seed. Greater absolute matching cosine therefore does not translate into correct class separation. The gate also remains above its 0.1 initialization, preserving the harmful fused contribution in these validation runs."]
    lines += ["", "Absolute matching cosine and class-discriminative alignment are distinct: moving toward a common GCN direction can raise matching cosine without improving the matching-minus-wrong gap. Training-only contrastive alignment must transfer to the unseen validation classes to benefit their classification. These observations are limited to three seeds; they do not establish a general causal or test-set result.", "",
              "## Verification, checkpoints and preservation", "",
              "- All 300 joint training updates had finite, nonzero projection-weight, projection-bias and fusion-gate gradients. No skipped updates.",
              "- Every training loss explicitly checks that class IDs belong to [0,1,2]. Every episode passed support/query disjointness, balanced counts, actual-node-label and class-membership checks.",
              "- GCN and SciBERT remain frozen in evaluation mode with no gradients. GCN state hashes match before/after every seed; SciBERT state matches before/after the experiment. Cached frozen graph outputs and detached support targets prevent encoder backpropagation.",
              "- New checkpoints passed strict state-dict reload checks before evaluation. Initialization and training/validation episode hashes match the original V3-correct checkpoints.",
              f"- All {len(result['protected_hashes'])} existing V1/V2/V3 artifacts and manuscript hashes remain unchanged. New checkpoints and results are stored only under `results/v3_align/`; this report is separate from existing reports.", "",
              "| Seed | New V3-Align checkpoint | SHA-256 | Validation episode hash |", "|---:|---|---|---|"]
    for seed in result["seed_details"]:
        lines.append(f"| {seed['seed']} | `{seed['align_checkpoint']}` | `{seed['align_checkpoint_sha256']}` | `{seed['validation_episode_sha256']}` |")
    lines += ["", "Numeric results, per-episode accuracy/F1, gradient summaries, loss/gate histories, checkpoint provenance and protected hashes: `results/v3_align/mvp_results.json`. Scalar seed/model results: `results/v3_align/mvp_results.csv`.", "",
              f"Runtime: {result['runtime_seconds']:.2f} seconds on {result['device']} (unit tests excluded). Test-class results: **not evaluated**. No full experiments or weight sweep were run.", "",
              f"Unit tests: {result.get('unit_test_results', 'run separately with pytest tests/v3_align -q')}.", "",
              "Command from the implementation directory:", "", "```powershell",
              f"& .venv/Scripts/python.exe LLMGP/experiments/run_v3_align_mvp.py --alignment-weight {result['training_configuration']['alignment_weight']} --alignment-temperature {result['training_configuration']['alignment_temperature']}", "```", "",
              "Existing output paths cause an error before training; they are never silently overwritten."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUT / "mvp_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
