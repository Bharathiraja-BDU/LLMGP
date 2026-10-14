"""One inference-only NoPrompt/V1 comparison; no existing files are changed."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
OUTPUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import f1_score

from experiments import audit_final_evaluation as audit
from src.classifier import ProtoClassifier
from src.data_loader import verify_cora_dataset
from src.gnn_backbone import GCNBackbone
from src.llm_encoder import SciBERTEncoder
from src.prompt_builder import PromptBuilder
from src.revised_experiment import confidence_interval, episode_loss_and_predictions, sample_split_episode
from src.revised_protocol import CLASS_DESCRIPTIONS, CLASS_SPLIT, PROTOCOL_NAME, validate_protocol

SEEDS = (11, 23, 37, 53, 71)
SHOTS = (1, 3, 5, 10)
MODELS = ("NoPrompt", "V1")
TEST_EPISODES = 600
QUERY_PER_CLASS = 15
TEMPERATURE = 0.1


def state_hash(model: torch.nn.Module) -> str:
    sha = hashlib.sha256()
    for name, value in model.state_dict().items():
        sha.update(name.encode())
        sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def frozen_check(*models: torch.nn.Module) -> None:
    for model in models:
        assert not model.training
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def serialize_episode(episode: dict, index: int, sampler_seed: int) -> dict:
    return {"episode": index, "sampler_seed": sampler_seed,
            **{key: np.asarray(value).tolist() for key, value in episode.items()}}


def validate_episode(episode: dict, labels: np.ndarray, shot: int) -> None:
    assert episode["class_ids"].tolist() == [5, 6]
    assert set(episode["support_idx"]).isdisjoint(episode["query_idx"])
    classes = np.asarray(episode["class_ids"])
    for index_key, label_key, count in (("support_idx", "support_labels", shot),
                                        ("query_idx", "query_labels", QUERY_PER_CLASS)):
        idx, local = np.asarray(episode[index_key]), np.asarray(episode[label_key])
        assert len(idx) == count * 2 and len(set(idx)) == len(idx)
        assert np.array_equal(np.bincount(local, minlength=2), [count, count])
        assert np.array_equal(labels[idx], classes[local]), "Test-node labels or remapping mismatch"


def metrics(labels: torch.Tensor, prediction: torch.Tensor) -> tuple[float, float]:
    return (float((prediction == labels).float().mean()),
            float(f1_score(labels.numpy(), prediction.numpy(), labels=[0, 1], average="macro", zero_division=0)))


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUTPUT / name).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float]) -> dict:
    mean, sd, low, high = confidence_interval(values, confidence=0.95)
    return {"mean": mean, "sample_sd": sd, "ci95_lower": low, "ci95_upper": high}


def main() -> None:
    names = ("checkpoint_audit.json", "test_episode_manifest.json", "episode_metrics.csv", "seed_metrics.csv",
             "paired_seed_differences.csv", "aggregate_metrics.csv", "paired_differences.csv",
             "test_predictions.npz", "results.json", "final_report.md")
    for name in names:
        if (OUTPUT / name).exists():
            raise FileExistsError(f"Refusing to overwrite or repeat a completed test evaluation: {OUTPUT / name}")
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    before = audit.snapshot()
    backbones = {seed: audit.audit_backbone(seed) for seed in SEEDS}
    checkpoint_rows = [audit.audit_model(model, seed, shot, backbones[seed])
                       for model in MODELS for seed in SEEDS for shot in SHOTS]
    assert len(checkpoint_rows) == 40 and all(row["status"] == "valid" for row in checkpoint_rows), "NoPrompt/V1 checkpoint preflight failed; no test evaluation allowed"
    with (OUTPUT / "checkpoint_audit.json").open("x", encoding="utf-8") as handle:
        json.dump({"required_configurations": 40, "all_valid": True, "rows": checkpoint_rows}, handle, indent=2)
    print("Preflight passed: all 40 NoPrompt/V1 seed/shot configurations validated; fixed 600-episode protocol.", flush=True)
    info = verify_cora_dataset()
    labels = info["y"].numpy()
    validate_protocol(labels)
    assert CLASS_SPLIT == {"train": (0, 1, 2), "validation": (3, 4), "test": (5, 6)}
    episodes, manifest_sets, set_hashes = {}, {}, {}
    for seed in SEEDS:
        for shot in SHOTS:
            group, serialized = [], []
            for index in range(TEST_EPISODES):
                sampler_seed = seed * 10000000 + 9000000 + shot * 100000 + index
                episode = sample_split_episode(labels, "test", shot, sampler_seed, query_shots=QUERY_PER_CLASS)
                validate_episode(episode, labels, shot)
                group.append(episode)
                serialized.append(serialize_episode(episode, index, sampler_seed))
            key = f"test|seed={seed}|shot={shot}"
            episodes[(seed, shot)] = group
            manifest_sets[key] = serialized
            set_hashes[key] = hashlib.sha256(json.dumps(serialized, sort_keys=True).encode()).hexdigest()
    manifest = {"protocol": PROTOCOL_NAME, "split": "test", "test_classes": [5, 6],
                "seeds": SEEDS, "shots": SHOTS, "episodes_per_seed_shot": TEST_EPISODES,
                "query_per_class": QUERY_PER_CLASS,
                "seed_rule": "seed * 10000000 + 9000000 + shot * 100000 + episode_index",
                "sampler": "src.revised_experiment.sample_split_episode; internal sampler seed XOR 0x5DEECE66D",
                "episode_set_sha256": set_hashes, "sets": manifest_sets}
    manifest_path = OUTPUT / "test_episode_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, separators=(",", ":"))
    encoder = SciBERTEncoder(model_name="allenai/scibert_scivocab_uncased", device="cpu")
    # Match the original V1 encoder batch and existing class descriptions exactly.
    semantic = encoder.encode_class_descriptions([CLASS_DESCRIPTIONS[i] for i in range(7)], batch_size=7)
    assert semantic.shape == (7, 768)
    frozen_check(encoder.model)
    encoder_sha = state_hash(encoder.model)
    episode_rows, seed_rows, paired_seed_rows, prediction_arrays, training_provenance = [], [], [], {}, []
    for seed in SEEDS:
        gcn_path = audit.checkpoint_path("NoPrompt", seed, 1)
        checkpoint = torch.load(gcn_path, map_location="cpu", weights_only=True)
        backbone = GCNBackbone(1433, 256, 256).eval().requires_grad_(False)
        backbone.load_state_dict(checkpoint["model_state_dict"], strict=True)
        backbone_sha = state_hash(backbone)
        with torch.inference_mode():
            original_nodes = backbone(info["x"], info["edge_index"])
        for shot in SHOTS:
            shot_started = time.perf_counter()
            prompt_path = audit.checkpoint_path("V1", seed, shot)
            c = torch.load(prompt_path, map_location="cpu", weights_only=True)
            prompt = PromptBuilder().eval().requires_grad_(False)
            prompt.load_state_dict(c["prompt_state_dict"], strict=True)
            prompt_sha = state_hash(prompt)
            metadata = c["metadata"]
            log_path = ROOT / "results/logs/revised" / f"shot_{shot}_seed_{seed}_training.csv"
            history = []
            if log_path.is_file():
                with log_path.open(newline="", encoding="utf-8") as handle:
                    history = list(csv.DictReader(handle))
            training_provenance.append({"seed": seed, "shot": shot, "best_epoch": metadata["best_epoch"],
                                        "logged_epochs": len(history) if history else None,
                                        "logged_training_episodes": len(history) * metadata["episodes_per_epoch"] if history else None,
                                        "metadata": metadata, "training_log": log_path.relative_to(ROOT).as_posix()})
            group_metrics = {model: {"accuracy": [], "macro_f1": []} for model in MODELS}
            group_predictions = {model: [] for model in MODELS}
            true_labels = []
            with torch.inference_mode():
                for index, episode in enumerate(episodes[(seed, shot)]):
                    support = torch.as_tensor(episode["support_idx"], dtype=torch.long)
                    query = torch.as_tensor(episode["query_idx"], dtype=torch.long)
                    support_labels = torch.as_tensor(episode["support_labels"], dtype=torch.long)
                    query_labels = torch.as_tensor(episode["query_labels"], dtype=torch.long)
                    prototypes = ProtoClassifier.prototypes(original_nodes[support], support_labels, 2)
                    baseline_logits = ProtoClassifier.logits(original_nodes[query], prototypes, temperature=TEMPERATURE)
                    _, v1_logits, v1_labels = episode_loss_and_predictions(
                        episode, semantic, prompt, backbone, info["x"], info["edge_index"], torch.device("cpu"))
                    assert torch.equal(v1_labels, query_labels)
                    assert torch.isfinite(baseline_logits).all() and torch.isfinite(v1_logits).all()
                    for model, logits in (("NoPrompt", baseline_logits), ("V1", v1_logits)):
                        prediction = logits.argmax(dim=-1)
                        accuracy, macro_f1 = metrics(query_labels, prediction)
                        group_metrics[model]["accuracy"].append(accuracy)
                        group_metrics[model]["macro_f1"].append(macro_f1)
                        group_predictions[model].append(prediction.numpy().astype(np.int8))
                        episode_rows.append({"seed": seed, "shot": shot, "episode": index, "model": model,
                                             "accuracy": accuracy, "macro_f1": macro_f1})
                    true_labels.append(query_labels.numpy().astype(np.int8))
            key = f"test|seed={seed}|shot={shot}"
            regenerated = [serialize_episode(ep, i, seed * 10000000 + 9000000 + shot * 100000 + i)
                           for i, ep in enumerate(episodes[(seed, shot)])]
            assert set_hashes[key] == hashlib.sha256(json.dumps(regenerated, sort_keys=True).encode()).hexdigest()
            for model in MODELS:
                seed_rows.append({"seed": seed, "shot": shot, "model": model, "episodes": TEST_EPISODES,
                                  "accuracy": float(np.mean(group_metrics[model]["accuracy"])),
                                  "macro_f1": float(np.mean(group_metrics[model]["macro_f1"])),
                                  "episode_set_sha256": set_hashes[key],
                                  "gcn_checkpoint_sha256": audit.digest(gcn_path),
                                  "prompt_checkpoint_sha256": audit.digest(prompt_path) if model == "V1" else ""})
                prediction_arrays[f"seed_{seed}_shot_{shot}_{model}"] = np.stack(group_predictions[model])
            prediction_arrays[f"seed_{seed}_shot_{shot}_labels"] = np.stack(true_labels)
            paired_seed_rows.append({"seed": seed, "shot": shot,
                                     "accuracy_difference_v1_minus_noprompt": float(np.mean(np.asarray(group_metrics["V1"]["accuracy"]) - group_metrics["NoPrompt"]["accuracy"])),
                                     "macro_f1_difference_v1_minus_noprompt": float(np.mean(np.asarray(group_metrics["V1"]["macro_f1"]) - group_metrics["NoPrompt"]["macro_f1"]))})
            frozen_check(backbone, prompt, encoder.model)
            assert state_hash(backbone) == backbone_sha and state_hash(prompt) == prompt_sha
            print(f"seed={seed}, shot={shot}, " + "; ".join(f"{model}: acc={np.mean(group_metrics[model]['accuracy']):.4f}, F1={np.mean(group_metrics[model]['macro_f1']):.4f}" for model in MODELS)
                  + f" ({time.perf_counter()-shot_started:.1f}s)", flush=True)
    assert state_hash(encoder.model) == encoder_sha
    assert all(audit.digest(Path(path)) == sha for path, sha in before.items()), "Existing artifact changed during inference"
    aggregate, paired = [], []
    for shot in SHOTS:
        for model in MODELS:
            group = [row for row in seed_rows if row["shot"] == shot and row["model"] == model]
            row = {"shot": shot, "model": model, "seeds": 5, "episodes_per_seed": TEST_EPISODES}
            for metric in ("accuracy", "macro_f1"):
                row.update({f"{metric}_{key}": value for key, value in summarize([r[metric] for r in group]).items()})
            aggregate.append(row)
        group = [row for row in paired_seed_rows if row["shot"] == shot]
        for metric in ("accuracy", "macro_f1"):
            values = [r[f"{metric}_difference_v1_minus_noprompt"] for r in group]
            paired.append({"shot": shot, "metric": metric, "direction": "V1 minus NoPrompt", "seeds": 5, **summarize(values)})
    result = {"status": "completed_inference_only_two_model_test", "protocol": PROTOCOL_NAME,
              "class_split": {k: list(v) for k, v in CLASS_SPLIT.items()}, "seeds": SEEDS, "shots": SHOTS,
              "models": MODELS, "episodes_per_seed_shot": TEST_EPISODES, "query_per_class": QUERY_PER_CLASS,
              "temperature": TEMPERATURE, "seed_metrics": seed_rows, "paired_seed_differences": paired_seed_rows,
              "aggregate_metrics": aggregate, "paired_differences": paired, "training_provenance": training_provenance,
              "checkpoint_audit": checkpoint_rows, "protected_hashes": before, "existing_artifacts_unchanged": True,
              "frozen_parameter_states_unchanged": True, "scibert_state_sha256": encoder_sha,
              "episode_manifest_sha256": audit.digest(manifest_path), "runtime_seconds": time.perf_counter()-started,
              "device": "cpu", "training_run": False, "tuning_run": False, "test_selection_run": False}
    write_csv("episode_metrics.csv", episode_rows)
    write_csv("seed_metrics.csv", seed_rows)
    write_csv("paired_seed_differences.csv", paired_seed_rows)
    write_csv("aggregate_metrics.csv", aggregate)
    write_csv("paired_differences.csv", paired)
    with (OUTPUT / "test_predictions.npz").open("xb") as handle:
        np.savez_compressed(handle, **prediction_arrays)
    with (OUTPUT / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    write_report(result)
    assert all(audit.digest(Path(path)) == sha for path, sha in before.items()), "Existing artifact changed while saving final results"
    print(f"Final two-model test complete: {result['runtime_seconds']:.2f}s, {len(before)} existing files unchanged. No additional experiments.", flush=True)


def write_report(result: dict) -> None:
    lines = ["# Final read-only NoPrompt versus V1 test evaluation", "", "## Fixed protocol", "",
             "Existing validated checkpoints only; five seeds 11/23/37/53/71 and shots 1/3/5/10. Train [0,1,2], validation [3,4], test [5,6] (Case-Based Reasoning / Rule Learning). Each seed/shot uses 600 reproducible 2-way test episodes with 15 queries per class. Both models receive the exact same support/query indices, labels and ordering. No episode has support/query overlap. Node IDs and local labels were checked against actual dataset labels, with unique indices and balanced support/query counts.", "",
             "The existing test protocol's seed rule is `seed * 10000000 + 9000000 + shot * 100000 + episode_index`. The unchanged `sample_split_episode` then XORs with `0x5DEECE66D` for its internal sampler. This reuses the established test sampling rule rather than choosing episodes based on performance. The full manifest and per-set hashes were fixed before inference.", "",
             "NoPrompt computes support-only cosine prototypes from the unaugmented frozen GCN graph. V1 uses its trained 768-to-1433 SciBERT projection, appends two prompt nodes and connects them only to matching support nodes, then classifies using GCN support prototypes. Temperature remains 0.1. The same seed-specific frozen GCN is used by both models. SciBERT uses the existing descriptions and original seven-description encoder batch. Query labels are used only for scoring, not graph augmentation or optimization.", "",
             "## Accuracy and Macro-F1", "",
             "Each seed value averages 600 episode metrics. Mean and sample SD (ddof=1) are computed across five seed values. Confidence intervals are two-sided 95% Student-t intervals, df=4: `mean +/- t(0.975,4)*sample_SD/sqrt(5)`. Intervals are not clipped to [0,1]. Repeated query nodes and episodes are not treated as independent seed replications.", "",
             "| Shot | Model | Accuracy mean +/- SD | Accuracy 95% CI | Macro-F1 mean +/- SD | Macro-F1 95% CI |", "|---:|---|---:|---|---:|---|"]
    for row in result["aggregate_metrics"]:
        lines.append(f"| {row['shot']} | {row['model']} | {row['accuracy_mean']:.6f} +/- {row['accuracy_sample_sd']:.6f} | [{row['accuracy_ci95_lower']:.6f}, {row['accuracy_ci95_upper']:.6f}] | {row['macro_f1_mean']:.6f} +/- {row['macro_f1_sample_sd']:.6f} | [{row['macro_f1_ci95_lower']:.6f}, {row['macro_f1_ci95_upper']:.6f}] |")
    lines += ["", "## Paired model differences", "",
              "Differences are V1 minus NoPrompt on the same episodes within each seed, then summarized across five paired seed differences. This uses the SD of paired differences, not an unpaired or episode-level confidence interval. Positive differences favor V1. Values are fractions (0.01 is one percentage point). Intervals are per metric/shot and are not adjusted for multiple comparisons.", "",
              "| Shot | Metric | Paired mean difference +/- sample SD | Paired 95% CI |", "|---:|---|---:|---|"]
    for row in result["paired_differences"]:
        lines.append(f"| {row['shot']} | {row['metric']} | {row['mean']:+.6f} +/- {row['sample_sd']:.6f} | [{row['ci95_lower']:+.6f}, {row['ci95_upper']:+.6f}] |")
    lines += ["", "## Seed-level results", "", "| Seed | Shot | Model | Accuracy | Macro-F1 |", "|---:|---:|---|---:|---:|"]
    for row in result["seed_metrics"]:
        lines.append(f"| {row['seed']} | {row['shot']} | {row['model']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} |")
    lines += ["", "## Training and checkpoint selection differ", "",
              "Both methods share the previously trained GCN: 100 epochs of masked-feature-reconstruction self-supervision, Adam lr=0.001/wd=0.0001, 15% feature masking. All Cora node features and graph edges, including unlabeled test nodes, were transductively visible. No node labels were used in GCN pretraining or GCN checkpoint selection. The original full graph remains visible during this inference run.", "",
              "NoPrompt has no prompt-training stage and no separately selected projection. V1 additionally trained its semantic projection on train classes [0,1,2] using episodic classification, with Adam lr=0.001/wd=0.0001, 100 episodes per epoch, maximum 50 epochs, patience 10, and 30 fixed validation episodes on [3,4]. The existing V1 checkpoints were selected by minimum validation cross-entropy. Best epochs and logged training lengths are listed below; they were not changed after test scoring.", "",
              "**This is not an equal-training-budget comparison.** V1 has additional supervised episodic training and validation-based projection selection; NoPrompt uses only the common pretrained backbone plus episode supports. Test results were not used for checkpoint selection, hyperparameter changes or follow-up experiments.", "",
              "| Seed | Shot | V1 selected epoch | Logged epochs run | Logged training episodes |", "|---:|---:|---:|---:|---:|"]
    for row in result["training_provenance"]:
        lines.append(f"| {row['seed']} | {row['shot']} | {row['best_epoch']} | {row['logged_epochs'] if row['logged_epochs'] is not None else '--'} | {row['logged_training_episodes'] if row['logged_training_episodes'] is not None else '--'} |")
    lines += ["", "## Exploratory results remain separate", "",
              "V2, V3 and V3-Align are not part of this test comparison. Their existing findings remain exploratory validation results in their original reports. No exploratory metric was pooled with, renamed as or substituted for a test result. The earlier five-model audit remains an incomplete-comparison audit; this report is the separately authorized NoPrompt/V1 final test only.", "",
              "The revised 3/2/2 transductive protocol differs from the manuscript's original protocol. This comparison does not claim manuscript reproduction or a new dataset split. Existing revised-protocol V1 test runs used the same test seed rule; this final run adds the explicitly paired NoPrompt comparison without changing checkpoint selection.", "",
              "## Integrity, artifacts and stop point", "",
              "- All 40 logical NoPrompt/V1 seed/shot configurations passed fresh checkpoint-state and protocol validation before any test scoring (five GCN files shared across shots and 20 V1 prompt files). Full identifiers, SHA-256 values and metadata are in `checkpoint_audit.json`.",
              "- SciBERT, GCN and V1 projection remain in eval mode, requires_grad=False, with no gradients. Their in-memory state hashes match before/after inference.",
              f"- All {len(result['protected_hashes'])} existing file hashes remain unchanged, including every checkpoint, script, result, report and manuscript included in the snapshot; runtime __pycache__/.pytest_cache files are excluded. No existing artifact was overwritten.",
              f"- Test episode manifest SHA-256: `{result['episode_manifest_sha256']}`.",
              "- `test_episode_manifest.json`: support/query indices, labels, sampler seeds and per-set hashes.",
              "- `episode_metrics.csv`, `seed_metrics.csv`, `aggregate_metrics.csv`: per-episode, per-seed and final metrics.",
              "- `paired_seed_differences.csv`, `paired_differences.csv`: paired seed differences and their confidence intervals.",
              "- `test_predictions.npz`: local class predictions and true labels (0=class 5, 1=class 6), shaped [600,30] per model/seed/shot; corresponding query IDs are in the manifest.",
              "- `results.json`: full numerical results, training provenance, frozen-state and protected-file hashes.", "",
              f"Runtime: {result['runtime_seconds']:.2f} seconds on {result['device']}. Exactly this one final paired test evaluation was run. No retraining, tuning, architecture change, test-based selection or additional experiment was performed.", "",
              "Command from the implementation directory:", "", "```powershell",
              "& .venv/Scripts/python.exe LLMGP/reports/final_evaluation/two_model_test/run_evaluation.py", "```", "",
              "The runner refuses existing output filenames before inference to prevent overwriting or accidental repeat evaluation. Work stops after reporting these results."]
    with (OUTPUT / "final_report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
