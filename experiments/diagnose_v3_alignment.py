"""Check checkpoint availability and document the V3 alignment stop condition.

The previous MVP saved metrics, not trained model weights. This preflight
never trains or substitutes freshly initialized models for the missing ones.
It deliberately performs no model evaluation when checkpoints are absent.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/v3_alignment_diagnosis.md"
SEEDS = (11, 23, 37)
CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".safetensors"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_snapshot() -> dict[str, str]:
    files = {}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == REPORT:
            continue
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        files[path.relative_to(ROOT).as_posix()] = sha256(path)
    for path in ROOT.parent.glob("*.docx"):
        files["../" + path.name] = sha256(path)
    return files


def main() -> None:
    started = time.perf_counter()
    before = artifact_snapshot()
    inventory = sorted(path for path in ROOT.rglob("*")
                       if path.is_file() and path.suffix.lower() in CHECKPOINT_EXTENSIONS)
    candidates = []
    for path in inventory:
        name = path.relative_to(ROOT).as_posix().lower()
        if any(marker in name for marker in ("v3", "semantic_prototype", "fusion")):
            candidates.append(path)
    if candidates:
        # Fail closed if availability changes: this preflight cannot infer a
        # checkpoint schema, arm/seed metadata, or training provenance by name.
        raise RuntimeError(
            "V3 checkpoint candidates are now present. Inspect their state and "
            "training metadata before implementing checkpoint-backed inference: "
            + ", ".join(str(path.relative_to(ROOT)) for path in candidates)
        )

    runner = ROOT / "experiments/run_v3_mvp_validation.py"
    source = runner.read_text(encoding="utf-8")
    if "torch.save(" in source or "Models remain in memory" not in source:
        raise RuntimeError("MVP persistence behavior changed; recheck checkpoint availability")
    historical_path = ROOT / "reports/v3_mvp_validation_details.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    rows = historical["rows"]
    lines = [
        "# LLMGP-v3 alignment diagnosis", "",
        "**Status: stopped because trained V3 checkpoints are unavailable. No retraining or inference evaluation was performed.**", "",
        "The requested alignment diagnosis cannot establish why V3 underperforms NoPrompt from the saved artifacts. The prior MVP created SemanticPrototypeFusion models in memory and discarded their trained projection weights and gate parameters when the process ended. It saved CSV/JSON metrics, not model state dictionaries. Existing GCN, V1 prompt and V2 prompt checkpoints cannot replace the missing V3 weights.", "",
        "## Availability and intended scope", "",
        f"The filesystem scan inspected {len(inventory)} checkpoint-format files (.pt/.pth/.ckpt/.safetensors) under LLMGP, including data caches and V1/V2 artifacts. No V3/fusion checkpoint candidates or V3 checkpoint directory were found.", "",
        "| Seed | Required V3 arms | Trained V3 checkpoints | Planned validation |", "|---:|---|---|---|",
    ]
    for seed in SEEDS:
        lines.append(f"| {seed} | correct, shuffled, random | unavailable | 3-shot, 100 identical episodes per arm; not run |")
    lines += ["", "The intended split remains training [0,1,2], validation [3,4], test [5,6]. This diagnostic loaded neither the Cora graph nor SciBERT descriptions, sampled no episodes, instantiated no models, and accessed no test classes for evaluation. No training was attempted.", "",
              "## Requested measurements", "",
              "| Check | Result |", "|---|---|",
              "| Projected semantic versus support-prototype cosine similarity | unavailable: trained 768-to-256 projection weights are missing |",
              "| Correct-class margin before/after fusion | unavailable: saved accuracy/F1 metrics do not contain logits or fused prototypes |",
              "| Learned lambda and movement toward zero | historical final lambda is available below; no new inference or gate trajectory measurement |",
              "| Correct/shuffled/random representation misalignment | unavailable: projected representations were not saved |",
              "| Collapse of projected semantic directions | unavailable: trained projection and projected prototype vectors were not saved |", "",
              "A checkpoint-backed margin comparison would use the correct-class logit minus the highest incorrect-class logit on the same query, before and after fusion. A collapse check would compare pairwise cosine similarity between projected class directions. Neither quantity can be recovered from episode accuracy, Macro-F1, initialization hashes or gradient norms.", "",
              "## Recoverable historical gate evidence", "",
              "The following are archived endpoint values from `v3_mvp_validation_details.json`, not measurements from a new evaluation. The MVP initialized lambda to 0.1. Every reported final lambda is higher than that initialization, so the endpoints moved away from zero. No per-update lambda history was saved; monotonic movement cannot be established.", "",
              "| Seed | Historical condition | Initial lambda | Final lambda | Final minus initial |", "|---:|---|---:|---:|---:|"]
    for seed in SEEDS:
        for condition in ("V3-correct", "V3-shuffled", "V3-random"):
            row = next(row for row in rows if row["seed"] == seed and row["model"] == condition)
            gate = float(row["lambda"])
            lines.append(f"| {seed} | {condition} | 0.100000 | {gate:.9f} | {gate - 0.1:+.9f} |")
    lines += ["", "Those endpoint values show that the gate did not suppress the semantic contribution toward zero during the prior training run. They do not establish representation misalignment, direction collapse, or a causal explanation for the validation loss of accuracy. These remain unmeasured.", "",
              "## Evidence and preservation", "",
              "- The existing MVP runner has no model-checkpoint save call and explicitly documents that models remain in memory.",
              f"- MVP runner SHA-256: `{sha256(runner)}`.",
              f"- Historical metric JSON SHA-256: `{sha256(historical_path)}`.",
              f"- Existing artifacts hash-checked: {len(before)}. V1/V2/V3 architectures, checkpoints, results and manuscript remain unchanged.",
              "- The checkpoint inventory is listed below for review. No checkpoint was overwritten.", "",
              "```text"]
    lines += [str(path.relative_to(ROOT).as_posix()) for path in inventory]
    lines += ["```", "", "## Command and stop point", "", "```powershell",
              "& .venv/Scripts/python.exe LLMGP/experiments/diagnose_v3_alignment.py", "```", "",
              f"Preflight runtime: {time.perf_counter() - started:.2f} seconds. New validation metrics and test results: not evaluated. The script checks availability, records the missing-checkpoint stop condition, and exits without training. If checkpoint candidates appear later, it fails closed for provenance inspection rather than assigning them to a seed or semantic arm by filename."]
    assert before == artifact_snapshot(), "An existing artifact changed during preflight"
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert before == artifact_snapshot(), "An existing artifact changed while saving the report"
    print(f"Stopped: no trained V3 checkpoints. Saved {REPORT.relative_to(ROOT)}; "
          f"{len(before)} existing artifact hashes unchanged. No retraining or evaluation.")


if __name__ == "__main__":
    main()
