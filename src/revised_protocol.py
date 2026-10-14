from __future__ import annotations

from typing import Any

import numpy as np


PROTOCOL_NAME = "revised_self_supervised_3_2_2"
PROTOCOL_SEED = 2026
RUN_SEEDS = (11, 23, 37, 53, 71)
CLASS_ID_TO_TOPIC = {
    0: "Theory",
    1: "Reinforcement Learning",
    2: "Genetic Algorithms",
    3: "Neural Networks",
    4: "Probabilistic Methods",
    5: "Case-Based Reasoning",
    6: "Rule Learning",
}
CLASS_DESCRIPTIONS = {
    0: "Research on theoretical foundations and mathematical analysis of machine learning.",
    1: "Research on reinforcement learning, sequential decisions, and reward-driven agents.",
    2: "Research on genetic algorithms and evolutionary optimization methods.",
    3: "Research on neural networks and connectionist machine learning models.",
    4: "Research on probabilistic models and statistical methods for machine learning.",
    5: "Research on case-based reasoning and solving problems by reusing prior cases.",
    6: "Research on rule learning and methods that infer symbolic decision rules.",
}
CLASS_COUNTS_BY_ID = {0: 351, 1: 217, 2: 418, 3: 818, 4: 426, 5: 298, 6: 180}
CLASS_SPLIT = {
    "train": (0, 1, 2),
    "validation": (3, 4),
    "test": (5, 6),
}


def validate_protocol(labels: np.ndarray) -> None:
    split_sets = [set(values) for values in CLASS_SPLIT.values()]
    if any(split_sets[i] & split_sets[j] for i in range(len(split_sets)) for j in range(i + 1, len(split_sets))):
        raise AssertionError("Class splits must be pairwise disjoint")
    if set().union(*split_sets) != set(CLASS_ID_TO_TOPIC):
        raise AssertionError("Every Cora class must appear in exactly one split")
    actual_counts = {class_id: int(np.count_nonzero(labels == class_id)) for class_id in CLASS_ID_TO_TOPIC}
    if actual_counts != CLASS_COUNTS_BY_ID:
        raise ValueError(f"Cora label-ID mapping does not match canonical class counts: {actual_counts}")


def labels_visible_in_split(labels: np.ndarray, class_ids: tuple[int, ...]) -> np.ndarray:
    """Return labels for the requested split classes and hide every other label as -1."""
    labels = np.asarray(labels, dtype=np.int64)
    visible = np.full(labels.shape, -1, dtype=np.int64)
    class_mask = np.isin(labels, np.asarray(class_ids, dtype=np.int64))
    visible[class_mask] = labels[class_mask]
    return visible


def remap_episode_labels(episode: dict[str, Any]) -> dict[str, Any]:
    """Convert sampled global class IDs to local 0..N-1 episode labels."""
    class_ids = [int(value) for value in episode["class_ids"]]
    class_to_local = {class_id: local_id for local_id, class_id in enumerate(class_ids)}
    output = dict(episode)
    output["support_labels"] = np.asarray(
        [class_to_local[int(value)] for value in episode["support_labels"]],
        dtype=np.int64,
    )
    output["query_labels"] = np.asarray(
        [class_to_local[int(value)] for value in episode["query_labels"]],
        dtype=np.int64,
    )
    return output