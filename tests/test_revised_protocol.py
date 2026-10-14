import numpy as np

from src.episode_sampler import EpisodeSampler
from src.revised_protocol import (
    CLASS_SPLIT,
    labels_visible_in_split,
    remap_episode_labels,
    validate_protocol,
)


def test_fixed_split_is_disjoint_and_covers_all_classes():
    labels = np.concatenate([np.full(count, class_id) for class_id, count in {
        0: 351, 1: 217, 2: 418, 3: 818, 4: 426, 5: 298, 6: 180
    }.items()])
    validate_protocol(labels)
    sets = [set(class_ids) for class_ids in CLASS_SPLIT.values()]
    assert all(not (sets[i] & sets[j]) for i in range(3) for j in range(i + 1, 3))
    assert set.union(*sets) == set(range(7))


def test_split_mask_hides_all_other_class_labels():
    labels = np.asarray([0, 1, 2, 3, 4, 5, 6])
    visible = labels_visible_in_split(labels, CLASS_SPLIT["train"])
    assert visible.tolist() == [0, 1, 2, -1, -1, -1, -1]


def test_2way_episode_uses_only_allowed_classes_and_disjoint_nodes():
    labels = np.repeat(np.arange(7), 40)
    visible = labels_visible_in_split(labels, CLASS_SPLIT["train"])
    sampler = EpisodeSampler(num_classes=2, support_shots=3, query_shots=15)
    episode = remap_episode_labels(
        sampler.sample_episode(labels=visible, class_ids=list(CLASS_SPLIT["train"][:2]), seed=17)
    )
    assert set(episode["class_ids"].tolist()).issubset(CLASS_SPLIT["train"])
    assert set(episode["support_labels"].tolist()) == {0, 1}
    assert set(episode["query_labels"].tolist()) == {0, 1}
    assert set(episode["support_idx"].tolist()).isdisjoint(episode["query_idx"].tolist())