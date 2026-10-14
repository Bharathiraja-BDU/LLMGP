import numpy as np

from src.episode_sampler import EpisodeSampler


def test_no_label_leakage():
    sampler = EpisodeSampler(num_classes=7, support_shots=3, query_shots=15)
    episode = sampler.sample_episode(class_ids=[0, 1], seed=0)
    support_labels = episode["support_labels"]
    query_labels = episode["query_labels"]
    assert set(support_labels.tolist()) == {0, 1}
    assert set(query_labels.tolist()) == {0, 1}
    assert set(support_labels.tolist()).isdisjoint(set(query_labels.tolist())) is False
