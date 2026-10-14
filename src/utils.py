from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_class_split(configuration: str) -> Dict[str, List[int]]:
    """Return a valid split for the manuscript protocol or an alternative alternative design."""
    config_map = {
        "A": {"train": [0, 1, 2, 3], "val": [4], "test": [5, 6]},
        "B": {"train": [0, 1, 2], "val": [3, 4], "test": [5, 6]},
    }
    if configuration not in config_map:
        raise ValueError(f"Unsupported class split configuration: {configuration}")
    return config_map[configuration]
