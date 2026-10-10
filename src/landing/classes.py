"""Semantic Drone Dataset classes and what each one means for landing.

The dataset's ``class_dict.csv`` defines the classes and their mask colours; the names below
decide only how a class is treated when picking a landing site. Matching is by name, so the
order in the CSV does not matter.
"""

from __future__ import annotations

import numpy as np

# Flat ground a small multirotor can set down on. AR markers are printed sheets lying on the
# ground, so they count as ground too.
LANDABLE = frozenset({"paved-area", "dirt", "grass", "gravel", "ar-marker"})
# People and animals: never land near them, whatever the ground looks like.
LIVING = frozenset({"person", "dog"})
# Not trained on and not scored (Ultralytics' ignore label).
IGNORE = frozenset({"unlabeled", "conflicting"})


def train_classes(names: list[str]) -> list[str]:
    """Class names the network learns, in dataset order, without the ignored ones."""
    return [n for n in names if n not in IGNORE]


def label_mapping(names: list[str]) -> dict[int, int | str]:
    """Raw mask id (position in class_dict.csv) -> train id, as an Ultralytics ``label_mapping``."""
    train = {n: i for i, n in enumerate(train_classes(names))}
    return {raw: train.get(n, "ignore_label") for raw, n in enumerate(names)}


def category_luts(classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Boolean lookup tables over train ids: landable ground and living beings."""
    landable = np.array([c in LANDABLE for c in classes], dtype=bool)
    living = np.array([c in LIVING for c in classes], dtype=bool)
    return landable, living
