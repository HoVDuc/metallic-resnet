"""Random selection of COCO annotation instances to remove."""

from __future__ import annotations

import random
from typing import Any, List, Mapping, Sequence


def select_random_annotations(
    annotations: Sequence[Mapping[str, Any]], rng: random.Random
) -> List[Mapping[str, Any]]:
    """Select one to three distinct annotations, capped by availability."""

    candidates = list(annotations)
    if not candidates:
        return []
    count = rng.randint(1, min(3, len(candidates)))
    return rng.sample(candidates, count)
