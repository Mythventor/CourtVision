"""Jersey number recognition: read numbers per frame, then vote per track."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def read_number(crop: np.ndarray) -> int | None:
    """Read a jersey number from a player crop, or None if unreadable.

    TODO (week 1): try a Roboflow jersey-number model or an OCR / small VLM model.
    """
    raise NotImplementedError("Jersey number reading is not implemented yet")


class JerseyVoter:
    """Collect per-frame number readings for each track and pick the most common one."""

    def __init__(self, min_votes: int = 3):
        self.min_votes = min_votes
        self.votes: dict[int, Counter[int]] = defaultdict(Counter)

    def add(self, track_id: int, number: int | None) -> None:
        if number is not None:
            self.votes[track_id][number] += 1

    def number_for(self, track_id: int) -> int | None:
        counts = self.votes.get(track_id)
        if not counts:
            return None
        number, n = counts.most_common(1)[0]
        return number if n >= self.min_votes else None

    def assignments(self) -> dict[int, int | None]:
        return {t: self.number_for(t) for t in self.votes}
