"""HeadlineGenerator interface.

Implementations take 1+ StatFlags (a "story") and return several candidate
headlines. Returning several candidates is deliberate: the selector downstream
picks the most surprising / least-repetitive one. Keeping the contract on a
LIST of flags is what lets us write multi-stat headlines (e.g. "X had Y points
AND Z blocks") in a single line.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import HeadlineCandidate, StatFlag


class HeadlineGenerator(ABC):
    @abstractmethod
    def generate(self, flags: list[StatFlag], n: int = 4) -> list[HeadlineCandidate]:
        raise NotImplementedError
