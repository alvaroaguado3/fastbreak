"""Provider-agnostic live data interface.

Any feed (Sportradar push, Genius Sports, nba_api polling, or a replay file)
implements `events()` as a generator of StatEvent. The rest of the pipeline
never knows which provider it is talking to -- migration is a one-line swap.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..models import StatEvent


class DataFeed(ABC):
    @abstractmethod
    def events(self) -> Iterator[StatEvent]:
        """Yield StatEvent objects as the game progresses.

        Implementations should yield as early as possible (do not batch) to
        keep end-to-end latency low.
        """
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        pass
