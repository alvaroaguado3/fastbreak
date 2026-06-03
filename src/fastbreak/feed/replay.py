"""Replay simulator -- the development workhorse.

Reads a JSON file of play-by-play snapshots and yields StatEvent objects,
optionally pacing them in real time (or sped up) so you can exercise the full
pipeline without a live game or paid feed.

JSON format (see feed/sample_pbp.json):
{
  "context": { ...GameContext fields... },
  "events": [
    {"t": 0.0, "subject_id": "jb", "subject_name": "Jalen Brunson",
     "subject_type": "player", "stat": "points", "value": 12,
     "quarter": 1, "clock_seconds": 540},
    ...
  ]
}
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

from ..models import GameContext, StatEvent
from .base import DataFeed


class ReplayFeed(DataFeed):
    def __init__(self, path: str | Path, speed: float = 0.0):
        """speed=0 -> emit as fast as possible (tests/benchmarks).
        speed=1 -> real time. speed=10 -> 10x faster than real time."""
        self.path = Path(path)
        self.speed = speed
        data = json.loads(self.path.read_text())
        self._base_ctx = data["context"]
        self._rows = data["events"]

    def events(self) -> Iterator[StatEvent]:
        prev_t = 0.0
        for row in self._rows:
            if self.speed > 0:
                dt = (row.get("t", prev_t) - prev_t) / self.speed
                if dt > 0:
                    time.sleep(dt)
                prev_t = row.get("t", prev_t)
            ctx = GameContext(
                game_id=self._base_ctx["game_id"],
                home_team=self._base_ctx["home_team"],
                away_team=self._base_ctx["away_team"],
                game_type=self._base_ctx.get("game_type", "regular"),
                quarter=row.get("quarter", 1),
                clock_seconds=row.get("clock_seconds", 720),
                venue=self._base_ctx.get("venue", ""),
                city=self._base_ctx.get("city", ""),
                state=self._base_ctx.get("state", ""),
                opponent=row.get("opponent", self._base_ctx.get("opponent", "")),
            )
            yield StatEvent(
                subject_id=row["subject_id"],
                subject_name=row["subject_name"],
                subject_type=row.get("subject_type", "player"),
                stat=row["stat"],
                value=float(row["value"]),
                context=ctx,
                raw=row,
            )
