"""Sportradar NBA adapter.

Sportradar offers two relevant *push* feeds (Realtime customers only):
  - Push Events     -> play-by-play (mirrors the Game Play-by-Play REST feed)
  - Push Statistics -> running box-score stats per player/team

Both stream newline-delimited JSON over a long-lived HTTP connection, so we
read incrementally and translate each payload into StatEvent objects. This is
the lowest-latency public option; a websocket/Genius Sports adapter would slot
in the same way.

Docs: https://developer.sportradar.com/basketball/docs/nba-ig-api-basics

NOTE: requires `pip install -e ".[live]"`. Without credentials this class is
import-safe but `events()` will raise a clear error.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator

from ..models import GameContext, StatEvent
from .base import DataFeed

PUSH_STATS_URL = (
    "https://api.sportradar.com/nba/{access}/stream/statistics/subscribe"
    "?api_key={key}"
)


class SportradarPushFeed(DataFeed):
    def __init__(self, context: GameContext, api_key: str | None = None,
                 access_level: str | None = None, url: str | None = None):
        self.context = context
        self.api_key = api_key or os.environ.get("SPORTRADAR_API_KEY", "")
        self.access_level = access_level or os.environ.get("SPORTRADAR_ACCESS_LEVEL", "trial")
        self.url = url
        self._resp = None

    def _endpoint(self) -> str:
        if self.url:
            return self.url
        access = "trial/v8/en" if self.access_level == "trial" else "production/v8/en"
        return PUSH_STATS_URL.format(access=access, key=self.api_key)

    def events(self) -> Iterator[StatEvent]:
        try:
            import requests  # local import keeps core dependency-free
        except ImportError as e:  # pragma: no cover
            raise RuntimeError('Live feeds need: pip install -e ".[live]"') from e
        if not self.api_key:
            raise RuntimeError("SPORTRADAR_API_KEY is not set (Realtime customer required).")

        self._resp = requests.get(self._endpoint(), stream=True, timeout=(5, None))
        self._resp.raise_for_status()
        for line in self._resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield from self._translate(payload)

    def _translate(self, payload: dict) -> Iterator[StatEvent]:
        """Map a Sportradar push payload to StatEvent objects.

        The push-statistics payload carries running totals per player. We emit
        one StatEvent per (player, tracked stat). Clock/quarter live on the
        enclosing game object; we update self.context accordingly.

        This mapping is deliberately defensive -- Sportradar's schema nests
        differently across v7/v8, so we read keys leniently and skip unknowns.
        """
        ctx = self._update_context(payload)
        statistics = payload.get("statistics") or payload.get("payload", {}).get("statistics")
        if not statistics:
            return
        for team in statistics.get("teams", []):
            for player in team.get("players", []):
                name = player.get("full_name") or player.get("name", "?")
                pid = player.get("id", name)
                stats = player.get("statistics", {})
                for stat_name, fb_name in _STAT_MAP.items():
                    if stat_name in stats:
                        yield StatEvent(
                            subject_id=pid, subject_name=name, subject_type="player",
                            stat=fb_name, value=float(stats[stat_name]),
                            context=ctx, raw=player,
                        )

    def _update_context(self, payload: dict) -> GameContext:
        game = payload.get("game", {}) or payload.get("payload", {}).get("game", {})
        clock = game.get("clock", "12:00")
        try:
            mm, ss = clock.split(":")
            clock_seconds = int(mm) * 60 + int(ss)
        except (ValueError, AttributeError):
            clock_seconds = self.context.clock_seconds
        return GameContext(
            game_id=self.context.game_id,
            home_team=self.context.home_team,
            away_team=self.context.away_team,
            game_type=self.context.game_type,
            quarter=int(game.get("quarter", self.context.quarter)),
            clock_seconds=clock_seconds,
            venue=self.context.venue, city=self.context.city, state=self.context.state,
            opponent=self.context.opponent,
        )

    def close(self) -> None:  # pragma: no cover
        if self._resp is not None:
            self._resp.close()


# Sportradar stat key -> FastBreak stat name
_STAT_MAP = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "steals": "steals",
    "blocks": "blocks",
    "three_points_made": "threes",
    "plus_minus": "plus_minus",
    "turnovers": "turnovers",
}
