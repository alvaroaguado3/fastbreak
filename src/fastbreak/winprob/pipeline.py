"""Live win-impact pipeline (separate from the headline pipeline).

Given a stream of GameSnapshots and a fitted WinProbabilityModel restricted to
VALIDATED predictive stats, emit a WinImpactInsight whenever a validated stat's
contribution to win probability exceeds a threshold. This is the win-odds analog
of the headline pipeline's detector+selector, minus the NLG.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

from .model import WinProbabilityModel
from .models import GameSnapshot, PredictiveStat, WinImpactInsight


class WinImpactPipeline:
    def __init__(self, model: WinProbabilityModel,
                 validated: list[PredictiveStat],
                 min_delta_wp: float = 0.03):
        self.model = model
        self.min_delta_wp = min_delta_wp
        self._q = {p.name: p.q_value for p in validated}
        self._eff = {p.name: p.effect for p in validated}

    def run(self, snapshots: Iterable[GameSnapshot]) -> Iterator[WinImpactInsight]:
        for s in snapshots:
            wp = self.model.win_prob(s)
            for stat in self.model.stat_names:
                if stat not in s.stats:
                    continue
                dwp = self.model.delta_wp(s, stat)
                if abs(dwp) >= self.min_delta_wp:
                    yield WinImpactInsight(
                        snapshot=s, stat=stat, win_prob=wp, delta_wp=dwp,
                        effect=self._eff.get(stat, 0.0),
                        q_value=self._q.get(stat, 1.0))
