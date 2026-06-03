"""Win-probability model: baseline scoreboard features + validated stats.

Train on historical snapshots. `delta_wp(snapshot, stat)` answers the product
question -- "how much does this stat move win probability right now?" -- by
comparing WP with the stat at its observed value vs at its historical median.
Only stats that passed the PredictivityFilter should be fed in; otherwise the
delta is noise dressed up as insight.
"""
from __future__ import annotations

import statistics

from .logistic import LogisticRegression
from .models import GameSnapshot


class WinProbabilityModel:
    def __init__(self, stat_names: list[str] | None = None, l2: float = 1.0,
                 epochs: int = 300, seed: int = 0):
        self.stat_names = list(stat_names or [])
        self.model = LogisticRegression(l2=l2, epochs=epochs, seed=seed)
        self._medians: dict[str, float] = {}

    def _features(self, s: GameSnapshot) -> list[float]:
        return s.base_features() + [float(s.stats.get(n, self._medians.get(n, 0.0)))
                                    for n in self.stat_names]

    def fit(self, snapshots: list[GameSnapshot]) -> "WinProbabilityModel":
        labeled = [s for s in snapshots if s.won is not None]
        for n in self.stat_names:
            vals = [s.stats[n] for s in labeled if n in s.stats]
            self._medians[n] = statistics.median(vals) if vals else 0.0
        X = [self._features(s) for s in labeled]
        y = [int(s.won) for s in labeled]
        self.model.fit(X, y)
        return self

    def win_prob(self, s: GameSnapshot) -> float:
        return self.model.predict_proba([self._features(s)])[0]

    def delta_wp(self, s: GameSnapshot, stat: str) -> float:
        """WP(stat at observed value) - WP(stat at historical median)."""
        if stat not in self.stat_names:
            return 0.0
        actual = self.win_prob(s)
        baseline_stats = dict(s.stats)
        baseline_stats[stat] = self._medians.get(stat, 0.0)
        s_base = GameSnapshot(s.game_id, s.team, s.score_diff, s.seconds_remaining,
                              s.is_home, baseline_stats, s.won)
        return actual - self.win_prob(s_base)
