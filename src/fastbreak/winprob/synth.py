"""Synthetic historical games for testing the screener.

We plant ONE genuinely predictive stat ("clutch_stops") whose value raises the
true win probability, alongside several PURE NOISE stats. A correct
PredictivityFilter should keep clutch_stops and reject the noise -- including a
'rare_event' stat that is present in only a few games (the low-frequency trap).
"""
from __future__ import annotations

import math
import random

from .models import GameSnapshot


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def make_games(n_games: int = 400, seed: int = 1) -> list[GameSnapshot]:
    rng = random.Random(seed)
    snaps: list[GameSnapshot] = []
    for g in range(n_games):
        # one snapshot per game, sampled at a random late-game moment
        secs = rng.uniform(0, 1200)
        score_diff = rng.gauss(0, 9)
        is_home = rng.randint(0, 1)

        # TRUE signal: clutch defensive stops, modest extra effect beyond margin
        clutch_stops = max(0, round(rng.gauss(4, 2)))
        # NOISE stats
        noise_a = rng.gauss(10, 4)
        noise_b = rng.uniform(0, 20)

        total = 2880.0
        time_frac = 1.0 - min(secs, total) / total
        # latent win logit: scoreboard dominates; clutch_stops adds a little;
        # home a touch; noise contributes nothing.
        logit = (0.18 * score_diff
                 + 1.1 * score_diff * time_frac / 10
                 + 0.22 * (clutch_stops - 4)
                 + 0.15 * is_home
                 + rng.gauss(0, 0.4))
        won = 1 if rng.random() < _sigmoid(logit) else 0

        stats = {"clutch_stops": clutch_stops, "noise_a": noise_a, "noise_b": noise_b}
        # 'rare_event' present in only ~6% of games -> low-frequency trap.
        if rng.random() < 0.06:
            stats["rare_event"] = rng.uniform(0, 5)

        snaps.append(GameSnapshot(
            game_id=f"g{g}", team="NYK", score_diff=score_diff,
            seconds_remaining=secs, is_home=is_home, stats=stats, won=won))
    return snaps


CANDIDATE_STATS = ["clutch_stops", "noise_a", "noise_b", "rare_event"]
