"""Historical distribution store with O(log n) percentile + surprisal lookup.

Each (stat, context_key) maps to an EmpiricalDistribution. In production you
would precompute these from a season+ of play-by-play (offline job) and load
them as quantile arrays; for development we seed synthetic distributions.

Why empirical + bisect rather than a parametric fit? In-game counting stats
(points-so-far, blocks, plus_minus) are heavily skewed and bounded by game
clock, so a Gaussian assumption is wrong in exactly the tail we care about.
Empirical quantiles are non-parametric and the lookup is still logarithmic.
For unbounded streaming you can swap EmpiricalDistribution for a t-digest
(Dunning & Ertl, 2019) without touching the detector.
"""
from __future__ import annotations

import bisect
import json
import math
import random
from collections import defaultdict
from pathlib import Path


class EmpiricalDistribution:
    """Sorted-sample empirical distribution."""

    def __init__(self, samples: list[float] | None = None):
        self._samples: list[float] = sorted(samples) if samples else []

    @property
    def n(self) -> int:
        return len(self._samples)

    def add(self, value: float) -> None:
        bisect.insort(self._samples, value)

    def percentile_of(self, value: float) -> float:
        """Fraction of historical samples strictly below `value`, in [0,1].

        Uses a midpoint rule for ties so an exactly-median value scores ~0.5.
        """
        if not self._samples:
            return 0.0
        lo = bisect.bisect_left(self._samples, value)
        hi = bisect.bisect_right(self._samples, value)
        # midpoint of the tie block -> stable percentile for repeated values
        rank = (lo + hi) / 2.0
        return rank / self.n

    def surprisal_bits(self, value: float) -> float:
        """Information content of "value or higher": -log2(P(X >= value)).

        Add-one (Laplace) smoothing avoids infinite surprisal on a new max.
        A 90th-pct event ~ 3.3 bits; a 1-in-1000 event ~ 10 bits.
        """
        if not self._samples:
            return 0.0
        ge = self.n - bisect.bisect_left(self._samples, value)
        p = (ge + 1) / (self.n + 1)
        return -math.log2(p)

    def to_list(self) -> list[float]:
        return list(self._samples)


class DistributionStore:
    """stat -> context_key -> EmpiricalDistribution."""

    def __init__(self):
        self._d: dict[str, dict[str, EmpiricalDistribution]] = defaultdict(dict)

    def get(self, stat: str, context_key: str) -> EmpiricalDistribution | None:
        return self._d.get(stat, {}).get(context_key)

    def add_sample(self, stat: str, context_key: str, value: float) -> None:
        dist = self._d[stat].setdefault(context_key, EmpiricalDistribution())
        dist.add(value)

    def bulk_load(self, stat: str, context_key: str, samples: list[float]) -> None:
        self._d[stat][context_key] = EmpiricalDistribution(samples)

    def save(self, path: str | Path) -> None:
        out = {
            stat: {ck: dist.to_list() for ck, dist in by_ctx.items()}
            for stat, by_ctx in self._d.items()
        }
        Path(path).write_text(json.dumps(out))

    @classmethod
    def load(cls, path: str | Path) -> "DistributionStore":
        store = cls()
        raw = json.loads(Path(path).read_text())
        for stat, by_ctx in raw.items():
            for ck, samples in by_ctx.items():
                store.bulk_load(stat, ck, samples)
        return store


# --- Synthetic seeding for development --------------------------------------

# Rough per-stat "single game" scales (mean-ish), used to draw plausible
# historical distributions. Replace with real precomputed data in production.
_STAT_SCALE = {
    "points": 14.0,
    "rebounds": 6.0,
    "assists": 4.0,
    "blocks": 1.0,
    "steals": 1.2,
    "threes": 2.0,
    "plus_minus": 0.0,
    "turnovers": 2.0,
}


def seed_synthetic_store(seed: int = 7, samples_per_context: int = 400) -> DistributionStore:
    """Build a store with plausible distributions across several contexts.

    Contexts mirror GameContext.context_keys() so the detector finds matches.
    Counting stats use a log-normal-ish draw; plus_minus uses a normal draw.
    """
    rng = random.Random(seed)
    store = DistributionStore()
    contexts = [
        "all", "regular", "playoff", "finals",
        "finals:q1", "finals:q2", "finals:q3", "finals:q4",
        "finals:q4:clutch", "finals:q4:normal",
        "finals:vs:San Antonio Spurs", "finals:vs:New York Knicks",
        "venue:Madison Square Garden",
        "finals:q4:venue:Madison Square Garden",
    ]
    for stat, scale in _STAT_SCALE.items():
        for ck in contexts:
            samples = []
            for _ in range(samples_per_context):
                if stat == "plus_minus":
                    samples.append(rng.gauss(0, 9))
                else:
                    # log-normal keeps it non-negative and right-skewed
                    mu = math.log(scale + 1e-6)
                    samples.append(max(0.0, rng.lognormvariate(mu, 0.6)))
            store.bulk_load(stat, ck, samples)
    return store
