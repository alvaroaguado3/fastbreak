"""Turns raw StatEvents into StatFlags when they land in a high percentile.

Strategy: try the MOST SPECIFIC context first (e.g. finals/Q4/clutch at MSG)
and fall back to broader contexts until one has at least `min_sample_size`
historical instances. Specific contexts are more relevant but data-sparse;
this fallback is the "minimum number of instances" rule from the spec.
"""
from __future__ import annotations

from ..models import StatEvent, StatFlag
from .store import DistributionStore


class AnomalyDetector:
    def __init__(self, store: DistributionStore, percentile_threshold: float = 0.90,
                 min_sample_size: int = 30):
        self.store = store
        self.percentile_threshold = percentile_threshold
        self.min_sample_size = min_sample_size

    def evaluate(self, event: StatEvent) -> StatFlag | None:
        """Return a StatFlag if the event is high-percentile, else None.

        We prefer the most specific context that (a) exists, (b) has enough
        samples, and (c) clears the percentile bar. Among qualifying contexts
        we keep the most specific one (last in the broad->specific ordering)
        because it is the most newsworthy framing.
        """
        best: StatFlag | None = None
        for ck in event.context.context_keys():
            dist = self.store.get(event.stat, ck)
            if dist is None or dist.n < self.min_sample_size:
                continue
            pct = dist.percentile_of(event.value)
            if pct >= self.percentile_threshold:
                flag = StatFlag(
                    event=event,
                    percentile=pct,
                    matched_context=ck,
                    sample_size=dist.n,
                    surprisal_bits=dist.surprisal_bits(event.value),
                )
                # keep the most specific qualifying context (later in list)
                best = flag
        return best
