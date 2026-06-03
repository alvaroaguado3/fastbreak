"""Core data structures shared across the pipeline.

These are intentionally plain dataclasses (no heavy deps) so the hot path
stays fast and the objects are trivially serializable for the feedback log.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# --- Context dimensions -----------------------------------------------------
# A "context" is the conditioning we compare a stat against. The richer the
# context, the more specific (and more surprising/relevant) the percentile.

GameType = str  # "regular" | "playoff" | "finals"


@dataclass(frozen=True, slots=True)
class GameContext:
    """The situational frame a stat occurs in.

    Used both to (a) look up the right historical distribution and
    (b) condition headline generation.
    """
    game_id: str
    home_team: str
    away_team: str
    game_type: GameType = "regular"
    quarter: int = 1
    # seconds remaining in the current period (regulation period = 720s)
    clock_seconds: int = 720
    venue: str = ""
    city: str = ""
    state: str = ""
    # opponent the player's team is currently facing (for matchup context)
    opponent: str = ""

    def context_keys(self) -> list[str]:
        """Ordered list of context keys, broad -> specific.

        The detector tries the most specific context that has a minimum
        number of historical instances, falling back to broader ones.
        """
        q = f"q{self.quarter}"
        clutch = "clutch" if (self.quarter >= 4 and self.clock_seconds <= 300) else "normal"
        return [
            "all",
            self.game_type,
            f"{self.game_type}:{q}",
            f"{self.game_type}:{q}:{clutch}",
            f"{self.game_type}:vs:{self.opponent}",
            f"venue:{self.venue}",
            f"{self.game_type}:{q}:venue:{self.venue}",
        ]


# --- Incoming events --------------------------------------------------------

@dataclass(slots=True)
class StatEvent:
    """A single (player|team, stat, value) observation at a point in time.

    Emitted by a DataFeed for every meaningful play-by-play update.
    """
    subject_id: str          # player id or team id
    subject_name: str
    subject_type: str        # "player" | "team"
    stat: str                # "points", "rebounds", "plus_minus", "assists", ...
    value: float
    context: GameContext
    wall_clock: float = field(default_factory=time.time)
    raw: dict[str, Any] = field(default_factory=dict)


# --- Detector output --------------------------------------------------------

@dataclass(slots=True)
class StatFlag:
    """A stat that landed in a high percentile for some context."""
    event: StatEvent
    percentile: float            # [0..1] within the matched context
    matched_context: str         # which context key was used
    sample_size: int             # n historical instances backing the percentile
    surprisal_bits: float        # -log2(P(value >= observed)) information content

    @property
    def stat(self) -> str:
        return self.event.stat


# --- Generator output -------------------------------------------------------

@dataclass(slots=True)
class HeadlineCandidate:
    """One candidate headline produced for one or more flags."""
    text: str
    flags: list[StatFlag]
    source: str                  # "template" | "neural"
    gen_latency_ms: float = 0.0
    logprob: Optional[float] = None  # model confidence if available


# --- Selector output --------------------------------------------------------

@dataclass(slots=True)
class SelectedHeadline:
    """A candidate that cleared the surprise/quality bar and will be published."""
    candidate: HeadlineCandidate
    surprise_score: float        # [0..1]
    novelty_score: float         # [0..1] vs recently published
    reasons: dict[str, float] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Flatten for the feedback log (JSONL)."""
        c = self.candidate
        return {
            "ts": time.time(),
            "text": c.text,
            "source": c.source,
            "surprise_score": self.surprise_score,
            "novelty_score": self.novelty_score,
            "reasons": self.reasons,
            "gen_latency_ms": c.gen_latency_ms,
            "flags": [
                {
                    "subject": f.event.subject_name,
                    "stat": f.stat,
                    "value": f.event.value,
                    "percentile": f.percentile,
                    "context": f.matched_context,
                    "surprisal_bits": f.surprisal_bits,
                    "n": f.sample_size,
                }
                for f in c.flags
            ],
        }
