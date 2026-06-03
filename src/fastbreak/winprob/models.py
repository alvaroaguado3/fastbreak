"""Data structures for the win-probability pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GameSnapshot:
    """Game state at a moment, used both for training the WP model and live.

    `stats` holds candidate per-team stat features whose predictive value we
    want to screen (e.g. {"fast_break_pts": 12, "second_chance_pts": 8}).
    `won` is the label (1 if this team eventually won) -- only present in
    historical training data, None live.
    """
    game_id: str
    team: str
    score_diff: float          # this team's score minus opponent's
    seconds_remaining: float   # in the whole game
    is_home: int = 0
    stats: dict[str, float] = field(default_factory=dict)
    won: int | None = None

    # --- baseline (scoreboard) features ---
    def base_features(self) -> list[float]:
        # score margin, time fraction elapsed, and their interaction:
        # a lead matters more as time runs out. This is the canonical
        # score-differential-vs-time win-probability skeleton (Stern, 1994).
        total = 2880.0  # 48 min regulation
        time_frac = 1.0 - min(self.seconds_remaining, total) / total
        return [self.score_diff, time_frac, self.score_diff * time_frac, float(self.is_home)]

    BASE_NAMES = ("score_diff", "time_frac", "score_diff*time", "is_home")


@dataclass(slots=True)
class PredictiveStat:
    """A stat that survived the predictivity filter."""
    name: str
    cv_lift: float             # mean out-of-sample log-loss improvement vs baseline
    p_value: float             # permutation-test p
    q_value: float             # Benjamini-Hochberg adjusted p (FDR)
    stability: float           # fraction of resamples where it helped [0..1]
    effect: float              # standardized coefficient (log-odds per SD)
    n: int                     # samples backing it

    def passed(self, alpha: float, min_stability: float) -> bool:
        return (self.q_value <= alpha and self.stability >= min_stability
                and self.cv_lift > 0)


@dataclass(slots=True)
class WinImpactInsight:
    """Live output: a validated stat is moving the needle right now."""
    snapshot: GameSnapshot
    stat: str
    win_prob: float            # model WP for this team at this moment
    delta_wp: float            # change in WP attributable to this stat
    effect: float              # historical standardized effect size
    q_value: float             # how confidently predictive (from screening)
