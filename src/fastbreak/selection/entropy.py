"""Surprise / entropy selector -- decides which candidates are worth publishing.

The "surprise" of a headline blends three signals:

  1. Information content of the underlying stats (surprisal in bits, from the
     detector). This is Shannon self-information: rarer stat -> more bits.
     Squashed to [0,1] via a logistic so ~3.3 bits (p90) ~ 0.5.
  2. Novelty vs recently published headlines (token Jaccard distance), so we
     don't tweet five variations of the same run.
  3. A light readability/quality prior (length sanity, has a subject+number).

score = sigmoid(w . features). The weights are exposed and updatable so the
feedback loop can learn what humans actually reshare (see feedback/ + the
`update_from_feedback` hook). This is the calibration target: optimize weights
so P(publish) tracks P(a journalist reuses it).

Reference framing: Shannon, "A Mathematical Theory of Communication" (1948) for
self-information; logistic calibration of scores is standard (Platt, 1999).
"""
from __future__ import annotations

import math
import re
from collections import deque

from ..models import HeadlineCandidate, SelectedHeadline

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_WORD.findall(s.lower()))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class NoveltyTracker:
    """Keeps a rolling window of recently published headlines."""

    def __init__(self, window: int = 50):
        self._recent: deque[set[str]] = deque(maxlen=window)

    def novelty(self, text: str) -> float:
        """1.0 = totally new; 0.0 = identical to something recent."""
        toks = _tokens(text)
        if not toks or not self._recent:
            return 1.0
        max_sim = 0.0
        for prev in self._recent:
            inter = len(toks & prev)
            union = len(toks | prev) or 1
            max_sim = max(max_sim, inter / union)
        return 1.0 - max_sim

    def remember(self, text: str) -> None:
        self._recent.append(_tokens(text))


class SurpriseSelector:
    # feature order: [bias, info_bits_norm, novelty, quality]
    DEFAULT_WEIGHTS = [-1.4, 2.6, 1.1, 0.8]

    def __init__(self, threshold: float = 0.62, weights: list[float] | None = None,
                 novelty: NoveltyTracker | None = None):
        self.threshold = threshold
        self.weights = list(weights or self.DEFAULT_WEIGHTS)
        self.novelty = novelty or NoveltyTracker()

    # --- feature extraction ---
    def _info_norm(self, cand: HeadlineCandidate) -> float:
        # aggregate surprisal across flags, diminishing returns, then squash
        total = sum(f.surprisal_bits for f in cand.flags)
        # logistic centered near 3.3 bits (~p90 single stat)
        return _sigmoid((total - 3.3) / 2.0)

    def _quality(self, cand: HeadlineCandidate) -> float:
        text = cand.text
        n = len(text)
        length_ok = 1.0 if 25 <= n <= 230 else 0.3
        has_number = 1.0 if re.search(r"\d", text) else 0.4
        return 0.5 * length_ok + 0.5 * has_number

    def score(self, cand: HeadlineCandidate) -> SelectedHeadline:
        info = self._info_norm(cand)
        nov = self.novelty.novelty(cand.text)
        qual = self._quality(cand)
        feats = [1.0, info, nov, qual]
        s = _sigmoid(sum(w * f for w, f in zip(self.weights, feats)))
        return SelectedHeadline(
            candidate=cand, surprise_score=s, novelty_score=nov,
            reasons={"info_bits_norm": info, "novelty": nov, "quality": qual},
        )

    def select(self, candidates: list[HeadlineCandidate]) -> SelectedHeadline | None:
        """Pick the single best candidate if it clears the bar, else None.

        We publish at most one headline per story to avoid spamming; the
        feedback loop can later raise/lower the threshold per stat type.
        """
        if not candidates:
            return None
        scored = [self.score(c) for c in candidates]
        best = max(scored, key=lambda s: s.surprise_score)
        if best.surprise_score >= self.threshold:
            self.novelty.remember(best.candidate.text)
            return best
        return None

    # --- feedback hook ---
    def update_from_feedback(self, selected: SelectedHeadline, reused: bool,
                             lr: float = 0.05) -> None:
        """Online logistic-regression update toward observed human reuse.

        `reused` = did a human (journalist/repost) actually use it? This nudges
        the weights so the surprise score becomes a calibrated predictor of
        reuse. Run it from the feedback consumer as labels arrive.
        """
        info = selected.reasons["info_bits_norm"]
        feats = [1.0, info, selected.reasons["novelty"], selected.reasons["quality"]]
        pred = selected.surprise_score
        err = (1.0 if reused else 0.0) - pred
        self.weights = [w + lr * err * f for w, f in zip(self.weights, feats)]
