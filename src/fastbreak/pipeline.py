"""End-to-end orchestrator: feed -> detect -> coalesce -> generate -> select -> publish.

Latency design:
  - Each StatEvent is evaluated the moment it arrives (no batching for detection).
  - Flags occurring in the same game "moment" (same quarter + clock bucket) are
    coalesced into ONE story so we can write multi-stat headlines, then flushed
    as soon as the moment advances. This satisfies "generate headlines for 1 or
    more stats at a time" without adding latency to single-stat stories.
  - Generation + selection are the only model calls; everything else is O(log n).

Swap any component (feed, generator, publisher) without touching this file.
"""
from __future__ import annotations

import time
from collections.abc import Iterator

from .config import Config
from .feed.base import DataFeed
from .feedback.store import FeedbackStore
from .generation.base import HeadlineGenerator
from .generation.templates import TemplateGenerator
from .models import SelectedHeadline, StatFlag
from .percentile.detector import AnomalyDetector
from .publish.base import Publisher
from .publish.console import ConsolePublisher
from .selection.entropy import SurpriseSelector


def _moment_key(flag: StatFlag) -> tuple[int, int]:
    ctx = flag.event.context
    # 6-second clock buckets -> flags within the same possession coalesce
    return (ctx.quarter, ctx.clock_seconds // 6)


class Pipeline:
    def __init__(self, feed: DataFeed, detector: AnomalyDetector,
                 generator: HeadlineGenerator | None = None,
                 selector: SurpriseSelector | None = None,
                 publisher: Publisher | None = None,
                 feedback: FeedbackStore | None = None,
                 config: Config | None = None):
        self.config = config or Config.from_env()
        self.feed = feed
        self.detector = detector
        self.generator = generator or TemplateGenerator()
        self.selector = selector or SurpriseSelector(threshold=self.config.surprise_threshold)
        self.publisher = publisher or ConsolePublisher()
        self.feedback = feedback
        self._buffer: list[StatFlag] = []
        self._buffer_key: tuple[int, int] | None = None

    def run(self) -> Iterator[SelectedHeadline]:
        """Drive the whole pipeline, yielding each published headline."""
        for event in self.feed.events():
            flag = self.detector.evaluate(event)
            if flag is None:
                continue
            key = _moment_key(flag)
            if self._buffer_key is not None and key != self._buffer_key:
                yield from self._flush()
            self._buffer_key = key
            self._buffer.append(flag)
        yield from self._flush()

    def _flush(self) -> Iterator[SelectedHeadline]:
        if not self._buffer:
            return
        flags = self._buffer[: self.config.max_flags_per_headline]
        self._buffer = []
        self._buffer_key = None

        candidates = self.generator.generate(flags, n=4)
        selected = self.selector.select(candidates)
        if selected is None:
            return
        self.publisher.publish(selected)
        if self.feedback is not None:
            self.feedback.log_published(selected)
        yield selected
