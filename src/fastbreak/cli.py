"""Command-line entry points."""
from __future__ import annotations

import argparse
import os
import time

from .config import Config
from .feed.replay import ReplayFeed
from .feedback.store import FeedbackStore
from .percentile.detector import AnomalyDetector
from .percentile.store import seed_synthetic_store
from .pipeline import Pipeline
from .publish.console import ConsolePublisher
from .publish.x_connector import XPublisher
from .selection.entropy import SurpriseSelector


def demo(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="FastBreak replay demo (Knicks vs Spurs sample).")
    here = os.path.dirname(__file__)
    p.add_argument("--pbp", default=os.path.join(here, "feed", "sample_pbp.json"))
    p.add_argument("--speed", type=float, default=0.0, help="0=instant, 1=realtime, 10=10x")
    p.add_argument("--threshold", type=float, default=None, help="surprise threshold")
    p.add_argument("--x", action="store_true", help="also route to X publisher (dry-run)")
    args = p.parse_args(argv)

    cfg = Config.from_env()
    if args.threshold is not None:
        cfg.surprise_threshold = args.threshold

    store = seed_synthetic_store()
    detector = AnomalyDetector(store, cfg.percentile_threshold, cfg.min_sample_size)
    feed = ReplayFeed(args.pbp, speed=args.speed)
    publisher = XPublisher(dry_run=True) if args.x else ConsolePublisher()

    pipe = Pipeline(
        feed=feed, detector=detector,
        selector=SurpriseSelector(threshold=cfg.surprise_threshold),
        publisher=publisher,
        feedback=FeedbackStore(),
        config=cfg,
    )

    print(f"=== FastBreak demo | generator={cfg.generator} | "
          f"surprise>={cfg.surprise_threshold} | percentile>={cfg.percentile_threshold} ===")
    t0 = time.perf_counter()
    n = sum(1 for _ in pipe.run())
    dt = (time.perf_counter() - t0) * 1000
    print(f"\n=== published {n} headlines in {dt:.1f}ms ===")


if __name__ == "__main__":
    demo()
