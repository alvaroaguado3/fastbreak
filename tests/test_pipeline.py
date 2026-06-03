import os
from fastbreak.feed.replay import ReplayFeed
from fastbreak.percentile.store import seed_synthetic_store
from fastbreak.percentile.detector import AnomalyDetector
from fastbreak.pipeline import Pipeline
from fastbreak.selection.entropy import SurpriseSelector
from fastbreak.publish.console import ConsolePublisher

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "src", "fastbreak", "feed", "sample_pbp.json")


def test_pipeline_runs_and_publishes():
    det = AnomalyDetector(seed_synthetic_store(), 0.90, 30)
    pipe = Pipeline(
        feed=ReplayFeed(SAMPLE, speed=0.0), detector=det,
        selector=SurpriseSelector(threshold=0.5), publisher=ConsolePublisher(),
    )
    published = list(pipe.run())
    assert len(published) >= 1
    assert all(h.surprise_score >= 0.5 for h in published)


def test_pipeline_coalesces_multistat():
    det = AnomalyDetector(seed_synthetic_store(), 0.90, 30)
    pipe = Pipeline(
        feed=ReplayFeed(SAMPLE, speed=0.0), detector=det,
        selector=SurpriseSelector(threshold=0.4), publisher=ConsolePublisher(),
    )
    published = list(pipe.run())
    # at least one published headline should reference 2 flags (multi-stat moment)
    assert any(len(h.candidate.flags) >= 2 for h in published)
