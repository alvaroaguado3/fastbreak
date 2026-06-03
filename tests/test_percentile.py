from fastbreak.percentile.store import EmpiricalDistribution, DistributionStore, seed_synthetic_store
from fastbreak.percentile.detector import AnomalyDetector
from fastbreak.models import GameContext, StatEvent


def test_percentile_monotonic():
    d = EmpiricalDistribution([float(i) for i in range(100)])
    assert d.percentile_of(0) < d.percentile_of(50) < d.percentile_of(99)
    assert 0.0 <= d.percentile_of(50) <= 1.0


def test_surprisal_higher_for_rarer():
    d = EmpiricalDistribution([float(i) for i in range(100)])
    assert d.surprisal_bits(99) > d.surprisal_bits(50)
    # new max is finite thanks to add-one smoothing
    assert d.surprisal_bits(1000) < float("inf")


def test_store_roundtrip(tmp_path):
    s = DistributionStore()
    s.bulk_load("points", "finals", [1.0, 2.0, 3.0])
    p = tmp_path / "dist.json"
    s.save(p)
    s2 = DistributionStore.load(p)
    assert s2.get("points", "finals").n == 3


def test_detector_flags_high_value_and_picks_specific_context():
    store = seed_synthetic_store()
    det = AnomalyDetector(store, percentile_threshold=0.90, min_sample_size=30)
    ctx = GameContext("g", "NYK", "SAS", game_type="finals", quarter=4,
                      clock_seconds=60, venue="Madison Square Garden",
                      opponent="San Antonio Spurs")
    ev = StatEvent("vw", "Victor Wembanyama", "player", "blocks", 9.0, ctx)
    flag = det.evaluate(ev)
    assert flag is not None
    assert flag.percentile >= 0.90
    # most-specific qualifying context should be chosen, not just "all"
    assert flag.matched_context != "all"


def test_detector_ignores_ordinary_value():
    store = seed_synthetic_store()
    det = AnomalyDetector(store, percentile_threshold=0.90, min_sample_size=30)
    ctx = GameContext("g", "NYK", "SAS", game_type="finals")
    ev = StatEvent("p", "Role Player", "player", "points", 2.0, ctx)
    assert det.evaluate(ev) is None
