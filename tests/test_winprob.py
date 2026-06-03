import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastbreak.winprob import LogisticRegression, PredictivityFilter, WinProbabilityModel
from fastbreak.winprob.screening import _bh_qvalues
from fastbreak.winprob.synth import make_games, CANDIDATE_STATS


def test_logistic_separable():
    X = [[x] for x in range(-20, 21)]
    y = [0 if x < 0 else 1 for x in range(-20, 21)]
    m = LogisticRegression(epochs=400).fit(X, y)
    assert m.predict_proba([[-10]])[0] < 0.3
    assert m.predict_proba([[10]])[0] > 0.7


def test_bh_monotone_and_bounded():
    q = _bh_qvalues([0.001, 0.04, 0.5, 0.9])
    assert all(0 <= v <= 1 for v in q)
    assert q[0] <= q[1] <= q[2] <= q[3]


def test_filter_keeps_real_signal_rejects_noise():
    games = make_games(n_games=400, seed=1)
    filt = PredictivityFilter(min_sample=50, n_permutations=120, n_bootstrap=30, alpha=0.05)
    results = {r.name: r for r in filt.screen(games, CANDIDATE_STATS)}
    keep = {r.name for r in filt.passing(list(results.values()))}
    # the planted predictor should survive
    assert "clutch_stops" in keep
    # pure noise should not
    assert "noise_a" not in keep and "noise_b" not in keep


def test_filter_rejects_low_frequency_trap():
    games = make_games(n_games=400, seed=1)
    filt = PredictivityFilter(min_sample=50)
    results = {r.name: r for r in filt.screen(games, CANDIDATE_STATS)}
    # 'rare_event' appears in ~6% of games -> below min_sample -> auto-rejected
    assert results["rare_event"].n < 50
    assert "rare_event" not in {r.name for r in filt.passing(list(results.values()))}


def test_winprob_model_calibration_beats_constant():
    games = make_games(n_games=300, seed=2)
    model = WinProbabilityModel(stat_names=["clutch_stops"]).fit(games)
    y = [g.won for g in games]
    p = [model.win_prob(g) for g in games]
    base_rate = sum(y) / len(y)
    model_ll = LogisticRegression.log_loss(y, p)
    const_ll = LogisticRegression.log_loss(y, [base_rate] * len(y))
    assert model_ll < const_ll  # model is better than predicting the base rate
