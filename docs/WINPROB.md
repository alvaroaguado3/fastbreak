# Win-probability & predictivity screening (`fastbreak/winprob/`)

A **separate pipeline** from headlines. Goal: estimate live win probability and
surface which stats genuinely raise win likelihood — without getting fooled by
low-frequency noise.

## The two problems (kept separate on purpose)

1. **Win-probability model** — the dominant predictors are score margin + time
   remaining (Stern, 1994). `WinProbabilityModel` fits a calibrated logistic
   model on `base_features()` = `[score_diff, time_frac, score_diff×time, is_home]`
   plus any *validated* stats.
2. **Predictivity screening** — does a candidate stat add signal *beyond the
   scoreboard*? `PredictivityFilter` answers this and is the heart of the module.

## Why screening is non-negotiable

You're testing many (stat × context) hypotheses on sparse data. In-sample
correlation will always find "signal." The filter requires a stat to survive
five independent guards before it's trusted:

| Guard | Rejects | Reference |
|---|---|---|
| Min sample size | claims from a handful of games | — |
| Out-of-sample CV lift > 0 | overfit / in-sample-only effects | standard CV |
| Permutation p-value | lucky associations | Good (2000) |
| Stability selection | one-lucky-split effects | Meinshausen & Bühlmann (2010) |
| Benjamini–Hochberg FDR | false positives from many tests | Benjamini & Hochberg (1995) |

A stat is kept only if `q ≤ alpha AND stability ≥ min_stability AND cv_lift > 0`.

## Demonstrated behavior

`examples/run_winprob_screening.py` plants one real predictor (`clutch_stops`)
among pure noise plus a `rare_event` present in ~6% of games. The filter keeps
`clutch_stops`, rejects the noise, and auto-rejects `rare_event` for
insufficient sample — i.e. it does exactly the right thing on the
low-frequency trap you were worried about.

## Workflow

```python
from fastbreak.winprob import PredictivityFilter, WinProbabilityModel, WinImpactPipeline

filt = PredictivityFilter(min_sample=40, alpha=0.05, min_stability=0.6)
results = filt.screen(historical_snapshots, candidate_stats)   # OFFLINE
keep = filt.passing(results)                                    # validated whitelist

model = WinProbabilityModel(stat_names=[r.name for r in keep]).fit(historical_snapshots)
pipe  = WinImpactPipeline(model, keep, min_delta_wp=0.03)       # LIVE
for insight in pipe.run(live_snapshots):
    ...  # insight.stat, insight.win_prob, insight.delta_wp, insight.q_value
```

## Production notes
- Pure-Python logistic regression keeps the core dependency-free and lets the
  screening run anywhere. For real datasets, swap in scikit-learn / xgboost
  (the `ml` extra) — the `fit / predict_proba / log_loss` interface matches.
- With numpy you can afford full k-fold permutations (the pure-Python version
  uses a single train/test split for the null to stay fast).
- Add isotonic/Platt calibration + a reliability diagram before trusting the WP
  numbers publicly.
- Re-screen periodically: predictive value drifts with the league / roster.
