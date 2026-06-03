# FastBreak — Roadmap & open decisions

A prioritized list of what stands between the current scaffold and something
you'd trust live. Tiers are ordered by impact-on-risk, not by effort.

---

## Tier 0 — Blockers (do these before trusting any live output)

- [ ] **Offline ETL for real historical distributions.** The detector is only
  as good as `data/distributions.json`. Build a job that replays a season+ of
  play-by-play and, for each `(stat, context_key)`, records the *stat-so-far at
  the same game-moment* (e.g. points by end of Q3, blocks in clutch). Save via
  `DistributionStore.save()`. Today `seed_synthetic_store()` fakes this.
  *Why first: every percentile claim downstream depends on it.*

- [ ] **Multiple-comparisons / false-discovery control.** Every possession you
  test dozens of (player × stat × context) cells at p90 with a fallback ladder,
  so you WILL fire on noise. Add a Benjamini–Hochberg FDR gate and/or a
  per-game "interesting moments" budget. *Single biggest quality risk.*

- [ ] **Faithfulness guard on the neural generator.** Hard-check that every
  number and subject name in a generated headline appears in the input flags;
  fail closed to the template if not. An LLM that prints "42" when the stat is
  "41" is worse than no system. (Stub noted in `training/README.md`.)

## Tier 1 — Core quality

- [ ] **Decide & wire the production feed.** Sportradar/Genius push (sub-second,
  paid) vs `stats.nba.com` polling (10–60 s, free). The feed is your latency
  floor — settle this and price it before optimizing anything else.
- [ ] **Calibrate the surprise score.** Replace the hand-set logistic weights
  with a fit on real reuse labels; check a reliability curve so `surprise=0.8`
  ≈ 80 % reuse odds. Use `FeedbackStore.iter_labeled()` as the training set.
- [ ] **Be disciplined about context depth.** Every context dimension halves
  sample sizes. Keep `min_sample_size` honest; drop context keys that never
  reach it. Audit which keys actually fire on real data.
- [ ] **Define the reuse signal concretely.** "A journalist used it" is hard to
  observe; "reposted/liked on our own X account" is easy. Pick what you can
  actually measure — it's the label the whole feedback loop optimizes.

## Tier 2 — Model & scale

- [ ] **Train the LoRA model only once templates prove insufficient.** Ship
  templates first; collect reuse data; then fine-tune and A/B against templates.
- [ ] **Latency tuning for neural path.** GPU, ≤24 new tokens, batch candidates;
  measure p50/p99 generation latency under game load.
- [ ] **Streaming distribution updates.** Swap `EmpiricalDistribution` for a
  t-digest so distributions can update online instead of a nightly rebuild.
- [ ] **Per-stat thresholds.** Let the feedback loop learn different surprise
  thresholds per stat type (a 9-block game is rarer-news than 30 points).

## Tier 3 — Product / ops

- [ ] **X cost guardrails.** Pay-per-use in 2026 (~$0.015/post, ~$0.20 if it
  has a URL). Add a daily spend cap + dry-run-by-default in CI/staging.
- [ ] **Additional channels.** Bluesky / Threads / internal Slack `Publisher`s.
- [ ] **Observability.** Log per-stage latency, fire-rate, and reuse-rate to a
  dashboard; alert if fire-rate spikes (usually a distribution bug).
- [ ] **Backtesting harness.** Replay full historical games and eyeball which
  headlines would have fired — fastest way to catch over/under-firing.

---

## Open questions to resolve (answer these in a few days)

1. What's the budget for a real-time data feed, and which provider?
2. What reuse signal can you actually collect, and from where?
3. Who is the consumer — your own social account, or a newsroom tool? (changes
   the latency bar and the surprise calibration target.)
4. Do you need the neural generator for v1 at all, or are templates enough?
5. How much historical play-by-play can you get access to for the ETL + the
   win-probability screening (see the new `winprob/` module)?

---

## New: Win-probability module (`winprob/`) — see its own backlog

- [ ] Replace pure-python logistic regression with sklearn/xgboost in the `ml`
  extra for the production screener.
- [ ] Build the offline screening job on REAL games and persist the validated
  predictive-stat whitelist.
- [ ] Add isotonic/Platt calibration + reliability diagrams to the WP model.
- [ ] Compare against the canonical baseline (score differential + time only) —
  any stat that doesn't beat that baseline out-of-sample is not predictive.
