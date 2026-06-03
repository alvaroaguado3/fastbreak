# Architecture

## Design goals (and how they're met)

| Goal | Mechanism |
|---|---|
| Low end-to-end latency | Per-event detection (no batching); `O(log n)` percentile via sorted-sample bisect; single small model call; stdlib-only hot path. |
| Context-aware "interesting" | Broad→specific context keys with a minimum-sample fallback in `AnomalyDetector`. |
| 1+ stats per headline | Moment-coalescing buffer in `Pipeline` (quarter + 6s clock bucket). |
| Only share the surprising | `SurpriseSelector`: surprisal bits + novelty + quality → logistic → threshold. |
| Learn from humans | `FeedbackStore` reuse labels → online weight update + next LoRA training set. |
| Swappable everything | ABCs: `DataFeed`, `HeadlineGenerator`, `Publisher`. |

## Data flow

```
StatEvent ─▶ AnomalyDetector.evaluate ─▶ StatFlag?
                                          │ (buffer by game moment)
                                          ▼
                        Pipeline._flush ─▶ HeadlineGenerator.generate ─▶ [HeadlineCandidate]
                                          ▼
                        SurpriseSelector.select ─▶ SelectedHeadline?
                                          ▼
                        Publisher.publish + FeedbackStore.log_published
                                          ▼ (later, async)
                        FeedbackStore.record_signal ─▶ SurpriseSelector.update_from_feedback
                                                     └▶ training/train_lora.py (nightly)
```

## Why these statistics

- **Percentile** answers "how rare, in this context?" but is scale-free and easy to communicate ("p97").
- **Surprisal** (`−log₂ P(X ≥ x)`) is additive across independent stats, which is exactly why a one-player-multi-stat line ("9 blocks AND 38 points") should score higher than either alone — and the selector aggregates bits across coalesced flags.
- We deliberately do **not** assume normality: in-game counting stats are right-skewed and clock-bounded, so empirical quantiles beat a parametric tail.

## Latency budget (rough, single moment, CPU)

| Stage | Typical |
|---|---|
| Detect (per event) | < 0.05 ms |
| Template generation | < 0.1 ms |
| Neural generation (0.5B + LoRA, 24 tok) | 20–80 ms GPU / 100–400 ms CPU |
| Selection | < 0.1 ms |
| **Feed delay (the real cost)** | **sub-second (push) … 10–60 s (polling)** |

Conclusion: invest in the feed and (if you need neural) a GPU + short generations. Everything else is already negligible.

## X API cost model (2026)

Pay-per-use: ~$0.015/write, **~$0.20/write if the post contains a URL**. At, say, 40 headlines/game URL-free, that's ~$0.60/game. Adding a link to each would be ~$8/game. `XPublisher` strips URLs by default; if you must link (e.g., to a highlight), do it as a reply or quote-tweet you've explicitly budgeted for.

## Extension points

- **New feed**: implement `DataFeed.events()` (Genius Sports, nba_api polling, Kafka topic).
- **New generator**: implement `HeadlineGenerator.generate()` (bigger model, RAG over player history).
- **New channel**: implement `Publisher.publish()` (Bluesky, Threads, internal Slack).
- **Better detection**: precompute real distributions offline; swap `EmpiricalDistribution` for a t-digest for streaming updates.
