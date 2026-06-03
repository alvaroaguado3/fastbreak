# Data feeds

## Interface
A feed yields `StatEvent`s; that's the whole contract. Yield as early as possible.

```python
class DataFeed(ABC):
    def events(self) -> Iterator[StatEvent]: ...
```

## ReplayFeed (development)
Reads a JSON file (`src/fastbreak/feed/sample_pbp.json` is the template). `speed=0` emits instantly (tests/benchmarks); `speed=1` is real time; `speed=10` is 10× faster. No dependencies.

## SportradarPushFeed (production)
- Requires a **Realtime** Sportradar contract + `SPORTRADAR_API_KEY` + `pip install -e ".[live]"`.
- Consumes the **Push Statistics** feed (running per-player box-score stats) as newline-delimited JSON over a long-lived HTTP connection — one connection, minimal calls, lowest latency.
- `_translate()` and `_STAT_MAP` map Sportradar keys to FastBreak stat names; the mapping is intentionally lenient because the nesting differs across v7/v8. Verify against your contracted version with the [Postman collection](https://www.postman.com/sportradar-media-apis/sportradar-media-apis/documentation/8u2ejra/sportradar-nba-v8) before trusting it in production.
- Docs: https://developer.sportradar.com/basketball/docs/nba-ig-api-basics

## Other options
- **Genius Sports** — official NBA real-time data partner; add an adapter the same way.
- **nba_api / stats.nba.com** — free but *polling* with 10–60 s delay + rate limits; fine for backfilling historical distributions, not for "low latency live."

## Building the historical distributions
The detector needs `data/distributions.json`. Offline, iterate a season+ of play-by-play, and for each (stat, context_key) accumulate the *in-game value at the same game-moment* (e.g., points-so-far at Q4/clutch). Save with `DistributionStore.save()`. Until you have that, `seed_synthetic_store()` gives you plausible distributions to develop against.
