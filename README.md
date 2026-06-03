# FastBreak 🏀⚡

**A low-latency engine that turns live NBA play-by-play into shareable insights.**

As stats stream in during a game, FastBreak (1) detects when a player/team stat lands in a high percentile *given the game context* (quarter, clock, playoff vs. regular vs. Finals, opponent, venue), (2) generates candidate headlines for one or more co-occurring stats, (3) scores them by **surprise** (information content + novelty + quality) and publishes only the ones worth sharing, and (4) records human reuse to recalibrate itself over time.

Built so the whole hot path is `O(log n)` plus one small model call. The core runtime is **dependency-free (Python stdlib only)** — `torch`/`requests` are optional extras.

```
feed → detect (percentile + surprisal) → coalesce → generate → select (surprise) → publish → feedback ↺
```

---

## Quickstart (no API keys, no GPU)

```bash
git clone <your-repo-url> fastbreak && cd fastbreak
python examples/run_replay_demo.py            # replays a sample Knicks–Spurs Finals game
python examples/run_replay_demo.py --x        # route to the X publisher in dry-run mode
```

Expected output (abridged):

```
HEADLINE [template] surprise=0.95 novelty=0.84
> Jalen Brunson stuffing the sheet with 28 plus/minus and 41 points at Madison Square Garden.
  (Jalen Brunson 28 plus_minus (p99, 7.6b), Jalen Brunson 41 points (p96, 4.6b))  gen=0.0ms
```

Run the tests:

```bash
pip install -e ".[dev]"
pytest -q
```

---

## How it works

### 1. Data feed (`fastbreak/feed/`)
Everything sits behind a `DataFeed` interface that yields `StatEvent`s as early as possible (no batching). Three implementations ship:

- **`ReplayFeed`** — replays a JSON play-by-play file, instantly or paced in real time. Your dev/test workhorse; needs nothing installed.
- **`SportradarPushFeed`** — adapter for Sportradar's *Push Statistics / Push Events* feeds (newline-delimited JSON over a long-lived connection — the lowest-latency public option). Requires a Realtime contract + `pip install -e ".[live]"`.
- Add Genius Sports (the official NBA data partner) the same way — implement `events()`, done.

> **Latency reality check:** the model is *not* your bottleneck — the data feed is. A push/websocket feed is sub-second; the free unofficial `stats.nba.com` endpoints are polling-based with 10–60 s delays and rate limits, so they're fine for dev but not for "as low latency as possible." Pick the feed first; the rest of the system is already fast.

### 2. Detection (`fastbreak/percentile/`)
For each incoming stat we look up a historical distribution conditioned on context and ask: *how extreme is this, here?*

- **`GameContext.context_keys()`** produces a broad→specific list: `all` → `finals` → `finals:q4` → `finals:q4:clutch` → `finals:vs:San Antonio Spurs` → `venue:Madison Square Garden` → `finals:q4:venue:Madison Square Garden`.
- **`AnomalyDetector`** tries the *most specific* context that has at least `min_sample_size` historical instances (your "minimum number of instances" rule) and clears the percentile bar.
- We report two numbers: the **percentile** and the **surprisal in bits** = `−log₂ P(X ≥ value)` (Shannon self-information). Add-one smoothing keeps a brand-new record finite. Surprisal is what feeds the surprise score downstream.
- Distributions are non-parametric empirical quantiles (sorted-sample + bisect). Counting stats are bounded by the clock and heavily skewed, so a Gaussian fit is wrong in exactly the tail we care about. Swap in a t-digest (Dunning & Ertl, 2019) for unbounded streaming without touching the detector.

### 3. Coalescing + generation (`fastbreak/generation/`)
The pipeline buffers flags from the same game *moment* (same quarter + 6-second clock bucket) so a single possession that produces several extreme stats becomes **one multi-stat headline** — then flushes the instant the moment advances, so single-stat stories add zero latency.

Two generators behind one `HeadlineGenerator` interface:

- **`TemplateGenerator`** — deterministic, instant, zero training data. Handles single-stat, one-player-multi-stat, and two-player-duel framings. This is day one *and* the permanent fallback.
- **`NeuralGenerator`** — a small base model (e.g. Qwen2.5-0.5B) + a LoRA adapter, prompted with the *same* structured features the templates use. Lazily imports `torch`; `available()` falls back to templates if the adapter or `ml` extras are missing, so the live system never hard-fails.

### 4. Selection (`fastbreak/selection/`)
`SurpriseSelector` scores each candidate with a small logistic model over three features:

1. **information content** — aggregated surprisal bits, squashed so ~p90 ≈ 0.5;
2. **novelty** — token-Jaccard distance from recently published headlines (don't tweet five variations of the same run);
3. **quality** — length sanity + has-a-number.

Only the best candidate above `surprise_threshold` is published. The weights are exposed and updated online from human reuse via `update_from_feedback`, so the surprise score becomes a *calibrated predictor of resharing* (Platt-style calibration; Shannon 1948 for self-information).

### 5. Publishing (`fastbreak/publish/`)
`XPublisher` posts via X API v2. **It defaults to dry-run** (formats + logs, hits nothing, costs nothing). See [X API notes](#x-api-notes-2026) before going live.

### 6. Feedback loop (`fastbreak/feedback/`)
Every published headline is appended to a JSONL log. When a journalist reuses it / it gets reposted, call `record_signal(headline_id, reused=True)`. `iter_labeled()` then yields `(record, reused)` pairs that feed both the selector's online update **and** the next LoRA training run (reused headlines = positive examples).

---

## Going live

```bash
# 1. Live data
pip install -e ".[live]"
export SPORTRADAR_API_KEY=...           # Realtime customer

# 2. (optional) train + enable the neural generator
pip install -e ".[ml]"
python training/generate_synthetic.py --n 4000
python -m training.train_lora --base Qwen/Qwen2.5-0.5B --out checkpoints/headline-lora
export FASTBREAK_GENERATOR=neural
export FASTBREAK_NEURAL_MODEL_PATH=checkpoints/headline-lora

# 3. enable real X posting (read the cost note first!)
export FASTBREAK_PUBLISH_DRY_RUN=false
export X_API_KEY=... X_API_SECRET=... X_ACCESS_TOKEN=... X_ACCESS_TOKEN_SECRET=...
```

Wire your own loop (see `fastbreak/cli.py` for the full version):

```python
from fastbreak.feed import SportradarPushFeed
from fastbreak.percentile import AnomalyDetector, DistributionStore
from fastbreak.publish import XPublisher
from fastbreak.pipeline import Pipeline
from fastbreak.models import GameContext

ctx = GameContext("2026-finals-g1", "New York Knicks", "San Antonio Spurs",
                  game_type="finals", venue="Madison Square Garden",
                  opponent="San Antonio Spurs")
store = DistributionStore.load("data/distributions.json")   # precomputed offline
pipe = Pipeline(
    feed=SportradarPushFeed(ctx),
    detector=AnomalyDetector(store, percentile_threshold=0.90, min_sample_size=30),
    publisher=XPublisher(),     # dry-run unless FASTBREAK_PUBLISH_DRY_RUN=false
)
for headline in pipe.run():
    ...   # already published; do extra logging/alerting here
```

---

## X API notes (2026)

The X API changed materially in 2026 — design accordingly:

- **No free tier for new developers** (since Feb 2026). Default is **pay-per-use**.
- A standard write is **~$0.015/post**; a post **containing a URL is ~$0.20/post** (≈13× more).
- Because of that, `XPublisher` **strips URLs from the post body by default** and runs in **dry-run** until you opt in.

Budget before you flip `FASTBREAK_PUBLISH_DRY_RUN=false`. See the cost discussion in `docs/ARCHITECTURE.md`.

---

## Project layout

```
fastbreak/
├── src/fastbreak/
│   ├── models.py            # GameContext, StatEvent, StatFlag, HeadlineCandidate, SelectedHeadline
│   ├── config.py            # env-driven config
│   ├── pipeline.py          # orchestrator (feed→detect→coalesce→generate→select→publish)
│   ├── cli.py               # `fastbreak-demo`
│   ├── feed/                # DataFeed: replay + Sportradar
│   ├── percentile/          # empirical distributions + surprisal detector
│   ├── generation/          # template + neural (LoRA) headline generators
│   ├── selection/           # surprise/entropy selector
│   ├── publish/             # console + X connector (dry-run default)
│   └── feedback/            # JSONL reuse log (retro-feed loop)
├── training/                # synthetic bootstrap + LoRA fine-tuning harness
├── tests/                   # pytest (stdlib-only, fast)
├── examples/run_replay_demo.py
├── docs/                    # ARCHITECTURE, DATA_FEED, TRAINING, CONTRIBUTING
└── .github/workflows/ci.yml
```

## Configuration

All via environment (see `.env.example`): `FASTBREAK_GENERATOR`, `FASTBREAK_SURPRISE_THRESHOLD`, `FASTBREAK_PERCENTILE_THRESHOLD`, `FASTBREAK_MIN_SAMPLE_SIZE`, `FASTBREAK_PUBLISH_DRY_RUN`, plus `SPORTRADAR_*` and `X_*` credentials.

## References

- Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal.
- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Dunning, T. & Ertl, O. (2019). *Computing Extremely Accurate Quantiles Using t-Digests.* arXiv:1902.04023.
- Platt, J. (1999). *Probabilistic Outputs for SVMs and Comparisons to Regularized Likelihood Methods.*

## License

MIT — see `LICENSE`.

> Built for the (hypothetical, gloriously fun) Knicks vs. Spurs Finals. Not affiliated with the NBA, Sportradar, or X.
