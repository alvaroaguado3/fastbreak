# Contributing

## Setup
```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

## Conventions
- Core (`src/fastbreak`, excluding `generation/neural.py`, `feed/sportradar.py`, `publish/x_connector.py` live paths) must stay **stdlib-only**. Heavy deps are imported lazily inside the methods that need them.
- Every new component implements the relevant ABC (`DataFeed` / `HeadlineGenerator` / `Publisher`) and ships with a test.
- Keep the hot path allocation-light; prefer `dataclass(slots=True)`.

## Tests
Fast, deterministic, no network. Use `seed_synthetic_store()` for distributions and `ReplayFeed` for end-to-end. Live adapters (Sportradar, X-live) are excluded from unit tests by design (`# pragma: no cover`).

## Pull requests
CI (GitHub Actions) runs lint + tests on Python 3.10–3.12. Green CI required.
