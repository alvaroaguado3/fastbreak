"""Runtime configuration, loaded from environment with safe defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _b(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    generator: str = os.environ.get("FASTBREAK_GENERATOR", "template")
    neural_model_path: str = os.environ.get("FASTBREAK_NEURAL_MODEL_PATH", "checkpoints/headline-lora")
    surprise_threshold: float = _f("FASTBREAK_SURPRISE_THRESHOLD", 0.62)
    percentile_threshold: float = _f("FASTBREAK_PERCENTILE_THRESHOLD", 0.90)
    min_sample_size: int = int(_f("FASTBREAK_MIN_SAMPLE_SIZE", 30))
    publish_dry_run: bool = _b("FASTBREAK_PUBLISH_DRY_RUN", True)
    max_flags_per_headline: int = int(_f("FASTBREAK_MAX_FLAGS_PER_HEADLINE", 3))

    @classmethod
    def from_env(cls) -> "Config":
        return cls()
