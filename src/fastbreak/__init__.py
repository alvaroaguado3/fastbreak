"""FastBreak: low-latency live NBA insight engine."""
__version__ = "0.1.0"

from .models import GameContext, StatEvent, StatFlag, HeadlineCandidate, SelectedHeadline

__all__ = [
    "GameContext",
    "StatEvent",
    "StatFlag",
    "HeadlineCandidate",
    "SelectedHeadline",
    "__version__",
]
