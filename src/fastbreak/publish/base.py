"""Publisher interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import SelectedHeadline


class Publisher(ABC):
    @abstractmethod
    def publish(self, headline: SelectedHeadline) -> dict:
        """Publish a headline. Returns a provider response / record dict."""
        raise NotImplementedError
