"""X (Twitter) API v2 publisher.

IMPORTANT 2026 pricing reality (designed around it):
  - There is no free tier for new developers; it's pay-per-use. A standard
    write is ~$0.015; a post CONTAINING A URL is ~$0.20. So this connector
    keeps headlines URL-free by default to avoid the 13x surcharge, and runs
    in DRY-RUN mode unless explicitly enabled.

Dry-run (default): formats and logs the tweet, hits nothing, costs nothing.
Live: posts via POST /2/tweets using OAuth 1.0a user context.

Enable live posting by setting FASTBREAK_PUBLISH_DRY_RUN=false AND providing
the four OAuth1 credentials in the environment. Requires `pip install -e ".[live]"`.
"""
from __future__ import annotations

import os
import re

from ..models import SelectedHeadline
from .base import Publisher

_URL_RE = re.compile(r"https?://\S+")
TWEET_LIMIT = 280


class XPublisher(Publisher):
    def __init__(self, dry_run: bool | None = None, strip_urls: bool = True,
                 hashtags: tuple[str, ...] = ("#NBAFinals",)):
        env_dry = os.environ.get("FASTBREAK_PUBLISH_DRY_RUN", "true").lower() in {"1", "true", "yes", "on"}
        self.dry_run = env_dry if dry_run is None else dry_run
        self.strip_urls = strip_urls
        self.hashtags = hashtags

    def format_tweet(self, headline: SelectedHeadline) -> str:
        text = headline.candidate.text
        if self.strip_urls:
            text = _URL_RE.sub("", text).strip()
        tags = " " + " ".join(self.hashtags) if self.hashtags else ""
        if len(text) + len(tags) > TWEET_LIMIT:
            text = text[: TWEET_LIMIT - len(tags) - 1].rstrip() + "…"
        return f"{text}{tags}"

    def publish(self, headline: SelectedHeadline) -> dict:
        tweet = self.format_tweet(headline)
        if self.dry_run:
            print(f"[X dry-run] would post ({len(tweet)} chars): {tweet}")
            return {"published": False, "dry_run": True, "text": tweet}
        return self._post_live(tweet)

    def _post_live(self, tweet: str) -> dict:  # pragma: no cover - needs creds
        try:
            from requests_oauthlib import OAuth1Session
        except ImportError as e:
            raise RuntimeError('Live X posting needs: pip install requests-oauthlib') from e

        creds = {k: os.environ.get(k) for k in (
            "X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET")}
        if not all(creds.values()):
            raise RuntimeError("Missing X OAuth1 credentials in environment.")

        oauth = OAuth1Session(
            creds["X_API_KEY"], client_secret=creds["X_API_SECRET"],
            resource_owner_key=creds["X_ACCESS_TOKEN"],
            resource_owner_secret=creds["X_ACCESS_TOKEN_SECRET"],
        )
        resp = oauth.post("https://api.twitter.com/2/tweets", json={"text": tweet})
        resp.raise_for_status()
        data = resp.json()
        return {"published": True, "dry_run": False, "id": data.get("data", {}).get("id"), "text": tweet}
