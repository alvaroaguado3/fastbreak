"""Feedback log -- the retro-feed loop's data layer.

Every published headline is appended as a JSONL record. External signals
(a journalist reused it, it got reposted, an editor killed it) are matched back
by `headline_id` and recorded as labels. `iter_labeled()` then yields
(SelectedHeadline-like record, reused: bool) pairs that the SurpriseSelector's
`update_from_feedback` consumes to recalibrate -- and that the LoRA trainer can
use as preference data (reused vs not).
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from ..models import SelectedHeadline


class FeedbackStore:
    def __init__(self, path: str | Path = "data/feedback.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_published(self, headline: SelectedHeadline) -> str:
        rec = headline.to_record()
        hid = uuid.uuid4().hex[:12]
        rec["headline_id"] = hid
        rec["reused"] = None  # filled in later
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return hid

    def record_signal(self, headline_id: str, reused: bool, source: str = "manual") -> None:
        """Append a label event (we keep an append-only log; readers take the last)."""
        with self.path.open("a") as f:
            f.write(json.dumps({
                "headline_id": headline_id, "reused": reused,
                "signal_source": source, "ts": time.time(), "_label": True,
            }) + "\n")

    def iter_labeled(self) -> Iterator[tuple[dict, bool]]:
        """Yield (record, reused) for headlines that received a label."""
        records: dict[str, dict] = {}
        labels: dict[str, bool] = {}
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("_label"):
                labels[obj["headline_id"]] = obj["reused"]
            elif "headline_id" in obj:
                records[obj["headline_id"]] = obj
        for hid, rec in records.items():
            if hid in labels:
                yield rec, labels[hid]
