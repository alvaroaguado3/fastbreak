"""Console publisher -- prints headlines. Useful for demos and as a tee."""
from __future__ import annotations

from ..models import SelectedHeadline
from .base import Publisher


class ConsolePublisher(Publisher):
    def publish(self, headline: SelectedHeadline) -> dict:
        c = headline.candidate
        flags = ", ".join(
            f"{f.event.subject_name} {int(f.event.value)} {f.stat} "
            f"(p{int(f.percentile*100)}, {f.surprisal_bits:.1f}b)" for f in c.flags
        )
        print(
            f"\n  HEADLINE [{c.source}] surprise={headline.surprise_score:.2f} "
            f"novelty={headline.novelty_score:.2f}\n"
            f"  > {c.text}\n"
            f"    ({flags})  gen={c.gen_latency_ms:.1f}ms"
        )
        return {"published": True, "channel": "console"}
