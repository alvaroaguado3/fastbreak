"""Deterministic template generator -- fast, reliable, zero training data.

This is the day-one generator and the permanent fallback when the neural model
is unavailable or low-confidence. It encodes the same "feature -> phrasing"
mapping you would later distill into the LoRA model, which also makes it a
convenient source of synthetic training pairs (see training/generate_synthetic.py).
"""
from __future__ import annotations

import time

from ..models import HeadlineCandidate, StatFlag

# Human-readable phrasing per stat
_STAT_NOUN = {
    "points": "points", "rebounds": "rebounds", "assists": "assists",
    "blocks": "blocks", "steals": "steals", "threes": "threes",
    "plus_minus": "plus/minus", "turnovers": "turnovers",
}

_CONTEXT_PHRASE = {
    "all": "all-time",
    "regular": "in the regular season",
    "playoff": "in the playoffs",
    "finals": "in the Finals",
}


def _ctx_phrase(ck: str) -> str:
    if ck in _CONTEXT_PHRASE:
        return _CONTEXT_PHRASE[ck]
    if ":q4:clutch" in ck:
        return "in clutch time"
    if "venue:" in ck:
        return f"at {ck.split('venue:')[-1]}"
    if ":vs:" in ck:
        return f"against the {ck.split(':vs:')[-1]}"
    if ck.endswith(("q1", "q2", "q3", "q4")):
        q = ck[-1]
        return f"by the end of Q{q}"
    return ""


def _pct_word(p: float) -> str:
    if p >= 0.99:
        return "an all-time"
    if p >= 0.97:
        return "an elite"
    if p >= 0.90:
        return "a top-tier"
    return "a strong"


class TemplateGenerator:
    source = "template"

    def generate(self, flags: list[StatFlag], n: int = 4) -> list[HeadlineCandidate]:
        t0 = time.perf_counter()
        if not flags:
            return []
        flags = sorted(flags, key=lambda f: f.surprisal_bits, reverse=True)
        cands: list[str] = []

        if len(flags) == 1:
            cands = self._single(flags[0])
        else:
            cands = self._multi(flags)

        latency = (time.perf_counter() - t0) * 1000.0
        # de-dup, keep order, cap at n
        seen, out = set(), []
        for text in cands:
            if text not in seen:
                seen.add(text)
                out.append(HeadlineCandidate(text=text, flags=flags,
                                             source=self.source, gen_latency_ms=latency))
            if len(out) >= n:
                break
        return out

    def _single(self, f: StatFlag) -> list[str]:
        name = f.event.subject_name
        val = int(f.event.value) if f.event.value == int(f.event.value) else round(f.event.value, 1)
        noun = _STAT_NOUN.get(f.stat, f.stat)
        cphrase = _ctx_phrase(f.matched_context)
        pctile = int(round(f.percentile * 100))
        pword = _pct_word(f.percentile)
        return [
            f"{name} has {val} {noun} {cphrase} — {pword} mark.".replace("  ", " "),
            f"{val} {noun} for {name}. That's {pctile}th-percentile {cphrase}.".replace("  ", " "),
            f"{name} is putting up {val} {noun} {cphrase}, a {f.surprisal_bits:.1f}-bit outlier.".replace("  ", " "),
            f"Watch {name}: {val} {noun} already, top {100 - pctile}% {cphrase}.".replace("  ", " "),
        ]

    def _multi(self, flags: list[StatFlag]) -> list[str]:
        # group by subject; if one player owns multiple flags, combine them
        by_subject: dict[str, list[StatFlag]] = {}
        for f in flags:
            by_subject.setdefault(f.event.subject_name, []).append(f)

        # Case A: single hot player across several stats
        hottest_name = max(by_subject, key=lambda k: sum(x.surprisal_bits for x in by_subject[k]))
        group = by_subject[hottest_name]
        if len(group) >= 2:
            parts = [f"{int(g.event.value)} {_STAT_NOUN.get(g.stat, g.stat)}" for g in group]
            joined = ", ".join(parts[:-1]) + f" and {parts[-1]}"
            cphrase = _ctx_phrase(group[0].matched_context)
            return [
                f"{hottest_name} is everywhere: {joined} {cphrase}.".replace("  ", " "),
                f"Stat line alert — {hottest_name}: {joined}, all top-tier {cphrase}.".replace("  ", " "),
                f"{hottest_name} stuffing the sheet with {joined} {cphrase}.".replace("  ", " "),
            ]

        # Case B: different players each hot -> dual-subject headline
        top2 = flags[:2]
        a, b = top2[0], top2[1]
        return [
            (f"Duel in the Finals: {a.event.subject_name} ({int(a.event.value)} "
             f"{_STAT_NOUN.get(a.stat, a.stat)}) vs {b.event.subject_name} "
             f"({int(b.event.value)} {_STAT_NOUN.get(b.stat, b.stat)})."),
            (f"{a.event.subject_name} and {b.event.subject_name} are both going off — "
             f"{int(a.event.value)} {_STAT_NOUN.get(a.stat, a.stat)} and "
             f"{int(b.event.value)} {_STAT_NOUN.get(b.stat, b.stat)}."),
        ]
