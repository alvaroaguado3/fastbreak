"""Bootstrap a seed dataset to solve the cold-start problem.

You have no human-labeled headlines on day one. So we generate (prompt -> headline)
pairs by sampling plausible game situations, running the SAME template engine the
live system falls back to, and recording the structured prompt + the template
headline as the target. This teaches the LoRA model to reproduce, then generalize
beyond, the template phrasings.

Once the live system has run and the FeedbackStore has human-reuse labels, mix
those REAL reused headlines in (and upweight them) for a far better model. This
file produces the v0 dataset to get you off the ground.

Output: data/headlines_synth.jsonl  with {"prompt": ..., "completion": ...}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastbreak.generation.neural import build_prompt  # noqa: E402
from fastbreak.generation.templates import TemplateGenerator  # noqa: E402
from fastbreak.models import GameContext, StatEvent, StatFlag  # noqa: E402

_PLAYERS = [
    ("Jalen Brunson", "New York Knicks"), ("Victor Wembanyama", "San Antonio Spurs"),
    ("Karl-Anthony Towns", "New York Knicks"), ("Stephon Castle", "San Antonio Spurs"),
    ("OG Anunoby", "New York Knicks"), ("Devin Vassell", "San Antonio Spurs"),
]
_STATS = ["points", "rebounds", "assists", "blocks", "steals", "threes", "plus_minus"]
_GAME_TYPES = ["regular", "playoff", "finals"]
_VENUES = ["Madison Square Garden", "Frost Bank Center"]


def _rand_flag(rng: random.Random) -> StatFlag:
    name, team = rng.choice(_PLAYERS)
    opp = "San Antonio Spurs" if team == "New York Knicks" else "New York Knicks"
    stat = rng.choice(_STATS)
    gt = rng.choice(_GAME_TYPES)
    q = rng.randint(1, 4)
    ctx = GameContext(
        game_id="synth", home_team="New York Knicks", away_team="San Antonio Spurs",
        game_type=gt, quarter=q, clock_seconds=rng.randint(0, 720),
        venue=rng.choice(_VENUES), city="New York", state="NY", opponent=opp,
    )
    if stat == "plus_minus":
        value = rng.randint(10, 35)
    elif stat in ("blocks", "steals", "threes"):
        value = rng.randint(4, 10)
    else:
        value = rng.randint(12, 50)
    ev = StatEvent(name.lower(), name, "player", stat, float(value), ctx)
    pct = rng.uniform(0.90, 0.999)
    bits = rng.uniform(3.3, 9.0)
    ck = rng.choice([gt, f"{gt}:q{q}", f"venue:{ctx.venue}", f"{gt}:vs:{opp}", f"{gt}:q4:clutch"])
    return StatFlag(event=ev, percentile=pct, matched_context=ck,
                    sample_size=rng.randint(40, 500), surprisal_bits=bits)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default="data/headlines_synth.jsonl")
    ap.add_argument("--multi-frac", type=float, default=0.3, help="fraction of multi-stat stories")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    gen = TemplateGenerator()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out.open("w") as f:
        for _ in range(args.n):
            if rng.random() < args.multi_frac:
                flags = [_rand_flag(rng) for _ in range(rng.randint(2, 3))]
            else:
                flags = [_rand_flag(rng)]
            cands = gen.generate(flags, n=4)
            if not cands:
                continue
            target = rng.choice(cands).text  # vary phrasing across samples
            f.write(json.dumps({"prompt": build_prompt(flags), "completion": " " + target}) + "\n")
            written += 1
    print(f"wrote {written} examples -> {out}")


if __name__ == "__main__":
    main()
