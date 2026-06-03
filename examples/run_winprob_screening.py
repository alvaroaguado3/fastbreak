"""Win-probability screening demo. Run: python examples/run_winprob_screening.py

Generates synthetic games (one real predictor + noise + a low-frequency trap),
screens candidate stats, fits a WP model on the survivors, and streams a few
live win-impact insights. Pure stdlib -- no deps.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastbreak.winprob import (PredictivityFilter, WinProbabilityModel,
                               WinImpactPipeline)
from fastbreak.winprob.synth import make_games, CANDIDATE_STATS


def main():
    games = make_games(n_games=400)
    print(f"=== screening {len(CANDIDATE_STATS)} candidate stats on {len(games)} games ===\n")

    filt = PredictivityFilter(min_sample=50, n_permutations=200, n_bootstrap=40, alpha=0.05)
    results = filt.screen(games, CANDIDATE_STATS)

    print(f"{'stat':<14}{'cv_lift':>9}{'p':>8}{'q(FDR)':>9}{'stab':>7}{'effect':>8}{'n':>6}  verdict")
    print("-" * 78)
    for r in results:
        ok = r.passed(filt.alpha, filt.min_stability)
        print(f"{r.name:<14}{r.cv_lift:>9.4f}{r.p_value:>8.3f}{r.q_value:>9.3f}"
              f"{r.stability:>7.2f}{r.effect:>8.3f}{r.n:>6}  {'KEEP' if ok else 'reject'}")

    passing = filt.passing(results)
    keep = [p.name for p in passing]
    print(f"\nvalidated predictive stats: {keep or 'none'}")

    if not keep:
        print("nothing survived the filter -- exactly what you want when there's no real signal.")
        return

    model = WinProbabilityModel(stat_names=keep).fit(games)
    pipe = WinImpactPipeline(model, passing, min_delta_wp=0.02)
    print("\n=== live win-impact insights (first 5) ===")
    for i, ins in enumerate(pipe.run(games[:50])):
        if i >= 5:
            break
        print(f"  {ins.stat}: WP={ins.win_prob:.2f}  ΔWP={ins.delta_wp:+.3f}  "
              f"(effect={ins.effect:+.2f}, q={ins.q_value:.3f}, "
              f"score_diff={ins.snapshot.score_diff:+.0f})")


if __name__ == "__main__":
    main()
