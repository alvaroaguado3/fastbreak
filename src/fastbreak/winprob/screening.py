"""Predictivity filter -- the guard against spurious low-frequency signal.

Question: does a candidate stat predict winning BEYOND score margin + time?

A stat is only kept if it clears ALL of:
  1. Minimum sample size               -> no claims from a handful of cases.
  2. Out-of-sample CV lift > 0         -> it must improve HELD-OUT prediction,
                                          not in-sample fit (which always rises).
  3. Permutation test p-value          -> beat its own shuffled null, so the
                                          lift isn't luck. (Good, 2000)
  4. Stability selection               -> help must be CONSISTENT across
                                          resamples, not one lucky split.
                                          (Meinshausen & Buhlmann, 2010)
  5. Benjamini-Hochberg FDR            -> correct for testing many stats at
                                          once. (Benjamini & Hochberg, 1995)

The baseline model is score margin + time (the dominant, well-known predictor;
Stern 1994). Lift is measured against THAT, so we never credit a stat for
information the scoreboard already carried.

Runs OFFLINE on historical games; emits a validated whitelist the live pipeline
consumes. Pure-Python so it runs anywhere; for large-scale screening swap
LogisticRegression for sklearn/xgboost in the `ml` extra.

PERFORMANCE NOTE: a full k-fold permutation test in pure Python is too slow, so
the permutation null uses a single fixed train/test split (the standard fast
form of permutation importance). The reported cv_lift still comes from k-fold;
only the p-value uses the split. With numpy/sklearn you can afford full k-fold
permutations -- the interface is unchanged.
"""
from __future__ import annotations

import random

from .logistic import LogisticRegression
from .models import GameSnapshot, PredictiveStat


def _fit_eval(Xtr, ytr, Xte, yte, l2, epochs, seed):
    if len(set(ytr)) < 2 or not Xte:
        return float("inf")
    m = LogisticRegression(l2=l2, epochs=epochs, seed=seed).fit(Xtr, ytr)
    return LogisticRegression.log_loss(yte, m.predict_proba(Xte))


def _kfold_logloss(X, y, k, l2, epochs, seed):
    idx = list(range(len(X)))
    random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    losses = []
    for f in range(k):
        test = set(folds[f])
        Xtr = [X[i] for i in idx if i not in test]
        ytr = [y[i] for i in idx if i not in test]
        Xte = [X[i] for i in folds[f]]
        yte = [y[i] for i in folds[f]]
        loss = _fit_eval(Xtr, ytr, Xte, yte, l2, epochs, seed)
        if loss != float("inf"):
            losses.append(loss)
    return sum(losses) / len(losses) if losses else float("inf")


def _bh_qvalues(pvals):
    """Benjamini-Hochberg adjusted p-values (preserves input order)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(prev, pvals[i] * m / (rank + 1))
        q[i] = val
        prev = val
    return q


class PredictivityFilter:
    def __init__(self, min_sample=40, k_folds=5, n_permutations=100,
                 n_bootstrap=15, alpha=0.05, min_stability=0.6, l2=1.0,
                 epochs=60, test_frac=0.3, seed=0):
        self.min_sample = min_sample
        self.k = k_folds
        self.n_perm = n_permutations
        self.n_boot = n_bootstrap
        self.alpha = alpha
        self.min_stability = min_stability
        self.l2 = l2
        self.epochs = epochs
        self.test_frac = test_frac
        self.seed = seed

    def screen(self, snapshots, candidate_stats):
        labeled = [s for s in snapshots if s.won is not None]
        y = [int(s.won) for s in labeled]
        Xbase = [s.base_features() for s in labeled]

        base_cv = _kfold_logloss(Xbase, y, self.k, self.l2, self.epochs, self.seed)
        split = self._make_split(len(labeled))
        base_split_loss = self._split_baseline_loss(Xbase, y, split)

        results, pvals = [], []
        for stat in candidate_stats:
            col = [float(s.stats.get(stat, 0.0)) for s in labeled]
            n_present = sum(1 for s in labeled if stat in s.stats)
            if n_present < self.min_sample:
                results.append(PredictiveStat(stat, 0.0, 1.0, 1.0, 0.0, 0.0, n_present))
                pvals.append(1.0)
                continue

            Xaug = [row + [c] for row, c in zip(Xbase, col)]
            cv_lift = base_cv - _kfold_logloss(Xaug, y, self.k, self.l2, self.epochs, self.seed)
            p = self._permutation_p(Xbase, col, y, split, base_split_loss)
            stab = self._stability(Xbase, col, y)
            effect = self._effect(Xaug, y)
            results.append(PredictiveStat(stat, cv_lift, p, 1.0, stab, effect, n_present))
            pvals.append(p)

        for r, q in zip(results, _bh_qvalues(pvals)):
            r.q_value = q
        results.sort(key=lambda r: (r.q_value, -r.cv_lift))
        return results

    def passing(self, results):
        return [r for r in results if r.passed(self.alpha, self.min_stability)]

    # --- internals ---
    def _make_split(self, n):
        idx = list(range(n))
        random.Random(self.seed + 7).shuffle(idx)
        cut = int(n * (1 - self.test_frac))
        return idx[:cut], idx[cut:]

    def _split_baseline_loss(self, Xbase, y, split):
        tr, te = split
        return _fit_eval([Xbase[i] for i in tr], [y[i] for i in tr],
                         [Xbase[i] for i in te], [y[i] for i in te],
                         self.l2, self.epochs, self.seed)

    def _aug_split_loss(self, Xbase, col, y, split):
        tr, te = split
        Xtr = [Xbase[i] + [col[i]] for i in tr]
        Xte = [Xbase[i] + [col[i]] for i in te]
        return _fit_eval(Xtr, [y[i] for i in tr], Xte, [y[i] for i in te],
                         self.l2, self.epochs, self.seed)

    def _permutation_p(self, Xbase, col, y, split, base_split_loss):
        obs_lift = base_split_loss - self._aug_split_loss(Xbase, col, y, split)
        rng = random.Random(self.seed + 1)
        shuffled = list(col)
        ge = 0
        for _ in range(self.n_perm):
            rng.shuffle(shuffled)
            lift = base_split_loss - self._aug_split_loss(Xbase, shuffled, y, split)
            if lift >= obs_lift:
                ge += 1
        return (1 + ge) / (1 + self.n_perm)

    def _stability(self, Xbase, col, y):
        rng = random.Random(self.seed + 2)
        n = len(y)
        helped, runs = 0, 0
        for b in range(self.n_boot):
            samp = [rng.randrange(n) for _ in range(n)]
            in_samp = set(samp)
            oob = [i for i in range(n) if i not in in_samp]
            ytr = [y[i] for i in samp]
            if not oob or len(set(ytr)) < 2:
                continue
            lb = _fit_eval([Xbase[i] for i in samp], ytr,
                           [Xbase[i] for i in oob], [y[i] for i in oob],
                           self.l2, self.epochs, b)
            la = _fit_eval([Xbase[i] + [col[i]] for i in samp], ytr,
                           [Xbase[i] + [col[i]] for i in oob], [y[i] for i in oob],
                           self.l2, self.epochs, b)
            helped += 1 if (lb - la) > 0 else 0
            runs += 1
        return helped / runs if runs else 0.0

    def _effect(self, Xaug, y):
        m = LogisticRegression(l2=self.l2, epochs=self.epochs, seed=self.seed).fit(Xaug, y)
        return m.coef()[-1]
