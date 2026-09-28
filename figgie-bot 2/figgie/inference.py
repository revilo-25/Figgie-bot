"""Hand-only Bayesian inference of the goal suit.

Hypothesis = a full deck composition (which suit has 12, goal size 8/10, layout of the rest).
Prior is uniform over the 16 equally likely compositions the engine can deal.
Likelihood of our initial hand = multivariate hypergeometric: prod_s C(n_s, h_s) / C(40, k).
The C(40, k) denominator is common to all hypotheses, so it cancels in the posterior.
"""
from functools import lru_cache
from math import comb
from .engine import SUITS, PARTNER


def _hypotheses():
    hyps = []
    for common in SUITS:
        goal = PARTNER[common]
        others = [s for s in SUITS if s not in (common, goal)]
        for g in (8, 10):
            rest = (10, 10) if g == 8 else (8, 10)
            for a, b in ((0, 1), (1, 0)):
                sizes = {common: 12, goal: g, others[a]: rest[0], others[b]: rest[1]}
                hyps.append((goal, sizes))
    return hyps


HYPS = _hypotheses()


@lru_cache(maxsize=None)
def _posterior(hand_key):
    w = dict.fromkeys(SUITS, 0.0)
    for goal, sizes in HYPS:
        like = 1
        for s, h in zip(SUITS, hand_key):
            like *= comb(sizes[s], h)
        w[goal] += like
    tot = sum(w.values())
    return tuple(w[s] / tot for s in SUITS)


def goal_posterior(initial_hand):
    return dict(zip(SUITS, _posterior(tuple(initial_hand[s] for s in SUITS))))


def initial_hand(obs):
    """Recover the dealt hand: current hand minus purchases plus sales."""
    h, me = dict(obs["hand"]), obs["me"]
    for tr in obs["trades"]:
        if tr.buyer == me:
            h[tr.suit] -= 1
        if tr.seller == me:
            h[tr.suit] += 1
    return h
