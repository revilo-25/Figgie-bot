"""Pot-aware card values.

Payoff for a player holding c goal-suit cards when the goal suit has N cards:
    10*c + B * share,   B = 200 - 10*N   (the majority bonus; ties split it)
`share` depends on what the opponents hold. Given c, the other N-c goal cards sit
uniformly at random in the opponents' h_opp*(n-1) card slots, so their goal counts are
multivariate hypergeometric and can be enumerated exactly (no Monte Carlo needed).

Marginal values (per suit s, current holding c, goal-suit hypotheses weighted by posterior):
    buy  value = E[payoff(c+1) - payoff(c) | goal = s] * P(goal = s)
    sell value = E[payoff(c)   - payoff(c-1) | goal = s] * P(goal = s)
They differ because the bonus makes payoff non-linear in c (majority is lumpy).

Approximation: opponents are assumed to hold 40//n cards each even after our own trades.
"""
from functools import lru_cache
from math import comb, prod
from .engine import SUITS, POT, CARD_VALUE
from .inference import HYPS


def weighted_hyps(initial, current):
    """Hypotheses weighted by likelihood of the INITIAL hand, ruling out any composition
    that has fewer cards of a suit than we currently hold."""
    out = []
    for goal, sizes in HYPS:
        if any(sizes[s] < current[s] for s in SUITS):
            continue
        w = prod(comb(sizes[s], initial[s]) for s in SUITS)
        if w:
            out.append((goal, sizes, w))
    tot = sum(w for _, _, w in out)
    return [(g, sz, w / tot) for g, sz, w in out]


def _comps(m, k, cap):
    if k == 1:
        if m <= cap:
            yield (m,)
        return
    for x in range(min(m, cap) + 1):
        for rest in _comps(m - x, k - 1, cap):
            yield (x,) + rest


@lru_cache(maxsize=None)
def win_share(c, m, n, h_opp):
    """Expected share of the bonus for me holding c goal cards while m goal cards are
    spread over n-1 opponents with h_opp cards each."""
    total = comb(h_opp * (n - 1), m)
    acc = 0
    for counts in _comps(m, n - 1, h_opp):
        top = max(counts)
        if c > top:
            share = 1.0
        elif c == top:
            share = 1.0 / (1 + counts.count(top))
        else:
            continue
        acc += prod(comb(h_opp, x) for x in counts) * share
    return acc / total


def payoff(c, n_goal, n, h_opp):
    return CARD_VALUE * c + (POT - CARD_VALUE * n_goal) * win_share(c, n_goal - c, n, h_opp)


@lru_cache(maxsize=None)
def _card_values(init_key, cur_key, n):
    initial, current = dict(zip(SUITS, init_key)), dict(zip(SUITS, cur_key))
    h_opp = 40 // n
    hyps = weighted_hyps(initial, current)
    res = {}
    for s in SUITS:
        c, buy, sell = current[s], 0.0, 0.0
        for goal, sizes, w in hyps:
            if goal != s:
                continue
            ng = sizes[s]
            base = payoff(c, ng, n, h_opp)
            up = payoff(c + 1, ng, n, h_opp) if c + 1 <= ng else base
            dn = payoff(c - 1, ng, n, h_opp) if c >= 1 else base
            buy += w * (up - base)
            sell += w * (base - dn)
        res[s] = (buy, sell)
    return res


def card_values(initial, current, n=4):
    return _card_values(tuple(initial[s] for s in SUITS), tuple(current[s] for s in SUITS), n)


def hand_value(hand, n=4):
    """Expected payout of a dealt hand (posterior-averaged). Averages to POT/n over deals."""
    h_opp = 40 // n
    return sum(w * payoff(hand[g], sizes[g], n, h_opp)
               for g, sizes, w in weighted_hyps(hand, hand))
