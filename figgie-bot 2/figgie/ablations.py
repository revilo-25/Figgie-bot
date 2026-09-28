"""Ablation bots: one shared quoting engine (identical to PotMM's), pluggable valuation.

Every variant differs from every other ONLY in `values(obs) -> {suit: (buy_v, sell_v)}`,
so PnL differences are attributable to the valuation, not to quoting mechanics.
"""
import math
import random
from .engine import SUITS, CARD_VALUE
from .inference import goal_posterior, initial_hand
from .valuation import card_values


class ValueMM:
    def __init__(self, edge=1.0, take_edge=1.0):
        self.edge, self.take_edge = edge, take_edge

    def values(self, obs):
        raise NotImplementedError

    def act(self, obs, rng):
        me = obs["me"]
        vals = self.values(obs)
        acts = []
        for s in SUITS:
            buy_v, sell_v = vals[s]
            top = obs["top"][s]
            ask, bid = top["offer"], top["bid"]
            if ask and ask[1] != me and ask[0] <= buy_v - self.take_edge and obs["cash"] >= ask[0]:
                acts.append(("bid", s, ask[0]))
                continue
            if bid and bid[1] != me and bid[0] >= sell_v + self.take_edge and obs["hand"][s] > 0:
                acts.append(("offer", s, bid[0]))
                continue
            b, a = math.floor(buy_v - self.edge), math.ceil(sell_v + self.edge)
            if b >= a:
                mid = (buy_v + sell_v) / 2
                b, a = math.floor(mid - self.edge), math.ceil(mid + self.edge)
            b, a = max(1, b), max(2, a)
            acts += [("cancel", s, "bid"), ("cancel", s, "offer")]
            if b <= obs["cash"]:
                acts.append(("bid", s, b))
            if obs["hand"][s] > 0:
                acts.append(("offer", s, a))
        return acts


class PotVal(ValueMM):
    """Full pot-aware values (should reproduce PotMM)."""
    def values(self, obs):
        return card_values(initial_hand(obs), obs["hand"], obs["n"])


class PotValShuffled(ValueMM):
    """Same price LEVELS and spreads as PotVal, but suit labels permuted per game
    (permutation derived from the dealt hand, so the bot stays stateless).
    Any edge that survives this control is NOT coming from goal-suit information."""
    def values(self, obs):
        init = initial_hand(obs)
        r = random.Random(hash(tuple(init[s] for s in SUITS)) ^ obs["me"])
        perm = list(SUITS)
        r.shuffle(perm)
        v = card_values(init, obs["hand"], obs["n"])
        return {s: v[p] for s, p in zip(SUITS, perm)}


class PotValSymmetric(ValueMM):
    """Pot-aware, but buy = sell = midpoint (removes the lumpy-majority asymmetry)."""
    def values(self, obs):
        v = card_values(initial_hand(obs), obs["hand"], obs["n"])
        return {s: ((b + a) / 2, (b + a) / 2) for s, (b, a) in v.items()}


class HandVal(ValueMM):
    """Hand-only posterior, fair = 10 * P(goal), same inventory skew as HandMM."""
    def __init__(self, edge=1.0, take_edge=1.0, skew=0.25):
        super().__init__(edge, take_edge)
        self.skew = skew

    def values(self, obs):
        init = initial_hand(obs)
        post = goal_posterior(init)
        return {s: (f, f) for s in SUITS
                for f in [CARD_VALUE * post[s] - self.skew * (obs["hand"][s] - init[s])]}


class HandValLevel(ValueMM):
    """Hand-only posterior SHAPE, but rescaled so the mean price level matches PotVal.
    Separates 'how informative is the signal' from 'how high are the prices'."""
    def __init__(self, edge=1.0, take_edge=1.0, level=8.8, skew=0.25):
        super().__init__(edge, take_edge)
        self.level, self.skew = level, skew

    def values(self, obs):
        init = initial_hand(obs)
        post = goal_posterior(init)
        return {s: (f, f) for s in SUITS
                for f in [4 * self.level * post[s] - self.skew * (obs["hand"][s] - init[s])]}


class ConstVal(ValueMM):
    """No information at all: every card valued at a constant, no skew."""
    def __init__(self, c=8.8, edge=1.0, take_edge=1.0):
        super().__init__(edge, take_edge)
        self.c = c

    def values(self, obs):
        return {s: (self.c, self.c) for s in SUITS}
