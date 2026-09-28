from .engine import SUITS


class RandomBot:
    """Quotes randomly around the last trade price. Baseline; ~0 PnL vs itself."""
    def act(self, obs, rng):
        s = rng.choice(SUITS)
        ref = obs["last_price"].get(s, 8)
        if rng.random() < 0.5:
            return [("bid", s, max(1, ref + rng.randint(-4, 2)))]
        return [("offer", s, max(1, ref + rng.randint(-2, 4)))]


import math
from .engine import CARD_VALUE
from .inference import goal_posterior, initial_hand


class HandMM:
    """Market maker priced off the hand-only posterior: fair = 10 * P(goal = suit).

    v1 ignores the majority-pot option value and order-flow information (next steps).
    Inventory skew shifts fair value down when long a suit (and up when short).
    """
    def __init__(self, edge=1.0, take_edge=1.0, skew=0.25):
        self.edge, self.take_edge, self.skew = edge, take_edge, skew

    def act(self, obs, rng):
        me = obs["me"]
        init = initial_hand(obs)
        post = goal_posterior(init)
        acts = []
        for s in SUITS:
            f = CARD_VALUE * post[s] - self.skew * (obs["hand"][s] - init[s])
            top = obs["top"][s]
            ask, bid = top["offer"], top["bid"]
            if ask and ask[1] != me and ask[0] <= f - self.take_edge and obs["cash"] >= ask[0]:
                acts.append(("bid", s, ask[0]))          # lift a cheap offer
                continue
            if bid and bid[1] != me and bid[0] >= f + self.take_edge and obs["hand"][s] > 0:
                acts.append(("offer", s, bid[0]))        # hit a rich bid
                continue
            acts += [("cancel", s, "bid"), ("cancel", s, "offer")]
            b, a = max(1, math.floor(f - self.edge)), max(2, math.ceil(f + self.edge))
            if b <= obs["cash"]:
                acts.append(("bid", s, b))
            if obs["hand"][s] > 0:
                acts.append(("offer", s, a))
        return acts


from .valuation import card_values


class PotMM:
    """Market maker using pot-aware marginal values: bid off the buy value, ask off the sell
    value (they differ because the majority bonus is lumpy). No order-flow inference yet."""
    def __init__(self, edge=1.0, take_edge=1.0):
        self.edge, self.take_edge = edge, take_edge

    def act(self, obs, rng):
        me, n = obs["me"], obs["n"]
        init = initial_hand(obs)
        vals = card_values(init, obs["hand"], n)
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
            if b >= a:                                   # non-convex region: quote around the mid
                mid = (buy_v + sell_v) / 2
                b, a = math.floor(mid - self.edge), math.ceil(mid + self.edge)
            b, a = max(1, b), max(2, a)
            acts += [("cancel", s, "bid"), ("cancel", s, "offer")]
            if b <= obs["cash"]:
                acts.append(("bid", s, b))
            if obs["hand"][s] > 0:
                acts.append(("offer", s, a))
        return acts


from .flow import flow_posterior
from .inference import goal_posterior as _gp


class FlowMM(PotMM):
    """PotMM whose goal-probability factor is replaced by the order-flow posterior.
    card_values already embeds P_hand(s), so rescale by P_flow(s)/P_hand(s)."""
    def __init__(self, edge=1.0, take_edge=1.0, model=None):
        super().__init__(edge, take_edge)
        self.model = model

    def act(self, obs, rng):
        init = initial_hand(obs)
        ph, pf = _gp(init), flow_posterior(obs, self.model)
        scale = {s: pf[s] / max(ph[s], 1e-9) for s in SUITS}
        real = card_values
        import figgie.bots as B
        B.card_values = lambda i, c, n=4: {s: (real(i, c, n)[s][0] * scale[s], real(i, c, n)[s][1] * scale[s]) for s in SUITS}
        try:
            return super().act(obs, rng)
        finally:
            B.card_values = real


class FixedSpreadBot:
    """Value-blind seller: bid 2 / offer 8 in one random suit each tick (rarely buys; sells ~9 cards/game)."""
    def act(self, obs, rng):
        s = rng.choice(SUITS)
        acts = [("cancel", s, "bid"), ("cancel", s, "offer"), ("bid", s, 2)]
        if obs["hand"][s] > 0:
            acts.append(("offer", s, 8))
        return acts


class MomentumBot:
    """Buys the suit with the most recent trades, sells the least-traded; chases flow."""
    def act(self, obs, rng):
        cnt = dict.fromkeys(SUITS, 0)
        for tr in obs["trades"][-12:]:
            cnt[tr.suit] += 1
        hot = max(SUITS, key=lambda s: (cnt[s], rng.random()))
        ask = obs["top_ex"][hot]["offer"]
        acts = []
        if ask and ask[0] <= 12 and obs["cash"] >= ask[0]:
            acts.append(("bid", hot, ask[0]))
        cold = min(SUITS, key=lambda s: (cnt[s], rng.random()))
        bid = obs["top_ex"][cold]["bid"]
        if bid and obs["hand"][cold] > 0:
            acts.append(("offer", cold, bid[0]))
        return acts
