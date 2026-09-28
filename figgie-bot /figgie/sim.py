import random
import statistics as st
from .engine import Game, IllegalAction


def play(bots, ticks=200, seed=None, on_tick=None):
    g = Game(len(bots), seed)
    rng = random.Random(None if seed is None else seed + 1)
    illegal = 0
    for t in range(ticks):
        g.t = t
        order = list(range(g.n))
        rng.shuffle(order)
        for p in order:
            for act in bots[p].act(g.observe(p), rng) or []:
                try:
                    if act[0] == "cancel":
                        g.cancel(p, act[1], act[2])
                    else:
                        g.place(p, act[1], act[0], act[2])
                except IllegalAction:
                    illegal += 1
        if on_tick:
            on_tick(g, t)
    return g, g.settle(), illegal


def tournament(bots, games=2000, ticks=200, seed=0):
    """Mean PnL per seat with 95% CI. Rotate bots across seats to control for seat effects."""
    n = len(bots)
    pnl = [[] for _ in bots]
    for k in range(games):
        shift = k % n
        seated = bots[shift:] + bots[:shift]
        _, res, _ = play(seated, ticks, seed + k)
        for seat, r in enumerate(res):
            pnl[(seat + shift) % n].append(r)
    out = []
    for xs in pnl:
        m, se = st.mean(xs), st.stdev(xs) / len(xs) ** 0.5
        out.append((m, 1.96 * se))
    return out
