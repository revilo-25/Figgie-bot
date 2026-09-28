import math
from figgie import SUITS, PARTNER, Game
from figgie.inference import goal_posterior, initial_hand


def test_posterior_normalised():
    for seed in range(100):
        g = Game(4, seed)
        assert math.isclose(sum(goal_posterior(g.hands[0]).values()), 1.0)


def test_heavy_suit_points_to_partner():
    p = goal_posterior({"S": 6, "C": 2, "H": 1, "D": 1})
    assert max(p, key=p.get) == PARTNER["S"]


def test_posterior_calibrated():
    # hand-only info is weak but real: measured over 20k deals, mean mass on the
    # true goal is ~0.292 and argmax accuracy ~0.38, versus 0.25 for the prior.
    tot, hit, n = 0.0, 0, 4000
    for seed in range(n):
        g = Game(4, seed)
        p = goal_posterior(g.hands[0])
        tot += p[g.goal]
        hit += max(p, key=p.get) == g.goal
    assert tot / n > 0.28
    assert hit / n > 0.33


def test_initial_hand_recovery():
    g = Game(4, 3)
    s = next(s for s in SUITS if g.hands[0][s] > 0)
    start = dict(g.hands[0])
    g.place(0, s, "offer", 5)
    g.place(1, s, "bid", 6)
    assert initial_hand(g.observe(0)) == start
