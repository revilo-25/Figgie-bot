import random
from figgie import Game, IllegalAction, SUITS, PARTNER
from figgie.bots import RandomBot
from figgie.sim import play


def test_deck_composition():
    for seed in range(300):
        g = Game(4, seed)
        assert sorted(g.suit_sizes.values()) in ([8, 10, 10, 12], )
        assert sum(sum(h.values()) for h in g.hands) == 40
        assert all(sum(h.values()) == 10 for h in g.hands)
        common = [s for s, n in g.suit_sizes.items() if n == 12][0]
        assert g.goal == PARTNER[common] and g.suit_sizes[g.goal] in (8, 10)


def test_trade_at_resting_price_and_conservation():
    g = Game(4, 1)
    s = next(s for s in SUITS if g.hands[0][s] > 0)
    g.place(0, s, "offer", 9)
    tr = g.place(1, s, "bid", 12)
    assert tr.price == 9 and tr.buyer == 1 and tr.seller == 0
    assert sum(g.cash) == 4 * (350 - 50)


def test_self_cross_and_funding():
    g = Game(4, 2)
    s = next(s for s in SUITS if g.hands[0][s] > 0)
    g.place(0, s, "offer", 9)
    try:
        g.place(0, s, "bid", 10); assert False
    except IllegalAction:
        pass
    try:
        g.place(1, s, "bid", 10_000); assert False
    except IllegalAction:
        pass


def test_zero_sum_over_random_games():
    for seed in range(50):
        _, pnl, _ = play([RandomBot() for _ in range(4)], 100, seed)
        assert abs(sum(pnl)) < 1e-9
