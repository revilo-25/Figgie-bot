from math import comb
from figgie import Game, SUITS
from figgie.valuation import win_share, hand_value, card_values


def test_win_share_exchangeable():
    # Averaged over my dealt goal count, my expected bonus share must be exactly 1/4.
    for ng in (8, 10):
        tot = 0.0
        for c in range(0, min(ng, 10) + 1):
            p = comb(ng, c) * comb(40 - ng, 10 - c) / comb(40, 10)
            tot += p * win_share(c, ng - c, 4, 10)
        assert abs(tot - 0.25) < 1e-12


def test_expected_payout_is_fair_share():
    n = 3000
    mean = sum(hand_value(Game(4, s).hands[0]) for s in range(n)) / n
    assert 48.5 < mean < 51.5, mean


def test_values_nonnegative_and_goal_suit_worth_more_than_ten_at_most_bounded():
    g = Game(4, 5)
    for s, (b, k) in card_values(g.hands[0], g.hands[0]).items():
        assert b >= -1e-9 and k >= -1e-9 and b < 40 and k < 40
