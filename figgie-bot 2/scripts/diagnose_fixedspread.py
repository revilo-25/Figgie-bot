"""Why does FixedSpreadBot (bid 2 / offer 8) earn +30..+150 per game?
Hypothesis: it sells cards at 8 to opponents whose price anchor is ~8, while the average
card is only worth ~2.5 in expectation, so the edge is opponent-driven, not a game property."""
import sys, random, statistics as st
from figgie.bots import FixedSpreadBot, RandomBot, PotMM
from figgie.ablations import ConstVal
from figgie.sim import play, tournament

G = int(sys.argv[1]) if len(sys.argv) > 1 else 300


class AnchoredRandom(RandomBot):
    """RandomBot with a configurable default reference price (original hardcodes 8)."""
    def __init__(self, default_ref=8):
        self.d = default_ref

    def act(self, obs, rng):
        lp = {s: obs["last_price"].get(s, self.d) for s in "SCHD"}
        return super().act({**obs, "last_price": lp}, rng)


# 1) Trade anatomy: where does FixedSpread's money come from? (seat 0 vs 3 Random)
sold, bought, sell_px, buy_px, goal_sold, pnl = 0, 0, [], [], 0, []
for k in range(G):
    g, res, _ = play([FixedSpreadBot(), RandomBot(), RandomBot(), RandomBot()], 200, k)
    pnl.append(res[0])
    for tr in g.trades:
        if tr.seller == 0:
            sold += 1; sell_px.append(tr.price); goal_sold += tr.suit == g.goal
        if tr.buyer == 0:
            bought += 1; buy_px.append(tr.price)
print(f"FixedSpread vs 3 Random ({G} games): PnL {st.mean(pnl):+.1f}")
print(f"  sells/game {sold/G:.1f} @ avg {st.mean(sell_px):.2f} | buys/game {bought/G:.1f} @ avg {st.mean(buy_px) if buy_px else 0:.2f}")
print(f"  share of its sales that were goal-suit cards: {goal_sold/max(sold,1):.1%} (prior 25%)")
print(f"  => cash from selling at ~8 a card whose unconditional value is ~2.5-4 (10 x P(goal) + bonus)")

# 2) Does the edge survive when the opponents' price anchor moves?
print("\nRandom-bot default reference price (original = 8):")
for d in (8, 6, 4, 2):
    r = tournament([FixedSpreadBot()] + [AnchoredRandom(d) for _ in range(3)], G)
    r2 = tournament([ConstVal()] + [AnchoredRandom(d) for _ in range(3)], G)
    r3 = tournament([PotMM()] + [AnchoredRandom(d) for _ in range(3)], G)
    print(f"  anchor {d}:  FixedSpread {r[0][0]:+6.1f}±{r[0][1]:.1f}   Const8.8 {r2[0][0]:+6.1f}±{r2[0][1]:.1f}   PotMM {r3[0][0]:+6.1f}±{r3[0][1]:.1f}")
