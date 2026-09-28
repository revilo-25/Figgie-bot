from figgie.bots import HandMM, RandomBot
from figgie.sim import tournament

names = ["HandMM", "Random", "Random", "Random"]
res = tournament([HandMM(), RandomBot(), RandomBot(), RandomBot()], games=1000)
for n, (m, ci) in zip(names, res):
    print(f"{n:7s} mean PnL {m:+.2f} ± {ci:.2f}")
