from figgie.bots import RandomBot
from figgie.sim import tournament

res = tournament([RandomBot() for _ in range(4)], games=1000)
for i, (m, ci) in enumerate(res):
    print(f"bot {i}: mean PnL {m:+.2f} ± {ci:.2f}")
