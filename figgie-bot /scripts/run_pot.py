from figgie.bots import HandMM, PotMM, RandomBot
from figgie.sim import tournament

for label, bot in (("HandMM", HandMM()), ("PotMM", PotMM())):
    res = tournament([bot, RandomBot(), RandomBot(), RandomBot()], games=1000)
    print(f"{label:7s} vs 3 random: {res[0][0]:+.2f} ± {res[0][1]:.2f}   (randoms: "
          + ", ".join(f"{m:+.1f}" for m, _ in res[1:]) + ")")
