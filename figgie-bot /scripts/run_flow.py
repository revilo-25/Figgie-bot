import sys, time
from figgie.bots import PotMM, FlowMM
from figgie.sim import tournament

games = int(sys.argv[1])
t0 = time.time()
r = tournament([FlowMM(), PotMM(), PotMM(), PotMM()], games=games)
print(f"1 FlowMM vs 3 PotMM : FlowMM {r[0][0]:+.2f} ± {r[0][1]:.2f}   PotMMs " +
      ", ".join(f"{m:+.1f}" for m, _ in r[1:]) + f"   [{time.time()-t0:.0f}s]")
r = tournament([FlowMM(), PotMM(), FlowMM(), PotMM()], games=games)
f = (r[0][0] + r[2][0]) / 2; p = (r[1][0] + r[3][0]) / 2
ci = (r[0][1] ** 2 + r[2][1] ** 2) ** 0.5 / 2
print(f"2 FlowMM vs 2 PotMM : FlowMM {f:+.2f} ± ~{ci:.2f} each seat-avg   PotMM {p:+.2f}   [{time.time()-t0:.0f}s]")
