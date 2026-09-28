"""Head-to-head, ablation, opponent-set and FixedSpread experiments.
Usage: PYTHONPATH=. python scripts/experiments.py [games]   (seat-rotated, seed base 900000)"""
import sys, time, json, statistics as st
from figgie.bots import RandomBot, HandMM, PotMM, FlowMM, FixedSpreadBot, MomentumBot
from figgie.ablations import (ConstVal, HandVal, HandValLevel, PotVal, PotValSymmetric, PotValShuffled)
from figgie.sim import tournament, play

G = int(sys.argv[1]) if len(sys.argv) > 1 else 600
SEED = 900_000
OUT = {}


def run(tag, bots, names, games=G):
    t = time.time()
    r = tournament(bots, games=games, seed=SEED)
    OUT[tag] = {n: [round(m, 1), round(c, 1)] for n, (m, c) in zip(names, r)}
    print(f"[{tag}] " + " | ".join(f"{n} {m:+.1f}±{c:.1f}" for n, (m, c) in zip(names, r)) + f"  ({time.time()-t:.0f}s)", flush=True)


# ---- E1: everyone at one table
run("E1 same table", [PotMM(), HandMM(), FlowMM(), FixedSpreadBot()], ["PotMM", "HandMM", "FlowMM", "FixedSpread"], games=800)

# ---- E2: ablation ladder vs two opponent sets (X in seat 0; opponents identical across variants)
LADDER = [("ConstVal(8.8)", ConstVal), ("HandMM(real)", HandMM), ("HandVal", HandVal), ("HandValLevel", HandValLevel),
          ("PotValSymmetric", PotValSymmetric), ("PotVal", PotVal), ("PotValShuffled", PotValShuffled), ("PotMM(real)", PotMM)]
for setname, opp, oppn in (("vs 3 Random", lambda: [RandomBot() for _ in range(3)], "Random"),
                           ("vs 3 PotMM", lambda: [PotMM() for _ in range(3)], "PotMM")):
    for name, cls in LADDER:
        run(f"E2 {name} {setname}", [cls()] + opp(), [name, oppn, oppn, oppn])

# ---- E3: chasers
for name, cls in (("PotMM", PotMM), ("HandMM", HandMM), ("ConstVal(8.8)", ConstVal), ("PotValShuffled", PotValShuffled), ("FixedSpread", FixedSpreadBot)):
    run(f"E3 {name} vs 3 Momentum", [cls()] + [MomentumBot() for _ in range(3)], [name, "Mom", "Mom", "Mom"])

# ---- E4: FixedSpread vs different tables
for setname, mk, oppn in (("3 Random", RandomBot, "Random"), ("3 PotMM", PotMM, "PotMM"), ("3 HandMM", HandMM, "HandMM"),
                          ("3 FixedSpread", FixedSpreadBot, "Fixed")):
    run(f"E4 FixedSpread vs {setname}", [FixedSpreadBot()] + [mk() for _ in range(3)], ["FixedSpread", oppn, oppn, oppn])

# ---- E4b: who pays FixedSpread? per-counterparty sales at table [Fixed, Random, Momentum, PotMM]
kinds = ["FixedSpread", "Random", "Momentum", "PotMM"]
by = {k: [0, 0.0] for k in kinds}
for k in range(G):
    bots = [FixedSpreadBot(), RandomBot(), MomentumBot(), PotMM()]
    g, res, _ = play(bots, 200, SEED + k)
    for tr in g.trades:
        if tr.seller == 0:
            by[kinds[tr.buyer]][0] += 1
            by[kinds[tr.buyer]][1] += tr.price
OUT["E4b sales by buyer"] = {k: [round(v[0] / G, 2), round(v[1] / max(v[0], 1), 2)] for k, v in by.items()}
print("[E4b] FixedSpread sales/game and avg price by buyer:", OUT["E4b sales by buyer"], flush=True)

json.dump(OUT, open("experiments_out.json", "w"), indent=1)
print("done", flush=True)
