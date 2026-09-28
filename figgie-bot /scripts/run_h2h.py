"""Head-to-head, ablation, opponent-robustness and FixedSpread diagnostics.
Usage: PYTHONPATH=. python scripts/run_h2h.py [games]   (default 600)"""
import sys, json
from figgie.bots import (HandMM, PotMM, FlowMM, FixedSpreadBot, MomentumBot, RandomBot)
from figgie.ablations import (PotVal, PotValShuffled, PotValSymmetric, HandVal, HandValLevel, ConstVal)
from figgie.sim import tournament

G = int(sys.argv[1]) if len(sys.argv) > 1 else 600
SEC = sys.argv[2] if len(sys.argv) > 2 else "1234"   # which sections to run, e.g. "34"
R, M = RandomBot, MomentumBot
out = {}

def show(tag, names, res):
    out[tag] = {n: [round(m, 2), round(c, 2)] for n, (m, c) in zip(names, res)}
    print(f"{tag}\n  " + "   ".join(f"{n} {m:+.1f}±{c:.1f}" for n, (m, c) in zip(names, res)), flush=True)

print(f"== games per config: {G} ==")
# 1. one table, four different strategies
names = ["PotMM", "HandMM", "FlowMM", "FixedSpread"]
if "1" in SEC:
    show("1. same table", names, tournament([PotMM(), HandMM(), FlowMM(), FixedSpreadBot()], G))

# 2. each strategy vs three copies of an opponent type
for opp_name, Opp in ((("Random", R), ("Momentum", M), ("PotMM", PotMM)) if "2" in SEC else ()):
    for n, mk in (("PotMM", PotMM), ("HandMM", HandMM), ("FlowMM", FlowMM), ("FixedSpread", FixedSpreadBot), ("Const8.8", ConstVal)):
        if opp_name == "PotMM" and n == "PotMM":
            continue
        r = tournament([mk(), Opp(), Opp(), Opp()], G)
        show(f"2. {n} vs 3 {opp_name}", [n, opp_name, opp_name, opp_name], r)

# 3. valuation ablation (identical quoting code), vs 3 random
for n, mk in ((("PotVal", PotVal), ("PotVal-shuffled-suits", PotValShuffled), ("PotVal-symmetric", PotValSymmetric),
              ("HandVal", HandVal), ("HandVal-level-matched", HandValLevel), ("Const8.8 (no info)", ConstVal)) if "3" in SEC else ()):
    r = tournament([mk(), R(), R(), R()], G)
    show(f"3. ablation {n} vs 3 Random", [n, "R", "R", "R"], r)

# 4. information test where levels are equal: informed vs uninformed at the SAME table
if "4" in SEC:
  show("4. PotVal vs Const8.8 vs 2 Random", ["PotVal", "Const8.8", "R", "R"], tournament([PotVal(), ConstVal(), R(), R()], G))
  show("4. PotVal vs PotVal-shuffled vs 2 Random", ["PotVal", "Shuffled", "R", "R"], tournament([PotVal(), PotValShuffled(), R(), R()], G))
  show("4. PotVal vs Const8.8 vs 2 Momentum", ["PotVal", "Const8.8", "M", "M"], tournament([PotVal(), ConstVal(), M(), M()], G))
json.dump(out, open(f"h2h_results_{SEC}.json", "w"), indent=1)
