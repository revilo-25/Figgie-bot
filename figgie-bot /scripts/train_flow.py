"""Fit the order-flow model on mixed-population play, evaluate on held-out games."""
import json, sys, time
import numpy as np
from figgie.engine import SUITS
import random
from figgie.bots import PotMM, HandMM, RandomBot, MomentumBot, FixedSpreadBot
from figgie.sim import play
from figgie.flow import features, fit, softmax, bucket, BUCKETS, MODEL_PATH


POOL = [PotMM, PotMM, HandMM, RandomBot, RandomBot, MomentumBot, FixedSpreadBot]


def lineup(sd):
    """Heterogeneous table: homogeneous PotMM self-play freezes after ~3 ticks (no flow to learn from)."""
    r = random.Random(sd)
    return [r.choice(POOL)() for _ in range(4)]


def collect(seeds):
    X, y, T = [], [], []
    for sd in seeds:
        def hook(g, t):
            if t % 10 == 9:
                for p in range(g.n):
                    X.append(features(g.observe(p))); y.append(SUITS.index(g.goal)); T.append(t)
        play(lineup(sd), 200, sd, on_tick=hook)
    return np.array(X), np.array(y), np.array(T)


ntr, nte = int(sys.argv[1]), int(sys.argv[2])
t0 = time.time()
Xtr, ytr, Ttr = collect(range(100_000, 100_000 + ntr))
Xte, yte, Tte = collect(range(200_000, 200_000 + nte))
print(f"collected {len(ytr)} train / {len(yte)} test snapshots in {time.time()-t0:.0f}s")

btr = np.array([bucket(t) for t in Ttr]); bte = np.array([bucket(t) for t in Tte])
model = {}
print(f"{'ticks':>9} {'hand logloss':>13} {'flow logloss':>13} {'hand acc':>9} {'flow acc':>9}  coefs")
for b in range(len(BUCKETS)):
    th = fit(Xtr[btr == b], ytr[btr == b]); model[b] = th.tolist()
    X, y = Xte[bte == b], yte[bte == b]
    ph = softmax(X[:, :, 0]); pf = softmax(np.einsum("nsf,f->ns", X, th)); i = np.arange(len(y))
    lo = BUCKETS[b]; hi = BUCKETS[b + 1] if b + 1 < len(BUCKETS) else 200
    print(f"{lo:>4}-{hi:<4} {-np.log(ph[i, y]).mean():13.4f} {-np.log(pf[i, y]).mean():13.4f} "
          f"{(ph.argmax(1) == y).mean():9.3f} {(pf.argmax(1) == y).mean():9.3f}  {np.round(th, 3)}")
json.dump(model, open(MODEL_PATH, "w"))
