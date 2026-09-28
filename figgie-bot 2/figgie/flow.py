"""Order-flow inference: a conditional-logit model that updates the goal posterior using
public market data. Score for suit s at time t (coefficients shared across suits, fitted
per time bucket on simulated self-play):

    score_s = th0*log P_hand(s) + th1*last_px_s + th2*best_bid_s + th3*best_offer_s + th4*n_trades_s

Price features are centred across suits; bid/offer exclude the observer's own quotes.
P(goal=s | everything) = softmax(score)_s.  th=[1,0,0,0,0] reproduces the hand-only posterior.
"""
import json, math, os
import numpy as np
from .engine import SUITS
from .inference import goal_posterior, initial_hand

MODEL_PATH = os.path.join(os.path.dirname(__file__), "flow_model.json")
BUCKETS = (0, 30, 80, 140)          # lower tick bound of each bucket
NF = 5
_MODEL = None


def bucket(t):
    return max(i for i, b in enumerate(BUCKETS) if t >= b)


def features(obs):
    """4 x NF matrix (rows = suits, in SUITS order)."""
    post = goal_posterior(initial_hand(obs))
    top, last = obs["top_ex"], obs["last_price"]
    cnt = dict.fromkeys(SUITS, 0)
    for tr in obs["trades"]:
        cnt[tr.suit] += 1
    cols = [[math.log(max(post[s], 1e-9)) for s in SUITS],
            [last.get(s, 0) for s in SUITS],
            [top[s]["bid"][0] if top[s]["bid"] else 0 for s in SUITS],
            [top[s]["offer"][0] if top[s]["offer"] else 10 for s in SUITS],
            [cnt[s] for s in SUITS]]
    X = np.array(cols, dtype=float).T
    X[:, 1:] -= X[:, 1:].mean(axis=0)
    return X


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit(X, y, l2=1e-3):
    """X: N x 4 x NF, y: N true-goal indices. Ridge on all coefficients except th0."""
    from scipy.optimize import minimize
    N = len(y)
    Y = np.eye(4)[y]
    reg = np.array([0.0] + [l2] * (NF - 1))

    def loss(th):
        P = softmax(np.einsum("nsf,f->ns", X, th))
        return -np.log(P[np.arange(N), y] + 1e-12).mean() + 0.5 * (reg * th * th).sum()

    def grad(th):
        P = softmax(np.einsum("nsf,f->ns", X, th))
        return np.einsum("nsf,ns->f", X, P - Y) / N + reg * th

    th0 = np.zeros(NF); th0[0] = 1.0
    return minimize(loss, th0, jac=grad, method="L-BFGS-B").x


def load_model():
    global _MODEL
    if _MODEL is None:
        with open(MODEL_PATH) as f:
            _MODEL = {int(k): np.array(v) for k, v in json.load(f).items()}
    return _MODEL


def flow_posterior(obs, model=None):
    model = model or load_model()
    p = softmax(features(obs) @ model[bucket(obs["t"])])
    return dict(zip(SUITS, p))
