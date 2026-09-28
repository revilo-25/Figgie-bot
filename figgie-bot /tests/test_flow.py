import numpy as np
from figgie import Game, SUITS
from figgie.inference import goal_posterior
from figgie.flow import features, flow_posterior, softmax, fit, NF, BUCKETS


def test_identity_model_equals_hand_posterior():
    g = Game(4, 11)
    model = {i: np.array([1.0, 0, 0, 0, 0]) for i in range(len(BUCKETS))}
    a = flow_posterior(g.observe(0), model)
    b = goal_posterior(g.hands[0])
    assert all(abs(a[s] - b[s]) < 1e-9 for s in SUITS)


def test_features_shape_and_own_quote_excluded():
    g = Game(4, 12)
    s = next(s for s in SUITS if g.hands[0][s] > 0)
    g.place(0, s, "offer", 20)
    obs = g.observe(0)
    assert features(obs).shape == (4, NF)
    assert obs["top"][s]["offer"] is not None and obs["top_ex"][s]["offer"] is None


def test_fit_recovers_synthetic_coefficients():
    rng = np.random.default_rng(0)
    th_true = np.array([1.0, 0.3, -0.2, 0.1, 0.05])
    X = rng.normal(size=(40000, 4, NF))
    P = softmax(np.einsum("nsf,f->ns", X, th_true))
    y = np.array([rng.choice(4, p=p) for p in P])
    assert np.allclose(fit(X, y), th_true, atol=0.06)
