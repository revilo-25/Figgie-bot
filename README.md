# Figgie Market-Making & Inference

A from-scratch simulator, Bayesian goal-suit inference, and a family of market-making bots for **Figgie** — the card-trading game used by Jane Street to teach market making. Includes an ablation suite that stress-tests every "profitable" result against a no-information control.

> **TL;DR:** A pot-aware market maker looked +46/game against naive bots. A control with zero information about the goal suit scored the same. The edge was a price-level effect (selling above the $5 fair average to mispriced counterparties), not signal. Full trace in [Findings](#findings) below.

## What's here

| Module | What it does |
|---|---|
| `figgie/engine.py` | Deal, per-suit order books (price-time priority), settlement. Turn-based simplification of real Figgie — see docstring for exact rule deviations. |
| `figgie/inference.py` | Exact hand-only posterior over the 16 deck compositions consistent with a dealt hand. |
| `figgie/valuation.py` | Pot-aware marginal card values — exact enumeration over opponents' possible goal-card counts, capturing the lumpy majority-bonus payoff. |
| `figgie/flow.py` | Conditional-logit model that updates the goal posterior from public order-flow (trades, quotes) as a game progresses. |
| `figgie/bots.py` | `RandomBot`, `HandMM`, `PotMM`, `FlowMM`, `FixedSpreadBot`, `MomentumBot`. |
| `figgie/ablations.py` | One shared quoting engine, pluggable valuation (`ConstVal`, `HandVal`, `HandValLevel`, `PotVal`, `PotValSymmetric`, `PotValShuffled`) — isolates *why* a bot makes money. |
| `scripts/experiments.py` | Head-to-head tables, the full ablation ladder, opponent-set tests, and the FixedSpread diagnostic. Writes `results.json`. |
| `scripts/train_flow.py` | Retrains `figgie/flow_model.json` on mixed-population self-play. |

## Quickstart

```bash
git clone <repo-url> && cd figgie-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=. pytest -q tests                       # 14 tests
PYTHONPATH=. python scripts/experiments.py 600      # ~8 min, reproduces results.json
PYTHONPATH=. python scripts/run_h2h.py              # quick head-to-head, ~2 min
PYTHONPATH=. python scripts/train_flow.py 1500 500  # retrain the flow model
```

## Findings

600 games/table, seats rotated to cancel seat effects, 95% CIs. Raw numbers in `results.json`; reproduce with `scripts/experiments.py`.

1. **Three market makers, one table** — PotMM, HandMM, FlowMM, FixedSpread: +5.8±4.1, +1.0±1.6, +3.2±4.3, −9.9±2.0. The three information-using bots are statistically indistinguishable from each other.
2. **PotMM's edge over Random bots is a price-level effect, not inference.** A constant valuation of 8.8 with zero goal-suit information (`ConstVal`) earns +46.6/game — matching PotMM's +46.5. The true average card is worth $5 (200 pot ÷ 40 cards), so quoting above that against price-anchored opponents pays regardless of what you know.
3. **Suit-level information isn't shown to add net edge.** Scrambling which suit the model thinks is the goal (`PotValShuffled`) costs 20–28/game — so the signal isn't pure noise — but a no-information constant still matches the real bot's PnL against these opponents.
4. **Pot-aware values alone don't explain PotMM either.** Swapping pot-aware values into HandMM's symmetric quoting (`PotValSymmetric`) scores ~0 vs Random and −40 vs a table of PotMMs. PotMM's asymmetric buy/sell quoting is doing real work the ablations haven't fully isolated yet.
5. **MomentumBot is value-blind, not smart** — it buys any offer ≤12 and dumps into any bid, no valuation at all. Every valuation-based bot farms it (HandMM +100, PotMM +60, FixedSpread +155/game), so beating it isn't evidence of skill.
6. **The FixedSpreadBot question, resolved.** A bot that blindly bids 2 / offers 8 in one random suit each tick: +28 vs Random, +155 vs Momentum, but −3.9 vs PotMM, −1.9 vs HandMM, +0.9 vs itself. It sells ~8.8 cards/game at ~8.3 against a $5 average (+3.3 × 8.8 ≈ +29, matching observed). **The edge is entirely counterparty-driven** — it evaporates against any opponent that prices correctly.
7. **Order flow improves prediction, not (yet) PnL.** Held-out goal-suit accuracy: 38% (hand-only) → ~44% (+ order flow); log loss 1.303 → 1.245. FlowMM's realized PnL is statistically tied with PotMM's.

## Known limitations

- Turn-based tick structure is a simplification of real-time Figgie (see `engine.py` docstring for the full list).
- The comparison opponents (Random, Momentum, FixedSpread) are all value-blind by construction — no bot here has been tested against a genuinely rational, adaptive counterparty, so "beats X" claims are scoped to these specific opponents, not the game in general.
- Ablation results at 600 games/table carry ±4–9 CIs; treat differences smaller than that as noise.
- `flow_model.json` is fit on a specific mixed-bot population; it may not generalize to a different opponent mix (train/deploy mismatch risk).
- `FlowMM` only rescales `PotMM`'s valuation by a likelihood ratio — it doesn't otherwise change the quoting logic.

## Tests

```bash
PYTHONPATH=. pytest -q tests
```
14 tests covering engine mechanics (deal/trade/settle correctness, illegal-action handling), the hand-only posterior (sums to 1, matches brute-force enumeration on small cases), pot-aware valuation, and the flow model.
