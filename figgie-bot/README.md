# Figgie quant project

Simulator, market-making bots and Bayesian goal-suit inference for Figgie (4 players).

- `figgie/engine.py` – deal, per-suit order books (price-time priority), settlement. Rule simplifications are in the docstring (turn-based ticks, one bid/offer per suit per player).
- `figgie/inference.py` – exact hand-only posterior over the 16 deck compositions.
- `figgie/valuation.py` – pot-aware marginal card values (exact enumeration of opponents' goal-card counts).
- `figgie/flow.py` – conditional-logit order-flow model updating the goal posterior from public market data.
- `figgie/bots.py` – RandomBot, HandMM, PotMM, FlowMM, FixedSpreadBot, MomentumBot.

## Run
    pip install -r requirements.txt
    PYTHONPATH=. pytest -q tests
    PYTHONPATH=. python scripts/run_h2h.py
    PYTHONPATH=. python scripts/train_flow.py 1500 500   # retrains figgie/flow_model.json

## Findings (600 games/table, seat-rotated, 95% CI; raw numbers in results.json, reproduce with scripts/experiments.py)
1. **Same table** (PotMM, HandMM, FlowMM, FixedSpread): +5.8 +/- 4.1, +1.0 +/- 1.6, +3.2 +/- 4.3, -9.9 +/- 2.0. The three market makers are statistically tied; FixedSpread loses.
2. **PotMM's +46 vs Random bots is a price-level effect, not inference.** A constant valuation of 8.8 (ConstVal) earns +46.6, hand-posterior rescaled to the same level earns +44.3, PotMM +46.5. The true average card is worth 5 (200 pot / 40 cards), so selling above 5 to price-anchored bots pays.
3. **Suit information is not shown to add edge.** PotMM ties ConstVal-style bots against informed tables; scrambling suit labels (PotValShuffled) costs 20-28, so the signal is not noise, but a no-information constant does as well as the real thing.
4. **Swapping pot-aware values into HandMM's quoting** (PotValSymmetric) earns ~0 vs Random and -40 vs PotMM tables: pot-aware midpoints alone do not help; PotMM's asymmetric buy/sell quoting matters.
5. **MomentumBot is value-blind, not smart**: it buys any ask <= 12 and dumps into any bid. Every valuation-based bot farms it (HandMM +100, PotMM +60, FixedSpread +155).
6. **FixedSpread resolved**: +28 vs Random, +155 vs Momentum, but -3.9 vs PotMM, -1.9 vs HandMM, +0.9 vs itself. It sells ~8.8 cards/game at ~8.3 against an average worth of 5 (~+3.3 x 8.8 = +29, matching). The edge comes from value-blind counterparties, not from the game.
7. Order flow lifts held-out goal-suit accuracy 0.38 -> ~0.44 (log loss 1.303 -> 1.245) but FlowMM's PnL is tied with PotMM.

## Known limitations
Turn-based simplification; opponents are simple (Random/Momentum/FixedSpread are value-blind); ablations shown at 600 games have CIs of +/-4-9; flow model is trained on a specific bot population; FlowMM only rescales PotMM values.
