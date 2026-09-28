"""Figgie engine: deal, per-suit order books (price-time priority), trades, settlement.

Simplifications (documented so results are interpretable):
- Turn-based ticks instead of real-time: each tick every player acts once, random order.
- Each player has at most one bid and one offer per suit; a new one replaces the old.
- A crossing order executes immediately at the RESTING order's price.
- After a trade, stale orders (unfundable bids / offers without a card) are removed.
"""
from __future__ import annotations
import random
from dataclasses import dataclass

SUITS = ("S", "C", "H", "D")
PARTNER = {"S": "C", "C": "S", "H": "D", "D": "H"}  # same-colour pairs
CARD_VALUE = 10
POT = 200


class IllegalAction(Exception):
    pass


@dataclass
class Order:
    player: int
    price: int
    seq: int


@dataclass
class Trade:
    t: int
    suit: str
    price: int
    buyer: int
    seller: int


class Game:
    def __init__(self, n_players: int = 4, seed: int | None = None):
        assert n_players in (4, 5)
        self.rng = random.Random(seed)
        self.n = n_players
        self.ante = POT // n_players
        self.start_cash = 350
        self.cash = [self.start_cash - self.ante] * n_players
        self._deal()
        self.books = {s: {"bid": {}, "offer": {}} for s in SUITS}
        self.trades: list[Trade] = []
        self.t = 0
        self._seq = 0

    # ---- setup -------------------------------------------------------
    def _deal(self):
        common = self.rng.choice(SUITS)
        self.goal = PARTNER[common]
        goal_n = self.rng.choice((8, 10))
        others = [s for s in SUITS if s not in (common, self.goal)]
        self.rng.shuffle(others)
        sizes = {common: 12, self.goal: goal_n,
                 others[0]: 10 if goal_n == 8 else 8, others[1]: 10}
        # goal_n=8 -> remaining suits are 10,10 ; goal_n=10 -> 10,8
        if goal_n == 8:
            sizes[others[0]] = 10
        deck = [s for s in SUITS for _ in range(sizes[s])]
        assert len(deck) == 40, sizes
        self.rng.shuffle(deck)
        k = 40 // self.n
        self.hands = [{s: 0 for s in SUITS} for _ in range(self.n)]
        for i in range(self.n):
            for c in deck[i * k:(i + 1) * k]:
                self.hands[i][c] += 1
        self.suit_sizes = sizes

    # ---- order book --------------------------------------------------
    def _best(self, suit, side, exclude=None):
        orders = [o for o in self.books[suit][side].values() if o.player != exclude]
        if not orders:
            return None
        if side == "bid":
            return max(orders, key=lambda o: (o.price, -o.seq))
        return min(orders, key=lambda o: (o.price, o.seq))

    def place(self, p: int, suit: str, side: str, price: int):
        """side in {'bid','offer'}. Returns a Trade if it crossed, else None."""
        if suit not in SUITS or side not in ("bid", "offer") or int(price) != price or price <= 0:
            raise IllegalAction("bad order")
        if side == "bid":
            if self.cash[p] < price:
                raise IllegalAction("insufficient cash")
            own = self.books[suit]["offer"].get(p)
            if own and own.price <= price:
                raise IllegalAction("self-cross")
            best = self._best(suit, "offer", exclude=p)
            if best and best.price <= price:
                return self._trade(suit, buyer=p, seller=best.player, price=best.price)
        else:
            if self.hands[p][suit] < 1:
                raise IllegalAction("no card")
            own = self.books[suit]["bid"].get(p)
            if own and own.price >= price:
                raise IllegalAction("self-cross")
            best = self._best(suit, "bid", exclude=p)
            if best and best.price >= price:
                return self._trade(suit, buyer=best.player, seller=p, price=best.price)
        self._seq += 1
        self.books[suit][side][p] = Order(p, int(price), self._seq)
        return None

    def cancel(self, p: int, suit: str, side: str):
        self.books[suit][side].pop(p, None)

    def _trade(self, suit, buyer, seller, price):
        self.cash[buyer] -= price
        self.cash[seller] += price
        self.hands[buyer][suit] += 1
        self.hands[seller][suit] -= 1
        tr = Trade(self.t, suit, price, buyer, seller)
        self.trades.append(tr)
        self._clean(buyer, seller)
        return tr

    def _clean(self, buyer, seller):
        for s in SUITS:
            b = self.books[s]["bid"].get(buyer)
            if b and b.price > self.cash[buyer]:
                del self.books[s]["bid"][buyer]
            o = self.books[s]["offer"].get(seller)
            if o and self.hands[seller][s] < 1:
                del self.books[s]["offer"][seller]

    # ---- observation -------------------------------------------------
    def observe(self, p: int) -> dict:
        """Public info (books, trades) + private info (own hand, cash)."""
        top = {}
        for s in SUITS:
            bb, bo = self._best(s, "bid"), self._best(s, "offer")
            top[s] = {"bid": (bb.price, bb.player) if bb else None,
                      "offer": (bo.price, bo.player) if bo else None}
        last = {}
        for tr in self.trades:
            last[tr.suit] = tr.price
        top_ex = {}
        for s in SUITS:
            bb, bo = self._best(s, "bid", exclude=p), self._best(s, "offer", exclude=p)
            top_ex[s] = {"bid": (bb.price, bb.player) if bb else None,
                         "offer": (bo.price, bo.player) if bo else None}
        return {"me": p, "t": self.t, "hand": dict(self.hands[p]), "cash": self.cash[p],
                "top": top, "top_ex": top_ex, "last_price": last, "trades": list(self.trades), "n": self.n}

    # ---- settlement --------------------------------------------------
    def settle(self) -> list[float]:
        """Return each player's PnL relative to starting cash (sums to 0)."""
        counts = [h[self.goal] for h in self.hands]
        bonus = POT - CARD_VALUE * sum(counts)
        top = max(counts)
        winners = [i for i, c in enumerate(counts) if c == top]
        pnl = []
        for i in range(self.n):
            final = self.cash[i] + CARD_VALUE * counts[i]
            if i in winners:
                final += bonus / len(winners)
            pnl.append(final - self.start_cash)
        return pnl
