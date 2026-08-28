"""
Strategy interface.

This is intentionally minimal: given the current order book (and, if you wire it
up, your current inventory), return the quotes you want resting on the book.
Everything about *how* to price those quotes is left to you.

Implement your own subclass of `Strategy` and pass it into MarketMaker instead of
PlaceholderStrategy. Ideas for what to build here:
  - Inventory skew (widen/shift quotes as your position grows to mean-revert it)
  - Volatility-aware spreads (widen when recent price moves are large)
  - Avellaneda-Stoikov style reservation pricing
  - Signal-driven skew (news, order flow imbalance, related-market correlation)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OrderBookView:
    """Simplified view of the book — best bid/ask and the derived midpoint."""

    best_bid: float | None
    best_ask: float | None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2


@dataclass
class Quote:
    """A two-sided quote the bot should try to keep resting on the book.

    Set bid_price/bid_size or ask_price/ask_size to None to skip quoting that side
    (e.g. if inventory limits mean you only want to quote one side right now).
    """

    bid_price: float | None
    bid_size: float | None
    ask_price: float | None
    ask_size: float | None


class Strategy(ABC):
    @abstractmethod
    def generate_quotes(self, book: OrderBookView, inventory: float, default_size: float) -> Quote | None:
        """
        Return the Quote to place, or None to pull all quotes this cycle
        (e.g. book is empty/too wide, or a risk limit was breached).

        Args:
            book: current best bid/ask/mid for the market being quoted.
            inventory: your current net position in shares (positive = long).
                       Wire this up to your own position tracking / the Polymarket
                       Data API — py-clob-client does not track it for you.
            default_size: the ORDER_SIZE from config, provided as a convenience.
        """
        raise NotImplementedError


class PlaceholderStrategy(Strategy):
    """
    Bare-bones example so the bot runs out of the box: quote a fixed spread
    symmetrically around the midpoint, ignoring inventory entirely.

    Replace this with your real strategy — this class exists only as a working
    placeholder / reference, not as something you should trade real size with.
    """

    def __init__(self, half_spread: float = 0.02):
        self.half_spread = half_spread

    def generate_quotes(self, book: OrderBookView, inventory: float, default_size: float) -> Quote | None:
        if book.mid is None:
            return None  # no two-sided market to quote against — stay flat

        # --- YOUR STRATEGY GOES HERE ---
        # Naive symmetric quotes around the mid. Consider replacing with:
        #   - inventory skew: shift both prices by -k * inventory
        #   - dynamic spread: widen self.half_spread with realized volatility
        bid_price = round(book.mid - self.half_spread, 3)
        ask_price = round(book.mid + self.half_spread, 3)

        # Keep prices inside Polymarket's valid (0, 1) probability range.
        bid_price = max(0.01, min(bid_price, 0.99))
        ask_price = max(0.01, min(ask_price, 0.99))

        return Quote(
            bid_price=bid_price,
            bid_size=default_size,
            ask_price=ask_price,
            ask_size=default_size,
        )
