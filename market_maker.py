"""
Core market-making loop.

Wraps py-clob-client (the official Polymarket CLOB SDK) and drives it with
whatever Strategy you plug in. Handles: connecting/authenticating, reading the
book, asking the strategy for quotes, replacing resting orders, and — most
importantly — checking the kill switch before anything that touches an order.
"""

import logging
import time

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OpenOrderParams, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

from config import Config
from kill_switch import KillSwitch
from strategy import OrderBookView, Quote, Strategy

logger = logging.getLogger("market_maker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class MarketMaker:
    def __init__(self, cfg: Config, strategy: Strategy):
        self.cfg = cfg
        self.strategy = strategy
        self.kill_switch = KillSwitch(cfg.kill_switch_file)
        self.client = self._build_client()

    def _build_client(self) -> ClobClient:
        client = ClobClient(
            self.cfg.clob_host,
            key=self.cfg.private_key,
            chain_id=self.cfg.chain_id,
            signature_type=self.cfg.signature_type,
            funder=self.cfg.funder,
        )
        # Derives (or creates, first run) L2 API credentials from the L1 wallet
        # signature and authenticates the client for order placement/cancellation.
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        logger.info("Authenticated as %s", client.get_address())
        return client

    # ------------------------------------------------------------------ #
    # Market data
    # ------------------------------------------------------------------ #

    def get_book_view(self) -> OrderBookView:
        book = self.client.get_order_book(self.cfg.token_id)

        best_bid = max((float(b.price) for b in (book.bids or [])), default=None)
        best_ask = min((float(a.price) for a in (book.asks or [])), default=None)
        return OrderBookView(best_bid=best_bid, best_ask=best_ask)

    def get_inventory(self) -> float:
        """
        Placeholder position lookup. py-clob-client doesn't track your net
        position for you — wire this up to your own fill bookkeeping, or to
        Polymarket's Data API (e.g. GET /positions for your wallet address),
        and return net shares held for cfg.token_id (positive = long).
        """
        return 0.0

    # ------------------------------------------------------------------ #
    # Order management
    # ------------------------------------------------------------------ #

    def cancel_our_orders(self):
        try:
            self.client.cancel_market_orders(asset_id=self.cfg.token_id)
        except Exception:
            logger.exception("Failed to cancel existing orders")

    def cancel_everything(self):
        """Used by the kill switch — cancels ALL resting orders for this account,
        not just the ones for the market currently being quoted."""
        try:
            self.client.cancel_all()
            logger.warning("Kill switch: cancelled all open orders.")
        except Exception:
            logger.exception("Kill switch: failed to cancel all orders — check manually!")

    def open_order_count(self) -> int:
        try:
            orders = self.client.get_orders(OpenOrderParams(asset_id=self.cfg.token_id))
            return len(orders)
        except Exception:
            logger.exception("Failed to fetch open orders")
            return 0

    def place_quote(self, quote: Quote):
        if self.open_order_count() >= self.cfg.max_open_orders:
            logger.warning("Max open orders reached (%d) — skipping new quotes this cycle.", self.cfg.max_open_orders)
            return

        for side, price, size in (
            (BUY, quote.bid_price, quote.bid_size),
            (SELL, quote.ask_price, quote.ask_size),
        ):
            if price is None or size is None:
                continue
            try:
                order_args = OrderArgs(price=price, size=size, side=side, token_id=self.cfg.token_id)
                signed_order = self.client.create_order(order_args)
                resp = self.client.post_order(signed_order, OrderType.GTC)
                logger.info("Placed %s %.2f @ %.3f -> %s", side, size, price, resp)
            except Exception:
                logger.exception("Failed to place %s order", side)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self):
        logger.info("Starting market maker for token_id=%s", self.cfg.token_id)
        try:
            while True:
                if self.kill_switch.is_triggered():
                    logger.warning("Kill switch active — halting and cancelling all orders.")
                    self.cancel_everything()
                    break

                book = self.get_book_view()
                inventory = self.get_inventory()
                quote = self.strategy.generate_quotes(book, inventory, self.cfg.order_size)

                # Re-check the kill switch right before touching orders — the
                # strategy call above could have taken nontrivial time.
                if self.kill_switch.is_triggered():
                    logger.warning("Kill switch active — halting and cancelling all orders.")
                    self.cancel_everything()
                    break

                self.cancel_our_orders()

                if quote is None:
                    logger.info("Strategy returned no quote this cycle — staying flat.")
                else:
                    self.place_quote(quote)

                time.sleep(self.cfg.quote_refresh_seconds)
        finally:
            # Belt-and-braces: never exit the loop (error, kill switch, Ctrl+C)
            # leaving resting orders behind unintentionally.
            self.cancel_everything()
            logger.info("Market maker stopped.")
