"""
Entry point.

Usage:
    1. cp .env.example .env   and fill in PK / TOKEN_ID (and FUNDER if you trade
       through a Polymarket proxy wallet rather than a raw EOA).
    2. pip install -r requirements.txt
    3. python main.py

Emergency stop while running:
    - Ctrl+C, or
    - `touch KILL_SWITCH` from another terminal (filename configurable via
      KILL_SWITCH_FILE in .env). The bot polls for this every cycle, cancels all
      open orders, and exits.

Replace PlaceholderStrategy with your own Strategy subclass (see strategy.py)
before trading real size.
"""

from config import load_config
from market_maker import MarketMaker
from strategy import PlaceholderStrategy


def main():
    cfg = load_config()

    # --- Swap this out for your own Strategy implementation ---
    strategy = PlaceholderStrategy(half_spread=0.02)

    bot = MarketMaker(cfg, strategy)
    bot.run()


if __name__ == "__main__":
    main()
