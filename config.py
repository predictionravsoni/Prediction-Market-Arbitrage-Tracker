"""
Loads bot configuration from environment variables (via a local .env file).

Keeping all tunables in one place makes it easy to see what the bot depends on
and avoids scattering os.environ calls throughout the codebase.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # --- Auth / connection ---
    private_key: str
    clob_host: str
    chain_id: int
    signature_type: int
    funder: str | None

    # --- Market ---
    token_id: str

    # --- Safety ---
    kill_switch_file: str

    # --- Quoting parameters (placeholders — tune these or drive them from your strategy) ---
    order_size: float          # size (in shares) per quote
    quote_refresh_seconds: float  # how often the main loop re-quotes
    max_open_orders: int       # hard cap on resting orders as a sanity check


def load_config() -> Config:
    pk = os.environ.get("PK")
    if not pk:
        raise RuntimeError("PK is not set. Copy .env.example to .env and fill in your private key.")

    token_id = os.environ.get("TOKEN_ID")
    if not token_id or token_id == "REPLACE_WITH_TOKEN_ID":
        raise RuntimeError("TOKEN_ID is not set. Set it in .env to the outcome token you want to quote.")

    return Config(
        private_key=pk,
        clob_host=os.environ.get("CLOB_HOST", "https://clob.polymarket.com"),
        chain_id=int(os.environ.get("CHAIN_ID", "137")),
        signature_type=int(os.environ.get("SIGNATURE_TYPE", "0")),
        funder=os.environ.get("FUNDER") or None,
        token_id=token_id,
        kill_switch_file=os.environ.get("KILL_SWITCH_FILE", "KILL_SWITCH"),
        order_size=float(os.environ.get("ORDER_SIZE", "5")),
        quote_refresh_seconds=float(os.environ.get("QUOTE_REFRESH_SECONDS", "10")),
        max_open_orders=int(os.environ.get("MAX_OPEN_ORDERS", "4")),
    )
