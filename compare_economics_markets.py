"""
Compares Polymarket and Kalshi ECONOMICS markets using sentence-embedding
cosine similarity, to flag markets that are likely asking the same question
on both platforms. Same methodology as compare_markets.py (which scopes to
politics/geopolitics) -- this file just narrows the market universe to
economics instead.

Polymarket markets are pulled from the "Economy" (tag_id=100328) and
"Economic Policy" (tag_id=101800) tags (the pair that best mirrors
Polymarket's own "Economy" section; "Economic Policy" adds a modest number
of markets not already covered by "Economy").

Kalshi has no equivalent tags, so its "Economics" event category is used as
the closest proxy (Kalshi also has a separate "Financials" category, which is
deliberately excluded here to stay scoped to economics specifically, not
markets/finance more broadly).

Each market title is embedded with a sentence-transformers model, and every
Polymarket title is compared against every Kalshi title with cosine
similarity (range -1 to 1). Any pair scoring below --threshold (default 0.9)
is discarded; the rest are printed sorted by score, descending.

Usage:
    pip install -r requirements.txt
    python compare_economics_markets.py
    python compare_economics_markets.py --limit-pm 200 --limit-kalshi 200   # quick test run
    python compare_economics_markets.py --threshold 0.95
"""

import argparse
import time
from dataclasses import dataclass

import requests
from sentence_transformers import SentenceTransformer

GAMMA_API = "https://gamma-api.polymarket.com"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

POLYMARKET_TAG_IDS = {
    "economy": 100328,
    "economic_policy": 101800,
}

# Kalshi's event `category` field is the closest equivalent to Polymarket's
# Economy/Economic Policy tags. Deliberately excludes "Financials" to stay
# scoped to economics rather than markets/finance broadly. The API's own
# `category` query param doesn't actually filter server-side, so we filter
# client-side after fetching (same as compare_markets.py).
KALSHI_CATEGORIES = {"Economics"}


@dataclass
class Market:
    source: str  # "polymarket" | "kalshi"
    id: str
    title: str
    url: str


def fetch_polymarket_markets(tag_ids, limit=None):
    markets: dict[str, Market] = {}
    page_size = 100
    for tag_id in tag_ids:
        offset = 0
        while True:
            resp = requests.get(
                f"{GAMMA_API}/markets",
                params={
                    "limit": page_size,
                    "offset": offset,
                    "active": "true",
                    "closed": "false",
                    "tag_id": tag_id,
                },
                timeout=20,
            )
            if resp.status_code == 422:
                # Gamma API caps offset-based pagination ("offset too large,
                # use /markets/keyset for deeper pagination"); treat that as
                # end-of-results rather than crashing.
                break
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for m in batch:
                question = (m.get("question") or "").strip()
                if not question:
                    continue
                markets[m["id"]] = Market(
                    source="polymarket",
                    id=m["id"],
                    title=question,
                    url=f"https://polymarket.com/event/{m.get('slug', '')}",
                )
            offset += page_size
            if len(batch) < page_size or (limit and len(markets) >= limit):
                break
        if limit and len(markets) >= limit:
            break
    values = list(markets.values())
    return values[:limit] if limit else values


def fetch_kalshi_markets(categories, limit=None):
    markets: dict[str, Market] = {}
    cursor = None
    page_size = 200
    while True:
        params = {"limit": page_size, "status": "open", "with_nested_markets": "true"}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{KALSHI_API}/events", params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        events = data.get("events", [])
        if not events:
            break
        for event in events:
            if event.get("category") not in categories:
                continue
            event_title = (event.get("title") or "").strip()
            for m in event.get("markets", []):
                ticker = m.get("ticker")
                if not ticker:
                    continue
                title = (m.get("title") or event_title).strip()
                sub_title = (m.get("yes_sub_title") or "").strip()
                # Multi-outcome events (e.g. "Which sector leads Q3 GDP
                # growth?") share one generic title across markets; fold in
                # the per-market sub-title so each option is distinguishable.
                if sub_title and sub_title.lower() not in title.lower():
                    title = f"{title} - {sub_title}"
                if not title:
                    continue
                markets[ticker] = Market(
                    source="kalshi",
                    id=ticker,
                    title=title,
                    url=f"https://kalshi.com/markets/{event.get('event_ticker', '')}",
                )
        cursor = data.get("cursor")
        if not cursor or (limit and len(markets) >= limit):
            break
        time.sleep(0.05)
    values = list(markets.values())
    return values[:limit] if limit else values


def embed(model, markets):
    texts = [m.title for m in markets]
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare Polymarket and Kalshi ECONOMICS markets by title embedding similarity."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Cosine similarity threshold below which a pair is discarded (default 0.9)",
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model name")
    parser.add_argument("--limit-pm", type=int, default=None, help="Cap Polymarket markets fetched (for quick test runs)")
    parser.add_argument("--limit-kalshi", type=int, default=None, help="Cap Kalshi markets fetched (for quick test runs)")
    parser.add_argument("--out", default=None, help="If set, dump matched pairs as JSON to this path for further analysis")
    args = parser.parse_args()

    print("Fetching Polymarket economy/economic-policy markets...")
    pm_markets = fetch_polymarket_markets(POLYMARKET_TAG_IDS.values(), limit=args.limit_pm)
    print(f"  -> {len(pm_markets)} Polymarket markets")

    print("Fetching Kalshi economics markets...")
    kalshi_markets = fetch_kalshi_markets(KALSHI_CATEGORIES, limit=args.limit_kalshi)
    print(f"  -> {len(kalshi_markets)} Kalshi markets")

    if not pm_markets or not kalshi_markets:
        print("Nothing to compare.")
        return

    print(f"Loading embedding model '{args.model}'...")
    model = SentenceTransformer(args.model)

    print("Embedding titles...")
    pm_vecs = embed(model, pm_markets)
    kalshi_vecs = embed(model, kalshi_markets)

    print("Computing cosine similarities...")
    sims = pm_vecs @ kalshi_vecs.T  # embeddings are L2-normalized, so dot product == cosine similarity

    matches = []
    for i, pm in enumerate(pm_markets):
        for j, k in enumerate(kalshi_markets):
            score = float(sims[i, j])
            if score >= args.threshold:
                matches.append((score, pm, k))

    matches.sort(key=lambda t: t[0], reverse=True)

    print(f"\n=== MATCHES >= {args.threshold} ({len(matches)}) ===")
    for score, pm, k in matches:
        print(f"cosine={score:.3f}")
        print(f"  Polymarket: {pm.title}\n    {pm.url}")
        print(f"  Kalshi:     {k.title}\n    {k.url}")

    print(
        f"\nSummary: {len(matches)} pairs >= {args.threshold} out of "
        f"{len(pm_markets) * len(kalshi_markets)} pairs compared "
        f"({len(pm_markets)} Polymarket x {len(kalshi_markets)} Kalshi)."
    )

    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump(
                [[score, pm.title, pm.url, k.title, k.url] for score, pm, k in matches],
                f,
                indent=1,
            )
        print(f"Dumped {len(matches)} pairs (score >= {args.threshold}) to {args.out} for manual review.")


if __name__ == "__main__":
    main()
