"""
Compares Polymarket and Kalshi politics/geopolitics markets using sentence-
embedding cosine similarity, to flag markets that are likely asking the same
question on both platforms.

Polymarket markets are pulled from the "Politics" (tag_id=2) and
"Geopolitics" (tag_id=100265) tags. Kalshi has no equivalent tags, so its
closest proxy -- the "Politics", "Elections", and "World" event categories --
is used instead (see KALSHI_CATEGORIES below).

Each market title is embedded with a sentence-transformers model, and every
Polymarket title is compared against every Kalshi title with cosine
similarity (range -1 to 1). Pairs are labeled:
    > --same threshold            -> SAME
    between --possible-low/-high  -> POSSIBLE

Usage:
    pip install -r requirements.txt
    python compare_markets.py
    python compare_markets.py --limit-pm 200 --limit-kalshi 200   # quick test run
"""

import argparse
import time
from dataclasses import dataclass

import requests
from sentence_transformers import SentenceTransformer

GAMMA_API = "https://gamma-api.polymarket.com"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

POLYMARKET_TAG_IDS = {
    "politics": 2,
    "geopolitics": 100265,
}

# Kalshi's event `category` field is the closest equivalent to Polymarket's
# politics/geopolitics tags; the API's own `category` query param doesn't
# actually filter server-side, so we filter client-side after fetching.
KALSHI_CATEGORIES = {"Politics", "Elections", "World"}


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
                # Multi-outcome events (e.g. "Who will be the next Pope?")
                # share one generic title across markets; fold in the
                # per-market sub-title so each candidate is distinguishable.
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
        description="Compare Polymarket and Kalshi politics/geopolitics markets by title embedding similarity."
    )
    parser.add_argument("--ultra-same", type=float, default=0.95, help="Cosine similarity threshold above which a pair is labeled ULTRA SAME")
    parser.add_argument("--same", type=float, default=0.9, help="Cosine similarity threshold above which a pair is labeled SAME (and below --ultra-same)")
    parser.add_argument("--possible-low", type=float, default=0.7, help="Lower bound of the POSSIBLE range")
    parser.add_argument("--possible-high", type=float, default=0.8, help="Upper bound of the POSSIBLE range")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model name")
    parser.add_argument("--limit-pm", type=int, default=None, help="Cap Polymarket markets fetched (for quick test runs)")
    parser.add_argument("--limit-kalshi", type=int, default=None, help="Cap Kalshi markets fetched (for quick test runs)")
    args = parser.parse_args()

    print("Fetching Polymarket politics/geopolitics markets...")
    pm_markets = fetch_polymarket_markets(POLYMARKET_TAG_IDS.values(), limit=args.limit_pm)
    print(f"  -> {len(pm_markets)} Polymarket markets")

    print("Fetching Kalshi politics/elections/world markets...")
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

    ultra_same_pairs = []
    same_pairs = []
    possible_pairs = []
    for i, pm in enumerate(pm_markets):
        for j, k in enumerate(kalshi_markets):
            score = float(sims[i, j])
            if score > args.ultra_same:
                ultra_same_pairs.append((score, pm, k))
            elif score > args.same:
                same_pairs.append((score, pm, k))
            elif args.possible_low <= score <= args.possible_high:
                possible_pairs.append((score, pm, k))

    ultra_same_pairs.sort(key=lambda t: t[0], reverse=True)
    same_pairs.sort(key=lambda t: t[0], reverse=True)
    possible_pairs.sort(key=lambda t: t[0], reverse=True)

    def show(pairs, label):
        print(f"\n=== {label} ({len(pairs)}) ===")
        for score, pm, k in pairs:
            print(f"[{label}] cosine={score:.3f}")
            print(f"  Polymarket: {pm.title}\n    {pm.url}")
            print(f"  Kalshi:     {k.title}\n    {k.url}")

    show(ultra_same_pairs, "ULTRA SAME")
    show(same_pairs, "SAME")
    show(possible_pairs, "POSSIBLE")

    print(
        f"\nSummary: {len(ultra_same_pairs)} ULTRA SAME, {len(same_pairs)} SAME, "
        f"{len(possible_pairs)} POSSIBLE out of {len(pm_markets) * len(kalshi_markets)} pairs compared "
        f"({len(pm_markets)} Polymarket x {len(kalshi_markets)} Kalshi)."
    )


if __name__ == "__main__":
    main()
