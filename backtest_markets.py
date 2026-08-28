"""
Backtests the "SAME market" cosine-similarity threshold identified via
compare_markets.py (>=0.972) against CLOSED/SETTLED Polymarket and Kalshi
politics/geopolitics markets, scoped to those that closed within the last
N days (default 365).

Same tag/category scoping as compare_markets.py:
    Polymarket: tag_id 2 (Politics) and 100265 (Geopolitics)
    Kalshi:     event category in {Politics, Elections, World}

Differences from compare_markets.py (needed because the live-market fetch
strategy doesn't work for historical data):
    - Polymarket: closed=true, sorted by closedTime descending, so we can
      stop paginating as soon as we cross the cutoff date.
    - Kalshi: has no server-side date filter on /events, and cursor order
      isn't chronological, so we must page through ALL settled events
      (status=settled, with_nested_markets=true) and filter client-side on
      both event category and each nested market's close_time.

Usage:
    python backtest_markets.py
    python backtest_markets.py --days 365 --threshold 0.972
"""

import argparse
import datetime
import time
from dataclasses import dataclass

import requests
from sentence_transformers import SentenceTransformer

GAMMA_API = "https://gamma-api.polymarket.com"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

POLYMARKET_TAG_IDS = {"politics": 2, "geopolitics": 100265}
KALSHI_CATEGORIES = {"Politics", "Elections", "World"}


@dataclass
class Market:
    source: str
    id: str
    title: str
    url: str
    closed_at: str


def _parse_pm_time(raw):
    return datetime.datetime.strptime(raw[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)


def fetch_polymarket_range(tag_ids, start, end, limit=None):
    """Fetch closed Polymarket markets with closedTime in [start, end).

    Uses the /markets/keyset endpoint (ascending by closedTime) instead of the
    offset-based /markets endpoint, because offset pagination on Gamma hard-caps
    at ~2100 rows regardless of filters/order (confirmed via direct probing:
    offset+limit > ~2100 always returns 422 "offset too large, use
    /markets/keyset for deeper pagination", even though the docs point at
    keyset as the fix).

    Also note: the response's `next_cursor` field is NOT the request param name.
    The API silently ignores an unrecognized `cursor` param and just re-returns
    page 1 forever (confirmed by passing garbage cursor values and getting an
    identical response). The real param, per Gamma's own OpenAPI spec
    (https://gamma-api.polymarket.com/openapi.json), is `after_cursor`.
    """
    markets: dict[str, Market] = {}
    page_size = 100
    for tag_id in tag_ids:
        cursor = None
        while True:
            params = {
                "limit": page_size,
                "closed": "true",
                "tag_id": tag_id,
                "order": "closedTime",
                "ascending": "true",
            }
            if cursor:
                params["after_cursor"] = cursor
            for attempt in range(6):
                resp = requests.get(f"{GAMMA_API}/markets/keyset", params=params, timeout=20)
                if resp.status_code >= 500:
                    wait = 2 ** attempt
                    print(f"  ...Gamma API {resp.status_code}, retrying in {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            else:
                raise RuntimeError("Gamma API failed after retries")
            data = resp.json()
            batch = data.get("markets", [])
            if not batch:
                break
            crossed_end = False
            for m in batch:
                closed_time = (m.get("closedTime") or "").strip()
                if not closed_time:
                    continue
                closed_dt = _parse_pm_time(closed_time)
                if closed_dt < start:
                    continue
                if closed_dt >= end:
                    crossed_end = True
                    break
                question = (m.get("question") or "").strip()
                if not question:
                    continue
                markets[m["id"]] = Market(
                    source="polymarket",
                    id=m["id"],
                    title=question,
                    url=f"https://polymarket.com/event/{m.get('slug', '')}",
                    closed_at=closed_time,
                )
            next_cursor = data.get("next_cursor")
            if crossed_end or not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
            if limit and len(markets) >= limit:
                break
        if limit and len(markets) >= limit:
            break
    values = list(markets.values())
    return values[:limit] if limit else values


def fetch_kalshi_closed(categories, start, end, limit=None):
    markets: dict[str, Market] = {}
    cursor = None
    page_size = 200
    pages = 0
    t0 = time.time()
    while True:
        params = {
            "limit": page_size,
            "status": "settled",
            "with_nested_markets": "true",
        }
        if cursor:
            params["cursor"] = cursor
        for attempt in range(6):
            resp = requests.get(f"{KALSHI_API}/events", params=params, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  ...rate limited, backing off {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            raise RuntimeError("Kalshi API rate limit exceeded retry budget")
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
                close_time = m.get("close_time")
                if not close_time:
                    continue
                close_dt = datetime.datetime.strptime(close_time, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=datetime.timezone.utc
                )
                if close_dt < start or close_dt >= end:
                    continue
                title = (m.get("title") or event_title).strip()
                sub_title = (m.get("yes_sub_title") or "").strip()
                if sub_title and sub_title.lower() not in title.lower():
                    title = f"{title} - {sub_title}"
                if not title:
                    continue
                markets[ticker] = Market(
                    source="kalshi",
                    id=ticker,
                    title=title,
                    url=f"https://kalshi.com/markets/{event.get('event_ticker', '')}",
                    closed_at=close_time,
                )
        pages += 1
        if pages % 20 == 0:
            print(f"  ...scanned {pages} pages of settled Kalshi events, "
                  f"{len(markets)} matching markets so far, {time.time()-t0:.0f}s elapsed")
        cursor = data.get("cursor")
        if not cursor or (limit and len(markets) >= limit):
            break
        time.sleep(0.15)
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
        description="Backtest the SAME-market cosine threshold against closed/settled markets from the last N days."
    )
    parser.add_argument("--days", type=int, default=365, help="How many days back to backtest (ignored if --start-date is given)")
    parser.add_argument("--start-date", default=None, help="Window start, YYYY-MM-DD (UTC). Overrides --days.")
    parser.add_argument("--end-date", default=None, help="Window end, YYYY-MM-DD (UTC), exclusive. Defaults to now.")
    parser.add_argument("--threshold", type=float, default=0.972, help="Cosine threshold to backtest as the SAME cutoff")
    parser.add_argument("--ultra-same", type=float, default=0.95, help="Lower bound for listing pairs (for manual review)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--limit-pm", type=int, default=None)
    parser.add_argument("--limit-kalshi", type=int, default=None)
    parser.add_argument("--out", default="/tmp/backtest_pairs.json", help="Where to dump pairs >= --ultra-same as JSON")
    args = parser.parse_args()

    now = datetime.datetime.now(datetime.timezone.utc)
    if args.start_date:
        start = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
        end = (
            datetime.datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
            if args.end_date
            else now
        )
    else:
        start = now - datetime.timedelta(days=args.days)
        end = now
    print(f"Backtest window: closed/settled in [{start.isoformat()}, {end.isoformat()})")

    print("Fetching closed Polymarket politics/geopolitics markets...")
    pm_markets = fetch_polymarket_range(POLYMARKET_TAG_IDS.values(), start, end, limit=args.limit_pm)
    print(f"  -> {len(pm_markets)} Polymarket markets")

    print("Fetching settled Kalshi politics/elections/world markets (scanning all settled events, no server-side date filter available)...")
    kalshi_markets = fetch_kalshi_closed(KALSHI_CATEGORIES, start, end, limit=args.limit_kalshi)
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
    sims = pm_vecs @ kalshi_vecs.T

    review_pairs = []
    for i, pm in enumerate(pm_markets):
        for j, k in enumerate(kalshi_markets):
            score = float(sims[i, j])
            if score >= args.ultra_same:
                review_pairs.append((score, pm, k))
    review_pairs.sort(key=lambda t: t[0])

    at_or_above_threshold = sum(1 for score, _, _ in review_pairs if score >= args.threshold)
    print(
        f"\n{len(review_pairs)} pairs >= {args.ultra_same} "
        f"({at_or_above_threshold} of them >= backtest threshold {args.threshold}) "
        f"out of {len(pm_markets)} x {len(kalshi_markets)} = {len(pm_markets)*len(kalshi_markets)} pairs compared."
    )

    import json
    with open(args.out, "w") as f:
        json.dump(
            [
                [score, pm.title, pm.url, pm.closed_at, k.title, k.url, k.closed_at]
                for score, pm, k in review_pairs
            ],
            f,
            indent=1,
        )
    print(f"Dumped {len(review_pairs)} pairs (score >= {args.ultra_same}) to {args.out} for manual review.")


if __name__ == "__main__":
    main()
