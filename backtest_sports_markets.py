"""
Backtests a cosine-similarity threshold for Sports-tagged markets against
CLOSED/SETTLED Polymarket and Kalshi sports markets, scoped to those that
closed within the last N days (default 365).

Same tag/category scoping as the "sports" entry in price_interface.py's
CATEGORY_CONFIGS:
    Polymarket: tag_id 1 (Sports)
    Kalshi:     event category == "Sports"

This is a straight adaptation of backtest_markets.py (the Politics backtest
script) -- same fetch/embed/compare pipeline, just different tag/category
scoping. See that file's docstring for the Gamma/Kalshi pagination notes.

Usage:
    python backtest_sports_markets.py --days 365 --threshold 0.93
"""

import argparse
import datetime
import json
import time
from dataclasses import dataclass

import requests
from sentence_transformers import SentenceTransformer

GAMMA_API = "https://gamma-api.polymarket.com"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

POLYMARKET_TAG_IDS = {"sports": 1}
KALSHI_CATEGORIES = {"Sports"}


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
    """NOTE: for tag_id=1 (Sports), the Gamma /markets/keyset endpoint returns
    a persistent HTTP 500 when order=closedTime&ascending=true is combined
    with this tag (confirmed via direct curl probing -- ascending=false and
    no-order both return 200 for the same tag, so this looks like a
    server-side bug/limitation specific to sorting this large, heavily
    populated tag ascending). Worked around by paginating DESCENDING
    (newest-closed-first) instead and stopping once we page past the window
    start -- equivalent result for a "last N days" query, since we only ever
    want the most-recently-closed markets anyway.
    """
    markets: dict[str, Market] = {}
    page_size = 100
    t0 = time.time()
    pages = 0
    for tag_id in tag_ids:
        cursor = None
        while True:
            params = {
                "limit": page_size,
                "closed": "true",
                "tag_id": tag_id,
                "order": "closedTime",
                "ascending": "false",
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
            pages += 1
            if pages % 20 == 0:
                oldest_on_page = (batch[-1].get("closedTime") or "")[:19]
                print(f"  ...scanned {pages} pages of closed PM sports markets, "
                      f"{len(markets)} kept so far, now at closedTime~{oldest_on_page}, "
                      f"{time.time()-t0:.0f}s elapsed")
            crossed_start = False
            for m in batch:
                closed_time = (m.get("closedTime") or "").strip()
                if not closed_time:
                    continue
                closed_dt = _parse_pm_time(closed_time)
                if closed_dt >= end:
                    continue
                if closed_dt < start:
                    crossed_start = True
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
            if crossed_start or not next_cursor or next_cursor == cursor:
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
                  f"{len(markets)} matching Sports markets so far, {time.time()-t0:.0f}s elapsed")
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
        description="Backtest a cosine threshold for Sports markets against closed/settled markets from the last N days."
    )
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--threshold", type=float, default=0.93, help="Cosine threshold being backtested")
    parser.add_argument("--ultra-same", type=float, default=0.93, help="Lower bound for listing pairs (for manual review)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--limit-pm", type=int, default=None)
    parser.add_argument("--limit-kalshi", type=int, default=None)
    parser.add_argument("--out", default="/tmp/backtest_sports_pairs.json")
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

    print("Fetching closed Polymarket sports markets...")
    pm_markets = fetch_polymarket_range(POLYMARKET_TAG_IDS.values(), start, end, limit=args.limit_pm)
    print(f"  -> {len(pm_markets)} Polymarket markets")

    print("Fetching settled Kalshi sports markets (scanning all settled events, no server-side date filter available)...")
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
