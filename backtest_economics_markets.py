"""
Backtests the SAME-market cosine-similarity threshold against CLOSED/SETTLED
Polymarket and Kalshi ECONOMICS markets, scoped to those that closed within
the last N days (default 365). Same methodology as backtest_markets.py
(which scopes to politics/geopolitics) -- this file narrows the market
universe to economics instead, matching compare_economics_markets.py's tag
scoping:
    Polymarket: tag_id 100328 (Economy) and 101800 (Economic Policy)
    Kalshi:     event category "Economics"

Differences from compare_economics_markets.py (needed because the
live-market fetch strategy doesn't work for historical data):
    - Polymarket: closed=true, sorted by closedTime ascending via
      /markets/keyset (after_cursor pagination), so we can stop paginating
      as soon as we cross the cutoff date.
    - Kalshi: has no server-side date filter on /events, and cursor order
      isn't chronological, so we must page through ALL settled events
      (status=settled, with_nested_markets=true) and filter client-side on
      both event category and each nested market's close_time.

Usage:
    python backtest_economics_markets.py
    python backtest_economics_markets.py --days 365 --threshold 0.95
"""

import argparse
import datetime
import json
import os
import time
from dataclasses import dataclass

import requests
from sentence_transformers import SentenceTransformer

GAMMA_API = "https://gamma-api.polymarket.com"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

POLYMARKET_TAG_IDS = {"economy": 100328, "economic_policy": 101800}
KALSHI_CATEGORIES = {"Economics"}


@dataclass
class Market:
    source: str
    id: str
    title: str
    url: str
    closed_at: str
    extra: str = ""  # PM description / Kalshi rules_primary+rules_secondary


def _parse_pm_time(raw):
    return datetime.datetime.strptime(raw[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)


def fetch_polymarket_range(tag_ids, start, end, limit=None):
    """Fetch closed Polymarket markets with closedTime in [start, end).

    Uses the /markets/keyset endpoint (ascending by closedTime) instead of the
    offset-based /markets endpoint, because offset pagination on Gamma hard-caps
    at ~2100 rows regardless of filters/order. The response's `next_cursor`
    field must be sent back as `after_cursor` (per Gamma's OpenAPI spec), not
    `cursor` -- the API silently ignores an unrecognized `cursor` param.
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
                    extra=(m.get("description") or "").strip(),
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


CHECKPOINT_PATH = "/tmp/kalshi_closed_checkpoint.json"


def _save_checkpoint(cursor, pages, markets):
    """Periodic checkpoint so a crash (e.g. a transient network timeout)
    partway through the full-history settled-events scan doesn't throw away
    everything scanned so far -- this scan has no server-side date filter
    and can run for 1000+ pages / tens of minutes, so losing all progress to
    one flaky request is a real cost, not a theoretical one (hit exactly
    this failure mode on 2026-08-28: ReadTimeout after ~1600 pages)."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({
            "cursor": cursor,
            "pages": pages,
            "markets": {k: vars(v) for k, v in markets.items()},
        }, f)


def _load_checkpoint():
    try:
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        markets = {k: Market(**v) for k, v in data["markets"].items()}
        return data["cursor"], data["pages"], markets
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None, 0, {}


def fetch_kalshi_closed(categories, start, end, limit=None, resume=True):
    cursor, pages, markets = _load_checkpoint() if resume else (None, 0, {})
    if pages:
        print(f"  ...resuming from checkpoint: {pages} pages already scanned, {len(markets)} matching markets so far")
    page_size = 200
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
            try:
                resp = requests.get(f"{KALSHI_API}/events", params=params, timeout=30)
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                print(f"  ...network error ({e.__class__.__name__}: {e}), retrying in {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  ...rate limited, backing off {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = 2 ** attempt
                print(f"  ...Kalshi API {resp.status_code}, retrying in {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            _save_checkpoint(cursor, pages, markets)
            raise RuntimeError("Kalshi API retry budget exceeded (checkpoint saved, re-run to resume)")
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
                rules = " ".join(
                    part.strip()
                    for part in (m.get("rules_primary"), m.get("rules_secondary"))
                    if part and part.strip()
                )
                markets[ticker] = Market(
                    source="kalshi",
                    id=ticker,
                    title=title,
                    url=f"https://kalshi.com/markets/{event.get('event_ticker', '')}",
                    closed_at=close_time,
                    extra=rules,
                )
        pages += 1
        if pages % 20 == 0:
            print(f"  ...scanned {pages} pages of settled Kalshi events, "
                  f"{len(markets)} matching markets so far, {time.time()-t0:.0f}s elapsed")
        if pages % 100 == 0:
            _save_checkpoint(data.get("cursor"), pages, markets)
        cursor = data.get("cursor")
        if not cursor or (limit and len(markets) >= limit):
            break
        time.sleep(0.15)
    values = list(markets.values())
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
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
        description="Backtest the SAME-market cosine threshold against closed/settled ECONOMICS markets from the last N days."
    )
    parser.add_argument("--days", type=int, default=365, help="How many days back to backtest (ignored if --start-date is given)")
    parser.add_argument("--start-date", default=None, help="Window start, YYYY-MM-DD (UTC). Overrides --days.")
    parser.add_argument("--end-date", default=None, help="Window end, YYYY-MM-DD (UTC), exclusive. Defaults to now.")
    parser.add_argument("--threshold", type=float, default=0.95, help="Cosine threshold to backtest as the SAME cutoff")
    parser.add_argument("--ultra-same", type=float, default=0.85, help="Lower bound for listing pairs (for manual review)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--limit-pm", type=int, default=None)
    parser.add_argument("--limit-kalshi", type=int, default=None)
    parser.add_argument("--out", default="/tmp/backtest_economics_pairs.json", help="Where to dump pairs >= --ultra-same as JSON")
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

    print("Fetching closed Polymarket economy/economic-policy markets...")
    pm_markets = fetch_polymarket_range(POLYMARKET_TAG_IDS.values(), start, end, limit=args.limit_pm)
    print(f"  -> {len(pm_markets)} Polymarket markets")

    print("Fetching settled Kalshi economics markets (scanning all settled events, no server-side date filter available)...")
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
                [score, pm.title, pm.url, pm.closed_at, k.title, k.url, k.closed_at, pm.extra, k.extra]
                for score, pm, k in review_pairs
            ],
            f,
            indent=1,
        )
    print(f"Dumped {len(review_pairs)} pairs (score >= {args.ultra_same}) to {args.out} for manual review.")


if __name__ == "__main__":
    main()
