"""
Live price-comparison interface for matched Polymarket vs Kalshi market
pairs, rendered as two collapsible dropdowns -- Politics and Economics.

Builds on compare_markets.py's matching logic, but instead of just listing
matched pairs, it also pulls the current "Yes" side price from both platforms
and renders a static HTML dashboard so you can eyeball where the two
platforms disagree on price for what's very likely the same real-world
question.

Categories (see CATEGORY_CONFIGS):
    - Politics:  cosine > 0.977 ("ULTRA SAME") is precise enough on its own
      for politics/geopolitics/elections titles.
    - Economics: raw cosine is only ~30-55% precise for economics markets
      (numeric brackets read as near-identical text to the embedding model),
      so a lower cosine > 0.85 floor is combined with the regex numeric-
      boundary match, live year/month/quarter alignment, and an EXACT
      (same-payoff) operator/direction classification from
      economics_quintuple_filter.py before a pair counts as matched. A full
      manual backtest audit found this combination is needed -- cosine +
      boundary-match + date-align alone still let ~77% of "matches" through
      that were actually bracket-edge/complement/threshold mismatches, not
      the same tradeable question.

Price sourcing:
    - Polymarket: Gamma API's /markets response already includes live
      outcomePrices / bestBid / bestAsk for active markets, no extra calls
      needed. outcomePrices[0] corresponds to the "Yes" outcome (outcomes[0]),
      confirmed by direct inspection of sample live markets.
    - Kalshi: the /events (and /markets) endpoints return yes_bid/yes_ask/
      last_price as null for the vast majority of markets even when they have
      resting liquidity -- confirmed by scanning 3,000+ live events and
      finding zero with a populated yes_bid. The actual live book is only
      available via GET /markets/{ticker}/orderbook, which returns two
      dollar-denominated bid ladders: `yes_dollars` (resting buy-Yes orders)
      and `no_dollars` (resting buy-No orders). Standard complementary-book
      convention is used to derive a synthetic Yes market:
          yes_bid = max(yes_dollars prices)          (highest buy-Yes order)
          yes_ask = 1 - max(no_dollars prices)        (buying No == selling Yes)
          yes_mid = mean(yes_bid, yes_ask) when both present

Live refresh mode (--interval, default 60s):
    The expensive step is the match computation (fetching the full PM + Kalshi
    market universe and embedding ~16k titles per category, ~30-60s). The
    prices themselves are cheap to re-poll. So each category's match set
    (score + which PM id / Kalshi ticker pair) is cached to a per-category
    file derived from --cache on first run, then the script loops forever
    re-fetching ONLY prices for those cached pairs every --interval seconds
    (default 60) and rewriting --out in place:
        - Polymarket: one batched GET /markets?id=...&id=...  (repeated `id`
          params in a single request, confirmed supported) for all cached
          PM ids at once.
        - Kalshi: still one GET /markets/{ticker}/orderbook per cached ticker
          (no batch endpoint), same as the one-shot path.
    The rendered HTML also carries a <meta http-equiv="refresh"> tag set to
    the same interval, so a browser tab left open on --out reloads itself and
    picks up each rewrite automatically. Pass --interval 0 for a single
    one-shot render instead of looping.

    Independently of price refresh, an incremental scanner checks for
    newly-listed markets on both platforms every --scan-interval seconds
    (default 900s = 15min), hard-capped at MAX_SCAN_INTERVAL_SECONDS (3600s
    = 1hr) -- new markets are guaranteed to be scanned and tested against
    the existing universe at least once an hour, for every category.

    Pass --rebuild to force recomputing every category's match set from
    scratch (e.g. after moving a similarity threshold) instead of reusing
    --cache/--universe-cache.

Usage:
    python price_interface.py                                    # loops, 60s price refresh
    python price_interface.py --interval 0                       # one-shot, no loop
    python price_interface.py --interval 60 --rebuild             # recompute matches first
"""

import argparse
import json
import time
import webbrowser
from dataclasses import dataclass, field

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from economics_quintuple_filter import (
    boundary_match,
    date_align_live,
    classify_relationship,
    extract_months,
    extract_years,
    is_year_end_terminal,
    CAT_EXACT,
    CAT_STRICT_VS_INCLUSIVE,
)

GAMMA_API = "https://gamma-api.polymarket.com"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

# How long a matched pair keeps its "NEW" badge in the rendered HTML after
# the incremental scanner first finds it.
NEW_BADGE_WINDOW_SECONDS = 15 * 60

# Hard ceiling on how long the --interval loop is allowed to go between
# scans for newly-listed markets, regardless of what --scan-interval is set
# to. New markets must never go undetected for more than an hour.
MAX_SCAN_INTERVAL_SECONDS = 3600

# Displayed page auto-refresh cadence (meta-refresh tag + per-price JS
# countdown + blackout-flash timing). Deliberately decoupled from
# --interval/args.interval, which continues to drive the *backend's* own
# refresh-loop polling cadence unchanged. In practice a full refresh cycle
# (Kalshi has no batch order-book endpoint, so each pair needs its own
# request) takes long enough that the backend's real cadence naturally runs
# closer to ~90s than to a nominal 60s --interval, so the on-page countdown
# is pinned to 90s to match what users actually observe, rather than the
# raw --interval value. The top-of-page status box is unaffected by this --
# it always shows the real, actual backend refresh/scan timestamps.
PAGE_REFRESH_DISPLAY_SECONDS = 90


def economics_pair_ok(pm_title, k_title, pm_extra="", k_extra=""):
    """Extra correctness gate applied ONLY to the Economics category, on top
    of the cosine threshold. Cosine similarity alone is only ~30-55%
    precise for economics markets (see economics_quintuple_filter.py's
    backtest audit) because embeddings can't tell a Polymarket bracket's
    edge from a Kalshi ladder rung that happens to share the same number.
    Require the regex numeric-boundary match, live year/month/quarter
    alignment, AND an operator/direction classification of EXACT or
    STRICT_VS_INCLUSIVE before two economics markets are treated as the
    same tradeable question.

    `pm_extra`/`k_extra` are optional resolution-criteria text (PM
    `description` / Kalshi `rules_primary`+`rules_secondary`) passed
    through to date_align_live's growth-basis/GDP-basis checks, so a
    real/nominal or QoQ/YoY qualifier stated only in the rules prose (not
    the title) still gets caught. NOT used for numeric boundary parsing or
    for classify_relationship -- those stay title-only by design, since
    rules text often lists a whole bracket ladder rather than just this
    specific market's own bracket.

    Returns (ok, risk):
        ok    -- False if the pair fails the gate entirely (not shown).
        risk  -- None for a normal EXACT match (identical payoff), or
                 "boundary" for a STRICT_VS_INCLUSIVE match -- same number,
                 same direction, same period, but one side is a strict
                 inequality (>  / <) and the other inclusive (>= / <=), so
                 the two markets only pay out differently in the knife-edge
                 case where the reported value lands exactly on the
                 threshold. Kept (not dropped) but flagged as higher risk
                 so it can be shown separately in the dashboard.
    """
    if not boundary_match(pm_title, k_title, pct_tol=0.05, dollar_tol_frac=0.01):
        return False, None
    if not date_align_live(pm_title, k_title, pm_extra, k_extra):
        return False, None
    cat, _ = classify_relationship(pm_title, k_title)
    if cat == CAT_EXACT:
        return True, None
    if cat == CAT_STRICT_VS_INCLUSIVE:
        return True, "boundary"
    return False, None


def resolve_terminal_siblings(pairs):
    """Drop matched pairs where a "terminal" PM question (e.g. "at the end
    of 2026", no specific month/quarter attached) has been matched against
    MULTIPLE Kalshi siblings that each pin to a different specific date
    within that year (e.g. one ticker per Fed meeting: Sep/Oct/Dec). Only
    the chronologically LATEST such sibling can actually be the same bet as
    "the end of the year" -- the rate (or whatever's being measured) can
    still change between an earlier meeting/reading and year-end, so an
    earlier sibling is a genuinely different, earlier-resolving question,
    not just a different phrasing of the same one. Surfaced by the user
    spotting PM's single "end of 2026" Fed-rate market matched against
    Kalshi's Sep 16, Oct 28, AND Dec 9 meeting-specific tickers at once.

    `pairs`: list of cache-shaped dicts (pm_id, pm_title, kalshi_ticker,
    kalshi_title, ...). Non-destructive; returns a filtered list.

    "Latest" is a real (year, month) comparison, not a bare month number --
    a Kalshi sibling with no *unambiguous* year attached (either directly on
    its own title, or inherited from a PM title mentioning exactly one year)
    is skipped rather than guessed at, so e.g. a "Jan" sibling is never
    silently treated as earlier than a "Dec" sibling just because 1 < 12
    when they actually belong to different years.
    """
    groups: dict[str, list] = {}
    for p in pairs:
        groups.setdefault(p["pm_id"], []).append(p)

    drop_keys = set()
    for pm_id, group in groups.items():
        if len(group) < 2 or not is_year_end_terminal(group[0]["pm_title"]):
            continue

        dated = []
        for p in group:
            months = extract_months(p["kalshi_title"])
            if not months:
                continue
            k_years = extract_years(p["kalshi_title"])
            if len(k_years) == 1:
                year = next(iter(k_years))
            elif not k_years:
                pm_years = extract_years(p["pm_title"])
                if len(pm_years) != 1:
                    continue  # no unambiguous year anywhere -> can't order safely
                year = next(iter(pm_years))
            else:
                continue  # Kalshi title mentions >1 year -> ambiguous, skip
            dated.append((p, (year, max(months))))

        if len(dated) < 2:
            continue
        latest_key = max(key for _, key in dated)
        for p, key in dated:
            if key != latest_key:
                drop_keys.add((p["pm_id"], p["kalshi_ticker"]))

    return [p for p in pairs if (p["pm_id"], p["kalshi_ticker"]) not in drop_keys]


# Each entry drives one collapsible dropdown section in the rendered HTML.
# "threshold" is the cosine floor; "extra_filter(pm_title, k_title) -> (ok,
# risk)", when set, is required IN ADDITION to the cosine floor before a
# pair counts as a match (used by Economics to layer on the regex/date/
# direction signals above, since raw cosine is too weak on its own for that
# category). risk is None for a normal match, or a short tag (e.g.
# "boundary") for a match that's kept but should be grouped/flagged as
# higher risk in the rendered dashboard.
CATEGORY_CONFIGS = {
    "politics": {
        "label": "Politics",
        "pm_tag_ids": {"politics": 2, "geopolitics": 100265},
        "kalshi_categories": {"Politics", "Elections", "World"},
        "threshold": 0.977,
        "extra_filter": None,
    },
    "economics": {
        "label": "Economics",
        "pm_tag_ids": {"economy": 100328, "economic_policy": 101800},
        "kalshi_categories": {"Economics"},
        "threshold": 0.85,
        "extra_filter": economics_pair_ok,
    },
}


@dataclass
class PMMarket:
    id: str
    title: str
    url: str
    yes_price: float | None = None
    yes_ask: float | None = None
    yes_bid: float | None = None
    description: str = ""

    @property
    def no_ask(self):
        # Best ask to buy No = 1 - best bid for Yes (selling Yes at the bid
        # is economically equivalent to buying No at 1-bid).
        return 1 - self.yes_bid if self.yes_bid is not None else None


@dataclass
class KalshiMarket:
    ticker: str
    title: str
    url: str
    yes_bid: float | None = None
    yes_ask: float | None = None
    rules: str = ""

    @property
    def yes_mid(self):
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2
        return self.yes_bid if self.yes_bid is not None else self.yes_ask

    @property
    def no_ask(self):
        return 1 - self.yes_bid if self.yes_bid is not None else None


def fetch_polymarket_markets(tag_ids, limit=None):
    markets: dict[str, PMMarket] = {}
    page_size = 100
    for tag_id in tag_ids:
        offset = 0
        while True:
            resp = requests.get(
                f"{GAMMA_API}/markets",
                params={"limit": page_size, "offset": offset, "active": "true", "closed": "false", "tag_id": tag_id},
                timeout=20,
            )
            if resp.status_code == 422:
                break
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for m in batch:
                question = (m.get("question") or "").strip()
                if not question:
                    continue
                yes_price = None
                try:
                    outcomes = json.loads(m.get("outcomes") or "[]")
                    prices = json.loads(m.get("outcomePrices") or "[]")
                    if outcomes and prices and outcomes[0].strip().lower() == "yes":
                        yes_price = float(prices[0])
                except Exception:
                    pass
                yes_bid = float(m["bestBid"]) if m.get("bestBid") is not None else None
                if yes_price is None and yes_bid is not None and m.get("bestAsk") is not None:
                    yes_price = (yes_bid + float(m["bestAsk"])) / 2
                yes_ask = float(m["bestAsk"]) if m.get("bestAsk") is not None else None
                # The market's own "slug" only resolves to a real page for
                # single-market events. For grouped/multi-outcome events
                # (e.g. "Fed rate cuts", "GDP growth band") each sibling
                # market has a distinct slug that Polymarket's router 404s
                # on -- the real, working URL is the parent EVENT's slug.
                events = m.get("events") or []
                event_slug = events[0].get("slug") if events else None
                markets[m["id"]] = PMMarket(
                    id=m["id"],
                    title=question,
                    url=f"https://polymarket.com/event/{event_slug or m.get('slug', '')}",
                    yes_price=yes_price,
                    yes_ask=yes_ask,
                    yes_bid=yes_bid,
                    description=(m.get("description") or "").strip(),
                )
            offset += page_size
            if len(batch) < page_size or (limit and len(markets) >= limit):
                break
        if limit and len(markets) >= limit:
            break
    values = list(markets.values())
    return values[:limit] if limit else values


def fetch_kalshi_markets(categories, limit=None):
    markets: dict[str, KalshiMarket] = {}
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
                if sub_title and sub_title.lower() not in title.lower():
                    title = f"{title} - {sub_title}"
                if not title:
                    continue
                # Kalshi's router needs the series segment too -- a bare
                # event-ticker path (the old behavior here) doesn't resolve.
                # Real URLs are /markets/{series_ticker}/{event_ticker}
                # (lowercased); the human-readable slug segment some pages
                # show is cosmetic and not required for routing.
                series_ticker = (event.get("series_ticker") or "").lower()
                event_ticker = (event.get("event_ticker") or "").lower()
                rules = " ".join(
                    part.strip()
                    for part in (m.get("rules_primary"), m.get("rules_secondary"))
                    if part and part.strip()
                )
                markets[ticker] = KalshiMarket(
                    ticker=ticker,
                    title=title,
                    url=f"https://kalshi.com/markets/{series_ticker}/{event_ticker}",
                    rules=rules,
                )
        cursor = data.get("cursor")
        if not cursor or (limit and len(markets) >= limit):
            break
        time.sleep(0.05)
    values = list(markets.values())
    return values[:limit] if limit else values


def fill_kalshi_orderbook_prices(kalshi_markets):
    """Fetch live Yes bid/ask for a (small) set of matched Kalshi markets via
    the per-market orderbook endpoint, since the bulk /events listing returns
    null bid/ask/last_price for almost all markets."""
    for km in kalshi_markets:
        for attempt in range(4):
            resp = requests.get(f"{KALSHI_API}/markets/{km.ticker}/orderbook", timeout=15)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                break
            data = resp.json()
            ob = data.get("orderbook_fp") or data.get("orderbook") or {}
            yes_levels = ob.get("yes_dollars") or ob.get("yes") or []
            no_levels = ob.get("no_dollars") or ob.get("no") or []
            try:
                if yes_levels:
                    km.yes_bid = max(float(p) for p, _ in yes_levels)
                if no_levels:
                    best_no_bid = max(float(p) for p, _ in no_levels)
                    km.yes_ask = 1 - best_no_bid
            except Exception:
                pass
            break
        time.sleep(0.05)


def fetch_polymarket_prices_batch(ids, chunk_size=50):
    """Refresh just the Yes price + Yes best-ask for a known list of PM market
    ids, via repeated `id` query params in one request per chunk (confirmed
    Gamma supports this -- much cheaper than re-listing the whole active
    market set)."""
    prices: dict[str, tuple[float | None, float | None, float | None]] = {}
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i : i + chunk_size]
        resp = requests.get(
            GAMMA_API + "/markets",
            params=[("id", mid) for mid in chunk] + [("limit", len(chunk))],
            timeout=20,
        )
        resp.raise_for_status()
        for m in resp.json():
            yes_price = None
            try:
                outcomes = json.loads(m.get("outcomes") or "[]")
                out_prices = json.loads(m.get("outcomePrices") or "[]")
                if outcomes and out_prices and outcomes[0].strip().lower() == "yes":
                    yes_price = float(out_prices[0])
            except Exception:
                pass
            yes_bid = float(m["bestBid"]) if m.get("bestBid") is not None else None
            if yes_price is None and yes_bid is not None and m.get("bestAsk") is not None:
                yes_price = (yes_bid + float(m["bestAsk"])) / 2
            yes_ask = float(m["bestAsk"]) if m.get("bestAsk") is not None else None
            prices[m["id"]] = (yes_price, yes_ask, yes_bid)
    return prices


def save_match_cache(matched, threshold, cache_path):
    now = time.time()
    payload = {
        "threshold": threshold,
        "built_at": now,
        "pairs": [
            {
                "score": score,
                "pm_id": pm.id,
                "pm_title": pm.title,
                "pm_url": pm.url,
                "kalshi_ticker": km.ticker,
                "kalshi_title": km.title,
                "kalshi_url": km.url,
                "first_seen": now,
                "risk": risk,
            }
            for score, pm, km, risk in matched
        ],
    }
    payload["pairs"] = resolve_terminal_siblings(payload["pairs"])
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=1)
    return payload


def load_match_cache(cache_path):
    with open(cache_path) as f:
        return json.load(f)


def merge_new_matches(cache, new_matches, cache_path):
    """Fold newly-discovered ULTRA SAME pairs (from scan_for_new_matches) into
    an existing match cache, tagging each with a first_seen timestamp so the
    HTML can badge them as NEW, and re-persist the cache to disk."""
    existing_keys = {(p["pm_id"], p["kalshi_ticker"]) for p in cache["pairs"]}
    now = time.time()
    added = 0
    for score, pm, km, risk in new_matches:
        key = (pm.id, km.ticker)
        if key in existing_keys:
            continue
        cache["pairs"].append(
            {
                "score": score,
                "pm_id": pm.id,
                "pm_title": pm.title,
                "pm_url": pm.url,
                "kalshi_ticker": km.ticker,
                "kalshi_title": km.title,
                "kalshi_url": km.url,
                "first_seen": now,
                "risk": risk,
            }
        )
        existing_keys.add(key)
        added += 1
    cache["pairs"] = resolve_terminal_siblings(cache["pairs"])
    cache["pairs"].sort(key=lambda p: p["score"], reverse=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=1)
    return cache, added


def save_universe(pm_markets, pm_vecs, kalshi_markets, kalshi_vecs, path):
    """Persist every market we've ever embedded (title + url + its embedding
    vector) so future scans can diff the live listing against it and only
    embed genuinely new markets. Embeddings are the expensive part, so this
    is what makes incremental scanning cheap."""
    np.savez(
        path,
        pm_ids=np.array([m.id for m in pm_markets], dtype=object),
        pm_titles=np.array([m.title for m in pm_markets], dtype=object),
        pm_urls=np.array([m.url for m in pm_markets], dtype=object),
        pm_vecs=np.asarray(pm_vecs, dtype=np.float32),
        kalshi_ids=np.array([m.ticker for m in kalshi_markets], dtype=object),
        kalshi_titles=np.array([m.title for m in kalshi_markets], dtype=object),
        kalshi_urls=np.array([m.url for m in kalshi_markets], dtype=object),
        kalshi_vecs=np.asarray(kalshi_vecs, dtype=np.float32),
    )


def load_universe(path):
    try:
        data = np.load(path, allow_pickle=True)
    except FileNotFoundError:
        return None
    pm_markets = [PMMarket(id=str(i), title=str(t), url=str(u)) for i, t, u in zip(data["pm_ids"], data["pm_titles"], data["pm_urls"])]
    kalshi_markets = [KalshiMarket(ticker=str(i), title=str(t), url=str(u)) for i, t, u in zip(data["kalshi_ids"], data["kalshi_titles"], data["kalshi_urls"])]
    return {
        "pm_markets": pm_markets,
        "pm_vecs": data["pm_vecs"],
        "kalshi_markets": kalshi_markets,
        "kalshi_vecs": data["kalshi_vecs"],
    }


def embed(model, texts):
    return model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)


def _magnitude_class(value):
    # Color by size of the guaranteed-hedge profit: >=0 only, since a
    # negative value means there's no arbitrage on this pair at all.
    if value is None or value <= 0:
        return ""
    if value >= 0.03:
        return "diff-large"
    elif value >= 0.01:
        return "diff-medium"
    else:
        return "diff-small"


def _render_rows(pairs):
    rows = []
    for score, pm, km, best_arb, best_arb_label, is_new, risk in pairs:
        pm_yes_ask = f"{pm.yes_ask:.3f}" if pm.yes_ask is not None else "n/a"
        pm_no_ask = f"{pm.no_ask:.3f}" if pm.no_ask is not None else "n/a"
        km_yes_ask = f"{km.yes_ask:.3f}" if km.yes_ask is not None else "n/a"
        km_no_ask = f"{km.no_ask:.3f}" if km.no_ask is not None else "n/a"
        # best_arb (computed upstream in build_section_pairs) is already
        # "whichever combo is available": if both PM-Yes+Kalshi-No and
        # PM-No+Kalshi-Yes are quoted it's the better of the two, and if only
        # one side has both asks quoted it's simply that one combo's value --
        # so the same threshold logic below handles the "missing one best
        # ask" case automatically, with no separate branch needed.
        #
        # A row shows one of three things in the Best-Arb column:
        #   - a genuine positive hedge (best_arb > 0, i.e. the two best asks
        #     sum to less than $1): "+0.XXX" plus a label naming which combo
        #     achieves it -- the normal profit display.
        #   - a guaranteed loss (best_arb <= 0, i.e. the best available
        #     combo's best asks sum to $1 or more): the actual (negative)
        #     value -- 1 minus that sum -- plus a red "LOSS" marker, so it's
        #     clear this pair was evaluated and found unprofitable rather
        #     than silently missing data.
        #   - truly missing data (best_arb is None, i.e. neither combo had
        #     both asks quoted at all): a plain dash.
        has_profit = best_arb is not None and best_arb > 0
        has_loss = best_arb is not None and best_arb <= 0
        if has_profit:
            arb_str = f"{best_arb:+.3f}"
            arb_class = _magnitude_class(best_arb)
            label_html = f'<div class="arb-label">{best_arb_label}</div>'
        elif has_loss:
            arb_str = f"{best_arb:.3f}"
            arb_class = ""
            # A value that rounds to 0.000 at the displayed precision isn't
            # really a "loss" -- the two best asks land on exactly $1, i.e.
            # break-even -- so it gets its own neutral "Balanced" label
            # rather than the red/lilac LOSS marker.
            is_balanced = arb_str in ("0.000", "-0.000")
            label_html = f'<div class="arb-loss">{"Balanced" if is_balanced else "LOSS"}</div>'
        else:
            arb_str = "-"
            arb_class = ""
            label_html = ""

        row_class = "row-new" if is_new else ""
        new_badge = '<span class="new-badge">NEW</span> ' if is_new else ""
        # Each ask price carries its own countdown badge (ticked down client-
        # side by the <script> at the bottom of render_html) showing seconds
        # remaining until the page's next auto-refresh pulls a new price.
        countdown = '<div class="countdown"></div>'
        rows.append(f"""
        <tr class="{row_class}">
          <td class="score">{score:.4f}</td>
          <td>{new_badge}<a href="{pm.url}" target="_blank">{pm.title}</a></td>
          <td class="price ask">{pm_yes_ask}{countdown}</td>
          <td class="price ask">{pm_no_ask}{countdown}</td>
          <td><a href="{km.url}" target="_blank">{km.title}</a></td>
          <td class="price ask">{km_yes_ask}{countdown}</td>
          <td class="price ask">{km_no_ask}{countdown}</td>
          <td class="price divider {arb_class}">{arb_str}{label_html}</td>
        </tr>""")
    return rows


def render_html(sections, out_path, refresh_seconds=None, last_scan_at=None, last_refresh_at=None):
    """sections: list of dicts, one per category dropdown, each with:
        key            -- short slug used for HTML element ids (e.g. "politics")
        label          -- display name (e.g. "Politics")
        pairs          -- list of (score, pm, km, best_arb, best_arb_label, is_new, risk)
                           risk is None for a normal match, or "boundary"
                           for a STRICT_VS_INCLUSIVE match -- rendered in a
                           separate "HIGHER RISK" sub-table at the bottom
                           of the section instead of being dropped.
        threshold      -- cosine floor used for this category
        match_built_at -- epoch seconds the match set was last fully rebuilt
        new_count      -- pairs newly found by the incremental scanner
        extra_filter   -- bool, whether an extra correctness filter (beyond
                           cosine) was applied for this category

    last_scan_at, last_refresh_at: epoch seconds of the most recent
    incremental new-market scan and most recent live-price refresh
    (independent schedules -- see refresh_and_render/main's loop). Rendered
    verbatim (to the second, with time zone) in the status box at the top of
    the page. last_refresh_at defaults to "now" since render_html is always
    called as part of a refresh.
    """
    if last_refresh_at is None:
        last_refresh_at = time.time()

    def _fmt_ts(epoch):
        return time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(epoch)) if epoch else "not yet run"

    total_pairs = sum(len(s["pairs"]) for s in sections)
    total_new = sum(s["new_count"] for s in sections)
    oldest_build = min((s["match_built_at"] for s in sections if s["match_built_at"]), default=None)
    # A "profit opportunity" is a pair with a genuine positive hedge (same
    # threshold _render_rows uses to decide between the +profit display and
    # the LOSS/Balanced/dash displays): best_arb is not None and > 0.
    profitable_arbs = [p[3] for s in sections for p in s["pairs"] if p[3] is not None and p[3] > 0]
    total_profit_opportunities = len(profitable_arbs)
    # Aggregate ROI across every profitable pair, assuming $1 notional per
    # pair: best_arb IS the profit per $1 notional (best_arb = 1 - cost, by
    # construction in build_section_pairs), so cost = 1 - best_arb. Summing
    # profit and cost separately (rather than averaging the per-pair %)
    # weights the aggregate by how much capital each opportunity actually
    # requires, e.g. a $0.98 total-cost pair barely moves the number even if
    # its individual % looks similar to a $0.60 total-cost pair.
    total_profit_dollars = sum(profitable_arbs)
    total_cost_dollars = sum(1.0 - a for a in profitable_arbs)
    profit_pct = (total_profit_dollars / total_cost_dollars * 100.0) if total_cost_dollars > 0 else None

    refresh_tag = f'<meta http-equiv="refresh" content="{refresh_seconds}">' if refresh_seconds else ""
    match_note = (
        f" Match sets were last fully rebuilt "
        f"{time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(oldest_build))} "
        f"(oldest of the categories below); the incremental scanner tops each one up with "
        f"newly-listed markets at least once an hour between rebuilds."
        if oldest_build
        else ""
    )
    new_note = f" <span class=\"new-badge\">{total_new} NEW</span> pair(s) found by the scanner since the last full rebuild." if total_new else ""

    thead = """<thead><tr>
      <th>Cosine</th><th>Polymarket question</th><th class="ask">PM Yes Ask</th><th class="ask">PM No Ask</th>
      <th>Kalshi question</th><th class="ask">Kalshi Yes Ask</th><th class="ask">Kalshi No Ask</th>
      <th class="divider">Best Arb</th>
    </tr></thead>"""

    section_blocks = []
    for s in sections:
        normal_pairs = [p for p in s["pairs"] if p[6] != "boundary"]
        risk_pairs = [p for p in s["pairs"] if p[6] == "boundary"]
        rows = _render_rows(normal_pairs)
        risk_rows = _render_rows(risk_pairs)
        section_new_note = f' <span class="new-badge">{s["new_count"]} NEW</span>' if s["new_count"] else ""
        filter_note = (
            " Also requires a regex numeric-boundary match and live year/month/quarter alignment, "
            "plus an operator/direction classification of EXACT or STRICT_VS_INCLUSIVE -- see economics_quintuple_filter.py."
            if s["extra_filter"]
            else ""
        )
        risk_block = ""
        if risk_rows:
            risk_block = f"""
  <h3 class="risk-heading">HIGHER RISK</h3>
  <div class="risk-note">Same threshold, direction, and time period on both sides, but one market uses a strict inequality (&gt; / &lt;) and the other is inclusive (&ge; / &le;) -- these two only pay out differently if the reported value lands exactly on the boundary itself.</div>
  <table id="t-{s['key']}-risk">
    {thead}
    <tbody>
      {''.join(risk_rows)}
    </tbody>
  </table>"""
        section_blocks.append(f"""
  <h2 class="category-heading" id="{s['key']}-heading" onclick="toggleSection('{s['key']}')">
    <span class="toggle-triangle"></span>{s['label']}
  </h2>
  <div class="sub">{len(s['pairs'])} live pairs (cosine &gt; {s['threshold']}){f', {len(risk_pairs)} higher-risk' if risk_pairs else ''}.{filter_note}{section_new_note}</div>
  <div id="{s['key']}-section">
  <table id="t-{s['key']}">
    {thead}
    <tbody>
      {''.join(rows) if rows else f'<tr><td colspan="8">No live {s["label"]} pairs found at this threshold.</td></tr>'}
    </tbody>
  </table>
  {risk_block}
  </div>""")

    toggle_keys_js = ", ".join(f"'{s['key']}'" for s in sections)
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
{refresh_tag}
<title>Prediction market price comparison</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 24px; background: #0b0e14; color: #e6e6e6; }}
  .page-header {{ display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid #2a2f3a; padding-bottom: 12px; margin-bottom: 16px; }}
  .page-header .brand {{ font-size: 20px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }}
  .page-header .author {{ font-size: 14px; color: #999; }}
  .status-box {{ display: flex; align-items: stretch; gap: 32px; flex-wrap: wrap; background: #12151c; border: 1px solid #2a2f3a; border-radius: 6px; padding: 10px 16px; margin: 16px 0 20px; }}
  .status-item {{ display: flex; flex-direction: column; justify-content: center; }}
  .status-label {{ font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #999; margin-bottom: 3px; }}
  .status-value {{ font-size: 14px; font-weight: 600; color: #e6e6e6; font-variant-numeric: tabular-nums; }}
  .status-value.profit-count {{ color: #51cf66; font-size: 16px; }}
  .status-divider {{ width: 1px; align-self: stretch; background: #2a2f3a; }}
  h1 {{ font-size: 18px; font-weight: 600; }}
  .category-heading {{ display: flex; align-items: center; font-size: 15px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #7aa2ff; border-bottom: 2px solid #2a2f3a; padding-bottom: 6px; margin: 20px 0 12px; cursor: pointer; user-select: none; }}
  .toggle-triangle {{ display: inline-block; width: 0; height: 0; margin-right: 8px; border-top: 5px solid transparent; border-bottom: 5px solid transparent; border-left: 7px solid #7aa2ff; transition: transform 0.15s ease; transform: rotate(90deg); }}
  .category-heading.collapsed .toggle-triangle {{ transform: rotate(0deg); }}
  .sub {{ color: #999; font-size: 13px; margin-bottom: 16px; }}
  .risk-heading {{ font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #ffa94d; margin: 22px 0 4px; }}
  .risk-note {{ color: #999; font-size: 12px; margin-bottom: 10px; max-width: 900px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; box-sizing: border-box; table-layout: fixed; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #2a2f3a; text-align: left; vertical-align: top; overflow-wrap: break-word; }}
  th {{ position: sticky; top: 0; background: #12151c; cursor: pointer; }}
  /* Fixed per-column widths (same across every table) so the Yes-Diff divider
     lines up at the same x-position in every category/risk table, regardless
     of how long that table's particular question text happens to be. */
  th:nth-child(1), td:nth-child(1) {{ width: 6%; }}
  th:nth-child(2), td:nth-child(2) {{ width: 25%; }}
  th:nth-child(3), td:nth-child(3) {{ width: 8%; }}
  th:nth-child(4), td:nth-child(4) {{ width: 8%; }}
  th:nth-child(5), td:nth-child(5) {{ width: 25%; }}
  th:nth-child(6), td:nth-child(6) {{ width: 8%; }}
  th:nth-child(7), td:nth-child(7) {{ width: 8%; }}
  th:nth-child(8), td:nth-child(8) {{ width: 12%; }}
  a {{ color: #7aa2ff; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .price {{ font-variant-numeric: tabular-nums; text-align: right; }}
  .ask {{ color: #c8a2c8; }}
  .divider {{ border-left: 1px solid #ffffff; }}
  .score {{ font-variant-numeric: tabular-nums; color: #9ad; }}
  .diff-small {{ color: #ff6b6b; font-weight: 600; }}
  .diff-medium {{ color: #ffa94d; font-weight: 600; }}
  .diff-large {{ color: #51cf66; font-weight: 700; }}
  .arb-label {{ font-size: 10px; font-weight: 400; color: #999; text-align: right; }}
  .arb-loss {{ font-size: 10px; font-weight: 700; color: #c8a2c8; text-align: right; text-transform: uppercase; letter-spacing: 0.04em; }}
  .countdown {{ font-size: 9px; font-weight: 400; color: #667; text-align: right; font-variant-numeric: tabular-nums; }}
  /* Flashed on briefly (via JS) right before each auto-refresh reload, so
     the price swap reads as a visible "blackout, then new price" beat
     instead of a silent in-place text change. Uses the page's own
     background color (not pure black) so the flashed cells blend into the
     surrounding page rather than punching a black hole in a dark-but-not-
     black UI. */
  body.price-refreshing td.price.ask, body.price-refreshing td.price.divider {{ background: #0b0e14; color: #0b0e14 !important; }}
  body.price-refreshing td.price.ask .countdown, body.price-refreshing td.price.divider .arb-label, body.price-refreshing td.price.divider .arb-loss {{ color: #0b0e14 !important; }}
  tr:hover {{ background: #161a24; }}
  .new-badge {{ display: inline-block; background: #2f9e44; color: #fff; font-size: 10px; font-weight: 700; letter-spacing: 0.04em; padding: 1px 5px; border-radius: 3px; vertical-align: middle; margin-right: 4px; }}
  tr.row-new {{ background: #132417; }}
  tr.row-new:hover {{ background: #1a3220; }}
</style>
</head>
<body>
  <div class="page-header">
    <div class="brand">PREDICTION MARKET ARBITRAGE TRACKER</div>
    <div class="author">Neerav Soni</div>
  </div>
  <div class="status-box">
    <div class="status-item">
      <span class="status-label">Last incremental market scan</span>
      <span class="status-value">{_fmt_ts(last_scan_at)}</span>
    </div>
    <div class="status-item">
      <span class="status-label">Last price refresh</span>
      <span class="status-value">{_fmt_ts(last_refresh_at)}</span>
    </div>
    <div class="status-divider"></div>
    <div class="status-item">
      <span class="status-label">Total profit opportunities</span>
      <span class="status-value profit-count">{total_profit_opportunities}</span>
    </div>
    <div class="status-divider"></div>
    <div class="status-item">
      <span class="status-label">Aggregate profit %</span>
      <span class="status-value profit-count">{f'{profit_pct:.2f}%' if profit_pct is not None else '-'}</span>
    </div>
  </div>
  <h1>Matched market pairs — cross-platform hedge arbitrage</h1>
  <div class="sub">
    {total_pairs} live pairs across {len(sections)} categories.
    <span class="ask">Lilac = best ask</span> for each side: PM Yes/No ask from Polymarket's bestBid/bestAsk (No ask = 1&minus;Yes bid);
    Kalshi Yes/No ask derived from its live orderbook the same way.
    Best Arb = the larger of the two guaranteed-hedge payoffs on a matched pair (buying the opposite outcome on each
    platform always pays exactly $1 if the two markets resolve identically, before fees/gas/slippage):
    1.00 &minus; (PM Yes ask + Kalshi No ask), or 1.00 &minus; (PM No ask + Kalshi Yes ask) &mdash; whichever is larger, with its direction labeled underneath.
    Colored by size when positive:
    <span style="color:#ff6b6b">red = smallest (&lt; 0.01)</span>,
    <span style="color:#ffa94d">amber = medium (0.01&ndash;0.03)</span>,
    <span style="color:#51cf66">green = largest (&ge; 0.03)</span>;
    shown as the (negative) value with a <span class="arb-loss">LOSS</span> marker when the best available combo's two best asks add up to $1 or more (i.e. no guaranteed hedge);
    shown as &ndash; only when a market is missing best-ask data on both possible combos entirely.
    Prices refreshed {time.strftime('%Y-%m-%d %H:%M:%S %Z')}{' (auto-refreshing every ' + str(refresh_seconds) + 's -- countdown shown under each ask price)' if refresh_seconds else ''}.{match_note}{new_note}
  </div>
  {''.join(section_blocks)}
  <script>
    function toggleSection(key) {{
      var heading = document.getElementById(key + '-heading');
      var section = document.getElementById(key + '-section');
      var collapsed = heading.classList.toggle('collapsed');
      section.style.display = collapsed ? 'none' : '';
    }}

    // Live per-price countdown to the next auto-refresh (the page itself
    // reloads via <meta http-equiv="refresh">, which restarts this timer
    // from the top on every reload).
    (function() {{
      var refreshSeconds = {refresh_seconds or 0};
      if (refreshSeconds > 0) {{
        var remaining = refreshSeconds;
        var els = document.querySelectorAll('.countdown');
        function tick() {{
          var label = remaining + 's';
          for (var i = 0; i < els.length; i++) els[i].textContent = label;
          if (remaining > 0) remaining--;
        }}
        tick();
        setInterval(tick, 1000);

        // Blacks out every live price cell for the last stretch before the
        // page's own <meta refresh> swaps in the freshly-written HTML, so
        // each refresh reads as a visible "blackout, then new price" beat
        // rather than a silent reload. BLACKOUT_MS is comfortably under
        // half a second and timed off the same page-load clock the meta
        // refresh itself uses, so the reveal lines up with the reload.
        var BLACKOUT_MS = 350;
        var blackoutDelay = Math.max(0, refreshSeconds * 1000 - BLACKOUT_MS);
        setTimeout(function() {{
          document.body.classList.add('price-refreshing');
        }}, blackoutDelay);
      }}
    }})();
  </script>
</body>
</html>"""
    with open(out_path, "w") as f:
        f.write(html)


def compute_matches(threshold, model, pm_tag_ids, kalshi_categories, extra_filter=None, label=""):
    """Full from-scratch build: fetch the ENTIRE live market universe on both
    platforms, embed every title, and compare every PM market against every
    Kalshi market. Expensive (O(n*m) comparisons + embedding ~thousands of
    titles) -- only meant to run once, on cold start or --rebuild. Ongoing
    discovery of newly-listed markets should go through
    scan_for_new_matches() instead, which only embeds/compares what's new."""
    print(f"Fetching live Polymarket {label} markets...")
    pm_markets = fetch_polymarket_markets(pm_tag_ids)
    print(f"  -> {len(pm_markets)} Polymarket markets")

    print(f"Fetching live Kalshi {label} markets...")
    kalshi_markets = fetch_kalshi_markets(kalshi_categories)
    print(f"  -> {len(kalshi_markets)} Kalshi markets")

    if not pm_markets or not kalshi_markets:
        return [], pm_markets, np.zeros((0, 384), dtype=np.float32), kalshi_markets, np.zeros((0, 384), dtype=np.float32)

    print("Embedding titles...")
    pm_vecs = embed(model, [m.title for m in pm_markets])
    kalshi_vecs = embed(model, [m.title for m in kalshi_markets])

    print("Computing cosine similarities...")
    sims = pm_vecs @ kalshi_vecs.T

    matched = []
    for i, pm in enumerate(pm_markets):
        for j, km in enumerate(kalshi_markets):
            score = float(sims[i, j])
            if score <= threshold:
                continue
            risk = None
            if extra_filter is not None:
                ok, risk = extra_filter(pm.title, km.title, pm.description, km.rules)
                if not ok:
                    continue
            matched.append((score, pm, km, risk))
    matched.sort(key=lambda t: t[0], reverse=True)
    n_risk = sum(1 for *_, risk in matched if risk)
    print(f"  -> {len(matched)} matched pairs (cosine > {threshold}{', + boundary/date/direction filter' if extra_filter else ''}{f', {n_risk} flagged higher-risk' if n_risk else ''})")
    return matched, pm_markets, pm_vecs, kalshi_markets, kalshi_vecs


def scan_for_new_matches(model, universe_path, threshold, pm_tag_ids, kalshi_categories, extra_filter=None):
    """Incremental scanner: fetch the current live market listing on both
    platforms (cheap -- no embedding), diff it against the persisted
    universe cache to find markets we haven't seen before, embed ONLY those
    new titles, and compare:
        - every NEW PM market against ALL Kalshi markets (old + new)
        - every OLD PM market against ONLY the NEW Kalshi markets
    which together cover every new-vs-anything pair exactly once, while
    never recomputing an old-vs-old comparison that a previous scan (or the
    initial compute_matches build) already did.

    Also prunes markets that have closed/delisted since the last scan out of
    the persisted universe, so it doesn't grow unbounded with dead markets.

    Returns (new_matches, n_new_pm, n_new_kalshi).
    """
    pm_markets = fetch_polymarket_markets(pm_tag_ids)
    kalshi_markets = fetch_kalshi_markets(kalshi_categories)

    dim = model.get_sentence_embedding_dimension()
    prev = load_universe(universe_path)
    if prev is None:
        prev_pm_markets, prev_pm_vecs = [], np.zeros((0, dim), dtype=np.float32)
        prev_kalshi_markets, prev_kalshi_vecs = [], np.zeros((0, dim), dtype=np.float32)
    else:
        prev_pm_markets, prev_pm_vecs = prev["pm_markets"], prev["pm_vecs"]
        prev_kalshi_markets, prev_kalshi_vecs = prev["kalshi_markets"], prev["kalshi_vecs"]

    # Drop anything no longer in the live listing (closed/delisted) so the
    # universe cache stays current and doesn't grow forever.
    live_pm_ids = {m.id for m in pm_markets}
    live_kalshi_ids = {m.ticker for m in kalshi_markets}
    keep_pm = [i for i, m in enumerate(prev_pm_markets) if m.id in live_pm_ids]
    keep_kalshi = [i for i, m in enumerate(prev_kalshi_markets) if m.ticker in live_kalshi_ids]
    prev_pm_markets = [prev_pm_markets[i] for i in keep_pm]
    prev_pm_vecs = prev_pm_vecs[keep_pm] if len(prev_pm_vecs) else prev_pm_vecs
    prev_kalshi_markets = [prev_kalshi_markets[i] for i in keep_kalshi]
    prev_kalshi_vecs = prev_kalshi_vecs[keep_kalshi] if len(prev_kalshi_vecs) else prev_kalshi_vecs

    known_pm_ids = {m.id for m in prev_pm_markets}
    known_kalshi_ids = {m.ticker for m in prev_kalshi_markets}
    new_pm = [m for m in pm_markets if m.id not in known_pm_ids]
    new_kalshi = [m for m in kalshi_markets if m.ticker not in known_kalshi_ids]

    if not new_pm and not new_kalshi:
        save_universe(prev_pm_markets, prev_pm_vecs, prev_kalshi_markets, prev_kalshi_vecs, universe_path)
        return [], 0, 0

    new_pm_vecs = embed(model, [m.title for m in new_pm]) if new_pm else np.zeros((0, dim), dtype=np.float32)
    new_kalshi_vecs = embed(model, [m.title for m in new_kalshi]) if new_kalshi else np.zeros((0, dim), dtype=np.float32)

    all_kalshi_markets = prev_kalshi_markets + new_kalshi
    all_kalshi_vecs = np.vstack([prev_kalshi_vecs, new_kalshi_vecs]) if len(prev_kalshi_vecs) or len(new_kalshi_vecs) else new_kalshi_vecs

    new_matches = []
    if new_pm and all_kalshi_markets:
        sims = new_pm_vecs @ all_kalshi_vecs.T  # new PM vs ALL kalshi (covers new-vs-new too)
        for i, pm in enumerate(new_pm):
            for j, km in enumerate(all_kalshi_markets):
                score = float(sims[i, j])
                if score <= threshold:
                    continue
                risk = None
                if extra_filter is not None:
                    ok, risk = extra_filter(pm.title, km.title, pm.description, km.rules)
                    if not ok:
                        continue
                new_matches.append((score, pm, km, risk))
    if new_kalshi and prev_pm_markets:
        sims2 = prev_pm_vecs @ new_kalshi_vecs.T  # old PM vs NEW kalshi only (new-vs-new already covered above)
        for i, pm in enumerate(prev_pm_markets):
            for j, km in enumerate(new_kalshi):
                score = float(sims2[i, j])
                if score <= threshold:
                    continue
                risk = None
                if extra_filter is not None:
                    ok, risk = extra_filter(pm.title, km.title, pm.description, km.rules)
                    if not ok:
                        continue
                new_matches.append((score, pm, km, risk))
    new_matches.sort(key=lambda t: t[0], reverse=True)

    all_pm_markets = prev_pm_markets + new_pm
    all_pm_vecs = np.vstack([prev_pm_vecs, new_pm_vecs]) if len(prev_pm_vecs) or len(new_pm_vecs) else new_pm_vecs
    save_universe(all_pm_markets, all_pm_vecs, all_kalshi_markets, all_kalshi_vecs, universe_path)

    return new_matches, len(new_pm), len(new_kalshi)


def build_section_pairs(cache):
    """Refresh live prices for one category's cached match set and return
    the (score, pm, km, best_arb, best_arb_label, is_new, risk) rows
    render_html expects, plus how many of them are freshly-scanner-found
    ("NEW").

    best_arb is the larger of the two guaranteed-hedge payoffs for a
    matched pair (buying the opposite side on each platform always pays
    out exactly $1 if the two markets truly resolve identically):
        1.00 - (Polymarket Yes ask + Kalshi No ask)   -- buy PM Yes + K No
        1.00 - (Polymarket No ask  + Kalshi Yes ask)   -- buy PM No + K Yes
    A positive value is the guaranteed profit per $1 of eventual payout,
    before fees/gas/slippage."""
    pairs_meta = cache["pairs"]
    pm_ids = [p["pm_id"] for p in pairs_meta]
    pm_prices = fetch_polymarket_prices_batch(pm_ids) if pm_ids else {}

    kalshi_markets = [KalshiMarket(ticker=p["kalshi_ticker"], title=p["kalshi_title"], url=p["kalshi_url"]) for p in pairs_meta]
    if kalshi_markets:
        fill_kalshi_orderbook_prices(kalshi_markets)

    now = time.time()
    pairs = []
    new_count = 0
    for p, km in zip(pairs_meta, kalshi_markets):
        pm_yes_price, pm_yes_ask, pm_yes_bid = pm_prices.get(p["pm_id"], (None, None, None))
        pm = PMMarket(id=p["pm_id"], title=p["pm_title"], url=p["pm_url"], yes_price=pm_yes_price, yes_ask=pm_yes_ask, yes_bid=pm_yes_bid)

        candidates = []
        if pm.yes_ask is not None and km.no_ask is not None:
            candidates.append((1.00 - (pm.yes_ask + km.no_ask), "Buy PM Yes + Kalshi No"))
        if pm.no_ask is not None and km.yes_ask is not None:
            candidates.append((1.00 - (pm.no_ask + km.yes_ask), "Buy PM No + Kalshi Yes"))
        best_arb, best_arb_label = max(candidates, key=lambda c: c[0]) if candidates else (None, None)

        is_new = (now - p.get("first_seen", 0)) < NEW_BADGE_WINDOW_SECONDS
        if is_new:
            new_count += 1
        pairs.append((p["score"], pm, km, best_arb, best_arb_label, is_new, p.get("risk")))
    return pairs, new_count


def refresh_and_render(caches, out_path, interval, last_scan_at=None):
    """caches: dict of category_key -> match cache. Refreshes live prices
    for every category and writes a single HTML file with one dropdown
    section per category.

    last_scan_at: epoch seconds of the most recent incremental new-market
    scan (across all categories), threaded through from main()'s loop so the
    top-of-page status box can show it alongside this refresh's own
    timestamp -- these run on independent schedules, so this function can't
    infer the scan time on its own."""
    last_refresh_at = time.time()
    sections = []
    for key, config in CATEGORY_CONFIGS.items():
        cache = caches[key]
        pairs, new_count = build_section_pairs(cache)
        sections.append({
            "key": key,
            "label": config["label"],
            "pairs": pairs,
            "threshold": cache["threshold"],
            "match_built_at": cache["built_at"],
            "new_count": new_count,
            "extra_filter": config["extra_filter"] is not None,
        })
    # The meta-refresh/countdown/blackout display cadence is intentionally
    # decoupled from `interval` (see PAGE_REFRESH_DISPLAY_SECONDS) -- only
    # whether we're looping at all (interval truthy) matters here, not its
    # exact value.
    display_refresh_seconds = PAGE_REFRESH_DISPLAY_SECONDS if interval else None
    render_html(sections, out_path, refresh_seconds=display_refresh_seconds, last_scan_at=last_scan_at, last_refresh_at=last_refresh_at)


def _category_path(base_path, key):
    """Derive a per-category cache/universe filename from a base path, e.g.
    /tmp/price_interface_cache.json -> /tmp/price_interface_cache_economics.json"""
    if "." in base_path.rsplit("/", 1)[-1]:
        root, ext = base_path.rsplit(".", 1)
        return f"{root}_{key}.{ext}"
    return f"{base_path}_{key}"


def load_or_build_category(key, config, args, model):
    """Load a category's cached match set (rebuilding from scratch if
    missing/stale/--rebuild), then run the incremental new-market scanner
    unless suppressed. Returns the up-to-date match cache."""
    cache_path = _category_path(args.cache, key)
    universe_path = _category_path(args.universe_cache, key)

    cache = None
    if not args.rebuild:
        try:
            cache = load_match_cache(cache_path)
            if abs(cache.get("threshold", -1) - config["threshold"]) > 1e-9:
                print(f"[{config['label']}] cached threshold ({cache.get('threshold')}) != configured ({config['threshold']}); rebuilding.")
                cache = None
        except (FileNotFoundError, json.JSONDecodeError):
            cache = None

    pm_tag_ids = list(config["pm_tag_ids"].values())
    kalshi_categories = config["kalshi_categories"]
    extra_filter = config["extra_filter"]

    if cache is None:
        matched, pm_markets, pm_vecs, kalshi_markets, kalshi_vecs = compute_matches(
            config["threshold"], model, pm_tag_ids, kalshi_categories, extra_filter, label=config["label"]
        )
        cache = save_match_cache(matched, config["threshold"], cache_path)
        save_universe(pm_markets, pm_vecs, kalshi_markets, kalshi_vecs, universe_path)
        print(f"[{config['label']}] cached {len(cache['pairs'])} matched pairs to {cache_path}; universe snapshot ({len(pm_markets)} PM, {len(kalshi_markets)} Kalshi) saved to {universe_path}")
    elif not args.no_scan:
        print(f"[{config['label']}] scanning for newly-listed markets since the last snapshot...")
        new_matches, n_new_pm, n_new_kalshi = scan_for_new_matches(model, universe_path, config["threshold"], pm_tag_ids, kalshi_categories, extra_filter)
        if new_matches:
            cache, added = merge_new_matches(cache, new_matches, cache_path)
            print(f"  -> {n_new_pm} new PM / {n_new_kalshi} new Kalshi markets found; {added} new matched pair(s) merged in.")
        else:
            print(f"  -> {n_new_pm} new PM / {n_new_kalshi} new Kalshi markets found; no new matched pairs.")

    return cache, universe_path


def main():
    parser = argparse.ArgumentParser(description="Live price-comparison interface (Politics + Economics dropdowns) with an incremental new-market scanner.")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--out", default="/tmp/price_interface.html")
    parser.add_argument("--cache", default="/tmp/price_interface_cache.json", help="Base path for the per-category matched-pair cache (a suffix is added per category, e.g. _politics/_economics)")
    parser.add_argument("--universe-cache", default="/tmp/price_interface_universe.npz", help="Base path for the per-category seen-market/embedding snapshot used by the incremental scanner")
    parser.add_argument("--interval", type=int, default=60, help="Loop forever, refreshing prices every N seconds (default 60s; pass 0 for a single one-shot render)")
    parser.add_argument("--scan-interval", type=int, default=900, help=f"How often (seconds) to scan for newly-listed markets while looping. Independent of --interval; hard-capped at {MAX_SCAN_INTERVAL_SECONDS}s (1hr) so new markets are never missed for longer than that (default 900s = 15min)")
    parser.add_argument("--rebuild", action="store_true", help="Force a full from-scratch rebuild (re-embeds and re-compares EVERY market) instead of reusing --cache/--universe-cache")
    parser.add_argument("--no-scan", action="store_true", help="Skip the incremental new-market scan on startup (still runs during --interval loop unless combined with --rebuild logic elsewhere)")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open the result in a browser")
    args = parser.parse_args()

    print("Loading embedding model...")
    model = SentenceTransformer(args.model)

    caches = {}
    universe_paths = {}
    for key, config in CATEGORY_CONFIGS.items():
        caches[key], universe_paths[key] = load_or_build_category(key, config, args, model)

    # The startup pass above already ran the incremental scan for each
    # category (unless --no-scan), so timestamp it as "now" for the status
    # box; --no-scan means no scan actually happened yet, so leave it unset
    # until the loop's first scan sets it for real.
    startup_scan_at = None if args.no_scan else time.time()

    refresh_and_render(caches, args.out, args.interval or None, last_scan_at=startup_scan_at)
    print(f"Wrote {args.out}")
    if not args.no_open:
        webbrowser.open(f"file://{args.out}")

    if args.interval and args.interval > 0:
        scan_interval = args.scan_interval
        if scan_interval > MAX_SCAN_INTERVAL_SECONDS:
            print(f"--scan-interval {scan_interval}s exceeds the {MAX_SCAN_INTERVAL_SECONDS}s (1hr) cap; clamping to {MAX_SCAN_INTERVAL_SECONDS}s so new markets are never missed for more than an hour.")
            scan_interval = MAX_SCAN_INTERVAL_SECONDS

        # Scan and price-refresh run on independent schedules -- a long
        # --interval (price refresh) never causes scans to be skipped, and
        # vice versa. The loop just wakes up on whichever cadence is
        # shorter and checks what's due. Both categories share the same
        # scan/refresh schedule.
        tick = max(1, min(args.interval, scan_interval))
        print(f"Looping: price refresh every {args.interval}s, market scan every {scan_interval}s for all categories (Ctrl+C to stop)...")
        last_scan = startup_scan_at or time.time()   # startup scan (if any) already ran above
        last_refresh = time.time()  # initial render already happened above
        try:
            while True:
                time.sleep(tick)
                now = time.time()
                found_new = False
                if now - last_scan >= scan_interval:
                    for key, config in CATEGORY_CONFIGS.items():
                        pm_tag_ids = list(config["pm_tag_ids"].values())
                        new_matches, n_new_pm, n_new_kalshi = scan_for_new_matches(
                            model, universe_paths[key], config["threshold"], pm_tag_ids, config["kalshi_categories"], config["extra_filter"]
                        )
                        if new_matches:
                            caches[key], added = merge_new_matches(caches[key], new_matches, _category_path(args.cache, key))
                            found_new = True
                            print(f"  ...[{time.strftime('%H:%M:%S')}][{config['label']}] scan found {n_new_pm} new PM / {n_new_kalshi} new Kalshi markets -> {added} new pair(s)")
                        else:
                            print(f"  ...[{time.strftime('%H:%M:%S')}][{config['label']}] scan found {n_new_pm} new PM / {n_new_kalshi} new Kalshi markets, no new pairs")
                    last_scan = now
                if found_new or now - last_refresh >= args.interval:
                    refresh_and_render(caches, args.out, args.interval, last_scan_at=last_scan)
                    last_refresh = now
                    print(f"  ...refreshed {time.strftime('%H:%M:%S')}")
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    main()
