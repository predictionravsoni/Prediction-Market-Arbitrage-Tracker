"""
Five-signal filter for flagging genuine SAME-market pairs between
Polymarket and Kalshi ECONOMICS markets:
    1. cosine similarity >= 0.85 (embedding-based topic match, computed
       upstream and baked into the pair dumps this module reads)
    2. boundary_match       -- shared numeric threshold (regex)
    3. date_align            -- shared time period (close-timestamp
                                 proximity for backtest pairs, title-parsed
                                 year/month/quarter for live pairs)
    4. growth_basis_align    -- shared growth-rate/GDP basis (QoQ/YoY/MoM,
                                 real/nominal), now also description/rules-
                                 text-aware, not title-only
    5. classify_relationship -- comparison-operator/direction classification
                                 (EXACT / STRICT_VS_INCLUSIVE pass; every
                                 other relationship is dropped)
Reads pair dumps produced upstream:
    /tmp/backtest_economics_pairs.json      (closed/settled, "last year")
    /tmp/compare_economics_live_085.json    (open, "this year")

Background: pure cosine similarity tops out around 30-55% precision for
economics markets (see backtest_economics_markets.py's review), because
sentence embeddings are largely blind to the ONE number that actually
distinguishes bracket-family markets (e.g. "Fed funds rate >= 4.00%" vs
"...>= 4.25%" read as nearly-identical text). A regex-based numeric-boundary
match recovers that signal -- but on its own it has a blind spot: it can
match the right NUMBER attached to the wrong TIME PERIOD (e.g. PM's "end of
2026" matched to Kalshi's "December 31, 2036" -- a decade off -- because the
threshold happens to line up). Signals 3-5 were added incrementally, each
one closing a specific blind spot found via manual backtest audit:
  - signal 3 (date_align) closes the wrong-time-period gap above.
      - BACKTEST (closed/settled) pairs carry real close/settle timestamps
        on both sides -- compared directly (stronger than parsing text).
      - LIVE (open) pairs have no close timestamp yet, so explicit 4-digit
        years/months/quarters are extracted from the title instead. Absence
        of a signal on either side isn't penalized -- it's not evidence of
        a mismatch.
  - signal 4 (growth_basis_align) closes the same-number-different-
    statistic gap (QoQ vs YoY, real vs nominal GDP).
  - signal 5 (classify_relationship) closes the bracket-edge-vs-ladder-rung
    gap (a PM range's edge and a Kalshi threshold can share a number while
    being genuinely different bets).

Usage:
    python economics_quintuple_filter.py
    python economics_quintuple_filter.py --pct-tol 0.1 --dollar-tol-frac 0.02
    python economics_quintuple_filter.py --samples 20
"""

import argparse
import datetime
import json
import random
import re
from collections import Counter

BACKTEST_PATH = "/tmp/backtest_economics_pairs.json"
LIVE_PATH = "/tmp/compare_economics_live_085.json"

PCT_RE = re.compile(r'(-?\d+(?:\.\d+)?)\s*%')
DOLLAR_RE = re.compile(
    r'\$\s*(-?\d[\d,]*(?:\.\d+)?)\s*(billion|bn|b\b|million|m\b|k\b|thousand)?', re.I
)
YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')
MONTH_RE = re.compile(
    r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|'
    r'aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b',
    re.I,
)
QUARTER_RE = re.compile(r'\bQ([1-4])\b', re.I)
_MONTH_KEY = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Growth-rate BASIS (as opposed to time period) -- QoQ, YoY, MoM, and
# annualized/SAAR figures for the same metric are NOT interchangeable (e.g.
# YoY compounds four quarters of growth into one number, so "GDP QoQ >=
# 1.5%" and "GDP YoY >= 1.5%" describe different bets even when the number,
# quarter, and comparison direction all line up). See date_align_live().
GROWTH_BASIS_RE = re.compile(
    r'\b(QoQ|YoY|MoM|quarter-over-quarter|quarter over quarter|'
    r'year-over-year|year over year|month-over-month|month over month|'
    r'seasonally adjusted annual rate|SAAR|annualized|annual rate)\b',
    re.I,
)
_GROWTH_BASIS_KEY = {
    "qoq": "QoQ", "quarter over quarter": "QoQ",
    "yoy": "YoY", "year over year": "YoY",
    "mom": "MoM", "month over month": "MoM",
    "saar": "ANNUALIZED", "seasonally adjusted annual rate": "ANNUALIZED",
    "annualized": "ANNUALIZED", "annual rate": "ANNUALIZED",
}

DOLLAR_UNIT_SCALE = {
    None: 1e-9,       # bare dollar figure (e.g. "$50000") -> billions
    "": 1e-9,
    "billion": 1.0,
    "bn": 1.0,
    "b": 1.0,
    "million": 1e-3,
    "m": 1e-3,
    "thousand": 1e-6,
    "k": 1e-6,
}


def extract_pct(title):
    return [float(v) for v in PCT_RE.findall(title)]


def extract_dollars_billions(title):
    out = []
    for value, unit in DOLLAR_RE.findall(title):
        v = float(value.replace(",", ""))
        scale = DOLLAR_UNIT_SCALE.get((unit or "").lower(), 1e-9)
        out.append(v * scale)
    return out


def extract_years(title):
    return {int(m.group(0)) for m in YEAR_RE.finditer(title)}


def extract_months(title):
    out = set()
    for m in MONTH_RE.finditer(title):
        key = m.group(1).lower()[:3]  # "sept" -> "sep" too
        out.add(_MONTH_KEY.get(key))
    out.discard(None)
    return out


def extract_quarters(title):
    return {int(m.group(1)) for m in QUARTER_RE.finditer(title)}


# "At the end of 2026" / "by year-end" style phrasing asks about the FINAL
# value for that year -- as opposed to a value pinned to one specific
# month/quarter/meeting within it. This matters when one such "terminal"
# title gets matched against several Kalshi siblings that are each anchored
# to a specific date within the year (e.g. one ticker per FOMC meeting):
# only the chronologically LAST such sibling can actually be the same bet
# as "the end of the year" -- anything can still change between an earlier
# snapshot and year-end, so the earlier siblings are a different question,
# not an equivalent phrasing of the same one. See resolve_terminal_siblings
# in price_interface.py, which uses this to dedupe exactly that pattern.
TERMINAL_RE = re.compile(
    r'\b(at the end of|by the end of|by year-?end|year-?end|end of (?:the )?year)\b',
    re.I,
)


def is_year_end_terminal(title):
    """True if `title` asks about a year-end/terminal value with no more
    specific month or quarter mentioned to pin it down further."""
    if not TERMINAL_RE.search(title):
        return False
    return not extract_months(title) and not extract_quarters(title)


def extract_growth_basis(title, extra=""):
    """Return the set of explicit growth-rate bases mentioned in a title
    (QoQ / YoY / MoM / ANNUALIZED). Absence of a basis mention on either
    side isn't evidence of a mismatch -- most titles don't bother stating
    the "obvious" convention -- so this is only used to REJECT pairs where
    both sides explicitly state DIFFERENT bases.

    `extra` is optional resolution-criteria text (Polymarket `description`
    or Kalshi `rules_primary`/`rules_secondary`) appended to the title
    before scanning. Titles are often terse ("Fed rate decision Dec 2026")
    while the basis convention is spelled out in the rules prose instead
    ("...resolves based on the QoQ annualized rate reported by BEA..."), so
    checking title-only text misses signal the platform itself states
    explicitly. Title stays primary; this only ADDS categorical-qualifier
    signal, never numeric-threshold signal (see extract_gdp_basis)."""
    text = f"{title} {extra}" if extra else title
    out = set()
    for m in GROWTH_BASIS_RE.finditer(text):
        key = re.sub(r'\s+', ' ', m.group(1).lower().replace('-', ' ')).strip()
        canon = _GROWTH_BASIS_KEY.get(key)
        if canon:
            out.add(canon)
    return out


def extract_gdp_basis(title, extra=""):
    """Real vs nominal GDP growth are different statistics (nominal bakes
    in inflation, real strips it out) -- not interchangeable even when the
    quarter/threshold/direction all line up. Unlike QoQ/YoY/MoM above,
    "absence isn't evidence of a mismatch" does NOT apply here: most "GDP
    growth" titles don't bother saying "real" because that IS the default/
    headline convention -- confirmed directly against Polymarket's own
    market rules, which state an unqualified "GDP growth" question resolves
    against the BEA's REAL-GDP Advance Estimate. So an unqualified GDP
    mention is treated as an *implicit* REAL basis rather than skipped,
    which is what actually catches a PM "GDP growth > X%" title silently
    matching a Kalshi *nominal*-GDP market (same number/quarter/direction,
    different statistic) -- found via manual backtest audit, see
    /tmp/backtest_survivors.json pair #19 vs #21. Only fires on titles that
    mention GDP at all; every other economics title is unaffected.

    `extra` is optional resolution-criteria text (PM `description` / Kalshi
    `rules_primary`+`rules_secondary`), appended before scanning for the
    same reason as extract_growth_basis above: the "real" vs "nominal"
    qualifier is sometimes stated only in the rules prose, not the title.
    This still only feeds the categorical real/nominal check -- it is
    deliberately NOT used to parse numeric bracket boundaries, since
    resolution text often lists an entire bracket ladder rather than just
    the one bracket this specific market covers, which would risk picking
    up the wrong number."""
    text = f"{title} {extra}" if extra else title
    if not re.search(r'\bgdp\b', text, re.I):
        return set()
    if re.search(r'\bnominal\b', text, re.I):
        return {"NOMINAL"}
    return {"REAL"}


def boundary_match(title_a, title_b, pct_tol=0.10, dollar_tol_frac=0.02):
    """True if the titles share a numeric threshold within tolerance.

    pct_tol: absolute percentage-point tolerance (e.g. 0.10 = +/-0.10%).
    dollar_tol_frac: relative tolerance as a fraction of the value being
    compared (dollar brackets span orders of magnitude, so relative makes
    more sense than absolute here).
    Fails closed: if neither side has a comparable numeric signal, no match.
    """
    a_pct, b_pct = extract_pct(title_a), extract_pct(title_b)
    if a_pct and b_pct:
        if any(abs(x - y) <= pct_tol for x in a_pct for y in b_pct):
            return True

    a_usd, b_usd = extract_dollars_billions(title_a), extract_dollars_billions(title_b)
    if a_usd and b_usd:
        if any(
            abs(x - y) <= dollar_tol_frac * max(abs(x), abs(y), 1e-9)
            for x in a_usd for y in b_usd
        ):
            return True

    return False


def date_align_live(title_a, title_b, extra_a="", extra_b=""):
    """For open markets (no close timestamp yet): require mentioned years,
    months, quarters, and growth-rate bases to each overlap wherever both
    sides mention that dimension. Absence of a dimension on either side
    isn't evidence of a mismatch, so it's skipped rather than penalized.

    This catches three distinct failure modes seen in practice:
      - year mismatch (PM "end of 2026" vs Kalshi "...2036")
      - same-year, different-MONTH mismatch (PM "August 2026 unemployment"
        vs Kalshi "Unemployment rate in Sep 2026") -- year-only checking
        misses this since both sides say "2026".
      - same number/quarter, different growth-rate BASIS (PM "Brazil GDP
        growth (QoQ) >= 1.5%" vs Kalshi "Brazilian GDP (YoY) >= 1.5%") --
        QoQ and YoY are different statistics, not interchangeable phrasings
        of the same one, even though the threshold and quarter line up.

    `extra_a`/`extra_b` are optional resolution-criteria text (PM
    `description` / Kalshi `rules_primary`+`rules_secondary`) for each
    side, respectively. They're ONLY fed to the growth-basis/GDP-basis
    extractors (categorical qualifiers, often only spelled out in the
    rules prose) -- year/month/quarter extraction stays title-only, since
    boilerplate rules text can mention unrelated years/dates that would
    add false signal rather than remove it.
    """
    for extractor in (extract_years, extract_months, extract_quarters):
        a, b = extractor(title_a), extractor(title_b)
        if a and b and not (a & b):
            return False
    for extractor in (extract_growth_basis, extract_gdp_basis):
        a, b = extractor(title_a, extra_a), extractor(title_b, extra_b)
        if a and b and not (a & b):
            return False
    return True


def growth_basis_align(title_a, title_b, extra_a="", extra_b=""):
    """Standalone growth-rate-basis check (QoQ/YoY/MoM/ANNUALIZED, plus
    real-vs-nominal GDP), usable independently of date_align_live -- e.g.
    for BACKTEST pairs, where period agreement is already verified via real
    close timestamps (date_align_backtest), which says nothing about
    growth-rate basis. `extra_a`/`extra_b` are optional resolution-criteria
    text, same as date_align_live."""
    a, b = extract_growth_basis(title_a, extra_a), extract_growth_basis(title_b, extra_b)
    if a and b and not (a & b):
        return False
    ga, gb = extract_gdp_basis(title_a, extra_a), extract_gdp_basis(title_b, extra_b)
    if ga and gb and not (ga & gb):
        return False
    return True


def _parse_pm_time(raw):
    return datetime.datetime.strptime(raw[:19], "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=datetime.timezone.utc
    )


def _parse_kalshi_time(raw):
    return datetime.datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc
    )


def date_align_backtest(pm_closed_at, kalshi_closed_at, tolerance_days):
    try:
        pm_dt = _parse_pm_time(pm_closed_at)
        k_dt = _parse_kalshi_time(kalshi_closed_at)
    except (ValueError, TypeError):
        return True  # can't parse -> don't penalize, fall back to regex+cosine
    return abs((pm_dt - k_dt).total_seconds()) <= tolerance_days * 86400


# ---------------------------------------------------------------------------
# Signal 4: comparison-operator / direction classification.
#
# Signals 1-3 establish that two titles are about the same topic, mention the
# same number, and refer to the same time period -- but none of them check
# *how* that number is used. Polymarket frequently posts bracket markets
# ("between 4.3% and 4.6%") while Kalshi posts a ladder of one-sided
# threshold markets ("4.3% or more", "4.6% or more", ...). A bracket's lower
# edge and a ladder rung can share the exact same number while being
# genuinely different bets (one is a superset of the other). This signal
# parses the comparison operator on each side of a title pair and classifies
# the *relationship* between the two markets, instead of just checking that
# their numbers are close.
#
# A full manual backtest audit (last year's closed/settled economics pairs
# that already pass signals 1-3) found only ~22.5% of survivors are actually
# in the EXACT category -- the rest are topically-correct but structurally
# different bets (bracket-vs-ladder edges, complements, point-vs-threshold).
# So this signal is NOT optional polish: without it, ~3 out of 4 "matched"
# pairs would silently feed a Yes-price diff that compares two different
# payoffs.
# ---------------------------------------------------------------------------

CAT_EXACT = "EXACT"                              # identical payoff -- safe to price-diff directly
CAT_SUBSET = "SUBSET"                             # K threshold = PM range's lower edge (K is a broader superset)
CAT_ADJACENT = "ADJACENT"                         # K threshold = PM range's upper edge (disjoint/opposite bracket)
CAT_PARTIAL = "PARTIAL"                           # K threshold falls inside PM range, not at an edge
CAT_COMPLEMENT = "COMPLEMENT"                     # same threshold, opposite direction (lt/lte vs gt/gte)
CAT_STRICT_VS_INCLUSIVE = "STRICT_VS_INCLUSIVE"   # same threshold + direction, but one side is > and the other is >= (or < vs <=) -- differ only in the knife-edge case where the value lands exactly on the threshold
CAT_BUCKET_VS_THRESHOLD = "BUCKET_VS_THRESHOLD"   # PM exact point vs K open-ended threshold at same number
CAT_POINT_VS_OPEN = "POINT_VS_OPEN"               # PM open-ended threshold vs K exact point (mirror of the above)
CAT_RANGE_MISMATCH = "RANGE_MISMATCH"             # both sides are ranges but edges don't line up
CAT_MISMATCH = "MISMATCH"                         # parsed fine, but the numbers don't actually line up
CAT_UNPARSED = "UNPARSED"                         # couldn't parse a comparison operator on one/both sides
CAT_UNHANDLED = "UNHANDLED"                       # parsed but this operator combination isn't covered yet

# Only these categories represent a genuinely identical payoff. Everything
# else is (at best) the same underlying event/topic but must NOT be treated
# as the same tradeable market for price-comparison purposes.
SAFE_CATEGORIES = {CAT_EXACT}

_EDGE_TOL = 0.06        # abs tolerance for "this K threshold sits at a PM range edge"
# abs tolerance for "these two ranges are the same range". Must stay tight
# (float-rounding noise only) -- NOT a fudge factor for "close enough"
# brackets. Polymarket and Kalshi use different tie-break conventions for
# adjacent bracket edges: PM's ties resolve to the HIGHER bracket (so its
# "1.5-2.0%" bucket is actually [1.5, 2.0)), while Kalshi labels its ladder
# with a non-overlapping gap (e.g. "1.6% to 2.0%" is the bucket right above
# "1.1% to 1.5%"). At 0.1-precision reported data (GDP/inflation/unemployment
# etc.) this means PM's "1.5-2.0%" and Kalshi's "1.6-2.0%" brackets differ by
# a full reporting tick at the lower edge -- NOT the same payoff (a reported
# value of exactly 1.5% pays YES on Polymarket and NO on Kalshi). The old
# value of 0.15 was loose enough to treat this one-tick offset as "the same
# range" and silently misclassified every PM/Kalshi GDP-bracket pair as
# CAT_EXACT. Caught by the user manually reading the live Polymarket/Kalshi
# rules pages side by side, not by the automated backtest audit -- this
# range-vs-range code path never appeared in the last-365-days backtest data
# (the annual "GDP growth in 2026" market hadn't settled yet), so it was a
# blind spot in every prior manual review.
_RANGE_TOL = 0.01
_VALUE_TOL = 0.06       # abs tolerance for "these two single values are the same value"


def parse_side(title):
    """Extract the comparison operator + numeric threshold(s) implied by a
    market title.

    Returns one of:
        ('range', lo, hi)   -- "between X and Y" / "X% to Y%"
        ('lte', v)          -- "at most / less than or equal to X"
        ('lt', v)           -- "less than X"
        ('gte', v)          -- "at least / X or more / X or above"
        ('gt', v)           -- "greater than / more than / above X"
        ('exact', v)        -- a bare/point value ("be X", "exactly X", or a
                                bare "X%" with no comparison word at all)
        None                -- no parseable numeric operator found
    """
    t = title
    m = re.search(r'between\s+\$?(-?\d+\.?\d*)\s*%?\s*[TtBbMmKk]?\s*and\s+\$?(-?\d+\.?\d*)', t, re.I)
    if m:
        return ('range', float(m.group(1)), float(m.group(2)))
    m = re.search(r'(\d+\.?\d*)\s*%\s*to\s*(\d+\.?\d*)\s*%', t)
    if m:
        return ('range', float(m.group(1)), float(m.group(2)))
    m = re.search(r'(less than or equal to|≤|at most)\s*\$?(-?\d+\.?\d*)', t, re.I)
    if m:
        return ('lte', float(m.group(2)))
    m = re.search(r'(less than)\s*\$?(-?\d+\.?\d*)', t, re.I)
    if m:
        return ('lt', float(m.group(2)))
    m = re.search(r'(at least|≥|or more|or above)\s*\$?(-?\d+\.?\d*)', t, re.I)
    if m:
        return ('gte', float(m.group(2)))
    m = re.search(r'\$?(-?\d+\.?\d*)\s*(?:%)?\s*or more', t, re.I)
    if m:
        return ('gte', float(m.group(1)))
    m = re.search(r'(greater than|more than|above)\s*\$?(-?\d+\.?\d*)', t, re.I)
    if m:
        return ('gt', float(m.group(2)))
    m = re.search(r'exactly\s*\$?(-?\d+\.?\d*)', t, re.I)
    if m:
        return ('exact', float(m.group(1)))
    m = re.search(r'\bbe\s*\$?(-?\d+\.?\d*)\s*%?\??\s*$', t, re.I)
    if m:
        return ('exact', float(m.group(1)))
    m = re.search(r'\$?(-?\d+\.?\d*)\s*%', t)
    if m:
        return ('exact', float(m.group(1)))
    return None


def classify_relationship(pm_title, k_title):
    """Classify how a PM/Kalshi title pair that already passed
    boundary_match + date_align actually relate to each other.

    Returns (category, detail) where detail is a tuple of the parsed
    values involved, for debugging/printing.
    """
    p = parse_side(pm_title)
    k = parse_side(k_title)
    if p is None or k is None:
        return CAT_UNPARSED, None

    if p[0] == 'range':
        _, lo, hi = p
        if k[0] in ('gt', 'gte'):
            kv = k[1]
            if abs(kv - lo) < _EDGE_TOL:
                return CAT_SUBSET, (lo, hi, kv)
            if abs(kv - hi) < _EDGE_TOL:
                return CAT_ADJACENT, (lo, hi, kv)
            if lo <= kv <= hi:
                return CAT_PARTIAL, (lo, hi, kv)
            return CAT_MISMATCH, (lo, hi, kv)
        if k[0] == 'range':
            klo, khi = k[1], k[2]
            if abs(klo - lo) < _RANGE_TOL and abs(khi - hi) < _RANGE_TOL:
                return CAT_EXACT, (lo, hi, klo, khi)
            return CAT_RANGE_MISMATCH, (lo, hi, klo, khi)
        return CAT_UNHANDLED, (p, k)

    if k[0] == 'range':
        # mirror image of the block above, with PM/K swapped
        _, lo, hi = k
        if p[0] in ('gt', 'gte'):
            pv = p[1]
            if abs(pv - lo) < _EDGE_TOL:
                return CAT_SUBSET, (lo, hi, pv)
            if abs(pv - hi) < _EDGE_TOL:
                return CAT_ADJACENT, (lo, hi, pv)
            if lo <= pv <= hi:
                return CAT_PARTIAL, (lo, hi, pv)
            return CAT_MISMATCH, (lo, hi, pv)
        return CAT_UNHANDLED, (p, k)

    # both sides are single-valued (exact/gt/gte/lt/lte)
    pop, pv = p
    kop, kv = k
    if abs(pv - kv) > _VALUE_TOL:
        return CAT_MISMATCH, (pv, kv)
    if pop == 'exact' and kop == 'exact':
        return CAT_EXACT, (pv, kv)
    if pop in ('gte', 'gt') and kop in ('gte', 'gt'):
        if pop != kop:
            # e.g. PM "more than 5%" (gt, excludes 5.0 exactly) vs Kalshi
            # "at least 5.0%" (gte, includes it) -- same number, same
            # direction, but NOT the same payoff at the boundary itself.
            return CAT_STRICT_VS_INCLUSIVE, (pv, kv)
        return CAT_EXACT, (pv, kv)
    if pop in ('lte', 'lt') and kop in ('lte', 'lt'):
        if pop != kop:
            return CAT_STRICT_VS_INCLUSIVE, (pv, kv)
        return CAT_EXACT, (pv, kv)
    if pop == 'exact' and kop in ('gt', 'gte'):
        return CAT_BUCKET_VS_THRESHOLD, (pv, kv)
    if pop in ('gt', 'gte') and kop == 'exact':
        return CAT_POINT_VS_OPEN, (pv, kv)
    if pop == 'exact' and kop in ('lt', 'lte'):
        return CAT_BUCKET_VS_THRESHOLD, (pv, kv)
    if pop in ('lt', 'lte') and kop == 'exact':
        return CAT_POINT_VS_OPEN, (pv, kv)
    if pop in ('lt', 'lte') and kop in ('gt', 'gte'):
        return CAT_COMPLEMENT, (pv, kv)
    if pop in ('gt', 'gte') and kop in ('lt', 'lte'):
        return CAT_COMPLEMENT, (pv, kv)
    return CAT_UNHANDLED, (pop, pv, kop, kv)


def _print_relationship_breakdown(labeled):
    """labeled: list of (category, ...pair fields...). Prints category
    counts and how many/what fraction are in a SAFE_CATEGORIES bucket."""
    counts = Counter(cat for cat, *_ in labeled)
    total = len(labeled)
    if not total:
        return
    print(f"\n  -- signal 4: comparison-operator/direction breakdown ({total} pairs) --")
    for cat, n in counts.most_common():
        flag = "safe" if cat in SAFE_CATEGORIES else "NOT safe to price-diff"
        print(f"    {cat:<22} {n:>4}  ({n/total*100:5.1f}%)  [{flag}]")
    safe_n = sum(n for cat, n in counts.items() if cat in SAFE_CATEGORIES)
    print(f"  -> {safe_n}/{total} ({safe_n/total*100:.1f}%) are genuinely the same market (EXACT) and safe to compare Yes prices directly.")


def analyze_backtest(path, pct_tol, dollar_tol_frac, tolerance_days, samples, seed, exact_only=False):
    data = json.load(open(path))
    regex_only, triple = [], []
    for row in data:
        # pm_extra/k_extra (PM description / Kalshi rules text) are only
        # present in dumps regenerated after backtest_economics_markets.py
        # started capturing them -- fall back to "" for older dump files.
        score, pm_title, pm_url, pm_closed, k_title, k_url, k_closed = row[:7]
        pm_extra, k_extra = (row[7], row[8]) if len(row) >= 9 else ("", "")
        if not boundary_match(pm_title, k_title, pct_tol, dollar_tol_frac):
            continue
        regex_only.append((score, pm_title, pm_url, pm_closed, k_title, k_url, k_closed))
        if date_align_backtest(pm_closed, k_closed, tolerance_days) and growth_basis_align(
            pm_title, k_title, pm_extra, k_extra
        ):
            triple.append((score, pm_title, pm_url, pm_closed, k_title, k_url, k_closed))

    print(f"\n=== BACKTEST (last year, closed/settled) — {len(data)} pairs >= cosine 0.85 ===")
    print(f"  regex boundary-match only:        {len(regex_only)} pairs")
    print(f"  regex + close-date + growth-basis alignment (<= {tolerance_days}d): {len(triple)} pairs")
    removed = len(regex_only) - len(triple)
    print(f"  -> date/basis-alignment check removed {removed} pairs ({removed/len(regex_only)*100:.1f}% of regex-only) as likely date- or growth-basis-mismatched false positives" if regex_only else "")

    if triple:
        scores = [t[0] for t in triple]
        scores.sort()
        print(f"  score range: {scores[0]:.3f}-{scores[-1]:.3f}, median {scores[len(scores)//2]:.3f}")

    labeled = [(classify_relationship(pm_title, k_title)[0],) + row
               for row in triple
               for pm_title, k_title in [(row[1], row[4])]]
    _print_relationship_breakdown(labeled)

    exact = [row[1:] for row in labeled if row[0] in SAFE_CATEGORIES]
    output_set = exact if exact_only else triple
    label = "EXACT-only" if exact_only else "3-signal-filtered"

    rng = random.Random(seed)
    sample = rng.sample(output_set, min(samples, len(output_set)))
    print(f"\n  -- {len(sample)} random samples from the {label} set --")
    for score, pm_title, pm_url, pm_closed, k_title, k_url, k_closed in sample:
        cat, _ = classify_relationship(pm_title, k_title)
        print(f"  {score:.3f} [{cat}] | PM({pm_closed[:10]}): {pm_title}")
        print(f"                    | K ({k_closed[:10]}): {k_title}")
    return regex_only, triple, exact


def analyze_live(path, pct_tol, dollar_tol_frac, samples, seed, exact_only=False):
    data = json.load(open(path))
    regex_only, triple = [], []
    for score, pm_title, pm_url, k_title, k_url in data:
        if not boundary_match(pm_title, k_title, pct_tol, dollar_tol_frac):
            continue
        regex_only.append((score, pm_title, pm_url, k_title, k_url))
        if date_align_live(pm_title, k_title):
            triple.append((score, pm_title, pm_url, k_title, k_url))

    print(f"\n=== LIVE (this year, open) — {len(data)} pairs >= cosine 0.85 ===")
    print(f"  regex boundary-match only:      {len(regex_only)} pairs")
    print(f"  regex + year-mention alignment: {len(triple)} pairs")
    removed = len(regex_only) - len(triple)
    print(f"  -> date-alignment check removed {removed} pairs ({removed/len(regex_only)*100:.1f}% of regex-only) as likely date-mismatched false positives" if regex_only else "")

    if triple:
        scores = [t[0] for t in triple]
        scores.sort()
        print(f"  score range: {scores[0]:.3f}-{scores[-1]:.3f}, median {scores[len(scores)//2]:.3f}")

    labeled = [(classify_relationship(pm_title, k_title)[0],) + row
               for row in triple
               for pm_title, k_title in [(row[1], row[3])]]
    _print_relationship_breakdown(labeled)

    exact = [row[1:] for row in labeled if row[0] in SAFE_CATEGORIES]
    output_set = exact if exact_only else triple
    label = "EXACT-only" if exact_only else "3-signal-filtered"

    rng = random.Random(seed)
    sample = rng.sample(output_set, min(samples, len(output_set)))
    print(f"\n  -- {len(sample)} random samples from the {label} set --")
    for score, pm_title, pm_url, k_title, k_url in sample:
        cat, _ = classify_relationship(pm_title, k_title)
        print(f"  {score:.3f} [{cat}] | PM: {pm_title}")
        print(f"                    | K : {k_title}")
    return regex_only, triple, exact


def main():
    parser = argparse.ArgumentParser(description="cosine>=0.85 AND regex-boundary-match AND date/year-alignment AND operator/direction filter for economics markets")
    parser.add_argument("--pct-tol", type=float, default=0.10, help="Absolute percentage-point tolerance for numeric-boundary matching (default 0.10)")
    parser.add_argument("--dollar-tol-frac", type=float, default=0.02, help="Relative tolerance for dollar-amount matching (default 0.02 = 2%%)")
    parser.add_argument("--close-date-tolerance-days", type=float, default=3.0, help="Backtest: max allowed gap between PM/Kalshi actual close timestamps (default 3 days)")
    parser.add_argument("--samples", type=int, default=20, help="Random samples to print per dataset for manual review")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exact-only", action="store_true", help="Restrict printed samples/output to the EXACT (same-payoff, safe-to-price-diff) category instead of all 3-signal-filtered pairs")
    args = parser.parse_args()

    analyze_backtest(BACKTEST_PATH, args.pct_tol, args.dollar_tol_frac, args.close_date_tolerance_days, args.samples, args.seed, args.exact_only)
    analyze_live(LIVE_PATH, args.pct_tol, args.dollar_tol_frac, args.samples, args.seed, args.exact_only)


if __name__ == "__main__":
    main()
