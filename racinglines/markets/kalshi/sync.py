"""
Sync Kalshi's F1 markets into market_links (exchange = 'kalshi'), map each to the model prediction that
prices it, and store its tape, price history and order books in the same tables as Polymarket's. Nothing runs
unless called (`racinglines markets --exchange kalshi ...`); built and tested on mocked responses only, since
the cloud network policy blocks Kalshi's API. F1 roadmap, F1-9.

One link per Kalshi market: its YES contract (token_id = the market ticker, condition_id = the event ticker;
NO is the mirror image). Titles are classified like Polymarket's. Checked against the live listing on
2026-09-28 (event title / market title, series):

    "Azerbaijan Grand Prix Winner" / "Oscar Piastri to finish in first"   KXF1RACE          race_win
    "... Main Race: Podium Finishers" / "... to finish"                   KXF1RACEPODIUM    race_podium
    "... Main Race: Top 10 Finishers" / "... to finish top 10"            KXF1TOP10         race_top10
    "... Main Race: Top 5 Finishers" / "... to finish top 5"              KXF1TOP5          race_top5
    "... Main Race: Biggest Mover" / "Biggest Mover: Oscar Piastri"       KXF1BIGGESTMOVER  race_biggest_mover
    "... Qualifying Session (Q3): Pole Position"                          KXF1POLE          race_pole
    "... Main Race: Top Constructor" / "McLaren to finish in first"       KXF1TOPCONSTRUCTOR race_constructor_top
    "... Main Race: Fastest Lap" / "Fastest Lap: Oscar Piastri"           KXF1FASTLAP       race_fastest_lap
    "F1 Drivers Champion" / "Will Lando Norris win the F1 Drivers ..."    KXF1              champion
    "F1 Constructors Champion"                                            KXF1CONSTRUCTORS  constructors_champion
    "Dutch Grand Prix: Sprint Race Winner" / "... finish in first in the
        Sprint Race at the 2026 Dutch Grand Prix?"                       KXF1RACESPRINT    race_sprint_win *
    "... Sprint Qualifying: Pole Position" / "... fastest valid
        qualifying lap time in the Sprint Qualifying session (SQ3) ..."  KXF1SPRINTPOLE    race_sprint_pole *
    "... Sprint Race: Top 5 / Top 10 Finishers"                          KXF1SPRINTTOP5/10 race_sprint_top5/10 *
    "... Sprint Race: Top Constructor" / "Will Alpine F1 Team finish in
        first in the Sprint Race at ...?"         KXF1SPRINTTOPCONSTRUCTOR  race_sprint_constructor_top *
    "... Sprint Race: Fastest Lap"                                        KXF1SPRINTFASTLAP unmodeled (no model draws it)
    "F1 Matchup: Verstappen vs Hamilton" / "Will Max Verstappen beat
        Lewis Hamilton in the racing matchup?" (main race)              KXF1H2H           race_h2h
    retirements, race occurrence, transfers, ...                                            unmodeled

The kind is read first from the series ticker, an exact key (sports/f1.toml [markets.kalshi.kinds]); the titles
classify a series the table doesn't list, and give the Grand Prix either way. A link filed before a series was
classified keeps its old `unmodeled` until a sync re-reads it: closed links only come back with --include-closed
(KXF1TOP5's 330 and the sprint series were filed so on 2026-09-28, before #48 and the sprint kinds).

yes_sub_title is the driver (or the team); race_h2h has params.opponent_id. Head-to-head titles don't name
the Grand Prix: it's read from the event ticker's race code (KXF1H2H-BRIGP26VERHAM -> BRIGP26), which the
other series' events of the same weekend name ("KXF1RACE-BRIGP26": "British Grand Prix Winner"). Safety car,
red flag and rain props would take their kinds, but none was listed to check. Unmodeled markets are listed
with prices and no model price.

Each link keeps Kalshi's resolution rules (params.rules, from rules_primary) so markets on two venues are only
compared when their rules agree (F1-9), and its series ticker (params.series, for the price history).

The open listing drops an event once its markets close, so a decided market (pole after qualifying) would sit
open in market_links with its last pre-close quote. Each pass therefore re-reads by ticker the open links of the
synced series that the listing no longer returns (up to REREAD_MAX, most recently synced first) and stores their
status, result and last quote. A ticker Kalshi no longer serves (settled before its historical cutoff: a 404)
closes once its end date has passed.

* Sprint markets (docs/todo.md U5) are classified by default (sprints_enabled; on since 2026-10-06, owner);
RACINGLINES_KALSHI_SPRINTS=0 (or false/no) turns it off and every sprint market stays unmodeled. The sprint winner,
sprint pole, top 5, top 10 and top constructor take the sprint kinds of markets/kinds.toml (race_sprint_win settles
after the Sprint, race_sprint_pole after SQ; sports/f1.toml closes them when the session starts); the winner and
pole are priced by db.reads.model_prob, the others only from a run's simulations (kinds.fair). The sprint's fastest
lap stays unmodeled: no model draws it.

Other series (docs/todo.md, U9): NASCAR Cup, MotoGP and IndyCar have no title classifier (only a schema with
[markets] titles = true does: F1). Their schemas name the ticker prefixes ([markets.kalshi] series, e.g. KXNASCAR);
`sync(..., sport="nascar")` upserts their markets under their own competition, no driver or race lookup by title, so
trades, history and books record them like F1's; the archive pass sends the rows to Kalshi's tree by
`market_links.exchange`. A sport whose schema names an [identity] resolver has its links' kind, driver and race
filled by it, and a link whose kind the schema's [markets] kinds list, fully identified, is filed as that kind
(identity.promote); every other link stays `unmodeled`. Nothing about them runs unless the sport is named
(`--sport nascar`), so the default F1 sync is unchanged.
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from sqlalchemy import select, text

from racinglines import sports
from racinglines.db import models as m
from racinglines.markets import identity
from racinglines.markets.kalshi import client as K

SERIES = ()          # Kalshi series tickers to sync; empty: every Sports series that looks like F1 (see f1_series)
F1_WORDS = re.compile(r"\b(formula\s*1|formula one|f1|grand prix)\b", re.I)
CLOSED = {"closed", "settled", "finalized", "determined"}
STOP = {"who", "will", "win", "the", "a", "an", "at", "in", "of", "formula", "one", "which", "driver", "drivers", "f1"}
PROP = [(r"fastest lap", "race_fastest_lap"), (r"safety car", "race_safety_car"), (r"red[- ]flag", "race_red_flag"),
        (r"\brain", "race_rain")]
SPRINT_FLAG = "RACINGLINES_KALSHI_SPRINTS"
SPRINT_TOP = {"5": "race_sprint_top5", "10": "race_sprint_top10"}      # KXF1SPRINTTOP5 / KXF1SPRINTTOP10
SPRINT_KINDS = ("race_sprint_win", "race_sprint_pole") + tuple(SPRINT_TOP.values())   # one market per driver
TEAM_KINDS = ("race_constructor_top", "race_sprint_constructor_top", "constructors_champion")   # yes_sub_title: a team
ATHLETE_KINDS = ("race_win", "race_podium", "race_top10", "race_top5", "race_pole", "race_fastest_lap",
                 "race_biggest_mover", "champion")       # one market per driver: the driver is the yes_sub_title
REREAD_MAX = 100     # open links re-read by ticker per pass when their event left the open listing


def sprints_enabled():
    """Sprint markets are classified unless RACINGLINES_KALSHI_SPRINTS is 0/false/no (on by default since 2026-10-06)."""
    return os.environ.get(SPRINT_FLAG, "1").strip().lower() not in ("0", "false", "no", "off")


def classify_sprint(low):
    """The sprint kind of a lower-cased sprint title: the top constructor ("Sprint Race: Top Constructor", "Will
    Alpine F1 Team finish in first in the Sprint Race"), top 5 / top 10 ("Sprint Race: Top 5 Finishers"), sprint
    pole ("Sprint Qualifying: Pole Position", "SQ3"), the sprint winner ("Sprint Race Winner", "finish in first in the
    Sprint Race"), else unmodeled: the sprint's fastest lap (no model draws it), any other top N, and any prop."""
    if re.search(r"fastest lap|podium|safety car|red[- ]flag|\brain|mover|retire", low):
        return "unmodeled"
    if "constructor" in low:
        return "race_sprint_constructor_top"
    if mm := re.search(r"\btop[- ]?(\d+)\b", low):
        return SPRINT_TOP.get(mm.group(1), "unmodeled")
    if re.search(r"\bpole\b|\bsq3\b", low):
        return "race_sprint_pole"
    if re.search(r"\bwin(ner)?\b|finish(es)? in (exactly )?first", low):
        return "race_sprint_win"
    return "unmodeled"


def gp_name(t):
    """The Grand Prix a title names ("Who will win the Singapore Grand Prix?" -> "Singapore Grand Prix"), or None.
    2025 titles also say "F1 Australian Grand Prix Winner?", "Las Vegas GP: ..." and "Gran Premio de Mexico Winner?"."""
    for mm in re.finditer(r"((?:[A-Z][A-Za-z\.'-]*\s+)+)(?:Grand Prix|GP)\b", t or ""):
        words = [w for w in mm.group(1).split() if w.lower() not in STOP]
        if words:
            return " ".join(words) + " Grand Prix"
    mm = re.search(r"Gran Premio (?:de |del |d')?([A-Z][\w-]+)", t or "")       # "Gran Premio de Mexico Winner?" (2025)
    return f"{mm.group(1)} Grand Prix" if mm else None


def is_sprint(kind):
    """A kind of the sprint weekend's own rounds (the sprint, sprint qualifying): its row's session or stage."""
    from racinglines.markets import kinds as K
    k = K.KINDS.get(kind)
    return k is not None and ("sprint" in (k.session, k.stage) or k.stage == "sprint_qual")


def classify(event_title, market_title="", gp=None, sprints=None, series=None, sport="f1"):
    """(prediction kind, Grand Prix name or None) for a Kalshi market. gp: the Grand Prix when the titles
    don't name it (head-to-heads). sprints: classify sprint markets (default: sprints_enabled(), on).
    series: the market's series ticker; when the sport's schema maps it ([markets.kalshi] kinds), that is the kind
    (an exact key), and the titles only give the Grand Prix. Otherwise the kind is read from the titles."""
    t = f"{event_title or ''} {market_title or ''}"
    gp = gp_name(event_title) or gp_name(market_title) or gp
    low = t.lower()
    kind = sports.kalshi_kinds(sport).get(series) if series else None
    if kind is not None:
        from racinglines.markets import kinds as K
        if K.KINDS[kind].payoff == "standings":
            return kind, None
        if not gp:
            return "unmodeled", None
        if is_sprint(kind) and not (sprints_enabled() if sprints is None else sprints):
            return "unmodeled", gp
        return kind, gp
    if re.search(r"constructors'? champion", low):
        return "constructors_champion", None
    if re.search(r"drivers'? champion", low) or (re.search(r"\bchampion(ship)?\b", low) and not gp):
        return "champion", None
    if not gp:
        return "unmodeled", None
    if "sprint" in low:                          # sprint race, sprint qualifying: not the model's race ...
        sprints = sprints_enabled() if sprints is None else sprints
        return (classify_sprint(low) if sprints else "unmodeled"), gp    # ... unless the sprint kinds are on
    for pat, kind in PROP:
        if re.search(pat, low):
            return kind, gp
    if re.search(r"biggest mover", low):
        return "race_biggest_mover", gp
    if re.search(r"finish ahead of|head[- ]to[- ]head|\bvs\.?\b|matchup", low):
        return "race_h2h", gp
    if "constructor" in low:
        return "race_constructor_top", gp
    if "podium" in low:
        return "race_podium", gp
    if re.search(r"\btop[- ]?10\b", low):
        return "race_top10", gp
    if re.search(r"\btop[- ]?5\b", low):
        return "race_top5", gp
    if "pole" in low:
        return "race_pole", gp
    if re.search(r"\bwin(ner)?\b", low):
        return "race_win", gp
    return "unmodeled", gp


def _time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def _volume(mk):
    """Contracts traded: `volume_fp` ("989756.84") on current responses, `volume` on older ones."""
    v = mk.get("volume_fp") if mk.get("volume_fp") not in (None, "") else mk.get("volume")
    return float(v) if v not in (None, "") else None


def _quote(mk):
    """(bid, ask, mid) of the YES contract; an empty side (0 bid / 1.00 ask) is None. A dead book's
    synthetic 0.50 midpoint is treated as no quote, but a real last trade such as 0.01 is kept."""
    bid, ask = K.price(mk, "yes_bid"), K.price(mk, "yes_ask")
    bid = None if bid in (None, 0) else bid
    ask = None if ask is None or ask >= 1.0 else ask
    last = K.price(mk, "last_price")
    if bid is None and ask is None:
        if last is None or abs(last - 0.5) > 1e-9:
            return None, None, last or None
        return None, None, None
    mid = (bid + ask) / 2 if bid is not None and ask is not None else (last or None)
    return bid, ask, mid


def _race_code(event_ticker):
    """The race part of an event ticker: KXF1RACE-BRIGP26 -> BRIGP26 (KXF1H2H-BRIGP26VERHAM -> BRIGP26VERHAM)."""
    parts = (event_ticker or "").split("-")
    return parts[1] if len(parts) > 1 else ""


def season(ev, default=None):
    """The season an event belongs to: the two digits ending its race code or ticker (KXF1RACE-ABUDGP25,
    KXF1-25, KXF1H2H-BRIGP26VERHAM), else the year its first market closes, else `default`."""
    code = _race_code(ev.get("event_ticker"))
    mm = re.match(r"[A-Z]+GP(\d{2})", code) or re.fullmatch(r"(\d{2})", code)
    if mm:
        return 2000 + int(mm.group(1))
    ends = [_time(mk.get("close_time") or mk.get("expiration_time")) for mk in ev.get("markets") or []]
    ends = [e for e in ends if e is not None]
    return min(ends).year if ends else default


def gp_codes(events):
    """{race code: Grand Prix name} from events whose title names the Grand Prix (BRIGP26 -> British Grand Prix)."""
    out = {}
    for ev in events:
        gp, code = gp_name(ev.get("title")), _race_code(ev.get("event_ticker"))
        if gp and re.fullmatch(r"[A-Z]+GP\d{2}", code):
            out[code] = gp
    return out


def link_rows(events, resolver, modeled=True):
    """market_links values (with token_id) for every market of `events` (Kalshi /events with nested markets).
    resolver: driver(name), team(name), race(gp, end_date) -> (race_id, event key), as polymarket.sync.Resolver.
    modeled=False (a tape-only sport): every market is `unmodeled`, the resolver is never called."""
    rows = []
    codes = gp_codes(events) if modeled else {}
    for ev in events:
        code = _race_code(ev.get("event_ticker"))
        ev_gp = next((g for c, g in codes.items() if code.startswith(c)), None)
        for mk in ev.get("markets") or []:
            kind, gp = (classify(ev.get("title"), mk.get("title"), ev_gp, series=ev.get("series_ticker")) if modeled
                        else ("unmodeled", None))
            sub = mk.get("yes_sub_title") or mk.get("subtitle") or ""
            end = _time(mk.get("close_time") or mk.get("expiration_time"))
            race_id, race_key = resolver.race(gp, end) if gp else (None, None)
            athlete_id, params = None, None
            if kind == "race_h2h":
                mm = re.search(r"Will (.+?) (?:finish ahead of|beat) (.+?)(?: at| in|\?|$)", mk.get("title") or "")
                a, b = (resolver.driver(mm.group(1)), resolver.driver(mm.group(2))) if mm else (None, None)
                athlete_id, params = a, {"opponent_id": b}
            elif kind in TEAM_KINDS:
                params = {"team": resolver.team(sub)}
            elif kind in ATHLETE_KINDS + SPRINT_KINDS:
                athlete_id = resolver.driver(sub)
            matched = kind == "unmodeled" or (
                (athlete_id is not None or (params or {}).get("team") or kind in ("race_safety_car", "race_red_flag", "race_rain"))
                and (params is None or all(v is not None for v in params.values()))
                and (not kind.startswith("race_") or race_id is not None))
            params = dict(params or {}, **({"event_key": race_key} if race_key else {}),
                          series=ev.get("series_ticker"), rules=mk.get("rules_primary") or None)
            bid, ask, mid = _quote(mk)
            status = (mk.get("status") or "").lower()
            result = (mk.get("result") or "").lower()
            tick = mk.get("tick_size")
            rows.append(dict(
                token_id=mk["ticker"], exchange="kalshi", market_slug=mk["ticker"],
                question=mk.get("title") or ev.get("title") or mk["ticker"], condition_id=mk.get("event_ticker") or ev.get("event_ticker"),
                outcome=sub or "Yes", neg_risk=bool(ev.get("mutually_exclusive")),
                tick_size=(float(tick) / 100 if tick else 0.01), min_size=1.0,
                athlete_id=athlete_id, race_id=race_id, prediction=kind if matched else "unmodeled",
                params={k: v for k, v in params.items() if v is not None}, invert=False,
                event_slug=ev.get("event_ticker"), event_title=ev.get("title"), group_title=sub or mk.get("title"),
                last_bid=bid, last_ask=ask, last_price=mid, volume=_volume(mk),
                end_date=end, closed=status in CLOSED, resolved_yes={"yes": True, "no": False}.get(result),
                active=status not in CLOSED))
    return rows


def f1_series(kc):
    """The series tickers to sync: SERIES, or every Sports series whose ticker starts KXF1 (some titles don't
    say F1: KXF1POLEPOSITION is "Qualify in Pole Position") or whose title looks like F1."""
    if SERIES:
        return list(SERIES)
    return [s["ticker"] for s in kc.series(category="Sports")
            if s.get("ticker", "").startswith("KXF1") or F1_WORDS.search(s.get("title") or "")]


def series_for(kc, sport="f1", series=None):
    """The series tickers to sync for a sport: `series` when given; F1's discovery (f1_series); else every
    Sports series whose ticker starts with one of the sport's [markets.kalshi] prefixes (sports/<code>.toml)."""
    if series:
        return list(series)
    if sport == "f1":
        return f1_series(kc)
    prefixes = sports.kalshi_series(sport)
    if not prefixes:
        raise ValueError(f"sport {sport!r} names no Kalshi series: set [markets.kalshi] series in sports/{sport}.toml")
    return [s["ticker"] for s in kc.series(category="Sports") if s.get("ticker", "").startswith(prefixes)]


def competition(session, sport):
    """(Competition, its first Category) of a sport, as the sync files links under. Creates the rows from the
    sport's schema when the database has not been seeded for it (db/ingest.ensure_competition)."""
    from racinglines.db import ingest
    return ingest.ensure_competition(session, sport)


def sync(session, conn, year=2026, include_closed=False, kc=None, sport="f1", series=None):
    """Upsert one market_links row per Kalshi market of `sport` (open ones; with include_closed, settled ones
    too). F1 (the default) is classified and matched to its races and drivers; a tape-only sport (nascar,
    motogp, indycar) is filed under its own competition with every link `unmodeled`. series: explicit series
    tickers instead of discovery. kc: a kalshi Client (default: a new one on the network)."""
    from racinglines.markets.polymarket.sync import Resolver
    comp, cat = competition(session, sport)
    modeled = sports.titles_classified(sport)   # [markets] titles: the title classifier and the F1 resolver
    kc = kc or K.Client()
    now = datetime.now(timezone.utc)
    events = []
    names = series_for(kc, sport, series)
    for s in names:
        events += kc.events(series_ticker=s, status="open")
        if include_closed:
            events += kc.events(series_ticker=s, status="settled")
    events = list({e["event_ticker"]: e for e in events}.values())      # an event can come back under both statuses
    for ev in events:                           # settled before Kalshi's historical cutoff: markets under /historical
        if not ev.get("markets") and include_closed:
            ev["markets"] = kc.historical_markets(ev["event_ticker"])
    stats = dict(events=len(events), links=0, modeled=0, unmatched=0, new=0)
    # Each event is matched against its own season's races and drivers: settled events of every year come
    # back whatever `year` is, and matching a 2025 event against 2026 left it unmodeled.
    if modeled:
        by_season = {}
        for ev in events:
            by_season.setdefault(season(ev, year), []).append(ev)
        rows = [r for y, evs in sorted(by_season.items()) for r in link_rows(evs, Resolver(conn, y))]
    else:
        rows = link_rows(events, None, modeled=False)
    who = identity.linker(sport, conn) if not modeled else None      # a tape-only sport with a resolver: driver, race, kind
    if who:
        who.fill(rows)
        identity.promote(sport, rows)            # prediction = params.kind where [markets] kinds files it, identified
        stats["identity"] = dict(who.counts)
    seen = {r["token_id"] for r in rows}
    for row in rows:
        tok = row.pop("token_id")
        values = dict(row, competition_id=comp.id, category_id=cat.id, synced_at=now)
        link = session.scalars(select(m.MarketLink).filter_by(exchange="kalshi", token_id=tok)).first()
        if link is None:
            session.add(m.MarketLink(token_id=tok, first_seen_at=now, **values))
            stats["new"] += 1
        else:
            for k, v in values.items():
                setattr(link, k, v)
        stats["links"] += 1
        stats["modeled" if row["prediction"] != "unmodeled" else "unmatched"] += 1
    session.flush()
    stats["reread"] = reread(session, kc, comp.id, names, seen, now)    # closed since the last pass: off the listing
    session.commit()
    return stats


def reread(session, kc, competition_id, series, seen, now, limit=REREAD_MAX):
    """Re-read by ticker the open links of `series` that this pass's listing (`seen` tickers) no longer returned
    and store each one's status, result, last quote and volume. Returns how many links changed."""
    ids = session.execute(text("""SELECT id FROM market_links
                                  WHERE exchange = 'kalshi' AND competition_id = :c AND closed IS NOT TRUE
                                    AND params->>'series' = ANY(:s) AND NOT (token_id = ANY(:seen))
                                  ORDER BY synced_at DESC NULLS LAST, id LIMIT :n"""),
                          dict(c=competition_id, s=list(series), seen=list(seen), n=limit)).scalars().all()
    n = 0
    for link in (session.get(m.MarketLink, i) for i in ids):
        try:
            mk = kc.market(link.token_id) or {}
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404 and link.end_date is not None and link.end_date < now:
                link.closed, link.active, link.synced_at = True, False, now      # settled before the historical cutoff
                n += 1
            continue
        except httpx.HTTPError:
            continue
        status, result = (mk.get("status") or "").lower(), (mk.get("result") or "").lower()
        if not status:
            continue
        link.last_bid, link.last_ask, link.last_price = _quote(mk)
        link.volume = _volume(mk) if _volume(mk) is not None else link.volume
        link.closed, link.active = status in CLOSED, status not in CLOSED
        link.resolved_yes = {"yes": True, "no": False}.get(result)
        link.synced_at = now
        n += 1
    return n


def trade_rows(ticker, event_ticker, trades):
    """market_trades rows for Kalshi trades: the taker's side of the YES contract (taker bought YES = BUY,
    bought NO = SELL YES), at the YES price; size in contracts."""
    rows = []
    for t in trades:
        yes = K.price(t, "yes_price")
        if yes is None:
            continue
        rows.append(dict(token_id=ticker, condition_id=event_ticker or ticker, outcome_index=0, ts=_time(t["created_time"]),
                         side="BUY" if (t.get("taker_side") or "").lower() == "yes" else "SELL", price=yes,
                         size=float(t.get("count_fp") or t.get("count") or 0), tx_hash=str(t["trade_id"])[:80],
                         wallet=""))            # no wallets on Kalshi; "" (not NULL) keeps uq_market_trade deduplicating
    return rows


def _tickers(conn, event_tickers=None, open_only=False, sport=None):
    """(token, event ticker, series) of the Kalshi links of the given events, or of every event of `sport`'s
    competition when no events are given (open_only: markets not closed)."""
    if not event_tickers and not sport:
        raise ValueError("give event tickers or a sport")
    where, params = ["l.exchange = 'kalshi'"], {}
    if event_tickers:
        where.append("l.condition_id = ANY(:e)")
        params["e"] = list(event_tickers)
    else:
        where.append("co.code = :c")
        params["c"] = sports.load(sport)["competition"]["code"]
    if open_only:
        where.append("NOT l.closed")
    return conn.execute(text(f"""SELECT l.token_id, l.condition_id, l.params->>'series' FROM market_links l
                                 LEFT JOIN competitions co ON co.id = l.competition_id
                                 WHERE {' AND '.join(where)} ORDER BY l.condition_id, l.token_id"""), params).all()


def fetch_trades(session, conn, event_tickers=None, since=None, kc=None, sport=None, open_only=False):
    """Store every trade on the markets of the given Kalshi events (or of every event of `sport`). Idempotent.
    since: only trades from then on; open_only: markets not closed (the recorder's pass). Returns trades stored."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    kc = kc or K.Client()
    n = 0
    for tok, ev, _ in _tickers(conn, event_tickers, open_only=open_only, sport=sport):
        rows = trade_rows(tok, ev, kc.trades(tok, min_ts=int(since.timestamp()) if since else None))
        for i in range(0, len(rows), 1000):
            session.execute(pg_insert(m.MarketTrade).values(rows[i:i + 1000]).on_conflict_do_nothing(constraint="uq_market_trade"))
        n += len(rows)
    session.commit()
    return n


def history_rows(ticker, candles):
    """market_price_history rows from Kalshi candlesticks: the close of the YES price, else the bid/ask mid when
    both sides quote, plus the bid and ask closes. An empty side (0 bid / 1.00 ask, as _quote reads it) is None.
    An hour with no price (no trade and a one-sided or dead book) gives no row: price is NOT NULL, a dead book's
    mid is 0.50, not a price, and the replays' staleness rule covers the gap."""
    rows = []
    for c in candles:
        bid = K.price(c.get("yes_bid") or {}, "close")
        ask = K.price(c.get("yes_ask") or {}, "close")
        bid = None if not bid else bid
        ask = None if ask is None or ask >= 1.0 else ask
        p = K.price(c.get("price") or {}, "close")
        if p is None and bid is not None and ask is not None:
            p = (bid + ask) / 2
        if p is not None:
            rows.append(dict(token_id=ticker, ts=datetime.fromtimestamp(int(c["end_period_ts"]), tz=timezone.utc),
                             price=p, bid=bid, ask=ask))
    return list({r["ts"]: r for r in rows}.values())       # one row per ts: an upsert can't touch a row twice


MAX_CANDLES = 4000      # per candlesticks request: Kalshi answers 400 above 5,000 (seen on the 2026-10-05 backfill)


def _candles(kc, series, ticker, start, end, period):
    """Every candle of [start, end], asked for in windows of at most MAX_CANDLES periods."""
    out, a, step = [], start, timedelta(minutes=period * MAX_CANDLES)
    while a < end:
        b = min(a + step, end)
        out += kc.candlesticks(series, ticker, a.timestamp(), b.timestamp(), period)
        a = b
    return out


def _raw_path(raw_dir, series, ticker):
    return Path(raw_dir) / (series or "_") / f"{ticker}.json"


def fetch_history(session, conn, event_tickers, start, end, period=60, kc=None, sport=None, save_raw=None,
                  from_raw=None):
    """Store price history (candlesticks, `period` minutes: 1, 60 or 1440) for the given Kalshi events (or
    every event of `sport`). save_raw: also write each market's candlesticks response, as Kalshi sent it, to
    <save_raw>/<series>/<ticker>.json. from_raw: read those files instead of calling Kalshi (a market with no
    file is skipped), so a saved pull can be re-imported, and its rows re-derived, without the network."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from racinglines import progress
    kc = kc or (None if from_raw else K.Client())
    n = 0
    for tok, _, series in progress.track(_tickers(conn, event_tickers, sport=sport), unit="market", name=lambda x: x[0]):
        if from_raw:
            f = _raw_path(from_raw, series, tok)
            if not f.exists():
                continue
            candles = json.loads(f.read_text())["candlesticks"]
        else:
            candles = _candles(kc, series, tok, start, end, period)
            if save_raw:
                f = _raw_path(save_raw, series, tok)
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(dict(ticker=tok, series=series, start=start.isoformat(), end=end.isoformat(),
                                             period=period, fetched_at=datetime.now(timezone.utc).isoformat(),
                                             candlesticks=candles)))
        rows = history_rows(tok, candles)
        if rows:
            ins = pg_insert(m.MarketPriceHistory).values(rows)
            session.execute(ins.on_conflict_do_update(index_elements=["token_id", "ts"], set_={
                "price": ins.excluded.price, "bid": ins.excluded.bid, "ask": ins.excluded.ask}))
            session.commit()                     # per market: a long backfill keeps what it has if it stops
        n += len(rows)
    return n


def coverage(conn, event_tickers=None, sport=None):
    """How much of the stored price history (Parquet and Postgres) carries the candle's bid and ask, for the given
    Kalshi events (or every event of `sport`): rows, with both sides, and prices outside the closing quote (a last
    trade made before the book moved, so some are expected)."""
    from racinglines.markets import store as MS
    toks = [t for t, _, _ in _tickers(conn, event_tickers, sport=sport)]
    ph = MS.read(conn, "prices", tokens=toks, root=MS.root_for("kalshi"))
    both = ph.dropna(subset=["bid", "ask"])
    gap = (both["bid"] - both["price"]).clip(lower=0) + (both["price"] - both["ask"]).clip(lower=0)
    return dict(markets=int(ph["token_id"].nunique()), rows=len(ph), with_bid=int(ph["bid"].notna().sum()),
                with_ask=int(ph["ask"].notna().sum()), both=len(both), outside=int((gap > 1e-9).sum()),
                outside_5c=int((gap > 0.05).sum()), first=str(ph["ts"].min()) if len(ph) else None,
                last=str(ph["ts"].max()) if len(ph) else None)


def book_row(ticker, ob, ts, depth=10):
    """A market_book_snapshots row from Kalshi's order book (YES bids and NO bids; a NO bid at p is a YES ask
    at 1 - p). Levels [[price, contracts]], best first."""
    def levels(side):
        d = ob.get(f"{side}_dollars")
        if d:
            return [(float(p), float(q)) for p, q in d]
        return [(float(p) / 100, float(q)) for p, q in ob.get(side) or []]
    bids = sorted(levels("yes"), reverse=True)
    asks = sorted((round(1 - p, 4), q) for p, q in levels("no"))
    return dict(token_id=ticker, ts=ts, best_bid=bids[0][0] if bids else None, best_ask=asks[0][0] if asks else None,
                bids=[list(x) for x in bids[:depth]], asks=[list(x) for x in asks[:depth]])


def snapshot_books(session, conn, event_tickers=None, depth=10, kc=None, sport=None):
    """One order-book snapshot for every open market of the given Kalshi events (or of `sport`). Returns
    snapshots stored."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    kc = kc or K.Client()
    now = datetime.now(timezone.utc)
    rows = [book_row(tok, kc.orderbook(tok, depth), now, depth)
            for tok, _, _ in _tickers(conn, event_tickers, open_only=True, sport=sport)]
    if rows:
        session.execute(pg_insert(m.MarketBookSnapshot).values(rows).on_conflict_do_nothing())
        session.commit()
    return len(rows)
