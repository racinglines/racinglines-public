"""
Generic sportsbook slips against our model and the prediction markets (docs/sportsbook/slips.md): the functions behind
`racinglines book map | price | settle` and the MCP tools `map_book`, `price_book` and `settle_book`, which wrap them
with the book passed as text. Read-only on the database: every query runs in a READ ONLY transaction.

    load_file(path) / load_text(text)          a validated book (books/schema.py)
    map_book(conn, book)                       each line's legs resolved to a race, athletes and a kind by exact keys
    price_book(conn, book, runs=, sims=)       + the model's fair, the linked market's price, the book's implied
                                               probability and the EV against the model and against the market
    settle_book(conn, book)                    + each leg's and line's result from the stored classification

A line is one bet: a single (its `market` one kind) or a slip of several legs (`kind = "combo"`, markets/combos.py).
Its legs may sit on other events and sports than the book's (`sport`, `event` or `race_id` on the leg or the combo
table), so one book can hold a parlay across F1, NASCAR and MotoGP.

Exact keys only (owner policy): a race is `race_id`, or (sport, season, round) as events.series_round of the sport's
competition and category; an athlete is an id or an exact display name of the race's field (predictions, results and
linked markets), through the book's own exact `[aliases]` table. Anything else is listed as unmapped with the reason,
never guessed.

Slip probability. Legs on the same race are priced jointly when the simulations can say so: an OutcomeSims for that
race passed in (`sims`, e.g. `--sims RACE_ID=FILE.npz`), else a stored `f1_combo` run (model_runs kind "combo") holding
exactly these legs; otherwise the product of the legs' marginals, flagged "correlated: EV approximate". Legs on
different races are taken as independent (the product). The market's slip probability is always the product of its
legs' prices (no exchange lists the combo), flagged the same way when legs share a race.

Model and market are shown side by side and never blended (owner, 2026-10-07: no blend)
(docs/sportsbook/slips.md, decision log placeholder).
"""

import tomllib
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from racinglines import sports as SP
from racinglines.books import schema as S
from racinglines.markets import kinds as K

COMPLEMENT = {"yes": False, "over": False, "no": True, "under": True}
EXCHANGE_ORDER = ("polymarket", "kalshi", "og")          # tie-break between equally fresh quotes
CORRELATED = "correlated: EV approximate"
ASSUMPTIONS = (
    "EV is per unit staked: probability x decimal payout - 1; decimal payout = 1 / the book's implied probability.",
    "Legs on different races are independent: the slip probability is the product of their probabilities.",
    "Legs on the same race are priced jointly from simulations when they are available (passed in, or a stored combo "
    "run with exactly these legs); otherwise the product of the marginals, flagged '" + CORRELATED + "'.",
    "The market's slip probability is the product of its legs' prices: no exchange lists the slip itself.",
    "Model and market are shown side by side, never blended (owner, 2026-10-07).",
    "The market price is the freshest linked exchange's mid (bid/ask midpoint, else its last price, else the tape's).",
)
LEG_FIELDS = ("line", "leg", "status", "reason", "sport", "competition", "season", "round", "event_key", "race_id",
              "event", "kind", "side", "athlete_id", "athlete", "opponent_id", "opponent", "team", "threshold", "odds")
PRICE_FIELDS = ("book_prob", "run_id", "run_source", "model_prob", "model_note", "market_prob", "market_exchange",
                "market_bid", "market_ask", "market_synced_utc", "ev_model", "ev_market")
LINE_FIELDS = ("line", "id", "title", "selection", "kind", "status", "reason", "n_legs", "races", "odds",
               "decimal_odds", "book_prob", "model_prob", "model_method", "market_prob", "market_method", "flags",
               "edge_model", "edge_market", "ev_model", "ev_market", "result", "payout", "profit", "box")


# --- loading -------------------------------------------------------------------------------------------------------

def load_file(path):
    return S.load_book(path)


def load_text(raw, name="book"):
    """A book passed as TOML text (the MCP tools), validated."""
    return S.validate_book(tomllib.loads(raw), name)


def readonly(conn):
    """Make the connection's current transaction read-only (Postgres): a write would fail, not happen."""
    conn.execute(text("SET TRANSACTION READ ONLY"))
    return conn


def _r(x, nd=6):
    return None if x is None else round(float(x), nd)


def decimal_odds(odds, fmt):
    """The payout per unit staked (stake included) of a quote in `fmt`: 1 / its implied probability."""
    return 1.0 / S.to_prob(odds, fmt)


# --- map -----------------------------------------------------------------------------------------------------------

class _Db:
    """Per-call caches of the read-only lookups."""

    def __init__(self, conn, asof=None):
        self.conn, self.races, self.fields, self.results, self.cache = conn, {}, {}, {}, {}
        self.asof = None if asof is None else pd.Timestamp(asof).tz_convert("UTC") if pd.Timestamp(asof).tzinfo \
            else pd.Timestamp(asof).tz_localize("UTC")            # price as of this moment (price_book's asof)

    def q(self, sql, **params):
        return pd.read_sql(text(sql), self.conn, params=params)

    def race(self, sport, key, race_id):
        """(info dict, None) or (None, reason) for an exact race key."""
        ck = (sport, key, race_id)
        if ck in self.races:
            return self.races[ck]
        self.races[ck] = out = self._race(sport, key, race_id)
        return out

    def _race(self, sport, key, race_id):
        if race_id is None and sport is None:
            return None, "no sport: set book.sport or the leg's sport (sports/<code>.toml)"
        comp = SP.load(sport)["competition"] if sport else None
        if race_id is None:
            if key is None:
                return None, "no event key"
            season, rnd = (int(x) for x in key.split("-"))
            cat = next(iter(comp["categories"]))
            df = self.q("""
                SELECT ra.id FROM races ra JOIN events e ON e.id = ra.event_id JOIN seasons s ON s.id = e.season_id
                JOIN competitions co ON co.id = s.competition_id JOIN categories c ON c.id = ra.category_id
                WHERE co.code = :comp AND s.year = :y AND e.series_round = :r AND c.code = :cat ORDER BY ra.id""",
                        comp=comp["code"], y=season, r=rnd, cat=cat)
            if len(df) != 1:
                return None, (f"no {sport} race with season {season} round {rnd} ({comp['code']} {cat}) in the database"
                              if not len(df) else f"{len(df)} {sport} races match {key}: give race_id")
            race_id = int(df["id"].iloc[0])
        from racinglines.markets.venues import race_info
        info = race_info(self.conn, int(race_id))
        if info is None:
            return None, f"race_id {race_id} is not in the database"
        if comp is not None and info["competition"] != comp["code"]:
            return None, f"race_id {race_id} is {info['competition']}, not {sport} ({comp['code']})"
        info["sport_code"] = sport or SP.by_competition(info["competition"])["sport"]["code"]
        rows = self.q("SELECT e.series_round FROM races ra JOIN events e ON e.id = ra.event_id WHERE ra.id = :r",
                      r=int(race_id))
        info["round"] = None if pd.isna(rows["series_round"].iloc[0]) else int(rows["series_round"].iloc[0])
        return info, None

    def field(self, race_id):
        """{athlete_id: display_name}: the race's predictions, results and linked markets."""
        if race_id not in self.fields:
            df = self.q("""
                SELECT a.id, a.display_name FROM athletes a WHERE a.id IN (
                    SELECT athlete_id FROM race_predictions WHERE race_id = :r
                    UNION SELECT r.athlete_id FROM results r JOIN rounds ro ON ro.id = r.round_id WHERE ro.race_id = :r
                    UNION SELECT athlete_id FROM market_links WHERE race_id = :r AND athlete_id IS NOT NULL)
                ORDER BY a.id""", r=race_id)
            self.fields[race_id] = dict(zip(df["id"].astype(int), df["display_name"]))
        return self.fields[race_id]


def _athlete(db, race_id, name, aid, aliases, what):
    """(athlete_id, display name, None) or (None, None, reason): an exact id or display name in the race's field."""
    field = db.field(race_id)
    if aid is not None:
        return (aid, field[aid], None) if aid in field else (None, None, f"{what} id {aid} is not in race {race_id}'s field")
    want = aliases.get(name, name)
    hits = [i for i, n in field.items() if n == want]
    if len(hits) == 1:
        return hits[0], want, None
    why = "is not an exact name in" if not hits else f"matches {len(hits)} athletes of"
    return None, None, f"{what} {name!r}" + (f" (alias {want!r})" if want != name else "") + f" {why} race {race_id}'s field"


def _map_leg(db, m, where, aliases):
    """One market table (a single, or a combo's leg) -> a leg row: mapped, or unmapped with the reason."""
    row = dict.fromkeys(LEG_FIELDS)
    kind = m.get("kind")
    side = m.get("side")
    row.update(kind=kind, side=side or ("over" if "line" in m else "yes"), team=m.get("team"), threshold=m.get("line"),
               odds=m.get("odds"))
    sport = m.get("sport", where.get("sport"))
    key = m.get("event", where.get("event"))
    race_id = m.get("race_id", where.get("race_id") if "event" not in m and "sport" not in m else None)
    info, why = db.race(sport, key, race_id)
    if info is None:
        return dict(row, status="unmapped", reason=why, sport=sport, event_key=key, race_id=race_id)
    row.update(sport=info["sport_code"], competition=info["competition"], season=int(info["season"]),
               round=info["round"], race_id=int(info["race_id"]), event=info["title"] if info.get("venue") else info["name"],
               event_key=f"{int(info['season'])}-{info['round']}" if info["round"] is not None else None)
    problems = []
    k = K.KINDS.get(kind)
    names_driver = "driver" in m or "driver_id" in m
    if names_driver:
        row["athlete_id"], row["athlete"], why = _athlete(db, row["race_id"], m.get("driver"), m.get("driver_id"),
                                                          aliases, "driver")
        if why:
            problems.append(why)
    if "opponent" in m or "opponent_id" in m:
        row["opponent_id"], row["opponent"], why = _athlete(db, row["race_id"], m.get("opponent"), m.get("opponent_id"),
                                                            aliases, "opponent")
        if why:
            problems.append(why)
    if k is None:
        row["reason"] = f"{kind} is a race prop (history rate), not in the kinds registry: not priced from a model run"
    if problems:
        return dict(row, status="unmapped", reason="; ".join(problems))
    return dict(row, status="mapped")


def _place(m):
    return {f: m[f] for f in ("sport", "event", "race_id") if f in m}


def map_book(conn, book, _db=None):
    """{"book": ..., "lines": [line rows, each with "legs"], "unmapped": [line ids]}."""
    db = _db or _Db(conn)
    b = book["book"]
    aliases = book.get("aliases", {})
    base = {"sport": b.get("sport"), "event": b["event"]}
    lines = []
    for i, ln in enumerate(book["lines"]):
        m = ln["market"]
        row = dict.fromkeys(LINE_FIELDS)
        row.update(line=i + 1, id=ln.get("id", str(i + 1)), title=ln["title"], selection=ln["selection"],
                   odds=ln.get("odds"), box=ln.get("box"))
        legs = []
        if m == "unmapped":
            row.update(kind=None, status="unmapped", reason="the book file marks this line unmapped", n_legs=0)
        else:
            where = dict(base, **_place(m))
            if "race_id" in m and "event" not in m:
                where.pop("event", None)
            tables = m["legs"] if m["kind"] == S.COMBO else [m]
            for j, t in enumerate(tables):
                leg = _map_leg(db, t, where, aliases)
                leg.update(line=i + 1, leg=f"{i + 1}.{j + 1}" if m["kind"] == S.COMBO else str(i + 1))
                legs.append(leg)
            bad = [lg for lg in legs if lg["status"] != "mapped"]
            row.update(kind=m["kind"], n_legs=len(legs),
                       status="unmapped" if bad else "mapped",
                       reason="; ".join(f"leg {lg['leg']}: {lg['reason']}" for lg in bad) or None,
                       races=sorted({lg["race_id"] for lg in legs if lg["race_id"] is not None}))
        row["legs"] = legs
        lines.append(row)
    return dict(book=dict(venue=b["venue"], event=b["event"], sport=b.get("sport"), captured_utc=b["captured_utc"],
                          odds=b["odds"], currency=b["currency"]),
                lines=lines, unmapped=[r["id"] for r in lines if r["status"] != "mapped"])


# --- price ---------------------------------------------------------------------------------------------------------

def _params(leg):
    p = {}
    if leg["opponent_id"] is not None:
        p["opponent_id"] = leg["opponent_id"]
    if leg["team"] is not None:
        p["team"] = leg["team"]
    if leg["threshold"] is not None:
        p["line"] = leg["threshold"]
    return p


def _flip(p, side):
    return None if p is None else (1.0 - p if COMPLEMENT.get(side, False) else p)


def _run_for(db, leg, runs):
    """(run_id, source, stale) the model's price comes from: a run passed in for this leg's competition, else the app's own
    choice for the race (markets/venues.pricing_run: the live stage or forecast before the race, the latest as-of run
    made before it once it has run). stale: pricing_run's verdict, or None (a --run is taken as given)."""
    from racinglines.markets.venues import pricing_run, race_info
    for rid, comp in (runs or {}).items():
        if comp == leg["competition"]:
            return rid, "--run", None
    ck = ("run", leg["race_id"])
    if ck not in db.cache and getattr(db, "asof", None) is not None:
        db.cache[ck] = (*_asof_run(db, leg["race_id"]), None)
    if ck not in db.cache:
        r = pricing_run(db.conn, race_info(db.conn, leg["race_id"]))
        db.cache[ck] = (r["run_id"], r["source"], r.get("stale"))
    return db.cache[ck]


def _asof_run(db, race_id):
    """(run_id, source) of the race's newest diagnostic or forecast run whose data cutoff (params.cutoff, else when it
    was made) is at or before db.asof: the model's price as it stood then. A run made later but cut off before asof
    (a stage re-priced after the fact) counts: it saw nothing after its cutoff."""
    df = db.q("""SELECT mr.id, coalesce((mr.params->>'cutoff')::timestamp, mr.created_at::timestamp) AS cutoff
                 FROM model_runs mr WHERE mr.kind IN ('diagnostic', 'forecast')
                   AND EXISTS (SELECT 1 FROM race_predictions rp WHERE rp.model_run_id = mr.id AND rp.race_id = :r)
                   AND coalesce((mr.params->>'cutoff')::timestamp, mr.created_at::timestamp) <= :t
                 ORDER BY cutoff DESC, mr.created_at DESC LIMIT 1""",
              r=race_id, t=db.asof.tz_localize(None).to_pydatetime())
    if not len(df):
        return None, f"no run cut off by {db.asof:%Y-%m-%dT%H:%MZ}"
    return int(df["id"].iloc[0]), f"as of {pd.Timestamp(df['cutoff'].iloc[0]):%Y-%m-%dT%H:%MZ}"


def _model(db, leg, runs):
    from racinglines.db.reads import model_prob
    if leg["kind"] not in K.KINDS:
        return None, None, None, "a race prop: no stored model price (models/props.py prices it live)"
    run_id, source, stale = _run_for(db, leg, runs)
    if run_id is None:
        return None, None, source, "no model run prices this race"
    if stale:              # a session ended and the model hasn't re-run on it: no price rather than an old one
        return None, run_id, source, stale["why"]
    comp = db.q("SELECT competition_id, category_id FROM races ra JOIN events e ON e.id = ra.event_id "
                "JOIN seasons s ON s.id = e.season_id WHERE ra.id = :r", r=leg["race_id"]).iloc[0]
    params = dict(_params(leg), event_key=f"{leg['season']}-{(leg['round'] or 0):02d}")
    link = dict(prediction=leg["kind"], competition_id=int(comp["competition_id"]), category_id=int(comp["category_id"]),
                athlete_id=leg["athlete_id"], race_id=leg["race_id"], params=params, invert=False)
    p, rid = model_prob(db.conn, link, db.cache.setdefault("model_prob", {}), run_id=run_id)
    note = None if p is not None else f"run {rid} stores no {leg['kind']} price for this selection"
    return _flip(p, leg["side"]), rid, source, note


def _quote_prob(link, tape):
    """A link's price of its own YES token: the bid/ask midpoint, else its last price, else the tape's last price."""
    bid, ask, last = (None if pd.isna(link[c]) else float(link[c]) for c in ("last_bid", "last_ask", "last_price"))
    p = (bid + ask) / 2 if bid is not None and ask is not None else last if last is not None else tape.get(link["token_id"])
    return p, bid, ask


def _market(db, leg):
    """(chosen quote or None, [every linked exchange's quote]) for the leg's YES, sides applied."""
    from racinglines.markets import store
    df = db.q("""SELECT id, exchange, token_id, outcome, athlete_id, params, invert, last_bid, last_ask, last_price,
                        synced_at, closed FROM market_links
                 WHERE race_id = :r AND prediction = :k AND lower(outcome) <> 'no' ORDER BY id""",
              r=leg["race_id"], k=leg["kind"])
    quotes = []
    for link in df.to_dict("records"):
        p = link["params"] or {}
        aid = None if pd.isna(link["athlete_id"]) else int(link["athlete_id"])
        opp = p.get("opponent_id")
        flip = False
        if leg["kind"] in ("race_h2h", "race_sprint_h2h"):
            if (aid, opp) == (leg["athlete_id"], leg["opponent_id"]):
                pass
            elif (aid, opp) == (leg["opponent_id"], leg["athlete_id"]):
                flip = True                  # the same pair the other way round: P(ours) = 1 - P(theirs)
            else:
                continue
        elif aid != leg["athlete_id"] or p.get("team") != leg["team"] or \
                (leg["threshold"] is not None and p.get("line") != leg["threshold"]):
            continue
        if db.asof is not None:             # the tape's last trade price then: the link's bid/ask are today's
            px, bid, ask = store.last_before(db.conn, [link["token_id"]], db.asof).get(link["token_id"]), None, None
            link = dict(link, synced_at=db.asof)
        else:
            tape = store.last_before(db.conn, [link["token_id"]], datetime.now(timezone.utc))
            px, bid, ask = _quote_prob(link, tape)
        if px is None:
            continue
        if link["invert"] or flip:
            px, bid, ask = 1 - px, (None if ask is None else 1 - ask), (None if bid is None else 1 - bid)
        if COMPLEMENT.get(leg["side"], False):
            px, bid, ask = 1 - px, (None if ask is None else 1 - ask), (None if bid is None else 1 - bid)
        synced = None if pd.isna(link["synced_at"]) else pd.Timestamp(link["synced_at"])
        quotes.append(dict(exchange=link["exchange"], link_id=int(link["id"]), token_id=link["token_id"],
                           prob=_r(px), bid=_r(bid), ask=_r(ask), closed=bool(link["closed"]),
                           synced_utc=None if synced is None else synced.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ")))
    if not quotes:
        return None, []
    best = {}
    for q in quotes:                         # one quote per exchange: its freshest link
        cur = best.get(q["exchange"])
        if cur is None or (q["synced_utc"] or "") > (cur["synced_utc"] or ""):
            best[q["exchange"]] = q
    order = {e: i for i, e in enumerate(EXCHANGE_ORDER)}
    chosen = sorted(best.values(), key=lambda q: (q["synced_utc"] or "", -order.get(q["exchange"], 99)))[-1]
    return chosen, sorted(best.values(), key=lambda q: order.get(q["exchange"], 99))


def _ev(p, dec):
    return None if p is None or dec is None else _r(p * dec - 1.0)


def _combo_leg(leg):
    """A mapped leg in markets/combos.py's form."""
    out = {"kind": leg["kind"]}
    if leg["opponent_id"] is not None:
        out["pair"] = [leg["athlete_id"], leg["opponent_id"]]
    elif leg["athlete_id"] is not None:
        out["athlete"] = leg["athlete_id"]
    if leg["team"] is not None:
        out["team"] = leg["team"]
    if leg["threshold"] is not None:
        out["line"] = leg["threshold"]
    out["side"] = leg["side"]
    return out


def _canon(legs):
    return sorted((lg["kind"], lg.get("athlete"), tuple(lg.get("pair") or ()), lg.get("team"), lg.get("line"),
                   lg.get("side") or "yes") for lg in legs)


def _stored_combo(db, group):
    """(fair, run_id, flags) from the newest stored combo run (Lab job f1_combo) whose combo has exactly these legs."""
    first = group[0]
    keys = [f"{first['season']}-{first['round']}", f"{first['season']}-{first['round']:02d}"] if first["round"] else []
    if not keys:
        return None
    df = db.q("""SELECT mr.id, mr.metrics FROM model_runs mr JOIN competitions co ON co.id = mr.competition_id
                 WHERE mr.kind = 'combo' AND co.code = :c AND mr.params->>'event_key' = ANY(:k)
                 ORDER BY mr.created_at DESC, mr.id DESC""", c=first["competition"], k=keys)
    want = _canon([_combo_leg(lg) for lg in group])
    for r in df.itertuples():
        for c in (r.metrics or {}).get("combos", []):
            if _canon(c.get("legs", [])) == want:
                flags = [] if c.get("calibrated", True) else [f"stored combo run {int(r.id)} not calibrated: "
                                                             + "; ".join(f.get("note", "") for f in c.get("flags", []))]
                return float(c["fair"]), int(r.id), flags
    return None


def _joint(db, group, sims):
    """(probability, method, flags) for legs on one race."""
    from racinglines.markets import combos as C
    probs = [lg["model_prob"] for lg in group]
    if len(group) == 1:
        return probs[0], "single", []
    s = (sims or {}).get(group[0]["race_id"])
    if s is not None:
        try:
            return C.combo_fair([_combo_leg(lg) for lg in group], s), "joint: simulations passed in", []
        except ValueError as e:
            fallback = [f"race {group[0]['race_id']}: the simulations can't price these legs jointly ({e})"]
    else:
        fallback = []
    hit = _stored_combo(db, group)
    if hit is not None:
        return hit[0], f"joint: stored combo run {hit[1]}", hit[2]
    p = None if any(x is None for x in probs) else float(pd.Series(probs).prod())
    return p, "product", fallback + [f"race {group[0]['race_id']}: {len(group)} legs on one race, {CORRELATED}"]


def price_book(conn, book, runs=None, sims=None, _db=None, asof=None):
    """map_book + per leg: book_prob (a leg's own odds), model_prob, market_prob, ev_model, ev_market; per line: the
    payout, book_prob, model_prob and market_prob of the whole bet, the method behind each, flags and EVs.
    runs: {run_id: competition code} to price from instead of the app's choice; sims: {race_id: OutcomeSims}.
    asof (UTC): price as it stood then, to follow a book through a weekend: the model from the newest run cut off by
    then (_asof_run), the market from each linked exchange's last tape price by then (no bid/ask)."""
    db = _db or _Db(conn, asof)
    out = map_book(conn, book, db)
    fmt = book["book"]["odds"]
    for row in out["lines"]:
        for leg in row["legs"]:
            leg.update(dict.fromkeys(PRICE_FIELDS), quotes=[])
            if leg["odds"] is not None:
                leg["book_prob"] = _r(S.to_prob(leg["odds"], fmt))
            if leg["status"] != "mapped":
                continue
            p, rid, src, note = _model(db, leg, runs)
            q, quotes = _market(db, leg)
            leg.update(run_id=rid, run_source=src, model_prob=_r(p), model_note=note, quotes=quotes)
            if q:
                leg.update(market_prob=q["prob"], market_exchange=q["exchange"], market_bid=q["bid"],
                           market_ask=q["ask"], market_synced_utc=q["synced_utc"])
            if leg["odds"] is not None:
                dec = decimal_odds(leg["odds"], fmt)
                leg.update(ev_model=_ev(leg["model_prob"], dec), ev_market=_ev(leg["market_prob"], dec))
        if row["status"] != "mapped":
            continue
        legs = row["legs"]
        dec = decimal_odds(row["odds"], fmt) if row["odds"] is not None else \
            float(pd.Series([decimal_odds(lg["odds"], fmt) for lg in legs]).prod())
        groups = {}
        for lg in legs:
            groups.setdefault(lg["race_id"], []).append(lg)
        flags, methods, pm = [], [], 1.0
        for g in groups.values():
            p, how, f = _joint(db, g, sims)
            flags += f
            methods.append(how)
            pm = None if p is None or pm is None else pm * p
        mk = [lg["market_prob"] for lg in legs]
        pk = None if any(x is None for x in mk) else float(pd.Series(mk).prod())
        same = any(len(g) > 1 for g in groups.values())
        if same and pk is not None:
            flags.append(f"market: product of leg prices on one race, {CORRELATED}")
        if len(groups) > 1:
            flags.append(f"legs on {len(groups)} races: independent across races (product)")
        book_p = 1.0 / dec
        row.update(decimal_odds=_r(dec, 4), book_prob=_r(book_p), model_prob=_r(pm), market_prob=_r(pk),
                   model_method=" x ".join(methods) if len(methods) > 1 else methods[0],
                   market_method=None if pk is None else ("single" if len(legs) == 1 else "product"),
                   flags=flags, edge_model=None if pm is None else _r(pm - book_p),
                   edge_market=None if pk is None else _r(pk - book_p), ev_model=_ev(pm, dec), ev_market=_ev(pk, dec))
    out.update(assumptions=list(ASSUMPTIONS), blend="none (owner, 2026-10-07; docs/sportsbook/slips.md)")
    if db.asof is not None:
        out.update(asof=db.asof.strftime("%Y-%m-%dT%H:%MZ"))
        out["assumptions"].append(f"As of {out['asof']}: the model's newest run cut off by then, and each exchange's "
                                  "last tape price by then (no bid/ask).")
    return out


# --- settle --------------------------------------------------------------------------------------------------------

def _leg_result(db, leg):
    """won / lost / void (the classification can't decide it) / manual (the results don't record it) / pending (no
    classification stored yet), by the kind's own rule (kinds.settle through private_book.outcome_for)."""
    from racinglines.markets import private_book as PB
    k = K.KINDS.get(leg["kind"])
    if k is None or (k.spec is None and k.payoff == "indicator"):
        return "manual"
    if leg["race_id"] not in db.results:
        db.results[leg["race_id"]] = PB.race_outcomes(db.conn, leg["race_id"])
    res = db.results[leg["race_id"]]
    if res.empty:
        return "pending"
    y = PB.outcome_for(leg["kind"], leg["athlete_id"], _params(leg), res)
    if y is None:
        return "void"
    return "won" if bool(y) != COMPLEMENT.get(leg["side"], False) else "lost"


def settle_book(conn, book, _db=None):
    """map_book + per leg: result; per line: result (won / lost / void / manual / pending) and the payout and profit
    per unit staked (a void returns the stake). A slip: any losing leg loses it; a void leg voids it ("void_all", the
    default) or is dropped ("drop_leg": the rest decide, the payout re-priced from the remaining legs' odds when the
    legs carry odds, else left to the book)."""
    db = _db or _Db(conn)
    out = map_book(conn, book, db)
    fmt = book["book"]["odds"]
    for row, ln in zip(out["lines"], book["lines"]):
        for leg in row["legs"]:
            leg["result"] = _leg_result(db, leg) if leg["status"] == "mapped" else None
        if row["status"] != "mapped":
            continue
        res = [lg["result"] for lg in row["legs"]]
        void_leg = ln["market"].get("void_leg", "void_all") if row["kind"] == S.COMBO else "void_all"
        dec = decimal_odds(row["odds"], fmt) if row["odds"] is not None else \
            float(pd.Series([decimal_odds(lg["odds"], fmt) for lg in row["legs"]]).prod())
        payout = None
        if "lost" in res:
            result, payout = "lost", 0.0
        elif "pending" in res:
            result = "pending"
        elif "manual" in res:
            result = "manual"
        elif "void" in res and (void_leg == "void_all" or all(r == "void" for r in res)):
            result, payout = "void", 1.0
        elif "void" in res:
            result = "won"
            kept = [lg for lg in row["legs"] if lg["result"] == "won"]
            payout = float(pd.Series([decimal_odds(lg["odds"], fmt) for lg in kept]).prod()) \
                if all(lg["odds"] is not None for lg in kept) else None
        else:
            result, payout = "won", dec
        row.update(decimal_odds=_r(dec, 4), result=result, payout=_r(payout, 4), profit=None if payout is None else _r(payout - 1.0, 4))
    return out


# --- tables --------------------------------------------------------------------------------------------------------

def leg_table(out):
    rows = [{k: v for k, v in lg.items() if k != "quotes"} for r in out["lines"] for lg in r["legs"]]
    df = pd.DataFrame(rows)
    for c in ("race_id", "athlete_id", "opponent_id", "run_id", "season", "round"):
        if c in df:
            df[c] = df[c].astype("Int64")
    return df


def line_table(out):
    return pd.DataFrame([{k: (", ".join(map(str, v)) if isinstance(v, list) else v) for k, v in r.items() if k != "legs"}
                         for r in out["lines"]])
