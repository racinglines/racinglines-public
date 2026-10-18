"""
House markets: YES/NO markets we quote ourselves for private bets.

Prices are what the counterparty pays per $1 of payout:
    yes_price = fair + spread / 2          (they buy YES from us)
    no_price  = (1 - fair) + spread / 2    (they buy NO from us)
rounded UP to the cent, so rounding only ever widens our margin. A side isn't
offered if its price would be above MAX_PRICE (0.99): at that point the payout
is too small to be worth quoting.

Settlement:
    race_win / race_podium   Final results in the database (official ChronoRace).
    race_make_final          The athlete appears on the Final's start list or results.
    rank_up / rank_down      Manual, from the OFFICIAL UCI standings (our points
                             tables are placeholders, so the model's rank is only a hint).
"""

import math
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select, text

from racinglines.db import models as m

MAX_PRICE = 0.99   # cents-based book: 0.99 still leaves the other side a 0.01+ quote
MIN_PRICE = 0.01

KINDS = {
    "race_win": "wins the {venue} {cat} Final",
    "race_podium": "finishes on the podium (top 3) in the {venue} {cat} Final",
    "race_make_final": "makes the {venue} {cat} Final",
    "rank_up": "is ranked higher than #{rank} in the {cat} championship after {venue}",
    "rank_down": "is ranked lower than #{rank} in the {cat} championship after {venue}",
}
# F1 (and other circuit racing) wording: races are Grands Prix, there's no "Final",
# and the points-paying top 10 is a natural market
KINDS_F1 = {
    "race_win": "wins the {venue} Grand Prix",
    "race_podium": "finishes on the podium (top 3) at the {venue} Grand Prix",
    "race_top10": "finishes in the points (top 10) at the {venue} Grand Prix",
    "rank_up": KINDS["rank_up"],
    "rank_down": KINDS["rank_down"],
}
KINDS["race_top10"] = "finishes in the top 10 in the {venue} {cat} Final"
PROB_FIELD = {"race_win": "win_prob", "race_podium": "podium_prob", "race_make_final": "make_final_prob",
              "race_top10": "top10_prob",
              "rank_up": "rank_up_prob", "rank_down": "rank_down_prob"}
AUTO_SETTLE = ("race_win", "race_podium", "race_top10", "race_make_final")
FINAL_ROUNDS = ("final", "race")   # the deciding round: DH "final", F1 "race"


def quote(fair, spread):
    """(yes_price, no_price); None for a side that isn't offered."""
    def side(p):
        price = math.ceil(round((p + spread / 2) * 100, 6)) / 100
        return price if MIN_PRICE <= price <= MAX_PRICE else None
    return side(fair), side(1 - fair)


def latest_race_predictions(conn, race_id):
    """Latest forecast run's per-athlete predictions for a race (incl. rank-move fields)."""
    df = pd.read_sql(text("""
        SELECT rp.model_run_id, rp.athlete_id, a.display_name AS athlete, rp.win_prob, rp.podium_prob,
               rp.top10_prob, rp.make_final_prob, rp.extra
        FROM race_predictions rp JOIN athletes a ON a.id = rp.athlete_id
        WHERE rp.race_id = :race AND rp.model_run_id = (
            SELECT max(model_run_id) FROM race_predictions rp2 JOIN model_runs mr ON mr.id = rp2.model_run_id
            WHERE rp2.race_id = :race AND mr.kind = 'forecast')"""), conn, params=dict(race=race_id))
    if len(df):
        extra = pd.json_normalize(df.pop("extra").fillna({}).tolist())
        for c in ("rank_up_prob", "rank_down_prob", "current_rank", "exp_rank_after"):
            df[c] = extra[c].to_numpy() if c in extra else None
    return df


def race_label(session, race_id):
    race = session.get(m.Race, race_id)
    venue = race.event.venue.name if race.event.venue else race.event.name
    return venue, race.category.code


def generate(session, conn, race_id, kinds, top_n, spread, maker_id=None):
    """Create markets for the top_n athletes (by win probability for race kinds,
    by current championship rank for rank kinds) and reprice open markets that
    still use the model price. Returns (created, repriced)."""
    preds = latest_race_predictions(conn, race_id)
    if preds.empty:
        raise ValueError("no forecast predictions for this race; run racinglines model forecast mtb_dh --db --save")
    venue, cat = race_label(session, race_id)
    is_f1 = (session.get(m.Race, race_id).format or {}).get("kind") == "f1"
    created = repriced = 0
    for kind in kinds:
        if kind.startswith("rank"):
            pool = preds.dropna(subset=["current_rank"]).sort_values("current_rank").head(top_n)
        else:
            pool = preds.sort_values("win_prob", ascending=False).head(top_n)
        for r in pool.itertuples():
            fair = getattr(r, PROB_FIELD[kind])
            if fair is None or pd.isna(fair):
                continue
            fair = float(fair)
            ctx = {"current_rank": int(r.current_rank)} if kind.startswith("rank") and pd.notna(r.current_rank) else None
            mk = session.scalars(select(m.HouseMarket).filter_by(race_id=race_id, athlete_id=int(r.athlete_id),
                                                                  kind=kind, maker_id=maker_id)).first()
            if mk is None:
                kinds = KINDS_F1 if is_f1 else KINDS
                if kind not in kinds:
                    continue
                title = f"{r.athlete} " + kinds[kind].format(venue=venue, cat=cat, rank=(ctx or {}).get("current_rank"))
                yes, no = quote(fair, spread)
                session.add(m.HouseMarket(race_id=race_id, athlete_id=int(r.athlete_id), kind=kind, title=title,
                                          model_run_id=int(r.model_run_id), fair_prob=fair, spread=spread,
                                          yes_price=yes, no_price=no, context=ctx, maker_id=maker_id))
                created += 1
            elif mk.status == "open" and mk.fair_source == "model":
                mk.fair_prob, mk.spread, mk.model_run_id = fair, spread, int(r.model_run_id)
                mk.yes_price, mk.no_price = quote(fair, spread)
                repriced += 1
    session.commit()
    return created, repriced


def set_price(session, market_id, fair, spread):
    mk = session.get(m.HouseMarket, market_id)
    mk.fair_prob, mk.spread, mk.fair_source = fair, spread, "manual"
    mk.yes_price, mk.no_price = quote(fair, spread)
    session.commit()


def record_bet(session, market_id, counterparty, side, stake, price=None, note=None, taker_id=None):
    mk = session.get(m.HouseMarket, market_id)
    if mk.status != "open":
        raise ValueError(f"market is {mk.status}")
    quoted = mk.yes_price if side == "YES" else mk.no_price
    price = price or quoted
    if price is None:
        raise ValueError(f"{side} is not offered on this market")
    if not (0 < price < 1) or stake <= 0:
        raise ValueError("price must be between 0 and 1 and stake positive")
    if taker_id is not None and taker_id == mk.maker_id:
        raise ValueError("makers can't bet against their own market")
    bet = m.HouseBet(market_id=market_id, counterparty=counterparty.strip(), side=side, price=price, stake=stake,
                     payout=round(stake / price, 2), note=note or None, taker_id=taker_id)
    session.add(bet)
    session.commit()
    return bet


def settle(session, market_id, outcome, note):
    """outcome: True (YES happened), False, or None to void (all stakes returned)."""
    mk = session.get(m.HouseMarket, market_id)
    mk.outcome = outcome
    mk.status = "void" if outcome is None else "settled"
    mk.settled_at = datetime.now(timezone.utc)
    mk.settle_note = note
    for bet in session.scalars(select(m.HouseBet).filter_by(market_id=market_id)):
        if outcome is None:
            bet.status = "void"
        else:
            bet.status = "won" if (bet.side == "YES") == outcome else "lost"
    session.commit()


def auto_settle(session, conn, race_id):
    """Settle open race_* markets for a race from its Final. Returns [(market_id, outcome, note)]."""
    fin = pd.read_sql(text("""
        SELECT r.athlete_id, r.position, r.status FROM results r JOIN rounds ro ON ro.id = r.round_id
        WHERE ro.race_id = :race AND ro.kind IN ('final', 'race')"""), conn, params=dict(race=race_id))
    finished = fin[fin["status"] == "OK"]
    done = []
    for mk in session.scalars(select(m.HouseMarket).where(
            m.HouseMarket.race_id == race_id, m.HouseMarket.status.in_(["open", "closed"]),
            m.HouseMarket.kind.in_(AUTO_SETTLE))).all():
        row = fin[fin["athlete_id"] == mk.athlete_id]
        if mk.kind == "race_make_final":
            if len(fin) < 20:          # Final start list not published yet
                continue
            outcome, note = len(row) > 0, "Final start list/results"
        else:
            if len(finished) < 10:     # Final not raced yet
                continue
            classified = len(row) and row["status"].iloc[0] == "OK"
            pos = row["position"].iloc[0] if classified and pd.notna(row["position"].iloc[0]) else None
            limit = {"race_win": 1, "race_podium": 3, "race_top10": 10}[mk.kind]
            outcome = pos is not None and pos <= limit
            note = f"Final position {int(pos)}" if pos is not None else "not classified in the Final"
        settle(session, mk.id, outcome, f"auto: {note}")
        done.append((mk.id, outcome, note))
    return done


ALL = object()   # book(maker_id=ALL): every maker's markets


def book(conn, race_id=None, maker_id=ALL, status=None):
    """Markets with bet totals and P&L if YES / if NO (the maker's side), plus EV at fair.
    maker_id: ALL (default), None (house/legacy markets) or a user id."""
    df = pd.read_sql(text("""
        SELECT hm.*, coalesce(a.display_name, hm.params->>'team', '') AS athlete, coalesce(u.username, 'house') AS maker,
               ml.last_price AS pm_price, ml.last_bid AS pm_bid, ml.last_ask AS pm_ask, ml.event_slug AS pm_slug,
               ml.event_title AS pm_event, ml.volume AS pm_volume,
               count(hb.id) FILTER (WHERE hb.status <> 'void') AS bets,
               coalesce(sum(hb.stake) FILTER (WHERE hb.status <> 'void'), 0) AS staked,
               coalesce(sum(CASE WHEN hb.side = 'YES' THEN hb.stake - hb.payout ELSE hb.stake END)
                        FILTER (WHERE hb.status <> 'void'), 0) AS pnl_if_yes,
               coalesce(sum(CASE WHEN hb.side = 'NO' THEN hb.stake - hb.payout ELSE hb.stake END)
                        FILTER (WHERE hb.status <> 'void'), 0) AS pnl_if_no
        FROM house_markets hm LEFT JOIN athletes a ON a.id = hm.athlete_id
        LEFT JOIN users u ON u.id = hm.maker_id LEFT JOIN market_links ml ON ml.id = hm.market_link_id
        LEFT JOIN house_bets hb ON hb.market_id = hm.id
        WHERE (CAST(:race AS int) IS NULL OR hm.race_id = CAST(:race AS int)
               OR (CAST(:race AS int) = 0 AND hm.race_id IS NULL))
          AND (CAST(:all_makers AS boolean) OR hm.maker_id IS NOT DISTINCT FROM CAST(:maker AS int))
          AND (CAST(:status AS text) IS NULL OR hm.status = CAST(:status AS text))
        GROUP BY hm.id, a.display_name, u.username, ml.id
        ORDER BY hm.kind, hm.fair_prob DESC"""), conn,
        params=dict(race=race_id, all_makers=maker_id is ALL, maker=None if maker_id is ALL else maker_id,
                    status=status))
    df["ev"] = df["fair_prob"] * df["pnl_if_yes"] + (1 - df["fair_prob"]) * df["pnl_if_no"]
    df["worst"] = df[["pnl_if_yes", "pnl_if_no"]].min(axis=1)
    df["settled_pnl"] = [(r.pnl_if_yes if r.outcome else r.pnl_if_no) if r.status == "settled" else None
                         for r in df.itertuples()]
    return df


def bets(conn, market_id):
    return pd.read_sql(text("""SELECT hb.*, coalesce(u.username, hb.counterparty) AS taker FROM house_bets hb
                               LEFT JOIN users u ON u.id = hb.taker_id WHERE hb.market_id = :m ORDER BY hb.id"""),
                       conn, params=dict(m=market_id))


def taker_bets(conn, taker_id=None):
    """Bets with market info and the taker's P&L (won: payout - stake, lost: -stake).
    taker_id None = every bet placed by a taker account."""
    df = pd.read_sql(text("""
        SELECT hb.id, hb.created_at, hb.side, hb.price, hb.stake, hb.payout, hb.status, hb.market_id,
               hm.title, hm.kind, hm.race_id, hm.status AS market_status, coalesce(mu.username, 'house') AS maker,
               tu.username AS taker
        FROM house_bets hb JOIN house_markets hm ON hm.id = hb.market_id
        LEFT JOIN users mu ON mu.id = hm.maker_id LEFT JOIN users tu ON tu.id = hb.taker_id
        WHERE (CAST(:t AS int) IS NULL AND hb.taker_id IS NOT NULL) OR hb.taker_id = CAST(:t AS int)
        ORDER BY hb.id DESC"""), conn, params=dict(t=taker_id))
    df["pnl"] = [(r.payout - r.stake) if r.status == "won" else (-r.stake if r.status == "lost" else 0.0)
                 for r in df.itertuples()]
    return df


# ---------------------------------------------------------------------------
# Mirroring exchange markets (Polymarket) into a maker's book
# ---------------------------------------------------------------------------

def mirror_title(link):
    q = link["question"] or ""
    if link["prediction"] in ("race_h2h",):
        return f"{link['outcome']} finishes ahead — {q}"
    return q


def mirror_event(session, conn, event_slug, maker_id, spread, only_modeled=True, run_id=None, tag=None):
    """Create/refresh house markets for every modeled outcome of an exchange event,
    priced at our fair value +- spread/2 (from the latest forecast, or `run_id`).
    `tag` is stored in the market's params (e.g. {"diagnostic_run": 9}).
    Returns (created, repriced, skipped)."""
    from racinglines.db import reads as data
    links = pd.read_sql(text("SELECT * FROM market_links WHERE event_slug = :s AND NOT closed ORDER BY id"), conn,
                        params=dict(s=event_slug))
    # A binary Polymarket market (e.g. a head-to-head) has two outcome tokens; our
    # market already quotes both YES and NO, so mirror only the first token.
    # A Kalshi condition_id is the whole event (every driver's market), so Kalshi pairs by market ticker instead.
    mine = {r[0] for r in conn.execute(text("""
        SELECT CASE WHEN ml.exchange = 'kalshi' THEN ml.token_id ELSE ml.condition_id END
        FROM house_markets hm JOIN market_links ml ON ml.id = hm.market_link_id
        WHERE hm.maker_id = :mk AND ml.event_slug = :s"""), dict(mk=maker_id, s=event_slug))}
    cache, created, repriced, skipped, seen = {}, 0, 0, 0, set()
    for link in links.to_dict("records"):
        cond = link["token_id"] if link["exchange"] == "kalshi" else link["condition_id"]
        if cond and link["prediction"] == "race_h2h" and cond in seen:
            continue
        seen.add(cond)
        fair, run_id_used = data.model_prob(conn, link, cache, run_id=run_id)
        if fair is None:
            skipped += 1
            continue
        mk = session.scalars(select(m.HouseMarket).filter_by(market_link_id=int(link["id"]), maker_id=maker_id)).first()
        if mk is None and link["prediction"] == "race_h2h" and cond in mine:
            continue   # mirrored earlier through the other token
        yes, no = quote(fair, spread)
        race = None if pd.isna(link["race_id"]) else int(link["race_id"])
        athlete = None if pd.isna(link["athlete_id"]) else int(link["athlete_id"])
        if mk is None:
            params = dict(link["params"] or {}, **(tag or {})) if isinstance(link["params"], dict) or tag else link["params"]
            session.add(m.HouseMarket(race_id=race, athlete_id=athlete, kind=link["prediction"], title=mirror_title(link),
                                      model_run_id=run_id_used, fair_prob=fair, spread=spread, yes_price=yes, no_price=no,
                                      maker_id=maker_id, market_link_id=int(link["id"]), params=params))
            created += 1
        elif mk.status == "open" and mk.fair_source == "model":
            mk.fair_prob, mk.spread, mk.model_run_id, mk.yes_price, mk.no_price = fair, spread, run_id_used, yes, no
            repriced += 1
    session.commit()
    return created, repriced, skipped


def settle_from_exchange(session, conn):
    """Settle open/closed mirrored markets whose exchange market has resolved."""
    rows = conn.execute(text("""
        SELECT hm.id, ml.resolved_yes, ml.question FROM house_markets hm JOIN market_links ml ON ml.id = hm.market_link_id
        WHERE hm.status IN ('open', 'closed') AND ml.resolved_yes IS NOT NULL""")).all()
    for mid, yes, q in rows:
        settle(session, mid, bool(yes), f"auto: Polymarket resolved {'YES' if yes else 'NO'} ({q})")
    return [r[0] for r in rows]


def race_outcomes(conn, race_id):
    """Official race classification for settlement: athlete -> (position, classified, team_key), and `grid`, the
    starting grid slot the race round stores (results.extra.grid: F1's FastF1 GridPosition after penalties, 0 = pit
    lane; NASCAR's starting_position; NaN where none is stored), which settles the biggest mover (payoffs.biggest_mover)."""
    df = pd.read_sql(text("""
        SELECT r.athlete_id, r.position, r.status, coalesce(r.extra->>'team_id', r.team) AS team_id,
               coalesce((r.extra->>'points')::float, 0) AS points, nullif(r.extra->>'grid', '')::float AS grid
        FROM results r JOIN rounds ro ON ro.id = r.round_id
        WHERE ro.race_id = :r AND ro.kind IN ('race', 'final')"""), conn, params=dict(r=race_id))
    q = pd.read_sql(text("""SELECT r.athlete_id, r.position AS qual_position FROM results r JOIN rounds ro ON ro.id = r.round_id
                            WHERE ro.race_id = :r AND ro.kind = 'qual'"""), conn, params=dict(r=race_id))
    out = df.merge(q, on="athlete_id", how="left") if len(df) else df.assign(qual_position=None)
    if not len(out):
        return out
    from racinglines import sports
    from racinglines.markets import payoffs as P
    out = P.mark_sport(out, sports.race_sport(conn, race_id))     # who is classified: the sport's [results] rule
    return out.merge(stage_outcomes(conn, race_id), on="athlete_id", how="left")


def stage_outcomes(conn, race_id):
    """Per athlete, each side stage of the weekend (sports/f1.toml [sessions.sim], the sprint): <stage>_position,
    _status, _points, and <grid session>_position (sprint_qual_position) for the stage-pole kinds. Sprint
    Qualifying is stored without positions, so its order is the stage's starting grid (FastF1's GridPosition,
    results.extra.grid: after penalties, 0 = pit lane = none), known once the stage has run; an SQ position
    stored in the classification wins when there is one. Empty frame (just athlete_id) on a weekend without one."""
    from racinglines.models.inputs.laps import MAIN_STAGE, SIM_SESSIONS
    # the current-format grid session (grid_from); 2021's per-year source (grid_from_by_year: qual) is not read here,
    # no venue listed 2021 sprint markets, so none are settled by this
    side = {k: v["grid_from"] for k, v in SIM_SESSIONS.items() if k != MAIN_STAGE}
    rows = pd.read_sql(text("""
        SELECT r.athlete_id, ro.kind, r.position, r.status, coalesce((r.extra->>'points')::float, 0) AS points,
               nullif(nullif(r.extra->>'grid', '')::float, 0) AS grid
        FROM results r JOIN rounds ro ON ro.id = r.round_id
        WHERE ro.race_id = :r AND ro.kind = ANY(:k)"""), conn,
        params=dict(r=race_id, k=sorted(set(side) | set(side.values()))))
    out = pd.DataFrame(dict(athlete_id=pd.Series(dtype="int64")))
    for stage, grid_from in side.items():
        st = rows[rows["kind"] == stage].drop_duplicates("athlete_id").set_index("athlete_id")
        gq = rows[rows["kind"] == grid_from].drop_duplicates("athlete_id").set_index("athlete_id")["position"]
        if st.empty and gq.notna().sum() == 0:
            continue
        idx = st.index.union(gq.index)
        f = pd.DataFrame({f"{stage}_position": st["position"].reindex(idx), f"{stage}_status": st["status"].reindex(idx),
                          f"{stage}_points": st["points"].reindex(idx),
                          f"{grid_from}_position": gq.reindex(idx).combine_first(st["grid"].reindex(idx))}, index=idx)
        out = out.merge(f.rename_axis("athlete_id").reset_index(), on="athlete_id", how="outer")
    return out


def outcome_for(kind, athlete_id, params, res):
    """YES/NO for a race market from the official classification (None if undecidable): the kind's
    settlement in racinglines/markets/kinds.py."""
    from racinglines.markets import kinds as K
    from racinglines.models.inputs.laps import team_key
    k = K.KINDS.get(kind)
    # a team market: a kind whose subject is a team (markets/kinds.toml: a team spec, or the legacy group_top)
    grouped = k is not None and k.subject == "team"
    return K.settle(kind, athlete_id, params, res, group_key=team_key if grouped else None)


def settle_from_results(session, conn, race_id, market_ids=None, rules=None):
    """Settle open race markets (incl. head-to-head and constructor-top) from the
    official classification. Returns settled market ids.
    rules: settle a cancelled race by each venue's rules (markets/settlement_rules.py; a mirrored market's
    venue is its exchange, the rest are the private book's). None reads RACINGLINES_CANCELLED_RACE_RULES,
    off by default: then a cancelled race never settles here and a relocated one settles on its result."""
    from racinglines.markets import settlement_rules as SR
    if SR.enabled(rules) and SR.race_status(_sport(session, race_id), _event_key(session, race_id),
                                            SR.db_status(conn, race_id)) == SR.CANCELLED:
        return _settle_cancelled(session, conn, race_id, market_ids)
    res = race_outcomes(conn, race_id)
    if res.empty or (res["status"] == "OK").sum() < 5:
        return []
    q = select(m.HouseMarket).where(m.HouseMarket.race_id == race_id, m.HouseMarket.status.in_(["open", "closed"]))
    if market_ids is not None:
        q = q.where(m.HouseMarket.id.in_(market_ids))
    done = []
    for mk in session.scalars(q).all():
        y = outcome_for(mk.kind, mk.athlete_id, mk.params, res)
        if y is None:
            continue
        settle(session, mk.id, y, "auto: official race classification")
        done.append(mk.id)
    return done


def _event_key(session, race_id):
    return session.get(m.Race, race_id).event.source_key


def _sport(session, race_id):
    race = session.get(m.Race, race_id)
    return "f1" if (race.format or {}).get("kind") == "f1" or race.event.source == "f1timing" else "mtb_dh"


def _settle_cancelled(session, conn, race_id, market_ids=None):
    """A cancelled race: every open market settles by its venue's rule (settlement_rules.RULES). A 0.5 payout
    (Polymarket's 50/50 on a head-to-head) or a price (Kalshi's fair-price settlement) can't be paid by a YES/NO
    book, so the market is voided with a note."""
    from racinglines.markets import settlement_rules as SR
    venue_of = dict(conn.execute(text("SELECT hm.id, ml.exchange FROM house_markets hm JOIN market_links ml "
                                      "ON ml.id = hm.market_link_id WHERE hm.race_id = :r"), dict(r=race_id)).all())
    name_of = dict(conn.execute(text("SELECT hm.id, coalesce(ml.group_title, ml.outcome) FROM house_markets hm "
                                     "JOIN market_links ml ON ml.id = hm.market_link_id WHERE hm.race_id = :r"),
                                dict(r=race_id)).all())
    q = select(m.HouseMarket).where(m.HouseMarket.race_id == race_id, m.HouseMarket.status.in_(["open", "closed"]))
    if market_ids is not None:
        q = q.where(m.HouseMarket.id.in_(market_ids))
    done = []
    for mk in session.scalars(q).all():
        venue = venue_of.get(mk.id) or "private"
        y = SR.payout(venue, SR.CANCELLED, mk.kind, name_of.get(mk.id))
        note = "auto: " + SR.describe(venue, SR.CANCELLED, mk.kind, name_of.get(mk.id))
        if y == SR.RESULT:
            continue
        if y == SR.VOID:
            y = None
        elif y == SR.FAIR:
            y, note = None, note + "; a YES/NO book can't settle at a price: voided"
        elif not isinstance(y, bool):
            y, note = None, note + f"; a YES/NO book can't pay {y}: voided"
        settle(session, mk.id, y, note)
        done.append(mk.id)
    return done
