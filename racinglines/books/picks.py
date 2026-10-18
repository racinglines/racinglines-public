"""
A strategy replay's entries as a book file: `racinglines strategy signals f1 --asof T --book OUT.toml`. Each buy the
replay made (pipelines/signals.compute: the taker's trades, stage by stage, up to T) becomes one [[lines]] entry on the
exchange it was made on, at the price it was made at, so `racinglines book price|settle|export` can follow the picks
like any other bets (docs/sportsbook/slips.md). Sells (trims and exits) are left out and counted in a comment.

Every market table names its athletes by id (driver_id, opponent_id) from the replayed market's own link
(market_links, by token): exact keys, never a name match. The odds are the venue's own presentation: Kalshi in cents,
Polymarket and OG.com in dollars, and the stake is what the entry cost (shares x price), in USD.
"""

import json
import tomllib

import pandas as pd
from sqlalchemy import text

from racinglines.books import schema as S
from racinglines.markets import kinds as K

ODDS = {"kalshi": "cents", "polymarket": "dollars", "og": "dollars"}


class PicksError(ValueError):
    pass


def _s(v):
    return json.dumps(v, ensure_ascii=False)


def _links(conn, tokens):
    if not tokens:
        return {}
    rows = conn.execute(text("""SELECT ml.token_id, ml.athlete_id, ml.params, ml.prediction, a.display_name
                                FROM market_links ml LEFT JOIN athletes a ON a.id = ml.athlete_id
                                WHERE ml.token_id = ANY(:t)"""), dict(t=list(tokens))).mappings().all()
    return {r["token_id"]: dict(r) for r in rows}


def _names(conn, ids):
    ids = [int(i) for i in ids if i is not None]
    if not ids:
        return {}
    rows = conn.execute(text("SELECT id, display_name FROM athletes WHERE id = ANY(:i)"), dict(i=ids)).all()
    return {int(i): n for i, n in rows}


def _market(sig, link, names):
    """The book's market table for one entry, or None when the link can't name its subject exactly."""
    kind = sig["kind"]
    k = K.KINDS.get(kind)
    if k is None:
        return None
    params = link.get("params") or {}
    m = {"kind": kind}
    if k.subject == "team":
        if not params.get("team"):
            return None
        m["team"] = params["team"]
    elif k.subject == "pair":
        opp = params.get("opponent_id")
        if link.get("athlete_id") is None or opp is None:
            return None
        m.update(driver=link["display_name"], driver_id=int(link["athlete_id"]),
                 opponent=names.get(int(opp), str(opp)), opponent_id=int(opp))
    elif k.subject == "driver":
        if link.get("athlete_id") is None:
            return None
        m.update(driver=link["display_name"], driver_id=int(link["athlete_id"]))
    else:
        return None
    if params.get("line") is not None:
        m["line"] = params["line"]
    if sig["side"] == "NO":
        m["side"] = "no"
    return m


def _odds(price, fmt):
    if fmt == "cents":
        return int(min(99, max(1, round(price * 100))))
    return round(min(0.999, max(0.001, float(price))), 3)


def book_toml(conn, out, asof, path_note=None):
    """The book file text (validated) for one compute() result `out`. -> (text, n_lines, n_left_out)."""
    w = out.get("event")
    if not w or not out.get("signals"):
        raise PicksError("the replay made no trades: nothing to write")
    venue = (out.get("venue") or next((s.get("detail", {}).get("venue") for s in out["signals"]
                                       if (s.get("detail") or {}).get("venue")), None) or "polymarket")
    fmt = ODDS.get(venue)
    if fmt is None:
        raise PicksError(f"no odds format for venue {venue!r}")
    buys = [s for s in out["signals"] if s["action"] == "buy" and s.get("limit_price") and s.get("shares")]
    links = _links(conn, {s["market_key"] for s in buys})
    names = _names(conn, {(lk.get("params") or {}).get("opponent_id") for lk in links.values()})
    captured = pd.Timestamp(asof, tz="UTC") if pd.Timestamp(asof).tzinfo is None else pd.Timestamp(asof).tz_convert("UTC")
    lines, skipped = [], []
    for s in buys:
        link = links.get(s["market_key"])
        m = _market(s, link, names) if link else None
        stake = round(float(s["shares"]) * float(s["limit_price"]), 2)
        if m is None or stake <= 0:
            skipped.append(f"{s['kind']} {s['subject']} ({'no exact link' if m is None else 'no stake'})")
            continue
        placed = pd.Timestamp(s["signal_ts"])
        placed = placed.tz_localize("UTC") if placed.tzinfo is None else placed.tz_convert("UTC")
        lines.append(dict(title=K.KINDS[s["kind"]].label or s["kind"], selection=s["subject"],
                          odds=_odds(s["limit_price"], fmt), stake=stake, placed=placed.strftime("%Y-%m-%dT%H:%MZ"),
                          stage=s["stage"], side=s["side"], market=m))
    if not lines:
        raise PicksError("no entry names its market exactly: nothing to write (" + "; ".join(skipped) + ")")
    sells = sum(1 for s in out["signals"] if s["action"] != "buy")
    prof = out["profile"]["name"]
    head = [f"# {prof}'s entries on {w['name']} ({venue}), from `racinglines strategy signals --asof "
            f"{captured:%Y-%m-%dT%H:%MZ} --book`" + (f" ({path_note})" if path_note else "") + ".",
            "# Paper picks at the price each was made at; the stake is what the entry cost (shares x price).",
            f"# Sells and trims left out: {sells}." + (" Not written (" + "; ".join(skipped) + ")." if skipped else "")]
    txt = head + ["", "[book]", f"venue = {_s(venue)}", f"event = {_s(w['event_key'])}", 'sport = "f1"',
                  f"captured_utc = {_s(captured.strftime('%Y-%m-%dT%H:%MZ'))}", 'source = "api"',
                  f"odds = {_s(fmt)}", 'currency = "USD"']
    for i, ln in enumerate(lines, 1):
        mk = ", ".join(f"{k} = {_s(v)}" for k, v in ln["market"].items())
        txt += ["", "[[lines]]", f'id = "{i}"', f"title = {_s(ln['title'])}",
                f"selection = {_s(ln['selection'] + (' (NO)' if ln['side'] == 'NO' else ''))}",
                f"odds = {ln['odds']}", f"stake = {ln['stake']}", f"placed_utc = {_s(ln['placed'])}",
                f"note = {_s('stage: ' + ln['stage'])}", f"market = {{ {mk} }}"]
    body = "\n".join(txt) + "\n"
    S.validate_book(tomllib.loads(body), "picks")
    return body, len(lines), len(skipped)
