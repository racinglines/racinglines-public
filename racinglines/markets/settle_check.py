"""
Settlement check, read-only: our outcome from the results beside each exchange's own resolution, per market link.

    df = check(conn, "og", "f1")       one row per resolved link of the sport on the exchange
    print(text_report(df, "og", "f1"))

For every closed link of the sport's competition on the exchange that the exchange has resolved (resolved_yes set;
a void recorded by `markets --exchange og settle` in params.settlement is listed as void), our outcome is the one
settlement every pipeline uses (private_book.outcome_for -> markets/kinds.settle, on private_book.race_outcomes of the
link's race). Each row is then:

    agree        ours == resolved_yes
    disagree     ours != resolved_yes: a settlement rule, a link or a result to look at
    undecided    the results can't settle it (no result yet, a kind the results don't decide such as the fastest lap,
                 a cancelled race, a biggest mover whose grid isn't stored, ...): `reason` says which
    unmodeled    the link has no kind, driver or race we can settle (prediction 'unmodeled' and no params.kind, ...)
    void         the exchange voided it (params.settlement = 'void')

"Settled counts match resolved_yes" (docs/parity-rebuild.md) means: no `disagree` row. Nothing is written. The kind is
the link's `prediction`, else params.kind (the NASCAR and MotoGP links keep their kind there). Any exchange:
`racinglines markets --exchange og settle --check`, `racinglines markets --exchange kalshi settle-check`, and the
Polymarket form `racinglines markets [--sport f1] settle-check`.
"""

import pandas as pd
from sqlalchemy import text

COLUMNS = ["link_id", "token_id", "race_id", "event_key", "kind", "athlete_id", "question", "outcome", "resolved_yes",
           "ours", "verdict", "reason"]


def links(conn, exchange, sport="f1"):
    """The sport's closed links on the exchange that are resolved (resolved_yes set) or voided (params.settlement)."""
    from racinglines import sports
    return pd.read_sql(text("""
        SELECT ml.id AS link_id, ml.token_id, ml.race_id, ml.athlete_id, ml.prediction, ml.params, ml.question,
               ml.outcome, ml.resolved_yes, e.source_key AS event_key, e.status AS event_status
        FROM market_links ml JOIN competitions co ON co.id = ml.competition_id
        LEFT JOIN races ra ON ra.id = ml.race_id LEFT JOIN events e ON e.id = ra.event_id
        WHERE ml.exchange = :x AND co.code = :c AND ml.closed
          AND (ml.resolved_yes IS NOT NULL OR ml.params->>'settlement' = 'void')
        ORDER BY ml.race_id NULLS LAST, ml.id"""), conn,
        params=dict(x=exchange, c=sports.load(sport)["competition"]["code"]))


def kind_of(link):
    """The link's market kind: `prediction` when it is a kind of markets/kinds.toml, else params.kind, else None."""
    from racinglines.markets import kinds as K
    p = link.get("params") or {}
    for k in (link.get("prediction"), p.get("kind")):
        if k in K.KINDS:
            return k
    return None


def _int(v):
    return None if v is None or pd.isna(v) else int(v)


def verdict(link, res, sport="f1"):
    """(ours, verdict, reason) for one link (a dict of links()) given its race's result frame (`res`, or None when the
    link names no race)."""
    from racinglines.markets import kinds as K
    from racinglines.markets import payoffs as P
    from racinglines.markets import private_book as house
    from racinglines.markets import settlement_rules as SR
    p = link.get("params") or {}
    if p.get("settlement") == "void" and link.get("resolved_yes") is None:
        return None, "void", "voided by the exchange"
    kind = kind_of(link)
    if kind is None:
        return None, "unmodeled", f"no kind (prediction {link.get('prediction')!r})"
    k = K.KINDS[kind]
    if res is None:
        return None, "unmodeled", "no race on the link"
    if SR.race_status(sport, link.get("event_key"), link.get("event_status")) == SR.CANCELLED:
        return None, "undecided", "cancelled race: the venue's cancelled-race rule decides (markets/settlement_rules.py)"
    ath = _int(link.get("athlete_id"))
    if ath is None and k.subject not in ("team", "field"):
        return None, "unmodeled", "no driver on the link"
    if res.empty:
        return None, "undecided", "no result stored for the race"
    ours = house.outcome_for(kind, ath, p, res)
    if ours is None:
        reason = "the results don't decide this kind"
        if k.payoff == "mover":
            reason = P.biggest_mover(res)[1]
        elif k.payoff == "h2h" and p.get("opponent_id") is None:
            reason = "no opponent on the link"
        return None, "undecided", reason
    return bool(ours), "agree" if bool(ours) == bool(link["resolved_yes"]) else "disagree", ""


def check(conn, exchange, sport="f1", results=None):
    """One row per resolved (or voided) link of `sport` on `exchange`: COLUMNS. results: race_id -> result frame
    (default private_book.race_outcomes, read once per race)."""
    from racinglines.markets import private_book as house
    df = links(conn, exchange, sport)
    results = results or (lambda rid: house.race_outcomes(conn, rid))
    cache, rows = {}, []
    for link in df.to_dict("records"):
        rid = _int(link.get("race_id"))
        if rid is not None and rid not in cache:
            cache[rid] = results(rid)
        ours, v, reason = verdict(link, cache.get(rid) if rid is not None else None, sport)
        ry = link.get("resolved_yes")
        rows.append(dict(link_id=int(link["link_id"]), token_id=link["token_id"], race_id=rid,
                         event_key=link.get("event_key"), kind=kind_of(link), athlete_id=_int(link.get("athlete_id")),
                         question=link.get("question"), outcome=link.get("outcome"),
                         resolved_yes=None if ry is None or pd.isna(ry) else bool(ry), ours=ours, verdict=v,
                         reason=reason))
    return pd.DataFrame(rows, columns=COLUMNS)


def text_report(df, exchange, sport="f1"):
    """Counts per verdict and per kind, then every disagreement."""
    out = [f"{exchange} {sport}: {len(df)} resolved links"]
    if not len(df):
        return out[0] + " (nothing to check)"
    counts = df["verdict"].value_counts()
    out.append("  " + "  ".join(f"{v} {int(counts.get(v, 0))}" for v in ("agree", "disagree", "undecided", "unmodeled",
                                                                            "void")))
    by = df[df["kind"].notna()].groupby(["kind", "verdict"]).size().unstack(fill_value=0)
    if len(by):
        out += ["", by.to_string()]
    und = df[df["verdict"] == "undecided"]["reason"].value_counts()
    if len(und):
        out += ["", "Undecided, by reason:"] + [f"  {n:5d}  {r}" for r, n in und.items()]
    bad = df[df["verdict"] == "disagree"]
    if len(bad):
        out += ["", f"Disagreements ({len(bad)}):",
                bad[["link_id", "token_id", "event_key", "kind", "athlete_id", "question", "resolved_yes",
                     "ours"]].to_string(index=False)]
    else:
        out += ["", "No disagreement: every settled outcome matches the exchange's resolution."]
    return "\n".join(out)
