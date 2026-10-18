"""
Which driver and race a market is about, for the sports whose schema names a resolver ([identity] resolver in
sports/<code>.toml). A sync builds its link rows, hands them to `linker(sport, conn).fill(rows)`, and stores what comes
back: the same answer whichever exchange the rows came from, and the same one the database pass gives the links
already stored (`racinglines data link <sport>`). A sport with no resolver gets None and nothing changes.

    L = linker("nascar", conn)
    if L:
        L.fill(rows)            # sets athlete_id / race_id / params on each row; never raises
        promote("nascar", rows)  # prediction = params.kind where the sport's schema files that kind and it is identified
        stats["identity"] = dict(L.counts)

`promote` is the schema-driven classification of a sport the title classifier doesn't know: a link takes its
params.kind as its prediction when the kind is a registry kind (markets/kinds.toml), the sport's [markets] kinds list it,
and the link names everything the kind needs (identified below). Anything else stays `unmodeled`, its params.kind kept.
"""

import importlib

from racinglines import sports


def linker(sport, conn):
    name = sports.identity(sport)
    if not name:
        return None
    return importlib.import_module(f"racinglines.sources.{name}.links").Linker(conn)


def identified(kind, row):
    """Whether a link row names everything `kind` needs: its subject (a driver: athlete_id; a pair: athlete_id and
    params.opponent_id; a team: params.team) and, for a race market, race_id."""
    from racinglines.markets import kinds as K
    k = K.KINDS.get(kind)
    if k is None:
        return False
    p = row.get("params") or {}
    who = {"driver": row.get("athlete_id") is not None,
           "pair": row.get("athlete_id") is not None and p.get("opponent_id") is not None,
           "team": p.get("team") is not None, "field": True}[k.subject]
    return who and (k.payoff == "standings" or row.get("race_id") is not None)


def promote(sport, rows):
    """Set prediction = params.kind on each row whose kind the sport's schema files ([markets] kinds) and that is
    identified; the others are left as they are. Returns how many rows were promoted."""
    kinds = set(sports.link_kinds(sport))
    n = 0
    for row in rows:
        kind = (row.get("params") or {}).get("kind")
        if kind in kinds and identified(kind, row):
            row["prediction"] = kind
            n += 1
    return n
