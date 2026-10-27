"""
The MCP server: `build()` registers the tools of racinglines/mcp/tools.py and the docs as resources on an
MCPServer (the `mcp` package, 2.x); `serve()` runs it over stdio (the default: a chat client launches
`racinglines mcp` as a subprocess) or streamable HTTP (`--http`, for the VM behind a tunnel), where every request
must carry an account's bearer token (racinglines/mcp/auth.py: `racinglines mcp token <account>`).

Every read tool runs in its own READ ONLY transaction. The job tools (run_job, cancel_job) write what the Lab's
Run form writes: a `jobs` row; the job's subprocess saves a model run (forecasts as scenarios, never promoted).
fantasy_order writes what the Trade page writes (fantasy/exchange.place, the caller's own account only), and the
fantasy reads first pay out resolved hand trades as the Trade and History pages do (fantasy/exchange.settle).
fantasy_follow writes what /strategy/pick writes (fantasy/follow.follow: the profile and an audit row, the caller's own
account only).
"""

import importlib
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import text

from racinglines.mcp import tools as T
from racinglines.mcp import view as V

NAME = "racinglines"
DOCS = Path(__file__).resolve().parents[2] / "docs"
SPORTS = Path(__file__).resolve().parents[2] / "sports"
DEFAULT_PORT = 8100

INSTRUCTIONS = """racinglines: fair prices and market making for race sports (Formula 1 on Polymarket and Kalshi, UCI downhill),
priced strictly as of any moment and tested on real market tapes. This server reads the app's database and archive
and runs its simulations.

How to use it:
- overview() first: sports, seasons, the live forecast, what runs and markets exist.
- Data: list_events / get_event, search_athletes / get_athlete, list_markets (our fair value vs every venue per
  outcome), get_market_history (exchange prices, trades, books; resampled), list_model_runs / get_model_run /
  get_predictions, compare_runs (two runs paired by event, ±2 SE), get_forecast, edge_finder (sweep results per
  strategy), list_diagnostics / get_diagnostic,
  track_record / list_positions / list_signals (paper trading), list_live_events, data_changes.
- Fantasy trading (F$ only, your own account): fantasy_balance, fantasy_positions, fantasy_orders, fantasy_leaderboard,
  fantasy_markets, fantasy_book, and fantasy_order (one immediate-or-cancel order, as the Trade page places it).
  fantasy_strategy shows the strategy you follow and the five styles; fantasy_follow follows one (or stops), as the
  Strategy page does, never during a race weekend.
- Anything else: sql(query) — read-only SELECT, paged. describe_schema() lists the tables.
- Scenarios: list_job_types() then run_job(job_type, params) queues a Lab job (backtest, forecast scenario, event
  diagnostic, Edge Finder sweep, season strategy, downhill scenario/backtest); get_job(job_id) follows it and gives the
  model run it saved. replay_maker(run_id, knobs...) re-quotes a past event against the real tape in seconds.
- Every list is paged: default 50 rows, at most 500, and a byte cap; `total`, `truncated` and `next_offset` say
  what is left. Ask for what you need (columns, limit, offset) rather than everything.
- Probabilities are fractions (0.31 = 31%); money is USD; times are UTC. Fair values are the model's; the exchange
  columns are the venue's quotes at the time shown.
- Resources racinglines://docs/<page> hold the documentation (model.md, market-making.md, f1-forecast.md, ...).
"""
INSTRUCTIONS += """
Hosted Pro accounts can read shared race data and Lab research, but account records and jobs are limited to
their own account. sql, describe_schema and data_changes are admin-only. Demo accounts cannot sign in.
"""


_URL = None


def _engine():
    from racinglines.db.config import get_engine
    return get_engine(_URL)


def accounts_url():
    """RACINGLINES_MCP_ACCOUNTS_URL: the database the accounts, tokens and sign-ins are read from, when it isn't the
    served one (a second server over a shared research store, e.g. a demo store with no account rows of its own).
    None: the served database holds the accounts too (the default)."""
    return os.environ.get("RACINGLINES_MCP_ACCOUNTS_URL") or None


def _accounts_engine():
    from racinglines.db.config import get_engine
    return get_engine(accounts_url()) if accounts_url() else _engine()


def _owner():
    """Who a job is filed under and whose jobs a caller sees: the caller, except on a shared store (accounts elsewhere:
    no account rows here, so jobs are filed without an owner and every signed-in caller sees them)."""
    return (None, None) if accounts_url() else (caller_id(), caller())


def _sql_engine():
    """The sql tool's connection: RACINGLINES_MCP_SQL_URL when set (a login that can SELECT only the allowed tables,
    made by scripts/vm/mcp_sql_role.sh), else the app's own. Only the role keeps users and orders out of reach; the
    text check in tools.check_sql is a first filter."""
    url = os.environ.get("RACINGLINES_MCP_SQL_URL")
    if not url:
        return _engine()
    from racinglines.db.config import get_engine
    return get_engine(url)


def _json(obj):
    """One line of JSON (an indented dump would double the bytes for nothing), led by a drawn table of every list of
    records in it when there is one (racinglines/mcp/view.py): the table for a person reading the terminal, the JSON,
    always the last block, for the CLI and any client that parses it."""
    data = json.dumps(obj, default=str)
    try:
        table = V.render(obj)
    except Exception:  # noqa: BLE001  (a view never costs the caller the data)
        table = None
    return [table, data] if table else data


@contextmanager
def _sports_shown():
    """The tool call's hidden exchanges (tools.HIDDEN_VENUES, [exchange] audience) and its hidden sports (tools.HIDDEN): the ones sports.is_public turns away from a non-admin caller, none for an
    admin or over stdio. Every tool reads them, so a hidden sport is missing from lists and 'not available' by name."""
    who = caller()
    token, vtoken = T.HIDDEN.set(T.hidden_sports(who)), T.HIDDEN_VENUES.set(T.hidden_venues(who))
    try:
        yield
    finally:
        T.HIDDEN_VENUES.reset(vtoken)
        T.HIDDEN.reset(token)


def _read(fn, engine=None, **kw):
    """Call a tools.py function inside a READ ONLY transaction; anticipated errors become tool errors the client sees."""
    from mcp.server.mcpserver.exceptions import ToolError
    if fn in (T.sql, T.describe_schema, T.data_changes, T.storage_status):
        _require_admin()
    try:
        with _sports_shown(), (engine or _engine()).begin() as c:
            c.execute(text("SET TRANSACTION READ ONLY"))
            return _json(fn(c, **kw))
    except (ValueError, KeyError, TypeError) as ex:
        raise ToolError(str(ex)) from ex
    except Exception as ex:  # noqa: BLE001  (a database error: its first line is safe and useful)
        raise ToolError(f"{type(ex).__name__}: {str(ex).splitlines()[0][:400]}") from ex


def _plain(fn, **kw):
    from mcp.server.mcpserver.exceptions import ToolError
    try:
        with _sports_shown():
            return _json(fn(**kw))
    except (ValueError, KeyError, TypeError) as ex:
        raise ToolError(str(ex)) from ex


def _write(fn, **kw):
    """The job tools: they write a jobs row (what the Lab's Run form does), nothing else."""
    from mcp.server.mcpserver.exceptions import ToolError
    try:
        with _sports_shown():
            return _json(fn(_engine(), **kw))
    except (ValueError, KeyError, TypeError) as ex:
        raise ToolError(str(ex)) from ex


def _books():
    from racinglines.fantasy import exchange as FX
    return FX.books_engine(_engine())


def _fantasy_read(fn, **kw):
    """A fantasy read: resolved hand trades are paid out first (fantasy/exchange.settle, the same step the Trade and
    History pages run; it writes only accounts.fantasy_positions and the ledger), then the READ ONLY read, with the
    recorded books and resolutions from the read-only books engine."""
    import logging
    from racinglines.fantasy import exchange as FX
    from racinglines.fantasy import ledger as FL
    if FL.on():
        try:
            FX.settle(_engine(), _books())
        except Exception:  # noqa: BLE001  (never block a read on settlement)
            logging.getLogger(__name__).exception("fantasy settle failed")
    return _read(fn, books=_books(), **kw)


def _fantasy_write(fn=None, **kw):
    """fantasy_order (the default: one F$ order through fantasy/exchange.place) or fantasy_follow (fantasy/follow.follow),
    on the caller's own account."""
    from mcp.server.mcpserver.exceptions import ToolError
    try:
        return _json((fn or T.fantasy_order)(_engine(), viewer=_viewer(), **kw))
    except (ValueError, KeyError, TypeError) as ex:
        raise ToolError(str(ex)) from ex


def caller():
    """The account behind this request in --http mode (dict(id, username, role)), None over stdio."""
    from mcp.server.auth.middleware.auth_context import get_access_token
    t = get_access_token()
    if t is None or not t.subject:
        return None
    return dict(id=int(t.subject), username=(t.claims or {}).get("username"), role=(t.claims or {}).get("role"))


NO_ACCOUNT = -1                                  # matches no users.id (serial ids start at 1)


def _viewer():
    """The viewer the account-scoped reads use: the caller, except a non-admin caller on a shared store (accounts
    elsewhere): its id belongs to the accounts database, not this one, where the same number can be another account's
    rows (the demo store's maker or taker). There it reads as an account with no rows here, so overview, track_record,
    list_positions, list_signals and the maker quotes in list_markets show none of the store's account data."""
    c = caller()
    if c and accounts_url():
        from racinglines.web import roles as R
        if R.canonical(c.get("role")) != "admin":
            return dict(c, id=NO_ACCOUNT)
    return c


def caller_id():
    c = caller()
    return c["id"] if c else None


def _require_admin():
    from mcp.server.mcpserver.exceptions import ToolError
    from racinglines.web import roles as R
    who = caller()
    if who and R.canonical(who.get("role")) != "admin":
        raise ToolError("This MCP tool is admin-only; use the account-scoped and research tools instead")


def build(jobs_worker=False, engine_url=None, oauth=False):
    """The server with every tool and resource registered. `jobs_worker`: run the Lab's job worker in this process
    (queued jobs run here when no web app is running). `engine_url`: a database other than $DATABASE_URL. `oauth`:
    every request needs a token (racinglines/mcp/oauth.py: OAuth sign-in, or an account's `rl_` token)."""
    global _URL
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ResourceError

    _URL = engine_url
    kw = {}
    if oauth:
        from racinglines.mcp import oauth as O
        kw = dict(auth_server_provider=O.provider(_accounts_engine), auth=O.auth_settings())
    srv = MCPServer(NAME, instructions=INSTRUCTIONS, log_level="WARNING", **kw)
    if importlib.util.find_spec("racinglines.mcp.calllog"):     # not in the package (oss/manifest.toml)
        calllog = importlib.import_module("racinglines.mcp.calllog")
        jevscreen = importlib.import_module("racinglines.mcp.jevscreen")
        if (log_dir := calllog.folder(http=oauth)) is not None:
            # one JSON line per call, written off the request path; Jev screens each tool call (advisory unless
            # JEV_SCREEN_GATE_AT is set, fail-open either way)
            calllog.install(srv, log_dir, caller, screen=jevscreen.Screen.from_env())

    # --- orientation --------------------------------------------------------------------------------
    @srv.tool(structured_output=False)
    def overview() -> str:
        """What the database holds: sports, seasons, the live forecast per competition, model runs by kind, market
        links per exchange, venues, users and the next races. Call this first."""
        return _read(T.overview, viewer=_viewer())

    @srv.tool(structured_output=False)
    def describe_schema(table: str | None = None) -> str:
        """The database tables with one-line meanings, or one table's columns and types (for the sql tool)."""
        return _read(T.describe_schema, table=table)

    # --- data ---------------------------------------------------------------------------------------
    @srv.tool(structured_output=False)
    def list_events(sport: str | None = None, competition: str | None = None, season: int | None = None,
                    status: str | None = None, limit: int = 50, offset: int = 0) -> str:
        """Events (race weekends / rounds), newest first. sport: a sport code (f1, nascar, motogp, mtb_dh, ...)
        or competition: a competition code (overview() lists both);
        status: completed or scheduled. Each row has its id (for get_event) and source_key (e.g. 2026-15)."""
        return _read(T.list_events, sport=sport, competition=competition, season=season, status=status, limit=limit, offset=offset)

    @srv.tool(structured_output=False)
    def get_event(event_id: int | None = None, source_key: str | None = None, include: str = "results") -> str:
        """One event by id or source_key ('2026-15', '20260821_mtb'): its races and, per `include` (comma-separated:
        results, predictions, markets), the final classification, every stored prediction, the venues matrix."""
        return _read(T.get_event, event_id=event_id, source_key=source_key, include=include, viewer=_viewer())

    @srv.tool(structured_output=False)
    def search_athletes(q: str | None = None, limit: int = 50) -> str:
        """Drivers and riders by (part of) name, with result counts, last race and wins."""
        return _read(T.search_athletes, q=q, limit=limit)

    @srv.tool(structured_output=False)
    def get_athlete(athlete_id: int, include: str = "results") -> str:
        """One athlete with identifiers and, per `include` (results, predictions), their results and prediction history."""
        return _read(T.get_athlete, athlete_id=athlete_id, include=include)

    @srv.tool(structured_output=False)
    def list_markets(race_id: int | None = None, event_id: int | None = None, competition: str | None = None,
                     sport: str | None = None, kinds: str | None = None, limit: int = 50, offset: int = 0) -> str:
        """The market matrix: one row per outcome (kind x subject) with our fair value, each venue's quote (Polymarket,
        Kalshi when enabled, the private book), the gap and, for a past race, the result and the exchange price at the
        time we priced. Give race_id or event_id for a race weekend, or competition/sport for the season-long markets.
        kinds: comma-separated (race_win, race_podium, race_top10, race_h2h, race_constructor_top, race_pole, champion, ...).
        freshness: per exchange, the links and the newest price sync; `stale` (with the reason) when an upcoming event has
        no linked market on a live exchange or its newest sync is older than RACINGLINES_STALE_HOURS (default 3):
        quote those prices as possibly out of date."""
        return _read(T.list_markets, race_id=race_id, event_id=event_id, competition=competition, sport=sport, kinds=kinds,
                     limit=limit, offset=offset, viewer=_viewer())

    @srv.tool(structured_output=False)
    def get_market_history(tokens: str | None = None, race_id: int | None = None, kind: str | None = None,
                           athlete_id: int | None = None, subject: str | None = None, exchange: str | None = None,
                           series: str = "prices", start: str | None = None, end: str | None = None,
                           resample: str | None = "1h", limit: int = 200, offset: int = 0) -> str:
        """Exchange time series from the archive and the database. series: prices (last price per bucket), trades (count,
        volume, VWAP per bucket; resample=None lists trades) or books (best bid/ask). Pick markets by tokens (comma-separated
        token ids from list_markets) or by race_id + kind / athlete_id / subject (a name) / exchange (polymarket, kalshi).
        start/end are UTC ISO timestamps; a race defaults to the week before its start. resample is a pandas offset
        ('15min', '1h', '1D'); at most 200 points per token whatever you ask."""
        return _read(T.get_market_history, tokens=tokens, race_id=race_id, kind=kind, athlete_id=athlete_id, subject=subject,
                     exchange=exchange, series=series, start=start, end=end, resample=resample, limit=limit, offset=offset)

    @srv.tool(structured_output=False)
    def list_model_runs(kind: str | None = None, competition: str | None = None, sport: str | None = None,
                        season: int | None = None, limit: int = 50, offset: int = 0) -> str:
        """Stored model runs, newest first. kind: forecast (live prices), scenario, diagnostic (as-of a cutoff), backtest,
        sweep (a season traded on the exchange), season_strategy, season_checkpoints, walk_forward, candidate."""
        return _read(T.list_model_runs, kind=kind, competition=competition, sport=sport, season=season, limit=limit, offset=offset)

    @srv.tool(structured_output=False)
    def get_model_run(run_id: int, path: str | None = None) -> str:
        """One run: params, metrics and its prediction targets. Large metrics come back as a key map with sizes; path
        (e.g. 'metrics.weekends', 'metrics.weekends.3', 'params.settings') returns that part, paged when it is a list."""
        return _read(T.get_model_run, run_id=run_id, path=path)

    @srv.tool(structured_output=False)
    def compare_runs(run_a: int, run_b: int, kinds: str | None = None, stage: str | None = None,
                     diagnostics: bool = False) -> str:
        """Two saved runs scored on the same outcomes, paired by event, from their prediction records: per kind and season
        (and pooled), log loss and Brier of each, the difference B - A with its standard error over events and whether 0
        is outside +-2 SE. kinds: comma-separated market kinds; stage: one stage label. diagnostics=True adds the pass test's
        tables: reliability buckets (priced vs happened per price bucket), favourite wins, and the pole-sitter win rate. Read-only."""
        return _read(T.compare_runs, run_a=run_a, run_b=run_b, kinds=kinds, stage=stage, diagnostics=diagnostics)

    @srv.tool(structured_output=False)
    def get_predictions(run_id: int, target: str | None = None, top: int = 20, standings: bool = False) -> str:
        """A run's per-athlete probabilities (win, podium, top 10, make the Final, expected points) for one target (a race;
        omit to list the targets), or with standings=true its projected championship standings."""
        return _read(T.get_predictions, run_id=run_id, target=target, top=top, standings=standings)

    @srv.tool(structured_output=False)
    def get_forecast(competition: str | None = None, sport: str | None = None, category: str | None = None, top: int = 10) -> str:
        """The live forecast (the run the web app shows as our fair prices): the next races' top probabilities and the
        championship standings, per competition and category."""
        return _read(T.get_forecast, competition=competition, sport=sport, category=category, top=top)

    # --- strategy research ---------------------------------------------------------------------------
    @srv.tool(structured_output=False)
    def edge_finder(year: int = 2026, strategy: str | None = None, venue: str | None = None, limit: int = 50,
                    offset: int = 0) -> str:
        """The Lab's Edge Finder from saved sweeps: every configuration (a full set of sweep settings) with a full-season
        sweep of `year`, and each strategy's full-season recap (P&L, volume, weekends up, drawdown, consistency, fills and
        markout for makers) with its closing-line value (clv: bets, decided, mean CLV in points and percent, share
        positive). Nothing is simulated. strategy: update, hold, last, early, maker, maker_flat, ... (omit for all).
        venue: polymarket or kalshi, the exchange the sweep traded (omit for all; each row names its venue)."""
        return _read(T.edge_finder, year=year, strategy=strategy, venue=venue, limit=limit, offset=offset)

    @srv.tool(structured_output=False)
    def list_candidates(limit: int = 50, offset: int = 0) -> str:
        """Named configuration x strategy candidates saved in the Lab (searches add theirs; the strategy profiles A and C)."""
        return _read(T.list_candidates, limit=limit, offset=offset)

    @srv.tool(structured_output=False)
    def list_diagnostics(limit: int = 50, offset: int = 0) -> str:
        """Events with as-of diagnostic runs (our prices at a cutoff before the race): the run ids and cutoffs per event.
        get_diagnostic(run_id) reads one; replay_maker(run_id, ...) re-quotes it."""
        return _read(T.list_diagnostics, limit=limit, offset=offset)

    @srv.tool(structured_output=False)
    def get_diagnostic(run_id: int, top: int = 25) -> str:
        """One diagnostic run: our prices vs Polymarket at the cutoff (and 24 h before), the edge per market, the paper
        strategy's P&L, Brier and log loss per kind for model and market, and the result."""
        return _read(T.get_diagnostic, run_id=run_id, top=top)

    @srv.tool(structured_output=False)
    def replay_maker(run_id: int, fill: str = "through", half_spread: float = 0.02, size: float = 50, max_pos: float = 250,
                     max_capital: float = 1000, skew: float = 1.0, max_disagree: float = 0.15, min_volume_24h: float = 100,
                     pull_min: float = 15, exchange: str = "polymarket", with_sweep: bool = False, top: int = 20) -> str:
        """Scenario: replay a maker quoting an exchange (polymarket, or kalshi with its maker fee) through an event's
        diagnostic runs (repricing at each cutoff) against the real trade tape, with the event diagnostic's knobs: fill rule
        (through, touch, queue), half-spread ($), shares per quote, max inventory per market, max worst-case loss ($),
        inventory skew, max |fair - market| to quote, min 24 h volume ($), pull time before the race (min). Synchronous
        (seconds); nothing is stored. with_sweep adds the half-spread x fill-rule sensitivity table."""
        return _read(T.replay_maker, run_id=run_id, fill=fill, half_spread=half_spread, size=size, max_pos=max_pos,
                     max_capital=max_capital, skew=skew, max_disagree=max_disagree, min_volume_24h=min_volume_24h,
                     pull_min=pull_min, exchange=exchange, with_sweep=with_sweep, top=top)

    # --- paper trading, live events, change log ------------------------------------------------------
    @srv.tool(structured_output=False)
    def track_record(user: str, venue: str = "polymarket", sport: str | None = None) -> str:
        """A user's paper-trading record, one row per weekend: strategy, trades taken or fills, positions, P&L (settled or
        marked), backtest replay or live. venue: polymarket, kalshi (the maker's replay on Kalshi's tape), private, or all
        (one row per weekend and venue, with a venue column and totals per venue). sport: f1, nascar, motogp, ... (omit
        for all; NASCAR and MotoGP paper rows show where RACINGLINES_SPORT_PAPER is on). Users: see overview()."""
        return _read(T.track_record, user=user, venue=venue, sport=sport, viewer=_viewer())

    @srv.tool(structured_output=False)
    def list_positions(user: str, venue: str | None = None, event_key: str | None = None, open_only: bool = False,
                       limit: int = 50, offset: int = 0) -> str:
        """A user's paper positions (YES/NO shares, cash, mark, outcome, P&L) with totals per venue."""
        return _read(T.list_positions, user=user, venue=venue, event_key=event_key, open_only=open_only, limit=limit,
                     offset=offset, viewer=_viewer())

    @srv.tool(structured_output=False)
    def list_signals(user: str | None = None, event_key: str | None = None, status: str | None = None,
                     action: str | None = None, limit: int = 50, offset: int = 0) -> str:
        """Paper signals (what a strategy profile would do: buy/sell recommendations, quote/pull, paper fills), newest first.
        status: new, alerted, expired, filled_paper; action: buy, sell, quote, pull, fill."""
        return _read(T.list_signals, user=user, event_key=event_key, status=status, action=action, limit=limit,
                     offset=offset, viewer=_viewer())

    # --- fantasy trading (F$ only; the Trade and History tabs) -------------------------------------------
    @srv.tool(structured_output=False)
    def fantasy_balance(user: str | None = None) -> str:
        """Your fantasy account: total F$ (the 1,000 plus both P&Ls), cash you can trade with now (shared with the
        strategy you follow), open hand trades at their mark, hand-trade P&L and strategy P&L. Hosted accounts see their
        own only; user is for the owner."""
        return _fantasy_read(T.fantasy_balance, user=user, viewer=_viewer())

    @srv.tool(structured_output=False)
    def fantasy_positions(user: str | None = None, open_only: bool = True, limit: int = 50, offset: int = 0) -> str:
        """Your hand-trade positions: market, side, shares, cost, fees, worth now, P&L; open_only=False adds the settled
        ones (outcome, payout)."""
        return _fantasy_read(T.fantasy_positions, user=user, open_only=open_only, limit=limit, offset=offset,
                             viewer=_viewer())

    @srv.tool(structured_output=False)
    def fantasy_orders(user: str | None = None, limit: int = 50, offset: int = 0) -> str:
        """Your orders by hand, newest first, with fills, average price, fee, and the reason for any refused order."""
        return _read(T.fantasy_orders, user=user, limit=limit, offset=offset, viewer=_viewer())

    @srv.tool(structured_output=False)
    def fantasy_leaderboard(limit: int = 50, offset: int = 0) -> str:
        """The fantasy leaderboard: rank, score (ROI x (trades / 10)^0.3), ROI, trades and P&L. Never balances."""
        return _fantasy_read(T.fantasy_leaderboard, limit=limit, offset=offset)

    @srv.tool(structured_output=False)
    def fantasy_markets(q: str | None = None, limit: int = 50, offset: int = 0) -> str:
        """Markets you can trade with F$ now (F1 on Kalshi and Polymarket) with the best recorded YES bid and ask, the
        book's time and closes_at (when the market closes: the deciding session's start; null for a season future).
        q searches the question, driver or event. Use market_link_id with fantasy_book, fantasy_quote and fantasy_order."""
        return _read(T.fantasy_markets, q=q, limit=limit, offset=offset, books=_books())

    @srv.tool(structured_output=False)
    def fantasy_book(market_link_id: int) -> str:
        """One market's latest recorded book, YES and NO sides, up to the 10 levels an order may walk, and closes_at."""
        return _read(T.fantasy_book, market_link_id=market_link_id, books=_books())

    @srv.tool(structured_output=False)
    def fantasy_quote(market_link_id: int, side: str, action: str, shares: int, limit_price: float,
                      user: str | None = None) -> str:
        """What fantasy_order would do with this order right now, without placing it: the fills against the latest
        recorded book, or the venue's live book when that one is over 15 minutes old (the order itself is checked against the venue's live book, so it can differ) at or better than
        your limit, the average price, the fee, the total it would cost (a buy) or
        bring (a sale), or the reason it would be refused. Same arguments as fantasy_order; nothing is written."""
        return _read(T.fantasy_quote, market_link_id=market_link_id, side=side, action=action, shares=shares,
                     limit_price=limit_price, user=user, viewer=_viewer(), books=_books())

    @srv.tool(structured_output=False)
    def fantasy_order(market_link_id: int, side: str, action: str, shares: int, limit_price: float,
                      client_key: str | None = None, user: str | None = None) -> str:
        """Place one immediate-or-cancel F$ order on your own account, exactly as the Trade page does: side yes|no,
        action buy|sell, whole shares, limit_price 0.01-0.99. It is checked against the venue's live book at that
        moment (Kalshi or Polymarket; a failed fetch is refused as book_unavailable) and fills up to 10 levels at or
        better than your limit; the rest is cancelled; book_source says which book it walked ("live", or "recorded"
        when the server runs with recorded books). F$100 per order at most,
        never more than your cash, never short. Every refusal, the rate limit (10 a minute, 200 a day) included, is
        ok=false with reason and message. client_key makes a retry safe. Nothing reaches an exchange. user: only
        over stdio (the owner naming the account)."""
        return _fantasy_write(market_link_id=market_link_id, side=side, action=action, shares=shares,
                              limit_price=limit_price, client_key=client_key, user=user)

    @srv.tool(structured_output=False)
    def fantasy_strategy(user: str | None = None) -> str:
        """The strategy you follow (style, sizing, Kelly fraction, tweaks; null when none), whether changes are paused
        for a race weekend and until when, the five styles you can follow (label, blurb, backtest return on F$1,000,
        maker or taker, the two tweak sliders with their ranges) and every code fantasy_follow takes. Hosted accounts
        see their own only; user is for the owner."""
        return _read(T.fantasy_strategy, user=user, viewer=_viewer())

    @srv.tool(structured_output=False)
    def fantasy_follow(code: str, sizing: str = "standard", kelly: float = 0.25, min_edge: float | None = None,
                       min_volume_24h: float | None = None, rolled: bool = False, user: str | None = None) -> str:
        """Follow a strategy on your own account, exactly as the Strategy page saves it: code one of the five styles
        (T1, T2, T8, T6, T5; fantasy_strategy lists them) or "none" to stop. sizing standard|kelly, kelly a fraction
        0.10-1.00 on a 5% step. min_edge (0.03-0.15) and min_volume_24h (0-400) tweak the style (clamped; untested,
        scored live); rolled marks tweaks as random. Paused during a race weekend's signal window. A refusal comes back
        as ok=false with a reason (demo_account, slot_not_open, weekend_locked, bad_sizing, bad_code, not_allowed,
        rate_limited) and the page's message; nothing is written then. user: only over stdio (the owner)."""
        return _fantasy_write(T.fantasy_follow, code=code, sizing=sizing, kelly=kelly, min_edge=min_edge,
                              min_volume_24h=min_volume_24h, rolled=rolled, user=user)

    @srv.tool(structured_output=False)
    def pull_manifest(sport: str, since: str | None = None) -> str:
        """`racinglines data pull SPORT`: the files of one sport's pull (the registry, its race data, model runs and
        exchange prices, trades and books by month). since: a UTC date (YYYY-MM-DD) for a top-up. pull_chunk fetches
        each file. Never accounts, positions, signals, orders or jobs."""
        return _read(T.pull_manifest, sport=sport, since=since)

    @srv.tool(structured_output=False)
    def pull_chunk(pull_id: str, name: str, offset: int = 0) -> str:
        """One base64 chunk of a pull_manifest file from byte `offset`; `done` says whether it was the last."""
        return _read(T.pull_chunk, pull_id=pull_id, name=name, offset=offset)

    @srv.tool(structured_output=False)
    def list_live_events(limit: int = 50, offset: int = 0) -> str:
        """Live private-book events: the settled ones (maker P&L vs the crowd and the demo taker, fills, volume) and the run
        folders on this machine, with whether one is live now."""
        return _read(T.list_live_events, limit=limit, offset=offset)

    @srv.tool(structured_output=False)
    def data_changes(sport: str | None = None, limit: int = 50, offset: int = 0) -> str:
        """The data change log: ingests that changed the race history, athlete merges and notes, newest first."""
        return _read(T.data_changes, sport=sport, limit=limit, offset=offset)

    # the VM's storage snapshot (racinglines/storage_status.py): the hosted server only, the public package leaves it out
    if importlib.util.find_spec("racinglines.storage_status"):
        _storage_tool(srv)

    # --- sportsbook slips ----------------------------------------------------------------------------
    @srv.tool(structured_output=False)
    def list_kinds(sport: str | None = None) -> str:
        """Every market kind (race winner, podium, top N, head-to-head, pole, sprint, props, ...) with the sports that
        model it: what a book's lines may name for map_book / price_book. sport: f1, nascar, motogp, ... to list only
        the kinds modeled there."""
        return _plain(T.list_kinds, sport=sport)

    @srv.tool(structured_output=False)
    def map_book(book: str) -> str:
        """Match a generic sportsbook's book (TOML text: [book] venue, event, sport, odds format; [[lines]] with kind,
        driver / opponent / team, odds; combos as legs; docs/sportsbook/slips.md) to exact races and athletes in the
        database. Each line and leg comes back mapped, or unmapped with the reason. Nothing is stored."""
        return _read(T.map_book, book=book)

    @srv.tool(structured_output=False)
    def price_book(book: str, run_ids: list[int] | None = None) -> str:
        """map_book, then price every leg and line: the book's implied probability, our model's (a stored model run:
        the app's own choice, or run_ids to pin one per competition), the linked prediction market's (Polymarket,
        Kalshi, OG.com) and the expected value of the bet against each. Same-race combos are priced jointly where the
        simulations allow, and flagged otherwise. The two probabilities are never blended (owner decision)."""
        return _read(T.price_book, book=book, run_ids=run_ids)

    @srv.tool(structured_output=False)
    def settle_book(book: str) -> str:
        """map_book, then settle each leg and line from the stored results: won, lost, void, manual or pending, with
        the payout and profit per line at the book's odds."""
        return _read(T.settle_book, book=book)

    # --- sql ----------------------------------------------------------------------------------------
    @srv.tool(structured_output=False)
    def sql(query: str, limit: int = 50, offset: int = 0) -> str:
        """Run one read-only SQL query (SELECT / WITH ... SELECT / EXPLAIN) against the database, in a READ ONLY transaction
        with a 10 s timeout. The result is paged (limit at most 500; an outer LIMIT/OFFSET is applied for you). The users and
        orders tables are not readable. describe_schema() lists the tables and columns."""
        _require_admin()
        return _read(T.sql, engine=_sql_engine(), query=query, limit=limit, offset=offset)

    # --- jobs ---------------------------------------------------------------------------------------
    @srv.tool(structured_output=False)
    def list_job_types() -> str:
        """The simulations run_job can queue (the Lab's job catalog) with their knobs, ranges and defaults, and every sweep
        setting (model, timing, taker, maker, markets) an Edge Finder sweep accepts."""
        return _plain(T.list_job_types)

    @srv.tool(structured_output=False)
    def run_job(job_type: str, params: dict | None = None) -> str:
        """Scenario: queue a simulation as a Lab job: f1_backtest, f1_scenario (a forward forecast saved as a scenario, never
        the live prices), f1_diagnostic (event '2026-15', cutoff '2026-09-25T13:30'), f1_sweep (year + settings: any sweep
        setting), f1_season_strategy, f1_combo (combo / same-game parlay prices for one event: event '2026-17', legs = a list
        of legs like [{"kind": "race_win", "driver": "Max Verstappen"}, {"kind": "race_fastest_lap", "driver": "Max
        Verstappen"}], several combos as {name: [legs]}, optional cutoff, variant; get_model_run(result_run_id) has
        metrics.combos and metrics.checks), dh_scenario, dh_backtest, walk_forward (sport, model: own / global / a
        challenger the sport's schema names, seasons '2025 2026', venue, stage: a [stages] label such as 'after Quali',
        settings: a dict of the chosen model's settings, e.g. {"covariates": "grid,qual_gap", "temperature": "fit"},
        validated against its schema). params: the job type's knobs (list_job_types). Returns the
        job id; get_job follows it and names the model run it saved (result_run_id)."""
        return _write(T.run_job, job_type=job_type, params=params, user_id=_owner()[0])

    @srv.tool(structured_output=False)
    def get_job(job_id: int, log_lines: int = 20) -> str:
        """A job's status (queued, running, done, failed), progress, the last log lines and result_run_id."""
        return _read(T.get_job, job_id=job_id, log_lines=log_lines, viewer=_owner()[1])

    @srv.tool(structured_output=False)
    def list_jobs(status: str | None = None, limit: int = 50, offset: int = 0) -> str:
        """Jobs, newest first (the Lab's and this server's)."""
        return _read(T.list_jobs, status=status, limit=limit, offset=offset, viewer=_owner()[1])

    @srv.tool(structured_output=False)
    def cancel_job(job_id: int) -> str:
        """Cancel a job that has not started yet."""
        return _write(T.cancel_job, job_id=job_id, viewer=_owner()[1])

    # --- resources: the docs and the sport schemas ---------------------------------------------------
    @srv.resource("racinglines://docs", name="docs-index", description="The documentation pages available as racinglines://docs/<page>")
    def docs_index() -> str:
        pages = sorted(p.name for p in DOCS.glob("*.md"))
        return "\n".join(pages)

    @srv.resource("racinglines://docs/{page}", name="doc", description="One documentation page (Markdown), e.g. model.md",
                  mime_type="text/markdown")
    def doc(page: str) -> str:
        p = DOCS / page
        if not page.endswith(".md") or "/" in page or not p.exists():
            raise ResourceError(f"no docs page {page!r}; read racinglines://docs for the list")
        return p.read_text()

    @srv.resource("racinglines://sports/{code}", name="sport", description="A sport's schema (sports/<code>.toml): f1, mtb_dh, or a tape-only sport (nascar, motogp, indycar)",
                  mime_type="application/toml")
    def sport(code: str) -> str:
        p = SPORTS / f"{code}.toml"
        if "/" in code or not p.exists():
            raise ResourceError(f"no sport {code!r}")
        if code in T.hidden_sports(caller()):
            raise ResourceError(f"{code} is not available")
        return p.read_text()

    if jobs_worker:
        from racinglines.web import jobs as J
        J.start_worker(reset_running=False)
    return srv


def http_app(srv):
    """The streamable-HTTP ASGI app of a server built with `oauth=True`: /mcp needs a bearer token (401 with the
    OAuth discovery header otherwise), and the SDK serves the sign-in endpoints next to it. Any Host header: the
    SDK's default for a loopback bind answers 421 to every Host but localhost (its DNS-rebinding guard: a browser
    page tricked into calling 127.0.0.1 would carry no bearer token and is refused 401 anyway), which would also
    refuse the tunnel's public hostname (mcp.racinglines.bet)."""
    from mcp.server.transport_security import TransportSecuritySettings
    return srv.streamable_http_app(stateless_http=True, json_response=True,
                                   transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))


SHUTDOWN_GRACE_SEC = 10
"""How long the HTTP server waits for in-flight requests on SIGTERM before cancelling them. uvicorn's default is
forever: a chat client keeps a GET /mcp event stream open for as long as it is connected, so with no limit every
`systemctl restart` (every deploy) sat on that stream until systemd's 90-second stop timeout killed the process, and
the connector was down for those 90 seconds. The unit's TimeoutStopSec stays above this number."""


def serve(http=False, host="127.0.0.1", port=DEFAULT_PORT, jobs_worker=True):
    """Run the server: stdio (default), or streamable HTTP on host:port/mcp, each request with an account's token.
    On SIGTERM the server stops within SHUTDOWN_GRACE_SEC seconds."""
    from racinglines.mcp import auth, oauth
    srv = build(jobs_worker=jobs_worker, oauth=http)
    if not http:
        srv.run("stdio")
        return
    oauth.warn_if_disabled()
    who = auth.holders(_accounts_engine())
    print("accounts with an rl_ token: " + (", ".join(f"{u} ({r})" for u, r, _ in who) or "none") +
          f"; OAuth sign-in for roles {', '.join(auth.ROLES)} via {oauth.WEB_URL}", file=sys.stderr)
    if host not in ("127.0.0.1", "localhost"):
        print(f"WARNING: listening on {host}; keep it behind a tunnel or a private network.", file=sys.stderr)
    import uvicorn
    uvicorn.run(http_app(srv), host=host, port=int(port), log_level="warning", timeout_graceful_shutdown=SHUTDOWN_GRACE_SEC)


def _storage_tool(srv):
    @srv.tool(structured_output=False)
    def storage_status(refresh: bool = False) -> str:
        """Admin only. Where the data lives: the VM's disk (free / used), each data folder's size, files and newest file,
        the database dumps, every database's size, and the data bucket per prefix (vendor/probalytics, backups/prod,
        archive/, ...) with size, objects and newest object. From the last snapshot; refresh=True starts a new one in
        the background (minutes). Sizes and counts only, never file contents."""
        return _read(T.storage_status, refresh=refresh)
