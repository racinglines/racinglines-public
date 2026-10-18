"""
SQLAlchemy models for the racinglines database (PostgreSQL).

Built for many sports and leagues, not just downhill:

    Sport            what is raced            mtb_dh (mountain bike downhill)
    League           who organizes it         uci_mtb (UCI Mountain Bike World Series)
    Competition      league x sport           uci_dhi_wc (UCI Downhill World Cup)
      Category       per competition          ME, MJ, WE, WJ
      Season         per competition          2026
        Event        one weekend / meeting    Les Gets 2026 (source key 20260821_mtb)
          Race       event x category         Les Gets 2026, Men Elite  (+ weekend format)
            Round    a session of the race    practice / qual1 / qual2 / final ...
              Result one athlete in a round   position, status, time
                Split intermediate times  cumulative time at split n

Athletes are global (one person can race several sports) and are matched
through AthleteIdentifier rows (scheme, value): "name" = normalized name key
today; "uci" (or any other federation / source ID) later.

Times are stored as integer milliseconds so nothing is lost to float rounding.
Anything sport-specific that doesn't deserve a column goes in a JSONB field
(Race.format, Result.extra, ModelRun.params / metrics, *.extra).
"""

from datetime import date, datetime

from sqlalchemy import (
    BigInteger, Date, DateTime, Float, ForeignKey, Integer, String, Text,
    UniqueConstraint, func, text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Sport(Base):
    __tablename__ = "sports"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(40), unique=True)          # mtb_dh
    name: Mapped[str] = mapped_column(String(120))                      # Mountain bike downhill
    result_kind: Mapped[str] = mapped_column(String(20), default="time", server_default="time")  # time | score | ...


class League(Base):
    __tablename__ = "leagues"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(40), unique=True)          # uci_mtb
    name: Mapped[str] = mapped_column(String(200))
    organizer: Mapped[str | None] = mapped_column(String(120))          # UCI


class Competition(Base):
    """A league's championship in one sport, e.g. the UCI Downhill World Cup."""
    __tablename__ = "competitions"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(40), unique=True)          # uci_dhi_wc
    name: Mapped[str] = mapped_column(String(200))
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"))
    sport_id: Mapped[int] = mapped_column(ForeignKey("sports.id"))

    league: Mapped[League] = relationship()
    sport: Mapped[Sport] = relationship()


class Category(Base):
    __tablename__ = "categories"
    __table_args__ = (UniqueConstraint("competition_id", "code"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"))
    code: Mapped[str] = mapped_column(String(20))                       # ME
    name: Mapped[str] = mapped_column(String(120))                      # Men Elite
    gender: Mapped[str | None] = mapped_column(String(1))               # M / W / X
    age_group: Mapped[str | None] = mapped_column(String(20))           # elite / junior / u23


class Season(Base):
    __tablename__ = "seasons"
    __table_args__ = (UniqueConstraint("competition_id", "year"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"))
    year: Mapped[int] = mapped_column(Integer)
    label: Mapped[str | None] = mapped_column(String(40))               # e.g. "2025/26" for winter sports


class Venue(Base):
    __tablename__ = "venues"
    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(80), unique=True)          # les-gets
    name: Mapped[str] = mapped_column(String(120))
    country: Mapped[str | None] = mapped_column(String(3))              # IOC/UCI code, FRA


class VenueAlias(Base):
    """Other spellings of a venue seen in source data -> the canonical venue."""
    __tablename__ = "venue_aliases"
    alias: Mapped[str] = mapped_column(String(80), primary_key=True)    # mont-ste-anne
    venue_id: Mapped[int] = mapped_column(ForeignKey("venues.id"))


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (UniqueConstraint("season_id", "source", "source_key"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id"))
    source: Mapped[str] = mapped_column(String(40))                     # chronorace
    source_key: Mapped[str] = mapped_column(String(80))                 # 20260821_mtb
    name: Mapped[str] = mapped_column(Text)
    start_date: Mapped[date] = mapped_column(Date)
    venue_id: Mapped[int | None] = mapped_column(ForeignKey("venues.id"))
    series_round: Mapped[int | None] = mapped_column(Integer)           # DHI #7
    status: Mapped[str] = mapped_column(String(20), default="completed", server_default="completed")  # scheduled | completed | cancelled

    season: Mapped[Season] = relationship()
    venue: Mapped[Venue | None] = relationship()


class Race(Base):
    """One category's race at an event. `format` holds the weekend format,
    e.g. {"kind": "q1q2", "q1_to_final": 20, "q2_to_final": 10}."""
    __tablename__ = "races"
    __table_args__ = (UniqueConstraint("event_id", "category_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"))
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))
    format: Mapped[dict | None] = mapped_column(JSONB)

    event: Mapped[Event] = relationship()
    category: Mapped[Category] = relationship()
    rounds: Mapped[list["Round"]] = relationship(back_populates="race", cascade="all, delete-orphan",
                                                 passive_deletes=True)


class Round(Base):
    __tablename__ = "rounds"
    __table_args__ = (UniqueConstraint("race_id", "kind"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(20))                       # practice | qual | qual1 | qual2 | semi | final
    ordinal: Mapped[int] = mapped_column(Integer)                       # running order within the race
    name: Mapped[str | None] = mapped_column(String(80))                # heading as published
    extra: Mapped[dict | None] = mapped_column(JSONB)                   # e.g. weather, total laps, session date

    race: Mapped[Race] = relationship(back_populates="rounds")


class Athlete(Base):
    __tablename__ = "athletes"
    id: Mapped[int] = mapped_column(primary_key=True)
    display_name: Mapped[str] = mapped_column(String(200))
    nation: Mapped[str | None] = mapped_column(String(3))
    birth_year: Mapped[int | None] = mapped_column(Integer)
    gender: Mapped[str | None] = mapped_column(String(1))


class AthleteIdentifier(Base):
    """How sources refer to an athlete: ("name", "o callaghan oisin"),
    ("uci", "10007307417"), ... One athlete can have many."""
    __tablename__ = "athlete_identifiers"
    scheme: Mapped[str] = mapped_column(String(20), primary_key=True)
    value: Mapped[str] = mapped_column(String(200), primary_key=True)
    athlete_id: Mapped[int] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)


class Result(Base):
    __tablename__ = "results"
    __table_args__ = (UniqueConstraint("round_id", "athlete_id"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    round_id: Mapped[int] = mapped_column(ForeignKey("rounds.id", ondelete="CASCADE"))
    athlete_id: Mapped[int] = mapped_column(ForeignKey("athletes.id"), index=True)
    position: Mapped[int | None] = mapped_column(Integer)               # official rank; null if not classified
    status: Mapped[str] = mapped_column(String(10))                     # OK | DNF | DNS | DSQ
    time_ms: Mapped[int | None] = mapped_column(BigInteger)
    bib: Mapped[str | None] = mapped_column(String(20))
    team: Mapped[str | None] = mapped_column(String(200))
    nation: Mapped[str | None] = mapped_column(String(3))
    extra: Mapped[dict | None] = mapped_column(JSONB)


class Split(Base):
    __tablename__ = "splits"
    result_id: Mapped[int] = mapped_column(ForeignKey("results.id", ondelete="CASCADE"), primary_key=True)
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)         # 1..n
    cum_time_ms: Mapped[int | None] = mapped_column(BigInteger)
    rank: Mapped[int | None] = mapped_column(Integer)


class SourceFile(Base):
    """Every ingested file, so re-ingesting unchanged files is a no-op."""
    __tablename__ = "source_files"
    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(Text, unique=True)
    sha256: Mapped[str] = mapped_column(String(64))
    parser: Mapped[str] = mapped_column(String(40))
    race_id: Mapped[int | None] = mapped_column(ForeignKey("races.id", ondelete="SET NULL"))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PointsScheme(Base):
    """Championship points by rank, per competition, era and round kind."""
    __tablename__ = "points_schemes"
    id: Mapped[int] = mapped_column(primary_key=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"))
    season_from: Mapped[int] = mapped_column(Integer)
    season_to: Mapped[int | None] = mapped_column(Integer)              # null = still current
    round_kind: Mapped[str] = mapped_column(String(20))                 # final | qual1 | semi ...
    points: Mapped[list[int]] = mapped_column(ARRAY(Integer))           # points[0] = 1st place
    is_official: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    note: Mapped[str | None] = mapped_column(Text)


class ModelRun(Base):
    """One run of a model: its settings, what it was fit on, and summary metrics."""
    __tablename__ = "model_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"))
    season_id: Mapped[int | None] = mapped_column(ForeignKey("seasons.id"))
    category_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id"))
    model: Mapped[str] = mapped_column(String(60))                      # season_sim
    kind: Mapped[str] = mapped_column(String(20))                       # forecast | backtest
    data_through: Mapped[date | None] = mapped_column(Date)             # last event date in training data
    code_version: Mapped[str | None] = mapped_column(String(64))        # git commit
    params: Mapped[dict | None] = mapped_column(JSONB)
    metrics: Mapped[dict | None] = mapped_column(JSONB)


class RacePrediction(Base):
    """Per-athlete prediction for one race. race_id is null for races that
    don't exist in the database yet (e.g. an unraced round with unknown venue);
    `target` then names it (e.g. "next_round")."""
    __tablename__ = "race_predictions"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"), index=True)
    race_id: Mapped[int | None] = mapped_column(ForeignKey("races.id", ondelete="CASCADE"))
    target: Mapped[str] = mapped_column(String(80))
    athlete_id: Mapped[int] = mapped_column(ForeignKey("athletes.id"))
    win_prob: Mapped[float | None] = mapped_column(Float)
    podium_prob: Mapped[float | None] = mapped_column(Float)
    top10_prob: Mapped[float | None] = mapped_column(Float)
    make_final_prob: Mapped[float | None] = mapped_column(Float)
    exp_points: Mapped[float | None] = mapped_column(Float)
    extra: Mapped[dict | None] = mapped_column(JSONB)


class StandingsPrediction(Base):
    __tablename__ = "standings_predictions"
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"), primary_key=True)
    athlete_id: Mapped[int] = mapped_column(ForeignKey("athletes.id"), primary_key=True)
    current_points: Mapped[float | None] = mapped_column(Float)
    exp_points: Mapped[float | None] = mapped_column(Float)
    points_p10: Mapped[float | None] = mapped_column(Float)
    points_p90: Mapped[float | None] = mapped_column(Float)
    champion_prob: Mapped[float | None] = mapped_column(Float)
    top3_prob: Mapped[float | None] = mapped_column(Float)
    top10_prob: Mapped[float | None] = mapped_column(Float)
    exp_rank: Mapped[float | None] = mapped_column(Float)
    extra: Mapped[dict | None] = mapped_column(JSONB)                   # e.g. wins_ge, ahead_of (pairwise)


class MarketLink(Base):
    """A prediction-market outcome token linked to one of our model's probabilities.

    prediction: race_win | race_podium | race_top10 | race_make_final   (per race; race_id
                null = the next unraced round, "remaining_round")
                champion | standings_top3                               (season standings)
    invert:     the token pays out when the prediction does NOT happen (e.g. the "No" side),
                so the model's fair price is 1 - p.
    """
    __tablename__ = "market_links"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    exchange: Mapped[str] = mapped_column(String(20), default="polymarket", server_default="polymarket")
    market_slug: Mapped[str | None] = mapped_column(String(300))
    question: Mapped[str] = mapped_column(Text)
    condition_id: Mapped[str | None] = mapped_column(String(80))
    token_id: Mapped[str] = mapped_column(String(100), index=True)
    outcome: Mapped[str] = mapped_column(String(200))                   # "Yes", "No", or a named outcome
    neg_risk: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    tick_size: Mapped[float | None] = mapped_column(Float)
    min_size: Mapped[float | None] = mapped_column(Float)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"))
    category_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id"))
    athlete_id: Mapped[int | None] = mapped_column(ForeignKey("athletes.id"))    # null for team / other subjects
    race_id: Mapped[int | None] = mapped_column(ForeignKey("races.id", ondelete="SET NULL"))
    prediction: Mapped[str] = mapped_column(String(30))                 # see racinglines/web/data.PREDICTION_KINDS
    params: Mapped[dict | None] = mapped_column(JSONB)                  # e.g. {"opponent_id": 12}, {"team": "mercedes"}, {"n": 3}
    event_slug: Mapped[str | None] = mapped_column(String(300), index=True)
    event_title: Mapped[str | None] = mapped_column(Text)
    group_title: Mapped[str | None] = mapped_column(String(200))       # outcome label within a multi-outcome event
    last_bid: Mapped[float | None] = mapped_column(Float)
    last_ask: Mapped[float | None] = mapped_column(Float)
    last_price: Mapped[float | None] = mapped_column(Float)            # mid if both sides quoted, else last/outcome price
    volume: Mapped[float | None] = mapped_column(Float)
    end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    closed: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    resolved_yes: Mapped[bool | None] = mapped_column()                # set when the exchange resolves the market
    synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # sync's first sight of the token (null: before alerts)
    invert: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    active: Mapped[bool] = mapped_column(default=True, server_default=text("true"))
    note: Mapped[str | None] = mapped_column(Text)


class Order(Base):
    """Every order the app builds: dry runs, submissions (with the exchange's
    response) and cancellations. Nothing is ever deleted."""
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    exchange: Mapped[str] = mapped_column(String(20), default="polymarket", server_default="polymarket")
    market_link_id: Mapped[int | None] = mapped_column(ForeignKey("market_links.id", ondelete="SET NULL"))
    token_id: Mapped[str] = mapped_column(String(100))
    side: Mapped[str] = mapped_column(String(4))                        # BUY | SELL
    price: Mapped[float] = mapped_column(Float)
    size: Mapped[float] = mapped_column(Float)                          # shares
    order_type: Mapped[str] = mapped_column(String(8), default="GTC", server_default="GTC")
    post_only: Mapped[bool] = mapped_column(default=True, server_default=text("true"))
    status: Mapped[str] = mapped_column(String(20))                     # dry_run | submitted | rejected | error | cancelled
    exchange_order_id: Mapped[str | None] = mapped_column(String(100))
    model_prob: Mapped[float | None] = mapped_column(Float)
    best_bid: Mapped[float | None] = mapped_column(Float)
    best_ask: Mapped[float | None] = mapped_column(Float)
    model_run_id: Mapped[int | None] = mapped_column(ForeignKey("model_runs.id", ondelete="SET NULL"))
    response: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)


class HouseMarket(Base):
    """A YES/NO market we quote ourselves (private bets), priced from the model.

    kind: race_win | race_podium | race_make_final   (race_id + athlete_id; settle from results)
          rank_up | rank_down                          (championship rank after the race vs before;
                                                        settle manually from official standings)
    Prices are what a counterparty pays per $1 payout:
        yes_price = fair + spread/2,  no_price = (1 - fair) + spread/2   (a side is not offered
        if its price would exceed max_price).
    """
    __tablename__ = "house_markets"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    race_id: Mapped[int | None] = mapped_column(ForeignKey("races.id"))            # null for season-level markets
    athlete_id: Mapped[int | None] = mapped_column(ForeignKey("athletes.id"))      # null for team markets
    kind: Mapped[str] = mapped_column(String(30))
    title: Mapped[str] = mapped_column(Text)
    market_link_id: Mapped[int | None] = mapped_column(ForeignKey("market_links.id", ondelete="SET NULL"), index=True)
    params: Mapped[dict | None] = mapped_column(JSONB)
    model_run_id: Mapped[int | None] = mapped_column(ForeignKey("model_runs.id", ondelete="SET NULL"))
    fair_prob: Mapped[float] = mapped_column(Float)                     # model probability (or admin override)
    fair_source: Mapped[str] = mapped_column(String(20), default="model", server_default="model")
    spread: Mapped[float] = mapped_column(Float)                        # e.g. 0.06 = 6 points total
    yes_price: Mapped[float | None] = mapped_column(Float)
    no_price: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(10), default="open", server_default="open")  # open | closed | settled | void
    outcome: Mapped[bool | None] = mapped_column()                      # True = YES happened
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    settle_note: Mapped[str | None] = mapped_column(Text)
    context: Mapped[dict | None] = mapped_column(JSONB)                 # e.g. current_rank for rank markets
    maker_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    # each maker quotes a market independently (null maker = legacy house markets)
    # (mirrored markets add market_link_id: e.g. several head-to-heads per driver and race)
    __table_args__ = (UniqueConstraint("race_id", "athlete_id", "kind", "maker_id", "market_link_id",
                                       name="uq_house_market_maker"),)


class HouseBet(Base):
    """A bet a counterparty took against our quote. Money is handled outside the app."""
    __tablename__ = "house_bets"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    market_id: Mapped[int] = mapped_column(ForeignKey("house_markets.id"), index=True)
    counterparty: Mapped[str] = mapped_column(String(120))
    side: Mapped[str] = mapped_column(String(3))                        # side the counterparty bought: YES | NO
    price: Mapped[float] = mapped_column(Float)                         # price paid per $1 payout
    stake: Mapped[float] = mapped_column(Float)                         # amount the counterparty puts up
    payout: Mapped[float] = mapped_column(Float)                        # stake / price, paid to them if they win
    status: Mapped[str] = mapped_column(String(10), default="open", server_default="open")  # open | won | lost | void
    note: Mapped[str | None] = mapped_column(Text)
    taker_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)



class Lap(Base):
    """One lap of a lap-based round (F1 qualifying/sprint/race; any sport with laps).
    Times in ms; speed traps in km/h (F1: I1 = sector-1 trap, I2 = sector-2 trap,
    FL = finish line, ST = speed trap on the fastest straight)."""
    __tablename__ = "laps"
    result_id: Mapped[int] = mapped_column(ForeignKey("results.id", ondelete="CASCADE"), primary_key=True)
    lap: Mapped[int] = mapped_column(Integer, primary_key=True)
    lap_time_ms: Mapped[int | None] = mapped_column(BigInteger)
    s1_ms: Mapped[int | None] = mapped_column(BigInteger)
    s2_ms: Mapped[int | None] = mapped_column(BigInteger)
    s3_ms: Mapped[int | None] = mapped_column(BigInteger)
    speed_i1: Mapped[float | None] = mapped_column(Float)
    speed_i2: Mapped[float | None] = mapped_column(Float)
    speed_fl: Mapped[float | None] = mapped_column(Float)
    speed_st: Mapped[float | None] = mapped_column(Float)
    compound: Mapped[str | None] = mapped_column(String(16))
    tyre_life: Mapped[float | None] = mapped_column(Float)
    stint: Mapped[int | None] = mapped_column(Integer)
    pit_in: Mapped[bool | None] = mapped_column()
    pit_out: Mapped[bool | None] = mapped_column()
    track_status: Mapped[str | None] = mapped_column(String(16))        # F1: "1" = green; 4 = SC, 6/7 = VSC ...
    position: Mapped[int | None] = mapped_column(Integer)
    is_accurate: Mapped[bool | None] = mapped_column()
    deleted: Mapped[bool | None] = mapped_column()


class TrackProfile(Base):
    """Per-event track properties computed from timing data, used as model features
    (sector time shares, speed-trap speeds, time-weighted speed index, overtaking,
    street circuit, weather). One row per event because layouts change over years."""
    __tablename__ = "track_profiles"
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), primary_key=True)
    venue_id: Mapped[int | None] = mapped_column(ForeignKey("venues.id"), index=True)
    features: Mapped[dict] = mapped_column(JSONB)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())



class User(Base):
    """Web-app accounts. role: admin | maker | taker. Passwords are scrypt hashes."""
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(60), unique=True)
    display_name: Mapped[str | None] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(10))
    password_hash: Mapped[str] = mapped_column(String(300))
    active: Mapped[bool] = mapped_column(default=True, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    prefs: Mapped[dict | None] = mapped_column(JSONB)                   # per-user UI settings (e.g. Edge Finder combos)


class ActivityLog(Base):
    """Every meaningful action in the web app, for the admin audit view."""
    __tablename__ = "activity_log"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    username: Mapped[str | None] = mapped_column(String(60))           # kept for failed logins / deleted users
    role: Mapped[str | None] = mapped_column(String(10))
    action: Mapped[str] = mapped_column(String(40), index=True)
    detail: Mapped[dict | None] = mapped_column(JSONB)
    ip: Mapped[str | None] = mapped_column(String(64))
    path: Mapped[str | None] = mapped_column(String(300))


class UserEmailPrefs(Base):
    """User email delivery settings: email address, exchange preferences (Polymarket/Kalshi/both),
    and sports interests (stored for future use; only F1 weekly picks sent now)."""
    __tablename__ = "user_email_prefs"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True)
    email: Mapped[str | None] = mapped_column(String(254), index=True)      # optional; if NULL, no email sent
    exchanges: Mapped[list[str]] = mapped_column(ARRAY(String), default=["polymarket"], server_default="'{polymarket}'")
    sports: Mapped[list[str]] = mapped_column(ARRAY(String), default=["f1_wdc"], server_default="'{f1_wdc}'")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship()


class MarketPriceHistory(Base):
    """Exchange price time series per outcome token, with optional Kalshi candle quotes."""
    __tablename__ = "market_price_history"
    token_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    price: Mapped[float] = mapped_column(Float)
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)


class MarketTrade(Base):
    """One exchange trade (Polymarket Data API, taker trades only): the taker's side,
    price and size for one outcome token. The tape a maker replay is filled against."""
    __tablename__ = "market_trades"
    id: Mapped[int] = mapped_column(primary_key=True)
    token_id: Mapped[str] = mapped_column(String(100), index=True)
    condition_id: Mapped[str] = mapped_column(String(100), index=True)
    outcome_index: Mapped[int] = mapped_column(Integer)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    side: Mapped[str] = mapped_column(String(4))            # taker side: BUY / SELL (of token_id)
    price: Mapped[float] = mapped_column(Float)
    size: Mapped[float] = mapped_column(Float)              # shares
    tx_hash: Mapped[str] = mapped_column(String(80))
    wallet: Mapped[str | None] = mapped_column(String(64))
    __table_args__ = (UniqueConstraint("tx_hash", "token_id", "wallet", "side", "price", "size",
                                       name="uq_market_trade"),)


class MarketBookSnapshot(Base):
    """Order-book snapshot per outcome token (CLOB /books), recorded by `racinglines markets record`.
    bids/asks are [[price, size], ...], best first. Polymarket has no historical book API,
    so depth only exists from when recording started."""
    __tablename__ = "market_book_snapshots"
    token_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    best_bid: Mapped[float | None] = mapped_column(Float)
    best_ask: Mapped[float | None] = mapped_column(Float)
    bids: Mapped[list | None] = mapped_column(JSONB)
    asks: Mapped[list | None] = mapped_column(JSONB)


class MarketDisagreement(Base):
    """The cross-venue disagreement log (`racinglines markets disagree`, off unless RACINGLINES_DISAGREE=1): one
    row per F1 outcome linked on both exchanges and per tick, with both venues' mids and top of book, our fair
    value at the time (the run in force then), each venue's fee-adjusted taker edge vs fair, and the
    cross-venue gap net of both taker fees (racinglines/markets/disagree.py). Prices are the outcome's YES
    side on both venues (an inverted Polymarket link is flipped)."""
    __tablename__ = "market_disagreements"
    pm_token: Mapped[str] = mapped_column(String(100), primary_key=True)        # Polymarket YES token
    kalshi_token: Mapped[str] = mapped_column(String(100), primary_key=True)    # Kalshi market ticker
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"))
    race_id: Mapped[int | None] = mapped_column(ForeignKey("races.id", ondelete="CASCADE"))   # null: a season market
    kind: Mapped[str] = mapped_column(String(30))
    athlete_id: Mapped[int | None] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"))
    team: Mapped[str | None] = mapped_column(String(40))
    opponent_id: Mapped[int | None] = mapped_column(Integer)
    n: Mapped[int | None] = mapped_column(Integer)
    pm_mid: Mapped[float | None] = mapped_column(Float)
    pm_bid: Mapped[float | None] = mapped_column(Float)
    pm_ask: Mapped[float | None] = mapped_column(Float)
    kalshi_mid: Mapped[float | None] = mapped_column(Float)
    kalshi_bid: Mapped[float | None] = mapped_column(Float)
    kalshi_ask: Mapped[float | None] = mapped_column(Float)
    fair: Mapped[float | None] = mapped_column(Float)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("model_runs.id", ondelete="SET NULL"))
    pm_edge: Mapped[float | None] = mapped_column(Float)        # fee-adjusted edge of the better taker trade on Polymarket
    pm_side: Mapped[str | None] = mapped_column(String(4))      # which trade that is: buy (YES at the ask) or sell (at the bid)
    kalshi_edge: Mapped[float | None] = mapped_column(Float)    # the same on Kalshi (its taker fee taken off)
    kalshi_side: Mapped[str | None] = mapped_column(String(4))
    gap: Mapped[float | None] = mapped_column(Float)            # kalshi_mid - pm_mid
    gap_net: Mapped[float | None] = mapped_column(Float)        # |gap| net of both venues' taker fees; > 0 = above fees
    pm_vol24: Mapped[float | None] = mapped_column(Float)       # USD traded on Polymarket in the 24 h before the tick (null: no tape stored)
    kalshi_vol24: Mapped[float | None] = mapped_column(Float)   # the same on Kalshi


class Job(Base):
    """A model run launched from the web app (backtest, scenario forecast, diagnostic, ...),
    executed as a CLI subprocess; `log` holds its output, `result_run_id` the model run it saved."""
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    kind: Mapped[str] = mapped_column(String(30))
    sport: Mapped[str] = mapped_column(String(20))
    params: Mapped[dict | None] = mapped_column(JSONB)
    argv: Mapped[list | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(12), default="queued")   # queued | running | done | failed
    progress: Mapped[str | None] = mapped_column(String(200))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result_run_id: Mapped[int | None] = mapped_column(ForeignKey("model_runs.id", ondelete="SET NULL"))
    log: Mapped[str | None] = mapped_column(Text)


class StrategySignal(Base):
    """What a user's strategy profile would do now (racinglines/pipelines/signals.py): a paper trade
    recommendation (taker), a quote starting / stopping or a paper fill (maker). Never an order.
    One row per (user, profile, market, dedupe, action, side); dedupe is the stage, or a fill's time."""
    __tablename__ = "strategy_signals"
    __table_args__ = (UniqueConstraint("user_id", "candidate_id", "market_key", "dedupe", "action", "side"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    candidate_id: Mapped[int | None] = mapped_column(Integer)            # the Lab candidate (may be deleted later)
    profile: Mapped[str] = mapped_column(String(120))
    strategy: Mapped[str] = mapped_column(String(20))                    # update | maker | ...
    race_id: Mapped[int | None] = mapped_column(ForeignKey("races.id", ondelete="SET NULL"))
    event_key: Mapped[str] = mapped_column(String(20))
    market_key: Mapped[str] = mapped_column(String(100))                 # taker: token id; maker: condition id
    kind: Mapped[str] = mapped_column(String(30))
    subject: Mapped[str | None] = mapped_column(Text)
    stage: Mapped[str] = mapped_column(String(20))
    dedupe: Mapped[str] = mapped_column(String(40))
    action: Mapped[str] = mapped_column(String(12))                      # buy | sell | quote | pull | fill
    side: Mapped[str] = mapped_column(String(8))                         # YES | NO (taker), bid | ask (maker)
    shares: Mapped[float | None] = mapped_column(Float)
    limit_price: Mapped[float | None] = mapped_column(Float)
    fair: Mapped[float | None] = mapped_column(Float)
    price: Mapped[float | None] = mapped_column(Float)                   # the market price the rule saw
    edge: Mapped[float | None] = mapped_column(Float)
    heat: Mapped[int | None] = mapped_column(Integer)                    # 1-3: modelled EV of the trade, qualitatively
    target_cost: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(14), default="new")       # new | alerted | expired | filled_paper
    run_id: Mapped[int | None] = mapped_column(ForeignKey("model_runs.id", ondelete="SET NULL"))
    signal_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    alerted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))   # the user opened Signals since
    detail: Mapped[dict | None] = mapped_column(JSONB)


class PaperPosition(Base):
    """A user's paper position in one market under their strategy profile, rebuilt on every signals run."""
    __tablename__ = "paper_positions"
    __table_args__ = (UniqueConstraint("user_id", "candidate_id", "market_key"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    candidate_id: Mapped[int | None] = mapped_column(Integer)
    race_id: Mapped[int | None] = mapped_column(ForeignKey("races.id", ondelete="SET NULL"))
    event_key: Mapped[str] = mapped_column(String(20))
    market_key: Mapped[str] = mapped_column(String(100))
    kind: Mapped[str] = mapped_column(String(30))
    subject: Mapped[str | None] = mapped_column(Text)
    yes_shares: Mapped[float] = mapped_column(Float, default=0.0)        # maker: YES-equivalent inventory
    no_shares: Mapped[float] = mapped_column(Float, default=0.0)
    cash: Mapped[float] = mapped_column(Float, default=0.0)               # paid (-) / received (+)
    mark: Mapped[float | None] = mapped_column(Float)                     # latest market price of YES
    outcome: Mapped[bool | None] = mapped_column()
    bid: Mapped[float | None] = mapped_column(Float)                      # maker: resting quotes now
    ask: Mapped[float | None] = mapped_column(Float)
    quote_state: Mapped[str | None] = mapped_column(String(30))           # quoting, or why not
    venue: Mapped[str] = mapped_column(String(20), default="polymarket", server_default="polymarket")  # polymarket | private


class LiveEvent(Base):
    """A live private-book event (pipelines/live.py), recorded when it is settled (racinglines live settle): its
    dates, the book's final P&L and the crowd's totals, so the demo accounts' story and the web app don't need
    the run folder. The positions themselves are in paper_positions (venue 'private')."""
    __tablename__ = "live_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    run: Mapped[str] = mapped_column(String(60), unique=True)             # the run folder: data/runs/live/<run>
    sport: Mapped[str] = mapped_column(String(20))
    event_key: Mapped[str] = mapped_column(String(40), index=True)        # paper_positions.event_key
    title: Mapped[str | None] = mapped_column(Text)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    maker_pnl: Mapped[float | None] = mapped_column(Float)                # vs the crowd + vs the demo taker
    crowd_pnl: Mapped[float | None] = mapped_column(Float)                # the maker's P&L vs the crowd
    taker_pnl: Mapped[float | None] = mapped_column(Float)                # the maker's P&L vs the demo taker
    fills: Mapped[int | None] = mapped_column(Integer)
    volume: Mapped[float | None] = mapped_column(Float)
    detail: Mapped[dict | None] = mapped_column(JSONB)                    # settings, crowd results, P&L curve
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DataChange(Base):
    """The data change log: one row per change to the race history the models read (an ingest that changed
    anything, an athlete merge, a manual note), so `racinglines db changes` answers "what changed, when, and
    why" without git or shell history. Written by `mtb_dh ingest`, `db merge-athletes` and `db changes --add`;
    docs/data-changes.md summarizes the major ones."""
    __tablename__ = "data_changes"
    id: Mapped[int] = mapped_column(primary_key=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    sport: Mapped[str | None] = mapped_column(String(20))                 # mtb_dh, f1; None for everything
    kind: Mapped[str] = mapped_column(String(30))                         # ingest | merge-athletes | note
    summary: Mapped[str] = mapped_column(Text)
    detail: Mapped[dict | None] = mapped_column(JSONB)                    # per-file statuses, ids moved, ...
    by: Mapped[str | None] = mapped_column(String(80))                    # user@host
