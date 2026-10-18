"""
A registered player's followed strategy, sized to their fantasy money (owner, 2026-10-12 19:47Z, decision card: "Size
followed strategies to the player's F$1,000": "Stakes scale with the balance and total spend is capped, so strategy and
manual P&L compare fairly"). Decision log: docs/f1-roadmap.md, 2026-10-12 "Followed strategies size to the player's
F$1,000".

The rule, for one player and the event being traded:

- strategy bankroll = the player's grant (F$1,000, the ledger's rows with no ref_table, as fantasy/ledger.summary
  counts it) + their hand trades' P&L + the strategy's own P&L on every *other* event (paper_positions, fantasy/ledger.STRATEGY_PNL's rule:
  settled at the outcome, open at the latest mark). The event being traded is left out, so a weekend's own marks
  never resize the weekend that made them (every pass replays the weekend from its first stage).
- scale = weekend_sweep.bankroll_scale(grant, bankroll): 1.0 at F$1,000, 1.5 after +500, 0 once it is gone (a busted
  strategy stops trading). A blend member's weight stays multiplied in.
- cap (TakerParams.max_deployed) = the bankroll less the cost still tied up in open hand trades and in the strategy's
  unsettled positions on other events, times a blend member's weight, so the open cost of everything it holds never exceeds the bankroll.

Kelly (owner, 2026-10-12 19:52Z / 19:53Z): a player may pick "kelly" with a fraction from 0.10 to 1.00 instead
(profiles.sizing_of reads it off their profile); the taker's existing Kelly sizing then runs with bankroll = the same
strategy bankroll, under the same cap (sized(kelly=...)).

One F$1,000 for both (owner, 2026-10-12 21:04Z: "Only 1k to use for both"): the bankroll also carries the player's
hand trades (accounts.fantasy_ledger, fantasy_positions): their P&L (cash + open positions at their links' last
price - the grant) is added to it, and the cost of their open positions is tied up like the strategy's own, so the
strategy never sizes or deploys money the player has already spent by hand. ledger.available is the other half:
hand trades can spend only what the strategy has not tied up.

Who: an account with a fantasy grant that is not a demo or system account. The demo accounts (web/demo.DEMO_USERS,
profiles.DEMO), backtests, sweeps, replays and golden tests never come here: player_bankroll is called only by the
live signal pass (signals.for_player), and it returns None for anyone else, which leaves the shared computation as is.
"""

from dataclasses import dataclass, replace

from sqlalchemy import text


@dataclass(frozen=True)
class StrategyBankroll:
    start: float            # the player's grant (F$1,000)
    pnl: float = 0.0        # hand trades' P&L + the strategy's P&L on other events
    open_cost: float = 0.0  # cost tied up in open hand trades and the strategy's unsettled positions on other events

    @property
    def balance(self):
        return self.start + self.pnl

    @property
    def scale(self):
        from racinglines.pipelines.weekend_sweep import bankroll_scale
        return bankroll_scale(self.start, self.balance)

    @property
    def cap(self):
        """F$ this event may have deployed at once: the balance less what other events still hold (never below 0)."""
        return max(self.balance - self.open_cost, 0.0)


def excluded_users():
    """Usernames never sized to fantasy money: the demo accounts and the reserved system names."""
    from racinglines.pipelines import profiles as PF
    from racinglines.web import accounts as ACC
    from racinglines.web import demo
    return set(PF.DEMO) | set(ACC.KEEP) | set(demo.DEMO_USERS) | set(ACC.RESERVED)


def player_bankroll(conn, user_id, event_key):
    """The followed strategy's bankroll for `user_id` trading `event_key`, or None when the account is not a fantasy
    player (no ledger, no grant, a demo or system account): None keeps the strategy's own sizing."""
    if not conn.execute(text("SELECT to_regclass('accounts.fantasy_ledger') IS NOT NULL")).scalar():
        return None
    name = conn.execute(text("SELECT username FROM users WHERE id = :u"), dict(u=user_id)).scalar()
    if name is None or name in excluded_users():
        return None
    grant = conn.execute(text("""SELECT sum(amount) FROM accounts.fantasy_ledger
                                 WHERE user_id = :u AND ref_table IS NULL"""), dict(u=user_id)).scalar()
    if grant is None or float(grant) <= 0:
        return None
    pnl, open_cost = conn.execute(text("""
        SELECT coalesce(sum(cash + yes_shares * y + no_shares * (1 - y)) FILTER (WHERE y IS NOT NULL), 0),
               coalesce(sum(greatest(-cash, 0)) FILTER (WHERE outcome IS NULL
                                                         AND (yes_shares <> 0 OR no_shares <> 0)), 0)
        FROM (SELECT cash, yes_shares, no_shares, outcome,
                     coalesce(CASE WHEN outcome IS NULL THEN NULL WHEN outcome THEN 1.0 ELSE 0.0 END, mark) AS y
              FROM paper_positions WHERE user_id = :u AND venue <> 'private' AND event_key IS DISTINCT FROM :e) p"""),
        dict(u=user_id, e=event_key)).one()
    from racinglines.fantasy import ledger as L
    hand = L.positions(conn, user_id, open_only=True)                   # marked at the links' last synced price
    hand_pnl = L.cash(conn, user_id) + sum(p["value"] or 0 for p in hand) - float(grant)
    hand_cost = sum(max(p["cost"] or 0, 0) for p in hand)
    return StrategyBankroll(start=float(grant), pnl=float(pnl) + hand_pnl, open_cost=float(open_cost) + hand_cost)


def sized(p, bank, kelly=None):
    """TakerParams `p` sized to a player's bankroll, max_deployed = bank.cap x a blend member's weight (p.scale; the
    tighter of it and the profile's own cap). bank None -> p unchanged.
    kelly None ("standard"): stakes x bank.scale, the weight kept (p.scale = weight x bank.scale).
    kelly a fraction (the player's choice, profiles.sizing_of, 0.10 to 1.00): the existing Kelly sizing
    (sizing.target_shares) with bankroll = the strategy bankroll (never below 0), so the stake is kelly x f* x bankroll
    x weight, still capped per market at max_stake x bank.scale x weight."""
    if bank is None:
        return p
    weight = p.scale
    cap = bank.cap * weight
    if p.max_deployed is not None:
        cap = min(cap, p.max_deployed)
    if kelly is None:
        return replace(p, scale=weight * bank.scale, max_deployed=cap)
    return replace(p, kelly=kelly, bankroll=max(bank.balance, 0.0), max_stake=p.max_stake * bank.scale,
                   max_deployed=cap)
