"""racinglines users  ->  web-app accounts and the fantasy-bucks ledger from the command line (web/accounts.py).

    racinglines users setup                         the accounts schema, ledger and manual-trading tables
                                                    (idempotent; fantasy/ledger.py); every active account
                                                    without its 1,000 signup grant gets it
    racinglines users list                          accounts, role, active, fantasy bucks
    racinglines users add NAME --role basic|pro|admin   a new account; prints a one-time password
    racinglines users reset-password NAME           a new one-time password, printed once
    racinglines users remove NAME                   delete an account with no markets or bets (else deactivate)
    racinglines users deactivate NAME | activate NAME
    racinglines users reset-fantasy [--keep maker,taker] [--apply]
                                                    every active account but --keep back to one 1,000 grant and no
                                                    trading history (paper positions, signals, house bets) and no
                                                    strategy profile; counts
                                                    only without --apply (on the VM: scripts/vm/reset_fantasy.sh)
    racinglines users season-demo [--seed N] [--year 2026] [--apply]
                                                    every fantasy player follows a random alpha style at a random
                                                    weekend share and follow rate through the season so far, replayed
                                                    on the real tapes (fantasy/season_demo.py); a database whose name
                                                    contains "demo" only; the draw only without --apply (on the VM:
                                                    scripts/vm/season_demo.sh)

Passwords are never read back: they are stored as scrypt hashes, and a reset prints a fresh one exactly once.
"""

import argparse

from sqlalchemy import text


def main(argv=None):
    ap = argparse.ArgumentParser(prog="racinglines users", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    sub.add_parser("list")
    a = sub.add_parser("add")
    a.add_argument("name")
    a.add_argument("--role", default="basic", choices=("basic", "pro", "admin"))
    r = sub.add_parser("reset-fantasy")
    r.add_argument("--keep", default="maker,taker", help="accounts left as they are (comma-separated)")
    r.add_argument("--apply", action="store_true", help="write; without it the command only counts")
    r.add_argument("--note", default="reset", help="the new grant's ledger note")
    d = sub.add_parser("season-demo")
    d.add_argument("--seed", type=int, default=2026, help="the draw's seed (default 2026): same seed, same demo")
    d.add_argument("--year", type=int, default=2026)
    d.add_argument("--apply", action="store_true", help="write; without it the command only prints the draw")
    for cmd in ("reset-password", "remove", "deactivate", "activate"):
        sub.add_parser(cmd).add_argument("name")
    args = ap.parse_args(argv)

    from racinglines.db.config import get_engine, get_session
    from racinglines.web import accounts as ACC
    from racinglines.web import users as U
    eng = get_engine()

    if args.cmd == "setup":
        n = ACC.setup(eng)
        print(f"accounts.fantasy_ledger and the fantasy trading tables ready; {n} signup grants of {ACC.SIGNUP_GRANT} added")
        return 0
    if args.cmd == "reset-fantasy":
        keep = tuple(k.strip() for k in args.keep.split(",") if k.strip())
        with (eng.begin() if args.apply else eng.connect()) as c:
            if not ACC.ready(c):
                print("no accounts ledger yet: racinglines users setup")
                return 1
            rows = ACC.reset(c, keep=keep, apply=args.apply, note=args.note)
        cols = [t for t, _ in ACC.HISTORY]
        print(f"{'id':>5}  {'username':<30} {'action':<7} {'ledger':>6} {'balance':>10} {'profiles':>8} "
              + " ".join(f"{t:>17}" for t in cols))
        for r in rows:
            act = "keep" if r["kept"] else ("reset" if args.apply else "would")
            print(f"{r['id']:>5}  {r['username']:<30} {act:<7} {r['ledger_rows']:>6} {r['balance']:>10,.0f} {r['profiles']:>8} "
                  + " ".join(f"{r[t]:>17,}" for t in cols))
        n = sum(not r["kept"] for r in rows)
        print(f"{'reset' if args.apply else 'dry run: would reset'} {n} accounts to {ACC.SIGNUP_GRANT}, no history and no strategy; "
              f"kept {', '.join(keep) or 'none'}" + ("" if args.apply else " (--apply writes)"))
        return 0
    if args.cmd == "season-demo":
        from racinglines.db.config import database_url
        from racinglines.fantasy import season_demo as SD
        try:
            _, rep = SD.build(eng, database_url(), seed=args.seed, year=args.year, apply=args.apply,
                              echo=lambda m: print(m, flush=True))
        except ValueError as e:
            print(e)
            return 1
        for u in sorted({r[0] for r in rep}):
            rs = [r for r in rep if r[0] == u]
            print(f"{u:<24} {rs[0][1]:3s} {len(rs):2d} weekends, {sum(r[3] for r in rs):4d} picks taken, "
                  f"paper P&L F${sum(r[4] for r in rs):+,.2f}")
        return 0
    if args.cmd == "list":
        with eng.connect() as c:
            bal = ACC.balances(c)
            for i, name, role, active in c.execute(text("SELECT id, username, role, active FROM users ORDER BY id")):
                print(f"{i:>5}  {name:<30} {U.canonical(role):<6} {'active' if active else 'inactive':<8} "
                      f"{bal.get(i, 0):>10,.0f}")
        return 0

    with get_session() as s:
        u = U.get_user(s, username=args.name)
        if args.cmd == "add":
            temp = ACC.temp_password()
            u = U.create_user(s, args.name, temp, args.role)
            if ACC.ready(s.connection()):
                ACC.grant(s.connection(), u.id, note="cli add")
                s.commit()
            U.log(eng, None, "user_create", username=args.name, role=args.role, via="cli")
            print(f"created {args.role} {args.name}; one-time password (not shown again): {temp}")
            return 0
        if u is None:
            print(f"no user {args.name!r}")
            return 1
        if args.cmd == "reset-password":
            temp = ACC.temp_password()
            u.password_hash = U.hash_password(temp)
            s.commit()
            U.log(eng, None, "user_password_reset", username=args.name, via="cli")
            print(f"new one-time password for {args.name} (not shown again): {temp}")
        elif args.cmd in ("deactivate", "activate"):
            u.active = args.cmd == "activate"
            s.commit()
            U.log(eng, None, f"user_{args.cmd}", username=args.name, via="cli")
            print(f"{args.name}: {args.cmd}d")
        elif args.cmd == "remove":
            history = s.execute(text("""SELECT (SELECT count(*) FROM house_markets WHERE maker_id = :u)
                                             + (SELECT count(*) FROM house_bets WHERE taker_id = :u)"""),
                                dict(u=u.id)).scalar()
            if history:
                print(f"{args.name} has markets or bets: not removed (racinglines users deactivate {args.name})")
                return 1
            s.delete(u)
            s.commit()
            U.log(eng, None, "user_delete", username=args.name, via="cli")
            print(f"removed {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
