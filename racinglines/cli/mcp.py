"""
racinglines mcp                          the MCP server over stdio (docs/mcp.md)
racinglines mcp --http [--port 8100]     streamable HTTP on 127.0.0.1:8100/mcp; each request needs an account's token
racinglines mcp token ACCOUNT [--revoke] issue (print once) or revoke an account's bearer token; `token` alone lists holders
"""

import argparse
import os


def main(argv=None):
    argv = list(argv or [])
    if argv and argv[0] == "token":
        return _token(argv[1:])
    ap = argparse.ArgumentParser(prog="racinglines mcp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="database URL (default $DATABASE_URL, else the local one)")
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP (per-account bearer tokens) instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--no-jobs", action="store_true", help="don't run the Lab's job worker here (leave queued jobs to the web app)")
    a = ap.parse_args(argv)
    try:
        import mcp  # noqa: F401
    except ImportError:
        print("The MCP server needs the `mcp` package: pip install \"racinglines[mcp]\" (or pip install -r requirements.txt in a checkout)")
        return 2
    if a.db:
        os.environ["DATABASE_URL"] = a.db
    from racinglines.mcp.server import serve
    serve(http=a.http, host=a.host, port=a.port, jobs_worker=not a.no_jobs)
    return 0


def _token(argv):
    ap = argparse.ArgumentParser(prog="racinglines mcp token")
    ap.add_argument("account", nargs="?", help="the web-app account (a real one: admin, or a maker once RACINGLINES_MCP_ROLES allows)")
    ap.add_argument("--revoke", action="store_true")
    ap.add_argument("--db")
    a = ap.parse_args(argv)
    from racinglines.db.config import get_engine
    from racinglines.mcp import auth
    eng = get_engine(a.db)
    if not a.account:
        rows = auth.holders(eng)
        print("\n".join(f"{u:20} {r:8} issued {t}" for u, r, t in rows) if rows else "no account has a token")
        return 0
    if a.revoke:
        print(f"revoked {a.account}" if auth.revoke(eng, a.account) else f"{a.account} had no token")
        return 0
    try:
        tok = auth.new_token(eng, a.account)
    except ValueError as ex:
        print(ex)
        return 2
    print(f"{a.account}'s MCP token (shown once; any earlier token is now void):\n{tok}\n"
          f"Clients send it as `Authorization: Bearer {tok[:8]}...`")
    return 0


if __name__ == "__main__":
    main()
