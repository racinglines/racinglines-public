"""racinglines web  ->  the web app on http://127.0.0.1:8000 (WEB_HOST / WEB_PORT to change)."""

import argparse
import os

import uvicorn

SHUTDOWN_GRACE_SEC = 10

# Printed when the app starts from an installed package (README step 9 says the same): what this copy is, and what
# has to change before it goes on a public address.
NOTICE = """\
NOTICE: this is your own copy of the racinglines web app, running on your machine. It is not racinglines.bet, and
  accounts here have nothing to do with that site. The code is MIT-licensed (see LICENSE): use it, change it, host it.
  The racinglines name and branding (the logo in racinglines/web/static/brand/, the wordmark on the sign-in page,
  the page titles and the top bar) are (c) 2026 Kevin Lalli and not covered by that licence: replace them before
  you put this app on any public link."""


HELP = """racinglines web [--demo]: serve the web app on http://127.0.0.1:8000.

    WEB_HOST / WEB_PORT      the address (default 127.0.0.1:8000)
    ADMIN_USERNAME / ADMIN_PASSWORD   the admin account (default admin; with no password, the first start
                             creates one and prints it once)
    RACINGLINES_FANTASY=1    F$ fantasy trading on
    RACINGLINES_TRUST_PROXY=1   trust cf-connecting-ip / x-forwarded-for from any peer (a reverse proxy that isn't
                             on this machine; from loopback they are always trusted)
    --demo [PACK]            the app on the demo store (the kart pack's synthetic data; docs/quickstart.md)"""


def main(argv=None):
    argparse.ArgumentParser(prog="racinglines web", description=HELP,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv or [])
    from racinglines.db import config as DBC
    if DBC.fallback and not os.environ.get("DATABASE_URL"):
        DBC.database_url()                        # the first-run choice now, not as a traceback in the app's startup
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "8000"))
    from racinglines.paths import CHECKOUT
    if not CHECKOUT:
        print(NOTICE, flush=True)
    if host not in ("127.0.0.1", "localhost"):
        print(f"WARNING: listening on {host}. The app can place exchange orders; keep it behind a private network or the tunnel.")
    # On SIGTERM wait at most this long for in-flight requests (uvicorn's default is forever, which held a restart until
    # systemd's 90 s stop timeout when a client kept a connection open); the unit's TimeoutStopSec stays above it.
    uvicorn.run("racinglines.web.app:app", host=host, port=port, reload=os.environ.get("WEB_RELOAD") == "1",
                timeout_graceful_shutdown=SHUTDOWN_GRACE_SEC)


if __name__ == "__main__":
    main()
