# Changelog

Every version on PyPI, newest first. Each one is a single commit and tag `vX` in this repository
(<https://github.com/racinglines/racinglines-public>), and that is the tree PyPI's build came from.

## 1.0.3 (2026-10-26)

- **Removed: ProCyclingStats (PCS).** racinglines no longer documents, recommends or supports fetching road-cycling
  results from ProCyclingStats. The quickstart's data-sources table, the CLI reference and the road-cycling page drop it,
  along with the extra packages that page told you to install for it. No released version shipped the PCS client, and
  none ships any scraping or anti-bot tool. The release export now fails if such a tool is named anywhere in the
  package or its docs.
- The downhill downloader (`racinglines data download mtb_dh`) identifies itself as `racinglines` with this
  repository's URL instead of a browser-style user agent.
- `racinglines remote` and `--remote`: when the hosted server answers 401 with a message, such as the sign-up prompt
  after the demo's free calls, the CLI prints that message instead of the generic "401 Unauthorized" line.
- Fantasy (F$): strategy picks stay open until lights out and close at the race start. A late pick trades only the
  stages priced after it, never a session that already ran. A player with no pick gets no trades: the random-style
  fallback and `racinglines users assign-random` are gone. Picking from a popup, an admin Pick locks page, an
  "available to follow" flag per style, and a pre-race stage 4 h before lights out for every player. Live strategy
  orders walk the recorded book at each stage, and a followed strategy has no per-market stake cap (bankroll, edge
  and sizing only). The leaderboard score is ROI × (trades / 10)^0.3. A by-hand ledger adjustment moves the balance
  only. F-bucks show as Ⓕ.
- Sign-in matches the username ignoring case, or a unique display name or email.
- Order confirmations say when an order was checked against the venue's live book.
- Models: every F1 live-book market is priced. Saved forecast runs record `calib_power`, which stays `off` by default
  in the package. New opt-in pieces: the `gridnoise` variant, `--trades-out` (every sweep trade to Parquet), a
  teammate head-to-head kind in walk-forward, and `racinglines eval compare`, which pairs two saved runs by event
  with ±2 SE. Default prices are unchanged. The package prices with the one engine (`model_global`) and its laps,
  run-times and results cores; the Plackett-Luce challenger stays out.
- The Markets board prices from the live default's runs only, not the newest live run of any profile.
- The web app serves `robots.txt`, `sitemap.xml` and meta tags, with private pages marked noindex.

## 1.0.2 (2026-10-19)

Install in steps now: `racinglines[db]` runs every command, `racinglines[mcp]` adds the MCP server, and
`racinglines[web]` adds the web app, last, with its notice. The README follows that order.

- **New extra `[db]`**: the database, the kart demo, backtests and forecasts, the live book and its reports,
  sportsbook pricing, FastF1 and the downhill sim, with no web packages. `[web]` is now `[db]` plus the web server.
- **New extra `[mcp]`**: `racinglines mcp` (44 tools over stdio or HTTP), its per-account tokens
  (`racinglines mcp token`), the hosted-server client (`racinglines remote`, `--remote`) and `racinglines data pull`.
  It is `[db]` plus `mcp`, with no web packages.
- A missing package now names the extra to install (`pip install "racinglines[db]"`), and the database hints point
  at the README's `docker run` line.
- Removed: the `/mybets` page, the old engine classes behind `--model f1_sector_sim` and `--model timed_runs`, and the
  deprecated import paths `racinglines.models.position_sim` and `racinglines.models.timed_runs` (use
  `racinglines.models.inputs.*`). F1 and downhill price on the same laps and run-times cores as before, so prices are
  unchanged.
- Fixed:
  - `racinglines web -h` printed the server's startup instead of help; `web` with no database now prints the
    first-run text.
  - `book render` failed because `markets/books.toml` wasn't in the package.
  - `live new` wrote its spec into site-packages; it now writes under the data folder.
  - `model props f1` crashed on an event with no history, and F1 model commands on an empty database ended in
    `No objects to concatenate`; both now say what to run.
  - A command that needs a module the package leaves out prints one line instead of a traceback.
  - The MCP tool `get_diagnostic` failed on a run with an unpriced market.
  - The web app's MCP settings routes and `/markets/og` returned 500 in the package.
  - The Markets page's taker call now nets the venue's taker cost and fee, as the live pass does.
  - The in-race chart takes the race distance from the race's `total_laps` (else 56).
  - Resetting a demo history deletes only that venue's paper positions.
  - `live agent` from an installed package runs the installed `racinglines` from the data home.
  - Quiet Polymarket books no longer read as stale: fantasy quotes fall back to the venue's live book when the recorded
    one is over 15 minutes old, and `book price` prices from the newest recorded book and flags a market price over 3 h old.
- Security: the web app trusts `cf-connecting-ip` / `x-forwarded-for` only from a loopback peer (a tunnel or proxy on
  the same machine), or from any peer with `RACINGLINES_TRUST_PROXY=1`; the failed-login table is bounded.
- A self-hosted app shows no racinglines.bet support address, board stamp or Pitch link; `/forgot` names
  `racinglines users reset-password`. Live reports find Chromium the same way books do (`RACINGLINES_CHROME`, PATH,
  then the usual install paths).
- Removed the unused `/live/f1/stream` page, its websocket and markets API, and the dead Coinbase driver paths.
- Docs: README step 10 and the quickstart say to sign up on racinglines.bet before the hosted MCP steps (`--remote` needs
  the token from that sign-in; the hosted demo store may allow a few calls without one first). The local `racinglines
  mcp --demo` needs no account.
- Docs: the live viewer is admin-only (`/live`), `RACINGLINES_DEMO_DATABASE_URL` is documented, the quickstart installs
  from PyPI, and the repository has a `CONTRIBUTING.md`. PyPI links the source repository, and the README carries the
  release workflow's status badge. Docs and source links point at the current repositories.

## 1.0.1 (2026-10-19)

- Published from this repository: PyPI's provenance for 1.0.1 names `racinglines/racinglines-public` and the tag `v1.0.1`.
- The web app's GitHub links (the nav and the footer) point at this repository.
- The racinglines 101 page links its two docs pages (F1 calibration, F1 evaluation) at <https://racinglines.bet/docs>
  instead of relative paths that led nowhere in an installed package.
- README: the from-source steps clone this repository and `pip install -e ".[web]"`.

## 1.0.0 (2026-10-18)

First release.

- The F1 pricing engine and the `racinglines` CLI, the UCI downhill sim and the MK8DX (kart) demo.
- `pip install "racinglines[web]"`: the web app (Markets, Trade, History, Lab, the live viewer) with F$ fantasy trading,
  its database schema and migrations.
- `racinglines web --demo` creates the demo accounts (Try the demo signs in as `maker`); `/` is the sign-in page.
