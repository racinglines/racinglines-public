# racinglines

[![PyPI](https://img.shields.io/pypi/v/racinglines)](https://pypi.org/project/racinglines/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)
[![release](https://github.com/racinglines/racinglines-public/actions/workflows/release.yml/badge.svg)](https://github.com/racinglines/racinglines-public/actions/workflows/release.yml)

Fair prices for Formula 1 prediction markets. racinglines turns official timing into win, podium, head-to-head and
championship odds, priced strictly as of any moment (a price never sees data from after its own time), and gives you
the tools to test trading strategies against those prices: backtests, season forecasts, a live private book that
reprices after every session and trades against simulated bettors, and a web app with fantasy trading and a live
viewer. The same engine runs a UCI downhill mountain-bike sim: season forecasts, backtests and a live final.

You can try all of it on your own machine with no account, no API key and no real money. The **kart demo** is a
synthetic F1 season with a Mario Kart grid (Mario, Luigi, Peach, Bowser and friends, racing on Rainbow Road and
Bowser's Castle). Every lap, result and price in it is invented, and the commands are the same ones that run on real
F1 data.

## How releases work

- **What you install is what you can read.** Every version on PyPI is built from a tagged commit in the public source repository, so `pip install racinglines==X` matches tag `vX` line for line.
- **Releases come now and then, not every day.** Day-to-day development happens elsewhere. Each release is one commit and tag in [racinglines-public](https://github.com/racinglines/racinglines-public), the same tree PyPI built ([CHANGELOG](https://github.com/racinglines/racinglines-public/blob/main/CHANGELOG.md)).
- **Contributions are welcome.** Issues and pull requests go to racinglines-public. Accepted changes are carried into the next release ([CONTRIBUTING](https://github.com/racinglines/racinglines-public/blob/main/CONTRIBUTING.md)).
- **Docs:** while the `racinglines[docs]` install is in the works, the documentation is live at https://racinglines.bet/docs.
- **Quickstart:** watch the demo below run in your browser at https://racinglines.bet/term.

## Two ways in

| | What you get | What you need |
|---|---|---|
| **Quick check** (2 minutes) | The engine and the command line, run on synthetic data | Python 3.11+ |
| **Everything** (about 15 minutes) | A local database, the kart demo, backtests, forecasts, a live private book, the downhill sim, the MCP server and, last, the web app with fantasy trading and its live viewer | Python 3.11+, Docker |

### Quick check: `pip install racinglines`

```bash
python3 --version                                  # 3.11 or newer; on an older Python pip says "no matching distribution"
python3 -m venv ~/rl-try && source ~/rl-try/bin/activate
pip install racinglines
racinglines --version
racinglines check --sport f1 --offline --no-db     # the pipelines on synthetic data: no network, no database
```

The last command should end with `9/9 checks passed`. That install is the engine and the command line only. For
the database, the demo, the live book and the MCP server, install `racinglines[db]` as below. The web app comes last
(step 11), with its own install and notice.

## Everything: database, demo, MCP server and the web app

Run these in order, one block at a time; each says what you should see. Steps 2, 4 to 8, 10 and 11 were run end to end from
the built package in a fresh virtualenv, with an empty home folder, against a PostgreSQL like the one step 3 starts.

### 1. What you need

- **Python 3.11 or newer.** Check with `python3 --version`. On a Mac the built-in `/usr/bin/python3` is too old;
  install a newer one from [python.org](https://www.python.org/downloads/) or with `brew install python@3.12`.
- **Docker** (Docker Desktop on a Mac or Windows) for the database. Any PostgreSQL 16+ server works instead; see
  step 3.

### 2. Install

```bash
python3 -m venv ~/rl && source ~/rl/bin/activate
pip install "racinglines[db]"
racinglines --version
```

The quotes matter in zsh (the Mac's shell). `[db]` adds the database, timing and report packages: every step up to
10 runs on it. The web app's packages come in step 11. Keep this terminal open: every later step runs with the virtualenv active (`source ~/rl/bin/activate`
again in a new terminal).

### 3. Start the database

```bash
docker run -d --name racinglines-db -p 5433:5432 -e POSTGRES_USER=racinglines -e POSTGRES_PASSWORD=racinglines \
  -e POSTGRES_DB=racinglines -v racinglines-pgdata:/var/lib/postgresql/data postgres:17
```

That starts PostgreSQL on `localhost:5433` (port 5433 so it doesn't clash with a Postgres you may already run on
5432), with its data kept in the Docker volume `racinglines-pgdata`. Give it a few seconds, then tell racinglines to
use it and create its tables:

```bash
racinglines book config source local
racinglines db init
```

`db init` ends with `Reference data seeded.` No Docker? Point `DATABASE_URL` at your own server instead, for example
`export DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/racinglines`. The user needs the right to
create databases, because the demo makes its own next to it. The server must use UTF-8 encoding.

### 4. Check the install

```bash
racinglines check --offline
```

It runs the F1 and downhill pipelines on synthetic data and checks the database, in about 5 seconds. It should end
with `14/14 checks passed`.

### 5. Build the kart demo and look at it

```bash
racinglines db stats --demo
```

`--demo` on any command runs that same command on the **demo store**: a separate database (`racinglines_demo_kart`)
and data folder (`~/.racinglines/data/demo/kart/`) filled with the kart world. The first `--demo` command builds it, with a progress
bar (under a minute here). Then it prints what's stored: events, sessions, results and market links per season.
Later commands reuse it. The demo commands below name the sport `kart`: that is the demo's own name for its F1
world (stored under the F1 schema), so `racinglines -h` lists only `f1` and `mtb_dh` as sports.

### 6. Backtest the model and forecast the season

```bash
racinglines model backtest kart --demo
racinglines model forecast kart --demo --save
```

The backtest prices every past race as of before the weekend, before qualifying and before the race, then scores
those prices against what happened (about 30 seconds). It ends with a best call and the biggest upset, and saves the
per-race rows as a CSV. The forecast prices the races still to come and both championships, and `--save` stores it
for the web app (`Saved forecast run ...`).

### 7. Run a live private book

```bash
racinglines live run kart/demo-2026-09 --demo --simulate --no-fetch
```

This replays one race weekend, the Bowser's Castle Grand Prix, on a simulated clock (about 30 seconds). A market maker
opens a book before practice, reprices it after every session from the model, and 100 simulated bettors trade against
it until the book closes at lights out. Then it settles from the results. It ends with a summary box: the fills, the
money traded, and how many bettors finished up or down.

Afterwards:

```bash
racinglines live status --demo
racinglines live report kart/demo-2026-09 --demo
```

`live status` lists the event as `settled`. `live report` writes the event report (Markdown and charts) under
`~/.racinglines/data/demo/kart/runs/live/demo-2026-09/report/`.

### 8. Price a sportsbook's lines

```bash
racinglines book price --demo
```

The demo carries a sample sportsbook's lines on two upcoming races. This prints each leg with the model's fair price,
the exchange price, the book's implied probability and the expected value against each. The last line, `typo`
(`Maryo`), is misspelt on purpose: it shows under **unmapped lines**, because names are matched exactly, never
guessed.

### 9. The downhill sim

`racinglines check` in step 4 already ran the downhill model, forecast and backtest on synthetic data. On real UCI
World Cup results it needs the network: `racinglines mtb_dh -h` lists its commands (download, parse, ingest,
forecast, backtest, walk-forward, points, live), and `racinglines data download mtb_dh -h` explains how results are found on
ChronoRace. No results ship with the package.

### 10. Ask Claude: the MCP server

> **Before the hosted MCP steps: sign up on [racinglines.bet](https://racinglines.bet/signup).** The hosted server
> (`racinglines remote`, `--remote` on any command, and Claude connected to racinglines.bet) needs a token, and that
> comes from your racinglines.bet sign-in: the first `--remote` command opens the browser to sign in, or copy an `rl_`
> token from Settings > Connect Claude (MCP) into `RACINGLINES_MCP_TOKEN`. The hosted demo store (`--demo --remote`) may
> let you try a few calls without a token first; after that it asks you to sign up. The local server below
> (`racinglines mcp --demo` over your own database) needs no account.

```bash
pip install "racinglines[mcp]"
claude mcp add racinglines -- ~/rl/bin/racinglines mcp --demo
```

`[mcp]` is `[db]` plus the MCP server, with no web packages. `racinglines mcp` serves 44 tools over stdio: events, markets and fair
prices, model runs and forecasts, the live events and read-only `sql`; its fantasy-trading tools work once step 11 has set
up fantasy trading. It runs on the same
database as the commands above, and `--demo` puts it on the demo store. The second line adds it to Claude Code. In
any other MCP client, run `~/rl/bin/racinglines mcp --demo` as a stdio server. Then ask, for example, "what does the
model give Luigi to win the next race, against the exchange?". The setup for Claude Desktop, the hosted server and
`racinglines remote` is at <https://racinglines.bet/docs/mcp/>.

### 11. Last: the web app and its live viewer

> **Read this first.** This runs your own copy of the racinglines web app on your machine. It is not
> [racinglines.bet](https://racinglines.bet), and accounts here have nothing to do with that site. The code is
> MIT-licensed ([LICENSE](LICENSE)): use it, change it, host it. The racinglines name and branding (the logo in
> `racinglines/web/static/brand/`, the wordmark on the sign-in page, the page titles and the top bar) are
> © 2026 Kevin Lalli and not covered by that licence: replace them before you put this app on any public link.
> `racinglines web` prints the same notice when it starts.

```bash
pip install "racinglines[web]"
RACINGLINES_FANTASY=1 ADMIN_PASSWORD=choose-one racinglines web --demo
```

`[web]` is `[db]` plus the web server and its templates.

Pick your own password in place of `choose-one`. The app runs at <http://127.0.0.1:8000> on your machine only; stop
it with Ctrl+C. With `--demo` it sets up the accounts and fantasy-trading tables on first start and creates the demo
accounts, so nothing else is needed before you sign in.

1. Open <http://127.0.0.1:8000> and click **Try the demo** (the demo account, `maker`, a pro). The top bar has four
   tabs: Markets, Trade, History and Lab.
2. **Markets** shows the forecast you saved in step 6, and **Lab** has the backtests and strategy tools.
3. **Trade** and **History** are fantasy trading with F$ 1,000 of play money (`RACINGLINES_FANTASY=1` turns them
   on). Trade lists exchange markets with a book recorded in the last day; the demo's recorded books are older, so
   in the demo it says `No F1 markets with a recorded book in the last day.`
4. **The live viewer is admin-only** and has no tab in the top bar. Click **Log out**, sign in as `admin` with the
   password you chose (a box asks for an email so you don't get locked out: click **Skip for now**, nothing leaves
   your machine), then open <http://127.0.0.1:8000/live>. That's the race you ran in step 7: the winner, the maker's
   settled P&L, the crowd's fills and volume, and a **Replay** bar to step or play through the weekend's seven
   updates as they were seen live. Signed in as any other account, `/live` answers 404.

To add more accounts, by role, stop the app (Ctrl+C) and run, for each one:

```bash
racinglines users add alice --role basic --demo
```

`--role` is `admin` (everything, plus user management, the database explorer and the live viewer), `pro` (fair
prices, quotes, every strategy and the Lab) or `basic`. With fantasy trading on (`RACINGLINES_FANTASY=1`, as above),
a `basic` account gets the same pages as `pro`. Each prints a one-time password; `racinglines users list --demo` shows who exists. Then start the app again as above.

### Where things are, and starting over

| What | Where |
|---|---|
| The main database | `racinglines` on `localhost:5433` (the Docker volume `racinglines-pgdata`) |
| The demo database | `racinglines_demo_kart`, on the same server, reused if it already exists (`RACINGLINES_DEMO_DATABASE_URL` points it elsewhere) |
| The demo's files, runs and reports | `~/.racinglines/data/demo/kart/` |
| Your settings | `~/.config/racinglines/cli.toml` |

To rebuild the demo from scratch, delete `~/.racinglines/data/demo/kart/demo-store.json` and run any `--demo` command
again. To remove everything: `docker rm -f racinglines-db && docker volume rm racinglines-pgdata`, then delete
`~/.racinglines` and the virtualenv `~/rl`.

### If something goes wrong

| What you see | What it means |
|---|---|
| `No matching distribution found` during `pip install` | Your Python is too old. Check `python3 --version`, then make the virtualenv with a newer one (step 1) |
| `Requirement already satisfied` with an old version | `pip` belongs to another environment (conda, for example). Activate this virtualenv first, or run `~/rl/bin/pip` |
| `zsh: no matches found: racinglines[db]` | Quote it: `pip install "racinglines[db]"` |
| `The MCP server needs the mcp package` | Install the extra: `pip install "racinglines[mcp]"` |
| `command not found: racinglines` | The virtualenv isn't active in this terminal: `source ~/rl/bin/activate` |
| `needs psycopg, which the core install leaves out` (or another package) | The plain install is missing an extra: the message names it, for example `pip install "racinglines[db]"` |
| `racinglines needs a database, and none is configured` | Step 3 was skipped: run `racinglines book config source local` |
| `demo: no Postgres server for ...` | The database isn't running: `docker start racinglines-db`, then `docker ps` should list it |
| `check` fails on `alembic_version` | The tables don't exist yet: `racinglines db init` |

### From source (contributors)

```bash
git clone https://github.com/racinglines/racinglines-public.git && cd racinglines-public
python3 -m venv .venv-rl && source .venv-rl/bin/activate
pip install -e ".[web]"                                 # Python 3.11+; [web] includes [db]; the database is step 3's container
```

Then steps 3 (`book config` and `db init`) to 11 as above; the demo's files go under `data/demo/kart/` in the checkout.

## More

- Documentation: https://racinglines.bet/docs (the quickstart, the model, market making, the CLI reference)
- The same demo, recorded, in your browser: https://racinglines.bet/term
- `racinglines <group> -h` lists any command group's commands and options.

MIT licensed.
