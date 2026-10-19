# racinglines

[![PyPI](https://img.shields.io/pypi/v/racinglines)](https://pypi.org/project/racinglines/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

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
- **Releases come now and then, not every day.** Day-to-day development happens elsewhere, and the public repository is updated in occasional releases. Each one lands as a reviewed pull request that follows the same contribution guidelines we ask of everyone.
- **Contributions are welcome.** Open a pull request against the public repository. Accepted changes are carried into the next release.
- **Docs:** while the `racinglines[docs]` install is in the works, the documentation is live at https://racinglines.bet/docs.
- **Quickstart:** watch the demo below run in your browser at https://racinglines.bet/term.

## Two ways in

| | What you get | What you need |
|---|---|---|
| **Quick check** (2 minutes) | The engine and the command line, run on synthetic data | Python 3.11+ |
| **Everything** (about 15 minutes) | A local database, the kart demo, backtests, forecasts, a live private book, the web app with fantasy trading and its live viewer, the downhill sim | Python 3.11+, Docker |

### Quick check: `pip install racinglines`

```bash
python3 --version                                  # 3.11 or newer; on an older Python pip says "no matching distribution"
python3 -m venv ~/rl-try && source ~/rl-try/bin/activate
pip install racinglines
racinglines --version
racinglines check --sport f1 --offline --no-db     # the pipelines on synthetic data: no network, no database
```

The last command should end with `9/9 checks passed`. That install is the engine and the command line only. For
the database, the demo, the live book and the web app, install `racinglines[web]` as below.

## Everything: database, demo and live viewer

Run these in order, one block at a time; each says what you should see. Steps 2 and 4 to 9 were run end to end from
the built package in a fresh virtualenv, with an empty home folder, against a PostgreSQL like the one step 3 starts.

### 1. What you need

- **Python 3.11 or newer.** Check with `python3 --version`. On a Mac the built-in `/usr/bin/python3` is too old;
  install a newer one from [python.org](https://www.python.org/downloads/) or with `brew install python@3.12`.
- **Docker** (Docker Desktop on a Mac or Windows) for the database. Any PostgreSQL 16+ server works instead; see
  step 3.

### 2. Install

```bash
python3 -m venv ~/rl && source ~/rl/bin/activate
pip install "racinglines[web]"
racinglines --version
```

The quotes matter in zsh (the Mac's shell). `[web]` adds the database, web and timing packages; nothing else needs
installing. Keep this terminal open: every later step runs with the virtualenv active (`source ~/rl/bin/activate`
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
Later commands reuse it.

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
the exchange price, the book's implied probability and the expected value against each.

### 9. Open the web app and the live viewer

> **Read this first.** This runs your own copy of the racinglines web app on your machine. It is not
> [racinglines.bet](https://racinglines.bet), and accounts here have nothing to do with that site. The code is
> MIT-licensed ([LICENSE](LICENSE)): use it, change it, host it. The racinglines name and branding (the logo in
> `racinglines/web/static/brand/`, the wordmark on the sign-in page, the page titles and the top bar) are
> © 2026 Kevin Lalli and not covered by that licence: replace them before you put this app on any public link.
> `racinglines web` prints the same notice when it starts.

```bash
RACINGLINES_FANTASY=1 ADMIN_PASSWORD=choose-one racinglines web --demo
```

Pick your own password in place of `choose-one`. The app runs at <http://127.0.0.1:8000> on your machine only; stop
it with Ctrl+C. With `--demo` it sets up the accounts and fantasy-trading tables on first start and creates the demo
accounts, so nothing else is needed before you sign in.

1. Open <http://127.0.0.1:8000> and click **Try the demo** (the demo account, `maker`, a pro). Or sign in as `admin`
   with the password you chose; a box then asks for an email so you don't get locked out: click **Skip for now**,
   nothing leaves your machine.
2. Click **Live** in the top bar. That's the live viewer for the race you ran in step 7: the winner, the maker's
   settled P&L, the crowd's fills and volume, and a **Replay** bar to step or play through the weekend's seven
   updates as they were seen live.
3. **Markets** shows the forecast you saved in step 6, and **Lab** has the backtests and strategy tools.
4. **Trade** and **History** are fantasy trading with F$ 1,000 of play money (`RACINGLINES_FANTASY=1` turns them
   on). Trade lists exchange markets with a book recorded in the last day; the demo's recorded books are older, so
   in the demo it opens with no markets.

To add more accounts, by role, stop the app (Ctrl+C) and run, for each one:

```bash
racinglines users add alice --role basic --demo
```

`--role` is `admin` (everything, plus user management and the database explorer), `pro` (fair prices, quotes,
every strategy and the Lab) or `basic` (open markets, a few taker strategies, their own bets). Each prints a one-time
password; `racinglines users list --demo` shows who exists. Then start the app again as above.

### 10. The downhill sim

`racinglines check` in step 4 already ran the downhill model, forecast and backtest on synthetic data. On real UCI
World Cup results it needs the network: `racinglines mtb_dh -h` lists its commands (download, parse, ingest,
forecast, backtest, points, live), and `racinglines data download mtb_dh -h` explains how results are found on
ChronoRace. No results ship with the package.

### Where things are, and starting over

| What | Where |
|---|---|
| The main database | `racinglines` on `localhost:5433` (the Docker volume `racinglines-pgdata`) |
| The demo database | `racinglines_demo_kart`, on the same server |
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
| `zsh: no matches found: racinglines[web]` | Quote it: `pip install "racinglines[web]"` |
| `command not found: racinglines` | The virtualenv isn't active in this terminal: `source ~/rl/bin/activate` |
| `needs psycopg, which the core install leaves out` (or another package) | The plain install is missing the extra: `pip install "racinglines[web]"` |
| `racinglines needs a database, and none is configured` | Step 3 was skipped: run `racinglines book config source local` |
| `demo: no Postgres server for ...` | The database isn't running: `docker start racinglines-db`, then `docker ps` should list it |
| `check` fails on `alembic_version` | The tables don't exist yet: `racinglines db init` |

### From source (contributors)

```bash
git clone https://github.com/racinglines/racinglines-public.git && cd racinglines-public
python3 -m venv .venv-rl && source .venv-rl/bin/activate
pip install -e ".[web]"                                 # Python 3.11+; the database is step 3's container
```

Then steps 3 (`book config` and `db init`) to 10 as above; the demo's files go under `data/demo/kart/` in the checkout.

## More

- Documentation: https://racinglines.bet/docs (the quickstart, the model, market making, the CLI reference)
- The same demo, recorded, in your browser: https://racinglines.bet/term
- `racinglines <group> -h` lists any command group's commands and options.

MIT licensed.
