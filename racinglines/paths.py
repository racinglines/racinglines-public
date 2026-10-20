"""
Where everything lives on disk. One place, so nothing else hard-codes a path.

    data/                                 $RACINGLINES_DATA overrides
      raw/<sport>/<source>/...            immutable downloads: the record
      archive/markets/<exchange>/<store>/month=YYYY-MM/*.parquet
      archive/<sport>/...                 stale heavy race data moved out of Postgres
      runs/<sport>/<kind>/...             generated outputs (backtests, sweeps, forecasts)
      runs/jobs/...                       outputs of jobs launched from the web app
      cache/<tool>/...                    disposable (e.g. FastF1's HTTP cache)

Sports use their database codes (f1, mtb_dh); sources name where the data came from.
"""

import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
CHECKOUT = (PACKAGE.parent / "pyproject.toml").is_file()    # a git checkout (also `pip install -e .`), not a wheel
# the committed configs (sports/, markets/, exchanges/): the checkout's, or the copies a wheel bundles (pyproject.toml)
# racinglines.bet's own copy: its branded landing page ships only there (oss/manifest.toml leaves it out of the package)
HOSTED = (PACKAGE / "web" / "templates" / "login.html").is_file()
ROOT = PACKAGE.parent if CHECKOUT else PACKAGE / "_bundled"
HOME = PACKAGE.parent if CHECKOUT else Path.home() / ".racinglines"     # where data/ goes when nothing says otherwise
DATA = Path(os.environ.get("RACINGLINES_DATA", HOME / "data"))


def raw(sport, source):
    return DATA / "raw" / sport / source


def archive_markets(exchange="polymarket"):
    return DATA / "archive" / "markets" / exchange


def archive(sport):
    return DATA / "archive" / sport


def runs(*parts, mkdir=True):
    p = DATA / "runs" / Path(*parts)
    if mkdir:
        p.mkdir(parents=True, exist_ok=True)
    return p


def cache(tool):
    return DATA / "cache" / tool


def rel(path):
    """A path relative to DATA (stable across machines), e.g. for source-file keys."""
    p = Path(path).resolve()
    try:
        return p.relative_to(DATA.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


F1_RAW = raw("f1", "fastf1")
DH_RAW = raw("mtb_dh", "chronorace")
NASCAR_RAW = raw("nascar", "cf")
