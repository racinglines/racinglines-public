"""Database connection settings.

Set DATABASE_URL to point somewhere else, e.g.
    export DATABASE_URL=postgresql+psycopg://user:pass@host:5432/racinglines
The default matches docker-compose.yml (host port 5433, so it doesn't clash
with a Postgres already running on 5432).
"""

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DEFAULT_URL = "postgresql+psycopg://racinglines:racinglines@localhost:5433/racinglines"
# What database_url() falls back to when neither a URL nor DATABASE_URL is given: None = DEFAULT_URL. The CLI sets it
# while a command runs (racinglines/cli/source.py: the defaults file's `db`, or the first-run choice).
fallback = None


def database_url(url=None):
    if url or os.environ.get("DATABASE_URL"):
        return url or os.environ["DATABASE_URL"]
    return fallback() if fallback else DEFAULT_URL


@lru_cache(maxsize=None)
def get_engine(url=None):
    return create_engine(database_url(url), future=True)


def get_session(url=None):
    return sessionmaker(get_engine(url), expire_on_commit=False)()
