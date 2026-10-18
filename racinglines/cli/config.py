"""
The CLI's defaults file, for people who run their own server: which data source `racinglines book` uses when no
--demo / --remote / --local flag is given, and where. ~/.config/racinglines/cli.toml (RACINGLINES_CLI_CONFIG to move it):

    source = "remote"                              # local (default) | remote | demo; set by the first-run question
    mcp_url = "https://mcp.example.com/mcp"        # the hosted MCP server --remote uses
    db = "postgresql+psycopg://...@host/racinglines"   # the database local mode uses when DATABASE_URL is unset

Flags win over environment variables, which win over this file: --mcp-url > RACINGLINES_MCP_URL > mcp_url;
--db > DATABASE_URL > db. `racinglines book config` shows the file, `book config KEY VALUE` sets a key, and
`book config KEY --unset` removes it.
"""

import os
import tomllib
from pathlib import Path

KEYS = {"source": ("local", "remote", "demo"), "mcp_url": None, "db": None}


def path():
    return Path(os.environ.get("RACINGLINES_CLI_CONFIG", "~/.config/racinglines/cli.toml")).expanduser()


def load():
    """The file's settings ({} when there is none). Raises ValueError on a bad file, naming it."""
    p = path()
    try:
        data = tomllib.loads(p.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError) as ex:
        raise ValueError(f"{p}: {ex}") from ex
    bad = [k for k in data if k not in KEYS]
    if bad:
        raise ValueError(f"{p}: unknown setting(s) {', '.join(bad)} (known: {', '.join(KEYS)})")
    if data.get("source") not in (None, *KEYS["source"]):
        raise ValueError(f"{p}: source must be one of {', '.join(KEYS['source'])}")
    return data


def set_key(key, value):
    """Set (value) or remove (None) one key; returns the new settings. Raises ValueError on an unknown key or value."""
    if key not in KEYS:
        raise ValueError(f"unknown setting {key!r} (known: {', '.join(KEYS)})")
    if value is not None and KEYS[key] and value not in KEYS[key]:
        raise ValueError(f"{key} must be one of {', '.join(KEYS[key])}")
    data = load()
    if value is None:
        data.pop(key, None)
    else:
        data[key] = value
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f'{k} = "{v}"\n' for k, v in data.items()))
    os.chmod(p, 0o600)                                    # a db URL can hold a password
    return data
