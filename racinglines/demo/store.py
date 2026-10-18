"""
Demo packs and the demo store. A pack (racinglines/demo/packs/<name>/pack.toml, or a folder on RACINGLINES_DEMO_PACKS)
is data: it names an existing sport schema, the files and/or generator that make its data, the real commands that
load it, its sportsbook book and its quickstart. `--demo [PACK]` (racinglines/cli/source.py) points every command at
that pack's store: a separate Postgres database and data folder, built here once by the REAL pipeline. Nothing here
prices, trades or reads a model.

    1. create the database (racinglines_demo_<pack> by default; never one whose name lacks "demo")
    2. the schema and reference rows: Alembic upgrade head and the seed, as `racinglines db init` runs them
    3. the pack's files copied into the data folder, then its generator (if any) writes the rest
    4. the pack's [load] commands, each a real `racinglines ...` command run on the store (e.g. the sport's ingest)
    5. a data_changes line and a marker (<data>/demo-store.json: a fingerprint of the pack and its generator; a
       changed pack rebuilds the store from scratch)

Seeded packs give the same store every time.
"""

import hashlib
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

MARKER = "demo-store.json"
BUNDLED = Path(__file__).resolve().parent / "packs"
DEFAULT_PACK = "kart"
DEFAULT_SECTION = ""                        # [[quickstart]] entries with no `section` key


@dataclass
class Pack:
    name: str
    path: Path
    manifest: dict = field(default_factory=dict)

    @property
    def sport(self):
        return self.manifest["sport"]

    @property
    def sport_name(self):
        """The name the pack's sport answers to on the command line under --demo (`sport_name`, default the stored
        sport): kart's commands say `kart` while the data sits under the f1 schema."""
        return self.manifest.get("sport_name") or self.sport

    @property
    def book(self):
        f = (self.manifest.get("book") or {}).get("file")
        return self.path / f if f else None

    @property
    def quickstart(self):
        return list(self.manifest.get("quickstart", []))

    def sections(self):
        """The quickstart grouped by its entries' optional `section` key, in first-seen order: [(name, [steps])]. An
        entry without one is in DEFAULT_SECTION (a pack with only that renders as one ungrouped list)."""
        out = {}
        for step in self.quickstart:
            out.setdefault(step.get("section") or DEFAULT_SECTION, []).append(step)
        return list(out.items())

    @property
    def remote(self):
        """[remote]: where the pack's hosted store is served (mcp_url), for `--demo PACK --remote`."""
        return dict(self.manifest.get("remote") or {})

    @property
    def load(self):
        return list((self.manifest.get("load") or {}).get("commands", []))


def search_path():
    """Where packs are looked up: RACINGLINES_DEMO_PACKS (folders, os.pathsep-separated), then the bundled ones."""
    extra = [Path(p) for p in os.environ.get("RACINGLINES_DEMO_PACKS", "").split(os.pathsep) if p.strip()]
    return extra + [BUNDLED]


def find(ref=None):
    """The pack named `ref` (a name on the search path, or a folder holding a pack.toml), or None."""
    ref = ref or DEFAULT_PACK
    cands = [Path(ref)] if os.sep in ref else []
    cands += [d / ref for d in search_path()]
    for p in cands:
        if (p / "pack.toml").is_file():
            m = tomllib.loads((p / "pack.toml").read_text())
            return Pack(m.get("name") or p.name, p.resolve(), m)
    return None


def packs():
    """Every pack on the search path, by name (the first found wins)."""
    out = {}
    for d in search_path():
        for p in sorted(d.glob("*/pack.toml")) if d.is_dir() else []:
            pk = find(str(p.parent.resolve()))
            out.setdefault(pk.name, pk)
    return out


def fingerprint(pack):
    """What the store is built from: every file of the pack, its generator's module and this builder."""
    h = hashlib.sha256()
    for f in sorted(p for p in pack.path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(f.relative_to(pack.path).as_posix().encode())
        h.update(f.read_bytes())
    gen = (pack.manifest.get("generator") or {}).get("entry")
    if gen:
        h.update(Path(importlib.import_module(gen.split(":")[0]).__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:16]


def _db_name(url):
    from sqlalchemy.engine import make_url
    return make_url(url).database


def _guard(url, data):
    if "demo" not in (_db_name(url) or ""):
        raise SystemExit(f"demo store: refusing to build into database {_db_name(url)!r} (its name must contain 'demo')")
    if "demo" not in Path(data).resolve().as_posix():
        raise SystemExit(f"demo store: refusing to build into {data} (the folder's path must contain 'demo')")


def _admin(url):
    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url
    return create_engine(make_url(url).set(database="postgres"), isolation_level="AUTOCOMMIT")


def db_exists(url):
    from sqlalchemy import text
    eng = _admin(url)
    try:
        with eng.connect() as c:
            return bool(c.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), dict(n=_db_name(url))).scalar())
    finally:
        eng.dispose()


def status(pack, url, data):
    """'ready', or why the store needs building ('no database', 'no data', 'outdated')."""
    m = Path(data) / MARKER
    if not db_exists(url):
        return "no database"
    if not m.exists():
        return "no data"
    try:
        built = json.loads(m.read_text())
    except ValueError:
        return "no data"
    if built.get("fingerprint") != fingerprint(pack) or built.get("database") != _db_name(url):
        return "outdated"
    return "ready"


@contextmanager
def _bar(console):
    """progress(done, total, what): a redrawing Rich bar on a terminal; otherwise one line per new step (styled when
    `console` is a Rich console, plain text without one)."""
    from racinglines.cli import tty
    if console is None or not tty.interactive(console.file):
        last = []

        def lines(done, total, what):
            key = what.split(" R")[0] if what.startswith("generating: ") else what     # a line per season generated
            if last and last[-1] == key:
                return
            last.append(key)
            if console is None:
                print(f"demo store: {what} ({done:.0f}/{total})", flush=True)
            else:
                from rich.markup import escape
                n = "" if what.startswith("generating: ") else f"[dim]{done:.0f}/{total}[/] "
                console.print(f"🔧 [bold]demo store[/] {n}{escape(what)}")
        yield lines
        return
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
    with Progress(TextColumn("[bold]demo store[/]"), BarColumn(bar_width=30), MofNCompleteColumn(), TimeElapsedColumn(),
                  TextColumn("[dim]{task.description}"), console=console, transient=False) as bar:
        task = bar.add_task("", total=None)

        def step(done, total, what):
            bar.update(task, completed=done, total=total, description=what[:70])
        yield step


def env_for(url, data):
    """The environment a command runs in on the store (what `--demo` sets; racinglines/cli/source.py)."""
    env = dict(os.environ, DATABASE_URL=url, RACINGLINES_DATA=str(data), RACINGLINES_OFFLINE="1")
    env.pop("RACINGLINES_DEMO", None)
    return env


def build(pack, url, data, console=None, options=None):
    """Build `pack`'s store into database `url` and folder `data` (wiping an older build of both). options: keyword
    arguments for the pack's generator (e.g. a smaller store). Returns a summary dict."""
    from sqlalchemy import text

    from racinglines import progress as PG
    _guard(url, data)
    data = Path(data)
    loads = pack.load
    total = 3 + len(loads)
    with _bar(console) as bar:
        def say(done, what):
            bar(done, total, what)
            PG.update(item=f"demo store: {what}")

        say(0, "creating the database")
        eng = _admin(url)
        name = _db_name(url)
        with eng.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
            c.execute(text(f'CREATE DATABASE "{name}"'))
        eng.dispose()
        if data.is_dir():
            shutil.rmtree(data)
        data.mkdir(parents=True)

        say(1, "schema and reference data (alembic upgrade head, seed)")
        _init_db(url)

        say(2, "the pack's data files")
        if pack.manifest.get("files") and (pack.path / pack.manifest["files"]).is_dir():
            shutil.copytree(pack.path / pack.manifest["files"], data, dirs_exist_ok=True)
        gen, made = pack.manifest.get("generator") or {}, {}
        if gen.get("entry"):
            mod, _, fn = gen["entry"].partition(":")
            kw = dict(options or {})
            if gen.get("input"):
                kw["input"] = pack.path / gen["input"]
            made = getattr(importlib.import_module(mod), fn)(
                data, echo=lambda d, t, what: say(2 + d / max(t, 1), f"generating: {what}"), **kw) or {}

        for i, cmd in enumerate(loads):
            say(3 + i, cmd)
            argv = shlex.split(cmd)
            argv = argv[1:] if argv[:1] == ["racinglines"] else argv
            p = subprocess.run([sys.executable, "-m", "racinglines", *argv], env=dict(env_for(url, data), NO_COLOR="1"),
                               capture_output=True, text=True)
            if p.returncode != 0:
                raise SystemExit(f"demo store: `{cmd}` failed ({p.returncode}):\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
        summary = dict(pack=pack.name, sport=pack.sport, fingerprint=fingerprint(pack), database=name,
                       generated={k: v for k, v in made.items() if isinstance(v, (int, str, list))}, loaded=loads)
        from racinglines.db import changes
        from racinglines.db.config import get_session
        with get_session(url) as s:
            changes.record(s, "demo", f"demo store built from pack {pack.name} (synthetic {pack.sport} data)",
                           sport=pack.sport, detail=summary)
            s.commit()
        (data / MARKER).write_text(json.dumps(summary, indent=1))
        bar(total, total, "ready")
    return summary


def _init_db(url):
    """`racinglines db init` on the demo database: Alembic upgrade head, then the seed."""
    from alembic import command
    from alembic.config import Config

    from racinglines import paths
    from racinglines.db.config import get_session
    from racinglines.db.ingest import seed
    cfg = Config()                    # alembic.ini's settings without its logging setup: the progress bar shows the step
    cfg.set_main_option("script_location", str(paths.ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    with get_session(url) as s:
        seed(s)
        s.commit()
