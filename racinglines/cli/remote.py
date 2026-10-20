"""
`--remote` for `racinglines book`: run map / price / settle on the VM's hosted MCP server (docs/mcp.md, Hosted) instead
of a local database, so nobody has to rebuild the data or serve the app to try the commands. The book file is read
and validated here and sent as text to the MCP tools `map_book`, `price_book`, `settle_book`, which return the same
dict `--json` prints locally (racinglines/books/slips.py); `--rich` and plain text render it the same way.

Sign-in, the same OAuth flow Claude's connector uses (docs/mcp.md, Hosted): the first `--remote` opens racinglines.bet in
the browser for one Allow (after the normal login if needed) and keeps the login in ~/.config/racinglines/mcp-login.json
(mode 600): an access token it refreshes on its own, for 30 days of disuse. `racinglines book logout` forgets it.

    RACINGLINES_MCP_TOKEN   an account's rl_ token instead of the browser sign-in (scripts, machines with no browser)
    RACINGLINES_MCP_URL     the server (default https://mcp.racinglines.bet/mcp); --mcp-url overrides it.
"""

import asyncio
import json
import os
import socket
import sys
import threading
from pathlib import Path

DEFAULT_URL = "https://mcp.racinglines.bet/mcp"
TOOLS = {"map": "map_book", "price": "price_book", "settle": "settle_book"}
LOGIN_FILE = Path(os.environ.get("RACINGLINES_MCP_LOGIN", "~/.config/racinglines/mcp-login.json")).expanduser()
CALLBACK_WAIT_SEC = 300
WARNING = ("warning: racinglines.bet remote access is in testing (#aintcare mode). A review of venue and data-feed "
           "terms is pending, so data served here is for testing only.")
_warned = False


def warn():
    """The remote-access warning on stderr, once per command run, before anything the server returns (stdout stays
    clean for --json). Every remote path goes through call(), which calls this first."""
    global _warned
    if not _warned:
        print(WARNING, file=sys.stderr, flush=True)
        _warned = True


class RemoteError(RuntimeError):
    """The server refused or failed the call; the message says why."""


def url(override=None):
    return override or os.environ.get("RACINGLINES_MCP_URL") or DEFAULT_URL


def token():
    """An rl_ token from the environment, or None (then the browser sign-in is used)."""
    return os.environ.get("RACINGLINES_MCP_TOKEN", "").strip() or None


# --- browser sign-in -----------------------------------------------------------------------------------------------

def _load():
    try:
        return json.loads(LOGIN_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _save(data):
    LOGIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOGIN_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(LOGIN_FILE)


def logout(server=None):
    """Forget the stored login for `server` (all servers when None). True if there was one."""
    data = _load()
    if server is None:
        had = bool(data)
        data = {}
    else:
        had = data.pop(url(server), None) is not None
    if data:
        _save(data)
    elif LOGIN_FILE.exists():
        LOGIN_FILE.unlink()
    return had


class _Storage:
    """The SDK's TokenStorage, one entry per server URL in LOGIN_FILE."""

    def __init__(self, server):
        self.server = server

    def _get(self, key, model):
        v = _load().get(self.server, {}).get(key)
        return None if v is None else model.model_validate(v)

    def _set(self, key, value):
        data = _load()
        data.setdefault(self.server, {})[key] = value.model_dump(mode="json", exclude_none=True)
        _save(data)

    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken
        return self._get("tokens", OAuthToken)

    async def set_tokens(self, tokens):
        self._set("tokens", tokens)

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull
        return self._get("client", OAuthClientInformationFull)

    async def set_client_info(self, client_info):
        self._set("client", client_info)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _callback_port(server):
    """The loopback port the registered client redirects to: the stored client's, else a free one (a new client)."""
    client = _load().get(server, {}).get("client")
    if client:
        from urllib.parse import urlparse
        return urlparse(client["redirect_uris"][0]).port
    return _free_port()


def _wait_for_callback(port, ready=None):
    """Serve one GET /callback on 127.0.0.1:port and return its query parameters (code, state, or error). `ready`
    (a threading.Event) is set once the port is bound, or binding failed, so the browser never opens first."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse
    got = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            q = urlparse(self.path)
            if q.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return
            got.update({k: v[0] for k, v in parse_qs(q.query).items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            msg = "Signed in. You can close this tab." if "code" in got else "Sign-in was not completed."
            self.wfile.write(f"<p style='font-family:sans-serif'>racinglines: {msg}</p>".encode())

        def log_message(self, *args):
            pass

    try:
        httpd = HTTPServer(("127.0.0.1", port), Handler)
    except OSError as ex:
        if ready is not None:
            ready.set()
        raise RemoteError(f"can't listen on 127.0.0.1:{port} for the sign-in ({ex}); "
                          "run `racinglines book logout` and try again") from ex
    httpd.timeout = 1
    if ready is not None:
        ready.set()
    with httpd:
        for _ in range(CALLBACK_WAIT_SEC):
            httpd.handle_request()
            if got:
                return got
    raise RemoteError(f"no sign-in within {CALLBACK_WAIT_SEC} s")


def oauth(server, open_browser=None):
    """The SDK's OAuth provider for `server`: stored tokens, refreshed as needed; a browser sign-in when there are
    none. `open_browser(url)` is for tests (default: the web browser, and the URL printed to stderr)."""
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import AuthorizationCodeResult, OAuthClientMetadata
    port = _callback_port(server)
    pending = {}

    async def redirect(auth_url):
        ready = threading.Event()
        pending["t"] = threading.Thread(target=lambda: pending.update(r=_safe_wait(port, ready)), daemon=True)
        pending["t"].start()
        await asyncio.to_thread(ready.wait, 10)        # listening before the browser can come back to it
        print(f"Sign in to racinglines in your browser (this URL if it does not open):\n  {auth_url}", file=sys.stderr)
        if open_browser:
            await asyncio.to_thread(open_browser, auth_url)
        else:
            import webbrowser
            webbrowser.open(auth_url)

    async def callback():
        await asyncio.to_thread(pending["t"].join)
        r = pending.get("r")
        if isinstance(r, Exception):
            raise r
        if "code" not in r:
            raise RemoteError(f"sign-in refused: {r.get('error', 'no code returned')}")
        return AuthorizationCodeResult(code=r["code"], state=r.get("state"))

    meta = OAuthClientMetadata(client_name="racinglines CLI", redirect_uris=[f"http://127.0.0.1:{port}/callback"],
                               grant_types=["authorization_code", "refresh_token"], response_types=["code"],
                               token_endpoint_auth_method="none")
    return OAuthClientProvider(server_url=server, client_metadata=meta, storage=_Storage(server),
                               redirect_handler=redirect, callback_handler=callback)


def _safe_wait(port, ready=None):
    try:
        return _wait_for_callback(port, ready)
    except Exception as ex:  # noqa: BLE001  (handed to the callback handler, which raises it)
        return ex


def call(tool, arguments, server=None, tok=None, http_client=None, open_browser=None):
    """Call one tool on the hosted server and return its parsed JSON: with an rl_ token (`tok`, else
    RACINGLINES_MCP_TOKEN) when there is one, else the browser sign-in. `http_client` (an httpx2.AsyncClient) and
    `open_browser` are for tests."""
    warn()
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    statuses = []                                          # the SDK's errors drop the HTTP status: keep it here

    async def seen(response):
        if response.status_code >= 400:
            statuses.append(response.status_code)

    async def go():
        bearer = tok or token()
        client = http_client or (create_mcp_http_client(headers={"Authorization": f"Bearer {bearer}"}) if bearer
                                 else create_mcp_http_client(auth=oauth(url(server), open_browser)))
        client.event_hooks.setdefault("response", []).append(seen)
        async with client, streamable_http_client(url(server), http_client=client) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool, arguments)

    try:
        r = asyncio.run(go())
    except RemoteError:
        raise
    except Exception as ex:  # noqa: BLE001  (network, TLS, 401: one line the user can act on)
        raise RemoteError(f"{url(server)}: {_reason(ex, statuses)}") from ex
    return _parse(tool, r)


def _parse(tool, r):
    text = r.content[-1].text if r.content else ""          # the JSON is the last block (a drawn table may lead it)
    if r.is_error:
        raise RemoteError(f"{tool}: {text}")
    return json.loads(text)


class Session:
    """One signed-in MCP session for many calls (`data pull`'s chunks, a tool listing): call() opens a session per
    call, which is a sign-in round trip each time. The session lives in one task of its own event loop in a thread
    (the SDK's streams must be entered and left by the same task); call() hands it each request. Use it as a context
    manager."""

    def __init__(self, server=None, tok=None, http_client=None, open_browser=None):
        import concurrent.futures
        warn()
        self.server, self._statuses = server, []
        self.loop = asyncio.new_event_loop()
        self._ready = concurrent.futures.Future()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()
        self._queue = None
        self._main = asyncio.run_coroutine_threadsafe(self._run(tok, http_client, open_browser), self.loop)
        try:
            self._ready.result()
        except Exception as ex:  # noqa: BLE001
            self._stop()
            raise RemoteError(f"{url(server)}: {_reason(ex, self._statuses)}") from ex

    async def _run(self, tok, http_client, open_browser):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        async def seen(response):
            if response.status_code >= 400:
                self._statuses.append(response.status_code)
        try:
            bearer = tok or token()
            client = http_client or (create_mcp_http_client(headers={"Authorization": f"Bearer {bearer}"}) if bearer
                                     else create_mcp_http_client(auth=oauth(url(self.server), open_browser)))
            client.event_hooks.setdefault("response", []).append(seen)
            async with client, streamable_http_client(url(self.server), http_client=client) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    self._queue = asyncio.Queue()
                    self._ready.set_result(None)
                    while (item := await self._queue.get()) is not None:
                        tool, args, fut = item
                        try:
                            fut.set_result(await (session.list_tools() if tool == "__list__" else
                                                  session.call_tool(tool, args)))
                        except Exception as ex:  # noqa: BLE001  (handed to the caller's thread)
                            fut.set_exception(ex)
        except Exception as ex:  # noqa: BLE001
            if not self._ready.done():
                self._ready.set_exception(ex)
            else:
                raise

    def tools(self):
        """The server's tools: [(name, description, input schema)]."""
        import concurrent.futures
        fut = concurrent.futures.Future()
        self.loop.call_soon_threadsafe(self._queue.put_nowait, ("__list__", None, fut))
        try:
            r = fut.result()
        except Exception as ex:  # noqa: BLE001
            raise RemoteError(f"{url(self.server)}: {_reason(ex, self._statuses)}") from ex
        return [(t.name, t.description or "", getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}) for t in r.tools]

    def call(self, tool, arguments):
        import concurrent.futures
        fut = concurrent.futures.Future()
        self.loop.call_soon_threadsafe(self._queue.put_nowait, (tool, arguments, fut))
        try:
            r = fut.result()
        except Exception as ex:  # noqa: BLE001
            raise RemoteError(f"{url(self.server)}: {_reason(ex, self._statuses)}") from ex
        return _parse(tool, r)

    def _stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(5)

    def close(self):
        if self._queue is not None:
            self.loop.call_soon_threadsafe(self._queue.put_nowait, None)
            try:
                self._main.result(10)
            except Exception:  # noqa: BLE001  (closing: nothing left to report)
                pass
        self._stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _reason(ex, statuses=()):
    """One line the user can act on: the HTTP status the server answered, else the innermost message of an exception
    group (anyio wraps transport errors)."""
    if 401 in statuses:
        return ("401 Unauthorized: the token or login is wrong, revoked or for an account that may not use MCP "
                "(`racinglines book logout` forgets a stored login)")
    while isinstance(ex, BaseExceptionGroup) and ex.exceptions:
        ex = ex.exceptions[0]
    msg = str(ex).splitlines()[0] if str(ex) else type(ex).__name__
    return (f"HTTP {statuses[-1]}: " if statuses else "") + msg[:300]


def run(cmd, raw_book, server=None, runs=(), open_browser=None):
    """The command's result dict from the hosted server: what `racinglines book CMD --json` prints locally."""
    args = {"book": raw_book}
    if cmd == "price" and runs:
        args["run_ids"] = list(runs)
    return call(TOOLS[cmd], args, server, open_browser=open_browser)


# --- model commands as hosted jobs ---------------------------------------------------------------------------------
# `racinglines model backtest|forecast SPORT --remote`: the command's Lab job (racinglines/web/jobs.py CATALOG) queued
# on the hosted server with run_job and followed with get_job, its log printed as it arrives. The server runs the same
# CLI command in its own worker; the flags below are the ones the job takes (its knobs), the rest are local only.

JOBS = {("f1", "backtest"): "f1_backtest", ("f1", "forecast"): "f1_scenario",
        ("mtb_dh", "backtest"): "dh_backtest", ("mtb_dh", "forecast"): "dh_scenario"}
POLL_SEC = 1.0
TAIL = 200                                 # log lines per get_job poll (the tool returns up to 500)


def _job_parser(job):
    import argparse
    ap = argparse.ArgumentParser(prog=f"--remote ({job})", add_help=False)
    ap.add_argument("--sims", type=int)
    if job.startswith("f1_"):
        ap.add_argument("--half-life", dest="half_life", type=float)
    else:
        ap.add_argument("--half-life-days", dest="half_life", type=float)
        ap.add_argument("--junior-weight", dest="junior_weight", type=float)
    if job == "f1_backtest":
        ap.add_argument("--races", type=int)
        ap.add_argument("--track", choices=["both", "on", "off"])
    if job.endswith("_scenario"):
        ap.add_argument("--scenario", dest="label")
    if job == "f1_scenario":
        ap.add_argument("--no-track", dest="track", action="store_const", const="off")
    ap.add_argument("--save", action="store_true")              # implied: a hosted job always saves its run
    ap.add_argument("--db", nargs="?", const="")                 # mtb_dh's "read the database": the server's always
    return ap


# more commands whose Lab job takes its knobs by name (C53): the flags are the job's own knobs, read from the server's
# list_job_types (`--<knob> VALUE`, and `--set NAME=VALUE` for a sweep setting), so nothing here lists them twice
AUTO_JOBS = {("f1", "sweep"): "f1_sweep", ("f1", "season-strategy"): "f1_season_strategy",
             ("f1", "diagnostic"): "f1_diagnostic", ("f1", "combo"): "f1_combo",
             ("nascar", "sweep"): "nascar_sweep", ("motogp", "sweep"): "motogp_sweep"}
COMMAND_GROUP = {"backtest": "model", "forecast": "model", "diagnostic": "model", "combo": "model",
                 "sweep": "strategy", "season-strategy": "strategy"}


def runs_remotely():
    """The `GROUP COMMAND SPORT` forms --remote takes."""
    return [f"{COMMAND_GROUP[c]} {c} {s}" for s, c in [*JOBS, *AUTO_JOBS]]


def knob_params(job_type, argv, catalog):
    """(job type, params) from `--<knob> VALUE` flags (dashes or underscores) and `--set NAME=VALUE` (a sweep setting;
    VALUE as JSON when it parses, else text), checked against `catalog` (list_job_types' job_types)."""
    jt = next((j for j in catalog if j["job_type"] == job_type), None)
    if jt is None:
        raise RemoteError(f"the hosted server has no job {job_type!r} (or not for this account)")
    knobs = {k["name"] for k in jt["knobs"] if k["name"] != "settings"}
    takes_settings = any(k["name"] == "settings" for k in jt["knobs"])
    params, settings, bad = {}, {}, []
    args = [a for a in argv if a != "--remote"]
    i = 0
    while i < len(args):
        a = args[i]
        name, val = (a[2:].split("=", 1) + [None])[:2] if a.startswith("--") else (None, None)
        if name is None:
            bad.append(a)
            i += 1
            continue
        if val is None:
            if i + 1 >= len(args):
                raise RemoteError(f"{a}: needs a value")
            val, i = args[i + 1], i + 2
        else:
            i += 1
        if name == "set" and takes_settings and "=" in val:
            k, v = val.split("=", 1)
            try:
                settings[k.replace("-", "_")] = json.loads(v)
            except ValueError:
                settings[k.replace("-", "_")] = v
        elif name.replace("-", "_") in knobs:
            params[name.replace("-", "_")] = val
        else:
            bad.append(a)
    if bad:
        takes = sorted(f"--{k.replace('_', '-')}" for k in knobs) + (["--set NAME=VALUE"] if takes_settings else [])
        raise RemoteError(f"{' '.join(bad)}: not a knob of {job_type} on the hosted server; it takes {', '.join(takes)}")
    if settings:
        params["settings"] = settings
    return job_type, params


def job_params(sport, command, argv):
    """(job type, params) for `model COMMAND SPORT argv --remote`. Raises RemoteError on a flag the job doesn't take."""
    job = JOBS.get((sport, command))
    if job is None:
        raise RemoteError(f"{command} {sport} doesn't run on the hosted server (it runs: {', '.join(runs_remotely())}; "
                          "anything else with an MCP tool: racinglines remote call TOOL, e.g. replay_maker)")
    known, extra = _job_parser(job).parse_known_args([a for a in argv if a != "--remote"])
    if extra:
        raise RemoteError(f"{' '.join(extra)}: local only; on the hosted server model {command} {sport} takes "
                          + ", ".join(sorted(o for a in _job_parser(job)._actions for o in a.option_strings
                                             if o not in ("--save", "--db"))))
    params = {k: v for k, v in vars(known).items() if v is not None and k not in ("save", "db")}
    if job.endswith("_scenario"):
        params.setdefault("label", f"cli {command}")
    return job, params


def run_model(sport, command, argv, server=None, out=None, poll=POLL_SEC, sleep=None):
    """Queue the job, print its log as it arrives (stdout), and return 0 when it is done, 1 when it failed."""
    import time
    out = out or sys.stdout
    sleep = sleep or time.sleep
    if (sport, command) in AUTO_JOBS:
        job, params = knob_params(AUTO_JOBS[(sport, command)], argv, call("list_job_types", {}, server)["job_types"])
    else:
        job, params = job_params(sport, command, argv)
    return follow(job, params, server, out=out, poll=poll, sleep=sleep)


def follow(job, params, server=None, out=None, poll=POLL_SEC, sleep=None):
    """Queue job `job` with `params` on the hosted server, print its log as it arrives, -> 0 done, 1 failed."""
    import time
    out = out or sys.stdout
    sleep = sleep or time.sleep
    q = call("run_job", dict(job_type=job, params=params), server)
    job_id = q["job_id"]
    print(f"queued {job} as job {job_id} on {url(server)} ({', '.join(f'{k}={v}' for k, v in q['params'].items())})",
          file=sys.stderr, flush=True)
    seen = 0
    from racinglines import progress as PG
    while True:
        j = call("get_job", dict(job_id=job_id, log_lines=TAIL), server)
        n, tail = j.get("log_lines", 0), j.get("log_tail") or []
        new = n - seen
        if new > len(tail):
            print(f"... ({new - len(tail)} log lines not shown; get_job({job_id}) on the server)", file=out, flush=True)
        for line in tail[len(tail) - min(max(new, 0), len(tail)):]:
            print(line, file=out, flush=True)
        seen = n
        PG.update(item=f"job {job_id} {j['status']}: {j.get('progress') or ''}"[:120])
        if j["status"] in ("done", "failed"):
            break
        sleep(poll)
    run = j.get("result_run_id")
    print(f"job {job_id} {j['status']}" + (f": model run {run} on the hosted server (get_model_run({run}))" if run else
                                           f" ({j.get('progress')})"), file=sys.stderr, flush=True)
    return 0 if j["status"] == "done" else 1


# --- `racinglines remote`: any hosted tool from the command line (C53) ---------------------------------------------
# The raw-data read-outs and every other MCP tool without a CLI command of its own (owner, 2026-10-14: exchange prices,
# books and trades may reach signed-in users; the terms-pending warning stays).

CLI_HELP = """racinglines remote <command>   (the hosted server's MCP tools, signed in as for --remote)

    tools                       the tools this account may call, with their arguments
    call TOOL [--ARG VALUE ...] call one tool and print its JSON; --csv FILE writes its rows instead
                                e.g. racinglines remote call get_market_history --race-id 812 --series trades
                                     racinglines remote call list_live_events
                                     racinglines remote call replay_maker --run-id 10833 --half-spread 0.03
                                     racinglines remote call fantasy_strategy
                                     racinglines remote call fantasy_follow --code T2 --sizing kelly --kelly 0.5
    job JOB_TYPE [--KNOB VALUE] queue a Lab job and follow its log (list_job_types names them; --set NAME=VALUE
                                for a sweep setting), e.g. racinglines remote job f1_sweep --year 2025
    logout                      forget the stored sign-in

    --mcp-url URL               the server (default RACINGLINES_MCP_URL, else https://mcp.racinglines.bet/mcp)
"""


def _typed(value, schema):
    """A command-line VALUE as the JSON type the tool's input schema gives (integer, number, boolean, array, object)."""
    types = [schema.get("type")] + [s.get("type") for s in schema.get("anyOf", [])]
    if "integer" in types:
        try:
            return int(value)
        except ValueError:
            pass
    if "number" in types:
        try:
            return float(value)
        except ValueError:
            pass
    if "boolean" in types and value.lower() in ("true", "false", "1", "0", "yes", "no"):
        return value.lower() in ("true", "1", "yes")
    if {"array", "object"} & set(types):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def tool_args(argv, schema):
    """{arg: value} from `--arg VALUE` / `--arg=VALUE` flags, typed by the tool's input schema. RemoteError on an
    argument the tool doesn't take."""
    props = schema.get("properties", {})
    out, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith("--"):
            raise RemoteError(f"{a}: arguments are --NAME VALUE")
        name, _, val = a[2:].partition("=")
        if not _:
            if i + 1 >= len(argv):
                raise RemoteError(f"{a}: needs a value")
            val, i = argv[i + 1], i + 1
        i += 1
        key = name.replace("-", "_")
        if key not in props:
            raise RemoteError(f"--{name}: not an argument; it takes {', '.join('--' + k.replace('_', '-') for k in props) or 'none'}")
        out[key] = _typed(val, props[key])
    return out


def _rows(result):
    """The first list of row dicts in a tool result (a paged envelope's rows), for --csv."""
    if isinstance(result, list) and (not result or isinstance(result[0], dict)):
        return result
    if isinstance(result, dict):
        if isinstance(result.get("rows"), list):
            return result["rows"]
        for v in result.values():
            r = _rows(v)
            if r is not None:
                return r
    return None


def main(argv):
    import argparse
    import csv
    args = list(argv)
    server = None
    if "--mcp-url" in args:
        i = args.index("--mcp-url")
        server, args = args[i + 1], args[:i] + args[i + 2:]
    if not args or args[0] in ("-h", "--help"):
        print(CLI_HELP)
        return 0 if args else 2
    cmd, rest = args[0], args[1:]
    try:
        if cmd == "logout":
            print("signed out" if logout(server) else "no stored sign-in")
            return 0
        if cmd == "tools":
            with Session(server) as s:
                for name, desc, schema in sorted(s.tools()):
                    params = ", ".join(schema.get("properties", {}))
                    print(f"{name}({params})\n    {desc.splitlines()[0] if desc else ''}")
            return 0
        if cmd == "call":
            if not rest:
                print("usage: racinglines remote call TOOL [--ARG VALUE ...] [--csv FILE]", file=sys.stderr)
                return 2
            tool, rest = rest[0], rest[1:]
            out_csv = None
            if "--csv" in rest:
                i = rest.index("--csv")
                out_csv, rest = rest[i + 1], rest[:i] + rest[i + 2:]
            with Session(server) as s:
                schema = next((sc for n, _, sc in s.tools() if n == tool), None)
                if schema is None:
                    raise RemoteError(f"no tool {tool!r} on {url(server)} (racinglines remote tools lists them)")
                result = s.call(tool, tool_args(rest, schema))
            if out_csv:
                rows = _rows(result)
                if rows is None:
                    raise RemoteError(f"{tool} returned no rows to write (print it without --csv)")
                cols = list(dict.fromkeys(k for r in rows for k in r))
                with open(out_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cols)
                    w.writeheader()
                    w.writerows(rows)
                print(f"{len(rows)} rows -> {out_csv}", file=sys.stderr)
                if isinstance(result, dict) and result.get("truncated"):
                    print(f"truncated: more rows from --offset {result.get('next_offset')}", file=sys.stderr)
            else:
                print(json.dumps(result, indent=1, default=str))
            return 0
        if cmd == "job":
            if not rest:
                print("usage: racinglines remote job JOB_TYPE [--KNOB VALUE ...] [--set NAME=VALUE ...]", file=sys.stderr)
                return 2
            job, params = knob_params(rest[0], rest[1:], call("list_job_types", {}, server)["job_types"])
            return follow(job, params, server)
    except RemoteError as e:
        print(f"remote {cmd}: {e}", file=sys.stderr)
        return 2
    argparse.ArgumentParser(prog="racinglines remote").error(f"no command {cmd!r}: tools, call, job, logout")
