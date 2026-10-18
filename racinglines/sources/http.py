"""
Polite HTTP for every external data source (Polymarket, ChronoRace, NASCAR's content feeds, Wikipedia; FastF1 has its own
limiter, see sources/fastf1/fetch.py).

- **Pacing:** a minimum interval between requests to the same host, shared by every thread in the
  process (HOST_INTERVAL; well under each API's published limits).
- **Retries:** timeouts, dropped connections, 429 and 5xx are retried with exponential backoff and
  jitter (2, 4, 8, 16 s ...), honouring a Retry-After header; 4xx other than 429 return at once.
- **Caching** lives with the callers: downloads skip what's already stored (weekends with prices,
  weeks of season history, FastF1 sessions on disk, the saved FastF1 schedule).

    r = http.get(client, "/prices-history", params={...})        # httpx.Client with base_url
    r = http.get(session, "https://en.wikipedia.org/wiki/...")    # requests.Session
"""

import random
import threading
import time
from urllib.parse import urlsplit

# seconds between requests per host: deliberately conservative (at most a few requests per second per
# host), not tuned to any published limit; ChronoRace and Wikipedia are shared public servers
HOST_INTERVAL = {
    "clob.polymarket.com": 0.15,
    "gamma-api.polymarket.com": 0.25,
    "data-api.polymarket.com": 0.25,
    "api.elections.kalshi.com": 0.25,
    "prod.chronorace.be": 0.5,
    "cf.nascar.com": 1.0,
    "api.motogp.pulselive.com": 0.5,
    "motorsport.com": 1.0,
    "www.motorsport.com": 1.0,
    "en.wikipedia.org": 1.0,
}
DEFAULT_INTERVAL = 0.25
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_TRIES = 5
MAX_WAIT = 120.0

_lock = threading.Lock()
_next_ok = {}                       # host -> earliest time the next request may start
stats = {"requests": 0, "retries": 0, "waited_s": 0.0}


def _host(client, url):
    if url.startswith("http"):
        return urlsplit(url).hostname or ""
    base = getattr(client, "base_url", None)
    return urlsplit(str(base)).hostname or "" if base is not None else ""


def _pace(host):
    gap = HOST_INTERVAL.get(host, DEFAULT_INTERVAL)
    with _lock:
        now = time.monotonic()
        start = max(now, _next_ok.get(host, 0.0))
        _next_ok[host] = start + gap
    if start > now:
        stats["waited_s"] += start - now
        time.sleep(start - now)


def _retry_after(r):
    v = (getattr(r, "headers", {}) or {}).get("Retry-After")
    try:
        return min(float(v), MAX_WAIT) if v is not None else None
    except ValueError:
        return None


def request(client, method, url, tries=MAX_TRIES, **kw):
    """One request through `client` (httpx.Client or requests.Session), paced and retried. Returns the
    last response (the caller checks its status), or raises the last network error."""
    import httpx
    try:
        import requests
        net_errors = (httpx.TimeoutException, httpx.TransportError, requests.ConnectionError, requests.Timeout)
    except ImportError:                                  # pragma: no cover
        net_errors = (httpx.TimeoutException, httpx.TransportError)
    host = _host(client, url)
    r = None
    for i in range(tries):
        _pace(host)
        stats["requests"] += 1
        try:
            r = client.request(method, url, **kw)
            if r.status_code not in RETRY_STATUS or i == tries - 1:
                return r
            wait = _retry_after(r)
        except net_errors:
            if i == tries - 1:
                raise
            wait = None
        wait = wait if wait is not None else min(2 ** (i + 1), MAX_WAIT) * (0.75 + 0.5 * random.random())
        stats["retries"] += 1
        stats["waited_s"] += wait
        time.sleep(wait)
        with _lock:                                      # back off the whole host, not just this call
            _next_ok[host] = max(_next_ok.get(host, 0.0), time.monotonic())
    return r


def get(client, url, **kw):
    return request(client, "GET", url, **kw)


def post(client, url, **kw):
    return request(client, "POST", url, **kw)
