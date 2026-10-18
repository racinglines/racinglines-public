"""
`racinglines book render`: one self-contained HTML page (and a PNG with a headless Chromium) of a priced sportsbook
board, in three styles (docs/sportsbook/render.md):

    hud        our prices overlaid on the book's own screenshot: a label in each line's `box` = [x, y, w, h]
               (pixels on the screenshot, written into the book file at transcription: a free area of the line's
               row, so the label covers neither the selection nor the odds); lines without a box go in a side panel
    card       the same per-line content in racinglines styling, without the screenshot
    agnostic   no book odds at all: the model's chance, exchange mids where the market is tight and the minimum
               decimal odds worth taking

Input is `price_book`'s result (racinglines/books/slips.py; `book price --json` saved to a file renders without a
database) or the road-cycling CLI's price outputs (`racinglines model price cycling` writes reports/<event>/futures.csv and
matchups.csv). Both become the same normalized line rows, `rows_from_priced` / `rows_from_cycling`.

Sizing is data (markets/books.toml [stake], overridable per call): one function, `size`, for every sport and venue.

    load_settings(overrides)            markets/books.toml with the CLI's overrides
    rows_from_priced(priced, settings)  (meta, rows) from a price_book result
    rows_from_cycling(folder)           (meta, rows) from a cycling price output folder (model only, no exchange)
    size(rows, settings)                + stake, verdict and note per line (the v2 rule, the cap, scale, drop-under)
    render(meta, rows, style, settings, image=None)   the HTML page
    to_png(html_path, png_path, chrome=None, dpr=1)    the PNG, with a headless Chromium (find_chrome)
    parse_text(text, odds) / book_toml(...)           loose "A over B 1.57" lines into a book file draft
"""

import base64
import glob
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from racinglines.books import schema as S
from racinglines.paths import ROOT

SETTINGS = ROOT / "markets" / "books.toml"
CSS = Path(__file__).with_name("render.css")
STYLES = ("hud", "card", "agnostic")
CALIBRATION = "calibration unvalidated (the C48 calibration check is not done)"
CHROME_NAMES = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "chrome")
CHROME_GLOBS = ("/opt/pw-browsers/chromium-*/chrome-linux/chrome",
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium")
EPS = 1e-9


class RenderError(ValueError):
    pass


# --- settings ------------------------------------------------------------------------------------------------------

def load_settings(overrides=None, path=SETTINGS):
    """markets/books.toml, with `overrides` ({"bank": 500, ...}, None values ignored) applied to [stake]."""
    st = tomllib.loads(Path(path).read_text())
    for k, v in (overrides or {}).items():
        if v is not None:
            if k not in st["stake"]:
                raise RenderError(f"unknown stake setting {k!r}: {', '.join(st['stake'])}")
            st["stake"][k] = v
    for k in ("bank", "kelly"):
        if not st["stake"][k] > 0:
            raise RenderError(f"stake setting {k} must be > 0, got {st['stake'][k]!r}")
    return st


# --- normalized rows -----------------------------------------------------------------------------------------------

ROW = dict(id=None, title=None, selection=None, opponent=None, kind=None, group=None, status="mapped", reason=None,
           odds=None, decimal_odds=None, book_prob=None, model_prob=None, market_prob=None, market_exchange=None,
           market_spread=None, quotes=(), flags=(), box=None, n_legs=1)


def _tight(quotes, max_spread):
    """The usable quotes: both a bid and an ask, and ask - bid <= max_spread; each with its mid."""
    out = []
    for q in quotes or ():
        bid, ask = q.get("bid"), q.get("ask")
        if bid is None or ask is None or ask - bid > max_spread + EPS:
            continue
        out.append(dict(q, mid=(bid + ask) / 2, spread=ask - bid))
    return out


def _pick(quotes, max_spread):
    """The tightest usable quote (the freshest on a tie), or None."""
    ok = _tight(quotes, max_spread)
    if not ok:
        return None
    best = min(round(q["spread"], 9) for q in ok)
    return max((q for q in ok if round(q["spread"], 9) == best), key=lambda q: q.get("synced_utc") or "")


def _kind_label(kind):
    from racinglines.markets import kinds as K
    k = K.KINDS.get(kind)
    return k.label if k is not None and k.label else (kind or "Other")


def rows_from_priced(priced, settings, book=None):
    """(meta, rows) from a price_book result. `book` (a loaded book file) supplies `box` and `[book] image` when the
    priced JSON predates them."""
    max_spread = settings["market"]["max_spread"]
    boxes = {}
    if book is not None:
        boxes = {ln.get("id", str(i + 1)): ln.get("box") for i, ln in enumerate(book["lines"])}
    rows, runs, syncs, names = [], {}, {}, []
    for r in priced["lines"]:
        legs = r.get("legs") or []
        row = dict(ROW, id=r["id"], title=r["title"], selection=r["selection"], kind=r["kind"], status=r["status"],
                   reason=r.get("reason"), odds=r.get("odds"), decimal_odds=r.get("decimal_odds"),
                   book_prob=r.get("book_prob"), model_prob=r.get("model_prob"), flags=tuple(r.get("flags") or ()),
                   box=r.get("box") or boxes.get(r["id"]), n_legs=r.get("n_legs") or len(legs) or 1)
        if row["decimal_odds"] is None and row["odds"] is not None:
            row["decimal_odds"] = 1.0 / S.to_prob(row["odds"], priced["book"]["odds"])
        for lg in legs:
            if lg.get("run_id") is not None:
                runs[lg["run_id"]] = lg.get("run_source")
            for q in lg.get("quotes") or ():
                if q.get("synced_utc") and q["synced_utc"] > syncs.get(q["exchange"], ""):
                    syncs[q["exchange"]] = q["synced_utc"]
            if lg.get("event") and lg["event"] not in names:
                names.append(lg["event"])
        if len(legs) == 1:
            row.update(opponent=legs[0].get("opponent"), quotes=tuple(legs[0].get("quotes") or ()),
                       group=_kind_label(row["kind"]))
            q = _pick(row["quotes"], max_spread)
            if q is not None:
                row.update(market_prob=q["mid"], market_exchange=q["exchange"], market_spread=q["spread"])
        elif legs:
            picks = [_pick(lg.get("quotes"), max_spread) for lg in legs]
            row["group"] = "Slips"
            if all(p is not None for p in picks):
                p = 1.0
                for q in picks:
                    p *= q["mid"]
                row.update(market_prob=p, market_exchange="product of legs",
                           market_spread=max(q["spread"] for q in picks))
        else:
            row["group"] = "Unmapped"
        rows.append(row)
    b = priced["book"]
    meta = dict(venue=b.get("venue"), event=b.get("event"), sport=b.get("sport"), captured_utc=b.get("captured_utc"),
                event_name=" + ".join(names) or f"event {b.get('event')}", odds=b.get("odds"),
                image=(book or {}).get("book", {}).get("image"),
                models=[f"run {rid}" + (f" ({src})" if src else "") for rid, src in sorted(runs.items())],
                model_source="racinglines model runs (markets/venues.pricing_run unless --run)",
                syncs=dict(sorted(syncs.items())))
    return meta, rows


def rows_from_cycling(path):
    """(meta, rows) from `racinglines model price cycling` output: a folder holding futures.csv and/or matchups.csv (and
    settings.csv), or one of those files. Model only: no exchange lists these races, so the v2 rule (which needs a
    market price) passes on every line and says so. The cycling model's own prices are used as they are."""
    import pandas as pd
    p = Path(path)
    folder = p if p.is_dir() else p.parent
    files = {f: folder / f for f in ("futures.csv", "matchups.csv", "settings.csv")}
    if p.is_file():
        files = {k: v for k, v in files.items() if v == p or k == "settings.csv"}
    if not any(files[f].exists() for f in ("futures.csv", "matchups.csv") if f in files):
        raise RenderError(f"{path}: no futures.csv or matchups.csv (racinglines model price cycling writes them)")
    rows = []
    fut = files.get("futures.csv")
    if fut is not None and fut.exists():
        for r in pd.read_csv(fut).to_dict("records"):
            odds = None if pd.isna(r.get("book")) else float(r["book"])
            rows.append(dict(ROW, id=f"win-{r['rider']}", title="Winner", selection=str(r["rider"]), kind="winner",
                             group="Win", odds=odds, decimal_odds=odds, book_prob=None if odds is None else 1 / odds,
                             model_prob=float(r["win_p"])))
    mu = files.get("matchups.csv")
    if mu is not None and mu.exists():
        for r in pd.read_csv(mu).to_dict("records"):
            for a, b, o, pr in ((r["a"], r["b"], r["a_odds"], r["a_p"]), (r["b"], r["a"], r["b_odds"], r["b_p"])):
                rows.append(dict(ROW, id=f"h2h-{a}-{b}", title="Head to Head", selection=str(a), opponent=str(b),
                                 kind="h2h", group="Head-to-head", odds=float(o), decimal_odds=float(o),
                                 book_prob=1 / float(o), model_prob=float(pr)))
    settings = files.get("settings.csv")
    sline = ""
    if settings is not None and settings.exists():
        s = pd.read_csv(settings).iloc[0].to_dict()
        sline = " · settings " + ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items())
    mtime = max(f.stat().st_mtime for f in files.values() if f.exists())
    meta = dict(venue=None, event=folder.name, sport="road_cycling", captured_utc=None, event_name=folder.name,
                odds="decimal", image=None, models=[f"road-cycling model, {folder.name}/" + sline],
                model_source=f"racinglines model price cycling output, written "
                             f"{datetime.fromtimestamp(mtime, timezone.utc):%Y-%m-%dT%H:%MZ}",
                syncs={})
    return meta, rows


# --- sizing --------------------------------------------------------------------------------------------------------

def _pct(x, signed=False):
    return "–" if x is None else (f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%")


def size(rows, settings):
    """Each row + ev_model, ev_market, kelly_stake (before the cap), stake, verdict (BET / PASS / –) and note.
    [stake]: bet when EV >= min_ev against every probability in `require`; kelly x Kelly on the lowest of them, times
    `bank`; at least `floor`; the whole book scaled down to `cap` (floor bets kept at the floor, the rest scaled into
    what's left); then x `scale`, and a stake under `drop_under` dropped. Returns (rows, summary)."""
    st = settings["stake"]
    out = []
    for r in rows:
        r = dict(r)
        d = r["decimal_odds"]
        probs = dict(model=r["model_prob"], market=r["market_prob"])
        r["ev_model"] = None if probs["model"] is None or d is None else probs["model"] * d - 1
        r["ev_market"] = None if probs["market"] is None or d is None else probs["market"] * d - 1
        r.update(kelly_stake=0.0, stake=0.0, verdict="PASS", note=None)
        if r["status"] != "mapped":
            r.update(verdict="–", note="unmapped" + (f": {r['reason']}" if r.get("reason") else ""))
        elif d is None:
            r.update(verdict="–", note="no book odds")
        else:
            missing = [n for n in st["require"] if probs.get(n) is None]
            short = [n for n in st["require"] if n not in missing and probs[n] * d - 1 < st["min_ev"] - EPS]
            if missing or short:
                r["note"] = "; ".join(([f"no {' or '.join(missing)} price: no bet under the rule"] if missing else []) +
                                      [f"EV vs {n} {_pct(probs[n] * d - 1, True)} < {_pct(st['min_ev'])}" for n in short])
            else:
                p = min(probs[n] for n in st["require"])
                r["kelly_stake"] = st["bank"] * st["kelly"] * (p * d - 1) / (d - 1)
        out.append(r)
    bets = [r for r in out if r["kelly_stake"] > 0]
    stakes = {id(r): max(r["kelly_stake"], st["floor"]) for r in bets}
    total = sum(stakes.values())
    capped = total > st["cap"] + EPS
    if capped:
        if len(bets) * st["floor"] >= st["cap"]:
            stakes = {k: st["cap"] / len(bets) for k in stakes}
        else:
            fixed = set()
            while True:
                free = [r for r in bets if id(r) not in fixed]
                room = st["cap"] - st["floor"] * len(fixed)
                k = room / sum(r["kelly_stake"] for r in free)
                low = [r for r in free if r["kelly_stake"] * k < st["floor"]]
                if not low:
                    stakes.update({id(r): r["kelly_stake"] * k for r in free})
                    stakes.update({id(r): st["floor"] for r in bets if id(r) in fixed})
                    break
                fixed |= {id(r) for r in low}
    for r in bets:
        s = stakes[id(r)] * st["scale"]
        if s < st["drop_under"] - EPS:
            r.update(stake=0.0, verdict="PASS", note=f"${s:.2f} after x{st['scale']:g} is under ${st['drop_under']:.2f}")
        else:
            r.update(stake=round(s, 2), verdict="BET")
    summary = dict(n_bets=sum(r["verdict"] == "BET" for r in out), total=round(sum(r["stake"] for r in out), 2),
                   kelly_total=round(sum(stakes.values()), 2) if not capped else round(total, 2), capped=capped,
                   rule=rule_text(settings))
    return out, summary


def rule_text(settings):
    st = settings["stake"]
    req = " and ".join(st["require"])
    kelly = {0.5: "half", 0.25: "quarter", 1.0: "full"}.get(st["kelly"], f"{st['kelly']:g}x")
    txt = (f"Bet when EV >= {st['min_ev'] * 100:g}% against the {req}; {kelly} Kelly on the lower of them on a "
           f"${st['bank']:,.0f} bank; ${st['floor']:g} minimum; the book's total scaled to ${st['cap']:,.0f}")
    if st["scale"] != 1:
        txt += f"; then x{st['scale']:g}"
    if st["drop_under"]:
        txt += f", dropping stakes under ${st['drop_under']:.2f}"
    return txt + ". Market = tightest two-sided exchange quote with a spread of " \
                 f"{settings['market']['max_spread'] * 100:g} points or less."


# --- agnostic numbers ----------------------------------------------------------------------------------------------

def ticks(row, settings):
    """[(exchange, mid)] for the agnostic graphic: two-sided quotes with ask - bid <= [agnostic] max_spread."""
    return [(q["exchange"], q["mid"]) for q in _tight(row["quotes"], settings["agnostic"]["max_spread"])]


def min_odds(row, settings):
    """margin / min(model, every tight mid): the lowest decimal odds worth taking (the model alone with no tight mid)."""
    ps = [row["model_prob"]] + [m for _, m in ticks(row, settings)]
    ps = [p for p in ps if p is not None and p > 0]
    return None if not ps else settings["agnostic"]["margin"] / min(ps)


# --- html ----------------------------------------------------------------------------------------------------------

def _e(x):
    return html.escape("" if x is None else str(x))


def _exname(settings, code):
    return settings["render"]["exchanges"].get(code, {}).get("name", code)


def _excolor(settings, code):
    return settings["render"]["exchanges"].get(code, {}).get("color", "#a9b4c0")


def footer(meta, settings, agnostic=False):
    """The provenance every output carries, starting with the compact source note (racinglines/attribution.py): the
    sport's results feed(s), the exchanges quoted and, unless agnostic, the book's lines as generic sportsbook lines
    (never the book's brand)."""
    from racinglines import attribution
    note = attribution.note(meta.get("sport"), list(meta.get("syncs") or ()), sportsbook=not agnostic)["text"]
    parts = ([note] if note else []) + ["Model: " + ("; ".join(meta["models"]) or "none") + f" · source: {meta['model_source']}"]
    if meta["syncs"]:
        parts.append("Markets synced: " + ", ".join(f"{_exname(settings, x)} {t}" for x, t in meta["syncs"].items()))
    else:
        parts.append("Markets: no exchange quote")
    if meta.get("captured_utc") and not agnostic:
        parts.append(f"Board captured {meta['captured_utc']}")
    parts.append(f"Model {CALIBRATION}")
    parts.append("Not betting advice")
    return " · ".join(parts) + "."


def _page(body, cls, title, extra_css=""):
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            f"<title>{_e(title)}</title><style>{CSS.read_text()}{extra_css}</style></head>"
            f'<body class="{cls}">{body}<script>document.body.setAttribute("data-h",'
            f"Math.ceil(document.documentElement.scrollHeight));document.body.setAttribute(\"data-w\","
            f"Math.ceil(document.documentElement.scrollWidth));</script></body></html>")


def _label(r, settings):
    """The per-line content of the hud and card: (verdict class, model, market, EVs, stake or note)."""
    mk = "no usable market" if r["market_prob"] is None else \
        f"{_exname(settings, r['market_exchange'])} {_pct(r['market_prob'])}"
    ev = f"EV model {_pct(r['ev_model'], True)} · market {_pct(r['ev_market'], True)}"
    act = f"BET ${r['stake']:.2f}" if r["verdict"] == "BET" else ("PASS" if r["verdict"] == "PASS" else "not priced")
    cls = {"BET": "bet", "PASS": "pass"}.get(r["verdict"], "na")
    return cls, f"model {_pct(r['model_prob'])}", mk, ev, act


def _name(r):
    return _e(r["selection"]) + (f' <small>over {_e(r["opponent"])}</small>' if r.get("opponent") else "")


def _side_item(r, settings):
    cls, m, mk, ev, act = _label(r, settings)
    note = f'<div class="why">{_e(r["note"])}</div>' if r["note"] and r["verdict"] != "BET" else ""
    odds = "" if r["decimal_odds"] is None else f" · {r['decimal_odds']:.2f}"
    return (f'<div class="si {cls}"><div class="sh"><span class="sn">{_name(r)}</span><span class="act">{_e(act)}</span>'
            f'</div><div class="sd">{_e(r["title"])}{_e(odds)} · {_e(m)} · {_e(mk)}</div><div class="sd">{_e(ev)}</div>'
            f"{note}</div>")


def _image(path):
    """(data URI, width, height) of a screenshot."""
    raw = Path(path).read_bytes()
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "png")
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
    except ImportError:
        if raw[:8] != b"\x89PNG\r\n\x1a\n":
            raise RenderError(f"{path}: without Pillow only PNG screenshots can be measured")
        w, h = int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    return f"data:image/{mime};base64,{base64.b64encode(raw).decode()}", w, h


def _summary(summary):
    return (f'<div class="tot"><b>{summary["n_bets"]} bet{"s" if summary["n_bets"] != 1 else ""} · '
            f'${summary["total"]:.2f}</b>' + (" (scaled to the cap)" if summary["capped"] else "") +
            f'</div><div class="rule">{_e(summary["rule"])}</div>')


def html_hud(meta, rows, summary, settings, image):
    uri, w, h = _image(image)
    side = settings["render"]["side_panel_px"]
    labels, panel = [], []
    for r in rows:
        box = r.get("box")
        if not box:
            panel.append(_side_item(r, settings))
            continue
        x, y, bw, bh = box
        cls, m, mk, ev, act = _label(r, settings)
        fs = max(8.0, min(15.0, bh / 4.0))                   # three lines of text fill the box's height
        labels.append(f'<div class="box {cls}" style="left:{x}px;top:{y}px;width:{bw}px;height:{bh}px;'
                      f'font-size:{fs:.1f}px"><div class="pill"><div><b>{_e(act)}</b></div><div>{_e(m)} · {_e(mk)}</div>'
                      f"<div>{_e(ev)}</div></div></div>")
    head = (f'<div class="kicker">{_e(meta["event_name"])}</div><h2>Our prices on this board</h2>'
            f'<div class="sub">{_e(meta["venue"] or "")} · captured {_e(meta["captured_utc"] or "–")}</div>')
    rest = ("<div class=\"ph\">Lines without a box on the screenshot</div>" + "".join(panel)) if panel else ""
    body = (f'<div class="stage" style="width:{w + side}px;min-height:{h}px">'
            f'<div class="shot" style="width:{w}px;height:{h}px;background-image:url({uri})">{"".join(labels)}</div>'
            f'<div class="side" style="width:{side}px;left:{w}px;min-height:{h}px">{head}{_summary(summary)}{rest}'
            f'<div class="sfoot">{_e(footer(meta, settings))}<div class="brand">racinglines.bet</div></div></div></div>')
    return _page(body, "hud", f"{meta['event_name']} board")


def _groups(rows):
    out = {}
    for r in rows:
        out.setdefault(r["group"] or "Other", []).append(r)
    return out


def html_card(meta, rows, summary, settings):
    panels = []
    for g, rs in _groups(rows).items():
        body = ('<div class="crow ch"><div class="nm">selection</div><div>odds</div><div>model</div><div>market</div>'
                '<div>EV model</div><div>EV market</div><div>stake</div></div>')
        for r in rs:
            cls, *_ = _label(r, settings)
            mk = "–" if r["market_prob"] is None else \
                (f'{_pct(r["market_prob"])}<small style="color:{_excolor(settings, r["market_exchange"])}">'
                 f'{_e(_exname(settings, r["market_exchange"]))}</small>')
            act = f"${r['stake']:.2f}" if r["verdict"] == "BET" else r["verdict"]
            why = f'<div class="why">{_e(r["note"])}</div>' if r["note"] and r["verdict"] != "BET" else ""
            odds = "–" if r["decimal_odds"] is None else f"{r['decimal_odds']:.2f}"
            body += (f'<div class="crow {cls}"><div class="nm">{_name(r)}{why}</div><div>{odds}</div>'
                     f'<div class="m">{_pct(r["model_prob"])}</div><div class="mk">{mk}</div>'
                     f'<div class="ev">{_pct(r["ev_model"], True)}</div><div class="ev">{_pct(r["ev_market"], True)}</div>'
                     f'<div class="act">{_e(act)}</div></div>')
        panels.append(f'<div class="panel"><div class="ph">{_e(g)} <em>{len(rs)} line{"s" if len(rs) != 1 else ""}</em>'
                      f"</div>{body}</div>")
    body = (f'<div class="kicker">{_e(meta["event_name"])}{" · " + _e(meta["venue"]) if meta["venue"] else ""}</div>'
            f"<h1>Our prices on this board</h1>"
            f'<div class="sub">The model\'s chance and the tightest prediction-market price for each line on the board, '
            f"the expected value against each, and the stake the sizing rule gives.</div>"
            f'<div class="panel sum">{_summary(summary)}</div>{"".join(panels)}'
            f'<div class="foot"><div>{_e(footer(meta, settings))}</div><div class="brand">racinglines.bet</div></div>')
    return _page(body, "social card", f"{meta['event_name']} prices")


def html_agnostic(meta, rows, settings):
    groups = {}
    for r in rows:
        if r["status"] == "mapped" and r["model_prob"] is not None and r["n_legs"] == 1:
            groups.setdefault(r["group"] or "Other", []).append(r)
    left_out = sum(1 for r in rows if r["status"] != "mapped" or r["model_prob"] is None or r["n_legs"] != 1)
    seen = set()
    panels = []
    for g, rs in groups.items():
        rs = sorted(rs, key=lambda r: -r["model_prob"])
        top = max([r["model_prob"] for r in rs] + [m for r in rs for _, m in ticks(r, settings)])
        scale = min(1.0, max(0.1, -(-top * 10 // 1) / 10))
        body = ('<div class="prow colh"><div></div><div></div><div>model</div><div>markets</div>'
                '<div>min odds</div></div>')
        for r in rs:
            tk = ticks(r, settings)
            seen |= {x for x, _ in tk}
            marks = "".join(f'<div class="tk" style="left:{min(m / scale, 1) * 100:.2f}%;'
                            f'background:{_excolor(settings, x)}"></div>' for x, m in tk)
            mk = "".join(f'<span style="color:{_excolor(settings, x)}">{_pct(m)}</span><br>' for x, m in tk) or \
                '<span class="muted">–</span>'
            mo = min_odds(r, settings)
            body += (f'<div class="prow"><div class="nm">{_name(r)}</div>'
                     f'<div class="track"><div class="fill" style="width:{min(r["model_prob"] / scale, 1) * 100:.2f}%">'
                     f'</div>{marks}</div><div class="pc">{_pct(r["model_prob"])}</div><div class="mk">{mk}</div>'
                     f'<div class="mo">{"–" if mo is None else f"{mo:.2f}"}</div></div>')
        sc = "".join(f"<span>{round(scale * i / 4 * 100):g}%</span>" for i in range(5))
        body += f'<div class="prow ticks"><div></div><div class="s">{sc}</div><div></div><div></div><div></div></div>'
        panels.append(f'<div class="panel"><div class="ph">{_e(g)} <em>our model\'s chance</em></div>{body}</div>')
    legend = ('<div class="legend"><span><i class="lm"></i>our model</span>' +
              "".join(f'<span><i style="background:{_excolor(settings, x)}"></i>{_e(_exname(settings, x))}</span>'
                      for x in sorted(seen)) +
              f"<span>market mid, shown where the spread is {settings['agnostic']['max_spread'] * 100:g} points or "
              f"less</span></div>")
    note = (f'<div class="note">Min odds = {settings["agnostic"]["margin"]:g} / the lower of our model and the market '
            f"mids shown.{f' Left out: {left_out} line(s) with no model price.' if left_out else ''}</div>")
    body = (f'<div class="kicker">{_e(meta["event_name"])}</div><h1>Matchups and potential winners to watch</h1>'
            f'<div class="sub">Our model\'s chance for each selection, beside the prediction markets where their '
            f"prices are tight.</div>{''.join(panels)}<div class=\"panel\">{legend}{note}</div>"
            f'<div class="foot"><div>{_e(footer(meta, settings, agnostic=True))}</div>'
            f'<div class="brand">racinglines.bet</div></div>')
    return _page(body, "social agnostic", f"{meta['event_name']} to watch")


def render(meta, rows, style, settings, image=None):
    """(html, sized rows, summary)."""
    if style not in STYLES:
        raise RenderError(f"style {style!r} not in {STYLES}")
    sized, summary = size(rows, settings)
    if style == "hud":
        if not image:
            raise RenderError("--style hud needs the book's screenshot: --image PNG (or [book] image in the book file)")
        return html_hud(meta, sized, summary, settings, image), sized, summary
    if style == "card":
        return html_card(meta, sized, summary, settings), sized, summary
    return html_agnostic(meta, sized, settings), sized, summary


# --- png -----------------------------------------------------------------------------------------------------------

def find_chrome(explicit=None):
    """A headless-capable Chromium: --chrome, then $RACINGLINES_CHROME, then common names on PATH, then the
    Playwright and macOS install locations. None when there is none."""
    for c in (explicit, os.environ.get("RACINGLINES_CHROME")):
        if c:
            if Path(c).exists() or shutil.which(c):
                return shutil.which(c) or c
            raise RenderError(f"Chromium {c!r} not found")
    for n in CHROME_NAMES:
        if shutil.which(n):
            return shutil.which(n)
    for g in CHROME_GLOBS:
        hits = sorted(glob.glob(g))
        if hits:
            return hits[-1]
    return None


def _chrome(chrome, *args, timeout=120):
    base = [chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
            "--allow-file-access-from-files", "--run-all-compositor-stages-before-draw", "--virtual-time-budget=5000"]
    with tempfile.TemporaryDirectory() as prof:
        return subprocess.run(base + [f"--user-data-dir={prof}", *args], capture_output=True, text=True,
                              timeout=timeout)


def to_png(html_path, png_path, chrome=None, dpr=1):
    """Screenshot the page at its full size. One pass in a small window reads the size the page measured (data-w,
    data-h on <body>); the second takes the screenshot in a window with room to spare (headless Chromium's viewport
    is shorter than its window), cropped to the page with Pillow when it is installed."""
    exe = find_chrome(chrome)
    if exe is None:
        raise RenderError("no Chromium found for --png: pass --chrome PATH or set RACINGLINES_CHROME "
                          f"(looked for {', '.join(CHROME_NAMES)} on PATH and {', '.join(CHROME_GLOBS)})")
    url = Path(html_path).resolve().as_uri()
    dom = _chrome(exe, "--window-size=320,240", "--dump-dom", url)
    w = re.search(r'data-w="(\d+)"', dom.stdout)
    h = re.search(r'data-h="(\d+)"', dom.stdout)
    if not (w and h):
        raise RenderError(f"Chromium could not measure the page: {dom.stderr.strip()[-400:]}")
    w, h = int(w.group(1)), int(h.group(1))
    png = Path(png_path).resolve()
    shot = _chrome(exe, f"--window-size={w},{h + 200}", f"--force-device-scale-factor={dpr}", f"--screenshot={png}", url)
    if not png.exists():
        raise RenderError(f"Chromium wrote no screenshot: {shot.stderr.strip()[-400:]}")
    try:
        from PIL import Image
    except ImportError:
        return png
    with Image.open(png) as im:
        box = (0, 0, min(im.width, round(w * dpr)), min(im.height, round(h * dpr)))
        im.crop(box).save(png)
    return png


# --- loose text into a book file -----------------------------------------------------------------------------------

_ODD = {"decimal": r"\d+(?:\.\d+)?", "american": r"[+-]\d{3,}", "fractional": r"\d+/\d+"}
_ODD["prob"] = _ODD["dollars"] = r"0?\.\d+"
_ODD["cents"] = r"\d{1,2}"
_EG = {"decimal": "1.57", "american": "+150", "fractional": "4/7", "prob": ".64", "dollars": ".64", "cents": "64"}
_H2H_WORD = re.compile(r"\s(?:vs\.?|v\.?|over|beats|ahead of)\s", re.I)


def _odds(tok, fmt):
    v = tok if fmt == "fractional" else (int(tok) if fmt in ("american", "cents") else float(tok))
    return None if S.check_odds(v, fmt) else v


def parse_text(text, odds="decimal"):
    """(lines, unparsed) from loose text, one bet per line:
        A over B 1.57            A finishes ahead of B, at 1.57
        A vs B 1.57 2.25         both sides of the matchup: A over B at 1.57, B over A at 2.25
        Name 4.35                Name, on the current title (default "Winner")
        Race Winner:             a line ending in a colon sets the title of the lines after it
    Titles and selections stay as written; each line's market is left "unmapped" for `book map` and review, with the
    shape it was read as in `parsed`. A line that matches none of these (or whose odds aren't odds in `odds`) is
    returned in `unparsed` with the reason, never guessed: so is a name that ends in a price ("A 1.57 2.25") and a
    single price on a head to head ("A vs B 1.57")."""
    if odds not in _ODD:
        raise RenderError(f"--odds {odds!r} not in {tuple(_ODD)}")
    o, o_eg = _ODD[odds], _EG[odds]
    pats = (("vs", re.compile(rf"^(?P<a>.+?)\s+(?:vs\.?|v\.?)\s+(?P<b>.+?)\s+(?P<oa>{o})\s+(?P<ob>{o})$", re.I)),
            ("over", re.compile(rf"^(?P<a>.+?)\s+(?:over|beats|ahead of)\s+(?P<b>.+?)\s+(?P<oa>{o})$", re.I)),
            ("single", re.compile(rf"^(?P<a>.*?[^\d\s+/.-].*?)\s+(?P<oa>{o})$")))
    lines, bad = [], []
    title = None
    for n, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.endswith(":"):
            title = s[:-1].strip()
            continue
        for shape, pat in pats:
            m = pat.match(s)
            if m:
                break
        else:
            bad.append((n, s, f"no pattern matched (odds read as {odds})"))
            continue
        g = m.groupdict()
        vals = [_odds(g[k], odds) for k in ("oa", "ob") if g.get(k)]
        if any(v is None for v in vals):
            bad.append((n, s, f"odds are not valid {odds} odds"))
            continue
        a, b = g["a"].strip(), (g.get("b") or "").strip()
        if any(x and re.fullmatch(o, x.split()[-1]) for x in (a, b)):
            bad.append((n, s, f"a name ends in a price: too many prices for a {shape} line (one name \"A {o_eg}\", "
                              f"one side \"A over B {o_eg}\", both sides \"A vs B {o_eg} {o_eg}\")"))
            continue
        if shape == "single" and _H2H_WORD.search(f" {a} "):
            bad.append((n, s, "a head to head needs a price per side (\"A vs B p1 p2\") or \"A over B p\""))
            continue
        if shape == "single":
            lines.append(dict(title=title or "Winner", selection=a, odds=vals[0], parsed=f"winner: {a}", src=n))
            continue
        t = title or "Head to Head"
        lines.append(dict(title=t, selection=a, odds=vals[0], parsed=f"h2h: {a} over {b}", src=n))
        if shape == "vs":
            lines.append(dict(title=t, selection=b, odds=vals[1], parsed=f"h2h: {b} over {a}", src=n))
    return lines, bad


def _toml_str(s):
    return json.dumps(s, ensure_ascii=False)


def book_toml(lines, unparsed, venue, event, odds, sport=None, currency="USD", captured_utc=None, source_ref=None):
    """A book file draft (validated): every line "unmapped" with the shape it was read as in its note, and the
    unparsed input listed as comments at the top."""
    captured = captured_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    out = ["# Book file draft from `racinglines book new --from-text`" + (f" ({source_ref})" if source_ref else "") + ".",
           "# Every market is left \"unmapped\": write each one's market table (docs/sportsbook/index.md), then",
           "# `racinglines book map` this file before pricing it."]
    if unparsed:
        out.append("# Input lines not parsed (fix and re-run, or add them by hand):")
        out += [f"#   line {n}: {s}  ({why})" for n, s, why in unparsed]
    out += ["", "[book]", f"venue = {_toml_str(venue)}", f"event = {_toml_str(event)}"]
    if sport:
        out.append(f"sport = {_toml_str(sport)}")
    out += [f"captured_utc = {_toml_str(captured)}", 'source = "paste"', f"odds = {_toml_str(odds)}",
            f"currency = {_toml_str(currency)}"]
    for i, ln in enumerate(lines, 1):
        v = ln["odds"]
        out += ["", "[[lines]]", f'id = "{i}"', f"title = {_toml_str(ln['title'])}",
                f"selection = {_toml_str(ln['selection'])}",
                f"odds = {_toml_str(v) if isinstance(v, str) else v}",
                f"note = {_toml_str('parsed as ' + ln['parsed'] + ' (input line ' + str(ln['src']) + ')')}",
                'market = "unmapped"']
    txt = "\n".join(out) + "\n"
    S.validate_book(tomllib.loads(txt), "draft")
    return txt
