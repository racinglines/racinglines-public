/* racinglines web app: the little client-side behaviour the server-rendered pages need.
   Partial updates (the Live tab, the Lab's sections and Edge Finder) are HTMX attributes in the templates;
   this file adds what attributes can't: the CSRF token on every HTMX POST, an error toast, browser-side view
   settings (which Lab sections are open, the jobs filter), the replay player, the sweep form's "what changed"
   highlighting, a generic pressed-button toggle (the Positions chart's Polymarket / Private book switch), and the
   three "less on the page at once" pieces: page tabs, remembered collapsed sections, and table row caps / filters. */
(function () {
  "use strict";
  const NS = document.body.dataset.ns || "";                // a demo session's own namespace (web/demo.py)
  const store = {
    get(k, d) { try { const v = localStorage.getItem(NS + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(NS + k, JSON.stringify(v)); } catch (e) {} },
  };
  const csrf = () => (document.querySelector("[name=csrf_token]") || {}).value || "";

  // --- HTMX: every non-GET request carries the form's CSRF token; a refused request shows its message ---
  document.addEventListener("htmx:configRequest", e => {
    if (e.detail.verb !== "get") e.detail.parameters.csrf_token = csrf();
  });
  let lastToast = 0;                                          // polling must not stack a toast per failed request
  document.addEventListener("htmx:responseError", e => {
    if (e.detail.requestConfig && e.detail.requestConfig.verb === "get") {   // polls: at most one toast a minute
      if (Date.now() - lastToast < 60000) return;
      lastToast = Date.now();
    }
    const x = e.detail.xhr, text = (x.responseText || "").trim();
    let msg = text;
    try { msg = JSON.parse(text).detail || text; } catch (err) {}
    toast(msg || `Request failed (${x.status})`);
  });
  function toast(msg) {
    let t = document.getElementById("toast");
    if (!t) { t = document.createElement("div"); t.id = "toast"; t.className = "flash bad toast"; t.setAttribute("role", "alert");
      document.body.appendChild(t); }
    t.textContent = msg; clearTimeout(toast.timer); toast.timer = setTimeout(() => t.remove(), 6000);
  }

  // --- generic pressed toggles: <button data-toggle="plot" data-value="x"> shows [data-plot="x"], hides the others;
  //     data-value="*" shows them all; special case: data-toggle="venue" toggles .m markers by data-venue ---
  document.addEventListener("click", e => {
    const b = e.target.closest("button[data-toggle]");
    if (!b) return;
    const group = b.dataset.toggle, v = b.dataset.value;
    document.querySelectorAll(`button[data-toggle="${group}"]`).forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    if (group === "venue") {
      // Toggle venue indicators on bars
      document.querySelectorAll(".bar .m").forEach(m => {
        m.style.display = m.dataset.venue === v ? "" : "none";
      });
      // Toggle gap values in the Gap column
      document.querySelectorAll(".gap-cell .gap-value").forEach(g => {
        g.style.display = g.dataset.venue === v ? "" : "none";
      });
    } else {
      document.querySelectorAll(`[data-${group}]:not(button)`).forEach(p => { p.hidden = v !== "*" && p.dataset[group] !== v; });
    }
  });

  // --- page tabs: <nav class="ptabs" data-tabs="name"><button data-tab="k">…</button></nav> shows [data-panel="k"] and
  //     hides the other panels. One set per page; the choice is remembered per page, and #k in the URL opens a tab
  //     (so does a link to any element inside one). Content re-rendered by HTMX gets the same tab back. ---
  // The panels of a tab nav are the [data-panel] elements under the nav's parent (not nested in another panel), so a
  // page with a second set (or a panel holding its own tabs) is not hidden by this one. Tabs get ids and the panels
  // role=tabpanel + aria-labelledby them; the selected tab is also the only one in the tab order.
  function tabPanels(nav) {
    const root = nav.parentElement || document;
    return [...root.querySelectorAll("[data-panel]")].filter(p => !p.parentElement.closest("[data-panel]"));
  }
  function tabsApply(nav, key, setHash) {
    const btns = [...nav.querySelectorAll("[data-tab]")];
    if (!btns.some(b => b.dataset.tab === key)) key = btns.length ? btns[0].dataset.tab : null;
    const idOf = k => "tab-" + nav.dataset.tabs + "-" + k;
    btns.forEach(b => { b.id = idOf(b.dataset.tab); b.setAttribute("aria-selected", String(b.dataset.tab === key)); });
    tabPanels(nav).forEach(p => {
      p.hidden = p.dataset.panel !== key; p.setAttribute("role", "tabpanel");
      if (btns.some(b => b.dataset.tab === p.dataset.panel)) p.setAttribute("aria-labelledby", idOf(p.dataset.panel));
    });
    store.set("tabs." + nav.dataset.tabs, key);
    if (setHash && key) history.replaceState(null, "", "#" + key);
  }
  // --- Lab Run: pick a sport, then a job; only that job's form shows (no JS: every form shows) ---
  function runPickInit(root) {
    (root.querySelectorAll ? root : document).querySelectorAll("[data-run-pick]").forEach(pick => {
      const box = pick.parentElement, sportSel = pick.querySelector("[data-run-sport]"), jobSel = pick.querySelector("[data-run-job]");
      const secs = [...box.querySelectorAll(".run-sport[data-sport]")];
      if (!secs.length) return;
      const show = () => {
        secs.forEach(sec => {
          const on = sec.dataset.sport === sportSel.value;
          sec.hidden = !on;
          sec.querySelectorAll(".run-job").forEach(j => { j.hidden = !on || j.dataset.job !== jobSel.value; });
        });
      };
      const fill = (want) => {
        const sec = secs.find(x => x.dataset.sport === sportSel.value);
        const jobs = [...sec.querySelectorAll(".run-job")];
        jobSel.replaceChildren(...jobs.map(j => new Option(j.dataset.label, j.dataset.job)));
        jobSel.value = jobs.some(j => j.dataset.job === want) ? want : (jobs[0] && jobs[0].dataset.job);
        show();
      };
      const sel = box.querySelector(".run-job[data-sel]");
      const first = sel ? sel.closest(".run-sport") : secs[0];
      sportSel.value = first.dataset.sport;
      fill(sel ? sel.dataset.job : null);
      sportSel.addEventListener("change", () => fill(null));
      jobSel.addEventListener("change", show);
      pick.hidden = false;
    });
  }
  function tabsInit(root) {
    const nav = root.querySelector ? root.querySelector("[data-tabs]") : null;
    if (!nav) return;
    let key = store.get("tabs." + nav.dataset.tabs, null);
    const h = location.hash.slice(1), el = h && document.getElementById(h);
    if (h && nav.querySelector(`[data-tab="${h}"]`)) key = h;
    else if (el && el.closest("[data-panel]")) key = el.closest("[data-panel]").dataset.panel;
    tabsApply(nav, key, false);
    if (el && el.closest("[data-panel]") && !nav.dataset.shown) { nav.dataset.shown = 1; el.scrollIntoView({block: "start"}); }
  }
  document.addEventListener("click", e => {
    const b = e.target.closest("[data-tabs] [data-tab]");
    if (b) tabsApply(b.closest("[data-tabs]"), b.dataset.tab, true);
  });

  // --- inline-handler replacements: <select data-autosubmit> submits its form on change; <form data-confirm="…">
  //     asks before submitting; <select data-sets-cutoff> copies the chosen option's data-cutoff into the form's
  //     cutoff field; <button data-copy="#textarea" data-copied="#status" data-copied-text="…"> copies the text ---
  document.addEventListener("change", e => {
    const t = e.target;
    if (t.matches && t.matches("[data-autosubmit]") && t.form) t.form.submit();
    if (t.matches && t.matches("[data-sets-cutoff]") && t.form && t.form.cutoff) t.form.cutoff.value = t.selectedOptions[0].dataset.cutoff;
  });
  document.addEventListener("submit", e => {
    const f = e.target;
    if (f.matches && f.matches("form[data-confirm]") && !confirm(f.dataset.confirm)) e.preventDefault();
  });
  document.addEventListener("click", e => {
    const b = e.target.closest("[data-copy]");
    if (!b) return;
    const t = document.querySelector(b.dataset.copy), done = document.querySelector(b.dataset.copied || "#none");
    if (!t) return;
    const ok = () => { if (done) done.textContent = b.dataset.copiedText || "Copied."; };
    const legacy = () => { t.select(); document.execCommand("copy"); ok(); };
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(t.value).then(ok, legacy);
    else legacy();
  });
  // the forgot-password page: the e-mail text follows the username box and feeds the mailto link
  function forgotInit() {
    const t = document.getElementById("forgot-email"), u = document.getElementById("forgot-user"), a = document.getElementById("forgot-send");
    if (!t || !u || !a) return;
    const base = t.value, support = t.dataset.support || "";
    function sync() {
      t.value = base.replace("YOUR USERNAME", (u.value || "").trim() || "YOUR USERNAME");
      const text = t.value.split("\n").slice(3).join("\n");           // the part after To and Subject
      a.href = "mailto:" + support + "?subject=" + encodeURIComponent("Password reset request") + "&body=" + encodeURIComponent(text);
    }
    u.addEventListener("input", sync); sync();
  }
  // the site notice: shown once per browser session (the sign-up variant has its own key and also sets the plain one;
  // the add-email popup, data-key="email", has its own key only)
  function noticeInit() {
    const d = document.getElementById("site-notice");
    if (!d) return;
    const k = "racinglines-notice-ok" + (d.dataset.signup ? "-signup" : d.dataset.key ? "-" + d.dataset.key : "");
    try { if (sessionStorage.getItem(k)) return; } catch (e) {}
    d.addEventListener("close", () => {
      try { sessionStorage.setItem(k, "1"); if (!d.dataset.key) sessionStorage.setItem("racinglines-notice-ok", "1"); } catch (e) {} });
    if (d.showModal) d.showModal(); else d.setAttribute("open", "");
  }

  // --- tooltips ([data-tooltip], CSS-only): open toward the left when the text would run past the window's right edge ---
  function tipSide(e) {
    const t = e.target.closest && e.target.closest("[data-tooltip]");
    if (t) t.classList.toggle("tip-end", t.getBoundingClientRect().left + 270 > document.documentElement.clientWidth);
  }
  document.addEventListener("mouseover", tipSide);
  document.addEventListener("focusin", tipSide);

  // --- collapsed sections: <details data-remember="id"> keeps its open / closed state per browser ---
  document.addEventListener("toggle", e => {
    const d = e.target;
    if (d.matches && d.matches("details[data-remember]")) store.set("open." + d.dataset.remember, d.open);
  }, true);
  function detailsInit(root) {
    root.querySelectorAll("details[data-remember]").forEach(d => {
      const v = store.get("open." + d.dataset.remember, null);
      if (v !== null) d.open = v;
    });
  }

  // --- a link to #id inside (or on) a closed <details> opens it and scrolls it into view ---
  function openHashDetails() {
    const el = location.hash && document.getElementById(location.hash.slice(1));
    if (!el) return;
    const d = el.matches("details") ? el : el.closest("details");
    if (d && !d.open) { d.open = true; el.scrollIntoView({block: "start"}); }
  }
  window.addEventListener("hashchange", openHashDetails);

  // --- table tools: <div class="scroll" data-rows="12" data-filter="Driver or team"> caps a long table at N rows
  //     behind a "Show all" button and, with data-filter, adds a box that filters the rows by their text ---
  function tableInit(root) {
    root.querySelectorAll(".scroll[data-rows], .scroll[data-filter]").forEach(box => {
      if (box.dataset.tt) return;
      box.dataset.tt = 1;
      const rows = [...box.querySelectorAll("tbody tr")], cap = +box.dataset.rows || 0;
      if (rows.length <= (cap || 0) && !box.dataset.filter) return;
      const bar = document.createElement("div"); bar.className = "tt";
      let all = !(cap && rows.length > cap), q = "";
      function apply() {
        let shown = 0;
        rows.forEach(r => {
          const ok = !q || r.textContent.toLowerCase().includes(q);
          r.hidden = !ok || (!all && !q && shown >= cap); if (!r.hidden) shown++;
        });
        if (more) { more.hidden = all || !!q; }
        if (count) count.textContent = q ? `${shown} of ${rows.length}` : (all ? "" : `${cap} of ${rows.length}`);
      }
      let inp = null, more = null, count = null;
      if (box.dataset.filter !== undefined) {
        inp = document.createElement("input"); inp.type = "search"; inp.placeholder = box.dataset.filter || "Filter…";
        inp.setAttribute("aria-label", "Filter rows");
        inp.addEventListener("input", () => { q = inp.value.trim().toLowerCase(); apply(); });
        bar.appendChild(inp);
      }
      count = document.createElement("span"); count.className = "legend"; bar.appendChild(count);
      if (cap && rows.length > cap) {
        more = document.createElement("button"); more.type = "button"; more.className = "link";
        more.textContent = `Show all ${rows.length}`;
        more.addEventListener("click", () => { all = true; apply(); });
        bar.appendChild(more);
      }
      box.parentNode.insertBefore(bar, box);
      apply();
    });
  }

  // --- calendar: <div data-calendar data-events="[...]"> a month grid (Mon-start). <select data-cal-filter="sport|
  //     exchange"> and <input data-cal-filter="q"> narrow which events show; a day over 3 events gets a "+N more".
  //     Month navigation and filtering are all client-side (every event for the page is already in data-events). ---
  function calendarInit(root) {
    root.querySelectorAll("[data-calendar]").forEach(box => {
      if (box.dataset.tt) return;
      box.dataset.tt = 1;
      let events = [];
      try { events = JSON.parse(box.dataset.events || "[]"); } catch (e) {}
      const byDay = new Map();
      events.forEach(e => {
        if (!e.date) return;
        const key = e.date.slice(0, 10);
        (byDay.get(key) || byDay.set(key, []).get(key)).push(e);
      });
      const grid = box.querySelector("[data-cal-grid]"), label = box.querySelector("[data-cal-label]"),
            sport = box.querySelector('[data-cal-filter="sport"]'), exch = box.querySelector('[data-cal-filter="exchange"]'),
            q = box.querySelector('[data-cal-filter="q"]');
      const today = new Date(); today.setHours(0, 0, 0, 0);
      const todayKey = today.toISOString().slice(0, 10);
      let cur = new Date(today.getFullYear(), today.getMonth(), 1);

      function matches(e) {
        const s = sport ? sport.value : "", x = exch ? exch.value : "", qq = q ? q.value.trim().toLowerCase() : "";
        return (!s || e.sport === s) && (!x || (e.exchanges || []).includes(x)) && (!qq || (e.title || "").toLowerCase().includes(qq));
      }
      function dayCell(dayNum, dayEvents) {
        const c = document.createElement("div");
        c.className = "cal-cell" + (dayNum == null ? " empty" : "");
        if (dayNum == null) return c;
        const key = `${cur.getFullYear()}-${String(cur.getMonth() + 1).padStart(2, "0")}-${String(dayNum).padStart(2, "0")}`;
        if (key === todayKey) c.classList.add("today");
        const n = document.createElement("div"); n.className = "cal-day"; n.textContent = dayNum; c.appendChild(n);
        const shown = dayEvents.slice(0, 3);
        shown.forEach(e => {
          const a = document.createElement(e.url ? "a" : "div");
          a.className = "cal-ev"; if (e.url) a.href = e.url;
          a.title = e.title; a.textContent = e.title;
          c.appendChild(a);
        });
        if (dayEvents.length > shown.length) {
          const more = document.createElement("div"); more.className = "cal-more";
          more.textContent = `+${dayEvents.length - shown.length} more`; more.title = dayEvents.slice(3).map(e => e.title).join(", ");
          c.appendChild(more);
        }
        return c;
      }
      function render() {
        const y = cur.getFullYear(), m = cur.getMonth();
        if (label) label.textContent = cur.toLocaleString(undefined, { month: "long", year: "numeric" });
        const startOffset = (new Date(y, m, 1).getDay() + 6) % 7;         // Monday-start week
        const daysInMonth = new Date(y, m + 1, 0).getDate();
        grid.innerHTML = "";
        for (let i = 0; i < startOffset; i++) grid.appendChild(dayCell(null, []));
        for (let d = 1; d <= daysInMonth; d++) {
          const key = `${y}-${String(m + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
          grid.appendChild(dayCell(d, (byDay.get(key) || []).filter(matches)));
        }
      }
      const prev = box.querySelector("[data-cal-prev]"), next = box.querySelector("[data-cal-next]"),
            todayBtn = box.querySelector("[data-cal-today]");
      if (prev) prev.addEventListener("click", () => { cur.setMonth(cur.getMonth() - 1); render(); });
      if (next) next.addEventListener("click", () => { cur.setMonth(cur.getMonth() + 1); render(); });
      if (todayBtn) todayBtn.addEventListener("click", () => { cur = new Date(today.getFullYear(), today.getMonth(), 1); render(); });
      [sport, exch, q].forEach(el => el && el.addEventListener("input", render));
      render();
    });
  }
  // --- local time: <time class="localtime" datetime="ISO"> (the recorders legend) shows the browser's local time
  //     once this runs; the server-rendered UTC text is the no-JS fallback ---
  function localTimeInit(root) {
    root.querySelectorAll("time.localtime[datetime]").forEach(t => {
      if (t.dataset.localized) return;
      const d = new Date(t.getAttribute("datetime"));
      if (isNaN(d)) return;
      t.textContent = d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
      t.dataset.localized = "1";
    });
  }
  function initAll(root) { tabsInit(root); runPickInit(root); detailsInit(root); tableInit(root); calendarInit(root); localTimeInit(root); }
  document.addEventListener("htmx:afterSwap", e => initAll(e.detail.target));

  document.addEventListener("DOMContentLoaded", () => {
  initAll(document);
  openHashDetails();
  forgotInit();
  noticeInit();
  // --- /term: each replay waits paused on its full command; Play re-types it, then plays the output (no JS: all shown) ---
  function termPlay(t) {
    const cmd = t.querySelector("[data-term-cmd]"), out = t.querySelector("[data-term-out]"), btn = t.querySelector("[data-term-play]");
    const full = cmd.dataset.full;
    btn.disabled = true;
    const done = () => { t.removeAttribute("data-typing"); t.removeAttribute("data-running"); if (out) out.style.maxHeight = ""; btn.disabled = false; btn.textContent = "Replay"; };
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { if (out) out.hidden = false; return done(); }
    if (out) out.hidden = true; t.setAttribute("data-typing", ""); cmd.textContent = "";
    let i = 0;
    const run = () => {     // the "run": the output scrolls in line by line, like the command printing it
      if (!out) return done();     // a step shown but not run (pack record = false): typing is all there is
      t.removeAttribute("data-typing"); t.setAttribute("data-running", "");
      out.style.maxHeight = "0px"; out.hidden = false;
      const h = out.scrollHeight, step = Math.max(16, h / 120), t0 = performance.now();
      const grow = () => {
        const shown = Math.min(h, step * Math.floor((performance.now() - t0) / 25));
        out.style.maxHeight = shown + "px";
        shown < h ? requestAnimationFrame(grow) : done();
      };
      requestAnimationFrame(grow);
    };
    const tick = () => {
      cmd.textContent = full.slice(0, ++i);
      if (i < full.length) return setTimeout(tick, 28 + Math.random() * 40);
      setTimeout(run, 350);
    };
    setTimeout(tick, 250);
  }
  document.querySelectorAll("[data-term]").forEach(t => {
    const cmd = t.querySelector("[data-term-cmd]"), btn = t.querySelector("[data-term-play]");
    cmd.dataset.full = cmd.textContent;
    const out = t.querySelector("[data-term-out]");
    if (out) out.hidden = true;     // paused: only the command, in full
    btn.hidden = false;
    btn.addEventListener("click", () => termPlay(t));
  });

  // --- /term parts: one part at a time, picked from the tabs (phone) or sidebar (wide); the URL hash keeps the part ---
  const parts = document.querySelector("[data-term-parts]");
  if (parts) {
    const boxes = [...parts.querySelectorAll(".term-partbox")], tabs = [...parts.querySelectorAll(".term-nav [data-term-tab]")];
    const showPart = (id, scroll) => {
      if (!boxes.some(b => b.id === id)) id = boxes[0].id;
      boxes.forEach(b => { b.hidden = b.id !== id; });
      tabs.forEach(a => a.toggleAttribute("aria-current", a.dataset.termTab === id));
      const cur = tabs.find(a => a.dataset.termTab === id), nav = cur && cur.parentElement;
      if (nav && nav.scrollWidth > nav.clientWidth)       // phone tabs: bring the current one into view
        nav.scrollLeft = cur.offsetLeft - nav.offsetLeft - (nav.clientWidth - cur.offsetWidth) / 2;
      if (scroll) parts.scrollIntoView({block: "start"});
    };
    const fromHash = () => {     // #part-2, or a step's id inside a part
      const el = location.hash && document.getElementById(location.hash.slice(1));
      const box = el && el.closest(".term-partbox");
      return box ? box.id : null;
    };
    parts.addEventListener("click", e => {
      const a = e.target.closest("[data-term-tab]");
      if (!a) return;
      e.preventDefault();
      history.replaceState(null, "", "#" + a.dataset.termTab);
      showPart(a.dataset.termTab, true);
    });
    const toTarget = () => {     // a hash inside a part (a step, a section): show its part, then scroll to it
      showPart(fromHash(), false);
      const el = location.hash && document.getElementById(location.hash.slice(1));
      if (el && !el.classList.contains("term-partbox")) el.scrollIntoView({block: "start"});
    };
    window.addEventListener("hashchange", toTarget);
    parts.classList.add("js");
    toTarget();
  }

  // --- Racinglines 101 scoring part: the penalty slider, and the Python boxes. Each <div data-py> shows its code and the output
  //     it printed when the page was written; Run loads Pyodide from the CDN (first Run only) and runs the edited code in the
  //     browser (no JS: code and output still show) ---
  document.querySelectorAll("[data-penalty]").forEach(box => {
    const r = box.querySelector("input"), o = box.querySelector("output");
    const svg = box.closest("section").querySelector("[data-pen-chart]");
    const [x0, x1, y0, y1, ymax] = svg ? svg.dataset.penChart.split(" ").map(Number) : [];
    const show = () => {
      const p = r.value / 100, brier = (1 - p) ** 2, logloss = -Math.log(p);
      o.textContent = `${r.value}% → Brier ${brier.toFixed(2)}, log loss ${logloss.toFixed(2)}`;
      if (!svg) return;
      // the chart's guide line and dots follow the slider, and the legend shows both values
      const x = x0 + p * (x1 - x0), y = v => y0 - v / ymax * (y0 - y1);
      svg.querySelectorAll("[data-pen]").forEach(el => {
        const k = el.dataset.pen, v = k === "ll" ? logloss : brier;
        if (k === "at") { el.setAttribute("x1", x); el.setAttribute("x2", x); }
        else if (el.tagName === "circle") { el.setAttribute("cx", x); el.setAttribute("cy", y(v)); }
        else el.textContent = `${k === "ll" ? "log loss: −ln(p)" : "Brier: (1 − p)²"} = ${v.toFixed(2)}`;
      });
    };
    r.addEventListener("input", show);
    show();
  });
  const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
  let pyReady = null;
  const pyodide = () => pyReady ||= new Promise((ok, fail) => {
    const s = document.createElement("script");
    s.src = PYODIDE + "pyodide.js";
    s.onload = () => loadPyodide({ indexURL: PYODIDE }).then(ok, fail);
    s.onerror = () => fail(new Error("Couldn't load Python from " + PYODIDE));
    document.head.appendChild(s);
  }).catch(e => { pyReady = null; throw e; });
  document.querySelectorAll("[data-py]").forEach(box => {
    const code = box.querySelector("[data-py-code]"), out = box.querySelector("[data-py-out]"), btn = box.querySelector("[data-py-run]");
    btn.hidden = false;
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      out.textContent = pyReady ? "Running…" : "Loading Python in your browser (first run only)…";
      try {
        const py = await pyodide();
        await py.loadPackagesFromImports(code.value);
        let text = "";
        py.setStdout({ batched: s => { text += s + "\n"; } });
        py.setStderr({ batched: s => { text += s + "\n"; } });
        await py.runPythonAsync(code.value);
        out.textContent = text || "(no output)";
      } catch (e) {
        out.textContent = String(e.message || e).trim();
      }
      btn.disabled = false;
    });
  });

  // --- /strategy/pick: each slider shows its value (points for probabilities, $ or shares otherwise); the dice roll
  // sliders at random, up to `spread` of each slider's range away from the strategy's own setting (no JS: no dice) ---
  const pick = document.querySelector("form.pick");
  if (pick) {
    const show = r => {
      const out = pick.querySelector(`[data-out="${r.name}"]`), v = Number(r.value), u = r.dataset.unit;
      out.textContent = u === "pts" ? `${Math.round(v * 100)} pts` : u === "$" ? `$${v}` : `${v} ${u}`;
      const moved = Math.abs(Number(r.dataset.base) - v) > 1e-9;
      const note = r.parentElement.querySelector("[data-untested]");     // the alpha drawer: "untested" instead
      if (note) note.hidden = !moved;
      else if (moved) out.textContent += " (adjusted)";
    };
    const spread = pick.querySelector("[data-spread]");
    const roll = opt => {
      const k = spread ? Number(spread.value) : 1;
      opt.querySelectorAll("input[type=range]").forEach(r => {
        const lo = Number(r.min), hi = Number(r.max), step = Number(r.step), base = Number(r.dataset.base);
        const v = base + (Math.random() * 2 - 1) * k * (hi - lo);
        r.value = Math.min(hi, Math.max(lo, lo + Math.round((v - lo) / step) * step));
        show(r);
      });
      const pre = opt.querySelector("input[type=checkbox]");
      if (pre && Math.random() < k / 2) pre.checked = !pre.checked;
      pick.querySelector("[data-rolled]").value = "1";
    };
    pick.querySelectorAll("input[type=range]:not([data-spread]):not([data-kelly])").forEach(r => { r.addEventListener("input", () => show(r)); show(r); });
    pick.querySelectorAll("[data-roll]").forEach(btn => { btn.hidden = false; btn.addEventListener("click", e => { e.preventDefault(); roll(btn.closest(".pick-opt, [data-tweak]")); }); });
    // the alpha's Kelly slider: its percent and its tier ("nitty to degen", strategy_pick.KELLY_LABELS), quick picks
    const kbox = pick.querySelector("[data-kelly-box]");
    if (kbox) {
      const k = kbox.querySelector("[data-kelly]"), tiers = JSON.parse(kbox.dataset.labels);
      const showK = () => {
        const v = Number(k.value);
        kbox.querySelector("[data-kelly-out]").textContent = `${Math.round(v * 100)}%`;
        kbox.querySelector("[data-kelly-label]").textContent = (tiers.find(t => v <= t[0] + 1e-9) || tiers[tiers.length - 1])[1];
      };
      k.addEventListener("input", showK);
      kbox.querySelectorAll("[data-kelly-pick]").forEach(b => b.addEventListener("click", () => {
        k.value = b.dataset.kellyPick; showK();
        const kr = pick.querySelector("input[name=sizing][value=kelly]"); if (kr) kr.checked = true;
      }));
      showK();
    }
    const any = pick.querySelector("[data-roll-any]");
    if (any) {
      pick.querySelector("[data-dice]").hidden = false;
      const so = pick.querySelector("[data-spread-out]");
      spread.addEventListener("input", () => { so.textContent = `${Math.round(spread.value * 100)}%`; });
      any.addEventListener("click", () => {     // a strategy with settings to roll, then its settings
        const opts = [...pick.querySelectorAll(".pick-opt")].filter(o => o.querySelector("[data-roll]"));
        const opt = opts[Math.floor(Math.random() * opts.length)];
        opt.querySelector("input[type=radio]").checked = true;
        roll(opt);
        opt.scrollIntoView({block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"});
      });
    }
  }

  // --- Lab: sections open on demand (HTMX loads them on their "open" event), remembered per browser ---
  const lab = document.getElementById("lab");
  if (lab) {
    const open = new Set(store.get("lab.open", []));
    (JSON.parse(lab.dataset.openNow || "[]")).forEach(k => open.add(k));
    function show(key, on, scroll) {
      const box = document.getElementById(key), btn = document.querySelector(`[data-section="${key}"]`);
      if (!box || !btn) return;
      box.hidden = !on; btn.setAttribute("aria-pressed", on);
      on ? open.add(key) : open.delete(key); store.set("lab.open", [...open]);
      if (on) { htmx.trigger(box, "open"); if (scroll) box.scrollIntoView({block: "start"}); }
    }
    document.querySelectorAll("[data-section]").forEach(b =>
      b.addEventListener("click", () => show(b.dataset.section, b.getAttribute("aria-pressed") !== "true")));
    document.addEventListener("click", e => {
      const a = e.target.closest("a[data-open]");
      if (a) { e.preventDefault(); show(a.dataset.open, true, true); }
    });
    // the jobs section: whose jobs (mine / everyone's) is a browser setting, sent as a query parameter
    const jobs = document.getElementById("jobs");
    if (jobs) {
      jobs.addEventListener("htmx:configRequest", e => { e.detail.parameters.scope = store.get("lab.jobsScope", "mine"); });
      document.addEventListener("change", e => {
        if (e.target.matches("[name=jobs-scope]")) { store.set("lab.jobsScope", e.target.value); htmx.trigger(jobs, "open"); }
      });
    }
    // Edge Finder buttons reflect the saved combos after every swap
    function mark() {
      const root = document.getElementById("edge-root");
      let cs = [];
      try { cs = JSON.parse(root.dataset.combos); } catch (e) {}
      const has = (v, s) => cs.some(c => c[0] === v && c[1] === s);
      document.querySelectorAll("[data-edge=toggle]").forEach(b => b.setAttribute("aria-pressed", has(b.dataset.variant, b.dataset.strategy)));
      document.querySelectorAll("[data-edge-model]").forEach(b => {
        const inEF = cs.some(c => c[0] === b.dataset.variant);
        b.setAttribute("aria-pressed", inEF);
        b.dataset.edge = inEF ? "remove_model" : "add_model";
        b.setAttribute("hx-vals", JSON.stringify({action: b.dataset.edge, variant: b.dataset.variant}));
        b.textContent = inEF ? "✓ in Edge Finder" : "+ Edge Finder";
      });
    }
    document.addEventListener("htmx:afterSwap", e => {
      mark();
      const sel = e.detail.target.querySelector && e.detail.target.querySelector("[data-start-from]");
      if (sel && sel.value) sel.dispatchEvent(new Event("change", {bubbles: true}));
      if (e.detail.target.id === "edge" && e.detail.requestConfig.path === "/lab/candidate") {
        const run = document.getElementById("run");
        if (run && !run.hidden) htmx.trigger(run, "open");             // the sweep form lists candidates
      }
    });
    // candidates: naming one takes two prompts, so it stays a click handler that fires the HTMX request
    document.addEventListener("click", e => {
      const b = e.target.closest("[data-cand=add]");
      if (!b) return;
      const name = prompt("Name this candidate (it can be loaded into the sweep form):", "");
      if (name === null) return;
      const why = prompt("Why is it worth exploring? (optional)", "") || "";
      htmx.ajax("POST", "/lab/candidate", {target: "#edge", swap: "innerHTML",
        values: {action: "add", name, why, variant: b.dataset.variant, strategy: b.dataset.strategy, csrf_token: csrf()}});
    });
    // the sweep form: start from a candidate or the defaults; highlight what differs from the starting point
    const fields = (form, name) => [...form.querySelectorAll(`[name="${name}"]`)];
    function readSettings(form) {
      const out = {};
      form.querySelectorAll("[data-setting]").forEach(l => {
        const name = l.dataset.setting, els = fields(form, name).filter(x => x.type !== "hidden");
        if (els.length && els[0].type === "checkbox" && els.length === 1) out[name] = els[0].checked;
        else if (els.length && els[0].type === "checkbox") out[name] = els.filter(x => x.checked).map(x => x.value);
        else if (els.length) out[name] = els[0].type !== "number" ? els[0].value : els[0].value === "" ? null : Number(els[0].value);
      });
      return out;
    }
    function fillSettings(form, st) {
      Object.entries(st).forEach(([name, v]) => {
        const els = fields(form, name).filter(x => x.type !== "hidden");
        if (!els.length) return;
        if (els[0].type === "checkbox" && els.length === 1) els[0].checked = !!v;
        else if (els[0].type === "checkbox") els.forEach(x => { x.checked = (v || []).includes(x.value); });
        else els[0].value = v === null ? "" : v;
      });
    }
    // a stored value and the field's value are the same setting: "not set" (null) reads as an empty field, an empty
    // checkbox group, or a select's first option (the setting's fallback, e.g. the default exchange)
    function same(form, k, v, stored) {
      if (stored === undefined) return true;
      if (stored === null) {
        const sel = form.querySelector(`select[name="${k}"]`);
        return v === null || v === "" || (Array.isArray(v) && !v.length) || (sel && sel.options.length && v === sel.options[0].value);
      }
      return JSON.stringify(v) === JSON.stringify(stored) || (typeof v === "number" && Number(stored) === v);
    }
    function markChanged(form) {
      const src = JSON.parse(form.dataset.source || "{}"), now = readSettings(form);
      let n = 0;
      form.querySelectorAll("[data-setting]").forEach(l => {
        const k = l.dataset.setting, changed = !same(form, k, now[k], src[k]);
        l.classList.toggle("changed", changed); if (changed) n++;
      });
      const c = form.querySelector("[data-changed-count]");
      if (c) c.textContent = n ? `${n} setting${n > 1 ? "s" : ""} changed from the starting point` : "";
      // anything off the defaults makes the run a scenario (not calibrated)
      const defs = JSON.parse(form.dataset.defaults || "{}"), note = form.querySelector("[data-scenario-note]");
      if (note) note.hidden = Object.keys(defs).every(k => !(k in now) || same(form, k, now[k], defs[k]));
    }
    // the model variant: the switch checkboxes and the text field mirror each other; ticked switches join with "+" in
    // the checkboxes' order, none ticked is "baseline"
    function syncSwitches(form) {
      const t = form.querySelector("[data-variant-text]");
      if (!t) return;
      const on = t.value.split("+").map(x => x.trim());
      form.querySelectorAll("[data-switch]").forEach(b => { b.checked = on.includes(b.dataset.switch); });
    }
    function switchesToText(form, changed) {
      if (changed && changed.checked && changed.dataset.exclusive)
        form.querySelectorAll("[data-switch]").forEach(b => {
          if (b !== changed && changed.dataset.exclusive.split(" ").includes(b.dataset.switch)) b.checked = false;
        });
      const on = [...form.querySelectorAll("[data-switch]")].filter(b => b.checked).map(b => b.dataset.switch);
      form.querySelector("[data-variant-text]").value = on.length ? on.join("+") : "baseline";
    }
    document.querySelectorAll("#sweep-form").forEach(syncSwitches);
    document.addEventListener("htmx:afterSwap", e => {
      const f = e.detail.target.querySelector && e.detail.target.querySelector("#sweep-form");
      if (f) syncSwitches(f);
    });
    document.addEventListener("change", e => {
      const sel = e.target.closest("[data-start-from]");
      if (sel) {
        const form = sel.form, st = JSON.parse(sel.selectedOptions[0].dataset.settings || "{}");
        form.dataset.source = JSON.stringify(Object.assign(JSON.parse(form.dataset.defaults), st));
        fillSettings(form, JSON.parse(form.dataset.source));
        syncSwitches(form);
      }
      const form = e.target.closest("#sweep-form");
      if (form && e.target.matches("[data-switch]")) switchesToText(form, e.target);
      if (form) markChanged(form);
    });
    document.addEventListener("input", e => {
      const form = e.target.closest("#sweep-form");
      if (form && e.target.matches("[data-variant-text]")) syncSwitches(form);
      if (form) markChanged(form);
    });
    const hashKey = location.hash.slice(1);
    if (document.querySelector(`[data-section="${hashKey}"]`)) open.add(hashKey);
    [...open].forEach(k => show(k, true, hashKey === k));
    mark();
  }

  // --- Live replay: the slider picks a saved snapshot (HTMX fetches it); play steps through them ---
  const rp = document.getElementById("replay");
  if (rp) {
    const T = JSON.parse(rp.dataset.times), sl = document.getElementById("rp-slider"), lab = document.getElementById("rp-time"),
          play = document.getElementById("rp-play"), link = document.getElementById("rp-link"), paced = rp.dataset.pace === "real", base = rp.dataset.url;
    const at = s => Date.UTC(+s.slice(0, 4), +s.slice(4, 6) - 1, +s.slice(6, 8), +s.slice(9, 11), +s.slice(11, 13), +s.slice(13, 15));
    let speed = +rp.dataset.speed || 30, timer = null, lastFetchMs = 0;
    function label() { const s = T[+sl.value]; lab.textContent = s.slice(9, 11) + ":" + s.slice(11, 13) + ":" + s.slice(13, 15) + " UTC · " + (+sl.value + 1) + " / " + T.length; }
    function show(i) {
      sl.value = Math.max(0, Math.min(T.length - 1, i)); label();
      const t = T[+sl.value];
      const u = new URL(base, location.href); u.searchParams.set("t", t);
      const page = u.pathname + u.search; u.searchParams.set("partial", "1");
      history.replaceState(null, "", page); if (link) link.href = page;
      return htmx.ajax("GET", u.pathname + u.search, {target: "#live", swap: "innerHTML"});
    }
    function stop() { clearTimeout(timer); timer = null; play.setAttribute("aria-pressed", "false"); play.textContent = "▶ Play"; }
    function tick() {
      const i = +sl.value;
      if (i >= T.length - 1) return stop();
      const target = paced ? Math.max(150, (at(T[i + 1]) - at(T[i])) / speed) : 60000 / speed;   // updates hours apart: a fixed pace
      // each fetch (network + server render) takes its own time on top of the wait; subtract the last one so
      // higher speeds keep pacing correctly instead of it just adding on top and swallowing the difference
      const wait = Math.max(0, target - lastFetchMs);
      timer = setTimeout(() => {
        const t0 = performance.now();
        show(i + 1).then(() => { lastFetchMs = performance.now() - t0; if (timer) tick(); });
      }, wait);
    }
    play.addEventListener("click", () => {
      if (timer) return stop();
      if (+sl.value >= T.length - 1) sl.value = 0;
      play.setAttribute("aria-pressed", "true"); play.textContent = "❚❚ Pause"; timer = 1; tick();
    });
    rp.querySelectorAll("[data-step]").forEach(b => b.addEventListener("click", () => { stop(); show(+sl.value + +b.dataset.step); }));
    rp.querySelectorAll("[data-speed]").forEach(b => b.addEventListener("click", () => {
      speed = +b.dataset.speed;
      rp.querySelectorAll("[data-speed]").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    }));
    sl.addEventListener("input", label);
    sl.addEventListener("change", () => { stop(); show(+sl.value); });
    label();
  }
  });
})();
