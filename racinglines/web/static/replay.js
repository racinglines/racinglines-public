/* The maker replay in the browser: the diagnostics page's "Live replay" panel. It downloads the event's tape once
   (/lab/diagnostics/<run>/tape.json, gzipped) and the replay compiled to WebAssembly (/static/replay.wasm, built
   from wasm/ by wasm/build.sh), then re-runs the whole event's replay on this computer every time a knob moves:
   the server only serves files. The Rust replay is checked against the Python one (tests/test_wasm_replay.py), so
   the numbers equal the server replay's for the same knobs.

   Markup: <div data-wasm-replay="<tape url>" data-wasm="<.wasm url>"> holding [data-knob="<Params field>"] inputs, a [data-start] button,
   [data-status] and [data-out] elements. Interface of the .wasm (no wasm-bindgen): alloc(len) -> ptr,
   dealloc(ptr, len), load(ptr, len) -> 0 | 1, run(ptr, len) -> ptr of the result JSON, result_len(). */
(function () {
  "use strict";
  const enc = new TextEncoder(), dec = new TextDecoder();

  async function kernel(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`replay.wasm: ${res.status}`);
    const { instance } = await WebAssembly.instantiate(await res.arrayBuffer(), {});
    const x = instance.exports;
    function call(fn, text) {                       // pass a string in linear memory, call, free it
      const b = enc.encode(text), p = x.alloc(b.length);
      new Uint8Array(x.memory.buffer, p, b.length).set(b);
      try { return fn(p, b.length); } finally { x.dealloc(p, b.length); }
    }
    return {
      load: text => { if (call(x.load, text) !== 0) throw new Error("the tape didn't parse"); },
      run(params) {
        const p = call(x.run, JSON.stringify(params));
        return JSON.parse(dec.decode(new Uint8Array(x.memory.buffer, p, x.result_len())));
      },
    };
  }

  const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "" :
    (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  const plain = v => "$" + num(v);                      // amounts that have no sign (notional)
  const money = v => (v < 0 ? "−$" : v > 0 ? "+$" : "$") + num(Math.abs(v));
  const cls = v => v > 0 ? "good" : v < 0 ? "bad" : "";
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const utc = ns => new Date(ns / 1e6).toISOString().slice(0, 16).replace("T", " ");

  function table(cols, rows) {
    const head = cols.map(([k, label]) => `<th class="${k === "kind" || k === "reason" ? "l" : ""}">${label}</th>`).join("");
    const body = rows.map(r => "<tr>" + cols.map(([k, , f]) => {
      const v = r[k], text = f ? f(v) : esc(v);
      return `<td class="${typeof v === "string" ? "l" : ""} ${f === money ? cls(v) : ""}">${text}</td>`;
    }).join("") + "</tr>").join("");
    return `<div class="scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  function render(out, r, ms) {
    if (r.error) { out.innerHTML = `<p class="flash bad">${esc(r.error)}</p>`; return; }
    const t = r.summary[r.summary.length - 1];
    const pct = v => num(v * 100, 1) + "%", int = v => num(v, 0), one = v => num(v, 1);
    out.innerHTML =
      `<p><b>${int(t.fills)}</b> fills · notional ${plain(t.notional)} · spread ` +
      `<b class="${cls(t.spread_pnl)}">${money(t.spread_pnl)}</b> · markout 60m <b class="${cls(t.markout_60m)}">${money(t.markout_60m)}</b>` +
      ` · P&amp;L held to the result <b class="${cls(t.pnl)}">${money(t.pnl)}</b> · worst case ${money(t.worst_case)}` +
      ` <span class="muted">(${int(r.quotes)} quote decisions, ${num(ms, 0)} ms in this browser)</span></p>` +
      "<h3>By market type</h3>" + table([["kind", "kind"], ["markets", "markets", int], ["quoted_share", "quoted", pct],
        ["fills", "fills", int], ["shares", "shares", one], ["notional", "notional", plain], ["spread_pnl", "spread_pnl", money],
        ["markout_60m", "markout_60m", money], ["model_edge", "model_edge", money], ["pnl", "pnl", money],
        ["worst_case", "worst_case", money]], r.summary) +
      '<div class="grid"><div><h3>Pricing stages</h3>' + table([["run_id", "run", v => String(v)], ["start", "start (UTC)", utc],
        ["fills", "fills", int], ["spread_pnl", "spread_pnl", money], ["markout_60m", "markout_60m", money],
        ["pnl", "pnl", money]], r.stages) +
      "</div><div><h3>Why markets weren't quoted</h3>" +
      table([["reason", "reason"], ["n", "market steps", int]], r.skips.map(([reason, n]) => ({ reason, n }))) + "</div></div>";
  }

  function knobs(root, base) {
    const p = Object.assign({}, base);
    root.querySelectorAll("[data-knob]").forEach(el => {
      const k = el.dataset.knob, v = el.type === "checkbox" ? el.checked : el.value;
      if (k === "fill") p.fill = v;
      else if (k === "max_disagree") p.max_disagree = Number(v) > 0 ? Number(v) : null;      // 0 = off
      else p[k] = Number(v);
      const shown = root.querySelector(`[data-show="${k}"]`);
      if (shown) shown.textContent = k === "max_disagree" && !(Number(v) > 0) ? "off" : v;
    });
    return p;
  }

  async function start(root) {
    const status = root.querySelector("[data-status]"), out = root.querySelector("[data-out]");
    status.textContent = "Downloading the tape and the replay…";
    try {
      const [k, tape] = await Promise.all([kernel(root.dataset.wasm), fetch(root.dataset.wasmReplay).then(r => {
        if (!r.ok) throw new Error(`tape: ${r.status}`); return r.text(); })]);
      const base = JSON.parse(tape).params;
      k.load(tape);
      let queued = false;
      const go = () => {
        if (queued) return;
        queued = true;
        requestAnimationFrame(() => {
          queued = false;
          const t0 = performance.now(), r = k.run(knobs(root, base));
          render(out, r, performance.now() - t0);
        });
      };
      root.querySelectorAll("[data-knob]").forEach(el => el.addEventListener("input", go));
      root.querySelector("[data-controls]").hidden = false;
      root.querySelector("[data-start]").hidden = true;
      status.textContent = "";
      go();
    } catch (e) {
      status.textContent = `The live replay couldn't start: ${e.message}. The server replay below still works.`;
    }
  }

  document.addEventListener("click", e => {
    const b = e.target.closest("[data-wasm-replay] [data-start]");
    if (b) start(b.closest("[data-wasm-replay]"));
  });
})();
