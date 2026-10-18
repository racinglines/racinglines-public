// Live timing on the Live page (templates/live.html #f1-stream, web/f1_live.py): off until the viewer turns it on.
(function () {
  const box = document.getElementById("f1-stream");
  if (!box) return;
  const btn = document.getElementById("fs-toggle"), state = document.getElementById("fs-state");
  const logEl = document.getElementById("fs-log"), table = document.getElementById("fs-table");
  let ws = null;
  const now = () => new Date().toISOString().slice(11, 19);
  function log(msg, level, ts) {
    if (!logEl) return;                                 // the debug log is on the page for admins only
    logEl.textContent += `${ts || now()} ${level === "error" ? "ERROR " : ""}${msg}\n`;
    logEl.scrollTop = logEl.scrollHeight;
  }
  function setState(on, text) {
    btn.textContent = on ? "Turn off" : "Turn on";
    btn.setAttribute("aria-pressed", on ? "true" : "false");
    state.textContent = text;
  }
  const t = (s) => (s == null ? "–" : `${Math.floor(s / 60)}:${(s % 60).toFixed(3).padStart(6, "0")}`);
  const esc = (x) => String(x == null ? "" : x).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  function render(d) {
    if (!d.session) { table.innerHTML = `<p class="muted small">${esc(d.event)}: ${esc(d.note)}</p>`; return; }
    const rows = d.drivers.map((r) => `<tr><td>${r.pos ?? ""}</td><td class="l">${esc(r.code)}</td><td class="l">${esc(r.name)}</td>` +
      `<td class="l">${esc(r.team)}</td><td class="num">${t(r.best)}</td><td class="num">${t(r.last)}</td><td class="num">${r.laps}</td></tr>`).join("");
    table.innerHTML = `<p class="small"><b>${esc(d.event)} · ${esc(d.session)}</b> (started ${esc(d.start).slice(0, 16).replace("T", " ")} UTC)` +
      ` · ${d.laps} laps · updated ${now()} UTC</p>` + (rows ? `<div class="scroll"><table><thead><tr><th>#</th><th class="l"></th><th class="l">Driver</th>` +
      `<th class="l">Team</th><th class="num">Best</th><th class="num">Last</th><th class="num">Laps</th></tr></thead><tbody>${rows}</tbody></table></div>` : "");
  }
  function start() {
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/f1/live`;
    log(`opening ${url} for ${box.dataset.event}`);
    setState(true, "connecting");
    ws = new WebSocket(url);
    ws.onopen = () => { setState(true, "on"); ws.send(JSON.stringify({type: "start", event: box.dataset.event})); };
    ws.onmessage = (e) => {
      let m;
      try { m = JSON.parse(e.data); } catch (err) { log(`unreadable message: ${e.data.slice(0, 200)}`, "error"); return; }
      if (m.type === "log") log(m.msg, m.level, m.ts);
      else if (m.type === "timing") render(m.data);
      else log(`unknown message type ${m.type}`, "error");
    };
    ws.onerror = () => log("websocket error (the server's reason, if any, is in the close below)", "error");
    ws.onclose = (e) => { log(`closed (code ${e.code}${e.reason ? ", " + e.reason : ""})`); ws = null; setState(false, "off"); };
  }
  btn.addEventListener("click", () => {
    if (ws) {
      const w = ws;
      try { w.send(JSON.stringify({type: "stop"})); } catch (err) { /* already closing */ }
      setTimeout(() => w.close(), 1000);
    }
    else start();
  });
  const copyBtn = document.getElementById("fs-copy");
  if (copyBtn) copyBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(logEl.textContent).then(() => log("log copied"), () => log("couldn't copy", "error"));
  });
  log(`ready: press Turn on (event ${box.dataset.event})`);
})();
