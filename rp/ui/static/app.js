const statusPill = document.getElementById("statusPill");
const overviewBody = document.getElementById("overviewBody");
const stBox = document.getElementById("stBox");
const logEl = document.getElementById("log");
const progressFill = document.getElementById("progressFill");
const metricsEl = document.getElementById("metrics");
const summaryBody = document.getElementById("summaryBody");
const sleepHint = document.getElementById("sleepHint");

const trainNarrBtn = document.getElementById("trainNarrBtn");
const evalNarrBtn = document.getElementById("evalNarrBtn");
const sleepBtn = document.getElementById("sleepBtn");
const stopJobBtn = document.getElementById("stopJobBtn");
const serverStartBtn = document.getElementById("serverStartBtn");
const serverStopBtn = document.getElementById("serverStopBtn");

let es = null;
let currentJob = null;

function pill(text, cls) {
  statusPill.textContent = text;
  statusPill.className = "status-pill " + (cls || "");
}

function fmtSignals(store) {
  const sig = store.signals || {};
  const parts = Object.entries(sig).map(([k, v]) => `${k}: ${v}`);
  return parts.length ? parts.join(", ") : "none yet";
}

function renderOverview(data) {
  const narr = data.narration_meta;
  const narrLine = narr
    ? `Last narration train: ${narr.train_minutes} min, mean NLL ${narr.train_nll_mean}`
    : "Bundled narration weights in repo (or train to refresh).";
  overviewBody.innerHTML = `
    <p><strong>Active adapter:</strong> ${escapeHtml(data.adapter_label)}</p>
    <p><strong>Data dir:</strong> <code>${escapeHtml(data.data_dir)}</code></p>
    <p><strong>Narration dataset:</strong> ${data.narration_train_examples} examples in train.jsonl</p>
    <p>${escapeHtml(narrLine)}</p>
    <p><strong>Chat collected:</strong> ${data.store.generations} generations · signals (${fmtSignals(data.store)}) · ${data.store.new_signals_since_sleep} new since last sleep</p>
  `;
  sleepHint.textContent = data.sleep.message;
  sleepBtn.disabled = !data.sleep.ready || (currentJob && ["running", "queued"].includes(currentJob.state));
  const srv = data.server;
  stBox.innerHTML = srv.running
    ? `<p class="ok">Server running · model <code>${escapeHtml(srv.model_id)}</code></p>
       <p>SillyTavern endpoint: <code class="copy">${escapeHtml(srv.endpoint)}</code></p>`
    : `<p class="muted">Server stopped. Start it to connect SillyTavern.</p>
       <p>Will use endpoint <code>${escapeHtml(srv.endpoint)}</code> · model id <code>${escapeHtml(srv.model_id)}</code></p>`;
  serverStartBtn.disabled = srv.running;
  serverStopBtn.disabled = !srv.running;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

async function refreshOverview() {
  const r = await fetch("/api/overview");
  const data = await r.json();
  renderOverview(data);
  if (data.job) applySnapshot(data.job);
}

function applySnapshot(snap) {
  currentJob = snap;
  const pct = Math.round((snap.progress || 0) * 100);
  progressFill.style.width = pct + "%";
  if (snap.state === "running") {
    pill(`${snap.kind}… ${pct}%`, "running");
    metricsEl.textContent = `Job ${snap.kind} (${snap.id}) — ${pct}%`;
    stopJobBtn.disabled = false;
    trainNarrBtn.disabled = true;
    evalNarrBtn.disabled = true;
    sleepBtn.disabled = true;
  } else if (snap.state === "done") {
    pill("Done", "done");
    metricsEl.textContent = `Finished ${snap.kind}`;
    stopJobBtn.disabled = true;
    trainNarrBtn.disabled = false;
    evalNarrBtn.disabled = false;
    showSummary(snap);
  } else if (snap.state === "error" || snap.state === "stopped") {
    pill(snap.state, snap.state);
    metricsEl.textContent = snap.error || snap.state;
    stopJobBtn.disabled = true;
    trainNarrBtn.disabled = false;
    evalNarrBtn.disabled = false;
  }
  if (snap.log_tail) logEl.textContent = snap.log_tail.join("\n");
}

function showSummary(snap) {
  const r = snap.result;
  if (!r) {
    summaryBody.textContent = "Job finished (see log).";
    return;
  }
  if (snap.kind === "narration_train" && r.status === "ok") {
    summaryBody.innerHTML = `<p>Narration adapter saved to <code>${escapeHtml(r.out || "")}</code></p>
      <p>Train mean NLL: <strong>${r.train_nll_mean}</strong> · ${r.train_minutes} min · ${r.examples} examples</p>`;
  } else if (snap.kind === "sleep") {
    if (r.status === "skipped") {
      summaryBody.innerHTML = `<p class="warn">${escapeHtml(r.reason || "skipped")}</p>`;
    } else {
      summaryBody.innerHTML = `<p>Sleep <strong>${escapeHtml(r.version || "")}</strong> · activated: ${r.activated ? "yes" : "no"}</p>
        <p>Chosen NLL: ${r.train_nll_chosen_before} → ${r.train_nll_chosen_after}</p>
        <p>Pair accuracy: ${r.pair_pref_acc_after} · canary drift ${r.canary_drift} (guard ${r.guard_passed ? "ok" : "FAILED"})</p>`;
    }
  } else if (snap.kind === "narration_eval" && r.eval_md) {
    summaryBody.innerHTML = "<pre class='summary-md'>" + escapeHtml(r.eval_md.split("\n").slice(0, 12).join("\n")) + "</pre>";
  } else {
    summaryBody.innerHTML = "<pre>" + escapeHtml(JSON.stringify(r, null, 2)) + "</pre>";
  }
  refreshOverview();
}

function connectEvents() {
  if (es) es.close();
  es = new EventSource("/api/events");
  es.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "snapshot") applySnapshot(msg.data);
    if (msg.type === "log") {
      logEl.textContent += (logEl.textContent ? "\n" : "") + msg.line;
      logEl.scrollTop = logEl.scrollHeight;
      if (msg.progress != null) progressFill.style.width = Math.round(msg.progress * 100) + "%";
      if (msg.result) showSummary({ kind: currentJob?.kind, result: msg.result, state: "running" });
    }
    if (msg.type === "state" && msg.snapshot) applySnapshot(msg.snapshot);
    if (msg.type === "state" && msg.state === "stopped") refreshOverview();
  };
}

async function post(url) {
  const r = await fetch(url, { method: "POST" });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

trainNarrBtn.addEventListener("click", async () => {
  logEl.textContent = "";
  summaryBody.textContent = "Training…";
  try {
    const snap = await post("/api/narration/train");
    applySnapshot(snap);
    connectEvents();
  } catch (e) {
    alert(e.message);
  }
});

evalNarrBtn.addEventListener("click", async () => {
  logEl.textContent = "";
  summaryBody.textContent = "Running held-out eval (base vs narration)…";
  try {
    const snap = await post("/api/narration/eval");
    applySnapshot(snap);
    connectEvents();
  } catch (e) {
    alert(e.message);
  }
});

sleepBtn.addEventListener("click", async () => {
  logEl.textContent = "";
  try {
    const snap = await post("/api/sleep");
    applySnapshot(snap);
    connectEvents();
  } catch (e) {
    alert(e.message);
  }
});

stopJobBtn.addEventListener("click", async () => {
  await post("/api/job/stop");
});

serverStartBtn.addEventListener("click", async () => {
  await post("/api/server/start");
  await refreshOverview();
});

serverStopBtn.addEventListener("click", async () => {
  await post("/api/server/stop");
  await refreshOverview();
});

connectEvents();
refreshOverview();
setInterval(refreshOverview, 8000);
