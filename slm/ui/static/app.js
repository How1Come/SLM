(() => {
  const statusPill = document.getElementById("statusPill");
  const progressFill = document.getElementById("progressFill");
  const metrics = document.getElementById("metrics");
  const logEl = document.getElementById("log");
  const resultsBody = document.getElementById("resultsBody");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const chatBtn = document.getElementById("chatBtn");
  const chatForm = document.getElementById("chatForm");
  const chatOut = document.getElementById("chatOut");
  const budgetEl = document.getElementById("budget");
  const seedEl = document.getElementById("seed");

  let selectedPreset = null;
  let checkpoint = null;
  let es = null;

  document.querySelectorAll(".preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".preset").forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
      selectedPreset = btn.dataset.preset;
      startBtn.disabled = false;
    });
  });

  function setStatus(state) {
    statusPill.textContent = state.replace("_", " ");
    statusPill.className = "status-pill " + state;
    const busy = state === "running" || state === "preparing" || state === "queued";
    startBtn.disabled = busy || !selectedPreset;
    stopBtn.disabled = !busy;
    chatBtn.disabled = !checkpoint;
  }

  function applySnapshot(s) {
    if (!s || s.state === "idle") {
      setStatus("idle");
      return;
    }
    setStatus(s.state);
    const pct = Math.round((s.progress || 0) * 100);
    progressFill.style.width = pct + "%";
    let m = `Run <strong>${s.name}</strong> — ${pct}%`;
    if (s.stories_ppl != null) {
      m += ` · stories ppl <strong>${s.stories_ppl}</strong>`;
    }
    if (s.childes_ppl != null) {
      m += ` · childes ppl <strong>${s.childes_ppl}</strong>`;
    }
    metrics.innerHTML = m;
    if (s.log_tail && s.log_tail.length) {
      logEl.textContent = s.log_tail.join("\n");
      logEl.scrollTop = logEl.scrollHeight;
    }
    if (s.checkpoint) {
      checkpoint = s.checkpoint;
      chatBtn.disabled = false;
    }
    if (s.state === "done") {
      renderResults(s);
    }
  }

  function renderResults(s) {
    let html = "";
    if (s.stories_ppl != null) {
      html += `<p>Held-out <strong>stories perplexity</strong>: ${s.stories_ppl}`;
      if (s.childes_ppl != null) {
        html += ` · <strong>CHILDES perplexity</strong>: ${s.childes_ppl}`;
      }
      html += "</p>";
    }
    if (s.sample) {
      html += `<div class="sample-block"><strong>Sample</strong><br>${escapeHtml(s.sample)}</div>`;
    }
    if (!html) html = '<p class="muted">Run finished. Check the log for details.</p>';
    resultsBody.innerHTML = html;
  }

  function escapeHtml(t) {
    return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function connectEvents() {
    if (es) es.close();
    es = new EventSource("/api/events");
    es.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "snapshot") {
        applySnapshot(msg.data);
      } else if (msg.type === "log") {
        logEl.textContent += (logEl.textContent ? "\n" : "") + msg.line;
        logEl.scrollTop = logEl.scrollHeight;
        if (msg.progress != null) {
          progressFill.style.width = Math.round(msg.progress * 100) + "%";
        }
      } else if (msg.type === "state") {
        if (msg.state) setStatus(msg.state);
        if (msg.snapshot) applySnapshot(msg.snapshot);
        if (msg.state === "done" || msg.state === "error" || msg.state === "stopped") {
          fetch("/api/status").then((r) => r.json()).then(applySnapshot);
        }
      }
    };
    es.onerror = () => {
      setTimeout(connectEvents, 2000);
    };
  }

  startBtn.addEventListener("click", async () => {
    if (!selectedPreset) return;
    logEl.textContent = "";
    resultsBody.innerHTML = '<p class="muted">Training…</p>';
    const body = { preset: selectedPreset };
    if (selectedPreset !== "smoke") {
      body.budget = budgetEl.value;
      body.seed = parseInt(seedEl.value, 10);
    } else {
      body.seed = parseInt(seedEl.value, 10);
    }
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || "Could not start");
      return;
    }
    applySnapshot(data);
  });

  stopBtn.addEventListener("click", async () => {
    await fetch("/api/stop", { method: "POST" });
  });

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const prompt = document.getElementById("prompt").value;
    chatOut.textContent = "Generating…";
    const res = await fetch("/api/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, checkpoint }),
    });
    const data = await res.json();
    if (!res.ok) {
      chatOut.textContent = data.error || "Failed";
      return;
    }
    chatOut.textContent = data.full;
  });

  fetch("/api/status")
    .then((r) => r.json())
    .then(applySnapshot)
    .then(() => {
      if (!new URLSearchParams(location.search).has("screenshot")) {
        connectEvents();
      }
    });
})();
