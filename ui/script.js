// CIFAR-10 Cycle – redesigned console
let chartInstances = {}, sessionUploadCount = 0;
const CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"];
const CLASS_COLORS = ["#34d399","#2dd4bf","#22d3ee","#60a5fa","#a78bfa","#f472b6","#fb7185","#fb923c","#fbbf24","#a3e635"];

function switchTab(tabName) {
  document.querySelectorAll(".panel").forEach(p => p.classList.toggle("active", p.id === "panel-" + tabName));
  document.querySelectorAll("[data-tab]").forEach(btn => {
    const on = btn.dataset.tab === tabName;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-selected", on ? "true" : "false");
    btn.classList.toggle("text-gray-400", !on);
  });
}
function formatUptime(seconds) {
  if (seconds == null || isNaN(seconds)) return "—";
  const s = Math.floor(seconds), h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = s%60;
  if (h > 0) return h + "h " + m + "m";
  if (m > 0) return m + "m " + sec + "s";
  return sec + "s";
}
function initDeploymentBadge() {
  const badge = document.getElementById("deployment-badge");
  if (!badge) return;
  const isLocal = ["localhost","127.0.0.1",""].includes(window.location.hostname);
  badge.textContent = isLocal ? "LOCAL" : "CLOUD";
  badge.className = isLocal
    ? "hidden sm:inline-flex items-center gap-1.5 text-[11px] font-mono px-2.5 py-1 rounded-full bg-ink-700 text-gray-400 border border-white/5"
    : "hidden sm:inline-flex items-center gap-1.5 text-[11px] font-mono px-2.5 py-1 rounded-full bg-mint-500/15 text-mint-400 border border-mint-500/30";
}
function wireDropZone(zoneId, onFile) {
  const zone = document.getElementById(zoneId);
  if (!zone) return;
  const input = zone.querySelector('input[type="file"]');
  zone.addEventListener("click", e => { if (e.target.tagName !== "BUTTON" && input) input.click(); });
  zone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); if (input) input.click(); }});
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", e => {
    e.preventDefault(); zone.classList.remove("dragover");
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) onFile(f);
  });
}
function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }

async function fetchHealth() {
  const statusBadge = document.getElementById("status-badge");
  const statusDot = document.getElementById("status-dot");
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("fail");
    const data = await res.json();
    const online = data.status === "healthy" && data.model_loaded;
    statusBadge.textContent = online ? "Online" : "Degraded";
    statusBadge.className = online ? "text-mint-400" : "text-amber-400";
    statusDot.className = "w-2 h-2 rounded-full animate-pulseDot " + (online ? "bg-mint-500" : "bg-amber-400");
    setText("metric-status", online ? "Healthy" : "Degraded");
    setText("metric-uptime", formatUptime(data.uptime_seconds));
    setText("metric-trained", data.model_last_trained ? new Date(data.model_last_trained).toLocaleString() : "—");
    setText("metric-params", data.model_parameters != null ? Number(data.model_parameters).toLocaleString() : (data.runtime || "ONNX"));
    setText("footer-uptime", "uptime " + formatUptime(data.uptime_seconds));
    setText("last-sync", new Date().toLocaleTimeString());
  } catch (e) {
    statusBadge.textContent = "Offline";
    statusBadge.className = "text-red-400";
    statusDot.className = "w-2 h-2 rounded-full bg-red-500 animate-pulseDot";
    setText("metric-status", "Offline");
    ["metric-uptime","metric-trained","metric-params"].forEach(id => setText(id, "—"));
  }
}

async function runPrediction(file, mode) {
  const formData = new FormData();
  formData.append("file", file);
  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");
    let conf = data.confidence != null ? data.confidence : 0;
    if (!conf && data.probabilities) {
      const vals = Array.isArray(data.probabilities) ? data.probabilities : Object.values(data.probabilities);
      conf = Math.max(...vals);
    }
    const confPct = conf <= 1 ? conf * 100 : conf;
    const cls = data.class_name || data.predicted_class || CLASS_NAMES[data.class_index] || "—";
    if (mode === "quick") {
      document.getElementById("quick-result").classList.remove("hidden");
      setText("quick-class", cls);
      setText("quick-conf", confPct.toFixed(1) + "%");
      document.getElementById("quick-conf-bar").style.width = Math.min(100, confPct) + "%";
    } else {
      document.getElementById("predict-empty").classList.add("hidden");
      document.getElementById("predict-result").classList.remove("hidden");
      setText("pred-class", cls);
      setText("pred-conf", confPct.toFixed(1) + "%");
      setText("pred-conf-label", confPct.toFixed(1) + "%");
      document.getElementById("pred-conf-bar").style.width = Math.min(100, confPct) + "%";
      setText("pred-latency", data.processing_time_ms != null ? data.processing_time_ms + " ms" : "—");
      renderProbChart(data);
    }
  } catch (err) { alert(err.message || String(err)); }
}

function renderProbChart(data) {
  const canvas = document.getElementById("prob-chart");
  if (!canvas || typeof Chart === "undefined") return;
  if (chartInstances.prob) chartInstances.prob.destroy();
  let labels = CLASS_NAMES, values = new Array(10).fill(0);
  if (data.probabilities) {
    if (Array.isArray(data.probabilities)) values = data.probabilities.map(v => v <= 1 ? v * 100 : v);
    else { labels = Object.keys(data.probabilities); values = Object.values(data.probabilities).map(v => v <= 1 ? v * 100 : v); }
  }
  chartInstances.prob = new Chart(canvas, {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: CLASS_COLORS.map(c => c + "cc"), borderRadius: 4, borderSkipped: false }] },
    options: {
      indexAxis: "y", plugins: { legend: { display: false } },
      scales: {
        x: { max: 100, grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#6b7280", font: { size: 10 } } },
        y: { grid: { display: false }, ticks: { color: "#9ca3af", font: { size: 10 } } },
      },
    },
  });
}

function clearPrediction() {
  document.getElementById("predict-empty").classList.remove("hidden");
  document.getElementById("predict-result").classList.add("hidden");
  if (chartInstances.prob) { chartInstances.prob.destroy(); delete chartInstances.prob; }
  const fi = document.getElementById("file-input");
  if (fi) fi.value = "";
}

async function handleUpload() {
  const files = document.getElementById("retrain-files").files;
  const label = document.getElementById("label-select").value;
  if (!files || !files.length) { setText("upload-status", "Select at least one image."); return; }
  const formData = new FormData();
  for (const f of files) formData.append("files", f);
  formData.append("label", label);
  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");
    sessionUploadCount += data.uploaded || 0;
    setText("session-count", sessionUploadCount);
    setText("upload-status", "Uploaded " + data.uploaded + " image(s) for " + (data.class_name || label) + ".");
    document.getElementById("step-1-dot").classList.add("done");
    document.getElementById("step-2-dot").classList.add("active");
  } catch (err) { setText("upload-status", err.message); }
}

async function handleZipUpload() {
  const file = document.getElementById("zip-file").files[0];
  if (!file) { setText("upload-status", "Select a ZIP file."); return; }
  const formData = new FormData();
  formData.append("files", file);
  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "ZIP upload failed");
    sessionUploadCount += data.uploaded || 0;
    setText("session-count", sessionUploadCount);
    setText("upload-status", "ZIP: " + data.uploaded + " image(s) across " + Object.keys(data.per_class || {}).length + " classes.");
    document.getElementById("step-1-dot").classList.add("done");
    document.getElementById("step-2-dot").classList.add("active");
  } catch (err) { setText("upload-status", err.message); }
}

async function triggerRetrain() {
  const btn = document.getElementById("retrain-btn");
  btn.disabled = true; btn.textContent = "Training…";
  document.getElementById("step-2-dot").classList.add("active");
  document.getElementById("step-3-dot").classList.add("active");
  try {
    const res = await fetch("/retrain", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Retrain failed");
    document.getElementById("step-3-dot").classList.add("done");
    document.getElementById("step-4-dot").classList.add("active");
    if (data.promoted) {
      document.getElementById("step-4-dot").classList.add("done");
      document.getElementById("step-5-dot").classList.add("done", "active");
    }
    document.getElementById("retrain-result").classList.remove("hidden");
    const line = document.getElementById("retrain-status-line");
    line.textContent = data.promoted ? "PROMOTED" : "REJECTED";
    line.className = "text-sm font-semibold " + (data.promoted ? "text-mint-400" : "text-amber-400");
    setText("rt-base", (data.baseline_accuracy * 100).toFixed(2) + "%");
    setText("rt-new", (data.new_accuracy * 100).toFixed(2) + "%");
    setText("rt-delta", (data.accuracy_change >= 0 ? "+" : "") + (data.accuracy_change * 100).toFixed(2) + " pp");
    setText("rt-samples", data.samples_used);
    setText("rt-reason", data.reason || "");
    fetchRetrainHistory();
    if (data.promoted) fetchVisualizations();
  } catch (err) { alert(err.message || String(err)); }
  finally { btn.disabled = false; btn.textContent = "Trigger retraining"; }
}

async function fetchRetrainHistory() {
  try {
    const res = await fetch("/retrain-history");
    if (!res.ok) return;
    const log = await res.json();
    const empty = document.getElementById("history-empty"), list = document.getElementById("history-list");
    if (!log || !log.length) { empty.classList.remove("hidden"); list.classList.add("hidden"); return; }
    empty.classList.add("hidden"); list.classList.remove("hidden");
    list.innerHTML = log.slice().reverse().map(e => {
      const ok = e.promoted, ts = e.timestamp ? new Date(e.timestamp).toLocaleString() : "—";
      return '<div class="timeline-item relative pl-10 pb-6">' +
        '<div class="absolute left-0 top-1 w-6 h-6 rounded-full flex items-center justify-center ' + (ok ? "bg-mint-500/20" : "bg-amber-500/20") + '">' +
        '<span class="w-2 h-2 rounded-full ' + (ok ? "bg-mint-500" : "bg-amber-400") + '"></span></div>' +
        '<div class="glass rounded-xl p-4"><div class="flex flex-wrap items-center gap-2 mb-2">' +
        '<span class="text-[11px] font-semibold px-2 py-0.5 rounded-full ' + (ok ? "bg-mint-500/15 text-mint-400" : "bg-amber-500/15 text-amber-400") + '">' +
        (e.status || (ok ? "promoted" : "rejected")) + '</span>' +
        '<span class="text-[11px] font-mono text-gray-500">' + ts + '</span></div>' +
        '<p class="text-sm text-gray-300">' + (e.reason || "") + '</p>' +
        '<div class="mt-2 flex flex-wrap gap-3 text-[11px] font-mono text-gray-500">' +
        '<span>base ' + (e.baseline_accuracy*100).toFixed(2) + '%</span>' +
        '<span>new ' + (e.new_accuracy*100).toFixed(2) + '%</span>' +
        '<span>Δ ' + (e.accuracy_change>=0?"+":"") + (e.accuracy_change*100).toFixed(2) + ' pp</span>' +
        '<span>' + e.samples_used + ' samples</span></div></div></div>';
    }).join("");
  } catch (_) {}
}

async function fetchVisualizations() {
  try {
    const res = await fetch("/visualizations");
    if (!res.ok) {
      document.getElementById("viz-empty").classList.remove("hidden");
      document.getElementById("viz-content").classList.add("hidden");
      return;
    }
    const m = await res.json();
    document.getElementById("viz-empty").classList.add("hidden");
    document.getElementById("viz-content").classList.remove("hidden");
    setText("viz-accuracy", m.accuracy != null ? (m.accuracy * 100).toFixed(1) + "%" : "—");
    setText("viz-f1", m.macro_f1 != null ? m.macro_f1.toFixed(3) : "—");
    setText("viz-updated", m.updated_at ? new Date(m.updated_at).toLocaleString() : "—");
    const names = m.class_names || CLASS_NAMES;
    if (typeof Chart !== "undefined") {
      if (chartInstances.dist) chartInstances.dist.destroy();
      chartInstances.dist = new Chart(document.getElementById("dist-chart"), {
        type: "bar",
        data: { labels: names, datasets: [{ data: m.class_distribution || [], backgroundColor: CLASS_COLORS.map(c => c+"bb"), borderRadius: 4 }] },
        options: { plugins: { legend: { display: false } }, scales: {
          x: { ticks: { color: "#9ca3af", maxRotation: 45, font: { size: 10 } }, grid: { display: false } },
          y: { ticks: { color: "#6b7280" }, grid: { color: "rgba(255,255,255,0.04)" } },
        }},
      });
      if (chartInstances.f1) chartInstances.f1.destroy();
      chartInstances.f1 = new Chart(document.getElementById("f1-chart"), {
        type: "bar",
        data: { labels: names, datasets: [{ data: m.per_class_f1 || [], backgroundColor: CLASS_COLORS.map(c => c+"bb"), borderRadius: 4 }] },
        options: { plugins: { legend: { display: false } }, scales: {
          x: { ticks: { color: "#9ca3af", maxRotation: 45, font: { size: 10 } }, grid: { display: false } },
          y: { max: 1, ticks: { color: "#6b7280" }, grid: { color: "rgba(255,255,255,0.04)" } },
        }},
      });
    }
    const cm = m.confusion_matrix || [], parent = document.getElementById("cm-container");
    if (cm.length && parent) {
      let html = '<table class="w-full text-center text-[11px] font-mono"><thead><tr><th class="p-1"></th>';
      names.forEach(n => { html += '<th class="p-1 text-gray-500 font-normal">' + n.slice(0,4) + '</th>'; });
      html += '</tr></thead><tbody>';
      cm.forEach((row, i) => {
        html += '<tr><th class="p-1 text-gray-500 font-normal text-left">' + names[i] + '</th>';
        row.forEach((v, j) => {
          const mx = Math.max(...row, 1), alpha = 0.1 + 0.75 * (v / mx);
          const bg = i === j ? 'rgba(16,185,129,' + alpha + ')' : 'rgba(255,255,255,' + (alpha * 0.25) + ')';
          html += '<td class="p-1 rounded" style="background:' + bg + '">' + v + '</td>';
        });
        html += '</tr>';
      });
      html += '</tbody></table>';
      parent.innerHTML = html;
    }
  } catch (_) {
    document.getElementById("viz-empty").classList.remove("hidden");
    document.getElementById("viz-content").classList.add("hidden");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initDeploymentBadge();
  switchTab("dashboard");
  document.querySelectorAll("[data-tab]").forEach(btn => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));
  fetchHealth();
  setInterval(fetchHealth, 10000);
  fetchVisualizations();
  fetchRetrainHistory();
  const fileInput = document.getElementById("file-input");
  if (fileInput) fileInput.addEventListener("change", e => { if (e.target.files && e.target.files[0]) runPrediction(e.target.files[0], "full"); });
  wireDropZone("drop-zone", f => runPrediction(f, "full"));
  const browseBtn = document.getElementById("browse-btn");
  if (browseBtn) browseBtn.addEventListener("click", e => { e.stopPropagation(); if (fileInput) fileInput.click(); });
  const qInput = document.getElementById("quick-file-input");
  if (qInput) qInput.addEventListener("change", e => { if (e.target.files && e.target.files[0]) runPrediction(e.target.files[0], "quick"); });
  wireDropZone("quick-drop-zone", f => runPrediction(f, "quick"));
  const clearBtn = document.getElementById("clear-btn");
  if (clearBtn) clearBtn.addEventListener("click", clearPrediction);
  const upBtn = document.getElementById("upload-images-btn");
  if (upBtn) upBtn.addEventListener("click", handleUpload);
  const zipBtn = document.getElementById("upload-zip-btn");
  if (zipBtn) zipBtn.addEventListener("click", handleZipUpload);
  const rtBtn = document.getElementById("retrain-btn");
  if (rtBtn) rtBtn.addEventListener("click", triggerRetrain);
});
