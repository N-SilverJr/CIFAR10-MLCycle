// CIFAR-10 ML Cycle – dashboard logic (plain JS, no framework)

let chartInstances = {};
let sessionUploadCount = 0;

const CLASS_NAMES = [
  "airplane", "automobile", "bird", "cat", "deer",
  "dog", "frog", "horse", "ship", "truck",
];

const CLASS_COLORS = [
  "#4F8FF7", "#F76E6E", "#4CD37B", "#F7C948", "#B27FF0",
  "#F7924F", "#4FD1F7", "#F74FA3", "#8AC24A", "#7B8CF7",
];

const TAB_TITLES = {
  dashboard: "Dashboard",
  predict: "Predict",
  visualizations: "Visualizations",
  retrain: "Retrain",
  history: "History",
};

const NAV_BTN_ACTIVE =
  "w-full text-left px-4 py-3 rounded-lg transition-all duration-200 border-l-2 flex items-start gap-3 text-white bg-blue-600/15 border-blue-500";
const NAV_BTN_INACTIVE =
  "w-full text-left px-4 py-3 rounded-lg transition-all duration-200 border-l-2 flex items-start gap-3 text-gray-400 hover:text-gray-200 hover:bg-gray-800/50 border-transparent";

function switchTab(tabName) {
  if (!TAB_TITLES[tabName]) return;
  document.querySelectorAll("[data-panel]").forEach((panel) => {
    panel.classList.toggle("hidden", panel.id !== `panel-${tabName}`);
  });
  document.querySelectorAll("[data-tab]").forEach((btn) => {
    btn.className = btn.dataset.tab === tabName ? NAV_BTN_ACTIVE : NAV_BTN_INACTIVE;
  });
  const titleEl = document.getElementById("top-bar-title");
  if (titleEl) titleEl.textContent = TAB_TITLES[tabName];
}

function openSidebar() {
  document.getElementById("sidebar").classList.add("sidebar-open");
  document.getElementById("sidebar-backdrop").classList.remove("hidden");
}
function closeSidebar() {
  document.getElementById("sidebar").classList.remove("sidebar-open");
  document.getElementById("sidebar-backdrop").classList.add("hidden");
}
function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  if (sidebar.classList.contains("sidebar-open")) closeSidebar();
  else openSidebar();
}

function initDeploymentBadge() {
  const badge = document.getElementById("deployment-badge");
  if (!badge) return;
  const isLocal = ["localhost", "127.0.0.1", ""].includes(window.location.hostname);
  if (isLocal) {
    badge.textContent = "Local";
    badge.className = "text-xs text-gray-400 bg-gray-800 rounded-full px-3 py-1";
  } else {
    badge.textContent = "Cloud: Live";
    badge.className = "text-xs text-green-400 bg-green-400/10 rounded-full px-3 py-1";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initDeploymentBadge();
  switchTab("dashboard");

  document.querySelectorAll("[data-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      switchTab(btn.dataset.tab);
      closeSidebar();
    });
  });
  document.getElementById("sidebar-toggle").addEventListener("click", toggleSidebar);
  document.getElementById("sidebar-backdrop").addEventListener("click", closeSidebar);

  fetchHealth();
  setInterval(fetchHealth, 10000);
  fetchVisualizations();
  fetchRetrainHistory();

  document.getElementById("file-input").addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) runPrediction(e.target.files[0], "full");
  });
  wireDropZone("drop-zone", (file) => runPrediction(file, "full"));

  document.getElementById("quick-file-input").addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) runPrediction(e.target.files[0], "quick");
  });
  wireDropZone("quick-drop-zone", (file) => runPrediction(file, "quick"));

  document.getElementById("clear-btn").addEventListener("click", clearPrediction);
  document.getElementById("upload-images-btn").addEventListener("click", handleUpload);
  document.getElementById("upload-zip-btn").addEventListener("click", handleZipUpload);
  document.getElementById("retrain-btn").addEventListener("click", triggerRetrain);
});

function wireDropZone(zoneId, onFile) {
  const zone = document.getElementById(zoneId);
  zone.addEventListener("click", () => {
    const input = zone.querySelector('input[type="file"]');
    if (input) input.click();
  });
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("border-blue-500");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("border-blue-500"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("border-blue-500");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
  });
}

function formatUptime(seconds) {
  const s = Math.max(0, Math.floor(seconds || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return `${h}h ${m}m ${sec}s`;
}

const ALERT_STYLES = {
  success: { border: "border-green-500", text: "text-green-400" },
  warning: { border: "border-amber-500", text: "text-amber-400" },
  error: { border: "border-red-500", text: "text-red-400" },
  info: { border: "border-blue-500", text: "text-blue-300" },
};

function showAlert(containerId, type, message) {
  const container = document.getElementById(containerId);
  const style = ALERT_STYLES[type] || ALERT_STYLES.info;
  container.innerHTML = `<div class="border-l-4 ${style.border} bg-gray-900/60 rounded-r-md px-3 py-2 text-sm ${style.text} mt-2">${message}</div>`;
}

async function fetchHealth() {
  const statusBadge = document.getElementById("status-badge");
  const statusDot = document.getElementById("status-dot");
  const uptimeEl = document.getElementById("stat-uptime");
  const lastTrainedEl = document.getElementById("stat-last-trained");
  const paramsEl = document.getElementById("stat-params");
  const sidebarDot = document.getElementById("sidebar-status-dot");
  const sidebarText = document.getElementById("sidebar-status-text");
  const sidebarUptime = document.getElementById("sidebar-uptime-compact");

  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("health check failed");
    const data = await res.json();
    const online = data.status === "healthy" && data.model_loaded;

    statusBadge.textContent = online ? "Online" : "Offline";
    statusBadge.className = online
      ? "inline-block text-sm font-semibold text-green-400"
      : "inline-block text-sm font-semibold text-red-400";
    statusDot.classList.toggle("hidden", !online);

    uptimeEl.textContent = formatUptime(data.uptime_seconds);
    lastTrainedEl.textContent = data.model_last_trained
      ? new Date(data.model_last_trained).toLocaleString()
      : "Unknown";
    paramsEl.textContent =
      data.model_parameters != null ? data.model_parameters.toLocaleString() : "Unknown";

    sidebarDot.className = `w-2 h-2 rounded-full shrink-0 ${online ? "bg-green-500" : "bg-red-500"}`;
    sidebarText.textContent = online ? "Online" : "Offline";
    sidebarUptime.textContent = formatUptime(data.uptime_seconds);
  } catch (err) {
    statusBadge.textContent = "Offline";
    statusBadge.className = "inline-block text-sm font-semibold text-red-400";
    statusDot.classList.add("hidden");
    uptimeEl.textContent = "Unknown";
    lastTrainedEl.textContent = "Unknown";
    paramsEl.textContent = "Unknown";
    sidebarDot.className = "w-2 h-2 rounded-full shrink-0 bg-red-500";
    sidebarText.textContent = "Offline";
    sidebarUptime.textContent = "--";
  }
}

async function runPrediction(file, mode) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");

    if (mode === "quick") {
      document.getElementById("quick-result").classList.remove("hidden");
      document.getElementById("quick-pred-class").textContent = data.predicted_class;
      document.getElementById("quick-pred-conf").textContent =
        (data.confidence * 100).toFixed(1) + "%";
      document.getElementById("quick-pred-ms").textContent = data.processing_time_ms;
    } else {
      const resultEl = document.getElementById("pred-result");
      resultEl.classList.remove("hidden");
      document.getElementById("pred-class").textContent = data.predicted_class;
      document.getElementById("pred-conf").textContent =
        (data.confidence * 100).toFixed(1) + "%";
      document.getElementById("pred-ms").textContent = data.processing_time_ms;

      const preview = document.getElementById("pred-preview");
      preview.src = URL.createObjectURL(file);

      // probability bar chart
      const probs = CLASS_NAMES.map((n) => (data.probabilities[n] || 0) * 100);
      if (chartInstances.prob) chartInstances.prob.destroy();
      chartInstances.prob = new Chart(document.getElementById("prob-chart"), {
        type: "bar",
        data: {
          labels: CLASS_NAMES,
          datasets: [{ data: probs, backgroundColor: CLASS_COLORS }],
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, max: 100, ticks: { color: "#9ca3af" } },
            x: { ticks: { color: "#9ca3af", maxRotation: 45 } },
          },
        },
      });
    }
  } catch (err) {
    alert("Prediction error: " + err.message);
  }
}

function clearPrediction() {
  document.getElementById("pred-result").classList.add("hidden");
  document.getElementById("file-input").value = "";
  if (chartInstances.prob) {
    chartInstances.prob.destroy();
    chartInstances.prob = null;
  }
}

async function handleUpload() {
  const label = document.getElementById("upload-label").value;
  const files = document.getElementById("upload-files").files;
  if (!files.length) {
    showAlert("upload-alert", "warning", "Select at least one image.");
    return;
  }
  const formData = new FormData();
  formData.append("label", label);
  for (const f of files) formData.append("files", f);

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");
    sessionUploadCount += data.uploaded;
    document.getElementById("upload-status").textContent =
      `${sessionUploadCount} images uploaded this session`;
    showAlert(
      "upload-alert",
      "success",
      `Uploaded ${data.uploaded} image(s) for class ${data.class_name} (${data.label}).`
    );
  } catch (err) {
    showAlert("upload-alert", "error", err.message);
  }
}

async function handleZipUpload() {
  const file = document.getElementById("upload-zip").files[0];
  if (!file) {
    showAlert("upload-alert", "warning", "Select a ZIP file.");
    return;
  }
  const formData = new FormData();
  formData.append("files", file);

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "ZIP upload failed");
    sessionUploadCount += data.uploaded;
    document.getElementById("upload-status").textContent =
      `${sessionUploadCount} images uploaded this session`;
    showAlert(
      "upload-alert",
      "success",
      `Uploaded ${data.uploaded} image(s) across classes: ${JSON.stringify(data.per_class)}`
    );
  } catch (err) {
    showAlert("upload-alert", "error", err.message);
  }
}

async function triggerRetrain() {
  const btn = document.getElementById("retrain-btn");
  btn.disabled = true;
  btn.textContent = "Retraining… (this may take a minute)";
  showAlert("retrain-alert", "info", "Retraining in progress…");

  try {
    const res = await fetch("/retrain", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Retrain failed");

    const type = data.promoted ? "success" : "warning";
    showAlert(
      "retrain-alert",
      type,
      `${data.status.toUpperCase()}: ${data.reason} (Δ ${data.accuracy_change >= 0 ? "+" : ""}${(data.accuracy_change * 100).toFixed(2)} pp, ${data.samples_used} samples, ${data.training_time_seconds}s)`
    );
    fetchRetrainHistory();
    if (data.promoted) fetchVisualizations();
  } catch (err) {
    showAlert("retrain-alert", "error", err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Retrain Model";
  }
}

async function fetchRetrainHistory() {
  try {
    const res = await fetch("/retrain-history");
    const log = await res.json();
    const body = document.getElementById("history-body");
    if (!log.length) {
      body.innerHTML = `<tr><td colspan="7" class="text-center text-gray-500">No retraining events yet.</td></tr>`;
      return;
    }
    body.innerHTML = log
      .slice()
      .reverse()
      .map(
        (e) => `
      <tr>
        <td class="text-xs">${e.timestamp ? new Date(e.timestamp).toLocaleString() : "--"}</td>
        <td><span class="badge ${e.promoted ? "badge-success" : "badge-warning"} badge-sm">${e.status}</span></td>
        <td>${e.baseline_accuracy?.toFixed(4) ?? "--"}</td>
        <td>${e.new_accuracy?.toFixed(4) ?? "--"}</td>
        <td class="${e.accuracy_change >= 0 ? "text-green-400" : "text-red-400"}">${e.accuracy_change >= 0 ? "+" : ""}${(e.accuracy_change * 100).toFixed(2)} pp</td>
        <td>${e.samples_used ?? "--"}</td>
        <td class="text-xs max-w-xs truncate" title="${e.reason || ""}">${e.reason || "--"}</td>
      </tr>`
      )
      .join("");
  } catch (err) {
    console.error(err);
  }
}

async function fetchVisualizations() {
  try {
    const res = await fetch("/visualizations");
    if (!res.ok) {
      document.getElementById("viz-empty").classList.remove("hidden");
      document.getElementById("viz-content").classList.add("hidden");
      return;
    }
    const data = await res.json();
    document.getElementById("viz-empty").classList.add("hidden");
    document.getElementById("viz-content").classList.remove("hidden");

    const names = data.class_names || CLASS_NAMES;

    // Class distribution
    if (chartInstances.dist) chartInstances.dist.destroy();
    chartInstances.dist = new Chart(document.getElementById("dist-chart"), {
      type: "bar",
      data: {
        labels: names,
        datasets: [{ data: data.class_distribution, backgroundColor: CLASS_COLORS }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, ticks: { color: "#9ca3af" } },
          x: { ticks: { color: "#9ca3af", maxRotation: 45 } },
        },
      },
    });

    // Per-class F1
    if (chartInstances.f1) chartInstances.f1.destroy();
    chartInstances.f1 = new Chart(document.getElementById("f1-chart"), {
      type: "bar",
      data: {
        labels: names,
        datasets: [{ data: data.per_class_f1, backgroundColor: CLASS_COLORS }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, max: 1, ticks: { color: "#9ca3af" } },
          x: { ticks: { color: "#9ca3af", maxRotation: 45 } },
        },
      },
    });

    // Confusion matrix as heatmap (simple bar-like using Chart.js matrix is overkill; use table-style canvas)
    // For simplicity we render a basic matrix visualization
    if (chartInstances.cm) chartInstances.cm.destroy();
    const cm = data.confusion_matrix;
    // Flatten for a simple heatmap-like bar chart is not ideal; use a table instead for clarity
    const cmCanvas = document.getElementById("cm-chart");
    const parent = cmCanvas.parentElement;
    let tableHtml = `<table class="table table-xs text-center"><thead><tr><th></th>${names
      .map((n) => `<th class="text-xs">${n.slice(0, 4)}</th>`)
      .join("")}</tr></thead><tbody>`;
    cm.forEach((row, i) => {
      tableHtml += `<tr><th class="text-xs">${names[i]}</th>`;
      row.forEach((v, j) => {
        const intensity = Math.min(1, v / 500);
        const bg = i === j ? `rgba(34,197,94,${0.2 + intensity * 0.6})` : `rgba(239,68,68,${intensity * 0.5})`;
        tableHtml += `<td style="background:${bg}" class="text-xs">${v}</td>`;
      });
      tableHtml += `</tr>`;
    });
    tableHtml += `</tbody></table>`;
    parent.innerHTML = tableHtml;
  } catch (err) {
    console.error(err);
  }
}
