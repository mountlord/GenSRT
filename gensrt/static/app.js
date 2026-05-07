/* GenSRT — frontend application */

"use strict";

// ── State ──────────────────────────────────────────────────────────────────

const state = {
  inputPath: null,
  outputDir: null,
  running: false,
  pollTimer: null,
  startedAt: null,
  elapsedTimer: null,
};

// ── DOM refs ───────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

const dropZone       = $("drop-zone");
const fileRow        = $("file-row");
const fileNameLabel  = $("file-name-label");
const btnPickFile    = $("btn-pick-file");
const btnClearFile   = $("btn-clear-file");
const btnPickOutput  = $("btn-pick-output");
const btnClearOutput = $("btn-clear-output");
const outputDirInput = $("output-dir");
const selEngine      = $("sel-engine");
const selLang        = $("sel-lang");
const chkVad         = $("chk-vad");
const vadLabel       = $("vad-label");
const btnStart       = $("btn-start");
const progressSec    = $("progress-section");
const progressBar    = $("progress-bar");
const progressMsg    = $("progress-message");
const progressElapsed= $("progress-elapsed");
const resultSec      = $("result-section");
const resultIcon     = $("result-icon");
const resultMsg      = $("result-message");
const gpuBadge       = $("gpu-badge");
const aboutVersion   = $("about-version");

// ── Navigation ─────────────────────────────────────────────────────────────

document.querySelectorAll(".nav-item").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    $(`panel-${btn.dataset.panel}`).classList.add("active");
  });
});

// ── File selection ──────────────────────────────────────────────────────────

function setInputFile(path) {
  state.inputPath = path;
  const name = path.split(/[\\/]/).pop();
  fileNameLabel.textContent = name;
  dropZone.style.display = "none";
  fileRow.style.display = "flex";
  btnStart.disabled = false;
  clearResult();
}

function clearInputFile() {
  state.inputPath = null;
  dropZone.style.display = "";
  fileRow.style.display = "none";
  btnStart.disabled = true;
  clearResult();
}

btnPickFile.addEventListener("click", async () => {
  if (!window.pywebview) return;
  const path = await window.pywebview.api.select_file();
  if (path) setInputFile(path);
});

btnClearFile.addEventListener("click", clearInputFile);

// ── Output dir ──────────────────────────────────────────────────────────────

btnPickOutput.addEventListener("click", async () => {
  if (!window.pywebview) return;
  const dir = await window.pywebview.api.select_output_folder();
  if (dir) {
    state.outputDir = dir;
    outputDirInput.value = dir;
  }
});

btnClearOutput.addEventListener("click", () => {
  state.outputDir = null;
  outputDirInput.value = "";
});

// ── VAD toggle ──────────────────────────────────────────────────────────────

chkVad.addEventListener("change", () => {
  vadLabel.textContent = chkVad.checked ? "Enabled" : "Disabled";
});

// ── Drag and drop ──────────────────────────────────────────────────────────

dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const files = e.dataTransfer?.files;
  if (files && files.length > 0) {
    const f = files[0];
    // In a real file system context pywebview provides the full path
    if (f.path || f.name) setInputFile(f.path || f.name);
  }
});

// Expose for pyWebView Python drop handler
window.gensrtSetInput = path => {
  if (path) setInputFile(path);
};

// ── Progress helpers ───────────────────────────────────────────────────────

function showProgress(msg = "Starting…") {
  progressSec.style.display = "";
  resultSec.style.display = "none";
  progressBar.style.width = "0%";
  progressMsg.textContent = msg;
  progressElapsed.textContent = "";
  state.startedAt = Date.now();

  if (state.elapsedTimer) clearInterval(state.elapsedTimer);
  state.elapsedTimer = setInterval(() => {
    const s = ((Date.now() - state.startedAt) / 1000).toFixed(0);
    progressElapsed.textContent = `${s}s elapsed`;
  }, 1000);
}

function updateProgress(data) {
  const pct = Math.round(data.percent || 0);
  progressBar.style.width = `${pct}%`;
  progressMsg.textContent = data.message || "Working…";
}

function showResult(success, message) {
  if (state.elapsedTimer) clearInterval(state.elapsedTimer);
  progressSec.style.display = "none";
  resultSec.style.display = "flex";
  resultIcon.textContent = success ? "✅" : "❌";
  resultMsg.textContent = message;
}

function clearResult() {
  resultSec.style.display = "none";
  progressSec.style.display = "none";
  if (state.elapsedTimer) clearInterval(state.elapsedTimer);
}

// ── Polling ────────────────────────────────────────────────────────────────

function startPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const res = await fetch("/api/operation_status");
      const data = await res.json();

      if (!data.active) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        state.running = false;
        btnStart.disabled = false;

        const lastMsg = data.message || progressMsg.textContent;
        const success = !lastMsg.toLowerCase().startsWith("error");
        showResult(success, lastMsg);
      } else {
        updateProgress(data);
      }
    } catch (_) {
      // transient network hiccup — ignore
    }
  }, 600);
}

// ── Transcription start ────────────────────────────────────────────────────

btnStart.addEventListener("click", async () => {
  if (!state.inputPath || state.running) return;

  const body = {
    input_path: state.inputPath,
    output_dir: state.outputDir || null,
    translation_engine: selEngine.value,
    source_language: selLang.value,
    no_translate: selEngine.value === "none",
    no_vad: !chkVad.checked,
  };

  state.running = true;
  btnStart.disabled = true;
  showProgress("Sending to GenSRT…");

  try {
    const res = await fetch("/api/transcribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await res.json();

    if (data.status === "started") {
      showProgress("Starting…");
      startPolling();
    } else {
      state.running = false;
      btnStart.disabled = false;
      showResult(false, data.message || "Failed to start transcription.");
    }
  } catch (err) {
    state.running = false;
    btnStart.disabled = false;
    showResult(false, `Network error: ${err}`);
  }
});

// ── Startup ────────────────────────────────────────────────────────────────

async function init() {
  // GPU badge
  try {
    const res = await fetch("/api/config");
    const cfg = await res.json();
    const backend = (cfg.backend || "cpu").toUpperCase();
    const device = cfg.device || "cpu";
    const isGpu = device !== "cpu";
    gpuBadge.textContent = isGpu ? `${backend} ✓` : "CPU mode";
    gpuBadge.className = `badge ${isGpu ? "badge--success" : "badge--warning"}`;

    // Pre-set settings panel
    const mdlSel = $("cfg-model");
    const cmpSel = $("cfg-compute");
    const engSel = $("cfg-engine");
    if (mdlSel && cfg.model) mdlSel.value = cfg.model;
    if (cmpSel && cfg.compute_type) cmpSel.value = cfg.compute_type;
    if (engSel && cfg.translation_engine) engSel.value = cfg.translation_engine;

  } catch (_) {
    gpuBadge.textContent = "Server error";
    gpuBadge.className = "badge badge--error";
  }

  // Version
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (aboutVersion) aboutVersion.textContent = `GenSRT v${data.version}`;
  } catch (_) {}

  // Pre-load path from CLI (e.g. gensrt video.mkv launched in GUI mode)
  if (window.pywebview) {
    try {
      const path = await window.pywebview.api.get_open_path();
      if (path) setInputFile(path);
    } catch (_) {}
  }
}

document.addEventListener("DOMContentLoaded", init);
