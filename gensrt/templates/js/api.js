// ── Load Button (pywebview native file dialog) ────────────
// In browser mode this stays hidden; revealed when pywebview is ready.
if (browseBtn) {
  browseBtn.style.display = window.__tilesterPyWebViewReady ? 'inline-block' : 'none';

  window.__tilesterOnPyWebViewReady = () => {
    try { browseBtn.style.display = 'inline-block'; } catch {}
  };

  browseBtn.addEventListener('click', () => {
    const api = (typeof window.pywebview !== 'undefined' && window.pywebview) ? window.pywebview.api : null;
    if (!api || typeof api.select_video !== 'function') {
      showErrorDialog('Load Unavailable', 'Native file picker is only available in the <strong>pywebview</strong> app window.');
      return;
    }
    api.select_video().then(path => {
      // Route via window.* so we hit the patched version installed by the
      // sidecar hook in project.js — bare `tilesterSetVideoFromPath` would
      // resolve to the IIFE-scoped original and skip sidecar discovery.
      if (path) window.tilesterSetVideoFromPath(path);
    }).catch(err => {
      console.error('Load dialog error:', err);
      showErrorDialog('Load Failed', 'Load dialog failed. See console for details.');
    });
  });

  if (window.__tilesterPyWebViewReady) {
    try { window.__tilesterOnPyWebViewReady(); } catch {}
  }
}

// ── Panel Header Action Listeners ────────────────────────
if (splitBtn)  splitBtn.addEventListener('click',  (e) => { e.preventDefault(); splitSegmentAtPlayhead(); });
if (mergeBtn)  mergeBtn.addEventListener('click',  (e) => { e.preventDefault(); mergeAdjacentSegments(); });
if (deleteBtn) deleteBtn.addEventListener('click', (e) => { e.preventDefault(); deleteSelectedSegments(); });

// ── Button State Management ───────────────────────────────
function updateButtonStates() {
  const hasVideoPath = videoPathInput.value.trim().length > 0;
  // currentProjectPath is set whenever an SRT is loaded or saved with a path
  // (drop, file picker, sidecar discovery, Save, Save As).  It's cleared when
  // a fresh video is loaded that hasn't been transcribed yet.
  const hasSrt = !!(typeof currentProjectPath !== 'undefined' && currentProjectPath);
  detectBtn.disabled = !hasVideoPath;
  const burnBtn = document.getElementById('burnBtn');
  if (burnBtn) burnBtn.disabled = !(hasVideoPath && hasSrt);
}

videoPathInput.addEventListener('input', updateButtonStates);
updateButtonStates();

// ── Per-job footer selectors (Source / Target lang + Translation engine) ─
//
// These three <select>s are populated on page load:
//   * Source language and Translation Engine pre-fill from /api/config defaults
//     (the user's saved gensrt-config.json).
//   * Translation Engine's option list is fetched from /api/engines so it
//     reflects what's actually installed.
//   * Target language is locked to English for now — placeholder UI for the
//     future multi-target translation feature.
//
// Values are not auto-persisted to gensrt-config.json (that would silently
// overwrite the user's defaults).  Drop G will read these on Generate SRT
// click and send them with the /api/transcribe request as per-job overrides.

const ENGINE_LABELS = {
  google:      'Google (GTX)',
  nllb:        'NLLB-200 (offline)',
  marian:      'MarianMT (offline)',
  passthrough: 'None (skip translation)',
  none:        'None (skip translation)',
};

async function _initFooterSelectors() {
  const selSrc    = document.getElementById('sel-source-lang');
  const selTgt    = document.getElementById('sel-target-lang');
  const selEngine = document.getElementById('sel-engine');
  const selVad    = document.getElementById('sel-vad');
  if (!selSrc && !selEngine && !selVad && !selTgt) return;

  // 1. Translation engine options come from the backend.
  if (selEngine) {
    try {
      const res  = await fetch('/api/engines');
      const data = await res.json();
      const list = (data && Array.isArray(data.engines)) ? data.engines : [];
      selEngine.innerHTML = '';
      for (const key of list) {
        const opt = document.createElement('option');
        opt.value       = key;
        opt.textContent = ENGINE_LABELS[key] || key;
        selEngine.appendChild(opt);
      }
    } catch (err) {
      console.error('Failed to load /api/engines:', err);
    }
  }

  // 2. Pre-fill source language, target language, engine, and VAD from saved config defaults.
  try {
    const res    = await fetch('/api/config');
    const result = await res.json();
    const cfg    = (result && result.config) ? result.config : {};
    if (selSrc && cfg.source_language) {
      // Only set if the value is one of our offered options; otherwise leave at 'auto'.
      const opt = selSrc.querySelector(`option[value="${cfg.source_language}"]`);
      if (opt) selSrc.value = cfg.source_language;
    }
    if (selTgt && cfg.target_language) {
      // Only set if it's one of the curated UI options.  If a power user has
      // a target outside our list set via CLI/config, the footer stays on
      // its default — they can adjust via the modal or use the CLI for that
      // job.  This keeps the footer dropdown deterministic.
      const opt = selTgt.querySelector(`option[value="${cfg.target_language}"]`);
      if (opt) selTgt.value = cfg.target_language;
    }
    if (selEngine && cfg.translation_engine) {
      const opt = selEngine.querySelector(`option[value="${cfg.translation_engine}"]`);
      if (opt) selEngine.value = cfg.translation_engine;
    }
    if (selVad && typeof cfg.vad_enabled === 'boolean') {
      selVad.value = cfg.vad_enabled ? 'on' : 'off';
    }
  } catch (err) {
    console.error('Failed to load /api/config for selector defaults:', err);
  }
}

_initFooterSelectors();

// Auto-fill video path when video is loaded via browser drag-drop
// Note: browser security prevents full path access — user can correct if needed
const _origLoadVideo = loadVideo;
loadVideo = function(file) {
  _origLoadVideo.call(this, file);
  videoPathInput.value = file.name;
  updateButtonStates();
};

// ── Server Mode Detection ─────────────────────────────────
async function checkServerMode() {
  try {
    await fetch('/api/detect', { method: 'OPTIONS' });
    isServerMode = true;
    console.log('Server mode detected - API available');

    try {
      const cfgResponse = await fetch('/api/config', { method: 'GET' });
      const cfgResult   = await cfgResponse.json();
      if (cfgResult.status === 'success') {
        applyConfigToUI(cfgResult.config);
        console.log('Config applied on startup:', cfgResult.config);
      }
    } catch (e) {
      console.warn('Could not load config on startup:', e);
    }
  } catch (error) {
    isServerMode = false;
    console.log('Offline mode - using script download');
  }
}

// ── Job Runner ────────────────────────────────────────────
async function startJobDirect(endpoint, payload, title, onComplete, workingMessage = 'Working...', options = {}) {
  const { useServerProgress = false, operationKind = '' } = options || {};

  showProgressModal(title);
  updateProgress(1, 'Starting...');

  let progressTick = null;
  if (useServerProgress) {
    startProgressPolling(operationKind);
  } else {
    let pct = 3;
    progressTick = setInterval(() => {
      pct = Math.min(92, pct + (pct < 20 ? 7 : pct < 55 ? 4 : 2));
      updateProgress(pct, workingMessage);
    }, 450);
  }

  try {
    const response = await fetch(endpoint, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload)
    });

    const raw = await response.text();
    let data = {};
    if (raw) {
      try { data = JSON.parse(raw); }
      catch (err) { console.error('Bad JSON response:', raw); throw new Error(raw || `HTTP ${response.status}`); }
    }

    if (!response.ok || data.status === 'error') {
      showProgressError('Failed', data.message || `HTTP ${response.status}`);
      return false;
    }

    if (progressTick) clearInterval(progressTick);
    updateProgress(100, 'Done');

    try {
      onComplete && onComplete(data);
    } catch (e) {
      console.error('onComplete handler failed:', e);
      showProgressError('Failed', `UI handler error: ${e.message || e}`);
      return false;
    }

    return true;
  } catch (err) {
    console.error('Request error:', err);
    showProgressError('Failed', err.message || 'Network error');
    return false;
  } finally {
    if (progressTick) clearInterval(progressTick);
  }
}

// ── API Operations ────────────────────────────────────────
//
// Function name kept as `callDetectAPI` because init.js (off-limits) calls
// it.  The body is now a Generate SRT flow: collect overrides from the
// footer selectors, POST /api/transcribe, then load the resulting .srt
// into the right pane via project.js's loadSrtFromPath().
async function callDetectAPI() {
  let videoPath = normalizeFullPath(videoPathInput.value.trim());

  if (!videoPath) {
    showErrorDialog('Invalid Path',
      'Please load a video first.<br><br>' +
      '<div style="font-family: var(--font-mono); font-size: 12px; color: var(--text-dim);">Use drag-and-drop, click the player area, or the Load button.</div>');
    return;
  }
  if (!videoPath.includes('\\') && !videoPath.includes('/')) {
    showErrorDialog('Invalid Path',
      `Invalid path: <span style="font-family: var(--font-mono);">${videoPath}</span><br><br>` +
      'Please enter the <strong>FULL</strong> path, not just the filename.');
    return;
  }

  // Collect per-job overrides from the footer selectors.
  const selSrc    = document.getElementById('sel-source-lang');
  const selTgt    = document.getElementById('sel-target-lang');
  const selEngine = document.getElementById('sel-engine');
  const selVad    = document.getElementById('sel-vad');
  const payload   = { input_path: videoPath };
  if (selSrc && selSrc.value)       payload.source_language    = selSrc.value;
  if (selTgt && selTgt.value)       payload.target_language    = selTgt.value;
  if (selEngine && selEngine.value) payload.translation_engine = selEngine.value;
  if (selVad && selVad.value === 'off') payload.no_vad = true;

  detectBtn.disabled    = true;
  detectBtn.textContent = 'Generating...';

  try {
    await startJobDirect(
      '/api/transcribe',
      payload,
      'Generating SRT...',
      async (msg) => {
        // /api/transcribe returns immediately with status=started and an
        // `output` field containing the destination .srt path; the actual
        // transcription runs in a background thread and progress flows via
        // /api/operation_status (server-progress mode below).
        const srtPath = msg && msg.output ? String(msg.output) : null;
        if (srtPath && typeof window.gensrtLoadSrtFromPath === 'function') {
          const ok = await window.gensrtLoadSrtFromPath(srtPath, { quiet: true });
          if (ok) {
            showProgressSuccess('SRT Generated',
              `Loaded: <span style="font-family: var(--font-mono);">${srtPath}</span>`);
            return;
          }
        }
        showProgressSuccess('SRT Generated',
          srtPath
            ? `Saved to <span style="font-family: var(--font-mono);">${srtPath}</span>`
            : 'Transcription complete.');
      },
      'Transcribing...',
      { useServerProgress: true, operationKind: 'transcribe' }
    );
  } finally {
    detectBtn.textContent = 'Generate SRT';
    updateButtonStates();
  }
}
