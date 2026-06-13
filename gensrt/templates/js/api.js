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

// ── Per-job footer selectors (Source / Target lang + Model + VAD) ────────
//
// These four <select>s are populated on page load:
//   * Source language and Target language pre-fill from /api/config defaults.
//   * Model is populated from /api/known_models (built-in recommended +
//     user-added).  A "New…" sentinel option opens the Add Custom Model
//     modal — on success, the new model joins the dropdown and is selected.
//   * VAD on/off pre-fills from the saved default.
//
// Translation engine is no longer in the footer (moved to Config modal
// only) — model has more impact on output quality and earns the slot.
// The backend still respects translation_engine from the saved config.
//
// Values are not auto-persisted to gensrt-config.json — that would silently
// overwrite the user's defaults.  callDetectAPI() reads these on Generate
// SRT click and sends them with /api/transcribe as per-job overrides.

const ENGINE_LABELS = {
  google:      'Google (GTX)',
  nllb:        'NLLB-200 (offline)',
  marian:      'MarianMT (offline)',
  passthrough: 'None (skip translation)',
  none:        'None (skip translation)',
};

// Sentinel value used in the model dropdown to mean "open the Add Custom
// Model modal".  Kept distinct from any real model name.
const NEW_MODEL_SENTINEL = '__new_model__';

// Globally accessible so config.js can rebuild the Config modal's model
// dropdown using the same merged list.
window.__gensrtKnownModels = [];

async function _fetchKnownModels() {
  try {
    const res  = await fetch('/api/known_models');
    const data = await res.json();
    const list = (data && Array.isArray(data.models)) ? data.models : [];
    window.__gensrtKnownModels = list;
    return list;
  } catch (err) {
    console.error('Failed to load /api/known_models:', err);
    return window.__gensrtKnownModels || [];
  }
}

function _populateModelSelect(selModel, models, selectedValue) {
  if (!selModel) return;
  selModel.innerHTML = '';
  const seen = new Set();
  for (const m of models) {
    if (seen.has(m)) continue;
    seen.add(m);
    const opt = document.createElement('option');
    opt.value       = m;
    opt.textContent = m;
    selModel.appendChild(opt);
  }
  // If the saved value isn't in the known-models list (e.g. typed into
  // config.json manually), add it as an extra option so it survives the
  // round-trip through this dropdown.
  if (selectedValue && !seen.has(selectedValue)) {
    const opt = document.createElement('option');
    opt.value       = selectedValue;
    opt.textContent = selectedValue + ' (from config)';
    selModel.insertBefore(opt, selModel.firstChild);
  }
  // Sentinel "New…" at the bottom.
  const sep = document.createElement('option');
  sep.disabled  = true;
  sep.textContent = '──────────';
  selModel.appendChild(sep);
  const neu = document.createElement('option');
  neu.value       = NEW_MODEL_SENTINEL;
  neu.textContent = 'New…';
  selModel.appendChild(neu);

  if (selectedValue) selModel.value = selectedValue;
}

async function _initFooterSelectors() {
  const selSrc    = document.getElementById('sel-source-lang');
  const selTgt    = document.getElementById('sel-target-lang');
  const selModel  = document.getElementById('sel-model');
  const selVad    = document.getElementById('sel-vad');
  if (!selSrc && !selModel && !selVad && !selTgt) return;

  // 1. Load model list (built-in + user-added) and current config in parallel.
  let cfg = {};
  let knownModels = [];
  try {
    const [cfgRes, modelsList] = await Promise.all([
      fetch('/api/config').then(r => r.json()),
      _fetchKnownModels(),
    ]);
    cfg = (cfgRes && cfgRes.config) ? cfgRes.config : {};
    knownModels = modelsList;
  } catch (err) {
    console.error('Footer init: failed to fetch config / known models:', err);
  }

  // 2. Populate model dropdown — saved value wins; if it's not in the
  //    known list, we add it as a "(from config)" option so it survives.
  if (selModel) {
    _populateModelSelect(selModel, knownModels, cfg.model || '');
    selModel.addEventListener('change', _onModelSelectChange);
  }

  // 3. Pre-fill source language, target language, and VAD from saved config.
  if (selSrc && cfg.source_language) {
    const opt = selSrc.querySelector(`option[value="${cfg.source_language}"]`);
    if (opt) selSrc.value = cfg.source_language;
  }
  if (selTgt && cfg.target_language) {
    const opt = selTgt.querySelector(`option[value="${cfg.target_language}"]`);
    if (opt) selTgt.value = cfg.target_language;
  }
  if (selVad && typeof cfg.vad_enabled === 'boolean') {
    selVad.value = cfg.vad_enabled ? 'on' : 'off';
  }
}

// ── Model dropdown "New…" handler ─────────────────────────────────────────
//
// When the user picks the sentinel option, we open the Add Custom Model
// modal.  On a successful Validate & Add, the new model is appended to the
// known-models side file via /api/add_known_model, then we rebuild the
// dropdown so the new entry appears and is selected.  On Cancel, we
// snap the dropdown back to whatever was selected before.

let _lastModelSelection = null;

function _onModelSelectChange(ev) {
  const sel = ev.target;
  if (sel.value !== NEW_MODEL_SENTINEL) {
    _lastModelSelection = sel.value;
    return;
  }
  // User picked "New…" — open the modal.
  _openAddModelModal();
}

function _openAddModelModal() {
  const modal  = document.getElementById('addModelModal');
  const input  = document.getElementById('addModelInput');
  const status = document.getElementById('addModelStatus');
  const cancel = document.getElementById('addModelCancel');
  const ok     = document.getElementById('addModelOk');
  if (!modal || !input || !ok || !cancel) return;

  input.value         = '';
  status.textContent  = '';
  status.style.color  = '';
  ok.disabled         = false;
  ok.textContent      = 'Validate & Add';

  const restoreSelection = () => {
    const selModel = document.getElementById('sel-model');
    if (!selModel) return;
    if (_lastModelSelection &&
        selModel.querySelector(`option[value="${CSS.escape(_lastModelSelection)}"]`)) {
      selModel.value = _lastModelSelection;
    } else {
      // Fall back to the first non-sentinel option.
      for (const opt of selModel.options) {
        if (!opt.disabled && opt.value && opt.value !== NEW_MODEL_SENTINEL) {
          selModel.value = opt.value;
          break;
        }
      }
    }
  };

  const cleanup = () => {
    modal.classList.remove('visible');
    ok.removeEventListener('click', onOk);
    cancel.removeEventListener('click', onCancel);
    input.removeEventListener('keydown', onKey);
  };

  const onCancel = () => {
    restoreSelection();
    cleanup();
  };
  const onKey = (e) => {
    if (e.key === 'Enter')   { e.preventDefault(); onOk(); }
    if (e.key === 'Escape')  { e.preventDefault(); onCancel(); }
  };
  const onOk = async () => {
    const name = (input.value || '').trim();
    if (!name) {
      status.textContent = 'Please enter a model name.';
      status.style.color = 'var(--seam-on)';
      return;
    }
    ok.disabled        = true;
    ok.textContent     = 'Validating…';
    status.textContent = 'Checking model on HuggingFace…';
    status.style.color = 'var(--text-dim)';

    try {
      const valResp = await fetch('/api/validate_model', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ model: name }),
      });
      const valData = await valResp.json().catch(() => ({}));
      if (!valResp.ok || valData.status !== 'ok') {
        status.textContent = valData.message || `Validation failed (HTTP ${valResp.status})`;
        status.style.color = '#ef4444';
        ok.disabled = false;
        ok.textContent = 'Validate & Add';
        return;
      }

      // Validation passed — persist to side file.
      const addResp = await fetch('/api/add_known_model', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ model: name }),
      });
      const addData = await addResp.json().catch(() => ({}));
      if (!addResp.ok || addData.status !== 'ok') {
        status.textContent = addData.message || `Could not save: HTTP ${addResp.status}`;
        status.style.color = '#ef4444';
        ok.disabled = false;
        ok.textContent = 'Validate & Add';
        return;
      }

      // Success — rebuild the dropdown with the new model, select it,
      // and close the modal.  Also sync window.__gensrtKnownModels for
      // config.js to pick up next time.
      window.__gensrtKnownModels = Array.isArray(addData.models) ? addData.models : window.__gensrtKnownModels;
      const selModel = document.getElementById('sel-model');
      if (selModel) {
        _populateModelSelect(selModel, window.__gensrtKnownModels, name);
        _lastModelSelection = name;
      }
      // Also tell the Config modal (if open) to refresh its dropdown.
      try { window.dispatchEvent(new CustomEvent('gensrt:known_models_updated', { detail: { models: window.__gensrtKnownModels, selected: name } })); }
      catch (e) {}

      cleanup();
    } catch (err) {
      console.error('Add model failed:', err);
      status.textContent = `Error: ${err.message || err}`;
      status.style.color = '#ef4444';
      ok.disabled = false;
      ok.textContent = 'Validate & Add';
    }
  };

  ok.addEventListener('click', onOk);
  cancel.addEventListener('click', onCancel);
  input.addEventListener('keydown', onKey);
  modal.classList.add('visible');
  setTimeout(() => input.focus(), 10);
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
  const selModel  = document.getElementById('sel-model');
  const selVad    = document.getElementById('sel-vad');
  const payload   = { input_path: videoPath };
  if (selSrc && selSrc.value)       payload.source_language = selSrc.value;
  if (selTgt && selTgt.value)       payload.target_language = selTgt.value;
  // Model: skip the "New…" sentinel if somehow selected (shouldn't happen
  // — the change handler opens the modal first — but be defensive).
  if (selModel && selModel.value && selModel.value !== NEW_MODEL_SENTINEL) {
    payload.model = selModel.value;
  }
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
