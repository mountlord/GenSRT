// ── Config Editor ─────────────────────────────────────────
let currentConfig = null;

// GenSRT configuration schema.  Every field listed here must also appear in
// the server-side _CONFIG_VALIDATORS dict in server.py — saves of fields
// outside that allow-list are rejected.
//
// Special case: the "model" field is rendered as a dynamic dropdown
// populated from /api/known_models (see renderConfigEditor).  The
// `options` list below is just the built-in fallback for first paint
// before the API responds.
const configSchema = {
  'Transcription': {
    'model': {
      type: 'select-dynamic',
      source: 'known_models',
      options: ['tiny', 'base', 'small', 'medium', 'large',
                'large-v1', 'large-v2', 'large-v3', 'large-v3-turbo'],
      hint: 'Whisper model.  Built-in sizes plus any custom HF models you have added.  Pick "New…" to add a custom one.',
    },
    'source_language': {
      type: 'select',
      options: ['auto', 'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh',
                'ja', 'ko', 'hi', 'ml', 'ta', 'te', 'bn', 'ar', 'tr',
                'vi', 'th', 'id', 'ms', 'pl', 'nl', 'sv', 'da', 'no',
                'fi', 'cs', 'he', 'ur', 'fa'],
      hint: 'Default audio language code (ISO 639-1).  "auto" detects per-file.  Footer selector overrides per job.',
    },
    'device': {
      type: 'select',
      options: ['auto', 'cuda', 'cpu'],
      hint: 'Computation device.  "auto" picks GPU when available; "cpu" forces CPU even on a CUDA machine.',
    },
    'compute_type': {
      type: 'select',
      options: ['auto', 'float32', 'float16', 'int8_float16', 'int8'],
      hint: 'Numeric precision.  "auto" asks CTranslate2 what this device supports.  float16 is fastest on modern GPUs; int8 for slow hardware.',
    },
    'backend': {
      type: 'select',
      options: ['cuda', 'rocm', 'xpu', 'cpu'],
      hint: 'GPU acceleration backend.',
    },
    'gpu_id': {
      type: 'number', min: 0, max: 7, step: 1,
      hint: 'GPU device index (0 = first GPU).',
    },
  },
  'VAD & Subtitle Timing': {
    'vad_enabled': {
      type: 'checkbox',
      hint: 'Filter silence before transcription.  Recommended.',
    },
    'vad_threshold': {
      type: 'number', min: 0, max: 1, step: 0.05,
      hint: 'Voice detection sensitivity (0–1).  Lower = more permissive (more audio treated as speech).',
    },
    'vad_min_speech_ms': {
      type: 'number', min: 50, max: 10000, step: 50,
      hint: 'Minimum speech segment duration (ms).  Shorter blips are dropped.',
    },
    'vad_min_silence_ms': {
      type: 'number', min: 100, max: 10000, step: 100,
      hint: 'Minimum silence to break segments (ms).  Larger = longer subtitle lines.',
    },
    'vad_speech_pad_ms': {
      type: 'number', min: 0, max: 2000, step: 50,
      hint: 'Padding added around detected speech (ms).  Higher = subtitles linger; lower = tighter timing.',
    },
    'min_subtitle_duration_s': {
      type: 'number', min: 0, max: 60, step: 0.1,
      hint: 'Minimum SRT line length (seconds).',
    },
    'max_subtitle_duration_s': {
      type: 'number', min: 0, max: 60, step: 0.5,
      hint: 'Maximum SRT line length (seconds).  Long Whisper segments are split when they exceed this.',
    },
    'max_line_chars': {
      type: 'number', min: 0, max: 200, step: 1,
      hint: 'Characters per subtitle line before wrapping.  42 is the broadcast convention.  Text is never discarded — a long cue wraps onto more lines instead.',
    },
    'max_lines': {
      type: 'number', min: 1, max: 10, step: 1,
      hint: 'Preferred lines per subtitle.  A soft target: cues that need more lines get them rather than losing text.',
    },
  },
  'Translation': {
    'translate': {
      type: 'checkbox',
      hint: 'Translate transcript to the target language.',
    },
    'translation_engine': {
      type: 'select',
      options: ['google', 'none'],
      hint: 'Default translation backend.  Footer selector overrides per job.',
    },
    'target_language': {
      type: 'select',
      options: ['en', 'es', 'fr', 'de', 'pt', 'ru', 'zh', 'ja', 'ko',
                'ar', 'hi', 'ml', 'ta', 'te', 'bn', 'id', 'vi', 'th',
                'tr', 'pl'],
      hint: 'Target language code (ISO 639-1), e.g. en, ko, ja, hi. Translation is skipped when it matches the detected source language.',
    },
  },
  'System': {
    'output': {
      type: 'folder',
      hint: 'Default output directory.  Leave empty to write the SRT next to the source video.',
    },
    'log_level': {
      type: 'select',
      options: ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
      hint: 'Logging verbosity.',
    },
  },
};

function renderConfigEditor(config) {
  currentConfig            = config;
  configEditorBody.innerHTML = '';

  Object.keys(configSchema).forEach((sectionName, sectionIndex) => {
    const section    = configSchema[sectionName];
    const isExpanded = sectionIndex === 0;

    const sectionDiv = document.createElement('div');
    sectionDiv.className = `config-section ${isExpanded ? 'expanded' : 'collapsed'}`;

    const headerDiv = document.createElement('div');
    headerDiv.className = 'config-section-header';
    headerDiv.innerHTML = `
      <span class="expand-icon">▼</span>
      <span class="config-section-title">${sectionName}</span>
    `;

    const bodyDiv = document.createElement('div');
    bodyDiv.className = 'config-section-body';

    Object.keys(section).forEach(fieldKey => {
      const fieldDef   = section[fieldKey];
      const fieldValue = config[fieldKey];
      const fieldDiv   = document.createElement('div');
      fieldDiv.className = 'config-field';

      const label = document.createElement('div');
      label.className = 'config-field-label';
      label.innerHTML = `
        ${fieldKey.replace(/_/g, ' ')}
        ${fieldDef.hint ? `<span class="hint">${fieldDef.hint}</span>` : ''}
      `;

      let input;
      if (fieldDef.type === 'checkbox') {
        input         = document.createElement('input');
        input.type    = 'checkbox';
        input.checked = !!fieldValue;
        input.className   = 'config-field-input';
        input.dataset.key = fieldKey;
      } else if (fieldDef.type === 'select-dynamic' && fieldDef.source === 'known_models') {
        // Dynamic model dropdown — populated from window.__gensrtKnownModels
        // (cached by api.js footer init) or the built-in fallback below.
        // Always includes the saved value as an option even if it's not in
        // the known list, so manual config.json edits aren't silently
        // overwritten when the user clicks Save.
        input             = document.createElement('select');
        input.className   = 'config-field-input';
        input.dataset.key = fieldKey;
        const knownModels = (window.__gensrtKnownModels && window.__gensrtKnownModels.length)
          ? window.__gensrtKnownModels
          : (fieldDef.options || []);
        const seen = new Set();
        for (const m of knownModels) {
          if (seen.has(m)) continue;
          seen.add(m);
          const opt        = document.createElement('option');
          opt.value        = m;
          opt.textContent  = m;
          if (m === fieldValue) opt.selected = true;
          input.appendChild(opt);
        }
        // Rescue: saved value isn't in the dropdown (manual edit, deleted
        // model, etc.) — add it at the top so Save round-trips correctly.
        if (fieldValue && !seen.has(fieldValue)) {
          const opt        = document.createElement('option');
          opt.value        = fieldValue;
          opt.textContent  = fieldValue + ' (from config)';
          opt.selected     = true;
          input.insertBefore(opt, input.firstChild);
        }
        // Separator and New… sentinel.
        const sep         = document.createElement('option');
        sep.disabled      = true;
        sep.textContent   = '──────────';
        input.appendChild(sep);
        const neu         = document.createElement('option');
        neu.value         = '__new_model__';
        neu.textContent   = 'New…';
        input.appendChild(neu);
        // Selecting New… opens the same Add Custom Model modal as the
        // footer.  api.js owns the modal — we reuse it here.
        input.addEventListener('change', (ev) => {
          if (ev.target.value !== '__new_model__') return;
          if (typeof _openAddModelModal === 'function') {
            _openAddModelModal();
          }
        });
      } else if (fieldDef.type === 'select') {
        input             = document.createElement('select');
        input.className   = 'config-field-input';
        input.dataset.key = fieldKey;
        fieldDef.options.forEach(opt => {
          const option      = document.createElement('option');
          option.value      = opt;
          option.textContent = opt;
          if (opt === fieldValue) option.selected = true;
          input.appendChild(option);
        });
      } else if (fieldDef.type === 'array') {
        input             = document.createElement('input');
        input.type        = 'text';
        input.className   = 'config-field-input array';
        input.value       = Array.isArray(fieldValue) ? JSON.stringify(fieldValue) : '';
        input.placeholder = '[3, 2, 4]';
        input.dataset.key = fieldKey;
      } else if (fieldDef.type === 'folder') {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'display:flex; gap:6px; align-items:center; width:100%;';
        input             = document.createElement('input');
        input.type        = 'text';
        input.className   = 'config-field-input';
        input.style.flex  = '1';
        input.value       = fieldValue !== null && fieldValue !== undefined ? fieldValue : '';
        input.placeholder = '(same folder as video)';
        input.dataset.key = fieldKey;
        wrapper.appendChild(input);
        if (_tilesterDetectPyWebView()) {
          const browseBtn2 = document.createElement('button');
          browseBtn2.textContent = 'Browse…';
          browseBtn2.className   = 'modal-btn';
          browseBtn2.style.cssText = 'padding:3px 10px; font-size:12px; flex-shrink:0;';
          browseBtn2.addEventListener('click', async () => {
            try {
              const chosen = await window.pywebview.api.select_folder();
              if (chosen) input.value = chosen;
            } catch (e) { console.warn('select_folder failed:', e); }
          });
          wrapper.appendChild(browseBtn2);
        }
        fieldDiv.appendChild(label);
        fieldDiv.appendChild(wrapper);
        bodyDiv.appendChild(fieldDiv);
        return; // already appended
      } else {
        input             = document.createElement('input');
        input.type        = 'number';
        input.className   = 'config-field-input';
        input.value       = fieldValue !== null && fieldValue !== undefined ? fieldValue : '';
        input.step        = fieldDef.step || 1;
        if (fieldDef.min !== undefined) input.min = fieldDef.min;
        if (fieldDef.max !== undefined) input.max = fieldDef.max;
        input.dataset.key = fieldKey;
      }

      fieldDiv.appendChild(label);
      fieldDiv.appendChild(input);
      bodyDiv.appendChild(fieldDiv);
    });

    headerDiv.addEventListener('click', () => {
      sectionDiv.classList.toggle('expanded');
      sectionDiv.classList.toggle('collapsed');
    });

    sectionDiv.appendChild(headerDiv);
    sectionDiv.appendChild(bodyDiv);
    configEditorBody.appendChild(sectionDiv);
  });
}

function collectConfigFromUI() {
  const config = { ...currentConfig };

  // Strip keys that aren't in our schema and aren't on the small allow-list
  // of valid-but-not-surfaced keys.  This protects users transitioning from
  // older configs (e.g., tilester schema) from save failures — the server's
  // validator rejects unknown keys, so anything stale here would 400.
  const validKeys = new Set([
    'output_filename', 'recurse',  // valid for /api/config but not exposed in UI
  ]);
  Object.values(configSchema).forEach(section => {
    Object.keys(section).forEach(k => validKeys.add(k));
  });
  for (const k of Object.keys(config)) {
    if (!validKeys.has(k)) delete config[k];
  }

  // Overlay current UI values.
  const inputs = configEditorBody.querySelectorAll('.config-field-input');
  inputs.forEach(input => {
    const key = input.dataset.key;
    if (input.type === 'checkbox') {
      config[key] = input.checked;
    } else if (input.classList.contains('array')) {
      try { config[key] = JSON.parse(input.value); } catch (e) { console.warn(`Invalid array for ${key}:`, input.value); }
    } else if (input.type === 'number') {
      config[key] = input.value === '' ? null : parseFloat(input.value);
    } else {
      // Special case: skip the model dropdown's "New…" sentinel — saving
      // that string would persist garbage to gensrt-config.json.  The
      // sentinel only opens the modal; the real model name lands here
      // after the modal completes and the dropdown is rebuilt.
      if (key === 'model' && input.value === '__new_model__') {
        return; // keep currentConfig.model unchanged
      }
      config[key] = input.value.trim() === '' ? null : input.value.trim();
    }
  });
  return config;
}

function applyConfigToUI(config) {
  if (config.fskip  !== undefined && config.fskip  !== null) skipForward.value  = config.fskip;
  if (config.fbskip !== undefined && config.fbskip !== null) skipBackward.value = config.fbskip;
}

function formatServerError(result) {
  if (!result || typeof result !== 'object') return String(result ?? 'Unknown error');
  const parts = [];
  if (result.message) parts.push(String(result.message));
  if (result.detail)  parts.push(String(result.detail));
  if (result.path) {
    let loc = String(result.path);
    if (result.line   != null) loc += `:${result.line}`;
    if (result.column != null) loc += `:${result.column}`;
    parts.push(loc);
  }
  if (parts.length === 0) parts.push(JSON.stringify(result, null, 2));
  return parts.join('\n');
}

async function loadConfigFromServer() {
  try {
    configLoad.disabled    = true;
    configLoad.textContent = 'Loading...';
    const response = await fetch('/api/config', { method: 'GET' });
    const result   = await response.json();
    if (result.status === 'success') {
      renderConfigEditor(result.config);
      applyConfigToUI(result.config);
      showInfoDialog('Configuration', 'Configuration loaded successfully.');
    } else {
      showErrorDialog('Configuration', `Failed to load config:<br><pre style="text-align:left; white-space:pre-wrap;">${formatServerError(result)}</pre>`);
    }
  } catch (error) {
    showErrorDialog('Configuration', `Failed to load config:<br><pre style="text-align:left; white-space:pre-wrap;">${String(error?.message || error)}</pre>`);
  } finally {
    configLoad.disabled    = false;
    configLoad.textContent = 'Load';
  }
}

async function saveConfigToServer() {
  try {
    const config = collectConfigFromUI();
    configSave.disabled    = true;
    configSave.textContent = 'Saving...';
    const response = await fetch('/api/config', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(config)
    });
    const result = await response.json();
    if (result.status === 'success') {
      const reloadResp   = await fetch('/api/config', { method: 'GET' });
      const reloadResult = await reloadResp.json();
      if (reloadResult.status === 'success') {
        renderConfigEditor(reloadResult.config);
        applyConfigToUI(reloadResult.config);
      }
      // Sync the footer selectors (source language, translation engine, VAD)
      // to the freshly-saved defaults so the next Generate SRT picks them up
      // without requiring a page refresh.
      if (typeof _initFooterSelectors === 'function') {
        try { await _initFooterSelectors(); }
        catch (e) { console.warn('Footer re-init after save failed:', e); }
      }
      showInfoDialog('Configuration', 'Configuration saved. Next transcription will use these settings.');
      configEditorModal.classList.remove('visible');
    } else {
      showErrorDialog('Configuration', `Failed to save config:<br><pre style="text-align:left; white-space:pre-wrap;">${formatServerError(result)}</pre>`);
    }
  } catch (error) {
    showErrorDialog('Configuration', `Failed to save config:<br><pre style="text-align:left; white-space:pre-wrap;">${String(error?.message || error)}</pre>`);
  } finally {
    configSave.disabled    = false;
    configSave.textContent = 'Save';
  }
}

configBtn.addEventListener('click', async () => {
  try {
    const response = await fetch('/api/config', { method: 'GET' });
    const result   = await response.json();
    if (result.status === 'success') {
      renderConfigEditor(result.config);
      configEditorModal.classList.add('visible');
    } else {
      showErrorDialog('Configuration', `Failed to load config:<br><pre style="text-align:left; white-space:pre-wrap;">${formatServerError(result)}</pre>`);
    }
  } catch (error) {
    showErrorDialog('Configuration', `Failed to load config:<br><pre style="text-align:left; white-space:pre-wrap;">${String(error?.message || error)}</pre>`);
  }
});

configCancel.addEventListener('click', () => configEditorModal.classList.remove('visible'));
configLoad.addEventListener('click', loadConfigFromServer);
configSave.addEventListener('click', saveConfigToServer);
configEditorModal.addEventListener('click', (e) => {
  if (e.target === configEditorModal) configEditorModal.classList.remove('visible');
});

// When a new model is added via the Add Custom Model modal (whether opened
// from the footer or from inside the Config modal), refresh the Config
// modal's model dropdown so the new entry appears and is selected.
window.addEventListener('gensrt:known_models_updated', (ev) => {
  if (!configEditorModal.classList.contains('visible')) return;
  if (!currentConfig) return;
  const newName = (ev && ev.detail && ev.detail.selected) ? ev.detail.selected : null;
  if (newName) currentConfig.model = newName;
  renderConfigEditor(currentConfig);
});
