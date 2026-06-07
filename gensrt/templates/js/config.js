// ── Config Editor ─────────────────────────────────────────
let currentConfig = null;

const configSchema = {
  'Detection': {
    'detection_threads':           { type: 'number',   hint: 'Parallel decode workers (1 = single-threaded, 2-4 for long files)' },
    'sample_interval':             { type: 'number',   hint: 'Analyze every N frames' },
    'tile_similarity_threshold':   { type: 'number',   step: 0.01, hint: 'Cosine similarity threshold (0–1)' },
    'tile_early_exit_margin':      { type: 'number',   step: 0.01, hint: 'Early-exit margin above threshold' },
    'tile_pad_before_seconds':     { type: 'number',   step: 0.1,  hint: 'Expand TILED chapters earlier by this many seconds (0 = off)' },
    'tile_pad_after_seconds':      { type: 'number',   step: 0.1,  hint: 'Expand TILED chapters later by this many seconds (0 = off)' },
    'merge_full_gaps_max_seconds': { type: 'number',   step: 0.5,  hint: 'Merge FULL gaps shorter than this after padding (0 = off)' },
    'min_stable_enter_seconds':    { type: 'number',   step: 0.1,  hint: 'Seconds of evidence required to enter TILED' },
    'min_stable_exit_seconds':     { type: 'number',   step: 0.1,  hint: 'Seconds of evidence required to exit TILED' },
    'min_stable_samples':          { type: 'number',              hint: 'Minimum consecutive samples to commit a change' },
  },
  'System': {
    'out_dir':   { type: 'folder',   hint: 'Output directory (empty = same folder as video)' },
    'debug':     { type: 'checkbox', hint: 'Write debug CSV alongside each video' },
    'fbskip':    { type: 'number',   step: 0.5, hint: 'Skip Backward button duration (seconds)' },
    'fskip':     { type: 'number',   step: 0.5, hint: 'Skip Forward button duration (seconds)' },
    'gpu_id':    { type: 'number',              hint: 'GPU device index (0 = first GPU)' },
    'log_level': { type: 'select',   options: ['DEBUG', 'INFO', 'WARNING', 'ERROR'], hint: 'Logging verbosity' },
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
      body:    JSON.stringify({ config })
    });
    const result = await response.json();
    if (result.status === 'success') {
      const reloadResp   = await fetch('/api/config', { method: 'GET' });
      const reloadResult = await reloadResp.json();
      if (reloadResult.status === 'success') {
        renderConfigEditor(reloadResult.config);
        applyConfigToUI(reloadResult.config);
      }
      showInfoDialog('Configuration', 'Configuration saved. Next detection will use these settings.');
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
