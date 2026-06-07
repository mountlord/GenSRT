// ── Normalize Project Data ────────────────────────────────
// Kept for backwards-compat with any code path that still calls it.
function normalizeProjectData(data) {
  try {
    if (!data || typeof data !== 'object') return { schema_version: 1, segments: [] };
    if ((data.schema_version === 1 || data.schema_version === '1') && Array.isArray(data.segments)) return data;
    if (Array.isArray(data.chapters)) {
      const out = { ...data, schema_version: 1, segments: data.chapters };
      delete out.chapters;
      delete out.transitions;
      return out;
    }
    return { ...data, schema_version: 1, segments: Array.isArray(data.segments) ? data.segments : [] };
  } catch {
    return { schema_version: 1, segments: [] };
  }
}

// ── Apply segments returned from /api/srt to the UI ───────
//
// Single source of truth for rendering an SRT into the right-pane list.
// Used by:
//   • loadSrtFromPath()        (sidecar auto-discover + drag/drop)
//   • the post-transcribe completion handler in api.js
//   • the legacy browser-mode loadJSON path (renamed below to loadSRT)
function _applySrtPayload(payload, srtPath) {
  const segments = (payload && Array.isArray(payload.segments)) ? payload.segments : [];
  // The right-pane renderer historically reads from `data.chapters`, so we
  // shape the payload to match while populating the canonical chaptersArr.
  const proj = { schema_version: 1, segments: segments };
  linksData          = proj;
  currentProjectPath = srtPath || (payload && payload.path) || null;

  renderLinks({ chapters: segments });
  try { jsonDrop.style.display    = 'none'; } catch {}
  try { navContainer.style.display = 'block'; } catch {}
}

// ── Load SRT from an absolute path (via the server) ───────
async function loadSrtFromPath(srtPath, options = {}) {
  const { quiet = false } = options;
  if (!srtPath) return false;
  try {
    const resp = await fetch(`/api/srt?path=${encodeURIComponent(srtPath)}`);
    if (!resp.ok) {
      if (!quiet) {
        const j = await resp.json().catch(() => ({}));
        showErrorDialog('SRT Load Failed', j.error || `HTTP ${resp.status}`);
      }
      return false;
    }
    const data = await resp.json();
    _applySrtPayload(data, srtPath);
    return true;
  } catch (err) {
    console.error('loadSrtFromPath failed:', err);
    if (!quiet) showErrorDialog('SRT Load Failed', err.message || String(err));
    return false;
  }
}

window.gensrtLoadSrtFromPath = loadSrtFromPath;

// ── Browser-mode file drop (loadJSON's slot in player.js) ─
//
// Player.js (off-limits) calls window-side `loadJSON` from its file-picker
// change handler.  We keep the function name but:
//
//   • In pywebview mode, the File object exposes its absolute path via
//     ``file.pywebviewFullPath``.  We route through the server (/api/srt),
//     which also returns any sibling video path.  If a sibling video is
//     present and different from what's currently loaded, we set it via
//     window.tilesterSetVideoFromPath — the sidecar hook then re-applies
//     this SRT.  (The minor cost of loading the SRT twice is the price of
//     a single straight-through code path; user-initiated file pick is
//     rare enough that the extra round-trip is fine.)
//
//   • In browser mode, ``pywebviewFullPath`` is undefined and we have no
//     way to ask the server about the file's location.  Fall back to the
//     client-side parse that's been there since Drop H.
function loadJSON(file) {
  const fullPath = file && file.pywebviewFullPath;
  if (fullPath) {
    // pywebview path-aware load — uses /api/srt and picks up any sibling video.
    (async () => {
      try {
        const resp = await fetch(`/api/srt?path=${encodeURIComponent(fullPath)}`);
        if (!resp.ok) {
          const j = await resp.json().catch(() => ({}));
          showErrorDialog('SRT Load Failed', j.error || `HTTP ${resp.status}`);
          return;
        }
        const data = await resp.json();
        _applySrtPayload(data, fullPath);
        // Chain to sibling video if found and not already loaded.
        if (data.sibling_video) {
          const currentVideo = (videoPathInput && videoPathInput.value || '').trim();
          if (data.sibling_video !== currentVideo) {
            window.tilesterSetVideoFromPath(data.sibling_video);
          }
        }
      } catch (err) {
        console.error('loadJSON (pywebview path mode) failed:', err);
        showErrorDialog('SRT Load Failed', err.message || String(err));
      }
    })();
    return;
  }

  // Browser-mode fallback — naive client-side parse, no sibling-video
  // discovery (no path to ask the server about).
  const reader = new FileReader();
  reader.onload = async (e) => {
    try {
      const text = String(e.target.result || '');
      // Quick sanity check: SRT files have "-->" timestamp arrows.
      if (!text.includes('-->')) {
        showErrorDialog('Not an SRT file',
          `<strong>${file.name}</strong> doesn't look like an SRT subtitle file.<br><br>` +
          `Expected lines containing <code>HH:MM:SS,mmm --> HH:MM:SS,mmm</code>.`);
        return;
      }
      // Naive client-side parse — server-side parse requires a path we don't
      // have for browser-mode drops.  Sufficient for the fallback path.
      const blocks = text.replace(/\r/g, '').split(/\n\n+/);
      const segments = [];
      const parseT = (s) => {
        const m = /^(\d+):(\d+):(\d+)[,\.](\d+)$/.exec(s.trim());
        if (!m) return NaN;
        return (+m[1]) * 3600 + (+m[2]) * 60 + (+m[3]) + (+m[4]) / 1000;
      };
      for (const blk of blocks) {
        const lines = blk.split('\n').filter(l => l.trim().length);
        if (lines.length < 2) continue;
        // Optional index line; timestamp line is the one with "-->".
        const tsIdx = lines.findIndex(l => l.includes('-->'));
        if (tsIdx < 0) continue;
        const [a, b] = lines[tsIdx].split('-->').map(s => s.trim());
        const start  = parseT(a);
        const end    = parseT(b);
        if (!isFinite(start) || !isFinite(end)) continue;
        const content = lines.slice(tsIdx + 1).join('\n');
        segments.push({ index: segments.length + 1, start_time: start, end_time: end, text: content });
      }
      if (!segments.length) {
        showErrorDialog('Empty SRT', `${file.name} contained no parseable subtitle blocks.`);
        return;
      }
      _applySrtPayload({ segments, path: null }, null);
    } catch (err) {
      console.error('SRT parse error:', err);
      showErrorDialog('SRT Parse Failed', err.message || String(err));
    }
  };
  reader.readAsText(file);
}

// ── Sidecar auto-discovery on video load ──────────────────
//
// player.js (off-limits) exposes `tilesterSetVideoFromPath` on `window` and
// internally chains tilesterSetVideoPath → tilesterSetVideoFromPath.  We
// wrap the former to also try loading a matching <basename>.srt.  Wrapping
// is non-invasive — we call the original first, then opportunistically
// fetch the sidecar SRT (quiet on miss).
(function installSidecarHook() {
  const orig = window.tilesterSetVideoFromPath;
  if (typeof orig !== 'function') return;
  window.tilesterSetVideoFromPath = function (fullPath) {
    const r = orig.call(this, fullPath);
    try {
      const p = String(fullPath || '');
      if (p) {
        const srtPath = p.replace(/\.[^/.\\]+$/, '.srt');
        if (srtPath && srtPath !== p) {
          loadSrtFromPath(srtPath, { quiet: true });
        }
      }
    } catch (e) {
      console.warn('Sidecar SRT lookup failed:', e);
    }
    return r;
  };
  // Keep the simple wrapper in sync (player.js exposes both names).
  if (typeof window.tilesterSetVideoPath === 'function') {
    window.tilesterSetVideoPath = window.tilesterSetVideoFromPath;
  }
})();

// (legacy export name retained for symmetry with the old code path)
window.tilesterApplyLinksJson = function () { /* no-op — Drop H replaces JSON model */ };

// ── Modal keystroke isolation ─────────────────────────────
//
// While any modal is open the user needs to type freely in its text fields
// without firing the page's global keyboard shortcuts (Space → play/pause,
// arrow keys → seek, F → fullscreen, etc.).  Those global handlers live in
// off-limits files (init.js / ui.js / player.js) and listen at the document
// level, so we can't modify them directly.  Instead we attach a bubble-phase
// listener to each modal that stops keydown events from propagating past
// the modal element — they reach the modal's own inputs and buttons, then
// stop short of the global handlers.
//
// Escape is intentionally exempted so future "Escape closes the modal"
// behaviour (if added at the document level) keeps working.
(function installModalKeystrokeIsolation() {
  document.querySelectorAll('.modal-overlay').forEach((modal) => {
    modal.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') return;
      e.stopPropagation();
    });
  });
})();

// ── Build Project for Save ────────────────────────────────
function _buildProjectForSave() {
  const proj = _ensureEditableProject();
  if (Array.isArray(chaptersArr)) {
    const segs = chaptersArr.map(ch => ({ ...ch }));
    _reindexSegments(segs);
    proj.segments = segs;
  } else if (!Array.isArray(proj.segments)) {
    proj.segments = [];
  }
  if ((!proj.fps || !isFinite(proj.fps)) && fps) proj.fps = fps;
  if ((!proj.duration || !isFinite(proj.duration)) && player && isFinite(player.duration)) {
    proj.duration = Number(player.duration);
  }
  return proj;
}

function _defaultProjectFilename(videoPath) {
  const base  = basenameFromPath(videoPath || '');
  const noExt = base.replace(/\.[^/.]+$/, '');
  return `${noExt}.srt`;
}

// Map the editable chaptersArr into the {start_time, end_time, text} shape
// the /api/srt POST endpoint expects.  Filters out anything without a usable
// time range — the server will reject invalid rows anyway, but failing fast
// in the UI gives a clearer error.
function _buildSrtSegments() {
  if (!Array.isArray(chaptersArr)) return [];
  const out = [];
  for (const ch of chaptersArr) {
    const s = Number(ch.start_time);
    const e = Number(ch.end_time);
    if (!isFinite(s) || !isFinite(e) || e <= s) continue;
    out.push({
      start_time: s,
      end_time:   e,
      text:       String(ch.text || ch.title || '').trim() || ' ',
    });
  }
  return out;
}

// Derive the SRT destination path for plain Save.  Use the path from the
// last load/save if we have it; otherwise place the SRT next to the video.
function _deriveSaveTargetPath(videoPath) {
  if (currentProjectPath) return currentProjectPath;
  if (!videoPath) return null;
  const sep   = videoPath.includes('\\') ? '\\' : '/';
  const idx   = videoPath.lastIndexOf(sep);
  const dir   = idx >= 0 ? videoPath.slice(0, idx + 1) : '';
  return dir + _defaultProjectFilename(videoPath);
}

// ── Save SRT ──────────────────────────────────────────────
async function saveProject() {
  if (!isServerMode) {
    showErrorDialog('Save Unavailable', 'Save requires <strong>server mode</strong> (Flask running).');
    return;
  }

  const videoPath = normalizeFullPath(videoPathInput.value.trim());
  if (!videoPath) {
    showErrorDialog('Invalid Path', 'Please load a video first.');
    return;
  }
  if (!videoPath.includes('\\') && !videoPath.includes('/')) {
    showErrorDialog('Invalid Path', `Invalid path: <span style="font-family: var(--font-mono);">${videoPath}</span><br><br>` +
      'Please enter the <strong>FULL</strong> path, not just the filename.');
    return;
  }

  const targetPath = _deriveSaveTargetPath(videoPath);
  if (!targetPath) {
    showErrorDialog('Save Failed', 'Could not determine where to save the SRT.');
    return;
  }

  const segments = _buildSrtSegments();
  if (!segments.length) {
    showErrorDialog('Nothing to Save', 'No valid SRT lines to write.');
    return;
  }

  try {
    const resp = await fetch('/api/srt', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ path: targetPath, segments }),
    });
    const j = await resp.json();
    if (!resp.ok || j.status !== 'ok') throw new Error(j && j.message ? j.message : `HTTP ${resp.status}`);

    if (j.path) currentProjectPath = normalizeFullPath(j.path);
    showProgressSuccess('Saved',
      `Saved: <span style="font-family: var(--font-mono);">${j.path}</span><br>` +
      `<small>${j.count} segment(s)</small>`);
  } catch (e) {
    console.error('saveProject failed:', e);
    showErrorDialog('Save Failed', e.message || String(e));
  }
}

async function saveProjectAs() {
  if (!isServerMode) {
    showErrorDialog('Save As Unavailable', 'Save As requires <strong>server mode</strong> (Flask running).');
    return;
  }

  const videoPath = normalizeFullPath(videoPathInput.value.trim());
  if (!videoPath) {
    showErrorDialog('Invalid Path', 'Please load a video first.');
    return;
  }
  if (!videoPath.includes('\\') && !videoPath.includes('/')) {
    showErrorDialog('Invalid Path', `Invalid path: <span style="font-family: var(--font-mono);">${videoPath}</span><br><br>` +
      'Please enter the <strong>FULL</strong> path, not just the filename.');
    return;
  }

  const defName     = _defaultProjectFilename(videoPath);
  const defaultDir  = (() => {
    const sep = videoPath.includes('\\') ? '\\' : '/';
    const idx = videoPath.lastIndexOf(sep);
    return idx >= 0 ? videoPath.slice(0, idx) : '';
  })();
  let pickedPath = null;

  if (typeof window.pywebview !== 'undefined' && window.pywebview.api && typeof window.pywebview.api.save_srt_as === 'function') {
    try {
      pickedPath = await window.pywebview.api.save_srt_as(defName, defaultDir);
    } catch (e) {
      console.warn('pywebview Save As dialog failed; falling back to prompt:', e);
      pickedPath = null;
    }
    if (!pickedPath) return;  // user cancelled
  }

  if (!pickedPath) {
    // Browser-mode fallback: ask for a filename, place beside the video.
    const name = prompt('Save As filename (same folder as the video):', defName);
    if (!name) return;
    const sep = videoPath.includes('\\') ? '\\' : '/';
    const idx = videoPath.lastIndexOf(sep);
    const dir = idx >= 0 ? videoPath.slice(0, idx + 1) : '';
    pickedPath = dir + name;
  }

  // Ensure .srt suffix — the server will reject otherwise.
  if (!pickedPath.toLowerCase().endsWith('.srt')) pickedPath += '.srt';

  const segments = _buildSrtSegments();
  if (!segments.length) {
    showErrorDialog('Nothing to Save', 'No valid SRT lines to write.');
    return;
  }

  try {
    const resp = await fetch('/api/srt', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ path: pickedPath, segments }),
    });
    const j = await resp.json();
    if (!resp.ok || j.status !== 'ok') throw new Error(j && j.message ? j.message : `HTTP ${resp.status}`);

    if (j.path) currentProjectPath = normalizeFullPath(j.path);
    showProgressSuccess('Saved',
      `Saved: <span style="font-family: var(--font-mono);">${j.path}</span><br>` +
      `<small>${j.count} segment(s)</small>`);
  } catch (e) {
    console.error('saveProjectAs failed:', e);
    showErrorDialog('Save As Failed', e.message || String(e));
  }
}

if (saveBtn)   saveBtn.addEventListener('click', saveProject);
if (saveAsBtn) saveAsBtn.addEventListener('click', saveProjectAs);

// ── Reset Project ─────────────────────────────────────────
function resetProject() {
  try {
    try { progressModal.classList.remove('visible'); }    catch {}
    try { chapterEditorModal.classList.remove('visible'); } catch {}
    try { configEditorModal.classList.remove('visible'); }  catch {}

    linksData          = null;
    fps                = null;
    transitionsArr     = [];
    chaptersArr        = [];
    chapterSelections  = [];
    transitionEls      = [];
    chapterEls         = [];
    activeTransitionEl = null;
    activeChapterEl    = null;
    editingChapterIndex = null;
    currentProjectPath  = null;
    currentFullVideoPath = null;

    try { renderLinks({ chapters: [] }); } catch {}
    try { if (videoControls) videoControls.classList.remove('video-ready'); } catch {}
    try { jsonDrop.style.display   = 'block'; }  catch {}
    try { navContainer.style.display = 'none'; } catch {}
    try { if (fpsDisplay) fpsDisplay.innerHTML = '<span class="box-value">—/—</span>'; } catch {}
    try { linkCount.textContent = '0'; } catch {}

    try { player.pause(); } catch {}
    try { player.removeAttribute('src'); player.load(); } catch {}
    try { player.style.display = 'none'; } catch {}
    try { if (videoDrop) videoDrop.style.display = 'block'; } catch {}
    try { if (videoName) videoName.textContent = 'Drop a video or click to browse'; } catch {}

    try { videoPathInput.value = ''; } catch {}
    videoFilename = '';
    updateButtonStates();
  } catch (e) {
    console.error('resetProject failed:', e);
    showErrorDialog('Reset Failed', e.message || String(e));
  }
}

resetBtn.addEventListener('click', resetProject);
