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
  if (typeof updateButtonStates === 'function') updateButtonStates();
}

// ── In-player subtitle track (WebVTT blob) ────────────────
//
// The video element has a hidden <track> child.  We feed it a fresh
// WebVTT blob URL whenever the SRT data changes — load, Edit, Split,
// Merge, Delete.  Hook is _refreshSubtitleTrack(), called from
// renderLinks (chapters.js) so every mutation flows through it.
//
// Why a blob, not the on-disk .vtt?  Edits aren't persisted to disk
// until the user hits Save; pointing the track at <video>.vtt would
// show stale subtitles after every edit.  An in-memory VTT mirrors
// chaptersArr exactly, so the player always shows what's in the
// editor.
//
// VTT format mirrors gensrt/srt/builder.py write_vtt() — same
// HH:MM:SS.mmm timestamp format, no cue identifiers, blank line
// between cues.

let _subtitleBlobUrl = null;

function _formatVttTime(seconds) {
  const total = Math.max(0, Number(seconds) || 0);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total - (h * 3600) - (m * 60);
  // Pad seconds to 6 chars total (e.g. 05.500), with 3 decimal places.
  const sStr = s.toFixed(3);
  const sPadded = sStr.length < 6 ? '0'.repeat(6 - sStr.length) + sStr : sStr;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${sPadded}`;
}

function _buildVttFromSegments(segments) {
  if (!Array.isArray(segments) || !segments.length) return '';
  const lines = ['WEBVTT', ''];
  for (const seg of segments) {
    const start = Number(seg.start_time);
    const end   = Number(seg.end_time);
    if (!isFinite(start) || !isFinite(end) || !(end > start)) continue;
    // "-->" inside cue text breaks the WebVTT parser; replace with a
    // visually-similar Unicode arrow if it ever appears in transcribed
    // subtitle text (rare, but defensive).
    const text = String(seg.text || '').replace(/-->/g, '→→');
    lines.push(`${_formatVttTime(start)} --> ${_formatVttTime(end)}`);
    lines.push(text);
    lines.push('');
  }
  return lines.join('\n') + '\n';
}

function _refreshSubtitleTrack() {
  const trackEl = document.getElementById('subtitleTrack');
  const playerEl = document.getElementById('player');
  if (!trackEl || !playerEl) return;

  const segs = (typeof chaptersArr !== 'undefined' && Array.isArray(chaptersArr))
    ? chaptersArr
    : [];
  const vttText = _buildVttFromSegments(segs);

  // Always revoke the previous URL — even if we're about to create a new
  // one, the old one is no longer referenced by anything.
  if (_subtitleBlobUrl) {
    try { URL.revokeObjectURL(_subtitleBlobUrl); } catch {}
    _subtitleBlobUrl = null;
  }

  if (!vttText) {
    // Empty-state teardown that survives a subsequent src change.
    //
    //   1. mode='disabled' stops the browser from displaying any cue.
    //      This is what was missing in v1.1 and made New Project leak
    //      the previous video's last cue.
    //   2. removeCue() drains the parsed cue list — defensive, since
    //      some browser embeds leak the cues until the next src parse.
    //   3. We do NOT remove the src attribute.  In CEF/WebView2 (which
    //      pywebview uses), removing src from a <track> can leave the
    //      element in an "unloaded" state from which setting a new src
    //      doesn't re-trigger parsing — breaking the populated-after-
    //      empty transition that the post-transcribe flow needs.
    //      Leaving the (now-revoked) blob URL on src keeps the track
    //      element "loaded" enough that a future src reassignment is
    //      treated as a real source change.
    try {
      const tt = trackEl.track;
      if (tt) {
        tt.mode = 'disabled';
        if (tt.cues && tt.cues.length) {
          for (let i = tt.cues.length - 1; i >= 0; i--) {
            try { tt.removeCue(tt.cues[i]); } catch {}
          }
        }
      }
    } catch {}
    return;
  }

  const blob = new Blob([vttText], { type: 'text/vtt' });
  _subtitleBlobUrl = URL.createObjectURL(blob);

  // We replace the <track> element entirely instead of reassigning .src
  // on the existing one.  Reassigning works for short video-to-SRT gaps
  // (Drop SRT path, sub-second), but fails for long gaps (Generate path,
  // ~7 minutes): once the parent <video> reaches readyState 4
  // (HAVE_ENOUGH_DATA), CEF/WebView2 silently skips re-parsing track src
  // assignments — the cue list stays empty even though src, mode, and
  // readyState all look correct.
  //
  // Creating a fresh <track> element guarantees a fresh TextTrack object
  // with no "we already missed the window" state.  CEF treats it as a
  // first-time track addition and parses normally.
  //
  // We preserve the element's attributes (id, kind, srclang, label,
  // default) so any external code that looks it up by id still works.
  // The video element's textTracks list adds the new TextTrack; the old
  // one falls off when the old <track> is removed from DOM.
  try {
    const oldTrack = trackEl;
    const parent = oldTrack.parentNode;
    if (parent) {
      const fresh = document.createElement('track');
      // Copy attributes from the old track.  This is the minimal set
      // GenSRT actually uses; if the template grows more attributes,
      // they should be copied here too.
      fresh.id = oldTrack.id;
      fresh.kind = oldTrack.kind || 'subtitles';
      if (oldTrack.srclang) fresh.srclang = oldTrack.srclang;
      if (oldTrack.label) fresh.label = oldTrack.label;
      if (oldTrack.default) fresh.default = true;
      // Set src BEFORE inserting into the DOM so the browser parses it
      // as part of the initial track activation rather than as a later
      // src change.
      fresh.src = _subtitleBlobUrl;
      // Replace in-place to preserve sibling order.
      parent.replaceChild(fresh, oldTrack);
      // Update the cached reference so subsequent calls operate on the
      // new element.  trackEl is a const in this function, but the
      // outer-scope lookup uses getElementById which now resolves to
      // the fresh element by id.
      // Force mode='showing' on the new track.  Browsers default new
      // tracks to 'disabled' even with the `default` attribute when
      // src is set programmatically before DOM insertion.
      try {
        if (fresh.track) fresh.track.mode = 'showing';
      } catch {}
    } else {
      // Fallback (shouldn't happen — track element is always inside
      // the video element in our template): old-style src reassignment.
      trackEl.src = _subtitleBlobUrl;
      try {
        if (trackEl.track) trackEl.track.mode = 'showing';
      } catch {}
    }
  } catch (err) {
    console.error('_refreshSubtitleTrack: track replacement failed, falling back to src reassignment:', err);
    // Last-resort fallback so we don't break the player on unexpected DOM state.
    trackEl.src = _subtitleBlobUrl;
    try {
      if (trackEl.track) trackEl.track.mode = 'showing';
    } catch {}
  }
}

window._refreshSubtitleTrack = _refreshSubtitleTrack;

// When a new video loads, the <track> element re-attaches but the
// TextTrack mode often resets to 'disabled'.  Re-apply on loadedmetadata
// so cues continue to render after a video swap.
(function installTrackRefreshOnVideoLoad() {
  const playerEl = document.getElementById('player');
  if (!playerEl) return;
  playerEl.addEventListener('loadedmetadata', () => {
    // Defer to next tick so the browser finishes its own track reset
    // before we override mode='showing'.
    setTimeout(_refreshSubtitleTrack, 0);
  });
})();

// ── Load SRT from an absolute path (via the server) ───────
async function loadSrtFromPath(srtPath, options = {}) {
  const { quiet = false, skipSiblingVideo = false } = options;
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
    // Chain to sibling video if the server found one and it differs from
    // what's currently loaded.  Suppressed via skipSiblingVideo for callers
    // that have a reason not to (e.g. tests, or future sidecar flows that
    // already loaded the video first).
    if (!skipSiblingVideo && data.sibling_video) {
      const currentVideo = (typeof videoPathInput !== 'undefined' && videoPathInput
        ? (videoPathInput.value || '').trim()
        : '');
      if (data.sibling_video !== currentVideo) {
        // skipSidecar: the user chose THIS SRT explicitly (drag/drop or
        // Open).  Loading its sibling video must not then trigger sidecar
        // discovery, which tries <basename>.srt first and would silently
        // replace the file the user just asked for.  Dropping
        // MalayalamNews-2.ml.srt loaded the video and then swapped in
        // MalayalamNews-2.srt.
        window.tilesterSetVideoFromPath(data.sibling_video, { skipSidecar: true });
      }
    }
    return true;
  } catch (err) {
    console.error('loadSrtFromPath failed:', err);
    if (!quiet) showErrorDialog('SRT Load Failed', err.message || String(err));
    return false;
  }
}

window.gensrtLoadSrtFromPath = loadSrtFromPath;

// ── Load SRT for a given video (server-side sidecar discovery) ─
//
// Uses ``GET /api/srt?video=<path>`` so the server's discovery rules
// (prefer ``movie.srt`` over ``movie.<lang>.srt``, fall back to any
// language variant when no canonical track exists) apply.
//
// Used by the sidecar hook below — never by direct user action.
// 404 from the server means "no sidecar SRT for this video", which is
// the silent-no-op case (right pane stays empty until user generates).
async function loadSrtFromVideo(videoPath, options = {}) {
  const { quiet = false } = options;
  if (!videoPath) return false;
  try {
    const resp = await fetch(`/api/srt?video=${encodeURIComponent(videoPath)}`);
    if (!resp.ok) {
      // 404 → no sidecar exists.  Treat as silent miss when quiet.
      if (resp.status === 404) return false;
      if (!quiet) {
        const j = await resp.json().catch(() => ({}));
        showErrorDialog('SRT Load Failed', j.error || `HTTP ${resp.status}`);
      }
      return false;
    }
    const data = await resp.json();
    // Use the path the server actually picked (could be movie.srt OR
    // movie.ml.srt etc.) so Save round-trips back to the same file.
    _applySrtPayload(data, data.path || null);
    return true;
  } catch (err) {
    console.error('loadSrtFromVideo failed:', err);
    if (!quiet) showErrorDialog('SRT Load Failed', err.message || String(err));
    return false;
  }
}

window.gensrtLoadSrtFromVideo = loadSrtFromVideo;

// ── Browser-mode file drop (loadJSON's slot in player.js) ─
//
// Player.js (off-limits) calls window-side `loadJSON` from its file-picker
// change handler.  We keep the function name but:
//
//   • In pywebview mode, the File object exposes its absolute path via
//     ``file.pywebviewFullPath``.  We delegate to loadSrtFromPath which
//     handles the server round-trip AND chains to any sibling video the
//     server discovers.  Note that File objects from <input type="file">
//     elements DON'T expose pywebviewFullPath (only drag-drop events do),
//     so this branch only fires for the drop path.  The click-on-pane
//     path goes through select_srt() instead — see player.js.
//
//   • In browser mode (and for File objects without pywebviewFullPath),
//     we fall back to a naive client-side parse with no sibling-video
//     discovery — there's no path to ask the server about.
function loadJSON(file) {
  const fullPath = file && file.pywebviewFullPath;
  if (fullPath) {
    loadSrtFromPath(fullPath);
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
  // Second parameter is ours; player.js only ever passes the path, and an
  // unrecognised value degrades to "run discovery", which is the old
  // behaviour and the safe default.
  window.tilesterSetVideoFromPath = function (fullPath, opts) {
    const r = orig.call(this, fullPath);
    const skipSidecar = !!(opts && opts.skipSidecar);
    try {
      const p = String(fullPath || '');
      if (p && !skipSidecar) {
        // Delegate sidecar discovery to the server.  /api/srt?video=...
        // tries <basename>.srt first, then any <basename>.*.srt, so the
        // user sees the canonical track when present and a language
        // variant when that's all that exists.  Constructing the path
        // client-side would force <basename>.srt and miss the variants.
        loadSrtFromVideo(p, { quiet: true });
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
    if (typeof updateButtonStates === 'function') updateButtonStates();
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
    if (typeof updateButtonStates === 'function') updateButtonStates();
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
