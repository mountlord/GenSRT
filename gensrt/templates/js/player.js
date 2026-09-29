// ── pywebview Integration ─────────────────────────────────
// pywebview injects `window.pywebview` asynchronously. Treat it as "not ready"
// until we see it (or we receive the `pywebviewready` event).
let isPyWebView = false;
let _pvwIsFullscreen = false; // tracks native pywebview fullscreen state

window.__tilesterPyWebViewReady = false;

function _tilesterDetectPyWebView() {
  return (typeof window.pywebview !== 'undefined') && window.pywebview && window.pywebview.api;
}

function _tilesterMarkPyWebViewReady() {
  isPyWebView = true;
  window.__tilesterPyWebViewReady = true;
  try {
    if (typeof window.__tilesterOnPyWebViewReady === 'function') {
      window.__tilesterOnPyWebViewReady();
    }
  } catch (e) {
    console.warn('__tilesterOnPyWebViewReady failed:', e);
  }
}

if (_tilesterDetectPyWebView()) {
  _tilesterMarkPyWebViewReady();
  console.log('Running in native pywebview window');
} else {
  console.log('Running in browser mode (pywebview not ready yet)');
}

window.addEventListener('pywebviewready', () => {
  _tilesterMarkPyWebViewReady();
  console.log('pywebviewready: native window APIs available');
});

// Legacy hook — delegates to the canonical setter below.
// NOTE: We intentionally avoid file:/// playback because it breaks seeking
// in some WebView2 configurations and fails with certain unicode paths.
window.tilesterLoadVideoFromPath = function(path) {
  try {
    if (window.tilesterSetVideoFromPath) {
      window.tilesterSetVideoFromPath(path);
      return;
    }
    const videoPathInput = document.getElementById('videoPathInput');
    if (videoPathInput) videoPathInput.value = path || '';
  } catch (e) {
    console.warn('tilesterLoadVideoFromPath failed:', e);
  }
};

// ── FPS Detection ─────────────────────────────────────────
function tryDetectFpsFromPlayer() {
  // Best effort: ask the HTML player via captureStream() track settings.
  // Use this only to populate NOMINAL FPS (player-reported).
  // Effective FPS should come from ffprobe (avg_frame_rate).
  try {
    if (player && typeof player.captureStream === 'function') {
      const stream  = player.captureStream();
      const tracks  = stream.getVideoTracks ? stream.getVideoTracks() : [];
      if (tracks && tracks.length) {
        const settings = tracks[0].getSettings ? tracks[0].getSettings() : {};
        const fr = settings && settings.frameRate ? Number(settings.frameRate) : null;
        try { tracks[0].stop(); } catch {}
        if (fr && isFinite(fr) && fr > 0) {
          if (!fpsNominal || !isFinite(fpsNominal) || fpsNominal <= 0) fpsNominal = fr;
          if (fpsDisplay) fpsDisplay.innerHTML = `<span class="box-value">${fmtFpsPair()}</span>`;
          return fps;
        }
      }
    }
  } catch (e) { /* ignore */ }
  return fps;
}

// Prefer server-probed nominal FPS (ffprobe r_frame_rate) when available.
async function tryFetchNominalFpsFromServer(fullPath) {
  const p = normalizeFullPath(fullPath);
  if (!p) return;

  const seq = ++_videoInfoSeq;
  try {
    const resp = await fetch(`/api/video_info?path=${encodeURIComponent(p)}`);
    if (!resp.ok) return;
    const data = await resp.json();
    if (seq !== _videoInfoSeq) return;

    const eff = (data && typeof data.avg_fps === 'number') ? data.avg_fps : null;
    if (eff && isFinite(eff) && eff > 0) fps = eff;

    const nom = (data && typeof data.r_fps === 'number') ? data.r_fps : null;
    if (nom && isFinite(nom) && nom > 0) fpsNominal = nom;

    if (fpsDisplay) fpsDisplay.innerHTML = `<span class="box-value">${fmtFpsPair()}</span>`;
  } catch (e) { /* ignore */ }
}

// ── Go-to Frame / Time ────────────────────────────────────
function parseTime(str) {
  str = str.trim();
  if (!str) return NaN;
  if (/^\d+(\.\d+)?$/.test(str)) return parseFloat(str);
  const parts = str.split(':');
  if (parts.length < 2 || parts.length > 3) return NaN;
  let h = 0, m = 0, s = 0;
  if (parts.length === 3) {
    h = parseInt(parts[0], 10); m = parseInt(parts[1], 10); s = parseFloat(parts[2]);
  } else {
    m = parseInt(parts[0], 10); s = parseFloat(parts[1]);
  }
  if (isNaN(h) || isNaN(m) || isNaN(s)) return NaN;
  return h * 3600 + m * 60 + s;
}

function seekToFrame() {
  const f = parseInt(gotoFrame.value, 10);
  if (!fps || isNaN(f) || f < 0) return;
  player.currentTime = f / fps;
  if (!player.paused) player.pause();
}

function seekToTime() {
  const t = parseTime(gotoTime.value);
  if (isNaN(t) || t < 0) return;
  player.currentTime = t;
  if (!player.paused) player.pause();
}

gotoFrame.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); seekToFrame(); gotoFrame.blur(); }
});
gotoTime.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); seekToTime(); gotoTime.blur(); }
});
if (gotoFrameBtn) {
  gotoFrameBtn.addEventListener('click', (e) => { e.preventDefault(); seekToFrame(); try { gotoFrame.blur(); } catch {} });
}
if (gotoTimeBtn) {
  gotoTimeBtn.addEventListener('click', (e) => { e.preventDefault(); seekToTime(); try { gotoTime.blur(); } catch {} });
}

// ── Load Video ────────────────────────────────────────────
function loadVideo(file) {
  // In the desktop window a dropped File carries its full path.  Use the
  // path flow: a blob URL of a transport stream cannot play (and the
  // native drop handler is about to load the same path anyway).
  const full = file && file.pywebviewFullPath;
  if (full && typeof window.tilesterSetVideoFromPath === 'function') {
    window.tilesterSetVideoFromPath(String(full));
    return;
  }
  const url = URL.createObjectURL(file);
  player.src         = url;
  player.style.display = 'block';
  videoDrop.style.display = 'none';
  videoName.textContent   = file.name;
  videoFilename           = file.name;

  player.addEventListener('loadedmetadata', () => {
    if (durationDisplay) durationDisplay.textContent = `Duration: ${fmtDuration(player.duration)}`;
    if (fps && fpsDisplay) fpsDisplay.innerHTML = `<span class="box-value">${fmtFpsPair()}</span>`;
  });
}

// ── Native Video Path (pywebview / server streaming) ──────
function tilesterSetVideoFromPath(fullPath) {
  const p = normalizeFullPath(fullPath);
  if (!p) return;
  if (!p.includes('\\') && !p.includes('/')) {
    console.warn('tilesterSetVideoFromPath: rejected non-path value:', p);
    return;
  }

  // The DOM drop handler and pywebview's native drop handler both fire for
  // one drop; the second arrival of the same path within a moment is a
  // duplicate, not a reload.
  if (p === _lastSetPath && Date.now() - _lastSetAt < 1500) return;
  _lastSetPath = p; _lastSetAt = Date.now();

  currentFullVideoPath = p;
  currentProjectPath   = null; // new video → clear project path until user loads/saves

  videoPathInput.value = p;
  updateButtonStates();

  const name = basenameFromPath(p);
  videoName.textContent = name;
  videoFilename         = name;

  try {
    try { player.pause(); } catch {}
    try { player.removeAttribute('src'); player.load(); } catch {}

    player.style.display = 'block';
    if (videoContainer) videoContainer.style.display = 'block';
    if (videoDrop) videoDrop.style.display = 'none';

    const seq = ++_videoLoadSeq;
    if (_needsRemux(p)) {
      // Transport streams are rewrapped to MP4 on the server first (the
      // embedded browser cannot open them — see gensrt/remux.py).
      _prepareThenLoad(p, seq);
    } else {
      _hidePrepOverlay();
      _setSaveMp4Visible(false);
      _attachSource(p);
    }

    try { tryFetchNominalFpsFromServer(p); } catch {}
  } catch (e) {
    console.warn('Failed to set player.src from path:', e);
  }
}

// ── Transport-stream rewrap (server-side, cached) ─────────
let _videoLoadSeq = 0;
let _lastSetPath = null;
let _lastSetAt = 0;
const _REMUX_EXT = /\.(ts|m2ts|mts)$/i;

function _needsRemux(p) { return _REMUX_EXT.test(p || ''); }

function _attachSource(p) {
  player.src = `/api/media?path=${encodeURIComponent(p)}`;
  player.load();
}

function _prepOverlay() {
  let el = document.getElementById('videoPrepOverlay');
  if (!el && videoContainer) {
    el = document.createElement('div');
    el.id = 'videoPrepOverlay';
    el.style.cssText = 'position:absolute; inset:0; display:flex; flex-direction:column; ' +
      'align-items:center; justify-content:center; gap:8px; background:rgba(0,0,0,0.72); ' +
      'color:var(--text); font-size:14px; z-index:5; text-align:center; padding:16px;';
    el.innerHTML = '<div id="videoPrepLabel"></div>' +
      '<div style="width:60%; max-width:360px; height:6px; background:var(--surface2); border-radius:3px; overflow:hidden;">' +
      '<div id="videoPrepBar" style="height:100%; width:0%; background:var(--accent, #7c5cff); transition:width .3s;"></div></div>' +
      '<div id="videoPrepNote" style="font-size:11px; opacity:.7;">One-time rewrap to MP4 (no re-encode); cached for next time.</div>';
    videoContainer.appendChild(el);
  }
  return el;
}

function _showPrepOverlay(label, frac) {
  const el = _prepOverlay();
  if (!el) return;
  el.style.display = 'flex';
  const lbl = document.getElementById('videoPrepLabel');
  const bar = document.getElementById('videoPrepBar');
  const withBar = typeof frac === 'number';
  if (lbl) lbl.textContent = label;
  if (bar) {
    bar.parentElement.style.display = withBar ? 'block' : 'none';
    bar.style.width = (withBar ? Math.round(frac * 100) : 0) + '%';
  }
  const note = document.getElementById('videoPrepNote');
  if (note) note.style.display = withBar ? 'block' : 'none';
}

// A message in the video area, no bar, no modal — for "cannot play".
function _showVideoMessage(html) {
  _showPrepOverlay('', null);
  const lbl = document.getElementById('videoPrepLabel');
  if (lbl) lbl.innerHTML = html;
}

function _hidePrepOverlay() {
  const el = document.getElementById('videoPrepOverlay');
  if (el) el.style.display = 'none';
}

async function _prepareThenLoad(p, seq) {
  const name = basenameFromPath(p);
  _showPrepOverlay(`Preparing ${name} for playback…`, 0);
  try {
    let resp = await fetch('/api/media/prepare', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: p }),
    });
    let st = await resp.json();
    if (!resp.ok) throw new Error(st.error || `HTTP ${resp.status}`);
    while (st.status === 'preparing' || st.status === 'unprepared') {
      if (seq !== _videoLoadSeq) return;           // another video was opened
      const pct = (typeof st.progress === 'number') ? Math.round(st.progress * 100) : null;
      _showPrepOverlay(pct === null ? `Preparing ${name} for playback…`
                       : pct >= 100 ? `Finalising ${name}…`
                       : `Preparing ${name} for playback… ${pct}%`, st.progress || 0);
      await new Promise(r => setTimeout(r, 700));
      if (st.status === 'unprepared') {
        resp = await fetch('/api/media/prepare', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: p }),
        });
      } else {
        resp = await fetch(`/api/media/status?path=${encodeURIComponent(p)}`);
      }
      st = await resp.json();
      if (!resp.ok) throw new Error(st.error || `HTTP ${resp.status}`);
    }
    if (seq !== _videoLoadSeq) return;
    if (st.status !== 'ready') throw new Error(st.error || `remux ${st.status}`);
    _hidePrepOverlay();
    _setSaveMp4Visible(true);
    _attachSource(p);
  } catch (e) {
    if (seq !== _videoLoadSeq) return;
    console.error('Preparing video for playback failed:', e);
    _showVideoMessage(`<b>${name}</b><br>Cannot prepare for playback: ${e.message || e}`);
  }
}

// ── Save the MP4 rewrap ───────────────────────────────────
function _setSaveMp4Visible(on) {
  const b = document.getElementById('saveMp4Btn');
  if (b) b.style.display = on ? '' : 'none';
}

async function saveMp4Copy() {
  const src = currentFullVideoPath;
  if (!src || !_needsRemux(src)) return;
  const sep = src.includes('\\') ? '\\' : '/';
  const idx = src.lastIndexOf(sep);
  const dir = idx >= 0 ? src.slice(0, idx) : '';
  const defName = basenameFromPath(src).replace(/\.[^.]+$/, '') + '.mp4';
  let dest = null;
  if (typeof window.pywebview !== 'undefined' && window.pywebview.api && typeof window.pywebview.api.save_mp4_as === 'function') {
    try { dest = await window.pywebview.api.save_mp4_as(defName, dir); }
    catch (e) {
      console.error('save_mp4_as dialog failed:', e);
      try { showErrorDialog('Save MP4 failed', 'The save dialog could not be opened: ' + (e.message || e)); } catch {}
      return;
    }
    if (!dest) return;                                   // cancelled
  } else {
    const name = prompt('Save MP4 as (same folder as the recording):', defName);
    if (!name) return;
    dest = dir + sep + name;
  }
  const btn = document.getElementById('saveMp4Btn');
  const label = btn ? btn.textContent : '';
  if (btn) { btn.disabled = true; btn.textContent = '⏳ Saving…'; }
  try {
    const resp = await fetch('/api/media/export', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: src, dest }),
    });
    const body = await resp.json();
    if (!resp.ok) throw new Error(body.error || `HTTP ${resp.status}`);
    if (btn) btn.textContent = '✓ Saved';
    setTimeout(() => { if (btn) btn.textContent = label; }, 2000);
  } catch (e) {
    console.error('Save MP4 failed:', e);
    if (btn) btn.textContent = label;
    try { showErrorDialog('Save MP4 failed', String(e.message || e)); } catch {}
  } finally {
    if (btn) btn.disabled = false;
  }
}
(function () {
  const b = document.getElementById('saveMp4Btn');
  if (b) b.addEventListener('click', saveMp4Copy);
})();

function tilesterSetVideoPath(fullPath) {
  tilesterSetVideoFromPath(fullPath);
}

// Expose for Python evaluate_js
window.tilesterSetVideoFromPath = tilesterSetVideoFromPath;
window.tilesterSetVideoPath     = tilesterSetVideoPath;

// Allow Python-side to surface errors in the UI
window.tilesterShowError = (title, message) => {
  try { showErrorDialog(title, message); } catch (e) { console.error('tilesterShowError:', e); }
};

// Auto-open a local file when launched with a query parameter (e.g. ?open=D%3A%5C...)
// Used by the CLI to launch the UI and pre-load a video.
(function tilesterAutoOpenFromQuery() {
  try {
    const params = new URLSearchParams(window.location.search || '');
    const raw    = params.get('open');
    if (!raw) return;
    const p = normalizeFullPath(raw);
    if (!p) return;
    setTimeout(() => {
      try { tilesterSetVideoPath(p); } catch (e) { console.warn('auto-open failed:', e); }
    }, 0);
  } catch (e) { /* URLSearchParams may be unavailable in very old engines */ }
})();

// ── Fullscreen ────────────────────────────────────────────
async function toggleFullscreenApp() {
  try {
    if (_tilesterDetectPyWebView() && window.pywebview && window.pywebview.api && window.pywebview.api.toggle_fullscreen) {
      const r = await window.pywebview.api.toggle_fullscreen();
      if (r && r.ok === false) {
        console.warn('Native fullscreen failed:', r.error || r);
        return;
      }
      _pvwIsFullscreen = !_pvwIsFullscreen;
      document.body.classList.toggle('isAppFullscreen', _pvwIsFullscreen);
      if (!_pvwIsFullscreen) {
        document.body.classList.remove('navOpen', 'headerOpen', 'footerOpen', 'controlsOpen');
      }
      return;
    }
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await appRoot.requestFullscreen();
    }
  } catch (err) {
    console.warn('Fullscreen failed:', err);
  }
}

function updateFullscreenClasses() {
  const fsEl   = document.fullscreenElement;
  const isAppFs = (fsEl === appRoot) || _pvwIsFullscreen;
  document.body.classList.toggle('isAppFullscreen', isAppFs);
  if (!isAppFs) {
    document.body.classList.remove('navOpen', 'headerOpen', 'footerOpen', 'controlsOpen');
  }
}

document.addEventListener('fullscreenchange', updateFullscreenClasses);
updateFullscreenClasses();

fsBtn.addEventListener('click', toggleFullscreenApp);

// Click video area to toggle play/pause
videoContainer.addEventListener('click', (e) => {
  if (e.target === player || e.target === videoContainer) {
    player.paused ? player.play() : player.pause();
    player.focus();
  }
});

// Double-click video area to toggle app fullscreen
videoContainer.addEventListener('dblclick', (e) => {
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON')) return;
  toggleFullscreenApp();
});

// Edge-reveal behavior (only when appRoot is fullscreen)
const OPEN_ZONE_PX    = 24;
const CLOSE_LEFT_PX   = 480;
const TOP_EDGE_PX     = 24;
const BOTTOM_EDGE_PX  = 2;

window.addEventListener('pointermove', (e) => {
  if (!document.body.classList.contains('isAppFullscreen')) return;

  const nearRight = e.clientX >= (window.innerWidth - OPEN_ZONE_PX);
  if (nearRight) document.body.classList.add('navOpen');
  else if (e.clientX <= (window.innerWidth - CLOSE_LEFT_PX)) document.body.classList.remove('navOpen');

  const nearTop = e.clientY <= TOP_EDGE_PX;
  if (nearTop) document.body.classList.add('headerOpen');
  else if (e.clientY > 80) document.body.classList.remove('headerOpen');

  const CONTROLS_ZONE_PX = 80;
  const nearBottom = e.clientY >= (window.innerHeight - CONTROLS_ZONE_PX);
  const atBottom   = e.clientY >= (window.innerHeight - BOTTOM_EDGE_PX);

  if (nearBottom) document.body.classList.add('controlsOpen');
  else if (e.clientY < (window.innerHeight - 120)) document.body.classList.remove('controlsOpen');

  if (atBottom) document.body.classList.add('footerOpen');
  else if (e.clientY < (window.innerHeight - 80)) document.body.classList.remove('footerOpen');
});

navEdgeHotzone.addEventListener('pointerenter', () => {
  if (document.body.classList.contains('isAppFullscreen')) document.body.classList.add('navOpen');
});

// ── Custom Video Controls ─────────────────────────────────
const _volSteps = [1, 0.75, 0.5, 0.25, 0];
const _volIcons = ['🔊', '🔉', '🔉', '🔈', '🔇'];

function _updateVcPlayBtn() {
  if (vcPlayBtn) vcPlayBtn.textContent = player.paused ? '▶' : '⏸';
}
function _updateVcVolumeBtn() {
  if (!vcVolumeBtn) return;
  if (player.muted || player.volume === 0) { vcVolumeBtn.textContent = '🔇'; return; }
  const idx = _volSteps.findIndex(v => player.volume >= v - 0.01);
  vcVolumeBtn.textContent = _volIcons[idx >= 0 ? idx : 0];
}
function _updateVcFsBtn() {
  if (vcFsBtn) vcFsBtn.textContent = document.body.classList.contains('isAppFullscreen') ? '✕' : '⛶';
}

if (vcPlayBtn) {
  vcPlayBtn.addEventListener('click', () => { player.paused ? player.play() : player.pause(); });
}
player.addEventListener('play',  _updateVcPlayBtn);
player.addEventListener('pause', _updateVcPlayBtn);

// A <video> that cannot play its source fails SILENTLY: no console line,
// no dialog, the element just stays black.  Surface the MediaError so a
// container or codec refusal (WebView2 declining an MPEG-TS, an HEVC
// stream without the codec pack) is on screen instead of guessed at.
const _MEDIA_ERR = {
  1: 'MEDIA_ERR_ABORTED — loading was aborted',
  2: 'MEDIA_ERR_NETWORK — the server stopped sending data',
  3: 'MEDIA_ERR_DECODE — the file was served but could not be decoded',
  4: 'MEDIA_ERR_SRC_NOT_SUPPORTED — this container/codec is not playable by the embedded browser',
};
player.addEventListener('error', () => {
  const err  = player.error;
  const code = err ? err.code : 0;
  const what = _MEDIA_ERR[code] || `unknown media error (code ${code})`;
  const detail = err && err.message ? ` — ${err.message}` : '';
  console.error(`video element error: ${what}${detail}`, { src: player.currentSrc, networkState: player.networkState, readyState: player.readyState });
  // An element with no source yet (cleared before a load) can report an
  // error too; that is not news.  Only a real source gets a message.
  if (!player.currentSrc) return;
  try {
    _showVideoMessage(`<b>${basenameFromPath(currentFullVideoPath || '') || 'The video'}</b><br>Cannot play: ${what}${detail}`);
  } catch (e) { /* overlay helper not ready */ }
});

if (vcVolumeBtn) {
  vcVolumeBtn.addEventListener('click', () => {
    if (player.muted) {
      player.muted  = false;
      player.volume = _volSteps[0];
    } else {
      const cur  = player.volume;
      const idx  = _volSteps.findIndex(v => cur >= v - 0.01);
      const next = (idx + 1) % _volSteps.length;
      if (_volSteps[next] === 0) { player.muted = true; }
      else { player.muted = false; player.volume = _volSteps[next]; }
    }
  });
}
player.addEventListener('volumechange', _updateVcVolumeBtn);

if (vcFsBtn) vcFsBtn.addEventListener('click', toggleFullscreenApp);

new MutationObserver(_updateVcFsBtn)
  .observe(document.body, { attributes: true, attributeFilter: ['class'] });

// ── Pin Toggle ────────────────────────────────────────────
let isPinned = false;

pinBtn.addEventListener('click', () => {
  isPinned = !isPinned;
  document.body.classList.toggle('pinned', isPinned);
  pinBtn.classList.toggle('pinned', isPinned);
  pinBtn.textContent = isPinned ? '📌 Unpin' : '📌 Pin';
});

// ── Copy Time to Clipboard ────────────────────────────────
copyTimeBtn.addEventListener('click', async () => {
  const timeText = currentTimeEl.textContent;
  try {
    await navigator.clipboard.writeText(timeText);
    copyTimeBtn.classList.add('copied');
    copyTimeBtn.textContent = '✓';
    setTimeout(() => {
      copyTimeBtn.classList.remove('copied');
      copyTimeBtn.textContent = '📋';
    }, 1000);
  } catch (err) {
    showErrorDialog('Copy Failed', 'Failed to copy time to clipboard.');
  }
});

// ── Drag & Drop ───────────────────────────────────────────
function setupDrop(el, onFile, accept) {
  el.addEventListener('dragover', (e) => { e.preventDefault(); el.classList.add('dragover'); });
  el.addEventListener('dragleave', () => el.classList.remove('dragover'));
  el.addEventListener('drop', (e) => {
    e.preventDefault();
    el.classList.remove('dragover');

    const files     = [...e.dataTransfer.files];
    const match     = files.find(f => accept(f));
    if (match) onFile(match);

    // Companion file handling: video drop zone also accepts JSON, and vice versa
    const jsonFile  = files.find(f => f.name.endsWith('.json'));
    const videoFile = files.find(f => f.type.startsWith('video/') || /\.(mp4|mkv|webm|avi|mov|ts|m2ts)$/i.test(f.name));
    if (jsonFile  && el === videoDrop) loadJSON(jsonFile);
    if (videoFile && el === jsonDrop)  loadVideo(videoFile);
  });
}

const _isVideoFile = f => f.type.startsWith('video/') || /\.(mp4|mkv|webm|avi|mov|ts|m2ts)$/i.test(f.name);
const _isJsonFile  = f => f.name.endsWith('.json');

setupDrop(videoDrop, loadVideo, _isVideoFile);
setupDrop(jsonDrop,  loadJSON,  _isJsonFile);

// Also allow the whole video container as a drop target after video loaded
videoContainer.addEventListener('dragover', (e) => e.preventDefault());
videoContainer.addEventListener('drop', (e) => {
  e.preventDefault();
  const files     = [...e.dataTransfer.files];
  const videoFile = files.find(_isVideoFile);
  const jsonFile  = files.find(_isJsonFile);
  if (videoFile) loadVideo(videoFile);
  if (jsonFile)  loadJSON(jsonFile);
});

// Click drop zone to browse
videoDrop.addEventListener('click', () => {
  if (_tilesterDetectPyWebView() && window.pywebview.api && typeof window.pywebview.api.select_video === 'function') {
    window.pywebview.api.select_video().then(path => {
      if (path) window.tilesterSetVideoFromPath(path);
    }).catch(err => { console.error('videoDrop native picker error:', err); });
  } else {
    videoInput.click();
  }
});
jsonDrop.addEventListener('click', () => {
  if (_tilesterDetectPyWebView() && window.pywebview.api && typeof window.pywebview.api.select_srt === 'function') {
    window.pywebview.api.select_srt().then(path => {
      if (path && typeof window.gensrtLoadSrtFromPath === 'function') {
        window.gensrtLoadSrtFromPath(path);
      }
    }).catch(err => { console.error('jsonDrop native picker error:', err); });
  } else {
    jsonInput.click();
  }
});

videoInput.addEventListener('change', (e) => { if (e.target.files[0]) loadVideo(e.target.files[0]); });
jsonInput.addEventListener('change', (e) => { if (e.target.files[0]) loadJSON(e.target.files[0]); });

// ── Playback Updates ──────────────────────────────────────
player.addEventListener('timeupdate', () => {
  currentTimeEl.textContent = fmtTime(player.currentTime);

  if (!fps) tryDetectFpsFromPlayer();

  if (fps) {
    const f = frameAtTime(player.currentTime);
    const v = fmtFrameTag(f);
    if (frameNum) {
      const inner = frameNum.querySelector('.box-value');
      if (inner) inner.textContent = v;
      else frameNum.textContent = v;
    }
  }

  updateChapterTimelineNeedle();
  updateActiveRow(player.currentTime);
});

// ── Spacebar: capture-phase so it fires before native <video controls> ──
document.addEventListener('keydown', (e) => {
  if (e.key !== ' ') return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  e.preventDefault();
  e.stopPropagation();
  player.paused ? player.play() : player.pause();
}, true); // capture = true

// ── Keyboard Shortcuts ────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;

  const skipBack = parseFloat(skipBackward.value) || 5;
  const skipFwd  = parseFloat(skipForward.value)  || 5;

  switch (e.key) {
    case 'Delete':
      e.preventDefault();
      deleteSelectedSegments();
      break;
    case 'f': case 'F':
      e.preventDefault();
      toggleFullscreenApp();
      break;
    case 'p': case 'P':
      e.preventDefault();
      pinBtn.click();
      break;
    case 'm': case 'M':
      e.preventDefault();
      player.muted = !player.muted;
      _updateVcVolumeBtn();
      break;
    case 'ArrowLeft':
      e.preventDefault();
      player.currentTime = Math.max(0, player.currentTime - skipBack);
      break;
    case 'ArrowRight':
      e.preventDefault();
      player.currentTime = Math.min(player.duration || 0, player.currentTime + skipFwd);
      break;
    case ',':
      e.preventDefault();
      if (!player.paused) player.pause();
      if (fps) player.currentTime = Math.max(0, player.currentTime - 1 / fps);
      break;
    case '.':
      e.preventDefault();
      if (!player.paused) player.pause();
      if (fps) player.currentTime = Math.min(player.duration || 0, player.currentTime + 1 / fps);
      break;
  }
});
