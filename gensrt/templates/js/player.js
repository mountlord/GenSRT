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

    player.src = `/api/media?path=${encodeURIComponent(p)}`;
    player.style.display = 'block';
    if (videoContainer) videoContainer.style.display = 'block';
    if (videoDrop) videoDrop.style.display = 'none';
    player.load();

    try { tryFetchNominalFpsFromServer(p); } catch {}
  } catch (e) {
    console.warn('Failed to set player.src from path:', e);
  }
}

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
      if (path) tilesterSetVideoFromPath(path);
    }).catch(err => { console.error('videoDrop native picker error:', err); });
  } else {
    videoInput.click();
  }
});
jsonDrop.addEventListener('click', () => jsonInput.click());

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
