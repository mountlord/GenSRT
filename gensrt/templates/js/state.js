// ── DOM References ────────────────────────────────────────
const chapterTimeline       = document.getElementById('chapterTimeline');
const chapterTimelineNeedle = document.getElementById('chapterTimelineNeedle');
const appRoot               = document.getElementById('appRoot');
const player                = document.getElementById('player');
const videoContainer        = document.getElementById('videoContainer');
const videoDrop             = document.getElementById('videoDrop');
const videoName             = document.getElementById('videoName');
const currentTimeEl         = document.getElementById('currentTime');
const copyTimeBtn           = document.getElementById('copyTimeBtn');
const durationDisplay       = document.getElementById('durationDisplay');
const jsonDrop              = document.getElementById('jsonDrop');
const navContainer          = document.getElementById('navContainer');
const linkCount             = document.getElementById('linkCount');
const videoInput            = document.getElementById('videoInput');
const jsonInput             = document.getElementById('jsonInput');
const skipBackward          = document.getElementById('skipBackward');
const skipForward           = document.getElementById('skipForward');
const fpsDisplay            = document.getElementById('fpsDisplay');
const frameNum              = document.getElementById('frameNum');
const segMarkBtn            = document.getElementById('segMarkBtn');
const segLenInput           = document.getElementById('segLenInput');
const segPendingDisplay     = document.getElementById('segPendingDisplay');
const gotoFrame             = document.getElementById('gotoFrame');
const gotoTime              = document.getElementById('gotoTime');
const gotoFrameBtn          = document.getElementById('gotoFrameBtn');
const gotoTimeBtn           = document.getElementById('gotoTimeBtn');
const fsBtn                 = document.getElementById('fsBtn');
const pinBtn                = document.getElementById('pinBtn');
const videoControls         = document.getElementById('videoControls');
const vcPlayBtn             = document.getElementById('vcPlayBtn');
const vcVolumeBtn           = document.getElementById('vcVolumeBtn');
const vcFsBtn               = document.getElementById('vcFsBtn');
const detectBtn             = document.getElementById('detectBtn');
const navEdgeHotzone        = document.getElementById('navEdgeHotzone');
const linksBody             = document.getElementById('linksBody');
const chapterEditorModal    = document.getElementById('chapterEditorModal');
const editStartTime         = document.getElementById('editStartTime');
const editStartFrame        = document.getElementById('editStartFrame');
const editEndTime           = document.getElementById('editEndTime');
const editEndFrame          = document.getElementById('editEndFrame');
const modalCancel           = document.getElementById('modalCancel');
const modalSave             = document.getElementById('modalSave');
const configBtn             = document.getElementById('configBtn');
const configEditorModal     = document.getElementById('configEditorModal');
const configEditorBody      = document.getElementById('configEditorBody');
const configCancel          = document.getElementById('configCancel');
const configLoad            = document.getElementById('configLoad');
const configSave            = document.getElementById('configSave');
const progressModal         = document.getElementById('progressModal');
const progressTitle         = document.getElementById('progressTitle');
const progressMessage       = document.getElementById('progressMessage');
const progressBar           = document.getElementById('progressBar');
const progressPercent       = document.getElementById('progressPercent');
const progressElapsed       = document.getElementById('progressElapsed');
const progressETA           = document.getElementById('progressETA');
const progressProcessing    = document.getElementById('progressProcessing');
const progressResult        = document.getElementById('progressResult');
const progressResultIcon    = document.getElementById('progressResultIcon');
const progressResultMessage = document.getElementById('progressResultMessage');
const progressCloseBtn      = document.getElementById('progressCloseBtn');
const videoPathInput        = document.getElementById('videoPathInput');
const resetBtn              = document.getElementById('resetBtn');
const browseBtn             = document.getElementById('browseBtn');
const saveBtn               = document.getElementById('saveBtn');
const saveAsBtn             = document.getElementById('saveAsBtn');
const addBtn                = document.getElementById('addBtn');
const splitBtn              = document.getElementById('splitBtn');
const mergeBtn              = document.getElementById('mergeBtn');
const deleteBtn             = document.getElementById('deleteBtn');
const confirmModal          = document.getElementById('confirmModal');
const confirmModalTitle     = document.getElementById('confirmModalTitle');
const confirmModalMessage   = document.getElementById('confirmModalMessage');
const confirmModalCancel    = document.getElementById('confirmModalCancel');
const confirmModalOk        = document.getElementById('confirmModalOk');

// ── Shared State ──────────────────────────────────────────
let linksData            = null;
let fps                  = null;
let fpsNominal           = null;
let videoFilename        = '';
let currentProjectPath   = null;
let currentFullVideoPath = null;
let isServerMode         = false;

let editingChapterIndex  = null;
let pendingChapterStart  = null;

let transitionsArr     = [];
let chaptersArr        = [];
let chapterSelections  = [];
let transitionEls      = [];
let chapterEls         = [];
let activeTransitionEl = null;
let activeChapterEl    = null;

let progressStartTime    = null;
let progressElapsedTimer = null;
let progressLastPercent  = 0;
let progressLastETA      = '';
let progressPollTimer    = null;
let progressPollInFlight = false;

let _videoInfoSeq = 0;

// ── Utility Helpers ───────────────────────────────────────
function fmtTime(s) {
  const total = Math.floor(s * 1000);
  const ms = total % 1000;
  const sec = Math.floor(total / 1000);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const ss = sec % 60;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}.${String(ms).padStart(3,'0')}`;
}

function fmtDuration(s) {
  const sec = Math.round(s);
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const r = sec % 60;
  if (m < 60) return `${m}m ${r}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m ${r}s`;
}

function fmtFrameTag(f) {
  if (f === null || f === undefined || isNaN(f)) return '—';
  return String(f);
}

function frameAtTime(t) {
  if (!fps || typeof t !== 'number' || isNaN(t)) return null;
  return Math.round(t * fps);
}

function fmtFpsPair() {
  if (!fps || !isFinite(fps) || fps <= 0) return '—/—';
  const eff = (Math.abs(fps - Math.round(fps)) < 0.0005) ? String(Math.round(fps)) : fps.toFixed(3);
  const nom = (fpsNominal && isFinite(fpsNominal) && fpsNominal > 0)
    ? (Math.abs(fpsNominal - Math.round(fpsNominal)) < 0.0005 ? String(Math.round(fpsNominal)) : fpsNominal.toFixed(3))
    : '—';
  return `${eff}/${nom}`;
}

function normalizeFullPath(p) {
  if (!p) return '';
  // Strip quotes (Windows Copy-As-Path uses quotes)
  return String(p).replace(/^["']|["']$/g, '');
}

function basenameFromPath(p) {
  const s = normalizeFullPath(p);
  const parts = s.split(/[\\/]/);
  return parts[parts.length - 1] || s;
}


// ── Copyable messages ───────────────────────────────────────────────────
//
// PROJECT CONVENTION: any message a user might need to report must be
// selectable and copyable. Asking someone to screenshot an error to report it
// is friction we control, and a screenshot loses the exact text — paths,
// model names and version numbers all have to be retyped by whoever reads it.
//
// Attach with makeCopyable(el). Idempotent, so it is safe to call on every
// render. Works on any element whose text is set via textContent.

function makeCopyable(el) {
  if (!el || el.dataset.copyableAttached === '1') return;
  el.dataset.copyableAttached = '1';
  el.classList.add('copyable-message');

  const btn = document.createElement('button');
  btn.className = 'copy-msg-btn';
  btn.type = 'button';
  btn.textContent = 'copy';
  btn.title = 'Copy this message to the clipboard';

  btn.addEventListener('click', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    // The button lives inside the element, so its own label would be copied
    // along with the message. Read the text nodes only.
    const text = Array.from(el.childNodes)
      .filter(n => n.nodeType === Node.TEXT_NODE)
      .map(n => n.textContent)
      .join('')
      .trim();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = 'copied';
    } catch (err) {
      // Clipboard API needs a secure context; WebView2 on 127.0.0.1 counts,
      // but fall back rather than failing silently.
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); btn.textContent = 'copied'; }
      catch (e2) { btn.textContent = 'select + Ctrl-C'; }
      document.body.removeChild(ta);
    }
    setTimeout(() => { btn.textContent = 'copy'; }, 1800);
  });

  // Callers set the message with `el.textContent = msg`, which REPLACES every
  // child — including this button. That is why the first version of this
  // silently did nothing: the button was appended once and destroyed by the
  // next message. Rather than requiring every call site to use a special
  // setter (and remembering to, forever), watch for it and put the button
  // back. Self-healing, and new call sites get it for free.
  const reattach = () => {
    if (!el.contains(btn)) el.appendChild(btn);
    btn.style.display = el.textContent.replace('copy', '').trim() ? '' : 'none';
  };

  new MutationObserver(reattach).observe(el, { childList: true });
  reattach();
}
