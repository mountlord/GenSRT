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
