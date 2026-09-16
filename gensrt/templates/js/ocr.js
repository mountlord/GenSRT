// ── On-screen text recognition (OCR) ──────────────────────────────────────
//
// "Pause on a frame, read the text, add it as a subtitle line."
//
// Three things here are load-bearing and worth knowing before editing:
//
// 1. CAPTURE SENDS PIXELS, NOT A TIMESTAMP.  The browser already holds the
//    decoded frame, so drawing it to a canvas costs nothing and needs no
//    re-seek on the server.  It also guarantees what gets read is exactly
//    what the user is looking at.
//
// 2. THE RIGHT PANE COMPOSES ONE CUE, NOT SEVERAL.  Several cues at one
//    timestamp would overlap, and build_srt's overlap-clamp would then
//    mangle their timing into something nobody asked for.  So the chosen
//    lines are joined into a single multi-line cue, and their ORDER matters
//    — which is why this is a dual-list with move up/down rather than
//    checkboxes.
//
// 3. INSERTION IS SYSTEM-DRIVEN, WHICH IS NOT THE SAME AS THE ADD BUTTON.
//    Add stays restricted to an empty list on purpose (the user already has
//    Split for insertion).  insertSegmentAt() below is internal plumbing for
//    an insertion the SYSTEM originates at a timestamp the user is parked
//    on; it does not give the user a second manual door.

let _ocrRegions = [];      // everything detected in the current frame
let _ocrChosen  = [];      // indices into _ocrRegions, in composition order
let _ocrSelDetected = null;
let _ocrSelChosen   = null;
let _ocrCapturedAt  = 0;   // playhead position of the captured frame

// ── Capture ───────────────────────────────────────────────────────────────

function _ocrCaptureFrame() {
  if (!player || !player.videoWidth) return null;
  const canvas  = document.createElement('canvas');
  canvas.width  = player.videoWidth;
  canvas.height = player.videoHeight;
  canvas.getContext('2d').drawImage(player, 0, 0, canvas.width, canvas.height);
  // PNG rather than JPEG: text edges are exactly what the recogniser reads,
  // and JPEG ringing around glyphs is the last thing to introduce here.
  return canvas.toDataURL('image/png');
}

async function _ocrLoadLanguages() {
  const select = document.getElementById('ocrLanguage');
  if (!select || select.options.length) return;
  try {
    const data = await (await fetch('/api/ocr/languages')).json();
    (data.languages || []).forEach(lang => {
      const opt = document.createElement('option');
      opt.value = lang.code;
      // The size hint matters on first use: picking a language triggers a
      // download, and silence during it looks like a hang.
      opt.textContent = lang.present
        ? lang.label
        : `${lang.label} (${lang.size_mb} MB download)`;
      select.appendChild(opt);
    });
    const codes = (data.languages || []).map(l => l.code);
    select.value = _ocrPreferredLanguage(codes) || data.default || codes[0];
  } catch (err) {
    _ocrSetStatus(`Could not load OCR languages: ${err}`, true);
  }
}

function _ocrSetStatus(message, isError = false) {
  const el = document.getElementById('ocrStatus');
  if (!el) return;
  el.textContent = message || '';
  el.classList.toggle('error', !!isError);
}

// ── Scan ──────────────────────────────────────────────────────────────────

async function _ocrScan() {
  const image = _ocrCaptureFrame();
  if (!image) {
    showErrorDialog('Read Frame', 'Load a video and pause on a frame first.');
    return;
  }
  _ocrCapturedAt = player.currentTime || 0;

  const language  = document.getElementById('ocrLanguage').value || 'ja';
  const translate = document.getElementById('ocrTranslate').checked;

  _ocrSetStatus('Reading frame…');
  document.getElementById('ocrRescan').disabled = true;
  try {
    const res = await fetch('/api/ocr', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ image, language, translate,
                                target_language: _ocrTargetLanguage() }),
    });
    const data = await res.json();
    if (!res.ok) {
      // 422 is "you picked a language with no model" — the user's choice to
      // fix.  Anything else is ours.
      _ocrSetStatus(data.error || `OCR failed (HTTP ${res.status})`, true);
      _ocrRegions = [];
      _ocrRenderPanes();
      return;
    }
    _ocrRegions = data.regions || [];
    _ocrChosen  = [];
    _ocrSelDetected = _ocrSelChosen = null;

    const failure = _ocrRegions.find(r => r.translation_error);
    if (failure) {
      _ocrSetStatus(
        `${_ocrRegions.length} region(s) — translation failed: ` +
        `${failure.translation_error}`, true);
    } else {
      _ocrSetStatus(
        _ocrRegions.length
          ? `${_ocrRegions.length} region(s) at ${formatTimeForInput(_ocrCapturedAt)}`
          : 'No text found in this frame.'
      );
    }
    _ocrRenderPanes();
    _ocrSyncCompose();
  } catch (err) {
    _ocrSetStatus(`OCR request failed: ${err}`, true);
  } finally {
    document.getElementById('ocrRescan').disabled = false;
  }
}

// Set once the user ticks or unticks the box themselves; until then the
// picker follows the app's translation setting.
let _ocrTranslateTouched = false;

function _ocrApplyTranslateDefault() {
  const box = document.getElementById('ocrTranslate');
  if (!box || _ocrTranslateTouched) return;
  const cfg = (typeof currentConfig !== 'undefined' && currentConfig) || {};
  const engineOff = String(cfg.translation_engine || '').toLowerCase() === 'none';
  box.checked = (cfg.translate !== false) && !engineOff;
}

function _ocrTargetLanguage() {
  // Mirror the footer's translation target, so the English produced here
  // matches the English in the rest of the file.
  const sel = document.getElementById('sel-target-lang');
  return (sel && sel.value) ? sel.value : 'en';
}

function _ocrPreferredLanguage(available) {
  // Default the OCR language to the job's source language when that
  // language has a model: someone transcribing Japanese audio is almost
  // certainly looking at Japanese on screen.  "auto" and unsupported
  // languages fall through to the server's default.
  const sel = document.getElementById('sel-source-lang');
  const code = (sel && sel.value || '').toLowerCase();
  return available.includes(code) ? code : null;
}

// ── Panes ─────────────────────────────────────────────────────────────────

function _ocrRowHtml(region, selected) {
  const conf = (region.confidence == null)
    ? ''
    : `<span class="ocr-conf">${region.confidence.toFixed(2)}</span>`;
  // A failed translation must be visible. The server keeps the OCR result
  // and attaches translation_error rather than failing the request — but
  // rendering only `translation` made a failure look identical to
  // translation being switched off, which is worse than either.
  let translation = '';
  if (region.translation) {
    translation = `<div class="ocr-tr">${_escapeHtml(region.translation)}</div>`;
  } else if (region.translation_error) {
    translation = `<div class="ocr-tr err">translation failed: ` +
                  `${_escapeHtml(region.translation_error)}</div>`;
  }
  // The crop is not decoration.  For a user who cannot read the script it is
  // the only way to tell which row is the sign, which is the caption, and
  // which is the channel watermark — and to see a misread at a glance.
  return `
    <div class="ocr-row${selected ? ' selected' : ''}" data-idx="${region.index}">
      <img class="ocr-crop" src="${region.crop || ''}" alt="">
      <div class="ocr-row-text">
        <div class="ocr-ja">${_escapeHtml(region.text)}${conf}</div>
        ${translation}
      </div>
    </div>`;
}

function _ocrRenderPanes() {
  const detected = document.getElementById('ocrDetected');
  const chosen   = document.getElementById('ocrChosen');
  if (!detected || !chosen) return;

  const remaining = _ocrRegions.filter(r => !_ocrChosen.includes(r.index));
  detected.innerHTML = remaining.length
    ? remaining.map(r => _ocrRowHtml(r, r.index === _ocrSelDetected)).join('')
    : '<div class="ocr-empty">Nothing detected.</div>';

  chosen.innerHTML = _ocrChosen.length
    ? _ocrChosen.map(i => {
        const region = _ocrRegions.find(r => r.index === i);
        return region ? _ocrRowHtml(region, i === _ocrSelChosen) : '';
      }).join('')
    : '<div class="ocr-empty">Pick the lines that belong in the subtitle.</div>';

  detected.querySelectorAll('.ocr-row').forEach(row => {
    row.addEventListener('click', () => {
      _ocrSelDetected = Number(row.dataset.idx);
      _ocrRenderPanes();
    });
    row.addEventListener('dblclick', () => {
      _ocrSelDetected = Number(row.dataset.idx);
      _ocrMove('add');
    });
  });
  chosen.querySelectorAll('.ocr-row').forEach(row => {
    row.addEventListener('click', () => {
      _ocrSelChosen = Number(row.dataset.idx);
      _ocrRenderPanes();
    });
    row.addEventListener('dblclick', () => {
      _ocrSelChosen = Number(row.dataset.idx);
      _ocrMove('remove');
    });
  });
}

function _ocrMove(action) {
  if (action === 'add' && _ocrSelDetected != null) {
    if (!_ocrChosen.includes(_ocrSelDetected)) _ocrChosen.push(_ocrSelDetected);
    _ocrSelDetected = null;
  } else if (action === 'all') {
    _ocrRegions.forEach(r => {
      if (!_ocrChosen.includes(r.index)) _ocrChosen.push(r.index);
    });
  } else if (action === 'remove' && _ocrSelChosen != null) {
    _ocrChosen = _ocrChosen.filter(i => i !== _ocrSelChosen);
    _ocrSelChosen = null;
  } else if (action === 'up' || action === 'down') {
    const at = _ocrChosen.indexOf(_ocrSelChosen);
    const to = at + (action === 'up' ? -1 : 1);
    if (at < 0 || to < 0 || to >= _ocrChosen.length) return;
    [_ocrChosen[at], _ocrChosen[to]] = [_ocrChosen[to], _ocrChosen[at]];
  }
  _ocrRenderPanes();
  _ocrSyncCompose();
}

// ── Compose ───────────────────────────────────────────────────────────────

function _ocrComposedText() {
  return _ocrChosen.map(i => {
    const region = _ocrRegions.find(r => r.index === i);
    if (!region) return '';
    // Prefer the translation when there is one: that is the text the user
    // actually wants in an English subtitle file.
    return (region.translation || region.text || '').trim();
  }).filter(Boolean).join('\n');
}

function _ocrSyncCompose() {
  const textarea = document.getElementById('ocrText');
  if (textarea) textarea.value = _ocrComposedText();
  _ocrUpdateLineInfo();
}

function _ocrUpdateLineInfo() {
  const info = document.getElementById('ocrLineInfo');
  const textarea = document.getElementById('ocrText');
  if (!info || !textarea) return;

  const lines = textarea.value.split('\n').filter(l => l.trim());
  // Warn rather than block: these are the user's own formatting settings,
  // and a three-line sign they deliberately picked is their call to make.
  const maxLines = _ocrConfigNumber('max_lines', 2);
  const maxChars = _ocrConfigNumber('max_line_chars', 42);
  const longest  = lines.reduce((m, l) => Math.max(m, l.length), 0);

  const problems = [];
  if (lines.length > maxLines) problems.push(`${lines.length} lines (max ${maxLines})`);
  if (longest > maxChars)      problems.push(`${longest} chars (max ${maxChars})`);

  info.textContent = problems.length ? `— ${problems.join(', ')}` : '';
  info.classList.toggle('warn', problems.length > 0);
}

function _ocrConfigNumber(key, fallback) {
  try {
    const value = ((typeof currentConfig !== 'undefined' && currentConfig) || {})[key];
    return (value == null || isNaN(Number(value))) ? fallback : Number(value);
  } catch { return fallback; }
}

// ── Ordered insert ────────────────────────────────────────────────────────

function insertSegmentAt(startTime, endTime, text) {
  const proj     = _ensureEditableProject();
  const segments = proj.segments;

  const seg = {
    index:      segments.length + 1,
    start_time: Number(startTime.toFixed(3)),
    end_time:   Number(endTime.toFixed(3)),
    text:       text,
    has_seams:  false,
    seam_count: 0,
    manual:     true,
    source:     'ocr',
  };
  if (fps && isFinite(fps)) {
    seg.start_frame = frameAtTime(seg.start_time);
    seg.end_frame   = frameAtTime(seg.end_time);
  }

  segments.push(seg);
  // _reindexSegments sorts by start_time, so pushing then reindexing places
  // the cue correctly wherever in the timeline it belongs — including into a
  // gap with no neighbouring cue selected, which is exactly the case Split
  // cannot serve.
  _reindexSegments(segments);
  proj.segments       = segments;
  proj.schema_version = 1;
  if (!proj.fps && fps) proj.fps = fps;

  linksData = proj;
  renderLinks(proj);
  if (typeof updateButtonStates === 'function') updateButtonStates();
  return seg;
}

function _ocrDuplicateNearby(text, startTime, windowS = 10) {
  // On-screen text persists across frames, so reading the same sign twice is
  // the natural user mistake rather than an exotic one.
  const segments = (linksData && linksData.segments) || [];
  const needle = text.trim();
  return segments.find(s =>
    (s.text || '').trim() === needle &&
    Math.abs((s.start_time ?? 0) - startTime) <= windowS
  ) || null;
}

// ── Open / save ───────────────────────────────────────────────────────────

async function openOcrModal() {
  if (!player || !player.videoWidth) {
    showErrorDialog('Read Frame', 'Load a video and pause on a frame first.');
    return;
  }
  if (!player.paused) player.pause();

  _ocrSetMode('frame');
  await _ocrLoadLanguages();

  _ocrRegions = [];
  _ocrChosen  = [];
  _ocrSelDetected = _ocrSelChosen = null;
  _ocrCapturedAt  = player.currentTime || 0;

  // Seed Translate from the job's own translation setting. Leaving it
  // unchecked by default meant someone with translation switched on for
  // the whole app opened the picker, got an immediate scan with translate
  // off, and saw bare source text with nothing explaining why. Once the
  // user touches the checkbox their choice sticks for the session.
  _ocrApplyTranslateDefault();

  document.getElementById('ocrStart').value = formatTimeForInput(_ocrCapturedAt);
  document.getElementById('ocrEnd').value   = formatTimeForInput(_ocrCapturedAt + 2.0);
  document.getElementById('ocrText').value  = '';
  _ocrRenderPanes();
  _ocrSetStatus('');
  document.getElementById('ocrModal').classList.add('visible');

  _ocrScan();
}

function _ocrSave() {
  const start = parseTimeInput(document.getElementById('ocrStart').value);
  const end   = parseTimeInput(document.getElementById('ocrEnd').value);
  const text  = document.getElementById('ocrText').value.trim();

  if (isNaN(start) || isNaN(end)) {
    showErrorDialog('Invalid Time', 'Times must be in <strong>HH:MM:SS.mmm</strong> format.');
    return;
  }
  if (start >= end) {
    showErrorDialog('Invalid Range', 'Start must be before End.');
    return;
  }
  if (!text) {
    showErrorDialog('Nothing to Add',
      'Pick at least one detected line, or type the subtitle text yourself.');
    return;
  }

  const duplicate = _ocrDuplicateNearby(text, start);
  if (duplicate) {
    showErrorDialog('Already Added',
      `A line with this exact text already exists at ` +
      `<span style="font-family: var(--font-mono);">${fmtTime(duplicate.start_time)}</span>. ` +
      `On-screen text persists across frames, so this is probably the same ` +
      `sign read twice.<br><br>Edit the wording or the times if you meant to ` +
      `add it again.`);
    return;
  }

  insertSegmentAt(start, end, text);
  document.getElementById('ocrModal').classList.remove('visible');
}

// ── Wiring ────────────────────────────────────────────────────────────────

(function _ocrWire() {
  const modal = document.getElementById('ocrModal');
  if (!modal) return;

  const on = (id, event, handler) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener(event, handler);
  };

  on('ocrCancel', 'click', () => modal.classList.remove('visible'));
  modal.addEventListener('click', e => {
    if (e.target === modal) modal.classList.remove('visible');
  });
  on('ocrSave',   'click', _ocrSave);
  on('ocrRescan', 'click', _ocrScan);
  on('ocrAdd',    'click', () => _ocrMove('add'));
  on('ocrAddAll', 'click', () => _ocrMove('all'));
  on('ocrRemove', 'click', () => _ocrMove('remove'));
  on('ocrUp',     'click', () => _ocrMove('up'));
  on('ocrDown',   'click', () => _ocrMove('down'));
  on('ocrText',   'input', _ocrUpdateLineInfo);
  // Re-scan on either change: a different language needs a new read, and
  // translation is done server-side so it cannot be applied retroactively.
  on('ocrLanguage',  'change', _ocrScan);
  on('ocrTranslate', 'change', () => { _ocrTranslateTouched = true; _ocrScan(); });

  // Gate the button on a frame existing. Listening on the player element
  // rather than hooking each load path (file drop, native path, server
  // stream) keeps this correct however the video arrived.
  if (typeof player !== 'undefined' && player) {
    ['loadedmetadata', 'loadeddata', 'emptied'].forEach(evt =>
      player.addEventListener(evt, () => {
        if (typeof _updateSrtLineButtons === 'function') _updateSrtLineButtons();
      })
    );
  }
  if (typeof ocrBtn !== 'undefined' && ocrBtn) ocrBtn.disabled = true;
})();

// ── Extract Subtitles: mode menu, region selector, run settings ───────────
//
// Second OCR mode: instead of reading one paused frame, walk the whole video
// and turn burned-in subtitles into cues.
//
// The design rests on one observation about this problem: the hard part is
// not recognition, it is knowing WHEN a subtitle starts and ends. Comparing
// the recognised TEXT between samples — rather than the pixels — is invariant
// to whatever the video is doing behind the caption, and it hands you several
// independent readings of the same subtitle, which can then be voted on.
// Both properties matter; the pixel approach has neither.
//
// This file holds only the UI. The backend is not written yet; Generate
// assembles the request and shows it, which keeps the contract visible and
// testable before a line of server code exists.

let _ocrMode = 'frame';            // 'frame' | 'extract'
let _ocrRegion = null;             // {x, y, w, h} in VIDEO pixel coordinates

// ── Mode menu ─────────────────────────────────────────────────────────────

function _ocrToggleMenu(show) {
  const menu = document.getElementById('ocrMenu');
  if (!menu) return;
  const open = (show === undefined) ? !menu.classList.contains('visible') : show;
  menu.classList.toggle('visible', open);
  if (open) {
    // Position under the button, and close on the next click anywhere else.
    const r = ocrBtn.getBoundingClientRect();
    menu.style.left = `${r.left}px`;
    menu.style.top  = `${r.bottom + 4}px`;
    setTimeout(() => document.addEventListener('click', _ocrCloseMenuOnce), 0);
  }
}

function _ocrCloseMenuOnce(e) {
  const menu = document.getElementById('ocrMenu');
  if (menu && !menu.contains(e.target)) {
    menu.classList.remove('visible');
    document.removeEventListener('click', _ocrCloseMenuOnce);
  }
}

function _ocrSetMode(mode) {
  _ocrMode = mode;
  const overlay = document.getElementById('ocrModal');
  if (!overlay) return;
  // The class goes on the inner .modal box, not the overlay: every rule in
  // review.css is written as `.modal.mode-extract ...`. Putting it on the
  // overlay (which is what #ocrModal is) left the stylesheet looking one
  // level below the class, so both menu options opened an identical modal.
  const modal = overlay.querySelector('.modal') || overlay;
  modal.classList.toggle('mode-extract', mode === 'extract');
  const header = modal.querySelector('.modal-header');
  if (header) {
    header.textContent = (mode === 'extract')
      ? 'Extract Subtitles from Video'
      : 'Read Frame Text';
  }
}

// ── Video geometry ────────────────────────────────────────────────────────

function _videoDisplayRect() {
  // Where the picture actually sits inside the <video> box. A video element
  // letterboxes when its aspect ratio differs from the element's, so screen
  // coordinates cannot be scaled naively — the bars would shift every
  // mapping by the letterbox offset.
  if (!player || !player.videoWidth) return null;
  const box = player.getBoundingClientRect();
  const scale = Math.min(box.width / player.videoWidth,
                         box.height / player.videoHeight);
  const w = player.videoWidth * scale;
  const h = player.videoHeight * scale;
  return {
    left:  box.left + (box.width  - w) / 2,
    top:   box.top  + (box.height - h) / 2,
    width: w, height: h, scale,
  };
}

function _screenToVideo(rectPx, disp) {
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const x = clamp((rectPx.left - disp.left) / disp.scale, 0, player.videoWidth);
  const y = clamp((rectPx.top  - disp.top)  / disp.scale, 0, player.videoHeight);
  const w = clamp(rectPx.width  / disp.scale, 0, player.videoWidth  - x);
  const h = clamp(rectPx.height / disp.scale, 0, player.videoHeight - y);
  return { x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) };
}

function _videoToScreen(region, disp) {
  return {
    left:   disp.left + region.x * disp.scale,
    top:    disp.top  + region.y * disp.scale,
    width:  region.w * disp.scale,
    height: region.h * disp.scale,
  };
}

// ── Region selector ───────────────────────────────────────────────────────

function startRegionSelection() {
  const disp = _videoDisplayRect();
  if (!disp) {
    showErrorDialog('Select Region', 'Load a video first.');
    return;
  }
  // The picker gets out of the way entirely: you are drawing on the picture.
  document.getElementById('ocrModal').classList.remove('visible');
  const overlay = document.getElementById('ocrRegionOverlay');
  overlay.classList.add('visible');

  // Start from whatever is already chosen, else the bottom band — where
  // burned-in subtitles almost always live.
  _ocrDrawRegion(_ocrRegion || _ocrBottomBand());

  let dragging = false, startX = 0, startY = 0;

  const onDown = (e) => {
    if (e.target.closest('.ocr-region-bar')) return;   // buttons are not canvas
    dragging = true;
    startX = e.clientX;
    startY = e.clientY;
    e.preventDefault();
  };
  const onMove = (e) => {
    if (!dragging) return;
    const d = _videoDisplayRect();
    const rect = {
      left:   Math.min(startX, e.clientX),
      top:    Math.min(startY, e.clientY),
      width:  Math.abs(e.clientX - startX),
      height: Math.abs(e.clientY - startY),
    };
    _ocrDrawRegion(_screenToVideo(rect, d));
  };
  const onUp = () => { dragging = false; };

  overlay.addEventListener('mousedown', onDown);
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);

  const onKey = (e) => { if (e.key === 'Escape') finish(false); };
  document.addEventListener('keydown', onKey);

  const finish = (accept) => {
    overlay.removeEventListener('mousedown', onDown);
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    document.removeEventListener('keydown', onKey);
    overlay.classList.remove('visible');
    document.getElementById('ocrModal').classList.add('visible');
    if (accept) _ocrApplyRegion();
  };

  document.getElementById('ocrRegionUse').onclick    = () => finish(true);
  document.getElementById('ocrRegionCancel').onclick = () => finish(false);
  document.getElementById('ocrRegionBottom').onclick = () => _ocrDrawRegion(_ocrBottomBand());
  document.getElementById('ocrRegionFull').onclick   = () =>
    _ocrDrawRegion({ x: 0, y: 0, w: player.videoWidth, h: player.videoHeight });
}

function _ocrBottomBand() {
  // Bottom ~26% of frame height, inset slightly from the edges. A starting
  // guess, not a rule — the user drags from here.
  const vw = player.videoWidth, vh = player.videoHeight;
  return {
    x: Math.round(vw * 0.02),
    y: Math.round(vh * 0.72),
    w: Math.round(vw * 0.96),
    h: Math.round(vh * 0.26),
  };
}

let _ocrPendingRegion = null;

function _ocrDrawRegion(region) {
  _ocrPendingRegion = region;
  const disp = _videoDisplayRect();
  if (!disp) return;
  const box = document.getElementById('ocrRegionBox');
  const s = _videoToScreen(region, disp);
  box.style.left   = `${s.left}px`;
  box.style.top    = `${s.top}px`;
  box.style.width  = `${s.width}px`;
  box.style.height = `${s.height}px`;
  document.getElementById('ocrRegionSize').textContent =
    `${region.w} × ${region.h} px at (${region.x}, ${region.y})`;
}

function _ocrApplyRegion() {
  if (!_ocrPendingRegion || _ocrPendingRegion.w < 8 || _ocrPendingRegion.h < 8) return;
  _ocrRegion = _ocrPendingRegion;
  document.getElementById('ocrRegionCoords').textContent =
    `${_ocrRegion.w} × ${_ocrRegion.h} px at (${_ocrRegion.x}, ${_ocrRegion.y})`;

  // Preview the exact pixels that will be read, cropped from the current
  // frame. Cheaper to look at than to reason about coordinates.
  try {
    const canvas = document.createElement('canvas');
    canvas.width  = _ocrRegion.w;
    canvas.height = _ocrRegion.h;
    canvas.getContext('2d').drawImage(
      player, _ocrRegion.x, _ocrRegion.y, _ocrRegion.w, _ocrRegion.h,
      0, 0, _ocrRegion.w, _ocrRegion.h);
    document.getElementById('ocrRegionPreview').src = canvas.toDataURL('image/png');
  } catch { /* preview is a convenience, not a requirement */ }
}

// ── Extract settings + request ────────────────────────────────────────────

function openExtractModal() {
  if (!player || !player.videoWidth) {
    showErrorDialog('Extract Subtitles', 'Load a video first.');
    return;
  }
  _ocrSetMode('extract');
  _ocrLoadLanguages();
  _ocrApplyTranslateDefault();

  const startEl = document.getElementById('ocrRangeStart');
  const endEl   = document.getElementById('ocrRangeEnd');
  if (!startEl.value) startEl.value = formatTimeForInput(0);
  if (!endEl.value && isFinite(player.duration)) {
    endEl.value = formatTimeForInput(player.duration);
  }
  if (!_ocrRegion) _ocrDrawRegion(_ocrBottomBand()), _ocrApplyRegion();

  _ocrSetStatus('');
  document.getElementById('ocrModal').classList.add('visible');
}

function _ocrExtractRequest() {
  // The contract the backend will implement. Kept here, in one place, so the
  // UI and the server cannot drift apart silently once it exists.
  const start = parseTimeInput(document.getElementById('ocrRangeStart').value);
  const endRaw = document.getElementById('ocrRangeEnd').value.trim();
  const end = endRaw ? parseTimeInput(endRaw) : (player.duration || 0);

  return {
    video_path:      normalizeFullPath((videoPathInput && videoPathInput.value) || ''),
    region:          _ocrRegion,
    language:        document.getElementById('ocrLanguage').value,
    translate:       document.getElementById('ocrTranslate').checked,
    target_language: _ocrTargetLanguage(),
    sample_fps:      Number(document.getElementById('ocrSampleFps').value) || 2,
    start_time:      isNaN(start) ? 0 : start,
    end_time:        isNaN(end) ? null : end,
    min_duration_s:  Number(document.getElementById('ocrMinDuration').value) || 0,
    similarity:      (Number(document.getElementById('ocrSimilarity').value) || 85) / 100,
    existing:        document.getElementById('ocrExtractMode').value,
  };
}

function _ocrGenerate() {
  if (!_ocrRegion) {
    showErrorDialog('Extract Subtitles',
      'Select the subtitle region on the video first.');
    return;
  }
  const req = _ocrExtractRequest();
  if (!req.video_path) {
    showErrorDialog('Extract Subtitles',
      'No video path available. Load the video by path rather than by drag-and-drop.');
    return;
  }
  if (req.end_time != null && req.end_time <= req.start_time) {
    showErrorDialog('Invalid Range', 'The end time must be after the start time.');
    return;
  }

  // Backend not built yet. Showing the assembled request is more useful than
  // a disabled button: it makes the contract inspectable, and it verifies the
  // UI produces sane values before anything consumes them.
  const samples = Math.round(
    ((req.end_time ?? 0) - req.start_time) * req.sample_fps);
  console.log('[ocr:extract] request', req);
  showErrorDialog('Not Built Yet',
    `The extraction backend is not implemented yet — this is the request the ` +
    `UI would send:<br><br>` +
    `<span style="font-family: var(--font-mono); font-size: 12px;">` +
    `region ${req.region.w}×${req.region.h} at (${req.region.x}, ${req.region.y})<br>` +
    `language ${req.language}${req.translate ? ' → ' + req.target_language : ''}<br>` +
    `${req.sample_fps} fps over ${fmtTime(req.start_time)}–${fmtTime(req.end_time ?? 0)} ` +
    `(~${samples.toLocaleString()} samples)<br>` +
    `min cue ${req.min_duration_s}s, similarity ${Math.round(req.similarity * 100)}%<br>` +
    `existing lines: ${req.existing}</span><br><br>` +
    `The full object is in the browser console.`);
}

// ── Wiring (mode menu, region, generate) ──────────────────────────────────

(function _ocrWireExtract() {
  const menu = document.getElementById('ocrMenu');
  if (!menu) return;

  menu.querySelectorAll('[data-ocr-mode]').forEach(btn => {
    btn.addEventListener('click', () => {
      _ocrToggleMenu(false);
      const mode = btn.dataset.ocrMode;
      if (mode === 'extract') openExtractModal();
      else { _ocrSetMode('frame'); openOcrModal(); }
    });
  });

  const on = (id, ev, fn) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener(ev, fn);
  };
  on('ocrPickRegion', 'click', startRegionSelection);
  on('ocrGenerate',   'click', _ocrGenerate);
})();
