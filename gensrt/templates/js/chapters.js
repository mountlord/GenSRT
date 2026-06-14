// ── Chapter Timeline Bar ──────────────────────────────────

// Inline HTML-escape used when injecting user-supplied SRT text into innerHTML.
function _escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

// Persistent loadedmetadata listener — fires whether the video was loaded
// via browser file drop or pywebview path. Rebuilds the timeline once duration is known.
player.addEventListener('loadedmetadata', () => {
  if (videoControls) videoControls.classList.add('video-ready');
  if (chaptersArr && chaptersArr.length) renderChapterTimeline(chaptersArr);
});

function renderChapterTimeline(chapters) {
  if (!chapterTimeline) return;
  // GenSRT no longer renders per-SRT-line markers on the play-position bar
  // (only the playhead needle is shown).  Clear any leftover segments from
  // earlier code paths and reposition the needle.
  Array.from(chapterTimeline.querySelectorAll('.chapter-timeline-seg')).forEach(el => el.remove());
  updateChapterTimelineNeedle();
}

function updateTimelineSegmentColor(chapterIndex, hasSeams) {
  if (!chapterTimeline) return;
  const seg = chapterTimeline.querySelector(`.chapter-timeline-seg[data-seg-index="${chapterIndex}"]`);
  if (!seg) return;
  seg.classList.toggle('seam', hasSeams);
  seg.classList.toggle('full', !hasSeams);
}

function updateChapterTimelineNeedle() {
  if (!chapterTimeline || !chapterTimelineNeedle) return;
  const duration = player.duration || 0;
  if (duration <= 0) return;
  const pct = Math.min(1, Math.max(0, (player.currentTime || 0) / duration));
  chapterTimelineNeedle.style.left = (pct * 100).toFixed(4) + '%';
}

// Seek on timeline click/drag
if (chapterTimeline) {
  function _timelineSeek(e) {
    const rect = chapterTimeline.getBoundingClientRect();
    const pct  = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    const duration = player.duration || 0;
    if (duration > 0) player.currentTime = pct * duration;
  }
  let _timelineDragging = false;
  chapterTimeline.addEventListener('mousedown', (e) => { _timelineDragging = true; _timelineSeek(e); });
  document.addEventListener('mousemove', (e)  => { if (_timelineDragging) _timelineSeek(e); });
  document.addEventListener('mouseup',   ()   => { _timelineDragging = false; });
}

// ── Active Row Tracking ───────────────────────────────────
function _findLastLE(arr, t, keyFn) {
  let lo = 0, hi = arr.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const v   = keyFn(arr[mid]);
    if (v <= t) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return ans;
}

function _setActive(el, kind) {
  if (!el) return;
  const prev = (kind === 'chapter') ? activeChapterEl : activeTransitionEl;
  if (prev === el) return;
  if (prev) prev.classList.remove('active');
  el.classList.add('active');
  try { el.scrollIntoView({ block: 'nearest' }); } catch (_) {}
  if (kind === 'chapter') activeChapterEl = el;
  else activeTransitionEl = el;
}

function updateActiveRow(t) {
  if (!linksData) return;
  if (chaptersArr.length && chapterEls.length === chaptersArr.length) {
    const i = _findLastLE(chaptersArr, t, ch => ch.start_time);
    if (i >= 0) {
      const ch = chaptersArr[i];
      if (t >= ch.start_time && t < ch.end_time) _setActive(chapterEls[i], 'chapter');
    }
  }
}

// ── Render Navigation ─────────────────────────────────────
function renderLinks(data) {
  linksBody.innerHTML = '';

  const chapters = (data.segments || data.chapters || []);

  chaptersArr       = chapters.map(ch => ({...ch}));
  chapterSelections = new Array(chapters.length).fill(false);
  chapterEls        = [];
  activeChapterEl   = null;

  // Sync the in-player subtitle <track> BEFORE the empty-state early-return.
  // resetProject() calls us with chapters=[], and we MUST clear the track
  // in that case — otherwise the previous video's cues linger in the
  // player overlay (pywebview/CEF doesn't always tear down the active-cue
  // layer when the player goes display:none).  See _refreshSubtitleTrack()
  // in project.js for the actual cue-teardown logic that runs in this path.
  if (typeof _refreshSubtitleTrack === 'function') _refreshSubtitleTrack();

  if (chapters.length === 0) {
    linksBody.innerHTML = `
      <div class="links-empty">
        <div class="icon">✅</div>
        <div class="label">No SRT lines<br><small>Generate or drop an .srt file</small></div>
      </div>`;
    linkCount.textContent = '0';
    return;
  }

  chapters.forEach((ch, i) => {
    const item = document.createElement('div');
    item.className    = 'link-item';
    item.dataset.kind  = 'chapter';
    item.dataset.index = i;

    const startFrameTag = fmtFrameTag(frameAtTime(chaptersArr[i].start_time));
    const endFrameTag   = fmtFrameTag(frameAtTime(chaptersArr[i].end_time));

    const checkbox        = document.createElement('input');
    checkbox.type         = 'checkbox';
    checkbox.className    = 'chapter-checkbox';
    checkbox.checked      = chapterSelections[i];
    checkbox.addEventListener('change', (e) => {
      e.stopPropagation();
      chapterSelections[i] = e.target.checked;
    });

    const editIcon        = document.createElement('div');
    editIcon.className    = 'edit-icon';
    editIcon.textContent  = '✏️';
    editIcon.title        = 'Edit SRT line';
    editIcon.addEventListener('click', (e) => { e.stopPropagation(); openChapterEditor(i); });

    const rowDiv = document.createElement('div');
    rowDiv.style.display = 'contents';
    rowDiv.innerHTML = `
      <div class="time">${fmtTime(chaptersArr[i].start_time)} - ${fmtTime(chaptersArr[i].end_time)}</div>
      <div class="label">${_escapeHtml(ch.text || ch.title || ('Line ' + (i + 1)))} <span style="opacity:0.6">(${fmtDuration(chaptersArr[i].end_time - chaptersArr[i].start_time)})</span></div>
      <div class="frame-tag">${startFrameTag} - ${endFrameTag}</div>
    `;

    item.appendChild(checkbox);
    item.appendChild(editIcon);
    item.appendChild(rowDiv);

    item.addEventListener('click', (e) => {
      if (e.target === checkbox || e.target === editIcon) return;
      // Single-select semantics: clicking a row replaces any existing
      // checkbox selection with just this row.  Multi-select is still
      // available via the row checkboxes themselves.
      for (let j = 0; j < chapterSelections.length; j++) chapterSelections[j] = (j === i);
      const allCbs = linksBody.querySelectorAll('.chapter-checkbox');
      allCbs.forEach((cb, j) => { cb.checked = (j === i); });
      // Seek to the line's start and pause (no auto-play on click).
      player.currentTime = chaptersArr[i].start_time;
      try { player.pause(); } catch {}
    });

    linksBody.appendChild(item);
    chapterEls.push(item);
  });

  linkCount.textContent = `${chapters.length}`;
  renderChapterTimeline(chapters);

  // Subtitle <track> sync happens at the top of this function (above the
  // empty-state early-return) so it fires regardless of whether the new
  // chapter list is empty or populated.
}

// ── Chapter Editor ────────────────────────────────────────
function formatTimeForInput(seconds) {
  const h   = Math.floor(seconds / 3600);
  const m   = Math.floor((seconds % 3600) / 60);
  const s   = seconds % 60;
  const ms  = Math.round((s % 1) * 1000);
  const sInt = Math.floor(s);
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(sInt).padStart(2,'0')}.${String(ms).padStart(3,'0')}`;
}

function parseTimeInput(timeStr) {
  const parts = timeStr.trim().split(':');
  if (parts.length !== 3) return NaN;
  const h     = parseInt(parts[0], 10);
  const m     = parseInt(parts[1], 10);
  const sParts = parts[2].split('.');
  const s     = parseInt(sParts[0], 10);
  const ms    = sParts[1] ? parseInt(sParts[1].padEnd(3, '0'), 10) : 0;
  if (isNaN(h) || isNaN(m) || isNaN(s) || isNaN(ms)) return NaN;
  return h * 3600 + m * 60 + s + ms / 1000;
}

function openChapterEditor(index) {
  if (!fps) {
    showErrorDialog('FPS Missing', 'FPS data not available. Please load a video first.');
    return;
  }
  editingChapterIndex    = index;
  const ch               = chaptersArr[index];
  editStartTime.value    = formatTimeForInput(ch.start_time);
  editEndTime.value      = formatTimeForInput(ch.end_time);
  editStartFrame.value   = Math.round(ch.start_time * fps);
  editEndFrame.value     = Math.round(ch.end_time   * fps);
  const editTextEl       = document.getElementById('editText');
  if (editTextEl) editTextEl.value = ch.text || ch.title || '';
  chapterEditorModal.classList.add('visible');
}

editStartTime.addEventListener('input', () => {
  const t = parseTimeInput(editStartTime.value);
  if (!isNaN(t) && fps) editStartFrame.value = Math.round(t * fps);
});
editEndTime.addEventListener('input', () => {
  const t = parseTimeInput(editEndTime.value);
  if (!isNaN(t) && fps) editEndFrame.value = Math.round(t * fps);
});

modalCancel.addEventListener('click', () => chapterEditorModal.classList.remove('visible'));

modalSave.addEventListener('click', () => {
  const startTime = parseTimeInput(editStartTime.value);
  const endTime   = parseTimeInput(editEndTime.value);
  if (isNaN(startTime) || isNaN(endTime)) {
    showErrorDialog('Invalid Time', 'Invalid time format. Use <strong>HH:MM:SS.mmm</strong>.');
    return;
  }
  if (startTime >= endTime) {
    showErrorDialog('Invalid Range', 'Start time must be before end time.');
    return;
  }
  const editTextEl = document.getElementById('editText');
  chaptersArr[editingChapterIndex].start_time = startTime;
  chaptersArr[editingChapterIndex].end_time   = endTime;
  if (editTextEl) chaptersArr[editingChapterIndex].text = editTextEl.value;
  renderLinks({ chapters: chaptersArr });
  chapterEditorModal.classList.remove('visible');
});

chapterEditorModal.addEventListener('click', (e) => {
  if (e.target === chapterEditorModal) chapterEditorModal.classList.remove('visible');
});

// ── Segment Operations ────────────────────────────────────
function _chapterHasSeams(ch) {
  try { return !!(ch && (ch.has_seams || ((ch.seam_count || 0) > 0))); } catch { return false; }
}

function _reindexSegments(segments) {
  segments.sort((a, b) => (a.start_time ?? 0) - (b.start_time ?? 0));
  segments.forEach((s, i) => {
    const newIndex = i + 1;
    s.index = newIndex;
    if (!s.title)                                          s.title = `Chapter ${newIndex}`;
    else if (/^Chapter\s+\d+$/i.test(String(s.title).trim())) s.title = `Chapter ${newIndex}`;
  });
}

function _ensureEditableProject() {
  linksData = normalizeProjectData(linksData || {});
  if (!linksData.schema_version) linksData.schema_version = 1;
  if (!Array.isArray(linksData.segments)) linksData.segments = [];

  const vp = normalizeFullPath((videoPathInput && videoPathInput.value) ? videoPathInput.value.trim() : '');
  if (vp) {
    linksData.video_path = vp;
    linksData.video      = basenameFromPath(vp);
  } else if (!linksData.video && videoFilename) {
    linksData.video = videoFilename;
  }

  if (!linksData.fps && fps) linksData.fps = fps;
  return linksData;
}

function _parseLenSeconds(val) {
  if (val == null) return null;
  const s = String(val).trim();
  if (!s) return null;
  const n = Number(s);
  if (!isFinite(n)) return null;
  return n;
}

function _setSegMarkArmed(armed, startTime = null) {
  if (!segMarkBtn) return;
  if (armed) {
    segMarkBtn.classList.add('armed');
    segMarkBtn.textContent = '▾';
    segMarkBtn.title       = 'End SRT line';
    if (segPendingDisplay && startTime != null) {
      segPendingDisplay.style.display = 'inline';
      segPendingDisplay.textContent   = `IN ${fmtTime(startTime)}`;
    }
  } else {
    segMarkBtn.classList.remove('armed');
    segMarkBtn.textContent = '▴';
    segMarkBtn.title       = 'Start SRT line';
    if (segPendingDisplay) { segPendingDisplay.style.display = 'none'; segPendingDisplay.textContent = ''; }
    if (segLenInput) segLenInput.value = '';
  }
}

function onSegMarkClick() {
  if (!player) return;
  if (!fps) tryDetectFpsFromPlayer();

  const t = Number(player.currentTime || 0);

  if (pendingChapterStart == null) {
    pendingChapterStart = t;
    _setSegMarkArmed(true, pendingChapterStart);
    return;
  }

  const start = pendingChapterStart;
  let end     = t;
  const val   = _parseLenSeconds(segLenInput ? segLenInput.value : null);
  if (val != null) end = start + val;

  if (!isFinite(end) || end <= start + 0.001) {
    showErrorDialog('Invalid Segment',
      `End time must be after start time.<br><br>` +
      `Start: <span style="font-family: var(--font-mono);">${fmtTime(start)}</span><br>` +
      `End: <span style="font-family: var(--font-mono);">${fmtTime(end)}</span>`);
    return;
  }

  const proj     = _ensureEditableProject();
  const segments = proj.segments;

  const seg = {
    index:      segments.length + 1,
    title:      `Chapter ${segments.length + 1}`,
    start_time: Number(start.toFixed(3)),
    end_time:   Number(end.toFixed(3)),
    has_seams:  false,
    seam_count: 0,
    manual:     true,
  };
  if (fps && isFinite(fps)) {
    seg.start_frame = frameAtTime(seg.start_time);
    seg.end_frame   = frameAtTime(seg.end_time);
  }

  segments.push(seg);
  _reindexSegments(segments);
  proj.segments       = segments;
  proj.schema_version = 1;
  if (!proj.fps && fps) proj.fps = fps;

  linksData = proj;
  renderLinks(proj);
  if (typeof updateButtonStates === 'function') updateButtonStates();

  pendingChapterStart = null;
  _setSegMarkArmed(false);
}

if (segMarkBtn) segMarkBtn.addEventListener('click', onSegMarkClick);

// Name kept for the api.js wiring (splitBtn click handler calls this).
// Selection logic: use the row checkbox(es) — chapterSelections[i] is the
// authoritative selection state.  Earlier behaviour scanned for the
// chapter whose timing contained player.currentTime, which ignored the
// user's explicit selection and felt random when no row was checked.
function splitSegmentAtPlayhead() {
  if (!chaptersArr || !chaptersArr.length) {
    showErrorDialog('Split', 'No SRT lines available to split.');
    return;
  }

  const selectedIndices = chapterSelections
    .map((sel, i) => (sel ? i : -1))
    .filter(i => i >= 0);

  if (selectedIndices.length === 0) {
    showErrorDialog('Split',
      'Select an SRT line to split first (tick the row checkbox).');
    return;
  }
  if (selectedIndices.length > 1) {
    showErrorDialog('Split',
      'Please select exactly one SRT line to split.');
    return;
  }

  const idx  = selectedIndices[0];
  const orig = chaptersArr[idx];
  const EPS  = 0.001;
  const dur  = Number(orig.end_time) - Number(orig.start_time);
  if (!isFinite(dur) || dur <= EPS * 2) {
    showErrorDialog('Split', 'Selected SRT line is too small to split.');
    return;
  }

  // Open the Split modal pre-filled.  start1 = original.start, end2 = original.end,
  // text1 = original.text.  end1 / start2 / text2 left blank for the user to fill in.
  _splitPendingIndex = idx;
  document.getElementById('splitStart1').value = formatTimeForInput(orig.start_time);
  document.getElementById('splitEnd1').value   = '';
  document.getElementById('splitText1').value  = orig.text || orig.title || '';
  document.getElementById('splitStart2').value = '';
  document.getElementById('splitEnd2').value   = formatTimeForInput(orig.end_time);
  document.getElementById('splitText2').value  = '';
  document.getElementById('splitModal').classList.add('visible');
}

// Track which row the Split modal is operating on (set when the modal opens).
let _splitPendingIndex = -1;

// Cancel — just hide the modal.
const _splitModal       = document.getElementById('splitModal');
const _splitModalCancel = document.getElementById('splitModalCancel');
const _splitModalSave   = document.getElementById('splitModalSave');
if (_splitModalCancel) _splitModalCancel.addEventListener('click', () => {
  _splitModal.classList.remove('visible');
  _splitPendingIndex = -1;
});
if (_splitModal) _splitModal.addEventListener('click', (e) => {
  if (e.target === _splitModal) { _splitModal.classList.remove('visible'); _splitPendingIndex = -1; }
});

// Save — validate per-half (start < end) only, then replace the row with two halves.
if (_splitModalSave) _splitModalSave.addEventListener('click', () => {
  const idx = _splitPendingIndex;
  if (idx < 0 || !chaptersArr[idx]) {
    showErrorDialog('Split', 'No SRT line is selected for splitting.');
    return;
  }

  const s1 = parseTimeInput(document.getElementById('splitStart1').value);
  const e1 = parseTimeInput(document.getElementById('splitEnd1').value);
  const s2 = parseTimeInput(document.getElementById('splitStart2').value);
  const e2 = parseTimeInput(document.getElementById('splitEnd2').value);
  const t1 = document.getElementById('splitText1').value;
  const t2 = document.getElementById('splitText2').value;

  if (isNaN(s1) || isNaN(e1) || isNaN(s2) || isNaN(e2)) {
    showErrorDialog('Invalid Time', 'All four times must be in <strong>HH:MM:SS.mmm</strong> format.');
    return;
  }
  if (s1 >= e1) { showErrorDialog('Invalid Range', 'Line 1: Start must be before End.'); return; }
  if (s2 >= e2) { showErrorDialog('Invalid Range', 'Line 2: Start must be before End.'); return; }

  const orig  = chaptersArr[idx];
  const left  = { ...orig, start_time: s1, end_time: e1, text: t1, manual: true };
  const right = { ...orig, start_time: s2, end_time: e2, text: t2, manual: true };

  if (fps && isFinite(fps)) {
    left.start_frame  = frameAtTime(left.start_time);
    left.end_frame    = frameAtTime(left.end_time);
    right.start_frame = frameAtTime(right.start_time);
    right.end_frame   = frameAtTime(right.end_time);
  }

  chaptersArr.splice(idx, 1, left, right);
  _reindexSegments(chaptersArr);

  const proj    = _ensureEditableProject();
  proj.segments = chaptersArr.map(ch => ({ ...ch }));
  linksData     = proj;
  renderLinks(proj);
  if (typeof updateButtonStates === 'function') updateButtonStates();

  _splitModal.classList.remove('visible');
  _splitPendingIndex = -1;
});

function mergeAdjacentSegments() {
  if (!chaptersArr || !chaptersArr.length)  { showErrorDialog('Merge', 'No SRT lines available to merge.');      return; }
  if (chaptersArr.length < 2)               { showErrorDialog('Merge', 'Need at least two SRT lines to merge.'); return; }

  const _seamCount = (ch) => { try { return Math.max(0, parseInt(ch && ch.seam_count || 0, 10) || 0); } catch { return 0; } };

  const selected = [];
  for (let i = 0; i < (chapterSelections ? chapterSelections.length : 0); i++) {
    if (chapterSelections[i]) selected.push(i);
  }

  let a = null, b = null;

  if (selected.length === 2) {
    selected.sort((x, y) => x - y);
    if (selected[1] !== selected[0] + 1) { showErrorDialog('Merge', 'Please select two adjacent SRT lines to merge.'); return; }
    a = selected[0]; b = selected[1];
  } else if (selected.length === 1) {
    a = selected[0];
    b = (a < chaptersArr.length - 1) ? a + 1 : (a > 0 ? a - 1 : null);
    if (b == null) { showErrorDialog('Merge', 'No adjacent SRT line found to merge.'); return; }
    if (b < a) { const tmp = a; a = b; b = tmp; }
  } else if (selected.length === 0) {
    const t   = Number(player && player.currentTime || 0);
    const EPS = 0.001;
    let idx   = -1;
    for (let i = 0; i < chaptersArr.length; i++) {
      const st = Number(chaptersArr[i].start_time);
      const en = Number(chaptersArr[i].end_time);
      if (isFinite(st) && isFinite(en) && t >= (st - EPS) && t <= (en + EPS)) { idx = i; break; }
    }
    if (idx < 0) { showErrorDialog('Merge', 'Could not determine which SRT line to merge. Select one (or two) lines first.'); return; }
    a = idx;
    b = (a < chaptersArr.length - 1) ? a + 1 : (a > 0 ? a - 1 : null);
    if (b == null) { showErrorDialog('Merge', 'No adjacent SRT line found to merge.'); return; }
    if (b < a) { const tmp = a; a = b; b = tmp; }
  } else {
    showErrorDialog('Merge', 'Please select at most two SRT lines to merge.');
    return;
  }

  if (b !== a + 1) { showErrorDialog('Merge', 'Please select two adjacent SRT lines to merge.'); return; }

  const left  = chaptersArr[a];
  const right = chaptersArr[b];
  if (!left || !right) { showErrorDialog('Merge', 'Invalid SRT line selection.'); return; }

  const merged       = { ...left };
  merged.start_time  = Number(left.start_time);
  merged.end_time    = Number(right.end_time);
  // Combine text from BOTH segments — without this the right segment's
  // content was silently discarded.  Newline separator preserves the
  // visual two-line layout (SRT and WebVTT both support multi-line cues);
  // user can flatten to a single line via the Edit modal if desired.
  const leftText  = String(left.text  || '').replace(/\s+$/, '');
  const rightText = String(right.text || '').replace(/^\s+/, '');
  if (leftText && rightText) {
    merged.text = leftText + '\n' + rightText;
  } else {
    merged.text = leftText || rightText;
  }
  if (fps && isFinite(fps)) {
    merged.start_frame = frameAtTime(merged.start_time);
    merged.end_frame   = frameAtTime(merged.end_time);
  }
  merged.has_seams  = !!(_chapterHasSeams(left) || _chapterHasSeams(right));
  merged.seam_count = Math.max(_seamCount(left), _seamCount(right));
  merged.manual     = true;

  chaptersArr.splice(a, 2, merged);
  _reindexSegments(chaptersArr);

  const proj    = _ensureEditableProject();
  proj.segments = chaptersArr.map(ch => ({ ...ch }));
  linksData     = proj;
  renderLinks(proj);
  if (typeof updateButtonStates === 'function') updateButtonStates();
}

function deleteSelectedSegments() {
  if (!chaptersArr || !chaptersArr.length) { showErrorDialog('Delete', 'No SRT lines available to delete.'); return; }

  const selected = [];
  for (let i = 0; i < (chapterSelections ? chapterSelections.length : 0); i++) {
    if (chapterSelections[i]) selected.push(i);
  }

  if (selected.length === 0) {
    const t   = Number(player && player.currentTime || 0);
    const EPS = 0.001;
    let idx   = -1;
    for (let i = 0; i < chaptersArr.length; i++) {
      const st = Number(chaptersArr[i].start_time);
      const en = Number(chaptersArr[i].end_time);
      if (isFinite(st) && isFinite(en) && t >= (st - EPS) && t <= (en + EPS)) { idx = i; break; }
    }
    if (idx < 0) { showErrorDialog('Delete', 'Could not determine which SRT line to delete. Select one (or more) lines first.'); return; }
    selected.push(idx);
  }

  const count = selected.length;
  const msg   = (count === 1) ? 'Remove the selected SRT line?' : `Remove ${count} selected SRT lines?`;

  const _doDelete = () => {
    selected.sort((a, b) => b - a);
    for (const i of selected) {
      if (i >= 0 && i < chaptersArr.length) chaptersArr.splice(i, 1);
    }
    _reindexSegments(chaptersArr);
    const proj    = _ensureEditableProject();
    proj.segments = chaptersArr.map(ch => ({ ...ch }));
    linksData     = proj;
    renderLinks(proj);
    if (typeof updateButtonStates === 'function') updateButtonStates();
  };

  showStyledConfirm('Remove SRT Line' + (count > 1 ? 's' : ''), msg).then(ok => {
    if (ok) _doDelete();
  });
}
