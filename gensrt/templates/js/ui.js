// ── Progress Modal Helpers ────────────────────────────────
function formatETA(seconds) {
  if (!seconds || seconds <= 0 || !isFinite(seconds)) return '';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  if (mins > 0) return `ETA: ${mins}m ${secs}s`;
  return `ETA: ${secs}s`;
}

function formatElapsed(seconds) {
  if (seconds == null || seconds < 0 || !isFinite(seconds)) return '';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  if (mins > 0) return `${mins}m ${secs}s`;
  return `${secs}s`;
}

function stopProgressPolling() {
  if (progressPollTimer) {
    try { clearInterval(progressPollTimer); } catch {}
    progressPollTimer = null;
  }
  progressPollInFlight = false;
}

async function pollOperationStatusOnce(expectedKind = '') {
  if (progressPollInFlight) return;
  progressPollInFlight = true;
  try {
    const response = await fetch('/api/operation_status', { cache: 'no-store' });
    const data = await response.json();
    if (!response.ok || !data || data.status !== 'active' || !data.operation) return;

    const op = data.operation;
    if (expectedKind && op.kind && op.kind !== expectedKind) return;

    const percent = Math.max(progressLastPercent || 0, Math.min(99, Math.round(Number(op.percent || 0))));
    const message = op.message || 'Working...';
    const current = Number.isFinite(Number(op.current)) ? Number(op.current) : 0;
    const total   = Number.isFinite(Number(op.total))   ? Number(op.total)   : 0;
    updateProgress(percent, message, current, total);
  } catch (err) {
    console.warn('Operation status poll failed:', err);
  } finally {
    progressPollInFlight = false;
  }
}

function startProgressPolling(expectedKind = '') {
  stopProgressPolling();
  progressPollTimer = setInterval(() => {
    void pollOperationStatusOnce(expectedKind);
  }, 250);
  void pollOperationStatusOnce(expectedKind);
}

function showProgressModal(title) {
  progressTitle.textContent   = title;
  progressMessage.textContent = 'Starting...';
  progressBar.style.width     = '0%';
  progressPercent.textContent = '0%';
  if (progressElapsed) progressElapsed.textContent = '';
  progressETA.textContent = '';

  progressStartTime   = Date.now();
  progressLastPercent = 0;
  progressLastETA     = '';

  // Tick elapsed wall-clock time even if SSE events are sparse
  if (progressElapsedTimer) { try { clearInterval(progressElapsedTimer); } catch {} }
  progressElapsedTimer = setInterval(() => {
    if (!progressStartTime) return;
    const elapsedS  = (Date.now() - progressStartTime) / 1000;
    const elapsedTxt = formatElapsed(elapsedS);
    if (progressElapsed) progressElapsed.textContent = elapsedTxt ? `Elapsed: ${elapsedTxt}` : '';
    progressETA.textContent = progressLastETA || '';
  }, 250);

  progressProcessing.style.display = 'block';
  progressResult.style.display     = 'none';
  progressModal.classList.add('visible');
}

function updateProgress(percent, message = '', current = 0, total = 0) {
  progressBar.style.width     = percent + '%';
  progressPercent.textContent = percent + '%';
  if (message) progressMessage.textContent = message;

  const elapsedS  = progressStartTime ? ((Date.now() - progressStartTime) / 1000) : 0;
  const elapsedTxt = formatElapsed(elapsedS);

  let etaTxt = '';
  if (progressStartTime && percent > 0 && percent < 100) {
    const rate      = percent / Math.max(0.001, elapsedS);
    const remaining = 100 - percent;
    etaTxt = formatETA(remaining / rate);
  }

  progressLastPercent = percent;
  progressLastETA     = etaTxt;
  if (progressElapsed) progressElapsed.textContent = elapsedTxt ? `Elapsed: ${elapsedTxt}` : '';
  progressETA.textContent = etaTxt || '';
}

function showProgressSuccess(title, message) {
  if (progressElapsedTimer) { try { clearInterval(progressElapsedTimer); } catch {} progressElapsedTimer = null; }
  stopProgressPolling();
  progressTitle.textContent          = title;
  progressResultIcon.textContent     = '✓';
  progressResultIcon.style.color     = '#10b981';
  progressResultMessage.innerHTML    = message;
  progressProcessing.style.display   = 'none';
  progressResult.style.display       = 'block';
}

function showProgressError(title, message) {
  if (progressElapsedTimer) { try { clearInterval(progressElapsedTimer); } catch {} progressElapsedTimer = null; }
  stopProgressPolling();
  progressTitle.textContent          = title;
  progressResultIcon.textContent     = '✗';
  progressResultIcon.style.color     = '#ef4444';
  progressResultMessage.innerHTML    = message;
  progressProcessing.style.display   = 'none';
  progressResult.style.display       = 'block';
}

function closeProgressModal() {
  if (progressElapsedTimer) { try { clearInterval(progressElapsedTimer); } catch {} progressElapsedTimer = null; }
  stopProgressPolling();
  progressModal.classList.remove('visible');
}

progressCloseBtn.addEventListener('click', closeProgressModal);

// ── Inline Dialogs (avoid native alert/confirm in pywebview) ─────────
function showErrorDialog(title, htmlMessage) {
  showProgressModal(title || 'Error');
  showProgressError(title || 'Error', htmlMessage);
}

function showInfoDialog(title, htmlMessage) {
  showProgressModal(title || 'Info');
  showProgressSuccess(title || 'Info', htmlMessage);
}

// ── Styled Confirm Dialog (replaces native confirm()) ─────────────────
function showStyledConfirm(title, htmlMessage) {
  return new Promise(resolve => {
    confirmModalTitle.textContent   = title;
    confirmModalMessage.innerHTML   = htmlMessage;
    confirmModal.classList.add('visible');

    function cleanup(result) {
      confirmModal.classList.remove('visible');
      confirmModalOk.removeEventListener('click', onOk);
      confirmModalCancel.removeEventListener('click', onCancel);
      resolve(result);
    }
    const onOk     = () => cleanup(true);
    const onCancel = () => cleanup(false);
    confirmModalOk.addEventListener('click', onOk);
    confirmModalCancel.addEventListener('click', onCancel);
  });
}

// Legacy wrapper — kept for any callers using native confirm() pattern
function showConfirmDialog(message, callback) {
  if (confirm(message)) callback();
}

// ── Draggable modals ──────────────────────────────────────
//
// Modals are centred by flex on their overlay, which is fine until the modal
// covers the thing you are trying to look at — choosing OCR regions means
// wanting to see the video frame behind the picker.
//
// Dragging uses left/top offsets rather than a CSS transform on purpose:
// .modal already animates transform (modalSlideIn), and a transform written
// by script fights that animation for the first 300 ms.  Offsets are
// animation-neutral.
//
// Only the header drags, so text selection inside inputs and textareas is
// untouched.
function makeModalDraggable(modal) {
  const header = modal.querySelector('.modal-header');
  if (!header || header.dataset.draggable === '1') return;
  header.dataset.draggable = '1';
  header.style.cursor = 'move';
  header.title = 'Drag to move';

  let startX = 0, startY = 0, baseX = 0, baseY = 0, dragging = false;

  const clampIntoView = () => {
    // Keep at least a corner reachable after a window resize, otherwise a
    // modal dragged to an edge can end up unreachable.
    const r = modal.getBoundingClientRect();
    let dx = parseFloat(modal.style.left || '0');
    let dy = parseFloat(modal.style.top  || '0');
    if (r.right  < 80)                  dx += 80 - r.right;
    if (r.left   > window.innerWidth  - 80) dx -= r.left - (window.innerWidth - 80);
    if (r.bottom < 40)                  dy += 40 - r.bottom;
    if (r.top    > window.innerHeight - 40) dy -= r.top - (window.innerHeight - 40);
    modal.style.left = `${dx}px`;
    modal.style.top  = `${dy}px`;
  };

  const onMove = (e) => {
    if (!dragging) return;
    modal.style.left = `${baseX + (e.clientX - startX)}px`;
    modal.style.top  = `${baseY + (e.clientY - startY)}px`;
  };

  const onUp = () => {
    if (!dragging) return;
    dragging = false;
    modal.classList.remove('dragging');
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    clampIntoView();
  };

  header.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    dragging = true;
    startX = e.clientX;
    startY = e.clientY;
    baseX  = parseFloat(modal.style.left || '0');
    baseY  = parseFloat(modal.style.top  || '0');
    modal.classList.add('dragging');
    modal.style.position = 'relative';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    e.preventDefault();          // no text-selection drag on the title
  });

  // Double-clicking the header re-centres — the escape hatch for a modal
  // dragged somewhere awkward.
  header.addEventListener('dblclick', () => {
    modal.style.left = '0px';
    modal.style.top  = '0px';
  });

  window.addEventListener('resize', clampIntoView);
}

function initDraggableModals() {
  document.querySelectorAll('.modal-overlay .modal').forEach(makeModalDraggable);
}
