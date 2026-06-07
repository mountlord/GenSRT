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
