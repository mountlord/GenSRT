// ── Startup ───────────────────────────────────────────────
checkServerMode();

// ── Auto-detect Helper ────────────────────────────────────
// If no chapters are loaded, warn the user that detection will run first
// (overwriting any existing project JSON), then proceed into the extract operation.
async function _autoDetectThenRun(extractFn) {
  if (chaptersArr.length === 0) {
    const ok = await showStyledConfirm(
      'Run Detection First?',
      'No chapters are loaded.<br><br>' +
      'Detection will run first. Any existing project JSON for this video will be overwritten.<br><br>' +
      'Continue?'
    );
    if (!ok) return;
    await callDetectAPI();
    if (chaptersArr.length === 0) return;
    chaptersArr.forEach((ch, i) => {
      chapterSelections[i] = ch.has_seams === true || (ch.seam_count > 0);
    });
    closeProgressModal();
    await new Promise(r => setTimeout(r, 80));
  }
  await extractFn();
}

// ── Action Button Wiring ──────────────────────────────────
detectBtn.addEventListener('click', () => {
  if (isServerMode) {
    callDetectAPI();
  } else {
    showErrorDialog('Server Mode Required',
      'Detect requires server mode. Run: <span style="font-family: var(--font-mono);">tilester</span>');
  }
});

// ── Save The Children ─────────────────────────────────────
const stcBtn = document.getElementById('stcBtn');
if (stcBtn) {
  stcBtn.addEventListener('click', () => {
    const url = 'https://www.savethechildren.org/us/ways-to-give';
    if (isPyWebView && window.pywebview && window.pywebview.api && typeof window.pywebview.api.open_url === 'function') {
      window.pywebview.api.open_url(url);
    } else {
      window.open(url, '_blank');
    }
  });
}
