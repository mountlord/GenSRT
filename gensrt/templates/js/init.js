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

// ── Burn Subtitles (fire-and-forget ffmpeg) ───────────────
const burnBtn = document.getElementById('burnBtn');
if (burnBtn) {
  burnBtn.addEventListener('click', async () => {
    if (!isServerMode) {
      showErrorDialog('Server Mode Required',
        'Burn requires server mode. Run: <span style="font-family: var(--font-mono);">gensrt</span>');
      return;
    }
    const videoPath = (videoPathInput && videoPathInput.value || '').trim();
    const srtPath   = (typeof currentProjectPath !== 'undefined') ? (currentProjectPath || '') : '';
    if (!videoPath || !srtPath) {
      showErrorDialog('Burn Unavailable',
        'Burn needs both a video and a loaded SRT.<br><br>' +
        'Generate or load an SRT first, then try again.');
      return;
    }

    burnBtn.disabled = true;
    const origText   = burnBtn.textContent;
    burnBtn.textContent = 'Burning…';
    try {
      const resp = await fetch('/api/burn', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ video_path: videoPath, srt_path: srtPath }),
      });
      const j = await resp.json().catch(() => ({}));
      if (!resp.ok || j.status !== 'ok') {
        showErrorDialog('Burn Failed',
          (j && j.message) ? j.message : `HTTP ${resp.status}`);
        return;
      }
      // Fire-and-forget — the spawn succeeded; ffmpeg is now running in the
      // background.  We don't track its exit code.  User keeps working.
      // showInfoDialog (not showProgressSuccess) — the latter only updates
      // an already-visible modal's contents; we need the modal to actually
      // open here since there was no preceding processing phase.
      showInfoDialog('Burn started',
        (j.message || 'Burning subtitles in the background.') +
        (j.output_path
          ? `<br><br><small style="opacity:0.7;font-family:var(--font-mono);">${j.output_path}</small>`
          : ''));
    } catch (err) {
      console.error('Burn request failed:', err);
      showErrorDialog('Burn Failed', err.message || String(err));
    } finally {
      burnBtn.textContent = origText;
      // Re-evaluate state (will re-enable since video + SRT haven't changed).
      if (typeof updateButtonStates === 'function') updateButtonStates();
    }
  });
}

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
