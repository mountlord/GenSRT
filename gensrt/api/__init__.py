"""HTTP routes for the GenSRT desktop window, one Flask blueprint per concern.

``server.py`` used to hold every route, the job state, config validation,
model validation, burn-in and the OCR translator cache in one 2,400-line
module; three of the five server bugs fixed in Sep 2026 lived there, and it
was the least-tested module per line.  The split is by *what the route is
about*, not by HTTP verb:

  api.jobs    /api/transcribe, /api/burn, /api/operation_status
  api.config  /api/config, /api/engines, /api/known_models,
              /api/validate_model, /api/add_known_model
  api.media   /, /static, /api/status, /api/srt, /api/video_info, /api/media
  api.ocr     /api/ocr, /api/ocr/languages, /api/ocr/extract[/cancel]

  api._state  the single-operation gate the job routes share
  api._paths  path validation and sidecar lookup every route goes through

URLs are unchanged.  ``server.py`` keeps the Flask app, the JSON error
handler, port discovery and ``launch_server``; it registers the blueprints
below and re-exports the names the CLI and tests import from it.
"""

from __future__ import annotations

from flask import Flask

from gensrt.api import config, jobs, media, ocr

#: Registration order is also the URL-map order; no two blueprints share a
#: rule, so it only affects how the map reads.
BLUEPRINTS = (media.bp, jobs.bp, config.bp, ocr.bp)


def register(app: Flask) -> None:
    """Attach every route blueprint to *app* (module-level imports above so
    PyInstaller's static analysis sees the route modules)."""
    for bp in BLUEPRINTS:
        app.register_blueprint(bp)
