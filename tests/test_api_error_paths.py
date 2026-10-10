"""Request-level error paths for every route in gensrt/api.

The routes had tests for their happy paths and this week's server bugs
(favicon 500 loop, tuple passed to Path(), wrong OCR error code) lived on
the error paths.  Each test here sends a malformed, missing or busy request
and checks three things: the status code, that the body is JSON, and that
the message names the problem.  Nothing heavy runs — ASR, ffmpeg and the
translators are stubbed or never reached.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gensrt.api import _state, jobs


@pytest.fixture
def client():
    from gensrt.server import app
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def busy():
    """Hold the single-operation gate for the duration of a test."""
    _state._begin_long_operation("other.mp4")
    try:
        yield
    finally:
        _state._end_long_operation()


def _err(resp):
    body = resp.get_json()
    assert body is not None, "error responses must be JSON"
    return body.get("message") or body.get("error") or str(body.get("errors"))


# ── app-level ──────────────────────────────────────────────────────────────

def test_unknown_api_route_is_json_404_and_pages_stay_html(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404 and r.is_json and r.get_json()["status"] == 404
    r = client.get("/does-not-exist")
    assert r.status_code == 404 and not r.is_json


def test_wrong_method_on_an_api_route_is_json_405(client):
    r = client.get("/api/transcribe")
    assert r.status_code == 405 and r.is_json


def test_status_and_favicon(client):
    assert client.get("/api/status").get_json()["status"] == "ok"
    assert client.get("/favicon.ico").status_code == 204


# ── jobs ───────────────────────────────────────────────────────────────────

def test_transcribe_rejects_missing_or_absent_input(client, tmp_path):
    r = client.post("/api/transcribe", json={})
    assert r.status_code == 400 and "input_path" in _err(r)
    r = client.post("/api/transcribe", json={"input_path": str(tmp_path / "gone.mp4")})
    assert r.status_code == 400 and "not found" in _err(r).lower()
    r = client.post("/api/transcribe", data="not json", content_type="text/plain")
    assert r.status_code == 400


def test_transcribe_rejects_a_removed_engine_before_any_work(client, tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    monkeypatch.setattr(jobs, "run_transcription", lambda **kw: pytest.fail("pipeline must not run"))
    r = client.post("/api/transcribe", json={"input_path": str(video), "translation_engine": "google"})
    assert r.status_code == 400 and "google" in _err(r).lower()


def test_transcribe_is_409_while_another_job_runs(client, tmp_path, busy):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    r = client.post("/api/transcribe", json={"input_path": str(video)})
    assert r.status_code == 409 and "other.mp4" in _err(r)


def test_transcribe_pipeline_failure_is_500_and_releases_the_gate(client, tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")

    def boom(**kw):
        raise RuntimeError("CUDA fell over")
    monkeypatch.setattr(jobs, "run_transcription", boom)
    r = client.post("/api/transcribe", json={"input_path": str(video), "no_translate": True})
    assert r.status_code == 500 and "CUDA fell over" in _err(r)
    assert client.get("/api/operation_status").get_json()["status"] == "idle"


def test_transcribe_success_reports_translation_error(client, tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")

    class R:
        translation_error = "CUDA out of memory"
    monkeypatch.setattr(jobs, "run_transcription", lambda **kw: R())
    r = client.post("/api/transcribe", json={"input_path": str(video), "no_translate": True})
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok" and body["translation_error"] == "CUDA out of memory"
    assert body["output"].endswith(".srt")


def test_operation_status_shapes(client, busy):
    body = client.get("/api/operation_status").get_json()
    assert body["status"] == "active"
    op = body["operation"]
    assert op["filename"] == "other.mp4" and 0 <= op["percent"] <= 100


def test_burn_input_validation(client, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    srt = tmp_path / "v.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
    cases = [
        ({}, "video_path"),
        ({"video_path": str(video)}, "srt_path"),
        ({"video_path": str(tmp_path / "gone.mp4"), "srt_path": str(srt)}, "video_path"),
        ({"video_path": str(video), "srt_path": str(tmp_path / "gone.srt")}, "srt_path"),
        ({"video_path": str(video), "srt_path": str(video)}, ".srt"),
    ]
    for body, needle in cases:
        r = client.post("/api/burn", json=body)
        assert r.status_code == 400, body
        assert needle in _err(r), (body, _err(r))


def test_burn_is_fire_and_forget_and_versions_the_output_name(client, tmp_path, monkeypatch):
    """Burn is not gated (it is a spawned ffmpeg, not a pipeline run), and a
    second burn must not clobber the first's output."""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    srt = tmp_path / "v.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
    (tmp_path / "v_subbed.mp4").write_bytes(b"earlier burn")
    spawned = []

    class FakeProc:
        pid = 4242
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda cmd, **kw: spawned.append(cmd) or FakeProc())
    r = client.post("/api/burn", json={"video_path": str(video), "srt_path": str(srt)})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["pid"] == 4242 and body["output_path"].endswith("v_subbed_1.mp4")
    assert spawned and str(video) in spawned[0]


def test_burn_reports_a_missing_ffmpeg_as_500(client, tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    srt = tmp_path / "v.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
    import gensrt.ffmpeg_util as fu
    monkeypatch.setattr(fu, "get_ffmpeg_exe", lambda: str(tmp_path / "no-ffmpeg"))
    r = client.post("/api/burn", json={"video_path": str(video), "srt_path": str(srt)})
    assert r.status_code == 500 and "ffmpeg" in _err(r).lower()
    assert client.get("/api/operation_status").get_json()["status"] == "idle"


# ── config ─────────────────────────────────────────────────────────────────

def test_config_post_rejects_non_object_unknown_key_and_bad_type(client):
    r = client.post("/api/config", data="[]", content_type="application/json")
    assert r.status_code == 400
    r = client.post("/api/config", json={"no_such_key": 1})
    assert r.status_code == 400 and "no_such_key" in _err(r)
    r = client.post("/api/config", json={"vad_threshold": "high"})
    assert r.status_code == 400 and "vad_threshold" in _err(r)
    r = client.post("/api/config", json={"translation_engine": "google"})
    assert r.status_code == 400 and "translation_engine" in _err(r)


def test_config_get_and_engines_and_known_models_are_json(client):
    assert isinstance(client.get("/api/config").get_json(), dict)
    body = client.get("/api/engines").get_json()
    assert "engines" in body or isinstance(body, (list, dict))
    assert "models" in client.get("/api/known_models").get_json()


def test_validate_model_and_add_known_model_reject_empty_names(client):
    r = client.post("/api/validate_model", json={})
    assert r.status_code == 400 and "empty" in _err(r).lower()
    r = client.post("/api/add_known_model", json={"name": "   "})
    assert r.status_code == 400 and "empty" in _err(r).lower()


# ── media ──────────────────────────────────────────────────────────────────

def test_srt_get_error_paths(client, tmp_path):
    r = client.get("/api/srt")
    assert r.status_code == 400 and "path" in _err(r)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    r = client.get(f"/api/srt?path={video}")
    assert r.status_code == 400 and ".srt" in _err(r)
    r = client.get(f"/api/srt?path={tmp_path / 'gone.srt'}")
    assert r.status_code == 404
    r = client.get(f"/api/srt?video={video}")
    assert r.status_code == 404 and "sidecar" in _err(r).lower()
    bad = tmp_path / "bad.srt"
    bad.write_text("this is not an srt\n--> nope\n")
    r = client.get(f"/api/srt?path={bad}")
    assert r.status_code == 400 and "parse" in _err(r).lower()
    r = client.get(f"/api/srt?path={tmp_path / 'x.exe'}")
    assert r.status_code == 400


def test_srt_post_error_paths(client, tmp_path):
    dest = tmp_path / "out.srt"
    cases = [
        ({}, "path"),
        ({"path": str(dest)}, "segments"),
        ({"path": str(tmp_path / "out.txt"), "segments": []}, ".srt"),
        ({"path": str(tmp_path / "nodir" / "out.srt"), "segments": []}, "directory"),
        ({"path": str(dest), "segments": [{"start_time": "x", "end_time": 1, "text": ""}]}, "numeric"),
        ({"path": str(dest), "segments": [{"start_time": 2, "end_time": 1, "text": ""}]}, "after"),
    ]
    for body, needle in cases:
        r = client.post("/api/srt", json=body)
        assert r.status_code == 400, body
        assert needle in _err(r), (body, _err(r))
    assert not dest.exists()


def test_srt_post_backs_up_then_writes(client, tmp_path):
    dest = tmp_path / "out.srt"
    dest.write_text("old")
    r = client.post("/api/srt", json={"path": str(dest),
                                      "segments": [{"start_time": 0, "end_time": 1, "text": "new"}]})
    assert r.status_code == 200
    assert "new" in dest.read_text() and (tmp_path / "out.srt.bak").read_text() == "old"


def test_video_info_and_media_error_paths(client, tmp_path):
    r = client.get("/api/video_info")
    assert r.status_code == 400
    r = client.get(f"/api/video_info?path={tmp_path / 'gone.mp4'}")
    assert r.status_code == 400 and "not found" in _err(r).lower()
    r = client.get("/api/media")
    assert r.status_code == 400
    r = client.get(f"/api/media?path={tmp_path / 'gone.mp4'}")
    assert r.status_code == 400
    r = client.get(f"/api/media?path={tmp_path / 'x.exe'}")
    assert r.status_code == 400
    video = tmp_path / "v.mp4"
    video.write_bytes(b"0123456789")
    r = client.get(f"/api/media?path={video}", headers={"Range": "items=0-1"})
    assert r.status_code == 416
    r = client.get(f"/api/media?path={video}", headers={"Range": "bytes=2-4"})
    assert r.status_code == 206 and r.data == b"234" and r.headers["Content-Range"] == "bytes 2-4/10"
    r = client.get(f"/api/media?path={video}", headers={"Range": "bytes=8-"})
    assert r.status_code == 206 and r.data == b"89"


def test_video_info_reports_missing_ffprobe_as_500(client, tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    import gensrt.ffmpeg_util as fu
    monkeypatch.setattr(fu, "get_ffprobe_exe", lambda: str(tmp_path / "no-ffprobe"))
    r = client.get(f"/api/video_info?path={video}")
    assert r.status_code == 500 and "ffprobe" in _err(r).lower()


# ── ocr ────────────────────────────────────────────────────────────────────

def test_ocr_extract_error_paths(client, tmp_path):
    r = client.post("/api/ocr/extract", json={})
    assert r.status_code == 400 and "video_path" in _err(r)
    r = client.post("/api/ocr/extract", json={"video_path": str(tmp_path / "gone.mp4")})
    assert r.status_code == 404
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    r = client.post("/api/ocr/extract", json={"video_path": str(video), "region": {"x": 1}})
    assert r.status_code == 400 and "region" in _err(r)


def test_ocr_extract_is_409_while_another_job_runs(client, tmp_path, busy, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    r = client.post("/api/ocr/extract", json={"video_path": str(video)})
    assert r.status_code == 409


def test_ocr_cancel_is_always_accepted(client):
    assert client.post("/api/ocr/extract/cancel").get_json()["status"] == "cancelling"


def test_ocr_read_frame_error_paths(client):
    r = client.post("/api/ocr", json={})
    assert r.status_code == 400 and "image" in _err(r)
    r = client.post("/api/ocr", json={"image": "data:image/png;base64,@@@"})
    assert r.status_code == 400 and "base64" in _err(r)


def test_srt_get_reads_a_file_and_finds_its_sibling_video(client, tmp_path):
    (tmp_path / "movie.mp4").write_bytes(b"\x00")
    srt = tmp_path / "movie.ko.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n안녕\n")
    body = client.get(f"/api/srt?path={srt}").get_json()
    assert body["segments"][0]["text"] == "안녕"
    assert Path(body["sibling_video"]).name == "movie.mp4"
