"""Transport-stream rewrap for the review player (gensrt/remux.py) and its
routes.  The manager is exercised with a fake ffmpeg (a Python script that
emits -progress lines and writes the output) so the tests do not need a
real binary
one end-to-end test runs the real ffmpeg when it is present.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gensrt import remux

# ── pure parts ─────────────────────────────────────────────────────────────

def test_needs_remux_is_by_extension_case_insensitive():
    assert remux.needs_remux(Path("a.ts")) and remux.needs_remux(Path("A.M2TS"))
    assert not remux.needs_remux(Path("a.mp4")) and not remux.needs_remux(Path("a.mkv"))


def test_cache_key_changes_with_content_identity_not_just_path():
    p = Path("F:/Live/x.ts")
    a = remux.cache_key(p, 100, 1)
    assert a == remux.cache_key(p, 100, 1)
    assert a != remux.cache_key(p, 101, 1)            # size
    assert a != remux.cache_key(p, 100, 2)            # mtime
    assert a != remux.cache_key(Path("F:/Live/y.ts"), 100, 1)
    assert len(a) == 20 and a.isalnum()


def test_cache_key_survives_non_ascii_paths():
    assert remux.cache_key(Path("F:/Live/【한갱】Han Geng.ts"), 1, 1)


def test_build_command_is_a_stream_copy_into_seekable_mp4():
    cmd = remux.build_command("ffmpeg", Path("in.ts"), Path("out.mp4.part"))
    assert cmd[0] == "ffmpeg" and cmd[-1] == "out.mp4.part"
    assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy"
    assert "+faststart" in cmd[cmd.index("-movflags") + 1]
    assert "-progress" in cmd and "pipe:1" in cmd
    assert "aac_adtstoasc" not in cmd          # the mp4 muxer inserts it itself
    assert "libx264" not in cmd and "-crf" not in cmd


@pytest.mark.parametrize("line, dur, want", [
    ("out_time_us=3000000", 6.0, 0.5),
    ("out_time_ms=3000000", 6.0, 0.5),        # older ffmpeg: microseconds despite the name
    ("out_time_us=9000000", 6.0, 1.0),        # clamped
    ("out_time_us=-1", 6.0, None),
    ("out_time_us=abc", 6.0, None),
    ("frame=12", 6.0, None),
    ("out_time_us=3000000", None, None),      # no duration → no fraction
    ("out_time_us=3000000", 0, None),
])
def test_parse_progress(line, dur, want):
    assert remux.parse_progress(line, dur) == want


# ── manager with a fake ffmpeg ─────────────────────────────────────────────

FAKE_FFMPEG = r'''
import sys, time
args = sys.argv[1:]
dst = args[-1]
mode = __import__("os").environ.get("FAKE_MODE", "ok")
if mode == "fail":
    sys.stderr.write("in.ts: Invalid data found when processing input\n")
    sys.exit(1)
for us in (2000000, 4000000, 6000000):
    sys.stdout.write(f"out_time_us={us}\nprogress=continue\n")
    sys.stdout.flush()
    time.sleep(0.05 if mode != "slow" else 0.4)
with open(dst, "wb") as f:
    f.write(b"\x00\x00\x00\x20ftypisom" + b"x" * 100)
sys.stdout.write("progress=end\n")
'''

FAKE_FFPROBE = r'''
import sys
print("6.0")
'''


@pytest.fixture
def fake_bins(tmp_path):
    ff = tmp_path / "fake_ffmpeg.py"
    ff.write_text(FAKE_FFMPEG)
    fp = tmp_path / "fake_ffprobe.py"
    fp.write_text(FAKE_FFPROBE)
    # The manager invokes "<exe> args…"; a .py needs the interpreter, so
    # wrap it in a shim on PATH-less systems via a tiny launcher script.
    if sys.platform.startswith("win"):
        ffl = tmp_path / "ffmpeg.cmd"
        ffl.write_text(f'@"{sys.executable}" "{ff}" %*\n')
        fpl = tmp_path / "ffprobe.cmd"
        fpl.write_text(f'@"{sys.executable}" "{fp}" %*\n')
    else:
        ffl = tmp_path / "ffmpeg"
        ffl.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{ff}" "$@"\n')
        ffl.chmod(0o755)
        fpl = tmp_path / "ffprobe"
        fpl.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fp}" "$@"\n')
        fpl.chmod(0o755)
    return str(ffl), str(fpl)


def _wait(m, src, timeout=5.0):
    t0 = time.time()
    while m.status(src).state in ("preparing",) and time.time() - t0 < timeout:
        time.sleep(0.02)
    return m.status(src)


def test_prepare_runs_once_reports_progress_and_caches(tmp_path, fake_bins):
    src = tmp_path / "rec.ts"
    src.write_bytes(b"\x47" * 188 * 10)
    m = remux.RemuxManager(directory=tmp_path / "cache", ffmpeg=fake_bins[0], ffprobe=fake_bins[1])

    assert m.status(src).state == "unprepared"
    first = m.prepare(src)
    assert first.state == "preparing"
    assert m.prepare(src) is first                     # second call joins, does not restart
    st = _wait(m, src)
    assert st.state == "ready" and st.output and st.output.is_file()
    assert st.output.suffix == ".mp4" and not list((tmp_path / "cache").glob("*.part"))
    # Cached: a fresh manager on the same directory is ready at once.
    m2 = remux.RemuxManager(directory=tmp_path / "cache", ffmpeg="/nonexistent", ffprobe="/nonexistent")
    assert m2.status(src).state == "ready" and m2.prepare(src).state == "ready"


def test_changed_source_gets_a_fresh_remux(tmp_path, fake_bins):
    src = tmp_path / "rec.ts"
    src.write_bytes(b"\x47" * 188)
    m = remux.RemuxManager(directory=tmp_path / "cache", ffmpeg=fake_bins[0], ffprobe=fake_bins[1])
    m.prepare(src)
    a = _wait(m, src).output
    src.write_bytes(b"\x47" * 188 * 2)                 # re-downloaded: new size
    assert m.status(src).state == "unprepared"
    m.prepare(src)
    b = _wait(m, src).output
    assert a != b and a.is_file() and b.is_file()


def test_ffmpeg_failure_is_reported_and_retried_on_next_prepare(tmp_path, fake_bins, monkeypatch):
    src = tmp_path / "rec.ts"
    src.write_bytes(b"\x47" * 188)
    m = remux.RemuxManager(directory=tmp_path / "cache", ffmpeg=fake_bins[0], ffprobe=fake_bins[1])
    monkeypatch.setenv("FAKE_MODE", "fail")
    m.prepare(src)
    st = _wait(m, src)
    assert st.state == "error" and "Invalid data" in st.error
    assert not list((tmp_path / "cache").iterdir())    # partial output removed
    assert m.status(src).state == "error"              # polling still sees it
    monkeypatch.setenv("FAKE_MODE", "ok")
    assert m.prepare(src).state == "preparing"         # a new request retries
    assert _wait(m, src).state == "ready"


def test_missing_ffmpeg_is_an_error_not_a_hang(tmp_path):
    src = tmp_path / "rec.ts"
    src.write_bytes(b"\x47" * 188)
    m = remux.RemuxManager(directory=tmp_path / "cache", ffmpeg="/no/such/ffmpeg", ffprobe="/no/such/ffprobe")
    m.prepare(src)
    st = _wait(m, src)
    assert st.state == "error" and "ffmpeg not found" in st.error


def test_cancel_stops_the_job_and_leaves_no_output(tmp_path, fake_bins, monkeypatch):
    src = tmp_path / "rec.ts"
    src.write_bytes(b"\x47" * 188)
    m = remux.RemuxManager(directory=tmp_path / "cache", ffmpeg=fake_bins[0], ffprobe=fake_bins[1])
    monkeypatch.setenv("FAKE_MODE", "slow")
    m.prepare(src)
    time.sleep(0.2)
    m.cancel(src)
    st = _wait(m, src)
    assert st.state == "error" and st.error == "cancelled"
    assert not list((tmp_path / "cache").glob("*.mp4"))


def test_sweep_evicts_oldest_over_cap_and_stale_partials(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    old = d / "a.mp4"
    old.write_bytes(b"x" * 300)
    new = d / "b.mp4"
    new.write_bytes(b"x" * 300)
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    stale = d / "c.mp4.part"
    stale.write_bytes(b"x" * 10)
    os.utime(stale, (time.time() - 2 * 86400,) * 2)
    fresh = d / "d.mp4.part"
    fresh.write_bytes(b"x" * 10)
    m = remux.RemuxManager(directory=d, max_bytes=400)
    freed = m.sweep()
    assert not old.exists() and new.exists()           # oldest evicted to get under 400
    assert not stale.exists() and fresh.exists()       # only day-old partials go
    assert freed == 310


# ── routes ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    from gensrt.server import app
    app.config["TESTING"] = True
    return app.test_client()


def test_prepare_for_a_native_container_is_ready_without_starting_anything(client, tmp_path, monkeypatch):
    mp4 = tmp_path / "v.mp4"
    mp4.write_bytes(b"\x00" * 16)
    called = []
    monkeypatch.setattr(remux.manager, "prepare", lambda p: called.append(p))
    body = client.post("/api/media/prepare", json={"path": str(mp4)}).get_json()
    assert body == {"status": "ready", "progress": 1.0, "error": None, "remux": False}
    assert called == []


def test_prepare_and_status_for_a_ts_go_through_the_manager(client, tmp_path, monkeypatch):
    ts = tmp_path / "v.ts"
    ts.write_bytes(b"\x47" * 188)
    monkeypatch.setattr(remux.manager, "prepare", lambda p: remux.RemuxStatus("preparing", 0.25))
    monkeypatch.setattr(remux.manager, "status", lambda p: remux.RemuxStatus("preparing", 0.5))
    body = client.post("/api/media/prepare", json={"path": str(ts)}).get_json()
    assert body["status"] == "preparing" and body["progress"] == 0.25 and body["remux"] is True
    body = client.get(f"/api/media/status?path={ts}").get_json()
    assert body["status"] == "preparing" and body["progress"] == 0.5


def test_prepare_validates_the_path(client, tmp_path):
    assert client.post("/api/media/prepare", json={}).status_code == 400
    assert client.post("/api/media/prepare", json={"path": str(tmp_path / "nope.ts")}).status_code == 400
    assert client.post("/api/media/prepare", json={"path": str(tmp_path / "x.exe")}).status_code == 400
    assert client.get("/api/media/status").status_code == 400


def test_media_serves_the_rewrap_for_a_ts_and_409s_until_it_is_ready(client, tmp_path, monkeypatch):
    ts = tmp_path / "v.ts"
    ts.write_bytes(b"\x47" * 188)
    out = tmp_path / "v.mp4"
    out.write_bytes(b"MP4BYTES")
    monkeypatch.setattr(remux.manager, "status", lambda p: remux.RemuxStatus("preparing", 0.1))
    r = client.get(f"/api/media?path={ts}")
    assert r.status_code == 409 and r.get_json()["status"] == "preparing"

    monkeypatch.setattr(remux.manager, "status", lambda p: remux.RemuxStatus("ready", 1.0, output=out))
    r = client.get(f"/api/media?path={ts}")
    assert r.status_code == 200 and r.mimetype == "video/mp4" and r.data == b"MP4BYTES"
    r = client.get(f"/api/media?path={ts}", headers={"Range": "bytes=0-2"})
    assert r.status_code == 206 and r.data == b"MP4" and r.mimetype == "video/mp4"


def test_media_for_a_native_container_never_touches_the_manager(client, tmp_path, monkeypatch):
    mp4 = tmp_path / "v.mp4"
    mp4.write_bytes(b"MP4BYTES")
    monkeypatch.setattr(remux.manager, "status", lambda p: (_ for _ in ()).throw(AssertionError("called")))
    r = client.get(f"/api/media?path={mp4}")
    assert r.status_code == 200 and r.data == b"MP4BYTES"


# ── the real thing, when ffmpeg is on the machine ──────────────────────────

@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="ffmpeg not installed")
def test_real_ffmpeg_rewrap_keeps_streams_and_duration(tmp_path):
    src = tmp_path / "sample.ts"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "2",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-f", "mpegts", str(src)],
        check=True,
    )
    m = remux.RemuxManager(directory=tmp_path / "cache", ffmpeg="ffmpeg", ffprobe="ffprobe")
    m.prepare(src)
    st = _wait(m, src, timeout=30)
    assert st.state == "ready", st.error
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name,duration:stream=codec_name",
         "-of", "compact", str(st.output)], capture_output=True, text=True, check=True).stdout
    assert "codec_name=h264" in probe and "codec_name=aac" in probe
    assert "format_name=mov,mp4" in probe
    dur = float([ln for ln in probe.splitlines() if "duration=" in ln][0].split("duration=")[1].split("|")[0])
    assert abs(dur - 2.0) < 0.2
    head = st.output.read_bytes()[:4096]
    assert b"moov" in head                            # faststart: index at the front


# ── saving the rewrap ──────────────────────────────────────────────────────

def test_export_copies_the_ready_rewrap_to_the_chosen_mp4(client, tmp_path, monkeypatch):
    ts = tmp_path / "v.ts"
    ts.write_bytes(b"\x47" * 188)
    out = tmp_path / "cache.mp4"
    out.write_bytes(b"MP4BYTES")
    monkeypatch.setattr(remux.manager, "status", lambda p: remux.RemuxStatus("ready", 1.0, output=out))
    dest = tmp_path / "keep" / "v.mp4"
    dest.parent.mkdir()
    r = client.post("/api/media/export", json={"path": str(ts), "dest": str(dest)})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["bytes"] == 8 and dest.read_bytes() == b"MP4BYTES"


def test_export_error_paths(client, tmp_path, monkeypatch):
    ts = tmp_path / "v.ts"
    ts.write_bytes(b"\x47" * 188)
    mp4 = tmp_path / "n.mp4"
    mp4.write_bytes(b"x")
    out = tmp_path / "cache.mp4"
    out.write_bytes(b"MP4BYTES")
    dest = tmp_path / "v.mp4"
    # not a transport stream
    assert client.post("/api/media/export", json={"path": str(mp4), "dest": str(dest)}).status_code == 400
    # wrong destination suffix / missing parent
    monkeypatch.setattr(remux.manager, "status", lambda p: remux.RemuxStatus("ready", 1.0, output=out))
    assert client.post("/api/media/export", json={"path": str(ts), "dest": str(tmp_path / "v.mkv")}).status_code == 400
    assert client.post("/api/media/export", json={"path": str(ts), "dest": str(tmp_path / "nodir" / "v.mp4")}).status_code == 400
    # the cache file itself
    assert client.post("/api/media/export", json={"path": str(ts), "dest": str(out)}).status_code == 400
    # not ready yet
    monkeypatch.setattr(remux.manager, "status", lambda p: remux.RemuxStatus("preparing", 0.4))
    r = client.post("/api/media/export", json={"path": str(ts), "dest": str(dest)})
    assert r.status_code == 409 and r.get_json()["progress"] == 0.4
    assert not dest.exists()
