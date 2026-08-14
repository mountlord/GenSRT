"""Model-format validation.

Adding 'adalat-ai/whisper-small-ml-rmft' returned "Found on HuggingFace", then
transcription failed with "Unable to open file 'model.bin'". The repo exists;
it is simply a PyTorch model, and GenSRT runs on CTranslate2. Validation only
checked existence, so it gave a green light to a model that can never load.
"""

from __future__ import annotations

from gensrt.server import _ct2_format_problem


def _payload(*filenames):
    return {"id": "org/repo",
            "siblings": [{"rfilename": f} for f in filenames]}


def test_ct2_model_passes():
    assert _ct2_format_problem(
        "adalat-ai/ct2-whisper-medium-ml-rmft",
        _payload("model.bin", "config.json", "tokenizer.json"),
    ) is None


def test_pytorch_model_is_rejected():
    msg = _ct2_format_problem(
        "adalat-ai/whisper-small-ml-rmft",
        _payload("pytorch_model.bin", "config.json", "preprocessor_config.json"),
    )
    assert msg is not None
    assert "CTranslate2" in msg


def test_rejection_suggests_the_ct2_variant():
    """The working model really is the same name with a ct2- prefix."""
    msg = _ct2_format_problem(
        "adalat-ai/whisper-small-ml-rmft", _payload("model.safetensors")
    )
    assert "adalat-ai/ct2-whisper-small-ml-rmft" in msg


def test_rejection_gives_a_conversion_command():
    msg = _ct2_format_problem("org/repo", _payload("pytorch_model.bin"))
    assert "ct2-transformers-converter" in msg


def test_safetensors_only_is_rejected():
    assert _ct2_format_problem("org/repo", _payload("model.safetensors")) is not None


def test_missing_file_list_is_not_guessed():
    """Unsure must mean allow — blocking a model that would have worked is
    worse than letting the load-time error speak."""
    assert _ct2_format_problem("org/repo", {"id": "org/repo"}) is None
    assert _ct2_format_problem("org/repo", {"id": "x", "siblings": []}) is None


def test_nested_model_bin_counts():
    assert _ct2_format_problem(
        "org/repo", _payload("float16/model.bin", "float16/config.json")
    ) is None


def test_unrecognised_layout_warns_without_naming_pytorch():
    msg = _ct2_format_problem("org/repo", _payload("README.md", "LICENSE"))
    assert msg is not None
    assert "PyTorch" not in msg


# ── TLS trust consistency ─────────────────────────────────────────────────

def test_validation_uses_requests_not_urllib():
    """Validation and download must share a TLS trust store.

    urllib goes through Python's ssl to the Windows certificate store;
    huggingface_hub downloads go through requests to the certifi bundle. With
    urllib, validation could reject a model that would have downloaded fine —
    observed on a fresh Windows install with an incomplete root store, where
    curl succeeded and Python did not.
    """
    import inspect

    from gensrt import server

    src = inspect.getsource(server.api_validate_model)
    assert "import requests" in src
    assert "urllib.request.urlopen" not in src


def test_ssl_failure_message_is_actionable(monkeypatch):
    """The bare OpenSSL text tells a non-technical user nothing."""
    import requests

    from gensrt import server

    def boom(*a, **kw):
        raise requests.exceptions.SSLError(
            "CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate"
        )

    monkeypatch.setattr(requests, "get", boom)
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        r = c.post("/api/validate_model", json={"model": "org/repo"})
    msg = r.get_json()["message"]
    assert "certificate" in msg.lower()
    assert "browser" in msg.lower()          # names the usual remedy
    assert "date and time" in msg.lower()    # and the second one


def test_bad_ca_bundle_path_names_the_env_vars(monkeypatch):
    """A stale REQUESTS_CA_BUNDLE breaks every requests-based tool.

    requests raises OSError here, not SSLError, so this needs its own branch.
    """
    import requests

    from gensrt import server

    def boom(*a, **kw):
        raise OSError(
            "Could not find a suitable TLS CA certificate bundle, "
            "invalid path: C:\\Certificates\\empty.pem"
        )

    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", r"C:\Certificates\empty.pem")
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        r = c.post("/api/validate_model", json={"model": "org/repo"})
    msg = r.get_json()["message"]
    assert "REQUESTS_CA_BUNDLE" in msg
    assert "empty.pem" in msg          # shows the offending value
    assert "Clearing those variables" in msg


def test_other_os_errors_are_not_misreported(monkeypatch):
    import requests

    from gensrt import server

    monkeypatch.setattr(
        requests, "get",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")),
    )
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        r = c.post("/api/validate_model", json={"model": "org/repo"})
    assert "certificate" not in r.get_json()["message"].lower()


def test_401_does_not_claim_the_repo_exists(monkeypatch):
    """HuggingFace returns 401 for private, gated AND nonexistent repos.

    Saying "exists but requires authentication" claims more than the API
    reported, and sends someone off to request access to something that may
    not be there.
    """
    import requests

    from gensrt import server

    class R:
        status_code = 401

        def json(self):
            return {}

    monkeypatch.setattr(requests, "get", lambda *a, **kw: R())
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        r = c.post("/api/validate_model", json={"model": "org/ct2-thing"})
    msg = r.get_json()["message"]
    assert "exists but requires" not in msg
    assert "does not exist" in msg
    assert "huggingface.co/org/ct2-thing" in msg


def test_401_on_a_user_added_ct2_prefix_points_at_the_original(monkeypatch):
    import requests

    from gensrt import server

    class R:
        status_code = 401

        def json(self):
            return {}

    monkeypatch.setattr(requests, "get", lambda *a, **kw: R())
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        r = c.post("/api/validate_model", json={"model": "adalat-ai/ct2-whisper-small-ml"})
    assert "huggingface.co/adalat-ai/whisper-small-ml" in r.get_json()["message"]
