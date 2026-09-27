import json
import threading
import urllib.error
import urllib.request

import pytest

from nanomesh import web
from nanomesh.devices import get_device


@pytest.fixture
def fixed_device(monkeypatch):
    device = get_device("hp-elitebook-840-g6")
    monkeypatch.setattr(web, "_local_device", lambda: device)
    return device


def test_summary_reports_real_measurements_only(fixed_device, isolated_home):
    s = web.api("/api/summary")
    assert s["stats"]["models_measured"] == 0 and s["results"] == []
    assert s["passport"]["device"]["id"] == "hp-elitebook-840-g6"
    assert "qwen2.5-7b" in s["known_models"]


def test_plan_endpoint(fixed_device):
    p = web.api("/api/plan", "model=qwen2.5-7b&device=rtx-3060-12gb&min_quality=99.9")
    assert p["recommended"] == "Q8_0"
    assert p["device"]["id"] == "rtx-3060-12gb"


@pytest.mark.parametrize("query, message", [
    ("", "model is required"),
    ("model=not-a-model", "Can't interpret"),
    ("model=7b&device=nokia-3310", "Unknown device"),
    ("model=7b&prefer=fastest", "prefer must be"),
    ("model=7b&min_quality=amazing", "min_quality must be"),
    ("model=7b&context=lots", "context must be"),
])
def test_plan_endpoint_rejects_bad_input(fixed_device, query, message):
    with pytest.raises(web.ApiError, match=message):
        web.api("/api/plan", query)


def test_passport_and_devices(fixed_device):
    assert web.api("/api/passport", "device=redmi-14c-4gb")["compute_class"] == "Constrained Edge"
    assert len(web.api("/api/devices")["devices"]) >= 16


def test_server_serves_page_and_api_and_refuses_foreign_hosts(fixed_device):
    server = web.make_server(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "<title>NanoMesh Dashboard</title>" in page
        # Everything the page needs is inline: it must work offline.
        assert "<script src" not in page and "<link rel=\"stylesheet\"" not in page

        body = json.loads(urllib.request.urlopen(base + "/api/plan?model=7b").read())
        assert body["model"]["params"] == 7_000_000_000

        with pytest.raises(urllib.error.HTTPError) as bad:
            urllib.request.urlopen(base + "/api/plan")
        assert bad.value.code == 400 and "model is required" in bad.value.read().decode()

        # DNS rebinding: a page on another origin resolving to 127.0.0.1.
        req = urllib.request.Request(base + "/api/summary", headers={"Host": "evil.example"})
        with pytest.raises(urllib.error.HTTPError) as forbidden:
            urllib.request.urlopen(req)
        assert forbidden.value.code == 403
    finally:
        server.shutdown()
        server.server_close()


def test_server_binds_loopback_only():
    server = web.make_server(0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()
