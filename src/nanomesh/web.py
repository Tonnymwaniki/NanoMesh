"""`nanomesh dashboard`: a local web UI over the planner, device passports and results.

Everything runs on this machine, bound to 127.0.0.1, with no internet or CDN
needed. The page is one static file; data comes from a small JSON API.
"""

from __future__ import annotations

import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

from nanomesh import __version__
from nanomesh import results as store
from nanomesh.devices import get_device, load_devices, recognise
from nanomesh.hardware import DeviceProfile, scan_device
from nanomesh.model import KNOWN_MODELS, analyze
from nanomesh.passport import passport
from nanomesh.planner import QUALITY_TIERS, Requirements, plan


class ApiError(Exception):
    pass


@lru_cache(maxsize=1)
def _local_device() -> DeviceProfile:
    # Scanning shells out on Windows (PowerShell/CIM), so do it once per session.
    return recognise(scan_device())


def _device(device_id: str | None) -> DeviceProfile:
    if not device_id or device_id == "local":
        return _local_device()
    try:
        return get_device(device_id)
    except KeyError as e:
        raise ApiError(e.args[0]) from None


def _one(q: dict, key: str, default: str | None = None) -> str | None:
    return q.get(key, [default])[0] or default


def _float(q: dict, key: str) -> float | None:
    raw = _one(q, key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ApiError(f"{key} must be a number") from None


def _requirements(q: dict) -> Requirements:
    min_quality = _one(q, "min_quality", "good")
    prefer = _one(q, "prefer", "balanced")
    if prefer not in ("balanced", "quality", "speed", "size"):
        raise ApiError("prefer must be balanced, quality, speed or size")
    try:
        context = int(_one(q, "context", "4096"))
    except ValueError:
        raise ApiError("context must be a whole number") from None
    req = dict(context=context, min_tokens_per_s=_float(q, "min_speed"), max_ram_gb=_float(q, "ram"), prefer=prefer)
    try:
        return Requirements(**req, min_quality_pct=float(min_quality.rstrip("%")))
    except ValueError:
        if min_quality not in QUALITY_TIERS:
            raise ApiError(f"min_quality must be a percentage or one of {', '.join(QUALITY_TIERS)}") from None
        return Requirements(**req, min_quality=min_quality)


def _stats(rows: list[store.Result], device: DeviceProfile) -> dict:
    mine = [r for r in rows if r.device_key == device.key]
    return {
        "models_measured": len({r.model_name for r in mine}),
        "variants_measured": len({(r.model_name, r.format) for r in mine}),
        "quality_measured": sum(1 for r in mine if r.quality_pct is not None),
        "fastest_tokens_per_s": max((r.gen_tokens_per_s or 0 for r in mine), default=0) or None,
        "known_devices": len(load_devices()),
    }


def api(path: str, query: str = "") -> dict:
    """Answer one API request. Raises ApiError for bad input."""
    q = parse_qs(query)
    if path == "/api/summary":
        device = _device("local")
        rows = store.load()
        return {
            "version": __version__,
            "passport": passport(device).model_dump(),
            "device_key": device.key,
            "stats": _stats(rows, device),
            "results": store.rows(rows),
            "results_path": str(store.results_path()),
            "known_models": sorted(KNOWN_MODELS),
        }
    if path == "/api/devices":
        return {"devices": [d.model_dump(exclude_none=True) for d in load_devices().values()]}
    if path == "/api/passport":
        return passport(_device(_one(q, "device"))).model_dump()
    if path == "/api/plan":
        spec = _one(q, "model")
        if not spec:
            raise ApiError("model is required")
        try:
            model = analyze(spec)
        except ValueError as e:
            raise ApiError(str(e)) from None
        device = _device(_one(q, "device"))
        p = plan(model, device, _requirements(q), store.evidence(device, model))
        return p.model_dump()
    if path == "/api/results":
        return {"results": store.rows(store.load())}
    raise ApiError(f"unknown endpoint {path}")


def index_html() -> bytes:
    return resources.files("nanomesh.data").joinpath("dashboard.html").read_bytes()


ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}


class Handler(BaseHTTPRequestHandler):
    server_version = f"nanomesh/{__version__}"

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        # Refuse other Host names so a web page can't reach this server via DNS
        # rebinding and read hardware details or local model paths.
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        if host not in ALLOWED_HOSTS:
            self._send(403, "text/plain; charset=utf-8", b"Forbidden host")
            return
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", index_html())
            return
        if url.path.startswith("/api/"):
            try:
                body, status = api(url.path, url.query), 200
            except ApiError as e:
                body, status = {"error": str(e)}, 400
            self._send(status, "application/json", json.dumps(body).encode())
            return
        self._send(404, "text/plain; charset=utf-8", b"Not found")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # keep the terminal quiet
        pass


def make_server(port: int = 8765) -> ThreadingHTTPServer:
    # Loopback only: the dashboard shows hardware details and local file paths.
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
