"""Search → download → serve, without the internet: Hugging Face's API is
replaced by recorded-style responses and a local HTTP server."""

import hashlib
import http.server
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from nanomesh import catalog, download, jobs, mcp, serve
from nanomesh import results as store
from nanomesh.catalog import CatalogError
from nanomesh.devices import get_device
from nanomesh.download import DownloadError
from nanomesh.planner import Requirements

DATA = Path(__file__).parent / "data"
GiB = 1024**3


def _lfs(path, size, sha=None):
    return {"type": "file", "path": path, "size": 135, "lfs": {"size": size, "oid": sha or "0" * 64}}


QWEN_TREE = [
    {"type": "file", "path": "README.md", "size": 1000},
    _lfs("qwen2.5-1.5b-instruct-fp16.gguf", int(3.32 * GiB)),
    _lfs("qwen2.5-1.5b-instruct-q8_0.gguf", int(1.76 * GiB)),
    _lfs("qwen2.5-1.5b-instruct-q4_k_m.gguf", int(1.04 * GiB)),
    _lfs("qwen2.5-1.5b-instruct-q4_0.gguf", int(0.99 * GiB)),
    _lfs("qwen2.5-1.5b-instruct-q3_k_m.gguf", int(0.86 * GiB)),
]
BIG_TREE = [
    _lfs("Big-70B-Q4_K_M-00001-of-00002.gguf", 21 * GiB),
    _lfs("Big-70B-Q4_K_M-00002-of-00002.gguf", 21 * GiB),
    _lfs("Big-70B-Q2_K-00001-of-00002.gguf", 13 * GiB),  # second part missing: incomplete upload
    _lfs("mmproj-Big-70B-f16.gguf", 1 * GiB),
]
LISTING = [
    {"id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "downloads": 250000, "likes": 300},
    {"id": "Qwen/Qwen2.5-7B-Instruct-GGUF", "downloads": 400000, "likes": 500},
    {"id": "someone/Qwen2.5-1.5B-Instruct-abliterated-GGUF", "downloads": 900, "likes": 3},
]


def fake_hf(tree_by_repo, listing=LISTING, info=None):
    calls = []

    def fetch(url):
        calls.append(url)
        path = url.removeprefix(catalog.HF)
        if path.startswith("/api/models?"):
            return listing
        for repo, tree in tree_by_repo.items():
            if path == f"/api/models/{repo}/tree/main?recursive=true":
                return tree
            if path == f"/api/models/{repo}":
                return (info or {}).get(repo, {"id": repo})
        raise CatalogError(f"Not found on Hugging Face: {path}")

    fetch.calls = calls
    return fetch


@pytest.fixture
def elitebook_results():
    home = Path(os.environ["NANOMESH_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "results.jsonl").write_text((DATA / "elitebook-840-g6-qwen2.5-1.5b.jsonl").read_text())


# ---- catalog ----

def test_split_files_are_grouped_and_projectors_and_incomplete_uploads_skipped():
    files = catalog.repo_files("x/Big-70B-GGUF", fake_hf({"x/Big-70B-GGUF": BIG_TREE}))
    assert [f.name for f in files] == ["Big-70B-Q4_K_M-00001-of-00002.gguf"]
    assert files[0].parts == ["Big-70B-Q4_K_M-00001-of-00002.gguf", "Big-70B-Q4_K_M-00002-of-00002.gguf"]
    assert files[0].size_bytes == 42 * GiB and files[0].format == "Q4_K_M"


def test_search_uses_this_laptops_measurements(elitebook_results):
    fetch = fake_hf({"Qwen/Qwen2.5-1.5B-Instruct-GGUF": QWEN_TREE,
                     "someone/Qwen2.5-1.5B-Instruct-abliterated-GGUF": QWEN_TREE})
    found = catalog.search("qwen2.5 1.5b", get_device("hp-elitebook-840-g6"), fetch=fetch)
    # Every word must be in the name: the 7B is left out; order is by downloads.
    assert [m.repo for m in found] == ["Qwen/Qwen2.5-1.5B-Instruct-GGUF", "someone/Qwen2.5-1.5B-Instruct-abliterated-GGUF"]
    q = found[0]
    # Same pick as `nanomesh plan` on this laptop: INT4 measured 92.5%, below "good".
    assert q.recommended == "qwen2.5-1.5b-instruct-q8_0.gguf"
    rec = next(o for o in q.options if o.file == q.recommended)
    assert (rec.tokens_per_s, rec.speed_source) == (10.44, "measured")
    assert q.params_b == 1.78  # from the benchmarked GGUF, not the 1.54B on the model card
    # Q4_0 maps to INT4 by size, but INT4's measurements were of Q4_K_M: only an estimate for Q4_0.
    assert next(o for o in q.options if "fp16" in o.file).format == "F16"  # Qwen names it fp16
    q40 = next(o for o in q.options if o.format == "Q4_0")
    assert q40.variant == "INT4" and q40.speed_source == "calibrated" and q40.quality_source == "calibrated"
    assert "INT8" in q.reason and "measured" in q.reason
    # Searching by the longest word, then filtering: one API query.
    assert sum("/api/models?" in c for c in fetch.calls) == 1
    assert "search=qwen2.5" in fetch.calls[0]


def test_prefer_speed_picks_the_faster_file(elitebook_results):
    fetch = fake_hf({"Qwen/Qwen2.5-1.5B-Instruct-GGUF": QWEN_TREE})
    m = catalog.evaluate_repo("Qwen/Qwen2.5-1.5B-Instruct-GGUF", get_device("hp-elitebook-840-g6"),
                              Requirements(prefer="speed", min_quality="fair"), fetch)
    assert m.recommended == "qwen2.5-1.5b-instruct-q4_k_m.gguf"  # exact INT4 beats Q4_0 of the same variant


def test_unknown_model_sized_from_hugging_face_metadata():
    tree = [_lfs("Foo-Q4_K_M.gguf", int(4.4 * GiB)), _lfs("Foo-Q8_0.gguf", int(7.6 * GiB))]
    fetch = fake_hf({"org/Foo-GGUF": tree}, info={"org/Foo-GGUF": {"gguf": {"total": 7_300_000_000}}})
    m = catalog.evaluate_repo("org/Foo-GGUF", get_device("redmi-14c-4gb"), Requirements(), fetch)
    assert m.params_b == 7.3
    assert m.recommended is None and m.reason.startswith("Nothing fits")
    assert all(not o.fits for o in m.options)


def test_nothing_matching():
    assert catalog.search("nonexistent model", get_device("rtx-3060-12gb"), fetch=fake_hf({})) == []
    with pytest.raises(CatalogError):
        catalog.search("  ", get_device("rtx-3060-12gb"), fetch=fake_hf({}))


# ---- download ----

class _Files(http.server.BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    drop_after: int | None = None  # close the first response after this many bytes
    ignore_range = False
    requests: list[str | None] = []

    def do_GET(self):  # noqa: N802
        body = self.files.get(self.path)
        if body is None:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        type(self).requests.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng and not self.ignore_range else 0
        if start >= len(body) and rng:
            self.send_error(416)
            return
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        data = body[start:]
        if type(self).drop_after is not None:
            data, type(self).drop_after = data[: type(self).drop_after], None
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def hf_files(monkeypatch):
    _Files.files, _Files.drop_after, _Files.ignore_range, _Files.requests = {}, None, False, []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Files)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(download, "HF", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(download, "_wait", lambda attempt: None)
    yield _Files
    server.shutdown()


def _remote(files: dict[str, bytes], repo="org/M-GGUF"):
    for name, body in files.items():
        _Files.files[f"/{repo}/resolve/main/{name}"] = body
    tree = [_lfs(n, len(b), hashlib.sha256(b).hexdigest()) for n, b in files.items()]
    return catalog.repo_files(repo, fake_hf({repo: tree}))[0]


def test_download_split_model_verified(hf_files, tmp_path):
    a, b = os.urandom(300_000), os.urandom(200_000)
    f = _remote({"M-Q4_K_M-00001-of-00002.gguf": a, "M-Q4_K_M-00002-of-00002.gguf": b})
    dp = download.plan_download("org/M-GGUF", f, tmp_path / "models")
    seen = []
    path = download.download(dp, lambda d, t: seen.append((d, t)))
    assert path == tmp_path / "models" / "M-Q4_K_M-00001-of-00002.gguf"
    assert path.read_bytes() == a and (path.parent / "M-Q4_K_M-00002-of-00002.gguf").read_bytes() == b
    assert seen[-1] == (500_000, 500_000)
    assert not list(path.parent.glob("*.part"))
    # A second pull finds it complete and fetches nothing.
    assert download.plan_download("org/M-GGUF", f, tmp_path / "models").remaining_gb == 0


def test_dropped_connection_resumes(hf_files, tmp_path):
    body = os.urandom(3 * download.CHUNK + 123)
    f = _remote({"M-Q8_0.gguf": body})
    hf_files.drop_after = download.CHUNK + 7  # the first response dies early
    path = download.download(download.plan_download("org/M-GGUF", f, tmp_path))
    assert path.read_bytes() == body
    assert hf_files.requests == [None, f"bytes={download.CHUNK + 7}-"]


def test_resumes_an_earlier_partial_download(hf_files, tmp_path):
    body = os.urandom(500_000)
    f = _remote({"M-Q8_0.gguf": body})
    (tmp_path / "M-Q8_0.gguf.part").write_bytes(body[:200_000])
    dp = download.plan_download("org/M-GGUF", f, tmp_path)
    assert dp.have_bytes == 200_000
    assert download.download(dp).read_bytes() == body
    assert hf_files.requests == ["bytes=200000-"]


def test_server_ignoring_range_restarts_cleanly(hf_files, tmp_path):
    body = os.urandom(500_000)
    f = _remote({"M-Q8_0.gguf": body})
    (tmp_path / "M-Q8_0.gguf.part").write_bytes(b"x" * 1000)
    hf_files.ignore_range = True
    assert download.download(download.plan_download("org/M-GGUF", f, tmp_path)).read_bytes() == body


def test_corrupted_download_is_rejected(hf_files, tmp_path):
    body = os.urandom(100_000)
    f = _remote({"M-Q8_0.gguf": body})
    hf_files.files["/org/M-GGUF/resolve/main/M-Q8_0.gguf"] = os.urandom(100_000)  # same size, other bytes
    with pytest.raises(DownloadError, match="checksum"):
        download.download(download.plan_download("org/M-GGUF", f, tmp_path))
    assert not list(tmp_path.glob("*"))  # the bad .part is gone, the next try starts over


def test_not_enough_disk(tmp_path):
    f = catalog.RemoteFile(name="M.gguf", parts=["M.gguf"], size_bytes=50 * GiB, part_sizes=[50 * GiB],
                           sha256=[None], format="Q8_0")
    dp = download.DownloadPlan(repo="o/M", file=f, dest_dir=tmp_path, have_bytes=0, disk_free_gb=20.0)
    with pytest.raises(DownloadError, match="only 20 GB is free"):
        download.download(dp)


def test_pull_picks_the_recommended_file_or_a_named_format(elitebook_results):
    fetch = fake_hf({"Qwen/Qwen2.5-1.5B-Instruct-GGUF": QWEN_TREE})
    elitebook = get_device("hp-elitebook-840-g6")
    assert download.pull("Qwen/Qwen2.5-1.5B-Instruct-GGUF", None, elitebook, fetch=fetch).file.format == "Q8_0"
    assert download.pull("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "q4_k_m", elitebook, fetch=fetch).file.name == \
        "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    with pytest.raises(CatalogError, match="Available"):
        download.pull("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "Q5_K_M", elitebook, fetch=fetch)


def test_downloads_land_where_models_looks(monkeypatch, tmp_path):
    from nanomesh.discover import default_locations

    monkeypatch.setenv("NANOMESH_MODELS_DIR", str(tmp_path / "m"))
    assert download.models_dir() == tmp_path / "m"
    assert ("folder", tmp_path / "m") in default_locations()


# ---- MCP: search + download as a job ----

def test_mcp_search_and_download_job(hf_files, monkeypatch, tmp_path):
    body = os.urandom(400_000)
    f = _remote({"M-Q4_K_M.gguf": body})
    tree = [_lfs("M-Q4_K_M.gguf", len(body), hashlib.sha256(body).hexdigest())]
    fetch = fake_hf({"org/M-GGUF": tree}, listing=[{"id": "org/M-GGUF", "downloads": 5}],
                    info={"org/M-GGUF": {"gguf": {"total": 700_000_000}}})
    monkeypatch.setattr(catalog, "fetch_json", fetch)
    monkeypatch.setenv("NANOMESH_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(mcp, "_local", lambda: get_device("hp-elitebook-840-g6"))

    found = mcp.call_tool("search_models", {"query": "m"})["structuredContent"]
    assert found["results"][0]["recommended"] == f.name and "download_gb" in json.dumps(found)

    started = mcp.call_tool("download_model", {"repo": "org/M-GGUF"})["structuredContent"]
    assert started["status"] == "started" and started["job_id"].startswith("download-")
    for _ in range(100):
        status = mcp.call_tool("job_status", {"job_id": started["job_id"]})["structuredContent"]
        if status["status"] != "running":
            break
        time.sleep(0.05)
    assert status["status"] == "done", status
    assert Path(status["result"]["path"]).read_bytes() == body
    again = mcp.call_tool("download_model", {"repo": "org/M-GGUF"})["structuredContent"]
    assert again["status"] == "already downloaded"


def test_failed_job_reports_its_error():
    def boom(h):
        h.update(0.5, "halfway")
        raise ValueError("disk full")

    job = jobs.start("test", "boom", boom)
    for _ in range(100):
        if job.status != "running":
            break
        time.sleep(0.01)
    assert (job.status, job.error, job.progress) == ("failed", "disk full", 0.5)


def test_background_output_never_reaches_the_protocol_stream(monkeypatch):
    import io

    out, real = io.StringIO(), sys.stdout
    stdin = io.StringIO('{"jsonrpc": "2.0", "id": 1, "method": "ping"}\n')
    monkeypatch.setattr(sys, "stdout", out)
    try:
        mcp.serve(stdin)
        print("a stray print from a background job")  # noqa: T201
    finally:
        sys.stdout = real
    assert out.getvalue() == '{"jsonrpc": "2.0", "id": 1, "result": {}}\n'


# ---- serve ----

FAKE_SERVER = """#!{python}
import http.server, sys, time
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
if "{crash}" in " ".join(args):
    print("error: failed to load model"); sys.exit(1)
start = time.time()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        ready = time.time() - start > 0.5
        self.send_response(200 if ready or self.path != "/health" else 503)
        self.end_headers()
        self.wfile.write(b'{{"status": "ok"}}')
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


@pytest.fixture
def fake_server(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("fake llama-server is a POSIX script")
    bin_dir = tmp_path / "llama"
    bin_dir.mkdir()
    exe = bin_dir / "llama-server"
    exe.write_text(FAKE_SERVER.format(python=sys.executable, crash="crash.gguf"))
    exe.chmod(0o755)
    monkeypatch.setenv("NANOMESH_LLAMA_CPP", str(bin_dir))
    yield
    serve.stop()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_serve_start_status_stop(fake_server, tmp_path):
    model = tmp_path / "Qwen2.5-1.5B-Instruct-Q8_0.gguf"
    model.write_bytes(b"GGUF")
    port = _free_port()
    s, state = serve.start(model, port=port, threads=4, wait_s=10)
    assert state == "ok" and s.model == "qwen2.5-1.5b-instruct-q8_0"
    assert s.base_url == f"http://127.0.0.1:{port}/v1"
    assert [r.port for r in serve.running()] == [port]
    # Starting the same model again reuses the server; another model on that port is refused.
    assert serve.start(model, port=port, wait_s=1)[0].pid == s.pid
    other = tmp_path / "other.gguf"
    other.write_bytes(b"GGUF")
    with pytest.raises(serve.ToolchainError, match="already serving"):
        serve.start(other, port=port, wait_s=1)
    snippets = serve.connect_snippets(s)
    assert f'base_url="{s.base_url}"' in snippets["Python (pip install openai)"]
    assert "apiBase" in snippets["VS Code · Continue extension (config.yaml)"]

    assert [x.port for x in serve.stop(port)] == [port]
    assert serve.running() == [] and serve.health(port) is None


def test_serve_reports_a_crash_with_the_log(fake_server, tmp_path):
    model = tmp_path / "crash.gguf"
    model.write_bytes(b"GGUF")
    with pytest.raises(serve.ToolchainError, match="failed to load model"):
        serve.start(model, port=_free_port(), wait_s=10)
    assert serve.running() == []


def test_serve_refuses_a_busy_port(fake_server, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        with pytest.raises(serve.ToolchainError, match="Something else is using port"):
            serve.start(model, port=busy.getsockname()[1], wait_s=1)


def test_forgets_servers_that_are_gone(monkeypatch):
    dead = serve.Server(pid=999_999_999, create_time=0, port=1234, model="m", model_path="/m.gguf", context=4096,
                        started=store.now(), log="/x.log")
    serve._save([dead])
    assert serve.running() == [] and serve._load() == []


def test_mcp_benchmark_job_saves_a_measurement(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("fake llama.cpp binaries are POSIX scripts")
    from conftest import write_gguf
    from test_cli import _fake_llama_cpp

    monkeypatch.setenv("NANOMESH_LLAMA_CPP", _fake_llama_cpp(tmp_path))
    monkeypatch.setattr(mcp, "_local", lambda: get_device("hp-elitebook-840-g6"))
    model = tmp_path / "model-Q4_K_M.gguf"
    write_gguf(model, file_type=15)

    started = mcp.call_tool("benchmark_model", {"path": str(model)})["structuredContent"]
    for _ in range(200):
        job = mcp.call_tool("job_status", {"job_id": started["job_id"]})["structuredContent"]
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert job["result"]["variant"] == "Q4_K_M" and job["result"]["gen_tokens_per_s"]
    assert [r.format for r in store.load()] == ["Q4_K_M"]
    assert mcp.call_tool("benchmark_model", {"path": str(tmp_path / "nope.gguf")})["isError"]
