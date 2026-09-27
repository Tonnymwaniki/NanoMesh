"""`nanomesh serve`: run a model as a local, OpenAI-compatible server.

Starts llama.cpp's llama-server in the background with this device's best
settings, so editors, scripts and apps can use the model at
http://127.0.0.1:<port>/v1. Servers keep running until `nanomesh serve --stop`.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil
from pydantic import BaseModel

from nanomesh.model import gguf_parts
from nanomesh.results import home, now
from nanomesh.toolchain import Toolchain, ToolchainError, find_toolchain

DEFAULT_PORT = 8080
LOAD_TIMEOUT_S = 90
# Talks to our own server on 127.0.0.1: never through a proxy from HTTP_PROXY
# or the system settings.
_LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Server(BaseModel):
    pid: int
    create_time: float  # tells our process apart from a later one reusing its pid
    port: int
    model: str  # the name clients use ("model" in API requests)
    model_path: str
    context: int
    threads: int | None = None
    started: str
    log: str

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"


def _state() -> Path:
    return home() / "servers.json"


def _load() -> list[Server]:
    try:
        return [Server(**s) for s in json.loads(_state().read_text(encoding="utf-8"))]
    except (OSError, ValueError, TypeError):
        return []


def _save(servers: list[Server]) -> None:
    _state().parent.mkdir(parents=True, exist_ok=True)
    _state().write_text(json.dumps([s.model_dump() for s in servers], indent=1), encoding="utf-8")


def _alive(s: Server) -> bool:
    try:
        p = psutil.Process(s.pid)
        return abs(p.create_time() - s.create_time) < 1 and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def health(port: int) -> str | None:
    """'ok' when serving, 'loading' while the model loads, None if nothing answers."""
    try:
        with _LOCAL.open(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return "ok" if r.status == 200 else "loading"
    except urllib.error.HTTPError as e:
        return "loading" if e.code == 503 else None
    except (urllib.error.URLError, OSError):
        return None


def _port_taken(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def running() -> list[Server]:
    """Servers NanoMesh started that are still up (forgets the rest)."""
    servers = _load()
    alive = [s for s in servers if _alive(s)]
    if len(alive) != len(servers):
        _save(alive)
    return alive


def start(model_path: Path, *, port: int = DEFAULT_PORT, context: int = 4096, threads: int | None = None,
          wait_s: float = LOAD_TIMEOUT_S, tc: Toolchain | None = None) -> tuple[Server, str]:
    """Start llama-server on 127.0.0.1 (never exposed to the network). Returns the
    server and its state: 'ok', or 'loading' if it's still loading after wait_s."""
    tc = tc or find_toolchain()
    if not tc.server:
        raise ToolchainError("llama-server not found. It ships with llama.cpp releases: unpack one to C:\\llama.cpp "
                             "(Windows) or ~/llama.cpp, or set NANOMESH_LLAMA_CPP to its folder.")
    model_path = gguf_parts(model_path.expanduser().resolve())[0]
    if not model_path.is_file():
        raise ToolchainError(f"No such model file: {model_path}")
    for s in running():
        if s.port == port:
            if Path(s.model_path) == model_path:
                return s, health(port) or "loading"
            raise ToolchainError(f"Port {port} is already serving {s.model}. Stop it (nanomesh serve --stop "
                                 f"--port {port}) or pick another port.")
    if _port_taken(port):
        raise ToolchainError(f"Something else is using port {port}. Pick another one, e.g. port {port + 1}.")

    name = model_path.name.removesuffix(".gguf")
    name = name.split("-00001-of-")[0].lower()
    log = home() / "logs" / f"server-{port}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(tc.server), "-m", str(model_path), "--host", "127.0.0.1", "--port", str(port),
           "-c", str(context), "--alias", name]
    if threads:
        cmd += ["-t", str(threads)]
    # Detached: the server outlives this command and the MCP session that started it.
    kwargs: dict = {"start_new_session": True}
    if sys.platform == "win32":
        kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
    with log.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kwargs)
    server = Server(pid=proc.pid, create_time=psutil.Process(proc.pid).create_time(), port=port, model=name,
                    model_path=str(model_path), context=context, threads=threads, started=now(), log=str(log))
    _save([s for s in running() if s.port != port] + [server])

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            _save([s for s in _load() if s.pid != server.pid])
            tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-12:]
            raise ToolchainError("llama-server stopped while loading the model:\n" + "\n".join(tail))
        if health(port) == "ok":
            return server, "ok"
        time.sleep(0.5)
    return server, "loading"


def stop(port: int | None = None) -> list[Server]:
    """Stop NanoMesh's servers (all, or the one on port)."""
    stopped, keep = [], []
    for s in running():
        if port is not None and s.port != port:
            keep.append(s)
            continue
        try:
            p = psutil.Process(s.pid)
            p.terminate()
            try:
                p.wait(timeout=10)
            except psutil.TimeoutExpired:
                p.kill()
        except psutil.NoSuchProcess:
            pass
        stopped.append(s)
    _save(keep)
    return stopped


def connect_snippets(s: Server) -> dict[str, str]:
    """How to use the server from common tools. It speaks the OpenAI API."""
    url = s.base_url
    body = json.dumps({"model": s.model, "messages": [{"role": "user", "content": "Hello"}]})
    return {
        "Base URL (any OpenAI-compatible client)": f"{url}  ·  model: {s.model}  ·  API key: any text",
        **({"PowerShell": (f"Invoke-RestMethod {url}/chat/completions -Method Post -ContentType application/json "
                           f"-Body '{body}' | ForEach-Object {{ $_.choices[0].message.content }}")}
           if sys.platform == "win32" else
           {"curl": f"curl {url}/chat/completions -H 'Content-Type: application/json' -d '{body}'"}),
        "Python (pip install openai)": (
            "from openai import OpenAI\n"
            f'client = OpenAI(base_url="{url}", api_key="local")\n'
            f'reply = client.chat.completions.create(model="{s.model}", '
            'messages=[{"role": "user", "content": "Hello"}])\n'
            "print(reply.choices[0].message.content)"),
        "JavaScript": (
            f'const r = await fetch("{url}/chat/completions", {{method: "POST", '
            'headers: {"Content-Type": "application/json"},\n'
            f'  body: JSON.stringify({{model: "{s.model}", messages: [{{role: "user", content: "Hello"}}]}})}});\n'
            "console.log((await r.json()).choices[0].message.content);"),
        "VS Code · Continue extension (config.yaml)": (
            "models:\n"
            f"  - name: {s.model} (NanoMesh)\n"
            "    provider: openai\n"
            f"    model: {s.model}\n"
            f"    apiBase: {url}\n"
            "    apiKey: local"),
    }
