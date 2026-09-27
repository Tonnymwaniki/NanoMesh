"""`nanomesh pull`: download a model from Hugging Face, safely on bad connections.

Downloads go to a .part file and resume where they stopped (after a dropped
connection, or a later `pull` of the same file), then are checked against the
SHA-256 Hugging Face publishes before being renamed into place.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from nanomesh.catalog import HF, CatalogError, RemoteFile, _headers
from nanomesh.hardware import GB

CHUNK = 1 << 20
RETRIES = 5
DISK_MARGIN_GB = 1.0


class DownloadError(RuntimeError):
    pass


class Cancelled(DownloadError):
    pass


def models_dir() -> Path:
    """Where downloads go: NANOMESH_MODELS_DIR, else C:\\models on Windows if it
    exists, else ~/models. `nanomesh models` looks in all of these."""
    if env := os.environ.get("NANOMESH_MODELS_DIR"):
        return Path(env).expanduser()
    if sys.platform == "win32" and Path("C:/models").is_dir():
        return Path("C:/models")
    return Path.home() / "models"


class DownloadPlan(BaseModel):
    repo: str
    file: RemoteFile
    dest_dir: Path
    have_bytes: int  # already on disk (complete parts and partial downloads)
    disk_free_gb: float | None

    @property
    def target(self) -> Path:
        return self.dest_dir / Path(self.file.name).name

    @property
    def remaining_gb(self) -> float:
        return round(max(0, self.file.size_bytes - self.have_bytes) / GB, 2)


def plan_download(repo: str, file: RemoteFile, dest: Path | None = None) -> DownloadPlan:
    dest_dir = (dest or models_dir() / repo.split("/")[-1]).expanduser()
    have = 0
    for part in file.parts:
        final = dest_dir / Path(part).name
        partial = final.with_name(final.name + ".part")
        have += final.stat().st_size if final.exists() else partial.stat().st_size if partial.exists() else 0
    probe = dest_dir
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free = round(shutil.disk_usage(probe).free / GB, 1)
    except OSError:
        free = None
    return DownloadPlan(repo=repo, file=file, dest_dir=dest_dir, have_bytes=have, disk_free_gb=free)


Progress = Callable[[int, int], None]  # (bytes done, bytes total)


def download(dp: DownloadPlan, progress: Progress = lambda d, t: None, cancel: threading.Event | None = None,
             opener: Callable = urllib.request.urlopen) -> Path:
    """Download every part; returns the path to point llama.cpp at."""
    if dp.disk_free_gb is not None and dp.remaining_gb + DISK_MARGIN_GB > dp.disk_free_gb:
        raise DownloadError(f"Needs {dp.remaining_gb:g} GB but only {dp.disk_free_gb:g} GB is free on that drive.")
    dp.dest_dir.mkdir(parents=True, exist_ok=True)
    total = dp.file.size_bytes
    done_before = 0
    for part, sha in zip(dp.file.parts, dp.file.sha256):
        size = _part_size(dp, part)
        final = dp.dest_dir / Path(part).name
        if not (final.exists() and final.stat().st_size == size):
            _fetch_part(f"{HF}/{dp.repo}/resolve/main/{part}", final, size, sha,
                        lambda n, base=done_before: progress(base + n, total), cancel, opener)
        done_before += size
        progress(done_before, total)
    return dp.target


def _part_size(dp: DownloadPlan, part: str) -> int:
    return dp.file.part_sizes[dp.file.parts.index(part)]


def _fetch_part(url: str, final: Path, size: int, sha: str | None, progress: Progress,
                cancel: threading.Event | None, opener: Callable) -> None:
    partial = final.with_name(final.name + ".part")
    attempt = 0
    while True:
        have = partial.stat().st_size if partial.exists() else 0
        if size and have >= size:
            break
        headers = _headers() | ({"Range": f"bytes={have}-"} if have else {})
        try:
            with opener(urllib.request.Request(url, headers=headers), timeout=60) as resp:
                if have and resp.status != 206:
                    have = 0  # the server ignored the range: start over
                with partial.open("ab" if have else "wb") as f:
                    while chunk := resp.read(CHUNK):
                        if cancel and cancel.is_set():
                            raise Cancelled("Download cancelled; run it again to resume.")
                        f.write(chunk)
                        have += len(chunk)
                        progress(have)
            if not size or have >= size:
                break
            raise ConnectionError("connection closed early")
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:  # asked for bytes past the end: already complete
                break
            if e.code in (401, 403):
                raise DownloadError("Hugging Face refused the download: this model is gated. Accept its licence on "
                                    "huggingface.co and set HF_TOKEN.") from None
            if e.code == 404:
                raise DownloadError(f"Not found: {url}") from None
            error: Exception = e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            error = e
        attempt += 1
        if attempt > RETRIES:
            raise DownloadError(f"Download kept failing ({error}). Run the same command again to resume.")
        _wait(attempt)

    if size and partial.stat().st_size != size:
        raise DownloadError(f"{final.name}: got {partial.stat().st_size} bytes, expected {size}.")
    if sha and _sha256(partial) != sha:
        partial.unlink()
        raise DownloadError(f"{final.name} was corrupted in transit (checksum mismatch); run again to re-download.")
    os.replace(partial, final)


def _wait(attempt: int) -> None:
    time.sleep(min(2 ** attempt, 30))  # 2, 4, 8, 16, 30 s: rides out a flaky connection


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK * 8):
            h.update(chunk)
    return h.hexdigest()


def pull(repo: str, file: str | None, device, req=None, dest: Path | None = None, fetch=None) -> DownloadPlan:
    """Resolve what to download: the named file, or the one NanoMesh recommends
    for this device."""
    from nanomesh.catalog import evaluate_repo, repo_files
    from nanomesh.planner import Requirements

    files = repo_files(repo, fetch)
    if not files:
        raise CatalogError(f"{repo} has no GGUF files.")
    if file is None:
        choice = evaluate_repo(repo, device, req or Requirements(), fetch)
        if not choice.recommended:
            raise CatalogError(f"No file in {repo} suits this device: {choice.reason} Name one with --file.")
        file = choice.recommended
    wanted = file.lower()
    match = next((f for f in files if f.name.lower() == wanted or Path(f.name).name.lower() == wanted), None)
    # Accept a format too: `--file Q4_K_M`.
    match = match or next((f for f in files if (f.format or "").lower() == wanted), None)
    if match is None:
        names = ", ".join(Path(f.name).name for f in files)
        raise CatalogError(f"No file '{file}' in {repo}. Available: {names}")
    return plan_download(repo, match, dest)
