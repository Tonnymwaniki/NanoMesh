"""`nanomesh search`: find GGUF models on Hugging Face that suit this device.

Each result is sized against the device's memory budget and, for models
NanoMesh has measured here, predicted from those measurements, so the answer
is "which file to download", not just a list of repositories.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from nanomesh import __version__
from nanomesh import results as store
from nanomesh.fit import FitCard
from nanomesh.hardware import GB, DeviceProfile
from nanomesh.model import KNOWN_MODELS, SPLIT_RE, ModelInfo, analyze
from nanomesh.planner import FORMATS, RUNTIME_OVERHEAD_BYTES, QuantFormat, Requirements, _choose, plan

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
SEARCH_POOL = 100  # repositories fetched per search before filtering by every query word
# llama.cpp quantization names, most specific first (Q4_K_M before Q4_K).
QUANT_RE = re.compile(r"(?<![A-Za-z0-9])(BF16|FP16|F16|FP32|F32|IQ\d_[A-Z]+|Q\d_K_[SML]|Q\d_K|Q\d_[01]|Q8_0)"
                      r"(?![A-Za-z0-9])", re.IGNORECASE)
SKIP_FILES = ("mmproj",)  # vision projectors ship next to the language model; they aren't one


class CatalogError(RuntimeError):
    pass


Fetch = Callable[[str], Any]


def fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise CatalogError("Hugging Face refused access: this model is gated or private. Accept its licence on "
                               "huggingface.co and set HF_TOKEN to an access token.") from None
        if e.code == 404:
            raise CatalogError(f"Not found on Hugging Face: {url.removeprefix(HF)}") from None
        raise CatalogError(f"Hugging Face returned HTTP {e.code}.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise CatalogError(f"Can't reach Hugging Face ({getattr(e, 'reason', e)}). Check the internet connection; "
                           "`nanomesh models` still lists what's already downloaded.") from None


def _headers() -> dict[str, str]:
    h = {"User-Agent": f"nanomesh/{__version__}"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


class RemoteFile(BaseModel):
    """One downloadable model: a single GGUF, or every part of a split one."""

    name: str  # first (or only) file, what llama.cpp is pointed at
    parts: list[str]
    size_bytes: int
    part_sizes: list[int]
    sha256: list[str | None]  # per part, when Hugging Face publishes it
    format: str | None  # as named, e.g. Q4_K_M, IQ4_XS

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / GB, 2)


class FileOption(BaseModel):
    file: str
    format: str | None
    variant: str | None  # closest NanoMesh variant (INT4, ...) by bits per weight
    download_gb: float
    parts: int
    memory_gb: float
    fits: bool
    tokens_per_s: float | None = None
    speed_source: str | None = None
    quality_pct: float | None = None
    quality_source: str | None = None


class CatalogModel(BaseModel):
    repo: str
    url: str
    downloads: int | None = None
    likes: int | None = None
    params_b: float | None = None
    recommended: str | None = None  # file name
    reason: str | None = None
    options: list[FileOption] = []


def repo_files(repo: str, fetch: Fetch | None = None) -> list[RemoteFile]:
    """GGUF files of a repository, with split models grouped into one entry."""
    tree = (fetch or fetch_json)(f"{HF}/api/models/{repo}/tree/main?recursive=true")
    files = {e["path"]: e for e in tree if e.get("type") == "file" and e["path"].lower().endswith(".gguf")
             and not any(s in e["path"].lower() for s in SKIP_FILES)}
    out = []
    for path, entry in sorted(files.items()):
        m = SPLIT_RE.search(path)
        if m and m.group(1) != "00001":
            continue  # later parts are listed with the first
        if m:
            total = int(m.group(2))
            stem = path[: m.start()]
            parts = [f"{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]
            if any(p not in files for p in parts):
                continue  # incomplete upload
        else:
            parts = [path]
        entries = [files[p] for p in parts]
        fmt = QUANT_RE.search(path.rsplit("/", 1)[-1])
        out.append(RemoteFile(
            name=path, parts=parts, size_bytes=sum(_size(e) for e in entries), part_sizes=[_size(e) for e in entries],
            sha256=[(e.get("lfs") or {}).get("oid") for e in entries],
            format=fmt.group(1).upper().replace("FP", "F") if fmt else None,
        ))
    return out


def _size(entry: dict) -> int:
    return int((entry.get("lfs") or {}).get("size") or entry.get("size") or 0)


def model_info_for(repo: str, files: list[RemoteFile], meta: dict | None = None) -> ModelInfo | None:
    """What NanoMesh knows about the model in a repository: its architecture if
    it's a known model, else its parameter count from Hugging Face or the name."""
    name = repo.split("/")[-1].lower()
    known = next((k for k in sorted(KNOWN_MODELS, key=len, reverse=True) if k in name.replace("_", "-")), None)
    if known:
        info = analyze(known)
        return info.model_copy(update={"name": repo.split("/")[-1]})
    params = ((meta or {}).get("gguf") or {}).get("total")
    if not params:
        m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", name)
        params = int(float(m.group(1)) * 1e9) if m else None
    if not params:
        # Last resort: an 8-bit file holds about one byte per parameter.
        q8 = next((f for f in files if f.format == "Q8_0"), None)
        params = int(q8.size_bytes / 1.06) if q8 else None
    return ModelInfo(name=repo.split("/")[-1], source="size", params=int(params)) if params else None


def _closest(bits: float) -> QuantFormat:
    return min(FORMATS, key=lambda f: abs(f.bits_per_weight - bits))


def evaluate_repo(repo: str, device: DeviceProfile, req: Requirements, fetch: Fetch | None = None,
                  listing: dict | None = None) -> CatalogModel:
    """Size every GGUF in a repository for this device and pick one."""
    fetch = fetch or fetch_json
    files = repo_files(repo, fetch)
    meta = listing or {}
    if "gguf" not in meta:
        try:
            meta = fetch(f"{HF}/api/models/{repo}") or {}
        except CatalogError:
            meta = listing or {}
    out = CatalogModel(repo=repo, url=f"{HF}/{repo}", downloads=meta.get("downloads"), likes=meta.get("likes"))
    if not files:
        out.reason = "No GGUF files in this repository."
        return out
    info = model_info_for(repo, files, meta)
    if info is None:
        out.reason = "Couldn't tell the model's size, so it can't be sized for this device."
        out.options = [FileOption(file=f.name, format=f.format, variant=None, download_gb=f.size_gb,
                                  parts=len(f.parts), memory_gb=f.size_gb, fits=False) for f in files]
        return out
    ev = store.evidence(device, info)
    p = plan(info, device, req, ev)
    info = p.model  # evidence may correct the parameter count
    out.params_b = round(info.params / 1e9, 2)
    by_format = {v.format.name: v for v in p.variants}
    budget = max(b.memory_gb for b in p.budgets)
    kv = info.kv_bytes_per_token() * req.context / GB

    options, candidates = [], []
    for f in files:
        # A standard format name says what the file is; other quantizations
        # (Q4_0, IQ4_XS...) are placed by their size in bits per weight.
        v = by_format.get(f.format or "") or (by_format[_closest(f.size_bytes * 8 / info.params).name]
                                              if info.params else None)
        memory = round(f.size_bytes / GB + kv + RUNTIME_OVERHEAD_BYTES / GB, 2)
        opt = FileOption(file=f.name, format=f.format, variant=v.format.label if v else None, download_gb=f.size_gb,
                         parts=len(f.parts), memory_gb=memory, fits=memory <= budget)
        if v:
            exact = f.format == v.format.name
            # A variant's measured figures belong to that exact format; a
            # neighbour (Q4_0 next to Q4_K_M) only gets them as an estimate.
            opt.tokens_per_s = v.tokens_per_s
            opt.speed_source = v.speed_source if exact or v.speed_source != "measured" else "calibrated"
            opt.quality_pct = v.quality_pct
            opt.quality_source = v.quality_source if exact or v.quality_source != "measured" else "calibrated"
            if opt.fits and v.meets_requirements:
                candidates.append((v, f, exact))
        options.append(opt)
    out.options = sorted(options, key=lambda o: -o.download_gb)

    if candidates:
        best = _choose([v for v, _, _ in candidates], req.prefer)
        # Several files can map to one variant (Q4_K_M, Q4_K_S, Q4_0): prefer
        # the exact format NanoMesh measures, then the smaller download.
        chosen = min((c for c in candidates if c[0] is best), key=lambda c: (not c[2], c[1].size_bytes))[1]
        out.recommended = chosen.name
        opt = next(o for o in options if o.file == chosen.name)
        speed = f", ~{opt.tokens_per_s:g} tok/s ({opt.speed_source})" if opt.tokens_per_s else ""
        out.reason = (f"{opt.variant} fits in {opt.memory_gb:g} GB of the {budget:.1f} GB budget{speed}, "
                      f"quality ~{opt.quality_pct:g}% ({opt.quality_source}).")
    elif any(o.fits for o in options):
        out.reason = "Files that fit fall below the required quality; ask for a lower min_quality or a bigger device."
    else:
        smallest = min(options, key=lambda o: o.memory_gb)
        out.reason = f"Nothing fits: the smallest file needs ~{smallest.memory_gb:g} GB, the budget is {budget:.1f} GB."
    return out


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[\s/_-]+", text.lower()) if w]


def search(query: str, device: DeviceProfile, *, limit: int = 5, req: Requirements | None = None,
           fetch: Fetch | None = None) -> list[CatalogModel]:
    """Most-downloaded GGUF repositories whose name contains every query word,
    each sized for the device."""
    words = _words(query)
    if not words:
        raise CatalogError("Say what to search for, e.g. 'qwen2.5 coder 7b' or 'llama 3.2 3b'.")
    req, fetch = req or Requirements(), fetch or fetch_json
    # Hugging Face matches one substring; search by the most specific word and
    # check the rest here ("qwen2.5 1.5b" matches Qwen2.5-1.5B-Instruct-GGUF).
    anchor = max(words, key=len)
    params = urllib.parse.urlencode({"search": anchor, "filter": "gguf", "sort": "downloads", "direction": "-1",
                                     "limit": SEARCH_POOL})
    listing = fetch(f"{HF}/api/models?{params}")
    flat = lambda s: re.sub(r"[\s/_-]+", "", s.lower())  # noqa: E731  ("1.5b" vs "1.5B", "3.2" vs "3-2")
    hits = [m for m in listing if all(flat(w) in flat(m.get("id", "")) for w in words)][:limit]
    return [evaluate_repo(m["id"], device, req, fetch, m) for m in hits]


# ---- search by task: speech, vision, embeddings ----

TASK_TAGS = {  # NanoMesh task -> Hugging Face pipeline tag
    "speech-to-text": "automatic-speech-recognition",
    "object detection": "object-detection",
    "image classification": "image-classification",
    "embeddings": "sentence-similarity",
    "text generation": "text-generation",
}
TASK_ALIASES = {"speech": "speech-to-text", "asr": "speech-to-text", "transcription": "speech-to-text",
                "detection": "object detection", "vision": "object detection", "classification": "image classification",
                "embedding": "embeddings", "search": "embeddings", "text": "text generation", "chat": "text generation"}


def task_name(task: str) -> str:
    t = task.lower().strip().replace("_", " ").replace("-", " ")
    for name in TASK_TAGS:
        if t == name.replace("-", " "):
            return name
    if t in TASK_ALIASES:
        return TASK_ALIASES[t]
    raise CatalogError(f"Unknown task '{task}'. Use one of: speech, detection, classification, embeddings, text.")


class TaskResult(BaseModel):
    source: str  # huggingface | kaggle
    id: str
    url: str
    downloads: int | None = None
    card: FitCard


def _license(meta: dict) -> str | None:
    lic = (meta.get("cardData") or {}).get("license")
    if isinstance(lic, list):
        lic = ", ".join(lic)
    return lic or next((t.split(":", 1)[1] for t in meta.get("tags", []) if t.startswith("license:")), None)


def search_task(query: str, task: str, device: DeviceProfile, *, limit: int = 5, fetch: Fetch | None = None,
                sources: tuple[str, ...] = ("huggingface",)) -> list[TaskResult]:
    """Models for a task that fit this device, from Hugging Face (and Kaggle),
    each with its Fit Card: fitting, fast-enough ones first."""
    from nanomesh.fit import fit

    task = task_name(task)
    fetch = fetch or fetch_json
    out: list[TaskResult] = []
    if "huggingface" in sources:
        words = _words(query)
        params = {"pipeline_tag": TASK_TAGS[task], "sort": "downloads", "direction": "-1", "limit": SEARCH_POOL}
        if words:
            params["search"] = max(words, key=len)
        flat = lambda s: re.sub(r"[\s/_-]+", "", s.lower())  # noqa: E731
        listing = [m for m in fetch(f"{HF}/api/models?{urllib.parse.urlencode(params)}")
                   if all(flat(w) in flat(m.get("id", "")) for w in words)]
        for m in listing[: limit * 2]:
            try:
                meta = fetch(f"{HF}/api/models/{m['id']}") or {}
            except CatalogError:
                meta = m
            total = (meta.get("safetensors") or {}).get("total")
            tags = list(m.get("tags") or meta.get("tags") or []) + [m.get("library_name") or meta.get("library_name")]
            c = fit(m["id"], device, license=_license(meta), task=task, params=total, tags=tags)
            if c:
                out.append(TaskResult(source="huggingface", id=m["id"], url=f"{HF}/{m['id']}",
                                      downloads=m.get("downloads"), card=c))
    if "kaggle" in sources:
        from nanomesh.kaggle import search_models as kaggle_search

        out += kaggle_search(query or task, device, task, limit=limit)
    # Fits and fast enough first, then the most downloaded.
    out.sort(key=lambda r: (not r.card.fits, r.card.usable is False, -(r.downloads or 0)))
    return out[:limit] if len(sources) == 1 else out[: limit * 2]


def runs_well_hint(found: list[TaskResult], task: str, device: DeviceProfile) -> str | None:
    """When none of the results runs well on the device, a model that does."""
    from nanomesh.fit import runs_well

    if any(r.card.fits and r.card.usable for r in found):
        return None
    return runs_well(task_name(task), device)
