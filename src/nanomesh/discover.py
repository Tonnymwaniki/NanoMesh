"""`nanomesh models`: find models already on this machine and what fits.

Looks in the usual places (Hugging Face cache, LM Studio, Ollama) plus any
folders the user names. Reads file headers only; nothing is loaded or sent.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

from pydantic import BaseModel

from nanomesh.hardware import GB, DeviceProfile
from nanomesh.model import SPLIT_RE, _analyze_gguf, analyze, gguf_parts, gguf_size, is_later_split_part, read_gguf
from nanomesh.planner import Requirements, memory_budgets, plan
from nanomesh.results import evidence, gguf_format

MAX_DEPTH = 5
OLLAMA_MODEL_LAYER = "application/vnd.ollama.image.model"


class FoundModel(BaseModel):
    name: str
    path: str
    source: str  # huggingface | lm-studio | ollama | folder
    kind: str  # gguf | safetensors
    format: str | None = None  # GGUF quantization, e.g. Q4_K_M
    params: int
    size_gb: float
    fits: bool  # does this exact file fit the device's memory budget?
    recommended: str | None = None  # best variant of this model for the device
    recommended_note: str | None = None


def default_locations() -> list[tuple[str, Path]]:
    home = Path.home()
    hf = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", home / ".cache" / "huggingface")) / "hub")
    places = [("huggingface", hf), ("lm-studio", home / ".lmstudio" / "models"),
              ("lm-studio", home / ".cache" / "lm-studio" / "models"),
              ("ollama", Path(os.environ.get("OLLAMA_MODELS", home / ".ollama" / "models"))),
              ("folder", home / "models")]
    if os.name == "nt":
        places.append(("folder", Path("C:/models")))
    for extra in filter(None, os.environ.get("NANOMESH_MODEL_DIRS", "").split(os.pathsep)):
        places.append(("folder", Path(extra)))
    return places


def _walk(root: Path, depth: int = 0):
    """Yield model candidates: .gguf files and folders holding safetensors + config.json."""
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    if (root / "config.json").is_file() and any(e.suffix == ".safetensors" for e in entries):
        yield root
        return
    for e in entries:
        if e.is_file() and e.suffix == ".gguf" and not is_later_split_part(e):
            yield e
        elif e.is_dir() and depth < MAX_DEPTH and not e.name.startswith("."):
            yield from _walk(e, depth + 1)


def _ollama(root: Path) -> list[tuple[str, Path]]:
    """Ollama stores GGUFs as sha256 blobs; manifests map names to them."""
    found = []
    manifests = root / "manifests"
    for m in manifests.rglob("*") if manifests.is_dir() else []:
        if not m.is_file():
            continue
        try:
            layers = json.loads(m.read_text(encoding="utf-8")).get("layers", [])
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        digest = next((l.get("digest") for l in layers if l.get("mediaType") == OLLAMA_MODEL_LAYER), None)
        blob = root / "blobs" / str(digest).replace(":", "-") if digest else None
        if blob and blob.is_file():
            rel = m.relative_to(manifests).parts
            found.append((f"{rel[-2]}:{rel[-1]}" if len(rel) >= 2 else m.name, blob))
    return found


def _is_gguf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def _describe(name: str, path: Path, source: str, device: DeviceProfile, context: int) -> FoundModel | None:
    is_gguf = path.is_file() and _is_gguf(path)
    try:
        # Ollama blobs are GGUF files without the extension.
        info = _analyze_gguf(gguf_parts(path)[0]) if is_gguf else analyze(str(path))
    except (ValueError, OSError, KeyError, struct.error, UnicodeDecodeError):
        return None
    fmt, size = None, (info.disk_bytes or 0)
    if is_gguf:
        meta, _ = read_gguf(gguf_parts(path)[0])
        fmt, size = gguf_format(meta, path), gguf_size(path)
    budget = max(b.memory_gb for b in memory_budgets(device))
    kv = info.kv_bytes_per_token() * context / GB
    p = plan(info, device, Requirements(context=context), evidence(device, info))
    rec = next((v for v in p.variants if v.format.name == p.recommended), None)
    return FoundModel(
        name=name, path=str(path), source=source, kind="gguf" if is_gguf else "safetensors", format=fmt,
        params=info.params, size_gb=round(size / GB, 2), fits=size / GB + kv + 0.3 <= budget,
        recommended=p.recommended,
        recommended_note=(f"{rec.format.label} · {rec.total_memory_gb:.1f} GB"
                          + (f" · {rec.tokens_per_s:g} tok/s" if rec and rec.tokens_per_s else "")) if rec else
        (p.advice[0] if p.advice else None),
    )


def find_models(device: DeviceProfile, extra: list[Path] | None = None, context: int = 4096) -> list[FoundModel]:
    seen, out = set(), []

    def add(name: str, path: Path, source: str):
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        m = _describe(name, path, source, device, context)
        if m:
            out.append(m)

    places = default_locations() + [("folder", p) for p in (extra or [])]
    for source, root in places:
        if not root.exists():
            continue
        if source == "ollama":
            for name, blob in _ollama(root):
                add(name, blob, "ollama")
            continue
        if root.is_file():
            add(root.stem, root, source)
            continue
        for cand in _walk(root):
            # qwen2.5-7b-q4_k_m-00001-of-00002.gguf -> qwen2.5-7b-q4_k_m
            name = SPLIT_RE.sub("", cand.name) if cand.is_file() else cand.name
            name = name.removesuffix(".gguf")
            if source == "huggingface" and "snapshots" in cand.parts:
                # models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/<hash> -> Qwen/Qwen2.5-1.5B-Instruct
                repo = next((p for p in cand.parts if p.startswith("models--")), cand.name)
                name = repo.removeprefix("models--").replace("--", "/")
            add(name, cand, source)
    return sorted(out, key=lambda m: (m.source, m.name))
