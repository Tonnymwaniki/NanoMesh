"""Dataset Cards: what a dataset holds and what it costs this device.

Like a model's Fit Card, but for data: rows, columns, a preview, licence and
languages; the download (and its data cost), disk and memory it needs here;
whether to stream it or take a slice; and, for a model to fine-tune, how many
tokens that is and how long training takes on this device versus a cloud GPU.

Sources: Hugging Face (sizes and previews from its dataset viewer), Kaggle
(with the user's token) and local files (CSV, TSV, JSONL, JSON).
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import urllib.parse
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from nanomesh.catalog import HF, CatalogError, Fetch, _words, fetch_json
from nanomesh.hardware import GB, DeviceProfile
from nanomesh.planner import memory_budgets

VIEWER = os.environ.get("HF_DATASETS_SERVER", "https://datasets-server.huggingface.co").rstrip("/")
CHARS_PER_TOKEN = 4.0  # English text; other languages often take more tokens
PREVIEW_ROWS = 3
CELL_CHARS = 120
LOCAL_EXT = {".csv", ".tsv", ".jsonl", ".json"}
# Dense bf16/fp16 throughput actually reached in fine-tuning (~35-40% of peak), in TFLOPS.
CLOUD_GPUS = {"Colab / Kaggle T4 (free tier)": 25.0, "A100": 120.0}


class Column(BaseModel):
    name: str
    type: str


class TrainingFit(BaseModel):
    model: str
    method: str | None  # recommended by train-plan: full | lora | qlora, None if nothing fits
    memory_gb: float | None
    fits: bool
    tokens: int  # tokens trained per epoch (each example cut to seq_len)
    epochs: int
    hours_here: float | None
    hours_cloud: dict[str, float]
    advice: list[str] = []


class DataCard(BaseModel):
    id: str
    source: str  # huggingface | kaggle | local
    url: str | None = None
    device: str
    license: str | None = None
    languages: list[str] = []
    modality: str | None = None  # text | image | audio | tabular | video
    tasks: list[str] = []
    rows: int | None = None
    splits: dict[str, int] = {}  # split -> rows
    columns: list[Column] = []
    preview: list[dict[str, Any]] = []
    download_gb: float | None = None
    data_cost: str | None = None
    disk_gb: float | None = None
    memory_gb: float | None = None  # loaded fully into memory (Arrow)
    fits_disk: bool | None = None
    fits_memory: bool | None = None
    avg_tokens_per_row: float | None = None
    downloads: int | None = None
    advice: list[str] = []
    how: str | None = None
    training: TrainingFit | None = None


# ---- helpers ----

def _cost(gb: float | None) -> str | None:
    from nanomesh.fit import _data_cost

    return _data_cost(gb) if gb else None


def _disk_free(path: Path | None = None) -> float | None:
    probe = path or Path.home()
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / GB
    except OSError:
        return None


def _cell(v: Any) -> Any:
    if isinstance(v, str):
        return v if len(v) <= CELL_CHARS else v[:CELL_CHARS] + "…"
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    if isinstance(v, dict) and v.get("src"):  # images and audio in the viewer come as URLs
        return f"<{v.get('type', 'file')}: {str(v['src'])[:60]}>"
    text = json.dumps(v, ensure_ascii=False, default=str)
    return text if len(text) <= CELL_CHARS else text[:CELL_CHARS] + "…"


def _text_chars(v: Any) -> int:
    """Characters of text in a value, including chat-style lists of messages."""
    if isinstance(v, str):
        return len(v)
    if isinstance(v, dict):
        if "src" in v:
            return 0  # media, not text
        return sum(_text_chars(x) for x in v.values())
    if isinstance(v, list):
        return sum(_text_chars(x) for x in v)
    return 0


def _tokens_per_row(rows: list[dict]) -> float | None:
    if not rows:
        return None
    chars = sum(_text_chars(r) for r in rows) / len(rows)
    return round(chars / CHARS_PER_TOKEN, 1) if chars else None


def _modality(tags: list[str], columns: list[Column], sample: list[dict] | None = None) -> str | None:
    for r in sample or []:
        for v in r.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and {"role", "content"} <= set(v[0]):
                return "chat (messages with roles: ready for fine-tuning)"
    for t in tags:
        if t.startswith("modality:"):
            return t.split(":", 1)[1]
    types = " ".join(c.type.lower() for c in columns)
    for word, modality in (("image", "image"), ("audio", "audio"), ("video", "video")):
        if word in types:
            return modality
    if columns and all(c.type in ("int64", "float64", "int32", "float32", "bool", "number") for c in columns):
        return "tabular"
    return "text" if columns else None


def _advise(card: DataCard, device: DeviceProfile) -> None:
    budget = max(b.memory_gb for b in memory_budgets(device))
    free = _disk_free()
    if card.disk_gb is not None and free is not None:
        card.fits_disk = card.disk_gb + 1 <= free
        if not card.fits_disk:
            card.advice.append(f"Needs ~{card.disk_gb:g} GB of disk and only {free:.0f} GB is free: stream it "
                               "instead of downloading, or take one split.")
    if card.memory_gb is not None:
        card.fits_memory = card.memory_gb <= budget
        if not card.fits_memory:
            card.advice.append(f"Loaded whole it needs ~{card.memory_gb:g} GB, more than this device's "
                               f"{budget:.1f} GB budget: stream it (datasets: load_dataset(..., streaming=True)) or "
                               "process it in chunks.")
    if card.download_gb and card.download_gb > 2 and card.rows:
        per_row = card.download_gb / card.rows
        slice_rows = int(0.5 / per_row) if per_row else None
        if slice_rows and slice_rows < card.rows:
            card.advice.append(f"A 0.5 GB slice is about {slice_rows:,} rows"
                               + (f" ({_cost(0.5)})" if _cost(0.5) else "")
                               + ": enough to prototype before paying for the whole download.")
    if not card.license:
        card.advice.append("No licence stated: check the dataset page before using it in a product.")


# ---- Hugging Face ----

def _viewer(path: str, fetch: Fetch) -> dict | None:
    try:
        data = fetch(f"{VIEWER}/{path}")
        return data if isinstance(data, dict) and "error" not in data else None
    except CatalogError:
        return None  # not every dataset has the viewer (scripts, gated, very new)


def hf_card(dataset_id: str, device: DeviceProfile, fetch: Fetch | None = None) -> DataCard:
    fetch = fetch or fetch_json
    info = fetch(f"{HF}/api/datasets/{dataset_id}") or {}
    card_data = info.get("cardData") or {}
    tags = info.get("tags") or []
    lic = card_data.get("license")
    lic = ", ".join(lic) if isinstance(lic, list) else lic or next(
        (t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
    langs = card_data.get("language") or [t.split(":", 1)[1] for t in tags if t.startswith("language:")]
    card = DataCard(id=dataset_id, source="huggingface", url=f"{HF}/datasets/{dataset_id}", device=device.name,
                    license=lic, languages=langs if isinstance(langs, list) else [langs],
                    tasks=card_data.get("task_categories") or [t.split(":", 1)[1] for t in tags
                                                               if t.startswith("task_categories:")],
                    downloads=info.get("downloads"),
                    how=f"from datasets import load_dataset; ds = load_dataset(\"{dataset_id}\")  "
                        f"# add streaming=True to avoid the full download")
    q = urllib.parse.quote(dataset_id, safe="")
    size = _viewer(f"size?dataset={q}", fetch)
    if size and size.get("size"):
        whole = size["size"].get("dataset") or {}
        card.rows = whole.get("num_rows")
        parquet = whole.get("num_bytes_parquet_files")
        original = whole.get("num_bytes_original_files")
        card.download_gb = round(min(x for x in (parquet, original) if x) / GB, 3) if (parquet or original) else None
        if whole.get("num_bytes_memory"):
            card.memory_gb = round(whole["num_bytes_memory"] / GB, 3)
        card.disk_gb = round(((original or parquet or 0) + (whole.get("num_bytes_memory") or 0)) / GB, 3) or None
        for s in size["size"].get("splits") or []:
            name = s["split"] if s.get("config") in (None, "default") else f"{s['config']}/{s['split']}"
            card.splits[name] = s.get("num_rows")
        if size.get("partial"):
            card.advice.append("The viewer measured only part of this dataset: the real size is larger.")
    elif info.get("usedStorage"):
        card.download_gb = round(info["usedStorage"] / GB, 3)
    splits = _viewer(f"splits?dataset={q}", fetch)
    first = (splits or {}).get("splits") or []
    if first:
        pick = next((s for s in first if s.get("split") == "train"), first[0])
        rows = _viewer(f"first-rows?dataset={q}&config={urllib.parse.quote(pick['config'])}"
                       f"&split={urllib.parse.quote(pick['split'])}", fetch)
        if rows:
            card.columns = [Column(name=f["name"], type=_feature_type(f.get("type"))) for f in rows.get("features", [])]
            sample = [r.get("row", {}) for r in rows.get("rows", [])]
            card.avg_tokens_per_row = _tokens_per_row(sample)
            card.preview = [{k: _cell(v) for k, v in r.items()} for r in sample[:PREVIEW_ROWS]]
            card.modality = _modality([], [], sample)  # chat, when the rows are role/content messages
    else:
        card.advice.append("Hugging Face's viewer has no preview for this dataset (it may use a loading script or "
                           "be gated): sizes come from the repository.")
    card.modality = card.modality or _modality(tags, card.columns)
    card.data_cost = _cost(card.download_gb)
    _advise(card, device)
    return card


def _feature_type(t: Any) -> str:
    if isinstance(t, dict):
        if t.get("_type") == "Value":
            return t.get("dtype", "value")
        if t.get("_type") == "ClassLabel":
            names = t.get("names") or []
            return f"label ({len(names)} classes)" if names else "label"
        return (t.get("_type") or "struct").lower()
    if isinstance(t, list):
        return "list"
    return str(t)


def hf_search(query: str, device: DeviceProfile, *, task: str | None = None, limit: int = 5,
              fetch: Fetch | None = None) -> list[DataCard]:
    fetch = fetch or fetch_json
    words = _words(query)
    params = {"sort": "downloads", "direction": "-1", "limit": 100}
    if words:
        params["search"] = max(words, key=len)
    if task:
        params["filter"] = f"task_categories:{task}"
    flat = lambda s: re.sub(r"[\s/_-]+", "", s.lower())  # noqa: E731
    listing = [d for d in fetch(f"{HF}/api/datasets?{urllib.parse.urlencode(params)}")
               if all(flat(w) in flat(d.get("id", "")) for w in words)]
    return [hf_card(d["id"], device, fetch) for d in listing[:limit]]


# ---- Kaggle ----

def kaggle_search(query: str, device: DeviceProfile, *, limit: int = 5, fetcher=None) -> list[DataCard]:
    from nanomesh import kaggle

    get = fetcher or kaggle.fetch
    data = get(f"{kaggle.API}/datasets/list?" + urllib.parse.urlencode({"search": query, "page": 1}))
    items = data if isinstance(data, list) else (data or {}).get("datasets", [])
    out = []
    for d in items[:limit]:
        ref = d.get("ref") or "/".join(x for x in (d.get("ownerRef") or d.get("creatorName"), d.get("slug")) if x)
        size = d.get("totalBytes") or d.get("totalBytesNullable")
        card = DataCard(id=ref, source="kaggle", url=d.get("url") or f"https://www.kaggle.com/datasets/{ref}",
                        device=device.name, license=d.get("licenseName"), downloads=d.get("downloadCount"),
                        download_gb=round(size / GB, 3) if isinstance(size, (int, float)) else None,
                        how=f"kaggle datasets download -d {ref}  (pip install kaggle)")
        if card.download_gb:
            card.disk_gb = round(card.download_gb * 2, 3)  # the zip, plus it unpacked
            card.data_cost = _cost(card.download_gb)
        card.advice.append("Kaggle doesn't preview rows through its API: download it (or a single file) to look "
                           "inside.")
        _advise(card, device)
        out.append(card)
    return out


# ---- local files ----

def local_card(path: Path, device: DeviceProfile) -> DataCard:
    path = path.expanduser()
    files = sorted(p for p in (path.rglob("*") if path.is_dir() else [path]) if p.suffix.lower() in LOCAL_EXT)
    if not files:
        raise ValueError(f"No CSV, TSV, JSONL or JSON files at {path}.")
    size = sum(f.stat().st_size for f in files)
    card = DataCard(id=str(path), source="local", device=device.name, download_gb=None,
                    disk_gb=round(size / GB, 3), memory_gb=round(size * 2 / GB, 3))  # parsed in memory: ~2x
    rows_total, sample = 0, []
    for f in files:
        n, cols, first = _read_local(f)
        rows_total += n
        if not card.columns:
            card.columns = cols
        sample += first
        card.splits[f.name] = n
    card.rows = rows_total
    card.avg_tokens_per_row = _tokens_per_row(sample)
    card.preview = [{k: _cell(v) for k, v in r.items()} for r in sample[:PREVIEW_ROWS]]
    card.modality = _modality([], card.columns, sample)
    if len(files) > 1:
        card.advice.append(f"{len(files)} files: rows are summed; columns and preview come from {files[0].name}.")
    _advise(card, device)
    return card


def _kind(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int64"
    if isinstance(v, float):
        return "float64"
    if isinstance(v, str):
        try:
            float(v)
            return "number"
        except ValueError:
            return "string"
    return "list" if isinstance(v, list) else "struct" if isinstance(v, dict) else "string"


def _read_local(f: Path) -> tuple[int, list[Column], list[dict]]:
    ext = f.suffix.lower()
    if ext in (".csv", ".tsv"):
        with f.open(encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t" if ext == ".tsv" else ",")
            first, n = [], 0
            for row in reader:
                if n < 20:
                    first.append(dict(row))
                n += 1
            cols = [Column(name=c, type=_kind(first[0].get(c, "")) if first else "string")
                    for c in reader.fieldnames or []]
            return n, cols, first
    if ext == ".jsonl":
        first, n = [], 0
        with f.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                if n < 20:
                    try:
                        first.append(json.loads(line))
                    except ValueError:
                        pass
                n += 1
    else:  # .json: a list of records, or {"data": [...]}
        data = json.loads(f.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        records = [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
        n, first = len(records), records[:20]
    cols = [Column(name=k, type=_kind(v)) for k, v in (first[0].items() if first else [])]
    return n, cols, first


# ---- the card ----

def card(dataset: str, device: DeviceProfile, fetch: Fetch | None = None) -> DataCard:
    """A Dataset Card for a Hugging Face id, kaggle:<owner>/<slug>, or a local file or folder."""
    path = Path(dataset).expanduser()
    if path.exists():
        return local_card(path, device)
    if path.is_absolute() or dataset.startswith((".", "~")) or "\\" in dataset or dataset.count("/") > 1:
        raise ValueError(f"No such file or folder: {dataset}")  # a path, not a Hugging Face id (owner/name)
    if dataset.startswith("kaggle:"):
        ref = dataset.removeprefix("kaggle:")
        found = [c for c in kaggle_search(ref.split("/")[-1], device, limit=20) if c.id == ref]
        if not found:
            raise CatalogError(f"Kaggle has no dataset {ref}.")
        return found[0]
    if "/" not in dataset and not re.fullmatch(r"[\w.-]+", dataset):
        raise ValueError(f"Not a dataset id or file: {dataset}")
    return hf_card(dataset, device, fetch)


def training_fit(data: DataCard, model_name: str, device: DeviceProfile, *, epochs: int = 1,
                 seq_len: int = 1024) -> TrainingFit:
    """How long fine-tuning a model on this dataset takes here and in the cloud."""
    from nanomesh.fit import _calibrated, cpu_gflops
    from nanomesh.model import analyze
    from nanomesh.training import train_plan

    info = analyze(model_name)
    tp = train_plan(info, device, seq_len=seq_len)
    best = next((o for o in tp.options if o.method == tp.recommended), None)
    per_row = min(data.avg_tokens_per_row or seq_len / 2, seq_len)
    # Fine-tuning uses the training split(s): test, validation and unlabelled splits don't count.
    train_rows = sum(n or 0 for name, n in data.splits.items() if re.search(r"(^|/)train\b", name))
    rows = train_rows or data.rows or 0
    tokens = int(rows * per_row)
    # Training compute: ~6 FLOPs per parameter per token for full fine-tuning; LoRA/QLoRA skip the
    # weight gradients of the frozen base, ~4.
    flops = (6 if tp.recommended == "full" else 4) * info.params * tokens * epochs
    advice = []
    if train_rows and data.rows and train_rows < data.rows:
        advice.append(f"Counting the training split: {train_rows:,} of {data.rows:,} rows.")
    if not tokens:
        advice.append("The dataset's size in rows isn't known, so training time can't be estimated.")
    hours_here = None
    if best and tokens:
        if tp.placement == "GPU":
            gpu = device.best_gpu
            tflops = _gpu_tflops(gpu.name if gpu else "")
            hours_here = round(flops / (tflops * 1e12) / 3600, 1) if tflops else None
        else:
            gflops = _calibrated(device).get("onnx") or cpu_gflops(device)
            hours_here = round(flops / (gflops * 1e9 * 0.5) / 3600, 1)  # backward passes run slower than inference
    cloud = {name: round(flops / (tf * 1e12) / 3600, 2) for name, tf in CLOUD_GPUS.items()} if tokens else {}
    if hours_here and hours_here > 24:
        advice.append(f"~{hours_here:,.0f} h here: use a cloud GPU, or train on a slice first "
                      f"({int(rows * 24 / hours_here):,} rows fit in a day here)" if rows else "")
    if data.avg_tokens_per_row and data.avg_tokens_per_row > seq_len:
        advice.append(f"Rows average ~{data.avg_tokens_per_row:,.0f} tokens but training cuts them at {seq_len}: "
                      "raise --seq-len (more memory) or split long rows.")
    advice += [a for a in tp.advice if a]
    return TrainingFit(model=info.name, method=tp.recommended, memory_gb=best.total_gb if best else None,
                       fits=best is not None, tokens=tokens, epochs=epochs, hours_here=hours_here,
                       hours_cloud=cloud, advice=[a for a in advice if a])


GPU_TFLOPS = {"4090": 60.0, "3090": 30.0, "3060": 10.0, "4060": 18.0, "a100": 120.0, "h100": 350.0, "t4": 25.0,
              "l4": 45.0, "a10": 45.0, "orin": 5.0}


def _gpu_tflops(name: str) -> float | None:
    n = name.lower()
    return next((v for k, v in GPU_TFLOPS.items() if k in n), None)
