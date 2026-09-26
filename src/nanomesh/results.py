"""Local store of benchmark results; measured numbers beat estimates.

Results are appended to ``$NANOMESH_HOME/results.jsonl`` (default
``~/.nanomesh``), one JSON object per line, so the file can later be
uploaded to a shared benchmark database as-is.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from pydantic import BaseModel

from nanomesh import __version__
from nanomesh.hardware import DeviceProfile
from nanomesh.model import ModelInfo
from nanomesh.planner import Evidence

# llama.cpp's llama_ftype values for the formats NanoMesh builds.
GGUF_FILE_TYPES = {0: "F32", 1: "F16", 7: "Q8_0", 10: "Q2_K", 12: "Q3_K_M", 15: "Q4_K_M",
                   17: "Q5_K_M", 18: "Q6_K", 32: "BF16"}
# Params may differ slightly between a safetensors model and its GGUF
# conversion (tied embeddings, etc.), so match within a tolerance.
PARAMS_TOLERANCE = 0.03


class Result(BaseModel):
    timestamp: str
    nanomesh_version: str = __version__
    device_key: str
    device_name: str
    model_name: str
    model_params: int
    model_architecture: str | None = None
    format: str | None
    file_size_gb: float
    prompt_tokens_per_s: float | None = None
    gen_tokens_per_s: float | None = None
    peak_rss_gb: float | None = None
    backend: str | None = None
    threads: int | None = None
    perplexity: float | None = None
    reference_format: str | None = None
    quality_pct: float | None = None


def home() -> Path:
    return Path(os.environ.get("NANOMESH_HOME", "~/.nanomesh")).expanduser()


def results_path() -> Path:
    return home() / "results.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gguf_format(meta: dict, path: Path) -> str | None:
    ftype = meta.get("general.file_type")
    if ftype in GGUF_FILE_TYPES:
        return GGUF_FILE_TYPES[ftype]
    m = re.search(r"(F16|BF16|Q8_0|Q6_K|Q5_K_M|Q4_K_M|Q3_K_M|Q2_K)", path.name, re.IGNORECASE)
    return m.group(1).upper() if m else None


def save(results: list[Result]) -> Path:
    path = results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
    return path


def load() -> list[Result]:
    path = results_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(Result.model_validate_json(line))
        except ValueError:
            continue  # tolerate partial writes / old formats
    return out


def _same_model(r: Result, model: ModelInfo) -> bool:
    return abs(r.model_params - model.params) <= model.params * PARAMS_TOLERANCE


def evidence(device: DeviceProfile, model: ModelInfo, results: list[Result] | None = None) -> Evidence:
    """Collect what's been measured on this device, for this model and overall."""
    results = [r for r in (load() if results is None else results) if r.device_key == device.key]
    speeds, quality = {}, {}
    for r in results:  # later results overwrite earlier ones
        if r.format and _same_model(r, model):
            if r.gen_tokens_per_s:
                speeds[r.format] = r.gen_tokens_per_s
            if r.quality_pct is not None:
                quality[r.format] = r.quality_pct
    # Generation streams the whole model once per token, so tok/s x model size
    # is the bandwidth this device actually delivers to llama.cpp.
    implied = [r.gen_tokens_per_s * r.file_size_gb * (1024**3 / 1e9)
               for r in results if r.gen_tokens_per_s and r.file_size_gb >= 0.05]
    return Evidence(speeds=speeds, quality=quality,
                    effective_bandwidth_gbps=round(median(implied), 1) if implied else None,
                    calibration_runs=len(implied))


def rows(results: list[Result]) -> list[dict]:
    return [json.loads(r.model_dump_json()) for r in results]
