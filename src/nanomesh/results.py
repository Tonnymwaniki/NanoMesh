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
from nanomesh.conditions import RunConditions
from nanomesh.hardware import DeviceProfile
from nanomesh.model import ModelInfo
from nanomesh.planner import FORMATS_BY_NAME, BatteryCost, Evidence

# llama.cpp's llama_ftype values for the formats NanoMesh builds.
GGUF_FILE_TYPES = {0: "F32", 1: "F16", 7: "Q8_0", 10: "Q2_K", 12: "Q3_K_M", 15: "Q4_K_M",
                   17: "Q5_K_M", 18: "Q6_K", 32: "BF16"}
# Params may differ slightly between a safetensors model and its GGUF
# conversion (tied embeddings, etc.), so match within a tolerance.
PARAMS_TOLERANCE = 0.03
GIB_TO_GB = 1024**3 / 1e9


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
    # "benchmark" (llama-bench defaults), "sustained" (minutes of generation) or
    # "threads" (one row of a thread-count sweep).
    kind: str = "benchmark"
    # Steady-state benchmarks warm up until speed settles before measuring, so
    # they reflect long sessions; quick ones may catch a laptop's turbo phase.
    steady: bool = False
    warmup_s: float | None = None
    burst_tokens_per_s: float | None = None  # speed when started cold (turbo)
    # Where continuous generation settled during the warm-up, and when speed
    # dropped. Compared with gen_tokens_per_s it shows whether llama-bench's
    # pauses (model load, prompt) let a laptop's turbo budget recover.
    warmup_settled_tokens_per_s: float | None = None
    warmup_drop_at_s: float | None = None
    conditions: RunConditions | None = None
    sustained: SustainedRun | None = None

    @property
    def on_battery(self) -> bool:
        return bool(self.conditions and self.conditions.on_battery_any)


class SustainedPoint(BaseModel):
    t_s: float
    tokens_per_s: float
    temp_c: float | None = None
    clock_pct: float | None = None
    battery_pct: float | None = None
    discharge_w: float | None = None
    battery_wh: float | None = None


class SustainedRun(BaseModel):
    """Generation kept up for minutes: what heat and power limits do to speed."""

    points: list[SustainedPoint]
    burst_tokens_per_s: float
    sustained_tokens_per_s: float
    drop_pct: float
    watts: float | None = None  # battery draw while generating
    joules_per_token: float | None = None
    battery_hours: float | None = None  # full charge at this load
    tokens_per_battery_pct: float | None = None
    drop_at_s: float | None = None  # when speed fell halfway to its sustained level
    drop_pattern: str | None = None  # "step" (turbo budget ran out) or "gradual" (heat)
    # How the battery figures were measured: "power sensor", "charge counter"
    # (remaining Wh falling) or "battery %" (1% steps: rough on short runs).
    energy_source: str | None = None
    battery_hours_range: tuple[float, float] | None = None


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


def _slug(name: str) -> str:
    return re.sub(r"[\s_/]+", "-", name.lower()).strip("-")


def _same_model(r: Result, model: ModelInfo) -> bool:
    """Same model family and size, even if the parameter counts disagree.

    GGUF conversions often store tied embeddings twice (Qwen2.5-1.5B: 1.54B
    params on Hugging Face, 1.78B in the official GGUF), so names are matched
    first: "qwen2.5-1.5b" matches "Qwen2.5 1.5B Instruct" at a word boundary.
    """
    a, b = _slug(r.model_name), _slug(model.name)
    shorter, longer = sorted((a, b), key=len)
    if shorter and longer.startswith(shorter) and longer[len(shorter):len(shorter) + 1] in ("", "-"):
        return abs(r.model_params - model.params) <= model.params * 0.25
    return abs(r.model_params - model.params) <= model.params * PARAMS_TOLERANCE


# A run delivering under this share of the device's best bandwidth was held
# back by something else: the CPU unpacking low-bit weights.
COMPUTE_BOUND_BELOW = 0.90
# Near-lossless formats' measured loss is mostly noise; don't calibrate from them.
MIN_TYPICAL_LOSS = 1.0


def evidence(device: DeviceProfile, model: ModelInfo, results: list[Result] | None = None) -> Evidence:
    """Collect what's been measured on this device, for this model and overall."""
    results = [r for r in (load() if results is None else results) if r.device_key == device.key]
    battery_cost = _battery_cost(results)
    # Laptops slow down on battery, so plugged-in runs are the reference: when
    # both exist, only plugged-in runs count.
    if any(r.on_battery for r in results) and not all(r.on_battery for r in results):
        results = [r for r in results if not r.on_battery]
    speeds, quality, params = {}, {}, None
    # Later results overwrite earlier ones, but a steady-state measurement
    # always beats a quick one: quick runs may have caught a turbo phase.
    for r in sorted(results, key=lambda r: r.steady):
        if r.kind != "benchmark":
            continue  # thread sweeps and sustained runs aren't default-settings speeds
        if r.format and _same_model(r, model):
            params = r.model_params
            if r.gen_tokens_per_s:
                speeds[r.format] = r.gen_tokens_per_s
            if r.quality_pct is not None:
                quality[r.format] = r.quality_pct

    # Roofline calibration. Generating a token streams the whole model through
    # memory once, so tok/s x file size is the bandwidth a run achieved; the
    # best run shows what the device can deliver. Runs well below that were
    # compute-bound: the CPU couldn't unpack the weights any faster.
    bench_runs = [r for r in results if r.gen_tokens_per_s and r.file_size_gb >= 0.05 and r.kind == "benchmark"]
    if any(r.steady for r in bench_runs):
        bench_runs = [r for r in bench_runs if r.steady]  # don't calibrate from turbo bursts
    runs = [(r.gen_tokens_per_s * r.file_size_gb * GIB_TO_GB, r.gen_tokens_per_s * r.model_params / 1e9)
            for r in bench_runs]
    bandwidth = max((bw for bw, _ in runs), default=None)
    # Every run proves the CPU manages at least tok/s x params, so the ceiling
    # is the best any run achieved (a 7B model gets more out of the CPU than a
    # 1.5B one: 30 vs 27 G params/s on an i5-8365U). It only applies once some
    # run was actually held back by it; otherwise it's just a loose lower bound.
    cpu_bound = any(bw < bandwidth * COMPUTE_BOUND_BELOW for bw, _ in runs)
    compute = max(c for _, c in runs) if cpu_bound else None

    threads, gain = _best_threads(results, device)
    sustained_runs = [r.sustained for r in results if r.kind == "sustained" and r.sustained]

    return Evidence(speeds=speeds, quality=quality, model_params=params,
                    best_threads=threads, best_threads_gain_pct=gain, battery_cost=battery_cost,
                    sustained_drop_pct=sustained_runs[-1].drop_pct if sustained_runs else None,
                    effective_bandwidth_gbps=round(bandwidth, 1) if bandwidth else None,
                    compute_gparams_per_s=round(compute, 1) if compute else None,
                    quality_loss_scale=_quality_loss_scale(quality),
                    calibration_runs=len(runs))


def _battery_cost(results: list[Result]) -> BatteryCost | None:
    """Compare the latest sustained runs of one model plugged in vs on battery."""
    runs = [r for r in results if r.kind == "sustained" and r.sustained and r.conditions
            and r.conditions.start.on_battery is not None]
    for model in {(r.model_name, r.format) for r in reversed(runs)}:
        mine = [r for r in runs if (r.model_name, r.format) == model]
        plugged = next((r for r in reversed(mine) if not r.on_battery), None)
        battery = next((r for r in reversed(mine) if r.on_battery), None)
        if plugged and battery:
            p, b = plugged.sustained, battery.sustained
            return BatteryCost(model=f"{model[0]} {model[1]}",
                               plugged_burst=p.burst_tokens_per_s, plugged_sustained=p.sustained_tokens_per_s,
                               battery_burst=b.burst_tokens_per_s, battery_sustained=b.sustained_tokens_per_s)
    return None


def _best_threads(results: list[Result], device: DeviceProfile) -> tuple[int | None, float | None]:
    """Best thread count from the latest `nanomesh tune` sweep on this device,
    and its gain over llama.cpp's default (one thread per physical core)."""
    sweep = [r for r in results if r.kind == "threads" and r.threads and r.gen_tokens_per_s]
    if not sweep:
        return None, None
    latest = [r for r in sweep if (r.model_name, r.format, r.timestamp) == (sweep[-1].model_name, sweep[-1].format, sweep[-1].timestamp)]
    best = max(latest, key=lambda r: r.gen_tokens_per_s)
    default_threads = device.physical_cores or max(r.threads for r in latest)
    baseline = next((r for r in latest if r.threads == default_threads), None)
    if baseline is None or baseline is best:
        return best.threads, None
    return best.threads, round(100 * (best.gen_tokens_per_s / baseline.gen_tokens_per_s - 1), 1)


def _quality_loss_scale(quality: dict[str, float]) -> float | None:
    """How much more (or less) quality this model loses than the typical figures.

    Small models lose far more to quantization than the 7B-class models the
    typical figures come from (Qwen2.5-1.5B lost ~3x at INT4 and INT3).
    """
    ratios = []
    for name, pct in quality.items():
        fmt = FORMATS_BY_NAME.get(name.lower())
        if fmt and 100 - fmt.typical_quality_pct >= MIN_TYPICAL_LOSS:
            ratios.append(max(0.0, 100 - pct) / (100 - fmt.typical_quality_pct))
    return round(median(ratios), 2) if ratios else None


def rows(results: list[Result]) -> list[dict]:
    return [json.loads(r.model_dump_json()) for r in results]
