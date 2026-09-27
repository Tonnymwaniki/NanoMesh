"""Fit Cards: how well a model fits this device, for any kind of model.

A model has requirements (memory, compute per unit of work, download size,
licence); a device has capabilities (memory budget, compute, bandwidth,
battery). The Fit Card is the match: does it fit, how fast in the unit that
matters for the task (x real-time for speech, images/s for vision, sentences/s
for embeddings, tok/s for text), what it costs to download, what a full battery
buys, and a better-fitting member of the same family when there is one.

Estimates come from published per-model compute figures and the device's
compute; a benchmark on the device (ONNX Runtime, whisper.cpp) replaces the
estimate for that runtime, the same way llama.cpp benchmarks calibrate text
models.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from nanomesh import config
from nanomesh import results as store
from nanomesh.hardware import GB, DeviceProfile
from nanomesh.planner import memory_budgets

AVG_WHISPER_TOKENS_PER_AUDIO_S = 3  # speech is ~2-3 words per second
EMBED_TOKENS = 32  # a typical sentence or search chunk


class Member(BaseModel):
    """One model of a family, with published figures."""

    id: str
    params_m: float
    gflops: float | None = None  # per unit of work (vision, embeddings)
    enc_m: float | None = None  # Whisper: encoder / decoder parameters (M)
    dec_m: float | None = None
    quality: str
    rank: int  # order within the family by accuracy (0: dominated by another member, never suggested)
    ref: str | None = None  # the name the usual tool knows it by, when it differs from id


class Family(BaseModel):
    series: str
    prefix: str = ""  # member display name = prefix + id: whisper-small, yolov8n
    task: str  # speech-to-text | object detection | image classification | embeddings
    unit: str  # what speed is counted in
    runtime: str  # onnx | whisper.cpp: which benchmarks calibrate it
    bytes_per_param: float  # of the usual download
    usable: float  # speed below which it's not practical for the task
    how: str
    members: list[Member]
    pattern: str  # regex; group "m" names the member when present


FAMILIES = [
    Family(series="whisper", prefix="whisper-", task="speech-to-text", unit="x real-time", runtime="whisper.cpp", bytes_per_param=2.0,
           usable=1.0, pattern=r"(distil-)?whisper[-_. ]?(?P<m>large-v3-turbo|turbo|large-v3|large-v2|large|medium|"
                               r"small|base|tiny)?|ggml-(?P<g>tiny|base|small|medium|large-v3-turbo|large-v3)",
           how="whisper.cpp with ggml-{m}.bin from huggingface.co/ggerganov/whisper.cpp, or faster-whisper "
               "(pip install faster-whisper) with model '{m}'.",
           members=[
               Member(id="tiny", params_m=39, enc_m=8, dec_m=31, rank=1,
                      quality="lowest Whisper accuracy: fine for clear speech and commands"),
               Member(id="base", params_m=74, enc_m=20, dec_m=52, rank=2,
                      quality="basic accuracy: clear speech, main languages"),
               Member(id="small", params_m=244, enc_m=88, dec_m=153, rank=3,
                      quality="good accuracy for most languages; the usual laptop choice"),
               Member(id="medium", params_m=769, enc_m=307, dec_m=457, rank=4,
                      quality="high accuracy, better on accents and noisy audio"),
               Member(id="large-v3-turbo", params_m=809, enc_m=635, dec_m=172, rank=5,
                      quality="close to large-v3 accuracy with a much smaller decoder"),
               Member(id="distil-large-v3", params_m=756, enc_m=635, dec_m=121, rank=5,
                      quality="close to large-v3 accuracy in English only, with a 2-layer decoder"),
               Member(id="large-v3", params_m=1550, enc_m=635, dec_m=907, rank=6,
                      quality="best Whisper accuracy, including low-resource languages"),
           ]),
    Family(series="yolo11", prefix="yolo11", task="object detection", unit="images/s", runtime="onnx", bytes_per_param=4.0, usable=2.0,
           pattern=r"yolo[-_ ]?v?11(?P<m>[nsmlx])?\b",
           how="pip install ultralytics, then: yolo export model=yolo11{m}.pt format=onnx (runs with ONNX Runtime "
               "on CPU; add format=openvino for Intel GPUs).",
           members=[Member(id="n", params_m=2.6, gflops=6.5, rank=1, quality="COCO mAP 39.5 (published)"),
                    Member(id="s", params_m=9.4, gflops=21.5, rank=2, quality="COCO mAP 47.0 (published)"),
                    Member(id="m", params_m=20.1, gflops=68.0, rank=3, quality="COCO mAP 51.5 (published)"),
                    Member(id="l", params_m=25.3, gflops=86.9, rank=4, quality="COCO mAP 53.4 (published)"),
                    Member(id="x", params_m=56.9, gflops=194.9, rank=5, quality="COCO mAP 54.7 (published)")]),
    Family(series="yolov8", prefix="yolov8", task="object detection", unit="images/s", runtime="onnx", bytes_per_param=4.0, usable=2.0,
           pattern=r"yolo[-_ ]?v8(?P<m>[nsmlx])?\b",
           how="pip install ultralytics, then: yolo export model=yolov8{m}.pt format=onnx.",
           members=[Member(id="n", params_m=3.2, gflops=8.7, rank=1, quality="COCO mAP 37.3 (published)"),
                    Member(id="s", params_m=11.2, gflops=28.6, rank=2, quality="COCO mAP 44.9 (published)"),
                    Member(id="m", params_m=25.9, gflops=78.9, rank=3, quality="COCO mAP 50.2 (published)"),
                    Member(id="l", params_m=43.7, gflops=165.2, rank=4, quality="COCO mAP 52.9 (published)"),
                    Member(id="x", params_m=68.2, gflops=257.8, rank=5, quality="COCO mAP 53.9 (published)")]),
    Family(series="rt-detr", prefix="rtdetr-r", task="object detection", unit="images/s", runtime="onnx",
           bytes_per_param=4.0, usable=2.0, pattern=r"rt-?detr(?:-?v2)?[-_]?r(?P<m>18|34|50|101)",
           how="transformers (RTDetrForObjectDetection, 'PekingU/rtdetr_r{m}vd'), or export it to ONNX with "
               "optimum-cli export onnx for ONNX Runtime.",
           members=[Member(id="18", params_m=20, gflops=60, rank=1, quality="COCO AP 46.5, v2 47.9 (published)"),
                    Member(id="34", params_m=31, gflops=92, rank=2, quality="COCO AP 48.9, v2 49.9 (published)"),
                    Member(id="50", params_m=42, gflops=136, rank=3, quality="COCO AP 53.1, v2 53.4 (published)"),
                    Member(id="101", params_m=76, gflops=259, rank=4, quality="COCO AP 54.3, v2 54.3 (published)")]),
    Family(series="detr", prefix="detr-resnet-", task="object detection", unit="images/s", runtime="onnx",
           bytes_per_param=4.0, usable=2.0, pattern=r"(^|/)detr-resnet-(?P<m>50|101)",
           how="transformers (DetrForObjectDetection, 'facebook/detr-resnet-{m}'); RT-DETR is faster and more "
               "accurate at the same size.",
           members=[Member(id="50", params_m=41, gflops=86, rank=1, quality="COCO AP 42.0 (published)"),
                    Member(id="101", params_m=60, gflops=152, rank=2, quality="COCO AP 43.5 (published)")]),
    Family(series="image classifiers", task="image classification", unit="images/s", runtime="onnx",
           bytes_per_param=4.0, usable=10.0,
           pattern=r"(?P<m>mobilenet[-_]?v3[-_]?small|mobilenet[-_]?v3[-_]?large|efficientnet[-_]?b0|resnet[-_]?18|"
                   r"resnet[-_]?50|vit[-_]base[-_]patch16)",
           how="ONNX from the ONNX Model Zoo or timm (timm.create_model('{m}', pretrained=True), then export to "
               "ONNX); on phones use the LiteRT/TFLite version.",
           members=[Member(id="mobilenet-v3-small", ref="mobilenetv3_small_100", params_m=2.5, gflops=0.06, rank=1,
                           quality="ImageNet top-1 67.4% (published)"),
                    Member(id="mobilenet-v3-large", ref="mobilenetv3_large_100", params_m=5.4, gflops=0.22, rank=2,
                           quality="ImageNet top-1 75.2% (published)"),
                    Member(id="efficientnet-b0", ref="efficientnet_b0", params_m=5.3, gflops=0.39, rank=4,
                           quality="ImageNet top-1 77.1% (published)"),
                    Member(id="resnet-18", ref="resnet18", params_m=11.7, gflops=1.8, rank=0,
                           quality="ImageNet top-1 69.8% (published); MobileNetV3-Large is faster and more accurate"),
                    Member(id="resnet-50", ref="resnet50", params_m=25.6, gflops=4.1, rank=3,
                           quality="ImageNet top-1 76.1% (published); EfficientNet-B0 is faster and more accurate"),
                    Member(id="vit-base-patch16", ref="vit_base_patch16_224", params_m=86.6, gflops=17.6, rank=5,
                           quality="ImageNet top-1 ~81% (published)")]),
    Family(series="embeddings", task="embeddings", unit="sentences/s", runtime="onnx", bytes_per_param=4.0,
           usable=20.0,
           pattern=r"(?P<m>all-minilm-l6-v2|bge-small-en|bge-base-en|nomic-embed-text|e5-small)",
           how="sentence-transformers (pip install sentence-transformers) with '{m}', or its GGUF with "
               "llama-server --embeddings.",
           members=[Member(id="all-minilm-l6-v2", ref="sentence-transformers/all-MiniLM-L6-v2", params_m=22.7, rank=1, quality="fast general English search"),
                    Member(id="e5-small", ref="intfloat/e5-small-v2", params_m=33.4, rank=2, quality="good English retrieval, small"),
                    Member(id="bge-small-en", ref="BAAI/bge-small-en-v1.5", params_m=33.4, rank=2, quality="good English retrieval, small"),
                    Member(id="bge-base-en", ref="BAAI/bge-base-en-v1.5", params_m=109.5, rank=3, quality="better English retrieval"),
                    Member(id="nomic-embed-text", ref="nomic-ai/nomic-embed-text-v1.5", params_m=137, rank=3,
                           quality="strong retrieval, long inputs (8k tokens)")]),
]


class FitCard(BaseModel):
    model: str
    task: str
    device: str
    fits: bool
    memory_gb: float | None = None
    budget_gb: float | None = None
    speed: float | None = None
    speed_unit: str | None = None
    speed_source: str | None = None  # measured | calibrated | estimate
    usable: bool | None = None  # fast enough for the task
    quality: str | None = None
    download_gb: float | None = None
    data_cost: str | None = None
    battery: str | None = None
    license: str | None = None
    alternative: str | None = None
    how: str | None = None
    notes: list[str] = []


def display(fam: Family, member: Member) -> str:
    return fam.prefix + member.id


def match(name: str) -> tuple[Family, Member | None] | None:
    """The family (and member, when the name says which) a model name belongs to."""
    n = re.sub(r"[\s_]+", "-", name.lower())
    for fam in FAMILIES:
        m = re.search(fam.pattern, n)
        if not m:
            continue
        which = (m.groupdict().get("m") or m.groupdict().get("g") or "").replace("_", "-")
        if fam.series == "whisper":
            if "distil" in n:
                which = "distil-large-v3"
            which = {"turbo": "large-v3-turbo", "large": "large-v3", "large-v2": "large-v3"}.get(which, which)
        which = which.replace("mobilenetv3", "mobilenet-v3")
        member = next((x for x in fam.members if x.id == which or _slug(x.id) == _slug(which)), None)
        return fam, member
    return None


def _slug(s: str) -> str:
    return re.sub(r"[-_ ]", "", s.lower())


# ---- device compute ----

def cpu_gflops(device: DeviceProfile) -> float:
    """Sustained CPU throughput for inference, roughly: cores x per-core rate by
    instruction set. A benchmark on the device replaces this."""
    cores = device.physical_cores or device.logical_cores or 2
    flags = {f.lower() for f in device.cpu_flags or []}
    cpu = (device.cpu or "").lower()
    if device.arch in ("x86_64", "amd64"):
        per_core = 25 if any(f.startswith("avx512") for f in flags) else 15 if "avx2" in flags or not flags else 8
    elif "apple" in cpu or re.search(r"\bm[1-4]\b", cpu):
        per_core = 30
    elif device.kind == "phone":
        per_core = 4
    else:
        per_core = 6
    return float(cores * per_core)


def _calibrated(device: DeviceProfile) -> dict[str, float]:
    """GFLOPS each runtime achieved in benchmarks on this device (quiet runs
    preferred: a busy machine understates it)."""
    runs = [r for r in store.load() if r.device_key == device.key and r.throughput and r.gflops_per_unit
            and r.kind in ("onnx", "whisper.cpp")]
    if any(not r.busy for r in runs):
        runs = [r for r in runs if not r.busy]
    out: dict[str, float] = {}
    for r in runs:
        out[r.kind] = max(out.get(r.kind, 0), r.throughput * r.gflops_per_unit)
    return out


def _bandwidth(device: DeviceProfile, model_name: str = "") -> float:
    from nanomesh.model import ModelInfo

    ev = store.evidence(device, ModelInfo(name=model_name or "-", source="size", params=1))
    return ev.effective_bandwidth_gbps or (device.memory_bandwidth_gbps or 10.0) * 0.55


def _benchmarked(device: DeviceProfile, fam: Family, member: Member):
    """The latest quiet benchmark of this exact model on this device."""
    runs = [r for r in store.load() if r.device_key == device.key and r.kind == fam.runtime and r.throughput
            and _slug(r.model_name) == _slug(display(fam, member))]
    return store._quiet_first(runs)[-1] if runs else None


def speed(fam: Family, member: Member, device: DeviceProfile) -> tuple[float | None, str]:
    run = _benchmarked(device, fam, member)
    if run and fam.series != "whisper":
        return run.throughput, "measured"
    cal = _calibrated(device).get(fam.runtime)
    gflops, source = (cal, "calibrated") if cal else (cpu_gflops(device), "estimate")
    if fam.series == "whisper":
        if run:  # whisper-bench measured this model's encoder: use exactly that
            gflops, source = run.throughput * run.gflops_per_unit, "measured"
        # Per second of audio: the encoder is compute-bound (50 frames/s), the
        # decoder reads its weights once per token, like a text model.
        enc_s = 2 * member.enc_m * 50 / 1000 / gflops
        dec_s = AVG_WHISPER_TOKENS_PER_AUDIO_S * member.dec_m * 1e6 * fam.bytes_per_param / 1e9 / _bandwidth(device)
        return _round(1 / (enc_s + dec_s)), source
    per_unit = member.gflops if member.gflops is not None else 2 * member.params_m * EMBED_TOKENS / 1000
    return _round(gflops / per_unit), source


def _round(x: float) -> float:
    """One decimal, but keep two significant digits for slow speeds (0.04 images/s is not 0)."""
    return round(x, 1) if x >= 1 else float(f"{x:.2g}")


def unit_gflops(fam: Family, member: Member) -> float:
    if member.gflops is not None:
        return member.gflops
    if fam.series == "whisper":
        return 2 * member.enc_m * 50 / 1000
    return 2 * member.params_m * EMBED_TOKENS / 1000


def memory_gb(fam: Family, member: Member) -> float:
    weights = member.params_m * 1e6 * fam.bytes_per_param / GB
    working = {"speech-to-text": 0.25, "object detection": 0.35, "image classification": 0.1,
               "embeddings": 0.1}[fam.task]
    return round(weights * 1.1 + working, 2)


# ---- the card ----

# Formats only some devices can run, whatever their memory.
PLATFORM_ONLY = [
    (re.compile(r"coreml|whisperkit|\.mlmodel|mlpackage"), "apple", "it's a Core ML model, which runs on Apple devices only"),
    (re.compile(r"(^|[^a-z])mlx([^a-z]|$)"), "apple-silicon", "it's an MLX model, which runs on Apple Silicon Macs only"),
    (re.compile(r"tensorrt|(^|[^a-z])(trt|awq|gptq|exl2)([^a-z]|$)"), "nvidia", "it needs an NVIDIA GPU (TensorRT, "
                                                                            "AWQ, GPTQ or EXL2 format)"),
]


def _is_apple(device: DeviceProfile) -> bool:
    os_name = (device.os or "").lower()
    return "mac" in os_name or "darwin" in os_name or "ios" in os_name or any(g.vendor == "Apple" for g in device.gpus)


def cannot_run(name: str, tags: list[str] | None, device: DeviceProfile) -> str | None:
    """Why this device can't run the model's format at all, or None."""
    text = " ".join([name.lower(), *[t.lower() for t in tags or [] if t]])
    for pattern, needs, why in PLATFORM_ONLY:
        if not pattern.search(text):
            continue
        ok = {"apple": _is_apple(device),
              "apple-silicon": _is_apple(device) and device.arch in ("arm64", "aarch64"),
              "nvidia": any(g.vendor == "NVIDIA" for g in device.gpus)}[needs]
        if not ok:
            return why
    return None


def _approx(n: float) -> str:
    """Round big counts to two significant figures: 112,608 -> 110,000."""
    if n < 100:
        return f"{n:.0f}"
    digits = len(str(int(n))) - 2
    return f"{round(n, -digits):,.0f}"


def _data_cost(gb: float) -> str | None:
    price = config.get("data_price")
    if not price or not gb:
        return None
    return f"~{gb * float(price['per_gb']):,.0f} {price.get('currency', '')}".strip() + " of data"


def _battery(device: DeviceProfile, fam: Family, rate: float | None) -> str | None:
    runs = [r.sustained for r in store._quiet_first(r for r in store.load() if r.device_key == device.key
                                                     and r.kind == "sustained" and r.sustained
                                                     and r.sustained.battery_hours)]
    if not runs or not rate:
        return None
    hours = runs[-1].battery_hours
    if fam.task == "speech-to-text":
        return f"~{hours * rate:.0f} h of audio per full charge (battery life measured under AI load: {hours:g} h)"
    what = {"images/s": "images", "sentences/s": "sentences"}.get(fam.unit, "units")
    return f"~{_approx(hours * 3600 * rate)} {what} per full charge (battery life measured under AI load: {hours:g} h)"


def card(fam: Family, member: Member, device: DeviceProfile, *, name: str | None = None,
         download_bytes: int | None = None, license: str | None = None) -> FitCard:
    budget = max(b.memory_gb for b in memory_budgets(device))
    mem = memory_gb(fam, member)
    rate, source = speed(fam, member, device)
    dl = round((download_bytes or member.params_m * 1e6 * fam.bytes_per_param) / GB, 2)
    c = FitCard(model=name or display(fam, member), task=fam.task, device=device.name, fits=mem <= budget,
                memory_gb=mem, budget_gb=round(budget, 1), speed=rate, speed_unit=fam.unit, speed_source=source,
                usable=rate is not None and rate >= fam.usable, quality=member.quality, download_gb=dl,
                data_cost=_data_cost(dl), battery=_battery(device, fam, rate), license=license,
                how=fam.how.format(m=member.ref or member.id))
    c.alternative = _alternative(fam, member, device, c)
    if source == "estimate":
        tool = "ONNX Runtime" if fam.runtime == "onnx" else fam.runtime
        c.notes.append("Speed is an estimate from published compute figures. Benchmark any model of this kind "
                       f"(nanomesh benchmark <file>, {tool}) to make it a measurement for this device.")
    if device.best_gpu and fam.runtime == "onnx":
        c.notes.append("CPU figures: with the GPU (ONNX Runtime CUDA/DirectML, or OpenVINO on Intel) it runs faster.")
    return c


def _alternative(fam: Family, member: Member, device: DeviceProfile, c: FitCard) -> str | None:
    ranked = sorted((m for m in fam.members if m.rank > 0), key=lambda m: m.rank)
    here = [m for m in ranked if memory_gb(fam, m) <= (c.budget_gb or 0)]

    def describe(m: Member) -> str:
        rate, _ = speed(fam, m, device)
        return f"{display(fam, m)}: {rate:g} {fam.unit}, {m.quality}"

    if not c.fits or not c.usable:
        smaller = [m for m in here if 0 < m.rank < member.rank and (speed(fam, m, device)[0] or 0) >= fam.usable]
        if smaller:
            return "Fits better: " + describe(smaller[-1])
        # Nothing in this family runs well here: another family for the same task might.
        for other in FAMILIES:
            if other.task == fam.task and other is not fam:
                m = best(other, device)
                rate, _ = speed(other, m, device)
                if rate and rate >= other.usable:
                    return f"Runs well here instead: {display(other, m)}: {rate:g} {other.unit}, {m.quality}"
        return None
    bigger = [m for m in here if m.rank > member.rank and (speed(fam, m, device)[0] or 0) >= fam.usable * 2]
    if bigger:
        return "More accurate and still fast enough here: " + describe(bigger[-1])
    return None


def best(fam: Family, device: DeviceProfile) -> Member:
    """The most accurate member that fits and runs comfortably (twice the
    usable speed), else at a usable speed, else the smallest."""
    budget = max(b.memory_gb for b in memory_budgets(device))
    fitting = [m for m in fam.members if m.rank > 0 and memory_gb(fam, m) <= budget]
    for factor in (2, 1):
        ok = [m for m in fitting if (speed(fam, m, device)[0] or 0) >= fam.usable * factor]
        if ok:
            return max(ok, key=lambda m: m.rank)
    return min((m for m in fam.members if m.rank > 0), key=lambda m: m.rank)


def text_card(name: str, device: DeviceProfile, *, download_bytes: int | None = None,
              license: str | None = None) -> FitCard | None:
    """A Fit Card for a text model, from the planner."""
    from nanomesh.model import analyze
    from nanomesh.planner import Requirements, plan

    try:
        info = analyze(name)
    except ValueError:
        return None
    p = plan(info, device, Requirements(), store.evidence(device, info))
    v = next((x for x in p.variants if x.format.name == p.recommended), None)
    dl = round((download_bytes or (v.weights_gb * GB if v else 0)) / GB, 2) or None
    return FitCard(model=name, task="text generation", device=device.name, fits=v is not None,
                   memory_gb=v.total_memory_gb if v else None, budget_gb=round(max(b.memory_gb for b in p.budgets), 1),
                   speed=v.tokens_per_s if v else None, speed_unit="tok/s", speed_source=v.speed_source if v else None,
                   usable=bool(v and (v.tokens_per_s or 0) >= 5),
                   quality=f"{v.format.label} · ~{v.quality_pct:g}% of full precision ({v.quality_source})" if v else None,
                   download_gb=dl, data_cost=_data_cost(dl or 0), license=license,
                   how=f"nanomesh search {name}, then nanomesh pull and nanomesh serve",
                   notes=p.advice[:2])


GENERIC = {  # task -> (unit, runtime, usable, working memory GB)
    "speech-to-text": ("x real-time", "whisper.cpp", 1.0, 0.25),
    "image classification": ("images/s", "onnx", 10.0, 0.1),
    "object detection": ("images/s", "onnx", 2.0, 0.35),
    "embeddings": ("sentences/s", "onnx", 20.0, 0.1),
}


def generic_card(name: str, task: str, params: int, device: DeviceProfile, *, download_bytes: int | None = None,
                 license: str | None = None) -> FitCard:
    """A model outside the known families: sized from its parameter count."""
    unit, runtime, usable, working = GENERIC[task]
    budget = max(b.memory_gb for b in memory_budgets(device))
    mem = round(params * 4 / GB * 1.1 + working, 2)
    gflops = _calibrated(device).get(runtime) or cpu_gflops(device)
    source = "calibrated" if _calibrated(device).get(runtime) else "estimate"
    p = params / 1e9
    rate = {"speech-to-text": None if not p else _round(1 / (2 * 0.4 * p * 50 / gflops +
                                                             3 * 0.6 * p * 2 / _bandwidth(device))),
            "image classification": _round(gflops / (2 * p * 197)) if p else None,  # ViT-style: 197 patches
            "embeddings": _round(gflops / (2 * p * EMBED_TOKENS)) if p else None,
            "object detection": None}[task]  # detectors differ too much by architecture to guess
    dl = round((download_bytes or params * 4) / GB, 2)
    c = FitCard(model=name, task=task, device=device.name, fits=mem <= budget, memory_gb=mem,
                budget_gb=round(budget, 1), speed=rate, speed_unit=unit if rate else None,
                speed_source=source if rate else None, usable=None if rate is None else rate >= usable,
                download_gb=dl, data_cost=_data_cost(dl), license=license,
                notes=[f"Not a model family NanoMesh knows: sized from its {p * 1000:.0f}M parameters"
                       + ("." if rate else "; detectors differ too much by architecture to guess a speed: export it to "
                                           "ONNX and run nanomesh benchmark on it.")])
    return c


def fit(name: str, device: DeviceProfile, *, download_bytes: int | None = None, license: str | None = None,
        task: str | None = None, params: int | None = None, tags: list[str] | None = None) -> FitCard | None:
    """The Fit Card for a model name, repository id or file. tags: what the
    source says about its format (Hugging Face tags and library)."""
    c = _fit(name, device, download_bytes=download_bytes, license=license, task=task, params=params)
    if c and (why := cannot_run(name, tags, device)):
        c.fits = False
        c.notes = [n for n in c.notes if "best fit" not in n]
        c.notes.insert(0, f"Can't run on {device.name}: {why}. Look for its GGUF, ONNX or original version.")
    return c


def _fit(name: str, device: DeviceProfile, *, download_bytes: int | None = None, license: str | None = None,
         task: str | None = None, params: int | None = None) -> FitCard | None:
    path = Path(name).expanduser()
    if path.is_file() and path.suffix == ".gguf":
        return text_card(str(path), device, download_bytes=path.stat().st_size, license=license)
    matched = match(path.name if path.is_file() else name)
    if matched:
        fam, member = matched
        chosen = member or best(fam, device)
        size = path.stat().st_size if path.is_file() else download_bytes
        c = card(fam, chosen, device, name=name if member else f"{name} → {display(fam, chosen)}",
                 download_bytes=size if member else None, license=license)
        if not member:
            c.notes.insert(0, f"'{name}' has several sizes: this is the best fit for {device.name}.")
        return c
    if task in (None, "text generation", "text-generation"):
        return text_card(name, device, download_bytes=download_bytes, license=license)
    if task in GENERIC and params:
        return generic_card(name, task, params, device, download_bytes=download_bytes, license=license)
    return None
