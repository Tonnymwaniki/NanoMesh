"""Benchmarks for non-text models: ONNX Runtime (vision, embeddings) and
whisper.cpp (speech). Each measures throughput in the task's unit on this
device; saved results calibrate the Fit Card of the whole family."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil
from pydantic import BaseModel

from nanomesh.fit import EMBED_TOKENS, display, match, unit_gflops
from nanomesh.hardware import GB
from nanomesh.toolchain import ToolchainError, _find_binary


class RunResult(BaseModel):
    model_name: str
    task: str | None
    throughput: float
    unit: str
    gflops_per_unit: float | None
    params: int
    peak_rss_gb: float | None = None
    threads: int | None = None
    runtime: str  # onnx | whisper.cpp
    note: str | None = None


# ---- ONNX Runtime ----

ONNX_TYPES = {"tensor(float)": "float32", "tensor(float16)": "float16", "tensor(double)": "float64",
              "tensor(int64)": "int64", "tensor(int32)": "int32", "tensor(bool)": "bool", "tensor(uint8)": "uint8"}


def _inputs(session, image_size: int):
    import numpy as np

    feeds = {}
    for inp in session.get_inputs():
        shape = list(inp.shape)
        dims = []
        for i, d in enumerate(shape):
            if isinstance(d, int) and d > 0:
                dims.append(d)
            elif i == 0:
                dims.append(1)  # batch
            elif len(shape) == 4 and i == 1:
                dims.append(3)  # channels
            elif len(shape) == 4:
                dims.append(image_size)
            else:
                dims.append(EMBED_TOKENS)  # sequence length
        dtype = ONNX_TYPES.get(inp.type, "float32")
        if dtype.startswith("float"):
            feeds[inp.name] = np.random.rand(*dims).astype(dtype)
        else:
            feeds[inp.name] = np.ones(dims, dtype=dtype)  # token ids / attention masks
    return feeds


def onnx_benchmark(path: Path, seconds: float = 5.0, threads: int | None = None) -> RunResult:
    try:
        import onnxruntime as ort
    except ImportError:
        raise ToolchainError("ONNX Runtime isn't installed: pip install onnxruntime") from None
    matched = match(path.name)
    fam, member = matched if matched else (None, None)
    image_size = 640 if fam and fam.task == "object detection" else 224
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    try:
        session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    except Exception as e:  # onnxruntime raises its own exception types
        raise ToolchainError(f"ONNX Runtime couldn't load {path.name}: {e}") from None
    feeds = _inputs(session, image_size)
    proc = psutil.Process()
    for _ in range(2):  # warm-up: first runs allocate and pick kernels
        session.run(None, feeds)
    runs, peak, t0 = 0, proc.memory_info().rss, time.perf_counter()
    while True:
        session.run(None, feeds)
        runs += 1
        peak = max(peak, proc.memory_info().rss)
        elapsed = time.perf_counter() - t0
        if elapsed >= seconds or runs >= 500:
            break
    rate = runs / elapsed
    return RunResult(
        model_name=display(fam, member) if member else path.stem, task=fam.task if fam else None,
        throughput=round(rate, 2), unit=fam.unit if fam else "runs/s",
        gflops_per_unit=unit_gflops(fam, member) if member else None,
        params=int(member.params_m * 1e6) if member else 0, peak_rss_gb=round(peak / GB, 3), threads=threads,
        runtime="onnx", note=None if member else "Unknown model family: measured runs/s only, no calibration.")


# ---- whisper.cpp ----

WHISPER_ENCODE_RE = re.compile(r"encode time\s*=\s*([\d.]+)\s*ms(?:\s*/\s*(\d+)\s*runs\s*\(\s*([\d.]+)\s*ms per run)?")


def find_whisper_bench() -> Path | None:
    from nanomesh import config

    env = [Path(p).expanduser() for p in os.environ.get("NANOMESH_WHISPER_CPP", "").split(os.pathsep) if p]
    saved = [Path(p) for p in config.get("whisper_cpp", []) or []]
    defaults = [Path.home() / "whisper.cpp"] + ([Path("C:/whisper.cpp")] if sys.platform == "win32" else [])
    roots = list(dict.fromkeys(env + saved + [d for d in defaults if d.is_dir()]))
    found = _find_binary("whisper-bench", roots)
    if not found:  # older builds call it "bench"; only trust that name inside a whisper.cpp folder
        found = next((b for r in roots for b in [_find_binary("bench", [r])] if b and r in b.parents), None)
    if found and env and any(r in found.parents for r in env):
        config.set("whisper_cpp", [str(r) for r in env])
    return found


def parse_whisper_bench(output: str) -> float:
    """Milliseconds to encode one 30 s window."""
    m = WHISPER_ENCODE_RE.search(output)
    if not m:
        raise ToolchainError("Couldn't read whisper-bench's output (no 'encode time').")
    total, runs, per_run = float(m.group(1)), m.group(2), m.group(3)
    return float(per_run) if per_run else total / (int(runs) if runs else 1)


def whisper_benchmark(path: Path, threads: int | None = None) -> RunResult:
    bench = find_whisper_bench()
    if not bench:
        raise ToolchainError("whisper-bench not found. Build or download whisper.cpp and set NANOMESH_WHISPER_CPP to "
                             "its folder (C:\\whisper.cpp and ~/whisper.cpp are checked automatically).")
    matched = match(path.name)
    if not matched or not matched[1]:
        raise ToolchainError(f"Can't tell which Whisper model {path.name} is; name it like ggml-small.bin.")
    fam, member = matched
    cmd = [str(bench), "-m", str(path)] + (["-t", str(threads)] if threads else [])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.SubprocessError) as e:
        raise ToolchainError(f"whisper-bench failed to run: {e}") from None
    if out.returncode != 0:
        tail = (out.stderr or out.stdout).strip().splitlines()[-8:]
        raise ToolchainError("whisper-bench failed:\n" + "\n".join(tail))
    encode_ms = parse_whisper_bench(out.stdout + "\n" + out.stderr)
    return RunResult(
        model_name=display(fam, member), task=fam.task, throughput=round(30_000 / encode_ms, 2),
        unit="x real-time (encoder)", gflops_per_unit=unit_gflops(fam, member), params=int(member.params_m * 1e6),
        threads=threads, runtime="whisper.cpp",
        note="whisper-bench times the encoder on a 30 s window; the Fit Card adds the decoder from memory speed.")


def is_whisper_file(path: Path) -> bool:
    m = match(path.name)
    return path.suffix == ".bin" and bool(m) and m[0].series == "whisper"


def save_run(run: RunResult, path: Path, device, conditions=None):
    """Record a run so Fit Cards on this device use it."""
    from nanomesh import results as store

    r = store.Result(timestamp=store.now(), device_key=device.key, device_name=device.name, model_name=run.model_name,
                     model_params=run.params, format=path.suffix.lstrip(".") or None,
                     file_size_gb=round(path.stat().st_size / GB, 4), peak_rss_gb=run.peak_rss_gb, threads=run.threads,
                     backend=run.runtime, kind=run.runtime, task=run.task, throughput=run.throughput,
                     throughput_unit=run.unit, gflops_per_unit=run.gflops_per_unit, conditions=conditions)
    store.save([r])
    return r


def benchmark_file(path: Path, device, threads: int | None = None) -> RunResult:
    """Benchmark an .onnx model or a whisper.cpp model and save the result."""
    from nanomesh.conditions import Sampler

    with Sampler() as sampler:
        run = whisper_benchmark(path, threads) if is_whisper_file(path) else onnx_benchmark(path, threads=threads)
    save_run(run, path, device, sampler.summary())
    return run


def can_benchmark(path: Path) -> bool:
    return path.is_file() and (path.suffix == ".onnx" or is_whisper_file(path))
