"""Measure GGUF variants on this machine: speed, peak memory and quality."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from nanomesh.conditions import Sampler
from nanomesh.hardware import GB, DeviceProfile
from nanomesh.model import analyze, gguf_size, read_gguf
from nanomesh.planner import FORMATS, memory_budgets
from nanomesh.results import Result, gguf_format, now
from nanomesh.stress import WARMUP_SECONDS, rewarm_seconds, warm_up
from nanomesh.toolchain import Toolchain, benchmark_gguf, measure_perplexity

# Best reference first: the closer to the original weights, the better.
REFERENCE_ORDER = ["F32", "BF16", "F16"] + [f.name for f in FORMATS if f.name != "F16"]


def pick_reference(files: list[Path], device: DeviceProfile) -> Path | None:
    """The highest-precision file that fits in memory (llama.cpp mmaps weights,
    so a reference larger than RAM would thrash rather than fail)."""
    budget = max(b.memory_gb for b in memory_budgets(device)) * GB
    ranked = sorted(files, key=lambda f: REFERENCE_ORDER.index(fmt) if (fmt := _format(f)) in REFERENCE_ORDER else 99)
    return next((f for f in ranked if gguf_size(f) < budget), None)


def _format(path: Path) -> str | None:
    meta, _ = read_gguf(path)
    return gguf_format(meta, path)


def evaluate(tc: Toolchain, files: list[Path], device: DeviceProfile, *, quality: bool = True,
             reference: Path | None = None, eval_text: Path | None = None, threads: int | None = None,
             warmup_s: float = WARMUP_SECONDS, log: Callable[[str], None] = lambda _: None) -> list[Result]:
    """Benchmark each file. With warmup_s > 0 (the default) each measurement is
    taken in the device's steady state: the first file warms up for at least
    warmup_s, later ones briefly, since the CPU is already warm."""
    results = []
    first_warm = None
    if warmup_s > 0:
        # Smallest first: the lowest-bit file works the CPU hardest, so its
        # warm-up shows the turbo window most clearly (a memory-bound FP16
        # barely changes speed when turbo ends).
        files = sorted(files, key=gguf_size)
    for i, f in enumerate(files):
        warm = None
        if warmup_s > 0:
            budget = warmup_s if first_warm is None else rewarm_seconds(first_warm, warmup_s)
            warm = warm_up(tc, f, budget, threads,
                           log=lambda t, v, name=f.name, b=budget: log(
                               f"Warming up {name} until speed settles… {t:.0f}s (≥{b:.0f}s), {v:.1f} tok/s"))
            first_warm = first_warm or warm
        log(f"Benchmarking {f.name}…")
        with Sampler() as sampler:
            bench = benchmark_gguf(tc, f, threads)
        info = analyze(str(f))
        results.append(Result(
            timestamp=now(), device_key=device.key, device_name=device.name,
            model_name=info.name, model_params=info.params, model_architecture=info.architecture,
            format=_format(f), file_size_gb=bench.size_gb, prompt_tokens_per_s=bench.prompt_tokens_per_s,
            gen_tokens_per_s=bench.gen_tokens_per_s, peak_rss_gb=bench.peak_rss_gb,
            backend=bench.backend, threads=bench.threads, conditions=sampler.summary(),
            steady=warm is not None, warmup_s=warm.seconds if warm else None,
            # Only a cold start shows the turbo speed; later files start warm.
            burst_tokens_per_s=warm.burst_tokens_per_s if warm and i == 0 else None,
        ))

    if quality:
        ref = reference or pick_reference(files, device)
        if ref is None:
            log("No variant fits in memory as a quality reference; skipping quality.")
            return results
        log(f"Measuring reference perplexity ({ref.name})…")
        ref_ppl = measure_perplexity(tc, ref, eval_text, threads)
        ref_fmt = _format(ref)
        for f, r in zip(files, results):
            if f.resolve() == ref.resolve():
                ppl = ref_ppl
            else:
                log(f"Measuring perplexity of {f.name}…")
                ppl = measure_perplexity(tc, f, eval_text, threads)
            r.perplexity = round(ppl, 4)
            r.reference_format = ref_fmt
            r.quality_pct = round(100 * ref_ppl / ppl, 2)
    return results
