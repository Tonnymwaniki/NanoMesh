"""Measurements beyond a quick benchmark: sustained load and thread tuning."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from statistics import median

import psutil

from nanomesh.conditions import Sampler, read_conditions, summarize
from nanomesh.hardware import DeviceProfile
from nanomesh.model import analyze, gguf_size, read_gguf
from nanomesh.results import Result, SustainedPoint, SustainedRun, gguf_format, now
from nanomesh.toolchain import Toolchain, bench_rows

TOKENS_PER_ROUND = 64
MIN_POINTS_FOR_PATTERN = 6
MIN_SECONDS_FOR_PATTERN = 60


def _identity(f: Path, device: DeviceProfile) -> dict:
    info = analyze(str(f))
    meta, _ = read_gguf(f)
    return dict(timestamp=now(), device_key=device.key, device_name=device.name, model_name=info.name,
                model_params=info.params, model_architecture=info.architecture, format=gguf_format(meta, f),
                file_size_gb=round(gguf_size(f) / 1024**3, 4))


def _drop_shape(points: list[SustainedPoint], burst: float, sustained: float) -> tuple[float | None, str | None]:
    """When and how speed fell. A sudden step (Intel laptops: the short-term
    turbo power budget runs out after ~30-60 s) reads differently from a
    gradual slide as the chassis heats up."""
    if not burst or sustained >= burst * 0.9:
        return None, None
    # Too short to tell a pattern from noise.
    if len(points) < MIN_POINTS_FOR_PATTERN or points[-1].t_s < MIN_SECONDS_FOR_PATTERN:
        return None, None
    threshold = burst - 0.5 * (burst - sustained)  # halfway down
    i = next((i for i, p in enumerate(points) if p.tokens_per_s <= threshold), None)
    if i is None or i == 0:
        return None, None
    biggest_step = max(points[j - 1].tokens_per_s - points[j].tokens_per_s for j in range(1, len(points)))
    pattern = "step" if biggest_step >= 0.6 * (burst - sustained) else "gradual"
    return points[i].t_s, pattern


def summarize_sustained(points: list[SustainedPoint], full_wh: float | None) -> SustainedRun:
    speeds = [p.tokens_per_s for p in points]
    burst = round(sum(speeds[:2]) / len(speeds[:2]), 2)
    tail = speeds[-max(1, len(speeds) // 3):]
    sustained = round(median(tail), 2)
    drop = round(max(0.0, 100 * (1 - sustained / burst)), 1) if burst else 0.0
    run = SustainedRun(points=points, burst_tokens_per_s=burst, sustained_tokens_per_s=sustained, drop_pct=drop)
    run.drop_at_s, run.drop_pattern = _drop_shape(points, burst, sustained)

    watts = [p.discharge_w for p in points if p.discharge_w]
    batt = [p.battery_pct for p in points if p.battery_pct is not None]
    hours = (points[-1].t_s - points[0].t_s) / 3600 if len(points) > 1 else 0
    if watts:
        # Direct battery power draw: the most precise energy figure available.
        run.watts = round(sum(watts) / len(watts), 2)
        run.joules_per_token = round(run.watts / sustained, 2) if sustained else None
        if full_wh:
            run.battery_hours = round(full_wh / run.watts, 1)
            if run.joules_per_token:
                run.tokens_per_battery_pct = round(full_wh * 3600 / 100 / run.joules_per_token)
    elif len(batt) > 1 and hours > 0 and batt[0] - batt[-1] >= 1:
        # Fall back to the battery percentage falling (1% steps: needs a longer run).
        pct_per_hour = (batt[0] - batt[-1]) / hours
        run.battery_hours = round(100 / pct_per_hour, 1)
        run.tokens_per_battery_pct = round(sustained * 3600 / pct_per_hour)
    return run


def sustained(tc: Toolchain, f: Path, device: DeviceProfile, minutes: float = 3.0, threads: int | None = None,
              log: Callable[[SustainedPoint], None] = lambda _: None) -> Result:
    start = read_conditions()
    args = ["-p", "0", "-n", str(TOKENS_PER_ROUND), "-r", "1"] + (["-t", str(threads)] if threads else [])
    points, readings, t0 = [], [], time.monotonic()
    while True:
        # Sample *while* generating: right after a round the CPU is already idle
        # and clocks down, which made a busy CPU look throttled.
        with Sampler(interval_s=1.5, final_reading=False) as sampler:
            rows = bench_rows(tc, f, args)
        tok_s = next((r["avg_ts"] for r in rows if r.get("n_gen")), None)
        during = sampler.readings or [read_conditions(slow_parts=False)]
        readings += during
        clocks = [r.clock_pct for r in during if r.clock_pct is not None]
        temps = [r.temp_c for r in during if r.temp_c is not None]
        # The battery power reading shells out on Windows; only worth it on battery.
        end = read_conditions(slow_parts=bool(start.on_battery))
        point = SustainedPoint(t_s=round(time.monotonic() - t0, 1), tokens_per_s=round(tok_s or 0, 2),
                               temp_c=max(temps) if temps else end.temp_c,
                               clock_pct=round(median(clocks), 1) if clocks else None,
                               battery_pct=end.battery_pct, discharge_w=end.discharge_w)
        points.append(point)
        log(point)
        if point.t_s >= minutes * 60:
            break
    run = summarize_sustained(points, start.battery_full_wh)
    return Result(**_identity(f, device), kind="sustained", gen_tokens_per_s=run.burst_tokens_per_s,
                  threads=threads, conditions=summarize(start, readings, points[-1].t_s), sustained=run)


def thread_counts() -> list[int]:
    physical = psutil.cpu_count(logical=False) or psutil.cpu_count() or 4
    logical = psutil.cpu_count() or physical
    counts = {2, max(1, physical // 2), physical, (physical + logical) // 2, logical}
    return sorted(c for c in counts if 1 <= c <= logical)


def tune_threads(tc: Toolchain, f: Path, device: DeviceProfile, counts: list[int] | None = None) -> list[Result]:
    """Generation speed at each thread count (llama-bench runs them in one go)."""
    counts = counts or thread_counts()
    start = read_conditions()
    rows = bench_rows(tc, f, ["-p", "0", "-n", str(TOKENS_PER_ROUND), "-r", "2", "-t", ",".join(map(str, counts))])
    identity = _identity(f, device)
    conditions = summarize(start, [], 0)
    return [Result(**identity, kind="threads", threads=r.get("n_threads"), gen_tokens_per_s=round(r["avg_ts"], 2),
                   backend=r.get("backends"), conditions=conditions)
            for r in rows if r.get("n_gen")]
