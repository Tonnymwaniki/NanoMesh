"""The optimizer: given a model, a device and requirements, find the best variant.

Numbers start as *estimates* derived from first principles (weights, KV cache,
memory bandwidth, typical quantization loss). Pass `Evidence` from
`nanomesh benchmark` runs and measurements replace estimates: exact speeds and
quality for variants that were measured, and a calibrated memory bandwidth
for everything else on that device.
"""

from __future__ import annotations

from pydantic import BaseModel

from nanomesh.hardware import GB, DeviceProfile
from nanomesh.model import ModelInfo

# Quality tiers, worst first, with the minimum quality % (reference perplexity /
# variant perplexity) each requires.
QUALITY_TIERS = ["severe", "fair", "good", "high", "lossless"]
TIER_FLOORS = {"lossless": 99.8, "high": 98.5, "good": 96.0, "fair": 90.0, "severe": 0.0}


def tier_for(quality_pct: float) -> str:
    return next(t for t in reversed(QUALITY_TIERS) if quality_pct >= TIER_FLOORS[t])


class QuantFormat(BaseModel):
    name: str
    label: str  # human-friendly bit width
    bits_per_weight: float
    typical_quality_pct: float
    note: str

    @property
    def quality(self) -> str:
        return tier_for(self.typical_quality_pct)


# llama.cpp GGUF formats. Bits-per-weight are effective averages including
# block scales. Typical quality is a conservative reading of llama.cpp's
# published perplexity deltas; newer, denser models (e.g. Llama 3) lose more
# than older ones, so measure with `nanomesh benchmark` before trusting it.
FORMATS = [
    QuantFormat(name="F16", label="FP16", bits_per_weight=16.0, typical_quality_pct=100.0, note="Reference quality"),
    QuantFormat(name="Q8_0", label="INT8", bits_per_weight=8.5, typical_quality_pct=99.9, note="Practically indistinguishable from FP16"),
    QuantFormat(name="Q6_K", label="INT6", bits_per_weight=6.56, typical_quality_pct=99.5, note="Negligible quality loss"),
    QuantFormat(name="Q5_K_M", label="INT5", bits_per_weight=5.69, typical_quality_pct=99.0, note="Very small quality loss"),
    QuantFormat(name="Q4_K_M", label="INT4", bits_per_weight=4.89, typical_quality_pct=97.5, note="Best size/quality balance for most devices"),
    QuantFormat(name="Q3_K_M", label="INT3", bits_per_weight=3.91, typical_quality_pct=93.0, note="Noticeable loss; use when INT4 won't fit"),
    QuantFormat(name="Q2_K", label="INT2", bits_per_weight=3.0, typical_quality_pct=82.0, note="Large loss; last resort, better to pick a smaller model"),
]
FORMATS_BY_NAME = {f.name.lower(): f for f in FORMATS} | {f.label.lower(): f for f in FORMATS}

RUNTIME_OVERHEAD_BYTES = int(0.3 * GB)  # compute buffers, tokenizer, runtime itself
# Leave room for the OS/app to breathe: prefer variants using <= 90% of the budget.
HEADROOM = 0.90
# Fraction of peak bandwidth llama.cpp typically achieves; `benchmark` calibrates per device.
CPU_BANDWIDTH_EFFICIENCY = 0.55
GPU_BANDWIDTH_EFFICIENCY = 0.80


class Requirements(BaseModel):
    context: int = 4096
    min_tokens_per_s: float | None = None
    min_quality: str = "good"  # a tier name...
    min_quality_pct: float | None = None  # ...or an explicit percentage, which wins
    max_ram_gb: float | None = None
    prefer: str = "quality"  # quality | speed | size


class Budget(BaseModel):
    placement: str  # "GPU" or "CPU"
    memory_gb: float
    bandwidth_gbps: float | None
    efficiency: float


class Evidence(BaseModel):
    """Measurements for this model on this device (see nanomesh.results)."""

    speeds: dict[str, float] = {}  # format -> measured generation tok/s
    quality: dict[str, float] = {}  # format -> measured quality %
    # Bandwidth implied by past benchmarks of *any* model on this device.
    effective_bandwidth_gbps: float | None = None
    calibration_runs: int = 0


class Variant(BaseModel):
    format: QuantFormat
    weights_gb: float
    kv_cache_gb: float
    total_memory_gb: float
    compression: float  # vs FP16
    placement: str | None  # where it runs; None if it doesn't fit anywhere
    tokens_per_s: float | None
    speed_source: str | None = None  # measured | calibrated | estimate
    quality_pct: float
    quality: str
    quality_measured: bool = False
    fits: bool
    comfortable: bool = False  # fits with headroom to spare
    meets_requirements: bool
    reasons: list[str] = []
    pareto: bool = False


class Plan(BaseModel):
    model: ModelInfo
    device: DeviceProfile
    requirements: Requirements
    budgets: list[Budget]
    variants: list[Variant]
    recommended: str | None
    max_practical_params_b: float
    advice: list[str]


def memory_budgets(device: DeviceProfile, max_ram_gb: float | None = None) -> list[Budget]:
    """How much memory a model may use on this device, per placement (best first)."""
    budgets = []
    gpu = device.best_gpu
    if gpu and gpu.vram_gb:
        budgets.append(Budget(placement="GPU", memory_gb=gpu.vram_gb * 0.92,
                              bandwidth_gbps=gpu.bandwidth_gbps, efficiency=GPU_BANDWIDTH_EFFICIENCY))

    unified = next((g for g in device.gpus if g.unified_memory and g.vram_gb and g.bandwidth_gbps), None)
    if device.kind == "phone":
        # Android kills apps well before RAM is exhausted.
        cpu_mem = device.ram_gb * 0.45
    else:
        cpu_mem = device.ram_gb * 0.70
    if device.is_local and device.available_ram_gb:
        # A live scan knows what's actually free right now.
        free = device.available_ram_gb * 0.85
        cpu_mem = min(cpu_mem, free) if device.kind == "phone" else free
    if max_ram_gb is not None:
        cpu_mem = min(cpu_mem, max_ram_gb)
        budgets = [b.model_copy(update={"memory_gb": min(b.memory_gb, max_ram_gb)}) for b in budgets]

    if unified:
        budgets.append(Budget(placement="GPU (unified)", memory_gb=cpu_mem,
                              bandwidth_gbps=unified.bandwidth_gbps, efficiency=GPU_BANDWIDTH_EFFICIENCY))
    else:
        budgets.append(Budget(placement="CPU", memory_gb=cpu_mem,
                              bandwidth_gbps=device.memory_bandwidth_gbps, efficiency=CPU_BANDWIDTH_EFFICIENCY))
    return budgets


def estimate(model: ModelInfo, fmt: QuantFormat, context: int) -> tuple[float, float, float]:
    """Return (weights, kv cache, total) memory in GB."""
    weights = model.params * fmt.bits_per_weight / 8
    kv = model.kv_bytes_per_token() * context
    total = weights + kv + RUNTIME_OVERHEAD_BYTES
    return weights / GB, kv / GB, total / GB


def _decode_speed(weights_gb: float, kv_gb: float, effective_gbps: float) -> float:
    # Token generation is memory-bandwidth bound: every token streams all weights
    # (and, at worst, the full KV cache) through the memory bus once.
    # Sizes here are GiB (1024**3 bytes); bandwidth is GB/s (1e9 bytes).
    bytes_per_token = (weights_gb + kv_gb * 0.5) * GB  # average KV fill over a session
    return round(effective_gbps * 1e9 / bytes_per_token, 1)


def _speed(fmt: QuantFormat, weights: float, kv: float, budget: Budget, primary: bool,
           ev: Evidence) -> tuple[float | None, str | None]:
    # Benchmarks run on the device's preferred placement, so measurements and
    # calibration only apply to variants that land there too.
    if primary and fmt.name in ev.speeds:
        return ev.speeds[fmt.name], "measured"
    if primary and ev.effective_bandwidth_gbps:
        return _decode_speed(weights, kv, ev.effective_bandwidth_gbps), "calibrated"
    if budget.bandwidth_gbps:
        return _decode_speed(weights, kv, budget.bandwidth_gbps * budget.efficiency), "estimate"
    return None, None


def max_practical_params(budget_gb: float, context: int, fmt: QuantFormat = FORMATS[4]) -> float:
    """Largest model (in billions of params) that fits the budget at a given format."""
    usable = budget_gb * GB - RUNTIME_OVERHEAD_BYTES
    per_b = 1e9 * fmt.bits_per_weight / 8 + 16_384 * context  # weights + generic KV per 1B params
    return max(0.0, round(usable / per_b, 1))


def plan(model: ModelInfo, device: DeviceProfile, req: Requirements | None = None,
         evidence: Evidence | None = None) -> Plan:
    req = req or Requirements()
    ev = evidence or Evidence()
    budgets = memory_budgets(device, req.max_ram_gb)
    min_pct = req.min_quality_pct if req.min_quality_pct is not None else TIER_FLOORS[req.min_quality]
    fp16_gb = model.params * 2 / GB

    variants = []
    for fmt in FORMATS:
        weights, kv, total = estimate(model, fmt, req.context)
        budget = next((b for b in budgets if total <= b.memory_gb), None)
        speed, source = _speed(fmt, weights, kv, budget, budget is budgets[0], ev) if budget else (None, None)
        measured_q = fmt.name in ev.quality
        quality_pct = min(ev.quality[fmt.name], 100.0) if measured_q else fmt.typical_quality_pct
        reasons = []
        if not budget:
            reasons.append(f"needs ~{total:.1f} GB, budget is {budgets[-1].memory_gb:.1f} GB")
        if quality_pct < min_pct:
            what = "measured" if measured_q else "typical"
            reasons.append(f"{what} quality {quality_pct:g}% below required {min_pct:g}%")
        if req.min_tokens_per_s and speed is not None and speed < req.min_tokens_per_s:
            reasons.append(f"{speed} tok/s below required {req.min_tokens_per_s}")
        variants.append(Variant(
            format=fmt, weights_gb=round(weights, 2), kv_cache_gb=round(kv, 2),
            total_memory_gb=round(total, 2), compression=round(1 - weights / fp16_gb, 3),
            placement=budget.placement if budget else None, tokens_per_s=speed, speed_source=source,
            quality_pct=quality_pct, quality=tier_for(quality_pct), quality_measured=measured_q,
            fits=budget is not None, comfortable=budget is not None and total <= budget.memory_gb * HEADROOM,
            meets_requirements=not reasons, reasons=reasons,
        ))

    _mark_pareto(variants)
    best = _choose([v for v in variants if v.meets_requirements], req.prefer)
    max_b = max_practical_params(max(b.memory_gb for b in budgets), req.context)
    return Plan(model=model, device=device, requirements=req, budgets=budgets, variants=variants,
                recommended=best.format.name if best else None, max_practical_params_b=max_b,
                advice=_advice(model, variants, best, max_b, req))


def _choose(ok: list[Variant], prefer: str) -> Variant | None:
    if not ok:
        return None
    if prefer == "speed":
        return max(ok, key=lambda v: (v.tokens_per_s or 0, -v.format.bits_per_weight))
    if prefer == "size":
        return min(ok, key=lambda v: v.format.bits_per_weight)
    # Highest quality tier first; within a tier the smaller variant wins
    # (so INT8 beats FP16: same quality, half the memory). Variants that only
    # just squeeze in are used only when nothing fits comfortably.
    ok = [v for v in ok if v.comfortable] or ok
    # (Memory grows strictly with bits-per-weight, and unlike rounded GB it never ties.)
    return max(ok, key=lambda v: (QUALITY_TIERS.index(v.quality), -v.format.bits_per_weight))


def _mark_pareto(variants: list[Variant]) -> None:
    """A variant is Pareto-optimal if no other fitting variant is at least as
    good on quality and strictly smaller in memory."""
    fitting = [v for v in variants if v.fits]
    for v in fitting:
        q = QUALITY_TIERS.index(v.quality)
        v.pareto = not any(
            QUALITY_TIERS.index(o.quality) >= q and o.format.bits_per_weight < v.format.bits_per_weight
            for o in fitting if o is not v
        )


def _advice(model: ModelInfo, variants: list[Variant], best: Variant | None,
            max_b: float, req: Requirements) -> list[str]:
    advice = []
    if best is None:
        if not any(v.fits for v in variants):
            advice.append(
                f"No variant of this {model.params_b:.1f}B model fits. The largest practical model "
                f"for this device is ~{max_b}B params at INT4 — consider a smaller model."
            )
        else:
            advice.append("Variants fit in memory but none meet every requirement; "
                          "relax --min-quality / --min-speed or pick a smaller model.")
        if req.context > 2048:
            advice.append(f"Reducing context from {req.context} tokens shrinks the KV cache.")
        return advice
    better_fits = any(v.fits and QUALITY_TIERS.index(v.quality) > QUALITY_TIERS.index(best.quality)
                      for v in variants)
    if best.quality in ("fair", "severe") and not better_fits:
        advice.append(
            f"Only a {best.format.label} variant fits; a smaller model (≤{max_b}B) at INT4 "
            "usually beats a larger model squeezed to 2–3 bits."
        )
    if not best.comfortable:
        advice.append("This variant barely fits; close other apps or reduce --context.")
    if best.tokens_per_s is not None and best.tokens_per_s < 5:
        advice.append("Expected speed is below ~5 tok/s, which feels sluggish for chat.")
    if best.tokens_per_s is None:
        advice.append("No bandwidth data for this device; run `nanomesh benchmark` on it for real speeds.")
    if not best.quality_measured:
        advice.append("Quality is a typical figure for this format; `nanomesh benchmark` measures it.")
    return advice
