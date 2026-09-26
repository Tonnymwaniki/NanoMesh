"""The optimizer: given a model, a device and requirements, find the best variant.

Every number here is an *estimate* derived from first principles (weights,
KV cache, memory bandwidth). `nanomesh benchmark` replaces estimates with
measurements on real hardware.
"""

from __future__ import annotations

from pydantic import BaseModel

from nanomesh.hardware import GB, DeviceProfile
from nanomesh.model import ModelInfo

# Quality tiers, best first. Higher rank == better quality.
QUALITY_TIERS = ["severe", "fair", "good", "high", "lossless"]


class QuantFormat(BaseModel):
    name: str
    label: str  # human-friendly bit width
    bits_per_weight: float
    quality: str
    note: str


# llama.cpp GGUF formats. Bits-per-weight are effective averages including
# block scales; quality tiers follow llama.cpp's published perplexity deltas.
FORMATS = [
    QuantFormat(name="F16", label="FP16", bits_per_weight=16.0, quality="lossless", note="Reference quality"),
    QuantFormat(name="Q8_0", label="INT8", bits_per_weight=8.5, quality="lossless", note="Practically indistinguishable from FP16"),
    QuantFormat(name="Q6_K", label="INT6", bits_per_weight=6.56, quality="high", note="Negligible quality loss"),
    QuantFormat(name="Q5_K_M", label="INT5", bits_per_weight=5.69, quality="high", note="Very small quality loss"),
    QuantFormat(name="Q4_K_M", label="INT4", bits_per_weight=4.89, quality="good", note="Best size/quality balance for most devices"),
    QuantFormat(name="Q3_K_M", label="INT3", bits_per_weight=3.91, quality="fair", note="Noticeable loss; use when INT4 won't fit"),
    QuantFormat(name="Q2_K", label="INT2", bits_per_weight=3.0, quality="severe", note="Large loss; last resort, better to pick a smaller model"),
]
FORMATS_BY_NAME = {f.name.lower(): f for f in FORMATS} | {f.label.lower(): f for f in FORMATS}

RUNTIME_OVERHEAD_BYTES = int(0.3 * GB)
# Leave room for the OS/app to breathe: prefer variants using <= 90% of the budget.
HEADROOM = 0.90  # compute buffers, tokenizer, runtime itself
CPU_BANDWIDTH_EFFICIENCY = 0.55
GPU_BANDWIDTH_EFFICIENCY = 0.70


class Requirements(BaseModel):
    context: int = 4096
    min_tokens_per_s: float | None = None
    min_quality: str = "good"
    max_ram_gb: float | None = None
    prefer: str = "quality"  # quality | speed | size


class Budget(BaseModel):
    placement: str  # "GPU" or "CPU"
    memory_gb: float
    bandwidth_gbps: float | None
    efficiency: float


class Variant(BaseModel):
    format: QuantFormat
    weights_gb: float
    kv_cache_gb: float
    total_memory_gb: float
    compression: float  # vs FP16
    placement: str | None  # where it runs; None if it doesn't fit anywhere
    tokens_per_s: float | None
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
    elif device.kind == "local" and device.available_ram_gb:
        cpu_mem = device.available_ram_gb * 0.85
    else:
        cpu_mem = device.ram_gb * 0.70
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


def _decode_speed(weights_gb: float, kv_gb: float, budget: Budget) -> float | None:
    # Token generation is memory-bandwidth bound: every token streams all weights
    # (and, at worst, the full KV cache) through the memory bus once.
    if not budget.bandwidth_gbps:
        return None
    bytes_per_token_gb = weights_gb + kv_gb * 0.5  # average KV fill over a session
    return round(budget.bandwidth_gbps * budget.efficiency / bytes_per_token_gb * (GB / 1e9), 1)


def max_practical_params(budget_gb: float, context: int, fmt: QuantFormat = FORMATS[4]) -> float:
    """Largest model (in billions of params) that fits the budget at a given format."""
    usable = budget_gb * GB - RUNTIME_OVERHEAD_BYTES
    per_b = 1e9 * fmt.bits_per_weight / 8 + 16_384 * context  # weights + generic KV per 1B params
    return max(0.0, round(usable / per_b, 1))


def plan(model: ModelInfo, device: DeviceProfile, req: Requirements | None = None) -> Plan:
    req = req or Requirements()
    budgets = memory_budgets(device, req.max_ram_gb)
    min_rank = QUALITY_TIERS.index(req.min_quality)
    fp16_gb = model.params * 2 / GB

    variants = []
    for fmt in FORMATS:
        weights, kv, total = estimate(model, fmt, req.context)
        budget = next((b for b in budgets if total <= b.memory_gb), None)
        speed = _decode_speed(weights, kv, budget) if budget else None
        reasons = []
        if not budget:
            reasons.append(f"needs ~{total:.1f} GB, budget is {budgets[-1].memory_gb:.1f} GB")
        if QUALITY_TIERS.index(fmt.quality) < min_rank:
            reasons.append(f"quality '{fmt.quality}' below required '{req.min_quality}'")
        if req.min_tokens_per_s and speed is not None and speed < req.min_tokens_per_s:
            reasons.append(f"~{speed} tok/s below required {req.min_tokens_per_s}")
        variants.append(Variant(
            format=fmt, weights_gb=round(weights, 2), kv_cache_gb=round(kv, 2),
            total_memory_gb=round(total, 2), compression=round(1 - weights / fp16_gb, 3),
            placement=budget.placement if budget else None, tokens_per_s=speed,
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
        return max(ok, key=lambda v: (v.tokens_per_s or 0, -v.total_memory_gb))
    if prefer == "size":
        return min(ok, key=lambda v: v.total_memory_gb)
    # Highest quality tier first; within a tier the smaller variant wins
    # (so INT8 beats FP16: same quality, half the memory). Variants that only
    # just squeeze in are used only when nothing fits comfortably.
    ok = [v for v in ok if v.comfortable] or ok
    return max(ok, key=lambda v: (QUALITY_TIERS.index(v.format.quality), -v.total_memory_gb))


def _mark_pareto(variants: list[Variant]) -> None:
    """A variant is Pareto-optimal if no other fitting variant is at least as
    good on quality and strictly smaller in memory."""
    fitting = [v for v in variants if v.fits]
    for v in fitting:
        q = QUALITY_TIERS.index(v.format.quality)
        v.pareto = not any(
            QUALITY_TIERS.index(o.format.quality) >= q and o.total_memory_gb < v.total_memory_gb
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
    better_fits = any(v.fits and QUALITY_TIERS.index(v.format.quality) > QUALITY_TIERS.index(best.format.quality)
                      for v in variants)
    if best.format.quality in ("fair", "severe") and not better_fits:
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
    return advice
