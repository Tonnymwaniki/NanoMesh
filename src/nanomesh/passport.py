"""Device Passport: what a device can realistically run. Shared by the CLI and dashboard."""

from __future__ import annotations

from pydantic import BaseModel

from nanomesh.hardware import DeviceProfile, compute_class
from nanomesh.model import analyze
from nanomesh.planner import Requirements, max_practical_params, memory_budgets, plan
from nanomesh.results import evidence

REFERENCE_SIZES = ["1b", "3b", "7b", "13b", "32b", "70b"]
PASSPORT_CONTEXT = 4096


class Fit(BaseModel):
    size: str
    fits: bool
    label: str | None = None  # INT4, INT8, ...
    format: str | None = None
    quality: str | None = None
    memory_gb: float | None = None
    tokens_per_s: float | None = None
    speed_source: str | None = None
    placement: str | None = None


class Passport(BaseModel):
    device: DeviceProfile
    compute_class: str
    budget_gb: float
    recommended_max_b: float  # fits comfortably at INT4
    possible_max_b: float  # fits, tightly, at INT4
    fits: list[Fit]
    low_free_ram: bool
    unrecognised: bool


def passport(device: DeviceProfile) -> Passport:
    budget = max(b.memory_gb for b in memory_budgets(device))
    fits = []
    for size in REFERENCE_SIZES:
        model = analyze(size)
        # Device-level calibration applies; per-model measurements don't (these are generic sizes).
        ev = evidence(device, model).model_copy(update={"speeds": {}, "quality": {}})
        p = plan(model, device, Requirements(min_quality="fair"), ev)
        v = next((x for x in p.variants if x.format.name == p.recommended), None)
        if v:
            fits.append(Fit(size=size.upper(), fits=True, label=v.format.label, format=v.format.name,
                            quality=v.quality, memory_gb=v.total_memory_gb, tokens_per_s=v.tokens_per_s,
                            speed_source=v.speed_source, placement=v.placement))
        else:
            fits.append(Fit(size=size.upper(), fits=False))
    return Passport(
        device=device, compute_class=compute_class(device), budget_gb=round(budget, 1),
        recommended_max_b=max_practical_params(budget * 0.75, PASSPORT_CONTEXT),
        possible_max_b=max_practical_params(budget, PASSPORT_CONTEXT), fits=fits,
        low_free_ram=bool(device.is_local and device.available_ram_gb is not None
                          and device.available_ram_gb < budget * 0.5),
        unrecognised=bool(device.is_local and not device.matched_id),
    )
