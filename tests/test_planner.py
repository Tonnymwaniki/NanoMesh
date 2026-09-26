from nanomesh.devices import get_device, search_devices
from nanomesh.model import analyze
from nanomesh.planner import Requirements, plan


def _variant(p, name):
    return next(v for v in p.variants if v.format.name == name)


def test_8gb_laptop_gets_int4_for_7b():
    p = plan(analyze("qwen2.5-7b"), get_device("hp-elitebook-840-g3"))
    assert p.recommended == "Q4_K_M"
    assert not _variant(p, "F16").fits
    assert _variant(p, "Q4_K_M").tokens_per_s is not None


def test_gpu_prefers_int8_when_it_fits():
    p = plan(analyze("llama-3.1-8b"), get_device("rtx-3060-12gb"))
    assert p.recommended == "Q8_0"
    assert _variant(p, "Q8_0").placement == "GPU"


def test_speed_requirement_pushes_to_smaller_variant():
    device = get_device("rtx-3060-12gb")
    p = plan(analyze("llama-3.1-8b"), device, Requirements(min_tokens_per_s=50))
    assert p.recommended == "Q4_K_M"


def test_nothing_fits_suggests_smaller_model():
    p = plan(analyze("70b"), get_device("low-end-android-4gb"))
    assert p.recommended is None
    assert p.max_practical_params_b < 3
    assert "smaller model" in p.advice[0]


def test_ram_cap_is_respected():
    p = plan(analyze("qwen2.5-3b"), get_device("rtx-4090-24gb"), Requirements(max_ram_gb=2.5))
    rec = _variant(p, p.recommended)
    assert rec.total_memory_gb <= 2.5


def test_pareto_excludes_dominated_variants():
    p = plan(analyze("qwen2.5-1.5b"), get_device("h100-sxm-80gb"))
    # FP16 and Q8_0 share a quality tier; Q8_0 is smaller, so FP16 is dominated.
    assert not _variant(p, "F16").pareto
    assert _variant(p, "Q8_0").pareto


def test_device_search():
    assert get_device("thinkpad").id == "lenovo-thinkpad-t480"
    assert {d.id for d in search_devices("android")} >= {"redmi-14c-4gb", "low-end-android-4gb"}


def test_tiny_model_prefers_int8_over_fp16_despite_rounding():
    # Both variants round to the same GB figure; INT8 must still win.
    p = plan(analyze("500m"), get_device("h100-sxm-80gb"))
    assert p.recommended == "Q8_0"
    p = plan(analyze("1m"), get_device("raspberry-pi-5-8gb"))
    assert p.recommended == "Q8_0"
