import pytest

from nanomesh import results as store
from nanomesh.devices import get_device
from nanomesh.model import analyze
from nanomesh.planner import Evidence, Requirements, plan, tier_for


def _result(**kw):
    base = dict(timestamp="2026-01-01T00:00:00+00:00", device_key="hp-elitebook-840-g3",
                device_name="HP EliteBook 840 G3", model_name="qwen", model_params=7_615_616_512,
                format="Q4_K_M", file_size_gb=4.36, gen_tokens_per_s=4.0)
    return store.Result(**(base | kw))


def test_save_load_roundtrip(isolated_home):
    store.save([_result(), _result(format="Q8_0")])
    assert [r.format for r in store.load()] == ["Q4_K_M", "Q8_0"]
    assert (isolated_home / "results.jsonl").exists()


def test_load_skips_corrupt_lines(isolated_home):
    store.save([_result()])
    with store.results_path().open("a") as f:
        f.write("{not json\n")
    assert len(store.load()) == 1


def test_evidence_matches_device_and_model():
    device, model = get_device("hp-elitebook-840-g3"), analyze("qwen2.5-7b")
    rows = [
        _result(gen_tokens_per_s=3.5, quality_pct=97.1),
        _result(format="Q3_K_M", gen_tokens_per_s=4.2, quality_pct=91.0),
        _result(device_key="lenovo-thinkpad-t480", gen_tokens_per_s=99.0),  # other device
        _result(model_params=1_500_000_000, file_size_gb=1.0, format="Q8_0", gen_tokens_per_s=15.0),  # other model
    ]
    ev = store.evidence(device, model, rows)
    assert ev.speeds == {"Q4_K_M": 3.5, "Q3_K_M": 4.2}
    assert ev.quality == {"Q4_K_M": 97.1, "Q3_K_M": 91.0}
    # Calibration uses every run on this device, including other models.
    assert ev.calibration_runs == 3
    assert ev.effective_bandwidth_gbps == pytest.approx(16.4, abs=0.1)  # median of 16.1, 16.4, 19.7


def test_measurements_override_estimates_in_plan():
    device, model = get_device("hp-elitebook-840-g3"), analyze("qwen2.5-7b")
    ev = Evidence(speeds={"Q4_K_M": 3.1}, quality={"Q4_K_M": 94.0}, effective_bandwidth_gbps=12.0)
    p = plan(model, device, Requirements(min_quality_pct=90), ev)
    q4 = next(v for v in p.variants if v.format.name == "Q4_K_M")
    q3 = next(v for v in p.variants if v.format.name == "Q3_K_M")
    assert (q4.tokens_per_s, q4.speed_source) == (3.1, "measured")
    assert (q4.quality_pct, q4.quality, q4.quality_measured) == (94.0, "fair", True)
    assert q3.speed_source == "calibrated"
    # 12 GB/s over 3.72 GB of weights + half of a 0.23 GB KV cache
    assert q3.tokens_per_s == pytest.approx(3.1, abs=0.05)


def test_measured_quality_can_disqualify_a_variant():
    device, model = get_device("hp-elitebook-840-g3"), analyze("qwen2.5-7b")
    p = plan(model, device, Requirements(min_quality_pct=95), Evidence(quality={"Q4_K_M": 92.0}))
    assert p.recommended == "Q5_K_M"
    q4 = next(v for v in p.variants if v.format.name == "Q4_K_M")
    assert "measured quality 92% below required 95%" in q4.reasons


@pytest.mark.parametrize("pct, tier", [(100, "lossless"), (99.9, "lossless"), (99.0, "high"),
                                       (97.5, "good"), (93, "fair"), (50, "severe")])
def test_tier_for(pct, tier):
    assert tier_for(pct) == tier


def test_gguf_format_detection(tmp_path):
    from pathlib import Path

    assert store.gguf_format({"general.file_type": 15}, Path("x.gguf")) == "Q4_K_M"
    assert store.gguf_format({}, Path("model-q8_0.gguf")) == "Q8_0"
    assert store.gguf_format({}, Path("model.gguf")) is None
