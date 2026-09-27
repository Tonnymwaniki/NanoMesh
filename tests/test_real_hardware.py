"""Regression tests from real benchmark runs.

tests/data/elitebook-840-g6-qwen2.5-1.5b.jsonl: HP EliteBook 840 G6
(i5-8365U, 16 GB DDR4-2400, Windows 11), official Qwen2.5-1.5B-Instruct GGUFs,
llama.cpp CPU build, `nanomesh benchmark` with 7.6 GB free.

tests/data/elitebook-840-g6-qwen2.5-7b.jsonl: the same laptop, the official
two-part Qwen2.5-7B-Instruct Q4_K_M GGUF, `--no-quality`, run afterwards to
check the 1.5B calibration's 7B prediction.
"""

from pathlib import Path

import pytest

from nanomesh import results as store
from nanomesh.devices import get_device
from nanomesh.model import analyze
from nanomesh.planner import Requirements, plan

DATA = Path(__file__).parent / "data"


@pytest.fixture
def elitebook_results(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "results.jsonl").write_text((DATA / "elitebook-840-g6-qwen2.5-1.5b.jsonl").read_text())
    monkeypatch.setenv("NANOMESH_HOME", str(home))
    return store.load()


def _v(p, name):
    return next(v for v in p.variants if v.format.name == name)


def test_gguf_with_duplicated_embeddings_still_matches_by_name(elitebook_results):
    # The official GGUF reports 1.78B params; Hugging Face says 1.54B.
    ev = store.evidence(get_device("hp-elitebook-840-g6"), analyze("qwen2.5-1.5b"))
    assert set(ev.speeds) == {"F16", "Q8_0", "Q4_K_M", "Q3_K_M"}
    assert ev.model_params == 1_777_088_000
    assert not store.evidence(get_device("hp-elitebook-840-g6"), analyze("qwen2.5-0.5b")).speeds


def test_roofline_learned_from_the_elitebook(elitebook_results):
    ev = store.evidence(get_device("hp-elitebook-840-g6"), analyze("qwen2.5-7b"))
    assert ev.effective_bandwidth_gbps == pytest.approx(20.1, abs=0.2)  # F16 run; spec sheet says 38.4
    assert ev.compute_gparams_per_s == pytest.approx(27.0, abs=0.2)  # Q3_K_M/Q4_K_M runs


def test_calibrated_speeds_respect_the_compute_ceiling(elitebook_results):
    device, model = get_device("hp-elitebook-840-g6"), analyze("qwen2.5-1.5b")
    p = plan(model, device, evidence=store.evidence(device, model))
    q5, q4 = _v(p, "Q5_K_M"), _v(p, "Q4_K_M")
    assert q5.speed_source == "calibrated" and q4.speed_source == "measured"
    # INT3 and INT4 measured the same speed, so INT5 can't be predicted faster.
    assert q5.tokens_per_s <= q4.tokens_per_s * 1.05
    assert any("no faster than the next step up" in a for a in p.advice)


def test_small_model_quality_loss_is_calibrated(elitebook_results):
    device, model = get_device("hp-elitebook-840-g6"), analyze("qwen2.5-1.5b")
    ev = store.evidence(device, model)
    # INT4 lost 7.46% (typical 2.5%), INT3 21.5% (typical 7%): ~3x.
    assert ev.quality_loss_scale == pytest.approx(3.03, abs=0.05)
    q5 = _v(plan(model, device, evidence=ev), "Q5_K_M")
    assert (q5.quality_source, q5.quality_pct) == ("calibrated", 97.0)


def test_recommends_int8_for_small_model_on_this_laptop(elitebook_results):
    device, model = get_device("hp-elitebook-840-g6"), analyze("qwen2.5-1.5b")
    p = plan(model, device, evidence=store.evidence(device, model))
    assert p.recommended == "Q8_0"
    assert "measured quality 92.54% below required 96%" in _v(p, "Q4_K_M").reasons


def test_balanced_falls_back_to_fastest_when_nothing_is_usable(elitebook_results):
    device = get_device("hp-elitebook-840-g6").model_copy(update={"ram_gb": 16})
    model = analyze("qwen2.5-7b")
    ev = store.evidence(device, model)
    p = plan(model, device, evidence=ev)
    rec = _v(p, p.recommended)
    assert rec.tokens_per_s < 5
    assert rec.tokens_per_s == max(v.tokens_per_s for v in p.variants if v.meets_requirements and v.comfortable)
    # Asking for quality instead keeps the higher-precision, slower variant.
    q = plan(model, device, Requirements(prefer="quality"), ev)
    assert q.recommended == "Q8_0"


def test_calibration_from_1_5b_predicted_the_7b_speed(elitebook_results):
    # Before the 7B run, only the 1.5B results existed. The plan then said:
    device, model = get_device("hp-elitebook-840-g6").model_copy(update={"ram_gb": 16}), analyze("qwen2.5-7b")
    q4 = _v(plan(model, device, evidence=store.evidence(device, model)), "Q4_K_M")
    measured = 3.96  # tests/data/elitebook-840-g6-qwen2.5-7b.jsonl
    assert q4.speed_source == "calibrated"
    # Predicted 3.5 tok/s: 12% low. (The uncalibrated spec-sheet estimate, 4.4,
    # was 11% high: calibration's real win on this laptop was the CPU ceiling
    # for low-bit formats, see test_calibrated_speeds_respect_the_compute_ceiling.)
    assert abs(q4.tokens_per_s - measured) / measured < 0.15


def test_split_7b_measurement_is_used_by_plan(elitebook_results):
    rows = (DATA / "elitebook-840-g6-qwen2.5-7b.jsonl").read_text()
    with store.results_path().open("a", encoding="utf-8") as f:
        f.write(rows)
    device, model = get_device("hp-elitebook-840-g6"), analyze("qwen2.5-7b")
    q4 = _v(plan(model, device, evidence=store.evidence(device, model)), "Q4_K_M")
    assert (q4.tokens_per_s, q4.speed_source) == (3.96, "measured")
    # The 7B run raised the CPU ceiling learned from the 1.5B runs.
    ev = store.evidence(device, model)
    assert ev.compute_gparams_per_s == pytest.approx(3.96 * 7.6156, abs=0.1)
    q3 = _v(plan(model, device, evidence=ev), "Q3_K_M")
    assert q3.speed_source == "calibrated" and q3.tokens_per_s >= 3.96
