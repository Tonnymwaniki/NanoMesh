import pytest

from nanomesh import conditions as cond
from nanomesh import results as store
from nanomesh.conditions import Conditions, advice, summarize
from nanomesh.devices import get_device
from nanomesh.model import analyze
from nanomesh.planner import plan
from nanomesh.results import SustainedPoint
from nanomesh.stress import summarize_sustained

# Captured from real Windows machines.
POWERCFG = "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)\r\n"
TYPEPERF = '''
"(PDH-CSV 4.0)","\\\\HP-840\\Processor Information(_Total)\\% Processor Performance"
"09/27/2026 10:00:01.123","87.531250"
Exiting, please wait...
The command completed successfully.
'''


def test_parsers():
    assert cond.parse_powercfg(POWERCFG) == "Balanced"
    assert cond.parse_powercfg("") is None
    assert cond.parse_typeperf(TYPEPERF) == 87.5
    assert cond.parse_typeperf("Error: No valid counters.") is None
    assert cond.parse_pmset_lowpower(" lowpowermode         1\n") is True
    assert cond.parse_pmset_lowpower("nothing") is None
    assert cond.parse_windows_battery('{"DischargeRate": 12850, "FullChargedCapacity": 41200, '
                                      '"DesignedCapacity": 50000}') == {
        "discharge_w": 12.85, "battery_full_wh": 41.2, "battery_design_wh": 50.0}
    assert cond.parse_windows_battery("not json") == {}


def test_read_conditions_never_raises():
    c = cond.read_conditions()
    assert c.available_ram_gb and c.available_ram_gb > 0


def test_clock_pct():
    assert Conditions(cpu_perf_pct=87.5).clock_pct == 87.5  # Windows counter wins
    assert Conditions(cpu_mhz=1600, cpu_max_mhz=3200).clock_pct == 50.0
    assert Conditions().clock_pct is None


def test_advice():
    assert advice(Conditions(on_battery=False, cpu_load_pct=5)) == []
    tips = " ".join(advice(Conditions(on_battery=True, power_mode="Best power efficiency", temp_c=95,
                                      cpu_load_pct=70, battery_full_wh=20, battery_design_wh=50)))
    for words in ("battery", "Best power efficiency", "95°C", "70%", "40% of its original"):
        assert words in tips
    # Clock speed only means something under load: idle CPUs downclock on purpose.
    assert not advice(Conditions(cpu_mhz=800, cpu_max_mhz=3000))
    assert "33.3% of its rated speed" in advice(Conditions(cpu_mhz=1000, cpu_max_mhz=3000), busy=True)[0]
    assert Conditions(power_plan="powersave").power_saving


def test_summarize_run():
    start = Conditions(on_battery=True, battery_pct=80, temp_c=60)
    readings = [Conditions(on_battery=True, battery_pct=79, temp_c=88, cpu_perf_pct=95, discharge_w=14),
                Conditions(on_battery=True, battery_pct=78, temp_c=93, cpu_perf_pct=71, discharge_w=16)]
    s = summarize(start, readings, 120)
    assert (s.temp_max_c, s.clock_pct_min, s.battery_drop_pct, s.discharge_w_avg) == (93, 71, 2, 15)
    assert s.on_battery_any


def _points(speeds, watts=None, batt=None, step=30):
    return [SustainedPoint(t_s=i * step, tokens_per_s=v, discharge_w=(watts or [None] * len(speeds))[i],
                           battery_pct=(batt or [None] * len(speeds))[i]) for i, v in enumerate(speeds)]


def test_sustained_throttling_and_energy():
    run = summarize_sustained(_points([4.0, 4.0, 3.8, 3.2, 3.0, 3.0], watts=[15] * 6), full_wh=45)
    assert (run.burst_tokens_per_s, run.sustained_tokens_per_s, run.drop_pct) == (4.0, 3.0, 25.0)
    assert run.watts == 15 and run.joules_per_token == 5.0
    assert run.battery_hours == 3.0
    assert run.tokens_per_battery_pct == round(45 * 3600 / 100 / 5)


def test_sustained_battery_life_from_percentage():
    # No power reading: fall back to the percentage dropping 2% in 5 minutes.
    run = summarize_sustained(_points([5.0] * 11, batt=[90 - i * 0.2 for i in range(11)]), full_wh=None)
    assert run.battery_hours == pytest.approx(4.2, abs=0.05)
    assert run.watts is None


def _row(**kw):
    base = dict(timestamp="2026-09-27T10:00:00+00:00", device_key="hp-elitebook-840-g6",
                device_name="HP EliteBook 840 G6", model_name="qwen2.5-1.5b-instruct", model_params=1_777_088_000,
                format="Q4_K_M", file_size_gb=1.04, gen_tokens_per_s=15.0)
    return store.Result(**(base | kw))


def _run_cond(on_battery):
    return summarize(Conditions(on_battery=on_battery), [], 10)


def test_plugged_in_runs_win_calibration():
    device, model = get_device("hp-elitebook-840-g6"), analyze("qwen2.5-1.5b")
    rows = [_row(gen_tokens_per_s=15.0, conditions=_run_cond(False)),
            _row(gen_tokens_per_s=9.0, timestamp="2026-09-27T11:00:00+00:00", conditions=_run_cond(True))]
    # The later battery run is slower; plugged-in results are the reference.
    assert store.evidence(device, model, rows).speeds["Q4_K_M"] == 15.0
    # With only battery runs, they are all there is.
    assert store.evidence(device, model, rows[1:]).speeds["Q4_K_M"] == 9.0


def test_thread_sweep_feeds_plan_advice():
    device, model = get_device("hp-elitebook-840-g6"), analyze("qwen2.5-1.5b")  # 4 physical cores
    sweep = [_row(kind="threads", threads=t, gen_tokens_per_s=v) for t, v in [(2, 11.0), (4, 13.0), (6, 14.6), (8, 12.1)]]
    ev = store.evidence(device, model, sweep)
    assert (ev.best_threads, ev.best_threads_gain_pct) == (6, 12.3)
    assert not ev.speeds  # sweep rows aren't default-settings speeds
    assert any("-t 6" in a for a in plan(model, device, evidence=ev).advice)


def test_sustained_drop_feeds_plan_advice():
    device, model = get_device("hp-elitebook-840-g6"), analyze("qwen2.5-1.5b")
    run = summarize_sustained(_points([15.0, 15.0, 12.0, 11.0, 11.0, 11.0]), full_wh=None)
    ev = store.evidence(device, model, [_row(kind="sustained", sustained=run)])
    assert ev.sustained_drop_pct == run.drop_pct
    assert any("Speed fell" in a for a in plan(model, device, evidence=ev).advice)


def test_old_results_without_conditions_still_load(isolated_home):
    isolated_home.mkdir(parents=True, exist_ok=True)
    with store.results_path().open("w", encoding="utf-8") as f:
        f.write(_row().model_dump_json(exclude={"kind", "conditions", "sustained"}) + "\n")
    [r] = store.load()
    assert r.kind == "benchmark" and r.conditions is None and not r.on_battery
