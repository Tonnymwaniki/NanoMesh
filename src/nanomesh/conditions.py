"""Live device conditions: power, clocks, heat and load, right now.

Hardware says what a device *can* do; conditions say what it will do at this
moment. Laptops on battery or in a power-saving mode cut CPU speed a lot, and
hot phones and thin laptops throttle. Each reader is best-effort: a value is
None when the OS won't reveal it (Windows hides CPU temperature from normal
users, for example), and nothing here ever raises.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import threading
import time
from pathlib import Path

import psutil
from pydantic import BaseModel

from nanomesh.hardware import GB, is_android

# Windows 11 "power mode" overlays (Settings > System > Power).
WINDOWS_POWER_MODES = {
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
    "00000000-0000-0000-0000-000000000000": "Balanced",
    "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
}
SLOW_POWER_WORDS = ("efficien", "saver", "powersave", "low-power", "low power", "quiet", "cool", "battery")


class Conditions(BaseModel):
    on_battery: bool | None = None
    battery_pct: float | None = None
    battery_full_wh: float | None = None  # capacity at full charge today (wear included)
    battery_design_wh: float | None = None
    discharge_w: float | None = None  # battery power draw; only meaningful on battery
    power_plan: str | None = None  # Windows scheme / Linux governor
    power_mode: str | None = None  # Windows 11 power mode / Linux platform profile / macOS Low Power Mode
    cpu_mhz: float | None = None
    cpu_max_mhz: float | None = None
    cpu_perf_pct: float | None = None  # Windows: % of base clock the CPU is running at
    temp_c: float | None = None
    cpu_load_pct: float | None = None
    available_ram_gb: float | None = None

    @property
    def clock_pct(self) -> float | None:
        """How fast the CPU runs relative to its rating (100 = base, >100 = turbo)."""
        if self.cpu_perf_pct is not None:
            return self.cpu_perf_pct
        if self.cpu_mhz and self.cpu_max_mhz:
            return round(100 * self.cpu_mhz / self.cpu_max_mhz, 1)
        return None

    @property
    def power_saving(self) -> bool:
        text = f"{self.power_plan or ''} {self.power_mode or ''}".lower()
        return any(w in text for w in SLOW_POWER_WORDS)


def _run(args: list[str], timeout: float = 6) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(errors="ignore").strip()
    except OSError:
        return None


# ---- parsers (pure, unit-tested with captured output) ----

def parse_powercfg(output: str | None) -> str | None:
    """`powercfg /getactivescheme` -> 'Balanced'."""
    m = re.search(r"\(([^()]+)\)\s*$", (output or "").strip())
    return m.group(1).strip() if m else None


def parse_typeperf(output: str | None) -> float | None:
    """Highest value in the data row of `typeperf <counter(*)> -sc 1` output.

    Queried per logical processor: the _Total average is dragged down by idle
    cores (llama.cpp uses 4 of an i5's 8 threads, so it read ~50% under load),
    while the busiest core shows the speed the model actually runs at.
    """
    rows = [line for line in (output or "").splitlines() if re.match(r'^"[^"]*\d{2}:\d{2}:\d{2}', line)]
    if not rows:
        return None
    values = [float(v) for v in re.findall(r'"([\d.]+)"', rows[-1])]
    return round(max(values), 1) if values else None


def parse_pmset_lowpower(output: str | None) -> bool | None:
    m = re.search(r"lowpowermode\s+(\d)", output or "")
    return None if m is None else m.group(1) == "1"


def parse_windows_battery(output: str | None) -> dict:
    """JSON from our PowerShell battery query -> watts and watt-hours."""
    try:
        data = json.loads(output or "{}")
    except json.JSONDecodeError:
        return {}
    if isinstance(data, list):
        data = data[0] if data else {}
    res = {}
    rate = data.get("DischargeRate")
    # Windows reports "unknown" as -2147483648 (0x80000000); real draws are 0-500 W.
    if isinstance(rate, (int, float)) and 0 < rate < 500_000:
        res["discharge_w"] = round(rate / 1000, 2)  # mW
    if data.get("FullChargedCapacity"):
        res["battery_full_wh"] = round(data["FullChargedCapacity"] / 1000, 1)  # mWh
    if data.get("DesignedCapacity"):
        res["battery_design_wh"] = round(data["DesignedCapacity"] / 1000, 1)
    return res


# ---- per-OS readers ----

WIN_BATTERY_PS = (
    "$s = Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | Select-Object -First 1; "
    "$f = Get-CimInstance -Namespace root/wmi -ClassName BatteryFullChargedCapacity -ErrorAction SilentlyContinue | Select-Object -First 1; "
    "$d = Get-CimInstance -Namespace root/wmi -ClassName BatteryStaticData -ErrorAction SilentlyContinue | Select-Object -First 1; "
    "@{DischargeRate=$s.DischargeRate; FullChargedCapacity=$f.FullChargedCapacity; DesignedCapacity=$d.DesignedCapacity} | ConvertTo-Json"
)
WIN_TEMP_PS = (
    "$t = Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop | "
    "Measure-Object -Property CurrentTemperature -Maximum; [math]::Round($t.Maximum / 10 - 273.15, 1)"
)


def _windows(c: Conditions, slow_parts: bool) -> None:
    c.power_plan = parse_powercfg(_run(["powercfg", "/getactivescheme"]))
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes")
        value = "ActiveOverlayDcPowerScheme" if c.on_battery else "ActiveOverlayAcPowerScheme"
        guid = str(winreg.QueryValueEx(key, value)[0]).lower()
        c.power_mode = WINDOWS_POWER_MODES.get(guid)
    except (ImportError, OSError):
        pass
    # English counter name; on other display languages this returns nothing.
    c.cpu_perf_pct = parse_typeperf(_run(["typeperf", r"\Processor Information(*)\% Processor Performance", "-sc", "1"]))
    if slow_parts:
        for k, v in parse_windows_battery(_run(["powershell", "-NoProfile", "-Command", WIN_BATTERY_PS], 10)).items():
            setattr(c, k, v)
        # Usually needs admin rights; None is the normal result.
        temp = _run(["powershell", "-NoProfile", "-Command", WIN_TEMP_PS], 8)
        try:
            c.temp_c = float(temp.strip()) if temp and temp.strip() else None
        except ValueError:
            pass


def _linux(c: Conditions) -> None:
    c.power_plan = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    c.power_mode = _read("/sys/firmware/acpi/platform_profile")
    for bat in sorted(Path("/sys/class/power_supply").glob("BAT*")) + sorted(Path("/sys/class/power_supply").glob("battery")):
        def num(name: str) -> float | None:
            raw = _read(str(bat / name))
            try:
                return float(raw) if raw else None
            except ValueError:
                return None
        power = num("power_now")  # µW
        if power is None and num("current_now") and num("voltage_now"):
            power = num("current_now") * num("voltage_now") / 1e6  # µA * µV -> µW
        if power:
            c.discharge_w = round(power / 1e6, 2)
        full, design = num("energy_full"), num("energy_full_design")  # µWh
        if full:
            c.battery_full_wh = round(full / 1e6, 1)
        if design:
            c.battery_design_wh = round(design / 1e6, 1)
        break
    if c.temp_c is None:
        # Phones and SBCs: thermal zones in millidegrees.
        temps = []
        for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            raw = _read(str(zone))
            if raw and raw.lstrip("-").isdigit() and 10_000 < int(raw) < 130_000:
                temps.append(int(raw) / 1000)
        if temps:
            c.temp_c = round(max(temps), 1)


def _psutil_temperature() -> float | None:
    try:
        sensors = psutil.sensors_temperatures()
    except (AttributeError, OSError):
        return None
    for name in ("coretemp", "k10temp", "zenpower", "cpu_thermal", "cpu-thermal", "soc_thermal", "acpitz"):
        readings = [t.current for t in sensors.get(name, []) if t.current]
        if readings:
            return round(max(readings), 1)
    return None


def read_conditions(slow_parts: bool = True) -> Conditions:
    """Snapshot current conditions. slow_parts=False skips the readers that
    shell out to PowerShell, for frequent sampling during a benchmark."""
    c = Conditions()
    try:
        bat = psutil.sensors_battery()
    except (AttributeError, OSError, NotImplementedError):
        bat = None
    if bat is not None:
        c.on_battery = not bat.power_plugged if bat.power_plugged is not None else None
        c.battery_pct = round(bat.percent, 1)
    try:
        freq = psutil.cpu_freq()
        if freq:
            c.cpu_mhz = round(freq.current) or None
            c.cpu_max_mhz = round(freq.max) or None
    except (AttributeError, OSError, NotImplementedError):
        pass
    c.cpu_load_pct = psutil.cpu_percent(interval=0.3)
    c.available_ram_gb = round(psutil.virtual_memory().available / GB, 1)
    c.temp_c = _psutil_temperature()

    system = platform.system()
    try:
        if system == "Windows":
            _windows(c, slow_parts)
        elif system == "Linux":
            _linux(c)
        elif system == "Darwin":
            low = parse_pmset_lowpower(_run(["pmset", "-g"]))
            c.power_mode = None if low is None else ("Low Power Mode" if low else "Normal")
    except Exception:  # noqa: BLE001 - conditions are advisory; never break a scan or benchmark
        pass
    if not c.on_battery:
        c.discharge_w = None  # a plugged-in laptop's battery draw says nothing about the workload
    if is_android() and c.on_battery is None:
        c.on_battery = True
    return c


# ---- sampling during a workload ----

class RunConditions(BaseModel):
    """Conditions before and during one measured run."""

    start: Conditions
    samples: int = 0
    duration_s: float = 0
    on_battery_any: bool | None = None
    temp_max_c: float | None = None
    clock_pct_min: float | None = None
    clock_pct_avg: float | None = None
    battery_drop_pct: float | None = None
    discharge_w_avg: float | None = None


class Sampler:
    """Samples conditions in a background thread while a workload runs."""

    def __init__(self, interval_s: float = 3.0, final_reading: bool = True):
        self.interval_s = interval_s
        self.final_reading = final_reading
        self.start = read_conditions(slow_parts=final_reading)
        self.readings: list[Conditions] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._t0 = 0.0

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.readings.append(read_conditions(slow_parts=False))

    def __enter__(self) -> Sampler:
        self._t0 = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        if self.final_reading:
            self.readings.append(read_conditions())  # includes the slower battery-power reading

    def summary(self) -> RunConditions:
        return summarize(self.start, self.readings, time.monotonic() - self._t0)


def summarize(start: Conditions, readings: list[Conditions], duration_s: float) -> RunConditions:
    during = readings or [start]
    temps = [r.temp_c for r in during if r.temp_c is not None]
    clocks = [r.clock_pct for r in during if r.clock_pct is not None]
    watts = [r.discharge_w for r in during if r.discharge_w]
    batt = [r.battery_pct for r in [start, *during] if r.battery_pct is not None]
    flags = [r.on_battery for r in [start, *during] if r.on_battery is not None]
    return RunConditions(
        start=start, samples=len(readings), duration_s=round(duration_s, 1),
        on_battery_any=any(flags) if flags else None,
        temp_max_c=max(temps) if temps else None,
        clock_pct_min=min(clocks) if clocks else None,
        clock_pct_avg=round(sum(clocks) / len(clocks), 1) if clocks else None,
        battery_drop_pct=round(batt[0] - batt[-1], 1) if len(batt) > 1 else None,
        discharge_w_avg=round(sum(watts) / len(watts), 2) if watts else None,
    )


# ---- advice ----

HOT_C = 90.0


def advice(c: Conditions, *, busy: bool = False) -> list[str]:
    """What to change about the current conditions. busy=True when the
    readings were taken under load (idle CPUs downclock on purpose)."""
    out = []
    if c.on_battery:
        out.append("Running on battery: laptops usually slow the CPU to save power. Plug in for full speed, "
                   "and benchmark plugged in so results are comparable.")
    if c.power_saving:
        mode = c.power_mode or c.power_plan
        out.append(f"Power mode is '{mode}', which caps the CPU. Switch to best performance "
                   "(Windows: Settings > System > Power > Power mode).")
    if c.temp_c is not None and c.temp_c >= HOT_C:
        out.append(f"CPU is at {c.temp_c:g}°C and will throttle. Give it airflow (raise the back, clear the vents).")
    if busy and c.clock_pct is not None and c.clock_pct < 80:
        out.append(f"Under load the CPU ran at only {c.clock_pct:g}% of its rated speed: it is being held back "
                   "by power or heat limits.")
    if not busy and c.cpu_load_pct is not None and c.cpu_load_pct >= 40:
        out.append(f"Other programs are using {c.cpu_load_pct:g}% of the CPU right now, which slows models down.")
    if c.battery_full_wh and c.battery_design_wh and c.battery_full_wh < 0.7 * c.battery_design_wh:
        pct = round(100 * c.battery_full_wh / c.battery_design_wh)
        out.append(f"The battery holds {pct}% of its original charge, so battery-powered runs will be short.")
    return out


def dump(c: Conditions) -> dict:
    return {**c.model_dump(exclude_none=True), **({"clock_pct": c.clock_pct} if c.clock_pct is not None else {})}

