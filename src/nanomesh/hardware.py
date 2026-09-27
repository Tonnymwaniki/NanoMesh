"""Hardware detection: build a device profile for the machine NanoMesh runs on."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import psutil
from pydantic import BaseModel, Field

GB = 1024**3

# CPU feature flags that matter for llama.cpp / ONNX Runtime CPU kernels.
INTERESTING_FLAGS = (
    "sse4_2", "avx", "avx2", "fma", "f16c", "avx512f", "avx512_vnni", "avx_vnni",
    "amx_int8", "neon", "asimd", "asimddp", "sve", "i8mm",
)


class GPU(BaseModel):
    name: str
    vendor: str
    vram_gb: float | None = None
    compute_capability: str | None = None
    # Memory bandwidth in GB/s, when known. Drives token/s estimates.
    bandwidth_gbps: float | None = None
    unified_memory: bool = False


class DeviceProfile(BaseModel):
    """Everything NanoMesh needs to know about a deployment target."""

    id: str = "local"
    name: str
    kind: str = "unknown"  # laptop, desktop, phone, sbc, gpu, server
    is_local: bool = False  # profiled live on this machine (vs. from the database)
    vendor: str | None = None
    model: str | None = None
    # Database entry this machine was recognised as, if any.
    matched_id: str | None = None
    # Regexes matched against "vendor model" to recognise a scanned machine.
    match: list[str] = Field(default_factory=list, exclude=True)
    os: str | None = None
    arch: str
    cpu: str | None = None
    physical_cores: int | None = None
    logical_cores: int | None = None
    cpu_flags: list[str] = Field(default_factory=list)
    ram_gb: float
    available_ram_gb: float | None = None
    # System memory bandwidth in GB/s, when known.
    memory_bandwidth_gbps: float | None = None
    disk_free_gb: float | None = None
    gpus: list[GPU] = Field(default_factory=list)
    npu: str | None = None
    runtimes: list[str] = Field(default_factory=list)
    notes: str | None = None

    @property
    def key(self) -> str:
        """Stable identity used to file benchmark results under."""
        if self.matched_id or not self.is_local:
            return self.matched_id or self.id
        who = " ".join(x for x in (self.vendor, self.model) if x) or self.cpu or "unknown"
        return f"local:{who}|{self.ram_gb:g}GB".lower()

    @property
    def best_gpu(self) -> GPU | None:
        dedicated = [g for g in self.gpus if g.vram_gb and not g.unified_memory]
        return max(dedicated, key=lambda g: g.vram_gb or 0, default=None)


# Placeholder strings firmware vendors leave in DMI fields.
_JUNK = re.compile(r"^(to be filled.*|system (product name|manufacturer|version)|default string|"
                   r"none|not applicable|o\.?e\.?m\.?|x\.x|0|)$", re.IGNORECASE)


def _clean(value: str | None) -> str | None:
    value = (value or "").strip().strip("\x00").strip()
    return None if _JUNK.match(value) else value


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(errors="ignore")
    except OSError:
        return None


def _cmd(args: list[str]) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def is_android() -> bool:
    return "ANDROID_ROOT" in os.environ or "ANDROID_DATA" in os.environ


def _android_identity() -> tuple[str | None, str | None]:
    def prop(name):
        return _clean(_cmd(["getprop", name]))

    vendor = prop("ro.product.brand") or prop("ro.product.manufacturer")
    # Market names ("Redmi 14C") beat model codes ("2409BRN2CA") when the OEM sets them.
    model = prop("ro.product.marketname") or prop("ro.vendor.product.marketname") or prop("ro.product.model")
    soc = prop("ro.soc.model")
    if soc and model:
        model = f"{model} ({soc})"
    return vendor, model


def system_identity() -> tuple[str | None, str | None]:
    """Return the machine's (vendor, model), e.g. ("HP", "HP EliteBook 840 G3")."""
    system = platform.system()
    if system == "Linux" and is_android():
        return _android_identity()
    if system == "Linux":
        dmi = "/sys/class/dmi/id/"
        vendor = _clean(_read(dmi + "sys_vendor"))
        name = _clean(_read(dmi + "product_name"))
        version = _clean(_read(dmi + "product_version"))
        # Lenovo puts the marketing name ("ThinkPad T480") in product_version.
        if version and re.search(r"[a-z]{3}", version, re.IGNORECASE):
            name = f"{version} ({name})" if name else version
        if name:
            return vendor, name
        # Single-board computers (Raspberry Pi, Jetson) describe themselves in the device tree.
        tree = _clean(_read("/proc/device-tree/model") or _read("/sys/firmware/devicetree/base/model"))
        return (tree.split()[0] if tree else None), tree
    if system == "Windows":
        out = _cmd(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model | ConvertTo-Json"])
        try:
            data = json.loads(out) if out else {}
        except json.JSONDecodeError:
            data = {}
        return _clean(data.get("Manufacturer")), _clean(data.get("Model"))
    if system == "Darwin":
        return "Apple", _clean(_cmd(["sysctl", "-n", "hw.model"]))
    return None, None


def _device_kind() -> str:
    if is_android():
        return "phone"
    try:
        has_battery = psutil.sensors_battery() is not None
    except (AttributeError, NotImplementedError, OSError):
        has_battery = False
    return "laptop" if has_battery else "desktop"


def _cpu_name() -> str | None:
    system = platform.system()
    try:
        if system == "Linux":
            text = Path("/proc/cpuinfo").read_text(errors="ignore")
            for key in ("model name", "Hardware", "Processor"):
                m = re.search(rf"^{key}\s*:\s*(.+)$", text, re.MULTILINE)
                if m:
                    return m.group(1).strip()
        elif system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            return out.stdout.strip() or None
        elif system == "Windows":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (OSError, subprocess.SubprocessError, ImportError):
        pass
    return platform.processor() or None


def _cpu_flags() -> list[str]:
    flags: set[str] = set()
    if platform.system() == "Linux":
        try:
            text = Path("/proc/cpuinfo").read_text(errors="ignore")
            m = re.search(r"^(?:flags|Features)\s*:\s*(.+)$", text, re.MULTILINE)
            if m:
                flags = set(m.group(1).split())
        except OSError:
            pass
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        flags = {"neon", "asimd", "asimddp", "i8mm"}
    return sorted(f for f in INTERESTING_FLAGS if f in flags)


def _nvidia_gpus() -> list[GPU]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            vram = round(float(parts[1]) / 1024, 1)
        except ValueError:
            vram = None
        cc = parts[2] if len(parts) > 2 and parts[2] not in ("", "[N/A]") else None
        gpus.append(GPU(name=parts[0], vendor="NVIDIA", vram_gb=vram, compute_capability=cc))
    return gpus


def _apple_gpu(ram_gb: float) -> list[GPU]:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return [GPU(name="Apple Silicon GPU", vendor="Apple", vram_gb=ram_gb, unified_memory=True)]
    return []


def _detect_runtimes(gpus: list[GPU]) -> list[str]:
    runtimes = ["llama.cpp (CPU)"]
    if any(g.vendor == "NVIDIA" for g in gpus):
        runtimes.append("llama.cpp (CUDA)")
    if any(g.vendor == "Apple" for g in gpus):
        runtimes.append("llama.cpp (Metal)")
    try:
        import onnxruntime  # type: ignore[import-not-found]

        runtimes += [f"onnxruntime ({p})" for p in onnxruntime.get_available_providers()]
    except ImportError:
        pass
    return runtimes


def _normalize_arch(machine: str) -> str:
    # Windows reports x86-64 as "AMD64" even on Intel CPUs.
    return {"amd64": "x86_64", "x64": "x86_64", "arm64": "arm64", "aarch64": "aarch64"}.get(machine.lower(), machine)


def scan_device() -> DeviceProfile:
    """Profile the current machine."""
    mem = psutil.virtual_memory()
    ram_gb = round(mem.total / GB, 1)
    gpus = _nvidia_gpus() + _apple_gpu(ram_gb)
    try:
        disk_free = round(shutil.disk_usage(Path.home()).free / GB, 1)
    except OSError:
        disk_free = None
    vendor, model = system_identity()
    if model:
        name = model if vendor and model.lower().startswith(vendor.lower()) else " ".join(filter(None, (vendor, model)))
    else:
        name = platform.node() or "This machine"
    return DeviceProfile(
        id="local",
        name=name,
        kind=_device_kind(),
        is_local=True,
        vendor=vendor,
        model=model,
        os="Android" if is_android() else f"{platform.system()} {platform.release()}",
        arch=_normalize_arch(platform.machine()),
        cpu=_cpu_name(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        cpu_flags=_cpu_flags(),
        ram_gb=ram_gb,
        available_ram_gb=round(mem.available / GB, 1),
        disk_free_gb=disk_free,
        gpus=gpus,
        runtimes=_detect_runtimes(gpus),
    )


def compute_class(device: DeviceProfile) -> str:
    """A coarse label for how much AI a device can realistically run."""
    gpu = device.best_gpu
    if gpu and gpu.vram_gb and gpu.vram_gb >= 40:
        return "Datacenter GPU"
    if gpu and gpu.vram_gb and gpu.vram_gb >= 8:
        return "Workstation GPU"
    budget = device.ram_gb
    if budget >= 24:
        return "High-memory CPU"
    if budget >= 10:
        return "Mid-range Edge"
    if budget >= 5:
        return "Entry-level Edge"
    return "Constrained Edge"


def dump(device: DeviceProfile) -> str:
    return json.dumps(device.model_dump(exclude_none=True), indent=2)
