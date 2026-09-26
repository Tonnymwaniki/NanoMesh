"""Hardware detection: build a device profile for the machine NanoMesh runs on."""

from __future__ import annotations

import json
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
    kind: str = "unknown"  # laptop, desktop, phone, sbc, gpu, server, generic
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
    def best_gpu(self) -> GPU | None:
        dedicated = [g for g in self.gpus if g.vram_gb and not g.unified_memory]
        return max(dedicated, key=lambda g: g.vram_gb or 0, default=None)


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
            return platform.processor() or None
    except (OSError, subprocess.SubprocessError):
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


def scan_device() -> DeviceProfile:
    """Profile the current machine."""
    mem = psutil.virtual_memory()
    ram_gb = round(mem.total / GB, 1)
    gpus = _nvidia_gpus() + _apple_gpu(ram_gb)
    try:
        disk_free = round(shutil.disk_usage(Path.home()).free / GB, 1)
    except OSError:
        disk_free = None
    node = platform.node() or "This machine"
    return DeviceProfile(
        id="local",
        name=node,
        kind="local",
        os=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
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
    budget = device.available_ram_gb or device.ram_gb
    if budget >= 24:
        return "High-memory CPU"
    if budget >= 10:
        return "Mid-range Edge"
    if budget >= 5:
        return "Entry-level Edge"
    return "Constrained Edge"


def dump(device: DeviceProfile) -> str:
    return json.dumps(device.model_dump(exclude_none=True), indent=2)
