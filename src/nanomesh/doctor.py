"""`nanomesh doctor`: what's installed for AI work on this machine, and what's wrong.

Runs only when asked; nothing is watched or uploaded.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata

import psutil
from pydantic import BaseModel

from nanomesh.hardware import GB, DeviceProfile
from nanomesh.toolchain import find_toolchain

# Python packages that matter for local AI work, with what each is for.
PACKAGES = {
    "torch": "PyTorch",
    "transformers": "Hugging Face models",
    "accelerate": "multi-device / offloading",
    "peft": "LoRA fine-tuning",
    "bitsandbytes": "4/8-bit loading, QLoRA",
    "safetensors": "safe weight files",
    "huggingface_hub": "downloading models",
    "llama_cpp_python": "llama.cpp from Python",
    "onnxruntime": "ONNX inference",
    "onnxruntime-gpu": "ONNX inference on GPU",
    "optimum": "export to ONNX / OpenVINO",
    "openvino": "Intel CPU/iGPU/NPU inference",
    "tensorflow": "TensorFlow",
    "mlx": "Apple Silicon inference",
    "vllm": "GPU serving",
}
TOOLS = ["nvidia-smi", "ollama", "lms", "docker", "git", "cmake"]
TORCH_PROBE = (
    "import json, torch; print(json.dumps({'version': torch.__version__, 'cuda_build': torch.version.cuda, "
    "'cuda': torch.cuda.is_available(), 'mps': getattr(torch.backends, 'mps', None) is not None "
    "and torch.backends.mps.is_available()}))"
)
LOW_DISK_GB = 15


class Finding(BaseModel):
    level: str  # ok | warn | fail
    topic: str
    message: str
    fix: str | None = None


class Environment(BaseModel):
    python: str
    python_path: str
    in_virtualenv: bool
    os: str
    packages: dict[str, str]  # name -> version, installed only
    torch: dict | None = None
    tools: dict[str, str]  # name -> path, found only
    llama_cpp: dict[str, str | None]
    disk_free_gb: float | None = None
    findings: list[Finding]


def _version(name: str) -> str | None:
    for candidate in (name, name.replace("_", "-"), name.replace("-", "_")):
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return None


def _torch_info() -> dict | None:
    # In a subprocess: importing torch takes seconds and a broken install can crash.
    try:
        out = subprocess.run([sys.executable, "-c", TORCH_PROBE], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return {"error": "timed out"}
    if out.returncode != 0:
        return {"error": (out.stderr.strip().splitlines() or ["import failed"])[-1]}
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": "unexpected output"}


def inspect(device: DeviceProfile | None = None) -> Environment:
    packages = {name: v for name in PACKAGES if (v := _version(name))}
    torch = _torch_info() if "torch" in packages else None
    tools = {t: p for t in TOOLS if (p := shutil.which(t))}
    tc = find_toolchain()
    llama = {"llama-bench": str(tc.bench) if tc.bench else None,
             "llama-quantize": str(tc.quantize) if tc.quantize else None,
             "llama-perplexity": str(tc.perplexity) if tc.perplexity else None,
             "convert_hf_to_gguf.py": str(tc.convert_script) if tc.convert_script else None}
    try:
        disk = round(shutil.disk_usage(os.path.expanduser("~")).free / GB, 1)
    except OSError:
        disk = None
    env = Environment(
        python=platform.python_version(), python_path=sys.executable,
        in_virtualenv=sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        os=f"{platform.system()} {platform.release()}", packages=packages, torch=torch, tools=tools,
        llama_cpp=llama, disk_free_gb=disk, findings=[],
    )
    env.findings = diagnose(env, device)
    return env


def diagnose(env: Environment, device: DeviceProfile | None = None) -> list[Finding]:
    """Turn what's installed into findings. Pure: tested with made-up environments."""
    f: list[Finding] = []
    has_nvidia = bool(device and any(g.vendor == "NVIDIA" for g in device.gpus)) or "nvidia-smi" in env.tools
    p = env.packages

    major, minor = (int(x) for x in env.python.split(".")[:2])
    if (major, minor) < (3, 10):
        f.append(Finding(level="fail", topic="Python", message=f"Python {env.python} is too old for NanoMesh and "
                         "most current AI libraries.", fix="Install Python 3.10 or newer."))
    elif (major, minor) >= (3, 13) and "torch" not in p:
        f.append(Finding(level="warn", topic="Python", message=f"Python {env.python} is very new; some AI "
                         "libraries don't publish wheels for it yet.", fix="Python 3.11 or 3.12 is the safe choice."))
    if not env.in_virtualenv:
        f.append(Finding(level="warn", topic="Python", message="Not running in a virtual environment.",
                         fix="python -m venv .venv, then activate it, so projects don't break each other."))

    t = env.torch
    if t and "error" in t:
        f.append(Finding(level="fail", topic="PyTorch", message=f"PyTorch is installed but fails to import: {t['error']}",
                         fix="Reinstall it: see pytorch.org/get-started for the right command."))
    elif t:
        if t.get("cuda_build") and not has_nvidia:
            f.append(Finding(level="warn", topic="PyTorch", message=f"PyTorch {t['version']} is a CUDA build, but this "
                             "machine has no NVIDIA GPU. It works on CPU, but the download is ~2 GB larger than needed.",
                             fix="pip install torch --index-url https://download.pytorch.org/whl/cpu"))
        if has_nvidia and not t.get("cuda"):
            what = "a CPU-only build" if not t.get("cuda_build") else "a CUDA build that can't see the GPU"
            f.append(Finding(level="fail", topic="PyTorch", message=f"This machine has an NVIDIA GPU but PyTorch is "
                             f"{what}, so everything runs on the CPU.",
                             fix="Update the NVIDIA driver and install the CUDA build from pytorch.org."))
        if t.get("cuda") or t.get("mps"):
            f.append(Finding(level="ok", topic="PyTorch",
                             message=f"PyTorch {t['version']} can use the {'GPU (CUDA)' if t.get('cuda') else 'Apple GPU (MPS)'}."))
    if "transformers" in p and "torch" not in p and "tensorflow" not in p:
        f.append(Finding(level="fail", topic="Transformers", message="transformers is installed without PyTorch, so "
                         "it can't load or run models.", fix="pip install torch"))
    if "bitsandbytes" in p and not has_nvidia:
        f.append(Finding(level="warn", topic="bitsandbytes", message="bitsandbytes (4/8-bit loading, QLoRA) needs an "
                         "NVIDIA GPU for its usual workflows; on this machine it won't speed anything up.",
                         fix="For local inference on CPU use GGUF with llama.cpp instead (nanomesh plan)."))
    if "onnxruntime" in p and "onnxruntime-gpu" in p:
        f.append(Finding(level="warn", topic="ONNX Runtime", message="Both onnxruntime and onnxruntime-gpu are "
                         "installed; they conflict and the CPU one often wins.",
                         fix="pip uninstall onnxruntime, keep onnxruntime-gpu."))

    llama = env.llama_cpp
    if not llama.get("llama-bench"):
        f.append(Finding(level="warn", topic="llama.cpp", message="llama.cpp tools not found, so NanoMesh can plan "
                         "but can't build or benchmark models.",
                         fix="Download a release from github.com/ggml-org/llama.cpp/releases and set NANOMESH_LLAMA_CPP "
                             "to its folder."))
    else:
        f.append(Finding(level="ok", topic="llama.cpp", message=f"llama.cpp tools found ({llama['llama-bench']})."))
        if not llama.get("llama-perplexity"):
            f.append(Finding(level="warn", topic="llama.cpp", message="llama-perplexity is missing, so quality can't "
                             "be measured.", fix="Use a full llama.cpp release, which includes it."))
        if not llama.get("convert_hf_to_gguf.py"):
            f.append(Finding(level="warn", topic="llama.cpp", message="convert_hf_to_gguf.py isn't on "
                             "NANOMESH_LLAMA_CPP, so Hugging Face models can't be converted (ready-made GGUFs still work).",
                             fix="Add a llama.cpp source checkout to NANOMESH_LLAMA_CPP (separate folders with ';' on "
                                 "Windows, ':' elsewhere)."))
    if has_nvidia and "nvidia-smi" not in env.tools:
        f.append(Finding(level="warn", topic="GPU", message="NVIDIA GPU present but nvidia-smi isn't on PATH, which "
                         "usually means the driver isn't installed properly.", fix="Install the latest NVIDIA driver."))

    if env.disk_free_gb is not None and env.disk_free_gb < LOW_DISK_GB:
        f.append(Finding(level="warn", topic="Disk", message=f"Only {env.disk_free_gb:g} GB free. A 7B model with a "
                         "couple of variants needs 15-20 GB.", fix="Free up space or keep models on another drive."))
    free_ram = psutil.virtual_memory().available / GB
    if device and free_ram < device.ram_gb * 0.3:
        f.append(Finding(level="warn", topic="Memory", message=f"Only {free_ram:.1f} GB of {device.ram_gb:g} GB RAM is "
                         "free right now.", fix="Close browsers and other heavy apps before running models."))
    return f
