"""Wrappers around llama.cpp for conversion, quantization and benchmarking.

NanoMesh doesn't reimplement quantization kernels; it drives battle-tested
tools and decides *which* variant to build. Point NANOMESH_LLAMA_CPP at a
llama.cpp checkout/build, or put its binaries on PATH.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel


class ToolchainError(RuntimeError):
    pass


class Toolchain(BaseModel):
    convert_script: Path | None = None
    quantize: Path | None = None
    bench: Path | None = None
    perplexity: Path | None = None

    @property
    def can_convert(self) -> bool:
        return self.convert_script is not None and self.quantize is not None


def _find_binary(name: str, roots: list[Path]) -> Path | None:
    names = [name + ".exe", name] if sys.platform == "win32" else [name]
    # CMake puts Windows binaries under build/bin/Release.
    subdirs = ("", "bin", "build/bin", "build/bin/Release")
    for root in roots:
        for sub in subdirs:
            for n in names:
                candidate = root / sub / n
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate
    found = shutil.which(name)
    return Path(found) if found else None


def find_toolchain() -> Toolchain:
    roots = [Path(p).expanduser() for p in os.environ.get("NANOMESH_LLAMA_CPP", "").split(os.pathsep) if p]
    convert = next((r / "convert_hf_to_gguf.py" for r in roots if (r / "convert_hf_to_gguf.py").is_file()), None)
    return Toolchain(
        convert_script=convert,
        quantize=_find_binary("llama-quantize", roots),
        bench=_find_binary("llama-bench", roots),
        perplexity=_find_binary("llama-perplexity", roots),
    )


def conversion_commands(tc: Toolchain, model_dir: Path, out_dir: Path, formats: list[str]) -> list[list[str]]:
    """Commands that turn a Hugging Face model directory into GGUF variants."""
    f16 = out_dir / "model-F16.gguf"
    convert = str(tc.convert_script or "convert_hf_to_gguf.py")
    quantize = str(tc.quantize or "llama-quantize")
    cmds = [[sys.executable, convert, str(model_dir), "--outtype", "f16", "--outfile", str(f16)]]
    for fmt in formats:
        if fmt != "F16":
            cmds.append([quantize, str(f16), str(out_dir / f"model-{fmt}.gguf"), fmt])
    return cmds


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-15:]
        raise ToolchainError(f"Command failed: {' '.join(cmd)}\n" + "\n".join(tail))


class BenchResult(BaseModel):
    model_file: str
    size_gb: float
    prompt_tokens_per_s: float | None = None
    gen_tokens_per_s: float | None = None
    peak_rss_gb: float | None = None
    backend: str | None = None
    threads: int | None = None


def parse_llama_bench(output: str, model_file: Path) -> BenchResult:
    rows = json.loads(output)
    from nanomesh.model import gguf_size

    res = BenchResult(model_file=str(model_file), size_gb=round(gguf_size(model_file) / 1024**3, 4))
    for row in rows:
        speed = row.get("avg_ts")
        if row.get("n_gen", 0) > 0 and row.get("n_prompt", 0) == 0:
            res.gen_tokens_per_s = round(speed, 2)
        elif row.get("n_prompt", 0) > 0 and row.get("n_gen", 0) == 0:
            res.prompt_tokens_per_s = round(speed, 2)
        res.backend = row.get("backends") or row.get("backend") or res.backend
        res.threads = row.get("n_threads", res.threads)
    return res


def benchmark_gguf(tc: Toolchain, model_file: Path, threads: int | None = None,
                   prompt: int = 512, gen: int = 128) -> BenchResult:
    if not tc.bench:
        raise ToolchainError("llama-bench not found. Install llama.cpp and set NANOMESH_LLAMA_CPP.")
    cmd = [str(tc.bench), "-m", str(model_file), "-p", str(prompt), "-n", str(gen), "-o", "json"]
    if threads:
        cmd += ["-t", str(threads)]
    peak = _run_tracking_memory(cmd)
    res = parse_llama_bench(peak[0], model_file)
    res.peak_rss_gb = peak[1]
    return res


def _run_tracking_memory(cmd: list[str]) -> tuple[str, float | None]:
    """Run a command; return (stdout, peak RSS in GB)."""
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True)
        if hasattr(os, "wait4"):
            # The kernel tracks the child's exact peak RSS; no sampling needed.
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            # ru_maxrss is bytes on macOS, kilobytes elsewhere.
            peak = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        else:
            peak = _sample_peak_rss(proc)
        out.seek(0)
        err.seek(0)
        if proc.returncode != 0:
            raise ToolchainError(f"llama-bench failed:\n{err.read().strip()[-2000:]}")
        return out.read(), (round(peak / 1024**3, 4) if peak else None)


def _sample_peak_rss(proc: subprocess.Popen) -> int:
    import psutil

    peak = 0
    try:
        ps = psutil.Process(proc.pid)
        while proc.poll() is None:
            peak = max(peak, ps.memory_info().rss)
            try:
                proc.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
    except psutil.Error:
        proc.wait()
    return peak


EVAL_CONTEXT = 512
EVAL_CHUNKS = 16


def default_eval_text() -> Path:
    from importlib import resources

    return Path(str(resources.files("nanomesh.data").joinpath("eval_text.txt")))


def parse_perplexity(output: str) -> float:
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", output)
    if not m:
        raise ToolchainError("Could not find the final perplexity in llama-perplexity output.")
    return float(m.group(1))


def measure_perplexity(tc: Toolchain, model_file: Path, text: Path | None = None,
                       threads: int | None = None, chunks: int = EVAL_CHUNKS) -> float:
    """Perplexity of a GGUF model on an evaluation text (lower is better)."""
    if not tc.perplexity:
        raise ToolchainError("llama-perplexity not found. Install llama.cpp and set NANOMESH_LLAMA_CPP.")
    cmd = [str(tc.perplexity), "-m", str(model_file), "-f", str(text or default_eval_text()),
           "-c", str(EVAL_CONTEXT), "--chunks", str(chunks)]
    if threads:
        cmd += ["-t", str(threads)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise ToolchainError(f"llama-perplexity failed:\n{output.strip()[-2000:]}")
    return parse_perplexity(output)


def bench_rows(tc: Toolchain, model_file: Path, args: list[str]) -> list[dict]:
    """Run llama-bench with extra arguments; return its JSON rows."""
    if not tc.bench:
        raise ToolchainError("llama-bench not found. Install llama.cpp and set NANOMESH_LLAMA_CPP.")
    result = subprocess.run([str(tc.bench), "-m", str(model_file), *args, "-o", "json"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise ToolchainError(f"llama-bench failed:\n{(result.stderr or result.stdout).strip()[-2000:]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ToolchainError("llama-bench did not return JSON output.") from None
