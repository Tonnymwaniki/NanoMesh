"""`nanomesh mcp`: NanoMesh as a local MCP server for coding agents.

Claude Code, Cursor, VS Code and other MCP clients start this over stdio and
call its tools to answer "can this run on my machine?" from this machine's
real hardware, models and benchmarks. Everything runs locally; nothing is sent
anywhere. Implements the stdio transport directly: newline-delimited JSON-RPC
2.0 on stdin/stdout (the protocol is small, so no SDK dependency).
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from nanomesh import __version__
from nanomesh import results as store
from nanomesh.conditions import advice as condition_advice
from nanomesh.conditions import dump as dump_conditions
from nanomesh.conditions import read_conditions
from nanomesh.devices import get_device, recognise
from nanomesh.hardware import DeviceProfile, scan_device
from nanomesh.model import analyze
from nanomesh.planner import QUALITY_TIERS, Plan, Requirements, add_live_advice, battery_advice, plan

PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]
MAX_WAIT_S = 50  # longest a tool call blocks; MCP clients time out calls after a minute or so
INSTRUCTIONS = (
    "NanoMesh knows this developer's actual machine: exact device model, live power/thermal conditions, models "
    "on disk and benchmarks measured here. Use it before recommending local models, quantization, runtimes or "
    "training setups: device_passport for what the machine can run, plan_model for which variant of a model "
    "to use (with measured speeds where available), list_local_models for models already downloaded, "
    "environment_doctor for installation problems, training_plan before fine-tuning. "
    "analyze_project finds the AI a codebase uses (cloud APIs and local models) and what could run locally "
    "instead, with the code change. fit_card sizes any model (speech, vision, embeddings, text) for this "
    "machine; search_models with a task finds them on Hugging Face and Kaggle. It can also get a model running "
    "end to end: search_models (Hugging Face, sized for this machine), "
    "download_model, benchmark_model, then start_model_server for an OpenAI-compatible endpoint. Downloads and "
    "benchmarks run as jobs: poll job_status. Always tell the user the download size before downloading."
)


LLAMA_MISSING = (
    "llama.cpp (llama-bench, llama-server) not found. Ask the user whether it's installed before downloading it: "
    "if it is, running `nanomesh doctor` once in a terminal where NANOMESH_LLAMA_CPP is set makes NanoMesh remember "
    "the folder; else they can unpack a release from github.com/ggml-org/llama.cpp/releases to C:\\llama.cpp "
    "(Windows) or ~/llama.cpp, where NanoMesh looks automatically."
)


class ToolError(Exception):
    pass


@lru_cache(maxsize=1)
def _local() -> DeviceProfile:
    return recognise(scan_device())


def _device(device_id: str | None) -> DeviceProfile:
    if not device_id or device_id == "local":
        return _local()
    try:
        return get_device(device_id)
    except KeyError as e:
        raise ToolError(e.args[0]) from None


def _model(spec: str):
    try:
        return analyze(spec)
    except ValueError as e:
        raise ToolError(str(e)) from None


def _device_summary(d: DeviceProfile) -> dict:
    return {k: v for k, v in {
        "name": d.name, "recognised_as": d.matched_id, "kind": d.kind, "os": d.os, "cpu": d.cpu,
        "cores": d.physical_cores, "threads": d.logical_cores, "ram_gb": d.ram_gb, "free_ram_gb": d.available_ram_gb,
        "memory_bandwidth_gbps": d.memory_bandwidth_gbps, "cpu_features": d.cpu_flags or None,
        "gpus": [{"name": g.name, "vram_gb": g.vram_gb, "shared_memory": g.unified_memory} for g in d.gpus] or None,
        "npu": d.npu,
    }.items() if v is not None}


def _plan_summary(p: Plan) -> dict:
    return {
        "model": p.model.name, "params_b": round(p.model.params / 1e9, 2), "device": p.device.name,
        "context": p.requirements.context, "memory_budget_gb": [round(b.memory_gb, 1) for b in p.budgets],
        "recommended": p.recommended,
        "variants": [{
            "variant": v.format.label, "format": v.format.name, "memory_gb": v.total_memory_gb,
            "tokens_per_s": v.tokens_per_s, "speed_source": v.speed_source,
            "quality_pct": v.quality_pct, "quality_source": v.quality_source, "runs_on": v.placement,
            "fits": v.fits, **({"ruled_out": "; ".join(v.reasons)} if v.reasons else {}),
        } for v in p.variants],
        "advice": p.advice,
        "legend": "speed/quality source: measured = benchmarked on this device; calibrated = predicted from this "
                  "device's benchmarks; estimate/typical = spec-sheet figures",
    }


# ---- tools ----

def device_passport(device: str | None = None) -> dict:
    from nanomesh.passport import passport

    pp = passport(_device(device))
    return {"device": _device_summary(pp.device), "compute_class": pp.compute_class,
            "model_memory_budget_gb": pp.budget_gb,
            "recommended_up_to_params_b_int4": pp.recommended_max_b, "possible_up_to_params_b_int4": pp.possible_max_b,
            "what_fits_4k_context": {f.size: ({"variant": f.label, "memory_gb": f.memory_gb, "tokens_per_s": f.tokens_per_s,
                                               "speed_source": f.speed_source} if f.fits else "does not fit")
                                     for f in pp.fits}}


def current_conditions() -> dict:
    c = read_conditions()
    note = battery_advice(store.battery_cost(_local()))
    return {"conditions": dump_conditions(c), "advice": condition_advice(c, battery_note=note)}


def plan_model(model: str, device: str | None = None, context: int = 4096, min_quality: str = "good",
               prefer: str = "balanced") -> dict:
    info, dev = _model(model), _device(device)
    req = _requirements(context, min_quality, prefer)
    ev = store.evidence(dev, info)
    p = plan(info, dev, req, ev)
    if dev.is_local:
        add_live_advice(p, ev, read_conditions())
    return _plan_summary(p)


def list_local_models(folders: list[str] | None = None) -> dict:
    from nanomesh.discover import find_models

    found = find_models(_local(), [Path(f).expanduser() for f in folders or []])
    return {"device": _local().name, "models": [m.model_dump(exclude_none=True) for m in found],
            "searched": "Hugging Face cache, LM Studio, Ollama, ~/models" + (", " + ", ".join(folders) if folders else "")}


def benchmark_results(model: str | None = None) -> dict:
    rows = [r for r in store.load() if not model or model.lower() in r.model_name.lower()]
    return {"results": [{k: v for k, v in {
        "when": r.timestamp, "device": r.device_name, "model": r.model_name, "variant": r.format, "kind": r.kind,
        "steady_state": r.steady or None, "gen_tokens_per_s": r.gen_tokens_per_s,
        "cold_start_tokens_per_s": r.burst_tokens_per_s, "prompt_tokens_per_s": r.prompt_tokens_per_s,
        "peak_ram_gb": r.peak_rss_gb, "quality_pct": r.quality_pct, "on_battery": r.on_battery or None,
        "threads": r.threads,
        "sustained": {"burst": r.sustained.burst_tokens_per_s, "sustained": r.sustained.sustained_tokens_per_s,
                      "drop_pct": r.sustained.drop_pct, "battery_hours": r.sustained.battery_hours}
        if r.sustained else None,
    }.items() if v is not None} for r in rows[-50:]]}


def _requirements(context: int, min_quality: str, prefer: str) -> Requirements:
    try:
        return Requirements(context=context, min_quality_pct=float(str(min_quality).rstrip("%")), prefer=prefer)
    except ValueError:
        if min_quality not in QUALITY_TIERS:
            raise ToolError(f"min_quality must be a percentage or one of {QUALITY_TIERS}") from None
        return Requirements(context=context, min_quality=min_quality, prefer=prefer)


def search_models(query: str = "", limit: int = 5, context: int = 4096, min_quality: str = "good",
                  prefer: str = "balanced", task: str | None = None, source: str = "huggingface") -> dict:
    from nanomesh.catalog import CatalogError, search, search_task, task_name

    try:
        task = task_name(task) if task else None
    except CatalogError as e:
        raise ToolError(str(e)) from None
    if task and task != "text generation":
        sources = ("huggingface", "kaggle") if source == "all" else (source,)
        try:
            found = search_task(query, task, _local(), limit=min(max(limit, 1), 10), sources=sources)
        except CatalogError as e:
            raise ToolError(str(e)) from None
        from nanomesh.catalog import runs_well_hint

        hint = runs_well_hint(found, task, _local())
        return {"device": _local().name, "task": task, "query": query,
                "results": [r.model_dump(exclude_none=True) for r in found],
                **({"runs_well_here": hint} if hint else {}),
                "note": "Each result's card is its Fit Card on this machine. fit_card(model) explains one in "
                        "detail; for vision/speech, download the file the card's 'how' names."}

    try:
        found = search(query, _local(), limit=min(max(limit, 1), 10), req=_requirements(context, min_quality, prefer))
    except CatalogError as e:
        raise ToolError(str(e)) from None
    return {"device": _local().name, "query": query,
            "results": [m.model_dump(exclude_none=True) for m in found],
            "note": "No repository name contains every word; try fewer or different words." if not found else
                    "recommended is the file to download for this device. Tell the user its download_gb and get "
                    "their OK, then call download_model(repo, file)."}


def download_model(repo: str, file: str | None = None) -> dict:
    from nanomesh import jobs
    from nanomesh.catalog import CatalogError
    from nanomesh.download import download, pull

    try:
        dp = pull(repo, file, _local())
    except CatalogError as e:
        raise ToolError(str(e)) from None
    info = {"file": dp.file.name, "parts": len(dp.file.parts), "download_gb": dp.file.size_gb,
            "remaining_gb": dp.remaining_gb, "destination": str(dp.target), "disk_free_gb": dp.disk_free_gb}
    if dp.remaining_gb == 0 and dp.target.exists():
        return {**info, "status": "already downloaded", "path": str(dp.target)}

    def run(h: jobs.Handle) -> dict:
        t0, start = time.monotonic(), dp.have_bytes

        def progress(done: int, total: int):
            elapsed = time.monotonic() - t0
            rate = (done - start) / elapsed if elapsed > 2 and done > start else None
            eta = (total - done) / rate if rate else None
            h.update(done / total if total else None,
                     f"{done / 1024**3:.2f} of {total / 1024**3:.2f} GB"
                     + (f" · {rate / 1e6:.1f} MB/s · ~{_duration(eta)} left" if eta is not None else ""), eta)
        path = download(dp, progress, h.cancel)
        return {"path": str(path), "next": "benchmark_model(path) to measure it here, or "
                                           "start_model_server(path) to use it."}

    job = jobs.start("download", f"{repo}/{dp.file.name}", run)
    return {**info, "job_id": job.id, "status": "started",
            "next": f"Call job_status(job_id, wait_seconds={MAX_WAIT_S}): it returns when the download finishes or "
                    "after that long, with speed and time left. Don't call it more often than that."}


def benchmark_model(path: str, quick: bool = True) -> dict:
    from nanomesh import jobs
    from nanomesh.evaluate import evaluate
    from nanomesh.runtimes import benchmark_file, can_benchmark
    from nanomesh.stress import WARMUP_SECONDS
    from nanomesh.toolchain import find_toolchain

    f = Path(path).expanduser()
    if can_benchmark(f):
        device = _local()

        def run_other(h: jobs.Handle) -> dict:
            h.update(message=f"Benchmarking {f.name}…")
            run = benchmark_file(f, device)
            return {**run.model_dump(exclude_none=True), "saved": True,
                    "note": "Saved: Fit Cards for this model's family on this machine now use it."}

        job = jobs.start("benchmark", f.name, run_other)
        return {"job_id": job.id, "status": "started", "expected": "under a minute",
                "next": f"Call job_status(job_id, wait_seconds={MAX_WAIT_S})."}
    if not f.is_file() or f.suffix != ".gguf":
        raise ToolError(f"Not a model NanoMesh can benchmark: {path} (.gguf, .onnx, or a whisper.cpp ggml-*.bin)")
    tc = find_toolchain()
    if not tc.bench:
        raise ToolError(LLAMA_MISSING)
    device = _local()

    def run(h: jobs.Handle) -> dict:
        rows = evaluate(tc, [f], device, quality=False, warmup_s=0 if quick else WARMUP_SECONDS,
                        log=lambda m: h.update(message=m))
        store.save(rows)
        r = rows[0]
        return {"model": r.model_name, "variant": r.format, "gen_tokens_per_s": r.gen_tokens_per_s,
                "prompt_tokens_per_s": r.prompt_tokens_per_s, "peak_ram_gb": r.peak_rss_gb,
                "steady_state": r.steady, "saved": True,
                "note": "Saved: plan_model and search_models now use this measurement."}

    job = jobs.start("benchmark", f.name, run)
    minutes = "about 1 minute" if quick else "2-4 minutes (warms the CPU up first)"
    return {"job_id": job.id, "status": "started", "expected": minutes,
            "next": f"Call job_status(job_id, wait_seconds={MAX_WAIT_S}) until it's done; ask the user to leave the "
                    "machine alone meanwhile."}


def _duration(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


def fit_card(model: str, device: str | None = None) -> dict:
    from nanomesh.fit import fit

    card = fit(model, _device(device))
    if card is None:
        raise ToolError(f"NanoMesh can't size '{model}'. Use a family name (whisper-small, yolo11n, "
                        "mobilenet-v3-large, bge-small-en, qwen2.5-7b), a file, or search_models with a task.")
    return card.model_dump(exclude_none=True)


def job_status(job_id: str | None = None, wait_seconds: float = 0) -> dict:
    from nanomesh import jobs

    if job_id:
        # Waiting here saves the agent (and the user's credits) dozens of polls.
        job = jobs.get(job_id, wait_s=min(max(wait_seconds or 0, 0), MAX_WAIT_S))
        if job is None:
            raise ToolError(f"No job {job_id} (jobs are forgotten when the MCP server restarts; downloads resume "
                            "if started again).")
        return job.model_dump(exclude_none=True)
    return {"jobs": [j.model_dump(exclude_none=True) for j in jobs.all_jobs()]}


def cancel_job(job_id: str) -> dict:
    from nanomesh import jobs

    job = jobs.cancel(job_id)
    if job is None:
        raise ToolError(f"No job {job_id}.")
    return {"job_id": job_id, "status": "cancelling" if job.status == "running" else job.status}


def _server_summary(srv, state: str | None = None) -> dict:
    from nanomesh.serve import connect_snippets, health

    return {"model": srv.model, "base_url": srv.base_url, "port": srv.port, "context": srv.context,
            "threads": srv.threads, "state": state or health(srv.port) or "not responding", "pid": srv.pid,
            "log": srv.log, "how_to_connect": connect_snippets(srv)}


def start_model_server(path: str, port: int = 8080, context: int = 4096) -> dict:
    from nanomesh.serve import start
    from nanomesh.toolchain import ToolchainError

    f = Path(path).expanduser()
    if not f.is_file():
        raise ToolError(f"No such model file: {path}")
    # Measured best thread count from `nanomesh tune`, when there is one.
    threads = store.evidence(_local(), analyze(str(f))).best_threads
    try:
        srv, state = start(f, port=port, context=context, threads=threads, wait_s=45)
    except ToolchainError as e:
        raise ToolError(str(e)) from None
    out = _server_summary(srv, state)
    if state == "loading":
        out["next"] = "Still loading the model; check model_server_status in a few seconds."
    return out


def model_server_status() -> dict:
    from nanomesh.serve import running

    servers = running()
    return {"servers": [_server_summary(s) for s in servers]} if servers else {
        "servers": [], "note": "No NanoMesh model servers running. start_model_server(path) starts one."}


def stop_model_server(port: int | None = None) -> dict:
    from nanomesh.serve import stop

    stopped = stop(port)
    return {"stopped": [{"model": s.model, "port": s.port} for s in stopped]} if stopped else {
        "stopped": [], "note": "Nothing to stop."}


def analyze_project(path: str = ".", device: str | None = None, include_tests: bool = False) -> dict:
    from nanomesh.project import analyze_project as scan_project

    try:
        report = scan_project(Path(path), _device(device), include_tests)
    except ValueError as e:
        raise ToolError(str(e)) from None
    return report.model_dump(exclude_none=True)


def environment_doctor() -> dict:
    from nanomesh.doctor import inspect

    env = inspect(_local())
    return {"python": env.python, "virtualenv": env.in_virtualenv, "os": env.os, "packages": env.packages,
            "torch": env.torch, "tools": sorted(env.tools), "llama_cpp": env.llama_cpp,
            "findings": [f.model_dump(exclude_none=True) for f in env.findings if f.level != "ok"],
            "ok": [f.message for f in env.findings if f.level == "ok"]}


def training_plan(model: str, device: str | None = None, seq_len: int = 1024, batch: int = 1,
                  lora_rank: int = 16) -> dict:
    from nanomesh.training import train_plan

    return train_plan(_model(model), _device(device), seq_len=seq_len, batch=batch,
                      lora_rank=lora_rank).model_dump()


_DEVICE = {"type": "string", "description": "Device id from the NanoMesh database (e.g. 'redmi-14c-4gb', "
                                            "'rtx-3060-12gb'), or 'local' / omitted for this machine."}
_MODEL = {"type": "string", "description": "A model directory or .gguf/.safetensors path, a known name like "
                                           "'qwen2.5-7b', or a size like '7b'."}

TOOLS: dict[str, tuple[Callable[..., dict], str, dict]] = {
    "device_passport": (device_passport,
        "What this machine (or a device from the database) can run: hardware, compute class, memory budget and "
        "which model sizes fit, with speeds calibrated from real benchmarks where available.",
        {"device": _DEVICE}),
    "current_conditions": (current_conditions,
        "Live conditions affecting AI performance right now: battery/plugged in, power mode, CPU speed, "
        "temperature where readable, CPU load, free RAM, plus advice.", {}),
    "plan_model": (plan_model,
        "Which quantized variant (FP16/INT8/INT6/INT5/INT4/INT3/INT2 GGUF) of a model to run on a device: memory, "
        "speed and quality for each, the recommendation, and why others were ruled out. Uses measured "
        "benchmarks from this device when they exist.",
        {"model": _MODEL, "device": _DEVICE,
         "context": {"type": "integer", "description": "Context length in tokens (default 4096)."},
         "min_quality": {"type": "string", "description": "Minimum quality: a percentage like '95' or a tier "
                                                          "(lossless, high, good, fair, severe). Default 'good'."},
         "prefer": {"type": "string", "enum": ["balanced", "quality", "speed", "size"]}}),
    "list_local_models": (list_local_models,
        "Models already on this machine (Hugging Face cache, LM Studio, Ollama, ~/models and given folders), with "
        "size, quantization, whether each fits in memory, and the best variant for this device.",
        {"folders": {"type": "array", "items": {"type": "string"}, "description": "Extra folders to search."}}),
    "benchmark_results": (benchmark_results,
        "Speeds, memory and quality measured on this machine with `nanomesh benchmark`, `sustained` and `tune`.",
        {"model": {"type": "string", "description": "Filter by model name substring."}}),
    "environment_doctor": (environment_doctor,
        "Check the local AI toolchain: Python, PyTorch (and whether it can use the GPU), transformers, "
        "bitsandbytes, ONNX Runtime, llama.cpp, disk and memory, with fixes for problems found.", {}),
    "training_plan": (training_plan,
        "Memory needed to fine-tune a model on a device with full fine-tuning, LoRA and QLoRA, which fits, and "
        "what to change if nothing does.",
        {"model": _MODEL, "device": _DEVICE,
         "seq_len": {"type": "integer", "description": "Training sequence length (default 1024)."},
         "batch": {"type": "integer", "description": "Micro-batch size (default 1)."},
         "lora_rank": {"type": "integer", "description": "LoRA rank (default 16)."}}),
    "analyze_project": (analyze_project,
        "Call this first when asked what AI a project uses or whether it could run locally: it's faster and more "
        "complete than searching files. Finds calls to OpenAI, Anthropic, Gemini (incl. Firebase/Vertex AI), "
        "Mistral, Cohere, Groq and LangChain in Python, JS/TS, Kotlin, Java, Swift and Dart, whether each runs on "
        "a server, in a mobile app or on this machine; also local models (transformers, llama.cpp, Ollama), AI "
        "dependencies and model files, with file:line. For each cloud use: a local, self-hosted or on-device "
        "model sized for where it runs, which devices run it, the code change, and caveats. Reads files only; "
        "skips .env files and masks API keys.",
        {"path": {"type": "string", "description": "Project folder (the workspace root). Default: current folder."},
         "device": _DEVICE,
         "include_tests": {"type": "boolean", "description": "Also count calls in test files (default false)."}}),
    "fit_card": (fit_card,
        "The Fit Card of any model on this machine (or a device from the database): fits in memory?, speed in "
        "the task's unit (x real-time for speech, images/s for vision, sentences/s for embeddings, tok/s for text; "
        "measured, calibrated or estimated), quality, download size and data cost, battery, licence, a "
        "better-fitting model of the same family, and how to run it. Knows Whisper, YOLOv8/YOLO11, image "
        "classifiers, embedding models and text models.",
        {"model": {"type": "string", "description": "Model name, repository id or file: whisper-small, "
                                                    "openai/whisper-base, yolo11n, BAAI/bge-small-en-v1.5, "
                                                    "qwen2.5-7b, C:/models/x.onnx. A family name alone "
                                                    "(whisper, yolo11) picks the best size."},
         "device": _DEVICE}),
    "search_models": (search_models,
        "Search for models and size each one for this machine. Without task: GGUF text models on Hugging Face "
        "(which file to download, size, speed, quality); turn tasks into model names: coding -> 'qwen2.5 coder "
        "7b'. With task (speech, detection, classification, embeddings): models for that task from Hugging "
        "Face and/or Kaggle, each with its Fit Card, fitting fast-enough ones first; query may be empty.",
        {"query": {"type": "string", "description": "Words that must all appear in the model's name."},
         "task": {"type": "string", "enum": ["speech", "detection", "classification", "embeddings", "text"]},
         "source": {"type": "string", "enum": ["huggingface", "kaggle", "all"],
                    "description": "Where to search with a task (default huggingface; kaggle needs the user's "
                                   "Kaggle API token)."},
         "limit": {"type": "integer", "description": "Repositories to return (default 5, max 10)."},
         "context": {"type": "integer", "description": "Context length in tokens (default 4096)."},
         "min_quality": {"type": "string", "description": "Minimum quality: a percentage or a tier (default 'good')."},
         "prefer": {"type": "string", "enum": ["balanced", "quality", "speed", "size"]}}),
    "download_model": (download_model,
        "Download a GGUF model from Hugging Face into the models folder, resuming and verifying checksums. "
        "Runs as a job: returns job_id at once. Tell the user the download size and get their OK first.",
        {"repo": {"type": "string", "description": "Repository id, e.g. 'Qwen/Qwen2.5-1.5B-Instruct-GGUF'."},
         "file": {"type": "string", "description": "File name or format (e.g. 'Q4_K_M'); default: NanoMesh's "
                                                   "recommendation for this machine."}}),
    "benchmark_model": (benchmark_model,
        "Measure a model's speed and memory on this machine and save it, so plans and Fit Cards use real "
        "numbers: GGUF with llama.cpp, ONNX with ONNX Runtime, Whisper with whisper.cpp. Runs as a job; about "
        "a minute (steady-state GGUF runs 2-4 minutes).",
        {"path": {"type": "string", "description": "Path to a .gguf file, an .onnx model (vision, embeddings) or "
                                                   "a whisper.cpp ggml-*.bin."},
         "quick": {"type": "boolean", "description": "GGUF only: skip the warm-up (default true)."}}),
    "job_status": (job_status,
        "Progress and result of a download or benchmark job (all jobs if no id is given). Set wait_seconds to wait "
        "for the job to finish before answering, instead of calling this repeatedly.",
        {"job_id": {"type": "string"},
         "wait_seconds": {"type": "number", "description": f"Wait up to this long (max {MAX_WAIT_S}) for the job "
                                                           "to finish. Default 0: answer at once."}}),
    "cancel_job": (cancel_job, "Cancel a running download or benchmark job. Downloads resume if started again.",
        {"job_id": {"type": "string"}}),
    "start_model_server": (start_model_server,
        "Serve a GGUF model on this machine as an OpenAI-compatible API at http://127.0.0.1:<port>/v1 "
        "(llama.cpp's llama-server, local only, using this machine's measured best thread count). Returns the "
        "endpoint and ready-to-paste snippets for Python, JavaScript, curl and VS Code (Continue). Keeps running "
        "until stop_model_server.",
        {"path": {"type": "string", "description": "Path to a .gguf file (from list_local_models or "
                                                   "download_model)."},
         "port": {"type": "integer", "description": "Port (default 8080)."},
         "context": {"type": "integer", "description": "Context length in tokens (default 4096)."}}),
    "model_server_status": (model_server_status,
        "Model servers NanoMesh is running, their endpoints and how to connect.", {}),
    "stop_model_server": (stop_model_server, "Stop a model server NanoMesh started (all if no port is given).",
        {"port": {"type": "integer"}}),
}
REQUIRED = {"plan_model": ["model"], "training_plan": ["model"], "fit_card": ["model"],
            "download_model": ["repo"], "benchmark_model": ["path"], "cancel_job": ["job_id"],
            "start_model_server": ["path"]}
# Most tools only read this machine's state and NanoMesh's own data, so
# clients such as VS Code can ask the user less often. The rest act, and say so.
READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
ANNOTATIONS = {name: READ_ONLY for name in TOOLS} | {
    "search_models": READ_ONLY | {"openWorldHint": True},
    "download_model": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    "benchmark_model": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False,
                        "openWorldHint": False},
    "cancel_job": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    "start_model_server": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True,
                           "openWorldHint": False},
    "stop_model_server": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True,
                          "openWorldHint": False},
}


def tool_list() -> list[dict]:
    return [{"name": name, "description": desc,
             "inputSchema": {"type": "object", "properties": props, "required": REQUIRED.get(name, []),
                             "additionalProperties": False},
             "annotations": {"title": name.replace("_", " ").capitalize(), **ANNOTATIONS[name]}}
            for name, (_, desc, props) in TOOLS.items()]


def call_tool(name: str, arguments: dict | None) -> dict:
    if name not in TOOLS:
        return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}
    fn, _, props = TOOLS[name]
    args = {k: v for k, v in (arguments or {}).items() if k in props}
    try:
        # Tools must never write to stdout: it carries the protocol.
        with contextlib.redirect_stdout(sys.stderr):
            result = fn(**args)
    except (ToolError, TypeError, ValueError) as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}
    return {"content": [{"type": "text", "text": json.dumps(result, default=str)}], "structuredContent": result,
            "isError": False}


def handle(message: dict) -> dict | None:
    """Answer one JSON-RPC message; None for notifications."""
    method, msg_id, params = message.get("method"), message.get("id"), message.get("params") or {}
    if msg_id is None:
        return None  # notifications (e.g. notifications/initialized) need no reply

    def ok(result: Any) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    if method == "initialize":
        asked = params.get("protocolVersion")
        return ok({"protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                   "capabilities": {"tools": {"listChanged": False}},
                   "serverInfo": {"name": "nanomesh", "version": __version__}, "instructions": INSTRUCTIONS})
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": tool_list()})
    if method == "tools/call":
        return ok(call_tool(params.get("name", ""), params.get("arguments")))
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def serve(stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    if stdout is None:
        # stdout carries the protocol. Background jobs outlive a tool call's
        # redirect, so anything else printed from now on goes to stderr.
        stdout, sys.stdout = sys.stdout, sys.stderr
    for line in stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        else:
            reply = handle(message) if isinstance(message, dict) else {
                "jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid request"}}
        if reply is not None:
            stdout.write(json.dumps(reply, default=str) + "\n")
            stdout.flush()


def client_configs(command: str) -> dict[str, str]:
    """Ready-to-paste setup for the common MCP clients."""
    cmd = json.dumps(command)
    return {
        "Claude Code": f"claude mcp add nanomesh -- {command} mcp",
        "VS Code (.vscode/mcp.json)": '{\n  "servers": {\n    "nanomesh": {"type": "stdio", "command": ' + cmd +
                                      ', "args": ["mcp"]}\n  }\n}',
        "Cursor (~/.cursor/mcp.json) / Claude Desktop": '{\n  "mcpServers": {\n    "nanomesh": {"command": ' + cmd +
                                                        ', "args": ["mcp"]}\n  }\n}',
    }
