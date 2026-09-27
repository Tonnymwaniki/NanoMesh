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
from nanomesh.planner import QUALITY_TIERS, Plan, Requirements, plan

PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]
INSTRUCTIONS = (
    "NanoMesh knows this developer's actual machine: exact device model, live power/thermal conditions, models "
    "on disk and benchmarks measured here. Use it before recommending local models, quantization, runtimes or "
    "training setups: device_passport for what the machine can run, plan_model for which variant of a model "
    "to use (with measured speeds where available), list_local_models for models already downloaded, "
    "environment_doctor for installation problems, training_plan before fine-tuning."
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
    return {"conditions": dump_conditions(c), "advice": condition_advice(c)}


def plan_model(model: str, device: str | None = None, context: int = 4096, min_quality: str = "good",
               prefer: str = "balanced") -> dict:
    info, dev = _model(model), _device(device)
    try:
        pct = float(str(min_quality).rstrip("%"))
        req = Requirements(context=context, min_quality_pct=pct, prefer=prefer)
    except ValueError:
        if min_quality not in QUALITY_TIERS:
            raise ToolError(f"min_quality must be a percentage or one of {QUALITY_TIERS}") from None
        req = Requirements(context=context, min_quality=min_quality, prefer=prefer)
    p = plan(info, dev, req, store.evidence(dev, info))
    if dev.is_local:
        p.advice += condition_advice(read_conditions())
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
}
REQUIRED = {"plan_model": ["model"], "training_plan": ["model"]}
# Every tool only reads this machine's state and NanoMesh's own data, so
# clients such as VS Code can ask the user less often.
ANNOTATIONS = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}


def tool_list() -> list[dict]:
    return [{"name": name, "description": desc,
             "inputSchema": {"type": "object", "properties": props, "required": REQUIRED.get(name, []),
                             "additionalProperties": False},
             "annotations": {"title": name.replace("_", " ").capitalize(), **ANNOTATIONS}}
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
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
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
