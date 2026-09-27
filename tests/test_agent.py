"""doctor, models, train-plan and the MCP server: the agent-facing layer."""

import io
import json
import os
import subprocess
import sys

import pytest

from nanomesh import mcp
from nanomesh.devices import get_device
from nanomesh.doctor import Environment, diagnose
from nanomesh.model import analyze
from nanomesh.training import train_plan


# ---- doctor ----

def _env(**kw):
    base = dict(python="3.11.9", python_path="python", in_virtualenv=True, os="Windows 11", packages={},
                tools={}, llama_cpp={"llama-bench": "C:/llama.cpp/llama-bench.exe",
                                     "llama-perplexity": "C:/llama.cpp/llama-perplexity.exe",
                                     "llama-quantize": "x", "convert_hf_to_gguf.py": "x"},
                disk_free_gb=100, findings=[])
    return Environment(**(base | kw))


def _levels(findings):
    return {(f.topic, f.level) for f in findings}


def test_doctor_flags_cuda_torch_on_a_cpu_only_laptop():
    elitebook = get_device("hp-elitebook-840-g6")
    env = _env(packages={"torch": "2.5.1"}, torch={"version": "2.5.1+cu121", "cuda_build": "12.1", "cuda": False})
    msgs = " ".join(f.message for f in diagnose(env, elitebook))
    assert "CUDA build" in msgs and "no NVIDIA GPU" in msgs


def test_doctor_flags_gpu_that_torch_cannot_use():
    env = _env(packages={"torch": "2.5.1"}, torch={"version": "2.5.1+cpu", "cuda_build": None, "cuda": False})
    found = diagnose(env, get_device("rtx-3060-12gb"))
    assert ("PyTorch", "fail") in _levels(found)


@pytest.mark.parametrize("packages, topic, level", [
    ({"transformers": "4.46"}, "Transformers", "fail"),  # no torch to run it
    ({"bitsandbytes": "0.44", "torch": "2.5"}, "bitsandbytes", "warn"),  # no NVIDIA GPU
    ({"onnxruntime": "1.19", "onnxruntime-gpu": "1.19"}, "ONNX Runtime", "warn"),
])
def test_doctor_package_problems(packages, topic, level):
    found = diagnose(_env(packages=packages), get_device("hp-elitebook-840-g6"))
    assert (topic, level) in _levels(found)


def test_doctor_environment_basics():
    found = diagnose(_env(in_virtualenv=False, disk_free_gb=6, python="3.9.7",
                          llama_cpp={"llama-bench": None}), None)
    assert {("Python", "fail"), ("Python", "warn"), ("Disk", "warn"), ("llama.cpp", "warn")} <= _levels(found)
    assert not [f for f in diagnose(_env(), None) if f.level != "ok"]


# ---- models ----

def test_models_found_in_hf_cache_lm_studio_and_ollama(tmp_path, monkeypatch):
    from conftest import write_gguf, write_safetensors

    from nanomesh.discover import find_models

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.delenv("NANOMESH_MODEL_DIRS", raising=False)

    snap = home / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"], "hidden_size": 64,
                                                  "num_hidden_layers": 2, "num_attention_heads": 4}))
    write_safetensors(snap / "model.safetensors", {"w": ("BF16", [64, 64])})

    lms = home / ".lmstudio/models/lmstudio-community/tiny"
    lms.mkdir(parents=True)
    write_gguf(lms / "tiny-Q4_K_M.gguf", file_type=15)
    write_gguf(lms / "big-Q8_0-00001-of-00002.gguf", file_type=7)
    write_gguf(lms / "big-Q8_0-00002-of-00002.gguf", file_type=7)
    write_gguf(lms / "half-Q4_K_M-00001-of-00003.gguf", file_type=15)  # download interrupted

    ollama = home / ".ollama/models"
    (ollama / "blobs").mkdir(parents=True)
    write_gguf(ollama / "blobs" / "sha256-deadbeef", file_type=15)
    manifest = ollama / "manifests/registry.ollama.ai/library/llama3.2/1b"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.model",
                                                "digest": "sha256:deadbeef"}]}))

    found = {m.name: m for m in find_models(get_device("hp-elitebook-840-g6"), [])}
    assert set(found) == {"Qwen/Qwen2.5-0.5B", "tiny-Q4_K_M", "big-Q8_0", "half-Q4_K_M", "llama3.2:1b"}
    assert (found["big-Q8_0"].parts, found["big-Q8_0"].missing_parts) == (2, 0)
    assert (found["half-Q4_K_M"].parts, found["half-Q4_K_M"].missing_parts) == (3, 2)
    assert found["tiny-Q4_K_M"].parts == 1
    assert found["Qwen/Qwen2.5-0.5B"].source == "huggingface" and found["Qwen/Qwen2.5-0.5B"].kind == "safetensors"
    assert found["tiny-Q4_K_M"].format == "Q4_K_M" and found["tiny-Q4_K_M"].fits
    assert found["llama3.2:1b"].source == "ollama" and found["llama3.2:1b"].kind == "gguf"


# ---- train-plan ----

def test_training_memory_matches_known_figures():
    # Widely reported: Llama 3.1 8B needs ~120 GB to fully fine-tune, ~16-18 GB with LoRA, ~6 GB with QLoRA.
    tp = train_plan(analyze("llama-3.1-8b"), get_device("rtx-3060-12gb"))
    full, lora, qlora = (o.total_gb for o in tp.options)
    assert 110 < full < 135 and 14 < lora < 20 and 5 < qlora < 8
    assert tp.recommended == "qlora"  # 12 GB card


def test_qlora_needs_nvidia_and_cpu_training_is_flagged():
    tp = train_plan(analyze("qwen2.5-1.5b"), get_device("hp-elitebook-840-g6"))
    qlora = next(o for o in tp.options if o.method == "qlora")
    assert not qlora.available and not qlora.fits
    assert tp.recommended == "lora" and tp.placement == "CPU"
    assert any("slower than a GPU" in a for a in tp.advice)


def test_nothing_fits_suggests_a_gpu_size_and_shorter_sequences():
    tp = train_plan(analyze("qwen2.5-7b"), get_device("redmi-14c-4gb"), seq_len=4096, batch=4)
    assert tp.recommended is None
    text = " ".join(tp.advice)
    assert "NVIDIA GPU" in text and "--seq-len 512" in text


def test_unknown_architecture_is_estimated():
    tp = train_plan(analyze("13b"), get_device("rtx-4090-24gb"))
    assert tp.shape_estimated and tp.recommended in ("lora", "qlora")


# ---- MCP ----

def _rpc(*messages):
    out = io.StringIO()
    mcp.serve(io.StringIO("".join(json.dumps(m) + "\n" for m in messages)), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_mcp_handshake_and_tool_list():
    init, tools = _rpc(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},  # notification: no reply
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert init["result"]["serverInfo"]["name"] == "nanomesh"
    names = {t["name"] for t in tools["result"]["tools"]}
    assert names == {"device_passport", "current_conditions", "plan_model", "list_local_models",
                     "benchmark_results", "environment_doctor", "training_plan"}
    assert all(t["inputSchema"]["type"] == "object" for t in tools["result"]["tools"])
    # Nothing changes the machine: clients may skip the confirmation prompt.
    assert all(t["annotations"]["readOnlyHint"] and not t["annotations"]["destructiveHint"]
               for t in tools["result"]["tools"])


def test_mcp_unknown_version_gets_latest():
    [init] = _rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}})
    assert init["result"]["protocolVersion"] == mcp.PROTOCOL_VERSIONS[0]


def test_mcp_plan_tool_for_a_database_device():
    [reply] = _rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "plan_model", "arguments": {"model": "llama-3.1-8b", "device": "rtx-3060-12gb"}}})
    result = reply["result"]
    assert not result["isError"]
    body = json.loads(result["content"][0]["text"])
    assert body == result["structuredContent"]
    assert body["recommended"] == "Q8_0" and body["device"].startswith("Desktop with NVIDIA RTX 3060")


def test_mcp_errors_are_reported_not_raised():
    bad_tool, bad_args, bad_method, garbage = _rpc(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "rm_rf", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "plan_model", "arguments": {"model": "nope"}}},
        {"jsonrpc": "2.0", "id": 6, "method": "resources/list"},
        "not an object",
    )
    assert bad_tool["result"]["isError"] and bad_args["result"]["isError"]
    assert "Can't interpret" in bad_args["result"]["content"][0]["text"]
    assert bad_method["error"]["code"] == -32601
    assert garbage["error"]["code"] == -32600


def test_mcp_over_a_real_stdio_subprocess(tmp_path):
    # What Claude Code / Cursor / VS Code actually do: spawn `nanomesh mcp` and talk JSON-RPC over pipes.
    env = {**os.environ, "NANOMESH_HOME": str(tmp_path), "PYTHONIOENCODING": "cp1252"}
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "training_plan", "arguments": {"model": "qwen2.5-1.5b", "device": "hp-elitebook-840-g6"}}}]
    proc = subprocess.run([sys.executable, "-m", "nanomesh.cli", "mcp"], input="".join(json.dumps(m) + "\n" for m in msgs),
                          capture_output=True, text=True, env=env, timeout=60, encoding="utf-8")
    lines = proc.stdout.splitlines()
    assert len(lines) == 2, proc.stdout + proc.stderr  # stdout carries only protocol messages
    reply = json.loads(lines[1])
    assert json.loads(reply["result"]["content"][0]["text"])["recommended"] == "lora"


def test_mcp_config_snippets():
    snippets = mcp.client_configs("C:\\Users\\hp\\NanoMesh\\.venv\\Scripts\\nanomesh.exe")
    vscode = json.loads(snippets["VS Code (.vscode/mcp.json)"])
    assert vscode["servers"]["nanomesh"]["args"] == ["mcp"]
    assert json.loads(snippets["Cursor (~/.cursor/mcp.json) / Claude Desktop"])["mcpServers"]["nanomesh"]["command"].endswith("nanomesh.exe")
    assert snippets["Claude Code"].startswith("claude mcp add nanomesh -- ")
