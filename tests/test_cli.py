import json

from typer.testing import CliRunner

from nanomesh.cli import app
from nanomesh.toolchain import parse_llama_bench

runner = CliRunner()


def test_plan_json():
    result = runner.invoke(app, ["plan", "qwen2.5-1.5b", "-d", "raspberry-pi-5-8gb", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["recommended"] == "Q8_0"


def test_plan_exit_code_when_nothing_fits():
    result = runner.invoke(app, ["plan", "70b", "-d", "redmi-14c-4gb"])
    assert result.exit_code == 2


def test_scan_device_json():
    result = runner.invoke(app, ["scan-device", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["ram_gb"] > 0


def test_optimize_dry_run_writes_package(tiny_model, tmp_path, monkeypatch):
    monkeypatch.delenv("NANOMESH_LLAMA_CPP", raising=False)
    out = tmp_path / "pkg"
    result = runner.invoke(app, ["optimize", str(tiny_model), "-d", "redmi-14c-4gb",
                                 "-q", "int4,int8", "-o", str(out), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert {"plan.json", "README.md", "config.json", "tokenizer.json"} <= {p.name for p in out.iterdir()}
    assert "llama-quantize" in result.output and "Q4_K_M" in result.output
    assert "model-Q8_0.gguf" in (out / "README.md").read_text()


def test_parse_llama_bench(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"\0" * 1024)
    out = json.dumps([
        {"n_prompt": 512, "n_gen": 0, "avg_ts": 101.234, "backends": "CPU", "n_threads": 4},
        {"n_prompt": 0, "n_gen": 128, "avg_ts": 12.345, "backends": "CPU", "n_threads": 4},
    ])
    r = parse_llama_bench(out, f)
    assert r.prompt_tokens_per_s == 101.23
    assert r.gen_tokens_per_s == 12.35
    assert r.threads == 4
