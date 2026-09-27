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


FAKE_BENCH = """#!{python}
import json, sys
m = sys.argv[sys.argv.index("-m") + 1]
speed = 20.0 if "Q4" in m else 10.0
print(json.dumps([{{"n_prompt": 512, "n_gen": 0, "avg_ts": speed * 10, "backends": "CPU", "n_threads": 4}},
                  {{"n_prompt": 0, "n_gen": 128, "avg_ts": speed, "backends": "CPU", "n_threads": 4}}]))
"""
FAKE_PPL = """#!{python}
import sys
m = sys.argv[sys.argv.index("-m") + 1]
print("some log line", file=sys.stderr)
print("Final estimate: PPL = %.4f +/- 0.1" % (10.5 if "Q4" in m else 10.0), file=sys.stderr)
"""


def _fake_llama_cpp(tmp_path):
    import os
    import stat
    import sys

    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    bin_dir.mkdir(parents=True)
    for name, src in (("llama-bench", FAKE_BENCH), ("llama-perplexity", FAKE_PPL)):
        f = bin_dir / name
        f.write_text(src.format(python=sys.executable))
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return os.pathsep.join([str(tmp_path / "llama.cpp")])


def test_benchmark_measures_quality_and_feeds_plan(tmp_path, monkeypatch):
    import sys

    import pytest

    if sys.platform == "win32":
        pytest.skip("fake llama.cpp binaries are POSIX scripts")
    from conftest import write_gguf

    monkeypatch.setenv("NANOMESH_LLAMA_CPP", _fake_llama_cpp(tmp_path))
    models = tmp_path / "models"
    models.mkdir()
    write_gguf(models / "model-F16.gguf", file_type=1)
    write_gguf(models / "model-Q4_K_M.gguf", file_type=15)

    result = runner.invoke(app, ["benchmark", str(models), "--quick"])
    assert result.exit_code == 0, result.output

    rows = json.loads(runner.invoke(app, ["results", "--json"]).output)
    by_fmt = {r["format"]: r for r in rows}
    assert by_fmt["F16"]["quality_pct"] == 100.0
    assert by_fmt["Q4_K_M"]["quality_pct"] == pytest.approx(95.24, abs=0.01)
    assert by_fmt["Q4_K_M"]["gen_tokens_per_s"] == 20.0
    assert by_fmt["Q4_K_M"]["reference_format"] == "F16"

    plan = json.loads(runner.invoke(app, ["plan", str(models / "model-F16.gguf"), "--json"]).output)
    q4 = next(v for v in plan["variants"] if v["format"]["name"] == "Q4_K_M")
    assert q4["speed_source"] == "measured" and q4["quality_measured"] is True


def test_min_quality_accepts_percentage():
    result = runner.invoke(app, ["plan", "qwen2.5-7b", "-d", "rtx-3060-12gb", "--min-quality", "99.9", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["recommended"] == "Q8_0"
    bad = runner.invoke(app, ["plan", "7b", "--min-quality", "amazing"])
    assert bad.exit_code == 1


def test_parse_perplexity():
    from nanomesh.toolchain import parse_perplexity

    assert parse_perplexity("...\nFinal estimate: PPL = 7.1234 +/- 0.05123\n") == 7.1234


def test_redirected_output_survives_a_legacy_windows_encoding(tmp_path):
    # `nanomesh plan > out.txt` in PowerShell used to crash: stdout fell back to cp1252.
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "NANOMESH_HOME": str(tmp_path)}
    out = subprocess.run([sys.executable, "-m", "nanomesh.cli", "plan", "qwen2.5-1.5b", "-d", "hp-elitebook-840-g6"],
                         capture_output=True, env=env)
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    assert "🏆" in out.stdout.decode("utf-8")


def test_benchmark_treats_split_gguf_as_one_model(tmp_path, monkeypatch):
    import sys

    import pytest

    if sys.platform == "win32":
        pytest.skip("fake llama.cpp binaries are POSIX scripts")
    from conftest import write_gguf

    from nanomesh.model import analyze

    monkeypatch.setenv("NANOMESH_LLAMA_CPP", _fake_llama_cpp(tmp_path))
    models = tmp_path / "qwen7b"
    models.mkdir()
    write_gguf(models / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", file_type=15)
    write_gguf(models / "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf", file_type=15,
               tensors=(("blk.1.attn_q.weight", [64, 64]),))

    # Either part stands for the whole model.
    whole = analyze(str(models / "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"))
    assert whole.params == 64 * 100 + 64 * 64 * 2

    result = runner.invoke(app, ["benchmark", str(models), "--no-quality", "--quick"])
    assert result.exit_code == 0, result.output
    rows = json.loads(runner.invoke(app, ["results", "--json"]).output)
    assert len(rows) == 1
    assert rows[0]["model_params"] == whole.params

    (models / "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf").unlink()
    missing = runner.invoke(app, ["benchmark", str(models), "--no-quality", "--quick"])
    assert missing.exit_code == 1 and "missing" in missing.output


def test_benchmark_is_steady_state_by_default(tmp_path, monkeypatch):
    import sys

    import pytest

    if sys.platform == "win32":
        pytest.skip("fake llama.cpp binaries are POSIX scripts")
    from conftest import write_gguf

    monkeypatch.setenv("NANOMESH_LLAMA_CPP", _fake_llama_cpp(tmp_path))
    models = tmp_path / "models"
    models.mkdir()
    write_gguf(models / "model-Q4_K_M.gguf", file_type=15)
    write_gguf(models / "model-Q8_0.gguf", file_type=7)
    result = runner.invoke(app, ["benchmark", str(models), "--no-quality", "--warmup", "0.3"])
    assert result.exit_code == 0, result.output
    assert "Steady-state mode" in result.output
    rows = json.loads(runner.invoke(app, ["results", "--json"]).output)
    assert all(r["steady"] and r["warmup_s"] >= 0.3 for r in rows)
    # Only the first file starts cold, so only it shows a cold-start speed.
    assert rows[0]["burst_tokens_per_s"] and rows[1]["burst_tokens_per_s"] is None
