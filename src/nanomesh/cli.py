import json
import shlex
import shutil
from pathlib import Path

import typer
from rich.console import Console

from nanomesh import __version__
from nanomesh.devices import get_device, load_devices, search_devices
from nanomesh.hardware import DeviceProfile, scan_device
from nanomesh.model import analyze
from nanomesh.planner import FORMATS_BY_NAME, QUALITY_TIERS, Plan, Requirements, plan
from nanomesh.report import bench_table, device_passport, model_summary, plan_view
from nanomesh.toolchain import ToolchainError, benchmark_gguf, conversion_commands, find_toolchain, run

app = typer.Typer(help="NanoMesh — find the best way to run an AI model on specific hardware.",
                  no_args_is_help=True)
console = Console()

DeviceOpt = typer.Option("local", "--device", "-d", help="Device id from `nanomesh devices`, or 'local' to scan this machine.")


def _resolve_device(device_id: str) -> DeviceProfile:
    if device_id == "local":
        return scan_device()
    try:
        return get_device(device_id)
    except KeyError as e:
        console.print(f"[red]{e.args[0]}")
        raise typer.Exit(1)


def _analyze(spec: str):
    try:
        return analyze(spec)
    except ValueError as e:
        console.print(f"[red]{e}")
        raise typer.Exit(1)


def _requirements(context, min_quality, min_speed, ram, prefer) -> Requirements:
    if min_quality not in QUALITY_TIERS:
        console.print(f"[red]--min-quality must be one of: {', '.join(QUALITY_TIERS)}")
        raise typer.Exit(1)
    if prefer not in ("quality", "speed", "size"):
        console.print("[red]--prefer must be one of: quality, speed, size")
        raise typer.Exit(1)
    return Requirements(context=context, min_quality=min_quality, min_tokens_per_s=min_speed,
                        max_ram_gb=ram, prefer=prefer)


@app.command()
def version():
    """Show the NanoMesh version."""
    console.print(f"nanomesh {__version__}")


@app.command("scan-device")
def scan_device_cmd(
    as_json: bool = typer.Option(False, "--json", help="Print the raw profile as JSON."),
    save: Path = typer.Option(None, help="Save the profile to a JSON file."),
):
    """Profile this machine and show its Device Passport."""
    device = scan_device()
    data = device.model_dump(exclude_none=True)
    if save:
        save.write_text(json.dumps(data, indent=2))
        console.print(f"[green]Saved device profile to {save}")
    if as_json:
        console.print_json(data=data)
    else:
        console.print(device_passport(device))


@app.command()
def devices(query: str = typer.Argument(None, help="Filter, e.g. 'thinkpad' or 'android'.")):
    """List devices in the NanoMesh device database."""
    found = search_devices(query) if query else list(load_devices().values())
    if not found:
        console.print(f"[yellow]No devices match '{query}'.")
        raise typer.Exit(1)
    for d in found:
        console.print(f"[bold cyan]{d.id:<24}[/] {d.name:<38} [dim]{d.ram_gb:g} GB {d.kind}", no_wrap=True)


@app.command()
def device(device_id: str = typer.Argument(..., help="Device id, name fragment, or 'local'.")):
    """Show the Device Passport for a known device."""
    console.print(device_passport(_resolve_device(device_id)))


@app.command("analyze")
def analyze_cmd(model: str = typer.Argument(..., help="Model dir, .safetensors/.gguf file, known name, or size like '7b'.")):
    """Inspect a model's size and architecture without loading it."""
    console.print(model_summary(_analyze(model)))


@app.command("plan")
def plan_cmd(
    model: str = typer.Argument(..., help="Model dir, weights file, known name (e.g. qwen2.5-7b), or size like '7b'."),
    device_id: str = DeviceOpt,
    context: int = typer.Option(4096, help="Context length to budget KV cache for."),
    min_quality: str = typer.Option("good", help=f"Minimum quality tier: {', '.join(QUALITY_TIERS)}."),
    min_speed: float = typer.Option(None, help="Minimum generation speed in tokens/s."),
    ram: float = typer.Option(None, help="Cap the memory the model may use, in GB."),
    prefer: str = typer.Option("quality", help="Tie-breaker among valid variants: quality, speed, size."),
    as_json: bool = typer.Option(False, "--json", help="Print the plan as JSON."),
):
    """Estimate every quantized variant on a device and recommend the best one."""
    p = plan(_analyze(model), _resolve_device(device_id),
             _requirements(context, min_quality, min_speed, ram, prefer))
    if as_json:
        console.print_json(data=p.model_dump(exclude_none=True))
    else:
        console.print(plan_view(p))
    if p.recommended is None:
        raise typer.Exit(2)


def _formats_to_build(p: Plan, quantization: str) -> list[str]:
    if quantization != "auto":
        names = []
        for q in quantization.split(","):
            fmt = FORMATS_BY_NAME.get(q.strip().lower())
            if not fmt:
                console.print(f"[red]Unknown quantization '{q}'. Use e.g. int4, int8, Q4_K_M, Q8_0.")
                raise typer.Exit(1)
            names.append(fmt.name)
        return names
    if p.recommended is None:
        return []
    # The winner plus the next smaller Pareto point, so users can A/B quality on-device.
    pareto = [v.format.name for v in p.variants if v.pareto and v.fits]
    idx = pareto.index(p.recommended) if p.recommended in pareto else -1
    extra = pareto[idx + 1: idx + 2] if idx >= 0 else []
    return [p.recommended, *extra]


@app.command()
def optimize(
    model_dir: Path = typer.Argument(..., help="Hugging Face model directory (config.json + safetensors)."),
    device_id: str = DeviceOpt,
    quantization: str = typer.Option("auto", "--quantization", "-q", help="'auto' or comma list like int4,int8."),
    out: Path = typer.Option(None, "--out", "-o", help="Output package directory."),
    context: int = typer.Option(4096),
    min_quality: str = typer.Option("good"),
    min_speed: float = typer.Option(None),
    ram: float = typer.Option(None),
    prefer: str = typer.Option("quality"),
    dry_run: bool = typer.Option(False, help="Only write the plan and print the conversion commands."),
    bench: bool = typer.Option(True, help="Benchmark built variants with llama-bench if available."),
):
    """Plan, convert and quantize a model into a deployment-ready package."""
    if not model_dir.is_dir():
        console.print(f"[red]{model_dir} is not a model directory.")
        raise typer.Exit(1)
    device = _resolve_device(device_id)
    p = plan(_analyze(str(model_dir)), device, _requirements(context, min_quality, min_speed, ram, prefer))
    console.print(plan_view(p))
    formats = _formats_to_build(p, quantization)
    if not formats:
        console.print("[red]Nothing to build: no variant meets the requirements.")
        raise typer.Exit(2)

    out = out or Path(f"{model_dir.name}-nanomesh-{device.id}")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        if (model_dir / name).exists():
            shutil.copy2(model_dir / name, out / name)
    (out / "plan.json").write_text(json.dumps(p.model_dump(exclude_none=True), indent=2))

    tc = find_toolchain()
    cmds = conversion_commands(tc, model_dir, out, formats)
    results = []
    if dry_run or not tc.can_convert:
        if not dry_run:
            console.print("\n[yellow]llama.cpp not found (set NANOMESH_LLAMA_CPP to your llama.cpp "
                          "checkout). Run these commands to build the variants:")
        else:
            console.print("\n[bold]Conversion commands:")
        for cmd in cmds:
            console.print(f"  {shlex.join(cmd)}", soft_wrap=True)
    else:
        with console.status("Converting to GGUF and quantizing…"):
            try:
                for cmd in cmds:
                    run(cmd)
            except ToolchainError as e:
                console.print(f"[red]{e}")
                raise typer.Exit(1)
        if "F16" not in formats:
            (out / "model-F16.gguf").unlink(missing_ok=True)
        if bench and tc.bench:
            for fmt in formats:
                with console.status(f"Benchmarking {fmt}…"):
                    results.append(benchmark_gguf(tc, out / f"model-{fmt}.gguf"))
            (out / "benchmark.json").write_text(json.dumps([r.model_dump() for r in results], indent=2))
            console.print(bench_table(results))

    (out / "README.md").write_text(_package_readme(p, formats, results))
    console.print(f"\n[green]Package written to {out}/")


def _package_readme(p: Plan, formats: list[str], results) -> str:
    lines = [
        f"# {p.model.name} — optimized by NanoMesh", "",
        f"Target device: **{p.device.name}**  ", f"Recommended variant: **{p.recommended}**", "",
        "| Variant | Est. memory | Est. speed | Quality |", "|---|---|---|---|",
    ]
    for v in p.variants:
        if v.format.name in formats:
            speed = f"~{v.tokens_per_s} tok/s" if v.tokens_per_s else "—"
            lines.append(f"| `model-{v.format.name}.gguf` | {v.total_memory_gb:.1f} GB | {speed} | {v.format.quality} |")
    if results:
        lines += ["", "## Measured", "", "| File | Generate | Peak RAM |", "|---|---|---|"]
        for r in results:
            lines.append(f"| `{Path(r.model_file).name}` | {r.gen_tokens_per_s} tok/s | {r.peak_rss_gb} GB |")
    lines += ["", "## Run", "", "```sh", f"llama-cli -m model-{formats[0]}.gguf -c {p.requirements.context}", "```", "",
              "Estimates come from `plan.json`; see `benchmark.json` for measurements when present.", ""]
    return "\n".join(lines)


@app.command()
def benchmark(
    path: Path = typer.Argument(..., help="A .gguf file or a directory of .gguf files."),
    threads: int = typer.Option(None, help="CPU threads for llama-bench."),
    save: Path = typer.Option(None, help="Write results to this JSON file."),
):
    """Measure real speed and memory of GGUF models on this machine."""
    files = sorted(path.glob("*.gguf")) if path.is_dir() else [path]
    if not files or not all(f.is_file() for f in files):
        console.print(f"[red]No .gguf files found at {path}")
        raise typer.Exit(1)
    tc = find_toolchain()
    results = []
    try:
        for f in files:
            with console.status(f"Benchmarking {f.name}…"):
                results.append(benchmark_gguf(tc, f, threads))
    except ToolchainError as e:
        console.print(f"[red]{e}")
        raise typer.Exit(1)
    console.print(bench_table(results))
    if save:
        save.write_text(json.dumps([r.model_dump() for r in results], indent=2))


if __name__ == "__main__":
    app()
