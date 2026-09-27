import json
import shlex
import shutil
import sys
from pathlib import Path

import typer
from rich.console import Console

from nanomesh import __version__
from nanomesh.devices import get_device, load_devices, recognise, search_devices
from nanomesh.hardware import DeviceProfile, scan_device
from nanomesh.model import analyze, gguf_parts, gguf_size
from nanomesh.planner import FORMATS_BY_NAME, QUALITY_TIERS, Plan, Requirements, free_ram_warning, plan
from nanomesh import results as store
from nanomesh.evaluate import evaluate
from nanomesh.report import bench_table, device_passport, model_summary, plan_view, results_table
from nanomesh.toolchain import ToolchainError, conversion_commands, find_toolchain, run

# Windows falls back to cp1252 when output is redirected (`nanomesh plan > out.txt`),
# which can't encode the emoji and box characters in reports. Always emit UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

app = typer.Typer(help="NanoMesh — find the best way to run an AI model on specific hardware.",
                  no_args_is_help=True)
console = Console()

DeviceOpt = typer.Option("local", "--device", "-d", help="Device id from `nanomesh devices`, or 'local' to scan this machine.")


def _resolve_device(device_id: str) -> DeviceProfile:
    if device_id == "local":
        return recognise(scan_device())
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
    pct = None
    try:
        pct = float(min_quality.rstrip("%"))
    except ValueError:
        if min_quality not in QUALITY_TIERS:
            console.print(f"[red]--min-quality must be a percentage (e.g. 95) or one of: {', '.join(QUALITY_TIERS)}")
            raise typer.Exit(1)
    if pct is not None:
        return Requirements(context=context, min_quality_pct=pct, min_tokens_per_s=min_speed,
                            max_ram_gb=ram, prefer=prefer)
    if prefer not in ("balanced", "quality", "speed", "size"):
        console.print("[red]--prefer must be one of: balanced, quality, speed, size")
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
    device = recognise(scan_device())
    data = device.model_dump(exclude_none=True)
    if save:
        save.write_text(json.dumps(data, indent=2), encoding="utf-8")
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


QualityOpt = typer.Option("good", help=f"Minimum quality: a percentage like 95, or a tier ({', '.join(QUALITY_TIERS)}).")


@app.command("plan")
def plan_cmd(
    model: str = typer.Argument(..., help="Model dir, weights file, known name (e.g. qwen2.5-7b), or size like '7b'."),
    device_id: str = DeviceOpt,
    context: int = typer.Option(4096, help="Context length to budget KV cache for."),
    min_quality: str = QualityOpt,
    min_speed: float = typer.Option(None, help="Minimum generation speed in tokens/s."),
    ram: float = typer.Option(None, help="Cap the memory the model may use, in GB."),
    prefer: str = typer.Option("balanced", help="balanced (best quality at a usable speed), quality, speed or size."),
    as_json: bool = typer.Option(False, "--json", help="Print the plan as JSON."),
):
    """Recommend the best variant of a model for a device, using measurements where available."""
    info, device = _analyze(model), _resolve_device(device_id)
    p = plan(info, device, _requirements(context, min_quality, min_speed, ram, prefer),
             store.evidence(device, info))
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


def _reference_format(p: Plan, formats: list[str]) -> str:
    """F16 if it fits in memory, else Q8_0 (near-lossless) as the quality yardstick."""
    f16 = next(v for v in p.variants if v.format.name == "F16")
    return "F16" if f16.fits or "Q8_0" in formats else "Q8_0"


@app.command()
def optimize(
    model_dir: Path = typer.Argument(..., help="Hugging Face model directory (config.json + safetensors)."),
    device_id: str = DeviceOpt,
    quantization: str = typer.Option("auto", "--quantization", "-q", help="'auto' or comma list like int4,int8."),
    out: Path = typer.Option(None, "--out", "-o", help="Output package directory."),
    context: int = typer.Option(4096),
    min_quality: str = QualityOpt,
    min_speed: float = typer.Option(None),
    ram: float = typer.Option(None),
    prefer: str = typer.Option("balanced"),
    dry_run: bool = typer.Option(False, help="Only write the plan and print the conversion commands."),
    bench: bool = typer.Option(True, help="Benchmark built variants (needs llama-bench)."),
    quality: bool = typer.Option(True, help="Measure quality loss vs the original (needs llama-perplexity)."),
):
    """Plan, convert and quantize a model into a deployment-ready package."""
    if not model_dir.is_dir():
        console.print(f"[red]{model_dir} is not a model directory.")
        raise typer.Exit(1)
    info, device = _analyze(str(model_dir)), _resolve_device(device_id)
    p = plan(info, device, _requirements(context, min_quality, min_speed, ram, prefer), store.evidence(device, info))
    console.print(plan_view(p))
    formats = _formats_to_build(p, quantization)
    if not formats:
        console.print("[red]Nothing to build: no variant meets the requirements.")
        raise typer.Exit(2)

    out = out or Path(f"{model_dir.name}-nanomesh-{device.matched_id or device.id}")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        if (model_dir / name).exists():
            shutil.copy2(model_dir / name, out / name)
    (out / "plan.json").write_text(json.dumps(p.model_dump(exclude_none=True), indent=2), encoding="utf-8")

    tc = find_toolchain()
    measure_quality = quality and tc.perplexity is not None
    # Build the quality reference too; it's deleted afterwards unless requested.
    reference = _reference_format(p, formats) if measure_quality else None
    to_build = formats + ([reference] if reference and reference not in formats else [])
    cmds = conversion_commands(tc, model_dir, out, to_build)
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
        if bench and tc.bench:
            files = [out / f"model-{f}.gguf" for f in to_build]
            results = _evaluate(tc, files, device, measure_quality, out / f"model-{reference}.gguf" if reference else None)
            results = [r for r in results if r.format in formats]
            (out / "benchmark.json").write_text(json.dumps(store.rows(results), indent=2), encoding="utf-8")
            console.print(bench_table(results))
        for extra in {"F16", *to_build} - set(formats):
            (out / f"model-{extra}.gguf").unlink(missing_ok=True)

    (out / "README.md").write_text(_package_readme(p, formats, results), encoding="utf-8")
    console.print(f"\n[green]Package written to {out}/")


def _evaluate(tc, files, device, quality, reference=None, eval_text=None, threads=None) -> list[store.Result]:
    with console.status("Measuring…") as status:
        try:
            results = evaluate(tc, files, device, quality=quality, reference=reference, eval_text=eval_text,
                               threads=threads, log=status.update)
        except ToolchainError as e:
            console.print(f"[red]{e}")
            raise typer.Exit(1)
    path = store.save(results)
    console.print(f"[dim]Saved {len(results)} result(s) to {path}; `nanomesh plan` will use them.")
    return results


def _package_readme(p: Plan, formats: list[str], results: list[store.Result]) -> str:
    lines = [
        f"# {p.model.name} — optimized by NanoMesh", "",
        f"Target device: **{p.device.name}**  ", f"Recommended variant: **{p.recommended}**", "",
        "| Variant | Est. memory | Est. speed | Quality |", "|---|---|---|---|",
    ]
    for v in p.variants:
        if v.format.name in formats:
            speed = f"~{v.tokens_per_s} tok/s" if v.tokens_per_s else "—"
            lines.append(f"| `model-{v.format.name}.gguf` | {v.total_memory_gb:.1f} GB | {speed} | {v.quality} ({v.quality_pct:g}%) |")
    if results:
        lines += ["", "## Measured on this machine", "", "| File | Generate | Peak RAM | Quality |", "|---|---|---|---|"]
        for r in results:
            ram = f"{r.peak_rss_gb:.2f} GB" if r.peak_rss_gb is not None else "—"
            q = f"{r.quality_pct:g}% of {r.reference_format}" if r.quality_pct is not None else "—"
            lines.append(f"| `model-{r.format}.gguf` | {r.gen_tokens_per_s} tok/s | {ram} | {q} |")
    lines += ["", "## Run", "", "```sh", f"llama-cli -m model-{formats[0]}.gguf -c {p.requirements.context}", "```", "",
              "Estimates come from `plan.json`; see `benchmark.json` for measurements when present.", ""]
    return "\n".join(lines)


@app.command()
def benchmark(
    path: Path = typer.Argument(..., help="A .gguf file or a directory of .gguf files."),
    threads: int = typer.Option(None, help="CPU threads for llama.cpp."),
    quality: bool = typer.Option(True, help="Measure quality (perplexity) vs a reference variant."),
    reference: Path = typer.Option(None, help="Reference .gguf for quality (default: highest precision that fits)."),
    eval_text: Path = typer.Option(None, help="Text file to measure perplexity on (default: bundled sample)."),
    save: Path = typer.Option(None, help="Also write results to this JSON file."),
):
    """Measure real speed, memory and quality of GGUF models on this machine."""
    files = sorted(path.glob("*.gguf")) if path.is_dir() else [path]
    if not files or not all(f.is_file() and f.suffix == ".gguf" for f in files):
        console.print(f"[red]No .gguf files found at {path}")
        raise typer.Exit(1)
    # A split model (x-00001-of-00002.gguf, ...) is one model: benchmark it via its first part.
    files = sorted({gguf_parts(f)[0] for f in files})
    for f in files:
        missing = [p.name for p in gguf_parts(f) if not p.exists()]
        if missing:
            console.print(f"[red]{f.name} is part of a split model, but these parts are missing: {', '.join(missing)}")
            raise typer.Exit(1)
    tc = find_toolchain()
    if not tc.bench:
        console.print("[red]llama-bench not found. Install llama.cpp and set NANOMESH_LLAMA_CPP.")
        raise typer.Exit(1)
    if quality and not tc.perplexity:
        console.print("[yellow]llama-perplexity not found; skipping quality measurement.")
        quality = False
    if quality and reference is None and len(files) == 1:
        console.print("[yellow]One file and no --reference: quality is relative to itself, so skipping it.")
        quality = False
    if reference and reference not in files:
        files.append(reference)
    device = _resolve_device("local")
    largest = max(gguf_size(f) for f in files) / 1024**3
    if warning := free_ram_warning(device, largest + 0.3):
        console.print(f"[yellow]⚠ {warning}")
    results = _evaluate(tc, files, device, quality, reference, eval_text, threads)
    console.print(bench_table(results))
    if save:
        save.write_text(json.dumps(store.rows(results), indent=2), encoding="utf-8")


@app.command()
def results(
    device_id: str = typer.Option(None, "--device", "-d", help="Only this device (id, or 'local')."),
    as_json: bool = typer.Option(False, "--json"),
):
    """Show benchmark results recorded on this machine."""
    rows = store.load()
    if device_id:
        key = _resolve_device(device_id).key
        rows = [r for r in rows if r.device_key == key]
    if as_json:
        console.print_json(data=store.rows(rows))
    elif not rows:
        console.print(f"[yellow]No results yet in {store.results_path()}. Run `nanomesh benchmark`.")
    else:
        console.print(results_table(rows))


if __name__ == "__main__":
    app()
