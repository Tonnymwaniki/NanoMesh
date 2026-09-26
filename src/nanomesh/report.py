"""Rich terminal rendering for device passports, plans and benchmarks."""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nanomesh.hardware import DeviceProfile, compute_class
from nanomesh.model import ModelInfo, analyze
from nanomesh.planner import Plan, Requirements, memory_budgets, max_practical_params, plan
from nanomesh.toolchain import BenchResult

QUALITY_STYLE = {"lossless": "green", "high": "green", "good": "cyan", "fair": "yellow", "severe": "red"}
REFERENCE_SIZES = ["1b", "3b", "7b", "13b", "32b", "70b"]


def _fmt_speed(v: float | None) -> str:
    return f"~{v:g} tok/s" if v is not None else "—"


def device_passport(device: DeviceProfile) -> Panel:
    spec = Table.grid(padding=(0, 2))
    spec.add_column(style="bold")
    spec.add_column()
    rows = [
        ("CPU", device.cpu), ("Cores", _cores(device)), ("Arch", device.arch), ("OS", device.os),
        ("RAM", f"{device.ram_gb:g} GB" + (f" ({device.available_ram_gb:g} GB free)" if device.available_ram_gb else "")),
        ("Memory BW", f"{device.memory_bandwidth_gbps:g} GB/s" if device.memory_bandwidth_gbps else None),
        ("CPU features", ", ".join(device.cpu_flags) or None),
        ("GPU", "; ".join(_gpu_str(g) for g in device.gpus) or None),
        ("NPU", device.npu), ("Runtimes", ", ".join(device.runtimes) or None),
        ("Disk free", f"{device.disk_free_gb:g} GB" if device.disk_free_gb else None),
    ]
    for k, v in rows:
        if v:
            spec.add_row(k, str(v))

    budget = max(b.memory_gb for b in memory_budgets(device))
    comfy = max_practical_params(budget * 0.75, 4096)
    limit = max_practical_params(budget, 4096)
    fit = Text()
    fit.append(f"\nAI COMPUTE CLASS  {compute_class(device)}\n", style="bold magenta")
    fit.append(f"Model memory budget ~{budget:.1f} GB\n\n")
    fit.append(f"🟢 Recommended  up to ~{comfy}B params (INT4)\n", style="green")
    fit.append(f"🟡 Possible     ~{comfy}B – {limit}B params (INT4, tight)\n", style="yellow")
    fit.append(f"🔴 Not advised  above ~{limit}B params\n", style="red")

    sizes = Table(title="What fits (4K context)", title_justify="left", box=None, header_style="bold")
    for col in ("Model size", "Best variant", "Memory", "Speed", "Runs on"):
        sizes.add_column(col)
    for size in REFERENCE_SIZES:
        p = plan(analyze(size), device, Requirements(min_quality="fair"))
        v = next((x for x in p.variants if x.format.name == p.recommended), None)
        if v:
            sizes.add_row(size.upper(), Text(v.format.label, style=QUALITY_STYLE[v.format.quality]),
                          f"{v.total_memory_gb:.1f} GB", _fmt_speed(v.tokens_per_s), v.placement or "")
        else:
            sizes.add_row(size.upper(), Text("won't fit", style="red"), "", "", "")

    notes = Text(f"\n{device.notes}", style="dim") if device.notes else Text("")
    return Panel(Group(spec, fit, sizes, notes), title=f"[bold]DEVICE PASSPORT · {device.name}",
                 subtitle="speeds are bandwidth-based estimates", border_style="magenta")


def _cores(d: DeviceProfile) -> str | None:
    if d.physical_cores and d.logical_cores:
        return f"{d.physical_cores} cores / {d.logical_cores} threads"
    return f"{d.logical_cores} threads" if d.logical_cores else None


def _gpu_str(g) -> str:
    extra = []
    if g.vram_gb:
        extra.append(f"{g.vram_gb:g} GB" + (" shared" if g.unified_memory else ""))
    if g.compute_capability:
        extra.append(f"sm {g.compute_capability}")
    return g.name + (f" ({', '.join(extra)})" if extra else "")


def model_summary(m: ModelInfo) -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold")
    t.add_column()
    rows = [
        ("Model", m.name), ("Architecture", m.architecture), ("Parameters", f"{m.params_b:.2f}B"),
        ("Weights dtype", m.source_dtype), ("Layers", m.num_layers), ("Hidden size", m.hidden_size),
        ("Attention", f"{m.num_attention_heads} heads, {m.num_kv_heads} KV heads" if m.num_attention_heads else None),
        ("Vocab", m.vocab_size), ("Max context", m.max_context),
        ("On disk", f"{m.disk_bytes / 1024**3:.2f} GB" if m.disk_bytes else None),
        ("KV cache", f"{m.kv_bytes_per_token() / 1024:.0f} KB/token (fp16)"),
    ]
    for k, v in rows:
        if v is not None:
            t.add_row(k, str(v))
    return t


def plan_view(p: Plan) -> Group:
    table = Table(title=f"{p.model.name} ({p.model.params_b:.1f}B) on {p.device.name}",
                  header_style="bold", title_justify="left")
    table.add_column("", width=2, no_wrap=True)
    table.add_column("Variant", no_wrap=True)
    for col in ("Memory", "Smaller", "Speed", "Quality", "Runs on"):
        table.add_column(col, no_wrap=True)
    for v in p.variants:
        mark = "🏆" if v.format.name == p.recommended else ("•" if v.pareto else "")
        table.add_row(
            mark, f"{v.format.label} [dim]{v.format.name}", f"{v.total_memory_gb:.1f} GB",
            f"{v.compression:.0%}", _fmt_speed(v.tokens_per_s),
            Text(v.format.quality, style=QUALITY_STYLE[v.format.quality]),
            v.placement or "✕", style=None if v.meets_requirements else "dim",
        )

    budget_txt = ", ".join(f"{b.placement} {b.memory_gb:.1f} GB" for b in p.budgets)
    summary = Text()
    summary.append(f"Memory budget: {budget_txt} · context {p.requirements.context} tokens\n", style="dim")
    rec = next((v for v in p.variants if v.format.name == p.recommended), None)
    if rec:
        summary.append(f"\n🏆 Recommended: {rec.format.label} ({rec.format.name}) on {rec.placement}", style="bold green")
        summary.append(f" — {rec.total_memory_gb:.1f} GB, {rec.compression:.0%} smaller than FP16")
        if rec.tokens_per_s:
            summary.append(f", {_fmt_speed(rec.tokens_per_s)}")
        summary.append(f"\n  {rec.format.note}")
        summary.append("\n✓ Meets your requirements\n", style="green")
    else:
        summary.append("\n✕ No variant meets your requirements\n", style="bold red")
    rejected = [v for v in p.variants if v.reasons]
    if rejected:
        summary.append("\nRuled out:\n", style="dim")
        for v in rejected:
            summary.append(f"  {v.format.label}: {'; '.join(v.reasons)}\n", style="dim")
    for a in p.advice:
        summary.append(f"→ {a}\n", style="yellow")
    summary.append("• = Pareto-optimal (no smaller variant of equal or better quality)", style="dim")
    return Group(table, summary)


def bench_table(results: list[BenchResult]) -> Table:
    t = Table(title="Benchmark (measured with llama-bench)", header_style="bold", title_justify="left")
    for col in ("Model file", "Size", "Prompt", "Generate", "Peak RAM", "Backend"):
        t.add_column(col)
    for r in results:
        t.add_row(r.model_file, f"{r.size_gb:.2f} GB",
                  f"{r.prompt_tokens_per_s} tok/s" if r.prompt_tokens_per_s else "—",
                  f"{r.gen_tokens_per_s} tok/s" if r.gen_tokens_per_s else "—",
                  f"{r.peak_rss_gb} GB" if r.peak_rss_gb else "—", r.backend or "—")
    return t

