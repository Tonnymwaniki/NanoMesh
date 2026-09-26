"""Rich terminal rendering for device passports, plans and benchmarks."""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nanomesh.hardware import DeviceProfile, compute_class
from nanomesh.model import ModelInfo, analyze
from nanomesh.planner import Plan, Requirements, Variant, memory_budgets, max_practical_params, plan
from nanomesh.results import Result, evidence

QUALITY_STYLE = {"lossless": "green", "high": "green", "good": "cyan", "fair": "yellow", "severe": "red"}
REFERENCE_SIZES = ["1b", "3b", "7b", "13b", "32b", "70b"]


def _fmt_gb(gb: float | None) -> str:
    if gb is None:
        return "—"
    return f"{gb * 1024:.1f} MB" if gb < 1 else f"{gb:.2f} GB"


def _fmt_speed(v: float | None, source: str | None = "estimate") -> str:
    if v is None:
        return "—"
    n = f"{v:.1f}" if v < 1000 else f"{v:,.0f}"
    return {"measured": f"{n} tok/s ✓", "calibrated": f"{n} tok/s *"}.get(source, f"~{n} tok/s")


def params_str(params: int) -> str:
    return f"{params / 1e9:.1f}B" if params >= 1e8 else f"{params / 1e6:.1f}M"


def _fmt_quality(v: Variant) -> Text:
    pct = f"{v.quality_pct:g}% ✓" if v.quality_measured else f"~{v.quality_pct:g}%"
    return Text(f"{v.quality} {pct}", style=QUALITY_STYLE[v.quality])


def device_passport(device: DeviceProfile) -> Panel:
    spec = Table.grid(padding=(0, 2))
    spec.add_column(style="bold")
    spec.add_column()
    rows = [
        ("Recognised as", device.matched_id if device.is_local else None),
        ("Type", device.kind),
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
        model = analyze(size)
        # Device-level calibration applies; per-model measurements don't (these are generic sizes).
        ev = evidence(device, model).model_copy(update={"speeds": {}, "quality": {}})
        p = plan(model, device, Requirements(min_quality="fair"), ev)
        v = next((x for x in p.variants if x.format.name == p.recommended), None)
        if v:
            sizes.add_row(size.upper(), Text(v.format.label, style=QUALITY_STYLE[v.quality]),
                          f"{v.total_memory_gb:.1f} GB", _fmt_speed(v.tokens_per_s, v.speed_source), v.placement or "")
        else:
            sizes.add_row(size.upper(), Text("won't fit", style="red"), "", "", "")

    notes = Text(f"\n{device.notes}", style="dim") if device.notes else Text("")
    if device.is_local and device.available_ram_gb is not None and device.available_ram_gb < budget * 0.5:
        notes.append(f"\n⚠ Only {device.available_ram_gb:g} GB of RAM is free right now. The table assumes "
                     "you close other apps (especially browsers) before running a model.", style="yellow")
    if device.is_local and not device.matched_id:
        notes.append("\nThis exact model isn't in the NanoMesh database yet, so speeds are unknown until you run "
                     "`nanomesh benchmark`.", style="dim")
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
        ("Model", m.name), ("Architecture", m.architecture), ("Parameters", params_str(m.params)),
        ("Weights dtype", m.source_dtype), ("Layers", m.num_layers), ("Hidden size", m.hidden_size),
        ("Attention", f"{m.num_attention_heads} heads, {m.num_kv_heads} KV heads" if m.num_attention_heads else None),
        ("Vocab", m.vocab_size), ("Max context", m.max_context),
        ("On disk", _fmt_gb(m.disk_bytes / 1024**3) if m.disk_bytes else None),
        ("KV cache", f"{m.kv_bytes_per_token() / 1024:.0f} KB/token (fp16)"),
    ]
    for k, v in rows:
        if v is not None:
            t.add_row(k, str(v))
    return t


def plan_view(p: Plan) -> Group:
    table = Table(title=f"{p.model.name} ({params_str(p.model.params)}) on {p.device.name}",
                  header_style="bold", title_justify="left")
    table.add_column("", width=2, no_wrap=True)
    table.add_column("Variant", no_wrap=True)
    for col in ("Memory", "Speed", "Quality", "Runs on"):
        table.add_column(col, no_wrap=True)
    for v in p.variants:
        mark = "🏆" if v.format.name == p.recommended else ("•" if v.pareto else "")
        table.add_row(
            mark, f"{v.format.label} [dim]{v.format.name}", f"{v.total_memory_gb:.1f} GB",
            _fmt_speed(v.tokens_per_s, v.speed_source), _fmt_quality(v),
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
            summary.append(f", {_fmt_speed(rec.tokens_per_s, rec.speed_source)}")
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
    summary.append("• Pareto-optimal (no smaller variant of equal or better quality)\n", style="dim")
    summary.append("✓ measured on this device · * calibrated from this device's benchmarks · ~ estimate", style="dim")
    return Group(table, summary)


def bench_table(results: list[Result]) -> Table:
    t = Table(title="Measured on this machine", header_style="bold", title_justify="left")
    for col in ("Variant", "Size", "Prompt", "Generate", "Peak RAM", "Quality"):
        t.add_column(col, no_wrap=col != "Variant")
    for r in results:
        q = f"{r.quality_pct:g}% of {r.reference_format}" if r.quality_pct is not None else "—"
        t.add_row(r.format or r.model_name, _fmt_gb(r.file_size_gb),
                  f"{r.prompt_tokens_per_s:g} tok/s" if r.prompt_tokens_per_s else "—",
                  f"{r.gen_tokens_per_s:g} tok/s" if r.gen_tokens_per_s else "—",
                  _fmt_gb(r.peak_rss_gb), q)
    return t


def results_table(results: list[Result]) -> Table:
    t = Table(title="Recorded benchmark results", header_style="bold", title_justify="left")
    for col in ("When", "Device", "Model", "Variant", "Generate", "Peak RAM", "Quality"):
        t.add_column(col)
    for r in results:
        size = params_str(r.model_params)
        t.add_row(r.timestamp[:16].replace("T", " "), r.device_name, f"{r.model_name} ({size})", r.format or "?",
                  f"{r.gen_tokens_per_s:g} tok/s" if r.gen_tokens_per_s else "—", _fmt_gb(r.peak_rss_gb),
                  f"{r.quality_pct:g}%" if r.quality_pct is not None else "—")
    return t
