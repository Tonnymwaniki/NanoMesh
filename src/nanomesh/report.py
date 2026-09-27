"""Rich terminal rendering for device passports, plans and benchmarks."""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nanomesh.conditions import Conditions
from nanomesh.conditions import advice as condition_advice
from nanomesh.hardware import DeviceProfile
from nanomesh.model import ModelInfo
from nanomesh.passport import passport
from nanomesh.planner import Plan, Variant
from nanomesh.results import Result

QUALITY_STYLE = {"lossless": "green", "high": "green", "good": "cyan", "fair": "yellow", "severe": "red"}


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
    pct = {"measured": f"{v.quality_pct:g}% ✓", "calibrated": f"{v.quality_pct:g}% *"}.get(
        v.quality_source, f"~{v.quality_pct:g}%")
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

    pp = passport(device)
    fit = Text()
    fit.append(f"\nAI COMPUTE CLASS  {pp.compute_class}\n", style="bold magenta")
    fit.append(f"Model memory budget ~{pp.budget_gb:.1f} GB\n\n")
    fit.append(f"🟢 Recommended  up to ~{pp.recommended_max_b}B params (INT4)\n", style="green")
    fit.append(f"🟡 Possible     ~{pp.recommended_max_b}B – {pp.possible_max_b}B params (INT4, tight)\n", style="yellow")
    fit.append(f"🔴 Not advised  above ~{pp.possible_max_b}B params\n", style="red")

    sizes = Table(title="What fits (4K context)", title_justify="left", box=None, header_style="bold")
    for col in ("Model size", "Best variant", "Memory", "Speed", "Runs on"):
        sizes.add_column(col)
    for f in pp.fits:
        if f.fits:
            sizes.add_row(f.size, Text(f.label, style=QUALITY_STYLE[f.quality]), f"{f.memory_gb:.1f} GB",
                          _fmt_speed(f.tokens_per_s, f.speed_source), f.placement or "")
        else:
            sizes.add_row(f.size, Text("won't fit", style="red"), "", "", "")

    notes = Text(f"\n{device.notes}", style="dim") if device.notes else Text("")
    if pp.low_free_ram:
        notes.append(f"\n⚠ Only {device.available_ram_gb:g} GB of RAM is free right now. The table assumes "
                     "you close other apps (especially browsers) before running a model.", style="yellow")
    if pp.unrecognised:
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
    steady = any(r.steady for r in results)
    burst = any(r.burst_tokens_per_s for r in results)
    t.title = "Measured on this machine · speeds in tok/s" + (" · steady state" if steady else "")
    warmed = any(r.warmup_settled_tokens_per_s for r in results)
    cols = ["Variant", "Size", "Prompt", "Generate"] + (["Cold"] if burst else []) + (["Warm-up"] if warmed else []) \
        + ["Peak RAM", "Quality"]
    for col in cols:
        t.add_column(col, no_wrap=True, min_width=6 if col == "Variant" else None)

    def num(v):
        return f"{v:.1f}" if v is not None and v < 1000 else (f"{v:,.0f}" if v else "—")

    for r in results:
        q = f"{r.quality_pct:g}% of {r.reference_format}" if r.quality_pct is not None else "—"
        row = [r.format or r.model_name, _fmt_gb(r.file_size_gb), num(r.prompt_tokens_per_s), num(r.gen_tokens_per_s)]
        if burst:
            row.append(num(r.burst_tokens_per_s) if r.burst_tokens_per_s else "")
        if warmed:
            row.append(num(r.warmup_settled_tokens_per_s) if r.warmup_settled_tokens_per_s else "")
        t.add_row(*row, _fmt_gb(r.peak_rss_gb), q)
    if burst or warmed:
        t.caption = ("Generate: llama-bench after warming up. Cold: the first seconds from idle, when laptops boost. "
                     "Warm-up: where continuous generation settled before measuring.")
    return t


def _run_label(r: Result) -> str:
    power = (" · battery" if r.on_battery else "") + (" · busy" if r.busy else "")
    if r.kind == "sustained":
        return "sustained" + power
    if r.kind == "threads":
        return f"thread sweep · {r.threads}t" if r.threads else "thread sweep"
    return ("steady" if r.steady else "quick") + power


def _generate(r: Result) -> str:
    if r.kind == "sustained" and r.sustained:
        # A sustained run's headline is where it settled, not its first seconds.
        return f"{r.sustained.burst_tokens_per_s:g} → {r.sustained.sustained_tokens_per_s:g} tok/s"
    return f"{r.gen_tokens_per_s:g} tok/s" if r.gen_tokens_per_s else "—"


def results_table(results: list[Result]) -> Table:
    t = Table(title="Recorded benchmark results", header_style="bold", title_justify="left")
    for col in ("When", "Device", "Model", "Variant", "Run", "Generate", "Peak RAM", "Quality"):
        t.add_column(col)
    for r in results:
        size = params_str(r.model_params)
        t.add_row(r.timestamp[:16].replace("T", " "), r.device_name, f"{r.model_name} ({size})", r.format or "?",
                  _run_label(r), _generate(r), _fmt_gb(r.peak_rss_gb),
                  f"{r.quality_pct:g}%" if r.quality_pct is not None else "—")
    return t


def conditions_view(c: Conditions, battery_note: str | None = None) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    power = None
    if c.on_battery is not None:
        power = "On battery" if c.on_battery else "Plugged in"
        if c.battery_pct is not None:
            power += f" · {c.battery_pct:g}%"
    rows = [
        ("Power", power),
        ("Power plan", " · ".join(x for x in (c.power_plan, c.power_mode) if x) or None),
        ("CPU speed", f"{c.clock_pct:g}% of rated" if c.clock_pct is not None else (f"{c.cpu_mhz:g} MHz" if c.cpu_mhz else None)),
        ("CPU temperature", f"{c.temp_c:g}°C" if c.temp_c is not None else "not readable on this system"),
        ("CPU busy", f"{c.cpu_load_pct:g}%" if c.cpu_load_pct is not None else None),
        ("Free RAM", f"{c.available_ram_gb:g} GB" if c.available_ram_gb is not None else None),
        ("Battery draw", f"{c.discharge_w:g} W" if c.discharge_w else None),
        ("Battery health", f"{c.battery_full_wh:g} of {c.battery_design_wh:g} Wh "
                           f"({round(100 * c.battery_full_wh / c.battery_design_wh)}%)"
                           if c.battery_full_wh and c.battery_design_wh else None),
    ]
    for k, v in rows:
        if v:
            grid.add_row(k, v)
    notes = Text()
    for a in condition_advice(c, battery_note=battery_note):
        notes.append(f"\n→ {a}", style="yellow")
    if not notes:
        notes.append("\n✓ Nothing in the current conditions should slow a model down.", style="green")
    return Panel(Group(grid, notes), title="[bold]RIGHT NOW", border_style="cyan")


def sustained_view(r: Result) -> Panel:
    s = r.sustained
    t = Text()
    t.append(f"{s.burst_tokens_per_s:g} → {s.sustained_tokens_per_s:g} tok/s", style="bold")
    t.append(f"  ({s.drop_pct:g}% slower after {s.points[-1].t_s / 60:.1f} min)")
    rc = r.conditions
    if rc and rc.temp_max_c is not None:
        t.append(f"\nHottest: {rc.temp_max_c:g}°C")
    if rc and rc.clock_pct_min is not None:
        t.append(f"\nSlowest CPU clock: {rc.clock_pct_min:g}% of rated")
    if s.watts:
        t.append(f"\nBattery draw {s.watts:g} W · {s.joules_per_token:g} J per token")
    if s.battery_hours:
        t.append(f"\nA full battery lasts ~{s.battery_hours:g} h of continuous generation")
        if s.battery_hours_range:
            lo, hi = s.battery_hours_range
            t.append(f" (somewhere between {lo:g} and {hi:g} h)")
    if s.tokens_per_battery_pct:
        t.append(f"\n~{s.tokens_per_battery_pct:,} tokens per 1% of battery")
    if s.energy_source == "battery %":
        t.append("\nRough: measured from the battery percentage, which only moves in 1% steps. "
                 "A 15-minute run narrows it.", style="dim")
    elif s.energy_source:
        t.append(f"\nMeasured with the battery's {s.energy_source}.", style="dim")
    if r.busy:
        t.append(f"\nOther programs were using {rc.start.cpu_load_pct:g}% of the CPU when this started, so these "
                 "figures include their load. NanoMesh prefers quiet runs; close them and run again for a clean "
                 "measurement.", style="yellow")
    if rc and rc.start.on_battery is False:
        t.append("\nPlugged in: unplug and run again to measure battery life and energy per token.", style="dim")
    if s.drop_pattern == "step":
        t.append(f"\nSudden drop at {int(s.drop_at_s // 60)}:{int(s.drop_at_s % 60):02d}: the CPU's short-term turbo "
                 "budget ran out (typical of Intel laptops). Long sessions run at the lower speed; quick benchmarks "
                 "taken in the first minute overstate it.")
    elif s.drop_pattern == "gradual":
        t.append(f"\nSpeed slid down gradually from {int(s.drop_at_s // 60)}:{int(s.drop_at_s % 60):02d} as the "
                 "device heated up. Better airflow helps.")
    if s.points[-1].t_s < 60:
        t.append("\nThis run was under a minute: run at least 3 minutes (the default) for a reliable reading.",
                 style="yellow")
        return Panel(t, title=f"[bold]SUSTAINED · {r.model_name} {r.format}", border_style="cyan")
    verdict = ("green", "Speed held steady.") if s.drop_pct < 10 else \
        ("yellow", "Noticeable slowdown under sustained load.") if s.drop_pct < 25 else \
        ("red", "Heavy throttling: long sessions run much slower than a quick test suggests.")
    t.append(f"\n{verdict[1]}", style=verdict[0])
    return Panel(t, title=f"[bold]SUSTAINED · {r.model_name} {r.format}", border_style="cyan")


def threads_view(rows: list[Result], default_threads: int | None) -> Table:
    best = max(rows, key=lambda r: r.gen_tokens_per_s)
    t = Table(title=f"Thread counts · {rows[0].model_name} {rows[0].format}", header_style="bold", title_justify="left")
    for col in ("Threads", "Generate", ""):
        t.add_column(col)
    top = best.gen_tokens_per_s
    for r in sorted(rows, key=lambda r: r.threads):
        bar = "█" * max(1, round(24 * r.gen_tokens_per_s / top))
        tags = " ".join(x for x in ("🏆 fastest" if r is best else "", "(llama.cpp default)" if r.threads == default_threads else "") if x)
        t.add_row(str(r.threads), f"{r.gen_tokens_per_s:g} tok/s", Text(f"{bar} {tags}", style="green" if r is best else "blue"))
    return t


LEVEL_STYLE = {"ok": ("✓", "green"), "warn": ("!", "yellow"), "fail": ("✕", "red")}


def doctor_view(env) -> Group:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold")
    t.add_column()
    torch = env.torch or {}
    rows = [("Python", f"{env.python}{'' if env.in_virtualenv else ' (not in a virtualenv)'}"),
            ("OS", env.os),
            ("PyTorch", (f"{torch.get('version')} · " + ("CUDA GPU" if torch.get("cuda") else "Apple GPU" if torch.get("mps")
                                                        else "CPU only")) if torch and "error" not in torch else
             (f"broken: {torch.get('error')}" if torch else "not installed")),
            ("AI packages", ", ".join(f"{k} {v}" for k, v in env.packages.items() if k != "torch") or "none"),
            ("llama.cpp", "found" if env.llama_cpp.get("llama-bench") else "not found"),
            ("Tools", ", ".join(sorted(env.tools)) or "none"),
            ("Disk free", f"{env.disk_free_gb:g} GB" if env.disk_free_gb is not None else None)]
    for k, v in rows:
        if v:
            t.add_row(k, v)
    out = Text()
    for f in sorted(env.findings, key=lambda f: ["fail", "warn", "ok"].index(f.level)):
        mark, style = LEVEL_STYLE[f.level]
        out.append(f"\n{mark} {f.topic}: {f.message}", style=style)
        if f.fix:
            out.append(f"\n    → {f.fix}", style="dim")
    problems = sum(f.level != "ok" for f in env.findings)
    out.append(f"\n\n{'No problems found.' if not problems else f'{problems} thing(s) to look at.'}",
               style="bold green" if not problems else "bold yellow")
    return Group(Panel(t, title="[bold]ENVIRONMENT", border_style="cyan"), out)


def models_view(found, device_name: str) -> Table | Text:
    if not found:
        return Text("No models found in the Hugging Face cache, LM Studio, Ollama or ~/models. "
                    "Point NanoMesh at a folder: nanomesh models C:\\models", style="yellow")
    t = Table(title=f"Models on this machine · fit for {device_name}", header_style="bold", title_justify="left")
    for col in ("Model", "Source", "Type", "Size", "Fits", "Best variant here"):
        t.add_column(col, overflow="fold")
    for m in found:
        kind = (m.format or m.kind) + (f" · {m.parts} parts" if m.parts > 1 else "")
        if m.missing_parts:
            fits = Text(f"{m.missing_parts} part(s) missing", style="red")
        else:
            fits = Text("yes", style="green") if m.fits else Text("no", style="red")
        t.add_row(m.name, m.source, kind, f"{m.size_gb:g} GB", fits, m.recommended_note or "—")
    return t


def train_plan_view(tp) -> Group:
    names = {"full": "Full fine-tuning", "lora": "LoRA", "qlora": "QLoRA"}
    t = Table(title=f"Fine-tuning {tp.model} ({tp.params / 1e9:.1f}B) on {tp.device}", header_style="bold",
              title_justify="left")
    for col in ("", "Method", "Memory", "Trainable", "Fits", "Notes"):
        t.add_column(col, overflow="fold")
    for o in tp.options:
        mark = "🏆" if o.method == tp.recommended else ""
        fits = Text("yes", style="green") if o.fits else Text("n/a" if not o.available else "no", style="red")
        trainable = f"{o.trainable_params / 1e6:,.1f}M" if o.trainable_params < 1e9 else f"{o.trainable_params / 1e9:.1f}B"
        t.add_row(mark, names[o.method], f"{o.total_gb:g} GB", trainable, fits, o.note)
    info = Text(f"{tp.placement} memory budget ~{tp.memory_gb:g} GB · sequence {tp.seq_len} tokens · batch {tp.batch}"
                f" · LoRA rank {tp.lora_rank} · gradient checkpointing {'on' if tp.checkpointing else 'off'}",
                style="dim")
    if tp.shape_estimated:
        info.append("\nArchitecture unknown: layer sizes estimated from the parameter count.", style="dim")
    advice = Text()
    for a in tp.advice:
        advice.append(f"\n→ {a}", style="yellow")
    return Group(t, info, advice)
