"""`nanomesh train-plan`: will fine-tuning this model fit on this device?

Standard memory accounting for mixed-precision AdamW training:
- full fine-tuning: 16 bytes per parameter (bf16 weights and grads, fp32
  master weights, two fp32 Adam moments)
- LoRA: frozen bf16 base (2 bytes/param) + 16 bytes per adapter parameter
- QLoRA: 4-bit NF4 base (~4.5 bits/param with double quantization) + adapters
plus activations (bf16, flash attention) and the loss layer's logits.
These are estimates; real runs vary with framework and settings.
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from nanomesh.hardware import GB, DeviceProfile
from nanomesh.model import ModelInfo

BYTES_FULL = 16
BYTES_ADAPTER = 16
BYTES_BF16 = 2
BYTES_NF4 = 4.5 / 8
ACT_BYTES_PER_HIDDEN = 34  # per token per layer without checkpointing (bf16, flash attention)
LOGIT_BYTES = 10  # bf16 logits, fp32 upcast and its gradient
FRAMEWORK_GB = 1.0  # CUDA context / PyTorch runtime
DEFAULT_VOCAB = 32_000
METHOD_NAMES = {"full": "full fine-tuning", "lora": "LoRA", "qlora": "QLoRA"}


class TrainOption(BaseModel):
    method: str  # full | lora | qlora
    total_gb: float
    breakdown: dict[str, float]  # GB per component
    trainable_params: int
    fits: bool
    available: bool  # e.g. QLoRA needs an NVIDIA GPU
    note: str


class TrainPlan(BaseModel):
    model: str
    params: int
    device: str
    placement: str  # "GPU" or "CPU"
    memory_gb: float
    seq_len: int
    batch: int
    lora_rank: int
    checkpointing: bool
    shape_estimated: bool  # architecture guessed from the parameter count
    options: list[TrainOption]
    recommended: str | None
    advice: list[str]


def _shape(model: ModelInfo) -> tuple[int, int, int, int, bool]:
    """(layers, hidden, kv_dim, vocab, estimated?)"""
    if model.num_layers and model.hidden_size:
        kv_dim = (model.num_kv_heads or model.num_attention_heads or 1) * (model.head_dim or 128)
        return model.num_layers, model.hidden_size, kv_dim, model.vocab_size or DEFAULT_VOCAB, False
    # Unknown architecture: transformer blocks hold ~12 * hidden^2 params each.
    layers = 24 if model.params < 2e9 else 32 if model.params < 20e9 else 80
    hidden = int(math.sqrt(model.params / (12 * layers)))
    return layers, hidden, hidden // 4, DEFAULT_VOCAB, True


def _budget(device: DeviceProfile) -> tuple[str, float]:
    gpu = device.best_gpu
    if gpu and gpu.vram_gb:
        return "GPU", gpu.vram_gb * 0.92
    return "CPU", device.ram_gb * 0.75


def train_plan(model: ModelInfo, device: DeviceProfile, *, seq_len: int = 1024, batch: int = 1,
               lora_rank: int = 16, checkpointing: bool = True) -> TrainPlan:
    layers, hidden, kv_dim, vocab, estimated = _shape(model)
    placement, memory = _budget(device)
    nvidia = any(g.vendor == "NVIDIA" for g in device.gpus)
    tokens = seq_len * batch

    # LoRA on the attention projections (q, k, v, o): r * (in + out) per matrix.
    adapters = layers * lora_rank * ((hidden + hidden) * 2 + (hidden + kv_dim) * 2)
    if checkpointing:
        acts = (layers * tokens * hidden * BYTES_BF16 + tokens * hidden * ACT_BYTES_PER_HIDDEN)
    else:
        acts = layers * tokens * hidden * ACT_BYTES_PER_HIDDEN
    logits = tokens * vocab * LOGIT_BYTES
    shared = {"activations": acts / GB, "loss layer": logits / GB, "runtime": FRAMEWORK_GB}

    def option(method: str, parts: dict[str, float], trainable: int, available: bool, note: str) -> TrainOption:
        parts = {k: round(v, 2) for k, v in {**parts, **shared}.items()}
        total = round(sum(parts.values()), 1)
        return TrainOption(method=method, total_gb=total, breakdown=parts, trainable_params=trainable,
                           fits=available and total <= memory, available=available, note=note)

    options = [
        option("full", {"weights + grads + optimizer": model.params * BYTES_FULL / GB}, model.params, True,
               "Updates every weight: best results, most memory."),
        option("lora", {"base model (bf16)": model.params * BYTES_BF16 / GB,
                        "adapters + optimizer": adapters * BYTES_ADAPTER / GB}, adapters, True,
               f"Trains small rank-{lora_rank} adapters on the attention layers; the base stays frozen."),
        option("qlora", {"base model (4-bit)": model.params * BYTES_NF4 / GB,
                         "adapters + optimizer": adapters * BYTES_ADAPTER / GB}, adapters, nvidia,
               "LoRA on a 4-bit base: the least memory, near-LoRA quality." if nvidia else
               "Needs an NVIDIA GPU (bitsandbytes), which this device doesn't have."),
    ]
    best = next((o for o in options if o.fits), None)
    return TrainPlan(model=model.name, params=model.params, device=device.name, placement=placement,
                     memory_gb=round(memory, 1), seq_len=seq_len, batch=batch, lora_rank=lora_rank,
                     checkpointing=checkpointing, shape_estimated=estimated, options=options,
                     recommended=best.method if best else None,
                     advice=_advice(model, device, placement, memory, options, best, seq_len, batch, nvidia))


def _max_params_qlora(memory_gb: float) -> float:
    """Rough largest model (billions) whose QLoRA run fits, at modest settings."""
    usable = memory_gb - FRAMEWORK_GB - 1.5  # adapters, activations, logits at seq 1024
    return max(0.0, round(usable * GB / (BYTES_NF4 * 1e9), 1))


def _advice(model, device, placement, memory, options, best, seq_len, batch, nvidia) -> list[str]:
    out = []
    if placement == "CPU":
        out.append("No usable GPU: training on the CPU works for tiny experiments but is typically tens of times "
                   "slower than a GPU. For real runs, rent a cloud GPU and use this plan to size it.")
    if best is None:
        smallest = min((o for o in options if o.available), key=lambda o: o.total_gb)
        out.append(f"Nothing fits in the ~{memory:.1f} GB budget; the smallest option ({METHOD_NAMES[smallest.method]}) needs "
                   f"~{smallest.total_gb:g} GB.")
        acts = smallest.breakdown["activations"] + smallest.breakdown["loss layer"]
        if acts > 1 and (seq_len > 512 or batch > 1):
            out.append(f"Sequences and batch take ~{acts:.1f} GB: try --seq-len 512 --batch 1 with gradient "
                       "accumulation to keep the same effective batch size.")
        for gb in (16, 24, 48):
            if _max_params_qlora(gb) * 1e9 >= model.params:
                out.append(f"A {gb} GB NVIDIA GPU (cloud or local) fits QLoRA for this model.")
                break
    elif best.method != "full":
        full = options[0]
        out.append(f"Full fine-tuning would need ~{full.total_gb:g} GB; {METHOD_NAMES[best.method]} fits in "
                   f"~{best.total_gb:g} GB and usually gets close to full fine-tuning quality.")
    if batch > 1 and best is not None:
        out.append("Tip: batch 1 with gradient accumulation gives the same effective batch using less memory.")
    return out
