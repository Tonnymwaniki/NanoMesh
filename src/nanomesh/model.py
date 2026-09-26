"""Model analysis: figure out how big a model is without loading its weights."""

from __future__ import annotations

import json
import re
import struct
from math import prod
from pathlib import Path

from pydantic import BaseModel

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}

# Well-known open models, so users can plan before downloading anything.
# (params in billions, layers, hidden size, attention heads, kv heads, head dim)
KNOWN_MODELS: dict[str, tuple[float, int, int, int, int, int]] = {
    "qwen2.5-0.5b": (0.49, 24, 896, 14, 2, 64),
    "qwen2.5-1.5b": (1.54, 28, 1536, 12, 2, 128),
    "qwen2.5-3b": (3.09, 36, 2048, 16, 2, 128),
    "qwen2.5-7b": (7.62, 28, 3584, 28, 4, 128),
    "llama-3.2-1b": (1.24, 16, 2048, 32, 8, 64),
    "llama-3.2-3b": (3.21, 28, 3072, 24, 8, 128),
    "llama-3.1-8b": (8.03, 32, 4096, 32, 8, 128),
    "llama-3.1-70b": (70.6, 80, 8192, 64, 8, 128),
    "mistral-7b": (7.25, 32, 4096, 32, 8, 128),
    "gemma-2-2b": (2.61, 26, 2304, 8, 4, 256),
    "phi-3-mini": (3.82, 32, 3072, 32, 32, 96),
}


class ModelInfo(BaseModel):
    name: str
    source: str  # "safetensors", "gguf", "known", "size"
    params: int
    architecture: str | None = None
    num_layers: int | None = None
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    vocab_size: int | None = None
    max_context: int | None = None
    source_dtype: str | None = None
    disk_bytes: int | None = None

    @property
    def params_b(self) -> float:
        return self.params / 1e9

    def kv_bytes_per_token(self, bytes_per_elem: int = 2) -> int:
        """KV-cache bytes needed per token of context (fp16 cache by default)."""
        if self.num_layers and self.num_kv_heads and self.head_dim:
            return 2 * self.num_layers * self.num_kv_heads * self.head_dim * bytes_per_elem
        # Unknown architecture: ~16 KB per billion params per token is typical for
        # modern grouped-query-attention models (Llama 3 8B is ~128 KB/token).
        return int(16_384 * max(self.params_b, 0.1))


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        if length > 100 * 1024 * 1024:
            raise ValueError(f"{path.name}: implausible safetensors header size")
        return json.loads(f.read(length))


def _count_safetensors(files: list[Path]) -> tuple[int, str | None]:
    params = 0
    dtypes: dict[str, int] = {}
    for file in files:
        for name, meta in read_safetensors_header(file).items():
            if name == "__metadata__":
                continue
            n = prod(meta["shape"]) if meta["shape"] else 1
            params += n
            dtypes[meta["dtype"]] = dtypes.get(meta["dtype"], 0) + n
    dominant = max(dtypes, key=dtypes.get) if dtypes else None
    return params, dominant


def _from_config(info: dict, config: dict) -> dict:
    text = config.get("text_config", config)  # multimodal configs nest the LM config
    heads = text.get("num_attention_heads")
    hidden = text.get("hidden_size")
    head_dim = text.get("head_dim") or (hidden // heads if hidden and heads else None)
    info.update(
        architecture=(config.get("architectures") or [config.get("model_type")])[0],
        num_layers=text.get("num_hidden_layers"),
        hidden_size=hidden,
        num_attention_heads=heads,
        num_kv_heads=text.get("num_key_value_heads", heads),
        head_dim=head_dim,
        vocab_size=text.get("vocab_size"),
        max_context=text.get("max_position_embeddings"),
    )
    return info


def parse_size(spec: str) -> float | None:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([bm])\s*", spec.lower())
    if not m:
        return None
    value = float(m.group(1))
    return value if m.group(2) == "b" else value / 1000


def analyze(spec: str) -> ModelInfo:
    """Analyze a local model directory/file, a known model name, or a size like '7b'."""
    path = Path(spec).expanduser()
    if path.is_dir():
        return _analyze_dir(path)
    if path.is_file() and path.suffix == ".gguf":
        # Without parsing the GGUF tensor table we estimate params from file size
        # assuming ~4.85 bits/weight; good enough for planning.
        size = path.stat().st_size
        return ModelInfo(name=path.stem, source="gguf", params=int(size * 8 / 4.85), disk_bytes=size)
    if path.is_file() and path.suffix == ".safetensors":
        params, dtype = _count_safetensors([path])
        return ModelInfo(name=path.stem, source="safetensors", params=params,
                         source_dtype=dtype, disk_bytes=path.stat().st_size)

    key = spec.lower().strip()
    if key in KNOWN_MODELS:
        p, layers, hidden, heads, kv, hd = KNOWN_MODELS[key]
        return ModelInfo(name=key, source="known", params=int(p * 1e9), num_layers=layers,
                         hidden_size=hidden, num_attention_heads=heads, num_kv_heads=kv,
                         head_dim=hd, source_dtype="BF16")
    size_b = parse_size(spec)
    if size_b:
        return ModelInfo(name=f"{spec.upper()} model", source="size", params=int(size_b * 1e9),
                         source_dtype="BF16")
    known = ", ".join(KNOWN_MODELS)
    raise ValueError(
        f"Can't interpret '{spec}'. Pass a model directory, a .safetensors/.gguf file, "
        f"a size like '7b', or one of: {known}"
    )


def _analyze_dir(path: Path) -> ModelInfo:
    info: dict = {"name": path.name, "source": "safetensors"}
    config_path = path / "config.json"
    if config_path.exists():
        _from_config(info, json.loads(config_path.read_text()))
    files = sorted(path.glob("*.safetensors"))
    ggufs = sorted(path.glob("*.gguf"))
    if files:
        info["params"], info["source_dtype"] = _count_safetensors(files)
        info["disk_bytes"] = sum(f.stat().st_size for f in files)
    elif ggufs:
        return analyze(str(ggufs[0]))
    else:
        raise ValueError(f"No .safetensors or .gguf weights found in {path}")
    return ModelInfo(**info)
