import json
import struct

import pytest


def write_safetensors(path, tensors: dict[str, tuple[str, list[int]]]):
    """Write a header-only-valid safetensors file with zeroed data."""
    header, offset = {}, 0
    sizes = {"F16": 2, "BF16": 2, "F32": 4}
    for name, (dtype, shape) in tensors.items():
        n = sizes[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + n]}
        offset += n
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(b"\0" * offset)


@pytest.fixture
def tiny_model(tmp_path):
    model = tmp_path / "tiny-llama"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "vocab_size": 100, "max_position_embeddings": 512,
    }))
    (model / "tokenizer.json").write_text("{}")
    write_safetensors(model / "model-00001-of-00002.safetensors",
                      {"embed.weight": ("BF16", [100, 64]), "layers.0.w": ("BF16", [64, 64])})
    write_safetensors(model / "model-00002-of-00002.safetensors",
                      {"layers.1.w": ("BF16", [64, 64]), "norm.weight": ("F32", [64])})
    return model
