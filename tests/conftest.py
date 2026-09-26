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


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Keep tests away from the real ~/.nanomesh results store."""
    monkeypatch.setenv("NANOMESH_HOME", str(tmp_path / "nanomesh-home"))
    return tmp_path / "nanomesh-home"


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


def write_gguf(path, file_type=1, tensors=(("token_embd.weight", [64, 100]), ("blk.0.attn_q.weight", [64, 64]))):
    """Write a header-only GGUF file (no tensor data) that NanoMesh can parse."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    kv = [("general.architecture", 8, s("llama")), ("general.name", 8, s("tiny")),
          ("general.file_type", 4, struct.pack("<I", file_type)),
          ("llama.block_count", 4, struct.pack("<I", 2)),
          ("llama.embedding_length", 4, struct.pack("<I", 64)),
          ("llama.attention.head_count", 4, struct.pack("<I", 4)),
          ("llama.attention.head_count_kv", 4, struct.pack("<I", 2)),
          ("tokenizer.ggml.scores", 9, struct.pack("<IQ", 6, 3) + struct.pack("<3f", 0, 0, 0)),
          ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + s("a") + s("b"))]
    data = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, t, v in kv:
        data += s(key) + struct.pack("<I", t) + v
    for name, dims in tensors:
        data += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", 1, 0)
    path.write_bytes(data)
    return path
