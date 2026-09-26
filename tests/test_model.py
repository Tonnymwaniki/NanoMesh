import pytest

from nanomesh.model import analyze, parse_size


def test_analyze_directory_counts_params_across_shards(tiny_model):
    info = analyze(str(tiny_model))
    assert info.params == 100 * 64 + 64 * 64 * 2 + 64
    assert info.source_dtype == "BF16"
    assert info.architecture == "LlamaForCausalLM"
    assert info.num_kv_heads == 2
    assert info.head_dim == 16
    # 2 (K and V) * layers * kv heads * head dim * 2 bytes
    assert info.kv_bytes_per_token() == 2 * 2 * 2 * 16 * 2


def test_analyze_known_model_and_size():
    qwen = analyze("Qwen2.5-7B")
    assert qwen.source == "known"
    assert round(qwen.params_b, 1) == 7.6
    assert analyze("7b").params == 7_000_000_000
    assert analyze("500m").params == 500_000_000


@pytest.mark.parametrize("spec", ["", "seven", "7x"])
def test_parse_size_rejects_garbage(spec):
    assert parse_size(spec) is None


def test_analyze_unknown_raises():
    with pytest.raises(ValueError):
        analyze("definitely-not-a-model")


def test_analyze_gguf_reads_header(tmp_path):
    import struct

    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    kv = [("general.architecture", 8, s("llama")), ("general.name", 8, s("tiny")),
          ("llama.block_count", 4, struct.pack("<I", 2)),
          ("llama.embedding_length", 4, struct.pack("<I", 64)),
          ("llama.attention.head_count", 4, struct.pack("<I", 4)),
          ("llama.attention.head_count_kv", 4, struct.pack("<I", 2)),
          ("tokenizer.ggml.scores", 9, struct.pack("<IQ", 6, 3) + struct.pack("<3f", 0, 0, 0)),
          ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + s("a") + s("b"))]
    tensors = [("token_embd.weight", [64, 100]), ("blk.0.attn_q.weight", [64, 64])]
    data = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, t, v in kv:
        data += s(key) + struct.pack("<I", t) + v
    for name, dims in tensors:
        data += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", 1, 0)
    f = tmp_path / "m.gguf"
    f.write_bytes(data)
    info = analyze(str(f))
    assert info.params == 64 * 100 + 64 * 64
    assert (info.name, info.architecture, info.num_layers, info.num_kv_heads, info.head_dim) == ("tiny", "llama", 2, 2, 16)
