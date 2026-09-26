# NanoMesh

**Give NanoMesh a model and a device; it finds the best way to run that model there.**

Quantization itself is a solved problem: llama.cpp, Olive, AIMET and others already do it well.
What's still slow and manual is deciding *which* variant to ship: INT8 or INT4? Will it fit
in 4 GB? How fast will it be on a ThinkPad T480? NanoMesh answers that question from the model's
architecture and the target's hardware profile, then drives llama.cpp to build and benchmark
the winner.

It is built first for the hardware people actually have: 4 GB Android phones, eight-year-old
laptops, Raspberry Pis and cheap VPSes.

## Install

```sh
pip install -e .            # add [dev] for pytest
```

To build and benchmark real GGUF files, install [llama.cpp](https://github.com/ggml-org/llama.cpp)
and point NanoMesh at it:

```sh
export NANOMESH_LLAMA_CPP=~/llama.cpp   # contains convert_hf_to_gguf.py and build/bin/
```

Planning (`plan`, `device`, `scan-device`) works without llama.cpp.

## Usage

### Device Passport: what can this machine run?

```sh
nanomesh scan-device                 # profile this machine
nanomesh devices                     # list the device database
nanomesh devices android             # search it
nanomesh device thinkpad-t480        # passport for a known device
```

```
╭──────────── DEVICE PASSPORT · Raspberry Pi 5 (8 GB) ────────────╮
│ CPU           Broadcom BCM2712 (4x Cortex-A76)                  │
│ RAM           8 GB                                              │
│ Memory BW     17.1 GB/s                                         │
│                                                                 │
│ AI COMPUTE CLASS  Entry-level Edge                              │
│ 🟢 Recommended  up to ~6.2B params (INT4)                       │
│ 🟡 Possible     ~6.2B – 8.4B params (INT4, tight)               │
│ 🔴 Not advised  above ~8.4B params                              │
│                                                                 │
│  Model size  Best variant  Memory  Speed       Runs on          │
│  1B          INT8          1.4 GB  ~9.9 tok/s  CPU              │
│  7B          INT4          4.7 GB  ~2.4 tok/s  CPU              │
│  13B         won't fit                                          │
╰─────────────────────────────────────────────────────────────────╯
```

### Plan: which variant should I ship?

`MODEL` can be a Hugging Face model directory, a `.safetensors`/`.gguf` file, a known model
name (`qwen2.5-7b`, `llama-3.1-8b`, …) or just a size (`7b`, `500m`).

```sh
nanomesh plan qwen2.5-7b --device hp-elitebook-840-g3
nanomesh plan llama-3.1-8b --device rtx-3060-12gb --min-speed 50
nanomesh plan ./my-model --device low-end-android-4gb --context 2048 --json
```

```
┏━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃    ┃ Variant     ┃ Memory  ┃ Smaller ┃ Speed      ┃ Quality  ┃ Runs on ┃
┡━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│    │ INT8 Q8_0   │ 8.1 GB  │ 47%     │ —          │ lossless │ ✕       │
│ •  │ INT5 Q5_K_M │ 5.6 GB  │ 64%     │ ~3.9 tok/s │ high     │ CPU     │
│ 🏆 │ INT4 Q4_K_M │ 4.9 GB  │ 69%     │ ~4.5 tok/s │ good     │ CPU     │
│ •  │ INT3 Q3_K_M │ 4.0 GB  │ 76%     │ ~5.6 tok/s │ fair     │ CPU     │
└────┴─────────────┴─────────┴─────────┴────────────┴──────────┴─────────┘
🏆 Recommended: INT4 (Q4_K_M) on CPU — 4.9 GB, 69% smaller than FP16, ~4.5 tok/s
```

Requirements you can set:

| Option | Meaning |
|---|---|
| `--context N` | Context length to reserve KV cache for (default 4096) |
| `--min-quality` | `lossless`, `high`, `good` (default), `fair`, `severe` |
| `--min-speed N` | Minimum generation speed in tokens/s |
| `--ram N` | Cap the memory the model may use, in GB |
| `--prefer` | Tie-break among valid variants: `quality` (default), `speed`, `size` |

`plan` exits with status 2 when no variant meets the requirements, so it can gate CI.

### Optimize: build the deployment package

```sh
nanomesh optimize ./my-model --device redmi-14c-4gb            # auto: winner + next Pareto point
nanomesh optimize ./my-model -q int4,int8 -o out/ --dry-run    # just print the llama.cpp commands
```

Produces:

```
my-model-nanomesh-redmi-14c-4gb/
├── model-Q4_K_M.gguf
├── model-Q3_K_M.gguf
├── plan.json          # every variant's estimates and why it won or lost
├── benchmark.json     # measured speed and peak RAM (when llama-bench is available)
├── config.json, tokenizer.json, …
└── README.md
```

### Benchmark: measure instead of estimate

```sh
nanomesh benchmark model-Q4_K_M.gguf
nanomesh benchmark ./my-model-nanomesh-local/ --save results.json
```

Runs `llama-bench` and records prompt/generation throughput and peak resident memory.

## How the estimates work

- **Memory** = weights (params × effective bits-per-weight of the GGUF format)
  + KV cache (2 × layers × KV heads × head dim × context × 2 bytes) + ~0.3 GB runtime overhead.
- **Budget** = 92% of VRAM on a discrete GPU; 45% of RAM on phones (Android kills apps early);
  70% of RAM elsewhere; 85% of *free* RAM for the local machine. Variants using more than 90%
  of the budget are only chosen when nothing else fits.
- **Speed** — token generation is memory-bandwidth bound, so
  tok/s ≈ bandwidth × efficiency ÷ bytes read per token. Devices without bandwidth data show `—`;
  run `nanomesh benchmark` on them.
- **Quality** tiers follow llama.cpp's published perplexity deltas per format. They are
  typical, not measured on your model.

Treat estimates as a way to narrow the search; `benchmark` is the ground truth.

## Roadmap

- [x] Model analyzer (safetensors headers, no torch), device scan and Device Passport
- [x] Device database, planner with Pareto selection, GGUF build + llama-bench measurement
- [ ] Quality measurement (perplexity on a held-out set) for each built variant
- [ ] ONNX / OpenVINO / LiteRT export; AWQ/GPTQ; vision models
- [ ] Shared benchmark database: measured results feed back into estimates per device
- [ ] Android on-device benchmarking, NPU profiles
- [ ] NanoMesh Cloud: upload a model, pick a device, download the optimized package

## Development

```sh
pip install -e '.[dev]'
pytest
```

Adding a device: append a profile to `src/nanomesh/data/devices.json`. Only include specs you
can source; leave `memory_bandwidth_gbps` out rather than guessing.
