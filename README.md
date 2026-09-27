# NanoMesh

[![tests](https://github.com/Tonnymwaniki/NanoMesh/actions/workflows/tests.yml/badge.svg)](https://github.com/Tonnymwaniki/NanoMesh/actions/workflows/tests.yml)

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
Windows, Linux, macOS and Android (via Termux) are supported; see
[docs/testing-on-your-device.md](docs/testing-on-your-device.md) for a step-by-step walkthrough.

## Usage

### Get a working local AI: search, pull, serve

```sh
nanomesh search qwen2.5 coder 7b                 # Hugging Face GGUFs, sized for this machine
nanomesh pull Qwen/Qwen2.5-1.5B-Instruct-GGUF    # downloads the best file for this machine
nanomesh serve ~/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q8_0.gguf
```

- **`search`** matches every word against repository names and, for each repository, picks the file
  that suits this device: its download size, memory, predicted speed and quality. Models you've
  benchmarked use your measurements.
- **`pull`** shows the download size and asks first (`--yes` to skip). It saves to `C:\models` (if it
  exists) or `~/models` (`NANOMESH_MODELS_DIR` to change), and resumes after a dropped connection or
  a Ctrl+C when run again. Each file is checked against Hugging Face's SHA-256. `--file Q4_K_M` picks
  a format. Gated models need `HF_TOKEN`.
- **`serve`** starts llama.cpp's `llama-server` in the background, on 127.0.0.1 only, with the
  thread count `nanomesh tune` measured as best. It prints ready-to-paste snippets for Python,
  JavaScript, curl/PowerShell and VS Code's Continue extension. The endpoint is OpenAI-compatible
  (`http://127.0.0.1:8080/v1`). `nanomesh serve --status` shows running servers and
  `nanomesh serve --stop` stops them.

### Fit Cards: speech, vision, embeddings and text models on this device

```sh
nanomesh fit whisper                       # the best Whisper size for this machine
nanomesh fit yolo11n -d redmi-14c-4gb      # a specific model on a specific device
nanomesh search --task speech              # popular speech models, sized for this machine
nanomesh search swahili --task speech --source all   # Hugging Face and Kaggle
nanomesh config --data-price 100 --currency KES      # show what downloads cost
```

A **Fit Card** is NanoMesh's answer for any model on any device:

- **Fits:** memory against the device's budget.
- **Speed**, in the unit that matters for the task: × real-time for speech, images/s for vision,
  sentences/s for embeddings, tok/s for text. Each figure is marked measured, calibrated or estimated.
- **Quality:** published figures, such as COCO mAP or ImageNet top-1, or where the model sits in its family.
- **Download size**, and its cost if you've set a data price.
- **Battery:** hours of audio or number of images per full charge, from a battery run on this device.
- **Licence.**
- **A better-fitting model of the same family**, when there is one.
- **How to run it.**

NanoMesh knows these families: Whisper (tiny to large-v3, turbo, distil), YOLOv8 and YOLO11, image
classifiers (MobileNetV3, EfficientNet, ResNet, ViT) and embedding models (MiniLM, BGE, E5, nomic).
Text models go through the planner. Models outside these families are sized from their parameter count.

Estimates come from published compute figures and the device's CPU. Benchmarking any model of a kind
turns its card into a measurement, and calibrates the rest of that kind on this device:

```sh
pip install onnxruntime                    # or: pip install -e ".[onnx]"
nanomesh benchmark yolo11n.onnx            # ONNX Runtime: vision and embedding models
nanomesh benchmark ggml-small.bin          # whisper.cpp (NANOMESH_WHISPER_CPP, or C:\whisper.cpp, ~/whisper.cpp)
```

Kaggle search needs your Kaggle API token: on kaggle.com go to Settings > API > Create New Token, then
save `kaggle.json` in `~/.kaggle/`.

### Datasets: what's in it, and what it costs this device

```sh
nanomesh data search swahili                         # Hugging Face datasets, sized for this machine
nanomesh data search --task translation --source all # add Kaggle (needs your kaggle.json)
nanomesh data card stanfordnlp/imdb                  # rows, columns, preview, licence, download, memory
nanomesh data card ./my-chats.jsonl --for-model qwen2.5-1.5b --epochs 2
```

A **Dataset Card** is the Fit Card for data. It shows:

- **What's in it:** rows per split, the columns and their types, a preview of the first rows, languages,
  licence, and the type (text, chat, image, audio, tabular).
- **What it costs this device:** download size and its data cost, disk space, and memory if loaded
  whole, with advice to stream it or take a slice when it's too big.
- **With `--for-model`, fine-tuning:** whether it fits (full, LoRA or QLoRA, from `train-plan`), the
  tokens, and hours on this device against a free Colab/Kaggle T4 and an A100.

Sizes and previews come from Hugging Face's dataset viewer. Local CSV, TSV, JSONL and JSON files and
folders work too, and chat-format rows (role/content messages) are recognised.

### Project: what AI does this codebase use, and what could run locally?

```sh
nanomesh project .                       # this project, sized for this machine
nanomesh project ~/code/app -d redmi-14c-4gb --json
```

It reads source files, notebooks, `requirements.txt`, `pyproject.toml` and `package.json`, and never runs
anything. It finds:

- **Cloud AI calls:** OpenAI, Anthropic, Google Gemini, Mistral, Cohere, Groq and LangChain, each with its
  file and line. Each is classified as chat, embeddings, speech-to-text or image generation.
- **Local AI:** transformers, sentence-transformers, llama.cpp and Ollama.
- **AI dependencies and model files** (`.gguf`, `.safetensors`, `.onnx`…) in the repository.

For each kind of cloud use it suggests a local replacement sized for the device:

- the biggest Qwen2.5 chat model that runs at a usable speed there, and which database devices can also run it
- a small embedding model
- Whisper for speech-to-text

It also shows the code change. `nanomesh serve` speaks the OpenAI API, so for the OpenAI SDK and LangChain
that change is a base URL and a model name. It's honest about the limits: GPT-4-class models are far stronger
on hard reasoning, and switching embedding models means re-embedding what's stored.

To stay useful and safe, it:

- only counts code: calls quoted in strings, docstrings and comments don't count
- skips tests unless you pass `--include-tests`
- never opens `.env` files and masks API keys in the snippets it shows

### Conditions, heat, battery and threads

`nanomesh scan-device` ends with a **Right now** panel: power source and battery, power plan and
power mode, CPU speed relative to its rating, temperature (Linux, Android and some laptops; Windows
usually hides it), background CPU load and battery health, plus what to change.

```sh
nanomesh sustained model.gguf --minutes 3   # speed over minutes: thermal slowdown
nanomesh tune model.gguf                    # fastest CPU thread count for this machine
```

`sustained` keeps generating and reports burst vs sustained speed. Unplugged, it also measures
battery draw, energy per token and how many hours of generation a full charge gives. `tune` tries
several thread counts; on CPUs with hyper-threading the default is often not the fastest. Both
feed `nanomesh plan`'s advice.

### Use NanoMesh from your coding agent (MCP)

NanoMesh runs as a local [MCP](https://modelcontextprotocol.io) server, so Claude Code, Cursor,
VS Code agents and other MCP clients can ask it about *this* machine before recommending a model,
quantization or training setup:

```sh
nanomesh mcp --config     # prints ready-to-paste setup for Claude Code, VS Code and Cursor
```

| Tool | Answers |
|---|---|
| `device_passport` | What can this machine (or a device from the database) run? |
| `current_conditions` | Battery, power mode, CPU speed, temperature, load: what's slowing things down now |
| `plan_model` | Which variant of a model to run, with measured speeds where they exist |
| `list_local_models` | Models already downloaded (Hugging Face, LM Studio, Ollama, folders) and what fits |
| `benchmark_results` | Everything measured here |
| `environment_doctor` | Broken or mismatched Python/PyTorch/GPU/llama.cpp setups, with fixes |
| `training_plan` | Will fine-tuning fit: full, LoRA or QLoRA |
| `analyze_project` | The AI a codebase uses (file:line) and what could run locally instead, with the code change |
| `fit_card` | Fit Card of any model (speech, vision, embeddings, text) on this machine or a database device |
| `search_datasets` / `dataset_card` | Datasets from Hugging Face and Kaggle (or local files): rows, preview, licence, download, memory, and fine-tuning time here vs the cloud |
| `search_models` | Hugging Face GGUFs sized for this machine; with a task, speech/vision/embedding models from Hugging Face and Kaggle with Fit Cards |
| `download_model` | Download a model (a background job), resuming and checksum-verified |
| `benchmark_model` | Measure a GGUF, ONNX or Whisper model here (a background job) so plans and Fit Cards use real numbers |
| `job_status` / `cancel_job` | Progress of downloads and benchmarks |
| `start_model_server` / `model_server_status` / `stop_model_server` | Serve a model as a local OpenAI-compatible API, with connection snippets |

So an agent can go from "I want a coding model on this laptop" to a running endpoint: search,
tell you the download size, download, benchmark, serve. Tools that act (download, benchmark,
serve, stop, cancel) are marked as such, so clients ask you first; the rest are read-only.

It runs over stdio on your machine. Only `search_models` and `download_model` go online, to
Hugging Face.

### Doctor, local models and training plans

```sh
nanomesh doctor                        # PyTorch that can't see the GPU, CUDA wheels on CPU laptops, missing llama.cpp…
nanomesh models C:\models              # models on disk + which fit this machine
nanomesh train-plan qwen2.5-7b -d rtx-3060-12gb --seq-len 2048
```

`train-plan` uses standard memory accounting for mixed-precision AdamW (16 bytes per parameter
for full fine-tuning; a frozen bf16 or 4-bit base plus adapters for LoRA and QLoRA; activations
with gradient checkpointing; the loss layer). QLoRA is marked unavailable without an NVIDIA GPU.

### Dashboard

```sh
nanomesh dashboard
```

Opens a local web dashboard in your browser: your Device Passport, everything you've measured,
an interactive planner for any model and device, and the device database. It runs on
`127.0.0.1` only, needs no internet connection, and your data never leaves the machine.

### Device Passport: what can this machine run?

`scan-device` recognises the exact machine (e.g. "HP EliteBook 840 G3", "ThinkPad T480",
"Redmi 14C") from its firmware or Android system properties, and merges in the curated
database profile for it: live facts like RAM win, the database fills in what can't be probed,
such as memory bandwidth.

```sh
nanomesh scan-device                 # profile and recognise this machine
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
| `--min-quality` | A percentage like `95`, or a tier: `lossless`, `high`, `good` (default), `fair`, `severe` |
| `--min-speed N` | Minimum generation speed in tokens/s |
| `--ram N` | Cap the memory the model may use, in GB |
| `--prefer` | `balanced` (default: best quality that still reaches ~5 tok/s, else the fastest), `quality`, `speed`, `size` |

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
nanomesh benchmark ./models/qwen1.5b/            # every .gguf in the folder
nanomesh benchmark model-Q4_K_M.gguf --reference model-F16.gguf
nanomesh results                                  # everything measured on this machine
```

Every benchmark also records the conditions it ran under: plugged in or on battery, power mode,
CPU speed and temperature where the system reports them. Calibration prefers plugged-in runs,
because laptops slow down on battery.

By default `benchmark` measures the **steady state**: before measuring, it generates until the
speed settles (at least 90 s for the first file, 20 s for the rest). Laptops boost for their first
minute or so: an HP EliteBook 840 G6 ran Qwen2.5-1.5B at 22 tok/s for about 70 s, then settled at
15.5. The cold-start speed is recorded too. `--quick` skips the warm-up; calibration prefers
steady results over quick ones.

For each file, `benchmark` measures prompt/generation speed and peak RAM (`llama-bench`)
and quality: perplexity relative to the highest-precision variant that fits in memory
(`llama-perplexity`, on a bundled public-domain text; `--eval-text` to use your own).

Results are appended to `~/.nanomesh/results.jsonl` (override with `NANOMESH_HOME`), and
from then on `plan` uses them:

- **✓ measured**: the exact speed/quality of variants you benchmarked
- **\* calibrated**: other variants and models on the same device, using the roofline
  your benchmarks revealed (the memory bandwidth the device actually delivers, and the
  CPU's ceiling for unpacking low-bit weights), and this model's measured quality loss
- **~ estimate**: nothing measured yet

## How the estimates work

- **Memory** = weights (params × effective bits-per-weight of the GGUF format)
  + KV cache (2 × layers × KV heads × head dim × context × 2 bytes) + ~0.3 GB runtime overhead.
- **Budget** = 92% of VRAM on a discrete GPU; 45% of RAM on phones (Android kills apps early);
  70% of RAM elsewhere; 85% of *free* RAM for the local machine. Variants using more than 90%
  of the budget are only chosen when nothing else fits.
- **Speed**: token generation is usually memory-bandwidth bound, so
  tok/s ≈ bandwidth × efficiency ÷ bytes read per token. On weak CPUs, low-bit formats hit a
  compute ceiling instead: on an HP EliteBook 840 G6, Qwen2.5-1.5B ran at 15 tok/s at both INT4
  and INT3. After a benchmark, speeds are min(bandwidth ÷ bytes, compute ceiling ÷ params),
  both learned from the device's runs.
- **Quality** is reference perplexity ÷ variant perplexity, as a percentage. Before measuring,
  each format uses a conservative typical value from llama.cpp's published perplexity deltas;
  tiers are derived from it (lossless ≥ 99.8%, high ≥ 98.5%, good ≥ 96%, fair ≥ 90%).
  Small models lose much more: Qwen2.5-1.5B kept 92.5% at INT4 and 78.5% at INT3, about 3×
  the typical loss. Once some variants are measured, the rest are scaled by that factor.

Treat estimates as a way to narrow the search; `benchmark` is the ground truth.

## Roadmap

- [x] Model analyzer (safetensors headers, no torch), device scan and Device Passport
- [x] Device database, planner with Pareto selection, GGUF build + llama-bench measurement
- [x] Exact device recognition (Windows, Linux, macOS, Android) matched to the database
- [x] Quality measurement (perplexity vs reference), local results store, per-device calibration
- [x] First real-hardware validation (HP EliteBook 840 G6): roofline speed model and
      per-model quality calibration came out of it
- [ ] ONNX / OpenVINO / LiteRT export; AWQ/GPTQ; vision models
- [x] Agent-ready: `doctor`, `models`, `train-plan` and a local MCP server for coding agents
- [x] Search, download and serve: from "I want a model" to a local OpenAI-compatible endpoint
- [x] Project scan: the AI a codebase uses and what could run locally instead
- [x] Fit Cards for speech, vision and embedding models; task search on Hugging Face and Kaggle
- [x] Datasets: find, size and preview (Hugging Face, Kaggle, local) and whether fine-tuning fits this device
- [ ] VS Code extension panel over the same tools
- [ ] Shared benchmark database: upload `results.jsonl` so every user of a device benefits
- [ ] Android on-device benchmarking, NPU profiles
- [ ] NanoMesh Cloud: upload a model, pick a device, download the optimized package

## Development

```sh
pip install -e '.[dev]'
pytest
```

Adding a device: append a profile to `src/nanomesh/data/devices.json`, with `match` regexes for
the vendor/model strings `nanomesh scan-device --json` reports on it. Only include specs you
can source; leave `memory_bandwidth_gbps` out rather than guessing.
