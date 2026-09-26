# Testing NanoMesh on your own device

This walks through validating NanoMesh's estimates on a real machine, using an
HP EliteBook 840 G6 as the example. Every step works the same on any laptop or
desktop; see the Android section for phones.

The goal: compare what `nanomesh plan` *predicts* with what `nanomesh benchmark`
*measures*, and let those measurements calibrate future plans for the device.

## 1. Install NanoMesh

```sh
git clone https://github.com/Tonnymwaniki/NanoMesh
cd NanoMesh
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Linux / macOS:  source .venv/bin/activate
pip install -e .
```

## 2. Check the Device Passport

```sh
nanomesh scan-device
```

It should say **Recognised as `hp-elitebook-840-g6`**. If it doesn't, run
`nanomesh scan-device --json` and send the `vendor` and `model` fields; that's
how new devices get added to the database.

## 3. Get llama.cpp

Download a prebuilt release from https://github.com/ggml-org/llama.cpp/releases
(on Windows: the `bin-win-cpu-x64` zip) and unzip it, e.g. to `C:\llama.cpp`.
Then point NanoMesh at it:

```sh
# Windows (PowerShell)
$env:NANOMESH_LLAMA_CPP = "C:\llama.cpp"
# Linux / macOS
export NANOMESH_LLAMA_CPP=~/llama.cpp
```

`nanomesh benchmark` needs `llama-bench` and `llama-perplexity` from that folder.

## 4. Download a small model in several variants

Qwen2.5 1.5B is a good first test for an 8 GB laptop. From
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF download these into one folder,
e.g. `models/qwen1.5b/`:

- `qwen2.5-1.5b-instruct-fp16.gguf` (quality reference, ~3.1 GB)
- `qwen2.5-1.5b-instruct-q8_0.gguf`
- `qwen2.5-1.5b-instruct-q4_k_m.gguf`
- `qwen2.5-1.5b-instruct-q3_k_m.gguf`

## 5. Predict, then measure

```sh
nanomesh plan qwen2.5-1.5b          # note the predicted speeds (~) and quality
nanomesh benchmark models/qwen1.5b  # takes a few minutes; close other apps first
nanomesh plan qwen2.5-1.5b          # now shows measured values (✓)
nanomesh plan qwen2.5-7b            # other models are now calibrated (*) too
```

`benchmark` measures, for every file:

- prompt and generation speed (`llama-bench`)
- peak RAM
- quality: perplexity compared with the highest-precision file that fits in memory
  (`llama-perplexity` on a bundled public-domain text)

Results are saved to `~/.nanomesh/results.jsonl`; view them with `nanomesh results`.

## 6. Share the results

Send `~/.nanomesh/results.jsonl` together with the output of both `plan` runs.
The gap between predicted and measured numbers is exactly what needs tuning next.

## Android (Termux)

NanoMesh runs inside [Termux](https://termux.dev) (install it from F-Droid; the
Play Store build is outdated):

```sh
pkg install python git clang cmake rust   # rust: pydantic-core may need to compile
git clone https://github.com/Tonnymwaniki/NanoMesh && cd NanoMesh
pip install -e .
nanomesh scan-device
```

Build llama.cpp in Termux with its CMake instructions, set `NANOMESH_LLAMA_CPP`,
then follow steps 4–6 with a smaller model (Qwen2.5 0.5B for 4 GB phones).
On phones, NanoMesh budgets at most 45% of RAM for the model, because Android
kills memory-hungry apps well before RAM is full.
