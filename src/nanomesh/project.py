"""`nanomesh project`: find the AI in a codebase and what could run locally instead.

Reads source files and manifests (never runs anything, never opens .env
files), finds calls to cloud AI APIs and local model libraries, and for each
cloud use suggests a local model sized for the device, with the code change
to make. NanoMesh's own server speaks the OpenAI API, so most swaps are a
base URL and a model name.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

from pydantic import BaseModel

from nanomesh import results as store
from nanomesh.devices import load_devices
from nanomesh.hardware import GB, DeviceProfile
from nanomesh.model import analyze
from nanomesh.planner import USABLE_TOKENS_PER_S, Requirements, plan

SOURCE_EXT = {".py", ".ipynb", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".kt", ".kts", ".java", ".swift", ".dart"}
MODEL_EXT = {".gguf", ".safetensors", ".onnx", ".pt", ".pth", ".tflite", ".mlmodel"}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "env", "__pycache__", "dist", "build", ".next", ".tox",
             "site-packages", ".mypy_cache", ".pytest_cache", "target", ".idea", ".vscode"}
MAX_FILE_BYTES = 1_000_000
MAX_FILES = 20_000
MAX_LOCATIONS = 5

# Secrets pasted into code must never be echoed back in a report.
SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{12,}|sk-ant-[A-Za-z0-9_\-]{12,}|AIza[0-9A-Za-z_\-]{20,}|hf_[A-Za-z0-9]{20,}"
                       r"|gsk_[A-Za-z0-9]{20,}|xai-[A-Za-z0-9]{20,})")

# (provider, how it's used in code); first match on a line wins.
PROVIDERS: list[tuple[str, re.Pattern]] = [
    ("LangChain + OpenAI", re.compile(r"langchain_openai|langchain/openai|\bChatOpenAI\b|\bOpenAIEmbeddings\b")),
    ("OpenAI", re.compile(r"\bfrom openai\b|\bimport openai\b|['\"]openai['\"]|\bOpenAI\(|\bAzureOpenAI\(|"
                          r"\.chat\.completions\.create|\.responses\.create|\bopenai\.")),
    ("Anthropic", re.compile(r"\banthropic\b|@anthropic-ai/sdk|\bAnthropic\(|\.messages\.create")),
    ("Google Gemini", re.compile(r"google\.generativeai|google\.genai|@google/generative-ai|@google/genai|"
                                 r"GoogleGenerativeAI|[gG]enerativeModel\(|\bgenai\.|@google-cloud/vertexai|"
                                 r"\bVertexAI\b|firebase/(ai|vertexai)|com\.google\.firebase\.(ai|vertexai)|"
                                 r"com\.google\.ai\.client\.generativeai|Firebase\.(ai|vertexAI)\b|FirebaseAI|"
                                 r"firebaseAI\(|package:(firebase_ai|firebase_vertexai|google_generative_ai)/")),
    ("Mistral", re.compile(r"\bmistralai\b|@mistralai/")),
    ("Cohere", re.compile(r"\bcohere\b")),
    ("Groq", re.compile(r"\bfrom groq\b|\bimport groq\b|groq-sdk|\bGroq\(")),
    ("Ollama (local)", re.compile(r"\bollama\b")),
    ("llama.cpp (local)", re.compile(r"\bLlama\(\s*model_path|llama_cpp\.Llama\(|\bgetLlama\(|"
                                    r"create_chat_completion\(")),
    ("Transformers (local)", re.compile(r"\bfrom transformers\b|\bimport transformers\b|@huggingface/transformers|"
                                       r"@xenova/transformers|\.from_pretrained\(|\bpipeline\(")),
    ("sentence-transformers (local)", re.compile(r"sentence_transformers|SentenceTransformer\(")),
]
LOCAL = {"Ollama (local)", "llama.cpp (local)", "Transformers (local)", "sentence-transformers (local)"}

# What a call does, from the API it uses or the model it names.
TASKS: list[tuple[str, re.Pattern]] = [
    ("embeddings", re.compile(r"embeddings?\.create|\.embed_content\(|\.embed\(|\.embed_(query|documents)\(|"
                              r"Embeddings\(|SentenceTransformer\(|\.encode\(|\.embedContent\(|"
                              r"\.batchEmbedContents\(")),
    ("speech-to-text", re.compile(r"audio\.transcriptions|\bwhisper\b|speech_to_text|automatic-speech-recognition",
                                  re.I)),
    ("image generation", re.compile(r"images\.generate|dall-e|gpt-image|imagen|text-to-image|StableDiffusion", re.I)),
    ("chat", re.compile(r"chat\.completions|responses\.create|messages\.create|generate_content|ChatOpenAI|"
                        r"ChatAnthropic|\.chat\(|\.invoke\(|create_chat_completion\(|\bLlama\(|"
                        r"\.generateContent(Stream)?\(|\.sendMessage(Stream)?\(|\.startChat\(")),
]
MODEL_BODY = (r"""((?:gpt-|o[134]-|o[134](?=["'`])|chatgpt-|claude-|gemini-|gemma-|text-embedding-|"""
              r"""whisper-|dall-e-|gpt-image|mistral-|command-|llama-?3|mixtral)[A-Za-z0-9.\-:_]*)""")
MODEL_NAME_RE = re.compile(r"""["'`]""" + MODEL_BODY + r"""["'`]""")
# GEMINI_MODEL = "gemini-2.5-flash", later used by name: getGenerativeModel({ model: GEMINI_MODEL })
CONSTANT_RE = re.compile(r"""\b([A-Z][A-Z0-9_]{2,})\s*[:=][^=\n]*?["'`]""" + MODEL_BODY + r"""["'`]""")
PRETRAINED_RE = re.compile(r"""(?:from_pretrained|SentenceTransformer|pipeline)\(\s*(?:[^)]*?model\s*=\s*)?"""
                           r"""["']([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+|all-[A-Za-z0-9\-]+)["']""")
IMPORT_RE = re.compile(r"^\s*(import\s|from\s+\S+\s+import\s|export\s+\*\s+from\s)|^\s*(const|let|var)\s+"
                       r"[\w{}\s,]+=\s*require\(")
DEFINITION_RE = re.compile(r"^\s*(export\s+)?(async\s+)?(def|function|class)\s")
# Where a file's code runs decides what "local" can mean for it.
MOBILE_RE = re.compile(r"\.(kt|kts|java|swift|dart)$|(^|/)(android|ios)/")
SERVER_PATH_RE = re.compile(r"(^|/)(functions|server|backend|api|cloud[-_]?functions|lambdas?|workers?)/")
SERVER_CODE_RE = re.compile(r"firebase-functions|functions\.https|\bonRequest\(|\bonCall\(|\bexpress\(\)|"
                            r"FastAPI\(|Flask\(|from django|@app\.(get|post|route)|exports\.handler|Deno\.serve|"
                            r"@google-cloud/functions-framework|functions_framework")
ANDROID_AI_RE = re.compile(r"com\.google\.firebase:firebase-(ai|vertexai)[\w-]*|firebase-(ai|vertexai)\b|"
                           r"com\.google\.ai\.client\.generativeai:[\w-]+|com\.google\.mediapipe:tasks-genai|"
                           r"com\.google\.mlkit:genai[\w-]*|org\.tensorflow:tensorflow-lite[\w-]*|"
                           r"com\.microsoft\.onnxruntime:[\w-]+")
FLUTTER_AI_RE = re.compile(r"^\s+(google_generative_ai|firebase_ai|firebase_vertexai|tflite_flutter|dart_openai|"
                           r"openai_dart|langchain\w*|llama_cpp_dart|fllama)\s*:", re.M)
AI_PACKAGES = {
    "openai", "anthropic", "google-generativeai", "google-genai", "mistralai", "cohere", "groq", "langchain",
    "langchain-openai", "langchain-anthropic", "langchain-community", "llama-index", "llama-cpp-python", "ollama",
    "transformers", "sentence-transformers", "torch", "tensorflow", "onnxruntime", "openai-whisper",
    "faster-whisper", "diffusers", "accelerate", "peft", "vllm",
    "@anthropic-ai/sdk", "@google/generative-ai", "@google/genai", "@mistralai/mistralai", "cohere-ai", "groq-sdk",
    "@langchain/openai", "@langchain/core", "langchain", "ollama", "node-llama-cpp", "@huggingface/transformers",
    "@xenova/transformers", "onnxruntime-node", "onnxruntime-web", "@huggingface/inference", "ai", "@ai-sdk/openai",
    "@google-cloud/vertexai", "@genkit-ai/googleai", "genkit", "@google-cloud/aiplatform", "google-cloud-aiplatform",
    "vertexai",
}
SERVER_TARGET, MOBILE_TARGET = "cheap-vps-4gb", "low-end-android-4gb"

# Local chat models to suggest, largest first, with their official GGUF repositories.
CHAT_LADDER = [("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct-GGUF"), ("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct-GGUF"),
               ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct-GGUF"), ("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")]


class Location(BaseModel):
    file: str
    line: int
    code: str


class Usage(BaseModel):
    provider: str
    task: str
    model: str | None = None
    local: bool
    where: str = "app"  # server | mobile | app (laptop, desktop, scripts)
    location: Location


class Alternative(BaseModel):
    model: str
    repo: str | None = None
    variant: str | None = None
    memory_gb: float | None = None
    tokens_per_s: float | None = None
    speed_source: str | None = None
    runs_on: list[str] = []  # database devices that run it at a usable speed
    target: str | None = None  # the device it was sized for
    how: str  # how to run it


class Finding(BaseModel):
    provider: str
    task: str
    where: str = "app"
    models: list[str]
    calls: int
    locations: list[Location]
    local: bool
    alternative: Alternative | None = None
    swap: str | None = None  # code change
    caveats: list[str] = []


class ProjectReport(BaseModel):
    path: str
    device: str
    files_scanned: int
    findings: list[Finding]
    dependencies: dict[str, list[str]]
    model_files: list[dict]
    summary: str


# ---- scanning ----

def _files(root: Path):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.startswith(".env"):
                continue  # secrets live there; never read them
            count += 1
            if count > MAX_FILES:
                return
            yield Path(dirpath) / name


def _lines(path: Path) -> list[str]:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return []
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if path.suffix == ".ipynb":
        try:
            cells = json.loads(text).get("cells", [])
            return [ln.rstrip("\n") for c in cells if c.get("cell_type") == "code" for ln in c.get("source", [])]
        except (ValueError, AttributeError):
            return []
    return text.splitlines()


def _strip_strings(line: str) -> str:
    """The line with string contents blanked, so a call counts only when it's
    code, not text (docs, examples, patterns like this scanner's own)."""
    out, quote, i = [], None, 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\":
                out.append("  ")
                i += 2
                continue
            if c == quote:
                quote = None
                out.append(c)
            else:
                out.append(" ")
        else:
            if c in "\"'`":
                quote = c
            elif c == "#" or line.startswith("//", i):
                break  # comment
            out.append(c)
        i += 1
    return "".join(out)


TEST_FILE_RE = re.compile(r"(^|/)(tests?|__tests__|spec|e2e)/|(^|/)test_[^/]*\.py$|_test\.py$|"
                          r"\.(test|spec)\.[jt]sx?$")


def _redact(line: str) -> str:
    return SECRET_RE.sub(lambda m: m.group(0)[:3] + "…[redacted]", line.strip())[:160]


def scan(root: Path, include_tests: bool = False) -> tuple[list[Usage], dict[str, list[str]], list[dict], int]:
    return _scan(root, include_tests)[:4]


def _scan(root: Path, include_tests: bool = False):
    usages, deps, model_files, scanned = [], {"python": [], "node": [], "android": [], "flutter": []}, [], 0
    mentioned: dict[str, set[str]] = {}
    for path in _files(root):
        rel = path.relative_to(root).as_posix()
        if not include_tests and TEST_FILE_RE.search(rel):
            continue  # mocked calls in tests aren't the product's AI use
        ext = path.suffix.lower()
        if ext in MODEL_EXT:
            try:
                model_files.append({"file": rel, "format": ext[1:], "size_gb": round(path.stat().st_size / GB, 2)})
            except OSError:
                pass
            continue
        if path.name == "package.json":
            deps["node"] += _node_deps(path)
        elif path.name == "pyproject.toml":
            deps["python"] += _pyproject_deps(path)
        elif re.fullmatch(r"requirements.*\.txt", path.name):
            deps["python"] += _requirements(path)
        elif re.fullmatch(r"build\.gradle(\.kts)?|libs\.versions\.toml", path.name):
            deps["android"] += [m.group(0) for m in ANDROID_AI_RE.finditer("\n".join(_lines(path)))]
        elif path.name == "pubspec.yaml":
            deps["flutter"] += FLUTTER_AI_RE.findall("\n".join(_lines(path)))
        if ext not in SOURCE_EXT:
            continue
        scanned += 1
        lines = _lines(path)
        # A file's imports tell which provider an unqualified call belongs to.
        file_provider = next((p for p, rx in PROVIDERS for ln in lines[:200] if rx.search(ln)), None)
        where = "mobile" if MOBILE_RE.search(rel) else "server" if SERVER_PATH_RE.search(rel) or any(
            SERVER_CODE_RE.search(ln) for ln in lines[:300]) else "app"
        constants = {m.group(1): m.group(2) for ln in lines if (m := CONSTANT_RE.search(ln))}
        mentioned[rel] = {m.group(1) for ln in lines for m in MODEL_NAME_RE.finditer(ln)}
        in_docstring = False
        for i, line in enumerate(lines, 1):
            if ext == ".py" or ext == ".ipynb":
                triple = line.count('"""') + line.count("'''")
                was_in = in_docstring
                in_docstring ^= triple % 2 == 1
                if was_in or (triple and in_docstring):
                    continue  # inside a multi-line string
            if len(line) > 1000 or IMPORT_RE.search(line) or DEFINITION_RE.search(line):
                continue  # importing or defining isn't using
            code = _strip_strings(line)
            if not code.strip():
                continue
            provider = next((p for p, rx in PROVIDERS if rx.search(code)), None)
            task = next((t for t, rx in TASKS if rx.search(code)), None)
            pretrained = PRETRAINED_RE.search(line) if re.search(r"from_pretrained|SentenceTransformer|pipeline",
                                                                  code) else None
            if task is None and pretrained:
                # Loading a named model locally: a language model unless it's an embedder.
                task = "embeddings" if re.search(r"SentenceTransformer|embed|MiniLM|bge-|e5-", line, re.I) else "chat"
            if not task or not (provider or pretrained or file_provider):
                continue
            model = pretrained.group(1) if pretrained else _model_near(lines, i - 1, constants)
            provider = provider or file_provider or _provider_for_model(model or "")
            if not provider:
                continue
            usages.append(Usage(provider=provider, task=task, model=model, local=provider in LOCAL, where=where,
                                location=Location(file=rel, line=i, code=_redact(line))))
    deps = {k: sorted(set(v)) for k, v in deps.items() if v}
    # mentioned: models named anywhere in each file, for calls that get theirs through a variable
    return usages, deps, model_files, scanned, mentioned


FUNCTION_START_RE = re.compile(r"^\s*(export\s+)?(async\s+|suspend\s+|private\s+|public\s+|static\s+)*"
                               r"(def|function|fun|func|class)\s|=>\s*\{?\s*$")


def _model_near(lines: list[str], idx: int, constants: dict[str, str]) -> str | None:
    """The model a call uses: on its line; below it when the call continues
    over several lines (create(\n  model="gpt-4o", ...)); or above it in the
    same function (model = getGenerativeModel(...) then model.generateContent()).
    Literally or through a constant. None when it's passed in from elsewhere."""
    def named(ln: str) -> str | None:
        if m := MODEL_NAME_RE.search(ln):
            return m.group(1)
        name = next((c for c in constants if re.search(rf"\b{c}\b", ln)), None)
        return constants[name] if name else None

    if found := named(lines[idx]):
        return found
    code = _strip_strings(lines[idx])
    if code.count("(") > code.count(")") or code.count("{") > code.count("}"):
        for ln in lines[idx + 1:idx + 6]:
            if found := named(ln):
                return found
    for ln in reversed(lines[max(0, idx - 15):idx]):
        if FUNCTION_START_RE.search(ln):
            break  # don't borrow a model from outside this function
        if found := named(ln):
            return found
    return None


def _provider_for_model(name: str) -> str | None:
    n = name.lower()
    for prefix, provider in (("gpt", "OpenAI"), ("o1", "OpenAI"), ("o3", "OpenAI"), ("o4", "OpenAI"),
                             ("text-embedding", "OpenAI"), ("whisper-", "OpenAI"), ("dall-e", "OpenAI"),
                             ("claude", "Anthropic"), ("gemini", "Google Gemini"), ("mistral", "Mistral"),
                             ("command", "Cohere")):
        if n.startswith(prefix):
            return provider
    return "Transformers (local)" if "/" in n or n.startswith("all-") else None


def _pkg(name: str) -> str:
    return re.split(r"[\s<>=!~\[;@]", name.strip(), maxsplit=1)[0].lower().replace("_", "-") if not \
        name.strip().startswith("@") else "@" + re.split(r"[\s<>=!~\[;]", name.strip()[1:], maxsplit=1)[0].lower()


def _requirements(path: Path) -> list[str]:
    out = []
    for line in _lines(path):
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            name = _pkg(line)
            if name in AI_PACKAGES:
                out.append(name)
    return out


def _pyproject_deps(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
        data = tomllib.loads(text) if tomllib else None
    except (OSError, ValueError):
        return []
    if data is None:  # no TOML parser: any quoted requirement that names an AI package
        return sorted({n for n in (_pkg(q) for q in re.findall(r'"([^"\n]+)"', text)) if n in AI_PACKAGES})
    project = data.get("project", {})
    reqs = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        reqs += extra
    reqs += list(data.get("tool", {}).get("poetry", {}).get("dependencies", {}))
    return [n for n in (_pkg(r) for r in reqs) if n in AI_PACKAGES]


def _node_deps(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    names = list(data.get("dependencies", {})) + list(data.get("devDependencies", {}))
    return [n for n in names if n in AI_PACKAGES]


# ---- recommendations ----

def _chat_alternative(device: DeviceProfile, devices: dict[str, DeviceProfile]) -> Alternative:
    """The biggest local chat model that runs at a usable speed on device."""
    chosen = None
    for name, repo in CHAT_LADDER:
        info = analyze(name)
        p = plan(info, device, Requirements(), store.evidence(device, info))
        v = next((x for x in p.variants if x.format.name == p.recommended), None)
        if v and (v.tokens_per_s is None or v.tokens_per_s >= USABLE_TOKENS_PER_S):
            chosen = (name, repo, info, v)
            break
        chosen = chosen or ((name, repo, info, v) if v else None)
    if chosen is None:
        name, repo = CHAT_LADDER[-1]
        return Alternative(model=name, repo=repo, how="Too big for this device's memory; try a smaller device budget.")
    name, repo, info, v = chosen
    return Alternative(
        model=name, repo=repo, variant=v.format.label if v else None, memory_gb=v.total_memory_gb if v else None,
        tokens_per_s=v.tokens_per_s if v else None, speed_source=v.speed_source if v else None,
        runs_on=_runs_on(info, devices), target=device.name,
        how=f"nanomesh pull {repo}, then nanomesh serve <file> (or ask your agent: download_model, "
            "start_model_server).")


def _runs_on(info, devices: dict[str, DeviceProfile]) -> list[str]:
    out = []
    for d in devices.values():
        p = plan(info, d, Requirements())
        v = next((x for x in p.variants if x.format.name == p.recommended), None)
        if v and (v.tokens_per_s or 0) >= USABLE_TOKENS_PER_S:
            out.append(d.name)
    return out


def _swap(provider: str, locations: list[Location], where: str) -> str | None:
    if where == "mobile":
        return None  # on-device inference is a different library, not a new endpoint
    suffixes = {Path(loc.file).suffix for loc in locations}
    js = bool(suffixes & {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"})
    py = bool(suffixes & {".py", ".ipynb"})
    url = "http://127.0.0.1:8080/v1" if where == "app" else "https://<your-model-server>/v1"
    key = '"local"' if where == "app" else "process.env.MODEL_SERVER_KEY" if js and not py else '"<your key>"'
    model = "<model name from nanomesh serve>" if where == "app" else "<model name on your server>"
    blocks = []
    if provider == "LangChain + OpenAI":
        if py:
            blocks.append(f'ChatOpenAI(model="{model}", base_url="{url}", api_key={key})')
        if js:
            blocks.append(f'new ChatOpenAI({{ model: "{model}", apiKey: {key}, configuration: {{ baseURL: "{url}" }} }})')
        return "\n".join(blocks) or None
    if provider not in ("OpenAI", "Groq", "Mistral", "Google Gemini", "Anthropic", "Cohere"):
        return None
    if provider != "OpenAI":
        blocks.append(f"{provider}'s SDK doesn't speak the OpenAI API: switch these calls to the openai package "
                      "(pip install openai / npm i openai), which works with llama-server.")
    if py:
        blocks.append(f'Python:     client = OpenAI(base_url="{url}", api_key={key})   # model="{model}"')
    if js:
        blocks.append(f'JavaScript: const client = new OpenAI({{ baseURL: "{url}", apiKey: {key} }});   '
                      f'// model: "{model}"')
    return "\n".join(blocks)


def _caveats(task: str, provider: str, models: list[str], code_related: bool = False) -> list[str]:
    out = []
    big = [m for m in models if re.match(r"(gpt-4(?!o-mini|\.1-mini|\.1-nano)|gpt-5|o[134]\b|o[134]-|claude-(opus|"
                                         r"sonnet|3-5|3-7|4)|gemini-.*pro)", m)]
    if task == "chat" and big:
        out.append(f"{', '.join(big)} {'is' if len(big) == 1 else 'are'} far stronger than a local model on complex "
                   "reasoning. Keep those calls in the cloud, or move only the simple ones (classification, "
                   "extraction, short replies) and compare outputs first.")
    if task == "chat" and code_related:
        out.append("For code generation, use the Qwen2.5-Coder model of the same size (same memory and speed).")
    if task == "embeddings":
        out.append("Embeddings from a different model aren't comparable: re-embed everything already stored "
                   "(vector database, cache) when switching.")
    if provider == "Anthropic" or provider == "Google Gemini":
        out.append("Prompts tuned for this provider may need adjusting for a small local model.")
    return out


def _alternative_for(task: str, where: str, device: DeviceProfile, devices, chat_alt) -> Alternative | None:
    if task == "chat":
        alt = chat_alt(where).model_copy()
        speed = f" at ~{alt.tokens_per_s:g} tok/s" if alt.tokens_per_s else ""
        if where == "server":
            alt.how = (f"This code runs on a server, so a model on your laptop can't serve its users. Self-host "
                       f"{alt.model} on a server you control: a 4 GB VPS runs it{speed}, one request at a time; a GPU "
                       "server handles many at once. Run llama-server there behind authentication and point this "
                       "code at it, or keep the cloud API.")
        elif where == "mobile":
            alt.how = (f"This code runs in a mobile app: the model would run on the phone itself (llama.cpp's "
                       f"Android/iOS bindings, or MediaPipe LLM Inference). Sized for a 4 GB Android phone"
                       f"{speed}. Works offline with no cost per call, but the app has to download the model "
                       f"({alt.memory_gb:g} GB in memory) once.")
        return alt
    device = devices[SERVER_TARGET] if where == "server" else devices[MOBILE_TARGET] if where == "mobile" else device
    if task == "embeddings":
        return Alternative(model="nomic-embed-text-v1.5", repo="nomic-ai/nomic-embed-text-v1.5-GGUF", variant="INT8",
                           memory_gb=0.3, runs_on=[d.name for d in devices.values()],
                           how="Small enough for any device. Serve it with llama-server --embeddings, or use "
                               "sentence-transformers with 'nomic-ai/nomic-embed-text-v1.5' in Python.")
    if task == "speech-to-text":
        size = "small" if device.ram_gb >= 8 else "base"
        return Alternative(model=f"whisper {size}", repo="ggerganov/whisper.cpp", runs_on=[],
                           how=f"whisper.cpp or faster-whisper with the '{size}' model: real-time or faster on most "
                               "laptops, 'base' or 'tiny' on phones.")
    if task == "image generation":
        gpu = device.best_gpu
        if gpu and (gpu.vram_gb or 0) >= 6:
            return Alternative(model="SDXL-Turbo", repo="stabilityai/sdxl-turbo",
                               how="Runs on this GPU with the diffusers library (a few seconds per image).")
        return Alternative(model="none practical", how="Image generation needs a GPU with 6 GB+; on this device keep "
                                                       "the cloud API.")
    return None


def analyze_project(root: Path, device: DeviceProfile, include_tests: bool = False) -> ProjectReport:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a folder: {root}")
    usages, deps, model_files, scanned, mentioned = _scan(root, include_tests)
    devices = load_devices()
    cache: dict = {}

    def chat_alt(where: str) -> Alternative:
        if where not in cache:
            target = devices[SERVER_TARGET] if where == "server" else devices[MOBILE_TARGET] if where == "mobile" \
                else device
            cache[where] = _chat_alternative(target, devices)
        return cache[where]

    groups: dict[tuple[str, str, str], list[Usage]] = {}
    for u in usages:
        groups.setdefault((u.provider, u.task, u.where), []).append(u)
    findings = []
    for (provider, task, where), us in sorted(groups.items(), key=lambda kv: (kv[1][0].local, -len(kv[1]))):
        models = {u.model for u in us if u.model}
        if not models:
            # The model came through a variable (callGemini(prompt, MEME_MODEL)): use the ones the files name.
            models = {m for u in us for m in mentioned.get(u.location.file, ())
                      if _provider_for_model(m) in (provider, None) or provider.startswith("LangChain")}
        models = sorted(models)
        f = Finding(provider=provider, task=task, where=where, models=models, calls=len(us), local=us[0].local,
                    locations=[u.location for u in us[:MAX_LOCATIONS]])
        if not f.local:
            f.alternative = _alternative_for(task, where, device, devices, chat_alt)
            if f.alternative and task == "chat":
                f.swap = _swap(provider, f.locations, where)
            code_related = any(re.search(r"cod(e|er|ing)|program|develop", x, re.I)
                               for x in models + [loc.file for loc in us_locations(us)])
            f.caveats = _caveats(task, provider, models, code_related)
        else:
            f.alternative = _local_model_note(models, device)
        findings.append(f)

    cloud = [f for f in findings if not f.local]
    movable = [f for f in cloud if f.alternative and f.alternative.model != "none practical"]
    places = {f.where for f in cloud}
    local_word = "a local or self-hosted option" if places & {"server", "mobile"} else "a local option"
    summary = (f"{sum(f.calls for f in cloud)} cloud AI call site(s) in {len({f.provider for f in cloud})} "
               f"provider(s); {len(movable)} of {len(cloud)} kinds of use have {local_word}."
               if cloud else "No cloud AI API calls found." if not findings else
               "Only local AI found: nothing leaves the machine.")
    return ProjectReport(path=str(root), device=device.name, files_scanned=scanned, findings=findings,
                         dependencies={k: v for k, v in deps.items() if v}, model_files=model_files, summary=summary)


def us_locations(us: list[Usage]) -> list[Location]:
    return [u.location for u in us]


def _local_model_note(models: list[str], device: DeviceProfile) -> Alternative | None:
    """For models already used locally (transformers etc.), whether a GGUF would be lighter."""
    from nanomesh.catalog import model_info_for

    for name in models:
        info = model_info_for(name, [])
        if not info:
            continue
        p = plan(info, device, Requirements(), store.evidence(device, info))
        v = next((x for x in p.variants if x.format.name == p.recommended), None)
        fp32 = round(info.params * 4 / GB, 1)
        if v:
            speed = f" at ~{v.tokens_per_s:g} tok/s ({v.speed_source})" if v.tokens_per_s else ""
            return Alternative(model=name, variant=v.format.label, memory_gb=v.total_memory_gb,
                               tokens_per_s=v.tokens_per_s, speed_source=v.speed_source,
                               how=f"transformers on a CPU loads {name} in full precision (~{fp32:g} GB). As "
                                   f"a GGUF ({v.format.label}) it needs {v.total_memory_gb:g} GB{speed}: "
                                   f"nanomesh search {name.split('/')[-1]}")
    return None
