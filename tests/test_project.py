"""`nanomesh project`: finding AI in a codebase and what could run locally."""

import json

import pytest

from nanomesh import mcp
from nanomesh.devices import get_device
from nanomesh.project import _redact, analyze_project, scan

FAKE_KEY = "sk-proj-" + "abcdefghijklmnopqrstuvwxyz123456"

FILES = {
    "app/chat.py": f'''from openai import OpenAI
client = OpenAI(api_key="{FAKE_KEY}")

def reply(q):
    r = client.chat.completions.create(model="gpt-4o-mini", messages=[{{"role": "user", "content": q}}])
    return r.choices[0].message.content

def embed(texts):
    return client.embeddings.create(model="text-embedding-3-small", input=texts)

def hard(q):
    return client.chat.completions.create(model="gpt-4o", messages=[{{"role": "user", "content": q}}])
''',
    "app/local.py": '''from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
''',
    "app/claude.py": '''import anthropic
c = anthropic.Anthropic()
msg = c.messages.create(model="claude-sonnet-4-5", max_tokens=100, messages=[])
''',
    "app/voice.py": '''from openai import OpenAI
client = OpenAI()
text = client.audio.transcriptions.create(model="whisper-1", file=open("a.mp3", "rb"))
''',
    "web/bot.ts": '''import OpenAI from "openai";
const client = new OpenAI();
export async function ask(q: string) {
  const r = await client.chat.completions.create({ model: "gpt-4.1-mini", messages: [{ role: "user", content: q }] });
  return r.choices[0].message.content;
}
''',
    "notebooks/explore.ipynb": json.dumps({"cells": [
        {"cell_type": "markdown", "source": ["client.chat.completions.create is how we call it"]},
        {"cell_type": "code", "source": ["from langchain_openai import ChatOpenAI\n",
                                         "llm = ChatOpenAI(model=\"gpt-4o-mini\")\n"]},
    ]}),
    ".env": "OPENAI_API_KEY=sk-secret-that-must-never-be-read-1234567890",
    "node_modules/openai/index.js": 'client.chat.completions.create({model: "gpt-4o"})',
    ".venv/lib/site.py": 'client.chat.completions.create(model="gpt-4o")',
    "requirements.txt": "openai>=1.0\nrequests\ntransformers==4.44  # local model\n",
    "package.json": json.dumps({"dependencies": {"openai": "^4", "react": "18"}}),
    "pyproject.toml": '[project]\nname = "x"\ndependencies = ["anthropic>=0.30", "fastapi"]\n',
    "README.md": "We call client.chat.completions.create with gpt-4o.",
}


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    for name, text in FILES.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    (root / "models").mkdir()
    (root / "models" / "tiny.gguf").write_bytes(b"\0" * 1000)
    return root


def _finding(report, provider, task):
    return next(f for f in report.findings if f.provider == provider and f.task == task)


def test_finds_every_use_with_file_and_line(project):
    usages, deps, model_files, scanned = scan(project)
    where = {(u.provider, u.task, u.location.file, u.location.line) for u in usages}
    assert ("OpenAI", "chat", "app/chat.py", 5) in where
    assert ("OpenAI", "embeddings", "app/chat.py", 9) in where
    assert ("OpenAI", "speech-to-text", "app/voice.py", 3) in where
    assert ("OpenAI", "chat", "web/bot.ts", 4) in where
    assert ("Anthropic", "chat", "app/claude.py", 3) in where
    assert ("LangChain + OpenAI", "chat", "notebooks/explore.ipynb", 2) in where
    assert ("Transformers (local)", "chat", "app/local.py", 2) in where
    assert not any(line == 1 for _, _, f, line in where if f.endswith((".py", ".ts")))  # imports aren't uses
    # Dependencies, vendored code, docs and markdown cells aren't call sites.
    files = {u.location.file for u in usages}
    assert not any(f.startswith(("node_modules", ".venv")) or f == "README.md" for f in files)
    assert deps == {"python": ["anthropic", "openai", "transformers"], "node": ["openai"]}
    assert model_files == [{"file": "models/tiny.gguf", "format": "gguf", "size_gb": 0.0}]
    assert scanned == 6


def test_secrets_are_never_read_or_echoed(project):
    report = analyze_project(project, get_device("hp-elitebook-840-g6"))
    text = report.model_dump_json()
    assert "abcdefghijklmnopqrstuvwxyz" not in text
    assert "must-never-be-read" not in text  # .env is skipped entirely
    # A key on a line that is reported gets masked.
    assert _redact(f'r = client.chat.completions.create(api_key="{FAKE_KEY}")') == \
        'r = client.chat.completions.create(api_key="sk-…[redacted]")'


def test_recommends_a_local_chat_model_for_this_laptop(project):
    report = analyze_project(project, get_device("hp-elitebook-840-g6"))
    chat = _finding(report, "OpenAI", "chat")
    assert chat.calls == 3 and chat.models == ["gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"]
    alt = chat.alternative
    assert alt.repo.startswith("Qwen/Qwen2.5-") and alt.tokens_per_s >= 5 and alt.memory_gb < 16
    assert "HP EliteBook 840 G6" in alt.runs_on
    # The swap is the two-line change to the local server, for both SDKs used.
    # OpenAI is called from Python and TypeScript here: the change is shown for both.
    assert 'base_url="http://127.0.0.1:8080/v1"' in chat.swap and 'baseURL: "http://127.0.0.1:8080/v1"' in chat.swap
    assert any("gpt-4o" in c and "stronger" in c for c in chat.caveats)  # honest about gpt-4o


def test_embeddings_speech_and_other_providers(project):
    report = analyze_project(project, get_device("hp-elitebook-840-g6"))
    emb = _finding(report, "OpenAI", "embeddings")
    assert emb.alternative.model == "nomic-embed-text-v1.5"
    assert any("re-embed" in c for c in emb.caveats)
    assert "whisper" in _finding(report, "OpenAI", "speech-to-text").alternative.model
    claude = _finding(report, "Anthropic", "chat")
    assert "openai package" in claude.swap and any("stronger" in c for c in claude.caveats)
    assert "ChatOpenAI(model=" in _finding(report, "LangChain + OpenAI", "chat").swap


def test_local_transformers_model_gets_a_lighter_option(project):
    report = analyze_project(project, get_device("hp-elitebook-840-g6"))
    local = next(f for f in report.findings if f.provider == "Transformers (local)")
    assert local.local and local.alternative.variant in ("INT8", "INT6", "INT5", "INT4")
    assert "full precision" in local.alternative.how
    # Cloud findings come first; the summary counts only cloud calls.
    assert not report.findings[0].local
    assert report.summary == ("7 cloud AI call site(s) in 3 provider(s); 5 of 5 kinds of use can run locally on "
                              "HP EliteBook 840 G6.")


def test_phone_gets_a_smaller_model(project):
    laptop = analyze_project(project, get_device("hp-elitebook-840-g6"))
    phone = analyze_project(project, get_device("redmi-14c-4gb"))
    size = lambda r: float(_finding(r, "OpenAI", "chat").alternative.model.split("-")[1].rstrip("b"))  # noqa: E731
    assert size(phone) < size(laptop)


def test_empty_and_missing_folders(tmp_path):
    (tmp_path / "empty").mkdir()
    assert analyze_project(tmp_path / "empty", get_device("rtx-3060-12gb")).summary == "No cloud AI API calls found."
    with pytest.raises(ValueError):
        analyze_project(tmp_path / "nope", get_device("rtx-3060-12gb"))


def test_mcp_tool(project, monkeypatch):
    monkeypatch.setattr(mcp, "_local", lambda: get_device("hp-elitebook-840-g6"))
    out = mcp.call_tool("analyze_project", {"path": str(project)})
    assert not out["isError"] and out["structuredContent"]["findings"][0]["provider"] == "OpenAI"
    assert mcp.call_tool("analyze_project", {"path": str(project / "missing")})["isError"]


EDGE = {
    "svc/multi.py": '''from openai import OpenAI
client = OpenAI()

def summarise(text):
    """Calls client.chat.completions.create(model="gpt-4o") under the hood."""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": text}],
    )
    return r

EXAMPLE = 'client.chat.completions.create(model="gpt-4o")'  # text, not a call
# client.embeddings.create(model="text-embedding-3-large")
''',
    "svc/gem.py": '''import google.generativeai as genai
model = genai.GenerativeModel("gemini-1.5-flash")
answer = model.generate_content("hi")
''',
    "svc/local.py": '''import ollama
reply = ollama.chat(model="llama3.2", messages=[])
''',
    "tests/test_multi.py": '''client.chat.completions.create(model="gpt-4o")''',
    "src/bot.test.ts": '''client.chat.completions.create({model: "gpt-4o"})''',
}


def test_edge_cases(tmp_path):
    for name, text in EDGE.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    usages = scan(tmp_path)[0]
    found = {(u.provider, u.task, u.location.file, u.location.line, u.model) for u in usages}
    assert found == {
        # The call spans lines: its model comes from below. The docstring, the
        # string example and the commented-out line aren't calls.
        ("OpenAI", "chat", "svc/multi.py", 6, "gpt-4o-mini"),
        ("Google Gemini", "chat", "svc/gem.py", 3, None),
        ("Ollama (local)", "chat", "svc/local.py", 2, "llama3.2"),
    }
    # Tests are left out unless asked for.
    with_tests = {u.location.file for u in scan(tmp_path, include_tests=True)[0]}
    assert {"tests/test_multi.py", "src/bot.test.ts"} <= with_tests
