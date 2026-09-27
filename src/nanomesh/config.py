"""NanoMesh's settings file, ~/.nanomesh/config.json (NANOMESH_HOME to move it)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def path() -> Path:
    from nanomesh.results import home

    return home() / "config.json"


def load() -> dict:
    try:
        data = json.loads(path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def get(key: str, default: Any = None) -> Any:
    return load().get(key, default)


def set(key: str, value: Any) -> None:  # noqa: A001 - config.set reads naturally
    data = load()
    if data.get(key) == value:
        return
    data[key] = value
    try:
        path().parent.mkdir(parents=True, exist_ok=True)
        path().write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass
