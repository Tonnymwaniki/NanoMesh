"""Kaggle Models, sized for this device.

Kaggle's API needs the user's token: create one at kaggle.com/settings (API >
Create New Token) and save kaggle.json in ~/.kaggle (C:\\Users\\<you>\\.kaggle on
Windows), or set KAGGLE_USERNAME and KAGGLE_KEY.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from nanomesh import __version__
from nanomesh.catalog import CatalogError, TaskResult
from nanomesh.hardware import DeviceProfile

API = os.environ.get("KAGGLE_API_ENDPOINT", "https://www.kaggle.com/api/v1").rstrip("/")
NO_TOKEN = ("Kaggle needs your API token: on kaggle.com go to Settings > API > Create New Token, then save the "
            "kaggle.json it downloads in ~/.kaggle/ (C:\\Users\\<you>\\.kaggle\\ on Windows), or set KAGGLE_USERNAME "
            "and KAGGLE_KEY.")


def credentials() -> tuple[str, str] | None:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]
    folder = Path(os.environ.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")
    try:
        data = json.loads((folder / "kaggle.json").read_text(encoding="utf-8"))
        return data["username"], data["key"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def fetch(url: str):
    creds = credentials()
    if not creds:
        raise CatalogError(NO_TOKEN)
    token = base64.b64encode(f"{creds[0]}:{creds[1]}".encode()).decode()
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {token}",
                                               "User-Agent": f"nanomesh/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise CatalogError("Kaggle refused the API token: create a new one at kaggle.com/settings.") from None
        raise CatalogError(f"Kaggle returned HTTP {e.code}.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise CatalogError(f"Can't reach Kaggle ({getattr(e, 'reason', e)}).") from None


def search_models(query: str, device: DeviceProfile, task: str | None = None, *, limit: int = 5,
                  fetcher=None) -> list[TaskResult]:
    """Kaggle models whose name or title matches, with Fit Cards for the ones
    NanoMesh can size (known families and text models)."""
    from nanomesh.fit import fit

    get = fetcher or fetch
    data = get(f"{API}/models/list?" + urllib.parse.urlencode({"search": query, "pageSize": 20}))
    models = data.get("models", []) if isinstance(data, dict) else data or []
    out = []
    for m in models:
        ref = m.get("ref") or "/".join(x for x in (m.get("owner") or m.get("author"), m.get("slug")) if x)
        if not ref:
            continue
        names = [ref, m.get("title") or "", m.get("slug") or ""]
        frameworks = [i.get("framework") for i in m.get("instances") or [] if isinstance(i, dict)]
        card = next((c for n in names if n and (c := fit(n, device, task=task if task != "text generation" else None,
                                                          license=m.get("licenseName"),
                                                          tags=[m.get("framework"), *frameworks]))), None)
        if card and (task is None or card.task == task):
            out.append(TaskResult(source="kaggle", id=ref, url=m.get("url") or f"https://www.kaggle.com/models/{ref}",
                                  downloads=m.get("downloadCount") or m.get("downloads"), card=card))
        if len(out) >= limit:
            break
    return out
