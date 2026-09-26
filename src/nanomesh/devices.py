"""The device database: curated profiles of real deployment targets."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources

from nanomesh.hardware import DeviceProfile


@lru_cache(maxsize=1)
def load_devices() -> dict[str, DeviceProfile]:
    raw = json.loads(resources.files("nanomesh.data").joinpath("devices.json").read_text(encoding="utf-8"))
    return {d["id"]: DeviceProfile(**d) for d in raw}


def get_device(device_id: str) -> DeviceProfile:
    devices = load_devices()
    if device_id in devices:
        return devices[device_id]
    matches = search_devices(device_id)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"Unknown device '{device_id}'. Run `nanomesh devices` to list known devices.")
    ids = ", ".join(d.id for d in matches)
    raise KeyError(f"'{device_id}' is ambiguous; did you mean one of: {ids}")


def search_devices(query: str) -> list[DeviceProfile]:
    terms = query.lower().replace("-", " ").split()
    results = []
    for device in load_devices().values():
        haystack = f"{device.id} {device.name} {device.cpu or ''} {device.kind} {device.os or ''}".lower().replace("-", " ")
        if all(t in haystack for t in terms):
            results.append(device)
    return results


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("_", " ")).strip()


def match_device(profile: DeviceProfile) -> DeviceProfile | None:
    """Find the database entry for a scanned machine, by vendor/model string."""
    identity = _normalize(" ".join(filter(None, (profile.vendor, profile.model))))
    if not identity:
        return None
    for known in load_devices().values():
        # Word boundaries keep "thinkpad t480" from matching a "ThinkPad T480s".
        if any(re.search(rf"(?<![\w-]){p}(?![\w-])", identity) for p in known.match):
            return known
    return None


def recognise(profile: DeviceProfile) -> DeviceProfile:
    """Enrich a live scan with curated data from its database entry.

    Measured facts about this machine (RAM, CPU, flags, GPUs found) win;
    the database fills in what can't be probed, like memory bandwidth.
    """
    known = match_device(profile)
    if not known:
        return profile
    fill = {
        "matched_id": known.id,
        "name": known.name,
        "kind": known.kind,
        "memory_bandwidth_gbps": profile.memory_bandwidth_gbps or known.memory_bandwidth_gbps,
        "npu": profile.npu or known.npu,
        "notes": profile.notes or known.notes,
        "cpu_flags": profile.cpu_flags or known.cpu_flags,
        "gpus": profile.gpus or known.gpus,
    }
    return profile.model_copy(update=fill)
