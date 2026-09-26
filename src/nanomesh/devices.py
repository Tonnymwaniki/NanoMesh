"""The device database: curated profiles of real deployment targets."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

from nanomesh.hardware import DeviceProfile


@lru_cache(maxsize=1)
def load_devices() -> dict[str, DeviceProfile]:
    raw = json.loads(resources.files("nanomesh.data").joinpath("devices.json").read_text())
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
