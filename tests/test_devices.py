import pytest

from nanomesh import hardware
from nanomesh.devices import match_device, recognise
from nanomesh.hardware import DeviceProfile
from nanomesh.planner import memory_budgets


def _local(vendor, model, **kw):
    return DeviceProfile(name="x", arch="x86_64", ram_gb=kw.pop("ram_gb", 8), is_local=True,
                         vendor=vendor, model=model, **kw)


@pytest.mark.parametrize("vendor, model, expected", [
    ("HP", "HP EliteBook 840 G3", "hp-elitebook-840-g3"),
    ("LENOVO", "ThinkPad T480 (20L5CTO1WW)", "lenovo-thinkpad-t480"),
    ("LENOVO", "ThinkPad T480s (20L7001SUS)", None),  # T480s is a different machine
    ("Dell Inc.", "Latitude 7490", "dell-latitude-7490"),
    ("Apple", "MacBookAir10,1", "macbook-air-m1-8gb"),
    ("Raspberry", "Raspberry Pi 5 Model B Rev 1.0", "raspberry-pi-5-8gb"),
    ("samsung", "SM-A256E (s5e8825)", "samsung-galaxy-a25-6gb"),
    ("Redmi", "Redmi 14C", "redmi-14c-4gb"),
    ("ACME", "Frobnicator 3000", None),
    (None, None, None),
])
def test_match_device(vendor, model, expected):
    found = match_device(_local(vendor, model))
    assert (found.id if found else None) == expected


def test_recognise_keeps_measured_facts_and_fills_gaps():
    live = _local("HP", "HP EliteBook 840 G3", ram_gb=16, cpu="Intel(R) Core(TM) i5-6300U", kind="laptop")
    d = recognise(live)
    assert d.matched_id == "hp-elitebook-840-g3"
    assert d.name == "HP EliteBook 840 G3"
    assert d.ram_gb == 16  # upgraded RAM: the live scan wins
    assert d.cpu == "Intel(R) Core(TM) i5-6300U"
    assert d.memory_bandwidth_gbps == 34.1  # only the database knows this
    assert d.key == "hp-elitebook-840-g3"


def test_unrecognised_machine_has_stable_key():
    d = recognise(_local("ACME", "Frobnicator 3000", ram_gb=12))
    assert d.matched_id is None
    assert d.key == "local:acme frobnicator 3000|12gb"


def test_linux_dmi_identity(monkeypatch):
    files = {
        "/sys/class/dmi/id/sys_vendor": "LENOVO\n",
        "/sys/class/dmi/id/product_name": "20L5CTO1WW\n",
        "/sys/class/dmi/id/product_version": "ThinkPad T480\n",
    }
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.delenv("ANDROID_ROOT", raising=False)
    monkeypatch.delenv("ANDROID_DATA", raising=False)
    monkeypatch.setattr(hardware, "_read", files.get)
    assert hardware.system_identity() == ("LENOVO", "ThinkPad T480 (20L5CTO1WW)")


def test_linux_dmi_ignores_placeholder_strings(monkeypatch):
    files = {
        "/sys/class/dmi/id/sys_vendor": "To Be Filled By O.E.M.\n",
        "/sys/class/dmi/id/product_name": "System Product Name\n",
        "/proc/device-tree/model": "Raspberry Pi 5 Model B Rev 1.0\x00",
    }
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.delenv("ANDROID_ROOT", raising=False)
    monkeypatch.delenv("ANDROID_DATA", raising=False)
    monkeypatch.setattr(hardware, "_read", files.get)
    assert hardware.system_identity() == ("Raspberry", "Raspberry Pi 5 Model B Rev 1.0")


def test_android_identity(monkeypatch):
    props = {"ro.product.brand": "Redmi", "ro.product.marketname": "Redmi 14C", "ro.soc.model": "MT6769Z"}
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setenv("ANDROID_ROOT", "/system")
    monkeypatch.setattr(hardware, "_cmd", lambda args: props.get(args[-1]))
    assert hardware.system_identity() == ("Redmi", "Redmi 14C (MT6769Z)")
    assert hardware._device_kind() == "phone"


def test_live_phone_budget_respects_both_limits():
    phone = DeviceProfile(name="p", arch="aarch64", kind="phone", is_local=True, ram_gb=8, available_ram_gb=6)
    assert memory_budgets(phone)[-1].memory_gb == pytest.approx(8 * 0.45)
    busy = phone.model_copy(update={"available_ram_gb": 2})
    assert memory_budgets(busy)[-1].memory_gb == pytest.approx(2 * 0.85)
