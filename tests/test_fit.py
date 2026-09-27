"""Fit Cards for speech, vision and embedding models; task search; benchmarks."""

import json
import sys

import pytest

from nanomesh import catalog, config, kaggle, mcp
from nanomesh import results as store
from nanomesh.catalog import CatalogError
from nanomesh.devices import get_device
from nanomesh.fit import FAMILIES, best, card, fit, match, speed
from nanomesh.results import SustainedRun


@pytest.mark.parametrize("name, family, member", [
    ("openai/whisper-small", "whisper", "small"),
    ("openai/whisper-large-v3-turbo", "whisper", "large-v3-turbo"),
    ("distil-whisper/distil-large-v3", "whisper", "distil-large-v3"),
    ("ggml-base.en.bin", "whisper", "base"),
    ("Systran/faster-whisper-medium", "whisper", "medium"),
    ("whisper", "whisper", None),
    ("yolo11n.onnx", "yolo11", "n"),
    ("Ultralytics/YOLOv8", "yolov8", None),
    ("yolov8s.pt", "yolov8", "s"),
    ("timm/mobilenetv3_large_100.ra_in1k", "image classifiers", "mobilenet-v3-large"),
    ("google/vit-base-patch16-224", "image classifiers", "vit-base-patch16"),
    ("microsoft/resnet-50", "image classifiers", "resnet-50"),
    ("BAAI/bge-small-en-v1.5", "embeddings", "bge-small-en"),
    ("sentence-transformers/all-MiniLM-L6-v2", "embeddings", "all-minilm-l6-v2"),
    ("nomic-ai/nomic-embed-text-v1.5", "embeddings", "nomic-embed-text"),
])
def test_names_match_families(name, family, member):
    fam, m = match(name)
    assert fam.series == family and (m.id if m else None) == member


def test_unknown_names_dont_match():
    assert match("qwen2.5-7b") is None and match("my-cool-model") is None


def test_every_family_is_ordered_small_to_large():
    for fam in FAMILIES:
        ranked = sorted((m for m in fam.members if m.rank > 0), key=lambda m: m.rank)
        compute = [m.gflops or m.enc_m or m.params_m for m in ranked]
        assert compute == sorted(compute) or fam.series in ("image classifiers", "embeddings"), fam.series


def test_family_name_alone_picks_the_best_size_for_the_device():
    laptop, phone = get_device("hp-elitebook-840-g6"), get_device("redmi-14c-4gb")
    c = fit("whisper", laptop)
    assert c.model == "whisper → whisper-small" and c.usable and c.speed_unit == "x real-time"
    assert "several sizes" in c.notes[0]
    # Bigger models are more accurate but slower: a phone gets a smaller or equal one.
    fam = match("whisper")[0]
    assert best(fam, phone).rank <= best(fam, laptop).rank


def test_a_too_slow_model_gets_a_better_fitting_one():
    c = fit("google/vit-base-patch16-224", get_device("redmi-14c-4gb"))
    assert c.fits and not c.usable
    # EfficientNet-B0 is both faster and more accurate than ResNet-50, so it's the one suggested.
    assert c.alternative.startswith("Fits better: efficientnet-b0")


def test_a_fast_model_is_offered_a_more_accurate_one():
    c = fit("mobilenet-v3-small", get_device("hp-elitebook-840-g6"))
    assert c.usable and c.alternative.startswith("More accurate and still fast enough here")


def test_download_cost_and_licence():
    config.set("data_price", {"per_gb": 120, "currency": "KES"})
    c = fit("openai/whisper-small", get_device("hp-elitebook-840-g6"), license="apache-2.0")
    assert c.download_gb == pytest.approx(0.45, abs=0.02) and c.data_cost == "~54 KES of data"
    assert c.license == "apache-2.0" and "whisper.cpp" in c.how


def test_battery_line_comes_from_a_measured_battery_run():
    elitebook = get_device("hp-elitebook-840-g6")
    run = SustainedRun(points=[], burst_tokens_per_s=14, sustained_tokens_per_s=13.6, drop_pct=5,
                       battery_hours=3.4)
    store.save([store.Result(timestamp=store.now(), device_key=elitebook.key, device_name=elitebook.name,
                             model_name="qwen", model_params=1, format="Q4_K_M", file_size_gb=1, kind="sustained",
                             sustained=run)])
    c = fit("whisper-small", elitebook)
    hours = 3.4 * c.speed
    assert c.battery.startswith(f"~{hours:.0f} h of audio per full charge")


def test_a_benchmark_measures_its_model_and_calibrates_the_family():
    elitebook = get_device("hp-elitebook-840-g6")
    fam, n = match("yolo11n")
    before, source = speed(fam, n, elitebook)
    assert source == "estimate"
    # Measured 30 images/s for yolo11n: the device delivers 30 x 6.5 = 195 GFLOPS with ONNX Runtime.
    store.save([store.Result(timestamp=store.now(), device_key=elitebook.key, device_name=elitebook.name,
                             model_name="yolo11n", model_params=2_600_000, format="onnx", file_size_gb=0.01,
                             kind="onnx", task="object detection", throughput=30.0, throughput_unit="images/s",
                             gflops_per_unit=6.5)])
    assert speed(fam, n, elitebook) == (30.0, "measured")
    s_rate, s_source = speed(fam, match("yolo11s")[1], elitebook)
    assert s_source == "calibrated" and s_rate == pytest.approx(195 / 21.5, abs=0.1)
    # Other ONNX families on this device are calibrated too; Whisper isn't (different runtime).
    assert speed(*match("mobilenet-v3-large"), elitebook)[1] == "calibrated"
    assert speed(*match("whisper-small"), elitebook)[1] == "estimate"


def test_unknown_model_is_sized_from_its_parameters():
    c = fit("someone/swahili-asr", get_device("hp-elitebook-840-g6"), task="speech-to-text", params=300_000_000)
    assert c.task == "speech-to-text" and c.memory_gb == pytest.approx(300e6 * 4 / 1024**3 * 1.1 + 0.25, abs=0.01)
    assert c.speed and "300M parameters" in c.notes[0]
    det = fit("someone/detector", get_device("hp-elitebook-840-g6"), task="object detection", params=40_000_000)
    assert det.speed is None and "benchmark it" in det.notes[0]


def test_text_models_get_cards_from_the_planner():
    c = fit("qwen2.5-1.5b", get_device("hp-elitebook-840-g6"))
    assert c.task == "text generation" and c.speed_unit == "tok/s" and c.fits


# ---- search by task ----

def fake_fetch(listing, infos):
    calls = []

    def fetch(url):
        calls.append(url)
        path = url.removeprefix(catalog.HF)
        if path.startswith("/api/models?"):
            return listing
        return infos.get(path.removeprefix("/api/models/"), {})
    fetch.calls = calls
    return fetch


def test_speech_search_ranks_what_fits_and_is_fast_enough():
    listing = [{"id": "openai/whisper-large-v3", "downloads": 5_000_000},
               {"id": "openai/whisper-small", "downloads": 2_000_000},
               {"id": "someone/swahili-asr", "downloads": 900},
               {"id": "openai/whisper-tiny", "downloads": 1_000_000}]
    infos = {"openai/whisper-small": {"cardData": {"license": "apache-2.0"}},
             "someone/swahili-asr": {"safetensors": {"total": 95_000_000}, "tags": ["license:cc-by-4.0"]}}
    fetch = fake_fetch(listing, infos)
    found = catalog.search_task("", "speech", get_device("redmi-14c-4gb"), limit=4, fetch=fetch)
    assert "pipeline_tag=automatic-speech-recognition" in fetch.calls[0] and "search=" not in fetch.calls[0]
    ids = [r.id for r in found]
    # large-v3 is the most downloaded, but too slow on this phone: it goes last.
    assert ids[-1] == "openai/whisper-large-v3" and found[-1].card.usable is False
    assert next(r for r in found if r.id == "openai/whisper-small").card.license == "apache-2.0"
    assert next(r for r in found if r.id == "someone/swahili-asr").card.license == "cc-by-4.0"


def test_search_words_and_tasks():
    fetch = fake_fetch([{"id": "Ultralytics/YOLOv8", "downloads": 10}, {"id": "hustvl/yolos-tiny", "downloads": 5}],
                       {})
    found = catalog.search_task("yolo v8", "detection", get_device("hp-elitebook-840-g6"), fetch=fetch)
    assert [r.id for r in found] == ["Ultralytics/YOLOv8"]
    assert found[0].card.model == "Ultralytics/YOLOv8 → yolov8n"  # the repo holds every size: the best one here
    with pytest.raises(CatalogError, match="Unknown task"):
        catalog.search_task("", "painting", get_device("hp-elitebook-840-g6"), fetch=fetch)


def test_kaggle_needs_a_token(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    with pytest.raises(CatalogError, match="Create New Token"):
        kaggle.fetch("https://www.kaggle.com/api/v1/models/list?search=x")
    (tmp_path / "kaggle.json").write_text(json.dumps({"username": "tonny", "key": "abc"}))
    assert kaggle.credentials() == ("tonny", "abc")


def test_kaggle_models_get_fit_cards():
    data = {"models": [{"ref": "google/gemma", "title": "Gemma"},
                       {"ref": "keras/yolo11", "title": "YOLO11", "licenseName": "AGPL-3.0", "downloadCount": 42},
                       {"ref": "someone/mystery", "title": "Mystery"}]}
    found = kaggle.search_models("yolo", get_device("hp-elitebook-840-g6"), "object detection",
                                 fetcher=lambda url: data)
    assert [r.id for r in found] == ["keras/yolo11"]
    r = found[0]
    assert r.source == "kaggle" and r.card.license == "AGPL-3.0" and r.downloads == 42
    assert r.url == "https://www.kaggle.com/models/keras/yolo11"


# ---- benchmarks ----

def _tiny_onnx(path, image=True):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    if image:
        w = numpy_helper.from_array(np.random.rand(8, 3, 3, 3).astype("float32"), "w")
        graph = helper.make_graph([helper.make_node("Conv", ["x", "w"], ["y"], strides=[4, 4])], "tiny",
                                  [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, "H", "W"])],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], [w])
    else:
        graph = helper.make_graph([helper.make_node("Cast", ["input_ids"], ["y"], to=TensorProto.FLOAT)], "tiny",
                                  [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["N", "S"])],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_onnx_benchmark_measures_and_calibrates(tmp_path):
    from nanomesh.runtimes import benchmark_file, onnx_benchmark

    model = _tiny_onnx(tmp_path / "yolo11n.onnx")
    elitebook = get_device("hp-elitebook-840-g6")
    run = benchmark_file(model, elitebook)
    assert run.runtime == "onnx" and run.model_name == "yolo11n" and run.unit == "images/s"
    assert run.throughput > 0 and run.gflops_per_unit == 6.5
    saved = store.load()[-1]
    assert (saved.kind, saved.task, saved.throughput) == ("onnx", "object detection", run.throughput)
    assert fit("yolo11n", elitebook).speed_source == "measured"
    # Text-style inputs (token ids) get a sequence; an unknown model is measured, not calibrated.
    other = onnx_benchmark(_tiny_onnx(tmp_path / "encoder.onnx", image=False), seconds=0.2)
    assert other.unit == "runs/s" and other.gflops_per_unit is None and "no calibration" in other.note


WHISPER_BENCH = """#!{python}
import sys
print("whisper_init_from_file: loading model", file=sys.stderr)
print("whisper_print_timings:   encode time =  1500.00 ms /     1 runs ( 1500.00 ms per run)", file=sys.stderr)
"""


def test_whisper_bench(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("fake whisper-bench is a POSIX script")
    from nanomesh.runtimes import benchmark_file, parse_whisper_bench

    assert parse_whisper_bench("encode time =  3000.00 ms /     2 runs ( 1500.00 ms per run)") == 1500.0
    assert parse_whisper_bench("whisper_print_timings: encode time = 900.5 ms") == 900.5
    folder = tmp_path / "whisper.cpp" / "build" / "bin"
    folder.mkdir(parents=True)
    exe = folder / "whisper-bench"
    exe.write_text(WHISPER_BENCH.format(python=sys.executable))
    exe.chmod(0o755)
    monkeypatch.setenv("NANOMESH_WHISPER_CPP", str(tmp_path / "whisper.cpp"))
    model = tmp_path / "ggml-small.bin"
    model.write_bytes(b"\0" * 1000)
    elitebook = get_device("hp-elitebook-840-g6")
    run = benchmark_file(model, elitebook)
    assert run.model_name == "whisper-small" and run.throughput == 20.0  # 30 s window in 1.5 s
    c = fit("whisper-small", elitebook)
    assert c.speed_source == "measured" and c.speed < 20  # the card adds the decoder
    # The folder is remembered for processes started without the variable (an editor's MCP server).
    monkeypatch.delenv("NANOMESH_WHISPER_CPP")
    from nanomesh.runtimes import find_whisper_bench
    assert find_whisper_bench() == exe


# ---- MCP and CLI ----

def test_mcp_fit_card_and_task_search(monkeypatch):
    monkeypatch.setattr(mcp, "_local", lambda: get_device("hp-elitebook-840-g6"))
    out = mcp.call_tool("fit_card", {"model": "whisper-small"})["structuredContent"]
    assert out["task"] == "speech-to-text" and out["speed_unit"] == "x real-time"
    assert mcp.call_tool("fit_card", {"model": "who-knows"})["isError"]
    monkeypatch.setattr(catalog, "fetch_json", fake_fetch([{"id": "BAAI/bge-small-en-v1.5", "downloads": 3}], {}))
    found = mcp.call_tool("search_models", {"task": "embeddings"})["structuredContent"]
    assert found["task"] == "embeddings" and found["results"][0]["card"]["speed_unit"] == "sentences/s"
    assert mcp.call_tool("search_models", {"task": "painting"})["isError"]


def test_cli_fit_and_config():
    from typer.testing import CliRunner

    from nanomesh.cli import app

    runner = CliRunner()
    assert runner.invoke(app, ["config", "--data-price", "100", "--currency", "KES"]).exit_code == 0
    out = runner.invoke(app, ["fit", "openai/whisper-small", "-d", "hp-elitebook-840-g6", "--json"])
    data = json.loads(out.output)
    assert data["data_cost"].endswith("KES of data") and data["fits"]
    assert runner.invoke(app, ["fit", "who-knows", "-d", "hp-elitebook-840-g6"]).exit_code == 1


def test_card_for_every_member_on_every_device():
    from nanomesh.devices import load_devices

    for device in load_devices().values():
        for fam in FAMILIES:
            for m in fam.members:
                c = card(fam, m, device)
                assert c.memory_gb > 0 and c.download_gb > 0 and c.speed and c.speed > 0


# ---- formats only some devices can run ----

@pytest.mark.parametrize("name, tags, runs_on, not_on", [
    ("argmaxinc/whisperkit-coreml", ["coreml", "whisper"], "macbook-air-m1-8gb", "hp-elitebook-840-g6"),
    ("mlx-community/whisper-small-mlx", ["mlx"], "macbook-air-m1-8gb", "rtx-3060-12gb"),
    ("TheBloke/some-7B-GPTQ", ["gptq"], "rtx-3060-12gb", "hp-elitebook-840-g6"),
])
def test_platform_only_formats(name, tags, runs_on, not_on):
    from nanomesh.fit import cannot_run

    assert cannot_run(name, tags, get_device(runs_on)) is None
    assert cannot_run(name, tags, get_device(not_on))
    assert cannot_run("openai/whisper-small", ["transformers", "pytorch"], get_device(not_on)) is None


def test_search_puts_models_this_device_cant_run_last():
    # The real top result for speech on the EliteBook: Core ML, Apple-only.
    listing = [{"id": "argmaxinc/whisperkit-coreml", "downloads": 10_997_197, "tags": ["whisper", "coreml"],
                "library_name": "whisperkit"},
               {"id": "Systran/faster-whisper-small", "downloads": 3_212_472, "library_name": "ctranslate2"}]
    found = catalog.search_task("", "speech", get_device("hp-elitebook-840-g6"), fetch=fake_fetch(listing, {}))
    assert [r.id for r in found] == ["Systran/faster-whisper-small", "argmaxinc/whisperkit-coreml"]
    coreml = found[1].card
    assert not coreml.fits and coreml.notes[0].startswith("Can't run on HP EliteBook 840 G6: it's a Core ML model")
    mac = catalog.search_task("", "speech", get_device("macbook-air-m1-8gb"), fetch=fake_fetch(listing, {}))
    assert mac[0].id == "argmaxinc/whisperkit-coreml" and mac[0].card.fits


def test_big_counts_are_rounded():
    from nanomesh.fit import _approx

    assert [_approx(x) for x in (42.4, 112_608, 1_234_567, 999)] == ["42", "110,000", "1,200,000", "1,000"]
