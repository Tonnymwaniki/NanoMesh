"""Dataset Cards: Hugging Face (viewer), Kaggle, local files, and fine-tuning time."""

import csv
import json

import pytest

from nanomesh import catalog, config, datasets, mcp
from nanomesh.catalog import CatalogError
from nanomesh.devices import get_device

GB = 1024**3
ELITEBOOK = "hp-elitebook-840-g6"

# Shaped like the real API and dataset viewer responses.
HF = {
    "/api/datasets?": [
        {"id": "stanfordnlp/imdb", "downloads": 140_000},
        {"id": "someone/swahili-news", "downloads": 900},
        {"id": "big/web-crawl", "downloads": 50_000},
    ],
    "/api/datasets/stanfordnlp/imdb": {"id": "stanfordnlp/imdb", "downloads": 140_000,
                                       "cardData": {"license": "other", "language": ["en"],
                                                    "task_categories": ["text-classification"]},
                                       "tags": ["modality:text", "size_categories:10K<n<100K"]},
    "/api/datasets/someone/swahili-news": {"id": "someone/swahili-news", "tags": ["language:sw"]},
    "/api/datasets/big/web-crawl": {"id": "big/web-crawl", "cardData": {"license": "odc-by"}},
}
VIEWER = {
    "size?dataset=stanfordnlp%2Fimdb": {"size": {
        "dataset": {"num_rows": 100_000, "num_bytes_original_files": 84_125_825,
                    "num_bytes_parquet_files": 83_446_840, "num_bytes_memory": 132_710_000},
        "splits": [{"config": "plain_text", "split": "train", "num_rows": 25_000},
                   {"config": "plain_text", "split": "test", "num_rows": 25_000},
                   {"config": "plain_text", "split": "unsupervised", "num_rows": 50_000}]}, "partial": False},
    "splits?dataset=stanfordnlp%2Fimdb": {"splits": [{"config": "plain_text", "split": "train"},
                                                     {"config": "plain_text", "split": "test"}]},
    "first-rows?dataset=stanfordnlp%2Fimdb&config=plain_text&split=train": {
        "features": [{"name": "text", "type": {"dtype": "string", "_type": "Value"}},
                     {"name": "label", "type": {"names": ["neg", "pos"], "_type": "ClassLabel"}}],
        "rows": [{"row_idx": i, "row": {"text": "I rented this movie " * 60, "label": 0}} for i in range(3)]},
    "size?dataset=big%2Fweb-crawl": {"size": {
        "dataset": {"num_rows": 400_000_000, "num_bytes_parquet_files": 900 * GB, "num_bytes_memory": 2000 * GB},
        "splits": [{"config": "default", "split": "train", "num_rows": 400_000_000}]}, "partial": True},
}


def fetch(url):
    if url.startswith(datasets.VIEWER):
        key = url.removeprefix(datasets.VIEWER + "/")
        if key not in VIEWER:
            raise CatalogError("Not found")
        return VIEWER[key]
    path = url.removeprefix(catalog.HF)
    if path.startswith("/api/datasets?"):
        return HF["/api/datasets?"]
    return HF.get(path, {})


def test_hugging_face_card_from_the_viewer():
    config.set("data_price", {"per_gb": 100, "currency": "KES"})
    c = datasets.hf_card("stanfordnlp/imdb", get_device(ELITEBOOK), fetch)
    assert (c.rows, c.license, c.languages, c.tasks) == (100_000, "other", ["en"], ["text-classification"])
    assert c.splits == {"plain_text/train": 25_000, "plain_text/test": 25_000, "plain_text/unsupervised": 50_000}
    assert [(col.name, col.type) for col in c.columns] == [("text", "string"), ("label", "label (2 classes)")]
    assert c.download_gb == pytest.approx(83_446_840 / GB, abs=0.001)  # the smaller of parquet and original
    assert c.data_cost == "~8 KES of data" and c.fits_memory and c.modality == "text"
    assert c.avg_tokens_per_row == pytest.approx(len("I rented this movie ") * 60 / 4, abs=1)
    assert c.preview[0]["text"].endswith("…") and len(c.preview) == 3
    assert "streaming=True" in c.how


def test_a_huge_dataset_gets_streaming_and_slice_advice():
    c = datasets.hf_card("big/web-crawl", get_device(ELITEBOOK), fetch)
    assert c.fits_memory is False and any("stream it" in a for a in c.advice)
    assert any("0.5 GB slice" in a for a in c.advice)
    assert any("only part" in a for a in c.advice)  # the viewer's measurement was partial
    assert any("no preview" in a for a in c.advice)


def test_no_viewer_no_licence():
    c = datasets.hf_card("someone/swahili-news", get_device(ELITEBOOK), fetch)
    assert c.languages == ["sw"] and c.rows is None and c.license is None
    assert any("No licence stated" in a for a in c.advice)


def test_search_filters_by_every_word():
    found = datasets.hf_search("swahili news", get_device(ELITEBOOK), fetch=fetch)
    assert [c.id for c in found] == ["someone/swahili-news"]
    assert len(datasets.hf_search("", get_device(ELITEBOOK), task="text-classification", limit=2, fetch=fetch)) == 2


def test_kaggle_datasets():
    data = [{"ref": "zynicide/wine-reviews", "title": "Wine Reviews", "totalBytes": 53_000_000,
             "downloadCount": 250_000, "licenseName": "CC BY-NC-SA 4.0"}]
    cards = datasets.kaggle_search("wine", get_device(ELITEBOOK), fetcher=lambda url: data)
    c = cards[0]
    assert (c.id, c.source, c.license, c.downloads) == ("zynicide/wine-reviews", "kaggle", "CC BY-NC-SA 4.0", 250_000)
    assert c.download_gb == pytest.approx(0.049, abs=0.001) and "kaggle datasets download -d" in c.how
    assert any("doesn't preview" in a for a in c.advice)


# ---- local files ----

@pytest.fixture
def local(tmp_path):
    with (tmp_path / "reviews.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "stars"])
        for i in range(500):
            w.writerow([f"Battery lasts long {i}", "pos", 5])
    with (tmp_path / "chat.jsonl").open("w") as f:
        for _ in range(200):
            f.write(json.dumps({"messages": [{"role": "user", "content": "Habari " * 20},
                                             {"role": "assistant", "content": "Nzuri sana " * 40}]}) + "\n")
    (tmp_path / "items.json").write_text(json.dumps({"data": [{"name": "a", "price": 1.5}, {"name": "b", "price": 2}]}))
    (tmp_path / "notes.tsv").write_text("q\ta\nhi\thello\n")
    return tmp_path


def test_local_csv(local):
    c = datasets.card(str(local / "reviews.csv"), get_device(ELITEBOOK))
    assert c.source == "local" and c.rows == 500
    assert [(col.name, col.type) for col in c.columns] == [("text", "string"), ("label", "string"), ("stars", "number")]
    assert c.preview[0] == {"text": "Battery lasts long 0", "label": "pos", "stars": "5"}


def test_local_chat_jsonl_is_recognised_as_chat(local):
    c = datasets.card(str(local / "chat.jsonl"), get_device(ELITEBOOK))
    assert c.rows == 200 and c.modality.startswith("chat")
    # Only the text counts: "Habari " * 20 + "Nzuri sana " * 40 + the two roles.
    assert c.avg_tokens_per_row == pytest.approx((7 * 20 + 11 * 40 + 4 + 9) / 4, abs=0.5)


def test_local_json_tsv_and_folders(local):
    j = datasets.card(str(local / "items.json"), get_device(ELITEBOOK))
    assert j.rows == 2 and [c.name for c in j.columns] == ["name", "price"]
    folder = datasets.card(str(local), get_device(ELITEBOOK))
    assert folder.rows == 500 + 200 + 2 + 1 and len(folder.splits) == 4
    with pytest.raises(ValueError):
        datasets.card(str(local / "nothing-here"), get_device(ELITEBOOK))


# ---- fine-tuning time ----

def test_fine_tuning_time_here_and_in_the_cloud(local):
    c = datasets.card(str(local / "chat.jsonl"), get_device(ELITEBOOK))
    tf = datasets.training_fit(c, "qwen2.5-1.5b", get_device(ELITEBOOK), epochs=2)
    assert tf.fits and tf.method == "lora"
    assert tf.tokens == int(200 * c.avg_tokens_per_row)
    flops = 4 * 1_540_000_000 * tf.tokens * 2
    assert tf.hours_cloud["A100"] == pytest.approx(flops / 120e12 / 3600, abs=0.01)
    assert tf.hours_here > tf.hours_cloud["Colab / Kaggle T4 (free tier)"] * 10  # a laptop CPU is far slower
    gpu = datasets.training_fit(c, "qwen2.5-1.5b", get_device("rtx-3060-12gb"))
    assert gpu.hours_here is not None and gpu.hours_here < tf.hours_here


def test_long_rows_are_flagged():
    c = datasets.DataCard(id="x", source="local", device="d", rows=1000, avg_tokens_per_row=3000)
    tf = datasets.training_fit(c, "qwen2.5-0.5b", get_device(ELITEBOOK), seq_len=1024)
    assert tf.tokens == 1000 * 1024 and any("cuts them at 1024" in a for a in tf.advice)


# ---- MCP and CLI ----

def test_mcp_tools(local, monkeypatch):
    monkeypatch.setattr(mcp, "_local", lambda: get_device(ELITEBOOK))
    out = mcp.call_tool("dataset_card", {"dataset": str(local / "chat.jsonl"), "for_model": "qwen2.5-0.5b"})
    card = out["structuredContent"]
    assert card["rows"] == 200 and card["training"]["fits"]
    assert mcp.call_tool("dataset_card", {"dataset": "!!"})["isError"]
    monkeypatch.setattr(catalog, "fetch_json", fetch)
    monkeypatch.setattr(datasets, "fetch_json", fetch)
    found = mcp.call_tool("search_datasets", {"query": "imdb"})["structuredContent"]
    assert found["datasets"][0]["id"] == "stanfordnlp/imdb"


def test_cli(local):
    from typer.testing import CliRunner

    from nanomesh.cli import app

    out = CliRunner().invoke(app, ["data", "card", str(local / "chat.jsonl"), "-d", ELITEBOOK,
                                   "--for-model", "qwen2.5-0.5b", "--json"])
    data = json.loads(out.output)
    assert data["modality"].startswith("chat") and data["training"]["method"] == "lora"


def test_fine_tuning_counts_only_the_training_split():
    # IMDb: 25k train, 25k test, 50k unsupervised. Only train is fine-tuned on.
    c = datasets.hf_card("stanfordnlp/imdb", get_device(ELITEBOOK), fetch)
    tf = datasets.training_fit(c, "qwen2.5-1.5b", get_device(ELITEBOOK))
    assert tf.tokens == int(25_000 * c.avg_tokens_per_row)
    assert tf.advice[0] == "Counting the training split: 25,000 of 100,000 rows."
