from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BEIR_URLS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
}


@dataclass(frozen=True)
class BeirDataset:
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def _download_and_extract(dataset: str, root: Path) -> Path:
    dataset_dir = root / dataset
    if dataset_dir.exists():
        return dataset_dir

    if dataset not in BEIR_URLS:
        raise ValueError(f"Nepoznat dataset: {dataset}. Dostupno: {list(BEIR_URLS)}")

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / f"{dataset}.zip"

    if not zip_path.exists():
        urllib.request.urlretrieve(BEIR_URLS[dataset], zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    return dataset_dir


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_beir(
    dataset: str = "scifact",
    root: str | Path = "data/cache/beir",
    split: str = "test",
) -> BeirDataset:
    root = Path(root)
    dataset_dir = _download_and_extract(dataset, root)

    corpus: dict[str, str] = {}
    for row in _read_jsonl(dataset_dir / "corpus.jsonl"):
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        corpus[row["_id"]] = f"{title} {text}".strip()

    queries: dict[str, str] = {}
    for row in _read_jsonl(dataset_dir / "queries.jsonl"):
        queries[row["_id"]] = row["text"]

    qrels: dict[str, dict[str, int]] = {}
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"

    with qrels_path.open("r", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            if not line.strip():
                continue
            qid, did, score = line.strip().split("\t")[:3]
            qrels.setdefault(qid, {})[did] = int(float(score))

    return BeirDataset(corpus=corpus, queries=queries, qrels=qrels)