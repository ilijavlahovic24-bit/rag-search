"""Latency benchmark for the API service.

Usage:
    python -m benchmarks.api_latency --sizes 1000 5000 --n-queries 100
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from api.service import MiniRagService


class FakeEncoder:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(0)
        out = rng.normal(size=(len(texts), self.dim)).astype(np.float32)
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out

    def encode_documents(self, texts, cache_key=None, show_progress=True):
        return self._encode(texts)

    def encode_queries(self, texts, cache_key=None, show_progress=False):
        return self._encode(texts)


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    idx = max(0, min(len(xs) - 1, int(p * len(xs)) - 1))
    return xs[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000])
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--mode", default="hybrid",
                    choices=["dense", "sparse", "hybrid"])
    args = ap.parse_args()

    print(f"| n_docs | mode | p50 (ms) | p95 (ms) | mean (ms) |")
    print(f"|---|---|---|---|---|")

    for n in args.sizes:
        svc = MiniRagService(model_name="fake")
        svc._encoder = FakeEncoder()  # type: ignore[attr-defined]

        docs = [(f"d{i}", f"document number {i} with some words") for i in range(n)]
        t0 = time.perf_counter()
        svc.index(docs)
        t_index = time.perf_counter() - t0
        print(f"# indexed {n} docs in {t_index:.2f}s", flush=True)

        latencies: list[float] = []
        for q in range(args.n_queries):
            t0 = time.perf_counter()
            svc.search(f"query {q}", top_k=args.top_k, mode=args.mode)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        p50 = statistics.median(latencies)
        p95 = percentile(latencies, 0.95)
        mean = statistics.mean(latencies)
        print(f"| {n} | {args.mode} | {p50:.2f} | {p95:.2f} | {mean:.2f} |", flush=True)


if __name__ == "__main__":
    main()