import heapq
import math
import random

import numpy as np


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.dot(a, b))


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

_METRICS = {
    "cosine": _cosine_distance,
    "euclidean": _euclidean_distance,
}


class HNSW:
    """HNSW graph from scratch (Malkov & Yashunin, 2018).

    - Layered structure with exponential level distribution (mL = 1/ln(M))
    - Insert: greedy descent to the level of the new node, then search_layer with
    ef_construction per layer + bidirectional links + prune via heuristic
    - Search: greedy descent from top to layer 1, then search_layer with ef_search
    on layer 0
    """
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        metric: str = "cosine",
        seed: int = 42,
        neighbor_selection: str = "heuristic",
    ) -> None:
        if metric not in _METRICS:
            raise ValueError(f"Unknown metrics: {metric}. Avilable: {list(_METRICS)}")
        if neighbor_selection not in ("simple", "heuristic"):
            raise ValueError("neighbor_selection must be 'simple' or 'heuristic'")
        if M < 2:
            raise ValueError("M mus be >= 2 (mL = 1/ln(M))")

        self.dim = dim
        self.M = M
        self.M_max = M          # upper layer
        self.M_max0 = 2 * M     # layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric = metric
        self.neighbor_selection = neighbor_selection

        self._dist = _METRICS[metric]
        self._rng = random.Random(seed)
        self._mL = 1.0 / math.log(M)

        # storage
        self.vectors: list[np.ndarray] = []
        self.ids: list[str] = []
        self.id_to_idx: dict[str, int] = {}

        # layers[layer_idx] = {node_idx: [neighbor_idx, ...]}
        self.layers: list[dict[int, list[int]]] = []

        self.entry_point: int | None = None
        self.max_level: int = -1

    def _prepare(self, vector: np.ndarray) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float32)
        if v.ndim != 1 or v.shape[0] != self.dim:
            raise ValueError(f"Očekivan vektor dim={self.dim}, dobio {v.shape}")
        if self.metric == "cosine":
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
        return v

    def _score(self, distance: float) -> float:
        # higher score -> better
        return 1.0 - distance if self.metric == "cosine" else -distance

    def _random_level(self) -> int:
        u = self._rng.random()
        while u == 0.0:
            u = self._rng.random()
        return int(-math.log(u) * self._mL)

    def _search_layer(
            self,
            query: np.ndarray,
            entry_points: list[int],
            ef: int,
            layer: int,
    ) -> list[tuple[float, int]]:
        """Returns list (distance, node_idx) sorted by increasing distance."""
        graph = self.layers[layer]
        visited: set[int] = set(entry_points)

        candidates: list[tuple[float, int]] = []  # min-heap
        W: list[tuple[float, int]] = []  # max-heap (neg. dist)

        for ep in entry_points:
            d = self._dist(query, self.vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(W, (-d, ep))

        while candidates:
            d_c, c = heapq.heappop(candidates)
            if W and d_c > -W[0][0]:
                break
            for nb in graph.get(c, ()):
                if nb in visited:
                    continue
                visited.add(nb)
                d_nb = self._dist(query, self.vectors[nb])
                if len(W) < ef or d_nb < -W[0][0]:
                    heapq.heappush(candidates, (d_nb, nb))
                    heapq.heappush(W, (-d_nb, nb))
                    if len(W) > ef:
                        heapq.heappop(W)

        return sorted((-d, i) for d, i in W)

    def _greedy_search(self, query: np.ndarray, ep: int, layer: int) -> int:
        res = self._search_layer(query, [ep], 1, layer)
        return res[0][1] if res else ep

    # ----------------------------------------------------------- neighbors
    def _select_neighbors(
            self,
            candidates: list[tuple[float, int]],
            M: int,
    ) -> list[int]:
        if self.neighbor_selection == "simple":
            return [idx for _, idx in candidates[:M]]

        selected: list[tuple[float, int]] = []
        for d_c, c in candidates:
            if len(selected) >= M:
                break
            ok = True
            vc = self.vectors[c]
            for _, r in selected:
                if self._dist(vc, self.vectors[r]) < d_c:
                    ok = False
                    break
            if ok:
                selected.append((d_c, c))
        return [idx for _, idx in selected]

    def _prune(
            self,
            node_idx: int,
            neighbor_list: list[int],
            M: int,
    ) -> list[int]:
        q = self.vectors[node_idx]
        cands = sorted(
            (self._dist(q, self.vectors[nb]), nb) for nb in neighbor_list
        )
        return self._select_neighbors(cands, M)

    # --------------------------------------------------------------- insert
    def insert(self, vector: np.ndarray, doc_id: str) -> None:
        if doc_id in self.id_to_idx:
            raise ValueError(f"Duplikat doc_id: {doc_id}")

        v = self._prepare(vector)
        idx = len(self.vectors)
        self.vectors.append(v)
        self.ids.append(doc_id)
        self.id_to_idx[doc_id] = idx

        level = self._random_level()
        # ensures that there are all layer from level i empty list
        while len(self.layers) <= level:
            self.layers.append({})
        for lc in range(level + 1):
            self.layers[lc][idx] = []

        if self.entry_point is None:
            self.entry_point = idx
            self.max_level = level
            return

        ep = self.entry_point

        # greedy descent
        for lc in range(self.max_level, level, -1):
            ep = self._greedy_search(v, ep, lc)
        # from min(level,max_level) to 0
        top = min(level, self.max_level)
        for lc in range(top, -1, -1):
            candidates = self._search_layer(v, [ep], self.ef_construction, lc)
            Mmax = self.M_max0 if lc == 0 else self.M_max
            neighbors = self._select_neighbors(candidates, Mmax)
            self.layers[lc][idx] = list(neighbors)

            for nb in neighbors:
                lst = self.layers[lc][nb]
                lst.append(idx)
                if len(lst) > Mmax:
                    self.layers[lc][nb] = self._prune(nb, lst, Mmax)

            if candidates:
                ep = candidates[0][1]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = idx

    # --------------------------------------------------------------- search
    def search(
            self,
            query: np.ndarray,
            k: int = 10,
            ef_search: int | None = None,
    ) -> list[tuple[str, float]]:
        if self.entry_point is None:
            return []
        q = self._prepare(query)
        ef = max(ef_search or self.ef_search, k)

        ep = self.entry_point
        for lc in range(self.max_level, 0, -1):
            ep = self._greedy_search(q, ep, lc)

        result = self._search_layer(q, [ep], ef, 0)
        return [(self.ids[i], self._score(d)) for d, i in result[:k]]

    # ----------------------------------------------------------------- meta
    def __len__(self) -> int:
        return len(self.vectors)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.id_to_idx

    def stats(self) -> dict:
        return {
            "size": len(self.vectors),
            "max_level": self.max_level,
            "layers": len(self.layers),
            "metric": self.metric,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "neighbor_selection": self.neighbor_selection,
        }