from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _normalize_numpy_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _normalize_numpy_query(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return vector
    return vector / norm


class SearchBackendBase:
    backend_name = "base"
    unavailable_reason = None

    def __init__(self, oversample_factor=8, extra_candidates=32):
        self.oversample_factor = max(2, int(oversample_factor))
        self.extra_candidates = max(8, int(extra_candidates))
        self.records = []

    def available(self):
        return True

    def build(self, records):
        self.records = list(records)

    def _candidate_search(self, query_vector: torch.Tensor, top_k: int):
        raise NotImplementedError

    def search(self, query_vector: torch.Tensor, top_k=4, namespaces=None):
        if not self.records:
            return []

        namespace_filter = set(namespaces) if namespaces is not None else None
        if namespace_filter is None:
            return self._candidate_search(query_vector, top_k=max(1, min(top_k, len(self.records))))

        candidate_k = min(
            len(self.records),
            max(top_k * self.oversample_factor, top_k + self.extra_candidates),
        )
        while True:
            hits = self._candidate_search(query_vector, top_k=max(1, candidate_k))
            filtered_hits = [
                (record, score)
                for record, score in hits
                if record.namespace in namespace_filter
            ]
            if len(filtered_hits) >= top_k or candidate_k >= len(self.records):
                return filtered_hits[:top_k]
            candidate_k = min(len(self.records), max(candidate_k * 2, candidate_k + top_k))


class ExactSearchBackend(SearchBackendBase):
    backend_name = "exact"

    def build(self, records):
        super().build(records)
        if not self.records:
            self.key_matrix = None
            self.record_namespaces = []
            return

        self.key_matrix = torch.stack(
            [record.key.detach().cpu().float() for record in self.records],
            dim=0,
        )
        self.key_matrix = F.normalize(self.key_matrix, dim=-1)
        self.record_namespaces = [record.namespace for record in self.records]

    def search(self, query_vector: torch.Tensor, top_k=4, namespaces=None):
        if self.key_matrix is None or len(self.records) == 0:
            return []

        normalized_query = F.normalize(query_vector.detach().cpu().float().unsqueeze(0), dim=-1)

        if namespaces is None:
            key_matrix = self.key_matrix
            candidate_records = self.records
        else:
            namespace_filter = set(namespaces)
            candidate_indices = [
                index
                for index, namespace in enumerate(self.record_namespaces)
                if namespace in namespace_filter
            ]
            if not candidate_indices:
                return []
            key_matrix = self.key_matrix[candidate_indices]
            candidate_records = [self.records[index] for index in candidate_indices]

        similarities = torch.matmul(normalized_query, key_matrix.transpose(0, 1)).squeeze(0)
        top_k = max(1, min(top_k, similarities.shape[0]))
        scores, indices = torch.topk(similarities, k=top_k)
        return [
            (candidate_records[int(index)], float(score))
            for score, index in zip(scores.tolist(), indices.tolist())
        ]

    def _candidate_search(self, query_vector: torch.Tensor, top_k: int):
        if self.key_matrix is None or len(self.records) == 0:
            return []

        normalized_query = F.normalize(query_vector.detach().cpu().float().unsqueeze(0), dim=-1)
        similarities = torch.matmul(normalized_query, self.key_matrix.transpose(0, 1)).squeeze(0)
        top_k = max(1, min(top_k, similarities.shape[0]))
        scores, indices = torch.topk(similarities, k=top_k)
        return [
            (self.records[int(index)], float(score))
            for score, index in zip(scores.tolist(), indices.tolist())
        ]


class FaissFlatSearchBackend(SearchBackendBase):
    backend_name = "faiss_flat"

    def __init__(self, oversample_factor=8, extra_candidates=32):
        super().__init__(oversample_factor=oversample_factor, extra_candidates=extra_candidates)
        try:
            import faiss  # type: ignore

            self.faiss = faiss
        except ImportError:
            self.faiss = None
            self.unavailable_reason = "faiss Python package is not installed"
        self.index = None

    def available(self):
        return self.faiss is not None

    def build(self, records):
        super().build(records)
        if not self.records or not self.available():
            self.index = None
            return

        matrix = np.stack(
            [record.key.detach().cpu().float().numpy() for record in self.records],
            axis=0,
        ).astype("float32")
        matrix = _normalize_numpy_rows(matrix)
        self.index = self.faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)

    def _candidate_search(self, query_vector: torch.Tensor, top_k: int):
        if self.index is None or not self.records:
            return []

        query = query_vector.detach().cpu().float().numpy().astype("float32").reshape(1, -1)
        query = _normalize_numpy_query(query)
        top_k = max(1, min(top_k, len(self.records)))
        scores, indices = self.index.search(query, top_k)
        hits = []
        for score, index in zip(scores[0].tolist(), indices[0].tolist()):
            if index < 0:
                continue
            hits.append((self.records[int(index)], float(score)))
        return hits


class HnswSearchBackend(SearchBackendBase):
    backend_name = "hnsw"

    def __init__(self, oversample_factor=8, extra_candidates=32):
        super().__init__(oversample_factor=oversample_factor, extra_candidates=extra_candidates)
        try:
            import hnswlib  # type: ignore

            self.hnswlib = hnswlib
        except ImportError:
            self.hnswlib = None
            self.unavailable_reason = "hnswlib Python package is not installed"
        self.index = None

    def available(self):
        return self.hnswlib is not None

    def build(self, records):
        super().build(records)
        if not self.records or not self.available():
            self.index = None
            return

        matrix = np.stack(
            [record.key.detach().cpu().float().numpy() for record in self.records],
            axis=0,
        ).astype("float32")
        matrix = _normalize_numpy_rows(matrix)

        dim = matrix.shape[1]
        self.index = self.hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=len(self.records), ef_construction=100, M=16)
        self.index.add_items(matrix, np.arange(len(self.records)))
        self.index.set_ef(min(max(64, self.extra_candidates * 2), max(64, len(self.records))))

    def _candidate_search(self, query_vector: torch.Tensor, top_k: int):
        if self.index is None or not self.records:
            return []

        query = query_vector.detach().cpu().float().numpy().astype("float32").reshape(1, -1)
        query = _normalize_numpy_query(query)
        top_k = max(1, min(top_k, len(self.records)))
        self.index.set_ef(min(max(64, top_k * 4), max(64, len(self.records))))
        labels, distances = self.index.knn_query(query, k=top_k)

        hits = []
        for label, distance in zip(labels[0].tolist(), distances[0].tolist()):
            if label < 0:
                continue
            score = 1.0 - float(distance)
            hits.append((self.records[int(label)], score))
        return hits


def build_search_backend(name: str, **kwargs):
    normalized = (name or "exact").strip().lower()
    if normalized == "exact":
        return ExactSearchBackend(**kwargs)
    if normalized in {"faiss", "faiss_flat"}:
        return FaissFlatSearchBackend(**kwargs)
    if normalized in {"hnsw", "hnswlib"}:
        return HnswSearchBackend(**kwargs)
    raise ValueError(f"Unknown search backend: {name}")


def search_backend_availability(names: Optional[Sequence[str]] = None):
    requested = list(names or ["exact", "faiss_flat", "hnsw"])
    availability = []
    for name in requested:
        backend = build_search_backend(name)
        availability.append(
            {
                "name": backend.backend_name,
                "available": backend.available(),
                "reason": backend.unavailable_reason,
            }
        )
    return availability


def available_search_backend_names():
    return [
        entry["name"]
        for entry in search_backend_availability()
        if entry["available"]
    ]
