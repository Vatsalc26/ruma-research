import hashlib
import math
import re
from collections import Counter
from pathlib import Path

import torch

from ann_retrieval import available_search_backend_names, build_search_backend, search_backend_availability
from memory_shards import MemoryRecord, MemoryShardStore


TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-]+")
WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_CORPUS_PATHS = [
    "Hyena/Hyena Paper.md",
    "Mamba/Mamba Paper.md",
    "Mamba-2/Mamba-2 (State Space Duality) Paper.md",
    "RETRO/RETRO (Retrieval) Paper.md",
    "Retrieval-Augmented Generation/Retrieval-Augmented Generation.md",
    "Improving language models by retrieving/Improving language models by retrieving.md",
    "Switch Transformers/Switch Transformers.md",
    "GLaM/GLaM.md",
    "Jamba/Jamba Paper.md",
    "Knowledge Editing for Large Language Models/Knowledge Editing for Large Language Models.md",
    "Mass-Editing Memory in a Transformer/Mass-Editing Memory in a Transformer.md",
    "Locating and Editing Factual Associations in GPT/Locating and Editing Factual Associations in GPT.md",
    "MEMORIZING TRANSFORMERS/MEMORIZING TRANSFORMERS.md",
    "Neural Turing Machines/Neural Turing Machines.md",
    "End-To-End Memory Networks/End-To-End Memory Networks.md",
    "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.md",
    "faiss-main/README.md",
]


def stable_hash_int(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def slugify(text):
    lowered = text.lower()
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_") or "default"


def clean_text(text):
    # Normalize common mojibake and whitespace for stable chunking.
    replacements = {
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€”": "-",
        "â€“": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return WHITESPACE_RE.sub(" ", text).strip()


def chunk_text(text, max_words=140, overlap_words=35):
    words = clean_text(text).split()
    if not words:
        return []

    chunks = []
    step = max(1, max_words - overlap_words)
    start = 0
    while start < len(words):
        chunk_words = words[start : start + max_words]
        if len(chunk_words) < 35 and chunks:
            break
        chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break
        start += step
    return chunks


class HashedTfidfEncoder:
    def __init__(self, d_model=256):
        self.d_model = d_model
        self.doc_freq = Counter()
        self.num_docs = 0

    def tokenize(self, text):
        return TOKEN_RE.findall(clean_text(text).lower())

    def fit(self, texts):
        self.doc_freq.clear()
        self.num_docs = len(texts)
        for text in texts:
            seen = set(self.tokenize(text))
            for token in seen:
                self.doc_freq[token] += 1

    def encode(self, text):
        tokens = self.tokenize(text)
        vector = torch.zeros(self.d_model, dtype=torch.float32)
        if not tokens:
            return vector

        counts = Counter(tokens)
        for token, count in counts.items():
            index = stable_hash_int(token) % self.d_model
            idf = math.log((1.0 + self.num_docs) / (1.0 + self.doc_freq.get(token, 0))) + 1.0
            vector[index] += (1.0 + math.log(count)) * idf

        norm = torch.norm(vector)
        if norm.item() > 0:
            vector = vector / norm
        return vector


class RealDocRUMAIndex:
    """
    A real-document memory prototype for Phase 2.
    It ingests local markdown papers, stores chunk-level records with provenance,
    and retrieves cited passages for user queries.
    """

    def __init__(
        self,
        repo_root,
        d_model=256,
        num_shards=24,
        chunk_words=140,
        overlap_words=35,
        namespace_bandwidth=3,
        search_backend="exact",
        search_backend_kwargs=None,
    ):
        self.repo_root = Path(repo_root)
        self.d_model = d_model
        self.num_shards = num_shards
        self.namespace_bandwidth = max(1, min(namespace_bandwidth, num_shards))
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words
        self.encoder = HashedTfidfEncoder(d_model=d_model)
        self.store_capacity = 4096
        self.store = self._make_store()
        self.documents = []
        self.requested_search_backend = search_backend
        self.search_backend_kwargs = dict(search_backend_kwargs or {})
        self._search_backend = None
        self._search_backend_stale = True
        self._search_backend_status = {
            "requested": self.requested_search_backend,
            "active": None,
            "fallback_reason": None,
            "available_backends": available_search_backend_names(),
        }

    def _make_store(self):
        return MemoryShardStore(
            num_shards=self.num_shards,
            d_model=self.d_model,
            capacity_per_shard=self.store_capacity,
        )

    def _mark_search_backend_stale(self):
        self._search_backend_stale = True

    def set_search_backend(self, backend_name, **kwargs):
        self.requested_search_backend = backend_name
        self.search_backend_kwargs = dict(kwargs)
        self._mark_search_backend_stale()

    def search_backend_info(self):
        return {
            "requested": self.requested_search_backend,
            "active": None if self._search_backend is None else self._search_backend.backend_name,
            "stale": self._search_backend_stale,
            "fallback_reason": self._search_backend_status.get("fallback_reason"),
            "available_backends": search_backend_availability(),
        }

    def _ensure_search_backend(self):
        if not self._search_backend_stale and self._search_backend is not None:
            return

        requested_backend = build_search_backend(
            self.requested_search_backend,
            **self.search_backend_kwargs,
        )
        fallback_reason = None
        if not requested_backend.available():
            fallback_reason = requested_backend.unavailable_reason
            requested_backend = build_search_backend("exact")

        requested_backend.build(self.store.active_records())
        self._search_backend = requested_backend
        self._search_backend_stale = False
        self._search_backend_status = {
            "requested": self.requested_search_backend,
            "active": self._search_backend.backend_name,
            "fallback_reason": fallback_reason,
            "available_backends": available_search_backend_names(),
        }

    def default_corpus_paths(self):
        paths = []
        for relative_path in DEFAULT_CORPUS_PATHS:
            path = self.repo_root / relative_path
            if path.exists():
                paths.append(path)
        return paths

    def _namespace_for_path(self, path):
        relative = path.relative_to(self.repo_root)
        parts = relative.parts
        if len(parts) <= 1:
            return slugify(path.stem)
        return slugify(parts[0])

    def _prepare_entries(self, paths):
        entries = []
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="ignore")
            relative_path = path.relative_to(self.repo_root).as_posix()
            namespace = self._namespace_for_path(path)
            for chunk_index, chunk_body in enumerate(
                chunk_text(text, max_words=self.chunk_words, overlap_words=self.overlap_words)
            ):
                entries.append(
                    {
                        "path": relative_path,
                        "namespace": namespace,
                        "chunk_index": chunk_index,
                        "text": chunk_body,
                    }
                )
        return entries

    def _namespace_shard_band(self, namespace):
        width = max(1, min(self.namespace_bandwidth, self.num_shards))
        start = stable_hash_int(namespace) % self.num_shards
        return [int((start + offset) % self.num_shards) for offset in range(width)]

    def _assign_shard_id(self, namespace, source, chunk_index, text):
        band = self._namespace_shard_band(namespace)
        local_index = stable_hash_int(f"{source}::{chunk_index}::{text[:64]}") % len(band)
        return band[int(local_index)]

    def _build_records(self, entries, timestamp, lineage_suffix=""):
        records = []
        for entry in entries:
            key = self.encoder.encode(entry["text"])
            lineage = f"{entry['namespace']}::{entry['path']}{lineage_suffix}"
            shard_band = self._namespace_shard_band(entry["namespace"])
            shard_id = self._assign_shard_id(
                entry["namespace"],
                entry["path"],
                entry["chunk_index"],
                entry["text"],
            )
            record_id = f"{entry['namespace']}::{entry['path']}::{timestamp}::{entry['chunk_index']}"
            records.append(
                MemoryRecord(
                    key=key,
                    value=key,
                    shard_id=shard_id,
                    record_id=record_id,
                    namespace=entry["namespace"],
                    content_type="doc_chunk",
                    status="active",
                    source=entry["path"],
                    timestamp=timestamp,
                    payload=entry["text"],
                    metadata={
                        "chunk_index": str(entry["chunk_index"]),
                        "path": entry["path"],
                        "word_count": str(len(entry["text"].split())),
                        "lineage": lineage,
                        "shard_assignment_policy": "namespace_banded_hash",
                        "shard_band": ",".join(str(shard) for shard in shard_band),
                    },
                )
            )
        return records

    def supersede_lineage(self, lineage):
        prior_records = [
            record.record_id
            for record in self.store.records(statuses={"active"})
            if record.metadata.get("lineage") == lineage
        ]
        if not prior_records:
            return 0
        return self.store.update_status(prior_records, "superseded")

    def build_from_paths(self, paths=None):
        selected_paths = [Path(path) for path in (paths or self.default_corpus_paths())]
        entries = self._prepare_entries(selected_paths)
        self.encoder.fit([entry["text"] for entry in entries])
        self.store = self._make_store()
        records = self._build_records(entries, timestamp="static_corpus")

        self.store.insert_many(records)
        self.documents = selected_paths
        self._mark_search_backend_stale()
        return records

    def ingest_text_update(
        self,
        text,
        source,
        namespace="manual_update",
        timestamp="manual_update",
        lineage=None,
        supersede_prior=False,
    ):
        normalized_text = clean_text(text)
        if not normalized_text:
            return []

        if self.encoder.num_docs == 0:
            self.encoder.fit([normalized_text])

        chunks = chunk_text(normalized_text, max_words=self.chunk_words, overlap_words=self.overlap_words)
        normalized_namespace = slugify(namespace)
        normalized_lineage = lineage or f"{normalized_namespace}::{source}"
        if supersede_prior:
            self.supersede_lineage(normalized_lineage)

        entries = [
            {
                "path": source,
                "namespace": normalized_namespace,
                "chunk_index": chunk_index,
                "text": chunk,
            }
            for chunk_index, chunk in enumerate(chunks)
        ]
        records = self._build_records(entries, timestamp=timestamp)
        for record in records:
            record.metadata["lineage"] = normalized_lineage
            record.metadata["update_kind"] = "append_update"

        self.store.insert_many(records)
        self._mark_search_backend_stale()
        return records

    def lineage_records(self, lineage, statuses=None):
        return [
            record
            for record in self.store.records(statuses=statuses)
            if record.metadata.get("lineage") == lineage
        ]

    def refresh_shard_assignments(self, namespaces=None):
        namespace_filter = set(namespaces) if namespaces is not None else None
        prior_records = self.store.records()
        refreshed_records = []

        for record in prior_records:
            if namespace_filter is not None and record.namespace not in namespace_filter:
                refreshed_records.append(record)
                continue

            updated_metadata = dict(record.metadata)
            shard_band = self._namespace_shard_band(record.namespace)
            updated_metadata["shard_assignment_policy"] = "namespace_banded_hash"
            updated_metadata["shard_band"] = ",".join(str(shard) for shard in shard_band)
            refreshed_records.append(
                MemoryRecord(
                    key=record.key,
                    value=record.value,
                    shard_id=self._assign_shard_id(
                        record.namespace,
                        record.source,
                        int(record.metadata.get("chunk_index", "0")),
                        str(record.payload),
                    ),
                    record_id=record.record_id,
                    namespace=record.namespace,
                    content_type=record.content_type,
                    status=record.status,
                    source=record.source,
                    timestamp=record.timestamp,
                    payload=record.payload,
                    metadata=updated_metadata,
                )
            )

        self.store = self._make_store()
        self.store.insert_many(refreshed_records)
        self._mark_search_backend_stale()
        return refreshed_records

    def namespace_layout(self):
        layout = {}
        for record in self.store.records():
            namespace = record.namespace
            namespace_layout = layout.setdefault(
                namespace,
                {
                    "record_count": 0,
                    "active_count": 0,
                    "superseded_count": 0,
                    "shard_ids": set(),
                    "shard_band": record.metadata.get("shard_band", ""),
                },
            )
            namespace_layout["record_count"] += 1
            if record.status == "active":
                namespace_layout["active_count"] += 1
            if record.status == "superseded":
                namespace_layout["superseded_count"] += 1
            namespace_layout["shard_ids"].add(int(record.shard_id))

        result = {}
        for namespace, stats in layout.items():
            result[namespace] = {
                "record_count": stats["record_count"],
                "active_count": stats["active_count"],
                "superseded_count": stats["superseded_count"],
                "shard_ids": sorted(stats["shard_ids"]),
                "shard_count": len(stats["shard_ids"]),
                "shard_band": stats["shard_band"],
            }
        return result

    def search(self, query, top_k=4, namespaces=None):
        query_vector = self.encoder.encode(query)
        self._ensure_search_backend()
        scored_records = self._search_backend.search(query_vector, top_k=top_k, namespaces=namespaces)
        results = []
        for record, score in scored_records:
            results.append(
                {
                    "score": score,
                    "namespace": record.namespace,
                    "source": record.source,
                    "record_id": record.record_id,
                    "chunk_index": int(record.metadata.get("chunk_index", "0")),
                    "excerpt": record.payload,
                    "timestamp": record.timestamp,
                    "status": record.status,
                    "lineage": record.metadata.get("lineage", ""),
                }
            )
        return results

    def stats(self):
        shard_stats = self.store.stats()
        return {
            "documents_indexed": len(self.documents),
            "chunks_indexed": shard_stats["total_records"],
            "namespace_bandwidth": self.namespace_bandwidth,
            "namespace_count": len(shard_stats["namespace_counts"]),
            "namespaces": sorted(shard_stats["namespace_counts"].keys()),
            "search_backend": self.search_backend_info(),
            "store": shard_stats,
        }
