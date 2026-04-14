from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn.functional as F


@dataclass
class MemoryRecord:
    key: torch.Tensor
    value: torch.Tensor
    shard_id: int
    record_id: str = "unknown"
    namespace: str = "default"
    content_type: str = "chunk"
    status: str = "active"
    source: str = "unknown"
    timestamp: str = "unknown"
    payload: object = None
    metadata: Dict[str, str] = field(default_factory=dict)


class MemoryShardStore:
    """
    A small in-memory shard store for routed memory experiments.
    This is a research scaffold, not a production vector database.
    """

    def __init__(self, num_shards, d_model, capacity_per_shard=256):
        self.num_shards = num_shards
        self.d_model = d_model
        self.capacity_per_shard = capacity_per_shard
        self.shards: List[List[MemoryRecord]] = [[] for _ in range(num_shards)]

    def insert(self, record: MemoryRecord):
        shard_id = int(record.shard_id) % self.num_shards
        record = MemoryRecord(
            key=record.key.detach().cpu().clone(),
            value=record.value.detach().cpu().clone(),
            shard_id=shard_id,
            record_id=record.record_id,
            namespace=record.namespace,
            content_type=record.content_type,
            status=record.status,
            source=record.source,
            timestamp=record.timestamp,
            payload=deepcopy(record.payload),
            metadata=dict(record.metadata),
        )
        self.shards[shard_id].append(record)
        if len(self.shards[shard_id]) > self.capacity_per_shard:
            self.shards[shard_id].pop(0)

    def insert_many(self, records):
        for record in records:
            self.insert(record)

    def update_status(self, record_ids, status):
        record_id_set = set(record_ids)
        updated = 0
        for shard in self.shards:
            for record in shard:
                if record.record_id in record_id_set:
                    record.status = status
                    updated += 1
        return updated

    def _active_records(self, shard_id):
        return [
            record
            for record in self.shards[int(shard_id) % self.num_shards]
            if record.status == "active"
        ]

    def active_records(self, namespaces=None):
        namespace_filter = set(namespaces) if namespaces is not None else None
        records = []
        for shard in self.shards:
            for record in shard:
                if record.status != "active":
                    continue
                if namespace_filter is not None and record.namespace not in namespace_filter:
                    continue
                records.append(record)
        return records

    def records(self, namespaces=None, statuses=None):
        namespace_filter = set(namespaces) if namespaces is not None else None
        status_filter = set(statuses) if statuses is not None else None
        records = []
        for shard in self.shards:
            for record in shard:
                if namespace_filter is not None and record.namespace not in namespace_filter:
                    continue
                if status_filter is not None and record.status not in status_filter:
                    continue
                records.append(record)
        return records

    def top_records(self, shard_id, query, top_k=1):
        records = self._active_records(shard_id)
        if not records:
            return []

        top_k = max(1, min(top_k, len(records)))
        keys = torch.stack([record.key.to(query.device) for record in records], dim=0)
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        key_norm = F.normalize(keys, dim=-1)
        similarities = torch.matmul(query_norm, key_norm.transpose(0, 1)).squeeze(0)
        scores, indices = torch.topk(similarities, k=top_k)

        return [
            (records[int(index)], float(score))
            for score, index in zip(scores.tolist(), indices.tolist())
        ]

    def query(self, shard_id, query, top_k=1):
        scored_records = self.top_records(shard_id=shard_id, query=query, top_k=top_k)
        if not scored_records:
            return torch.zeros(self.d_model, device=query.device, dtype=query.dtype)

        scores = torch.tensor(
            [score for _, score in scored_records],
            device=query.device,
            dtype=query.dtype,
        )
        weights = torch.softmax(scores, dim=0)
        selected_values = torch.stack(
            [record.value.to(query.device) for record, _ in scored_records],
            dim=0,
        )
        return torch.sum(selected_values * weights.unsqueeze(-1), dim=0)

    def search(self, query, top_k=4, namespaces=None):
        records = self.active_records(namespaces=namespaces)
        if not records:
            return []

        top_k = max(1, min(top_k, len(records)))
        keys = torch.stack([record.key.to(query.device) for record in records], dim=0)
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        key_norm = F.normalize(keys, dim=-1)
        similarities = torch.matmul(query_norm, key_norm.transpose(0, 1)).squeeze(0)
        scores, indices = torch.topk(similarities, k=top_k)

        return [
            (records[int(index)], float(score))
            for score, index in zip(scores.tolist(), indices.tolist())
        ]

    def stats(self):
        counts = [len(shard) for shard in self.shards]
        namespace_counts: Dict[str, int] = {}
        content_type_counts: Dict[str, int] = {}
        status_counts: Dict[str, int] = {}
        payload_bytes = 0
        active_payload_bytes = 0
        for shard in self.shards:
            for record in shard:
                namespace_counts[record.namespace] = namespace_counts.get(record.namespace, 0) + 1
                content_type_counts[record.content_type] = (
                    content_type_counts.get(record.content_type, 0) + 1
                )
                status_counts[record.status] = status_counts.get(record.status, 0) + 1
                payload_size = len(str(record.payload).encode("utf-8")) if record.payload is not None else 0
                payload_bytes += payload_size
                if record.status == "active":
                    active_payload_bytes += payload_size
        return {
            "num_shards": self.num_shards,
            "capacity_per_shard": self.capacity_per_shard,
            "records_per_shard": counts,
            "total_records": sum(counts),
            "active_records": status_counts.get("active", 0),
            "namespace_counts": namespace_counts,
            "content_type_counts": content_type_counts,
            "status_counts": status_counts,
            "payload_bytes": payload_bytes,
            "active_payload_bytes": active_payload_bytes,
        }
