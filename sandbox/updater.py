from typing import Iterable, Optional

import torch

from memory_shards import MemoryRecord


class SequenceMemoryUpdater:
    """
    Encodes sequences into shard-routed memory records.
    This is the first-class write path for the routed-memory sandbox.
    """

    def __init__(self, embedding, context_encoder, router, memory_store):
        self.embedding = embedding
        self.context_encoder = context_encoder
        self.router = router
        self.memory_store = memory_store

    @torch.no_grad()
    def insert_batch(
        self,
        input_ids,
        target_ids=None,
        sources: Optional[Iterable[str]] = None,
        namespaces: Optional[Iterable[str]] = None,
        timestamps: Optional[Iterable[str]] = None,
        causal: bool = False,
    ):
        x = self.embedding(input_ids)
        contextual = self.context_encoder(x, causal=causal)
        summaries = contextual.mean(dim=1)
        shard_ids = self.router(summaries.unsqueeze(1)).squeeze(1)

        batch_size = input_ids.shape[0]
        source_list = list(sources) if sources is not None else ["unknown"] * batch_size
        namespace_list = list(namespaces) if namespaces is not None else ["default"] * batch_size
        timestamp_list = list(timestamps) if timestamps is not None else ["unknown"] * batch_size

        records = []
        for i in range(batch_size):
            record_id = f"{namespace_list[i]}::{source_list[i]}::{timestamp_list[i]}::{i}"
            payload = input_ids[i].detach().cpu().tolist()
            if target_ids is not None:
                payload = payload + [int(target_ids[i, -1].item())]
            record = MemoryRecord(
                key=summaries[i],
                value=summaries[i],
                shard_id=int(shard_ids[i].item()),
                record_id=record_id,
                namespace=namespace_list[i],
                content_type="sequence_chunk",
                status="active",
                source=source_list[i],
                timestamp=timestamp_list[i],
                payload=payload,
                metadata={
                    "sequence_length": str(int(input_ids.shape[1])),
                    "payload_length": str(len(payload)),
                    "ingestion_policy": "sequence_chunk",
                    "batch_index": str(i),
                },
            )
            records.append(record)

        self.memory_store.insert_many(records)
        return records
