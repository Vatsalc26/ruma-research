from typing import Iterable, Optional

import torch

from memory_shards import MemoryRecord


def sequence_prefix_signature(payload, width=2):
    tokens = list(payload) if payload is not None else []
    if not tokens:
        return "unknown"
    prefix = [str(int(token)) for token in tokens[: max(1, int(width))]]
    return ":".join(prefix)


def sequence_anchor_signatures(payload, width=2, max_anchors=12):
    tokens = list(payload) if payload is not None else []
    width = max(1, int(width))
    if not tokens:
        return []
    if len(tokens) <= width:
        return [":".join(str(int(token)) for token in tokens)]

    anchors = []
    seen = set()
    max_start = max(0, len(tokens) - width)
    for start in range(max_start + 1):
        anchor = ":".join(str(int(token)) for token in tokens[start : start + width])
        if anchor in seen:
            continue
        seen.add(anchor)
        anchors.append(anchor)
        if len(anchors) >= max_anchors:
            break
    return anchors


def sequence_lineage_from_payload(payload, namespace="default", width=2):
    prefix_signature = sequence_prefix_signature(payload, width=width)
    return f"{namespace}::sequence::{prefix_signature}"


def sequence_lineage_key_from_input_ids(input_ids_row, width=2):
    payload = input_ids_row.detach().cpu().tolist()
    return sequence_prefix_signature(payload, width=width)


def sequence_anchor_keyset_from_input_ids(input_ids_row, width=2, max_anchors=12):
    payload = input_ids_row.detach().cpu().tolist()
    return set(sequence_anchor_signatures(payload, width=width, max_anchors=max_anchors))


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

    def _supersede_prior_lineage_records(self, namespace, lineage_key):
        prior_record_ids = []
        for record in self.memory_store.records(namespaces=[namespace], statuses=["active"]):
            if record.content_type != "sequence_chunk":
                continue
            if record.metadata.get("lineage_key") != lineage_key:
                continue
            prior_record_ids.append(record.record_id)

        if not prior_record_ids:
            return 0
        return self.memory_store.update_status(prior_record_ids, "superseded")

    @torch.no_grad()
    def insert_batch(
        self,
        input_ids,
        target_ids=None,
        sources: Optional[Iterable[str]] = None,
        namespaces: Optional[Iterable[str]] = None,
        timestamps: Optional[Iterable[str]] = None,
        causal: bool = False,
        supersede_prior_lineage: bool = False,
    ):
        device = self.embedding.weight.device
        input_ids = input_ids.to(device)
        if target_ids is not None:
            target_ids = target_ids.to(device)
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
            lineage_key = sequence_prefix_signature(payload)
            anchor_signatures = sequence_anchor_signatures(payload)
            payload_token_set = sorted({str(int(token)) for token in payload})
            if supersede_prior_lineage:
                self._supersede_prior_lineage_records(
                    namespace=namespace_list[i],
                    lineage_key=lineage_key,
                )
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
                    "prefix_signature": lineage_key,
                    "lineage_key": lineage_key,
                    "anchor_signatures": "|".join(anchor_signatures),
                    "payload_token_set": "|".join(payload_token_set),
                    "lineage": sequence_lineage_from_payload(payload, namespace=namespace_list[i]),
                },
            )
            records.append(record)

        self.memory_store.insert_many(records)
        return records
