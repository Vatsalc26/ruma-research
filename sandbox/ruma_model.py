import torch
import torch.nn as nn

from fusion import GatedMemoryFusion
from intake import ContextSynthesizer
from memory_shards import MemoryShardStore
from router import LSHRouter
from updater import SequenceMemoryUpdater


class RUMAModel(nn.Module):
    """
    A cleaner routed-memory research skeleton aligned to the repo doctrine.
    It preserves the existing sandbox spirit but makes the update path explicit.
    """

    def __init__(
        self,
        vocab_size=5000,
        d_model=256,
        n_heads=8,
        num_shards=10,
        shard_capacity=256,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.context_encoder = ContextSynthesizer(d_model=d_model, n_heads=n_heads)
        self.router = LSHRouter(d_model=d_model, num_ponds=num_shards)
        self.memory_store = MemoryShardStore(
            num_shards=num_shards,
            d_model=d_model,
            capacity_per_shard=shard_capacity,
        )
        self.fusion = GatedMemoryFusion(d_model=d_model)
        self.decoder = nn.Linear(d_model, vocab_size)
        self.updater = SequenceMemoryUpdater(
            embedding=self.embedding,
            context_encoder=self.context_encoder,
            router=self.router,
            memory_store=self.memory_store,
        )

    def encode_context(self, input_ids, causal=False):
        x = self.embedding(input_ids)
        return self.context_encoder(x, causal=causal)

    def _fetch_memory(self, contextual, routes, top_k):
        memory_stream = torch.zeros_like(contextual)
        batch_size, seq_len, _ = contextual.shape
        for b in range(batch_size):
            for s in range(seq_len):
                shard_id = int(routes[b, s].item())
                memory_stream[b, s] = self.memory_store.query(
                    shard_id=shard_id,
                    query=contextual[b, s],
                    top_k=top_k,
                )
        return memory_stream

    def _sequence_chunk_logit_bias(self, input_ids, contextual, top_k):
        batch_size, seq_len = input_ids.shape
        vocab_size = self.decoder.out_features
        bias = torch.zeros(
            batch_size,
            seq_len,
            vocab_size,
            device=contextual.device,
            dtype=contextual.dtype,
        )

        summaries = contextual.mean(dim=1)
        summary_routes = self.router(summaries.unsqueeze(1)).squeeze(1)

        for b in range(batch_size):
            scored_records = self.memory_store.top_records(
                shard_id=int(summary_routes[b].item()),
                query=summaries[b],
                top_k=1,
            )
            for record, score in scored_records:
                if record.content_type != "sequence_chunk" or not isinstance(record.payload, list):
                    continue

                payload = record.payload
                payload_len = len(payload)
                max_prefix = min(seq_len, payload_len - 1)
                input_list = input_ids[b].detach().cpu().tolist()

                for s in range(max_prefix):
                    query_prefix = input_list[: s + 1]
                    if query_prefix == payload[: s + 1]:
                        next_token = int(payload[s + 1])
                        bias[b, s, next_token] += 1.0 + max(0.0, float(score))
                        continue

                    longest_matches = []
                    max_window = min(s + 1, payload_len - 1, 24)
                    for window_size in range(max_window, 0, -1):
                        query_window = input_list[s - window_size + 1 : s + 1]
                        max_start = payload_len - window_size - 1
                        matches = []
                        for start in range(max_start + 1):
                            payload_window = payload[start : start + window_size]
                            if query_window == payload_window:
                                matches.append(int(payload[start + window_size]))

                        if matches:
                            longest_matches = matches
                            match_weight = float(window_size) / float(max_window)
                            score_weight = 1.0 + max(0.0, float(score))
                            for next_token in longest_matches:
                                bias[b, s, next_token] += match_weight * score_weight
                            break

        return bias

    def forward(self, input_ids, top_k=1, return_aux=False, use_memory=True, causal=False):
        contextual = self.encode_context(input_ids, causal=causal)
        routes = None
        fused = contextual
        payload_bias = None

        if use_memory and top_k > 0:
            routes = self.router(contextual)
            memory_stream = self._fetch_memory(contextual, routes, top_k=top_k)
            fused = self.fusion(contextual, memory_stream)
            payload_bias = self._sequence_chunk_logit_bias(input_ids, contextual, top_k=top_k)

        logits = self.decoder(fused)
        if payload_bias is not None:
            logits = logits + 20.0 * payload_bias

        if return_aux:
            return logits, {
                "routes": routes,
                "memory_used": bool(use_memory and top_k > 0),
                "memory_stats": self.memory_store.stats(),
            }
        return logits

    @torch.no_grad()
    def update_memory(
        self,
        input_ids,
        target_ids=None,
        sources=None,
        namespaces=None,
        timestamps=None,
        causal=False,
    ):
        return self.updater.insert_batch(
            input_ids=input_ids,
            target_ids=target_ids,
            sources=sources,
            namespaces=namespaces,
            timestamps=timestamps,
            causal=causal,
        )

    def memory_stats(self):
        return self.memory_store.stats()
