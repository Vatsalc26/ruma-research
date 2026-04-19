import torch
import torch.nn as nn

from intake import ContextSynthesizer
from memory_shards import MemoryShardStore
from router import LSHRouter
from ruma_v2_blocks import MambaBackboneBlock, RUMABlock, TransformerBackboneBlock
from updater import (
    SequenceMemoryUpdater,
    sequence_anchor_keyset_from_input_ids,
    sequence_lineage_key_from_input_ids,
)


def build_interleaved_schedule(num_backbone_blocks=8, num_ruma_blocks=4):
    schedule = []
    backbone_remaining = int(num_backbone_blocks)
    ruma_remaining = int(num_ruma_blocks)

    while backbone_remaining > 0 or ruma_remaining > 0:
        for _ in range(2):
            if backbone_remaining <= 0:
                break
            schedule.append("backbone")
            backbone_remaining -= 1
        if ruma_remaining > 0:
            schedule.append("ruma")
            ruma_remaining -= 1
        elif backbone_remaining > 0:
            schedule.append("backbone")
            backbone_remaining -= 1

    return schedule


def build_backbone_block(backbone_type, d_model, n_heads):
    normalized = str(backbone_type or "transformer").lower()
    if normalized == "transformer":
        return TransformerBackboneBlock(d_model=d_model, n_heads=n_heads)
    if normalized == "mamba":
        return MambaBackboneBlock(d_model=d_model)
    raise ValueError(f"Unsupported backbone_type: {backbone_type}")


class InterleavedRUMAModel(nn.Module):
    """
    First serious reference model following the reverse-designed blueprint:
    a stable Transformer backbone with repeated explicit RUMA blocks.
    """

    def __init__(
        self,
        vocab_size=5000,
        d_model=256,
        n_heads=8,
        num_shards=10,
        shard_capacity=256,
        num_backbone_blocks=8,
        num_ruma_blocks=4,
        top_k=4,
        backbone_type="transformer",
        use_selectivity=True,
        use_sparse_experts=True,
        use_lineage_filtering=True,
    ):
        super().__init__()
        self.backbone_type = str(backbone_type).lower()
        self.use_selectivity = bool(use_selectivity)
        self.use_sparse_experts = bool(use_sparse_experts)
        self.use_lineage_filtering = bool(use_lineage_filtering)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.router = LSHRouter(d_model=d_model, num_ponds=num_shards)
        self.memory_store = MemoryShardStore(
            num_shards=num_shards,
            d_model=d_model,
            capacity_per_shard=shard_capacity,
        )
        self.layer_schedule = build_interleaved_schedule(
            num_backbone_blocks=num_backbone_blocks,
            num_ruma_blocks=num_ruma_blocks,
        )
        self.layers = nn.ModuleList()
        for layer_type in self.layer_schedule:
            if layer_type == "backbone":
                self.layers.append(build_backbone_block(self.backbone_type, d_model=d_model, n_heads=n_heads))
            else:
                self.layers.append(
                    RUMABlock(
                        d_model=d_model,
                        router=self.router,
                        memory_store=self.memory_store,
                        top_k=top_k,
                        use_selectivity=self.use_selectivity,
                        use_sparse_experts=self.use_sparse_experts,
                        use_lineage_filtering=self.use_lineage_filtering,
                    )
                )

        self.final_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

        # Keep the write path simple and explicit while the interleaved stack is still maturing.
        self.write_encoder = ContextSynthesizer(d_model=d_model, n_heads=n_heads)
        self.updater = SequenceMemoryUpdater(
            embedding=self.embedding,
            context_encoder=self.write_encoder,
            router=self.router,
            memory_store=self.memory_store,
        )
        self.supersede_prior_lineage = True

    def iter_ruma_blocks(self):
        for layer_type, layer in zip(self.layer_schedule, self.layers):
            if layer_type == "ruma":
                yield layer

    def enable_controller_training(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

        for block in self.iter_ruma_blocks():
            for module in (
                block.query_projector,
                block.selectivity_head,
                block.fusion_bank,
                block.expert_router,
                block.post_norm,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = True

        for module in (self.final_norm, self.decoder):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def enable_parameter_edit_training(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

        for layer_type, layer in zip(self.layer_schedule, self.layers):
            if layer_type != "backbone":
                continue
            for parameter in layer.parameters():
                parameter.requires_grad = True

        for module in (self.embedding, self.final_norm, self.decoder):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def enable_memory_conditioned_text_training(self):
        """
        Broader standalone-text adaptation mode for the interleaved reference model.

        Compared with the narrower controller-only path, this mode keeps the
        memory-aware RUMA blocks trainable while also letting the backbone,
        embedding, and decoder adapt to memory-conditioned natural-language
        continuation.
        """
        for parameter in self.parameters():
            parameter.requires_grad = False

        for module in (self.embedding, self.write_encoder, self.final_norm, self.decoder):
            for parameter in module.parameters():
                parameter.requires_grad = True

        for layer_type, layer in zip(self.layer_schedule, self.layers):
            if layer_type == "backbone":
                for parameter in layer.parameters():
                    parameter.requires_grad = True
                continue

            for parameter in layer.parameters():
                parameter.requires_grad = True

    def encode_memory_queries(self, input_ids, causal=False):
        x = self.embedding(input_ids)
        return self.write_encoder(x, causal=causal)

    def encode_hidden(self, input_ids, top_k=4, use_memory=True, causal=False):
        x = self.embedding(input_ids)
        block_summaries = []
        memory_queries = self.encode_memory_queries(input_ids, causal=causal) if use_memory and top_k > 0 else None
        query_lineage_keys = (
            [sequence_lineage_key_from_input_ids(input_ids[i]) for i in range(input_ids.shape[0])]
            if use_memory and top_k > 0
            else None
        )
        query_anchor_keysets = (
            [sequence_anchor_keyset_from_input_ids(input_ids[i]) for i in range(input_ids.shape[0])]
            if use_memory and top_k > 0
            else None
        )
        query_entity_tokens = (
            [int(input_ids[i, 0].item()) for i in range(input_ids.shape[0])]
            if use_memory and top_k > 0
            else None
        )

        for layer_type, layer in zip(self.layer_schedule, self.layers):
            if layer_type == "backbone":
                x = layer(x, causal=causal)
                continue

            if use_memory and top_k > 0:
                x, block_aux = layer(
                    x,
                    top_k=top_k,
                    memory_queries=memory_queries,
                    query_lineage_keys=query_lineage_keys,
                    query_anchor_keysets=query_anchor_keysets,
                    query_entity_tokens=query_entity_tokens,
                )
            else:
                x, block_aux = layer(
                    x,
                    top_k=0,
                    memory_queries=None,
                    query_lineage_keys=None,
                    query_anchor_keysets=None,
                    query_entity_tokens=None,
                )
            block_summaries.append(block_aux)

        return (
            self.final_norm(x),
            block_summaries,
            memory_queries,
            query_lineage_keys,
            query_anchor_keysets,
            query_entity_tokens,
        )

    def _sequence_chunk_logit_bias(
        self,
        input_ids,
        hidden_states,
        block_summaries,
        top_k,
        memory_queries=None,
        query_lineage_keys=None,
        query_anchor_keysets=None,
        query_entity_tokens=None,
    ):
        batch_size, seq_len = input_ids.shape
        vocab_size = self.decoder.out_features
        bias = torch.zeros(
            batch_size,
            seq_len,
            vocab_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if top_k <= 0:
            return bias

        avg_selectivity = 0.0
        if block_summaries:
            avg_selectivity = sum(block["avg_selectivity"] for block in block_summaries) / len(block_summaries)
        sufficiency_gate = max(0.5, float(avg_selectivity))

        summaries = hidden_states.mean(dim=1)
        if memory_queries is not None:
            summaries = memory_queries.mean(dim=1)
        summary_routes = self.router(summaries.unsqueeze(1)).squeeze(1)

        for batch_index in range(batch_size):
            query_lineage_key = None
            if query_lineage_keys is not None:
                query_lineage_key = query_lineage_keys[batch_index]
            query_anchor_keyset = set()
            if query_anchor_keysets is not None:
                query_anchor_keyset = query_anchor_keysets[batch_index]
            query_entity_token = None
            if query_entity_tokens is not None:
                query_entity_token = str(int(query_entity_tokens[batch_index]))
            scored_records = self.memory_store.top_records(
                shard_id=int(summary_routes[batch_index].item()),
                query=summaries[batch_index],
                top_k=max(1, min(int(top_k), 8)),
            )
            if not scored_records:
                scored_records = self.memory_store.search(
                    summaries[batch_index],
                    top_k=max(1, min(int(top_k), 8)),
                )
                if not scored_records:
                    continue

            reranked_candidates = []
            for record, score in scored_records:
                if record.content_type != "sequence_chunk" or not isinstance(record.payload, list):
                    continue
                record_lineage_key = record.metadata.get("lineage_key")
                record_anchor_text = record.metadata.get("anchor_signatures", "")
                record_anchor_set = {anchor for anchor in record_anchor_text.split("|") if anchor}
                record_token_text = record.metadata.get("payload_token_set", "")
                record_token_set = {token for token in record_token_text.split("|") if token}
                exact_lineage = query_lineage_key is not None and record_lineage_key == query_lineage_key
                anchor_overlap = 0.0
                if query_anchor_keyset and record_anchor_set:
                    anchor_overlap = len(query_anchor_keyset & record_anchor_set) / max(1, len(query_anchor_keyset))
                entity_match = 1.0 if query_entity_token is not None and query_entity_token in record_token_set else 0.0
                if query_lineage_key is not None and record_lineage_key != query_lineage_key:
                    if not query_anchor_keyset or not record_anchor_set:
                        continue
                    if anchor_overlap <= 0.0:
                        continue
                    if query_entity_token is not None and query_entity_token not in record_token_set:
                        continue
                rerank_score = (
                    2.0 * float(score)
                    + 3.0 * float(exact_lineage)
                    + 1.5 * float(entity_match)
                    + 1.5 * float(anchor_overlap)
                )
                reranked_candidates.append((record, float(score), rerank_score))

            if not reranked_candidates:
                scored_records = self.memory_store.search(
                    summaries[batch_index],
                    top_k=max(4, min(int(top_k) * 2, 16)),
                )
                for record, score in scored_records:
                    if record.content_type != "sequence_chunk" or not isinstance(record.payload, list):
                        continue
                    record_lineage_key = record.metadata.get("lineage_key")
                    record_anchor_text = record.metadata.get("anchor_signatures", "")
                    record_anchor_set = {anchor for anchor in record_anchor_text.split("|") if anchor}
                    record_token_text = record.metadata.get("payload_token_set", "")
                    record_token_set = {token for token in record_token_text.split("|") if token}
                    exact_lineage = query_lineage_key is not None and record_lineage_key == query_lineage_key
                    anchor_overlap = 0.0
                    if query_anchor_keyset and record_anchor_set:
                        anchor_overlap = len(query_anchor_keyset & record_anchor_set) / max(1, len(query_anchor_keyset))
                    entity_match = 1.0 if query_entity_token is not None and query_entity_token in record_token_set else 0.0
                    if query_lineage_key is not None and record_lineage_key != query_lineage_key:
                        if not query_anchor_keyset or not record_anchor_set:
                            continue
                        if anchor_overlap <= 0.0:
                            continue
                        if query_entity_token is not None and query_entity_token not in record_token_set:
                            continue
                    rerank_score = (
                        2.0 * float(score)
                        + 3.0 * float(exact_lineage)
                        + 1.5 * float(entity_match)
                        + 1.5 * float(anchor_overlap)
                    )
                    reranked_candidates.append((record, float(score), rerank_score))

            reranked_candidates.sort(key=lambda item: item[2], reverse=True)

            for record, score, _ in reranked_candidates:

                payload = record.payload
                payload_len = len(payload)
                max_prefix = min(seq_len, payload_len - 1)
                input_list = input_ids[batch_index].detach().cpu().tolist()

                for token_index in range(max_prefix):
                    query_prefix = input_list[: token_index + 1]
                    if query_prefix == payload[: token_index + 1]:
                        next_token = int(payload[token_index + 1])
                        bias[batch_index, token_index, next_token] += sufficiency_gate * (
                            1.5 + max(0.0, float(score))
                        )
                        continue

                    longest_matches = []
                    max_window = min(token_index + 1, payload_len - 1, 24)
                    for window_size in range(max_window, 0, -1):
                        query_window = input_list[token_index - window_size + 1 : token_index + 1]
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
                                bias[batch_index, token_index, next_token] += (
                                    sufficiency_gate * max(0.5, match_weight) * score_weight
                                )
                            break

        return bias

    def forward(self, input_ids, top_k=4, return_aux=False, use_memory=True, causal=False):
        (
            hidden_states,
            block_summaries,
            memory_queries,
            query_lineage_keys,
            query_anchor_keysets,
            query_entity_tokens,
        ) = self.encode_hidden(
            input_ids=input_ids,
            top_k=top_k,
            use_memory=use_memory,
            causal=causal,
        )
        logits = self.decoder(hidden_states)
        if use_memory and top_k > 0:
            payload_bias = self._sequence_chunk_logit_bias(
                input_ids=input_ids,
                hidden_states=hidden_states,
                block_summaries=block_summaries,
                top_k=top_k,
                memory_queries=memory_queries,
                query_lineage_keys=query_lineage_keys,
                query_anchor_keysets=query_anchor_keysets,
                query_entity_tokens=query_entity_tokens,
            )
            logits = logits + 20.0 * payload_bias

        if return_aux:
            avg_sufficiency = 0.0
            avg_conflict = 0.0
            if block_summaries:
                avg_sufficiency = sum(block["avg_sufficiency"] for block in block_summaries) / len(block_summaries)
                avg_conflict = sum(block["avg_conflict"] for block in block_summaries) / len(block_summaries)
            return logits, {
                "backbone_type": self.backbone_type,
                "ablation_config": {
                    "use_selectivity": self.use_selectivity,
                    "use_sparse_experts": self.use_sparse_experts,
                    "use_lineage_filtering": self.use_lineage_filtering,
                },
                "layer_schedule": list(self.layer_schedule),
                "ruma_blocks": block_summaries,
                "avg_ruma_sufficiency": round(float(avg_sufficiency), 4),
                "avg_ruma_conflict": round(float(avg_conflict), 4),
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
            supersede_prior_lineage=self.supersede_prior_lineage,
        )

    def memory_stats(self):
        return self.memory_store.stats()
