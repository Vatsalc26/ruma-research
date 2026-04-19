from collections import Counter
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion import GatedMemoryFusion


class TransformerBackboneBlock(nn.Module):
    """
    First serious local sequence-modeling block for the reverse-designed RUMA path.
    It keeps the standard pre-norm attention + MLP structure as the stable reference
    backbone before we compare backbone variants like Mamba or hybrids.
    """

    def __init__(self, d_model, n_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_size = max(d_model, int(d_model * mlp_ratio))
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, d_model),
        )

    def _build_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, x, causal=False):
        norm_x = self.norm1(x)
        attn_mask = self._build_causal_mask(x.size(1), x.device) if causal else None
        attn_out, _ = self.attention(norm_x, norm_x, norm_x, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class MambaBackboneBlock(nn.Module):
    """
    Lightweight Mamba-inspired state-space mixer for local RUMA comparison.

    This is not a CUDA-kernel reproduction of official Mamba code. It is a
    bounded selective state-space block that keeps the key local design idea:
    recurrent state updates with learned input-dependent gates and decay.
    """

    def __init__(self, d_model, d_state=32, mlp_ratio=2.0, kernel_size=3):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = max(2, int(kernel_size))
        hidden_size = max(d_model, int(d_model * mlp_ratio))

        self.norm1 = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.dw_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=self.kernel_size,
            groups=d_model,
            bias=True,
        )
        self.dt_proj = nn.Linear(d_model, d_model)
        self.b_proj = nn.Linear(d_model, d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.a_log = nn.Parameter(torch.zeros(d_model))
        self.d_skip = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, d_model),
        )

    def _causal_depthwise_conv(self, x):
        x_t = x.transpose(1, 2)
        x_t = F.pad(x_t, (self.kernel_size - 1, 0))
        x_t = self.dw_conv(x_t)
        return x_t.transpose(1, 2)

    def forward(self, x, causal=False):
        norm_x = self.norm1(x)
        x_proj, gate = self.in_proj(norm_x).chunk(2, dim=-1)
        mixed = self._causal_depthwise_conv(x_proj) if causal else self.dw_conv(
            F.pad(x_proj.transpose(1, 2), (self.kernel_size // 2, self.kernel_size // 2))
        ).transpose(1, 2)
        if mixed.size(1) != x_proj.size(1):
            mixed = mixed[:, : x_proj.size(1), :]

        dt = F.softplus(self.dt_proj(mixed))
        b = torch.tanh(self.b_proj(mixed))
        c = torch.tanh(self.c_proj(mixed))
        a = -torch.exp(self.a_log).view(1, -1)
        d_skip = self.d_skip.view(1, -1)

        state = torch.zeros(x.size(0), self.d_model, device=x.device, dtype=x.dtype)
        outputs = []
        for token_index in range(x.size(1)):
            dt_t = dt[:, token_index, :]
            mixed_t = mixed[:, token_index, :]
            decay = torch.exp(dt_t * a)
            state = decay * state + dt_t * b[:, token_index, :] * mixed_t
            y_t = c[:, token_index, :] * state + d_skip * mixed_t
            outputs.append(y_t * torch.sigmoid(gate[:, token_index, :]))

        y = torch.stack(outputs, dim=1)
        x = x + self.out_proj(y)
        x = x + self.ffn(self.norm2(x))
        return x


class RUMASelectivityHead(nn.Module):
    """
    Retention-aware control head for deciding when retrieved memory should
    meaningfully alter the live residual stream.

    The final-form direction is not a purely hand-authored gate or a purely learned
    gate in isolation. We keep a strong analytic prior so the architecture remains
    stable before calibration, then learn a residual correction on top of it.
    """

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, memory_packet, route_confidence):
        memory_norm = memory_packet["memory_stream"].norm(dim=-1)
        memory_presence = (memory_norm > 1e-6).to(memory_norm.dtype)
        lineage_match = memory_packet["lineage_match_scores"]
        alignment = memory_packet["alignment_scores"]
        stale_penalty = memory_packet["stale_penalty_scores"]
        feature_tensor = torch.stack(
            [
                memory_packet["sufficiency_scores"],
                memory_packet["conflict_scores"],
                lineage_match,
                memory_packet["top_source_counts"],
                torch.clamp(memory_packet["top_scores"], min=0.0),
                alignment,
                stale_penalty,
                route_confidence,
                memory_presence,
            ],
            dim=-1,
        )
        prior_logit = (
            2.0 * memory_packet["sufficiency_scores"]
            + 1.25 * lineage_match
            + 0.75 * alignment
            + 0.35 * memory_packet["top_source_counts"]
            + 0.25 * torch.clamp(memory_packet["top_scores"], min=0.0)
            - 1.5 * memory_packet["stale_penalty_scores"]
            - 1.0 * memory_packet["conflict_scores"]
            - 0.5 * (1.0 - route_confidence)
        )
        learned_residual = self.net(feature_tensor).squeeze(-1)
        selectivity_logit = prior_logit + 0.5 * learned_residual
        selectivity = torch.sigmoid(selectivity_logit) * memory_presence
        return selectivity


class RUMAFusionExpertBank(nn.Module):
    """
    Sparse fusion bank for the interleaved final-form RUMA path.

    The bank keeps a small stable set of experts that bias toward conservative,
    base, bridge, or memory-dominant behavior. A sparse top-k router chooses which
    experts are active per token instead of forcing one monolithic fusion rule.
    """

    def __init__(self, d_model):
        super().__init__()
        self.expert_names = [
            "conservative",
            "base",
            "bridge",
            "memory_dominant",
        ]
        self.experts = nn.ModuleDict(
            {
                name: GatedMemoryFusion(d_model=d_model)
                for name in self.expert_names
            }
        )
        self.default_expert = "base"
        self.top_experts = 2

    def _expert_stack(self, context_state, memory_state):
        base_fused = self.experts["base"](context_state, memory_state)
        conservative_core = self.experts["conservative"](context_state, memory_state)
        bridge_core = self.experts["bridge"](context_state, memory_state)
        dominant_core = self.experts["memory_dominant"](context_state, memory_state)

        conservative = context_state + 0.4 * (conservative_core - context_state)
        bridge = context_state + 0.85 * (bridge_core - context_state) + 0.15 * memory_state
        memory_dominant = context_state + 1.15 * (dominant_core - context_state)
        return torch.stack(
            [
                conservative,
                base_fused,
                bridge,
                memory_dominant,
            ],
            dim=-2,
        )

    def forward(self, context_state, memory_state, expert_weights=None):
        expert_stack = self._expert_stack(context_state, memory_state)
        if expert_weights is None:
            default_index = self.expert_names.index(self.default_expert)
            expert_weights = torch.zeros(
                (*context_state.shape[:-1], len(self.expert_names)),
                device=context_state.device,
                dtype=context_state.dtype,
            )
            expert_weights[..., default_index] = 1.0
        else:
            k = min(self.top_experts, expert_weights.size(-1))
            top_values, top_indices = torch.topk(expert_weights, k=k, dim=-1)
            sparse_weights = torch.zeros_like(expert_weights)
            sparse_weights.scatter_(-1, top_indices, top_values)
            expert_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        fused = torch.sum(expert_stack * expert_weights.unsqueeze(-1), dim=-2)
        mean_weights = expert_weights.mean(dim=(0, 1))
        active_index = int(torch.argmax(mean_weights).item())
        active_expert = self.expert_names[active_index]
        expert_usage = {
            name: round(float(mean_weights[idx].item()), 4)
            for idx, name in enumerate(self.expert_names)
        }
        return fused, active_expert, expert_usage


class RUMABlock(nn.Module):
    """
    First explicit interleaved RUMA block.
    It makes the query projection, routing, retrieval, reranking, sufficiency scoring,
    conflict exposure, and residual merge visible as one architecture unit.
    """

    def __init__(
        self,
        d_model,
        router,
        memory_store,
        top_k=4,
        use_selectivity=True,
        use_sparse_experts=True,
        use_lineage_filtering=True,
    ):
        super().__init__()
        self.router = router
        self.memory_store = memory_store
        self.default_top_k = int(top_k)
        self.use_selectivity = bool(use_selectivity)
        self.use_sparse_experts = bool(use_sparse_experts)
        self.use_lineage_filtering = bool(use_lineage_filtering)
        self.pre_norm = nn.LayerNorm(d_model)
        self.query_projector = nn.Linear(d_model, d_model)
        self.selectivity_head = RUMASelectivityHead()
        self.fusion_bank = RUMAFusionExpertBank(d_model=d_model)
        self.expert_router = nn.Sequential(
            nn.Linear(8, 32),
            nn.SiLU(),
            nn.Linear(32, len(self.fusion_bank.expert_names)),
        )
        self.post_norm = nn.LayerNorm(d_model)

    def _route(self, queries):
        projections = torch.matmul(queries, self.router.hyperplanes)
        route_probs = torch.softmax(projections, dim=-1)
        routes = torch.argmax(route_probs, dim=-1)
        route_confidence = torch.max(route_probs, dim=-1).values
        return routes, route_confidence

    def _score_records(self, query_state, records_with_scores):
        if not records_with_scores:
            return []

        reranked = []
        query_norm = F.normalize(query_state.unsqueeze(0), dim=-1)
        for record, score in records_with_scores:
            record_key = F.normalize(record.key.to(query_state.device).unsqueeze(0), dim=-1)
            overlap_score = torch.matmul(query_norm, record_key.transpose(0, 1)).squeeze(0).squeeze(0)
            rerank_score = 0.6 * float(score) + 0.4 * float(overlap_score.detach().item())
            reranked.append((record, rerank_score))

        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked

    def _fetch_reranked_records(self, shard_id, query_state, top_k):
        scored_records = self.memory_store.top_records(
            shard_id=shard_id,
            query=query_state,
            top_k=top_k,
        )
        reranked_records = self._score_records(query_state, scored_records)
        if reranked_records:
            return reranked_records
        return self._score_records(
            query_state,
            self.memory_store.search(query_state, top_k=max(top_k, 8)),
        )

    def _build_memory_stream(
        self,
        queries,
        routes,
        route_confidence,
        top_k,
        query_lineage_keys=None,
        query_anchor_keysets=None,
        query_entity_tokens=None,
    ):
        batch_size, seq_len, d_model = queries.shape
        memory_stream = torch.zeros_like(queries)
        sufficiency_scores = torch.zeros(batch_size, seq_len, device=queries.device, dtype=queries.dtype)
        conflict_scores = torch.zeros(batch_size, seq_len, device=queries.device, dtype=queries.dtype)
        lineage_match_scores = torch.zeros(batch_size, seq_len, device=queries.device, dtype=queries.dtype)
        top_source_counts = torch.zeros(batch_size, seq_len, device=queries.device, dtype=queries.dtype)
        top_score_matrix = torch.zeros(batch_size, seq_len, device=queries.device, dtype=queries.dtype)
        alignment_scores = torch.zeros(batch_size, seq_len, device=queries.device, dtype=queries.dtype)
        stale_penalty_scores = torch.zeros(batch_size, seq_len, device=queries.device, dtype=queries.dtype)
        route_histogram = Counter()

        for batch_index in range(batch_size):
            for token_index in range(seq_len):
                shard_id = int(routes[batch_index, token_index].item())
                route_histogram[shard_id] += 1
                query_state = queries[batch_index, token_index]
                reranked_records = self._fetch_reranked_records(
                    shard_id=shard_id,
                    query_state=query_state,
                    top_k=top_k,
                )
                if not reranked_records:
                    continue

                query_lineage_key = None
                if query_lineage_keys is not None:
                    query_lineage_key = query_lineage_keys[batch_index]

                if not self.use_lineage_filtering:
                    query_lineage_key = None

                if query_lineage_key is not None:
                    query_anchor_keyset = set()
                    if query_anchor_keysets is not None:
                        query_anchor_keyset = query_anchor_keysets[batch_index]
                    query_entity_token = None
                    if query_entity_tokens is not None:
                        query_entity_token = str(int(query_entity_tokens[batch_index]))
                    matching_records = [
                        (record, score)
                        for record, score in reranked_records
                        if record.metadata.get("lineage_key") == query_lineage_key
                    ]
                    lineage_match = 1.0 if matching_records else 0.0
                    if not matching_records and query_anchor_keyset:
                        anchor_matching_records = []
                        best_anchor_overlap = 0.0
                        for record, score in reranked_records:
                            record_anchor_text = record.metadata.get("anchor_signatures", "")
                            record_anchor_set = {
                                anchor
                                for anchor in record_anchor_text.split("|")
                                if anchor
                            }
                            record_token_text = record.metadata.get("payload_token_set", "")
                            record_token_set = {
                                token
                                for token in record_token_text.split("|")
                                if token
                            }
                            if not record_anchor_set:
                                continue
                            overlap = len(query_anchor_keyset & record_anchor_set) / max(1, len(query_anchor_keyset))
                            if overlap <= 0.0:
                                continue
                            if query_entity_token is not None and query_entity_token not in record_token_set:
                                continue
                            anchor_matching_records.append((record, score))
                            best_anchor_overlap = max(best_anchor_overlap, float(overlap))
                        if anchor_matching_records:
                            matching_records = anchor_matching_records
                            lineage_match = min(0.85, max(0.35, best_anchor_overlap))
                        elif top_k > 0:
                            global_records = self._score_records(
                                query_state,
                                self.memory_store.search(query_state, top_k=max(top_k, 8)),
                            )
                            anchor_matching_records = []
                            best_anchor_overlap = 0.0
                            for record, score in global_records:
                                record_anchor_text = record.metadata.get("anchor_signatures", "")
                                record_anchor_set = {
                                    anchor
                                    for anchor in record_anchor_text.split("|")
                                    if anchor
                                }
                                record_token_text = record.metadata.get("payload_token_set", "")
                                record_token_set = {
                                    token
                                    for token in record_token_text.split("|")
                                    if token
                                }
                                if not record_anchor_set:
                                    continue
                                overlap = len(query_anchor_keyset & record_anchor_set) / max(1, len(query_anchor_keyset))
                                if overlap <= 0.0:
                                    continue
                                if query_entity_token is not None and query_entity_token not in record_token_set:
                                    continue
                                anchor_matching_records.append((record, score))
                                best_anchor_overlap = max(best_anchor_overlap, float(overlap))
                            if anchor_matching_records:
                                matching_records = anchor_matching_records
                                lineage_match = min(0.85, max(0.35, best_anchor_overlap))
                    if matching_records:
                        reranked_records = matching_records
                    else:
                        top_source_counts[batch_index, token_index] = 0.0
                        top_score_matrix[batch_index, token_index] = 0.0
                        sufficiency_scores[batch_index, token_index] = 0.0
                        conflict_scores[batch_index, token_index] = 0.0
                        lineage_match_scores[batch_index, token_index] = 0.0
                        alignment_scores[batch_index, token_index] = 0.0
                        stale_penalty_scores[batch_index, token_index] = 0.0
                        continue
                else:
                    lineage_match = 0.0

                scores = torch.tensor(
                    [score for _, score in reranked_records],
                    device=queries.device,
                    dtype=queries.dtype,
                )
                weights = torch.softmax(scores, dim=0)
                values = torch.stack(
                    [record.value.to(queries.device) for record, _ in reranked_records],
                    dim=0,
                )
                token_memory_state = torch.sum(values * weights.unsqueeze(-1), dim=0)
                memory_stream[batch_index, token_index] = token_memory_state

                top_score = float(scores[0].item())
                margin = float((scores[0] - scores[1]).item()) if scores.numel() > 1 else top_score
                sources = [record.source for record, _ in reranked_records]
                lineages = [
                    record.metadata.get("lineage", record.namespace)
                    for record, _ in reranked_records
                ]
                source_consensus = Counter(sources).most_common(1)[0][1] / max(1, len(sources))
                lineage_diversity = len(set(lineages)) / max(1, len(lineages))
                conflict = 1.0 if len(set(sources)) > 1 and len(set(lineages)) > 1 else 0.0

                lineage_penalty = 0.0
                if query_lineage_key is not None:
                    lineage_penalty = 2.5 * (1.0 - lineage_match)

                alignment = torch.cosine_similarity(
                    query_state.unsqueeze(0),
                    token_memory_state.unsqueeze(0),
                    dim=-1,
                ).squeeze(0)
                stale_penalty = (
                    0.75 * (1.0 - lineage_match)
                    + 0.5 * conflict
                    + 0.35 * (1.0 - source_consensus)
                    + 0.25 * max(0.0, 0.5 - float(alignment.item()))
                )

                sufficiency_logit = (
                    2.0 * top_score
                    + 1.0 * margin
                    + 0.75 * float(route_confidence[batch_index, token_index].item())
                    + 0.5 * source_consensus
                    + 1.25 * lineage_match
                    + 0.75 * float(alignment.item())
                    - 0.75 * lineage_diversity
                    - lineage_penalty
                    - 0.5 * conflict
                    - stale_penalty
                )
                sufficiency = torch.sigmoid(
                    torch.tensor(sufficiency_logit, device=queries.device, dtype=queries.dtype)
                )
                memory_stream[batch_index, token_index] = (
                    memory_stream[batch_index, token_index] * sufficiency
                )
                sufficiency_scores[batch_index, token_index] = sufficiency
                conflict_scores[batch_index, token_index] = float(conflict)
                lineage_match_scores[batch_index, token_index] = float(lineage_match)
                top_source_counts[batch_index, token_index] = float(source_consensus)
                top_score_matrix[batch_index, token_index] = float(top_score)
                alignment_scores[batch_index, token_index] = float(alignment.item())
                stale_penalty_scores[batch_index, token_index] = float(stale_penalty)

        return {
            "memory_stream": memory_stream,
            "sufficiency_scores": sufficiency_scores,
            "conflict_scores": conflict_scores,
            "lineage_match_scores": lineage_match_scores,
            "top_source_counts": top_source_counts,
            "top_scores": top_score_matrix,
            "alignment_scores": alignment_scores,
            "stale_penalty_scores": stale_penalty_scores,
            "route_histogram": dict(sorted(route_histogram.items())),
        }

    def _build_expert_weights(self, memory_packet, route_confidence, selectivity):
        stale_penalty = memory_packet["stale_penalty_scores"]
        lineage_match = memory_packet["lineage_match_scores"]
        sufficiency = memory_packet["sufficiency_scores"]
        conflict = memory_packet["conflict_scores"]
        top_source_consensus = memory_packet["top_source_counts"]
        alignment = memory_packet["alignment_scores"]

        prior_conservative_logit = (
            1.75 * stale_penalty
            + 1.0 * conflict
            + 0.75 * (1.0 - selectivity)
            + 0.35 * (1.0 - route_confidence)
        )
        prior_base_logit = (
            1.0 * selectivity
            + 0.5 * sufficiency
            + 0.25 * alignment
        )
        prior_bridge_logit = (
            1.1 * alignment
            + 0.8 * (1.0 - lineage_match)
            + 0.55 * (1.0 - top_source_consensus)
            + 0.45 * selectivity
            - 0.25 * conflict
        )
        prior_memory_dominant_logit = (
            1.85 * lineage_match
            + 1.2 * sufficiency
            + 0.6 * route_confidence
            + 0.45 * top_source_consensus
            - 1.4 * stale_penalty
            - 1.0 * conflict
        )

        prior_logits = torch.stack(
            [
                prior_conservative_logit,
                prior_base_logit,
                prior_bridge_logit,
                prior_memory_dominant_logit,
            ],
            dim=-1,
        )
        feature_tensor = torch.stack(
            [
                stale_penalty,
                lineage_match,
                sufficiency,
                conflict,
                top_source_consensus,
                alignment,
                route_confidence,
                selectivity,
            ],
            dim=-1,
        )
        learned_logits = self.expert_router(feature_tensor)
        logits = prior_logits + learned_logits
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        x,
        top_k=None,
        memory_queries=None,
        query_lineage_keys=None,
        query_anchor_keysets=None,
        query_entity_tokens=None,
    ):
        if top_k is None:
            top_k = self.default_top_k
        if top_k <= 0:
            return x, {
                "memory_used": False,
                "avg_sufficiency": 0.0,
                "avg_selectivity": 0.0,
                "avg_conflict": 0.0,
                "avg_lineage_match": 0.0,
                "avg_route_confidence": 0.0,
                "avg_alignment": 0.0,
                "avg_stale_penalty": 0.0,
                "active_fusion_expert": self.fusion_bank.default_expert,
                "fusion_expert_usage": {
                    name: 1.0 if name == self.fusion_bank.default_expert else 0.0
                    for name in self.fusion_bank.expert_names
                },
                "route_histogram": {},
            }

        norm_x = self.pre_norm(x)
        projected_queries = self.query_projector(norm_x)
        if memory_queries is not None:
            queries = 0.5 * projected_queries + 0.5 * memory_queries
        else:
            queries = projected_queries
        routes, route_confidence = self._route(queries)
        memory_packet = self._build_memory_stream(
            queries=queries,
            routes=routes,
            route_confidence=route_confidence,
            top_k=top_k,
            query_lineage_keys=query_lineage_keys,
            query_anchor_keysets=query_anchor_keysets,
            query_entity_tokens=query_entity_tokens,
        )
        if self.use_selectivity:
            selectivity = self.selectivity_head(memory_packet=memory_packet, route_confidence=route_confidence)
        else:
            selectivity = (memory_packet["memory_stream"].norm(dim=-1) > 1e-6).to(x.dtype)

        expert_weights = None
        if self.use_sparse_experts:
            expert_weights = self._build_expert_weights(
                memory_packet=memory_packet,
                route_confidence=route_confidence,
                selectivity=selectivity,
            )
        candidate_fused, active_fusion_expert, expert_usage = self.fusion_bank(
            x,
            memory_packet["memory_stream"],
            expert_weights=expert_weights,
        )
        candidate_fused = self.post_norm(candidate_fused)
        fused = x + selectivity.unsqueeze(-1) * (candidate_fused - x)

        aux = {
            "memory_used": True,
            "avg_sufficiency": round(float(memory_packet["sufficiency_scores"].mean().item()), 4),
            "avg_selectivity": round(float(selectivity.mean().item()), 4),
            "avg_conflict": round(float(memory_packet["conflict_scores"].mean().item()), 4),
            "avg_lineage_match": round(float(memory_packet["lineage_match_scores"].mean().item()), 4),
            "avg_route_confidence": round(float(route_confidence.mean().item()), 4),
            "avg_top_source_consensus": round(float(memory_packet["top_source_counts"].mean().item()), 4),
            "avg_top_score": round(float(memory_packet["top_scores"].mean().item()), 4),
            "avg_alignment": round(float(memory_packet["alignment_scores"].mean().item()), 4),
            "avg_stale_penalty": round(float(memory_packet["stale_penalty_scores"].mean().item()), 4),
            "active_fusion_expert": active_fusion_expert,
            "fusion_expert_usage": expert_usage,
            "ablation_config": {
                "use_selectivity": self.use_selectivity,
                "use_sparse_experts": self.use_sparse_experts,
                "use_lineage_filtering": self.use_lineage_filtering,
            },
            "route_histogram": memory_packet["route_histogram"],
        }
        return fused, aux
