import math
from collections import Counter

from real_doc_answerer import sentence_candidates, tokenize_lower


class ExtractiveRagAnswerer:
    """
    Lightweight retrieval baseline answerer.
    It uses the same index as RUMA but does not rely on lineage supersession.
    """

    def __init__(self, index, source_priority=None, detect_multi_source=False):
        self.index = index
        self.source_priority = {
            key.replace("\\", "/"): int(value)
            for key, value in (source_priority or {}).items()
        }
        self.detect_multi_source = detect_multi_source

    def _priority(self, source):
        normalized = source.replace("\\", "/")
        return self.source_priority.get(normalized, 0)

    def _ordered_hits(self, query, top_k, namespaces):
        expanded_top_k = max(top_k, top_k * 2)
        hits = self.index.search(query, top_k=expanded_top_k, namespaces=namespaces)
        if self.source_priority:
            hits.sort(
                key=lambda hit: (
                    self._priority(hit["source"]),
                    float(hit["score"]),
                ),
                reverse=True,
            )
        return hits[:top_k]

    def _score_sentence(self, query_tokens, sentence, retrieval_score):
        sentence_tokens = tokenize_lower(sentence)
        if not sentence_tokens:
            return 0.0, 0, sentence_tokens

        overlap = len(set(query_tokens) & set(sentence_tokens))
        if overlap == 0:
            return 0.0, 0, sentence_tokens

        density = overlap / max(1, len(set(query_tokens)))
        return float(retrieval_score) + density, overlap, sentence_tokens

    def answer(self, query, top_k=4, namespaces=None, max_sentences=2):
        hits = self._ordered_hits(query=query, top_k=top_k, namespaces=namespaces)
        query_tokens = tokenize_lower(query)
        query_token_set = set(query_tokens)

        sentence_pool = []
        for hit in hits:
            for sentence in sentence_candidates(hit["excerpt"]):
                score, overlap, sentence_tokens = self._score_sentence(
                    query_tokens,
                    sentence,
                    hit["score"],
                )
                if score <= 0:
                    continue
                sentence_pool.append(
                    {
                        "score": score,
                        "overlap": overlap,
                        "sentence": sentence,
                        "sentence_tokens": sentence_tokens,
                        "source": hit["source"],
                        "chunk_index": hit["chunk_index"],
                        "namespace": hit["namespace"],
                    }
                )

        sentence_pool.sort(key=lambda item: item["score"], reverse=True)

        chosen = []
        seen_sentences = set()
        covered_tokens = set()
        top_score = sentence_pool[0]["score"] if sentence_pool else 0.0

        while len(chosen) < max_sentences:
            best_item = None
            best_key = None
            for item in sentence_pool:
                normalized = item["sentence"].lower()
                if normalized in seen_sentences:
                    continue
                if chosen and item["score"] < max(0.6, top_score * 0.6):
                    continue

                overlap_tokens = query_token_set & set(item["sentence_tokens"])
                new_overlap = len(overlap_tokens - covered_tokens)
                if chosen and new_overlap == 0:
                    continue

                candidate_key = (
                    self._priority(item["source"]),
                    new_overlap if chosen else len(overlap_tokens),
                    item["score"],
                    item["overlap"],
                )
                if best_key is None or candidate_key > best_key:
                    best_key = candidate_key
                    best_item = item

            if best_item is None:
                break

            seen_sentences.add(best_item["sentence"].lower())
            covered_tokens.update(query_token_set & set(best_item["sentence_tokens"]))
            chosen.append(best_item)

        if not chosen:
            return {
                "query": query,
                "answer": "I could not find grounded evidence for that query in the current retrieval baseline.",
                "citations": [],
                "conflicts": [],
                "retrieval_hits": hits,
            }

        citations = [
            {
                "source": item["source"],
                "chunk_index": item["chunk_index"],
                "namespace": item["namespace"],
                "score": item["score"],
            }
            for item in chosen
        ]
        conflicts = []
        if self.detect_multi_source:
            distinct_sources = {(item["source"], item["chunk_index"]) for item in chosen}
            if len(distinct_sources) > 1:
                conflicts.append(
                    {
                        "type": "multi_source_answer",
                        "message": "The retrieval baseline answer was assembled from multiple active sources.",
                        "sources": citations,
                    }
                )

        return {
            "query": query,
            "answer": " ".join(item["sentence"] for item in chosen),
            "citations": citations,
            "conflicts": conflicts,
            "retrieval_hits": hits,
        }


class HybridExtractiveRagAnswerer(ExtractiveRagAnswerer):
    """
    Retrieval baseline that fuses the existing dense search with a lexical BM25-style pass.
    This is infrastructure borrowing, not a change to the core RUMA mechanism.
    """

    def __init__(
        self,
        index,
        source_priority=None,
        detect_multi_source=False,
        dense_weight=1.0,
        lexical_weight=1.0,
        rrf_k=60,
        bm25_k1=1.5,
        bm25_b=0.75,
    ):
        super().__init__(index, source_priority=source_priority, detect_multi_source=detect_multi_source)
        self.dense_weight = float(dense_weight)
        self.lexical_weight = float(lexical_weight)
        self.rrf_k = max(1, int(rrf_k))
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self._lexical_cache = None

    def _ensure_lexical_cache(self):
        active_records = self.index.store.active_records()
        cache_key = tuple(sorted(record.record_id for record in active_records))
        if self._lexical_cache is not None and self._lexical_cache["cache_key"] == cache_key:
            return

        doc_freq = Counter()
        record_rows = []
        total_length = 0
        for record in active_records:
            tokens = tokenize_lower(str(record.payload))
            counts = Counter(tokens)
            total_length += len(tokens)
            for token in counts.keys():
                doc_freq[token] += 1
            record_rows.append(
                {
                    "record_id": record.record_id,
                    "namespace": record.namespace,
                    "source": record.source,
                    "chunk_index": int(record.metadata.get("chunk_index", "0")),
                    "excerpt": record.payload,
                    "timestamp": record.timestamp,
                    "status": record.status,
                    "lineage": record.metadata.get("lineage", ""),
                    "tf": counts,
                    "length": len(tokens),
                }
            )

        avg_doc_len = total_length / max(1, len(record_rows))
        self._lexical_cache = {
            "cache_key": cache_key,
            "records": record_rows,
            "doc_freq": doc_freq,
            "doc_count": len(record_rows),
            "avg_doc_len": avg_doc_len,
        }

    def _bm25_score(self, query_tokens, record_row):
        if not query_tokens:
            return 0.0

        tf = record_row["tf"]
        doc_len = max(1, int(record_row["length"]))
        avg_doc_len = max(1.0, float(self._lexical_cache["avg_doc_len"]))
        doc_count = max(1, int(self._lexical_cache["doc_count"]))
        score = 0.0

        for token in query_tokens:
            term_freq = tf.get(token, 0)
            if term_freq <= 0:
                continue
            doc_freq = self._lexical_cache["doc_freq"].get(token, 0)
            idf = math.log(1.0 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
            numerator = term_freq * (self.bm25_k1 + 1.0)
            denominator = term_freq + self.bm25_k1 * (1.0 - self.bm25_b + self.bm25_b * (doc_len / avg_doc_len))
            score += idf * (numerator / max(1e-12, denominator))
        return float(score)

    def _lexical_hits(self, query, top_k, namespaces):
        self._ensure_lexical_cache()
        query_tokens = tokenize_lower(query)
        namespace_filter = set(namespaces) if namespaces is not None else None

        scored = []
        for row in self._lexical_cache["records"]:
            if namespace_filter is not None and row["namespace"] not in namespace_filter:
                continue
            score = self._bm25_score(query_tokens, row)
            if score <= 0.0:
                continue
            scored.append(
                {
                    "score": score,
                    "namespace": row["namespace"],
                    "source": row["source"],
                    "record_id": row["record_id"],
                    "chunk_index": row["chunk_index"],
                    "excerpt": row["excerpt"],
                    "timestamp": row["timestamp"],
                    "status": row["status"],
                    "lineage": row["lineage"],
                }
            )

        scored.sort(key=lambda hit: hit["score"], reverse=True)
        return scored[:top_k]

    def _ordered_hits(self, query, top_k, namespaces):
        expanded_top_k = max(top_k * 4, top_k + 16)
        dense_hits = self.index.search(query, top_k=expanded_top_k, namespaces=namespaces)
        lexical_hits = self._lexical_hits(query, top_k=expanded_top_k, namespaces=namespaces)

        fused = {}
        for rank, hit in enumerate(dense_hits, start=1):
            row = fused.setdefault(hit["record_id"], dict(hit))
            row["dense_rank"] = rank
            row["dense_score"] = hit["score"]
            row["hybrid_score"] = row.get("hybrid_score", 0.0) + self.dense_weight / (self.rrf_k + rank)

        for rank, hit in enumerate(lexical_hits, start=1):
            row = fused.setdefault(hit["record_id"], dict(hit))
            row["lexical_rank"] = rank
            row["lexical_score"] = hit["score"]
            row["hybrid_score"] = row.get("hybrid_score", 0.0) + self.lexical_weight / (self.rrf_k + rank)

        ordered = list(fused.values())
        ordered.sort(
            key=lambda hit: (
                hit.get("hybrid_score", 0.0),
                self._priority(hit["source"]),
                hit.get("dense_score", 0.0),
                hit.get("lexical_score", 0.0),
            ),
            reverse=True,
        )
        return ordered[:top_k]


class RerankedHybridRagAnswerer(HybridExtractiveRagAnswerer):
    """
    Hybrid retrieval plus a lightweight overlap/title reranking pass.
    This is still a baseline borrowing move, not a core RUMA mechanism change.
    """

    def __init__(
        self,
        index,
        source_priority=None,
        detect_multi_source=False,
        dense_weight=1.0,
        lexical_weight=1.0,
        rrf_k=60,
        bm25_k1=1.5,
        bm25_b=0.75,
        overlap_weight=0.7,
        title_weight=0.35,
    ):
        super().__init__(
            index=index,
            source_priority=source_priority,
            detect_multi_source=detect_multi_source,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
            rrf_k=rrf_k,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )
        self.overlap_weight = float(overlap_weight)
        self.title_weight = float(title_weight)

    def _source_title_tokens(self, source):
        base = str(source or "").replace("\\", "/").split("/")[-1]
        base = base.replace("_", " ")
        return set(tokenize_lower(base))

    def _ordered_hits(self, query, top_k, namespaces):
        candidate_hits = super()._ordered_hits(
            query=query,
            top_k=max(top_k * 3, top_k + 8),
            namespaces=namespaces,
        )
        query_tokens = set(tokenize_lower(query))
        if not query_tokens:
            return candidate_hits[:top_k]

        for hit in candidate_hits:
            excerpt_tokens = set(tokenize_lower(hit.get("excerpt", "")))
            title_tokens = self._source_title_tokens(hit.get("source", ""))
            overlap = len(query_tokens & excerpt_tokens) / max(1, len(query_tokens))
            title_overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
            hit["rerank_score"] = (
                float(hit.get("hybrid_score", 0.0))
                + self.overlap_weight * overlap
                + self.title_weight * title_overlap
            )

        candidate_hits.sort(
            key=lambda hit: (
                hit.get("rerank_score", 0.0),
                self._priority(hit["source"]),
                hit.get("hybrid_score", 0.0),
                hit.get("dense_score", 0.0),
                hit.get("lexical_score", 0.0),
            ),
            reverse=True,
        )
        return candidate_hits[:top_k]


class AbstentionAwareHybridRagAnswerer(HybridExtractiveRagAnswerer):
    """
    Hybrid retrieval plus a lightweight answer-control rule.
    The goal is to keep hybrid retrieval gains while refusing unsupported answers
    when dense/lexical agreement is too weak.
    """

    def __init__(
        self,
        index,
        source_priority=None,
        detect_multi_source=False,
        dense_weight=1.0,
        lexical_weight=1.0,
        rrf_k=60,
        bm25_k1=1.5,
        bm25_b=0.75,
        min_query_coverage=0.3,
        require_dual_signal=True,
        dual_signal_top_k=3,
        min_top_source_count=1,
        top_source_window=4,
    ):
        super().__init__(
            index=index,
            source_priority=source_priority,
            detect_multi_source=detect_multi_source,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
            rrf_k=rrf_k,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )
        self.min_query_coverage = float(min_query_coverage)
        self.require_dual_signal = bool(require_dual_signal)
        self.dual_signal_top_k = max(1, int(dual_signal_top_k))
        self.min_top_source_count = max(1, int(min_top_source_count))
        self.top_source_window = max(1, int(top_source_window))

    def _query_coverage(self, query, answer):
        query_tokens = set(tokenize_lower(query))
        if not query_tokens:
            return 0.0
        answer_tokens = set(tokenize_lower(answer))
        return len(query_tokens & answer_tokens) / max(1, len(query_tokens))

    def _has_dual_signal(self, retrieval_hits):
        for hit in retrieval_hits[: self.dual_signal_top_k]:
            if hit.get("dense_rank") is not None and hit.get("lexical_rank") is not None:
                return True
        return False

    def _top_source_count(self, retrieval_hits):
        source_counts = Counter()
        for hit in retrieval_hits[: self.top_source_window]:
            source_counts[str(hit.get("source", ""))] += 1
        if not source_counts:
            return 0
        return max(source_counts.values())

    def answer(self, query, top_k=4, namespaces=None, max_sentences=2):
        packet = super().answer(query, top_k=top_k, namespaces=namespaces, max_sentences=max_sentences)
        if not packet["citations"]:
            return packet

        dual_signal = self._has_dual_signal(packet["retrieval_hits"])
        coverage = self._query_coverage(query, packet["answer"])
        top_source_count = self._top_source_count(packet["retrieval_hits"])
        distinct_sources = {
            (citation["source"], citation["chunk_index"])
            for citation in packet["citations"]
        }

        should_abstain = False
        if self.require_dual_signal and not dual_signal:
            should_abstain = True
        if coverage < self.min_query_coverage:
            should_abstain = True
        if top_source_count < self.min_top_source_count and coverage < max(0.45, self.min_query_coverage):
            should_abstain = True
        if len(distinct_sources) > 1 and not dual_signal and coverage < max(0.45, self.min_query_coverage):
            should_abstain = True

        if not should_abstain:
            packet["guardrail"] = {
                "dual_signal": dual_signal,
                "query_coverage": coverage,
                "top_source_count": top_source_count,
                "abstained": False,
            }
            return packet

        packet["answer"] = "I could not find grounded evidence for that query in the current retrieval baseline."
        packet["citations"] = []
        packet["conflicts"] = []
        packet["guardrail"] = {
            "dual_signal": dual_signal,
            "query_coverage": coverage,
            "top_source_count": top_source_count,
            "abstained": True,
        }
        return packet


class AbstentionAwareRerankedHybridRagAnswerer(RerankedHybridRagAnswerer):
    """
    Reranked hybrid retrieval plus the same lightweight answer-control rule.
    This tests whether stronger reranking and abstention control can be combined
    before any deeper architectural change is justified.
    """

    def __init__(
        self,
        index,
        source_priority=None,
        detect_multi_source=False,
        dense_weight=1.0,
        lexical_weight=1.0,
        rrf_k=60,
        bm25_k1=1.5,
        bm25_b=0.75,
        overlap_weight=0.7,
        title_weight=0.35,
        min_query_coverage=0.3,
        require_dual_signal=True,
        dual_signal_top_k=3,
        min_top_source_count=1,
        top_source_window=4,
    ):
        super().__init__(
            index=index,
            source_priority=source_priority,
            detect_multi_source=detect_multi_source,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
            rrf_k=rrf_k,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            overlap_weight=overlap_weight,
            title_weight=title_weight,
        )
        self.min_query_coverage = float(min_query_coverage)
        self.require_dual_signal = bool(require_dual_signal)
        self.dual_signal_top_k = max(1, int(dual_signal_top_k))
        self.min_top_source_count = max(1, int(min_top_source_count))
        self.top_source_window = max(1, int(top_source_window))

    def _query_coverage(self, query, answer):
        query_tokens = set(tokenize_lower(query))
        if not query_tokens:
            return 0.0
        answer_tokens = set(tokenize_lower(answer))
        return len(query_tokens & answer_tokens) / max(1, len(query_tokens))

    def _has_dual_signal(self, retrieval_hits):
        for hit in retrieval_hits[: self.dual_signal_top_k]:
            if hit.get("dense_rank") is not None and hit.get("lexical_rank") is not None:
                return True
        return False

    def _top_source_count(self, retrieval_hits):
        source_counts = Counter()
        for hit in retrieval_hits[: self.top_source_window]:
            source_counts[str(hit.get("source", ""))] += 1
        if not source_counts:
            return 0
        return max(source_counts.values())

    def answer(self, query, top_k=4, namespaces=None, max_sentences=2):
        packet = super().answer(query, top_k=top_k, namespaces=namespaces, max_sentences=max_sentences)
        if not packet["citations"]:
            return packet

        dual_signal = self._has_dual_signal(packet["retrieval_hits"])
        coverage = self._query_coverage(query, packet["answer"])
        top_source_count = self._top_source_count(packet["retrieval_hits"])
        distinct_sources = {
            (citation["source"], citation["chunk_index"])
            for citation in packet["citations"]
        }

        should_abstain = False
        if self.require_dual_signal and not dual_signal:
            should_abstain = True
        if coverage < self.min_query_coverage:
            should_abstain = True
        if top_source_count < self.min_top_source_count and coverage < max(0.45, self.min_query_coverage):
            should_abstain = True
        if len(distinct_sources) > 1 and not dual_signal and coverage < max(0.45, self.min_query_coverage):
            should_abstain = True

        if not should_abstain:
            packet["guardrail"] = {
                "dual_signal": dual_signal,
                "query_coverage": coverage,
                "top_source_count": top_source_count,
                "abstained": False,
            }
            return packet

        packet["answer"] = "I could not find grounded evidence for that query in the current retrieval baseline."
        packet["citations"] = []
        packet["conflicts"] = []
        packet["guardrail"] = {
            "dual_signal": dual_signal,
            "query_coverage": coverage,
            "top_source_count": top_source_count,
            "abstained": True,
        }
        return packet


class InterleavedControllerRagAnswerer(RerankedHybridRagAnswerer):
    """
    Interleaved-style answer controller over the stronger hybrid+rereanked retriever.

    It mirrors the current RUMA-vNext direction:
    - support-oriented evidence expert
    - bridge/composition expert
    - explicit abstention/calibration control
    """

    def __init__(
        self,
        index,
        source_priority=None,
        detect_multi_source=False,
        dense_weight=1.0,
        lexical_weight=1.0,
        rrf_k=60,
        bm25_k1=1.5,
        bm25_b=0.75,
        overlap_weight=0.7,
        title_weight=0.35,
        min_query_coverage=0.4,
        min_controller_confidence=0.5,
        require_dual_signal=False,
        top_source_window=6,
    ):
        super().__init__(
            index=index,
            source_priority=source_priority,
            detect_multi_source=detect_multi_source,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
            rrf_k=rrf_k,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            overlap_weight=overlap_weight,
            title_weight=title_weight,
        )
        self.min_query_coverage = float(min_query_coverage)
        self.min_controller_confidence = float(min_controller_confidence)
        self.require_dual_signal = bool(require_dual_signal)
        self.top_source_window = max(1, int(top_source_window))

    def _chosen_conflict_summary(self, chosen):
        distinct_sources = {(item["source"], item["chunk_index"]) for item in chosen}
        if len(distinct_sources) <= 1:
            return []

        return [
            {
                "type": "multi_source_answer",
                "message": (
                    "The grounded answer uses multiple active sources. "
                    "Treat this as active disagreement or context-sensitive guidance."
                ),
                "sources": [
                    {
                        "source": item["source"],
                        "chunk_index": item["chunk_index"],
                        "namespace": item["namespace"],
                        "score": item["support_score"],
                    }
                    for item in chosen
                ],
            }
        ]

    def _detect_conflicts(self, query_tokens, hits, chosen):
        if not hits or not chosen:
            return []

        notices = []
        seen = set()
        chosen_keys = {(item["source"], item["chunk_index"]) for item in chosen}
        top_namespace = chosen[0]["namespace"]

        for hit in hits[1:]:
            if hit["namespace"] != top_namespace:
                continue
            if (hit["source"], hit["chunk_index"]) in chosen_keys:
                continue
            if hit.get("rerank_score", 0.0) < 0.2:
                continue
            other_tokens = set(tokenize_lower(hit["excerpt"]))
            overlap = len(set(query_tokens) & other_tokens)
            if overlap < 2:
                continue

            key = (hit["source"], hit["chunk_index"])
            if key in seen:
                continue
            seen.add(key)
            notices.append(
                {
                    "type": "conflict_candidate",
                    "source": hit["source"],
                    "chunk_index": hit["chunk_index"],
                    "namespace": hit["namespace"],
                    "score": hit.get("rerank_score", hit.get("hybrid_score", hit.get("score", 0.0))),
                    "lineage": hit.get("lineage", ""),
                    "message": (
                        "Another active source with overlapping evidence was retrieved. "
                        "This answer may need manual comparison across citations."
                    ),
                }
            )
            if len(notices) >= 2:
                break

        return notices

    def _query_coverage(self, query_tokens, answer_tokens):
        query_token_set = set(query_tokens)
        if not query_token_set:
            return 0.0
        return len(query_token_set & set(answer_tokens)) / max(1, len(query_token_set))

    def _controller_confidence(self, coverage, avg_support, avg_bridge, dual_signal_ratio, top_source_consensus):
        support_norm = min(1.0, avg_support / 3.0)
        bridge_norm = min(1.0, avg_bridge / 3.0)
        return (
            0.35 * coverage
            + 0.25 * support_norm
            + 0.15 * bridge_norm
            + 0.15 * dual_signal_ratio
            + 0.10 * top_source_consensus
        )

    def answer(self, query, top_k=4, namespaces=None, max_sentences=2):
        hits = self._ordered_hits(
            query=query,
            top_k=max(top_k * 3, top_k + 8),
            namespaces=namespaces,
        )
        query_tokens = tokenize_lower(query)
        query_token_set = set(query_tokens)
        source_counter = Counter(
            str(hit.get("source", ""))
            for hit in hits[: self.top_source_window]
        )
        top_source_consensus = (
            max(source_counter.values()) / max(1, sum(source_counter.values()))
            if source_counter
            else 0.0
        )

        sentence_pool = []
        for hit in hits:
            retrieval_score = float(hit.get("rerank_score", hit.get("hybrid_score", hit.get("score", 0.0))))
            source_support = source_counter.get(str(hit.get("source", "")), 0) / max(1, self.top_source_window)
            dual_signal = 1.0 if hit.get("dense_rank") is not None and hit.get("lexical_rank") is not None else 0.0
            title_overlap = len(query_token_set & self._source_title_tokens(hit.get("source", ""))) / max(1, len(query_token_set))

            for sentence in sentence_candidates(hit["excerpt"]):
                score, overlap, sentence_tokens = self._score_sentence(
                    query_tokens,
                    sentence,
                    retrieval_score,
                )
                if score <= 0.0:
                    continue
                overlap_tokens = query_token_set & set(sentence_tokens)
                overlap_density = len(overlap_tokens) / max(1, len(query_token_set))
                support_score = (
                    score
                    + 0.45 * source_support
                    + 0.25 * dual_signal
                    + 0.20 * title_overlap
                    + 0.25 * overlap_density
                )
                bridge_score = (
                    score
                    + 0.55 * overlap_density
                    + 0.30 * dual_signal
                    + 0.15 * (1.0 - source_support)
                )
                sentence_pool.append(
                    {
                        "score": score,
                        "support_score": support_score,
                        "bridge_score": bridge_score,
                        "dual_signal": dual_signal,
                        "source_support": source_support,
                        "title_overlap": title_overlap,
                        "sentence": sentence,
                        "sentence_tokens": sentence_tokens,
                        "source": hit["source"],
                        "chunk_index": hit["chunk_index"],
                        "namespace": hit["namespace"],
                    }
                )

        sentence_pool.sort(key=lambda item: (item["support_score"], item["bridge_score"]), reverse=True)

        chosen = []
        seen_sentences = set()
        covered_tokens = set()

        while len(chosen) < max_sentences:
            best_item = None
            best_key = None
            for item in sentence_pool:
                normalized = item["sentence"].lower()
                if normalized in seen_sentences:
                    continue

                overlap_tokens = query_token_set & set(item["sentence_tokens"])
                new_overlap = len(overlap_tokens - covered_tokens)
                if chosen and new_overlap == 0:
                    continue

                candidate_key = (
                    new_overlap if chosen else len(overlap_tokens),
                    item["bridge_score"] if chosen else item["support_score"],
                    item["support_score"],
                    item["source_support"],
                )
                if best_key is None or candidate_key > best_key:
                    best_key = candidate_key
                    best_item = item

            if best_item is None:
                break

            seen_sentences.add(best_item["sentence"].lower())
            covered_tokens.update(query_token_set & set(best_item["sentence_tokens"]))
            chosen.append(best_item)

        if not chosen:
            return {
                "query": query,
                "answer": "I could not find grounded evidence for that query in the current retrieval baseline.",
                "citations": [],
                "conflicts": [],
                "retrieval_hits": hits,
                "controller": {
                    "abstained": True,
                    "query_coverage": 0.0,
                    "controller_confidence": 0.0,
                    "active_experts": ["support", "bridge", "guardrail"],
                },
            }

        answer_tokens = []
        for item in chosen:
            answer_tokens.extend(item["sentence_tokens"])
        coverage = self._query_coverage(query_tokens, answer_tokens)
        dual_signal_ratio = sum(item["dual_signal"] for item in chosen) / max(1, len(chosen))
        avg_support = sum(item["support_score"] for item in chosen) / max(1, len(chosen))
        avg_bridge = sum(item["bridge_score"] for item in chosen) / max(1, len(chosen))
        controller_confidence = self._controller_confidence(
            coverage=coverage,
            avg_support=avg_support,
            avg_bridge=avg_bridge,
            dual_signal_ratio=dual_signal_ratio,
            top_source_consensus=top_source_consensus,
        )

        should_abstain = False
        if coverage < self.min_query_coverage:
            should_abstain = True
        if controller_confidence < self.min_controller_confidence:
            should_abstain = True
        if self.require_dual_signal and dual_signal_ratio <= 0.0:
            should_abstain = True

        citations = [
            {
                "source": item["source"],
                "chunk_index": item["chunk_index"],
                "namespace": item["namespace"],
                "score": item["support_score"],
            }
            for item in chosen
        ]
        conflicts = self._chosen_conflict_summary(chosen) + self._detect_conflicts(query_tokens, hits, chosen)
        controller = {
            "abstained": should_abstain,
            "query_coverage": round(float(coverage), 4),
            "controller_confidence": round(float(controller_confidence), 4),
            "avg_support_score": round(float(avg_support), 4),
            "avg_bridge_score": round(float(avg_bridge), 4),
            "dual_signal_ratio": round(float(dual_signal_ratio), 4),
            "top_source_consensus": round(float(top_source_consensus), 4),
            "active_experts": ["support", "bridge", "guardrail"],
        }

        if should_abstain:
            return {
                "query": query,
                "answer": "I could not find grounded evidence for that query in the current retrieval baseline.",
                "citations": [],
                "conflicts": [],
                "retrieval_hits": hits,
                "controller": controller,
            }

        return {
            "query": query,
            "answer": " ".join(item["sentence"] for item in chosen),
            "citations": citations,
            "conflicts": conflicts,
            "retrieval_hits": hits,
            "controller": controller,
        }
