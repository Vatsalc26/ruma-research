import re

from real_doc_memory import TOKEN_RE, clean_text


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "now",
    "of",
    "on",
    "or",
    "say",
    "the",
    "to",
    "uses",
    "what",
    "when",
    "which",
    "with",
}


def tokenize_lower(text):
    return [
        token
        for token in TOKEN_RE.findall(clean_text(text).lower())
        if token not in STOPWORDS
    ]


def sentence_candidates(text):
    cleaned = clean_text(text)
    parts = SENTENCE_SPLIT_RE.split(cleaned)
    return [part.strip() for part in parts if len(part.strip().split()) >= 6]


class CitationFirstAnswerer:
    """
    A lightweight answer layer over the real document memory index.
    It does not pretend to be a full language model. Instead, it turns
    retrieved evidence into a grounded extractive answer with citations.
    """

    def __init__(self, index):
        self.index = index

    def _score_sentence(self, query_tokens, sentence, retrieval_score):
        sentence_tokens = tokenize_lower(sentence)
        if not sentence_tokens:
            return 0.0, 0

        overlap = len(set(query_tokens) & set(sentence_tokens))
        if overlap == 0:
            return 0.0, 0

        density = overlap / max(1, len(set(query_tokens)))
        return float(retrieval_score) + density, overlap

    def _detect_conflicts(self, query_tokens, hits, chosen):
        if not hits or not chosen:
            return []

        top_hit = hits[0]
        top_source = chosen[0]["source"]
        top_namespace = chosen[0]["namespace"]
        notices = []
        seen = set()
        chosen_keys = {(item["source"], item["chunk_index"]) for item in chosen}

        for hit in hits[1:]:
            if hit["namespace"] != top_namespace:
                continue
            if (hit["source"], hit["chunk_index"]) in chosen_keys:
                continue
            if hit["score"] < 0.2:
                continue
            other_tokens = set(tokenize_lower(hit["excerpt"]))
            overlap = len(set(query_tokens) & other_tokens)
            if overlap < 2:
                continue
            overlap_density = overlap / max(1, len(set(query_tokens)))
            if overlap_density < 0.45:
                continue
            if hit["source"] == top_source:
                continue
            if top_hit["score"] - hit["score"] > 0.45:
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
                    "score": hit["score"],
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
                        "score": item["score"],
                    }
                    for item in chosen
                ],
            }
        ]

    def answer(self, query, top_k=4, namespaces=None, max_sentences=2):
        hits = self.index.search(query, top_k=top_k, namespaces=namespaces)
        query_tokens = tokenize_lower(query)

        sentence_pool = []
        for hit in hits:
            for sentence in sentence_candidates(hit["excerpt"]):
                sentence_tokens = tokenize_lower(sentence)
                score, overlap = self._score_sentence(query_tokens, sentence, hit["score"])
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
        top_score = sentence_pool[0]["score"] if sentence_pool else 0.0
        query_token_set = set(query_tokens)
        covered_tokens = set()

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
                if item["overlap"] < 2 and new_overlap == 0:
                    continue

                candidate_key = (
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
                "answer": "I could not find grounded evidence for that query in the current memory index.",
                "citations": [],
                "conflicts": [],
                "retrieval_hits": hits,
            }

        answer_text = " ".join(item["sentence"] for item in chosen)
        citations = [
            {
                "source": item["source"],
                "chunk_index": item["chunk_index"],
                "namespace": item["namespace"],
                "score": item["score"],
            }
            for item in chosen
        ]
        conflicts = self._chosen_conflict_summary(chosen) + self._detect_conflicts(query_tokens, hits, chosen)

        return {
            "query": query,
            "answer": answer_text,
            "citations": citations,
            "conflicts": conflicts,
            "retrieval_hits": hits,
        }
