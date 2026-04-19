import math
from collections import Counter

import torch
import torch.nn.functional as F

from memory_shards import MemoryRecord, MemoryShardStore
from real_doc_memory import TOKEN_RE, clean_text, stable_hash_int


class PromptEncoder:
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


class PromptEditMemoryIndex:
    """
    Minimal official-benchmark memory index for prompt->answer edit evaluation.
    This is intentionally simple and inspectable before tying the same datasets
    to stronger generative model baselines.
    """

    def __init__(self, d_model=256, num_shards=32, capacity_per_shard=16384):
        self.d_model = d_model
        self.num_shards = num_shards
        self.encoder = PromptEncoder(d_model=d_model)
        self.store = MemoryShardStore(
            num_shards=num_shards,
            d_model=d_model,
            capacity_per_shard=capacity_per_shard,
        )

    def fit(self, prompts):
        self.encoder.fit(prompts)

    def supersede_lineage(self, lineage):
        prior_records = [
            record.record_id
            for record in self.store.records(statuses={"active"})
            if record.metadata.get("lineage") == lineage
        ]
        if not prior_records:
            return 0
        return self.store.update_status(prior_records, "superseded")

    def _insert_prompt_answer(self, prompt, answer, lineage, namespace, timestamp, record_kind):
        key = self.encoder.encode(prompt)
        shard_id = stable_hash_int(f"{namespace}::{lineage}::{prompt}") % self.num_shards
        record_id = f"{namespace}::{lineage}::{timestamp}::{stable_hash_int(prompt) % 10_000_000}"
        self.store.insert(
            MemoryRecord(
                key=key,
                value=key,
                shard_id=shard_id,
                record_id=record_id,
                namespace=namespace,
                content_type="prompt_answer",
                status="active",
                source=f"{namespace}:{record_kind}",
                timestamp=timestamp,
                payload=answer,
                metadata={
                    "prompt": prompt,
                    "lineage": lineage,
                    "record_kind": record_kind,
                },
            )
        )

    def insert_main_record(self, record, answer, timestamp, supersede_prior=False):
        namespace = f"{record.dataset_name}_main"
        lineage = f"{record.dataset_name}::{record.case_id}::main"
        if supersede_prior:
            self.supersede_lineage(lineage)

        prompts = [record.canonical_prompt] + list(record.paraphrase_prompts)
        for prompt in prompts:
            self._insert_prompt_answer(
                prompt=prompt,
                answer=answer,
                lineage=lineage,
                namespace=namespace,
                timestamp=timestamp,
                record_kind="main",
            )

    def insert_retention_case(self, record, prompt_case, timestamp="base_retention"):
        namespace = f"{record.dataset_name}_retention"
        lineage = f"{record.dataset_name}::{record.case_id}::retention::{prompt_case.case_kind}::{stable_hash_int(prompt_case.prompt)}"
        self._insert_prompt_answer(
            prompt=prompt_case.prompt,
            answer=prompt_case.expected_answer,
            lineage=lineage,
            namespace=namespace,
            timestamp=timestamp,
            record_kind=prompt_case.case_kind,
        )

    def answer(self, query, top_k=4, latest_timestamp_bias=None):
        candidates = self.store.active_records()
        if not candidates:
            return {"answer": "", "hits": []}

        query_vector = self.encoder.encode(query)
        keys = torch.stack([record.key for record in candidates], dim=0)
        query_norm = F.normalize(query_vector.unsqueeze(0), dim=-1)
        key_norm = F.normalize(keys, dim=-1)
        similarities = torch.matmul(query_norm, key_norm.transpose(0, 1)).squeeze(0)

        scored = []
        for index, record in enumerate(candidates):
            score = float(similarities[index])
            if latest_timestamp_bias:
                score += latest_timestamp_bias(record.timestamp)
            scored.append((record, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        top_hits = scored[: max(1, min(top_k, len(scored)))]
        best_record, _ = top_hits[0]
        return {
            "answer": str(best_record.payload),
            "hits": [
                {
                    "answer": str(record.payload),
                    "score": score,
                    "timestamp": record.timestamp,
                    "namespace": record.namespace,
                    "lineage": record.metadata.get("lineage", ""),
                    "record_kind": record.metadata.get("record_kind", ""),
                    "prompt": record.metadata.get("prompt", ""),
                }
                for record, score in top_hits
            ],
        }
