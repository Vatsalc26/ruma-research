import argparse
import json
import statistics
import time
from pathlib import Path

from fever_data import load_fever_records, load_fever_wiki_subset
from rag_baselines import (
    AbstentionAwareHybridRagAnswerer,
    AbstentionAwareRerankedHybridRagAnswerer,
    HybridExtractiveRagAnswerer,
    InterleavedControllerRagAnswerer,
    RerankedHybridRagAnswerer,
)
from real_doc_answerer import CitationFirstAnswerer
from real_doc_memory import RealDocRUMAIndex, clean_text, slugify


NO_ANSWER_MARKER = "I could not find grounded evidence for that query in the current memory index."


def normalize_text(text):
    return clean_text(str(text or "")).lower()


def build_fever_index(repo_root, sentence_records):
    index = RealDocRUMAIndex(
        repo_root=repo_root,
        num_shards=24,
        namespace_bandwidth=3,
    )
    entries = []
    for sentence in sentence_records:
        entries.append(
            {
                "path": f"fever/{sentence.page_id}",
                "namespace": sentence.namespace,
                "chunk_index": sentence.sentence_id,
                "text": sentence.text,
            }
        )

    index.encoder.fit([entry["text"] for entry in entries])
    records = index._build_records(entries, timestamp="fever_dev")
    index.store.insert_many(records)
    index.documents = [Path(entry["path"]) for entry in entries]
    index._mark_search_backend_stale()
    return index


def hit_contains_gold_sentence(packet, gold_keys):
    hit_keys = {
        (
            citation["source"].replace("fever/", "", 1) if citation["source"].startswith("fever/") else citation["source"],
            int(citation["chunk_index"]),
        )
        for citation in packet["retrieval_hits"]
    }
    return bool(hit_keys & set(gold_keys))


def page_hit_contains_gold(packet, gold_pages):
    hit_pages = {
        citation["source"].replace("fever/", "", 1) if citation["source"].startswith("fever/") else citation["source"]
        for citation in packet["retrieval_hits"]
    }
    return bool(hit_pages & set(gold_pages))


def answer_contains_gold_sentence(packet, gold_sentence_texts):
    normalized_answer = normalize_text(packet["answer"])
    for sentence_text in gold_sentence_texts:
        if normalize_text(sentence_text) and normalize_text(sentence_text) in normalized_answer:
            return True
    return False


def is_abstained(packet):
    answer = str(packet.get("answer", "")).strip()
    if answer == NO_ANSWER_MARKER:
        return True
    return not bool(packet.get("citations"))


def build_gold_sentence_map(sentence_records):
    mapping = {}
    for record in sentence_records:
        mapping[(record.page_id, int(record.sentence_id))] = record.text
    return mapping


def evaluate_verifiable(answerer, records, mode_name, gold_sentence_map, oracle_namespace):
    evidence_recall = 0
    page_recall = 0
    answer_hit = 0
    latencies_ms = []
    cases = []

    for record in records:
        namespaces = None
        if oracle_namespace:
            namespaces = [slugify(f"fever_{page}") for page in record.evidence_pages]
        started = time.perf_counter()
        packet = answerer.answer(
            record.claim,
            top_k=5,
            namespaces=namespaces,
            max_sentences=2,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

        gold_sentence_texts = [
            gold_sentence_map[key]
            for key in record.evidence_keys
            if key in gold_sentence_map
        ]
        evidence_ok = hit_contains_gold_sentence(packet, record.evidence_keys)
        page_ok = page_hit_contains_gold(packet, record.evidence_pages)
        answer_ok = answer_contains_gold_sentence(packet, gold_sentence_texts)

        if evidence_ok:
            evidence_recall += 1
        if page_ok:
            page_recall += 1
        if answer_ok:
            answer_hit += 1

        cases.append(
            {
                "claim_id": record.claim_id,
                "label": record.label,
                "mode": mode_name,
                "evidence_hit": evidence_ok,
                "page_hit": page_ok,
                "answer_hit": answer_ok,
                "answer": packet["answer"],
                "citations": packet["citations"],
            }
        )

    return {
        "claim_count": len(records),
        "evidence_recall_at_5": evidence_recall / max(1, len(records)),
        "page_recall_at_5": page_recall / max(1, len(records)),
        "answer_contains_gold_sentence": answer_hit / max(1, len(records)),
        "avg_query_latency_ms": round(sum(latencies_ms) / max(1, len(latencies_ms)), 4),
        "median_query_latency_ms": round(statistics.median(latencies_ms), 4) if latencies_ms else 0.0,
        "cases": cases,
    }


def evaluate_nei(answerer, records, mode_name):
    abstain_passes = 0
    latencies_ms = []
    cases = []

    for record in records:
        started = time.perf_counter()
        packet = answerer.answer(
            record.claim,
            top_k=5,
            namespaces=None,
            max_sentences=2,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        abstained = is_abstained(packet)
        if abstained:
            abstain_passes += 1
        cases.append(
            {
                "claim_id": record.claim_id,
                "mode": mode_name,
                "abstained": abstained,
                "answer": packet["answer"],
                "citations": packet["citations"],
            }
        )

    return {
        "claim_count": len(records),
        "abstain_exact_match": abstain_passes / max(1, len(records)),
        "avg_query_latency_ms": round(sum(latencies_ms) / max(1, len(latencies_ms)), 4),
        "median_query_latency_ms": round(statistics.median(latencies_ms), 4) if latencies_ms else 0.0,
        "cases": cases,
    }


def run_v2_fever_bridge_benchmark(
    fever_dev_path,
    fever_wiki_dir,
    support_limit=64,
    refute_limit=64,
    nei_limit=64,
    distractor_limit=256,
):
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    datasets = load_fever_records(
        fever_dev_path,
        support_limit=support_limit,
        refute_limit=refute_limit,
        nei_limit=nei_limit,
    )
    verifiable = datasets["supports"] + datasets["refutes"]
    required_pages = set()
    for record in verifiable:
        required_pages.update(record.evidence_pages)

    wiki_subset = load_fever_wiki_subset(
        fever_wiki_dir,
        required_page_ids=required_pages,
        distractor_limit=distractor_limit,
    )
    gold_sentence_map = build_gold_sentence_map(wiki_subset["sentence_records"])
    index = build_fever_index(repo_root, wiki_subset["sentence_records"])
    answerer = CitationFirstAnswerer(index)
    hybrid_answerer = HybridExtractiveRagAnswerer(index)
    reranked_hybrid_answerer = RerankedHybridRagAnswerer(index)
    guarded_hybrid_answerer = AbstentionAwareHybridRagAnswerer(index)
    guarded_reranked_hybrid_answerer = AbstentionAwareRerankedHybridRagAnswerer(index)
    interleaved_controller_answerer = InterleavedControllerRagAnswerer(
        index,
        min_query_coverage=0.4,
        min_controller_confidence=0.5,
        require_dual_signal=False,
    )
    consensus_hybrid_answerer = AbstentionAwareHybridRagAnswerer(
        index,
        min_query_coverage=0.35,
        require_dual_signal=True,
        dual_signal_top_k=2,
        min_top_source_count=2,
        top_source_window=4,
    )
    consensus_reranked_hybrid_answerer = AbstentionAwareRerankedHybridRagAnswerer(
        index,
        min_query_coverage=0.35,
        require_dual_signal=True,
        dual_signal_top_k=2,
        min_top_source_count=2,
        top_source_window=4,
    )

    results = {
        "fever_dev_path": str(Path(fever_dev_path).resolve()),
        "fever_wiki_dir": str(Path(fever_wiki_dir).resolve()),
        "dataset_stats": {
            "supports": len(datasets["supports"]),
            "refutes": len(datasets["refutes"]),
            "nei": len(datasets["nei"]),
            "required_page_count": len(required_pages),
            "selected_page_count": wiki_subset["selected_page_count"],
            "distractor_page_count": wiki_subset["distractor_page_count"],
            "sentence_record_count": len(wiki_subset["sentence_records"]),
        },
        "index_stats": index.stats(),
        "modes": {
            "global_rag": {
                "verifiable": evaluate_verifiable(answerer, verifiable, "global_rag", gold_sentence_map, oracle_namespace=False),
                "nei": evaluate_nei(answerer, datasets["nei"], "global_rag"),
            },
            "hybrid_rag": {
                "verifiable": evaluate_verifiable(hybrid_answerer, verifiable, "hybrid_rag", gold_sentence_map, oracle_namespace=False),
                "nei": evaluate_nei(hybrid_answerer, datasets["nei"], "hybrid_rag"),
            },
            "reranked_hybrid_rag": {
                "verifiable": evaluate_verifiable(reranked_hybrid_answerer, verifiable, "reranked_hybrid_rag", gold_sentence_map, oracle_namespace=False),
                "nei": evaluate_nei(reranked_hybrid_answerer, datasets["nei"], "reranked_hybrid_rag"),
            },
            "guarded_hybrid_rag": {
                "verifiable": evaluate_verifiable(guarded_hybrid_answerer, verifiable, "guarded_hybrid_rag", gold_sentence_map, oracle_namespace=False),
                "nei": evaluate_nei(guarded_hybrid_answerer, datasets["nei"], "guarded_hybrid_rag"),
            },
            "consensus_hybrid_rag": {
                "verifiable": evaluate_verifiable(consensus_hybrid_answerer, verifiable, "consensus_hybrid_rag", gold_sentence_map, oracle_namespace=False),
                "nei": evaluate_nei(consensus_hybrid_answerer, datasets["nei"], "consensus_hybrid_rag"),
            },
            "guarded_reranked_hybrid_rag": {
                "verifiable": evaluate_verifiable(guarded_reranked_hybrid_answerer, verifiable, "guarded_reranked_hybrid_rag", gold_sentence_map, oracle_namespace=False),
                "nei": evaluate_nei(guarded_reranked_hybrid_answerer, datasets["nei"], "guarded_reranked_hybrid_rag"),
            },
            "interleaved_controller_rag": {
                "verifiable": evaluate_verifiable(interleaved_controller_answerer, verifiable, "interleaved_controller_rag", gold_sentence_map, oracle_namespace=False),
                "nei": evaluate_nei(interleaved_controller_answerer, datasets["nei"], "interleaved_controller_rag"),
            },
            "consensus_reranked_hybrid_rag": {
                "verifiable": evaluate_verifiable(consensus_reranked_hybrid_answerer, verifiable, "consensus_reranked_hybrid_rag", gold_sentence_map, oracle_namespace=False),
                "nei": evaluate_nei(consensus_reranked_hybrid_answerer, datasets["nei"], "consensus_reranked_hybrid_rag"),
            },
            "page_oracle_rag": {
                "verifiable": evaluate_verifiable(answerer, verifiable, "page_oracle_rag", gold_sentence_map, oracle_namespace=True),
            },
        },
    }

    output_path = results_dir / "v2_fever_bridge_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\n[NOTE] Results written to: {output_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fever-dev", default="benchmark_data/FEVER/shared_task_dev.jsonl")
    parser.add_argument("--fever-wiki-dir", default="benchmark_data/FEVER/wiki-pages")
    parser.add_argument("--support-limit", type=int, default=64)
    parser.add_argument("--refute-limit", type=int, default=64)
    parser.add_argument("--nei-limit", type=int, default=64)
    parser.add_argument("--distractor-limit", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run_v2_fever_bridge_benchmark(
            args.fever_dev,
            args.fever_wiki_dir,
            support_limit=args.support_limit,
            refute_limit=args.refute_limit,
            nei_limit=args.nei_limit,
            distractor_limit=args.distractor_limit,
        )
    )
