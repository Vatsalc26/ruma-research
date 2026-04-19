import argparse
import json
import re
import statistics
import string
import time
from pathlib import Path

from hotpotqa_data import load_hotpotqa_records
from rag_baselines import (
    HybridExtractiveRagAnswerer,
    InterleavedControllerRagAnswerer,
    RerankedHybridRagAnswerer,
)
from real_doc_answerer import CitationFirstAnswerer
from real_doc_memory import RealDocRUMAIndex


ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text):
    text = str(text or "").lower().translate(PUNCT_TABLE)
    text = ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def answer_contains_gold(answer_text, gold_answer):
    normalized_answer = normalize_answer(answer_text)
    normalized_gold = normalize_answer(gold_answer)
    return bool(normalized_gold) and normalized_gold in normalized_answer


def build_hotpotqa_index(repo_root, docs):
    index = RealDocRUMAIndex(
        repo_root=repo_root,
        num_shards=24,
        namespace_bandwidth=3,
    )

    entries = []
    for doc in docs:
        source = f"hotpotqa/{doc.title}"
        for sent_id, sentence in enumerate(doc.sentences):
            sentence = str(sentence).strip()
            if not sentence:
                continue
            entries.append(
                {
                    "path": source,
                    "namespace": doc.namespace,
                    "chunk_index": sent_id,
                    "text": sentence,
                }
            )

    index.encoder.fit([entry["text"] for entry in entries])
    records = index._build_records(entries, timestamp="hotpotqa_validation")
    index.store.insert_many(records)
    index.documents = [Path(entry["path"]) for entry in entries]
    index._mark_search_backend_stale()
    return index


def supporting_fact_metrics(packet, supporting_facts):
    hit_keys = {
        (
            citation["source"].replace("hotpotqa/", "", 1) if citation["source"].startswith("hotpotqa/") else citation["source"],
            int(citation["chunk_index"]),
        )
        for citation in packet["retrieval_hits"]
    }
    gold_keys = set(supporting_facts)
    gold_titles = {title for title, _ in gold_keys}
    hit_titles = {title for title, _ in hit_keys}

    supporting_fact_recall = len(hit_keys & gold_keys) / max(1, len(gold_keys))
    supporting_title_coverage = len(hit_titles & gold_titles) / max(1, len(gold_titles))
    full_support_chain_hit = 1.0 if gold_titles and gold_titles.issubset(hit_titles) else 0.0

    return {
        "supporting_fact_recall": supporting_fact_recall,
        "supporting_title_coverage": supporting_title_coverage,
        "full_support_chain_hit": full_support_chain_hit,
    }


def evaluate_questions(answerer, questions, mode_name, use_context_oracle):
    fact_recall_total = 0.0
    title_coverage_total = 0.0
    chain_hit_total = 0.0
    answer_hit_total = 0
    latencies_ms = []
    cases = []

    for record in questions:
        namespaces = record.context_namespaces if use_context_oracle else None
        started_at = time.perf_counter()
        packet = answerer.answer(
            record.question,
            top_k=8,
            namespaces=namespaces,
            max_sentences=3,
        )
        latencies_ms.append((time.perf_counter() - started_at) * 1000.0)

        metrics = supporting_fact_metrics(packet, record.supporting_facts)
        fact_recall_total += metrics["supporting_fact_recall"]
        title_coverage_total += metrics["supporting_title_coverage"]
        chain_hit_total += metrics["full_support_chain_hit"]

        answer_hit = answer_contains_gold(packet["answer"], record.answer)
        if answer_hit:
            answer_hit_total += 1

        cases.append(
            {
                "question_id": record.question_id,
                "question_type": record.question_type,
                "level": record.level,
                "mode": mode_name,
                "supporting_fact_recall": metrics["supporting_fact_recall"],
                "supporting_title_coverage": metrics["supporting_title_coverage"],
                "full_support_chain_hit": bool(metrics["full_support_chain_hit"]),
                "answer_hit": answer_hit,
                "answer": packet["answer"],
                "citations": packet["citations"],
            }
        )

    count = max(1, len(questions))
    return {
        "question_count": len(questions),
        "supporting_fact_recall_at_8": fact_recall_total / count,
        "supporting_title_coverage_at_8": title_coverage_total / count,
        "full_support_chain_hit_at_8": chain_hit_total / count,
        "answer_contains_gold": answer_hit_total / count,
        "avg_query_latency_ms": round(sum(latencies_ms) / count, 4),
        "median_query_latency_ms": round(statistics.median(latencies_ms), 4) if latencies_ms else 0.0,
        "cases": cases,
    }


def run_v2_hotpotqa_bridge_benchmark(
    parquet_path,
    bridge_limit=64,
    comparison_limit=64,
):
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    dataset = load_hotpotqa_records(
        parquet_path,
        bridge_limit=bridge_limit,
        comparison_limit=comparison_limit,
    )
    index = build_hotpotqa_index(repo_root, dataset["corpus_docs"])
    answerer = CitationFirstAnswerer(index)
    hybrid_answerer = HybridExtractiveRagAnswerer(index)
    reranked_hybrid_answerer = RerankedHybridRagAnswerer(index)
    interleaved_controller_answerer = InterleavedControllerRagAnswerer(
        index,
        min_query_coverage=0.25,
        min_controller_confidence=0.35,
        require_dual_signal=False,
        top_source_window=8,
    )

    all_questions = dataset["records"]
    results = {
        "dataset_path": str(Path(parquet_path).resolve()),
        "dataset_stats": dataset["stats"],
        "index_stats": index.stats(),
        "modes": {
            "global_rag": evaluate_questions(answerer, all_questions, "global_rag", use_context_oracle=False),
            "hybrid_rag": evaluate_questions(hybrid_answerer, all_questions, "hybrid_rag", use_context_oracle=False),
            "reranked_hybrid_rag": evaluate_questions(reranked_hybrid_answerer, all_questions, "reranked_hybrid_rag", use_context_oracle=False),
            "interleaved_controller_rag": evaluate_questions(interleaved_controller_answerer, all_questions, "interleaved_controller_rag", use_context_oracle=False),
            "context_oracle_rag": evaluate_questions(answerer, all_questions, "context_oracle_rag", use_context_oracle=True),
        },
    }

    output_path = results_dir / "v2_hotpotqa_bridge_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\n[NOTE] Results written to: {output_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-path", default="benchmark_data/HotpotQA/validation-00000-of-00001.parquet")
    parser.add_argument("--bridge-limit", type=int, default=64)
    parser.add_argument("--comparison-limit", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run_v2_hotpotqa_bridge_benchmark(
            args.parquet_path,
            bridge_limit=args.bridge_limit,
            comparison_limit=args.comparison_limit,
        )
    )
