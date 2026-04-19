import argparse
import json
import statistics
import time
from pathlib import Path

from beir_subset_data import load_beir_subset_records
from rag_baselines import HybridExtractiveRagAnswerer, RerankedHybridRagAnswerer
from real_doc_memory import RealDocRUMAIndex, clean_text, slugify


def normalize_text(text):
    return clean_text(str(text or "")).lower()


def build_beir_index(repo_root, dataset_name, corpus):
    index = RealDocRUMAIndex(
        repo_root=repo_root,
        num_shards=24,
        namespace_bandwidth=3,
    )
    entries = []
    for document in corpus:
        title = str(document.title or "").strip()
        body = str(document.text or "").strip()
        text = f"{title}\n{body}".strip()
        entries.append(
            {
                "path": f"beir/{dataset_name}/{document.doc_id}",
                "namespace": slugify(f"beir_{dataset_name}"),
                "chunk_index": 0,
                "text": text,
            }
        )

    index.encoder.fit([entry["text"] for entry in entries])
    records = index._build_records(entries, timestamp=f"beir_{dataset_name}_subset")
    index.store.insert_many(records)
    index.documents = [Path(entry["path"]) for entry in entries]
    index._mark_search_backend_stale()
    return index


def rank_of_first_hit(hits, relevant_doc_ids):
    for rank, hit in enumerate(hits, start=1):
        doc_id = hit["source"].rsplit("/", 1)[-1]
        if doc_id in relevant_doc_ids:
            return rank
    return None


def evaluate_mode(mode_name, search_fn, queries, top_k):
    recall_at_k = 0
    mrr_at_k = 0.0
    latencies_ms = []
    cases = []

    for query in queries:
        started = time.perf_counter()
        hits = search_fn(query.text, top_k=top_k)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        relevant_set = set(query.relevant_doc_ids)
        first_hit_rank = rank_of_first_hit(hits, relevant_set)

        if first_hit_rank is not None:
            recall_at_k += 1
            mrr_at_k += 1.0 / first_hit_rank

        cases.append(
            {
                "query_id": query.query_id,
                "mode": mode_name,
                "question": query.text,
                "relevant_doc_ids": query.relevant_doc_ids,
                "first_hit_rank": first_hit_rank,
                "top_hits": [
                    {
                        "doc_id": hit["source"].rsplit("/", 1)[-1],
                        "score": hit.get("score", hit.get("hybrid_score", hit.get("rerank_score", 0.0))),
                    }
                    for hit in hits[:top_k]
                ],
            }
        )

    return {
        "query_count": len(queries),
        "recall_at_k": recall_at_k / max(1, len(queries)),
        "mrr_at_k": mrr_at_k / max(1, len(queries)),
        "avg_query_latency_ms": round(sum(latencies_ms) / max(1, len(latencies_ms)), 4),
        "median_query_latency_ms": round(statistics.median(latencies_ms), 4) if latencies_ms else 0.0,
        "cases": cases,
    }


def run_dataset(repo_root, dataset_name, dataset_root, query_limit, distractor_limit, top_k):
    dataset = load_beir_subset_records(
        dataset_root=dataset_root,
        query_limit=query_limit,
        distractor_limit=distractor_limit,
        split_preference="test",
    )
    index = build_beir_index(repo_root, dataset_name, dataset["corpus"])
    hybrid_answerer = HybridExtractiveRagAnswerer(index)
    reranked_answerer = RerankedHybridRagAnswerer(index)

    return {
        "paths": dataset["paths"],
        "dataset_stats": dataset["stats"],
        "index_stats": index.stats(),
        "modes": {
            "dense_rag": evaluate_mode(
                "dense_rag",
                lambda query, top_k: index.search(query, top_k=top_k, namespaces=None),
                dataset["queries"],
                top_k=top_k,
            ),
            "hybrid_rag": evaluate_mode(
                "hybrid_rag",
                lambda query, top_k: hybrid_answerer._ordered_hits(query=query, top_k=top_k, namespaces=None),
                dataset["queries"],
                top_k=top_k,
            ),
            "reranked_hybrid_rag": evaluate_mode(
                "reranked_hybrid_rag",
                lambda query, top_k: reranked_answerer._ordered_hits(query=query, top_k=top_k, namespaces=None),
                dataset["queries"],
                top_k=top_k,
            ),
        },
    }


def run_v2_beir_subset_benchmark(
    fever_root,
    hotpotqa_root,
    nq_root,
    scifact_root,
    query_limit=128,
    distractor_limit=4096,
    top_k=10,
):
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    datasets = {
        "fever": fever_root,
        "hotpotqa": hotpotqa_root,
        "nq": nq_root,
        "scifact": scifact_root,
    }

    results = {
        "query_limit": int(query_limit),
        "distractor_limit": int(distractor_limit),
        "top_k": int(top_k),
        "datasets": {},
    }

    for dataset_name, dataset_root in datasets.items():
        results["datasets"][dataset_name] = run_dataset(
            repo_root=repo_root,
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            query_limit=query_limit,
            distractor_limit=distractor_limit,
            top_k=top_k,
        )

    output_path = results_dir / "v2_beir_subset_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\n[NOTE] Results written to: {output_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fever-root", default="benchmark_data/Beir/BeIRfever")
    parser.add_argument("--hotpotqa-root", default="benchmark_data/Beir/BeIRhotpotqa")
    parser.add_argument("--nq-root", default="benchmark_data/Beir/BeIRnq")
    parser.add_argument("--scifact-root", default="benchmark_data/Beir/BeIRscifact")
    parser.add_argument("--query-limit", type=int, default=128)
    parser.add_argument("--distractor-limit", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run_v2_beir_subset_benchmark(
            fever_root=args.fever_root,
            hotpotqa_root=args.hotpotqa_root,
            nq_root=args.nq_root,
            scifact_root=args.scifact_root,
            query_limit=args.query_limit,
            distractor_limit=args.distractor_limit,
            top_k=args.top_k,
        )
    )
