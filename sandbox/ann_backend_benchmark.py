import json
from pathlib import Path

from ann_retrieval import search_backend_availability
from multi_namespace_manual_benchmark import build_cases, evaluate_mode
from real_doc_answerer import CitationFirstAnswerer
from retrieval_scaling_benchmark import build_index_with_noise, load_noise_chunks


BACKENDS = ["exact", "faiss_flat", "hnsw"]
NOISE_POINTS = [0, 1000]


def evaluate_backend(repo_root, backend_name, noise_chunks, noise_count):
    index = build_index_with_noise(repo_root, noise_chunks, noise_count)
    index.set_search_backend(backend_name)
    answerer = CitationFirstAnswerer(index)
    cases = build_cases()
    global_metrics = evaluate_mode(answerer, cases, mode_name="global")
    routed_metrics = evaluate_mode(answerer, cases, mode_name="routed")
    return {
        "backend_info": index.search_backend_info(),
        "global_search_eval": {
            "exact_match": global_metrics["exact_match"],
            "namespace_hit_rate": global_metrics["namespace_hit_rate"],
            "avg_query_latency_ms": global_metrics["avg_query_latency_ms"],
        },
        "keyword_routed_eval": {
            "exact_match": routed_metrics["exact_match"],
            "namespace_hit_rate": routed_metrics["namespace_hit_rate"],
            "avg_query_latency_ms": routed_metrics["avg_query_latency_ms"],
        },
    }


def run_ann_backend_benchmark():
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("\n=======================================================")
    print("      RUMA ANN BACKEND BENCHMARK")
    print("=======================================================\n")

    print("[1/3] Checking backend availability...")
    availability = search_backend_availability(BACKENDS)
    print(json.dumps(availability, indent=2))

    print("[2/3] Loading distractor chunks for backend comparison...")
    noise_chunks = load_noise_chunks(repo_root)
    print(f"Loaded {len(noise_chunks)} reusable distractor chunks.")

    print("[3/3] Running backend comparisons...")
    comparisons = []
    for noise_count in NOISE_POINTS:
        for backend_name in BACKENDS:
            print(f"  - backend={backend_name} noise_chunks={noise_count}")
            comparisons.append(
                {
                    "noise_chunks": noise_count,
                    "requested_backend": backend_name,
                    **evaluate_backend(
                        repo_root=repo_root,
                        backend_name=backend_name,
                        noise_chunks=noise_chunks,
                        noise_count=noise_count,
                    ),
                }
            )

    results = {
        "backend_availability": availability,
        "comparisons": comparisons,
        "note": (
            "If FAISS or HNSW are unavailable locally, the benchmark records the requested "
            "backend and the exact-search fallback. This keeps the adapter path wired "
            "without blocking the current evidence loop on environment setup."
        ),
    }
    print(json.dumps(results, indent=2))
    (results_dir / "ann_backend_benchmark.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n[NOTE] This benchmark checks whether the optional ANN backend path is wired,")
    print("       whether FAISS or HNSW are available locally, and how exact fallback")
    print("       behaves at small and noisier corpus scales.")
    print(f"       Results written to: {results_dir / 'ann_backend_benchmark.json'}")


if __name__ == "__main__":
    run_ann_backend_benchmark()
