import json
from pathlib import Path

from multi_namespace_manual_benchmark import (
    build_cases,
    evaluate_mode,
    lineage_for_base_path,
    namespace_from_path,
)
from real_doc_answerer import CitationFirstAnswerer
from real_doc_memory import DEFAULT_CORPUS_PATHS, RealDocRUMAIndex, chunk_text, clean_text
from versioned_manual_benchmark import BASE_FILES, CONFLICT_FILES, UPDATE_FILES, read_text


SCALE_POINTS = [0, 100, 500, 1000]


def load_noise_chunks(repo_root, limit=1500):
    candidates = [repo_root / "alice.txt"]
    candidates.extend(repo_root / relative_path for relative_path in DEFAULT_CORPUS_PATHS)

    chunks = []
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk_index, chunk in enumerate(chunk_text(text, max_words=120, overlap_words=30)):
            cleaned = clean_text(chunk)
            if len(cleaned.split()) < 40:
                continue
            chunks.append(
                {
                    "source": path.name,
                    "chunk_index": chunk_index,
                    "text": cleaned,
                }
            )
            if len(chunks) >= limit:
                return chunks
    return chunks


def build_index_with_noise(repo_root, noise_chunks, noise_count):
    index = RealDocRUMAIndex(
        repo_root=repo_root,
        num_shards=64,
        namespace_bandwidth=2,
        chunk_words=140,
        overlap_words=35,
    )

    all_texts = [read_text(repo_root, relative_path) for relative_path in BASE_FILES]
    all_texts.extend(read_text(repo_root, update_path) for _, update_path in UPDATE_FILES)
    all_texts.extend(read_text(repo_root, conflict_path) for conflict_path, _ in CONFLICT_FILES)
    all_texts.extend(noise_chunk["text"] for noise_chunk in noise_chunks[:noise_count])
    index.encoder.fit(all_texts)

    for base_path in BASE_FILES:
        index.ingest_text_update(
            read_text(repo_root, base_path),
            source=base_path,
            namespace=namespace_from_path(base_path),
            timestamp=Path(base_path).stem,
            lineage=lineage_for_base_path(base_path),
            supersede_prior=False,
        )

    for base_path, update_path in UPDATE_FILES:
        index.ingest_text_update(
            read_text(repo_root, update_path),
            source=update_path,
            namespace=namespace_from_path(update_path),
            timestamp=Path(update_path).stem,
            lineage=lineage_for_base_path(base_path),
            supersede_prior=True,
        )

    for conflict_path, lineage in CONFLICT_FILES:
        index.ingest_text_update(
            read_text(repo_root, conflict_path),
            source=conflict_path,
            namespace=namespace_from_path(conflict_path),
            timestamp=Path(conflict_path).stem,
            lineage=lineage,
            supersede_prior=False,
        )

    for noise_index, noise_chunk in enumerate(noise_chunks[:noise_count]):
        namespace = f"noise_{noise_index:04d}"
        source = f"scaling_noise/{namespace}_{noise_chunk['source']}_{noise_chunk['chunk_index']}.md"
        index.ingest_text_update(
            noise_chunk["text"],
            source=source,
            namespace=namespace,
            timestamp=f"noise_{noise_index:04d}",
            lineage=f"{namespace}::{source}",
            supersede_prior=False,
        )

    return index


def run_retrieval_scaling_benchmark():
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("\n=======================================================")
    print("      RUMA RETRIEVAL SCALING BENCHMARK")
    print("=======================================================\n")

    print("[1/4] Loading distractor chunks...")
    noise_chunks = load_noise_chunks(repo_root)
    print(f"Loaded {len(noise_chunks)} reusable distractor chunks.")

    print("[2/4] Building the shared evaluation cases...")
    cases = build_cases()

    print("[3/4] Running scale points...")
    scale_results = []
    for noise_count in SCALE_POINTS:
        print(f"  - evaluating with {noise_count} distractor chunks")
        index = build_index_with_noise(repo_root, noise_chunks, noise_count)
        answerer = CitationFirstAnswerer(index)
        global_metrics = evaluate_mode(answerer, cases, mode_name="global")
        routed_metrics = evaluate_mode(answerer, cases, mode_name="routed")
        scale_results.append(
            {
                "noise_chunks": noise_count,
                "total_records": index.stats()["store"]["total_records"],
                "namespace_count": index.stats()["namespace_count"],
                "global_search_eval": {
                    "exact_match": global_metrics["exact_match"],
                    "namespace_hit_rate": global_metrics["namespace_hit_rate"],
                    "avg_query_latency_ms": global_metrics["avg_query_latency_ms"],
                    "median_query_latency_ms": global_metrics["median_query_latency_ms"],
                },
                "keyword_routed_eval": {
                    "exact_match": routed_metrics["exact_match"],
                    "namespace_hit_rate": routed_metrics["namespace_hit_rate"],
                    "avg_query_latency_ms": routed_metrics["avg_query_latency_ms"],
                    "median_query_latency_ms": routed_metrics["median_query_latency_ms"],
                },
            }
        )

    print("[4/4] Writing results...")
    results = {
        "case_count": len(cases),
        "scale_points": scale_results,
        "note": (
            "This benchmark measures when unrestricted exact search starts to become "
            "meaningfully more expensive than simple namespace-routed exact search "
            "as irrelevant document noise grows."
        ),
    }
    print(json.dumps(results, indent=2))
    (results_dir / "retrieval_scaling_benchmark.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n[NOTE] This benchmark is the systems trigger for borrowing ANN infrastructure.")
    print("       When exact search cost grows but routed quality still holds,")
    print("       RUMA should borrow retrieval infrastructure next rather than a heavier router.")
    print(f"       Results written to: {results_dir / 'retrieval_scaling_benchmark.json'}")


if __name__ == "__main__":
    run_retrieval_scaling_benchmark()
