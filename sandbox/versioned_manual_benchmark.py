import json
import re
import statistics
import time
from pathlib import Path

from real_doc_answerer import CitationFirstAnswerer
from real_doc_memory import RealDocRUMAIndex, clean_text
from versioned_manual_corpus import (
    BASE_FILES,
    BASE_KNOWN_CASES,
    CONFLICT_CASES,
    CONFLICT_FILES,
    MANUAL_NAMESPACE,
    RETENTION_CASES,
    UPDATE_CASES,
    UPDATE_FILES,
)


def normalized_contains(text, phrase):
    normalized_phrase = clean_text(phrase).lower()
    normalized_text = clean_text(text).lower()
    pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(normalized_phrase)}(?![a-z0-9_])")
    return bool(pattern.search(normalized_text))


def source_matches(packet, expected_suffix):
    return any(citation["source"].endswith(expected_suffix) for citation in packet["citations"])


def evaluate_cases(answerer, cases):
    results = []
    passes = 0
    latencies_ms = []
    for case in cases:
        started_at = time.perf_counter()
        packet = answerer.answer(
            case["query"],
            top_k=case.get("top_k", 4),
            namespaces=[MANUAL_NAMESPACE],
            max_sentences=case.get("max_sentences", 2),
        )
        latencies_ms.append((time.perf_counter() - started_at) * 1000.0)
        ok = True
        if "must_contain" in case:
            ok = ok and normalized_contains(packet["answer"], case["must_contain"])
        if "must_not_contain" in case:
            ok = ok and (not normalized_contains(packet["answer"], case["must_not_contain"]))
        if "expected_source_suffix" in case:
            ok = ok and source_matches(packet, case["expected_source_suffix"])
        if "expect_conflict" in case:
            has_conflict = bool(packet["conflicts"])
            ok = ok and (has_conflict == case["expect_conflict"])

        if ok:
            passes += 1

        results.append(
            {
                "name": case["name"],
                "passed": ok,
                "answer": packet["answer"],
                "citations": packet["citations"],
                "conflicts": packet["conflicts"],
            }
        )

    return {
        "exact_match": passes / max(1, len(cases)),
        "avg_query_latency_ms": round(sum(latencies_ms) / max(1, len(latencies_ms)), 4),
        "median_query_latency_ms": round(statistics.median(latencies_ms), 4) if latencies_ms else 0.0,
        "cases": results,
    }


def read_text(repo_root, relative_path):
    return (repo_root / relative_path).read_text(encoding="utf-8")


def base_lineage(base_relative_path):
    normalized = base_relative_path.replace("\\", "/")
    return f"{MANUAL_NAMESPACE}::{normalized}"


def build_base_index(repo_root):
    started_at = time.perf_counter()
    index = RealDocRUMAIndex(
        repo_root=repo_root,
        num_shards=16,
        namespace_bandwidth=3,
    )
    index.build_from_paths(paths=[repo_root / relative_path for relative_path in BASE_FILES])
    return index, time.perf_counter() - started_at


def apply_updates(index, repo_root, supersede_prior):
    started_at = time.perf_counter()
    for base_path, update_path in UPDATE_FILES:
        index.ingest_text_update(
            read_text(repo_root, update_path),
            source=update_path,
            namespace=MANUAL_NAMESPACE,
            timestamp=Path(update_path).stem,
            lineage=base_lineage(base_path),
            supersede_prior=supersede_prior,
        )
    return time.perf_counter() - started_at


def apply_conflicts(index, repo_root):
    started_at = time.perf_counter()
    for conflict_path, lineage in CONFLICT_FILES:
        index.ingest_text_update(
            read_text(repo_root, conflict_path),
            source=conflict_path,
            namespace=MANUAL_NAMESPACE,
            timestamp=Path(conflict_path).stem,
            lineage=lineage,
            supersede_prior=False,
        )
    return time.perf_counter() - started_at


def run_versioned_manual_benchmark():
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("\n=======================================================")
    print("      RUMA VERSIONED MANUAL BENCHMARK")
    print("=======================================================\n")

    print("[1/5] Building the base manual corpus...")
    base_index, base_build_seconds = build_base_index(repo_root)
    base_answerer = CitationFirstAnswerer(base_index)
    base_metrics = evaluate_cases(base_answerer, BASE_KNOWN_CASES)
    no_update_future_metrics = evaluate_cases(base_answerer, UPDATE_CASES)

    print("[2/5] Applying updates with explicit same-lineage supersession...")
    update_index, _ = build_base_index(repo_root)
    superseded_update_seconds = apply_updates(update_index, repo_root, supersede_prior=True)
    update_answerer = CitationFirstAnswerer(update_index)
    updated_metrics = evaluate_cases(update_answerer, UPDATE_CASES)
    retention_metrics = evaluate_cases(update_answerer, RETENTION_CASES)

    print("[3/5] Applying updates without supersession as a weaker baseline...")
    append_index, _ = build_base_index(repo_root)
    naive_append_update_seconds = apply_updates(append_index, repo_root, supersede_prior=False)
    append_answerer = CitationFirstAnswerer(append_index)
    naive_append_metrics = evaluate_cases(append_answerer, UPDATE_CASES)

    print("[4/5] Adding a conflicting active operator guide...")
    conflict_update_seconds = apply_conflicts(update_index, repo_root)
    conflict_answerer = CitationFirstAnswerer(update_index)
    conflict_metrics = evaluate_cases(conflict_answerer, CONFLICT_CASES)

    print("[5/5] Summarizing results...")
    results = {
        "base_corpus_stats": base_index.stats(),
        "systems_summary": {
            "base_build_seconds": round(base_build_seconds, 6),
            "superseded_update_seconds": round(superseded_update_seconds, 6),
            "naive_append_update_seconds": round(naive_append_update_seconds, 6),
            "conflict_update_seconds": round(conflict_update_seconds, 6),
            "base_avg_query_latency_ms": base_metrics["avg_query_latency_ms"],
            "updated_avg_query_latency_ms": updated_metrics["avg_query_latency_ms"],
            "retention_avg_query_latency_ms": retention_metrics["avg_query_latency_ms"],
            "naive_append_avg_query_latency_ms": naive_append_metrics["avg_query_latency_ms"],
            "conflict_avg_query_latency_ms": conflict_metrics["avg_query_latency_ms"],
        },
        "base_known_eval": base_metrics,
        "no_update_future_eval": no_update_future_metrics,
        "superseded_update_eval": updated_metrics,
        "retention_eval": retention_metrics,
        "naive_append_eval": naive_append_metrics,
        "conflict_eval": conflict_metrics,
    }
    print(json.dumps(results, indent=2))
    (results_dir / "versioned_manual_benchmark.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n[NOTE] This is the first curated versioned-manual corpus benchmark.")
    print("       It tests same-lineage supersession, retained unchanged guidance,")
    print("       conflict surfacing, and first-pass systems costs on an inspectable corpus.")
    print(f"       Results written to: {results_dir / 'versioned_manual_benchmark.json'}")


if __name__ == "__main__":
    run_versioned_manual_benchmark()
