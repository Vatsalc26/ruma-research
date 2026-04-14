import argparse
import json
import statistics
import time
from pathlib import Path

from external_corpus_benchmark import (
    apply_conflicts,
    apply_updates,
    build_base_index,
    normalized_contains,
)
from external_corpus_manifest import (
    load_external_corpus_manifest,
    missing_files_for_manifest,
)
from real_doc_answerer import CitationFirstAnswerer


def source_suffixes(packet):
    return [citation["source"].replace("\\", "/") for citation in packet["citations"]]


def build_answer_quality_cases(manifest):
    compositional_cases = []
    conflict_cases = []

    for entry in manifest["documents"]:
        update_suffix = entry["update_file"].replace("\\", "/")
        combined_query = (
            f"{entry['update_query']} "
            f"Also, {entry['retention_query']}"
        )
        compositional_cases.append(
            {
                "name": f"{entry['slug']}_updated_plus_retained",
                "query": combined_query,
                "must_contain_all": [entry["updated_value"], entry["retained_value"]],
                "must_not_contain_any": [entry["base_value"]],
                "expected_source_suffixes_all": [update_suffix],
                "min_citation_count": 1,
                "namespaces": [entry["namespace"]],
                "top_k": 5,
                "max_sentences": 2,
            }
        )

        for conflict_index, conflict in enumerate(entry["conflicts"]):
            conflict_suffix = conflict["file"].replace("\\", "/")
            conflict_query = (
                f"{entry['update_query']} "
                f"Also, {conflict['query']}"
            )
            conflict_cases.append(
                {
                    "name": f"{entry['slug']}_conflict_synthesis_{conflict_index + 1}",
                    "query": conflict_query,
                    "must_contain_all": [entry["updated_value"], conflict["must_contain"]],
                    "expected_source_suffixes_all": [update_suffix, conflict_suffix],
                    "min_citation_count": 2,
                    "min_distinct_sources": 2,
                    "expect_conflict": True,
                    "min_conflict_count": 1,
                    "namespaces": [entry["namespace"]],
                    "top_k": 6,
                    "max_sentences": 3,
                }
            )

    return {
        "compositional_cases": compositional_cases,
        "conflict_cases": conflict_cases,
    }


def evaluate_answer_cases(answerer, cases):
    results = []
    passes = 0
    latencies_ms = []
    total_citations = 0
    total_conflicts = 0

    for case in cases:
        started_at = time.perf_counter()
        packet = answerer.answer(
            case["query"],
            top_k=case.get("top_k", 4),
            namespaces=case.get("namespaces"),
            max_sentences=case.get("max_sentences", 2),
        )
        latencies_ms.append((time.perf_counter() - started_at) * 1000.0)

        answer_text = packet["answer"]
        citations = packet["citations"]
        conflicts = packet["conflicts"]
        citation_paths = source_suffixes(packet)
        distinct_sources = len({citation["source"] for citation in citations})

        ok = True
        for phrase in case.get("must_contain_all", []):
            ok = ok and normalized_contains(answer_text, phrase)
        for phrase in case.get("must_not_contain_any", []):
            ok = ok and (not normalized_contains(answer_text, phrase))
        for suffix in case.get("expected_source_suffixes_all", []):
            ok = ok and any(path.endswith(suffix) for path in citation_paths)
        if "min_citation_count" in case:
            ok = ok and len(citations) >= case["min_citation_count"]
        if "min_distinct_sources" in case:
            ok = ok and distinct_sources >= case["min_distinct_sources"]
        if "expect_conflict" in case:
            ok = ok and (bool(conflicts) == case["expect_conflict"])
        if "min_conflict_count" in case:
            ok = ok and len(conflicts) >= case["min_conflict_count"]

        if ok:
            passes += 1

        total_citations += len(citations)
        total_conflicts += len(conflicts)
        results.append(
            {
                "name": case["name"],
                "passed": ok,
                "answer": answer_text,
                "citations": citations,
                "conflicts": conflicts,
            }
        )

    avg_latency = round(sum(latencies_ms) / max(1, len(latencies_ms)), 4)
    median_latency = round(statistics.median(latencies_ms), 4) if latencies_ms else 0.0

    return {
        "exact_match": passes / max(1, len(cases)),
        "avg_query_latency_ms": avg_latency,
        "median_query_latency_ms": median_latency,
        "avg_citation_count": round(total_citations / max(1, len(cases)), 4),
        "avg_conflict_count": round(total_conflicts / max(1, len(cases)), 4),
        "cases": results,
    }


def run_external_answer_quality_benchmark(manifest_path):
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    manifest = load_external_corpus_manifest(manifest_path)
    missing_files = missing_files_for_manifest(manifest)
    if missing_files:
        print("\n[ERROR] The external corpus manifest is present, but the markdown files are missing.")
        print("Populate the files below, then rerun the benchmark:\n")
        for path in missing_files:
            print(f"- {path}")
        return 1

    quality_cases = build_answer_quality_cases(manifest)

    print("\n=======================================================")
    print("  RUMA EXTERNAL ANSWER QUALITY BENCHMARK")
    print("=======================================================\n")

    print("[1/4] Building base corpus and running harder compositional queries...")
    base_index, base_build_seconds = build_base_index(
        repo_root,
        [
            {
                "path": str(manifest["corpus_root"] / entry["base_file"]),
                "namespace": entry["namespace"],
                "slug": entry["slug"],
            }
            for entry in manifest["documents"]
        ],
    )
    base_answerer = CitationFirstAnswerer(base_index)
    base_compositional_metrics = evaluate_answer_cases(
        base_answerer,
        quality_cases["compositional_cases"],
    )

    print("[2/4] Applying superseding updates and re-evaluating compositional queries...")
    updated_index, _ = build_base_index(
        repo_root,
        [
            {
                "path": str(manifest["corpus_root"] / entry["base_file"]),
                "namespace": entry["namespace"],
                "slug": entry["slug"],
            }
            for entry in manifest["documents"]
        ],
    )
    superseded_update_seconds = apply_updates(
        updated_index,
        [
            {
                "base_path": str(manifest["corpus_root"] / entry["base_file"]),
                "update_path": str(manifest["corpus_root"] / entry["update_file"]),
                "namespace": entry["namespace"],
                "slug": entry["slug"],
            }
            for entry in manifest["documents"]
        ],
        supersede_prior=True,
    )
    updated_answerer = CitationFirstAnswerer(updated_index)
    superseded_compositional_metrics = evaluate_answer_cases(
        updated_answerer,
        quality_cases["compositional_cases"],
    )

    print("[3/4] Applying naive append baseline and conflict notes...")
    append_index, _ = build_base_index(
        repo_root,
        [
            {
                "path": str(manifest["corpus_root"] / entry["base_file"]),
                "namespace": entry["namespace"],
                "slug": entry["slug"],
            }
            for entry in manifest["documents"]
        ],
    )
    naive_append_seconds = apply_updates(
        append_index,
        [
            {
                "base_path": str(manifest["corpus_root"] / entry["base_file"]),
                "update_path": str(manifest["corpus_root"] / entry["update_file"]),
                "namespace": entry["namespace"],
                "slug": entry["slug"],
            }
            for entry in manifest["documents"]
        ],
        supersede_prior=False,
    )
    append_answerer = CitationFirstAnswerer(append_index)
    naive_append_compositional_metrics = evaluate_answer_cases(
        append_answerer,
        quality_cases["compositional_cases"],
    )

    conflict_files = []
    for entry in manifest["documents"]:
        for conflict in entry["conflicts"]:
            conflict_files.append(
                {
                    "path": str(manifest["corpus_root"] / conflict["file"]),
                    "lineage": conflict["lineage"] or f"{entry['namespace']}::operator_guide::{entry['slug']}",
                    "namespace": entry["namespace"],
                }
            )
    conflict_seconds = apply_conflicts(updated_index, conflict_files, manifest["namespace"])
    conflict_answerer = CitationFirstAnswerer(updated_index)
    conflict_metrics = evaluate_answer_cases(
        conflict_answerer,
        quality_cases["conflict_cases"],
    )

    print("[4/4] Writing results...")
    all_case_count = len(quality_cases["compositional_cases"]) + len(quality_cases["conflict_cases"])
    overall_passes = (
        sum(1 for case in superseded_compositional_metrics["cases"] if case["passed"])
        + sum(1 for case in conflict_metrics["cases"] if case["passed"])
    )
    results = {
        "corpus_name": manifest["corpus_name"],
        "manifest_path": str(manifest["manifest_path"]),
        "base_corpus_stats": base_index.stats(),
        "systems_summary": {
            "base_build_seconds": round(base_build_seconds, 6),
            "superseded_update_seconds": round(superseded_update_seconds, 6),
            "naive_append_update_seconds": round(naive_append_seconds, 6),
            "conflict_update_seconds": round(conflict_seconds, 6),
            "base_compositional_avg_query_latency_ms": base_compositional_metrics["avg_query_latency_ms"],
            "superseded_compositional_avg_query_latency_ms": superseded_compositional_metrics["avg_query_latency_ms"],
            "naive_append_compositional_avg_query_latency_ms": naive_append_compositional_metrics["avg_query_latency_ms"],
            "conflict_synthesis_avg_query_latency_ms": conflict_metrics["avg_query_latency_ms"],
        },
        "case_counts": {
            "compositional": len(quality_cases["compositional_cases"]),
            "conflict_synthesis": len(quality_cases["conflict_cases"]),
            "total": all_case_count,
        },
        "base_compositional_eval": base_compositional_metrics,
        "superseded_compositional_eval": superseded_compositional_metrics,
        "naive_append_compositional_eval": naive_append_compositional_metrics,
        "conflict_synthesis_eval": conflict_metrics,
        "overall_quality_eval": {
            "exact_match": overall_passes / max(1, all_case_count),
            "avg_citation_count": round(
                (
                    superseded_compositional_metrics["avg_citation_count"] * len(quality_cases["compositional_cases"])
                    + conflict_metrics["avg_citation_count"] * len(quality_cases["conflict_cases"])
                )
                / max(1, all_case_count),
                4,
            ),
            "avg_conflict_count": round(
                (
                    superseded_compositional_metrics["avg_conflict_count"] * len(quality_cases["compositional_cases"])
                    + conflict_metrics["avg_conflict_count"] * len(quality_cases["conflict_cases"])
                )
                / max(1, all_case_count),
                4,
            ),
        },
    }

    output_path = results_dir / "external_answer_quality_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\n[NOTE] Results written to: {output_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="external_corpora/python_ecosystem_changes/manifest.json",
        help="Path to a filled external corpus manifest JSON file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_external_answer_quality_benchmark(parse_args().manifest))
