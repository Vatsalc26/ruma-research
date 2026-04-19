import argparse
import json
import statistics
import time
from pathlib import Path

from external_answer_quality_benchmark import build_answer_quality_cases, evaluate_answer_cases
from external_corpus_benchmark import (
    apply_conflicts,
    apply_updates,
    build_base_index,
    normalized_contains,
)
from external_corpus_manifest import load_external_corpus_manifest, missing_files_for_manifest
from rag_baselines import ExtractiveRagAnswerer, InterleavedControllerRagAnswerer
from real_doc_answerer import CitationFirstAnswerer


def source_matches(packet, expected_suffix):
    expected_suffix = expected_suffix.replace("\\", "/")
    return any(citation["source"].replace("\\", "/").endswith(expected_suffix) for citation in packet["citations"])


def evaluate_fact_cases(answerer, cases):
    results = []
    passes = 0
    latencies_ms = []
    for case in cases:
        started_at = time.perf_counter()
        packet = answerer.answer(
            case["query"],
            top_k=case.get("top_k", 4),
            namespaces=case.get("namespaces"),
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
            ok = ok and (bool(packet["conflicts"]) == case["expect_conflict"])

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


def build_corpus_lists(manifest):
    base_files = []
    update_files = []
    conflict_files = []
    latest_source_priority = {}

    for entry in manifest["documents"]:
        base_path = str(manifest["corpus_root"] / entry["base_file"])
        update_path = str(manifest["corpus_root"] / entry["update_file"])
        base_files.append(
            {
                "path": base_path,
                "namespace": entry["namespace"],
                "slug": entry["slug"],
            }
        )
        update_files.append(
            {
                "base_path": base_path,
                "update_path": update_path,
                "namespace": entry["namespace"],
                "slug": entry["slug"],
            }
        )
        latest_source_priority[base_path] = 0
        latest_source_priority[update_path] = 1

        for conflict in entry["conflicts"]:
            conflict_path = str(manifest["corpus_root"] / conflict["file"])
            conflict_files.append(
                {
                    "path": conflict_path,
                    "lineage": conflict["lineage"] or f"{entry['namespace']}::operator_guide::{entry['slug']}",
                    "namespace": entry["namespace"],
                }
            )
            latest_source_priority[conflict_path] = 2

    return {
        "base_files": base_files,
        "update_files": update_files,
        "conflict_files": conflict_files,
        "latest_source_priority": latest_source_priority,
    }


def baseline_result_bundle(answerer, manifest, quality_cases):
    update_cases = []
    retention_cases = []
    for entry in manifest["documents"]:
        update_cases.append(
            {
                "name": f"{entry['slug']}_updated_value",
                "query": entry["update_query"],
                "must_contain": entry["updated_value"],
                "must_not_contain": entry["base_value"],
                "expected_source_suffix": entry["update_file"].replace("\\", "/"),
                "namespaces": [entry["namespace"]],
            }
        )
        retention_cases.append(
            {
                "name": f"{entry['slug']}_retained_value",
                "query": entry["retention_query"],
                "must_contain": entry["retained_value"],
                "namespaces": [entry["namespace"]],
            }
        )

    updated_metrics = evaluate_fact_cases(answerer, update_cases)
    retention_metrics = evaluate_fact_cases(answerer, retention_cases)
    compositional_metrics = evaluate_answer_cases(answerer, quality_cases["compositional_cases"])
    conflict_metrics = evaluate_answer_cases(answerer, quality_cases["conflict_cases"])

    overall_case_count = len(quality_cases["compositional_cases"]) + len(quality_cases["conflict_cases"])
    overall_passes = (
        sum(1 for case in compositional_metrics["cases"] if case["passed"])
        + sum(1 for case in conflict_metrics["cases"] if case["passed"])
    )

    return {
        "updated_eval": updated_metrics,
        "retention_eval": retention_metrics,
        "compositional_eval": compositional_metrics,
        "conflict_eval": conflict_metrics,
        "overall_quality_eval": {
            "exact_match": overall_passes / max(1, overall_case_count),
            "avg_citation_count": round(
                (
                    compositional_metrics["avg_citation_count"] * len(quality_cases["compositional_cases"])
                    + conflict_metrics["avg_citation_count"] * len(quality_cases["conflict_cases"])
                )
                / max(1, overall_case_count),
                4,
            ),
            "avg_conflict_count": round(
                (
                    compositional_metrics["avg_conflict_count"] * len(quality_cases["compositional_cases"])
                    + conflict_metrics["avg_conflict_count"] * len(quality_cases["conflict_cases"])
                )
                / max(1, overall_case_count),
                4,
            ),
        },
    }


def summarize_for_table(bundle):
    return {
        "updated_exact_match": bundle["updated_eval"]["exact_match"],
        "retention_exact_match": bundle["retention_eval"]["exact_match"],
        "compositional_exact_match": bundle["compositional_eval"]["exact_match"],
        "conflict_exact_match": bundle["conflict_eval"]["exact_match"],
        "overall_quality_exact_match": bundle["overall_quality_eval"]["exact_match"],
        "avg_query_latency_ms": {
            "updated": bundle["updated_eval"]["avg_query_latency_ms"],
            "retention": bundle["retention_eval"]["avg_query_latency_ms"],
            "compositional": bundle["compositional_eval"]["avg_query_latency_ms"],
            "conflict": bundle["conflict_eval"]["avg_query_latency_ms"],
        },
    }


def run_v2_external_baseline_benchmark(manifest_path):
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

    corpus_lists = build_corpus_lists(manifest)
    quality_cases = build_answer_quality_cases(manifest)

    print("\n=======================================================")
    print("      RUMA V2 EXTERNAL BASELINE BENCHMARK")
    print("=======================================================\n")

    print("[1/4] Building RUMA-updated index...")
    ruma_index, base_build_seconds = build_base_index(repo_root, corpus_lists["base_files"])
    ruma_update_seconds = apply_updates(ruma_index, corpus_lists["update_files"], supersede_prior=True)
    ruma_conflict_seconds = apply_conflicts(ruma_index, corpus_lists["conflict_files"], manifest["namespace"])
    ruma_answerer = CitationFirstAnswerer(ruma_index)
    interleaved_controller_answerer = InterleavedControllerRagAnswerer(
        ruma_index,
        min_query_coverage=0.38,
        min_controller_confidence=0.48,
        require_dual_signal=False,
    )
    ruma_bundle = baseline_result_bundle(ruma_answerer, manifest, quality_cases)
    interleaved_controller_bundle = baseline_result_bundle(interleaved_controller_answerer, manifest, quality_cases)

    print("[2/4] Building plain-RAG append baseline...")
    append_index, _ = build_base_index(repo_root, corpus_lists["base_files"])
    append_update_seconds = apply_updates(append_index, corpus_lists["update_files"], supersede_prior=False)
    append_conflict_seconds = apply_conflicts(append_index, corpus_lists["conflict_files"], manifest["namespace"])
    plain_rag_answerer = ExtractiveRagAnswerer(append_index)
    plain_rag_bundle = baseline_result_bundle(plain_rag_answerer, manifest, quality_cases)

    print("[3/4] Building latest-doc retrieval baseline...")
    latest_doc_answerer = ExtractiveRagAnswerer(
        append_index,
        source_priority=corpus_lists["latest_source_priority"],
    )
    latest_doc_bundle = baseline_result_bundle(latest_doc_answerer, manifest, quality_cases)

    print("[4/4] Writing results...")
    results = {
        "corpus_name": manifest["corpus_name"],
        "manifest_path": str(manifest["manifest_path"]),
        "base_corpus_stats": ruma_index.stats(),
        "systems_summary": {
            "base_build_seconds": round(base_build_seconds, 6),
            "ruma_update_seconds": round(ruma_update_seconds, 6),
            "ruma_conflict_seconds": round(ruma_conflict_seconds, 6),
            "plain_rag_append_seconds": round(append_update_seconds, 6),
            "plain_rag_conflict_seconds": round(append_conflict_seconds, 6),
        },
        "baselines": {
            "ruma_supersession": ruma_bundle,
            "interleaved_controller_ruma": interleaved_controller_bundle,
            "plain_rag": plain_rag_bundle,
            "latest_doc_rag": latest_doc_bundle,
        },
        "table_summary": {
            "ruma_supersession": summarize_for_table(ruma_bundle),
            "interleaved_controller_ruma": summarize_for_table(interleaved_controller_bundle),
            "plain_rag": summarize_for_table(plain_rag_bundle),
            "latest_doc_rag": summarize_for_table(latest_doc_bundle),
        },
    }
    output_path = results_dir / "v2_external_baseline_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["table_summary"], indent=2))
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
    raise SystemExit(run_v2_external_baseline_benchmark(parse_args().manifest))
