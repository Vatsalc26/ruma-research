import argparse
import json
import re
import statistics
import time
from pathlib import Path

from external_corpus_manifest import build_external_corpus_cases, load_external_corpus_manifest, missing_files_for_manifest
from real_doc_answerer import CitationFirstAnswerer
from real_doc_memory import RealDocRUMAIndex, clean_text


def normalized_contains(text, phrase):
    normalized_phrase = clean_text(phrase).lower()
    normalized_text = clean_text(text).lower()
    pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(normalized_phrase)}(?![a-z0-9_])")
    return bool(pattern.search(normalized_text))


def source_matches(packet, expected_suffix):
    return any(citation["source"].endswith(expected_suffix) for citation in packet["citations"])


def evaluate_cases(answerer, cases, namespaces):
    results = []
    passes = 0
    latencies_ms = []
    for case in cases:
        started_at = time.perf_counter()
        packet = answerer.answer(
            case["query"],
            top_k=case.get("top_k", 4),
            namespaces=case.get("namespaces", namespaces),
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


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def build_base_index(repo_root, base_files):
    started_at = time.perf_counter()
    index = RealDocRUMAIndex(
        repo_root=repo_root,
        num_shards=16,
        namespace_bandwidth=3,
    )
    index.documents = [Path(entry["path"]) for entry in base_files]
    for entry in base_files:
        base_path = entry["path"]
        index.ingest_text_update(
            read_text(base_path),
            source=str(Path(base_path).as_posix()),
            namespace=entry["namespace"],
            timestamp="base_corpus",
            lineage=f"{entry['namespace']}::{entry['slug']}",
            supersede_prior=False,
        )
    return index, time.perf_counter() - started_at


def apply_updates(index, update_files, supersede_prior):
    started_at = time.perf_counter()
    for entry in update_files:
        base_path = entry["base_path"]
        update_path = entry["update_path"]
        update_name = Path(update_path).stem
        lineage = f"{entry['namespace']}::{entry['slug']}"
        index.ingest_text_update(
            read_text(update_path),
            source=Path(update_path).as_posix(),
            namespace=entry["namespace"],
            timestamp=update_name,
            lineage=lineage,
            supersede_prior=supersede_prior,
        )
    return time.perf_counter() - started_at


def apply_conflicts(index, conflict_files, namespace):
    started_at = time.perf_counter()
    for entry in conflict_files:
        conflict_path = entry["path"]
        index.ingest_text_update(
            read_text(conflict_path),
            source=Path(conflict_path).as_posix(),
            namespace=entry["namespace"],
            timestamp=Path(conflict_path).stem,
            lineage=entry["lineage"],
            supersede_prior=False,
        )
    return time.perf_counter() - started_at


def run_external_corpus_benchmark(manifest_path):
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

    corpus = build_external_corpus_cases(manifest)
    namespace_filter = None

    print("\n=======================================================")
    print("      RUMA EXTERNAL CORPUS BENCHMARK")
    print("=======================================================\n")

    print(f"[1/5] Building base corpus from: {manifest['corpus_name']}")
    base_index, base_build_seconds = build_base_index(repo_root, corpus["base_files"])
    base_answerer = CitationFirstAnswerer(base_index)
    base_metrics = evaluate_cases(base_answerer, corpus["base_known_cases"], namespace_filter)
    no_update_future_metrics = evaluate_cases(base_answerer, corpus["update_cases"], namespace_filter)

    print("[2/5] Applying same-lineage superseding updates...")
    update_index, _ = build_base_index(repo_root, corpus["base_files"])
    superseded_update_seconds = apply_updates(
        update_index,
        corpus["update_files"],
        supersede_prior=True,
    )
    update_answerer = CitationFirstAnswerer(update_index)
    updated_metrics = evaluate_cases(update_answerer, corpus["update_cases"], namespace_filter)
    retention_metrics = evaluate_cases(update_answerer, corpus["retention_cases"], namespace_filter)

    print("[3/5] Applying updates without supersession...")
    append_index, _ = build_base_index(repo_root, corpus["base_files"])
    naive_append_update_seconds = apply_updates(
        append_index,
        corpus["update_files"],
        supersede_prior=False,
    )
    append_answerer = CitationFirstAnswerer(append_index)
    naive_append_metrics = evaluate_cases(append_answerer, corpus["update_cases"], namespace_filter)

    print("[4/5] Applying conflict snippets...")
    if corpus["conflict_files"]:
        conflict_update_seconds = apply_conflicts(update_index, corpus["conflict_files"], corpus["namespace"])
        conflict_answerer = CitationFirstAnswerer(update_index)
        conflict_metrics = evaluate_cases(conflict_answerer, corpus["conflict_cases"], namespace_filter)
    else:
        conflict_update_seconds = 0.0
        conflict_metrics = {
            "exact_match": None,
            "avg_query_latency_ms": 0.0,
            "median_query_latency_ms": 0.0,
            "cases": [],
        }

    print("[5/5] Writing results...")
    results = {
        "corpus_name": manifest["corpus_name"],
        "manifest_path": str(manifest["manifest_path"]),
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
    output_path = results_dir / "external_corpus_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

    print("\n[NOTE] This benchmark is the bridge from the internal starter corpus")
    print("       to manually curated external changing-document corpora.")
    print(f"       Results written to: {output_path}")
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
    raise SystemExit(run_external_corpus_benchmark(parse_args().manifest))
