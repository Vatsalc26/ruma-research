import json
import re
import statistics
import time
from pathlib import Path

from real_doc_answerer import CitationFirstAnswerer
from real_doc_memory import RealDocRUMAIndex, clean_text
from versioned_manual_benchmark import (
    BASE_FILES,
    CONFLICT_CASES,
    CONFLICT_FILES,
    RETENTION_CASES,
    UPDATE_CASES,
    UPDATE_FILES,
    read_text,
)


ROUTE_ALIASES = {
    "orchid": ["orchid"],
    "nova": ["nova"],
    "ruma": ["ruma"],
    "cinder": ["cinder"],
    "helios": ["helios"],
    "marlin": ["marlin"],
    "quartz": ["quartz"],
    "atlas": ["atlas"],
    "ember": ["ember"],
    "lumen": ["lumen"],
    "sable": ["sable"],
    "tidal": ["tidal"],
}


def normalized_contains(text, phrase):
    normalized_phrase = clean_text(phrase).lower()
    normalized_text = clean_text(text).lower()
    pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(normalized_phrase)}(?![a-z0-9_])")
    return bool(pattern.search(normalized_text))


def namespace_from_path(relative_path):
    stem = Path(relative_path).stem.lower()
    for namespace, aliases in ROUTE_ALIASES.items():
        if any(alias in stem for alias in aliases):
            return namespace
    return "misc"


def lineage_for_base_path(base_relative_path):
    namespace = namespace_from_path(base_relative_path)
    normalized = base_relative_path.replace("\\", "/")
    return f"{namespace}::{normalized}"


def expected_namespace_from_case(case_name):
    prefix = case_name.split("_", 1)[0]
    return prefix


def route_query_to_namespaces(query):
    lowered = clean_text(query).lower()
    matches = []
    for namespace, aliases in ROUTE_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            matches.append(namespace)
    return matches or None


def build_multi_namespace_index(repo_root):
    index = RealDocRUMAIndex(
        repo_root=repo_root,
        num_shards=24,
        namespace_bandwidth=2,
    )

    all_paths = list(BASE_FILES)
    all_paths.extend(update_path for _, update_path in UPDATE_FILES)
    all_paths.extend(conflict_path for conflict_path, _ in CONFLICT_FILES)
    index.encoder.fit([read_text(repo_root, relative_path) for relative_path in all_paths])

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
        conflict_namespace = namespace_from_path(conflict_path)
        index.ingest_text_update(
            read_text(repo_root, conflict_path),
            source=conflict_path,
            namespace=conflict_namespace,
            timestamp=Path(conflict_path).stem,
            lineage=lineage,
            supersede_prior=False,
        )

    return index


def build_cases():
    cases = []
    for case in UPDATE_CASES:
        copied = dict(case)
        copied["expected_namespace"] = expected_namespace_from_case(case["name"])
        cases.append(copied)
    for case in RETENTION_CASES:
        copied = dict(case)
        copied["expected_namespace"] = expected_namespace_from_case(case["name"])
        cases.append(copied)
    for case in CONFLICT_CASES:
        copied = dict(case)
        copied["expected_namespace"] = expected_namespace_from_case(case["name"])
        cases.append(copied)
    return cases


def evaluate_mode(answerer, cases, mode_name):
    latencies_ms = []
    passes = 0
    namespace_hits = 0
    results = []

    for case in cases:
        if mode_name == "global":
            namespaces = None
        elif mode_name == "routed":
            namespaces = route_query_to_namespaces(case["query"])
        elif mode_name == "oracle":
            namespaces = [case["expected_namespace"]]
        else:
            raise ValueError(f"Unknown mode: {mode_name}")

        started_at = time.perf_counter()
        packet = answerer.answer(
            case["query"],
            top_k=case.get("top_k", 5),
            namespaces=namespaces,
            max_sentences=case.get("max_sentences", 2),
        )
        latencies_ms.append((time.perf_counter() - started_at) * 1000.0)

        passed = True
        if "must_contain" in case:
            passed = passed and normalized_contains(packet["answer"], case["must_contain"])
        if "must_not_contain" in case:
            passed = passed and (not normalized_contains(packet["answer"], case["must_not_contain"]))
        if "expect_conflict" in case:
            passed = passed and (bool(packet["conflicts"]) == case["expect_conflict"])

        first_namespace = packet["citations"][0]["namespace"] if packet["citations"] else None
        namespace_hit = first_namespace == case["expected_namespace"]

        if passed:
            passes += 1
        if namespace_hit:
            namespace_hits += 1

        results.append(
            {
                "name": case["name"],
                "passed": passed,
                "namespace_hit": namespace_hit,
                "expected_namespace": case["expected_namespace"],
                "routed_namespaces": namespaces,
                "answer": packet["answer"],
                "citations": packet["citations"],
                "conflicts": packet["conflicts"],
            }
        )

    return {
        "exact_match": passes / max(1, len(cases)),
        "namespace_hit_rate": namespace_hits / max(1, len(cases)),
        "avg_query_latency_ms": round(sum(latencies_ms) / max(1, len(latencies_ms)), 4),
        "median_query_latency_ms": round(statistics.median(latencies_ms), 4) if latencies_ms else 0.0,
        "cases": results,
    }


def run_multi_namespace_manual_benchmark():
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("\n=======================================================")
    print("      RUMA MULTI-NAMESPACE MANUAL BENCHMARK")
    print("=======================================================\n")

    print("[1/3] Building the multi-namespace index...")
    index = build_multi_namespace_index(repo_root)
    answerer = CitationFirstAnswerer(index)
    cases = build_cases()

    print("[2/3] Evaluating global, routed, and oracle namespace search...")
    global_metrics = evaluate_mode(answerer, cases, mode_name="global")
    routed_metrics = evaluate_mode(answerer, cases, mode_name="routed")
    oracle_metrics = evaluate_mode(answerer, cases, mode_name="oracle")

    print("[3/3] Summarizing routing results...")
    results = {
        "index_stats": index.stats(),
        "case_count": len(cases),
        "global_search_eval": global_metrics,
        "keyword_routed_eval": routed_metrics,
        "oracle_namespace_eval": oracle_metrics,
    }
    print(json.dumps(results, indent=2))
    (results_dir / "multi_namespace_manual_benchmark.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n[NOTE] This benchmark asks whether simple namespace routing is already")
    print("       sufficient on the current changing-document corpus, or whether")
    print("       a heavier borrowed router is likely to be the next bottleneck.")
    print(f"       Results written to: {results_dir / 'multi_namespace_manual_benchmark.json'}")


if __name__ == "__main__":
    run_multi_namespace_manual_benchmark()
