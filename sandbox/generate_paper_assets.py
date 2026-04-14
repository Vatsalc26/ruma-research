import csv
import json
from pathlib import Path


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path, text):
    path.write_text(text, encoding="utf-8")


def build_main_results_rows(results):
    return [
        {
            "benchmark": "versioned_manual_benchmark",
            "base_known_exact_match": results["base_known_eval"]["exact_match"],
            "no_update_future_exact_match": results["no_update_future_eval"]["exact_match"],
            "superseded_update_exact_match": results["superseded_update_eval"]["exact_match"],
            "retention_exact_match": results["retention_eval"]["exact_match"],
            "naive_append_exact_match": results["naive_append_eval"]["exact_match"],
            "conflict_exact_match": results["conflict_eval"]["exact_match"],
        }
    ]


def build_system_rows(results):
    systems = results["systems_summary"]
    corpus = results["base_corpus_stats"]
    store = corpus["store"]
    return [
        {"metric": "base_corpus_documents", "value": corpus["documents_indexed"], "scope": "starter corpus"},
        {"metric": "base_corpus_chunks", "value": corpus["chunks_indexed"], "scope": "starter corpus"},
        {"metric": "active_payload_bytes", "value": store["active_payload_bytes"], "scope": "starter corpus"},
        {"metric": "base_build_seconds", "value": systems["base_build_seconds"], "scope": "local run"},
        {"metric": "superseded_update_seconds", "value": systems["superseded_update_seconds"], "scope": "local run"},
        {"metric": "naive_append_update_seconds", "value": systems["naive_append_update_seconds"], "scope": "local run"},
        {"metric": "conflict_update_seconds", "value": systems["conflict_update_seconds"], "scope": "local run"},
        {"metric": "base_avg_query_latency_ms", "value": systems["base_avg_query_latency_ms"], "scope": "local run"},
        {"metric": "updated_avg_query_latency_ms", "value": systems["updated_avg_query_latency_ms"], "scope": "local run"},
        {"metric": "retention_avg_query_latency_ms", "value": systems["retention_avg_query_latency_ms"], "scope": "local run"},
        {"metric": "naive_append_avg_query_latency_ms", "value": systems["naive_append_avg_query_latency_ms"], "scope": "local run"},
        {"metric": "conflict_avg_query_latency_ms", "value": systems["conflict_avg_query_latency_ms"], "scope": "local run"},
    ]


def build_case_rows(results, section_key):
    rows = []
    for case in results[section_key]["cases"]:
        rows.append(
            {
                "section": section_key,
                "name": case["name"],
                "passed": case["passed"],
                "answer": case["answer"],
                "citation_count": len(case["citations"]),
                "conflict_count": len(case["conflicts"]),
            }
        )
    return rows


def build_update_policy_ablation_rows(results):
    systems = results["systems_summary"]
    return [
        {
            "policy": "no_update",
            "metric_scope": "future_queries",
            "exact_match": results["no_update_future_eval"]["exact_match"],
            "avg_query_latency_ms": results["no_update_future_eval"]["avg_query_latency_ms"],
            "update_seconds": 0.0,
        },
        {
            "policy": "superseded_update",
            "metric_scope": "future_queries",
            "exact_match": results["superseded_update_eval"]["exact_match"],
            "avg_query_latency_ms": results["superseded_update_eval"]["avg_query_latency_ms"],
            "update_seconds": systems["superseded_update_seconds"],
        },
        {
            "policy": "naive_append",
            "metric_scope": "future_queries",
            "exact_match": results["naive_append_eval"]["exact_match"],
            "avg_query_latency_ms": results["naive_append_eval"]["avg_query_latency_ms"],
            "update_seconds": systems["naive_append_update_seconds"],
        },
        {
            "policy": "superseded_update",
            "metric_scope": "retention_queries",
            "exact_match": results["retention_eval"]["exact_match"],
            "avg_query_latency_ms": results["retention_eval"]["avg_query_latency_ms"],
            "update_seconds": systems["superseded_update_seconds"],
        },
    ]


def build_failure_rows(results):
    rows = []
    for section_key in [
        "no_update_future_eval",
        "naive_append_eval",
    ]:
        for case in results[section_key]["cases"]:
            if case["passed"]:
                continue
            rows.append(
                {
                    "section": section_key,
                    "name": case["name"],
                    "answer": case["answer"],
                    "citation_count": len(case["citations"]),
                    "conflict_count": len(case["conflicts"]),
                }
            )
    return rows


def build_benchmark_flow_mermaid():
    return """flowchart TD
    A[Base manuals v1] --> B[Build base memory index]
    B --> C[Evaluate base-known queries]
    B --> D[Evaluate future queries without updates]
    B --> E[Apply versioned updates with supersession]
    E --> F[Evaluate updated queries]
    E --> G[Evaluate retained queries]
    E --> H[Add active conflict docs]
    H --> I[Evaluate conflict queries]
    B --> J[Apply updates without supersession]
    J --> K[Evaluate naive append baseline]
"""


def build_result_snapshot_markdown(results):
    systems = results["systems_summary"]
    return f"""# Versioned Manual Benchmark Snapshot

- base corpus documents: `{results["base_corpus_stats"]["documents_indexed"]}`
- no-update future exact match: `{results["no_update_future_eval"]["exact_match"]}`
- superseded update exact match: `{results["superseded_update_eval"]["exact_match"]}`
- retention exact match: `{results["retention_eval"]["exact_match"]}`
- naive append exact match: `{results["naive_append_eval"]["exact_match"]}`
- conflict exact match: `{results["conflict_eval"]["exact_match"]}`
- base build seconds: `{systems["base_build_seconds"]}`
- superseded update seconds: `{systems["superseded_update_seconds"]}`
- conflict update seconds: `{systems["conflict_update_seconds"]}`
"""


def build_external_result_rows(results):
    return [
        {
            "benchmark": results["corpus_name"],
            "base_known_exact_match": results["base_known_eval"]["exact_match"],
            "no_update_future_exact_match": results["no_update_future_eval"]["exact_match"],
            "superseded_update_exact_match": results["superseded_update_eval"]["exact_match"],
            "retention_exact_match": results["retention_eval"]["exact_match"],
            "naive_append_exact_match": results["naive_append_eval"]["exact_match"],
            "conflict_exact_match": results["conflict_eval"]["exact_match"],
        }
    ]


def build_external_snapshot_markdown(results):
    systems = results["systems_summary"]
    return f"""# External Corpus Benchmark Snapshot

- corpus name: `{results["corpus_name"]}`
- base corpus documents: `{results["base_corpus_stats"]["documents_indexed"]}`
- no-update future exact match: `{results["no_update_future_eval"]["exact_match"]}`
- superseded update exact match: `{results["superseded_update_eval"]["exact_match"]}`
- retention exact match: `{results["retention_eval"]["exact_match"]}`
- naive append exact match: `{results["naive_append_eval"]["exact_match"]}`
- conflict exact match: `{results["conflict_eval"]["exact_match"]}`
- base build seconds: `{systems["base_build_seconds"]}`
- superseded update seconds: `{systems["superseded_update_seconds"]}`
- conflict update seconds: `{systems["conflict_update_seconds"]}`
"""


def build_external_answer_quality_rows(results):
    return [
        {
            "condition": "base_compositional",
            "exact_match": results["base_compositional_eval"]["exact_match"],
            "avg_query_latency_ms": results["base_compositional_eval"]["avg_query_latency_ms"],
            "avg_citation_count": results["base_compositional_eval"]["avg_citation_count"],
            "avg_conflict_count": results["base_compositional_eval"]["avg_conflict_count"],
        },
        {
            "condition": "superseded_compositional",
            "exact_match": results["superseded_compositional_eval"]["exact_match"],
            "avg_query_latency_ms": results["superseded_compositional_eval"]["avg_query_latency_ms"],
            "avg_citation_count": results["superseded_compositional_eval"]["avg_citation_count"],
            "avg_conflict_count": results["superseded_compositional_eval"]["avg_conflict_count"],
        },
        {
            "condition": "naive_append_compositional",
            "exact_match": results["naive_append_compositional_eval"]["exact_match"],
            "avg_query_latency_ms": results["naive_append_compositional_eval"]["avg_query_latency_ms"],
            "avg_citation_count": results["naive_append_compositional_eval"]["avg_citation_count"],
            "avg_conflict_count": results["naive_append_compositional_eval"]["avg_conflict_count"],
        },
        {
            "condition": "conflict_synthesis",
            "exact_match": results["conflict_synthesis_eval"]["exact_match"],
            "avg_query_latency_ms": results["conflict_synthesis_eval"]["avg_query_latency_ms"],
            "avg_citation_count": results["conflict_synthesis_eval"]["avg_citation_count"],
            "avg_conflict_count": results["conflict_synthesis_eval"]["avg_conflict_count"],
        },
        {
            "condition": "overall_quality",
            "exact_match": results["overall_quality_eval"]["exact_match"],
            "avg_query_latency_ms": "",
            "avg_citation_count": results["overall_quality_eval"]["avg_citation_count"],
            "avg_conflict_count": results["overall_quality_eval"]["avg_conflict_count"],
        },
    ]


def build_external_answer_quality_failure_rows(results):
    rows = []
    for section_key in [
        "base_compositional_eval",
        "naive_append_compositional_eval",
        "conflict_synthesis_eval",
    ]:
        for case in results[section_key]["cases"]:
            if case["passed"]:
                continue
            rows.append(
                {
                    "section": section_key,
                    "name": case["name"],
                    "answer": case["answer"],
                    "citation_count": len(case["citations"]),
                    "conflict_count": len(case["conflicts"]),
                }
            )
    return rows


def build_external_answer_quality_snapshot_markdown(results):
    systems = results["systems_summary"]
    return f"""# External Answer Quality Benchmark Snapshot

- corpus name: `{results["corpus_name"]}`
- compositional cases: `{results["case_counts"]["compositional"]}`
- conflict synthesis cases: `{results["case_counts"]["conflict_synthesis"]}`
- base compositional exact match: `{results["base_compositional_eval"]["exact_match"]}`
- superseded compositional exact match: `{results["superseded_compositional_eval"]["exact_match"]}`
- naive append compositional exact match: `{results["naive_append_compositional_eval"]["exact_match"]}`
- conflict synthesis exact match: `{results["conflict_synthesis_eval"]["exact_match"]}`
- overall quality exact match: `{results["overall_quality_eval"]["exact_match"]}`
- superseded compositional avg latency ms: `{systems["superseded_compositional_avg_query_latency_ms"]}`
- conflict synthesis avg latency ms: `{systems["conflict_synthesis_avg_query_latency_ms"]}`
"""


def build_routing_rows(results):
    return [
        {
            "mode": "global_search",
            "exact_match": results["global_search_eval"]["exact_match"],
            "namespace_hit_rate": results["global_search_eval"]["namespace_hit_rate"],
            "avg_query_latency_ms": results["global_search_eval"]["avg_query_latency_ms"],
        },
        {
            "mode": "keyword_routed",
            "exact_match": results["keyword_routed_eval"]["exact_match"],
            "namespace_hit_rate": results["keyword_routed_eval"]["namespace_hit_rate"],
            "avg_query_latency_ms": results["keyword_routed_eval"]["avg_query_latency_ms"],
        },
        {
            "mode": "oracle_namespace",
            "exact_match": results["oracle_namespace_eval"]["exact_match"],
            "namespace_hit_rate": results["oracle_namespace_eval"]["namespace_hit_rate"],
            "avg_query_latency_ms": results["oracle_namespace_eval"]["avg_query_latency_ms"],
        },
    ]


def build_scaling_rows(results):
    rows = []
    for point in results["scale_points"]:
        rows.append(
            {
                "noise_chunks": point["noise_chunks"],
                "total_records": point["total_records"],
                "namespace_count": point["namespace_count"],
                "global_exact_match": point["global_search_eval"]["exact_match"],
                "global_avg_query_latency_ms": point["global_search_eval"]["avg_query_latency_ms"],
                "routed_exact_match": point["keyword_routed_eval"]["exact_match"],
                "routed_avg_query_latency_ms": point["keyword_routed_eval"]["avg_query_latency_ms"],
            }
        )
    return rows


def build_ann_backend_rows(results):
    rows = []
    for comparison in results["comparisons"]:
        backend_info = comparison["backend_info"]
        rows.append(
            {
                "noise_chunks": comparison["noise_chunks"],
                "requested_backend": comparison["requested_backend"],
                "active_backend": backend_info["active"],
                "fallback_reason": backend_info["fallback_reason"] or "",
                "global_exact_match": comparison["global_search_eval"]["exact_match"],
                "global_avg_query_latency_ms": comparison["global_search_eval"]["avg_query_latency_ms"],
                "routed_exact_match": comparison["keyword_routed_eval"]["exact_match"],
                "routed_avg_query_latency_ms": comparison["keyword_routed_eval"]["avg_query_latency_ms"],
            }
        )
    return rows


def main():
    repo_root = Path(__file__).resolve().parent.parent
    results_path = repo_root / "sandbox" / "results" / "versioned_manual_benchmark.json"
    external_results_path = repo_root / "sandbox" / "results" / "external_corpus_benchmark.json"
    external_answer_quality_path = repo_root / "sandbox" / "results" / "external_answer_quality_benchmark.json"
    routing_results_path = repo_root / "sandbox" / "results" / "multi_namespace_manual_benchmark.json"
    scaling_results_path = repo_root / "sandbox" / "results" / "retrieval_scaling_benchmark.json"
    ann_results_path = repo_root / "sandbox" / "results" / "ann_backend_benchmark.json"
    assets_dir = repo_root / "paper_assets"
    assets_dir.mkdir(exist_ok=True)

    results = json.loads(results_path.read_text(encoding="utf-8"))

    write_csv(
        assets_dir / "versioned_manual_main_results.csv",
        [
            "benchmark",
            "base_known_exact_match",
            "no_update_future_exact_match",
            "superseded_update_exact_match",
            "retention_exact_match",
            "naive_append_exact_match",
            "conflict_exact_match",
        ],
        build_main_results_rows(results),
    )

    write_csv(
        assets_dir / "versioned_manual_systems.csv",
        ["metric", "value", "scope"],
        build_system_rows(results),
    )

    case_rows = []
    for section_key in [
        "base_known_eval",
        "no_update_future_eval",
        "superseded_update_eval",
        "retention_eval",
        "naive_append_eval",
        "conflict_eval",
    ]:
        case_rows.extend(build_case_rows(results, section_key))
    write_csv(
        assets_dir / "versioned_manual_cases.csv",
        ["section", "name", "passed", "answer", "citation_count", "conflict_count"],
        case_rows,
    )

    write_csv(
        assets_dir / "update_policy_ablation.csv",
        ["policy", "metric_scope", "exact_match", "avg_query_latency_ms", "update_seconds"],
        build_update_policy_ablation_rows(results),
    )

    write_csv(
        assets_dir / "failure_cases.csv",
        ["section", "name", "answer", "citation_count", "conflict_count"],
        build_failure_rows(results),
    )

    write_text(
        assets_dir / "figure_benchmark_flow.mmd",
        build_benchmark_flow_mermaid(),
    )
    write_text(
        assets_dir / "versioned_manual_snapshot.md",
        build_result_snapshot_markdown(results),
    )

    if routing_results_path.exists():
        routing_results = json.loads(routing_results_path.read_text(encoding="utf-8"))
        write_csv(
            assets_dir / "routing_modes.csv",
            ["mode", "exact_match", "namespace_hit_rate", "avg_query_latency_ms"],
            build_routing_rows(routing_results),
        )

    if scaling_results_path.exists():
        scaling_results = json.loads(scaling_results_path.read_text(encoding="utf-8"))
        write_csv(
            assets_dir / "retrieval_scaling.csv",
            [
                "noise_chunks",
                "total_records",
                "namespace_count",
                "global_exact_match",
                "global_avg_query_latency_ms",
                "routed_exact_match",
                "routed_avg_query_latency_ms",
            ],
            build_scaling_rows(scaling_results),
        )

    if ann_results_path.exists():
        ann_results = json.loads(ann_results_path.read_text(encoding="utf-8"))
        write_csv(
            assets_dir / "ann_backends.csv",
            [
                "noise_chunks",
                "requested_backend",
                "active_backend",
                "fallback_reason",
                "global_exact_match",
                "global_avg_query_latency_ms",
                "routed_exact_match",
                "routed_avg_query_latency_ms",
            ],
            build_ann_backend_rows(ann_results),
        )

    if external_results_path.exists():
        external_results = json.loads(external_results_path.read_text(encoding="utf-8"))
        write_csv(
            assets_dir / "external_corpus_main_results.csv",
            [
                "benchmark",
                "base_known_exact_match",
                "no_update_future_exact_match",
                "superseded_update_exact_match",
                "retention_exact_match",
                "naive_append_exact_match",
                "conflict_exact_match",
            ],
            build_external_result_rows(external_results),
        )
        write_text(
            assets_dir / "external_corpus_snapshot.md",
            build_external_snapshot_markdown(external_results),
        )

    if external_answer_quality_path.exists():
        external_answer_quality = json.loads(external_answer_quality_path.read_text(encoding="utf-8"))
        write_csv(
            assets_dir / "external_answer_quality.csv",
            [
                "condition",
                "exact_match",
                "avg_query_latency_ms",
                "avg_citation_count",
                "avg_conflict_count",
            ],
            build_external_answer_quality_rows(external_answer_quality),
        )
        write_csv(
            assets_dir / "external_answer_quality_failures.csv",
            ["section", "name", "answer", "citation_count", "conflict_count"],
            build_external_answer_quality_failure_rows(external_answer_quality),
        )
        write_text(
            assets_dir / "external_answer_quality_snapshot.md",
            build_external_answer_quality_snapshot_markdown(external_answer_quality),
        )

    print(f"Wrote paper assets to: {assets_dir}")


if __name__ == "__main__":
    main()
