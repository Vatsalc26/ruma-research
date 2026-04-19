import json
from pathlib import Path

from v2_beir_subset_benchmark import run_v2_beir_subset_benchmark
from v2_external_baseline_benchmark import run_v2_external_baseline_benchmark
from v2_fever_bridge_benchmark import run_v2_fever_bridge_benchmark
from v2_final_form_ruma_stabilization_suite import run_v2_final_form_ruma_stabilization_suite
from v2_hotpotqa_bridge_benchmark import run_v2_hotpotqa_bridge_benchmark
from v2_official_edit_benchmark import run_v2_official_edit_benchmark
from v2_standalone_text_maturity_suite import run_v2_standalone_text_maturity_suite


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_result(path, refresh, runner):
    target = Path(path)
    if refresh or not target.exists():
        runner()
    if not target.exists():
        raise FileNotFoundError(f"Expected benchmark result was not produced: {target}")


def run_v2_benchmark_eval_package(refresh=False):
    sandbox_root = Path(__file__).resolve().parent
    results_dir = sandbox_root / "results"
    results_dir.mkdir(exist_ok=True)

    ensure_result(
        results_dir / "v2_final_form_ruma_stabilization_suite.json",
        refresh,
        run_v2_final_form_ruma_stabilization_suite,
    )
    ensure_result(
        results_dir / "v2_standalone_text_maturity_suite.json",
        refresh,
        run_v2_standalone_text_maturity_suite,
    )
    ensure_result(
        results_dir / "v2_official_edit_benchmark.json",
        refresh,
        lambda: run_v2_official_edit_benchmark("benchmark_data/rome_dsets", counterfact_limit=128, zsre_limit=128),
    )
    ensure_result(
        results_dir / "v2_external_baseline_benchmark.json",
        refresh,
        lambda: run_v2_external_baseline_benchmark("external_corpora/python_ecosystem_changes/manifest.json"),
    )
    ensure_result(
        results_dir / "v2_fever_bridge_benchmark.json",
        refresh,
        lambda: run_v2_fever_bridge_benchmark(
            fever_dev_path="benchmark_data/FEVER/shared_task_dev.jsonl",
            fever_wiki_dir="benchmark_data/FEVER/wiki-pages",
            support_limit=64,
            refute_limit=64,
            nei_limit=64,
            distractor_limit=256,
        ),
    )
    ensure_result(
        results_dir / "v2_hotpotqa_bridge_benchmark.json",
        refresh,
        lambda: run_v2_hotpotqa_bridge_benchmark(
            parquet_path="benchmark_data/HotpotQA/validation-00000-of-00001.parquet",
            bridge_limit=64,
            comparison_limit=64,
        ),
    )
    ensure_result(
        results_dir / "v2_beir_subset_benchmark.json",
        refresh,
        lambda: run_v2_beir_subset_benchmark(
            fever_root="benchmark_data/Beir/BeIRfever",
            hotpotqa_root="benchmark_data/Beir/BeIRhotpotqa",
            nq_root="benchmark_data/Beir/BeIRnq",
            scifact_root="benchmark_data/Beir/BeIRscifact",
            query_limit=96,
            distractor_limit=2048,
            top_k=10,
        ),
    )

    stabilization = load_json(results_dir / "v2_final_form_ruma_stabilization_suite.json")
    standalone_text = load_json(results_dir / "v2_standalone_text_maturity_suite.json")
    official_edit = load_json(results_dir / "v2_official_edit_benchmark.json")
    external = load_json(results_dir / "v2_external_baseline_benchmark.json")
    fever = load_json(results_dir / "v2_fever_bridge_benchmark.json")
    hotpot = load_json(results_dir / "v2_hotpotqa_bridge_benchmark.json")
    beir = load_json(results_dir / "v2_beir_subset_benchmark.json")

    package = {
        "update_editing": {
            "standalone_final_form": stabilization["summary"],
            "standalone_text_maturity": standalone_text["summary"],
            "official_edit": {
                dataset_name: {
                    mode_name: {
                        "canonical_update_exact_match": round(float(mode_result["canonical_update_exact_match"]), 4),
                        "paraphrase_exact_match": round(float(mode_result["paraphrase_exact_match"]), 4),
                        "retention_exact_match": round(float(mode_result["retention_exact_match"]), 4),
                    }
                    for mode_name, mode_result in dataset_result.items()
                }
                for dataset_name, dataset_result in official_edit["datasets"].items()
            },
        },
        "grounded_retrieval_verification": {
            "external_baseline": external["table_summary"],
            "fever": {
                mode_name: {
                    "evidence_recall_at_5": round(float(mode_result["verifiable"]["evidence_recall_at_5"]), 4),
                    "page_recall_at_5": round(float(mode_result["verifiable"]["page_recall_at_5"]), 4),
                    "answer_contains_gold_sentence": round(float(mode_result["verifiable"]["answer_contains_gold_sentence"]), 4),
                    "nei_abstain_exact_match": round(float(mode_result.get("nei", {}).get("abstain_exact_match", 0.0)), 4),
                }
                for mode_name, mode_result in fever["modes"].items()
            },
            "beir": {
                dataset_name: {
                    mode_name: {
                        "recall_at_k": round(float(mode_result["recall_at_k"]), 4),
                        "mrr_at_k": round(float(mode_result["mrr_at_k"]), 4),
                    }
                    for mode_name, mode_result in dataset_result["modes"].items()
                }
                for dataset_name, dataset_result in beir["datasets"].items()
            },
        },
        "multi_hop_broader_answer_behavior": {
            "hotpotqa": {
                mode_name: {
                    "supporting_fact_recall_at_8": round(float(mode_result["supporting_fact_recall_at_8"]), 4),
                    "full_support_chain_hit_at_8": round(float(mode_result["full_support_chain_hit_at_8"]), 4),
                    "answer_contains_gold": round(float(mode_result["answer_contains_gold"]), 4),
                }
                for mode_name, mode_result in hotpot["modes"].items()
            },
        },
        "standalone_natural_language": {
            "text_maturity": standalone_text["summary"],
        },
        "ablations": stabilization["summary"]["comparison"],
        "artifacts": {
            "stabilization_suite": "sandbox/results/v2_final_form_ruma_stabilization_suite.json",
            "standalone_text_maturity": "sandbox/results/v2_standalone_text_maturity_suite.json",
            "official_edit": "sandbox/results/v2_official_edit_benchmark.json",
            "external_baseline": "sandbox/results/v2_external_baseline_benchmark.json",
            "fever": "sandbox/results/v2_fever_bridge_benchmark.json",
            "hotpotqa": "sandbox/results/v2_hotpotqa_bridge_benchmark.json",
            "beir": "sandbox/results/v2_beir_subset_benchmark.json",
        },
    }

    output_path = results_dir / "v2_benchmark_eval_package.json"
    output_path.write_text(json.dumps(package, indent=2), encoding="utf-8")
    print(json.dumps(package, indent=2))
    print(f"\n[NOTE] Results written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_v2_benchmark_eval_package())
