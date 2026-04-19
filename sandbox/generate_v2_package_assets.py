import csv
import json
from pathlib import Path


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_text_maturity_rows(results):
    rows = []
    for dataset_name, dataset_results in results.items():
        if dataset_name in {"summary", "config"}:
            continue
        for system_name, payload in dataset_results.items():
            post = payload["post_update"]
            rows.append(
                {
                    "dataset": dataset_name,
                    "system": system_name,
                    "update_teacher_exact_match": post["update_teacher_forced"]["exact_match"],
                    "retention_teacher_exact_match": post["retention_teacher_forced"]["exact_match"],
                    "update_generation_exact_match": post["update_generation"]["exact_match"],
                    "retention_generation_exact_match": post["retention_generation"]["exact_match"],
                    "update_generation_token_accuracy": post["update_generation"]["token_accuracy"],
                    "retention_generation_token_accuracy": post["retention_generation"]["token_accuracy"],
                    "old_text_suppression": round(float(1.0 - post["replaced_generation"]["exact_match"]), 4),
                }
            )
    return rows


def build_ablation_rows(package):
    rows = []
    for comparison_name, metrics in package["ablations"].items():
        row = {"comparison": comparison_name}
        row.update(metrics)
        rows.append(row)
    return rows


def build_snapshot_markdown(package):
    fever = package["grounded_retrieval_verification"]["fever"]["interleaved_controller_rag"]
    hotpot = package["multi_hop_broader_answer_behavior"]["hotpotqa"]["interleaved_controller_rag"]
    text_macro = package["standalone_natural_language"]["text_maturity"]["macro"]
    transformer_text = text_macro["transformer_final_form_ruma"]
    mamba_text = text_macro["mamba_final_form_ruma"]
    return f"""# V2 Benchmark Package Snapshot

- transformer standalone text update-generation exact: `{transformer_text["update_generation_exact_match"]}`
- transformer standalone text retention-generation exact: `{transformer_text["retention_generation_exact_match"]}`
- transformer standalone text update-generation token accuracy: `{transformer_text["update_generation_token_accuracy"]}`
- transformer standalone old-text suppression: `{transformer_text["old_text_suppression"]}`
- mamba standalone text update-generation exact: `{mamba_text["update_generation_exact_match"]}`
- mamba standalone text retention-generation exact: `{mamba_text["retention_generation_exact_match"]}`
- mamba standalone text update-generation token accuracy: `{mamba_text["update_generation_token_accuracy"]}`
- mamba standalone old-text suppression: `{mamba_text["old_text_suppression"]}`
- FEVER interleaved answer-hit: `{fever["answer_contains_gold_sentence"]}`
- FEVER interleaved evidence recall@5: `{fever["evidence_recall_at_5"]}`
- HotpotQA interleaved answer-hit: `{hotpot["answer_contains_gold"]}`
- HotpotQA interleaved full support-chain hit@8: `{hotpot["full_support_chain_hit_at_8"]}`
"""


def run_generate_v2_package_assets():
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "sandbox" / "results"
    paper_assets_dir = repo_root / "paper_assets"
    paper_assets_dir.mkdir(exist_ok=True)

    package = load_json(results_dir / "v2_benchmark_eval_package.json")
    text_results = load_json(results_dir / "v2_standalone_text_maturity_suite.json")

    write_csv(
        paper_assets_dir / "v2_standalone_text_results.csv",
        [
            "dataset",
            "system",
            "update_teacher_exact_match",
            "retention_teacher_exact_match",
            "update_generation_exact_match",
            "retention_generation_exact_match",
            "update_generation_token_accuracy",
            "retention_generation_token_accuracy",
            "old_text_suppression",
        ],
        build_text_maturity_rows(text_results),
    )

    write_csv(
        paper_assets_dir / "v2_ablation_summary.csv",
        [
            "comparison",
            "update_exact_match",
            "retention_exact_match",
            "old_fact_suppression",
            "second_update_exact_match",
            "prior_update_suppression_after_second",
            "expert_sharpness",
        ],
        build_ablation_rows(package),
    )

    (paper_assets_dir / "v2_benchmark_package_snapshot.md").write_text(
        build_snapshot_markdown(package),
        encoding="utf-8",
    )
    print("[NOTE] Wrote v2 package assets to paper_assets/")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_generate_v2_package_assets())
