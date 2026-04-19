import argparse
import json
from pathlib import Path

from official_benchmark_data import (
    load_counterfact_records,
    load_zsre_records,
    validate_json_file,
)


def build_supervised_examples(records):
    examples = []
    for record in records:
        examples.append(
            {
                "dataset": record.dataset_name,
                "case_id": record.case_id,
                "split_role": "canonical_update",
                "input": record.canonical_prompt,
                "target": record.target_new,
            }
        )
        for paraphrase in record.paraphrase_prompts:
            examples.append(
                {
                    "dataset": record.dataset_name,
                    "case_id": record.case_id,
                    "split_role": "paraphrase_update",
                    "input": paraphrase,
                    "target": record.target_new,
                }
            )
        for retention_case in record.retention_cases:
            examples.append(
                {
                    "dataset": record.dataset_name,
                    "case_id": record.case_id,
                    "split_role": retention_case.case_kind,
                    "input": retention_case.prompt,
                    "target": retention_case.expected_answer,
                }
            )
    return examples


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_official_finetune_dataset_prep(rome_dir, output_dir, counterfact_limit=512, zsre_eval_limit=512, zsre_train_limit=4096):
    rome_root = Path(rome_dir)
    output_root = Path(output_dir)

    summary = {"rome_dir": str(rome_root.resolve()), "output_dir": str(output_root.resolve()), "files": {}}

    train_records = []
    eval_records = []

    counterfact_path = rome_root / "counterfact.json"
    if counterfact_path.exists() and validate_json_file(counterfact_path)["valid"]:
        counterfact_records = load_counterfact_records(counterfact_path, limit=counterfact_limit)
        eval_records.extend(counterfact_records)
        summary["files"]["counterfact_eval_count"] = len(counterfact_records)

    zsre_eval_path = rome_root / "zsre_mend_eval.json"
    if zsre_eval_path.exists() and validate_json_file(zsre_eval_path)["valid"]:
        zsre_eval_records = load_zsre_records(zsre_eval_path, limit=zsre_eval_limit)
        eval_records.extend(zsre_eval_records)
        summary["files"]["zsre_eval_count"] = len(zsre_eval_records)

    zsre_train_path = rome_root / "zsre_mend_train.json"
    if zsre_train_path.exists() and validate_json_file(zsre_train_path)["valid"]:
        zsre_train_records = load_zsre_records(zsre_train_path, limit=zsre_train_limit)
        train_records.extend(zsre_train_records)
        summary["files"]["zsre_train_count"] = len(zsre_train_records)

    train_rows = build_supervised_examples(train_records)
    eval_rows = build_supervised_examples(eval_records)

    train_path = output_root / "official_edit_train.jsonl"
    eval_path = output_root / "official_edit_eval.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    summary["files"]["train_jsonl"] = str(train_path)
    summary["files"]["eval_jsonl"] = str(eval_path)
    summary["files"]["train_example_count"] = len(train_rows)
    summary["files"]["eval_example_count"] = len(eval_rows)

    summary_path = output_root / "official_edit_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\n[NOTE] Wrote prepared datasets to: {output_root}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rome-dir", default="benchmark_data/rome_dsets")
    parser.add_argument("--output-dir", default="benchmark_data/prepared_official_edits")
    parser.add_argument("--counterfact-limit", type=int, default=512)
    parser.add_argument("--zsre-eval-limit", type=int, default=512)
    parser.add_argument("--zsre-train-limit", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run_official_finetune_dataset_prep(
            args.rome_dir,
            args.output_dir,
            counterfact_limit=args.counterfact_limit,
            zsre_eval_limit=args.zsre_eval_limit,
            zsre_train_limit=args.zsre_train_limit,
        )
    )
