import argparse
import json
import statistics
import time
from pathlib import Path

from official_benchmark_data import (
    load_counterfact_records,
    load_zsre_records,
    validate_json_file,
)
from official_edit_memory import PromptEditMemoryIndex
from official_edit_subject_baseline import SubjectOverrideBaseline
from real_doc_memory import clean_text


def normalized_equal(left, right):
    return clean_text(left).lower() == clean_text(right).lower()


def latest_doc_bias(timestamp):
    if timestamp == "update":
        return 0.15
    if timestamp == "base":
        return 0.0
    if timestamp.startswith("retention"):
        return 0.05
    return 0.0


def build_index(records, mode):
    if mode == "subject_override":
        baseline = SubjectOverrideBaseline()
        for record in records:
            baseline.insert(
                dataset_name=record.dataset_name,
                subject=record.subject,
                answer=record.target_true,
                timestamp="base",
                case_id=record.case_id,
            )
            baseline.insert(
                dataset_name=record.dataset_name,
                subject=record.subject,
                answer=record.target_new,
                timestamp="update",
                case_id=record.case_id,
            )
        return baseline

    prompts = []
    for record in records:
        prompts.append(record.canonical_prompt)
        prompts.extend(record.paraphrase_prompts)
        prompts.extend(case.prompt for case in record.retention_cases)

    index = PromptEditMemoryIndex()
    index.fit(prompts)

    for record in records:
        index.insert_main_record(record, answer=record.target_true, timestamp="base", supersede_prior=False)
        for retention_case in record.retention_cases:
            index.insert_retention_case(record, retention_case, timestamp=f"retention_{retention_case.case_kind}")

    if mode == "ruma_supersession":
        for record in records:
            index.insert_main_record(record, answer=record.target_new, timestamp="update", supersede_prior=True)
    elif mode in {"naive_append", "latest_doc"}:
        for record in records:
            index.insert_main_record(record, answer=record.target_new, timestamp="update", supersede_prior=False)

    return index


def evaluate_records(index, records, mode):
    canonical_passes = 0
    paraphrase_total = 0
    paraphrase_passes = 0
    retention_total = 0
    retention_passes = 0
    query_latencies_ms = []

    for record in records:
        start = time.perf_counter()
        if mode == "subject_override":
            packet = index.answer(record.canonical_prompt)
        else:
            packet = index.answer(
                record.canonical_prompt,
                latest_timestamp_bias=latest_doc_bias if mode == "latest_doc" else None,
            )
        query_latencies_ms.append((time.perf_counter() - start) * 1000.0)
        if normalized_equal(packet["answer"], record.target_new):
            canonical_passes += 1

        for paraphrase in record.paraphrase_prompts:
            paraphrase_total += 1
            start = time.perf_counter()
            if mode == "subject_override":
                packet = index.answer(paraphrase)
            else:
                packet = index.answer(
                    paraphrase,
                    latest_timestamp_bias=latest_doc_bias if mode == "latest_doc" else None,
                )
            query_latencies_ms.append((time.perf_counter() - start) * 1000.0)
            if normalized_equal(packet["answer"], record.target_new):
                paraphrase_passes += 1

        for retention_case in record.retention_cases:
            retention_total += 1
            start = time.perf_counter()
            if mode == "subject_override":
                packet = index.answer(retention_case.prompt)
            else:
                packet = index.answer(
                    retention_case.prompt,
                    latest_timestamp_bias=latest_doc_bias if mode == "latest_doc" else None,
                )
            query_latencies_ms.append((time.perf_counter() - start) * 1000.0)
            if normalized_equal(packet["answer"], retention_case.expected_answer):
                retention_passes += 1

    return {
        "canonical_update_exact_match": canonical_passes / max(1, len(records)),
        "paraphrase_exact_match": paraphrase_passes / max(1, paraphrase_total),
        "retention_exact_match": retention_passes / max(1, retention_total),
        "avg_query_latency_ms": round(sum(query_latencies_ms) / max(1, len(query_latencies_ms)), 4),
        "median_query_latency_ms": round(statistics.median(query_latencies_ms), 4) if query_latencies_ms else 0.0,
        "record_count": len(records),
        "paraphrase_count": paraphrase_total,
        "retention_count": retention_total,
    }


def load_available_datasets(rome_dir, counterfact_limit, zsre_limit):
    available = {}

    counterfact_path = Path(rome_dir) / "counterfact.json"
    if counterfact_path.exists():
        validation = validate_json_file(counterfact_path)
        if validation["valid"]:
            available["counterfact"] = load_counterfact_records(counterfact_path, limit=counterfact_limit)
        else:
            available["counterfact_error"] = validation["error"]

    zsre_eval_path = Path(rome_dir) / "zsre_mend_eval.json"
    if zsre_eval_path.exists():
        validation = validate_json_file(zsre_eval_path)
        if validation["valid"]:
            available["zsre"] = load_zsre_records(zsre_eval_path, limit=zsre_limit)
        else:
            available["zsre_error"] = validation["error"]

    return available


def run_v2_official_edit_benchmark(rome_dir, counterfact_limit=128, zsre_limit=128):
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    datasets = load_available_datasets(rome_dir, counterfact_limit=counterfact_limit, zsre_limit=zsre_limit)

    if "counterfact" not in datasets and "zsre" not in datasets:
        print(json.dumps({"error": "No valid official benchmark files are currently available.", "details": datasets}, indent=2))
        return 1

    results = {"rome_dir": str(Path(rome_dir).resolve()), "datasets": {}, "validation": {}}

    for key in ["counterfact_error", "zsre_error"]:
        if key in datasets:
            results["validation"][key] = datasets[key]

    for dataset_name in ["counterfact", "zsre"]:
        records = datasets.get(dataset_name)
        if not records:
            continue

        dataset_result = {}
        for mode in ["ruma_supersession", "naive_append", "latest_doc", "subject_override"]:
            index = build_index(records, mode)
            dataset_result[mode] = evaluate_records(index, records, mode)
        results["datasets"][dataset_name] = dataset_result

    output_path = results_dir / "v2_official_edit_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\n[NOTE] Results written to: {output_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rome-dir",
        default="benchmark_data/rome_dsets",
        help="Directory containing official ROME benchmark JSON files.",
    )
    parser.add_argument("--counterfact-limit", type=int, default=128)
    parser.add_argument("--zsre-limit", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run_v2_official_edit_benchmark(
            args.rome_dir,
            counterfact_limit=args.counterfact_limit,
            zsre_limit=args.zsre_limit,
        )
    )
