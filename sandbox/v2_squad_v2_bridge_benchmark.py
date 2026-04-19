import argparse
import json
import re
import statistics
import string
import time
from pathlib import Path

from real_doc_answerer import CitationFirstAnswerer
from real_doc_memory import RealDocRUMAIndex, chunk_text
from squad_v2_data import load_squad_v2_records


ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)
NO_ANSWER_MARKER = "I could not find grounded evidence for that query in the current memory index."


def normalize_answer(text):
    text = str(text or "").lower().translate(PUNCT_TABLE)
    text = ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def any_gold_match(text, gold_answers):
    normalized_text = normalize_answer(text)
    for answer in gold_answers:
        normalized_answer = normalize_answer(answer)
        if normalized_answer and normalized_answer in normalized_text:
            return True
    return False


def is_abstained(packet):
    answer = str(packet.get("answer", "")).strip()
    if answer == NO_ANSWER_MARKER:
        return True
    return not bool(packet.get("citations"))


def build_squad_index(repo_root, paragraphs):
    index = RealDocRUMAIndex(
        repo_root=repo_root,
        num_shards=24,
        namespace_bandwidth=3,
    )
    entries = []
    for paragraph in paragraphs:
        source = f"squad_v2/{paragraph.title}/{paragraph.paragraph_index}"
        for chunk_index, chunk in enumerate(chunk_text(paragraph.context, max_words=120, overlap_words=30)):
            entries.append(
                {
                    "path": source,
                    "namespace": paragraph.namespace,
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )

    index.encoder.fit([entry["text"] for entry in entries])
    records = index._build_records(entries, timestamp="squad_v2_dev")
    index.store.insert_many(records)
    index.documents = [Path(entry["path"]) for entry in entries]
    index._mark_search_backend_stale()
    return index


def evaluate_answerable(answerer, questions, mode_name, oracle_namespace):
    answer_passes = 0
    retrieval_passes = 0
    latencies_ms = []
    cases = []

    for record in questions:
        namespaces = [record.namespace] if oracle_namespace else None
        started_at = time.perf_counter()
        packet = answerer.answer(
            record.question,
            top_k=5,
            namespaces=namespaces,
            max_sentences=2,
        )
        latencies_ms.append((time.perf_counter() - started_at) * 1000.0)

        retrieval_ok = any(any_gold_match(hit["excerpt"], record.answers) for hit in packet["retrieval_hits"])
        answer_ok = any_gold_match(packet["answer"], record.answers)
        if retrieval_ok:
            retrieval_passes += 1
        if answer_ok:
            answer_passes += 1

        cases.append(
            {
                "question_id": record.question_id,
                "title": record.title,
                "mode": mode_name,
                "retrieval_hit": retrieval_ok,
                "answer_hit": answer_ok,
                "answer": packet["answer"],
                "citations": packet["citations"],
            }
        )

    return {
        "question_count": len(questions),
        "retrieval_recall_at_5": retrieval_passes / max(1, len(questions)),
        "answer_contains_gold": answer_passes / max(1, len(questions)),
        "avg_query_latency_ms": round(sum(latencies_ms) / max(1, len(latencies_ms)), 4),
        "median_query_latency_ms": round(statistics.median(latencies_ms), 4) if latencies_ms else 0.0,
        "cases": cases,
    }


def evaluate_impossible(answerer, questions, mode_name, oracle_namespace):
    abstain_passes = 0
    latencies_ms = []
    cases = []

    for record in questions:
        namespaces = [record.namespace] if oracle_namespace else None
        started_at = time.perf_counter()
        packet = answerer.answer(
            record.question,
            top_k=5,
            namespaces=namespaces,
            max_sentences=2,
        )
        latencies_ms.append((time.perf_counter() - started_at) * 1000.0)
        abstained = is_abstained(packet)
        if abstained:
            abstain_passes += 1

        cases.append(
            {
                "question_id": record.question_id,
                "title": record.title,
                "mode": mode_name,
                "abstained": abstained,
                "answer": packet["answer"],
                "citations": packet["citations"],
            }
        )

    return {
        "question_count": len(questions),
        "abstain_exact_match": abstain_passes / max(1, len(questions)),
        "avg_query_latency_ms": round(sum(latencies_ms) / max(1, len(latencies_ms)), 4),
        "median_query_latency_ms": round(statistics.median(latencies_ms), 4) if latencies_ms else 0.0,
        "cases": cases,
    }


def run_v2_squad_v2_bridge_benchmark(
    squad_path,
    answerable_limit=128,
    impossible_limit=64,
    distractor_limit=256,
):
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    dataset = load_squad_v2_records(
        squad_path,
        answerable_limit=answerable_limit,
        impossible_limit=impossible_limit,
        distractor_limit=distractor_limit,
    )

    index = build_squad_index(repo_root, dataset["corpus_paragraphs"])
    answerer = CitationFirstAnswerer(index)

    results = {
        "dataset_path": str(Path(squad_path).resolve()),
        "dataset_stats": dataset["stats"],
        "index_stats": index.stats(),
        "modes": {},
    }

    for mode_name, oracle_namespace in [("global_rag", False), ("title_oracle_rag", True)]:
        results["modes"][mode_name] = {
            "answerable": evaluate_answerable(answerer, dataset["answerable"], mode_name, oracle_namespace),
            "impossible": evaluate_impossible(answerer, dataset["impossible"], mode_name, oracle_namespace),
        }

    output_path = results_dir / "v2_squad_v2_bridge_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\n[NOTE] Results written to: {output_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad-path", default="benchmark_data/SQuAD v2/dev-v2.0.json")
    parser.add_argument("--answerable-limit", type=int, default=128)
    parser.add_argument("--impossible-limit", type=int, default=64)
    parser.add_argument("--distractor-limit", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run_v2_squad_v2_bridge_benchmark(
            args.squad_path,
            answerable_limit=args.answerable_limit,
            impossible_limit=args.impossible_limit,
            distractor_limit=args.distractor_limit,
        )
    )
